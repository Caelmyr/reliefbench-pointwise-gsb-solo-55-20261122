"""Browser and local workflow HTTP routes."""
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from threading import BoundedSemaphore
import json
import os

import numpy as np
from flask import Flask, jsonify, request, send_file, send_from_directory
from werkzeug.exceptions import BadRequest, RequestEntityTooLarge

from .archive import build_packet, verify_packet
from .demo import DEMOS, cover, sample
from .models import (CaptureSpec, CropSpec, Point, Problem, SurfaceSpec, choice, fields, filename,
                     flag, identifier, mapping, text, whole)
from .printbook import report_pdf
from .procedures import preview_procedure, save_procedure
from .provenance import public_signature, require_match, seal
from .raster import decode, png
from .registration import Registration, examine_registration
from .results import Result
from .storage import Library
from .visual import GOLD, MINT, flatten_rgb, overlay, scale_bar, split_visual
from .transfer import export_capture_set, import_capture_set, inspect_capture_set
from .workflow import catalog, run


PHOTO_LIMIT = 18 * 1024 * 1024
PACKET_LIMIT = 72 * 1024 * 1024
STATIC = Path(__file__).resolve().parent.parent / "static"
VERSION = "1.0.0"


class WorkerBudget:
    def __init__(self, concurrent):
        self.latch = BoundedSemaphore(concurrent)

    @contextmanager
    def enter(self):
        if not self.latch.acquire(timeout=8):
            raise Problem("An image job is using the workbench. Wait briefly and retry.", 503)
        try:
            yield
        finally:
            self.latch.release()


def data():
    try:
        return mapping(request.get_json())
    except BadRequest:
        raise Problem("Enter a valid JSON object.") from None


def multipart_meta(key, default=None):
    raw = request.form.get(key)
    if raw is None and default is not None:
        return default
    if not raw or len(raw) > 90_000:
        raise Problem(f"Attach a small {key} document.")
    try:
        return mapping(json.loads(raw), key)
    except (json.JSONDecodeError, RecursionError):
        raise Problem(f"The {key} field must contain a JSON object.") from None


def upload(key, limit):
    file = request.files.get(key)
    if not file or not file.filename:
        raise Problem(f"Choose a {key} file.")
    payload = file.stream.read(limit + 1)
    if not payload:
        raise Problem("The uploaded file is empty.")
    if len(payload) > limit:
        raise Problem("The uploaded file is larger than the workbench limit.", 413)
    return filename(file.filename), payload


def resource(payload, mime, name, download=False):
    response = send_file(BytesIO(payload), mimetype=mime, download_name=name, as_attachment=download,
                         max_age=0, conditional=True)
    response.headers["X-Content-Type-Options"] = "nosniff"
    return response


def _registration_trial(photo, spec, physical):
    model = Registration.from_data({"id": "preview", **spec.public()})
    raw = decode(photo[1], photo[0]["filename"])
    code, visible, scale, metrics = examine_registration(raw, model, physical["width_mm"], physical["height_mm"])
    if visible.mean() < .12:
        raise Problem("Too little usable surface area lies inside these corners.")
    source = flatten_rgb(raw.rgb, raw.alpha >= .98, False)
    proto = scale_bar(flatten_rgb(code, visible, False), scale.width_mm)
    drawn = overlay(source, [("closed", model.corners.tolist(), MINT, "Surface outline"),
                             *[("point", [p], GOLD, str(i)) for i, p in enumerate(model.corners.tolist(), 1)]])
    report = Result("registration", "Four-corner registration",
                    "Compare original coordinates against the measured planar crop.",
                    {"scale": scale.public(), **metrics})
    report.image("registration-preview", split_visual(drawn, proto, "Camera", "Physical crop"))
    report.image("rectified-frame", proto)
    return report, metrics


def register_library(app, library, work):
    @app.get("/api/health")
    def health():
        return {"status": "ready", "application": "ReliefBench", "version": VERSION}

    @app.get("/api/catalog")
    def index():
        return {"surfaces": library.surfaces(), "tools": catalog(), "samples": [s.public() for s in DEMOS.values()],
                "limits": {"photos_per_surface": 48, "max_upload_bytes": PHOTO_LIMIT}}

    @app.get("/api/examples/<key>/cover")
    def sample_cover(key):
        return resource(cover(identifier(key)), "image/png", "training-scene.png")

    @app.post("/api/examples/<key>")
    def load_example(key):
        with work.enter():
            item, inserted = sample(identifier(key), library)
        return {"surface": item, "created": inserted}, 201 if inserted else 200

    @app.post("/api/surfaces")
    def surface_create():
        spec = SurfaceSpec.parse(data())
        return {"surface": library.create_surface(spec)}, 201

    @app.get("/api/surfaces/<sid>")
    def surface_details(sid):
        return library.get_surface(identifier(sid))

    @app.put("/api/surfaces/<sid>")
    def surface_edit(sid):
        doc = data()
        fields(doc, "version", "surface")
        expected = whole(doc.get("version"), "Surface version", 1, 1_000_000)
        spec = SurfaceSpec.parse(doc.get("surface"))
        return {"surface": library.edit_surface(identifier(sid), expected, spec)}

    @app.post("/api/surfaces/<sid>/captures")
    def new_capture(sid):
        title, image = upload("photo", PHOTO_LIMIT)
        params = multipart_meta("metadata", {})
        if "title" not in params:
            params["title"] = title.rsplit(".", 1)[0]
        spec = CaptureSpec.parse(params)
        with work.enter():
            record = library.add_capture(identifier(sid), title, image, spec)
        return {"capture": record}, 201

    @app.put("/api/captures/<cid>")
    def edit_capture(cid):
        obj = data()
        fields(obj, "version", "metadata")
        version = whole(obj.get("version"), "Capture version", 1, 1_000_000)
        return {"capture": library.edit_capture(identifier(cid), version, CaptureSpec.parse(obj.get("metadata")))}

    @app.get("/api/captures/<cid>/image/<kind>")
    def capture_image(cid, kind):
        record, (raw, mime) = library.capture_asset(identifier(cid))
        if kind == "original":
            return resource(raw, mime, record["filename"], True)
        if kind != "preview":
            raise Problem("Unknown image view.", 404)
        with work.enter():
            tinted = decode(raw, record["filename"]).preview()
        return resource(tinted, "image/png", "camera-preview.png")

    @app.get("/api/captures/<cid>/registrations")
    def capture_registrations(cid):
        return {"items": library.registrations(identifier(cid))}


def register_registration(app, library, work):
    @app.post("/api/captures/<cid>/registration")
    def register_photo(cid):
        obj = data()
        fields(obj, "corners", "note", "save")
        store = flag(obj.get("save", False), "Retain this registration")
        spec = CropSpec.parse({k: v for k, v in obj.items() if k != "save"})
        cid = identifier(cid)
        photo, (image, _) = library.capture_asset(cid)
        parent = library.get_surface(photo["surface_id"])["surface"]
        with work.enter():
            result, metrics = _registration_trial((photo, image), spec, parent)
            if store:
                revision = library.add_registration(cid, spec, metrics)
                return {"registration": revision, **result.public({"capture": photo["id"]})}, 201
        return result.public({"capture": photo["id"], "retained": False})

    @app.post("/api/captures/<cid>/registration-readout")
    def coordinates(cid):
        obj = data()
        fields(obj, "registration", "points", "space")
        space = choice(obj.get("space", "surface"), "Readout space", ("camera", "surface"))
        points = obj.get("points")
        if not isinstance(points, list) or not 1 <= len(points) <= 80:
            raise Problem("Choose one to eighty coordinate pairs.")
        values = [Point.parse(point, "Readout point").public() for point in points]
        cid = identifier(cid)
        photo, _ = library.capture_asset(cid)
        sid = photo["surface_id"]
        registrations = {cid: identifier(obj["registration"])} if obj.get("registration") else {}
        physical, fetch = library.acquire(sid, [cid], registrations)
        resolved = fetch[0][0]["registration"]
        current = Registration.from_data(resolved)
        camera = np.array(values) if space == "camera" else current.camera(values)
        surface = np.array(values) if space == "surface" else current.surface(values)
        width, height = photo["image"]["size"]
        positions = [{"camera_normalized": np.round(view, 7).tolist(),
                      "camera_px": np.round(view * [width - 1, height - 1], 4).tolist(),
                      "surface_normalized": np.round(pos, 7).tolist(),
                      "surface_mm": np.round(pos * [physical["width_mm"], physical["height_mm"]], 5).tolist(),
                      "in_camera": bool((view >= 0).all() and (view <= 1).all())}
                     for view, pos in zip(camera, surface, strict=True)]
        breaks = int(np.count_nonzero(~np.isfinite(camera).all(axis=1) | ~np.isfinite(surface).all(axis=1)))
        if breaks:
            raise Problem("One or more points cross the projective horizon on this view.")
        return {"registration": current.id, "registration_revision": resolved["revision"] if resolved else None,
                "positions": positions, "units": {"camera_px": "pixel", "surface_mm": "mm"},
                "surface_id": sid}



def register_transfer(app, library, work):
    @app.get("/api/surfaces/<sid>/capture-set")
    def export_sources(sid):
        with work.enter():
            name, raw = export_capture_set(library, identifier(sid))
        return resource(raw, "application/zip", name, True)

    @app.post("/api/capture-sets")
    def import_sources():
        _, blob = upload("packet", PACKET_LIMIT)
        applied = request.form.get("apply", "0") == "1"
        with work.enter():
            prepared = inspect_capture_set(blob)
            if not applied:
                return prepared.preview()
            saved = import_capture_set(library, prepared, request.form.get("preview_id"), request.form.get("title"))
        return saved, 201


def register_procedures(app, library):
    @app.get("/api/surfaces/<sid>/procedures")
    def get_procedures(sid):
        return {"procedures": library.procedures(identifier(sid))}

    @app.post("/api/surfaces/<sid>/procedures")
    def persist_procedure(sid):
        raw = data()
        fields(raw, "version", "procedure")
        v = whole(raw.get("version"), "Surface version", 1, 1_000_000)
        result = save_procedure(library, identifier(sid), mapping(raw.get("procedure")), v)
        return {"procedure": result}, 201

    @app.get("/api/surfaces/<sid>/procedures/<pid>")
    def review_procedure(sid, pid):
        sid = identifier(sid)
        record = library.get_procedure(sid, identifier(pid))
        return {"procedure": record, "preview": preview_procedure(record, library.get_surface(sid))}


def register_results(app, library, work):
    @app.post("/api/surfaces/<sid>/analysis/<kind>")
    def analyze(sid, kind):
        obj = data()
        retained = flag(obj.get("retain", False), "Retain result")
        expected = obj.pop("expected_preview", None)
        sid = identifier(sid)
        with work.enter():
            result, evidence = run(library, sid, identifier(kind, "Tool"), obj)
            signature = seal(result, evidence)
            evidence["data_signature"] = public_signature(signature)
            if retained:
                if expected is not None:
                    require_match(expected, signature)
                title = text(obj.get("title", result.title), "Record title")
                saved = library.save_inspection(sid, title, result, evidence)
                return {"record": saved, **result.public(evidence)}, 201
        return result.public(evidence)

    @app.get("/api/records/<rid>")
    def read_inspection(rid):
        item = library.get_inspection(identifier(rid))
        for part in item["files"]:
            part["url"] = f'/api/records/{rid}/files/{part["key"]}'
        return {"record": item}

    @app.put("/api/records/<rid>/review")
    def review_decision(rid):
        obj = data()
        fields(obj, "revision", "reviewer", "state", "note")
        version = whole(obj.get("revision"), "Record revision", 1, 1_000_000)
        who = text(obj.get("reviewer"), "Reviewer name", 80)
        decision = choice(obj.get("state"), "Review outcome", ("draft", "accepted", "revisit"))
        note = text(obj.get("note", ""), "Review note", 500, False)
        updated = library.mark_inspection(identifier(rid), version, who, decision, note)
        return {"record": updated}

    @app.get("/api/records/<rid>/files/<name>")
    def record_media(rid, name):
        media, mime, actual = library.inspection_asset(identifier(rid), identifier(name))
        download = request.args.get("download") == "1" or mime != "image/png"
        return resource(media, mime, actual, download)

    @app.get("/api/records/<rid>/packet")
    def export_packet(rid):
        include_sources = request.args.get("originals", "1") == "1"
        with work.enter():
            name, content = build_packet(library, identifier(rid), include_sources)
        return resource(content, "application/zip", name, True)

    @app.get("/api/records/<rid>/print")
    def print_inspection(rid):
        paper = request.args.get("paper", "a4")
        orientation = request.args.get("orientation", "portrait")
        selected = [identifier(part, "Picture") for part in request.args.getlist("picture")]
        with work.enter():
            filename, content = report_pdf(library, identifier(rid), paper, orientation, selected)
        return resource(content, "application/pdf", filename, True)

    @app.post("/api/packets/check")
    def packet_readout():
        _, payload = upload("packet", PACKET_LIMIT)
        with work.enter():
            return verify_packet(payload)


def create_app(data_dir=None):
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = PACKET_LIMIT + 1024 * 1024
    app.config["MAX_FORM_MEMORY_SIZE"] = 240 * 1024
    app.config["MAX_FORM_PARTS"] = 48
    library = Library(data_dir or os.getenv("RELIEFBENCH_DATA", "var"))
    work = WorkerBudget(1)
    register_library(app, library, work)
    register_registration(app, library, work)
    register_results(app, library, work)
    register_transfer(app, library, work)
    register_procedures(app, library)

    @app.get("/")
    def homepage():
        return send_from_directory(STATIC, "index.html")

    @app.get("/static/<path:name>")
    def static(name):
        return send_from_directory(STATIC, name)

    @app.errorhandler(Problem)
    def domain_problem(exc):
        return jsonify(error=str(exc)), exc.status

    @app.errorhandler(RequestEntityTooLarge)
    def oversized(_exc):
        return jsonify(error="The browser upload exceeds the request limits."), 413

    return app
