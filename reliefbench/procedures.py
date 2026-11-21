"""Named repeatable inspections that account for changed source settings."""
from dataclasses import dataclass
from hashlib import sha256
import json

from .models import JobRequest, Problem, choice, fields, identifier, text, whole
from .workflow import TOOLS


TRACKED_CAPTURE = ("filename", "version", "hash", "metadata", "image")
TRACKED_SURFACE = ("accession", "width_mm", "height_mm", "version")
MAX_PROCEDURES = 28


def _data_hash(doc):
    raw = json.dumps(doc, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return sha256(raw).hexdigest()


def _capture_snapshot(frame, registration):
    return {**{key: frame[key] for key in TRACKED_CAPTURE}, "registration": {
        key: registration[key] for key in ("id", "revision", "corners")
    } if registration else None}


def _surface_snapshot(surface):
    return {key: surface[key] for key in TRACKED_SURFACE}


def _find_registration(frame, specified, strict=True):
    if specified == "identity":
        return None
    if not specified:
        return next(iter(frame["registrations"]), None)
    item = next((entry for entry in frame["registrations"] if specified == entry["id"]), None)
    if not item and strict:
        raise Problem("An inspection procedure names a registration missing from its own source.", 409)
    return item


def _instance(tool, data):
    if tool not in TOOLS:
        raise Problem("This procedure uses an unavailable inspection tool.")
    if not isinstance(data, dict):
        raise Problem("Procedure needs reusable analysis inputs.")
    fields(data, "captures", "corrections", "region", "options")
    request = JobRequest.parse(tool, data)
    cfg = TOOLS[tool]
    if not cfg.minimum <= len(request.captures) <= cfg.maximum:
        raise Problem("Procedure source count does not match the tool.")
    requested_fields = set(request.options)
    allowed = {spec["key"] for spec in cfg.controls}
    if requested_fields - allowed:
        raise Problem("The saved tool procedure contains unknown controls.")
    return request


@dataclass(frozen=True)
class Procedure:
    name: str
    kind: str
    note: str
    options: dict
    policy: str

    @classmethod
    def parse(cls, raw):
        fields(raw, "name", "kind", "note", "inputs", "policy")
        name = text(raw.get("name"), "Procedure name", 96)
        note = text(raw.get("note", ""), "Procedure note", 430, False)
        kind = identifier(raw.get("kind"), "Procedure tool")
        request = _instance(kind, raw.get("inputs"))
        policy = choice(raw.get("policy", "recorded"), "Registration policy", ("recorded", "latest"))
        options = {"captures": list(request.captures), "corrections": request.corrections,
                   "region": request.region.public(), "options": request.options}
        return cls(name, kind, note, options, policy)

    def bind(self, surface):
        def verify(frame):
            if TOOLS[self.kind].input_kind == "raking" and frame["metadata"]["role"] != "raking":
                raise Problem("This procedure expects all selected photos to be directional.")

        current = {cap["id"]: cap for cap in surface["captures"]}
        available = {}
        revised = {}
        for cid in self.options["captures"]:
            frame = current.get(cid)
            if not frame:
                raise Problem("A selected source is absent from this surface.")
            verify(frame)
            crop = _find_registration(frame, self.options["corrections"].get(cid))
            available[cid] = _capture_snapshot(frame, crop)
            revised[cid] = crop["id"] if crop else "identity"
        settings = {**self.options, "corrections": revised}
        return {"name": self.name, "kind": self.kind, "note": self.note, "policy": self.policy,
                "inputs": settings, "sources": available, "surface": _surface_snapshot(surface["surface"]),
                "source_kind": TOOLS[self.kind].input_kind}


def _compare_metadata(expected, current):
    old = expected["metadata"]
    now = current["metadata"]
    changed = []
    for name in ("role", "azimuth", "elevation", "lamp_power", "exposure", "black_code", "camera"):
        if old.get(name) != now.get(name):
            changed.append({"field": name, "recorded": old.get(name), "current": now.get(name)})
    return changed


def _compare_source(frame, target, saved_registration, policy):
    absent = frame is None
    if absent:
        return {"status": "missing-source", "issues": ["Original source was not found."]}, None
    registration = (_find_registration(frame, None, strict=False) if policy == "latest"
                    else _find_registration(frame, saved_registration, strict=False))
    state = []
    if frame["hash"] != target["hash"]:
        state.append("Source bytes differ.")
    if registration is None and saved_registration not in ("identity", None):
        state.append("Recorded source registration is missing.")
    cid = registration["id"] if registration else "identity"
    if cid != saved_registration and policy == "latest":
        state.append("Latest registration differs from saved one.")
    settings = _compare_metadata(target, frame)
    if settings:
        state.append("Photo settings affecting computation changed.")
    return ({"status": "changed" if state else "unchanged", "file": frame["filename"], "issues": state,
             "metadata_changes": settings, "chosen_registration": cid, "hash": frame["hash"]},
            cid)


def preview_procedure(document, workspace):
    inputs = document["inputs"].copy()
    surface = workspace["surface"]
    current = _surface_snapshot(surface)
    last = document["surface"]
    changes = [{"field": field, "recorded": last[field], "current": current[field]}
               for field in ("width_mm", "height_mm") if last[field] != current[field]]
    available = {source["id"]: source for source in workspace["captures"]}
    corrections = {}
    items = []
    for cid in inputs["captures"]:
        item, chosen = _compare_source(available.get(cid), document["sources"][cid], inputs["corrections"][cid],
                                       document["policy"])
        items.append({"capture": cid, **item})
        if chosen is not None:
            corrections[cid] = chosen
    inputs["corrections"] = corrections
    possible = (len(corrections) == len(items) and
                all(document["source_kind"] != "raking" or available[c]["metadata"]["role"] == "raking"
                    for c in corrections))
    snapshot = {"procedure": document["id"], "kind": document["kind"], "inputs": inputs,
                "photos": items, "changed_surface": changes,
                "status": "ready" if possible else "unavailable",
                "needs_reconsent": bool(changes or any(item["status"] != "unchanged" for item in items))}
    return {**snapshot, "material_sha256": _data_hash(snapshot)}


def save_procedure(library, surface_id, raw, threshold):
    plan = Procedure.parse(raw)
    workspace = library.get_surface(surface_id)
    with library.connect(writing=True) as conn:
        source = library._surface(conn, surface_id)
        if source["version"] != threshold or workspace["surface"]["version"] != threshold:
            raise Problem("Surface dimensions were changed while preparing this procedure.", 409)
        bound = plan.bind(workspace)
        if len(library._procedures(conn, surface_id)) >= MAX_PROCEDURES:
            raise Problem("Keep at most 28 inspection procedures per surface.", 413)
        return library._add_procedure(conn, surface_id, bound)
