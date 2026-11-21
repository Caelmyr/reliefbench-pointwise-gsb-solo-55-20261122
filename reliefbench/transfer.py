"""Transport complete capture sets with their registration histories."""
from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import BadZipFile, ZIP_DEFLATED, ZipFile
import json
import re

from .archive import _safe_name, _zip_write
from .models import CaptureSpec, CropSpec, Problem, SurfaceSpec, filename, text
from .raster import RawCapture, decode
from .registration import Registration


CAPTURE_FORMAT = "reliefbench.capture-set.v1"
CAPTURE_CONTROL = "capture-manifest.json"
MAX_CAPTURE_SET = 65 * 1024 * 1024
MAX_FILES = 49
CONTROL_CAPACITY = 960_000
ITEM_LIMIT = 18 * 1024 * 1024
TOKEN = re.compile("[0-9a-f]{32}")
HASH = re.compile("[0-9a-f]{64}")


def _checksum(raw):
    return sha256(raw).hexdigest()


def _public_spec(known):
    return {k: known[k] for k in ("title", "note", "width_mm", "height_mm", "accession", "synthetic")}


def _export_rows(library, acquisitions):
    rows = []
    for num, known in enumerate(acquisitions, 1):
        data, mime = library.read_asset(known["hash"])
        filename_key = filename(known["filename"]).replace(" ", "-")
        relative = f"photos/{num:02d}-{filename_key}"
        records = [{"corners": item["corners"], "note": item["note"], "revision": item["revision"]}
                   for item in reversed(known["registrations"])]
        row = {"source_id": known["id"], "name": known["filename"], "metadata": known["metadata"],
               "source_version": known["version"], "sha256": _checksum(data),
               "bytes": len(data), "mime": mime, "path": relative, "registrations": records}
        rows.append((row, data))
    return rows


def export_capture_set(library, surface_id):
    known = library.get_surface(surface_id)
    rows = _export_rows(library, known["captures"])
    if not rows:
        raise Problem("There are no source photographs in this surface.")
    if sum(row["bytes"] for row, _ in rows) > MAX_CAPTURE_SET:
        raise Problem("This capture set exceeds the one-package size limit.", 413)
    manifest = {"format": CAPTURE_FORMAT, "surface": _public_spec(known["surface"]),
                "source_surface": known["surface"]["id"], "captures": [row for row, _ in rows]}
    raw = json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8")
    if len(raw) > CONTROL_CAPACITY:
        raise Problem("The capture manifest exceeds the read limit.", 413)
    memory = BytesIO()
    with ZipFile(memory, "w", compression=ZIP_DEFLATED, compresslevel=3) as arc:
        _zip_write(arc, CAPTURE_CONTROL, raw)
        for row, content in rows:
            _zip_write(arc, row["path"], content)
    return f"captures-{surface_id[:10]}.zip", memory.getvalue()


def _inside_bundle(key):
    if not _safe_name(key):
        return False
    parts = PurePosixPath(key).parts
    return len(parts) == 2 and parts[0] == "photos"


def _read_control(arc, registration):
    if CAPTURE_CONTROL not in registration or registration[CAPTURE_CONTROL].file_size > CONTROL_CAPACITY:
        raise Problem("The ZIP does not contain a supported capture manifest.")
    try:
        doc = json.loads(arc.read(CAPTURE_CONTROL))
    except (UnicodeError, ValueError, RuntimeError) as exc:
        raise Problem("The capture manifest could not be decoded.") from exc
    if not isinstance(doc, dict) or doc.get("format") != CAPTURE_FORMAT:
        raise Problem("Choose a ReliefBench capture-set ZIP.")
    if not isinstance(doc.get("captures"), list) or not 1 <= len(doc["captures"]) <= MAX_FILES - 1:
        raise Problem("The capture manifest contains an unsupported number of photos.")
    source = doc.get("source_surface")
    if not isinstance(source, str) or TOKEN.fullmatch(source) is None:
        raise Problem("The capture set cannot identify its original surface.")
    return doc


@dataclass(frozen=True)
class PreparedPhoto:
    name: str
    source_id: str
    digest: str
    raw: bytes
    metadata: CaptureSpec
    image: RawCapture
    registrations: tuple


@dataclass(frozen=True)
class PreparedSet:
    inspection_ref: str
    original: str
    surface: SurfaceSpec
    photos: tuple[PreparedPhoto, ...]

    def preview(self):
        return {"status": "ready_to_import", "original_surface": self.original, "preview_id": self.inspection_ref,
                "surface": self.surface.public(), "capture_count": len(self.photos), "photos": [
                    {"name": i.name, "title": i.metadata.title, "role": i.metadata.role,
                     "source_id": i.source_id, "sha256": i.digest, "image": i.image.info,
                     "registrations": len(i.registrations), "azimuth": i.metadata.azimuth,
                     "elevation": i.metadata.elevation}
                    for i in self.photos]}


def _shape_registration(raw, captured):
    if not isinstance(raw, list) or len(raw) > 36:
        raise Problem("The package has an unsupported registration history.")
    results = []
    for idx, obj in enumerate(raw, 1):
        if not isinstance(obj, dict) or obj.get("revision") != idx:
            raise Problem("Registration revisions in the capture packet must be consecutive.")
        spec = CropSpec.parse({"corners": obj.get("corners"), "note": obj.get("note", "")})
        homography = Registration.from_data({"id": f"trial-{idx}", **spec.public()})
        geometry = homography.overview(captured.shape)
        if geometry["geometrically_observed_fraction"] < .10:
            raise Problem("A photo registration in this capture set falls outside the photograph.")
        results.append((spec, geometry))
    return tuple(results)


def _capture_row(archive, info, description):
    if not isinstance(description, dict):
        raise Problem("The capture manifest has an invalid photo description.")
    name = description.get("name")
    if not isinstance(name, str) or name != filename(name):
        raise Problem("The capture manifest has an invalid photo filename.")
    root_path = description.get("path")
    if not _inside_bundle(root_path) or root_path not in info:
        raise Problem("A source image is missing or names an unsupported path.")
    source_id = description.get("source_id")
    if not isinstance(source_id, str) or TOKEN.fullmatch(source_id) is None:
        raise Problem("A packaged photo has no valid original identity.")
    digest = description.get("sha256")
    if not isinstance(digest, str) or not HASH.fullmatch(digest):
        raise Problem("A packaged photo has no valid content fingerprint.")
    declared = description.get("bytes")
    entry = info[root_path]
    if type(declared) is not int or not 1 <= declared <= ITEM_LIMIT or declared != entry.file_size:
        raise Problem("A photo size does not agree with the capture manifest.")
    try:
        with archive.open(entry) as fp:
            payload = fp.read(ITEM_LIMIT + 1)
    except (BadZipFile, RuntimeError, NotImplementedError, OSError) as error:
        raise Problem("A packaged photo could not be safely decoded.") from error
    if len(payload) != declared or _checksum(payload) != digest:
        raise Problem("A capture image fingerprint did not match.")
    capture = decode(payload, name)
    if capture.mime != description.get("mime"):
        raise Problem("An archive photo type does not match its decoded image.")
    spec = CaptureSpec.parse(description.get("metadata"))
    registrations = _shape_registration(description.get("registrations"), capture)
    return PreparedPhoto(name, source_id, digest, payload, spec, capture, registrations)


def _safe_entries(archive):
    info = archive.infolist()
    if len(info) > MAX_FILES or sum(e.file_size for e in info) > MAX_CAPTURE_SET + CONTROL_CAPACITY:
        raise Problem("The expanded capture ZIP is too large.", 413)
    paths = Counter(e.filename for e in info)
    if any(value != 1 for value in paths.values()):
        raise Problem("The archive names an image more than once.")
    for key in paths:
        if key != CAPTURE_CONTROL and not _inside_bundle(key):
            raise Problem("Only the manifest and its photo entries belong in the capture archive.")
    return {e.filename: e for e in info}


def inspect_capture_set(raw):
    if len(raw) > MAX_CAPTURE_SET:
        raise Problem("The capture set is too large.", 413)
    try:
        archive = ZipFile(BytesIO(raw))
    except (BadZipFile, OSError) as error:
        raise Problem("The photograph set is not a complete ZIP.") from error
    with archive:
        available = _safe_entries(archive)
        bundle = _read_control(archive, available)
        indicated = {row.get("path") for row in bundle["captures"] if isinstance(row, dict)
                     and isinstance(row.get("path"), str)}
        if indicated != set(available) - {CAPTURE_CONTROL} or len(indicated) != len(bundle["captures"]):
            raise Problem("The capture ZIP and the manifest list different source images.")
        photos = []
        frame_pixels = 0
        for row in bundle["captures"]:
            photo = _capture_row(archive, available, row)
            frame_pixels += photo.image.shape[0] * photo.image.shape[1]
            if frame_pixels > 30_000_000:
                raise Problem("The expanded capture set has too many camera pixels.", 413)
            photos.append(photo)
        if len({row.digest for row in photos}) != len(photos) or len({row.source_id for row in photos}) != len(photos):
            raise Problem("A declared capture is repeated with the same bytes or identity.")
        return PreparedSet(_checksum(raw), bundle["source_surface"], SurfaceSpec.parse(bundle.get("surface")), tuple(photos))


def import_capture_set(library, document, expected, title=None):
    if not isinstance(expected, str) or not HASH.fullmatch(expected) or document.inspection_ref != expected:
        raise Problem("The packet is different from the reviewed copy. Check it again before importing.", 409)
    original = document.surface.public()
    if title:
        original["title"] = text(title, "Imported surface title", 120)
    spec = SurfaceSpec.parse(original)
    return library.import_capture_set(spec, document)
