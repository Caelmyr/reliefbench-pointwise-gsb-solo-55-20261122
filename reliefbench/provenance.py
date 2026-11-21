"""Bind an inspection preview to its selected raster inputs and display outputs."""
from hashlib import sha256
import json

from .models import Problem


def fingerprint(value):
    return sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("utf-8")).hexdigest()


def _asset_occurrences(result):
    return [{"key": key, "name": item.name, "mime": item.mime,
             "sha256": sha256(item.content).hexdigest(), "size": len(item.content)}
            for key, item in sorted(result.assets.items())]


def _source_revision_data(sources):
    return [{"key": photo["id"], "source_hash": photo["hash"],
             "registrations": photo["registration_id"], "camera_settings": photo["settings"],
             "capture_version": photo["capture_version"]}
            for photo in sources]


def seal(result, evidence):
    images = _asset_occurrences(result)
    revisions = _source_revision_data(evidence["sources"])
    frame = evidence["surface"]
    frozen = {"format": "reliefbench.inspection-preview.v1",
              "originals": revisions, "surface_version": frame["version"],
              "surface_id": frame["id"], "dimensions_mm": [frame["width_mm"], frame["height_mm"]],
              "parameters": evidence["parameters"], "working_scale": evidence["scale"],
              "measurements_sha256": fingerprint(result.report), "report_files": images}
    return {"preview": fingerprint(frozen), **frozen}


def require_match(expected, candidate):
    if not isinstance(expected, str) or len(expected) != 64:
        raise Problem("Reopen the current inspection preview before retaining it.", 409)
    equal = expected == candidate["preview"]
    if not equal:
        raise Problem("The source settings or image result changed after preview. Run it again before retaining.", 409)


def public_signature(signature):
    return {"identity": signature["preview"], "measurements_sha256": signature["measurements_sha256"],
            "pictures": len(signature["report_files"]), "source_materials": len(signature["originals"])}
