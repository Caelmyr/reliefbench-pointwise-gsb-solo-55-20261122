"""Explain one surface site: which photographs vote, which are set aside and why."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

from .lightmodel import (AGREEMENT_FLOOR, AGREEMENT_TOLERANCE, CUTOFF_SIGNAL, CONDITION_LIMIT,
                         STATUS_BELOW, STATUS_CLIPPED, STATUS_OUTSIDE, STATUS_USED,
                         _fit_block, _fit_guarded_block)
from .models import CaptureSpec, Problem, Region, Scale, rounded
from .raster import decode, intensity, sample_camera, srgb_to_linear
from .registration import Registration
from .results import Result
from .visual import GOLD, HOT, MINT, PAPER, contact_sheet, flatten_rgb, overlay

PATCH_HALF = 78


@dataclass(frozen=True)
class CameraView:
    """The frozen source readings a site decision is recomputed from."""
    title: str
    capture_id: str
    spec: CaptureSpec
    raw: object
    registration: object

    @property
    def light(self):
        return self.spec.direction()


def _cell(shape, point):
    height, width = shape
    return (min(height - 1, max(0, int(point.y * height))),
            min(width - 1, max(0, int(point.x * width))))


def _readings(views, shape, row, column):
    u, v = (column + .5) / shape[1], (row + .5) / shape[0]
    rows = []
    for view in views:
        camera = view.registration.camera([[u, v]])[0]
        colour, inside = sample_camera(view.raw, camera[None, None, :])
        colour, inside = colour[0, 0], bool(inside[0, 0])
        black = srgb_to_linear(np.float32(view.spec.black_code))
        linear = np.maximum(srgb_to_linear(colour) - black, 0.) / (2 ** view.spec.exposure * view.spec.lamp_power)
        signal = float(intensity(linear))
        rows.append({"view": view, "camera": [float(camera[0]), float(camera[1])], "inside": inside,
                     "colour": colour, "signal": signal,
                     "clipped": bool(inside and (colour >= .992).any())})
    return rows


def _classify(item):
    if not item["inside"]:
        return STATUS_OUTSIDE
    if item["clipped"]:
        return STATUS_CLIPPED
    if item["signal"] < CUTOFF_SIGNAL:
        return STATUS_BELOW
    return STATUS_USED


def _fit_site(views, readings, candidates, mode):
    """Run the same fit the full solve uses, restricted to this site."""
    matrix = np.asarray([views[i].light for i in candidates], dtype=np.float64)
    values = np.asarray([readings[i]["signal"] for i in candidates], dtype=np.float64).reshape(-1, 1)
    colour = np.zeros((len(candidates), 1, 3), dtype=np.float32)
    weights = np.zeros(len(readings))
    agreeing = {}
    if mode == "guarded":
        normal, reflectance, _rgb, _error, good, kept, agreed, _rel, _model = _fit_guarded_block(
            matrix, values, colour, np.ones((len(candidates), 1)))
        for local, index in enumerate(candidates):
            weights[index] = kept[local, 0]
            agreeing[index] = bool(agreed[local, 0] > 0)
        return normal[0], bool(good[0]), float(reflectance[0]), weights, agreeing
    normal, metrics, _rgb, _error, good = _fit_block(matrix, values, colour)
    for index in candidates:
        weights[index] = 1.
        agreeing[index] = True
    return normal[0], bool(good[0]), float(metrics[0]), weights, agreeing


def _patch(item, index):
    view = item["view"]
    tile = Image.new("RGB", (2 * PATCH_HALF, 2 * PATCH_HALF), PAPER)
    if not item["inside"]:
        ImageDraw.Draw(tile).text((10, 10), "Outside this photograph", fill=HOT)
        return f"{index}. {view.title} · outside", tile
    height, width = view.raw.shape
    px, py = item["pixel"]
    left, top = max(0, px - PATCH_HALF), max(0, py - PATCH_HALF)
    right, bottom = min(width, px + PATCH_HALF), min(height, py + PATCH_HALF)
    rgb = view.raw.rgb[top:bottom, left:right]
    alpha = view.raw.alpha[top:bottom, left:right]
    tile = flatten_rgb(rgb, alpha >= .98, False)
    mark = [((px - left) / max(1, tile.width - 1), (py - top) / max(1, tile.height - 1))]
    tile = overlay(tile, [("point", mark, MINT if item["used"] else GOLD, None)])
    return f"{index}. {view.title} · {item['status_text']}", tile


def explain_site(views, shape, scale, selection, point, mode="ordinary"):
    """Account for one site from frozen source bytes: normal, votes and reasons."""
    row, column = _cell(shape, point)
    inside_region = bool(selection[row, column])
    readings = _readings(views, shape, row, column)
    for item in readings:
        item["status"] = _classify(item)
        item["pixel"] = ([int(round(item["camera"][0] * (item["view"].raw.shape[1] - 1))),
                          int(round(item["camera"][1] * (item["view"].raw.shape[0] - 1)))]
                         if item["inside"] else None)
    candidates = [i for i, item in enumerate(readings) if item["status"] == STATUS_USED]
    normal = np.array([0., 0., 1.])
    reflectance = 0.
    good = False
    weights = np.zeros(len(readings))
    agreeing = {i: False for i in range(len(readings))}
    if len(candidates) >= 3 and inside_region:
        normal, good, reflectance, weights, agreeing = _fit_site(views, readings, candidates, mode)
    for i, item in enumerate(readings):
        predicted = reflectance * float(np.maximum(np.dot(normal, item["view"].light), 0.)) if good else None
        tolerance = None if predicted is None else AGREEMENT_FLOOR + AGREEMENT_TOLERANCE * predicted
        residual = None if predicted is None else abs(item["signal"] - predicted)
        item.update(predicted=predicted, tolerance=tolerance, residual=residual,
                    agrees=bool(residual is not None and residual <= tolerance),
                    used=bool(weights[i] > 0 and good),
                    excluded=bool(weights[i] == 0 and item["status"] == STATUS_USED and good))
        item["status_text"] = _status_text(mode, item, good, inside_region)
    used_count = sum(1 for item in readings if item["used"])
    agreed_count = sum(1 for value in agreeing.values() if value)
    state = _state_text(mode, good, len(candidates), agreed_count, inside_region)
    summary = {
        "mode": mode, "site_row": int(row), "site_column": int(column),
        "surface_normalized": [rounded((column + .5) / shape[1], 7), rounded((row + .5) / shape[0], 7)],
        "surface_mm": [rounded((column + .5) / shape[1] * scale.width_mm),
                       rounded((row + .5) / shape[0] * scale.height_mm)],
        "inside_selected_region": inside_region, "state": state, "pending": state != "retained",
        "normal": [rounded(float(v)) for v in normal],
        "reflectance": rounded(reflectance) if good else None,
        "candidate_photos": len(candidates), "used_photos": used_count,
        "agreeing_photos": sum(agreeing.values()) if good else 0,
        "excluded_photos": [item["view"].title for item in readings if item["excluded"]],
    }
    rows = [{
        "photo": item["view"].title, "capture_id": item["view"].capture_id,
        "azimuth": rounded(item["view"].spec.azimuth, 3), "elevation": rounded(item["view"].spec.elevation, 3),
        "camera_normalized": [rounded(v, 7) for v in item["camera"]], "camera_pixel": item["pixel"],
        "participation": item["status_text"], "used": item["used"], "agrees": item["agrees"],
        "linear_signal": rounded(item["signal"]),
        "predicted_linear": rounded(item["predicted"]) if item["predicted"] is not None else None,
        "residual": rounded(item["residual"]) if item["residual"] is not None else None,
        "tolerance": rounded(item["tolerance"]) if item["tolerance"] is not None else None,
    } for item in readings]
    result = Result("normal-site", "Site decision review",
                    "Locate the chosen surface site in every source photograph and show why each was kept or set aside.",
                    summary, rows)
    tiles = [_patch(item, i) for i, item in enumerate(readings, 1)]
    result.image("camera-sites", contact_sheet(tiles, columns=min(3, len(tiles))))
    result.warnings.append("This recomputes the saved per-site decision from the same source bytes, light settings and registrations. Photographs changed after saving are not used.")
    return result


def _status_text(mode, item, good, inside_region):
    if not item["inside"]:
        return "outside this photograph"
    if item["clipped"]:
        return "clipped in this photograph"
    if item["signal"] < CUTOFF_SIGNAL:
        return "below the minimum signal"
    if not inside_region:
        return "usable, but outside the selected region"
    if not good:
        return "candidate; site is pending"
    if mode == "guarded" and not item["used"]:
        return "set aside: disagrees with the agreeing photos"
    return "agrees and was used" if mode == "guarded" else "usable sample"


def _state_text(mode, good, candidates, agreeing, inside_region):
    if not inside_region:
        return "outside the selected region"
    if candidates < 3:
        return "pending: fewer than three usable photographs"
    if good:
        return "retained"
    if mode == "guarded" and agreeing < 3:
        return "pending: too few photographs agree at this site"
    return "pending: the agreeing light directions do not separate the normal"


def _evidence_scale(evidence):
    size = evidence["scale"]
    return Scale(size["size_mm"][0], size["size_mm"][1], size["size_px"][0], size["size_px"][1])


def _evidence_views(library, evidence):
    """Reopen the exact source bytes and recorded light settings a saved inspection used."""
    current = {item["id"]: item for item in library.get_surface(evidence["surface"]["id"])["captures"]}
    for source in evidence["sources"]:
        record = current.get(source["id"])
        if record is None:
            raise Problem("A saved source photograph is no longer in this surface.", 409)
        if record["hash"] != source["hash"]:
            raise Problem("A saved source photograph has changed since this inspection was retained.", 409)
        data, _mime = library.read_asset(record["hash"])
        raw = decode(data, record["filename"])
        registration = None
        if source.get("registration_id") not in (None, "identity"):
            match = next((entry for entry in record["registrations"] if entry["id"] == source["registration_id"]), None)
            if match is None:
                raise Problem("A registration this inspection used is no longer stored.", 409)
            registration = Registration.from_data(match)
        settings = CaptureSpec.parse(source["settings"])
        yield CameraView(settings.title, source["id"], settings, raw, registration or Registration.from_data(None))


def saved_site_review(library, evidence, point):
    """Recompute one site's saved decision from the frozen source bytes."""
    scale = _evidence_scale(evidence)
    shape = (scale.height, scale.width)
    region = Region.parse(evidence["parameters"]["region"])
    selection = np.zeros(shape, dtype=bool)
    selection[region.window(scale.width, scale.height)] = True
    mode = evidence["parameters"]["options"].get("mode", "ordinary")
    review = explain_site(list(_evidence_views(library, evidence)), shape, scale, selection, point, mode=mode)
    review.report["summary"] = ("Recomputed from the same source bytes, light settings and registrations this "
                                "inspection used. " + review.report["summary"])
    review.metrics["selected_region"] = region.public()
    review.metrics["inspection_mode"] = mode
    return review


def review_current_site(dataset, point, mode="guarded"):
    """Explain one site from the photographs currently on the workbench."""
    views = [CameraView(frame.title, frame.id, frame.spec, frame.raw, frame.registration)
             for frame in dataset.frames]
    review = explain_site(views, (dataset.scale.height, dataset.scale.width), dataset.scale,
                          dataset.selection, point, mode=mode)
    review.metrics["selected_region"] = dataset.region.public()
    review.metrics["inspection_mode"] = mode
    return review
