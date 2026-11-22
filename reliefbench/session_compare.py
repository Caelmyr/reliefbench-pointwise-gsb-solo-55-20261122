"""Compare two retained normal/reflectance reports on one physical rectangle."""
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .color_reference import delta_lab
from .dataset import MAX_WORKSIDE, DataSet, make_frame
from .lightmodel import solve
from .models import Problem, Region, Scale, fields, identifier, list_of, number, rounded, whole
from .raster import decode, intensity, normalized_grid, safe_fraction, statistics, tiff_float
from .results import Result
from .visual import (COOL, FONT, GAP, GOLD, HOT, INK, MINT, PAPER, THERMAL, Extent, flatten_rgb, heatmap,
                     overlay, scale_bar, split_visual)


COMPARE_SIDE = 560
MAX_SESSION_PIXELS = 35_000_000
MIN_REFERENCE_SAMPLES = 48
COMPUTED_SETTINGS = ("role", "azimuth", "elevation", "lamp_power", "exposure", "black_code")
CLASSES = (
    (1, "Colour and contour both changed", HOT),
    (2, "Colour changed, contour did not", GOLD),
    (3, "Contour changed, colour did not", COOL),
    (4, "Held pending: colour and contour moved", MINT),
    (5, "Held pending: colour moved only", MINT),
    (6, "Held pending: contour moved only", MINT),
)
PENDING = (4, 5, 6)
SUPPORTED = (1, 2, 3)


@dataclass(frozen=True)
class Request:
    records: tuple[str, str]
    region: Region | None
    drawn: Region | None
    reference: Region
    colour_limit: float
    normal_limit: float
    support_distance_mm: float
    min_photos: int
    residual_limit: float
    min_area_mm2: float

    @classmethod
    def parse(cls, body):
        fields(body, "captures", "corrections", "region", "options", "title", "retain")
        options = body.get("options")
        if not isinstance(options, dict):
            raise Problem("Inspection settings are required.")
        fields(options, "records", "region", "reference", "colour_limit", "normal_limit",
               "support_distance_mm", "min_photos", "residual_limit", "min_area_mm2")
        ids = tuple(identifier(value, "Retained report")
                    for value in list_of(options.get("records"), "Retained reports", 2, 2))
        if ids[0] == ids[1]:
            raise Problem("Choose two different retained reports.")
        if options.get("reference") in (None, []):
            raise Problem("Select the untouched reference patch as a normalized rectangle.")
        region = None if options.get("region") in (None, []) else Region.parse(options.get("region"))
        drawn = None if body.get("region") in (None, []) else Region.parse(body.get("region"))
        return cls(
            ids, region, drawn, Region.parse(options.get("reference")),
            number(options.get("colour_limit"), "Colour change limit", 1., 40.),
            number(options.get("normal_limit"), "Contour change limit", 1., 60.),
            number(options.get("support_distance_mm"), "Reference support distance", 1., 500.),
            whole(options.get("min_photos"), "Minimum usable photos", 3, 12),
            number(options.get("residual_limit"), "Maximum fit residual", .005, .5),
            number(options.get("min_area_mm2"), "Smallest reported change", .01, 500.),
        )

    def public(self):
        return {"records": list(self.records), "region": self.region.public() if self.region else [],
                "drawn_region": self.drawn.public() if self.drawn else [],
                "reference": self.reference.public(), "colour_limit": self.colour_limit,
                "normal_limit": self.normal_limit, "support_distance_mm": self.support_distance_mm,
                "min_photos": self.min_photos, "residual_limit": self.residual_limit,
                "min_area_mm2": self.min_area_mm2}


@dataclass(frozen=True)
class Epoch:
    record: dict
    field: object
    frames: tuple
    sources: tuple
    photos: int


def _intersection(regions):
    left = max(region.left for region in regions)
    top = max(region.top for region in regions)
    right = min(region.right for region in regions)
    bottom = min(region.bottom for region in regions)
    if right - left < .012 or bottom - top < .012:
        raise Problem("The two reports have no common analysis window.")
    return Region(left, top, right, bottom)


def _saved_record(library, record_id, surface_id):
    record = library.get_inspection(record_id)
    if record["surface_id"] != surface_id:
        raise Problem("Both retained reports must belong to the surface being compared.")
    if record["kind"] != "normals":
        raise Problem("Choose retained normal and reflectance reports for both sessions.")
    evidence = record["evidence"]
    settings = evidence.get("parameters")
    if not isinstance(settings, dict) or not isinstance(settings.get("captures"), list):
        raise Problem("A retained report has no reusable source settings.")
    saved = evidence.get("surface", {})
    current = library.get_surface(surface_id)["surface"]
    for key in ("width_mm", "height_mm"):
        if abs(float(saved.get(key, 0.)) - float(current[key])) > 1e-6:
            raise Problem("The surface size changed after a report was saved. Compare reports on the same rectangle.")
    return record


def _source_sizes(record):
    sizes = []
    for source in record["evidence"]["sources"]:
        size = source.get("size")
        if not (isinstance(size, list) and len(size) == 2 and
                all(isinstance(value, int) and value > 0 for value in size)):
            raise Problem("A retained report does not record its source pixel sizes.")
        sizes.append(tuple(size))
    return sizes


def _check_sources(record, workspace):
    current = {item["id"]: item for item in workspace["captures"]}
    for source in record["evidence"]["sources"]:
        photo = current.get(source.get("id"))
        if photo is None:
            raise Problem("A source photograph of a retained report is no longer in this surface.")
        if photo["hash"] != source.get("hash"):
            raise Problem("A retained report refers to source bytes that are no longer present.")
        saved = source.get("settings", {})
        for name in COMPUTED_SETTINGS:
            if saved.get(name) != photo["metadata"].get(name):
                raise Problem("A photograph setting changed after a report was saved. Run that report again before comparing.")


def _common_scale(surface, records):
    width_mm, height_mm = float(surface["width_mm"]), float(surface["height_mm"])
    rates = [min(width / width_mm, height / height_mm)
             for record in records for width, height in _source_sizes(record)]
    rate = min(rates + [COMPARE_SIDE / max(width_mm, height_mm)])
    width, height = max(28, round(width_mm * rate)), max(28, round(height_mm * rate))
    return _working_scale(Scale(width_mm, height_mm, width, height))


def _working_scale(scale):
    if max(scale.width, scale.height) <= MAX_WORKSIDE:
        return scale
    rate = MAX_WORKSIDE / max(scale.width, scale.height)
    return Scale(scale.width_mm, scale.height_mm,
                 max(28, round(scale.width * rate)), max(28, round(scale.height * rate)))


def _epoch_fields(library, surface_id, record, surface, scale, region):
    sources = record["evidence"]["sources"]
    if not sources:
        raise Problem("A retained report does not name its source photographs.")
    captures = tuple(identifier(source["id"], "Capture") for source in sources)
    corrections = {identifier(source["id"], "Capture"): identifier(source["registration_id"], "Registration")
                   for source in sources}
    _, entries = library.acquire(surface_id, captures, corrections)
    grid = normalized_grid(scale.width, scale.height)
    frames = []
    for row, raw in entries:
        frames.append(make_frame(row, decode(raw, row["filename"]), grid))
    selection = np.zeros((scale.height, scale.width), dtype=bool)
    selection[region.window(scale.width, scale.height)] = True
    dataset = DataSet(surface, tuple(frames), scale, region, selection)
    dataset.lamps()
    field = solve(dataset)
    sources = tuple(frame.present(region) for frame in frames)
    return Epoch(record, field, tuple(frames), sources, len(frames))


def _reference_mask(shape, reference):
    mask = np.zeros(shape, dtype=bool)
    mask[reference.window(shape[1], shape[0])] = True
    return mask


def _reference_distance(scale, reference):
    """Physical distance in millimetres from each working pixel to the reference patch."""
    width_mm, height_mm = scale.width_mm, scale.height_mm
    x = (np.arange(scale.width) + .5) / scale.width * width_mm
    y = (np.arange(scale.height) + .5) / scale.height * height_mm
    left, right = reference.left * width_mm, reference.right * width_mm
    top, bottom = reference.top * height_mm, reference.bottom * height_mm
    dx = np.maximum(np.maximum(left - x, x - right), 0.)
    dy = np.maximum(np.maximum(top - y, y - bottom), 0.)
    return np.hypot(dy[:, None], dx[None, :])


def _minimum_area(mask, pixels):
    if pixels <= 1 or not mask.any():
        return mask
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=bool))
    if not count:
        return mask
    sizes = np.bincount(labels.ravel())
    keep = np.flatnonzero(sizes >= pixels)
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def _base_image(earlier, later, gain, common):
    rgb = .5 * (earlier.field.rgb * gain + later.field.rgb)
    luma = intensity(rgb)
    picks = luma[common & np.isfinite(luma)]
    if picks.size >= 2:
        low, high = np.percentile(picks, [5, 95])
    else:
        low, high = 0., 1.
    shaped = np.clip((luma - low) / max(float(high - low), 1e-6), 0., 1.)
    grey = (30 + 150 * shaped).astype(np.uint8)
    out = np.repeat(grey[..., None], 3, axis=2)
    out[~common] = GAP
    return out


def _paint(base, classes):
    """Colour-class overlay; pending classes are hatched so they read as held, not settled."""
    out = base.copy()
    x = np.arange(out.shape[1])[None, :]
    y = np.arange(out.shape[0])[:, None]
    hatch = ((x + y) // 4) % 2 == 0
    for code, _, colour in CLASSES:
        mask = classes == code
        if not mask.any():
            continue
        tint = np.asarray(colour, dtype=float)
        if code in PENDING:
            out[mask] = (base[mask] * .45 + tint * .55).astype(np.uint8)
            out[mask & hatch] = (base[mask & hatch] * .12 + tint * .88).astype(np.uint8)
        else:
            out[mask] = (base[mask] * .30 + tint * .70).astype(np.uint8)
    return out


def _legend(counts, common_pixels, request, thresholds, reference):
    panel = Image.new("RGB", (318, 452), PAPER)
    draw = ImageDraw.Draw(panel)
    y = 10
    draw.text((10, y), "Two-session change map", fill=INK, font=FONT)
    y += 22
    for code, label, colour in CLASSES:
        draw.rectangle((10, y + 2, 24, y + 14), fill=colour, outline=INK)
        draw.text((31, y), label, fill=INK, font=FONT)
        y += 17
        draw.text((31, y), f"{counts.get(code, 0)} px", fill=(168, 181, 171), font=FONT)
        y += 15
    draw.text((10, y + 1), "Pending = hatching, beyond reference support", fill=(168, 181, 171), font=FONT)
    draw.text((10, y + 16), "Dark = not clearly measured in both sessions", fill=(168, 181, 171), font=FONT)
    y += 38
    lines = [
        f"Colour limit: {thresholds[0]:.1f} dE (set {request.colour_limit:g})",
        f"Contour limit: {thresholds[1]:.1f} deg (set {request.normal_limit:g})",
        f"Reference support: {request.support_distance_mm:g} mm",
        f"Reference noise: {reference[0]:.1f} dE / {reference[1]:.1f} deg",
        f"Common measured area: {common_pixels} px",
        "Reference patch assumed unchanged.",
    ]
    for line in lines:
        draw.text((10, y), line, fill=INK, font=FONT)
        y += 17
    return panel


def _class_rows(classes, common, scale):
    """Per-class pixel counts, physical areas and shares of the measured window."""
    pixel_mm2 = scale.dx * scale.dy
    total = int(common.sum())
    rows = []
    for code, label, _ in CLASSES:
        count = int((classes == code).sum())
        rows.append({"class": code, "reading": label, "pixels": count,
                     "area_mm2": rounded(count * pixel_mm2),
                     "share_of_common": rounded(safe_fraction(count, total)),
                     "share_of_changes": None})
    changes = sum(row["pixels"] for row in rows)
    for row in rows:
        row["share_of_changes"] = rounded(safe_fraction(row["pixels"], changes))
    unchanged = int((common & (classes == 0)).sum())
    rows.append({"class": 0, "reading": "No screening change", "pixels": unchanged,
                 "area_mm2": rounded(unchanged * pixel_mm2),
                 "share_of_common": rounded(safe_fraction(unchanged, total)), "share_of_changes": None})
    return rows


def _support_share(reference_distance, limit):
    return safe_fraction(reference_distance <= limit, reference_distance.size)


def _distance_band(reference_distance, limit):
    share = _support_share(reference_distance, limit)
    if share >= .5:
        return "most of the common window"
    if share >= .15:
        return "part of the common window"
    return "little of the common window"


def _epoch_rows(records, epochs, scale):
    rows = []
    for role, record, epoch in zip(("earlier", "later"), records, epochs, strict=True):
        settings = record["evidence"]["parameters"]
        rows.append({"role": role, "record": record["id"], "title": record["title"],
                     "saved_at": record["created_at"], "review": record["state"],
                     "photos": epoch.photos, "saved_window": settings.get("region"),
                     "common_grid": scale.public()})
    return rows


def _surface_row(surface):
    keys = ("id", "title", "accession", "width_mm", "height_mm", "synthetic", "version")
    return {key: surface[key] for key in keys}


def _evidence(surface, scale, records, region, request, epochs, sources):
    return {
        "surface": _surface_row(surface), "scale": scale.public(),
        "parameters": {"kind": "sessions", "records": [record["id"] for record in records],
                       "region": region.public(), "reference": request.reference.public(),
                       "options": request.public(), "captures": [], "corrections": {}},
        "sources": list({source["id"]: source for source in sources}.values()),
        "epochs": [{"record": record["id"], "title": record["title"], "created_at": record["created_at"],
                    "state": record["state"], "photos": epoch.photos,
                    "captures": [frame.id for frame in epoch.frames],
                    "registrations": {frame.id: frame.registration.id for frame in epoch.frames}}
                   for record, epoch in zip(records, epochs, strict=True)],
    }


def compare_sessions(library, surface_id, body):
    request = Request.parse(body)
    workspace = library.get_surface(surface_id)
    surface = workspace["surface"]
    fetched = [_saved_record(library, record_id, surface_id) for record_id in request.records]
    for record in fetched:
        _check_sources(record, workspace)
    records = sorted(fetched, key=lambda item: item["created_at"])
    for record in records:
        pixels = sum(width * height for width, height in _source_sizes(record))
        if pixels > MAX_SESSION_PIXELS:
            raise Problem("A session exceeds the source-pixel read budget.", 413)
    scale = _common_scale(surface, records)
    saved_windows = [Region.parse(record["evidence"]["parameters"].get("region")) for record in records]
    common = _intersection(saved_windows)
    for extra in (request.region, request.drawn):
        if extra is not None:
            common = _intersection([common, extra])
    reference = request.reference
    if not (common.left <= reference.left and reference.right <= common.right and
            common.top <= reference.top and reference.bottom <= common.bottom):
        raise Problem("Keep the untouched reference patch inside the common analysis window.")
    earlier, later = (_epoch_fields(library, surface_id, record, surface, scale, common)
                      for record in records)
    common_valid = earlier.field.valid & later.field.valid
    enough = (earlier.field.input_count >= request.min_photos) & (later.field.input_count >= request.min_photos)
    clean = (earlier.field.error <= request.residual_limit) & (later.field.error <= request.residual_limit)
    common_valid &= enough & clean
    mask = np.zeros((scale.height, scale.width), dtype=bool)
    mask[common.window(scale.width, scale.height)] = True
    common_valid &= mask
    if int(common_valid.sum()) < max(200, round(.05 * mask.sum())):
        raise Problem("Too little of the common window is clearly measured in both sessions.")
    ref_mask = _reference_mask(common_valid.shape, reference) & common_valid
    if int(ref_mask.sum()) < MIN_REFERENCE_SAMPLES or ref_mask.sum() < .20 * _reference_mask(common_valid.shape, reference).sum():
        raise Problem("The reference patch has too few clearly measured pixels in both sessions.")
    earlier_ref = np.median(earlier.field.rgb[ref_mask], axis=0)
    later_ref = np.median(later.field.rgb[ref_mask], axis=0)
    if float(np.min(earlier_ref)) < .01:
        raise Problem("The reference patch has too little measured reflectance.")
    gain = later_ref / earlier_ref
    if not (np.all(gain >= .2) and np.all(gain <= 5.)):
        raise Problem("The reference patch separates too strongly between sessions. Check its position and the saved corners.")
    balanced_earlier = earlier.field.rgb * gain
    colour = delta_lab(balanced_earlier, later.field.rgb)
    dot = np.sum(earlier.field.normals * later.field.normals, axis=-1)
    angle = np.degrees(np.arccos(np.clip(dot, -1., 1.)))
    reference_colour = float(np.percentile(colour[ref_mask], 95))
    reference_angle = float(np.percentile(angle[ref_mask], 95))
    colour_limit = max(request.colour_limit, reference_colour * 1.4)
    normal_limit = max(request.normal_limit, reference_angle * 1.4)
    colour_moved = common_valid & (colour > colour_limit)
    shape_moved = common_valid & (angle > normal_limit)
    pixel_mm2 = scale.dx * scale.dy
    minimum_pixels = max(1, round(request.min_area_mm2 / pixel_mm2))
    changed = _minimum_area(colour_moved | shape_moved, minimum_pixels)
    distance = _reference_distance(scale, reference)
    far = distance > request.support_distance_mm
    both = changed & colour_moved & shape_moved
    colour_only = changed & colour_moved & ~shape_moved
    shape_only = changed & shape_moved & ~colour_moved
    classes = np.zeros(common_valid.shape, dtype=np.uint8)
    classes[both & ~far] = 1
    classes[colour_only & ~far] = 2
    classes[shape_only & ~far] = 3
    classes[both & far] = 4
    classes[colour_only & far] = 5
    classes[shape_only & far] = 6
    counts = {code: int((classes == code).sum()) for code, _, _ in CLASSES}
    rows = _class_rows(classes, common_valid, scale)
    counted = sum(counts.values())
    supported_changes = sum(counts[code] for code in SUPPORTED)
    pending_changes = sum(counts[code] for code in PENDING)
    disagreement = counts[2] + counts[3] + counts[5] + counts[6]
    base = _base_image(earlier, later, gain, common_valid)
    painted = _paint(base, classes)
    map_image = overlay(scale_bar(Image.fromarray(painted), scale.width_mm), [
        ("closed", common.corners(), INK, "Common window"),
        ("closed", reference.corners(), (250, 250, 245), "Reference"),
    ])
    panel = _legend(counts, int(common_valid.sum()), request, (colour_limit, normal_limit),
                    (reference_colour, reference_angle))
    plate = Image.new("RGB", (map_image.width + panel.width + 10, max(map_image.height, panel.height)), PAPER)
    plate.paste(map_image, (0, 0))
    plate.paste(panel, (map_image.width + 10, 0))
    result = Result("sessions", "Two-session colour and contour change map",
                    "Compare two retained normal/reflectance reports where both sessions measured clearly, "
                    "using an untouched patch to balance conditions and to limit how far changes are reported.",
                    {
                        "earlier_report": records[0]["title"], "later_report": records[1]["title"],
                        "earlier_saved_at": records[0]["created_at"], "later_saved_at": records[1]["created_at"],
                        "earlier_photos": earlier.photos, "later_photos": later.photos,
                        "common_grid": scale.public(),
                        "common_window": common.public(), "reference_patch": reference.public(),
                        "reference_support_distance_mm": request.support_distance_mm,
                        "minimum_photos_per_session": request.min_photos,
                        "maximum_fit_residual": request.residual_limit,
                        "colour_limit_requested": request.colour_limit,
                        "colour_limit_used": rounded(colour_limit),
                        "normal_limit_requested_deg": request.normal_limit,
                        "normal_limit_used_deg": rounded(normal_limit),
                        "reference_colour_p95_delta_e": rounded(reference_colour),
                        "reference_normal_p95_deg": rounded(reference_angle),
                        "reference_gain_rgb": np.round(gain, 6).tolist(),
                        "supported_area_share": rounded(_support_share(distance, request.support_distance_mm)),
                        "reference_support_band": _distance_band(distance, request.support_distance_mm),
                        "common_measured_fraction": rounded(safe_fraction(common_valid, mask.sum())),
                        "common_measured_pixels": int(common_valid.sum()),
                        "reported_change_pixels": counted,
                        "supported_change_pixels": supported_changes,
                        "held_pending_pixels": pending_changes,
                        "evidence_disagreement_fraction": rounded(safe_fraction(disagreement, counted)),
                        "both_evidence_fraction_of_changes": rounded(safe_fraction(counts[1], counted)),
                        "colour_only_pixels": counts[2] + counts[5],
                        "contour_only_pixels": counts[3] + counts[6],
                        "colour_delta_e": statistics(colour, common_valid),
                        "normal_change_degrees": statistics(angle, common_valid),
                    }, rows)
    result.image("session-change-map", plate)
    result.image("colour-change", scale_bar(heatmap(colour, common_valid,
                                                    Extent(0., max(colour_limit * 2, reference_colour * 2, 1.)),
                                                    THERMAL, True), scale.width_mm))
    result.image("contour-change", scale_bar(heatmap(angle, common_valid,
                                                     Extent(0., max(normal_limit * 2, reference_angle * 2, 1.)),
                                                     THERMAL, True), scale.width_mm))
    result.image("balanced-pair", split_visual(flatten_rgb(balanced_earlier, common_valid),
                                               flatten_rgb(later.field.rgb, common_valid),
                                               "Earlier, reference-balanced", "Later"))
    result.add("colour-deltae", "colour-difference-deltae.tiff", "image/tiff", tiff_float(colour, common_valid))
    result.add("contour-degrees", "contour-change-degrees.tiff", "image/tiff", tiff_float(angle, common_valid))
    result.add("class-raster", "change-classes.tiff", "image/tiff", tiff_float(classes.astype(float), common_valid))
    result.table("change-classes", rows)
    result.table("compared-reports", _epoch_rows(records, (earlier, later), scale))
    result.table("reference-readings", [
        {"role": role, "record": record["id"], "median_linear_rgb": np.round(values, 6).tolist(),
         "gain_to_later": np.round(ratio, 6).tolist()}
        for role, record, values, ratio in (("earlier", records[0], earlier_ref, gain),
                                            ("later", records[1], later_ref, np.ones(3)))
    ])
    result.links = [{"label": f"Earlier report: {records[0]['title']} ({records[0]['created_at'][:10]})",
                     "record": records[0]["id"]},
                    {"label": f"Later report: {records[1]['title']} ({records[1]['created_at'][:10]})",
                     "record": records[1]["id"]}]
    result.warnings.append("This map screens two separate photographic sessions. A colour or contour difference is not by itself a measurement of lost material.")
    result.warnings.append("Colour is compared after one gain per channel measured on the reference patch; contours are compared as surface normal direction.")
    band = _distance_band(distance, request.support_distance_mm)
    result.warnings.append(f"Changes farther than the reference support distance are held pending. This small reference covers {band}, so pending areas must be re-examined before their change is accepted.")
    result.warnings.append("The reference patch is assumed unchanged. If it moved or changed, every class shifts.")
    if reference_colour > request.colour_limit:
        result.warnings.append("The reference patch itself varies by more than the requested colour limit, so the effective colour limit was widened.")
    if reference_angle > request.normal_limit:
        result.warnings.append("The reference patch itself varies by more than the requested contour limit, so the effective contour limit was widened.")
    if pending_changes:
        result.warnings.append(f"{pending_changes} changed pixels lie farther than {request.support_distance_mm:g} mm from the reference and remain pending.")
    if _support_share(distance, request.support_distance_mm) < .5:
        result.warnings.append("More than half of the common window lies beyond reference support; rely on the supported classes there and treat the rest as a follow-up list.")
    excluded = int(mask.sum() - common_valid.sum())
    if excluded > .2 * mask.sum():
        result.warnings.append("A substantial part of the common window was not clearly measured in both sessions and is left dark.")
    evidence = _evidence(surface, scale, records, common, request, (earlier, later),
                         [source for epoch in (earlier, later) for source in epoch.sources])
    return result, evidence
