"""Split a glazed surface into matte diffuse colour and specular reflection.

The two registered photographs are a parallel-polarized frame (diffuse plus
specular reflection) and a cross-polarized frame (specular reflection is
blocked, so only diffuse remains). A matte box visible in both frames anchors
the exposure match. Every output pixel carries a reliability classification
because clipped camera codes cannot be turned into definite readings.
"""
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .models import Point, Problem, Region, fields, number, rounded, whole
from .raster import LUMA, safe_fraction
from .results import Result
from .visual import (COOL, GAP, GOLD, HOT, MINT, PAPER, INK, FONT, contact_sheet,
                     flatten_rgb, heatmap, overlay, scale_bar)


GAIN_TRUST = (.5, 2.)
PURPLE = (198, 137, 221)
STATUS_COLORS = {
    "definite": COOL, "lower-bound": GOLD, "diffuse-unknown": HOT,
    "range-only": PURPLE, "unobserved": GAP,
}
STATUS_LEGEND = (
    ("definite", "Definite: both frames unclipped"),
    ("lower-bound", "Specular lower bound: parallel frame clipped"),
    ("diffuse-unknown", "Diffuse unknown: cross frame clipped"),
    ("range-only", "Highlight range only: not a measured value"),
    ("unobserved", "Unobserved or single-frame coverage"),
)
MIN_COMPONENT_MM2 = .08


def _reference_mask(shape, region):
    mask = np.zeros(shape, dtype=bool)
    mask[region.window(shape[1], shape[0])] = True
    return mask


def _legend_band(picture, entries):
    width = picture.width
    draw = ImageDraw.Draw(Image.new("RGB", (width, 10)))
    x, row = 8, 6
    for color, label in entries:
        block = draw.textlength(label, font=FONT) + 30
        if x + block > width - 8:
            x, row = 8, row + 20
        x += int(block)
    band_height = row + 16
    band = Image.new("RGB", (width, band_height), PAPER)
    draw = ImageDraw.Draw(band)
    x, row = 8, 6
    for color, label in entries:
        block = draw.textlength(label, font=FONT) + 30
        if x + block > width - 8:
            x, row = 8, row + 20
        draw.rectangle((x, row, x + 12, row + 12), fill=tuple(color), outline=INK)
        draw.text((x + 17, row), label, fill=INK, font=FONT)
        x += int(block)
    out = Image.new("RGB", (width, picture.height + band_height), PAPER)
    out.paste(picture, (0, 0))
    out.paste(band, (0, picture.height))
    return out


def _source_tile(frame, reference, title):
    tinted = np.where(frame.clipped[..., None], np.asarray([.89, .42, .31]), frame.code)
    picture = flatten_rgb(tinted, frame.alpha_valid, False)
    return title, overlay(picture, [("closed", reference.corners(), MINT, "Matte reference")])


def _component_rows(bright, para, definite, lower_bound, dataset):
    closed_regions = ndimage.binary_closing(bright, structure=np.ones((3, 3)))
    labels, count = ndimage.label(closed_regions, structure=np.ones((3, 3)))
    area = dataset.scale.dx * dataset.scale.dy
    height, width = para.alpha_valid.shape
    rows, kept = [], []
    sizes = np.bincount(labels.ravel())[1:]
    for index in sorted(range(count), key=lambda i: -sizes[i])[:80]:
        member = labels == index + 1
        if member.sum() * area < MIN_COMPONENT_MM2:
            continue
        kept.append(index + 1)
        yy, xx = np.nonzero(member)
        box = Region(xx.min() / width, yy.min() / height, (xx.max() + 1) / width, (yy.max() + 1) / height)
        total = member.sum()
        rows.append({
            "kind": "highlight", "component": len(kept), "area_mm2": rounded(total * area, 4),
            "peak_code": rounded(para.code_y[member].max()),
            "clipped_fraction": rounded((member & para.clipped).sum() / total),
            "definite_fraction": rounded((member & definite).sum() / total),
            "lower_bound_fraction": rounded((member & lower_bound).sum() / total),
            "range_only_fraction": rounded((member & ~definite & ~lower_bound).sum() / total),
            "region_box": box.public(),
        })
    range_only = np.isin(labels, kept) & ~definite & ~lower_bound
    return rows, range_only


def polarization_report(dataset, options):
    fields(options, "reference", "parallel_order", "highlight_code")
    para, cross = dataset.pair()
    if whole(options.get("parallel_order", 1), "Parallel-polarized position", 1, 2) == 2:
        para, cross = cross, para
    limit = number(options.get("highlight_code", .92), "Highlight code threshold", .6, 1.)
    reference = Region.parse(options.get("reference", [.12, .12, .26, .26]))
    if not all(dataset.region.contains(point) for point in (
            Point(reference.left, reference.top), Point(reference.right, reference.bottom))):
        raise Problem("Keep the shared matte reference box inside the analysis region.")

    shape = para.alpha_valid.shape
    refmask = _reference_mask(shape, reference)
    ref_rows, medians = [], {}
    for frame, role in ((para, "parallel"), (cross, "cross")):
        covered = refmask & frame.alpha_valid
        usable = covered & ~frame.clipped
        if usable.sum() < max(28, round(covered.sum() * .68)):
            raise Problem(f"The matte reference is clipped or off-frame in the {role} photograph; choose a clearer box.")
        medians[role] = np.median(frame.linear[usable], axis=0)
        ref_rows.append({"kind": "reference", "photo": frame.title, "polarization": role,
                         "median_linear_rgb": np.round(medians[role], 6).tolist(),
                         "median_luma": rounded(float(medians[role] @ LUMA)),
                         "visible_fraction": rounded(covered.sum() / refmask.sum()),
                         "clipped_fraction": rounded((covered & frame.clipped).sum() / max(covered.sum(), 1)),
                         "usable_samples": int(usable.sum())})

    gain_raw = medians["cross"] / np.maximum(medians["parallel"], 1e-4)
    gain = np.clip(gain_raw, GAIN_TRUST[0], GAIN_TRUST[1]).astype(np.float32)
    gain_trusted = bool(np.all((gain_raw >= GAIN_TRUST[0]) & (gain_raw <= GAIN_TRUST[1])))
    matched_cross = cross.linear / gain
    noise = para.signal[refmask & para.alpha_valid & ~para.clipped] - (
        matched_cross @ LUMA)[refmask & para.alpha_valid & ~para.clipped]
    mad = float(np.median(np.abs(noise - np.median(noise))))
    spec_floor = max(.008, 3. * mad)

    both = para.alpha_valid & cross.alpha_valid & dataset.selection
    unobserved = dataset.selection & ~both
    para_ok, cross_ok = ~para.clipped, ~cross.clipped
    definite = both & para_ok & cross_ok
    lower_bound = both & para.clipped & cross_ok
    diffuse_unknown = both & cross.clipped
    spec_rgb = np.maximum(para.linear - matched_cross, 0.)
    spec_luma = spec_rgb @ LUMA
    definite_spec = definite & (spec_luma >= spec_floor)

    bright = (para.code_y >= limit) & both
    component_rows, range_only = _component_rows(
        bright, para, definite_spec, lower_bound, dataset)
    range_only &= dataset.selection
    masks = {"definite": definite, "lower-bound": lower_bound,
             "diffuse-unknown": diffuse_unknown, "range-only": range_only, "unobserved": unobserved}

    known_diffuse = definite | lower_bound
    diffuse_rgb = np.where(known_diffuse[..., None], matched_cross, 0.)
    diffuse_alpha = np.where(known_diffuse, 255, 0).astype(np.uint8)
    result = Result(
        "polarization", "Polarized diffuse and specular separation",
        "Matte-anchored exposure match separates definite diffuse colour and glare; clipped zones keep their reason instead of a reading.",
        {"parallel_photo": para.title, "cross_photo": cross.title, "reference_box": reference.public(),
         "gain_rgb": np.round(gain, 5).tolist(), "gain_luma": rounded(float(gain @ LUMA)),
         "gain_trusted": gain_trusted, "specular_noise_floor": rounded(mad, 6),
         "specular_detect_floor": rounded(spec_floor, 6), "highlight_code_threshold": limit,
         "definite_fraction": rounded(safe_fraction(definite, dataset.selection.sum())),
         "specular_lower_bound_fraction": rounded(safe_fraction(lower_bound, dataset.selection.sum())),
         "diffuse_indeterminate_fraction": rounded(safe_fraction(diffuse_unknown, dataset.selection.sum())),
         "highlight_range_area_mm2": rounded(range_only.sum() * dataset.scale.dx * dataset.scale.dy, 4),
         "highlight_component_count": len(component_rows)})
    result.payload = {
        "diffuse_rgb": diffuse_rgb.astype(np.float32), "diffuse_alpha": diffuse_alpha,
        "gain": gain.tolist(), "reference": reference.public(), "spec_floor": spec_floor,
        "parallel": {"id": para.id, "title": para.title, "hash": para.record["hash"],
                     "registration_id": para.registration.id, "capture_version": para.record["version"]},
        "cross": {"id": cross.id, "title": cross.title, "hash": cross.record["hash"],
                  "registration_id": cross.registration.id, "capture_version": cross.record["version"]},
        "counts": {name: int(mask.sum()) for name, mask in masks.items()},
        "highlight_code": limit,
    }

    status = np.full((*shape, 3), GAP, dtype=np.uint8)
    for name, mask in masks.items():
        status[mask] = STATUS_COLORS[name]
    spec_view = heatmap(np.where(definite, spec_luma, np.nan), definite)
    spec_view = np.asarray(spec_view).copy()
    spec_view[unobserved] = GAP
    spec_view[diffuse_unknown & ~range_only] = GAP
    spec_view[lower_bound] = GOLD
    spec_view[range_only] = PURPLE
    boxes = [("closed", Region.parse(row["region_box"]).corners(), GOLD, str(row["component"]))
             for row in component_rows]
    result.image("polarization-sources", contact_sheet([
        _source_tile(para, reference, f"1. Parallel · {para.title}"),
        _source_tile(cross, reference, f"2. Cross · {cross.title}")], columns=2))
    result.image("polarization-diffuse", scale_bar(
        overlay(flatten_rgb(diffuse_rgb, known_diffuse), boxes), dataset.scale.width_mm))
    result.image("polarization-specular", _legend_band(
        scale_bar(Image.fromarray(spec_view), dataset.scale.width_mm),
        [(STATUS_COLORS["definite"], "Measured specular"), (GOLD, "Lower bound (clipped)"),
         (PURPLE, "Range only"), (GAP, "No reading")]))
    result.image("polarization-status", _legend_band(Image.fromarray(status),
                 [(STATUS_COLORS[key], label) for key, label in STATUS_LEGEND]))

    total = max(dataset.selection.sum(), 1)
    rows = [*ref_rows,
            {"kind": "gain", "gain_rgb": np.round(gain_raw, 5).tolist(), "applied_gain_rgb": np.round(gain, 5).tolist(),
             "gain_luma": rounded(float(gain @ LUMA)), "gain_trusted": gain_trusted,
             "specular_floor": rounded(spec_floor, 6)},
            *[{"kind": "status", "status": name, "pixels": int(mask.sum()),
               "fraction": rounded(mask.sum() / total)} for name, mask in masks.items()],
            *component_rows]
    result.table("polarization-summary", rows)
    result.document("polarization-derivation", {
        "tool": "polarization", "gain": result.payload["gain"], "reference": reference.public(),
        "highlight_code": limit, "specular_floor": spec_floor,
        "sources": [result.payload["parallel"], result.payload["cross"]],
        "classification": result.payload["counts"], "scale": dataset.scale.public()})
    if not gain_trusted:
        result.warnings.append(
            f"Matte reference exposure ratio {np.round(gain_raw, 3).tolist()} lies outside {GAIN_TRUST}; gain was clamped and every separated value is provisional.")
    para_ref_luma = float(medians["parallel"] @ LUMA)
    cross_ref_luma = float(medians["cross"] @ LUMA)
    if cross_ref_luma > para_ref_luma * 1.05:
        result.warnings.append(
            f"The cross-polarized matte reference is brighter ({cross_ref_luma:.3f} vs {para_ref_luma:.3f}); check the photo position order.")
    result.warnings.append("Specular values are lower bounds wherever the parallel-polarized frame clips to white.")
    result.warnings.append("Range-only highlight zones show extent from bright codes; they are not quantified reflection readings.")
    return result
