"""Read acquisition quality, evaluate held-out illumination and compare coverage."""
import math

import numpy as np
from PIL import Image
from scipy import ndimage

from .dataset import DataSet
from .lightmodel import solve
from .models import Point, Problem, Region, choice, fields, number, rounded, whole
from .raster import intensity, robust_local, safe_fraction, statistics
from .results import Result
from .visual import (COOL, DIVERGE, GOLD, HOT, MINT, Extent, Trace, contact_sheet, flatten_rgb, heatmap,
                     histogram, overlay, plot, scale_bar, split_visual)


def _image_window(dataset, valid):
    return dataset.region.window(valid.shape[1], valid.shape[0])


def _square_grid(rows, columns):
    for row in range(rows):
        for column in range(columns):
            yield row, column, Region(column / columns, row / rows, (column + 1) / columns, (row + 1) / rows)


def _normalize_lamp(measurement, expected):
    sample = measurement[expected]
    low, high = np.percentile(sample, [5, 95])
    if high - low < .012:
        raise Problem("Registration marks have too little exposure contrast in this region.")
    return np.clip((measurement - low) / (high - low), 0., 1.)


def _sharpness_maps(signal, valid, scale, noise_radius):
    usable = ndimage.binary_erosion(valid, iterations=3)
    smoothed, local = robust_local(signal, valid, noise_radius)
    dy, dx = np.gradient(smoothed, scale.dy, scale.dx)
    gradient = np.hypot(dx, dy)
    laplace = ndimage.laplace(smoothed)
    twice = math.hypot(scale.dx, scale.dy)
    return gradient, laplace / max(twice ** 2, .000001), usable & (local >= .994)


def _score_tiles(gradient, laplace, valid, rows, columns, sample_mask):
    records = []
    for row, column, box in _square_grid(rows, columns):
        zone = box.window(valid.shape[1], valid.shape[0])
        seen = valid[zone]
        intended = sample_mask[zone]
        count = int(seen.sum())
        portion = count / max(1, intended.sum())
        if count < 20 or portion < .5:
            records.append({"row": row + 1, "column": column + 1, "area": box.public(),
                            "status": "unobserved", "samples": count, "fraction": rounded(portion)})
            continue
        amplitude = gradient[zone][seen]
        lp = laplace[zone][seen]
        p90, p97 = np.percentile(amplitude, [90, 97])
        spread = np.percentile(lp, 95) - np.percentile(lp, 5)
        records.append({"row": row + 1, "column": column + 1, "area": box.public(),
                        "status": "enough image", "samples": count, "fraction": rounded(portion),
                        "p90_edge_code_per_mm": rounded(p90), "p97_edge_code_per_mm": rounded(p97),
                        "laplace_90_span": rounded(spread), "min_support": count})
    return records


def focus_scout(dataset, options):
    fields(options, "rows", "columns", "smoothness", "gain_normalized")
    frame = dataset.one()
    rows = whole(options.get("rows", 5), "Focus grid rows", 2, 12)
    cols = whole(options.get("columns", 7), "Focus grid columns", 2, 12)
    smooth = number(options.get("smoothness", .65), "Pre-smooth pixels", .3, 2.8)
    active = dataset.require(frame.alpha_valid & ~frame.clipped, "Unclipped focus input")
    luma = _normalize_lamp(frame.code_y, active) if options.get("gain_normalized", False) is True else frame.code_y
    edge, laplace, usable = _sharpness_maps(luma, active, dataset.scale, smooth)
    readings = _score_tiles(edge, laplace, usable, rows, cols, dataset.selection)
    strongest = max((r for r in readings if r["status"] == "enough image"),
                    key=lambda r: r["p97_edge_code_per_mm"], default=None)
    inferred = Extent.fit(edge, usable, opts=(8, 99), zero="minimum")
    result = Result("focus", "Focus and local detail scout",
                    "Review coverage and spatial detail relative to the same photograph.",
                    {"median_edge_strength": statistics(edge, usable)["median"], "strongest_tile": strongest,
                     "detail_color_range": inferred.public(),
                     "complete_focus_tiles": sum(row["status"] == "enough image" for row in readings)}, readings)
    guides = [("closed", Region.parse(row["area"]).corners(), MINT if row["status"] == "enough image" else GOLD,
               f'{row["row"]}.{row["column"]}') for row in readings]
    result.image("acquisition-focus", scale_bar(overlay(heatmap(edge, usable, inferred, grid=True), guides),
                                                dataset.scale.width_mm))
    result.image("focus-context", overlay(flatten_rgb(frame.code, frame.alpha_valid, False), guides))
    result.warnings.append("Detail varies with the actual subject and shadows. Compare regions containing a comparable focus mark.")
    return result.table("detail-tile-observations")


def _refit_subset(dataset, omitted):
    remaining = tuple(frame for i, frame in enumerate(dataset.frames) if i != omitted)
    return DataSet(dataset.surface, remaining, dataset.scale, dataset.region, dataset.selection)


def _heldout_error(dataset, index, predicted_field):
    test = dataset.frames[index]
    all_ok = predicted_field.valid & dataset.selection & test.alpha_valid & ~test.clipped
    predicted = predicted_field.predicted(test)
    err = intensity(predicted) - test.signal
    return predicted, err, all_ok


def held_out_light(dataset, options):
    fields(options, "photo_index", "gain_range")
    inputs = dataset.lamps()
    if len(inputs) < 4:
        raise Problem("Select four or more photos to reserve one as a check.")
    index = whole(options.get("photo_index", len(inputs)), "Reserved light position", 1, len(inputs)) - 1
    fixed_range = number(options.get("gain_range", .08), "Residual scale", .008, .75)
    rest = _refit_subset(dataset, index)
    field = solve(rest)
    test = inputs[index]
    predicted, delta, valid = _heldout_error(dataset, index, field)
    active = dataset.require(valid, "Held-out comparison")
    observed_statistics = statistics(test.signal, active)
    fit_statistics = statistics(np.abs(delta), active)
    rows = [{"set": "holdout" if at == index else "fit input", "photo": frame.title, "capture_id": frame.id,
             "angle": frame.spec.azimuth, "elevation": frame.spec.elevation,
             "visible_fraction": rounded(safe_fraction(dataset.selection & frame.alpha_valid, dataset.selection.sum()))}
            for at, frame in enumerate(inputs)]
    result = Result("holdout", "Reserved lamp evidence",
                    "Use the selected remaining lamps to predict the reserved source and identify disagreement in supported pixels.",
                    {"holdout": test.title, "index": index + 1, "fit_sources": len(rest.frames),
                     "observed_linear_readout": observed_statistics, "absolute_residual_summary": fit_statistics,
                     "coherent_fraction": rounded(safe_fraction(active, dataset.selection.sum())),
                     "signed_color_limit": [-fixed_range, fixed_range]}, rows)
    result.image("heldout-pair", split_visual(flatten_rgb(test.linear, active),
                                               flatten_rgb(predicted, active), "Observed reserved", "Predicted from other lamps"))
    result.image("heldout-disagreement", heatmap(delta, active, Extent(-fixed_range, fixed_range), DIVERGE, True))
    hist_x, hist_y = histogram(np.clip(delta[active], -fixed_range, fixed_range), (-fixed_range, fixed_range), 60)
    result.image("holdout-histogram", plot([Trace("Difference", hist_x, hist_y)],
                                          "Predicted minus observed", "Samples"))
    result.warnings.append("Reserved source is a validity check for the diffuse model and chosen exposure. It is not used in this normal estimate.")
    return result.table("reserved-fit-inputs")


def _transition_flags(covered, min_neighbours):
    count = ndimage.convolve(covered.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    return covered & (count < min_neighbours)


def _source_support_summary(frame, accepted, dataset):
    base = dataset.selection
    counts = int(base.sum())
    cover = frame.alpha_valid & base
    cuts = frame.clipped & cover
    y, x = np.where(cover)
    touches = False
    if x.size:
        touches = (x.min() == 0 or y.min() == 0 or x.max() == cover.shape[1] - 1 or y.max() == cover.shape[0] - 1)
    return {"photo": frame.title, "exposure_ev": frame.spec.exposure, "source_hash": frame.record["hash"],
            "visible": int(cover.sum()), "clipped": int(cuts.sum()),
            "usable": int(accepted.sum()), "visible_fraction": rounded(cover.sum() / counts),
            "about_boundary": bool(touches)}


def stack_support(dataset, options):
    fields(options, "minimum", "discard_clips")
    photos = dataset.frames
    minimum = whole(options.get("minimum", min(3, len(photos))), "Minimum supported frames", 1, len(photos))
    clip = options.get("discard_clips", True)
    if type(clip) is not bool:
        raise Problem("Ignore clipped samples needs on or off.")
    covers = [frame.alpha_valid & dataset.selection & (~frame.clipped if clip else True) for frame in photos]
    total = np.asarray(covers).sum(axis=0)
    enough = (total >= minimum) & dataset.selection
    segments, number_segments = ndimage.label(enough)
    sizes = np.bincount(segments.ravel(), minlength=number_segments + 1)
    boundary = _transition_flags(enough, 7)
    plots = []
    for i, (frame, usable) in enumerate(zip(photos, covers, strict=True), 1):
        clip_color = np.where(frame.clipped[..., None], np.asarray([.89, .42, .31]), frame.code)
        skin = flatten_rgb(clip_color, frame.alpha_valid, False)
        plots.append((f"{i}. {frame.title}", skin))
    color = np.array(heatmap(total, dataset.selection, Extent(0, len(photos)), grid=True))
    color[boundary] = GOLD
    profiles = [{"usable_photos": i, "pixels": int(((total == i) & dataset.selection).sum()),
                "fraction": rounded(safe_fraction((total == i) & dataset.selection, dataset.selection.sum()))}
                for i in range(len(photos) + 1)]
    result = Result("support", "Multi-photo support map",
                    "See the number of images contributing usable samples at each site before solving.",
                    {"target_frame_count": minimum, "discard_clipped": clip,
                     "eligible_fraction": rounded(safe_fraction(enough, dataset.selection.sum())),
                     "largest_contiguous_share": rounded((np.max(sizes[1:]) if number_segments else 0) /
                                                        max(1, dataset.selection.sum())),
                     "regions_at_minimum": int(number_segments)}, profiles)
    result.image("stack-coverage-map", scale_bar(Image.fromarray(color), dataset.scale.width_mm))
    result.image("stack-source-thumbnails", contact_sheet(plots, columns=min(4, len(plots))))
    result.table("photo-support", [_source_support_summary(p, mask, dataset) for p, mask in zip(photos, covers, strict=True)])
    return result.table("coverage-distribution")


def _spatial_probe_values(frame, point):
    camera = frame.registration.camera([point.public()])[0]
    h, w = frame.raw.shape
    safe = 0 <= camera[0] <= 1 and 0 <= camera[1] <= 1
    coords = None
    if safe:
        x, y = round(camera[0] * (w - 1)), round(camera[1] * (h - 1))
        safe = frame.raw.alpha[y, x] >= .98
        if safe:
            coords = [x, y]
    rgbs = np.round(frame.raw.rgb[coords[1], coords[0]], 6).tolist() if coords else None
    return {"camera_pixel": coords, "camera_normalized": [rounded(x) for x in camera], "pixel_rgb_code": rgbs,
            "usable": bool(safe), "clipping": bool(rgbs and max(rgbs) >= .992)}


def physical_probe(dataset, options):
    fields(options, "points")
    raw = options.get("points", [[.24, .35], [.43, .52], [.76, .61]])
    if not isinstance(raw, list) or not 1 <= len(raw) <= 20:
        raise Problem("Enter one to twenty surface marker points.")
    points = [Point.parse(p, "Readout point") for p in raw]
    if any(not dataset.region.contains(p) for p in points):
        raise Problem("Marker points should be inside the selected area.")
    result_rows = []
    for n, point in enumerate(points, 1):
        xy = [rounded(point.x * dataset.scale.width_mm), rounded(point.y * dataset.scale.height_mm)]
        for frame in dataset.frames:
            result_rows.append({"point": n, "photo": frame.title,
                                "surface_mm": xy, "surface_normalized": point.public(),
                                **_spatial_probe_values(frame, point)})
    result = Result("probe", "Revisit measured points",
                    "Locate markers on original photos and inspect native camera codes.",
                    {"points": len(points), "photos": len(dataset.frames), "readings": len(result_rows),
                     "physical_scale": dataset.scale.public()}, result_rows)
    drawings = []
    for frame in dataset.frames[:8]:
        drawn = []
        for i, point in enumerate(points, 1):
            mapped = frame.registration.camera([point.public()])[0]
            if (mapped >= 0).all() and (mapped <= 1).all():
                drawn.append(("point", [mapped.tolist()], MINT, str(i)))
        raw_view = flatten_rgb(frame.raw.rgb, frame.raw.alpha >= .98, False)
        drawings.append((frame.title, overlay(raw_view, drawn)))
    result.image("camera-point-locations", contact_sheet(drawings, columns=min(len(drawings), 3)))
    return result.table("native-camera-counts")
