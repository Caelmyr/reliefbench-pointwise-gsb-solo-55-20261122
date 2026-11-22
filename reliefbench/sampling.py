"""Source-coordinate inspection cards and common support diagnostics."""
import numpy as np
from PIL import ImageDraw
from scipy import ndimage

from .dataset import GRID_INK, flatten_numbers, region_row
from .models import Point, Problem, Region, choice, fields, number, rounded, whole
from .raster import line_samples, safe_fraction, sample_field, statistics
from .results import Result
from .visual import (COOL, DIVERGE, GOLD, HOT, MINT, Extent, Trace, contact_sheet, flatten_rgb,
                     heatmap, histogram, overlay, plot, scale_bar, split_visual, support_image)


def _photo_view(frame, dataset, shown=None):
    mask = frame.alpha_valid if shown is None else shown
    picture = flatten_rgb(frame.code, mask, False)
    guides = [("closed", dataset.region.corners(), MINT, "Analysis region")]
    guides.extend(("closed", camera, GRID_INK, label)
                  for owner, _shape, camera, _ink, label in dataset.grid_guides("camera") if owner == frame.id)
    return overlay(picture, guides)


def frames_review(dataset, options):
    fields(options, "bins")
    bins = whole(options.get("bins", 44), "Histogram bins", 16, 100)
    rows, graphs, tables = [], [], []
    colors = [MINT, COOL, GOLD, HOT]
    for i, frame in enumerate(dataset.frames):
        row = region_row(frame, dataset.selection)
        clean = frame.alpha_valid & dataset.selection
        rows.append(row)
        if clean.any() and len(graphs) < len(colors):
            x, y = histogram(frame.code_y[clean], (0, 1), bins)
            graphs.append(Trace(frame.title[:23], x, y / max(1, clean.sum()), colors[i]))
        tables.append((f"{i+1}. {frame.title}", _photo_view(frame, dataset)))
    if not graphs:
        raise Problem("None of these photographs cover the selected area.")
    result = Result("intake", "Photograph signal and clipping",
                    "Compare readout in a rectangular region before estimating surface shape.",
                    {"photos": len(rows), "registered_grid": dataset.scale.public(),
                     "shown_histograms": len(graphs)}, rows)
    result.image("capture-contact", contact_sheet(tables, columns=min(3, len(tables))))
    result.image("capture-histogram", plot(graphs, "sRGB code", "Share of visible observations", (0, 1)))
    return result.table("capture-statistics", [flatten_numbers(row) for row in rows])


def overlap_review(dataset, options):
    fields(options, "mode", "edge_weight")
    mode = choice(options.get("mode", "checker"), "Alignment view", ("checker", "stereo", "contours"))
    a, b = dataset.pair()
    both = a.alpha_valid & b.alpha_valid & dataset.selection
    either = (a.alpha_valid | b.alpha_valid) & dataset.selection
    dataset.require(both, "Pair overlap")
    edge_weight = number(options.get("edge_weight", .55), "Contour sensitivity", .12, .95)
    mask = np.where(both, 3, np.where(a.alpha_valid & dataset.selection, 2,
                    np.where(b.alpha_valid & dataset.selection, 1, 0)))
    width, height = dataset.scale.width, dataset.scale.height
    restore = _pair_view(a, b, mode, edge_weight, both, width, height)
    colors = (np.array([(42, 54, 61), (125, 136, 237), (234, 163, 87), (99, 202, 164)]))[mask]
    keys = [
        {"support": "common", "pixels": int(both.sum()), "share": rounded(safe_fraction(both, dataset.selection.sum()))},
        {"support": "first only", "pixels": int((mask == 2).sum()), "share": rounded(safe_fraction(mask == 2, dataset.selection.sum()))},
        {"support": "second only", "pixels": int((mask == 1).sum()), "share": rounded(safe_fraction(mask == 1, dataset.selection.sum()))},
        {"support": "neither", "pixels": int((dataset.selection & ~either).sum()),
         "share": rounded(safe_fraction(dataset.selection & ~either, dataset.selection.sum()))},
    ]
    result = Result("overlap", "Overlap and alignment inspection",
                    "Check matched edges on the physical working grid before any shape inversion.",
                    {"mode": mode, "common_fraction": keys[0]["share"], "a": a.title, "b": b.title,
                     "region": dataset.region.public(), "scale": dataset.scale.public()}, keys)
    result.image("registration-overlay", restore)
    result.image("registration-coverage", flatten_rgb(colors / 255., dataset.selection, False))
    result.warnings.append("A change in light direction changes shadows. Treat aligned fixed marks rather than shadow edges as registration landmarks.")
    return result.table("alignment-support")


def _edge(frame, weight):
    luma = frame.code_y
    softened = ndimage.gaussian_filter(luma, .65, mode="nearest")
    magnitude = np.hypot(*np.gradient(softened))
    band = np.clip(magnitude * 15., 0., 1.)
    return np.clip((band - weight / 2) / max(.08, 1 - weight / 2), 0., 1.)


def _pair_view(a, b, mode, weight, valid, width, height):
    if mode == "checker":
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        toggle = ((x // max(16, width // 7)) + (y // max(16, height // 7))) % 2 == 0
        rgb = np.where(toggle[..., None], a.code, b.code)
    elif mode == "stereo":
        first, second = a.code_y, b.code_y
        rgb = np.stack((first, second, second), axis=-1)
    else:
        first = _edge(a, weight)
        second = _edge(b, weight)
        rgb = np.stack((first, np.clip(second + first * .40, 0, 1), second), axis=-1)
    return flatten_rgb(rgb, valid, False)


def light_patch(dataset, options):
    fields(options, "material", "patches", "levels")
    frame = dataset.one()
    expected = options.get("levels", [.18, .40, .72])
    raw = options.get("patches", [[.25, .25, .34, .34], [.45, .25, .54, .34], [.65, .25, .74, .34]])
    if not isinstance(raw, list) or not isinstance(expected, list) or len(raw) != len(expected) or not 2 <= len(raw) <= 8:
        raise Problem("Supply two to eight patch boxes and a matching known grayscale for each.")
    label = options.get("material", "Neutral reference")
    from .models import text
    label = text(label, "Matte target", 80)
    ox = []
    targets = []
    for patch, target in zip(raw, expected, strict=True):
        region = Region.parse(patch)
        value = number(target, "Known neutral reflectance", .015, .99)
        ox.append(region)
        targets.append(value)
    if len(set(targets)) != len(targets):
        raise Problem("Each patch should have a distinct known reflectance.")
    if not all(dataset.region.contains(Point(p.left, p.top)) and dataset.region.contains(Point(p.right, p.bottom)) for p in ox):
        raise Problem("Keep neutral patches inside the selected region.")
    shape = frame.alpha_valid.shape
    rows, guides = [], []
    for i, (patch, target) in enumerate(zip(ox, targets, strict=True), 1):
        selection = np.zeros(shape, dtype=bool)
        selection[patch.window(shape[1], shape[0])] = True
        covered = selection & frame.alpha_valid
        valid = covered & ~frame.clipped
        usable = valid.sum() >= max(28, round(selection.sum() * .68))
        centers = [float(np.median(frame.linear[..., k][valid])) for k in range(3)] if usable else None
        observed = float(np.median(frame.signal[valid])) if usable else None
        rows.append({"patch": i, "region": patch.public(), "known_reflectance": target,
                     "observed_linear": rounded(observed), "visible_fraction": rounded(covered.sum() / selection.sum()),
                     "clipped_fraction": rounded((selection & frame.clipped).sum() / selection.sum()),
                     "usable": bool(usable), "rgb_linear": centers,
                     "deviation": rounded(observed - target) if usable else None})
        guides.append(("closed", patch.corners(), MINT if usable else HOT, f"{i} | {target:.2f}"))
    usable = sorted([r for r in rows if r["usable"]], key=lambda row: row["known_reflectance"])
    if len(usable) < 2:
        raise Problem("The target needs two observed, unclipped patches.")
    targets = np.array([row["known_reflectance"] for row in usable])
    actual = np.array([row["observed_linear"] for row in usable])
    gains = targets / np.maximum(actual, .001)
    rank_problem = any(float(x) < 0 for x in np.diff(actual))
    result = Result("tone-target", "Neutral reference readings",
                    "Read known matte reference patches before choosing capture exposure.",
                    {"material": label, "valid_patches": len(usable), "rank_reversal": rank_problem,
                     "median_linear_gain": rounded(np.median(gains)),
                     "spread_of_gains": rounded(np.max(gains) - np.min(gains)),
                     "range_of_valid_targets": [float(targets.min()), float(targets.max())]}, rows)
    result.image("neutral-review", overlay(_photo_view(frame, dataset), guides))
    result.image("neutral-response", plot([Trace("Observed", targets, actual), Trace("Reference", targets, targets, GOLD)],
                                          "Known reflectance", "Observed linear signal"))
    if rank_problem:
        result.warnings.append("At least one adjacent neutral level reverses. Check patch selection or mixed glare before using a correction.")
    return result.table("neutral-readings")


def camera_trace(dataset, options):
    fields(options, "start", "end", "samples", "mode")
    a = Point.parse(options.get("start", [.18, .42]), "Trace start")
    b = Point.parse(options.get("end", [.82, .55]), "Trace end")
    if not (dataset.region.contains(a) and dataset.region.contains(b)):
        raise Problem("Keep both trace endpoints in the analysis rectangle.")
    n = whole(options.get("samples", 180), "Trace samples", 18, 600)
    mode = choice(options.get("mode", "encoded"), "Signal scale", ("encoded", "normalized"))
    dist = dataset.scale.distance(a, b)
    if dist < .1:
        raise Problem("Trace endpoints are too close to read.")
    points = line_samples(a, b, n)
    positions = np.linspace(0, dist, n)
    traces = []
    rows = []
    photos = []
    palette = [MINT, GOLD, COOL, HOT]
    for i, frame in enumerate(dataset.frames[:4]):
        image = frame.code_y if mode == "encoded" else frame.signal
        read, visible = sample_field(image, frame.alpha_valid & ~frame.clipped, points)
        traces.append(Trace(frame.title[:22], positions, np.where(visible, read, np.nan), palette[i]))
        camera = frame.registration.camera(points)
        for idx in range(n):
            rows.append({"photo": frame.title, "distance_mm": rounded(positions[idx]), "observed": bool(visible[idx]),
                         "readout": rounded(read[idx]) if visible[idx] else None,
                         "camera_x": rounded(camera[idx, 0]), "camera_y": rounded(camera[idx, 1]),
                         "surface_x": rounded(points[idx, 0]), "surface_y": rounded(points[idx, 1])})
        photos.append((frame.title, overlay(_photo_view(frame, dataset),
                                           [("line", [a.public(), b.public()], palette[i], "Readout")])))
    result = Result("camera-trace", "Signal profile in physical coordinates",
                    "Compare archived camera values along a nominated line with unusable samples left open.",
                    {"line_length_mm": rounded(dist), "samples_per_photo": n, "signal_scale": mode,
                     "sources_shown": len(traces)}, rows)
    result.image("line-location", contact_sheet(photos, columns=min(3, len(photos))))
    result.image("line-readout", plot(traces, "Along line (mm)", "Code" if mode == "encoded" else "Linear signal"))
    return result.table("camera-line-readings")


def glare_census(dataset, options):
    fields(options, "code_limit", "min_area_mm2")
    frame = dataset.one()
    cutoff = number(options.get("code_limit", .92), "Bright code threshold", .60, 1.)
    min_area = number(options.get("min_area_mm2", .08), "Minimum connected area", .01, 500.)
    valid = dataset.require(frame.alpha_valid, "The photograph")
    signal = frame.code_y
    seen = (signal >= cutoff) & valid
    expanded = ndimage.binary_closing(seen, structure=np.ones((3, 3))) & valid
    region, n = ndimage.label(seen | expanded, structure=np.ones((3, 3)))
    area = dataset.scale.dx * dataset.scale.dy
    sizes = np.bincount(region.ravel())[1:]
    kept = sorted([index for index in range(n) if sizes[index] * area >= min_area],
                  key=lambda i: -sizes[i])[:80]
    boxes, rows = [], []
    width, height = dataset.scale.width, dataset.scale.height
    for num, index in enumerate(kept, 1):
        member = region == index + 1
        yy, xx = np.nonzero(member)
        box = Region(xx.min() / width, yy.min() / height, (xx.max() + 1) / width, (yy.max() + 1) / height)
        px = (xx.mean() + .5) / width
        py = (yy.mean() + .5) / height
        at = frame.registration.camera([[px, py]])[0]
        rows.append({"region": num, "area_mm2": rounded(member.sum() * area, 4), "peak_code": rounded(signal[member].max()),
                     "sensor_clip_fraction": rounded((member & frame.clipped).sum() / member.sum()),
                     "surface_center": [rounded(px), rounded(py)], "camera_center": [rounded(x) for x in at],
                     "region_box": box.public()})
        boxes.append(("closed", box.corners(), GOLD, f"Bright {num}"))
    layer = np.where(seen[..., None], [.91, .45, .22], frame.code)
    result = Result("glare", "Bright and clipped candidates",
                    "Mark connected high codes and report the portion clipping in the original exposure.",
                    {"code_limit": cutoff, "component_count": len(rows),
                     "clipped_fraction": rounded(safe_fraction(frame.clipped & valid, valid.sum())),
                     "detected_area_mm2": rounded(seen.sum() * area, 3)}, rows)
    result.image("glare-census", overlay(flatten_rgb(layer, valid, False), boxes))
    result.warnings.append("A bright area can be diffuse paint. This map does not identify reflection material by itself.")
    return result.table("bright-regions")
