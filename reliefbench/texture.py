"""Quantify surface texture and raised components without treating missing as zeros."""
import math
import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .heightfield import integrate
from .lightmodel import solve
from .models import Problem, Region, choice, fields, number, rounded, whole
from .raster import robust_local, safe_fraction, statistics
from .results import Result
from .visual import (COOL, DIVERGE, GOLD, HOT, MINT, Extent, Trace, flatten_rgb, heatmap, overlay, plot,
                     scale_bar, support_boundary)


def _floor_remove(values, valid, sigma, preserve=.97):
    background, supported = robust_local(np.asarray(values, dtype=np.float64), valid, sigma)
    return values - background, valid & (supported >= preserve)


def _roughness(values, dx, dy):
    directional = np.gradient(values, dy, dx)
    return np.hypot(*directional)


def _local_moments(sq, keep, winsize):
    values, support = robust_local(sq, keep, winsize)
    return np.sqrt(np.maximum(values, 0.)), support


def _region_feature(values, mask, step_mm, exclude=0):
    counts = int(mask.sum())
    mean = statistics(values, mask)
    return {"observed_samples": counts, "covered_area_mm2": rounded(counts * step_mm),
            "excluded_samples": int(exclude), "median": mean["median"], "spread_mad": mean["mad"],
            "maximum": mean["max"], "p95": mean["p95"]}


def _weights(shape):
    h, w = shape
    return np.hanning(h)[:, None] * np.hanning(w)[None, :]


def _frequencies(height, step_y, width, step_x):
    y = np.fft.fftfreq(height, step_y)
    x = np.fft.rfftfreq(width, step_x)
    uu, vv = np.meshgrid(x, y)
    return np.hypot(uu, vv), np.mod(np.arctan2(vv, uu), np.pi)


def _frequency_table(energy, radii, angles, max_frequency, bands, wedges):
    selected = (radii > 0) & (radii <= max_frequency)
    ring = np.minimum((radii[selected] / max_frequency * bands).astype(int), bands - 1)
    sector = np.minimum((angles[selected] / np.pi * wedges).astype(int), wedges - 1)
    energy = energy[selected]
    by_band = np.bincount(ring, weights=energy, minlength=bands)
    by_sector = np.bincount(sector, weights=energy, minlength=wedges)
    s = by_band.sum()
    cols = []
    for i in range(bands):
        cols.append({"band": i + 1, "from_cycles_per_mm": rounded(i * max_frequency / bands),
                     "to_cycles_per_mm": rounded((i + 1) * max_frequency / bands),
                     "energy_fraction": rounded(by_band[i] / s) if s > 0 else None})
    lines = []
    for i in range(wedges):
        lines.append({"sector": i + 1, "from_degrees": rounded(i * 180 / wedges),
                      "to_degrees": rounded((i + 1) * 180 / wedges),
                      "energy_fraction": rounded(by_sector[i] / s) if s > 0 else None})
    return cols, lines, float(s)


def _frequent_texture(height, valid, scale, bands, angles):
    if min(valid.shape) < 26 or valid.mean() < .83:
        raise Problem("Find a larger nearly complete texture window for directional spectrum.")
    heights = height - np.median(height[valid])
    windows = _weights(valid.shape)
    weighted = heights * windows * valid
    energy = np.abs(np.fft.rfft2(weighted)) ** 2
    radii, orientation = _frequencies(*[height.shape[0], scale.dy, height.shape[1], scale.dx])
    upper = .5 / max(scale.dx, scale.dy)
    freq, angle, total = _frequency_table(energy, radii, orientation, upper, bands, angles)
    return freq, angle, total, upper


def _texture_settings(options):
    fields(options, "kind", "bands", "directions", "window_mm", "smoothing_mm")
    kind = choice(options.get("kind", "height"), "Texture source", ("height", "slope", "albedo"))
    bands = whole(options.get("bands", 18), "Frequency bands", 8, 30)
    directions = whole(options.get("directions", 12), "Directional groups", 6, 24)
    window = number(options.get("window_mm", 2.3), "Local RMS window", .25, 45.)
    smoothing = number(options.get("smoothing_mm", 1.4), "Background scale", .2, 28.)
    return kind, bands, directions, window, smoothing


def _source_texture(field, dataset, kind):
    if kind == "height":
        detail = integrate(field, dataset.scale)
        return detail.height, detail.visible, detail.scale, "mm"
    if kind == "slope":
        dx, dy, valid = field.gradients()
        return np.hypot(dx, dy), valid, dataset.scale, "gradient"
    return field.reflectance, field.valid, dataset.scale, "linear reflectance"


def _domain_crop(data, valid, region):
    section = region.window(data.shape[1], data.shape[0])
    return data[section], valid[section]


def texture_report(dataset, options):
    kind, bands, angles, window, smoothing = _texture_settings(options)
    field = solve(dataset)
    source, good, scale, units = _source_texture(field, dataset, kind)
    sigma_bg = (smoothing / scale.dy, smoothing / scale.dx)
    detail, mask = _floor_remove(source, good, sigma_bg)
    local, reliability = _local_moments(detail**2, mask, (window / scale.dy, window / scale.dx))
    seen = (reliability >= .94) & mask
    if seen.sum() < 48:
        raise Problem("The supported texture area is too small for this physical window.")
    display = Extent.fit(local, seen, zero="minimum")
    result = Result("texture", "Surface texture and spectral direction",
                    "Read local detail amplitude at physical scale plus the overall direction and frequency distribution.",
                    {"source": kind, "window_mm": window, "detrend_mm": smoothing, "value_unit": units,
                     "map_color_range": display.public(),
                     "local_rms": _region_feature(local, seen, scale.dx * scale.dy, good.sum() - seen.sum())})
    result.image("local-texture-rms", scale_bar(heatmap(local, seen, display, grid=True), scale.width_mm))
    details = heatmap(detail, mask, Extent.fit(detail, mask, zero="symmetric"), DIVERGE)
    result.image("texture-detail", scale_bar(details, scale.width_mm))
    patch, measured = _domain_crop(detail, mask, dataset.region)
    try:
        freq, angle, energy, upper = _frequent_texture(patch, measured, scale, bands, angles)
    except Problem as exc:
        result.warnings.append(str(exc))
    else:
        result.metrics["working_nyquist_cycles_mm"] = rounded(upper)
        result.metrics["relative_transform_energy"] = rounded(energy / max(1, patch.size))
        result.metrics["spectrum_coverage"] = rounded(measured.mean())
        result.rows.extend(freq)
        if energy > 0:
            x = np.array([.5 * (r["from_cycles_per_mm"] + r["to_cycles_per_mm"]) for r in freq])
            y = np.array([r["energy_fraction"] for r in freq])
            a = np.array([.5 * (r["from_degrees"] + r["to_degrees"]) for r in angle])
            va = np.array([r["energy_fraction"] for r in angle])
            result.image("texture-frequency", plot([Trace("Energy", x, y)], "Cycles per mm", "Fraction of energy"))
            result.image("texture-direction", plot([Trace("Frequency normal", a, va, GOLD)], "Degrees", "Fraction of energy"))
        result.table("frequency-bands")
        result.table("direction-bands", angle)
    result.warnings.append("The spectral result describes the whole chosen region. Local maps describe broadband amplitude.")
    return result


def _tile_entries(values, good, grid_rows, grid_columns, physical_dimensions):
    h, w = values.shape
    step = physical_dimensions[0] / w * physical_dimensions[1] / h
    y_breaks = np.linspace(0, h, grid_rows + 1).astype(int)
    x_breaks = np.linspace(0, w, grid_columns + 1).astype(int)
    for j, (y1, y2) in enumerate(zip(y_breaks[:-1], y_breaks[1:], strict=True), 1):
        for i, (x1, x2) in enumerate(zip(x_breaks[:-1], x_breaks[1:], strict=True), 1):
            patch = np.s_[y1:y2, x1:x2]
            visible = good[patch]
            yield (i, j, Region(x1 / w, y1 / h, x2 / w, y2 / h), patch,
                   _region_feature(values[patch], visible, step, visible.size - visible.sum()))


def tile_material(dataset, options):
    fields(options, "rows", "columns", "kind")
    rows = whole(options.get("rows", 6), "Rows", 2, 16)
    columns = whole(options.get("columns", 8), "Columns", 2, 16)
    kind = choice(options.get("kind", "albedo"), "Reading", ("albedo", "redness", "incline"))
    field = solve(dataset)
    if kind == "redness":
        values = field.rgb[..., 0] - (field.rgb[..., 1] + field.rgb[..., 2]) / 2.
    elif kind == "incline":
        values = field.angular_incline()
    else:
        values = field.reflectance
    physical = (dataset.scale.width_mm, dataset.scale.height_mm)
    entries = list(_tile_entries(values, field.valid, rows, columns, physical))
    summary = {"rows": rows, "columns": columns, "reading": kind}
    fields_by_tile = np.zeros_like(values, dtype=float)
    occupied = np.zeros_like(field.valid)
    display_entries = []
    reports = []
    for i, j, box, patch, numbers in entries:
        measured = numbers["observed_samples"]
        whole_area = measured + numbers["excluded_samples"]
        ratio = measured / max(1, whole_area)
        allowed = ratio >= .60 and measured >= 12
        reports.append({"row": j, "column": i, "boundary": box.public(), "observed_fraction": rounded(ratio),
                        "readable": bool(allowed), **(numbers if allowed else {"observed_samples": measured})})
        if allowed:
            occupied[patch] = True
            fields_by_tile[patch] = numbers["median"]
        display_entries.append(("closed", box.corners(), MINT if allowed else HOT, f"{j}.{i}"))
    span = Extent.fit(fields_by_tile, occupied)
    summary["color_readout_range"] = span.public()
    summary["tiles_with_enough_support"] = int(sum(bool(item["readable"]) for item in reports))
    result = Result("tiles", "Regional material readings",
                    "Read the same metric in a physical grid with insufficiently observed cells marked.",
                    summary, reports)
    average = heatmap(fields_by_tile, occupied, span, grid=True)
    result.image("tile-reading", scale_bar(overlay(average, display_entries), physical[0]))
    return result.table("surface-cell-data")


def _isolate(values, support, positive, cutoff, px_limit):
    raw = np.where(positive, values, -values)
    found = (raw >= cutoff) & support
    first = ndimage.binary_opening(found, np.ones((3, 3), dtype=bool))
    allowed = first & support
    labels, count = ndimage.label(allowed, np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    indices = [i for i in range(1, len(sizes)) if sizes[i] >= px_limit]
    return labels, sorted(indices, key=lambda i: -sizes[i])[:100]


def topographic_census(dataset, options):
    fields(options, "sense", "min_area", "displacement_mm", "local_scale_mm")
    sense = choice(options.get("sense", "ridge"), "Feature", ("ridge", "dent"))
    area = number(options.get("min_area", .25), "Connected area (mm²)", .02, 500.)
    cutoff = number(options.get("displacement_mm", .12), "Local relief signal (mm)", .015, 18.)
    estimate = number(options.get("local_scale_mm", 4.5), "Local zero scale (mm)", .6, 40.)
    field = solve(dataset)
    reconstructed = integrate(field, dataset.scale)
    step = reconstructed.scale
    values, supported = _floor_remove(reconstructed.height, reconstructed.visible,
                                      (estimate / step.dy, estimate / step.dx))
    positive = sense == "ridge"
    labels, chosen = _isolate(values, supported, positive, cutoff, max(5, round(area / (step.dx * step.dy))))
    output = []
    outlines = []
    size = labels.shape
    pic = np.array(heatmap(values, supported, Extent.fit(values, supported, zero="symmetric"), DIVERGE, True))
    for i, selected in enumerate(chosen, 1):
        part = labels == selected
        ys, xs = np.nonzero(part)
        sample_area = part.sum() * step.dx * step.dy
        bounds = Region(xs.min() / size[1], ys.min() / size[0], (xs.max() + 1) / size[1], (ys.max() + 1) / size[0])
        extremum = np.argmax(np.where(part, values if positive else -values, -np.inf))
        point = np.unravel_index(extremum, labels.shape)
        touches = (ndimage.binary_dilation(part) & ~supported).any()
        output.append({"feature": i, "footprint_mm2": rounded(sample_area), "peak_difference_mm": rounded(values[point]),
                       "peak_normalized": [rounded((point[1] + .5) / size[1]), rounded((point[0] + .5) / size[0])],
                       "truncated_by_support": bool(touches), "region": bounds.public()})
        pic[support_boundary(part)] = GOLD
        outlines.append(("closed", bounds.corners(), MINT, str(i)))
    result = Result("features", "Raised or indented local areas",
                    "Locate connected surface relief beyond the chosen threshold and distinguish clipped extents.",
                    {"sense": sense, "minimum_area_mm2": area, "height_cutoff_mm": cutoff,
                     "reference_radius_mm": estimate, "observed_area_fraction": rounded(supported.mean()),
                     "areas": len(output)}, output)
    result.image("connected-relief", scale_bar(overlay(Image.fromarray(pic), outlines), step.width_mm))
    result.warnings.append("These are relative local deviations. Review their original photographs before classifying damage.")
    return result.table("relief-feature-extents")
