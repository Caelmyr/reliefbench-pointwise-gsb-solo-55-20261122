"""Sensor reference checks used before acquiring directional material photographs."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image
from scipy import ndimage

from .models import Problem, Region, fields, number, rounded, whole
from .raster import intensity, masked_rgba, quantize, robust_local, safe_fraction, statistics
from .results import Result
from .visual import COOL, DIVERGE, GOLD, HOT, MINT, Extent, Trace, flatten_rgb, heatmap, histogram, overlay, plot, split_visual


def _check_cameras(frames):
    filled = {f.spec.camera for f in frames if f.spec.camera}
    if len(filled) > 1:
        raise Problem("Sensor references must come from the same camera.")
    elevations = [f.spec.exposure for f in frames if f.spec.role == "sensor-dark"]
    if elevations and max(elevations) - min(elevations) > .04:
        raise Problem("Use repeated camera-dark references at the same recorded EV.")
    dimensions = {f.raw.shape for f in frames}
    if len(dimensions) != 1:
        raise Problem("Sensor reference and input must share original dimensions.")
    if any(f.registration.id != "identity" for f in frames):
        raise Problem("Use the whole-photograph view for native sensor references.")


def _padded_box(valid, x0, y0, x1, y1):
    box = np.s_[y0:y1, x0:x1]
    return box, valid[box]


def _tile_support(valid, grid):
    height, width = valid.shape
    xb = np.rint(np.linspace(0, width, grid + 1)).astype(int)
    yb = np.rint(np.linspace(0, height, grid + 1)).astype(int)
    for row in range(grid):
        for column in range(grid):
            x1, x2 = xb[column], xb[column + 1]
            y1, y2 = yb[row], yb[row + 1]
            box, seen = _padded_box(valid, x1, y1, x2, y2)
            region = Region(x1 / width, y1 / height, x2 / width, y2 / height)
            yield row, column, box, seen, region


def _baseline(captures, validity):
    if len(captures) == 1:
        brightness = captures[0]
        smooth, support = robust_local(brightness, validity[0], 2.)
        return brightness, np.abs(brightness - smooth), validity[0] & (support > .92)
    stack = np.stack([np.where(good, values, np.nan) for values, good in zip(captures, validity, strict=True)])
    common = np.all(validity, axis=0)
    safe = np.where(common[None, ...], stack, 0)
    center = np.median(safe, axis=0)
    spread = np.median(np.abs(safe - center[None, ...]), axis=0) * 1.4826
    return center, spread, common


def _dark_summary(level, residual, good, img_count):
    return {"frame_count": img_count, "visible_fraction": rounded(good.mean()),
            "background_code": statistics(level, good), "noise_code": statistics(residual, good),
            "repeatability": "temporal median absolute difference" if img_count > 1 else "spatial local absolute difference"}


def _axis_pattern(values, valid, axis):
    support = np.sum(valid, axis=axis)
    sums = np.sum(np.where(valid, values, 0.), axis=axis)
    supported = support >= .66 * valid.shape[axis]
    result = np.full(support.shape, np.nan)
    result[supported] = sums[supported] / support[supported]
    return result, supported


def _defects(raw, estimated, valid, factor):
    miss = np.where(valid, raw - estimated, 0.)
    passed = miss[valid]
    center = float(np.median(passed))
    noise = max(.001, 1.4826 * np.median(np.abs(passed - center)))
    mask = valid & (miss >= center + factor * noise)
    return mask, float(noise), float(center + factor * noise)


def dark_report(dataset, options):
    fields(options, "tiles", "hot_factor")
    frames = dataset.frames
    if any(item.spec.role != "sensor-dark" for item in frames):
        raise Problem("Select covered-lens dark captures for this check.")
    _check_cameras(frames)
    grid = whole(options.get("tiles", 8), "Dark grid size", 3, 16)
    level = number(options.get("hot_factor", 10), "Local hot cutoff, noise units", 4, 35)
    arrays = [f.code_y for f in frames]
    support = [f.alpha_valid & dataset.selection for f in frames]
    center, variation, visible = _baseline(arrays, support)
    valid = dataset.require(visible, "Camera dark reference")
    local, local_support = robust_local(center, valid, 3.)
    local_good = valid & (local_support > .95)
    hot, sigma, limit = _defects(center, local, local_good, level)
    metrics = {**_dark_summary(center, variation, valid, len(frames)),
               "distinct_high_samples": int(hot.sum()), "cutoff_code_above_background": rounded(limit),
               "local_code_noise": rounded(sigma)}
    rows = []
    guides = []
    for row, column, box, seen, area in _tile_support(valid, grid):
        enough = seen.mean() >= .56 and seen.sum() >= 16
        rows.append({"row": row + 1, "column": column + 1, "area": area.public(),
                     "enough": bool(enough), "valid_fraction": rounded(seen.mean()),
                     "bright_hits": int(hot[box].sum()) if enough else None,
                     "median_code": statistics(center[box], seen)["median"] if enough else None,
                     "noise_mad": statistics(variation[box], seen)["median"] if enough else None})
        guides.append(("closed", area.corners(), MINT if enough else GOLD, f"{row+1}.{column+1}"))
    chart = np.array(heatmap(center, valid, Extent.fit(center, valid, floor=.002), grid=True))
    chart[hot] = HOT
    summarizer = Result("dark-check", "Dark frame and fixed-pattern readout",
                        "Review dark pedestal, support and repeat variability before choosing a camera black level.",
                        metrics, rows)
    summarizer.image("dark-pedestal", overlay(Image.fromarray(chart), guides))
    row_slope, _ = _axis_pattern(center, valid, axis=1)
    col_slope, _ = _axis_pattern(center, valid, axis=0)
    summarizer.image("dark-row-pattern", plot([Trace("Mean code", np.arange(len(row_slope)), row_slope)],
                                               "Camera row index", "Code"))
    summarizer.image("dark-column-pattern", plot([Trace("Mean code", np.arange(len(col_slope)), col_slope, COOL)],
                                                  "Camera column index", "Code"))
    summarizer.warnings.append("Single dark exposure estimates spatial variation, including fixed mottling. Use repeated exposures for a temporal noise reading.")
    return summarizer.table("dark-tile-readings")


def _elliptic_rings(shape, center, aspect, count):
    y, x = np.indices(shape)
    u = (x + .5) / shape[1] - center[0]
    v = ((y + .5) / shape[0] - center[1]) / aspect
    d = np.hypot(u, v)
    max_distance = np.percentile(d, 98)
    idx = np.minimum(np.floor(d / max_distance * count), count - 1).astype(int)
    return idx, float(max_distance)


def _radial_table(values, occupied, idx, max_distance, count):
    result = []
    for k in range(count):
        region = idx == k
        usable = region & occupied
        data = statistics(values, usable) if usable.sum() >= 18 else {}
        result.append({"ring": k + 1, "normalized_radius_from": rounded(max_distance * k / count),
                       "normalized_radius_to": rounded(max_distance * (k + 1) / count),
                       "p50": data.get("median"), "p05": data.get("p05"), "p95": data.get("p95"),
                       "visible_fraction": rounded(safe_fraction(usable, region.sum())),
                       "observed_samples": int(usable.sum())})
    return result


def _find_flat_center(values, valid):
    smooth, support = robust_local(values, valid, max(values.shape) / 12)
    usable = valid & (support > .97)
    region = np.zeros(values.shape, dtype=bool)
    h, w = values.shape
    region[h // 6: h - h // 6, w // 6: w - w // 6] = True
    eligible = usable & region
    if not eligible.any():
        return (.5, .5), None
    top = np.percentile(smooth[eligible], 96)
    y, x = np.nonzero(eligible & (smooth >= top))
    return ((float(x.mean()) + .5) / w, (float(y.mean()) + .5) / h), float(top)


def flat_report(dataset, options):
    fields(options, "rings", "aspect", "reference")
    frame = dataset.one()
    if frame.spec.role not in ("diffuse", "reference"):
        raise Problem("Select a photo of a uniform matte reference for falloff.")
    rings = whole(options.get("rings", 12), "Radial divisions", 6, 32)
    aspect = number(options.get("aspect", 1.), "Elliptical ratio", .5, 2.)
    valid = dataset.require(frame.alpha_valid & ~frame.clipped, "The unclipped reference")
    signal = frame.signal
    reference = options.get("reference", [])
    from .models import Point
    center, peak = _find_flat_center(signal, valid) if reference == [] else (Point.parse(reference).public(), None)
    h, w = signal.shape
    xi, yi = center
    local_ref = Region(max(0, xi-.05), max(0, yi-.05), min(1, xi+.05), min(1, yi+.05)).window(w, h)
    if valid[local_ref].sum() < 20:
        raise Problem("There is too little valid signal around the reference center.")
    baseline = float(np.median(signal[local_ref][valid[local_ref]]))
    if baseline < .02:
        raise Problem("The center of the flat reference is too dim.")
    normalized = signal / baseline
    idx, length = _elliptic_rings(signal.shape, center, aspect, rings)
    rows = _radial_table(normalized, valid, idx, length, rings)
    used = [r for r in rows if r["p50"] is not None]
    rads = np.asarray([.5 * (r["normalized_radius_from"] + r["normalized_radius_to"]) for r in used])
    measured = np.asarray([r["p50"] for r in used])
    shade = Extent.fit(normalized - 1., valid, floor=.03, zero="symmetric")
    result = Result("flat-check", "Flat reference field and falloff",
                    "Inspect uneven illumination relative to a chosen reference location.",
                    {"center": list(center), "center_automatic_peak": rounded(peak), "center_linear_value": rounded(baseline),
                     "relative_illumination": statistics(normalized, valid), "display_deviation_range": shade.public(),
                     "valid_fraction": rounded(safe_fraction(valid, dataset.selection.sum()))}, rows)
    result.image("flat-deviation", overlay(heatmap(normalized - 1., valid, shade, DIVERGE, True),
                                           [("point", [center], GOLD, "Reference")]))
    result.image("flat-radial-profile", plot([Trace("Median", rads, measured), Trace("Reference", rads, np.ones_like(rads), GOLD)],
                                              "Relative radius", "Relative light"))
    result.warnings.append("This check measures the combined flat target, exposure and lens falloff in the current acquisition.")
    return result.table("radial-light-readings")


@dataclass(frozen=True)
class CameraCorrection:
    rgb: np.ndarray
    supported: np.ndarray
    correction_gain: np.ndarray
    clipping: np.ndarray
    lost_reference: np.ndarray
    central_median: np.ndarray

    def metrics(self):
        return {"usable_fraction": rounded(self.supported.mean()), "outside_reference_fraction": rounded(self.lost_reference.mean()),
                "would_clip_fraction": rounded(safe_fraction(self.clipping, self.supported.sum())),
                "median_flat_rgb": np.round(self.central_median, 5).tolist(),
                "gain_readout": statistics(intensity(self.correction_gain), self.supported)}


def _apply_flat(frame, dark, flat, selection, minimum, cap):
    overlap = frame.alpha_valid & dark.alpha_valid & flat.alpha_valid & selection
    dark_code = np.maximum(dark.linear, 0)
    reference = flat.linear - dark_code
    enough = overlap & ~flat.clipped & ~dark.clipped & ~frame.clipped & (reference.min(axis=2) > minimum)
    if enough.sum() < max(64, round(selection.sum() * .29)):
        raise Problem("Directional reference lacks enough unclipped non-zero flat samples.")
    target = np.median(reference[enough], axis=0)
    gain = np.divide(target, np.maximum(reference, minimum), out=np.ones_like(reference), where=enough[..., None])
    safe_gain = gain.max(axis=2) <= cap
    supported = enough & safe_gain
    corrected = np.maximum(frame.linear - dark_code, 0.) * gain
    clipping = (corrected.max(axis=2) > 1) & supported
    return CameraCorrection(corrected, supported, gain, clipping, selection & ~supported, target)


def camera_compensation(dataset, options):
    fields(options, "minimum_flat_signal", "max_gain")
    source, dark, flat = _separate_reference_roles(dataset.frames)
    _check_cameras((source, dark, flat))
    min_signal = number(options.get("minimum_flat_signal", .05), "Minimum useful flat signal", .01, .65)
    max_gain = number(options.get("max_gain", 2.1), "Maximum flat gain", 1.1, 5.)
    applied = _apply_flat(source, dark, flat, dataset.selection, min_signal, max_gain)
    dataset.require(applied.supported, "Flat-field correction")
    marked = flatten_rgb(np.clip(applied.rgb, 0, 1), applied.supported)
    result = Result("camera-correction", "Camera flat reference compensation",
                    "Preview per-channel sensor correction where both references constrain the response.",
                    {"minimum_signal": min_signal, "maximum_gain": max_gain, "input": source.title,
                     "dark": dark.title, "flat": flat.title, **applied.metrics()})
    before = flatten_rgb(source.linear, source.alpha_valid & dataset.selection)
    result.image("flat-correction-side-by-side", split_visual(before, marked, "Original linear response", "Relative correction"))
    block = intensity(applied.correction_gain) - 1
    spread = Extent.fit(block, applied.supported, zero="symmetric")
    result.image("flat-gain-map", heatmap(block, applied.supported, spread, DIVERGE, True))
    result.image("camera-compensated", masked_rgba(quantize(applied.rgb), applied.supported), high_resolution=True)
    result.warnings.append("The correction is a derived view. Original photographs and light settings are preserved.")
    return result


def _separate_reference_roles(frames):
    if len(frames) != 3:
        raise Problem("Select one primary exposure, one dark frame and one flat frame.")
    dark = [f for f in frames if f.spec.role == "sensor-dark"]
    flat = [f for f in frames if f.spec.role == "reference"]
    inputs = [f for f in frames if f.spec.role in ("diffuse", "raking")]
    if len(dark) != 1 or len(flat) != 1 or len(inputs) != 1:
        raise Problem("Label the closed-lens capture as sensor-dark and the flat image as reference.")
    return inputs[0], dark[0], flat[0]
