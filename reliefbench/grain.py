"""Direction of brush grain at one chosen physical line width.

A scratch crossing the plate is much wider than the fine brushed lines.  Its
energy sits at far lower spatial frequencies, so an annular pass band tuned to
the requested line width (wavelength) leaves the scratch out while keeping the
repeating grain.  Each physical tile is read independently: tiles where two
real grain fields meet, the window is too small, or not enough fine pixels were
captured are reported as uncertain instead of being forced into a direction.
"""
import math

import numpy as np
from PIL import Image, ImageDraw

from .heightfield import integrate
from .lightmodel import solve
from .models import Problem, Region, choice, fields, number, rounded, whole
from .raster import robust_local
from .results import Result
from .visual import (DIVERGE, GAP, GOLD, HOT, MINT, Extent, heatmap, overlay, plot,
                     scale_bar, Trace)


# A line of width w repeats on a wavelength near w; one octave of wavelengths
# ([w, 2w] millimetres) is a stable band that rejects the much wider scratch.
BAND_OCTAVE = 2.
WEDGES = 72
MIN_TILE_SIDE_PX = 24
MIN_VALID_PX = 48
MIN_SUPPORT_FRACTION = .80
MIN_CYCLES = 2.2
MIN_ASPECT = .55
MIN_BAND_PROMINENCE = 1.3
MIN_PEAK_RATIO = 1.45
ORIENTATION_GROUPS = 12
DIRECTION_BREAK_DEGREES = 30.


def _grain_settings(options):
    fields(options, "kind", "line_mm", "rows", "columns", "detrend_mm", "boundary_margin_mm")
    kind = choice(options.get("kind", "slope"), "Grain source", ("height", "slope", "albedo"))
    line = number(options.get("line_mm", .8), "Brushed line width (mm)", .05, 12.)
    rows = whole(options.get("rows", 8), "Tile rows", 3, 20)
    columns = whole(options.get("columns", 10), "Tile columns", 3, 24)
    detrend = number(options.get("detrend_mm", max(1.6, 3. * line)),
                     "Background removal radius (mm)", .2, 40.)
    if detrend <= line * 1.5:
        raise Problem("Background removal radius must stay wider than the brushed line width.")
    margin = number(options.get("boundary_margin_mm", max(line, .5)),
                    "Boundary keep-out margin (mm)", 0., 12.)
    return kind, line, rows, columns, detrend, margin


def _source_field(field, dataset, kind):
    if kind == "height":
        detail = integrate(field, dataset.scale)
        return detail.height, detail.visible, detail.scale, "mm"
    if kind == "slope":
        dx, dy, valid = field.gradients()
        return np.hypot(dx, dy), valid, dataset.scale, "gradient"
    return field.reflectance, field.valid, dataset.scale, "linear reflectance"


def _detrended(source, support, scale, detrend_mm):
    sigma = (detrend_mm / scale.dy, detrend_mm / scale.dx)
    background, covered = robust_local(source.astype(np.float64), support, sigma)
    return source - background, support & (covered >= .97)


def _tile_grid(shape, rows, columns):
    height, width = shape
    y_breaks = np.linspace(0, height, rows + 1).astype(int)
    x_breaks = np.linspace(0, width, columns + 1).astype(int)
    for j, (y1, y2) in enumerate(zip(y_breaks[:-1], y_breaks[1:], strict=True), 1):
        for i, (x1, x2) in enumerate(zip(x_breaks[:-1], x_breaks[1:], strict=True), 1):
            yield i, j, Region(x1 / width, y1 / height, x2 / width, y2 / height), np.s_[y1:y2, x1:x2]


def _band_limits(line_mm):
    return 1. / (line_mm * BAND_OCTAVE), 1. / line_mm  # cycles per mm


def _annulus(radii, low, high, taper):
    """Soft annular mask so the band edge never splits one sinusoid in two."""
    mask = np.zeros_like(radii)
    mask[(radii >= low) & (radii <= high)] = 1.
    wings = ((radii > high) & (radii < high + taper)) | ((radii < low) & (radii > max(0., low - taper)))
    if np.any(wings):
        mask[wings] = .5
    mask[radii <= 0] = 0.
    return mask


def _wedge_totals(energy, angles, wedges):
    """Wavevector orientation bins over 0..180 degrees (rfft angles are
    already folded modulo pi, so opposite wavevectors share one bin)."""
    sector = np.minimum((angles / math.pi * wedges).astype(int), wedges - 1)
    return np.bincount(sector.ravel(), weights=energy.ravel(), minlength=wedges)


def _orientation_energy(values, support, scale, low_f, high_f, wedges=WEDGES):
    height, width = values.shape
    window = np.hanning(height)[:, None] * np.hanning(width)[None, :]
    signal = (values - np.median(values[support])) * window * support
    spectrum2 = np.abs(np.fft.rfft2(signal)) ** 2
    fy = np.fft.fftfreq(height, scale.dy)
    fx = np.fft.rfftfreq(width, scale.dx)
    uu, vv = np.meshgrid(fx, fy)
    radii = np.hypot(uu, vv)
    angles = np.mod(np.arctan2(vv, uu), math.pi)
    taper = .18 * (high_f - low_f)
    band = _annulus(radii, low_f, high_f, taper)
    positive = radii > 0
    total_energy = float(spectrum2[positive].sum())
    wavevector_bins = _wedge_totals(spectrum2 * band, angles, wedges)
    # Brushed lines run 90 degrees across the wavevector that builds them:
    # energy whose wavevector falls in wedge theta is relabelled as line
    # orientation theta + 90 degrees.  A positive roll shifts the contents of
    # each wedge forward by half a turn, which adds 90 degrees to its label.
    grain_bins = np.roll(wavevector_bins, wedges // 2)
    grain_angles = np.linspace(0, math.pi, wedges, endpoint=False)
    band_total = float(spectrum2[band > 0].sum())
    band_bins = float((band > 0).sum())
    # Reference level: median radial-ring density over the resolved spectrum,
    # so a strong DC/scratch peak at other radii cannot set the baseline.
    ring_index = np.floor(radii / max(high_f, 1e-9) * 12).astype(int)
    ring_density = []
    for ring in range(1, 13):
        members = positive & (ring_index == ring)
        count = float(members.sum())
        if count:
            ring_density.append(float(spectrum2[members].sum()) / count)
    reference = float(np.median(ring_density)) if ring_density else 1e-12
    prominence = (band_total / max(band_bins, 1.)) / max(reference, 1e-12)
    weighted_direction = _weighted_direction(spectrum2, angles, band)
    return grain_angles, grain_bins, band_total, total_energy, prominence, weighted_direction


def _peak_label(grain_angles, wedged):
    peak = int(np.argmax(np.maximum(wedged, 0.)))
    return math.degrees(grain_angles[peak]) % 180., peak


def _weighted_direction(spectrum2, angles, band):
    """Continuous energy-weighted line orientation inside the pass band.

    The wedge histogram only quantises orientation into coarse bins for the
    readout table.  The reported direction is a circular (vector) mean of the
    in-band Fourier energy weighted bin by bin, keeping sub-wedge precision
    and handling the 0/180 degree wrap automatically.  Brushed lines run 90
    degrees across the wavevector that builds them.
    """
    selected = band > 0
    energy = spectrum2[selected]
    wave_angles = angles[selected]
    total = float(energy.sum())
    if total <= 0.:
        return None
    doubled = 2. * wave_angles
    mean_wave = math.atan2(float(np.sum(energy * np.sin(doubled))),
                           float(np.sum(energy * np.cos(doubled))))
    return (math.degrees(mean_wave) / 2. + 90.) % 180.


def _competing_peak(wedged, peak):
    order = np.argsort(wedged)[::-1]
    for candidate in order:
        if abs(candidate - peak) >= 2 and abs(candidate - peak) <= len(wedged) - 2:
            return float(wedged[candidate])
    return 0.


def _read_tile(patch, support_patch, scale, low_f, high_f):
    height, width = patch.shape
    observed = int(support_patch.sum())
    fraction = observed / support_patch.size
    side_mm = (height * scale.dy, width * scale.dx)
    aspect = min(side_mm) / max(side_mm)
    reading = {
        "observed_samples": observed, "observed_fraction": rounded(fraction),
        "tile_size_px": [width, height], "tile_size_mm": [rounded(side_mm[1]), rounded(side_mm[0])],
        "cycles_across_tile": rounded(min(side_mm) * low_f),
        "band_energy_share": None, "band_prominence": None, "direction_degrees": None,
        "peak_to_runner_up": None, "dominant_wedge_degrees": None,
        "status": "uncertain", "uncertain_reasons": [],
        "wedge_energy_fraction": [None] * WEDGES,
    }
    reasons = reading["uncertain_reasons"]
    if min(height, width) < MIN_TILE_SIDE_PX:
        reasons.append("window_smaller_than_minimum_pixels")
    if observed < MIN_VALID_PX or fraction < MIN_SUPPORT_FRACTION:
        reasons.append("not_enough_fine_pixels")
    if aspect < MIN_ASPECT:
        reasons.append("tile_window_too_narrow")
    if reading["cycles_across_tile"] < MIN_CYCLES:
        reasons.append("window_too_short_for_line_width")
    if reasons:
        return reading
    grain_angles, wedged, band_total, all_total, prominence, weighted_direction = _orientation_energy(
        patch, support_patch, scale, low_f, high_f)
    share = band_total / max(all_total, 1e-12)
    label, peak = _peak_label(grain_angles, wedged)
    runner = _competing_peak(wedged, peak)
    ratio = float(wedged[peak]) / max(runner, 1e-12)
    wedge_fraction = wedged / max(wedged.sum(), 1e-12)
    reading.update({
        "band_energy_share": rounded(share), "band_prominence": rounded(prominence),
        "direction_degrees": rounded(weighted_direction), "peak_to_runner_up": rounded(ratio),
        "dominant_wedge_degrees": rounded(label),
        "wedge_energy_fraction": [rounded(float(v)) for v in wedge_fraction],
    })
    if prominence < MIN_BAND_PROMINENCE:
        reasons.append("little_energy_at_this_line_width")
    if ratio < MIN_PEAK_RATIO:
        reasons.append("two_directions_mixed_in_window")
    reading["status"] = "confirmed" if not reasons else "uncertain"
    return reading


def _direction_color(direction, confirmed):
    if not confirmed or direction is None:
        return GAP
    # Map the cyclic 0..180 degree line direction onto a full hue circle so
    # orthogonal grain fields get clearly separated, readable colours.
    hue = direction / 180.
    r, g, b = _hsl_to_rgb(hue, .42, .55)
    return (r, g, b)


def _hsl_to_rgb(hue, saturation, lightness):
    def tone(p, q, t):
        if t < 0.:
            t += 1.
        if t > 1.:
            t -= 1.
        if t < 1 / 6:
            return p + (q - p) * 6 * t
        if t < .5:
            return q
        if t < 2 / 3:
            return p + (q - p) * (2 / 3 - t) * 6
        return p
    q = lightness + saturation - lightness * saturation
    p = 2 * lightness - q
    return tuple(int(round(255 * tone(p, q, (hue + offset) % 1.))) for offset in (1 / 3, 0., -1 / 3))


def _contact_lines(entries):
    """Return contact segments between confirmed tiles whose grain differs."""
    grid = {(entry["row"], entry["column"]): entry for entry in entries}
    contacts = []
    for (j, i), entry in grid.items():
        if entry["status"] != "confirmed":
            continue
        for dj, di, side in ((1, 0, "h"), (0, 1, "v")):
            neighbour = grid.get((j + dj, i + di))
            if neighbour is None or neighbour["status"] != "confirmed":
                continue
            delta = abs(entry["direction_degrees"] - neighbour["direction_degrees"])
            delta = min(delta, 180. - delta)
            if delta >= DIRECTION_BREAK_DEGREES:
                contacts.append((entry, neighbour, side))
    return contacts


def _within_margin(entry, contacts, margin_mm, scale):
    """Distance in millimetres from the tile rectangle to each contact segment.
    Using the rectangle (not its centre) lets a physical keep-out margin mark
    the tiles that actually touch the grain boundary."""
    y1, y2, x1, x2 = entry["pixel_bounds"][1], entry["pixel_bounds"][3], entry["pixel_bounds"][0], entry["pixel_bounds"][2]
    tile_x0, tile_x1 = x1 * scale.dx, x2 * scale.dx
    tile_y0, tile_y1 = y1 * scale.dy, y2 * scale.dy
    for first, second, side in contacts:
        a, b = first["pixel_bounds"], second["pixel_bounds"]  # [x1,y1,x2,y2]
        if side == "h":  # shared horizontal edge: constant y, segment spans x
            line_y = .5 * (a[1] + b[1]) * scale.dy
            segment_x0 = min(a[0], b[0]) * scale.dx
            segment_x1 = max(a[2], b[2]) * scale.dx
            spans_overlap = min(tile_x1, segment_x1) >= max(tile_x0, segment_x0)
            gap = abs(line_y - np.clip(line_y, tile_y0, tile_y1))
        else:  # shared vertical edge: constant x, segment spans y
            line_x = .5 * (a[0] + b[0]) * scale.dx
            segment_y0 = min(a[1], b[1]) * scale.dy
            segment_y1 = max(a[3], b[3]) * scale.dy
            spans_overlap = min(tile_y1, segment_y1) >= max(tile_y0, segment_y0)
            gap = abs(line_x - np.clip(line_x, tile_x0, tile_x1))
        if spans_overlap and gap <= margin_mm:
            return True
    return False


def _direction_map(shape, entries):
    height, width = shape
    picture = np.full((height, width, 3), GAP, dtype=np.uint8)
    outlines = []
    for entry in entries:
        x1, y1, x2, y2 = entry["pixel_bounds"]
        confirmed = entry["status"] == "confirmed"
        picture[y1:y2, x1:x2] = _direction_color(entry["direction_degrees"], confirmed)
        if entry["near_grain_boundary"]:
            line = HOT
        elif confirmed:
            line = MINT
        else:
            line = GOLD
        left, top, right, bottom = entry["boundary"]
        corners = [[left, top], [right, top], [right, bottom], [left, bottom]]
        outlines.append(("closed", corners, line, ""))
    image = overlay(Image.fromarray(picture), outlines)
    draw = ImageDraw.Draw(image)
    for entry in entries:
        if entry["status"] != "confirmed":
            continue
        x1, y1, x2, y2 = entry["pixel_bounds"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        radians = math.radians(entry["direction_degrees"])
        reach = .34 * min(x2 - x1, y2 - y1)
        dx, dy = math.cos(radians) * reach, math.sin(radians) * reach
        draw.line((cx - dx, cy - dy, cx + dx, cy + dy), fill=(20, 28, 30), width=2)
    return image


def _direction_groups(directions, tolerance=18.):
    groups = []
    for value in sorted(directions):
        for group in groups:
            delta = abs(value - group["representative_degrees"])
            if min(delta, 180. - delta) <= tolerance:
                count = group["members"]
                group["representative_degrees"] = rounded(
                    (group["representative_degrees"] * count + value) / (count + 1))
                group["members"] = count + 1
                break
        else:
            groups.append({"representative_degrees": rounded(value), "members": 1})
    return sorted(groups, key=lambda g: -g["members"])


def _source_windows(dataset, region):
    """Where a working-surface rectangle lands in each selected original.

    The four tile corners are mapped through every frame's registration so the
    review view can show the exact camera pixels that fed the reading rather
    than the resampled working grid.
    """
    corners = np.array(region.corners(), dtype=float)
    windows = []
    for frame in dataset.frames:
        camera = frame.registration.camera(corners)
        windows.append({
            "capture": frame.id, "title": frame.title,
            "registration_id": frame.registration.id,
            "camera_corners": np.round(camera, 6).tolist(),
            "camera_bbox": [rounded(float(camera[:, 0].min())), rounded(float(camera[:, 1].min())),
                            rounded(float(camera[:, 0].max())), rounded(float(camera[:, 1].max()))],
        })
    return windows


def grain_report(dataset, options):
    kind, line_mm, rows, columns, detrend_mm, margin_mm = _grain_settings(options)
    field = solve(dataset)
    source, good, scale, units = _source_field(field, dataset, kind)
    detail, mask = _detrended(source, good, scale, detrend_mm)
    low_f, high_f = _band_limits(line_mm)
    nyquist = .5 / max(scale.dx, scale.dy)
    if high_f >= nyquist * .92:
        raise Problem("This line width is finer than the registered pixels can resolve. Use a wider line or sharper originals.")
    entries = []
    for i, j, region, patch_slice in _tile_grid(detail.shape, rows, columns):
        reading = _read_tile(detail[patch_slice], mask[patch_slice], scale, low_f, high_f)
        # Region.pixels returns (left, top, right, bottom) = (x1, y1, x2, y2).
        x1, y1, x2, y2 = region.pixels(detail.shape[1], detail.shape[0])
        reading.update({"tile": f"{j}.{i}", "row": j, "column": i,
                        "boundary": region.public(), "pixel_bounds": [x1, y1, x2, y2],
                        "near_grain_boundary": False,
                        "source_windows": _source_windows(dataset, region)})
        entries.append(reading)
    contacts = _contact_lines(entries)
    for entry in entries:
        if margin_mm > 0. and contacts and _within_margin(entry, contacts, margin_mm, scale):
            entry["near_grain_boundary"] = True
            if entry["status"] == "confirmed":
                entry["status"] = "uncertain"
                entry["uncertain_reasons"].append("inside_boundary_keep_out_margin")
    confirmed_directions = [e["direction_degrees"] for e in entries if e["status"] == "confirmed"]
    result = Result(
        "grain", "Brush grain direction at one line width",
        "Read the orientation of brushed texture inside a physical wavelength band, tile by tile, with mixed and unsupported windows marked uncertain.",
        {"source": kind, "line_width_mm": line_mm,
         "bandpass_wavelength_mm": [rounded(line_mm), rounded(line_mm * BAND_OCTAVE)],
         "bandpass_cycles_per_mm": [rounded(low_f), rounded(high_f)],
         "nyquist_cycles_per_mm": rounded(nyquist),
         "detrend_mm": detrend_mm, "boundary_margin_mm": margin_mm,
         "tile_rows": rows, "tile_columns": columns, "value_unit": units,
         "grain_boundary_contacts": len(contacts),
         "confirmed_tiles": sum(e["status"] == "confirmed" for e in entries),
         "uncertain_tiles": sum(e["status"] != "confirmed" for e in entries),
         "distinct_confirmed_directions": _direction_groups(confirmed_directions)})
    result.image("grain-direction-map", scale_bar(_direction_map(detail.shape, entries), scale.width_mm))
    result.image("grain-band-detail",
                 scale_bar(heatmap(detail, mask, Extent.fit(detail, mask, zero="symmetric"), DIVERGE),
                           scale.width_mm))
    grain_angles = np.linspace(0, 180., ORIENTATION_GROUPS, endpoint=False)
    pooled = np.zeros(ORIENTATION_GROUPS)
    for entry in entries:
        if entry["status"] == "confirmed":
            pooled[int(round(entry["direction_degrees"] / 180. * ORIENTATION_GROUPS)) % ORIENTATION_GROUPS] += 1
    if pooled.sum():
        result.image("grain-direction-summary",
                     plot([Trace("Confirmed tiles", grain_angles, pooled / pooled.sum(), GOLD)],
                          "Brush direction, degrees", "Confirmed tile fraction"))
    # Nested camera windows and the full wedge vector live only in the JSON
    # evidence document; the CSV table keeps flat scalar columns.
    table_only = ("wedge_energy_fraction", "source_windows")
    result.rows.extend({k: v for k, v in entry.items() if k not in table_only} for entry in entries)
    result.table("grain-tiles")
    document = {
        "schema": "reliefbench.grain-readings.v1", "settings": dict(result.metrics),
        "orientation_convention": "degrees clockwise from image right, modulo 180, along the brushed lines",
        "wedge_centres_degrees": [rounded(math.degrees(a)) for a in
                                  np.linspace(0, math.pi, WEDGES, endpoint=False)],
        "tiles": entries,
    }
    result.document("grain-readings", document)
    result.warnings.append("Directions describe the selected wavelength band only. Wider scratches and coarser marks are intentionally excluded.")
    if result.metrics["uncertain_tiles"]:
        result.warnings.append("Uncertain tiles mix two grain fields, lack fine pixels, or sit inside the boundary keep-out margin.")
    return result
