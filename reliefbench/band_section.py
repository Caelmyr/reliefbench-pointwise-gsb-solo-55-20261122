"""Banded cross-sections: read a narrow strip around a section line instead of one pixel line.

A single reconstructed line can catch one isolated spike (a burr) and show it as a sharp
needle on the depth curve. Reading several parallel lines across a narrow band and combining
them with a median keeps the central groove while suppressing isolated outliers.

The module also encodes the rule for where a station may be reported at all:

* the centre of the band must itself have measured support, otherwise the station is empty;
* the measured band members used for the median must stay on the same connected height
  island as the centre, otherwise the section has drifted onto the opposite slope and the
  station is empty;
* partial bands (one side missing) are accepted as long as the two rules above hold.
"""
from dataclasses import dataclass

import numpy as np

from .models import Point, Problem, number, rounded, whole
from .raster import line_samples, sample_field


MIN_BAND_LINES = 3
MAX_BAND_LINES = 15
MIN_MEDIAN_MEMBERS = 3


@dataclass(frozen=True)
class BandSpec:
    start: Point
    end: Point
    samples: int
    width_mm: float
    lines: int

    @classmethod
    def parse(cls, dataset, options):
        start = Point.parse(options.get("start", [.16, .48]), "Section start")
        end = Point.parse(options.get("end", [.84, .52]), "Section end")
        if not (dataset.region.contains(start) and dataset.region.contains(end)):
            raise Problem("Keep both section endpoints in the analysis rectangle.")
        length = dataset.scale.distance(start, end)
        if length < .2:
            raise Problem("Place the section endpoints further apart.")
        samples = whole(options.get("samples", 210), "Section samples", 24, 720)
        width = number(options.get("band_width_mm", 1.2), "Band width, mm", .05, 60.)
        if width > length * .45:
            raise Problem("Band width must stay well below the section length.")
        lines = whole(options.get("band_lines", 5), "Band lines", MIN_BAND_LINES, MAX_BAND_LINES)
        if lines % 2 == 0:
            raise Problem("Use an odd number of band lines so the central line is included.")
        return cls(start, end, samples, width, lines)

    def offsets_mm(self):
        """Signed perpendicular offsets of each parallel line, centred on zero."""
        return np.linspace(-self.width_mm / 2., self.width_mm / 2., self.lines)

    def footprint(self, scale):
        """Normalized corners of the strip rectangle, ordered around the perimeter."""
        a = np.array(self.start.public(), dtype=float)
        b = np.array(self.end.public(), dtype=float)
        direction = b - a
        perp_norm = np.array([-direction[1], direction[0]]) / np.hypot(*direction)
        half = perp_norm * (self.width_mm / 2.) / np.array([scale.width_mm, scale.height_mm])
        return np.vstack((a + half, b + half, b - half, a - half))


def _parallel_lines(spec, offsets, scale):
    """Sample points for every parallel band line, shape (lines, samples, 2).

    Offsets are physical millimetres measured perpendicular to the centre line.
    """
    base = line_samples(spec.start, spec.end, spec.samples)
    direction = base[-1] - base[0]
    perp_norm = np.array([-direction[1], direction[0]]) / np.hypot(*direction)
    unit_px = np.array([scale.width_mm, scale.height_mm])
    return np.stack([base + perp_norm * (offset / unit_px) for offset in offsets])


def _grid_members(points, shape):
    """Integer grid cell (row, col) under each sample for direct raster reads."""
    rows = np.clip(np.floor(points[..., 1] * shape[0]).astype(int), 0, shape[0] - 1)
    cols = np.clip(np.floor(points[..., 0] * shape[1]).astype(int), 0, shape[1] - 1)
    return rows, cols


def _station_median(height, visible, islands, points, centre_island):
    """Median of the measured members that share the centre line's island."""
    rows, cols = _grid_members(points, visible.shape)
    same_island = visible[rows, cols] & (islands[rows, cols] == centre_island)
    values = height[rows, cols][same_island]
    other_slope = bool(np.any(visible[rows, cols] & (islands[rows, cols] != centre_island)))
    if values.size < MIN_MEDIAN_MEMBERS:
        return None, int(same_island.sum()), other_slope
    return float(np.median(values)), int(same_island.sum()), other_slope


def read_band(dataset, field, height_field, spec):
    """Read the banded section and the original single-line section together."""
    offsets = spec.offsets_mm()
    line_points = _parallel_lines(spec, offsets, dataset.scale)
    centre_index = len(offsets) // 2
    centre_points = line_points[centre_index]
    distance = np.linspace(0, dataset.scale.distance(spec.start, spec.end), spec.samples)

    centre_values, centre_supported = height_field.at(centre_points)
    _, island_sampled = sample_field(height_field.island_labels, height_field.visible, centre_points)
    centre_rows, centre_cols = _grid_members(centre_points, height_field.visible.shape)
    centre_island = height_field.island_labels[centre_rows, centre_cols]
    incline, angle_ok = sample_field(field.angular_incline(), field.valid, centre_points)

    band_height = np.full(spec.samples, np.nan)
    rows = []
    centre_kept = 0
    off_slope = 0
    no_centre = 0
    sparse = 0
    member_counts = np.zeros(spec.samples, dtype=int)

    for i in range(spec.samples):
        island = int(centre_island[i])
        has_centre = bool(centre_supported[i] and island_sampled[i] and island > 0)
        median_value, same_island_members, other_slope = _station_median(
            height_field.height, height_field.visible, height_field.island_labels,
            line_points[:, i, :], island)
        member_counts[i] = same_island_members

        gr, gc = _grid_members(line_points[:, i, :], height_field.visible.shape)
        available = int(height_field.visible[gr, gc].sum())

        reason = None
        if not has_centre:
            reason = "no-central-support"
            no_centre += 1
        elif median_value is None:
            reason = "too-few-band-members"
            sparse += 1
        elif other_slope:
            reason = "off-slope"
            off_slope += 1
        usable = reason is None
        if usable:
            band_height[i] = median_value
            centre_kept += 1

        rows.append({
            "station": i,
            "distance_mm": rounded(distance[i]),
            "surface_point": [rounded(x) for x in centre_points[i]],
            "connected_island": island if island > 0 else None,
            "single_line_height_mm": rounded(centre_values[i]) if centre_supported[i] else None,
            "band_height_mm": rounded(band_height[i]) if usable else None,
            "incline_deg": rounded(incline[i]) if (angle_ok[i] and centre_supported[i]) else None,
            "usable": bool(usable),
            "centre_supported": has_centre,
            "band_members_measured": same_island_members,
            "band_members_available": available,
            "empty_reason": reason,
        })

    line_rows, line_cols = _grid_members(line_points, height_field.visible.shape)
    line_visible = height_field.visible[line_rows, line_cols]
    line_islands = height_field.island_labels[line_rows, line_cols]
    line_heights = height_field.height[line_rows, line_cols]

    geometry = {
        "offsets_mm": [rounded(float(v)) for v in offsets],
        "line_points": [[[rounded(x) for x in p] for p in line] for line in line_points],
        "footprint": [[rounded(x) for x in p] for p in spec.footprint(dataset.scale)],
        "line_visible": line_visible.astype(bool).tolist(),
        "line_islands": line_islands.astype(int).tolist(),
        "line_heights": [[rounded(float(v)) if ok else None for v, ok in zip(heights, seen, strict=True)]
                         for heights, seen in zip(line_heights, line_visible, strict=True)],
    }

    if centre_kept < 8:
        raise Problem("The band misses reconstructed central support along most of this section.")

    positive = member_counts[member_counts > 0]
    metrics = {
        "section_length_mm": rounded(distance[-1]),
        "band_width_mm": rounded(spec.width_mm),
        "band_lines": spec.lines,
        "stations": spec.samples,
        "usable_band_stations": centre_kept,
        "empty_stations": spec.samples - centre_kept,
        "no_central_support_stations": no_centre,
        "off_slope_stations": off_slope,
        "sparse_band_stations": sparse,
        "median_band_members_per_station": rounded(float(np.median(positive))) if positive.size else 0,
        "connected_islands_along_section": sorted({r["connected_island"] for r in rows if r["usable"]}),
    }
    return {
        "rows": rows,
        "geometry": geometry,
        "metrics": metrics,
        "distance": distance,
        "centre_values": centre_values,
        "centre_supported": centre_supported,
        "band_height": band_height,
        "incline": incline,
        "angle_ok": angle_ok,
        "spec": spec,
    }
