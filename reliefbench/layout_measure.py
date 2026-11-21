"""Read operator marks in calibrated physical coordinates and link them to camera pixels."""
from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from .models import Point, Problem, choice, fields, list_of, number, rounded, text
from .raster import normalized_grid, sample_field, safe_fraction, statistics
from .results import Result
from .visual import COOL, GOLD, HOT, MINT, flatten_rgb, overlay, scale_bar


SHAPES = ("segment", "polyline", "polygon", "angle")


@dataclass(frozen=True)
class Figure:
    name: str
    points: tuple[Point, ...]
    shape: str

    @classmethod
    def parse(cls, raw, region):
        if not isinstance(raw, dict):
            raise Problem("Each measurement figure needs named fields.")
        fields(raw, "name", "points", "shape")
        shape = choice(raw.get("shape", "segment"), "Figure shape", SHAPES)
        counts = {"segment": (2, 2), "polyline": (2, 24), "polygon": (3, 24), "angle": (3, 3)}
        name = text(raw.get("name"), "Measurement label", 58)
        points = tuple(Point.parse(pt, name) for pt in list_of(raw.get("points"), name, *counts[shape]))
        if len(set(points)) != len(points):
            raise Problem(f"{name} has repeated mark positions.")
        if any(not region.contains(p) for p in points):
            raise Problem(f"{name} lies outside the analysis region.")
        return cls(name, points, shape)

    @property
    def unit(self):
        return np.asarray([p.public() for p in self.points])

    def physical(self, step):
        return self.unit * np.array([step.width_mm, step.height_mm])


def _cross(a, b):
    return a[0] * b[1] - b[0] * a[1]


def _proper_cross(a, b, c, d):
    spans = (_cross(b - a, c - a), _cross(b - a, d - a),
             _cross(d - c, a - c), _cross(d - c, b - c))
    return spans[0] * spans[1] < -1e-12 and spans[2] * spans[3] < -1e-12


def _simple_ring(vertices):
    closed = np.vstack((vertices, vertices[0]))
    for i in range(len(vertices)):
        for j in range(i + 2, len(vertices)):
            if i == 0 and j == len(vertices) - 1:
                continue
            if _proper_cross(closed[i], closed[i + 1], closed[j], closed[j + 1]):
                return False
    return True


def _polygon_area(vertices):
    nxt = np.roll(vertices, -1, axis=0)
    x, y = np.moveaxis(vertices, -1, 0)
    u, v = np.moveaxis(nxt, -1, 0)
    return abs(float(np.sum(x * v - y * u))) * .5


def _side_readout(fig, scale):
    xy = fig.physical(scale)
    outer = fig.shape == "polygon"
    segments = np.diff(np.vstack((xy, xy[0])) if outer else xy, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.min(lengths) < min(scale.dx, scale.dy) * .25:
        raise Problem(f"{fig.name} includes a mark interval below the acquisition sampling step.")
    data = {"total_length_mm": rounded(lengths.sum(), 5), "piece_lengths_mm": np.round(lengths, 5).tolist(),
            "span_x_mm": rounded(np.ptp(xy[:, 0]), 5), "span_y_mm": rounded(np.ptp(xy[:, 1]), 5)}
    if outer:
        if not _simple_ring(xy):
            raise Problem(f"{fig.name} overlaps itself. Trace an outline without crossings.")
        data["area_mm2"] = rounded(_polygon_area(xy), 5)
    if fig.shape == "angle":
        a, b = segments
        angle = float(np.clip(-np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1., 1.))
        data["included_angle_degrees"] = rounded(math.degrees(math.acos(angle)), 4)
    return data


def _line_walk(fig, density):
    vertices = fig.unit
    points = np.vstack((vertices, vertices[0])) if fig.shape == "polygon" else vertices
    samples = []
    for pos, (a, b) in enumerate(zip(points[:-1], points[1:], strict=True), 1):
        samples.append((pos, np.linspace(a, b, density)))
    return samples


def _inside_ring(uv, ring):
    x = uv[..., 0]
    y = uv[..., 1]
    inside = np.zeros(x.shape, dtype=bool)
    for a, b in zip(ring, np.roll(ring, -1, axis=0), strict=True):
        ax, ay = a
        bx, by = b
        if abs(by - ay) < 1e-12:
            continue
        cross = ((ay > y) != (by > y)) & (x < (bx - ax) * (y - ay) / (by - ay) + ax)
        inside ^= cross
    return inside


def _region_reads(frame, scale, inside, valid):
    mask = valid & inside
    return {"observed_fraction": rounded(safe_fraction(mask, inside.sum())),
            "observed_area_mm2": rounded(mask.sum() * scale.dx * scale.dy, 5),
            "rounded_raster_region_mm2": rounded(inside.sum() * scale.dx * scale.dy, 5),
            "visible_code_readings": statistics(frame.code_y, mask)}


def _edge_values(fig, frame, valid, n):
    members = _line_walk(fig, n)
    rows = []
    for ordinal, points in members:
        observed, good = sample_field(frame.code_y, valid, points)
        cam = frame.registration.camera(points)
        out = statistics(observed, good)
        rows.append({"measurement": fig.name, "segment": ordinal,
                     "camera_start": np.round(cam[0], 7).tolist(), "camera_end": np.round(cam[-1], 7).tolist(),
                     "available_fraction": rounded(good.mean()), "points": len(points),
                     "usable_locations": out["count"], "median_code": out["median"]})
    return rows


def _figure_vertices(figures, frame, scale):
    rows = []
    h, w = frame.raw.shape
    for fig in figures:
        measured = fig.physical(scale)
        projected = frame.registration.camera(fig.unit)
        for i, (uv, xy) in enumerate(zip(projected, measured, strict=True), 1):
            xx = uv[0] * (w - 1)
            yy = uv[1] * (h - 1)
            valid = (uv >= 0).all() and (uv <= 1).all()
            if valid:
                valid = frame.raw.alpha[round(yy), round(xx)] >= .98
            rows.append({"measurement": fig.name, "vertex": i, "panel_mm": np.round(xy, 7).tolist(),
                         "camera_coordinates": np.round(uv, 7).tolist(),
                         "camera_pixel_estimate": [rounded(xx, 4), rounded(yy, 4)],
                         "vertex_observed": bool(valid)})
    return rows


DEFAULT_FIGURES = [
    {"name": "Ruling A", "shape": "segment", "points": [[.21, .30], [.77, .32]]},
    {"name": "Observed area", "shape": "polygon", "points": [[.32, .41], [.59, .42], [.67, .59], [.34, .64]]},
    {"name": "Mark angle", "shape": "angle", "points": [[.71, .41], [.82, .50], [.80, .68]]},
]


def mark_geometry(dataset, options):
    fields(options, "figures", "segment_sample_density")
    raw = options.get("figures", DEFAULT_FIGURES)
    figures = [Figure.parse(obj, dataset.region) for obj in list_of(raw, "Measurements", 1, 16)]
    if len(set(fig.name for fig in figures)) != len(figures):
        raise Problem("Enter a distinct label for each measurement.")
    n = int(number(options.get("segment_sample_density", 120), "Segment samples", 20, 400))
    frame = dataset.one()
    valid = dataset.require(frame.alpha_valid & ~frame.clipped, "Physical marks")
    grid = normalized_grid(dataset.scale.width, dataset.scale.height)
    summary = []
    visible_edges = []
    image = flatten_rgb(frame.code, valid, False)
    camera = flatten_rgb(frame.raw.rgb, frame.raw.alpha >= .98, False)
    flat_shapes, camera_shapes = [], []
    for fig in figures:
        coordinates = fig.unit.tolist()
        col = GOLD if fig.shape == "polygon" else COOL if fig.shape == "angle" else MINT
        pattern = "closed" if fig.shape == "polygon" else "line"
        detail = _side_readout(fig, dataset.scale)
        if fig.shape == "polygon":
            region = _inside_ring(grid, fig.unit)
            detail.update(_region_reads(frame, dataset.scale, region, valid))
        plan = {"measurement": fig.name, "type": fig.shape,
                "normalized_vertices": coordinates, **detail}
        summary.append(plan)
        visible_edges.extend(_edge_values(fig, frame, valid, n))
        mapped = frame.registration.camera(coordinates)
        finite = np.isfinite(mapped).all()
        flat_shapes.append((pattern, coordinates, col, fig.name))
        if finite and np.min(mapped) >= -.04 and np.max(mapped) <= 1.04:
            camera_shapes.append((pattern, mapped.tolist(), col, fig.name))
    vertices = _figure_vertices(figures, frame, dataset.scale)
    result = Result("marks", "Physical marks and original coordinates",
                    "Measure marked spans and outlines in the plate plane with original camera witnesses.",
                    {"marks": len(figures), "original": frame.title, "scale": dataset.scale.public(),
                     "coordinate_registration": frame.registration.id}, summary)
    result.image("calibrated-measurements", scale_bar(overlay(image, flat_shapes), dataset.scale.width_mm))
    result.image("original-mark-locations", overlay(camera, camera_shapes))
    result.table("mark-measurements")
    result.table("original-vertices", vertices)
    result.table("span-support", visible_edges)
    result.warnings.append("Lengths and areas lie in the plane of the marked corners. Surface bumps do not add travel distance to these measures.")
    return result
