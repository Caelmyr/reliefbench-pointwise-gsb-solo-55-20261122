"""Editable geometry and readable physical overlays from the reconstructed surface."""
from dataclasses import dataclass
from io import StringIO
from xml.etree.ElementTree import Element, SubElement, tostring
import math

import numpy as np
from PIL import Image
from scipy import ndimage

from .heightfield import integrate
from .lightmodel import solve
from .models import Problem, choice, fields, number, rounded, whole
from .results import Result
from .visual import DIVERGE, Extent, heatmap, scale_bar


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray
    triangles: np.ndarray
    support: np.ndarray
    axial_step_mm: tuple[float, float]
    original_labels: np.ndarray

    def public(self):
        if len(self.vertices) == 0:
            raise Problem("The measured grid does not contain safe mesh faces.")
        low = self.vertices.min(axis=0)
        high = self.vertices.max(axis=0)
        sizes = self.area_statistics()
        return {"vertices": len(self.vertices), "triangles": len(self.triangles), "units": "mm",
                "axis": "right, away from top, height toward camera",
                "bounds_mm": [np.round(low, 5).tolist(), np.round(high, 5).tolist()],
                "maximum_step_mm": [round(v, 5) for v in self.axial_step_mm], **sizes}

    def area_statistics(self):
        ends = self.vertices[self.triangles]
        first = ends[:, 1] - ends[:, 0]
        second = ends[:, 2] - ends[:, 0]
        areas = np.linalg.norm(np.cross(first, second), axis=1) / 2
        return {"total_mesh_area_mm2": rounded(np.sum(areas)), "mean_face_mm2": rounded(np.mean(areas)),
                "largest_face_mm2": rounded(np.max(areas)), "mesh_shape": list(self.support.shape),
                "cells_with_observed_faces": int(np.count_nonzero(self.support))}


def _indices(length, target):
    return np.unique(np.round(np.linspace(0, length - 1, min(length, target))).astype(int))


def _all_cell_coverage(valid, labels, ys, xs):
    sum_table = np.pad(np.cumsum(np.cumsum(valid.astype(np.int64), axis=0), axis=1), ((1, 0), (1, 0)))
    l = xs[:-1][None, :]
    r = (xs[1:] + 1)[None, :]
    t = ys[:-1][:, None]
    b = (ys[1:] + 1)[:, None]
    observed = sum_table[b, r] - sum_table[t, r] - sum_table[b, l] + sum_table[t, l]
    n = (r - l) * (b - t)
    corners = labels[np.ix_(ys, xs)]
    same = (corners[:-1, :-1] == corners[1:, 1:]) & (corners[:-1, 1:] == corners[:-1, :-1])
    same &= (corners[1:, :-1] == corners[:-1, :-1]) & (corners[:-1, :-1] > 0)
    return (n == observed) & same


def _compact_mesh(grid, indices, faces, node_island):
    used = np.unique(faces.ravel())
    new = np.full(grid.size // 3, -1, dtype=np.int64)
    new[used] = np.arange(len(used))
    return grid.reshape(-1, 3)[used], new[faces], node_island.ravel()[used]


def _choose_diagonal(samples):
    top_left = samples[:-1, :-1]
    bottom_left = samples[1:, :-1]
    top_right = samples[:-1, 1:]
    bottom_right = samples[1:, 1:]
    return np.abs(top_left - bottom_right) <= np.abs(top_right - bottom_left)


def from_height(field, density, z_scale):
    width, height = field.scale.width, field.scale.height
    xa = max(3, round(density * width / max(width, height)))
    ya = max(3, round(density * height / max(width, height)))
    xs, ys = _indices(width, xa), _indices(height, ya)
    support = _all_cell_coverage(field.visible, field.island_labels, ys, xs)
    if not support.any():
        raise Problem("No sufficiently observed cells at the requested mesh division.")
    u, v = np.meshgrid((xs + .5) * field.scale.dx, (ys + .5) * field.scale.dy)
    level = field.height[np.ix_(ys, xs)] * z_scale
    points = np.stack((u, v, level), axis=-1)
    idx = np.arange(u.size).reshape(u.shape)
    l1, r1, l2, r2 = idx[:-1, :-1], idx[:-1, 1:], idx[1:, :-1], idx[1:, 1:]
    diag_a = support & _choose_diagonal(level)
    diag_b = support & ~diag_a
    faces = np.concatenate((
        np.stack((l1[diag_a], r1[diag_a], r2[diag_a]), axis=-1),
        np.stack((l1[diag_a], r2[diag_a], l2[diag_a]), axis=-1),
        np.stack((l1[diag_b], r1[diag_b], l2[diag_b]), axis=-1),
        np.stack((r1[diag_b], r2[diag_b], l2[diag_b]), axis=-1),
    ))
    vertex_islands = field.island_labels[np.ix_(ys, xs)]
    points, faces, ids = _compact_mesh(points, idx, faces, vertex_islands)
    return Mesh(points, faces, support,
                (float(np.max(np.diff(xs)) * field.scale.dx),
                 float(np.max(np.diff(ys)) * field.scale.dy)), ids)


def obj_bytes(mesh):
    file = StringIO()
    file.write("# ReliefBench relative surface, coordinates in millimetres\n")
    file.write("# Islands have independent vertical datums\n")
    for x, y, z in mesh.vertices:
        file.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
    colors = mesh.original_labels[mesh.triangles[:, 0]]
    for island in np.unique(colors):
        file.write(f"g island_{island}\n")
        for a, b, c in mesh.triangles[colors == island] + 1:
            file.write(f"f {a} {b} {c}\n")
    return file.getvalue().encode()


def ply_bytes(mesh):
    file = StringIO()
    lines = (
        "ply", "format ascii 1.0", "comment ReliefBench surface millimetres",
        "comment Independent z origin per island", f"element vertex {len(mesh.vertices)}",
        "property float x", "property float y", "property float z", "property int island",
        f"element face {len(mesh.triangles)}", "property list uchar int vertex_indices", "end_header",
    )
    file.write("\n".join(lines) + "\n")
    for (x, y, z), island in zip(mesh.vertices, mesh.original_labels, strict=True):
        file.write(f"{x:.7f} {y:.7f} {z:.7f} {island}\n")
    for a, b, c in mesh.triangles:
        file.write(f"3 {a} {b} {c}\n")
    return file.getvalue().encode()


def mesh_export(dataset, options):
    fields(options, "density", "vertical_scale", "format")
    density = whole(options.get("density", 105), "Mesh divisions", 25, 225)
    exaggerate = number(options.get("vertical_scale", 1.), "Vertical exaggeration", .2, 8.)
    format_key = choice(options.get("format", "obj"), "Mesh format", ("obj", "ply"))
    shape = solve(dataset)
    relief = integrate(shape, dataset.scale)
    mesh = from_height(relief, density, exaggerate)
    output = Result("mesh", "Open surface geometry",
                    "Export triangles only where the underlying connected height grid is observed.",
                    {**mesh.public(), "vertical_scale": exaggerate, "format": format_key,
                     "grid_to_source_units": relief.scale.public()})
    if format_key == "obj":
        output.add("mesh-file", "observed-relief.obj", "application/obj", obj_bytes(mesh))
    else:
        output.add("mesh-file", "observed-relief.ply", "application/x-ply", ply_bytes(mesh))
    view = np.asarray(heatmap(relief.height, relief.visible, Extent.fit(relief.height, relief.visible, zero="symmetric"), DIVERGE))
    evidence = Image.fromarray(np.where(mesh.support, 255, 0).astype(np.uint8))
    evidence = np.asarray(evidence.resize((view.shape[1], view.shape[0]), Image.Resampling.NEAREST))
    output.image("mesh-observed", scale_bar(Image.fromarray(np.where(evidence[..., None] > 0, view, [45, 54, 61]).astype(np.uint8)),
                                             dataset.scale.width_mm))
    output.warnings.append("The mesh is open and relative. It has no mantle, underside, filled holes or mold allowances.")
    return output


EDGE_POINTS = (
    ((0., 0.), (1., 0.)), ((1., 0.), (1., 1.)),
    ((1., 1.), (0., 1.)), ((0., 1.), (0., 0.)),
)
CORNER_EDGES = ((0, 1), (1, 2), (2, 3), (3, 0))


def _crossing(level, edge, corners):
    a, b = CORNER_EDGES[edge]
    lo, hi = corners[a], corners[b]
    xy_a, xy_b = EDGE_POINTS[edge]
    ratio = (level - lo) / (hi - lo)
    return (xy_a[0] + ratio * (xy_b[0] - xy_a[0]), xy_a[1] + ratio * (xy_b[1] - xy_a[1]))


def _march_square(level, corners):
    edges = [e for e, (a, b) in enumerate(CORNER_EDGES) if (corners[a] > level) != (corners[b] > level)]
    if len(edges) == 2:
        start, end = edges
        return [(_crossing(level, start, corners), _crossing(level, end, corners))]
    if len(edges) != 4:
        return []
    high_center = sum(corners) / 4.0 > level
    first_high = corners[0] > level
    split = [(0, 1), (2, 3)] if high_center == first_high else [(0, 3), (1, 2)]
    return [(_crossing(level, a, corners), _crossing(level, b, corners)) for a, b in split]


def _interval_levels(field, interval):
    low, high = np.percentile(field.height[field.visible], [2, 98])
    lo, hi = math.ceil(low / interval), math.floor(high / interval)
    if hi - lo >= 28:
        raise Problem("Use a larger contour spacing for this height range.")
    return [k * interval for k in range(lo, hi + 1)]


def segments(field, interval):
    sample = field.height
    valid = field.visible
    labels = field.island_labels
    all_corners = (valid[:-1, :-1] & valid[:-1, 1:] & valid[1:, 1:] & valid[1:, :-1]
                   & (labels[:-1, :-1] == labels[1:, 1:]))
    c = (sample[:-1, :-1], sample[:-1, 1:], sample[1:, 1:], sample[1:, :-1])
    lower = np.minimum.reduce(c)
    upper = np.maximum.reduce(c)
    results = {}
    for height in _interval_levels(field, interval):
        used = all_corners & (lower < height) & (upper >= height)
        yy, xx = np.nonzero(used)
        per_level = []
        for y, x in zip(yy, xx, strict=True):
            vals = [values[y, x] for values in c]
            coords = _march_square(height, vals)
            for a, b in coords:
                per_level.append((*[(x + q[0] + .5) * field.scale.dx for q in (a, b)],
                                  *[(y + q[1] + .5) * field.scale.dy for q in (a, b)],
                                  int(labels[y, x])))
        if per_level:
            results[height] = per_level
    return results


def _level_group(root, level, color):
    return SubElement(root, "g", {"data-relative-height-mm": f"{level:.5g}", "stroke": color,
                                 "fill": "none", "stroke-width": ".14"})


def contour_svg(level_rows, scale, label):
    root = Element("svg", {"xmlns": "http://www.w3.org/2000/svg",
                           "width": f"{scale.width_mm}mm", "height": f"{scale.height_mm}mm",
                           "viewBox": f"0 0 {scale.width_mm} {scale.height_mm}"})
    SubElement(root, "title").text = label
    SubElement(root, "desc").text = "Relative contour heights with an independent origin for each island."
    SubElement(root, "rect", {"x": "0", "y": "0", "width": str(scale.width_mm), "height": str(scale.height_mm),
                             "fill": "none", "stroke": "#829193", "stroke-width": ".2"})
    for height, items in sorted(level_rows.items()):
        parent = _level_group(root, height, "#a64a35" if height > 0 else ("#455055" if height == 0 else "#356d95"))
        groups = {}
        for x1, x2, y1, y2, island in items:
            groups.setdefault(island, []).append(f"M{x1:.4f},{y1:.4f}L{x2:.4f},{y2:.4f}")
        for island, parts in groups.items():
            SubElement(parent, "path", {"d": " ".join(parts), "data-island": str(island)})
    return tostring(root, encoding="utf-8", xml_declaration=True)


def contour_export(dataset, options):
    fields(options, "spacing_mm", "slope_limit")
    delta = number(options.get("spacing_mm", .25), "Contour spacing", .015, 50.)
    slope = number(options.get("slope_limit", 3.), "Maximum usable slope", .3, 8.)
    shape = solve(dataset)
    relief = integrate(shape, dataset.scale, max_slope=slope, limit=270)
    records = segments(relief, delta)
    rows = [{"height_mm": rounded(level), "segments": len(entries), "islands": sorted({k[-1] for k in entries}),
             "mm_path_length": rounded(sum(math.hypot(x1 - x2, y1 - y2) for x1, x2, y1, y2, _ in entries))}
            for level, entries in sorted(records.items())]
    result = Result("contours", "Physical-size contour tracing",
                    "Export measured height contour segments in physical coordinates without connecting across gaps.",
                    {"surface_mm": [relief.scale.width_mm, relief.scale.height_mm],
                     "relative_level_interval_mm": delta, "relative_levels": len(rows),
                     "island_count": len(relief.islands)}, rows)
    media = contour_svg(records, relief.scale, dataset.surface["title"])
    result.add("vector-contours", "observed-height-contours.svg", "image/svg+xml", media)
    rendered = np.array(heatmap(relief.height, relief.visible, Extent.fit(relief.height, relief.visible, zero="symmetric"),
                                DIVERGE, True))
    for members in records.values():
        for x1, x2, y1, y2, _ in members:
            x, y = int((x1 + x2) * .5 / relief.scale.dx), int((y1 + y2) * .5 / relief.scale.dy)
            rendered[y, x] = (250, 240, 204)
    result.image("contour-placement", scale_bar(Image.fromarray(rendered), dataset.scale.width_mm))
    return result.table("level-summary")
