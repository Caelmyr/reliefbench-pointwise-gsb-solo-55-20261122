"""Planar registration used for source inspection and same-camera batches."""
from dataclasses import dataclass

import numpy as np

from .gridfit import GridRefinement
from .models import CropSpec, Problem, Scale, rounded
from .raster import normalized_grid, sample_camera


UNIT = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)
INVERSE_STEPS = 7


def polygon_area(points):
    return float(np.sum(points[:, 0] * np.roll(points[:, 1], -1)
                        - points[:, 1] * np.roll(points[:, 0], -1))) * .5


def _validate(points):
    edges = np.roll(points, -1, axis=0) - points
    next_edge = np.roll(edges, -1, axis=0)
    cross = edges[:, 0] * next_edge[:, 1] - edges[:, 1] * next_edge[:, 0]
    if np.min(cross) < .0003 or polygon_area(points) < .025:
        raise Problem("Set four corners clockwise from upper left around the visible surface.")
    if np.min(np.linalg.norm(edges, axis=1)) < .08:
        raise Problem("Each surface edge needs at least eight percent of the photograph span.")


def _from_corners(corners):
    _validate(corners)
    rows = []
    for (x, y), (u, v) in zip(UNIT, corners, strict=True):
        rows.extend([[x, y, 1, 0, 0, 0, -x * u, -y * u], [0, 0, 0, x, y, 1, -x * v, -y * v]])
    system = np.asarray(rows, dtype=float)
    if np.linalg.cond(system) > 1500:
        raise Problem("The photograph is too oblique to map this rectangle.")
    matrix = np.r_[np.linalg.solve(system, corners.reshape(-1)), 1.].reshape(3, 3)
    perspective = UNIT @ matrix[2, :2] + matrix[2, 2]
    if np.min(perspective) <= .09:
        raise Problem("The drawing crosses the projective horizon.")
    return matrix


def _apply(matrix, points):
    divisor = points @ matrix[2, :2] + matrix[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (points @ matrix[:2, :2].T + matrix[:2, 2]) / divisor[..., None]


@dataclass(frozen=True)
class Registration:
    id: str
    corners: np.ndarray
    to_camera: np.ndarray
    to_surface: np.ndarray
    grid: object = None

    @classmethod
    def from_data(cls, row=None):
        if row is None:
            return cls("identity", UNIT.copy(), np.eye(3), np.eye(3))
        spec = CropSpec.parse({"corners": row["corners"], "note": row["note"], "grid": row.get("grid")})
        points = np.asarray([p.public() for p in spec.corners])
        transform = _from_corners(points)
        refinement = GridRefinement.load(spec.grid).fitted(np.linalg.inv(transform)) if spec.grid is not None else None
        return cls(row["id"], points, transform, np.linalg.inv(transform), refinement)

    @property
    def has_grid(self):
        return self.grid is not None

    def camera(self, surface):
        surface = np.asarray(surface, dtype=float)
        if self.grid is None:
            return _apply(self.to_camera, surface)
        return _apply(self.to_camera, surface + self.grid.delta(surface))

    def surface(self, camera):
        camera = np.asarray(camera, dtype=float)
        base = _apply(self.to_surface, camera)
        if self.grid is None:
            return base
        # Solve t + delta(t) = base by Newton steps on the small correction.
        guess = base.copy()
        flat = guess.reshape(-1, 2)
        target = base.reshape(-1, 2)
        for _ in range(INVERSE_STEPS):
            delta = self.grid.delta(flat)
            residual = flat + delta - target
            if np.max(np.abs(residual)) < 1e-9:
                break
            step = 1e-4
            dx = (self.grid.delta(flat + [step, 0.]) - delta) / step
            dy = (self.grid.delta(flat + [0., step]) - delta) / step
            jacobian = np.empty((len(flat), 2, 2))
            jacobian[:, 0, 0] = 1. + dx[:, 0]
            jacobian[:, 1, 0] = dx[:, 1]
            jacobian[:, 0, 1] = dy[:, 0]
            jacobian[:, 1, 1] = 1. + dy[:, 1]
            try:
                increment = np.linalg.solve(jacobian, residual[..., None])[..., 0]
            except np.linalg.LinAlgError:
                increment = residual
            flat = flat - increment
        return flat.reshape(guess.shape)

    def observed_frame(self, image_shape, frame):
        height, width = image_shape
        vertices = self.camera(frame.corners())
        return {"normalized": np.round(vertices, 7).tolist(),
                "camera_pixels": np.round(vertices * [width - 1, height - 1], 2).tolist()}

    def overview(self, image_shape):
        height, width = image_shape
        pix = self.corners * [width - 1, height - 1]
        sides = np.linalg.norm(pix - np.roll(pix, -1, axis=0), axis=1)
        dense = self.camera(normalized_grid(100, 100))
        coverage = ((dense >= 0) & (dense <= 1)).all(axis=2).mean()
        report = {"camera_area_fraction": rounded(polygon_area(self.corners)),
                  "geometrically_observed_fraction": rounded(coverage),
                  "edge_spans_px": np.round(sides, 2).tolist(),
                  "forward_matrix": np.round(self.to_camera, 9).tolist(),
                  "inverse_matrix": np.round(self.to_surface, 9).tolist()}
        if self.grid is not None:
            report["interior_grid"] = self.grid_summary(image_shape)
        return report

    def grid_summary(self, image_shape=None, include_points=True):
        if self.grid is None:
            return None
        summary = dict(self.grid.summary)
        if include_points:
            summary["points"] = self.grid.diagnostics.get("points", [])
        summary["marks"] = self.grid.public()["points"]
        if image_shape is not None:
            height, width = image_shape
            summary["support_outlines_px"] = [
                np.round(self.camera(np.asarray(outline)) * [width - 1, height - 1], 2).tolist()
                for outline in summary.get("support_polygons", [])]
        summary["record"] = ("Interior plate cells corrected from "
                             f"{summary.get('control_points', 0)} marked intersections; "
                             f"corners stay fixed (corner movement "
                             f"{summary.get('corner_correction_px', 0)} px, edge "
                             f"{summary.get('edge_correction_px', 0)} px).")
        return summary


def dimensions(surface, source_shapes, max_side):
    widths, heights = zip(*[(shape[1], shape[0]) for shape in source_shapes])
    w_mm, h_mm = float(surface["width_mm"]), float(surface["height_mm"])
    px_per_mm = min(min(widths) / w_mm, min(heights) / h_mm, max_side / max(w_mm, h_mm))
    width = max(28, round(w_mm * px_per_mm))
    height = max(28, round(h_mm * px_per_mm))
    return Scale(w_mm, h_mm, width, height)


def examine_registration(raw, registration, width_mm, height_mm):
    pseudo_surface = {"width_mm": width_mm, "height_mm": height_mm}
    scale = dimensions(pseudo_surface, [raw.shape], 840)
    uv = normalized_grid(scale.width, scale.height)
    rgb, visible = sample_camera(raw, registration.camera(uv))
    ranges = registration.overview(raw.shape)
    ranges["alpha_observed_fraction"] = rounded(visible.mean())
    return rgb, visible, scale, ranges
