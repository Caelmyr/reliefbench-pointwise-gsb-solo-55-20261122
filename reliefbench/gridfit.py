"""Small, local correction from marked interior plate-grid intersections.

The four-corner homography remains the base mapping.  Marks on known plate
intersections add a compact, smoothly tapered displacement that reaches zero
before the plate border, so a small interior patch never bends an already good
edge.  A fitted result records where it actually acts -- in plate millimetres
and in camera pixels -- so later camera views and surface inspections can say
which part of the image truly used those interior cells.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage

from .models import Problem, Point, list_of, number, rounded


FORMAT = "reliefbench.interior-grid.v1"
MIN_MARKS = 3
MAX_MARKS = 24
SUB_PIXEL = .5
SUPPORT_RESOLUTION = 181
MAX_RELATIVE_RESIDUAL = .22


def apply_homography(matrix, points):
    points = np.asarray(points, dtype=float)
    divisor = points @ matrix[2, :2] + matrix[2, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        return (points @ matrix[:2, :2].T + matrix[:2, 2]) / divisor[..., None]


def _kernel(distance, radius):
    ratio = np.clip(distance / max(radius, 1e-9), 0., 1.)
    return (1. - ratio) ** 4 * (4. * ratio + 1.)


def _envelope(surface, width_mm, height_mm, margin_mm):
    """One away from the border, exactly zero on it (smoothstep)."""
    mm = np.asarray(surface, dtype=float) * [width_mm, height_mm]
    margin = max(margin_mm, 1e-9)
    room = np.minimum.reduce([mm[..., 0] / margin, (width_mm - mm[..., 0]) / margin,
                              mm[..., 1] / margin, (height_mm - mm[..., 1]) / margin])
    room = np.clip(room, 0., 1.)
    return room * room * (3. - 2. * room)


def _distances_mm(query, origins, width_mm, height_mm):
    delta = (np.asarray(query)[..., None, :] - np.asarray(origins)[None, ...]) * [width_mm, height_mm]
    return np.hypot(delta[..., 0], delta[..., 1])


def _nearest_spacing(surface, width_mm, height_mm):
    distance = _distances_mm(surface, surface, width_mm, height_mm)
    np.fill_diagonal(distance, np.inf)
    return float(np.median(np.min(distance, axis=1)))


def _residual_px(first, second, image_shape):
    scale = [image_shape[1] - 1, image_shape[0] - 1]
    return np.linalg.norm((np.asarray(first) - np.asarray(second)) * scale, axis=-1)


@dataclass(frozen=True)
class GridSubmission:
    camera: np.ndarray
    plate_mm: np.ndarray
    width_mm: float
    height_mm: float
    radius_mm: float
    edge_margin_mm: float

    @property
    def surface(self):
        return self.plate_mm / [self.width_mm, self.height_mm]

    @classmethod
    def parse(cls, raw, physical, radius=0., edge_margin=0.):
        if not isinstance(raw, dict):
            raise Problem("Interior grid marks need a named document with points.")
        if set(raw) - {"points", "radius_mm", "edge_margin_mm"}:
            raise Problem("Interior grid marks contain an unrecognized field.")
        width, height = float(physical["width_mm"]), float(physical["height_mm"])
        values = list_of(raw.get("points"), "Interior grid marks", MIN_MARKS, MAX_MARKS)
        camera, plate = [], []
        for entry in values:
            if not isinstance(entry, dict) or set(entry) != {"camera", "plate_mm"}:
                raise Problem("Each interior mark needs only a camera position and plate millimetres.")
            seen = Point.parse(entry.get("camera"), "Marked camera position")
            mm = list_of(entry.get("plate_mm"), "Plate position in mm", 2, 2)
            camera.append(seen.public())
            plate.append([number(mm[0], "Plate x in mm", 0., width),
                          number(mm[1], "Plate y in mm", 0., height)])
        camera = np.asarray(camera, dtype=float)
        plate = np.asarray(plate, dtype=float)
        if np.min(np.linalg.norm(camera[:, None] - camera[None, :], axis=-1)
                  + np.eye(len(camera)) * 9.) < .0025:
            raise Problem("Two marked camera positions are almost the same. Mark each intersection once.")
        if np.min(np.max(np.abs(plate[:, None] - plate[None, :]), axis=-1)
                  + np.eye(len(plate)) * 9.) < .05:
            raise Problem("Two interior marks carry the same plate millimetre position.")
        radius = number(raw.get("radius_mm", radius) or 0., "Interior influence radius", 0., 2. * max(width, height))
        edge_margin = number(raw.get("edge_margin_mm", edge_margin) or 0., "Interior edge margin",
                             0., min(width, height) * .25)
        if edge_margin <= 0.:
            edge_margin = float(np.clip(.02 * min(width, height), .8, 14.))
        return cls(camera, plate, width, height, radius, edge_margin)


@dataclass(frozen=True)
class GridRefinement:
    camera: np.ndarray
    plate_mm: np.ndarray
    width_mm: float
    height_mm: float
    radius_mm: float
    edge_margin_mm: float
    coefficients: np.ndarray
    diagnostics: dict = field(default_factory=dict)

    @property
    def surface(self):
        return self.plate_mm / [self.width_mm, self.height_mm]

    def delta(self, points):
        """Normalized plate-plane correction at plate coordinates."""
        points = np.asarray(points, dtype=float)
        flat = points.reshape(-1, 2)
        result = np.zeros_like(flat)
        for start in range(0, len(flat), 65536):
            block = flat[start:start + 65536]
            distance = _distances_mm(block, self.surface, self.width_mm, self.height_mm)
            weights = _kernel(distance, self.radius_mm)
            envelope = _envelope(block, self.width_mm, self.height_mm, self.edge_margin_mm)
            result[start:start + 65536] = (weights * envelope[:, None]) @ self.coefficients
        return result.reshape(points.shape)

    def shift_px(self, points, to_camera, image_shape):
        points = np.asarray(points, dtype=float)
        before = apply_homography(to_camera, points)
        after = apply_homography(to_camera, points + self.delta(points))
        scale = [image_shape[1] - 1, image_shape[0] - 1]
        return np.linalg.norm((after - before) * scale, axis=-1)

    @property
    def summary(self):
        return self.diagnostics.get("summary", {})

    def public(self):
        body = {"format": FORMAT, "width_mm": rounded(self.width_mm), "height_mm": rounded(self.height_mm),
                "radius_mm": rounded(self.radius_mm), "edge_margin_mm": rounded(self.edge_margin_mm),
                "points": [{"mark": index,
                            "camera": np.round(camera, 7).tolist(),
                            "surface": np.round(surface, 7).tolist(),
                            "plate_mm": [rounded(x), rounded(y)]}
                           for index, (camera, surface, (x, y)) in enumerate(
                               zip(self.camera, self.surface, self.plate_mm, strict=True), 1)]}
        if self.diagnostics:
            # Only the compact summary is stored with the revision; the per-mark
            # rows stay in the live preview and can be recomputed on demand.
            body["diagnostics"] = {"summary": self.summary, "points": []}
        return body

    @classmethod
    def load(cls, raw):
        if not isinstance(raw, dict):
            raise Problem("The saved interior grid alignment is not a document.")
        if raw.get("format") != FORMAT:
            raise Problem("The saved interior grid alignment has an unsupported format.")
        width = number(raw.get("width_mm"), "Saved plate width", .1, 100000.)
        height = number(raw.get("height_mm"), "Saved plate height", .1, 100000.)
        radius = number(raw.get("radius_mm"), "Saved influence radius", .1, 100000.)
        margin = number(raw.get("edge_margin_mm"), "Saved edge margin", 0., 100000.)
        entries = list_of(raw.get("points"), "Saved interior marks", MIN_MARKS, MAX_MARKS)
        camera, surface, plate = [], [], []
        for entry in entries:
            if not isinstance(entry, dict) or set(entry) - {"mark"} != {"camera", "surface", "plate_mm"}:
                raise Problem("A saved interior mark is malformed.")
            seen = Point.parse(entry.get("camera"), "Saved camera position")
            place = Point.parse(entry.get("surface"), "Saved plate position")
            mm = list_of(entry.get("plate_mm"), "Saved plate millimetres", 2, 2)
            px, py = number(mm[0], "Saved plate x", 0., width), number(mm[1], "Saved plate y", 0., height)
            if abs(place.x * width - px) > .05 or abs(place.y * height - py) > .05:
                raise Problem("A saved interior mark disagrees with its plate millimetres.")
            camera.append(seen.public())
            surface.append(place.public())
            plate.append([px, py])
        camera, surface, plate = np.asarray(camera), np.asarray(surface), np.asarray(plate)
        details = raw.get("diagnostics")
        details = details if isinstance(details, dict) else {}
        return cls(camera, plate, width, height, radius, margin, np.zeros((len(camera), 2)),
                   {"summary": details.get("summary", {}), "points": details.get("points", [])})

    def fitted(self, to_surface):
        """Restore the coefficients for a stored correction without re-diagnosing."""
        targets = apply_homography(to_surface, self.camera) - self.surface
        design = _design(self.surface, self.width_mm, self.height_mm, self.radius_mm, self.edge_margin_mm)
        ridge = 1e-7 * max(float(np.trace(design)) / max(len(self.camera), 1), 1e-9)
        coefficients = np.linalg.solve(design + np.eye(len(self.camera)) * ridge, targets)
        return GridRefinement(self.camera, self.plate_mm, self.width_mm, self.height_mm,
                              self.radius_mm, self.edge_margin_mm, coefficients, self.diagnostics)

    def solve(self, to_surface, to_camera, image_shape):
        """Fit coefficients against the base mapping, then measure what changed."""
        fitted = self.fitted(to_surface)
        diagnosed = diagnose(fitted, to_surface, to_camera, image_shape)
        return GridRefinement(self.camera, self.plate_mm, self.width_mm, self.height_mm,
                              self.radius_mm, self.edge_margin_mm, fitted.coefficients, diagnosed)


def _design(surface, width_mm, height_mm, radius_mm, margin_mm):
    distance = _distances_mm(surface, surface, width_mm, height_mm)
    return _kernel(distance, radius_mm) * _envelope(surface, width_mm, height_mm, margin_mm)[:, None]


def fit(submission, to_surface, to_camera, image_shape):
    """Fit a compact correction and describe where it really acts."""
    plate = submission.plate_mm
    centred = plate - np.median(plate, axis=0)
    spread = np.linalg.svd(centred, compute_uv=False)
    if len(spread) < 2 or spread[1] < max(.6, .02 * min(submission.width_mm, submission.height_mm)):
        raise Problem("The marked interior cells lie almost on one line. Mark intersections in two directions.")
    base_error = _residual_px(submission.camera,
                              apply_homography(to_camera, submission.surface), image_shape)
    worst = int(np.argmax(base_error))
    if base_error[worst] > MAX_RELATIVE_RESIDUAL * min(image_shape):
        raise Problem(f"Interior mark {worst + 1} sits {base_error[worst]:.1f} px from where the four corners "
                      "place its plate millimetres. Check that millimetre pair before fitting.")
    radius = submission.radius_mm
    if radius <= 0.:
        radius = float(np.clip(1.8 * _nearest_spacing(submission.surface, submission.width_mm,
                                                      submission.height_mm),
                               5., .45 * min(submission.width_mm, submission.height_mm)))
    draft = GridRefinement(submission.camera, plate, submission.width_mm, submission.height_mm,
                           radius, submission.edge_margin_mm, np.zeros((len(plate), 2)), {})
    return draft.solve(to_surface, to_camera, image_shape)


def diagnose(refinement, to_surface, to_camera, image_shape):
    before = _residual_px(refinement.camera, apply_homography(to_camera, refinement.surface), image_shape)
    moved = np.clip(refinement.surface + refinement.delta(refinement.surface), -1., 2.)
    after = _residual_px(refinement.camera, apply_homography(to_camera, moved), image_shape)
    leave_out = _leave_one_out(refinement, to_surface, to_camera, image_shape)
    suggestion = _suggestion(leave_out)
    neighbours = _neighbour_support(refinement, leave_out)
    envelope = _envelope(refinement.surface, refinement.width_mm, refinement.height_mm,
                         refinement.edge_margin_mm)
    limited = {index + 1 for index, value in enumerate(envelope) if value < .6}
    points = []
    for index, (camera, surface, plate) in enumerate(zip(refinement.camera, refinement.surface,
                                                          refinement.plate_mm, strict=True), 1):
        points.append({"mark": index, "plate_mm": [rounded(v) for v in plate],
                       "camera": np.round(camera, 7).tolist(),
                       "residual_before_px": rounded(float(before[index - 1]), 3),
                       "residual_after_px": rounded(float(after[index - 1]), 3),
                       "leave_one_out_px": rounded(float(leave_out[index - 1]), 3),
                       "neighbour_count": int(neighbours[index - 1]),
                       "near_border": index in limited,
                       "inconsistent": index == suggestion})
    support = _support(refinement, to_camera, image_shape)
    summary = {
        "control_points": len(refinement.camera),
        "radius_mm": rounded(refinement.radius_mm),
        "edge_margin_mm": rounded(refinement.edge_margin_mm),
        "rms_before_px": rounded(float(np.sqrt(np.mean(before ** 2)))),
        "rms_after_px": rounded(float(np.sqrt(np.mean(after ** 2)))),
        "max_before_px": rounded(float(np.max(before))),
        "max_after_px": rounded(float(np.max(after))),
        "suggested_exclusion": None if suggestion is None else {
            "mark": suggestion,
            "plate_mm": [rounded(v) for v in refinement.plate_mm[suggestion - 1]],
            "leave_one_out_px": rounded(float(leave_out[suggestion - 1]), 3),
            "message": (f"Interior mark {suggestion} at "
                        f"{refinement.plate_mm[suggestion - 1][0]:g}, {refinement.plate_mm[suggestion - 1][1]:g} mm "
                        f"is {leave_out[suggestion - 1]:.2f} px away from what its neighbours predict. "
                        "Drop that mark first, then preview the rest again."),
        },
        **support,
    }
    if limited:
        summary["border_marks"] = sorted(limited)
        summary["border_note"] = ("These marks sit inside the tapered border band, so their correction is "
                                  "deliberately reduced to keep the four corners and outer edge in place.")
    return {"summary": summary, "points": points}


def _neighbour_support(refinement, _errors):
    distance = _distances_mm(refinement.surface, refinement.surface,
                             refinement.width_mm, refinement.height_mm)
    return ((distance > 0) & (distance <= refinement.radius_mm)).sum(axis=1)


def _leave_one_out(refinement, to_surface, to_camera, image_shape):
    """Refit without each mark and report how far the others then miss it."""
    count = len(refinement.camera)
    errors = np.full(count, np.nan)
    if count < MIN_MARKS + 1:
        return errors
    for omitted in range(count):
        keep = np.arange(count) != omitted
        camera = refinement.camera[keep]
        surface = refinement.surface[keep]
        plate = refinement.plate_mm[keep]
        design = _design(surface, refinement.width_mm, refinement.height_mm,
                         refinement.radius_mm, refinement.edge_margin_mm)
        targets = apply_homography(to_surface, camera) - surface
        ridge = 1e-7 * max(float(np.trace(design)) / max(len(surface), 1), 1e-9)
        if np.linalg.cond(design + np.eye(len(surface)) * ridge) > 1e12:
            continue
        coefficients = np.linalg.solve(design + np.eye(len(surface)) * ridge, targets)
        trial = GridRefinement(camera, plate, refinement.width_mm, refinement.height_mm,
                               refinement.radius_mm, refinement.edge_margin_mm, coefficients, {})
        predicted = np.clip(refinement.surface[omitted] + trial.delta(refinement.surface[omitted]), -1., 2.)
        errors[omitted] = float(_residual_px(apply_homography(to_camera, predicted),
                                             refinement.camera[omitted], image_shape))
    return errors


def _suggestion(errors):
    """Name a mark only when its misfit is far outside the spread of the rest."""
    if len(errors) < 4 or not np.isfinite(errors).all():
        return None
    worst = int(np.nanargmax(errors))
    others = np.delete(errors, worst)
    middle = float(np.median(others))
    spread = 1.4826 * float(np.median(np.abs(others - middle)))
    score = (errors[worst] - middle) / max(spread, .35)
    if score >= 4. and errors[worst] >= max(8., 1.5 * middle):
        return worst + 1
    return None


def _support(refinement, to_camera, image_shape):
    x, y = np.meshgrid(np.linspace(0., 1., SUPPORT_RESOLUTION), np.linspace(0., 1., SUPPORT_RESOLUTION))
    grid = np.stack((x, y), axis=-1)
    shift = refinement.shift_px(grid, to_camera, image_shape)
    support = np.isfinite(shift) & (shift >= SUB_PIXEL)
    border = np.concatenate([
        np.stack((np.linspace(0., 1., 181), np.zeros(181)), axis=-1),
        np.stack((np.linspace(0., 1., 181), np.ones(181)), axis=-1),
        np.stack((np.zeros(181), np.linspace(0., 1., 181)), axis=-1),
        np.stack((np.ones(181), np.linspace(0., 1., 181)), axis=-1)])
    edge = refinement.shift_px(border, to_camera, image_shape)
    corners = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    frame_shift = refinement.shift_px(corners, to_camera, image_shape)
    inside = (grid[:, :, 0] >= .04) & (grid[:, :, 0] <= .96) & (grid[:, :, 1] >= .04) & (grid[:, :, 1] <= .96)
    interior = support & inside
    rows = np.nonzero(interior.any(axis=1))[0]
    columns = np.nonzero(interior.any(axis=0))[0]
    extent_mm = None
    if interior.any():
        extent_mm = [rounded(max(0., (columns.min() + .5) / SUPPORT_RESOLUTION - .5 / SUPPORT_RESOLUTION) * refinement.width_mm),
                     rounded(max(0., (rows.min() + .5) / SUPPORT_RESOLUTION - .5 / SUPPORT_RESOLUTION) * refinement.height_mm),
                     rounded(min(1., (columns.max() + .5) / SUPPORT_RESOLUTION + .5 / SUPPORT_RESOLUTION) * refinement.width_mm),
                     rounded(min(1., (rows.max() + .5) / SUPPORT_RESOLUTION + .5 / SUPPORT_RESOLUTION) * refinement.height_mm)]
    cells = _cells(interior, refinement, SUPPORT_RESOLUTION)
    return {"support_fraction": rounded(float(support.mean())),
            "max_correction_px": rounded(float(np.nanmax(shift)) if np.isfinite(shift).any() else 0.),
            "edge_correction_px": rounded(float(np.nanmax(edge)) if np.isfinite(edge).any() else 0.),
            "corner_correction_px": rounded(float(np.nanmax(frame_shift)) if np.isfinite(frame_shift).any() else 0.),
            "interior_correction_px": rounded(float(np.nanmax(np.where(inside, shift, np.nan))) if
                                              np.isfinite(shift[inside]).any() else 0.),
            "support_extent_mm": extent_mm,
            "support_cells_mm": cells,
            "support_polygons": _outlines(support)}


def _cells(interior, refinement, resolution):
    """Contiguous interior cell rectangles, in plate millimetres."""
    labels, count = ndimage.label(interior, structure=np.ones((3, 3)))
    if not count:
        return []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    minimum = max(24, int(.004 * sizes.sum()))
    step_x = refinement.width_mm / resolution
    step_y = refinement.height_mm / resolution
    cells = []
    for index in np.argsort(sizes)[::-1][:6]:
        if sizes[index] < minimum:
            continue
        ys, xs = np.nonzero(labels == index + 1)
        cells.append({"left_mm": rounded((xs.min() + .5) * step_x), "top_mm": rounded((ys.min() + .5) * step_y),
                      "right_mm": rounded((xs.max() + .5) * step_x), "bottom_mm": rounded((ys.max() + .5) * step_y),
                      "share_of_plate": rounded(float(sizes[index]) / interior.size)})
    return sorted(cells, key=lambda cell: (cell["top_mm"], cell["left_mm"]))


def _outlines(support):
    labels, count = ndimage.label(support, structure=np.ones((3, 3)))
    if not count:
        return []
    sizes = np.bincount(labels.ravel(), minlength=count + 1)[1:]
    minimum = max(12, int(.008 * sizes.sum()))
    outlines = []
    for index in np.argsort(sizes)[::-1][:4]:
        if sizes[index] < minimum:
            continue
        mask = labels == index + 1
        ys, xs = np.nonzero(mask)
        centre_x, centre_y = float(xs.mean()), float(ys.mean())
        angle = (np.arctan2(ys - centre_y, xs - centre_x) + np.pi) / (2. * np.pi)
        bins = np.minimum((angle * 64).astype(int), 63)
        radius = (xs - centre_x) ** 2 + (ys - centre_y) ** 2
        points = []
        for step in range(64):
            members = bins == step
            if not members.any():
                continue
            pick = np.flatnonzero(members)[int(np.argmax(radius[members]))]
            points.append([round((float(xs[pick]) + .5) / support.shape[1], 5),
                           round((float(ys[pick]) + .5) / support.shape[0], 5)])
        if len(points) >= 3:
            outlines.append(points)
    return outlines
