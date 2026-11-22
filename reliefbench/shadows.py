"""Cast-shadow projection along the measured relative-height grid.

The virtual lamp in :mod:`reliefbench.lightmodel` only darkens a surface
element from its own local normal, so the back face of a ridge and the region
the ridge blocks look the same.  This module marches a ray from each measured
cell toward the directional lamp and records *why* the light does not arrive:

* ``BACKFACE``  - the element itself faces away from the lamp (self shading);
* ``SHADOW``    - another measured rise stands on the ray to the lamp;
* ``UNCERTAIN`` - the ray crosses a cell with no measured height, or the
  boundary between independent height datums, before any measured blocker.

Uncertain rays are kept separate from measured shadows because an
unmeasured crack edge could hide a blocker that the height integration
never saw.
"""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .models import Point, Problem, rounded
from .visual import GAP, PAPER, FONT


LIT = 0
BACKFACE = 1
SHADOW = 2
UNCERTAIN = 3

BIAS_DEGREES = 1.0
_BIAS_SLOPE = math.tan(math.radians(BIAS_DEGREES))
_MIN_BIAS_MM = 0.003

LIT_TINT = (225, 228, 220)
BACKFACE_TINT = (150, 110, 196)
SHADOW_TINT = (52, 70, 110)
UNCERTAIN_TINT = (214, 150, 78)
HATCH = (58, 42, 28)

STATE_NAME = {LIT: "lit", BACKFACE: "self-backface", SHADOW: "measured-cast-shadow",
              UNCERTAIN: "uncertain-shadow"}
LEGEND = ((LIT_TINT, "Lit"), (BACKFACE_TINT, "Self backface"),
          (SHADOW_TINT, "Cast shadow"), (UNCERTAIN_TINT, "Uncertain (unmeasured gap)"),
          (GAP, "No measured height"))
LEGEND_STRIP_HEIGHT = 48


@dataclass(frozen=True)
class ShadowMap:
    state: np.ndarray
    visible: np.ndarray
    blocker_y: np.ndarray
    blocker_x: np.ndarray
    blocker_height_mm: np.ndarray
    blocker_distance_mm: np.ndarray
    crossed_gap: np.ndarray
    gap_ray_height_mm: np.ndarray
    first_gap_cell: np.ndarray
    direction: tuple
    step_mm: tuple
    ray_steps: int

    def fractions(self):
        total = max(int(self.visible.sum()), 1)
        return {
            "lit_fraction": rounded(np.count_nonzero((self.state == LIT) & self.visible) / total),
            "self_backface_fraction": rounded(np.count_nonzero((self.state == BACKFACE) & self.visible) / total),
            "measured_cast_shadow_fraction": rounded(np.count_nonzero((self.state == SHADOW) & self.visible) / total),
            "uncertain_shadow_fraction": rounded(np.count_nonzero((self.state == UNCERTAIN) & self.visible) / total),
        }


def _surface_normals(height, visible, step_y, step_x):
    gy, gx = np.gradient(height, step_y, step_x)
    normal = np.stack((-gx, -gy, np.ones_like(height)), axis=-1)
    length = np.maximum(np.linalg.norm(normal, axis=-1), 1e-12)
    normal /= length[..., None]
    return np.where(visible[..., None], normal, 0.)


def cast_shadows(heightfield, direction):
    """March every measured cell toward the lamp over the height grid.

    Rays are advanced together as long as their receiver cell still has no
    verdict.  The first sampled grid cell above the ray decides the outcome;
    an unmeasured cell before that leaves the ray permanently uncertain.
    """
    height = np.asarray(heightfield.height, dtype=np.float64)
    visible = np.asarray(heightfield.visible, dtype=bool)
    h, w = visible.shape
    step_y, step_x = heightfield.scale.dy, heightfield.scale.dx
    light = np.asarray(direction, dtype=np.float64)
    horizontal = math.hypot(light[0], light[1])
    if horizontal <= 1e-9:
        raise Problem("Cast shadows need a lamp elevation below 90 degrees.")
    dx = light[0] / horizontal
    dy = light[1] / horizontal
    rise = light[2] / horizontal
    step_length = math.hypot(dy * step_y, dx * step_x)
    max_steps = math.ceil(math.hypot(h, w)) + 2

    normals = _surface_normals(height, visible, step_y, step_x)
    backface = visible & ((normals @ light) <= 0.)

    state = np.zeros(visible.shape, dtype=np.uint8)
    state[backface] = BACKFACE
    blocker_y = np.full(visible.shape, -1, dtype=np.int64)
    blocker_x = np.full(visible.shape, -1, dtype=np.int64)
    blocker_h = np.zeros(visible.shape, dtype=np.float64)
    blocker_d = np.zeros(visible.shape, dtype=np.float64)
    crossed_gap = np.zeros(visible.shape, dtype=bool)
    gap_ray = np.full(visible.shape, np.nan, dtype=np.float64)
    first_gap = np.full((*visible.shape, 2), -1, dtype=np.int64)

    starts = np.argwhere(visible & ~backface)
    if not starts.size:
        return ShadowMap(state, visible, blocker_y, blocker_x, blocker_h, blocker_d,
                         crossed_gap, gap_ray, first_gap,
                         tuple(rounded(v) for v in light),
                         (rounded(step_y), rounded(step_x)), max_steps)
    rows = starts[:, 0].astype(np.int64)
    cols = starts[:, 1].astype(np.int64)
    px = cols.astype(np.float64)
    py = rows.astype(np.float64)
    base_z = height[rows, cols].copy()
    gap_seen = np.zeros(len(rows), dtype=bool)
    gap_height = np.full(len(rows), np.nan, dtype=np.float64)
    gap_cell_x = np.full(len(rows), -1, dtype=np.int64)
    gap_cell_y = np.full(len(rows), -1, dtype=np.int64)
    alive = np.ones(len(rows), dtype=bool)

    for k in range(1, max_steps + 1):
        if not alive.any():
            break
        idx = np.flatnonzero(alive)
        sx = px[idx] + dx * k
        sy = py[idx] + dy * k
        inbounds = (sx >= 0.) & (sx <= w - 1.) & (sy >= 0.) & (sy <= h - 1.)
        # Rays leaving the plate keep whatever evidence they gathered.
        alive_idx = idx[inbounds]
        if not alive_idx.size:
            alive[idx[~inbounds]] = False
            continue
        ix = np.rint(sx[inbounds]).astype(np.int64)
        iy = np.rint(sy[inbounds]).astype(np.int64)
        distance = k * step_length
        bias = np.full(alive_idx.shape, max(_MIN_BIAS_MM, distance * _BIAS_SLOPE), dtype=np.float64)
        ray_z = base_z[alive_idx] + rise * distance
        supported = visible[iy, ix]

        gap_here = alive_idx[~supported]
        if gap_here.size:
            gap_local = np.flatnonzero(~supported)
            first = ~gap_seen[gap_here]
            fresh = gap_here[first]
            fresh_local = gap_local[first]
            gap_height[fresh] = ray_z[fresh_local] + bias[fresh_local]
            gap_cell_x[fresh] = ix[fresh_local]
            gap_cell_y[fresh] = iy[fresh_local]
            gap_seen[gap_here] = True

        hit = np.flatnonzero(supported)
        if hit.size:
            ray_idx = alive_idx[hit]
            blocks = height[iy[hit], ix[hit]] > ray_z[hit] + bias[hit]
            blocked = ray_idx[blocks]
            if blocked.size:
                by, bx = iy[hit][blocks], ix[hit][blocks]
                state[rows[blocked], cols[blocked]] = np.where(gap_seen[blocked], UNCERTAIN, SHADOW)
                blocker_y[rows[blocked], cols[blocked]] = by
                blocker_x[rows[blocked], cols[blocked]] = bx
                blocker_h[rows[blocked], cols[blocked]] = height[by, bx]
                blocker_d[rows[blocked], cols[blocked]] = distance
                crossed_gap[rows[blocked], cols[blocked]] = gap_seen[blocked]
                gap_marked = blocked[gap_seen[blocked]]
                if gap_marked.size:
                    gap_ray[rows[gap_marked], cols[gap_marked]] = gap_height[gap_marked]
                    first_gap[rows[gap_marked], cols[gap_marked]] = np.stack(
                        (gap_cell_y[gap_marked], gap_cell_x[gap_marked]), axis=1)
                alive[blocked] = False
        # Out-of-bounds rays retire; a gap encountered earlier makes them
        # uncertain because the missing region could have hidden a blocker.
        retired = idx[~inbounds]
        uncertain_retire = retired[gap_seen[retired]]
        if uncertain_retire.size:
            state[rows[uncertain_retire], cols[uncertain_retire]] = UNCERTAIN
            crossed_gap[rows[uncertain_retire], cols[uncertain_retire]] = True
            gap_ray[rows[uncertain_retire], cols[uncertain_retire]] = gap_height[uncertain_retire]
            first_gap[rows[uncertain_retire], cols[uncertain_retire]] = np.stack(
                (gap_cell_y[uncertain_retire], gap_cell_x[uncertain_retire]), axis=1)
        alive[retired] = False

    # Rays stopped because every remaining step was out of bounds above are
    # handled; mark gap-only rays that ended inside the grid as well.
    gap_only = alive & gap_seen
    if gap_only.any():
        leftover = np.flatnonzero(gap_only)
        state[rows[leftover], cols[leftover]] = UNCERTAIN
        crossed_gap[rows[leftover], cols[leftover]] = True
        gap_ray[rows[leftover], cols[leftover]] = gap_height[leftover]
        first_gap[rows[leftover], cols[leftover]] = np.stack(
            (gap_cell_y[leftover], gap_cell_x[leftover]), axis=1)

    return ShadowMap(state, visible, blocker_y, blocker_x, blocker_h, blocker_d,
                     crossed_gap, gap_ray, first_gap,
                     tuple(rounded(v) for v in light),
                     (rounded(step_y), rounded(step_x)), max_steps)


def resample_state(shadow, shape):
    """Nearest-neighbour projection of the coarse shadow grid onto a view."""
    target_h, target_w = shape
    source_h, source_w = shadow.state.shape
    if (source_h, source_w) == (target_h, target_w):
        return shadow.state.copy(), shadow.visible.copy()
    rows = np.linspace(0., source_h - 1., target_h)
    cols = np.linspace(0., source_w - 1., target_w)
    grid = np.meshgrid(cols, rows)
    state = ndimage.map_coordinates(shadow.state, grid[::-1], order=0, mode="nearest").astype(np.uint8)
    visible = ndimage.map_coordinates(shadow.visible.astype(np.uint8), grid[::-1],
                                      order=0, mode="nearest").astype(bool)
    return state, visible


def _hatch(mask):
    h, w = mask.shape
    yy, xx = np.indices((h, w))
    return mask & (((xx + yy) % 12) < 3)


def classification_image(shadow):
    """Colour-coded map of *why* each cell receives no direct light."""
    canvas = np.full((*shadow.state.shape, 3), LIT_TINT, dtype=np.uint8)
    canvas[shadow.state == BACKFACE] = BACKFACE_TINT
    canvas[shadow.state == SHADOW] = SHADOW_TINT
    uncertain = shadow.state == UNCERTAIN
    canvas[uncertain] = UNCERTAIN_TINT
    canvas[_hatch(uncertain)] = HATCH
    checker = ((np.add.outer(np.arange(shadow.state.shape[0]), np.arange(shadow.state.shape[1])) // 5) % 2 == 0)
    canvas[~shadow.visible] = GAP
    canvas[~shadow.visible & checker] = (53, 64, 71)
    return with_legend(Image.fromarray(canvas, "RGB"), LEGEND)


def tinted_image(base_rgb, shadow, legend=True):
    """Apply the shadow classes on top of the local-normal relit view."""
    state, visible = resample_state(shadow, base_rgb.shape[:2])
    rgb = base_rgb.copy()
    rgb[~visible] = GAP
    shade = state == SHADOW
    back = state == BACKFACE
    uncertain = state == UNCERTAIN
    rgb[shade] = (rgb[shade] * 0.32).astype(np.uint8)
    rgb[back] = (rgb[back] * 0.55 + np.asarray(BACKFACE_TINT) * 0.30).astype(np.uint8)
    rgb[uncertain] = (rgb[uncertain] * 0.5 + np.asarray(UNCERTAIN_TINT) * 0.5).astype(np.uint8)
    rgb[_hatch(uncertain)] = HATCH
    picture = Image.fromarray(rgb, "RGB")
    if not legend:
        return picture
    return with_legend(picture,
                       ((LIT_TINT, "Local response"), (BACKFACE_TINT, "Self backface"),
                        (SHADOW_TINT, "Cast shadow"),
                        (UNCERTAIN_TINT, "Uncertain (unmeasured gap)"), (GAP, "No measured height")))


def with_legend(image, entries, strip=LEGEND_STRIP_HEIGHT):
    width = image.width
    plate = Image.new("RGB", (width, image.height + strip), PAPER)
    plate.paste(image, (0, strip))
    draw = ImageDraw.Draw(plate)
    rows, current, row_width = [], [], 8

    def flush():
        nonlocal current, row_width
        if current:
            rows.append(current)
        current, row_width = [], 8

    for color, label in entries:
        span = 12 + 6 + int(draw.textlength(label, font=FONT)) + 22
        if current and row_width + span > width - 4:
            flush()
        current.append((color, label, span))
        row_width += span
    flush()
    for row_index, row in enumerate(rows[:2]):
        x = 8
        y = 5 + row_index * 20
        for color, label, _span in row:
            draw.rectangle((x, y, x + 12, y + 12), fill=tuple(color), outline=(210, 216, 210))
            draw.text((x + 17, y - 1), label, fill=(228, 234, 230), font=FONT)
            x += 17 + int(draw.textlength(label, font=FONT)) + 22
    return plate


def probe_shadow(shadow, point: Point, height_at=None):
    """Return the occlusion evidence for one clicked normalized point."""
    h, w = shadow.state.shape
    col = min(w - 1, max(0, int(round(point.x * (w - 1)))))
    row = min(h - 1, max(0, int(round(point.y * (h - 1)))))
    base = {"point": point.public(), "grid_cell": [col, row],
            "height_at_probe_mm": rounded(height_at[row, col]) if height_at is not None else None}
    if not shadow.visible[row, col]:
        return {**base, "classification": "unmeasured", "occluder": None,
                "occluder_height_mm": None, "occluder_distance_mm": None,
                "ray_crossed_unmeasured_gap": None, "first_gap_point": None,
                "ray_height_at_gap_mm": None}
    code = int(shadow.state[row, col])
    has_blocker = shadow.blocker_y[row, col] >= 0
    occluder = ([rounded(shadow.blocker_x[row, col] / max(w - 1, 1)),
                 rounded(shadow.blocker_y[row, col] / max(h - 1, 1))] if has_blocker else None)
    gap_cell = shadow.first_gap_cell[row, col]
    first_gap = ([rounded(gap_cell[1] / max(w - 1, 1)), rounded(gap_cell[0] / max(h - 1, 1))]
                 if gap_cell[0] >= 0 else None)
    return {
        **base, "classification": STATE_NAME[code],
        "occluder": occluder,
        "occluder_height_mm": rounded(shadow.blocker_height_mm[row, col]) if has_blocker else None,
        "occluder_distance_mm": rounded(shadow.blocker_distance_mm[row, col]) if has_blocker else None,
        "ray_crossed_unmeasured_gap": bool(shadow.crossed_gap[row, col]),
        "first_gap_point": first_gap,
        "ray_height_at_gap_mm": (rounded(shadow.gap_ray_height_mm[row, col])
                                 if np.isfinite(shadow.gap_ray_height_mm[row, col]) else None),
    }
