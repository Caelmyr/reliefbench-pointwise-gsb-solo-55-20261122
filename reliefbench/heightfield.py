"""Integrate directional gradients at physical pixel steps and preserve missing support."""
from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage, sparse
from scipy.sparse.linalg import spsolve

from .models import Problem, rounded
from .raster import statistics, sample_field


MAX_INTEGRATION_SIDE = 340
MIN_ISLAND = 16


def _edge_pairs(present, ids, dx, dy, step):
    present_x = present[:, :-1] & present[:, 1:]
    present_y = present[:-1, :] & present[1:, :]
    yids = (ids[:-1, :][present_y], ids[1:, :][present_y])
    xids = (ids[:, :-1][present_x], ids[:, 1:][present_x])
    difference_x = ((dx[:, :-1] + dx[:, 1:]) * (.5 * step.dx))[present_x]
    difference_y = ((dy[:-1, :] + dy[1:, :]) * (.5 * step.dy))[present_y]
    return ((np.r_[xids[0], yids[0]], np.r_[xids[1], yids[1]]), np.r_[difference_x, difference_y])


def _mask_trim(mask):
    labels, count = ndimage.label(mask)
    sizes = np.bincount(labels.ravel(), minlength=count + 1)
    keep = np.flatnonzero(sizes[1:] >= MIN_ISLAND) + 1
    return np.isin(labels, keep), labels, keep


def _coarsen(field, scale, limit, slope):
    dx, dy, present = field.gradients(slope)
    stride = max(1, math.ceil(max(scale.width, scale.height) / limit))
    if stride == 1:
        return dx, dy, present, scale, 1
    from .models import Scale
    smooth_support = ndimage.uniform_filter(present.astype(np.float64), size=stride, mode="nearest")
    totals = [ndimage.uniform_filter(np.where(present, g, 0.), size=stride, mode="nearest") for g in (dx, dy)]
    sampled = [np.divide(a, smooth_support, out=np.zeros_like(a), where=smooth_support > .25)[::stride, ::stride]
               for a in totals]
    support = present[::stride, ::stride] & (smooth_support[::stride, ::stride] >= .98)
    used = Scale(scale.width_mm, scale.height_mm, support.shape[1], support.shape[0])
    return *sampled, support, used, stride


def _solve_gauge(count, pair, differences, anchors):
    first, second = pair
    rows = np.r_[first, second, first, second, anchors]
    cols = np.r_[first, second, second, first, anchors]
    values = np.r_[np.ones(2 * len(first)), -np.ones(2 * len(first)), np.repeat(4., len(anchors))]
    operator = sparse.coo_matrix((values, (rows, cols)), shape=(count, count)).tocsc()
    target = np.zeros(count, dtype=float)
    np.add.at(target, first, -differences)
    np.add.at(target, second, differences)
    return spsolve(operator, target)


def _edge_mismatch(first, second, values, difference):
    residual = values[second] - values[first] - difference
    return rounded(np.sqrt(np.mean(residual**2))) if residual.size else None


def _island_table(heights, labels, keep):
    rows = []
    for n, label in enumerate(keep, start=1):
        mask = labels == label
        py, px = np.where(mask)
        h, w = mask.shape
        data = statistics(heights, mask)
        rows.append({"island": n, "samples": int(mask.sum()), "relative_min_mm": data["min"],
                     "relative_max_mm": data["max"], "median_mm": data["median"],
                     "normalized_box": [rounded(px.min() / w), rounded(py.min() / h),
                                        rounded((px.max() + 1) / w), rounded((py.max() + 1) / h)]})
    return rows


@dataclass(frozen=True)
class HeightField:
    height: np.ndarray
    visible: np.ndarray
    island_labels: np.ndarray
    scale: object
    residual_mm: float | None
    stride: int
    islands: tuple[dict, ...]

    def at(self, points):
        return sample_field(self.height, self.visible, points)

    def report(self):
        span = statistics(self.height, self.visible)
        return {"height_mm": span, "islands": list(self.islands), "height_grid": self.scale.public(),
                "integration_stride": self.stride, "edge_fit_rmse_mm": self.residual_mm,
                "observed_fraction": rounded(self.visible.mean()),
                "datum": "Each connected island has an independent median zero."}


def integrate(field, input_scale, max_slope=3, limit=MAX_INTEGRATION_SIDE):
    dx, dy, mask, scale, stride = _coarsen(field, input_scale, limit, max_slope)
    valid, labels, keep = _mask_trim(mask)
    if valid.sum() < 64:
        raise Problem("There is too little contiguous slope support to integrate height.")
    labels = np.where(valid, labels, 0)
    indices = np.full(valid.shape, -1, dtype=np.int64)
    indices[valid] = np.arange(valid.sum())
    pairs, difference = _edge_pairs(valid, indices, dx, dy, scale)
    anchor = np.asarray([indices[labels == label][0] for label in keep])
    values = _solve_gauge(int(valid.sum()), pairs, difference, anchor)
    if not np.isfinite(values).all():
        raise Problem("The height reconstruction was ill-conditioned.")
    measured = np.zeros(valid.shape, dtype=float)
    measured[valid] = values
    from_islands = np.zeros_like(labels)
    for idx, island in enumerate(keep, start=1):
        current = labels == island
        from_islands[current] = idx
        measured[current] -= np.median(measured[current])
    rows = _island_table(measured, labels, keep)
    return HeightField(measured, valid, from_islands, scale,
                       _edge_mismatch(*pairs, values, difference), stride, tuple(rows))


def contour_levels(field, interval, maximum=40):
    selected = field.height[field.visible]
    low = math.floor(float(np.percentile(selected, 2)) / interval)
    high = math.ceil(float(np.percentile(selected, 98)) / interval)
    if high - low > maximum:
        raise Problem("Increase the interval so that contours can be distinguished.")
    integer_level = np.floor(field.height / interval)
    edge_x = field.visible[:, :-1] & field.visible[:, 1:] & (integer_level[:, :-1] != integer_level[:, 1:])
    edge_y = field.visible[:-1] & field.visible[1:] & (integer_level[:-1] != integer_level[1:])
    edge = np.zeros_like(field.visible)
    edge[:, 1:] |= edge_x
    edge[1:] |= edge_y
    return edge, [rounded(k * interval) for k in range(low, high + 1)]


def derivative_consistency(field, scale):
    dx, dy, valid = field.gradients()
    curl_x = np.diff(dy, axis=1) / scale.dx
    curl_y = np.diff(dx, axis=0) / scale.dy
    center = .5 * (curl_x[:-1] + curl_x[1:]) - .5 * (curl_y[:, :-1] + curl_y[:, 1:])
    common = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    Z = np.zeros_like(valid, dtype=float)
    complete = np.zeros_like(valid)
    Z[:-1, :-1] = center
    complete[:-1, :-1] = common
    return Z, complete
