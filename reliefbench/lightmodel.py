"""Directional reflectance inversion on common, registered image support."""
from dataclasses import dataclass
from itertools import combinations
import math

import numpy as np

from .models import Problem, number, rounded


CUTOFF_SIGNAL = .006
CONDITION_LIMIT = 38.
MAX_FRAMES = 12


def _conditioning(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    ratio = s[0] / s[-1] if len(s) == 3 and s[-1] > 1e-7 else np.inf
    return float(ratio), s


def directions(frames):
    return np.asarray([item.light for item in frames], dtype=np.float64)


def diversity(matrix):
    condition, values = _conditioning(matrix)
    gram = matrix.T @ matrix
    mutual = matrix @ matrix.T
    eigen = np.linalg.eigvalsh(gram)
    upper = mutual[np.triu_indices(len(matrix), 1)]
    spacing = np.degrees(np.arccos(np.clip(upper, -1, 1)))
    return {"condition": rounded(condition), "rank": int(np.linalg.matrix_rank(matrix, tol=1e-5)),
            "singular_values": np.round(values, 5).tolist(), "minimum_direction_spacing_degrees":
            rounded(float(np.min(spacing)) if spacing.size else None, 3),
            "maximum_direction_spacing_degrees": rounded(float(np.max(spacing)) if spacing.size else None, 3),
            "z_independence": rounded(eigen[0] / max(eigen[-1], 1e-10), 7)}


def _bits(mask):
    return np.einsum("i,ijk->jk", 1 << np.arange(mask.shape[0], dtype=np.uint16), mask,
                     dtype=np.uint16, optimize=True)


def _mask_groups(packed):
    order = np.argsort(packed.ravel(), kind="stable")
    sorted_values = packed.ravel()[order]
    offsets = np.flatnonzero(np.diff(sorted_values, prepend=65535))
    limits = np.r_[offsets[1:], len(order)]
    for start, end in zip(offsets, limits, strict=True):
        yield int(sorted_values[start]), order[start:end]


def matrix_for(frames, selection):
    return np.stack([frame.alpha_valid & ~frame.clipped & (frame.signal >= CUTOFF_SIGNAL) & selection
                     for frame in frames])


def _fit_block(matrix, values, rgb):
    vector = np.linalg.pinv(matrix, rcond=1e-6) @ values
    norm = np.linalg.norm(vector, axis=0)
    good = (norm >= .008) & (vector[2] > .035 * norm) & np.isfinite(vector).all(axis=0)
    at = np.maximum(norm, 1e-8)
    normal = vector.T / at[:, None]
    lambert = matrix @ normal.T
    estimate = lambert * norm[None, :]
    err = np.sqrt(np.mean((estimate - values) ** 2, axis=0))
    positive = lambert > .035
    weights = np.where(positive, lambert, 0.)
    energy = np.einsum("ij,ij->j", weights, weights)
    color = np.einsum("ij,ijk->jk", weights, rgb) / np.maximum(energy, 1e-7)[:, None]
    return normal, norm, color, err, good


@dataclass(frozen=True)
class SurfaceField:
    valid: np.ndarray
    normals: np.ndarray
    reflectance: np.ndarray
    rgb: np.ndarray
    error: np.ndarray
    support: np.ndarray
    input_count: np.ndarray
    usable_codes: np.ndarray
    global_condition: float

    def gradients(self, max_slope=3.):
        nz = np.maximum(self.normals[..., 2], .03)
        dx = -self.normals[..., 0] / nz
        dy = -self.normals[..., 1] / nz
        allowed = self.valid & (np.abs(dx) <= max_slope) & (np.abs(dy) <= max_slope)
        return dx, dy, allowed

    def predicted(self, frame):
        lambert = np.maximum(self.normals @ np.asarray(frame.light), 0.)
        return self.rgb * lambert[..., None]

    def angular_incline(self):
        return np.degrees(np.arccos(np.clip(self.normals[..., 2], -1., 1.)))

    def angular_aspect(self):
        return (np.degrees(np.arctan2(self.normals[..., 1], self.normals[..., 0])) + 360) % 360


def solve(dataset):
    frames = dataset.lamps()
    if len(frames) > MAX_FRAMES:
        raise Problem(f"Use at most {MAX_FRAMES} photographs per fit.")
    design = directions(frames)
    condition, _ = _conditioning(design)
    if condition > CONDITION_LIMIT:
        raise Problem("These light positions do not constrain the three normal directions reliably.")
    masks = matrix_for(frames, dataset.selection)
    shape = masks.shape[1:]
    packed = _bits(masks)
    signals = np.stack([frame.signal.ravel() for frame in frames])
    colors = np.stack([frame.linear.reshape(-1, 3) for frame in frames])
    n = int(np.prod(shape))
    normal = np.zeros((n, 3), dtype=np.float32)
    reflectance = np.zeros(n, dtype=np.float32)
    rgb = np.zeros((n, 3), dtype=np.float32)
    error = np.zeros(n, dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    good_support = np.zeros(n, dtype=np.uint8)
    for token, order in _mask_groups(packed):
        slots = [i for i in range(len(frames)) if token & (1 << i)]
        if len(slots) < 3:
            continue
        matrix = design[slots]
        local_condition, _ = _conditioning(matrix)
        if local_condition > CONDITION_LIMIT:
            continue
        points = np.ix_(slots, order)
        v, length, albedo, fit_error, good = _fit_block(matrix, signals[points], colors[points])
        valid[order] = good
        normal[order] = v
        reflectance[order] = length
        rgb[order] = albedo
        error[order] = fit_error
        good_support[order] = len(slots)
    valid = valid.reshape(shape)
    dataset.require(valid, "The normal fit", floor=.10)
    return SurfaceField(valid, normal.reshape((*shape, 3)), reflectance.reshape(shape), rgb.reshape((*shape, 3)),
                        error.reshape(shape), good_support.reshape(shape), masks.sum(axis=0), packed, condition)


def compare_positions(frames):
    ranked = []
    for a, b in combinations(range(len(frames)), 2):
        angle = math.degrees(math.acos(float(np.clip(np.dot(frames[a].light, frames[b].light), -1, 1))))
        ranked.append({"first_photo": frames[a].title, "second_photo": frames[b].title, "separation_degrees": rounded(angle, 3)})
    return sorted(ranked, key=lambda row: row["separation_degrees"])


def spherical_coverage(frames, steps=18):
    azimuth = np.linspace(0, 360, steps * 2, endpoint=False)
    elevation = np.linspace(7, 83, steps)
    a, e = np.meshgrid(np.radians(azimuth), np.radians(elevation))
    candidate = np.stack((np.cos(a) * np.cos(e), np.sin(a) * np.cos(e), np.sin(e)), axis=-1)
    active = directions(frames)
    affinity = np.einsum("hwi,ni->hwn", candidate, active)
    distance = np.degrees(np.arccos(np.clip(np.max(affinity, axis=-1), -1, 1)))
    index = np.unravel_index(np.argmax(distance), distance.shape)
    return distance, {"largest_sampled_gap_degrees": rounded(distance[index], 3),
                      "largest_gap_azimuth": rounded(azimuth[index[1]], 3),
                      "largest_gap_elevation": rounded(elevation[index[0]], 3)}


def color_relit(field, direction, ambient, contrast, colored=True):
    vector = np.asarray(direction, dtype=float)
    direct = np.maximum(field.normals @ vector, 0.)
    brightness = ambient + (1. - ambient) * np.power(direct, contrast)
    color = field.rgb if colored else np.ones_like(field.rgb) * .56
    return color * brightness[..., None]


def exposure_settings(body):
    return number(body.get("contrast", 1), "Lighting contrast", .5, 2.5), number(body.get("ambient", .16), "Ambient", .0, .7)
