#!/usr/bin/env python3
"""Unit-level checks for the band reading validity rules on a fabricated heightfield."""
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reliefbench.band_section import BandSpec, read_band  # noqa: E402
from reliefbench.models import Scale  # noqa: E402


def make_field(height, visible, islands, nx=120, ny=90):
    """Minimal stand-ins for SurfaceField and HeightField used by read_band."""
    normals = np.zeros((ny, nx, 3))
    normals[..., 2] = 1.0
    field = SimpleNamespace(valid=visible, normals=normals)
    field.angular_incline = lambda: np.degrees(np.arccos(np.clip(normals[..., 2], -1, 1)))
    height_field = SimpleNamespace(height=height, visible=visible, island_labels=islands,
                                   at=lambda points: _fake_at(height, visible, points))
    return field, height_field


def _fake_at(height, visible, points):
    rows = np.clip(np.floor(points[:, 1] * height.shape[0]).astype(int), 0, height.shape[0] - 1)
    cols = np.clip(np.floor(points[:, 0] * height.shape[1]).astype(int), 0, height.shape[1] - 1)
    return height[rows, cols], visible[rows, cols]


def main():
    nx, ny = 120, 90
    scale = Scale(120.0, 90.0, nx, ny)

    # Geometry: parallel lines must be spaced by the requested physical millimetres,
    # including on a diagonal section where normalized x/y units differ in length.
    diag_dataset = SimpleNamespace(region=SimpleNamespace(contains=lambda p: True), scale=scale)
    diag_spec = BandSpec.parse(diag_dataset, {
        "start": [.2, .3], "end": [.8, .7], "samples": 60,
        "band_width_mm": 8.0, "band_lines": 5})
    from reliefbench.band_section import _parallel_lines
    lines = _parallel_lines(diag_spec, diag_spec.offsets_mm(), scale)
    gaps = []
    for k in range(len(lines) - 1):
        dx = (lines[k + 1][:, 0] - lines[k][:, 0]) * scale.width_mm
        dy = (lines[k + 1][:, 1] - lines[k][:, 1]) * scale.height_mm
        gaps.append(np.hypot(dx, dy))
    gaps = np.asarray(gaps)
    expected_gap = 8.0 / (5 - 1)
    assert np.allclose(gaps, expected_gap, atol=1e-6), gaps
    print("physical line spacing mm:", np.round(gaps.mean(axis=1), 4), "expected", expected_gap)
    footprint = diag_spec.footprint(scale)
    assert np.isfinite(footprint).all() and footprint.min() > -0.01 and footprint.max() < 1.01
    print("footprint within normalized bounds:", footprint.tolist())

    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.02, (ny, nx))
    # a groove along the horizontal centre
    groove = -0.8 * np.exp(-((np.arange(ny)[:, None] - ny // 2) / 4.0) ** 2)
    height = base + groove
    visible = np.ones((ny, nx), dtype=bool)
    islands = np.ones((ny, nx), dtype=int)

    # punch a missing-support hole across the band around x ~ 0.5
    visible[ny // 2 - 3:ny // 2 + 4, 58:64] = False
    # a separate island (opposite slope/datum) on the lower half near x ~ 0.75
    islands[ny // 2 + 2:ny // 2 + 9, 86:94] = 2
    height[ny // 2 + 2:ny // 2 + 9, 86:94] = 3.0

    field, hf = make_field(height, visible, islands, nx, ny)
    dataset = SimpleNamespace(region=SimpleNamespace(
        contains=lambda p: True), scale=Scale(120.0, 90.0, nx, ny))

    options = {"start": [.05, .5], "end": [.95, .5], "samples": 120,
               "band_width_mm": 12.0, "band_lines": 9}
    spec = BandSpec.parse(dataset, options)
    band = read_band(dataset, field, hf, spec)
    rows = band["rows"]

    reasons = {}
    for r in rows:
        reasons.setdefault(r["empty_reason"] or "usable", 0)
        reasons[r["empty_reason"] or "usable"] += 1
    print("station outcomes:", reasons)

    hole = [r for r in rows if r["empty_reason"] == "no-central-support"]
    assert hole, "central-support hole did not create empty stations"
    for r in hole:
        assert r["band_height_mm"] is None and not r["centre_supported"]
    print("no-central-support stations:", len(hole))

    slope = [r for r in rows if r["empty_reason"] == "off-slope"]
    assert slope, "opposite-slope island did not create off-slope stations"
    for r in slope:
        assert r["band_height_mm"] is None
    print("off-slope stations:", len(slope))

    # usable stations keep both traces and at least three same-island members
    usable = [r for r in rows if r["usable"]]
    assert len(usable) > 50
    for r in usable:
        assert r["band_members_measured"] >= 3
        assert r["single_line_height_mm"] is not None and r["band_height_mm"] is not None

    # Median suppression of an isolated burr on a locally flat mark floor.
    # On flat ground the centre-line spike is one outlier among several neighbours, so the
    # median must not move while the single (interpolated) line still shows the needle.
    flat = rng.normal(0, 0.01, (ny, nx))
    flat_visible = np.ones((ny, nx), dtype=bool)
    flat_islands = np.ones((ny, nx), dtype=int)
    f_field, f_hf = make_field(flat, flat_visible, flat_islands, nx, ny)
    flat_spec = BandSpec.parse(dataset, {
        "start": [.05, .5], "end": [.95, .5], "samples": 120,
        "band_width_mm": 16.0, "band_lines": 7})
    flat_band = read_band(dataset, f_field, f_hf, flat_spec)

    spiked = flat.copy()
    cy, cx = ny // 2, 40
    spiked[cy, cx] += 5.0
    s_field, s_hf = make_field(spiked, flat_visible, flat_islands, nx, ny)
    spiked_band = read_band(dataset, s_field, s_hf, flat_spec)

    finite = np.isfinite(flat_band["band_height"]) & np.isfinite(spiked_band["band_height"])
    residual = np.abs(flat_band["band_height"][finite] - spiked_band["band_height"][finite]).max()
    assert residual < 0.02, residual
    print("flat-floor band median residual under a 5 mm burr:", round(float(residual), 5), "mm")

    near = [i for i, r in enumerate(spiked_band["rows"])
            if abs(r["surface_point"][0] - (cx + .5) / nx) < .02 and r["single_line_height_mm"] is not None]
    assert near, "no single-line samples near the burr"
    lifted = sum(spiked_band["rows"][i]["single_line_height_mm"]
                 - flat_band["rows"][i]["single_line_height_mm"] for i in near) / len(near)
    assert lifted > 0.5, lifted
    print("single-line needle near burr (mean lift mm):", round(float(lifted), 4))
    print("ALL BAND RULE CHECKS PASSED")


if __name__ == "__main__":
    main()
