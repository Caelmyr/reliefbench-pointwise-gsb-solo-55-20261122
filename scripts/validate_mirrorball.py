#!/usr/bin/env python3
"""Standalone validation for the mirror ball lamp inference tool.

Renders synthetic mirror ball photographs with known lamp directions, marks the
ball and highlight boxes, and checks that the recovered angles match.
"""
import math
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace")

from reliefbench.dataset import DataSet, StackFrame
from reliefbench.mirrorball import mirror_ball_report
from reliefbench.models import CaptureSpec, Region
from reliefbench.raster import RawCapture


def render_ball(lamps, size=360, centre=(0.5, 0.52), radius=0.30, clip_one=False, seed=0):
    rng = np.random.default_rng(seed)
    rgb = np.full((size, size, 3), 0.16, dtype=np.float32)
    alpha = np.ones((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    nx = (xx + 0.5) / (size - 1)
    ny = (yy + 0.5) / (size - 1)
    dx = (nx - centre[0]) / radius
    dy = (ny - centre[1]) / radius
    dist = np.hypot(dx, dy)
    inside = dist <= 1.0
    sz = np.sqrt(np.clip(1.0 - dx**2 - dy**2, 0, 1))
    # Dark ball shading facing the camera.
    shade = 0.10 + 0.12 * sz
    rgb[inside] = shade[inside, None]
    for index, (az, el) in enumerate(lamps):
        phi, theta = math.radians(az), math.radians(el)
        light = np.array([math.cos(theta)*math.cos(phi), math.cos(theta)*math.sin(phi), math.sin(theta)])
        # mirror point p = normalize(V + L)
        p = light + np.array([0, 0, 1.0])
        p /= np.linalg.norm(p)
        glow = np.exp(-((dx - p[0])**2 + (dy - p[1])**2) / (2 * 0.012**2))
        if clip_one and index == 0:
            block = glow > 0.05
            for channel in range(3):
                rgb[..., channel] = np.where(inside & block, 1.0, rgb[..., channel])
        else:
            tint = np.array([0.95, 0.9, 0.78])
            rgb = np.where(inside[..., None], rgb + (glow * 0.95)[..., None] * tint, rgb)
    rgb = np.clip(rgb + rng.normal(0, 0.002, rgb.shape), 0, 1)
    return np.round(rgb * 255).astype(np.uint8), alpha, centre, radius, inside


def make_frame(lamps, title, az, el, clip_one=False, seed=0):
    img8, alpha, centre, radius, inside = render_ball(lamps, clip_one=clip_one, seed=seed)
    rgb = img8.astype(np.float32) / 255.
    raw = RawCapture(rgb=rgb, alpha=alpha, encoding="8-bit RGB code", icc="assumed sRGB",
                     mime="image/png", bits=8, filename=f"{title}.png")
    spec = CaptureSpec.parse({"title": title, "azimuth": az, "elevation": el})
    frame = StackFrame(id=title.replace(" ", "-"), record={"id": title, "hash": "x", "version": 1,
                       "filename": raw.filename, "registration": None}, spec=spec, raw=raw,
                       registration=None, code=rgb, alpha_valid=alpha >= .98,
                       linear=rgb, clipped=(rgb >= .992).all(axis=-1))
    return frame, centre, radius


def box_around(nx, ny, half=0.035):
    return [nx - half, ny - half, nx + half, ny + half]


def mirror_point(az, el):
    phi, theta = math.radians(az), math.radians(el)
    light = np.array([math.cos(theta)*math.cos(phi), math.cos(theta)*math.sin(phi), math.sin(theta)])
    p = light + np.array([0, 0, 1.0])
    p /= np.linalg.norm(p)
    return p


def main():
    # Two lamps whose reflections fall close together on the ball: cannot tell which is which.
    lamps = [(35, 42), (48, 38)]
    frame, centre, radius = make_frame(lamps, "ambiguous pair", 35, 42, seed=1)
    points = [mirror_point(*lamp) for lamp in lamps]
    marks = [{
        "label": "Ball A", "center": list(centre), "radius": radius,
        "highlights": [box_around(centre[0] + radius * p[0], centre[1] + radius * p[1]) for p in points],
    }]
    surface = {"id": "s1", "title": "T", "accession": "", "width_mm": 100., "height_mm": 80.,
               "synthetic": True, "version": 1}
    dataset = DataSet(surface, (frame,), None, Region(0, 0, 1, 1), np.ones(frame.raw.shape, dtype=bool))
    result = mirror_ball_report(dataset, {"mirror_marks": marks})
    print("=== ambiguous case (two close lamps) ===")
    for row in result.rows:
        print({k: row[k] for k in ("status", "inferred_azimuth", "inferred_elevation",
              "azimuth_interval", "elevation_interval", "recorded_azimuth")})
    assert all(r["status"] == "ambiguous" for r in result.rows), "close highlights must be flagged ambiguous"

    # Single clear lamp, no clipping.
    lamp = (210, 55)
    frame, centre, radius = make_frame([lamp], "clear lamp", 0, 45, seed=2)
    p = mirror_point(*lamp)
    marks = [{"label": "Ball B", "center": list(centre), "radius": radius,
              "highlights": [box_around(centre[0] + radius * p[0], centre[1] + radius * p[1])]}]
    dataset = DataSet(surface, (frame,), None, Region(0, 0, 1, 1), np.ones(frame.raw.shape, dtype=bool))
    result = mirror_ball_report(dataset, {"mirror_marks": marks})
    row = result.rows[0]
    print("\n=== clear case ===")
    print({k: row[k] for k in ("status", "inferred_azimuth", "inferred_elevation",
          "azimuth_interval", "elevation_interval", "clipped_fraction")})
    assert row["status"] == "inferred"
    assert abs(row["inferred_azimuth"] - 210) < 3, row["inferred_azimuth"]
    assert abs(row["inferred_elevation"] - 55) < 3, row["inferred_elevation"]
    start, end = row["azimuth_interval"]
    offset = 360 if end < start else 0
    assert start <= row["inferred_azimuth"] <= end + offset

    # Clipped white centre, halo still visible -> bounded with a range.
    frame, centre, radius = make_frame([lamp], "clipped lamp", 0, 45, clip_one=True, seed=3)
    marks = [{"label": "Ball C", "center": list(centre), "radius": radius,
              "highlights": [box_around(centre[0] + radius * p[0], centre[1] + radius * p[1], half=0.05)]}]
    dataset = DataSet(surface, (frame,), None, Region(0, 0, 1, 1), np.ones(frame.raw.shape, dtype=bool))
    result = mirror_ball_report(dataset, {"mirror_marks": marks})
    row = result.rows[0]
    print("\n=== clipped case ===")
    print({k: row[k] for k in ("status", "inferred_azimuth", "inferred_elevation",
          "azimuth_interval", "elevation_interval", "clipped_fraction")})
    assert row["status"] == "bounded"
    assert row["clipped_fraction"] > 0.02
    assert row["azimuth_span_degrees"] > 0
    # centre estimate should still be close to the true direction.
    assert abs(row["inferred_azimuth"] - 210) < 8, row["inferred_azimuth"]
    assert abs(row["inferred_elevation"] - 55) < 8, row["inferred_elevation"]
    assert row["recorded_azimuth"] == 0 and row["recorded_elevation"] == 45

    # Assets present.
    keys = set(result.assets)
    print("\nassets:", sorted(keys))
    assert {"mirror-marks", "inferred-lamp-plate", "mirror-lamp-directions"} <= keys
    print("\nALL MIRROR BALL CHECKS PASSED")


if __name__ == "__main__":
    main()
