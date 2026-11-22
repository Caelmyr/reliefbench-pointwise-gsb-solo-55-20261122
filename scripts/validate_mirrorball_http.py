#!/usr/bin/env python3
"""End-to-end check of the mirror-ball tool through the Flask HTTP API."""
import io
import math
import sys
import tempfile

import numpy as np
from PIL import Image

sys.path.insert(0, "/workspace")

from reliefbench.http import create_app


def render(az, el, size=300, centre=(0.5, 0.5), radius=0.32):
    rgb = np.full((size, size, 3), 0.14, dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    nx = (xx + .5) / (size - 1)
    ny = (yy + .5) / (size - 1)
    dx = (nx - centre[0]) / radius
    dy = (ny - centre[1]) / radius
    inside = dx**2 + dy**2 <= 1.
    sz = np.sqrt(np.clip(1. - dx**2 - dy**2, 0, 1))
    rgb[inside] = (0.1 + 0.1 * sz[inside])[:, None]
    phi, theta = math.radians(az), math.radians(el)
    light = np.array([math.cos(theta)*math.cos(phi), math.cos(theta)*math.sin(phi), math.sin(theta)])
    p = light + np.array([0, 0, 1.]); p /= np.linalg.norm(p)
    glow = np.exp(-((dx - p[0])**2 + (dy - p[1])**2) / (2 * .014**2))
    rgb = np.where(inside[..., None], rgb + (glow * .9)[..., None] * np.array([.95, .9, .8]), rgb)
    img = Image.fromarray(np.round(np.clip(rgb, 0, 1) * 255).astype(np.uint8))
    buf = io.BytesIO(); img.save(buf, format="PNG")
    sx = centre[0] + radius * p[0]
    sy = centre[1] + radius * p[1]
    return buf.getvalue(), [sx - .04, sy - .04, sx + .04, sy + .04]


def main():
    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(tmp)
        client = app.test_client()
        surface = client.post("/api/surfaces", json={"title": "Ball test", "width_mm": 100,
                              "height_mm": 80, "accession": "", "note": "", "synthetic": True}).get_json()["surface"]
        sid = surface["id"]
        lamps = [(40, 35), (220, 60)]
        capture_ids, marks, boxes = [], [], []
        for i, (az, el) in enumerate(lamps, 1):
            blob, box = render(az, el)
            data = {"photo": (io.BytesIO(blob), f"ball{i}.png"),
                    "metadata": '{"title":"L%d","role":"raking","azimuth":0,"elevation":45}' % i}
            reply = client.post(f"/api/surfaces/{sid}/captures", data=data,
                                content_type="multipart/form-data")
            assert reply.status_code == 201, reply.get_json()
            capture_ids.append(reply.get_json()["capture"]["id"])
            marks.append({"label": f"Ball {i}", "center": [.5, .5], "radius": .32, "highlights": [box]})
            boxes.append(box)
        body = {"captures": capture_ids, "corrections": {}, "region": None,
                "options": {"mirror_marks": marks}}
        reply = client.post(f"/api/surfaces/{sid}/analysis/mirror-ball", json=body)
        assert reply.status_code == 200, reply.get_json()
        payload = reply.get_json()
        rows = payload["report"]["rows"]
        print("HTTP inference rows:")
        for row in rows:
            print({k: row[k] for k in ("photo", "status", "inferred_azimuth", "inferred_elevation",
                                       "azimuth_interval", "elevation_interval")})
        assert len(rows) == 2
        for row, (az, el) in zip(rows, lamps):
            assert abs(row["inferred_azimuth"] - az) < 4
            assert abs(row["inferred_elevation"] - el) < 4
        files = {f["key"]: f for f in payload["files"]}
        assert {"mirror-marks", "inferred-lamp-plate", "mirror-lamp-directions"} <= set(files)

        # Validation: mark count mismatch must be rejected.
        bad = dict(body); bad["options"] = {"mirror_marks": marks[:1]}
        reply = client.post(f"/api/surfaces/{sid}/analysis/mirror-ball", json=bad)
        assert reply.status_code == 422, reply.status_code
        print("rejected mismatched marks:", reply.get_json()["error"])

        # Validation: highlight outside ball must be rejected.
        bad_marks = [dict(m, highlights=[[.01, .01, .05, .05]]) for m in marks]
        reply = client.post(f"/api/surfaces/{sid}/analysis/mirror-ball",
                            json={**body, "options": {"mirror_marks": bad_marks}})
        assert reply.status_code == 422, reply.status_code
        print("rejected out-of-ball highlight:", reply.get_json()["error"])

        # Apply inferred angles back to a capture and verify persistence.
        first = rows[0]
        cid = capture_ids[0]
        current = client.get(f"/api/surfaces/{sid}").get_json()["captures"][0]
        upd = client.put(f"/api/captures/{cid}", json={"version": current["version"],
                         "metadata": {**current["metadata"], "azimuth": first["inferred_azimuth"],
                                      "elevation": first["inferred_elevation"]}})
        assert upd.status_code == 200, upd.get_json()
        saved = upd.get_json()["capture"]["metadata"]
        print("persisted lamp settings:", saved["azimuth"], saved["elevation"])
        assert abs(saved["azimuth"] - 40) < 4 and abs(saved["elevation"] - 35) < 4

        # The tool appears in the public catalog.
        catalog = client.get("/api/catalog").get_json()
        assert any(t["key"] == "mirror-ball" for t in catalog["tools"])
        print("\nHTTP END-TO-END MIRROR BALL CHECK PASSED")


if __name__ == "__main__":
    main()
