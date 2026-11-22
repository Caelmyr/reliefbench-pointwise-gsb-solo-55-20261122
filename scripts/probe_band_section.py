#!/usr/bin/env python3
"""End-to-end exercise for the banded physical section tool.

Run with an isolated data directory; the synthetic slate demo provides raking photos and a
heightfield with an unobserved margin so the empty-station rules are exercised.
"""
import base64
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reliefbench.http import create_app  # noqa: E402


def main():
    with tempfile.TemporaryDirectory() as data_dir:
        app = create_app(data_dir)
        client = app.test_client()

        created = client.post("/api/examples/slate")
        assert created.status_code in (200, 201), created.data
        sid = created.get_json()["surface"]["id"]
        surface = client.get(f"/api/surfaces/{sid}").get_json()
        captures = [c["id"] for c in surface["captures"] if c["metadata"]["role"] == "raking"]
        assert len(captures) >= 3

        options = {
            "start": [.10, .50], "end": [.92, .50], "samples": 160,
            "band_width_mm": 2.0, "band_lines": 5, "show_angles": True,
        }
        body = {"captures": captures, "corrections": {}, "region": None, "options": options}
        reply = client.post(f"/api/surfaces/{sid}/analysis/section", json=body)
        assert reply.status_code == 200, reply.data
        output = reply.get_json()
        report = output["report"]
        print("metrics:", json.dumps(report["metrics"], indent=2))

        keys = {f["key"]: f for f in output["files"]}
        for required in ("section-location", "section-height", "section-slope",
                         "section-band-probe", "physical-cross-section", "section-band-support"):
            assert required in keys, f"missing asset {required}"
        print("assets:", sorted(keys))

        probe_doc = json.loads(base64.b64decode(keys["section-band-probe"]["data"].split(",", 1)[1]))
        stations = probe_doc["stations"]
        assert len(stations) == 160
        assert probe_doc["band"]["width_mm"] == 2.0
        assert probe_doc["band"]["lines"] == 5
        assert len(probe_doc["line_points"]) == 5

        usable = [s for s in stations if s["usable"]]
        empties = [s for s in stations if not s["usable"]]
        print(f"usable={len(usable)} empty={len(empties)}")
        assert usable, "no usable stations"
        for s in usable:
            assert s["band_height_mm"] is not None
            assert s["band_members_measured"] >= 3
            assert s["empty_reason"] is None
        for s in empties:
            assert s["band_height_mm"] is None
            assert s["empty_reason"] in ("no-central-support", "too-few-band-members", "off-slope")

        # single line and band median are both present for comparison on usable stations
        compared = [s for s in usable if s["single_line_height_mm"] is not None]
        assert compared, "single line trace missing"
        print("example station:", json.dumps(compared[0], indent=2))

        # bad inputs are rejected
        bad = dict(body)
        bad["options"] = {**options, "band_lines": 4}
        rejected = client.post(f"/api/surfaces/{sid}/analysis/section", json=bad)
        assert rejected.status_code == 422, rejected.data
        print("even line count rejected:", rejected.get_json()["error"])

        bad = dict(body)
        bad["options"] = {**options, "band_width_mm": 200.0}
        rejected = client.post(f"/api/surfaces/{sid}/analysis/section", json=bad)
        assert rejected.status_code == 422, rejected.data
        print("oversized band rejected:", rejected.get_json()["error"])

        # saving evidence retains the probe so the saved view restores the exact band width
        retained = client.post(f"/api/surfaces/{sid}/analysis/section",
                               json={**body, "retain": True, "title": "Band section check"})
        assert retained.status_code == 201, retained.data
        record_id = retained.get_json()["record"]["id"]
        record = client.get(f"/api/records/{record_id}").get_json()["record"]
        saved_files = {f["key"]: f for f in record["files"]}
        assert "section-band-probe" in saved_files
        packet = client.get(f"/api/records/{record_id}/packet")
        assert packet.status_code == 200 and packet.data[:2] == b"PK"
        print("evidence retained with probe; packet bytes:", len(packet.data))
        print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
