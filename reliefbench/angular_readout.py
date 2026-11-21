"""Compare genuinely observed responses while changing directional illumination."""
from dataclasses import dataclass

import numpy as np

from .models import Problem, Region, fields, list_of, number, rounded, text
from .raster import safe_fraction, statistics
from .results import Result
from .visual import COOL, GOLD, HOT, MINT, Trace, contact_sheet, flatten_rgb, overlay, plot


DEFAULT_REGIONS = [
    {"label": "Upper shoulder", "region": [.40,.27,.56,.37]},
    {"label": "Central impression", "region": [.42,.44,.60,.59]},
    {"label": "Outer surface", "region": [.69,.53,.83,.70]},
]


@dataclass(frozen=True)
class AngularRegion:
    label: str
    area: Region

    @classmethod
    def parse(cls, obj, bounds):
        if not isinstance(obj, dict):
            raise Problem("Each readout region has a name and a rectangle.")
        fields(obj, "label", "region")
        name = text(obj.get("label"), "Readout region name", 65)
        area = Region.parse(obj.get("region"))
        if not (bounds.left <= area.left < area.right <= bounds.right and
                bounds.top <= area.top < area.bottom <= bounds.bottom):
            raise Problem(f"{name} is outside the analysis rectangle.")
        return cls(name, area)

    def mask(self, dimensions):
        height, width = dimensions
        value = np.zeros((height, width), dtype=bool)
        value[self.area.window(width, height)] = True
        return value


def _one_read(frame, mask, minimum):
    points = mask & frame.alpha_valid
    reading = points & ~frame.clipped
    ratio = safe_fraction(reading, mask.sum())
    if ratio < minimum or reading.sum() < 20:
        value = None
        inside = (None, None)
    else:
        metric = statistics(frame.signal, reading)
        value = metric["median"]
        inside = (metric["p05"], metric["p95"])
    return {"capture": frame.title, "id": frame.id, "azimuth": frame.spec.azimuth,
            "elevation": frame.spec.elevation, "exposure_ev": frame.spec.exposure,
            "normalized_response": value, "visible_fraction": rounded(ratio),
            "readable_samples": int(reading.sum()),
            "clip_fraction": rounded(safe_fraction(points & frame.clipped, points.sum())),
            "p05": inside[0], "p95": inside[1]}


def _summarize(rows):
    chosen = [row for row in rows if row["normalized_response"] is not None]
    if not chosen:
        return {"readable_sources": 0, "strongest_azimuth": None, "spread": None}
    low, high = min(chosen, key=lambda p:p["normalized_response"]), max(chosen, key=lambda p:p["normalized_response"])
    return {"readable_sources": len(chosen), "strongest_azimuth": high["azimuth"],
            "strongest_elevation": high["elevation"], "strongest_source": high["capture"],
            "spread": rounded(high["normalized_response"] - low["normalized_response"])}


def _collapsed_azimuth(rows):
    angles = []
    values = []
    prior = None
    for row in rows:
        az = row["azimuth"]
        if az == prior:
            # A repeated azimuth with another elevation is a separate physical point.
            # An explicit gap leaves the chart at the two observed responses.
            angles.append(az)
            values.append(np.nan)
        angles.append(az)
        value = row["normalized_response"]
        values.append(value if value is not None else np.nan)
        prior = az
    return np.array(angles, dtype=float), np.array(values, dtype=float)


def _guides(probes):
    palette = (MINT, GOLD, COOL, HOT)
    return [(probe, palette[i]) for i, probe in enumerate(probes)]


def angles_report(dataset, options):
    fields(options, "reference_regions", "min_coverage")
    frames = sorted(dataset.lamps(), key=lambda f:(f.spec.azimuth, f.spec.elevation))
    probes = [AngularRegion.parse(o, dataset.region) for o in list_of(
        options.get("reference_regions", DEFAULT_REGIONS), "Response regions", 1, 4)]
    if len(set(r.label for r in probes)) != len(probes):
        raise Problem("Use a distinct name for each response region.")
    minimum = number(options.get("min_coverage", .72), "Minimum unclipped region fraction", .38, 1.)
    parts = []
    lines = []
    summary = {}
    guide = _guides(probes)
    for desc, color in guide:
        mask = desc.mask(dataset.selection.shape)
        collection = [_one_read(frame, mask, minimum) for frame in frames]
        summary[desc.label] = _summarize(collection)
        u, v = _collapsed_azimuth(collection)
        lines.append(Trace(desc.label[:23], u, v, color))
        parts.extend({"region": desc.label, "bounds": desc.area.public(), **row} for row in collection)
    if not any(r["normalized_response"] is not None for r in parts):
        raise Problem("The selected lamp views do not reliably cover any readout region.")
    drawings = []
    for frame in frames[:8]:
        base = flatten_rgb(frame.code, frame.alpha_valid, False)
        boxes = [("closed", region.area.corners(), color, region.label) for region, color in guide]
        painted = overlay(base, boxes)
        drawings.append((f"{frame.spec.azimuth:g} / {frame.spec.elevation:g} | {frame.title}", painted))
    result = Result("angular-readout", "Measured response by lamp angle",
                    "Compare repeatable sample regions across mounted lamp photos and keep missing or clipped directions visible in the table.",
                    {"requested_minimum_coverage": minimum, "photos": len(frames), "region_summary": summary}, parts)
    result.image("response-angles", plot(lines, "Azimuth °", "Median linear signal", x_window=(0, 360)))
    result.image("angular-photographs", contact_sheet(drawings, columns=min(3, len(drawings))))
    result.warnings.append("Each point is a separate exposure. Lamp elevations and measured signal are listed, and the connecting line is a visual guide.")
    return result.table("marked-response-observations")
