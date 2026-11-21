"""Predict coverage of physically reachable light placements for a measured surface."""
from dataclasses import dataclass
from itertools import product
import math

import numpy as np
from PIL import Image, ImageDraw

from .lightmodel import directions, diversity, solve
from .models import Problem, fields, flag, light_vector, list_of, number, rounded, whole
from .raster import safe_fraction
from .results import Result
from .visual import COOL, GOLD, HOT, MINT, PAPER, Extent, heatmap, overlay


DEFAULT_AZIMUTHS = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
DEFAULT_ELEVATIONS = [24, 40, 62]
DEFAULT_KEEPOUT = [[102, 142], [246, 282]]


@dataclass(frozen=True)
class Position:
    azimuth: float
    elevation: float

    @property
    def vector(self):
        return np.asarray(light_vector(self.azimuth, self.elevation), dtype=float)


@dataclass(frozen=True)
class StandSpec:
    azimuths: tuple[float, ...]
    elevations: tuple[float, ...]
    exclusion: tuple
    minimum_direct: float
    concurrent: int
    prefer_support: bool

    @classmethod
    def parse(cls, options):
        fields(options, "azimuths", "elevations", "keepout", "surface_threshold", "count", "prefer_support")
        az_raw = list_of(options.get("azimuths", DEFAULT_AZIMUTHS), "Mount azimuths", 3, 32)
        azimuth = tuple(dict.fromkeys(number(x, "Mount azimuth", 0, 360) % 360 for x in az_raw))
        ele_raw = list_of(options.get("elevations", DEFAULT_ELEVATIONS), "Mount elevations", 1, 6)
        elevation = tuple(dict.fromkeys(number(e, "Mount elevation", 6, 87) for e in ele_raw))
        raw_exclude = list_of(options.get("keepout", DEFAULT_KEEPOUT), "Blocked sectors", 0, 8)
        exclusions = tuple(tuple(number(edge, "Blocked azimuth", 0, 360) % 360 for edge in
                                 list_of(span, "Blocked azimuth interval", 2, 2)) for span in raw_exclude)
        if any(a == b for a, b in exclusions):
            raise Problem("Use different ends for each blocked azimuth interval.")
        amount = number(options.get("surface_threshold", .34), "Minimum directed response", .1, .85)
        count = whole(options.get("count", 3), "Additional positions", 1, 5)
        prioritization = flag(options.get("prefer_support", True), "Prioritize low-response area")
        return cls(azimuth, elevation, exclusions, amount, count, prioritization)

    def candidates(self):
        return [Position(a, e) for a, e in product(self.azimuths, self.elevations)
                if not any(_intersects(a, start, stop) for start, stop in self.exclusion)]


def _intersects(value, a, b):
    return (a <= value <= b) if a <= b else (value >= a or value <= b)


def _separation(vector, others):
    closest = np.max(others @ vector)
    return math.degrees(math.acos(float(np.clip(closest, -1., 1.))))


def _initial_readout(normals, matrix, chunk=55_000):
    response = np.empty(len(normals), dtype=np.float32)
    for start in range(0, len(normals), chunk):
        part = np.s_[start:start + chunk]
        response[part] = np.max(normals[part] @ matrix.T, axis=1)
    return response


@dataclass(frozen=True)
class Candidate:
    mount: Position
    conditioning: float
    cover: float
    novel: float
    worst: float
    distance: float
    utility: float

    def public(self, order=None):
        data = {"azimuth": rounded(self.mount.azimuth, 3), "elevation": rounded(self.mount.elevation, 3),
                "nearest_other_light_angle": rounded(self.distance, 3),
                "requested_response_fraction": rounded(self.cover), "newly_covered_fraction": rounded(self.novel),
                "fifth_percentile_response": rounded(self.worst), "conditioning": rounded(self.conditioning),
                "utility": rounded(self.utility)}
        if order is not None:
            data["choice"] = order
        return data


def _estimate(mount, matrix, normals, reachable, spec):
    v = mount.vector
    distance = _separation(v, matrix)
    if distance < 4:
        return None
    new_response = normals @ v
    fit = np.maximum(reachable, new_response)
    stronger = fit >= spec.minimum_direct
    added = stronger & (reachable < spec.minimum_direct)
    potential = np.vstack((matrix, v))
    condition = diversity(potential)["condition"]
    if condition is None:
        condition = 250.
    cover = float(stronger.mean())
    new = float(added.mean())
    fifth = float(np.percentile(fit, 5))
    spread_penalty = min(4., condition)
    if spec.prefer_support:
        score = 4. * new + .45 * fifth + .0014 * distance - .006 * spread_penalty
    else:
        score = .8 * new + .25 * fifth + .005 * distance - .064 * spread_penalty
    return Candidate(mount, condition, cover, new, fifth, distance, score)


def rank(candidates, normals, matrix, known, spec):
    result = [_estimate(candidate, matrix, normals, known, spec) for candidate in candidates]
    return sorted((r for r in result if r is not None),
                  key=lambda r: (-r.utility, r.mount.azimuth, r.mount.elevation))


def _choose(candidates, normals, first, spec):
    selected = []
    design = first.copy()
    known = _initial_readout(normals, first)
    initial_candidates = rank(candidates, normals, design, known, spec)
    current = initial_candidates
    for _ in range(spec.concurrent):
        if not current:
            break
        candidate = current[0]
        selected.append(candidate)
        light = candidate.mount.vector
        design = np.vstack((design, light))
        known = np.maximum(known, normals @ light)
        unused = [member.mount for member in current[1:]]
        current = rank(unused, normals, design, known, spec)
    return selected, initial_candidates


def _polar(theta, elevation, center, scale):
    radius = scale * (90 - elevation) / 90
    angle = math.radians(theta)
    return (center + radius * math.cos(angle), center + radius * math.sin(angle))


def _polar_plate(frames, selected, exclusion):
    width = 505
    center = width / 2
    r = center - 30
    plate = Image.new("RGB", (width, width), PAPER)
    draw = ImageDraw.Draw(plate)
    for height in (0, 22.5, 45, 67.5):
        radius = r * (90 - height) / 90
        draw.ellipse((center - radius, center - radius, center + radius, center + radius), outline=(82, 102, 103), width=1)
        draw.text((center + 5, center + radius - 12), f"{height:g}", fill=(166, 187, 183))
    for theta in range(0, 360, 45):
        x, y = _polar(theta, 0, center, r)
        draw.line((center, center, x, y), fill=(64, 80, 83), width=1)
        t = _polar(theta, 0, center, r + 12)
        draw.text((t[0] - 8, t[1] - 7), str(theta), fill=(215, 230, 227))
    for a, b in exclusion:
        span = (b - a) % 360
        draw.arc((center - r, center - r, center + r, center + r), start=a, end=a + span, fill=HOT, width=5)
    for frame in frames:
        point = _polar(frame.spec.azimuth, frame.spec.elevation, center, r)
        draw.ellipse((point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill=COOL, outline=PAPER)
    for i, entry in enumerate(selected, 1):
        u, v = _polar(entry.mount.azimuth, entry.mount.elevation, center, r)
        draw.ellipse((u - 10, v - 10, u + 10, v + 10), fill=GOLD, outline=PAPER)
        draw.text((u - 3, v - 6), str(i), fill=PAPER)
    draw.text((10, 9), "Existing", fill=COOL)
    draw.text((75, 9), "Proposed", fill=GOLD)
    draw.text((142, 9), "Blocked", fill=HOT)
    return plate


def _support_layers(field, actual, extra, spec):
    normals = field.normals[field.valid]
    known = _initial_readout(normals, actual)
    later = _initial_readout(normals, np.vstack((actual, extra))) if len(extra) else known
    image = np.zeros(field.valid.shape, dtype=float)
    image[field.valid] = later
    improvement = np.zeros(field.valid.shape, dtype=bool)
    improvement[field.valid] = (later >= spec.minimum_direct) & (known < spec.minimum_direct)
    counts = {"originally_above_light_threshold": rounded(np.mean(known >= spec.minimum_direct)),
              "projected_above_light_threshold": rounded(np.mean(later >= spec.minimum_direct)),
              "newly_supported_fraction": rounded(safe_fraction(improvement, field.valid.sum()))}
    return image, improvement, counts


def plan_lamps(dataset, options):
    spec = StandSpec.parse(options)
    frames = dataset.lamps()
    candidate_positions = spec.candidates()
    if not candidate_positions:
        raise Problem("All physical lamp choices are in blocked sectors.")
    field = solve(dataset)
    flat = field.normals[field.valid]
    stride = max(1, math.ceil(len(flat) / 10_400))
    sampled = flat[::stride]
    origin = directions(frames)
    selected, rank_initial = _choose(candidate_positions, sampled, origin, spec)
    extra = np.array([m.mount.vector for m in selected])
    picture, improved, summary = _support_layers(field, origin, extra, spec)
    empty_count = len(candidate_positions) - len(rank_initial)
    result = Result("lamp-plan", "Plan additional raking light positions",
                    "Compare feasible mounting positions using photographed-facing surfaces and lamp conditioning.",
                    {**summary, "surface_response_limit": spec.minimum_direct, "proposed_positions": len(selected),
                     "blocked_sectors": [list(span) for span in spec.exclusion],
                     "orientation_fit_samples": int(len(flat)), "orientation_ranking_samples": int(len(sampled)),
                     "spacing_rejected_positions": empty_count,
                     "selected_established_lamps": len(frames)},
                    [entry.public(i) for i, entry in enumerate(selected, 1)])
    result.image("mounting-placement", _polar_plate(frames, selected, spec.exclusion))
    shown = np.array(heatmap(picture, field.valid, Extent(0., 1.), grid=True))
    shown[improved] = MINT
    result.image("projected-light-reach", Image.fromarray(shown))
    result.table("additional-positions")
    result.table("legal-first-candidates", [candidate.public() for candidate in rank_initial])
    result.warnings.append("This is planning from local surface orientation. It does not predict cast occlusion, glare or the actual future photograph.")
    return result
