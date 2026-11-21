"""Independent physical scale check from a photographed rule."""
from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage, signal

from .models import Point, Problem, fields, number, rounded, whole
from .raster import sample_field
from .results import Result
from .visual import COOL, GOLD, HOT, MINT, Trace, flatten_rgb, overlay, plot


@dataclass(frozen=True)
class RuleSpec:
    start: Point
    end: Point
    known_step: float
    width: float
    count: int
    dark_ticks: bool

    @classmethod
    def parse(cls, options, dataset):
        fields(options, "start", "end", "known_step_mm", "band_mm", "samples", "dark_ticks")
        a = Point.parse(options.get("start", [.13, .26]), "Rule start")
        b = Point.parse(options.get("end", [.87, .28]), "Rule end")
        if not (dataset.region.contains(a) and dataset.region.contains(b)):
            raise Problem("Rule endpoints should be inside the analysis area.")
        if dataset.scale.distance(a, b) < 5.:
            raise Problem("Mark a longer span of the photographed rule.")
        types = options.get("dark_ticks", True)
        if type(types) is not bool:
            raise Problem("Choose the relative brightness of marked ticks.")
        return cls(a, b, number(options.get("known_step_mm", 5.), "Expected tick spacing", .1, 100.),
                   number(options.get("band_mm", 2.6), "Rule reading band", .3, 25.),
                   whole(options.get("samples", 620), "Along rule samples", 100, 1900), types)


@dataclass(frozen=True)
class RuleSamples:
    positions: np.ndarray
    profile: np.ndarray
    observed: np.ndarray
    core: np.ndarray
    borders: np.ndarray
    reach: np.ndarray

    @property
    def sampling_step(self):
        return float(self.positions[1] - self.positions[0])


def _axis(spec, dims):
    a, b = np.array([spec.start.public(), spec.end.public()])
    delta = (b - a) * dims
    length = np.hypot(*delta)
    tangent = delta / length
    normal = np.array([-tangent[1], tangent[0]])
    return a, b, length, normal


def _read_band(frame, spec, shape):
    dims = np.array([shape.width_mm, shape.height_mm])
    a, b, distance, orth = _axis(spec, dims)
    half = spec.width * .5
    offsets = np.linspace(-half, half, 11)
    core = np.linspace(a, b, spec.count)
    path = np.asarray([core + v * orth / dims for v in offsets])
    values, good = sample_field(frame.code_y, frame.alpha_valid & ~frame.clipped,
                                path.reshape(-1, 2).clip(0, 1))
    support = good.reshape((len(offsets), spec.count)) & ((path >= 0) & (path <= 1)).all(axis=-1)
    observations = values.reshape(support.shape)
    reliable = support.sum(axis=0) >= 7
    if reliable.mean() < .64:
        raise Problem("The chosen photograph does not show enough of the tick band.")
    data = np.zeros(spec.count, dtype=float)
    data[reliable] = np.nanmedian(np.where(support[:, reliable], observations[:, reliable], np.nan), axis=0)
    points = np.linspace(0, distance, spec.count)
    shapes = np.array([a - half * orth / dims, b - half * orth / dims,
                       b + half * orth / dims, a + half * orth / dims])
    return RuleSamples(points, data, reliable, core, shapes, support.sum(axis=0))


def _unclipped_smooth(values, valid, sigma):
    support = ndimage.gaussian_filter1d(valid.astype(float), sigma, mode="nearest")
    content = ndimage.gaussian_filter1d(np.where(valid, values, 0.), sigma, mode="nearest")
    density = np.divide(content, support, out=np.zeros_like(content), where=support > .01)
    return density, support


def _detect(profiles, spec):
    stride = profiles.sampling_step
    samples_per_tick = spec.known_step / stride
    if samples_per_tick < 3:
        raise Problem("The line resampling is too coarse to separate this expected tick spacing.")
    near, support = _unclipped_smooth(profiles.profile, profiles.observed, max(samples_per_tick * .05, .5))
    trend, wide = _unclipped_smooth(profiles.profile, profiles.observed, max(samples_per_tick * .50, 2.))
    deviation = (trend - near) if spec.dark_ticks else (near - trend)
    valid = (support > .94) & (wide > .84) & profiles.observed
    values = deviation[valid]
    if values.size < 30:
        raise Problem("Visible tick signal is fragmented or too short.")
    floor = max(.009, 1.4826 * np.median(np.abs(values - np.median(values))) * 1.8)
    picks, props = signal.find_peaks(np.where(valid, deviation, 0.), height=floor, prominence=floor / 2,
                                     distance=max(2, round(samples_per_tick * .46)))
    return picks, props, deviation, valid


def _interp_vertex(i, trace):
    if i <= 0 or i + 1 >= len(trace):
        return float(i)
    left, center, right = trace[i - 1 : i + 2]
    denominator = left - 2 * center + right
    if abs(denominator) < 1e-8:
        return float(i)
    return float(i) + float(np.clip(.5 * (left - right) / denominator, -.45, .45))


def _ticks(meta, raw_picks, strengths, profiles, deviation, spec):
    dist = profiles.sampling_step
    positions = np.array([_interp_vertex(i, deviation) * dist for i in raw_picks])
    if len(positions) < 4:
        raise Problem("Could not find four distinct rule ticks. Check direction, tone and endpoints.")
    pairs = np.diff(positions)
    multiple = np.clip(np.rint(pairs / spec.known_step).astype(int), 1, 6)
    residual = np.abs(pairs / multiple - spec.known_step)
    fit_pairs = residual < spec.known_step * .25
    estimates = pairs[fit_pairs] / multiple[fit_pairs]
    if len(estimates) < 3:
        raise Problem("Detected marks do not agree with the declared tick spacing.")
    length_mm = float(np.median(estimates))
    support_rows = profiles.observed
    records = []
    for p, at, peak in zip(positions, raw_picks, strengths, strict=True):
        u = profiles.core[at]
        cam = meta.registration.camera([u.tolist()])[0]
        extends = p / max(1e-7, profiles.positions[-1])
        records.append({"mark": len(records) + 1, "distance_mm": rounded(p, 4),
                        "along_line_fraction": rounded(extends), "camera": np.round(cam, 7).tolist(),
                        "prominence_code": rounded(peak), "supported_cross_band_lines": int(profiles.reach[at])})
    between = [{"from": i + 1, "to": i + 2, "image_distance_mm": rounded(delta, 5),
                "inferred_intervals": int(multi), "readable": bool(verified),
                "normalized_interval_mm": rounded(delta / multi) if verified else None}
               for i, (delta, multi, verified) in enumerate(zip(pairs, multiple, fit_pairs, strict=True))]
    summary = {"declared_step_mm": spec.known_step, "measured_step_mm": rounded(length_mm, 5),
               "percentage_from_known": rounded(100 * (length_mm / spec.known_step - 1), 4),
               "matching_intervals": int(fit_pairs.sum()), "detected_marks": len(positions),
               "supported_fraction": rounded(np.mean(support_rows)),
               "interval_mad_mm": rounded(np.median(np.abs(estimates - length_mm)), 6)}
    return records, between, summary


def ruler_readout(dataset, options):
    spec = RuleSpec.parse(options, dataset)
    frame = dataset.one()
    source = _read_band(frame, spec, dataset.scale)
    picks, props, deviation, valid = _detect(source, spec)
    rows, intervals, summary = _ticks(frame, picks, props["prominences"], source, deviation, spec)
    result = Result("ruler", "Photographed rule readback",
                    "Compare tick spacing measured in the registered dimensions with the actual marked separation.",
                    summary, rows)
    draw = [("closed", source.borders.tolist(), GOLD, "Tick band")]
    draw.extend(("point", [source.core[index].tolist()], MINT, str(i))
                for i, index in enumerate(picks, 1))
    image = flatten_rgb(frame.code, frame.alpha_valid & dataset.selection, False)
    result.image("rule-location", overlay(image, draw))
    strip = np.where(valid, deviation, np.nan)
    impulses = np.full(source.positions.shape, np.nan)
    impulses[picks] = strip[picks]
    x = source.positions
    result.image("tick-response", plot([Trace("Tick contrast", x, strip), Trace("Observed mark", x, impulses, GOLD)],
                                       "From start (mm)", "Tick readout code"))
    result.table("tick-spacing", intervals)
    result.warnings.append("A ruler needs the same plane as the imaged surface to be an independent scale check. This report does not update the stored physical dimensions.")
    return result.table("tick-positions")
