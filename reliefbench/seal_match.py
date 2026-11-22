"""Compare a drawn seal fragment with retained relative-height records.

The fragment is described by a few strokes drawn on one of its photographs, a physical
radius and two clicks on the surviving rim. Each retained height record supplies depth
on a physical grid. The drawing is swept over every record at its true size (rotation and
translation only) with a matched filter that contrasts the drawn strokes against the near
background, so a shared round edge cannot carry a record on its own. Rim agreement is
reported next to the stroke evidence but does not enter the score. When two records agree
on the same short stretch and nothing else separates them, both stay in the candidate list
under one unresolved label.
"""
from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage
from scipy.signal import correlate

from .models import Point, Problem, fields, flag, identifier, list_of, number, rounded, whole
from .raster import decode
from .registration import Registration
from .results import Result
from .visual import DIVERGE, GOLD, HOT, INK, MINT, PAPER, Extent, FONT, heatmap, scale_bar, split_visual


SAMPLE_COUNTS = (24, 48, 96, 192, 384)
DEFAULT_SAMPLES = 192
DEFAULT_RADIUS_MM = 40.
DEFAULT_STROKE_WIDTH_MM = 1.2
DEFAULT_ANGLE_STEP = 2.
DEFAULT_MIN_VISIBLE = .55
DEFAULT_TOLERANCE_MM = .35
WEAK_CONTRAST = .16
RIM_ONLY_EVIDENCE = 1.35
TIE_SCORE_MARGIN = .07
SHORT_SUPPORT_MM = 6.
DETAIL_LIMIT = 8
MAX_CANDIDATES = 12
COARSE_ROTATION_FACTOR = 3.


@dataclass(frozen=True)
class CandidateRef:
    record: str
    surface: str

    @classmethod
    def parse(cls, raw):
        if not isinstance(raw, dict):
            raise Problem("Each candidate needs a retained record and its surface.")
        fields(raw, "record", "surface")
        return cls(identifier(raw.get("record"), "Candidate record"),
                   identifier(raw.get("surface"), "Candidate surface"))


@dataclass(frozen=True)
class SealSpec:
    surface: str
    capture: str
    registration: str
    strokes: tuple[tuple[Point, ...], ...]
    centre: Point
    radius_mm: float
    stroke_width_mm: float
    rim_points: tuple[Point, Point]
    samples: int
    angle_step: float
    min_visible: float
    tolerance_mm: float
    keep_unresolved: bool
    candidates: tuple[CandidateRef, ...]

    @classmethod
    def parse(cls, raw):
        fields(raw, "surface", "capture", "registration", "strokes", "centre", "radius_mm",
               "stroke_width_mm", "rim_points", "samples", "angle_step", "min_visible",
               "tolerance_mm", "keep_unresolved", "candidates")
        strokes = []
        for index, stroke in enumerate(list_of(raw.get("strokes"), "Drawn strokes", 1, 8), 1):
            points = list_of(stroke, f"Stroke {index}", 2, 24)
            strokes.append(tuple(Point.parse(value, f"Stroke {index} point") for value in points))
        rim = list_of(raw.get("rim_points"), "Surviving rim clicks", 2, 2)
        samples = whole(raw.get("samples", DEFAULT_SAMPLES), "Outline samples", 24, 384)
        if samples not in SAMPLE_COUNTS:
            raise Problem("Use one of the offered outline sample counts.")
        selected = [CandidateRef.parse(item) for item in list_of(raw.get("candidates"), "Candidate records", 1, MAX_CANDIDATES)]
        if len({(item.record, item.surface) for item in selected}) != len(selected):
            raise Problem("Use each candidate record once.")
        return cls(identifier(raw.get("surface"), "Fragment surface"),
                   identifier(raw.get("capture"), "Fragment photograph"),
                   identifier(raw.get("registration", "identity"), "Fragment registration"),
                   tuple(strokes), Point.parse(raw.get("centre"), "Fragment centre"),
                   number(raw.get("radius_mm", DEFAULT_RADIUS_MM), "Largest dial radius (mm)", 4., 120.),
                   number(raw.get("stroke_width_mm", DEFAULT_STROKE_WIDTH_MM), "Drawn stroke width (mm)", .25, 6.),
                   (Point.parse(rim[0], "First rim click"), Point.parse(rim[1], "Second rim click")),
                   samples, number(raw.get("angle_step", DEFAULT_ANGLE_STEP), "Rotation step", 1., 30.),
                   number(raw.get("min_visible", DEFAULT_MIN_VISIBLE), "Minimum visible fragment share", .2, 1.),
                   number(raw.get("tolerance_mm", DEFAULT_TOLERANCE_MM), "Depth tolerance (mm)", .02, 4.),
                   flag(raw.get("keep_unresolved", True), "Keep unresolved pairs"), tuple(selected))

    def public(self):
        return {"surface": self.surface, "capture": self.capture, "registration": self.registration,
                "strokes": [[point.public() for point in stroke] for stroke in self.strokes],
                "centre": self.centre.public(), "radius_mm": self.radius_mm,
                "stroke_width_mm": self.stroke_width_mm,
                "rim_points": [self.rim_points[0].public(), self.rim_points[1].public()],
                "samples": self.samples, "angle_step": self.angle_step,
                "min_visible": self.min_visible, "tolerance_mm": self.tolerance_mm,
                "keep_unresolved": self.keep_unresolved,
                "candidates": [{"record": item.record, "surface": item.surface} for item in self.candidates]}


@dataclass(frozen=True)
class HeightSheet:
    surface_id: str
    record: str
    title: str
    width_mm: float
    height_mm: float
    cell: float
    heights: np.ndarray
    observed: np.ndarray
    detail: np.ndarray
    gradient: np.ndarray
    source: dict | None
    signature: str

    def sample(self, values, row, column):
        data = ndimage.map_coordinates(np.asarray(values, dtype=np.float32), [row, column], order=1, mode="nearest")
        cover = ndimage.map_coordinates(self.observed.astype(np.float32), [row, column], order=1, mode="nearest")
        return data, np.clip(cover, 0., 1.)

    def cells(self, x_mm, y_mm):
        return np.asarray(y_mm) / self.cell - .5, np.asarray(x_mm) / self.cell - .5


@dataclass(frozen=True)
class Fragment:
    centre_mm: tuple[float, float]
    radius_mm: float
    rim_radius_mm: float
    sampled_mm: tuple[np.ndarray, ...]
    arc: tuple[float, float]

    @property
    def total_length_mm(self):
        return float(sum(float(np.sum(np.hypot(*np.diff(stroke, axis=0).T)))
                         for stroke in self.sampled_mm if len(stroke) > 1))

    @property
    def stroke_count(self):
        return len(self.sampled_mm)


def _registration_row(library, capture_id, wanted):
    rows = library.registrations(capture_id)
    if wanted == "identity":
        return None
    if wanted in ("", None):
        return rows[0] if rows else None
    for row in rows:
        if row["id"] == wanted:
            return row
    raise Problem("That registration is not stored for the fragment photograph.")


def _surface_points(points, registration, surface):
    normal = registration.surface(np.asarray([point.public() for point in points], dtype=float))
    return normal * [float(surface["width_mm"]), float(surface["height_mm"])]


def _rim_arc(rim_mm, centre_mm):
    vectors = rim_mm - centre_mm
    angles = np.degrees(np.arctan2(vectors[:, 1], vectors[:, 0]))
    start, end = float(angles[0]), float(angles[1])
    span = (end - start) % 360.
    if span > 180.:
        start, end = end, start
    return start, end


def build_fragment(spec, registration, surface):
    width_mm, height_mm = float(surface["width_mm"]), float(surface["height_mm"])
    centre = _surface_points((spec.centre,), registration, surface)[0]
    rim = _surface_points(spec.rim_points, registration, surface)
    dials = np.hypot(*(rim - centre).T)
    if float(dials.min()) < 2.:
        raise Problem("Click the surviving rim away from the fragment centre.")
    if float(dials.max() - dials.min()) > .35 * float(dials.max()):
        raise Problem("The two rim clicks disagree about the dial radius. Click the same surviving circle twice.")
    radius = float(dials.mean())
    arc = _rim_arc(rim, centre)
    drawn = [np.hypot(*points.T).max() for points in
             (_surface_points(stroke, registration, surface) - centre for stroke in spec.strokes)]
    if max(drawn) > max(radius, spec.radius_mm):
        raise Problem("A drawn stroke reaches beyond the declared dial radius.")
    length = sum(math.hypot(second.x - first.x, second.y - first.y)
                 for stroke in spec.strokes for first, second in zip(stroke[:-1], stroke[1:], strict=True))
    length = max(1e-6, length * width_mm)
    spacing = max(.1, min(spec.stroke_width_mm * .5, length / max(spec.samples, 24)))
    spans = []
    for stroke in spec.strokes:
        points = _surface_points(stroke, registration, surface) - centre
        path = []
        for first, second in zip(points[:-1], points[1:], strict=True):
            count = max(2, int(math.ceil(float(np.hypot(*(second - first))) / spacing)))
            path.append(np.linspace(first, second, count, endpoint=False))
        path.append(points[-1:])
        spans.append(np.vstack(path))
    return Fragment(tuple(float(v) for v in centre), radius, float(dials.mean()),
                    tuple(spans), arc)


def _float_raster(raw, label):
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except OSError as exc:
        raise Problem(f"The stored {label} raster could not be read.", 409) from exc
    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2:
        raise Problem(f"The stored {label} raster is not a single mono image.", 409)
    return values


def _preferred_source(record):
    sources = record["evidence"].get("sources") or []
    preferred = [item for item in sources if item.get("role") == "raking"] or sources
    if not preferred:
        return None
    source = preferred[0]
    return {"id": source["id"], "title": source.get("title", "Photograph"),
            "role": source.get("role", ""), "hash": source["hash"],
            "filename": source.get("filename", "capture"),
            "settings": source.get("settings") or {},
            "size": source.get("size") or [],
            "encoding": source.get("encoding", ""),
            "registration": source.get("registration"),
            "registration_id": (source.get("registration") or {}).get("id") or source.get("registration_id")}


def load_sheet(library, record_id, surface_id):
    record = library.get_inspection(record_id)
    if record["surface_id"] != surface_id:
        raise Problem("A candidate record belongs to a different surface.")
    if record["kind"] != "height":
        raise Problem("A seal candidate needs a retained relative height record.")
    keys = {item["key"] for item in record["files"]}
    if not {"height-samples", "height-support"} <= keys:
        raise Problem("This height record kept no stored height samples.")
    heights = _float_raster(library.inspection_asset(record_id, "height-samples")[0], "height samples")
    support = _float_raster(library.inspection_asset(record_id, "height-support")[0], "height support")
    if heights.shape != support.shape:
        raise Problem("The stored height samples and support do not share one grid.")
    metrics = record["report"]["metrics"]
    grid = metrics.get("height_grid") or record["evidence"]["scale"]
    width_mm, height_mm = float(grid["size_mm"][0]), float(grid["size_mm"][1])
    dx, dy = (float(value) for value in grid["sampling_mm"][:2])
    cell = .5 * (dx + dy)
    observed = np.isfinite(heights) & (support > .5)
    if observed.sum() < 64:
        raise Problem("This height record has too little retained depth support to compare.")
    clean = np.where(observed, heights, 0.).astype(np.float32)
    sigma = max(1., 1.6 / cell)
    weight = ndimage.gaussian_filter(observed.astype(np.float32), sigma, mode="nearest")
    broad = ndimage.gaussian_filter(clean, sigma, mode="nearest")
    detail = np.where(observed, clean - np.divide(broad, np.maximum(weight, 1e-4)), 0.).astype(np.float32)
    edge_sigma = max(1., 1.2 / cell)
    magnitude = np.hypot(*np.gradient(clean, cell, cell))
    gradient = np.where(observed, ndimage.gaussian_filter(magnitude, edge_sigma, mode="nearest"), 0.).astype(np.float32)
    return HeightSheet(surface_id, record["id"], record["title"], width_mm, height_mm, cell,
                       clean, observed, detail, gradient, _preferred_source(record),
                       record["evidence"]["data_signature"]["identity"])


def _template(fragment, cell, stroke_width, angle):
    margin = max(3. * stroke_width, 3. * cell, 1.)
    reach = fragment.radius_mm + margin
    half = int(math.ceil(reach / cell))
    extract = int(math.ceil(margin / cell)) + 1
    work = 2 * (half + extract) + 1
    centre = half + extract
    scale = 8
    big = work * scale
    canvas = Image.new("L", (big, big), 0)
    draw = ImageDraw.Draw(canvas)
    turn = math.radians(angle)
    cosine, sine = math.cos(turn), math.sin(turn)
    width_px = max(1, int(round(stroke_width / cell)))
    for stroke in fragment.sampled_mm:
        turned = np.column_stack((stroke[:, 0] * cosine - stroke[:, 1] * sine,
                                  stroke[:, 0] * sine + stroke[:, 1] * cosine))
        pixels = [((u / cell + centre) * scale, (v / cell + centre) * scale) for u, v in turned]
        draw.line(pixels, fill=255, width=width_px * scale, joint="curve")
        radius = width_px * scale / 2.
        for u, v in pixels:
            draw.ellipse((u - radius, v - radius, u + radius, v + radius), fill=255)
    ink = np.asarray(canvas, dtype=np.uint8) > 0
    raster = ink.reshape(work, scale, work, scale).mean(axis=(1, 3)) >= .5
    yy, xx = np.mgrid[0:work, 0:work]
    distance = np.hypot((xx - centre) * cell, (yy - centre) * cell)
    disk = distance <= fragment.radius_mm
    rim_band = (distance >= .86 * fragment.radius_mm) & (distance <= fragment.radius_mm * 1.05)
    guard = ndimage.binary_dilation(raster, iterations=width_px + 1)
    background = disk & ~guard & ~rim_band
    if raster.sum() < 6 or background.sum() < 24:
        return None
    stroke_weight = raster.astype(np.float32) / float(raster.sum())
    background_weight = background.astype(np.float32) / float(background.sum())
    core = np.s_[extract:extract + 2 * half + 1, extract:extract + 2 * half + 1]
    mask_core = raster[core].astype(np.float32)
    if mask_core.sum() < 6 or background[core].sum() < 24:
        return None
    mask_core /= mask_core.sum()
    background_core = background[core].astype(np.float32)
    background_core /= background_core.sum()
    kernel = np.zeros_like(stroke_weight)
    kernel[core] = mask_core - background_core
    return {"kernel": kernel, "mask": stroke_weight, "background": background_weight,
            "half": half, "extract": extract, "side": work, "width_px": width_px, "margin": margin,
            "raster": raster}


def _correlate(values, kernel):
    """Full correlation; a template feature at ``t`` lands at ``data + t - half``."""
    return correlate(values, kernel, mode="full", method="fft")


def _scan(sheet, fragment, spec, angle):
    template = _template(fragment, sheet.cell, spec.stroke_width_mm, angle)
    if template is None:
        return None
    observed = sheet.observed.astype(np.float32)
    response = _correlate(sheet.detail * observed, template["kernel"])
    cover = _correlate(observed, template["mask"])
    background = _correlate(observed, template["background"])
    offset = template["half"] + template["extract"]
    eligible = (cover >= spec.min_visible) & (background >= spec.min_visible * .8)
    if not eligible.any():
        return None
    scored = np.where(eligible, response, -np.inf)
    peak = np.unravel_index(int(np.argmax(scored)), scored.shape)
    return {"angle": float(angle), "peak": peak, "response": float(response[peak]),
            "cover": float(cover[peak]), "background": float(background[peak]),
            "template": template, "offset": offset}


def _centre_of(best, shape):
    offset = best["offset"]
    height, width = shape
    return int(best["peak"][0]) - offset, int(best["peak"][1]) - offset


def _align(sheet, fragment, spec):
    best = None
    coarse = max(spec.angle_step * COARSE_ROTATION_FACTOR, 4.)
    samples = []
    for angle in np.arange(0., 360., coarse):
        found = _scan(sheet, fragment, spec, angle)
        if found is None:
            continue
        row, column = _centre_of(found, sheet.observed.shape)
        if not (0 <= row < sheet.observed.shape[0] and 0 <= column < sheet.observed.shape[1]):
            continue
        samples.append(found)
        if best is None or found["response"] > best["response"]:
            best = found
    if best is None:
        raise Problem("No tested rotation places the fragment's centre inside this record.")
    fine = np.mod(np.arange(best["angle"] - coarse, best["angle"] + coarse + 1e-9, spec.angle_step), 360.)
    for angle in fine:
        found = _scan(sheet, fragment, spec, angle)
        if found is None:
            continue
        row, column = _centre_of(found, sheet.observed.shape)
        if not (0 <= row < sheet.observed.shape[0] and 0 <= column < sheet.observed.shape[1]):
            continue
        samples.append(found)
        if found["response"] > best["response"]:
            best = found
    best["alternatives"] = _rotation_alternatives(samples, best)
    return best


def _rotation_alternatives(samples, best, share=.90, separation=20.):
    """Count distinct rotations that reach nearly the best matched-filter response."""
    if best is None or best["response"] <= 1e-9:
        return 0
    peaks = []
    for item in samples:
        if item["response"] >= share * best["response"]:
            peaks.append(item["angle"] % 360.)
    if not peaks:
        return 0
    peaks.sort()
    groups = 1
    for previous, current in zip(peaks[:-1], peaks[1:], strict=True):
        if current - previous >= separation:
            groups += 1
    wrap = peaks[0] + 360. - peaks[-1]
    if wrap < separation and groups > 1:
        groups -= 1
    return groups


def _centre_cell(best, sheet):
    """Map the correlation peak to the record cell under the template centre."""
    row, column = _centre_of(best, sheet.observed.shape)
    if not (0 <= row < sheet.observed.shape[0] and 0 <= column < sheet.observed.shape[1]):
        raise Problem("The best placement falls outside this record.")
    return row, column


def _rotated(stroke, angle):
    turn = math.radians(angle)
    cosine, sine = math.cos(turn), math.sin(turn)
    return np.column_stack((stroke[:, 0] * cosine - stroke[:, 1] * sine,
                            stroke[:, 0] * sine + stroke[:, 1] * cosine))


def _probe(sheet, points_mm):
    rows, columns = sheet.cells(points_mm[:, 0], points_mm[:, 1])
    detail, cover = sheet.sample(sheet.detail, rows, columns)
    inside = ((rows >= 0) & (columns >= 0) & (rows < sheet.observed.shape[0] - 1)
              & (columns < sheet.observed.shape[1] - 1))
    return detail, cover, inside


def _stroke_rows(sheet, fragment, best):
    centre_row, centre_col = _centre_cell(best, sheet)
    centre = np.asarray([(centre_col + .5) * sheet.cell, (centre_row + .5) * sheet.cell])
    rows = []
    for index, stroke in enumerate(fragment.sampled_mm, 1):
        points = _rotated(stroke, best["angle"]) + centre
        detail, cover, inside = _probe(sheet, points)
        observed = inside & (cover >= .55)
        length = float(np.sum(np.hypot(*np.diff(points, axis=0).T))) if len(points) > 1 else 0.
        share = float(observed.mean()) if len(observed) else 0.
        value = float(np.mean(detail[observed])) if observed.any() else None
        rows.append({"stroke": index, "observed_fraction": rounded(share),
                     "signed_depth_mm": rounded(value, 5) if value is not None else None,
                     "sampled_length_mm": rounded(length),
                     "observed_length_mm": rounded(length * share),
                     "points_mm": np.round(points, 4).tolist(),
                     "observed": observed.tolist()})
    return rows


def _rim(sheet, fragment, best):
    centre_row, centre_col = _centre_cell(best, sheet)
    centre = np.asarray([(centre_col + .5) * sheet.cell, (centre_row + .5) * sheet.cell])
    start, end = fragment.arc
    samples = max(24, int(abs(end - start) * 1.5))
    angles = np.radians(np.linspace(start, end, samples)) + math.radians(best["angle"])
    profile = []
    for offset in (-1.4, -.9, -.45, 0., .45, .9, 1.4):
        points = centre + (fragment.rim_radius_mm + offset) * np.column_stack((np.cos(angles), np.sin(angles)))
        magnitude, cover = sheet.sample(sheet.gradient, *sheet.cells(points[:, 0], points[:, 1]))
        profile.append((magnitude, cover >= .55))
    stacked = np.vstack([item[0] for item in profile])
    usable = np.vstack([item[1] for item in profile])
    if usable.sum() < 8 or not sheet.observed.any():
        return {"evidence": None, "coverage": 0., "arc_degrees": rounded(abs(end - start))}
    baseline = float(np.median(sheet.gradient[sheet.observed]))
    if baseline <= 1e-9:
        return {"evidence": None, "coverage": 0., "arc_degrees": rounded(abs(end - start))}
    values = np.ma.masked_invalid(np.where(usable, stacked, np.nan))
    height = values.shape[0]
    half = height // 2
    center_band = np.ma.median(values[max(0, half - 1):half + 2], axis=0)
    outer_bands = np.ma.median(np.ma.vstack((values[:max(1, half - 2)], values[half + 3:])), axis=0)
    ridge = np.ma.filled(center_band, 0.).astype(float) - np.ma.filled(outer_bands, 0.).astype(float)
    ridge = np.where(np.ma.getmaskarray(center_band) | np.ma.getmaskarray(outer_bands), 0., ridge)
    peak = float(np.percentile(ridge, 90))
    return {"evidence": rounded(peak / baseline), "coverage": rounded(float(usable.mean())),
            "arc_degrees": rounded(float(abs(end - start)))}


def _measure(sheet, fragment, spec):
    best = _align(sheet, fragment, spec)
    strokes = _stroke_rows(sheet, fragment, best)
    total = sum(row["sampled_length_mm"] for row in strokes) or 1e-9
    supporting = [row for row in strokes if row["observed_fraction"] >= .5]
    supporting_length = float(sum(row["observed_length_mm"] for row in supporting))
    support = float(sum(row["observed_length_mm"] for row in strokes) / total)
    signed = float(best["response"])
    strength = float(np.clip(abs(signed) / (2. * spec.tolerance_mm), 0., 1.))
    signs = []
    for row in supporting:
        if row["signed_depth_mm"] is None:
            continue
        sign = 1. if row["signed_depth_mm"] >= 0 else -1.
        signs.append((sign, row["observed_length_mm"]))
    if signs and signed != 0.:
        overall = 1. if signed > 0 else -1.
        agreeing = sum(length for sign, length in signs if sign == overall)
        coherence = float(agreeing / max(1e-9, sum(length for _, length in signs)))
    else:
        coherence = 0.
    score = .65 * strength + .20 * coherence + .15 * min(1., support / max(spec.min_visible, 1e-6))
    alternatives = int(best.get("alternatives", 0))
    if alternatives >= 2:
        score *= 1. / (1. + .15 * (alternatives - 1))
    rows = [{"stroke": row["stroke"], "observed_fraction": row["observed_fraction"],
             "signed_depth_mm": row["signed_depth_mm"], "observed_length_mm": row["observed_length_mm"],
             "sampled_length_mm": row["sampled_length_mm"]} for row in strokes]
    centre_row, centre_col = _centre_cell(best, sheet)
    return {"best": best, "strokes": strokes, "stroke_rows": rows,
            "supporting": [row["stroke"] for row in supporting],
            "supporting_strokes": [row["stroke"] for row in supporting],
            "supporting_length_mm": rounded(supporting_length),
            "stroke_support": rounded(support), "contrast_mm": rounded(abs(signed), 5),
            "signed_contrast_mm": rounded(signed, 5), "strength": rounded(strength, 5),
            "coherence": rounded(coherence, 4), "score": rounded(float(score), 5),
            "penalised": alternatives >= 2,
            "rotation_degrees": rounded(best["angle"] % 360., 3),
            "rotation_alternatives": alternatives,
            "centre_mm": [rounded((centre_col + .5) * sheet.cell, 4),
                          rounded((centre_row + .5) * sheet.cell, 4)],
            "cover": rounded(best["cover"]), "background_cover": rounded(best["background"])}


def _status(row, spec):
    if row["score"] is None:
        return "unavailable"
    if row["stroke_support"] < spec.min_visible * .8:
        return "too little support"
    if row["strength"] <= WEAK_CONTRAST:
        return "outline resemblance only" if (row.get("rim_evidence") or 0.) >= RIM_ONLY_EVIDENCE else "weak relief"
    return "stroke evidence"


def _relative_status(rows, spec, fragment):
    ranked = [row for row in rows if row["status"] == "stroke evidence"]
    if not ranked:
        return
    best = max(row["strength"] for row in ranked)
    floor = max(WEAK_CONTRAST * .45, best * .45)
    for row in ranked:
        if row["strength"] >= floor:
            continue
        row["status"] = ("outline resemblance only" if (row.get("rim_evidence") or 0.) >= RIM_ONLY_EVIDENCE
                         else "weak relief")


def _label_unresolved(rows, spec, fragment):
    ordered = [row for row in sorted(rows, key=lambda r: -(r["score"] if r["score"] is not None else -1.))
               if row["status"] == "stroke evidence"]
    threshold = max(SHORT_SUPPORT_MM, .45 * fragment.total_length_mm)
    for row in ordered:
        row["short_support"] = bool(row["supporting_length_mm"] < threshold)
        row["weak_fix"] = bool(row.get("score_penalised"))
        row["comparable"] = bool(row["short_support"] or row["weak_fix"])
    label = 0
    for index, row in enumerate(ordered):
        if not row["comparable"] or row.get("unresolved"):
            continue
        peers = [other for other in ordered[:index + 1] if other is not row and other["comparable"]
                 and not other.get("unresolved")
                 and set(other["supporting_strokes"]) == set(row["supporting_strokes"])
                 and abs(other["score"] - row["score"]) <= TIE_SCORE_MARGIN]
        if not peers:
            continue
        label += 1
        for member in [*peers, row]:
            member["unresolved"] = f"U{label}"
            member["status"] = "unresolved together"


def _photo_panel(library, sheet, fragment, spec, measurement, title):
    source = sheet.source
    if not source:
        return _depth_panel(sheet, fragment, measurement, title)
    try:
        record, (blob, _mime) = library.capture_asset(source["id"])
        raw = decode(blob, record["filename"])
        registration_row = _registration_row(library, source["id"], source.get("registration_id"))
        registration = Registration.from_data(registration_row)
    except Problem:
        return _depth_panel(sheet, fragment, measurement, title)
    base = Image.open(BytesIO(raw.preview())).convert("RGB")
    draw = ImageDraw.Draw(base)
    centre_row, centre_col = _centre_cell(measurement["best"], sheet)
    centre = np.asarray([(centre_col + .5) * sheet.cell, (centre_row + .5) * sheet.cell])

    def pixels(points_mm):
        normal = np.column_stack((points_mm[:, 0] / sheet.width_mm, points_mm[:, 1] / sheet.height_mm))
        camera = registration.camera(normal) if registration_row is not None else normal
        return [(float(x) * (base.width - 1), float(y) * (base.height - 1)) for x, y in camera]

    matched = 0
    missing = 0
    for stroke in measurement["strokes"]:
        points = np.asarray(stroke["points_mm"])
        seen = stroke["observed"]
        start = None
        for index, flag_value in enumerate(seen + [False]):
            if flag_value and start is None:
                start = index
            elif not flag_value and start is not None:
                piece = pixels(points[max(0, start - 1):index + 1])
                if len(piece) > 1:
                    draw.line(piece, fill=MINT, width=3)
                    matched += index - start
                start = None
        gaps = _gaps(seen)
        for gap_start, gap_end in gaps:
            piece = pixels(points[max(0, gap_start - 1):min(len(points), gap_end + 1)])
            if len(piece) > 1:
                draw.line(piece, fill=GOLD, width=3)
        missing += sum(1 for value in seen if not value)
    theta = np.radians(np.linspace(*fragment.arc, 60)) + math.radians(measurement["rotation_degrees"])
    rim_points = np.column_stack((np.cos(theta), np.sin(theta))) * fragment.radius_mm + centre
    for x, y in pixels(rim_points):
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=HOT)
    centre_px = pixels(centre[None, :])[0]
    draw.ellipse((centre_px[0] - 5, centre_px[1] - 5, centre_px[0] + 5, centre_px[1] + 5), outline=INK, width=2)
    draw.text((8, 8), title, fill=INK, font=FONT)
    draw.text((8, base.height - 18), f"matched {matched} samples · missing {missing} samples · rim red",
              fill=INK, font=FONT)
    return base


def _gaps(seen):
    spans = []
    start = None
    for index, value in enumerate(seen + [True]):
        if not value and start is None:
            start = index
        elif value and start is not None:
            spans.append((start, index - 1))
            start = None
    return spans


def _depth_panel(sheet, fragment, measurement, title):
    span = Extent.fit(sheet.heights, sheet.observed, zero="symmetric")
    base = scale_bar(heatmap(sheet.heights, sheet.observed, span, DIVERGE, True), sheet.width_mm).convert("RGB")
    draw = ImageDraw.Draw(base)
    centre_row, centre_col = _centre_cell(measurement["best"], sheet)
    centre = np.asarray([(centre_col + .5) * sheet.cell, (centre_row + .5) * sheet.cell])
    scale = (base.width - 1) / sheet.width_mm

    def pixels(points_mm):
        return [(float(x) * scale, float(y) * scale) for x, y in points_mm]

    for stroke in measurement["strokes"]:
        seen = stroke["observed"]
        points = pixels(np.asarray(stroke["points_mm"]))
        for index, value in enumerate(seen):
            if value and len(points) > 1:
                draw.ellipse((points[index][0] - 1, points[index][1] - 1,
                              points[index][0] + 1, points[index][1] + 1), fill=MINT)
    theta = np.radians(np.linspace(*fragment.arc, 60)) + math.radians(measurement["rotation_degrees"])
    rim = np.column_stack((np.cos(theta), np.sin(theta))) * fragment.radius_mm + centre
    for x, y in pixels(rim):
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=HOT)
    draw.text((8, 8), title, fill=INK, font=FONT)
    draw.text((8, base.height - 18), "no original photo available · depth map shown", fill=INK, font=FONT)
    return base


def _fragment_panel(raw, spec, fragment):
    base = Image.open(BytesIO(raw.preview())).convert("RGB")
    draw = ImageDraw.Draw(base)
    width, height = base.size

    def camera(point):
        return float(point[0]) * (width - 1), float(point[1]) * (height - 1)

    for index, stroke in enumerate(spec.strokes, 1):
        points = [camera((point.x, point.y)) for point in stroke]
        draw.line(points, fill=MINT, width=3)
        draw.text((points[0][0] + 4, points[0][1] + 4), f"S{index}", fill=MINT, font=FONT)
    cx, cy = camera(spec.centre.public())
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), outline=INK, width=2)
    source_a = camera(spec.rim_points[0].public())
    source_b = camera(spec.rim_points[1].public())
    for px, py in (source_a, source_b):
        draw.ellipse((px - 4, py - 4, px + 4, py + 4), outline=HOT, width=2)
    radius_px = np.hypot(source_a[0] - cx, source_a[1] - cy)
    first = math.degrees(math.atan2(source_a[1] - cy, source_a[0] - cx))
    second = math.degrees(math.atan2(source_b[1] - cy, source_b[0] - cx))
    if (second - first) % 360. > 180.:
        first, second = second, first
    radians = np.radians(np.linspace(first, second, 80))
    for dx, dy in zip(np.cos(radians), np.sin(radians), strict=True):
        x, y = cx + dx * radius_px, cy + dy * radius_px
        draw.ellipse((x - 1.5, y - 1.5, x + 1.5, y + 1.5), fill=GOLD)
    gone = np.radians(np.linspace(second, first + 360., 80))
    for dx, dy in zip(np.cos(gone), np.sin(gone), strict=True):
        x, y = cx + dx * radius_px, cy + dy * radius_px
        draw.ellipse((x - 1.2, y - 1.2, x + 1.2, y + 1.2), fill=(120, 120, 120))
    draw.text((8, 8), "Fragment photograph · drawn strokes mint · surviving rim gold · missing arc grey",
              fill=INK, font=FONT)
    return base


def _overview_panel(fragment, spec, rows):
    ordered = sorted(rows, key=lambda row: -(row["score"] if row["score"] is not None else -1.))
    width = 780
    canvas = Image.new("RGB", (width, max(120, 46 + 30 * len(ordered))), PAPER)
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 10), "Candidate scores", fill=INK, font=FONT)
    for index, row in enumerate(ordered):
        y = 32 + index * 30
        draw.text((12, y + 3), f'{index + 1:02d}', fill=MINT, font=FONT)
        draw.text((42, y + 3), (row.get("candidate_title") or "Unavailable record")[:34], fill=INK, font=FONT)
        bar = 0. if row["score"] is None else float(row["score"])
        draw.rectangle((360, y, 640, y + 16), fill=(31, 46, 51), outline=(64, 84, 88))
        tone = (106, 230, 196) if row["status"] in ("stroke evidence", "unresolved together") else (249, 197, 108)
        draw.rectangle((360, y, 360 + int(280 * min(1., bar)), y + 16), fill=tone)
        draw.text((648, y + 3), f'{row["score"] if row["score"] is not None else "-"}', fill=INK, font=FONT)
        tag = row["status"] if not row.get("unresolved") else f'{row["status"]} {row["unresolved"]}'
        draw.text((684, y + 3), tag[:20], fill=INK, font=FONT)
    return canvas


def _record_label(library, candidate):
    try:
        record = library.get_inspection(candidate.record)
        surface = library.get_surface(candidate.surface)["surface"]
        return record["title"], surface["title"]
    except Problem:
        return "Unavailable record", ""


def _compare_candidate(library, candidate, fragment, spec):
    try:
        sheet = load_sheet(library, candidate.record, candidate.surface)
    except Problem as issue:
        title, surface_title = _record_label(library, candidate)
        return {"record": candidate.record, "surface": candidate.surface,
                "candidate_title": title, "surface_title": surface_title,
                "score": None, "status": "unavailable", "note": str(issue)}
    try:
        measured = _measure(sheet, fragment, spec)
        surface_title = library.get_surface(candidate.surface)["surface"]["title"]
    except Problem as issue:
        return {"record": candidate.record, "surface": candidate.surface,
                "candidate_title": sheet.title, "surface_title": "",
                "score": None, "status": "unavailable", "note": str(issue)}
    rim = _rim(sheet, fragment, measured["best"])
    row = {"record": candidate.record, "surface": candidate.surface,
           "candidate_title": sheet.title, "surface_title": surface_title,
           "candidate_size_mm": [rounded(sheet.width_mm), rounded(sheet.height_mm)],
           "contrast_mm": measured["contrast_mm"], "signed_contrast_mm": measured["signed_contrast_mm"],
           "strength": measured["strength"], "coherence": measured["coherence"],
           "score": measured["score"], "rotation_degrees": measured["rotation_degrees"],
           "centre_mm": measured["centre_mm"],
           "stroke_support": measured["stroke_support"],
           "supporting_strokes": measured["supporting_strokes"],
           "supporting_length_mm": measured["supporting_length_mm"],
           "rim_evidence": rim["evidence"], "rim_coverage": rim["coverage"],
           "rim_arc_degrees": rim["arc_degrees"],
           "rotation_alternatives": measured["rotation_alternatives"],
           "score_penalised": measured["penalised"],
           "short_support": False}
    row["status"] = _status(row, spec)
    row["_measurement"] = measured
    row["_sheet"] = sheet
    return row


def run_search(library, spec):
    workspace = library.get_surface(spec.surface)
    surface = workspace["surface"]
    capture = next((item for item in workspace["captures"] if item["id"] == spec.capture), None)
    if capture is None:
        raise Problem("The drawing photograph is not on this surface.")
    blob, _mime = library.capture_asset(spec.capture)[1]
    raw = decode(blob, capture["filename"])
    registration_row = _registration_row(library, spec.capture, spec.registration)
    registration = Registration.from_data(registration_row)
    fragment = build_fragment(spec, registration, surface)
    fragment_source = {"id": capture["id"], "title": capture["metadata"]["title"], "hash": capture["hash"],
                       "filename": capture["filename"], "role": capture["metadata"]["role"],
                       "settings": capture["metadata"], "capture_version": capture["version"],
                       "size": capture["image"]["size"], "encoding": capture["image"]["encoding"],
                       "registration": registration_row,
                       "registration_id": registration_row["id"] if registration_row else "identity"}
    results = []
    for candidate in spec.candidates:
        results.append(_compare_candidate(library, candidate, fragment, spec))
    rows = [row for row in results if row is not None]
    _relative_status(rows, spec, fragment)
    _label_unresolved(rows, spec, fragment)
    if not spec.keep_unresolved:
        keep = []
        collapsed = set()
        for row in sorted(rows, key=lambda r: -(r["score"] if r["score"] is not None else -1.)):
            group = row.get("unresolved")
            if group:
                if group in collapsed:
                    continue
                collapsed.add(group)
            elif row["status"] != "stroke evidence":
                continue
            keep.append(row)
        rows = keep
    return {"surface": surface, "capture": capture, "registration": registration, "raw": raw,
            "fragment": fragment, "fragment_source": fragment_source, "rows": rows,
            "workspace": workspace}


def _public_row(row):
    clean = {key: value for key, value in row.items() if not key.startswith("_")}
    if not clean.get("surface_title"):
        clean["surface_title"] = ""
    return clean


def search_report(library, spec):
    found = run_search(library, spec)
    fragment = found["fragment"]
    rows = found["rows"]
    ordered = sorted(rows, key=lambda row: -(row["score"] if row["score"] is not None else -1.))
    stroke_rows = []
    comparisons = []
    assets = 0
    details = []
    for row in ordered[:DETAIL_LIMIT]:
        if row["score"] is None:
            continue
        assets += 1
        key = f"candidate-{assets:02d}"
        title = f'{assets:02d} · {row["candidate_title"]}'
        left = _fragment_panel(found["raw"], spec, fragment)
        right = _photo_panel(library, row["_sheet"], fragment, spec, row["_measurement"], title)
        details.append((key, title, split_visual(left, right, "Fragment", "Candidate original")))
        comparisons.append({"record": row["record"], "surface": row["surface"], "file": key,
                            "title": f'{row["candidate_title"]} · {row["surface_title"]}',
                            "status": row["status"], "group": row.get("unresolved")})
        for stroke in row["_measurement"]["stroke_rows"]:
            stroke_rows.append({"record": row["record"], "candidate": row["candidate_title"],
                                "rotation_degrees": row["rotation_degrees"],
                                "centre_mm": row["centre_mm"], **stroke})
    result = Result("seal-match", "Seal fragment candidates",
                    "Compare drawn strokes at their physical size against stored relative-height records. "
                    "Rim similarity is reported separately and does not move a record up the list.",
                    {"fragment": found["capture"]["metadata"]["title"],
                     "dial_radius_mm": rounded(fragment.radius_mm, 4),
                     "measured_rim_radius_mm": rounded(fragment.rim_radius_mm, 4),
                     "rim_arc_degrees": rounded(abs(fragment.arc[1] - fragment.arc[0])),
                     "drawn_strokes": fragment.stroke_count,
                     "drawn_length_mm": rounded(fragment.total_length_mm),
                     "records_compared": len(rows),
                     "stroke_supported": sum(1 for row in rows if row["status"] == "stroke evidence"),
                     "outline_only": sum(1 for row in rows if row["status"] == "outline resemblance only"),
                     "unresolved_groups": len({row.get("unresolved") for row in rows if row.get("unresolved")}),
                     "rotation_step_degrees": spec.angle_step, "depth_tolerance_mm": spec.tolerance_mm,
                     "comparisons": comparisons,
                     "ranking": [{"record": row["record"], "title": row["candidate_title"],
                                  "score": row["score"], "status": row["status"],
                                  "group": row.get("unresolved")} for row in ordered]},
                    [_public_row(row) for row in ordered])
    result.image("fragment-query", _fragment_panel(found["raw"], spec, fragment))
    result.image("candidate-overview", _overview_panel(fragment, spec, rows))
    for key, label, picture in details:
        result.image(key, picture)
    result.table("candidate-scores", [_public_row(row) for row in ordered])
    result.table("stroke-readout", stroke_rows)
    result.warnings.append("Scores rank stroke relief at the fragment's actual size. A matching outer circle alone "
                           "is shown as rim evidence and is never counted as stroke support.")
    result.warnings.append("Two records that agree on the same short, poorly fixed stretch without further evidence "
                           "stay together under one unresolved label.")
    result.warnings.append("Each retained height island has its own relative zero; compare local relief, not absolute depth.")
    return found, result


def evidence(found, spec):
    candidate_sources = []
    seen = set()
    for row in found["rows"]:
        sheet = row.get("_sheet")
        if sheet is None or not sheet.source:
            continue
        source = dict(sheet.source)
        if not source.get("size"):
            source["size"] = [rounded(sheet.observed.shape[1] * sheet.cell, 4),
                              rounded(sheet.observed.shape[0] * sheet.cell, 4)]
            source["pixel_size"] = [int(sheet.observed.shape[1]), int(sheet.observed.shape[0])]
        source["encoding"] = source.get("encoding", "8-bit RGB code")
        source["registration"] = source.get("registration") or None
        source["settings"] = source.get("settings") or {"azimuth": 0, "elevation": 45,
                                                        "exposure": 0, "role": source.get("role", "raking")}
        source["capture_version"] = source.get("capture_version") or 1
        source["registration_id"] = source.get("registration_id") or "identity"
        source["candidate_record"] = row["record"]
        source["candidate_surface"] = row["surface"]
        if source["id"] in seen:
            continue
        seen.add(source["id"])
        candidate_sources.append(source)
    surface = found["surface"]
    return {"surface": {key: surface[key] for key in ("id", "title", "accession", "width_mm", "height_mm",
                                                       "synthetic", "version")},
            "scale": {"size_mm": [surface["width_mm"], surface["height_mm"]],
                      "size_px": [found["raw"].shape[1], found["raw"].shape[0]]},
            "parameters": spec.public(),
            "sources": [found["fragment_source"], *candidate_sources]}
