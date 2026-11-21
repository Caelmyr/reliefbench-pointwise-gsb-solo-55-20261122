"""Evaluate a photographed color chart against declared linear patch values."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .models import Problem, Region, fields, list_of, number, rounded, text
from .raster import quantize, srgb_to_linear, statistics
from .results import Result
from .visual import COOL, GOLD, HOT, MINT, Trace, flatten_rgb, overlay, plot, split_visual


XYZ_D65 = np.asarray([.95047, 1., 1.08883])
TO_XYZ = np.asarray([[.4124564, .3575761, .1804375],
                     [.2126729, .7151522, .0721750],
                     [.0193339, .1191920, .9503041]])


def _brightness_xyz(lin):
    return lin @ TO_XYZ.T


def _lab(linear):
    XYZ = _brightness_xyz(np.maximum(linear, 0)) / XYZ_D65
    delta = 6 / 29
    n = np.where(XYZ > delta**3, np.cbrt(XYZ), XYZ / (3 * delta**2) + 4 / 29)
    a, b, c = np.moveaxis(n, -1, 0)
    return np.stack((116 * b - 16, 500 * (a - b), 200 * (b - c)), axis=-1)


def delta_lab(left, right):
    return np.linalg.norm(_lab(left) - _lab(right), axis=-1)


def _coordinates(label, rgb):
    values = list_of(rgb, label, 3, 3)
    return np.array([number(v, label, 0., 1.) for v in values], dtype=float)


@dataclass(frozen=True)
class Swatch:
    label: str
    region: Region
    target: np.ndarray
    check_only: bool

    @classmethod
    def parse(cls, obj):
        if not isinstance(obj, dict):
            raise Problem("Each patch needs a label, a rectangle and a reference RGB.")
        fields(obj, "name", "region", "target_srgb", "check")
        label = text(obj.get("name"), "Patch label", 60)
        target = srgb_to_linear(_coordinates("Known sRGB", obj.get("target_srgb")))
        manual = obj.get("check", False)
        if type(manual) is not bool:
            raise Problem("A patch is either a control or a fitting swatch.")
        return cls(label, Region.parse(obj.get("region")), target, manual)


@dataclass(frozen=True)
class PatchData:
    swatch: Swatch
    rgb: np.ndarray | None
    spread: float | None
    covered: int
    geometrical: int
    clipped: int
    notes: str

    @property
    def available(self):
        return self.rgb is not None

    def report(self, matrix):
        row = {"label": self.swatch.label, "region": self.swatch.region.public(),
               "known_linear_rgb": np.round(self.swatch.target, 7).tolist(),
               "role": "check" if self.swatch.check_only else "fit",
               "availability": "valid" if self.available else self.notes,
               "clipped_samples": self.clipped, "usable_pixels": self.covered,
               "pixel_count": self.geometrical, "channel_mad": rounded(self.spread)}
        if not self.available:
            return row
        pred = self.rgb @ matrix
        row.update(measured_linear_rgb=np.round(self.rgb, 7).tolist(),
                   corrected_linear_rgb=np.round(pred, 7).tolist(),
                   deviation_before=rounded(delta_lab(self.rgb, self.swatch.target), 3),
                   deviation_after=rounded(delta_lab(pred, self.swatch.target), 3),
                   gamut_clip=bool(np.min(pred) < -.01 or np.max(pred) > 1.01))
        return row


def _patch_samples(scene, swatch):
    h, w = scene.alpha_valid.shape
    area = swatch.region.window(w, h)
    possible = scene.alpha_valid[area]
    clipped = scene.clipped[area] & possible
    visible = possible & ~clipped
    good = int(visible.sum())
    if good < max(24, round(.70 * visible.size)):
        return PatchData(swatch, None, None, good, visible.size, int(clipped.sum()), "too little unclipped coverage")
    samples = scene.linear[area][visible]
    group = np.median(samples, axis=0)
    deviation = np.median(np.abs(samples - group), axis=0)
    spread = float(np.max(deviation))
    if spread > .055:
        return PatchData(swatch, None, spread, good, visible.size, int(clipped.sum()), "non-uniform patch")
    return PatchData(swatch, group, spread, good, visible.size, int(clipped.sum()), "")


def _region_inside(outer, inner):
    return outer.left <= inner.left < inner.right <= outer.right and outer.top <= inner.top < inner.bottom <= outer.bottom


def _parse_chart(raw, layout):
    swatches = [Swatch.parse(item) for item in list_of(raw, "Chart swatches", 4, 22)]
    if len(set(s.label for s in swatches)) != len(swatches):
        raise Problem("Name each chart patch independently.")
    for i, s in enumerate(swatches):
        if not _region_inside(layout, s.region):
            raise Problem("The selected chart window must cover every swatch.")
        for prev in swatches[:i]:
            a, b = s.region, prev.region
            if min(a.right, b.right) - max(a.left, b.left) > .002 and min(a.bottom, b.bottom) - max(a.top, b.top) > .002:
                raise Problem("Chart swatches should not overlap.")
    return swatches


def _matrix(use, method, neutral_weight):
    source = np.array([row.rgb for row in use])
    goal = np.array([row.swatch.target for row in use])
    spread = np.array([row.spread for row in use])
    target_spread = np.ptp(goal, axis=1)
    weights = (1 + neutral_weight * (target_spread < .012)) / (.005 + spread)
    weights /= weights.max()
    strengths = np.sqrt(weights)
    if method == "channels":
        counts = np.sum((source * strengths[:, None])**2, axis=0)
        denom = np.maximum(counts, 1e-6)
        diag = np.sum(source * goal * weights[:, None], axis=0) / denom
        return np.diag(np.clip(diag, .3, 3.))
    design = source * strengths[:, None]
    singular = np.linalg.svd(design, compute_uv=False)
    if singular[-1] < .045 * singular[0]:
        raise Problem("The measured colors do not constrain a cross-channel correction.")
    lhs = design.T @ design + .02 * np.eye(3)
    rhs = design.T @ (goal * strengths[:, None]) + .02 * np.eye(3)
    transform = np.linalg.solve(lhs, rhs)
    if np.linalg.cond(transform) > 15:
        raise Problem("The extrapolated color transform is too strong for a reference reading.")
    return transform


def _report_comparison(rows):
    groups = {"fit": [], "check": []}
    for row in rows:
        if "deviation_after" in row:
            groups[row["role"]].append(row)
    returns = {}
    for name, group in groups.items():
        prior = [r["deviation_before"] for r in group]
        after = [r["deviation_after"] for r in group]
        returns[name] = {"readable": len(group),
                         "median_before": rounded(np.median(prior), 3) if prior else None,
                         "median_after": rounded(np.median(after), 3) if after else None,
                         "gamut_flags": sum(row["gamut_clip"] for row in group)}
    return returns


def _swatch_plate(data, matrix):
    width, height = 600, len(data) * 39 + 32
    canvas = Image.new("RGB", (width, height), (24, 37, 41))
    draw = ImageDraw.Draw(canvas)
    for x, label in [(14, "Patch"), (225, "Measured"), (355, "Transformed"), (490, "Reference")]:
        draw.text((x, 10), label, fill=(227, 234, 231))
    for idx, patch in enumerate(data):
        y = 29 + 39 * idx
        draw.text((14, y + 10), patch.swatch.label[:42], fill=MINT if patch.available else HOT)
        colors = [patch.rgb, patch.rgb @ matrix if patch.available else None, patch.swatch.target]
        for j, rgb in enumerate(colors):
            x = 220 + j * 130
            fill = tuple(quantize(rgb).tolist()) if rgb is not None else (44, 58, 67)
            draw.rectangle((x, y + 1, x + 100, y + 30), fill=fill, outline=(98, 108, 104))
    return canvas


DEFAULT_CHART = [
    {"name": "Blue", "region": [.18, .22, .28, .33], "target_srgb": [.18, .33, .64]},
    {"name": "Red", "region": [.39, .22, .49, .33], "target_srgb": [.71, .29, .25]},
    {"name": "Green", "region": [.60, .22, .70, .33], "target_srgb": [.24, .61, .31]},
    {"name": "Gray", "region": [.18, .50, .28, .61], "target_srgb": [.53, .53, .53]},
    {"name": "Ochre check", "region": [.39, .50, .49, .61], "target_srgb": [.73, .56, .22], "check": True},
    {"name": "Light check", "region": [.60, .50, .70, .61], "target_srgb": [.79, .80, .80], "check": True},
]


def chart_report(dataset, options):
    fields(options, "swatches", "transform", "neutral_priority")
    frame = dataset.one()
    raw = options.get("swatches", DEFAULT_CHART)
    probes = _parse_chart(raw, dataset.region)
    method = options.get("transform", "channels")
    if method not in ("channels", "cross-channel"):
        raise Problem("Choose channel gains or a cross-channel correction.")
    weight = number(options.get("neutral_priority", .65), "Neutral priority", 0, 4)
    measured = [_patch_samples(frame, swatch) for swatch in probes]
    fit = [row for row in measured if row.available and not row.swatch.check_only]
    if len(fit) < 3:
        raise Problem("The captured chart needs at least three usable fitting swatches.")
    matrix = _matrix(fit, method, weight)
    rows = [row.report(matrix) for row in measured]
    response = _report_comparison(rows)
    result = Result("color-chart", "Color reference check",
                    "Solve a restrained photo color correction and show the error on independent check patches.",
                    {"transform": method, "linear_rgb_row_matrix": np.round(matrix, 7).tolist(), **response}, rows)
    result.image("swatch-plate", _swatch_plate(measured, matrix))
    regions = [("closed", p.swatch.region.corners(), (COOL if p.swatch.check_only else MINT) if p.available else HOT,
                p.swatch.label) for p in measured]
    context = flatten_rgb(frame.code, frame.alpha_valid, False)
    result.image("chart-source", overlay(context, regions))
    source = frame.linear
    balanced = source @ matrix
    common = frame.alpha_valid & dataset.selection
    previous = flatten_rgb(source, common)
    current = flatten_rgb(balanced, common)
    result.image("color-correction", split_visual(previous, current, "Linear view", "Corrected preview"))
    if not response["check"]["readable"]:
        result.warnings.append("No usable check-only patch. The fit samples describe training error only.")
    result.warnings.append("The correction estimates a chart-space transform and stores a preview. It does not change the photo.")
    return result.table("color-reference-swatches").document("chart-linear-response", {"matrix": matrix.tolist(), "patches": rows})
