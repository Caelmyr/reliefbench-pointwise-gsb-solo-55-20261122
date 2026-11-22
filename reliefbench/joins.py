"""Join two relative height records on one physical millimetre grid.

Each record keeps its own zero and its own connected islands. Matched marks on
the two records fit one similarity transform (rotation, scale, translation), so
a slightly skewed second record is placed in the left record's millimetre frame.
A vertical datum is estimated per right-hand island, because height islands have
independent zeros. Overlap samples that disagree beyond the stated limit are
left unmerged and remain visible in both source readings.
"""
from dataclasses import dataclass
from io import BytesIO
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .models import Point, Problem, choice, fields, list_of, number, rounded
from .provenance import fingerprint
from .raster import png, safe_fraction, tiff_float
from .results import Result
from .visual import (COOL, DIVERGE, FONT, GAP, GOLD, HOT, INK, MINT, PAPER,
                     Extent, heatmap, scale_bar)

NONE, LEFT, RIGHT, BOTH, CONFLICT, UNSET = 0, 1, 2, 3, 4, 5
CLASS_LABELS = {NONE: "no evidence", LEFT: "left only", RIGHT: "right only",
                BOTH: "both agree", CONFLICT: "both disagree", UNSET: "right, no datum"}
CLASS_COLORS = {NONE: GAP, LEFT: MINT, RIGHT: COOL, BOTH: GOLD, CONFLICT: HOT,
                UNSET: (157, 131, 198)}
MAX_POINTS = 24
MAX_CELLS = 1_200_000
HEIGHT_FILE = "relative-height-mm.tiff"
ISLAND_FILE = "height-island-id.tiff"


def _tiff(raw, label):
    try:
        image = Image.open(BytesIO(raw))
        image.load()
        values = np.asarray(image)
    except (OSError, ValueError) as exc:
        raise Problem(f"The stored {label} raster could not be read.") from exc
    if values.ndim != 2:
        raise Problem(f"The stored {label} raster is not a single channel.")
    return values.astype(np.float32)


def _asset(record, key, name):
    found = [item for item in record["files"] if item["key"] == key]
    if not found:
        found = [item for item in record["files"] if item["name"].split("/")[-1] == name]
    return (key, found[0]) if found else (None, None)


def _grid_metrics(record):
    report = record.get("report") or {}
    candidates = [(report.get("metrics") or {}).get("height_grid"),
                  (record.get("evidence") or {}).get("scale")]
    for grid in candidates:
        if not isinstance(grid, dict):
            continue
        try:
            width_mm, height_mm = (float(v) for v in grid["size_mm"])
            width_px, height_px = (int(v) for v in grid["size_px"])
        except (KeyError, TypeError, ValueError):
            continue
        if width_mm > 0 and height_mm > 0 and width_px > 1 and height_px > 1:
            return width_mm, height_mm, width_px, height_px
    raise Problem("This height record does not carry a physical working grid.")


@dataclass(frozen=True)
class HeightRecord:
    id: str
    title: str
    surface_id: str
    surface_title: str
    width_mm: float
    height_mm: float
    height: np.ndarray
    valid: np.ndarray
    islands: np.ndarray
    height_hash: str
    height_key: str
    island_hash: str | None
    island_key: str | None
    island_source: str

    @property
    def width(self):
        return int(self.valid.shape[1])

    @property
    def rows(self):
        return int(self.valid.shape[0])

    @property
    def dx(self):
        return self.width_mm / self.width

    @property
    def dy(self):
        return self.height_mm / self.rows

    def sample(self, points):
        points = np.asarray(points, dtype=np.float64)
        inside = ((points[:, 0] >= 0) & (points[:, 0] <= self.width_mm)
                  & (points[:, 1] >= 0) & (points[:, 1] <= self.height_mm))
        coords = [points[:, 1] / self.dy - .5, points[:, 0] / self.dx - .5]
        weight = ndimage.map_coordinates(self.valid.astype(np.float32), coords, order=1, mode="nearest")
        values = ndimage.map_coordinates(np.where(self.valid, self.height, 0.), coords,
                                         order=1, mode="nearest")
        islands = ndimage.map_coordinates(self.islands.astype(np.float32), coords, order=0, mode="nearest")
        okay = inside & (weight >= .98)
        result = np.divide(values, weight, out=np.full(values.shape, math.nan), where=weight >= .98)
        return result, okay, np.where(okay, np.round(islands).astype(np.int32), 0)

    def public(self):
        return {"id": self.id, "title": self.title, "surface_id": self.surface_id,
                "surface_title": self.surface_title, "size_mm": [rounded(self.width_mm), rounded(self.height_mm)],
                "size_px": [self.width, self.rows], "sampling_mm": [rounded(self.dx), rounded(self.dy)],
                "observed_fraction": rounded(float(self.valid.mean())),
                "islands": int(self.islands.max()), "island_source": self.island_source,
                "height_sha256": self.height_hash, "island_sha256": self.island_hash}

    def preview(self):
        extent = Extent.fit(self.height, self.valid, zero="symmetric")
        return png(scale_bar(heatmap(self.height, self.valid, extent, DIVERGE, True), self.width_mm))


def load_record(library, inspection_id):
    record = library.get_inspection(inspection_id)
    if record["kind"] != "height":
        raise Problem("Choose a saved Relative height inspection for each side.")
    height_key, height_meta = _asset(record, "height-samples", HEIGHT_FILE)
    if height_meta is None:
        raise Problem(f'{record["title"]} has no retained relative-height raster.')
    height = _tiff(library.inspection_asset(record["id"], height_key)[0], "height")
    width_mm, height_mm, width_px, height_px = _grid_metrics(record)
    if height.shape != (height_px, width_px):
        raise Problem(f'{record["title"]} does not match its recorded working grid.')
    valid = np.isfinite(height)
    island_key, island_meta = _asset(record, "height-support", ISLAND_FILE)
    if island_meta is not None:
        islands = _tiff(library.inspection_asset(record["id"], island_key)[0], "island")
        if islands.shape != height.shape:
            raise Problem(f'{record["title"]} has an island map of a different size.')
        islands = np.round(np.nan_to_num(islands)).astype(np.int32)
        islands[~valid] = 0
        valid = valid & (islands > 0)
        island_source = "stored island map"
    else:
        islands = np.where(valid, 1, 0).astype(np.int32)
        island_source = "assumed one island"
    if int(valid.sum()) < 64:
        raise Problem(f'{record["title"]} has too little observed height to join.')
    surface = (record.get("evidence") or {}).get("surface") or {}
    return HeightRecord(record["id"], record["title"], surface.get("id", ""), surface.get("title", ""),
                        width_mm, height_mm, height, valid, islands,
                        height_meta["hash"], height_key, island_meta["hash"] if island_meta else None,
                        island_key, island_source)


def available(library):
    rows = []
    for summary in library.height_records():
        try:
            loaded = load_record(library, summary["id"])
        except Problem as issue:
            rows.append({"id": summary["id"], "title": summary["title"], "surface_id": summary["surface_id"],
                         "surface_title": summary["surface_title"], "created_at": summary["created_at"],
                         "available": False, "reason": str(issue)})
            continue
        rows.append({**loaded.public(), "created_at": summary["created_at"], "available": True,
                     "preview": f"/api/heights/{summary['id']}/preview.png"})
    return rows


def preview_record(library, inspection_id):
    return load_record(library, inspection_id).preview()


@dataclass(frozen=True)
class Similarity:
    scale: float
    rotation: np.ndarray
    translation: np.ndarray

    def forward(self, points):
        points = np.asarray(points, dtype=np.float64)
        return self.scale * (points @ self.rotation.T) + self.translation

    def inverse(self, points):
        points = np.asarray(points, dtype=np.float64)
        return (points - self.translation) @ self.rotation / self.scale

    @property
    def angle_deg(self):
        return math.degrees(math.atan2(self.rotation[1, 0], self.rotation[0, 0]))

    def matrix(self):
        return np.r_[np.c_[self.scale * self.rotation, self.translation], [[0., 0., 1.]]]

    def public(self):
        return {"scale": rounded(self.scale), "rotation_deg": rounded(self.angle_deg, 6),
                "rotation": np.round(self.rotation, 9).tolist(),
                "translation_mm": [rounded(v) for v in self.translation],
                "forward_matrix": np.round(self.matrix(), 9).tolist()}


def fit_similarity(source, target):
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 3:
        raise Problem("Mark at least three matched positions on each height record.")
    if not (np.isfinite(source).all() and np.isfinite(target).all()):
        raise Problem("Every marked position needs finite coordinates.")
    source_center, target_center = source.mean(axis=0), target.mean(axis=0)
    x, y = source - source_center, target - target_center
    spread = np.linalg.svd(x, compute_uv=False)
    if spread[0] < .35 or spread[1] < .08 * spread[0]:
        raise Problem("Marked positions lie almost on one line. Include a matched position off that line "
                      "so rotation and scale can be fixed.")
    left, values, right = np.linalg.svd(x.T @ y)
    turn = float(np.sign(np.linalg.det(right.T @ left.T))) or 1.
    rotation = right.T @ np.diag([1., turn]) @ left.T
    denominator = float(np.sum(x * x))
    scale = float((values * np.array([1., turn])).sum() / denominator) if denominator else math.nan
    if not math.isfinite(scale) or not .5 <= scale <= 2.:
        raise Problem("These matched marks imply an implausible size change. Check the pairing and units.")
    transform = Similarity(scale, rotation, target_center - scale * (rotation @ source_center))
    residual = np.linalg.norm(transform.forward(source) - target, axis=1)
    fit = {"rms_mm": rounded(float(np.sqrt(np.mean(residual ** 2))), 6),
           "p95_mm": rounded(float(np.percentile(residual, 95)), 6),
           "max_mm": rounded(float(np.max(residual)), 6)}
    return transform, residual, fit


@dataclass(frozen=True)
class JoinedGrid:
    height: np.ndarray
    code: np.ndarray
    difference: np.ndarray
    left_height: np.ndarray
    right_native: np.ndarray
    right_adjusted: np.ndarray
    left_island: np.ndarray
    right_island: np.ndarray
    origin: tuple
    spacing: float
    left: HeightRecord
    right: HeightRecord
    transform: Similarity
    datum: dict
    options: dict
    overlap: dict
    fit: dict
    control_points: list
    transform_public: dict

    @property
    def shape(self):
        return self.code.shape

    @property
    def extent(self):
        return [self.origin[0], self.origin[1],
                self.origin[0] + self.shape[1] * self.spacing,
                self.origin[1] + self.shape[0] * self.spacing]

    @property
    def width_mm(self):
        return self.extent[2] - self.extent[0]

    def counts(self):
        area = self.spacing ** 2
        total = int(self.code.size)
        rows = []
        for code in (NONE, LEFT, RIGHT, BOTH, CONFLICT, UNSET):
            cells = int(np.count_nonzero(self.code == code))
            rows.append({"class": CLASS_LABELS[code], "cells": cells, "area_mm2": rounded(cells * area),
                         "share": rounded(cells / total) if total else None})
        checked = int(np.count_nonzero(np.isfinite(self.difference)))
        rows.append({"class": "checked overlap", "cells": checked, "area_mm2": rounded(checked * area),
                     "share": rounded(safe_fraction(checked, total))})
        return rows


def _mark_datum(groups):
    offsets = []
    for island, members in sorted(groups.items()):
        differences = np.asarray(members["differences"], dtype=float)
        offsets.append({"right_island": int(island), "offset_mm": rounded(float(np.median(differences))),
                        "samples": int(differences.size),
                        "difference_mad_mm": rounded(float(np.median(np.abs(differences
                                                                             - np.median(differences))))),
                        "difference_range_mm": [rounded(float(differences.min())),
                                                rounded(float(differences.max()))],
                        "left_islands": sorted(int(v) for v in members["left_islands"]),
                        "method": "matched marks"})
    return offsets


def _overlap_datum(valid, left_island, right_island, differences, minimum=16):
    offsets = []
    uncalibrated = []
    for island in sorted(set(int(v) for v in right_island[valid] if v > 0)):
        member = valid & (right_island == island)
        if int(member.sum()) < minimum:
            uncalibrated.append(island)
            continue
        values = differences[member]
        offsets.append({"right_island": island, "offset_mm": rounded(float(np.median(values))),
                        "samples": int(member.sum()),
                        "difference_mad_mm": rounded(float(np.median(np.abs(values - np.median(values))))),
                        "difference_range_mm": [rounded(float(values.min())),
                                                rounded(float(values.max()))],
                        "left_islands": sorted(set(int(v) for v in left_island[member] if v > 0)),
                        "method": "shared overlap"})
    return offsets, uncalibrated


def _manifest(joined):
    return {"format": "reliefbench.height-join.v1",
            "left": joined.left.public(), "right": joined.right.public(),
            "transform": joined.transform_public, "datum": joined.datum,
            "options": joined.options, "control_points": joined.control_points,
            "fit": joined.fit, "overlap": joined.overlap,
            "grid": {"origin_mm": [rounded(v) for v in joined.origin],
                     "spacing_mm": rounded(joined.spacing),
                     "size_px": [int(joined.shape[1]), int(joined.shape[0])],
                     "extent_mm": [rounded(v) for v in joined.extent]},
            "classes": {str(code): CLASS_LABELS[code] for code in CLASS_LABELS}}


def pack(joined):
    return {"height": joined.height.astype(np.float32),
            "code": joined.code.astype(np.uint8),
            "difference": joined.difference.astype(np.float32),
            "left_height": joined.left_height.astype(np.float32),
            "right_native": joined.right_native.astype(np.float32),
            "right_adjusted": joined.right_adjusted.astype(np.float32),
            "left_island": joined.left_island.astype(np.int32),
            "right_island": joined.right_island.astype(np.int32)}


def grid_bytes(joined):
    buffer = BytesIO()
    np.savez_compressed(buffer, **pack(joined))
    return buffer.getvalue()


def build(library, left_id, right_id, raw_points, raw_options):
    fields(raw_options, "sample_mm", "agreement_mm", "contours_mm", "datum", "offset_mm")
    sample_mm = number(raw_options.get("sample_mm", .25), "Output sampling, mm", .02, 5.)
    agreement = number(raw_options.get("agreement_mm", .05), "Agreement limit, mm", .002, 5.)
    contours = number(raw_options.get("contours_mm", .08), "Contour interval, mm", .005, 5.)
    datum_method = choice(raw_options.get("datum", "marks"), "Vertical datum", ("marks", "overlap", "manual"))
    manual_offset = number(raw_options.get("offset_mm", 0.), "Vertical offset, mm", -50., 50.)
    fields(raw_points, "left", "right")
    left_raw = list_of(raw_points.get("left"), "Left marked positions", 3, MAX_POINTS)
    right_raw = list_of(raw_points.get("right"), "Right marked positions", 3, MAX_POINTS)
    if len(left_raw) != len(right_raw):
        raise Problem("Pair every left marked position with one right position.")
    left_points = [Point.parse(item, "Left marked position") for item in left_raw]
    right_points = [Point.parse(item, "Right marked position") for item in right_raw]
    left = load_record(library, left_id)
    right = load_record(library, right_id)
    if left.id == right.id:
        raise Problem("Choose two different height records.")
    left_mm = np.array([p.public() for p in left_points]) * [left.width_mm, left.height_mm]
    right_mm = np.array([p.public() for p in right_points]) * [right.width_mm, right.height_mm]
    transform, residual, fit = fit_similarity(right_mm, left_mm)

    marks_left, okay_left, mark_left_island = left.sample(left_mm)
    marks_right, okay_right, mark_right_island = right.sample(right_mm)
    usable_marks = okay_left & okay_right
    difference_marks = np.subtract(marks_left, marks_right)
    if datum_method == "marks":
        groups = {}
        for index in np.flatnonzero(usable_marks):
            island = int(mark_right_island[index])
            groups.setdefault(island, {"differences": [], "left_islands": []})
            groups[island]["differences"].append(float(difference_marks[index]))
            groups[island]["left_islands"].append(int(mark_left_island[index]))
        groups = {island: value for island, value in groups.items() if len(value["differences"]) >= 2}
        if not groups:
            raise Problem("Fewer than two marked positions share observed height on both records, so the vertical "
                          "datum cannot be estimated from the marks. Use the overlap or enter an offset.")
    else:
        groups = {}

    corners = transform.forward(np.array([[0., 0.], [right.width_mm, 0.],
                                          [right.width_mm, right.height_mm], [0., right.height_mm]]))
    x0 = min(0., float(corners[:, 0].min()))
    y0 = min(0., float(corners[:, 1].min()))
    x1 = max(left.width_mm, float(corners[:, 0].max()))
    y1 = max(left.height_mm, float(corners[:, 1].max()))
    nx = int(math.ceil((x1 - x0) / sample_mm))
    ny = int(math.ceil((y1 - y0) / sample_mm))
    if nx * ny > MAX_CELLS:
        raise Problem(f"The joined grid would need {nx} by {ny} samples. Choose a coarser output sampling "
                      f"(at least {math.sqrt((x1 - x0) * (y1 - y0) / MAX_CELLS):.3f} mm).")
    xs = x0 + (np.arange(nx) + .5) * sample_mm
    ys = y0 + (np.arange(ny) + .5) * sample_mm
    xv, yv = np.meshgrid(xs, ys)
    points = np.stack((xv.ravel(), yv.ravel()), axis=-1)
    left_height, left_ok, left_island = left.sample(points)
    right_local = transform.inverse(points)
    right_native, right_ok, right_island = right.sample(right_local)
    raw_difference = np.subtract(left_height, right_native)
    both_native = left_ok & right_ok

    if datum_method == "overlap":
        offsets, uncalibrated = _overlap_datum(both_native, left_island, right_island, raw_difference)
    elif datum_method == "manual":
        offsets = [{"right_island": int(island), "offset_mm": rounded(manual_offset), "samples": 0,
                    "difference_mad_mm": None, "difference_range_mm": None, "left_islands": [],
                    "method": "entered offset"}
                   for island in sorted(set(int(v) for v in right_island[right_ok] if v > 0))]
        uncalibrated = []
    else:
        offsets = _mark_datum(groups)
        uncalibrated = [int(island) for island in sorted(set(int(v) for v in right_island[right_ok] if v > 0))
                        if int(island) not in {row["right_island"] for row in offsets}]
    if not offsets:
        raise Problem("No vertical datum could be established for any right-record island from this method. "
                      "Mark positions on a shared island, use the overlap, or enter an offset.")

    shift = np.full(points.shape[0], math.nan)
    for row in offsets:
        shift[(right_island == row["right_island"]) & right_ok] = row["offset_mm"]
    have_right = right_ok & np.isfinite(shift)
    right_adjusted = right_native + shift
    both = left_ok & have_right
    difference = np.full(points.shape[0], math.nan)
    difference[both] = left_height[both] - right_adjusted[both]
    agreement_met = both & (np.abs(difference) <= agreement)

    code = np.full(points.shape[0], NONE, dtype=np.uint8)
    code[right_ok & ~have_right] = UNSET
    code[left_ok] = LEFT
    code[have_right] = RIGHT
    code[agreement_met] = BOTH
    code[both & ~agreement_met] = CONFLICT

    height = np.full(points.shape[0], math.nan)
    height[code == LEFT] = left_height[code == LEFT]
    height[code == RIGHT] = right_adjusted[code == RIGHT]
    height[code == BOTH] = .5 * (left_height[code == BOTH] + right_adjusted[code == BOTH])

    overlap_values = difference[np.isfinite(difference)]
    overlap = {"cells": int(both.sum()), "area_mm2": rounded(float(both.sum()) * sample_mm ** 2),
               "fraction_of_joined": rounded(safe_fraction(both, points.shape[0])),
               "difference_median_mm": rounded(float(np.median(overlap_values))) if overlap_values.size else None,
               "difference_mad_mm": rounded(float(np.median(np.abs(overlap_values - np.median(overlap_values)))))
               if overlap_values.size else None,
               "difference_p95_abs_mm": rounded(float(np.percentile(np.abs(overlap_values), 95)))
               if overlap_values.size else None,
               "conflict_cells": int(np.count_nonzero(code == CONFLICT)),
               "conflict_share_of_overlap": rounded(safe_fraction(code == CONFLICT, both.sum()))}

    datum = {"method": {"marks": "matched marks", "overlap": "shared overlap",
                        "manual": "entered offset"}[datum_method],
             "island_offsets": offsets, "uncalibrated_right_islands": uncalibrated,
             "mark_rows": [{"pair": index + 1,
                            "right_island": int(mark_right_island[index]) if okay_right[index] else None,
                            "left_island": int(mark_left_island[index]) if okay_left[index] else None,
                            "left_height_mm": rounded(marks_left[index]) if usable_marks[index] else None,
                            "right_height_mm": rounded(marks_right[index]) if usable_marks[index] else None,
                            "difference_mm": rounded(difference_marks[index]) if usable_marks[index] else None}
                           for index in range(len(left_mm))]}
    transform_public = transform.public()
    control_points = []
    for index in range(len(left_mm)):
        control_points.append({"pair": index + 1,
                               "left_mm": [rounded(v) for v in left_mm[index]],
                               "right_mm": [rounded(v) for v in right_mm[index]],
                               "right_mapped_mm": [rounded(v) for v in transform.forward(right_mm[index])],
                               "residual_mm": rounded(residual[index]),
                               "left_island": int(mark_left_island[index]) if okay_left[index] else None,
                               "right_island": int(mark_right_island[index]) if okay_right[index] else None,
                               "datum_difference_mm": datum["mark_rows"][index]["difference_mm"]})
    shape = (ny, nx)
    joined = JoinedGrid(height.reshape(shape), code.reshape(shape), difference.reshape(shape),
                        left_height.reshape(shape), right_native.reshape(shape), right_adjusted.reshape(shape),
                        left_island.reshape(shape), right_island.reshape(shape),
                        (float(x0), float(y0)), float(sample_mm), left, right, transform, datum,
                        {"sample_mm": sample_mm, "agreement_mm": agreement, "contours_mm": contours,
                         "datum": datum_method, "offset_mm": manual_offset},
                        overlap, fit, control_points, transform_public)
    result = _result(joined, contours)
    return result, joined


def _legend(image, entries):
    draw = ImageDraw.Draw(image)
    width = max(draw.textlength(label, font=FONT) for _color, label in entries) + 26
    draw.rounded_rectangle((8, 8, 8 + width, 8 + len(entries) * 13 + 8), radius=4, fill=PAPER)
    for index, (color, label) in enumerate(entries):
        y = 14 + index * 13
        draw.rectangle((14, y, 22, y + 8), fill=color)
        draw.text((27, y - 2), label, fill=INK, font=FONT)
    return image


def _contour(values, observed, interval):
    level = np.zeros(values.shape, dtype=np.int64)
    level[observed] = np.floor(values[observed] / interval)
    edge_x = observed[:, :-1] & observed[:, 1:] & (level[:, :-1] != level[:, 1:])
    edge_y = observed[:-1, :] & observed[1:, :] & (level[:-1, :] != level[1:, :])
    mask = np.zeros(values.shape, dtype=bool)
    mask[:, 1:] |= edge_x
    mask[1:, :] |= edge_y
    return mask


def _heatmap(joined):
    observed = np.isfinite(joined.height)
    extent = Extent.fit(joined.height, observed, zero="symmetric")
    return heatmap(joined.height, observed, extent, DIVERGE, True)


def _segment_image(joined):
    pixels = np.empty((*joined.shape, 3), dtype=np.uint8)
    for code, color in CLASS_COLORS.items():
        pixels[joined.code == code] = color
    return _legend(Image.fromarray(pixels), [(CLASS_COLORS[code], CLASS_LABELS[code])
                                             for code in (LEFT, RIGHT, BOTH, CONFLICT, UNSET, NONE)])


def _overlap_image(joined, interval):
    canvas = np.asarray(_heatmap(joined)).copy()
    left_lines = _contour(joined.left_height, np.isfinite(joined.left_height), interval)
    right_lines = _contour(joined.right_adjusted, np.isfinite(joined.right_adjusted), interval)
    canvas[left_lines] = MINT
    canvas[right_lines] = COOL
    canvas[left_lines & right_lines] = GOLD
    disagree = joined.code == CONFLICT
    canvas[disagree] = (canvas[disagree].astype(int) // 2 + np.array(HOT) // 2).astype(np.uint8)
    return _legend(Image.fromarray(canvas), [(MINT, "left contours"), (COOL, "right contours"),
                                             (GOLD, "coincident"), (HOT, "disagreement")])


def _conflict_regions(joined, limit=48):
    labels, count = ndimage.label(joined.code == CONFLICT, structure=np.ones((3, 3)))
    rows = []
    for label in range(1, count + 1):
        member = labels == label
        values = np.abs(joined.difference[member])
        values = values[np.isfinite(values)]
        yy, xx = np.nonzero(member)
        rows.append({"cells": int(member.sum()), "area_mm2": rounded(int(member.sum()) * joined.spacing ** 2),
                     "box_mm": [rounded(joined.origin[0] + xx.min() * joined.spacing),
                                rounded(joined.origin[1] + yy.min() * joined.spacing),
                                rounded(joined.origin[0] + (xx.max() + 1) * joined.spacing),
                                rounded(joined.origin[1] + (yy.max() + 1) * joined.spacing)],
                     "median_abs_difference_mm": rounded(float(np.median(values))) if values.size else None,
                     "max_abs_difference_mm": rounded(float(np.max(values))) if values.size else None})
    rows.sort(key=lambda row: -row["cells"])
    for index, row in enumerate(rows, 1):
        row["region"] = index
    return rows[:limit]


def _island_rows(joined):
    rows = []
    for side, values in (("left", joined.left_island), ("right", joined.right_island)):
        for island in sorted(set(int(v) for v in values.ravel() if v > 0)):
            member = values == island
            rows.append({"side": side, "island": island, "cells": int(member.sum()),
                         "area_mm2": rounded(int(member.sum()) * joined.spacing ** 2)})
    return rows


def _result(joined, contours):
    counts = joined.counts()
    conflicts = _conflict_regions(joined)
    observed_left_islands = sorted(set(int(v) for v in joined.left_island.ravel() if v > 0))
    offsets = joined.datum["island_offsets"]
    result = Result("height-join", "Joined relative height",
                    "Place two relative height records on one millimetre grid. Disagreeing overlap stays unmerged.",
                    {"left_record": joined.left.title, "right_record": joined.right.title,
                     "left_surface": joined.left.surface_title, "right_surface": joined.right.surface_title,
                     "output_origin_mm": [rounded(joined.origin[0]), rounded(joined.origin[1])],
                     "output_spacing_mm": joined.spacing, "output_size_px": [int(joined.shape[1]), int(joined.shape[0])],
                     "output_size_mm": [rounded(joined.width_mm),
                                        rounded(joined.extent[3] - joined.extent[1])],
                     "transform_scale": joined.transform_public["scale"],
                     "transform_rotation_deg": rounded(joined.transform.angle_deg, 6),
                     "transform_rms_mm": joined.fit["rms_mm"],
                     "vertical_datum": joined.datum["method"],
                     "island_offsets_mm": {str(row["right_island"]): row["offset_mm"] for row in offsets},
                     "uncalibrated_right_islands": joined.datum["uncalibrated_right_islands"] or None,
                     "agreement_limit_mm": joined.options["agreement_mm"],
                     "overlap": joined.overlap, "class_counts": {row["class"]: row["cells"] for row in counts},
                     "conflict_regions": len(conflicts)}, counts)
    result.image("joined-height", scale_bar(_heatmap(joined), joined.width_mm), high_resolution=True)
    result.image("source-segments", scale_bar(_segment_image(joined), joined.width_mm))
    if joined.overlap["cells"]:
        result.image("overlap-contours", scale_bar(_overlap_image(joined, contours), joined.width_mm))
        allowed = np.isfinite(joined.difference)
        extent = Extent.fit(joined.difference, allowed, zero="symmetric")
        result.image("overlap-difference", scale_bar(heatmap(joined.difference, allowed, extent, DIVERGE, True),
                                                     joined.width_mm))
    result.add("height-file", "joined-height-mm.tiff", "image/tiff",
               tiff_float(joined.height, np.isfinite(joined.height)))
    result.add("segment-file", "segment-source.tiff", "image/tiff",
               tiff_float(joined.code.astype(np.float64), joined.code > 0))
    result.add("difference-file", "overlap-difference-mm.tiff", "image/tiff",
               tiff_float(joined.difference, np.isfinite(joined.difference)))
    result.add("grid-file", "join-grid.npz", "application/octet-stream", grid_bytes(joined))
    manifest = _manifest(joined)
    manifest["identity"] = fingerprint(manifest)
    result.document("join-index", manifest)
    result.table("segment-coverage", counts)
    result.table("control-points", joined.control_points)
    if conflicts:
        result.table("conflict-regions", conflicts)
    result.table("source-islands", _island_rows(joined))
    result.warnings.append("Cells marked both disagree are blank in the joined height. The overlap contour and "
                           "difference views keep the two readings visibly separate rather than forcing one line.")
    result.warnings.append("Each right-hand island has its own estimated vertical offset so its own zero is not "
                           "silently reused. Right-only readings are reported both as measured and as adjusted.")
    if len(observed_left_islands) > 1:
        result.warnings.append(f"The left record contains islands {observed_left_islands} with independent zeros. "
                               "Joined heights compare reliably within a left island, not across the gap between them.")
    result.warnings.append("Heights are relative to the left record. They are not an absolute distance to a mount.")
    if abs(joined.transform.angle_deg) > 12.:
        result.warnings.append("The estimated rotation is large. Recheck the paired marks before treating the join as final.")
    if joined.fit["rms_mm"] and joined.fit["rms_mm"] > max(.5, 2 * joined.options["agreement_mm"]):
        result.warnings.append("The marked positions do not agree closely after alignment. Inspect the residuals "
                               "in the control-point table.")
    if not joined.overlap["cells"]:
        result.warnings.append("The two records share no observed overlap. There is nothing to compare in the join.")
    elif joined.overlap["conflict_cells"]:
        result.warnings.append(f'{joined.overlap["conflict_cells"]} overlap samples differ by more than '
                               f'{joined.options["agreement_mm"]:g} mm. They are kept as separate evidence.')
    if joined.datum["uncalibrated_right_islands"]:
        result.warnings.append("These right-record islands had no usable datum evidence and are blank in the join: "
                               f'{joined.datum["uncalibrated_right_islands"]}. Mark positions on them or use the overlap.')
    if joined.left.island_source != "stored island map" or joined.right.island_source != "stored island map":
        result.warnings.append("An island map was missing, so each record was treated as one connected island.")
    if abs(joined.transform.scale - 1.) > .03:
        result.warnings.append("The fitted size change differs from one by more than three percent. Confirm that the "
                               "two records were measured on the same physical scale.")
    return result


def manifest_for(joined):
    manifest = _manifest(joined)
    manifest["identity"] = fingerprint(manifest)
    return manifest


def readout(manifest, arrays, raw_points, space):
    grid = manifest["grid"]
    x0, y0 = grid["origin_mm"]
    spacing = grid["spacing_mm"]
    width, height = grid["size_px"]
    extent = grid["extent_mm"]
    left, right = manifest["left"], manifest["right"]
    rotation = np.asarray(manifest["transform"]["rotation"], dtype=float)
    scale = float(manifest["transform"]["scale"])
    translation = np.asarray(manifest["transform"]["translation_mm"], dtype=float)
    classes = {int(code): label for code, label in manifest["classes"].items()}
    shifts = {int(row["right_island"]): row for row in manifest["datum"]["island_offsets"]}
    place = choice(space, "Readout space", ("mm", "normalized"))
    rows = []
    for index, point in enumerate(raw_points, 1):
        a, b = (point.x, point.y) if isinstance(point, Point) else (float(point[0]), float(point[1]))
        if place == "normalized":
            x = a * (extent[2] - extent[0]) + extent[0]
            y = b * (extent[3] - extent[1]) + extent[1]
        else:
            x, y = a, b
        column = int(math.floor((x - x0) / spacing))
        row = int(math.floor((y - y0) / spacing))
        inside = 0 <= column < width and 0 <= row < height
        code = int(arrays["code"][row, column]) if inside else NONE
        center = [x0 + (column + .5) * spacing, y0 + (row + .5) * spacing] if inside else None
        left_value = float(arrays["left_height"][row, column]) if inside else math.nan
        native = float(arrays["right_native"][row, column]) if inside else math.nan
        adjusted = float(arrays["right_adjusted"][row, column]) if inside else math.nan
        left_island = int(arrays["left_island"][row, column]) if inside else 0
        right_island = int(arrays["right_island"][row, column]) if inside else 0
        difference = float(arrays["difference"][row, column]) if inside else math.nan
        joined = float(arrays["height"][row, column]) if inside else math.nan
        right_local = (rotation.T @ (np.array([x, y]) - translation) / scale)
        offset = shifts.get(right_island)
        if not inside:
            explanation = "Outside the joined grid."
        elif code == NONE:
            explanation = "Neither height record observed this cell, so the joined height is left blank."
        elif code == UNSET:
            explanation = ("The right record observed this cell, but its island has no datum evidence, so no joined "
                           "height is claimed. Provide marks on that island or estimate from overlap.")
        elif code == CONFLICT:
            explanation = ("Both records observed this cell but disagree beyond the agreement limit. The joined "
                           "height is blank; compare the two source readings.")
        elif code == LEFT:
            explanation = "Only the left record observed this cell. Its measured height is used unchanged."
        elif code == RIGHT:
            explanation = ("Only the right record observed this cell. Its measured height plus the island's "
                           "vertical offset is used.")
        else:
            explanation = "Both records observed this cell within the agreement limit; the two adjusted readings are averaged."
        rows.append({
            "point": index, "query_mm": [rounded(x), rounded(y)],
            "cell_center_mm": [rounded(v) for v in center] if center else None,
            "source": "outside" if not inside else classes.get(code, "no evidence"),
            "joined_height_mm": rounded(joined) if math.isfinite(joined) else None,
            "left": {"record": left["title"], "record_id": left["id"],
                     "asset_sha256": left["height_sha256"],
                     "island": left_island or None, "island_calibrated": None,
                     "local_mm": [rounded(x), rounded(y)],
                     "normalized": [rounded(x / left["size_mm"][0]), rounded(y / left["size_mm"][1])],
                     "height_mm": rounded(left_value) if math.isfinite(left_value) else None},
            "right": {"record": right["title"], "record_id": right["id"],
                      "asset_sha256": right["height_sha256"],
                      "island": right_island or None,
                      "island_calibrated": right_island in shifts if right_island else None,
                      "local_mm": [rounded(v) for v in right_local],
                      "normalized": [rounded(right_local[0] / right["size_mm"][0]),
                                     rounded(right_local[1] / right["size_mm"][1])],
                      "native_height_mm": rounded(native) if math.isfinite(native) else None,
                      "island_offset_mm": offset["offset_mm"] if offset else None,
                      "island_datum_samples": offset["samples"] if offset else None,
                      "adjusted_height_mm": rounded(adjusted) if math.isfinite(adjusted) else None},
            "difference_mm": rounded(difference) if math.isfinite(difference) else None,
            "explanation": explanation,
        })
    return rows
