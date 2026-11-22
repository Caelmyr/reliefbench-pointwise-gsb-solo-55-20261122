"""Infer lamp directions from specular highlights marked on a round mirror ball.

The camera looks at the mirror ball with the view direction V = (0, 0, 1).  For a
unit-sphere point p the mirror reflection of an incoming lamp direction L reaches
the camera when the reflected ray equals V, giving L = 2 (p . V) p - V.  Operators
circle the ball once per photograph and rough-box each visible bright spot; the box
becomes an angular range so clipped white centres still yield a usable spread.
"""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .models import Point, Problem, Region, fields, list_of, number, rounded, text
from .raster import safe_fraction
from .results import Result
from .visual import COOL, GOLD, HOT, MINT, PAPER, contact_sheet, flatten_rgb, overlay


MIN_BALL_RADIUS = .035
MAX_MARKS = 12
MAX_SPOTS = 6
# Directions separated by less than this inside one highlight box are treated as one spot.
AMBIGUITY_GAP = 14.


@dataclass(frozen=True)
class BallMark:
    label: str
    center: Point
    radius: float
    spots: tuple[Region, ...]

    @classmethod
    def parse(cls, obj, index):
        if not isinstance(obj, dict):
            raise Problem("Each mirror mark needs a ball centre, radius and highlight boxes.")
        fields(obj, "label", "center", "radius", "highlights")
        label = text(obj.get("label", f"Ball {index}"), f"Mirror ball label {index}", 65)
        center = Point.parse(obj.get("center"), "Mirror ball centre")
        radius = number(obj.get("radius"), "Mirror ball radius", MIN_BALL_RADIUS, .5)
        raw = list_of(obj.get("highlights", []), "Highlight boxes", 1, MAX_SPOTS)
        spots = tuple(Region.parse(box) for box in raw)
        return cls(label, center, radius, spots)


def parse_marks(options):
    raw = list_of(options.get("mirror_marks", []), "Mirror ball marks", 1, MAX_MARKS)
    return tuple(BallMark.parse(item, i) for i, item in enumerate(raw, 1))


def _ball_pixels(raw, mark):
    height, width = raw.shape
    yy, xx = np.mgrid[0:height, 0:width]
    nx = (xx + .5) / (width - 1)
    ny = (yy + .5) / (height - 1)
    # Normalize by the ball radius so that the unit circle is the photographed ball.
    dx = (nx - mark.center.x) / mark.radius
    dy = (ny - mark.center.y) / mark.radius
    distance = np.hypot(dx, dy)
    inside = distance <= 1.
    return dx, dy, distance, inside


def _spot_window(raw, mark, box):
    """Highlight box intersected with the ball, in camera pixel coordinates."""
    height, width = raw.shape
    left, top, right, bottom = box.pixels(width, height)
    cx0, cy0 = mark.center.pixel(width, height)
    rx, ry = mark.radius * (width - 1), mark.radius * (height - 1)
    px0 = max(left, int(math.floor(cx0 - rx)))
    px1 = min(right, int(math.ceil(cx0 + rx)) + 1)
    py0 = max(top, int(math.floor(cy0 - ry)))
    py1 = min(bottom, int(math.ceil(cy0 + ry)) + 1)
    return slice(max(0, py0), min(height, py1)), slice(max(0, px0), min(width, px1))


def _box_ball_overlap(raw, mark, box, geometry):
    """Fraction of the box's visible pixels that fall on the mirror ball disc."""
    wy, wx = _spot_window(raw, mark, box)
    if wx.stop <= wx.start or wy.stop <= wy.start:
        return 0.
    inside = geometry[3]
    visible = raw.alpha[wy, wx] >= .98
    return safe_fraction(inside[wy, wx] & visible, max(1, int(visible.size)))


def _estimate_center(raw, mark, box, geometry):
    """Brightness-weighted centre of the visible halo; the white block only loses precision."""
    overlap = _box_ball_overlap(raw, mark, box, geometry)
    if overlap < .5:
        raise Problem("A highlight box must sit mostly inside the marked mirror ball.")
    wy, wx = _spot_window(raw, mark, box)
    height, width = raw.shape
    sub = raw.rgb[wy, wx]
    luma = sub @ np.array([.2126, .7152, .0722], dtype=np.float32)
    support = (raw.alpha[wy, wx] >= .98) & geometry[3][wy, wx]
    if support.sum() < 12:
        raise Problem("A highlight box has too little visible mirror surface.")
    weights = np.clip(luma, 0., 1.) ** 2
    weights = np.where(support, weights, 0.)
    total = float(weights.sum())
    if total <= 1e-8:
        raise Problem("No bright highlight was found inside the box. Widen it around the halo.")
    gx = np.arange(wx.start, wx.stop, dtype=np.float32) + .5
    gy = np.arange(wy.start, wy.stop, dtype=np.float32) + .5
    cx = float((weights.sum(axis=0) @ gx) / total) / (width - 1)
    cy = float((weights.sum(axis=1) @ gy) / total) / (height - 1)
    clipped = safe_fraction((luma >= .992) & support, support.sum())
    return Point(cx, cy), float(clipped)


def _point_to_unit(mark, point):
    sx = (point.x - mark.center.x) / mark.radius
    sy = (point.y - mark.center.y) / mark.radius
    length2 = sx * sx + sy * sy
    if length2 > 1. + 1e-9:
        raise Problem("A highlight centre lies outside the marked mirror ball.")
    sz = math.sqrt(max(0., 1. - length2))
    return sx, sy, sz


def _lamp_angles(sx, sy, sz):
    # V = (0, 0, 1), L = 2 (p . V) p - V.
    lx, ly, lz = 2. * sz * sx, 2. * sz * sy, 2. * sz * sz - 1.
    azimuth = (math.degrees(math.atan2(ly, lx)) + 360.) % 360.
    elevation = math.degrees(math.asin(max(-1., min(1., lz))))
    return azimuth, elevation


def _angles_at(mark, point):
    return _lamp_angles(*_point_to_unit(mark, point))


def _ring_candidates(mark, box):
    """Sample the box corners and edge mid-points, clipped onto the ball, for the range."""
    xs = (box.left, (box.left + box.right) / 2., box.right)
    ys = (box.top, (box.top + box.bottom) / 2., box.bottom)
    candidates = []
    for x in xs:
        for y in ys:
            sx = (x - mark.center.x) / mark.radius
            sy = (y - mark.center.y) / mark.radius
            norm = math.hypot(sx, sy)
            if norm > 1.:
                sx, sy = sx / norm, sy / norm
            sz = math.sqrt(max(0., 1. - sx * sx - sy * sy))
            candidates.append(_lamp_angles(sx, sy, sz))
    return candidates


def _azimuth_span(values):
    """Smallest circular arc covering the azimuths; returns (start, end, span)."""
    values = sorted(values)
    gaps = [(values[(i + 1) % len(values)] - values[i]) % 360. for i in range(len(values))]
    cut = int(np.argmax(gaps))
    start = values[(cut + 1) % len(values)]
    end = values[cut]
    span = (end - start) % 360.
    return start, end, span


def _wrap_difference(a, b):
    return (a - b + 180.) % 360. - 180.


def _spot_report(raw, mark, box, index, geometry):
    center, clipped = _estimate_center(raw, mark, box, geometry)
    azimuth, elevation = _angles_at(mark, center)
    candidates = _ring_candidates(mark, box)
    az_start, az_end, az_span = _azimuth_span([a for a, _ in candidates])
    elevations = [e for _, e in candidates]
    el_low, el_high = min(elevations), max(elevations)
    status = "bounded" if clipped > .02 else "inferred"
    return {
        "spot": index, "box": box.public(), "center": [rounded(center.x), rounded(center.y)],
        "clipped_fraction": rounded(clipped), "status": status,
        "azimuth": rounded(azimuth, 2), "elevation": rounded(elevation, 2),
        "azimuth_interval": [rounded(az_start, 2), rounded(az_end, 2)],
        "azimuth_span_degrees": rounded(az_span, 2),
        "elevation_interval": [rounded(el_low, 2), rounded(el_high, 2)],
        "elevation_span_degrees": rounded(el_high - el_low, 2),
    }


def _is_ambiguous(spots):
    if len(spots) < 2:
        return False
    for i, a in enumerate(spots):
        for b in spots[i + 1:]:
            gap = abs(_wrap_difference(a["azimuth"], b["azimuth"]))
            if gap < AMBIGUITY_GAP:
                return True
    return False


def _polar_point(azimuth, elevation, center, radius):
    theta = math.radians(azimuth)
    radial = radius * (90. - elevation) / 90.
    return center + radial * math.cos(theta), center + radial * math.sin(theta)


def _uncertainty_wedge(draw, spot, color, center, radius):
    """Fill the azimuth/elevation range each rough highlight box still allows."""
    az0, az1 = spot["azimuth_interval"]
    el0, el1 = spot["elevation_interval"]
    r0 = radius * (90. - max(0., min(89.9, el1))) / 90.
    r1 = radius * (90. - max(0., min(89.9, el0))) / 90.
    span = (az1 - az0) % 360.
    steps = max(6, int(math.ceil(span / 6.)))
    angles = [math.radians(az0 + span * k / steps) for k in range(steps + 1)]
    outer = [(center + r1 * math.cos(a), center + r1 * math.sin(a)) for a in angles]
    inner = [(center + r0 * math.cos(a), center + r0 * math.sin(a)) for a in reversed(angles)]
    draw.polygon(outer + inner, fill=color + (46,))


def _direction_plate(entries):
    width = 480
    center = width / 2
    radius = center - 34
    plate = Image.new("RGBA", (width, width), PAPER)
    draw = ImageDraw.Draw(plate)
    for elevation in (0, 22.5, 45, 67.5):
        ring = radius * (90. - elevation) / 90.
        draw.ellipse((center - ring, center - ring, center + ring, center + ring),
                     outline=(82, 102, 103), width=1)
        draw.text((center + 5, center + ring - 11), f"{elevation:g}", fill=(166, 187, 183))
    for theta in range(0, 360, 45):
        x, y = _polar_point(theta, 0, center, radius)
        draw.line((center, center, x, y), fill=(64, 80, 83), width=1)
        tx, ty = _polar_point(theta, 0, center, radius + 13)
        draw.text((tx - 9, ty - 7), str(theta), fill=(215, 230, 227))
    for entry in entries:
        color = HOT if entry["status"] == "ambiguous" else (GOLD if entry["status"] == "bounded" else MINT)
        for spot in entry["spots"]:
            _uncertainty_wedge(draw, spot, color, center, radius)
    for entry in entries:
        color = HOT if entry["status"] == "ambiguous" else (GOLD if entry["status"] == "bounded" else MINT)
        for spot in entry["spots"]:
            x, y = _polar_point(spot["azimuth"], spot["elevation"], center, radius)
            draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color, outline=PAPER)
    draw.text((10, 9), "Inferred", fill=MINT)
    draw.text((86, 9), "Bounded", fill=GOLD)
    draw.text((160, 9), "Ambiguous", fill=HOT)
    return plate.convert("RGB")


def mirror_ball_report(dataset, options):
    fields(options, "mirror_marks")
    marks = parse_marks(options)
    frames = dataset.frames
    if len(marks) != len(frames):
        raise Problem(f"Supply one mirror ball mark per selected photograph ({len(frames)} photos, {len(marks)} marks).")
    rows, entries, drawings = [], [], []
    for frame, mark in zip(frames, marks, strict=True):
        geometry = _ball_pixels(frame.raw, mark)
        inside = geometry[3]
        coverage = safe_fraction(inside & (frame.raw.alpha >= .98), inside.sum())
        if coverage < .55:
            raise Problem(f"{mark.label}: too little visible surface lies inside the marked ball.")
        spots = [_spot_report(frame.raw, mark, box, i, geometry) for i, box in enumerate(mark.spots, 1)]
        ambiguous = _is_ambiguous(spots)
        if ambiguous:
            for spot in spots:
                spot["status"] = "ambiguous"
        entry = {"photo": frame.title, "capture_id": frame.id, "ball_label": mark.label,
                 "recorded_azimuth": frame.spec.azimuth, "recorded_elevation": frame.spec.elevation,
                 "status": "ambiguous" if ambiguous else spots[0]["status"],
                 "ball_coverage": rounded(coverage), "spots": spots}
        entries.append(entry)
        for spot in spots:
            rows.append({
                "photo": frame.title, "capture_id": frame.id, "spot": spot["spot"],
                "status": spot["status"], "recorded_azimuth": frame.spec.azimuth,
                "recorded_elevation": frame.spec.elevation,
                "inferred_azimuth": spot["azimuth"], "inferred_elevation": spot["elevation"],
                "azimuth_interval": spot["azimuth_interval"], "azimuth_span_degrees": spot["azimuth_span_degrees"],
                "elevation_interval": spot["elevation_interval"], "elevation_span_degrees": spot["elevation_span_degrees"],
                "clipped_fraction": spot["clipped_fraction"], "highlight_center": spot["center"],
                "highlight_box": spot["box"],
            })
        base = flatten_rgb(frame.raw.rgb, frame.raw.alpha >= .98, False)
        guides = [("closed", mark_ellipse(mark), COOL, mark.label[:18])]
        for i, spot in enumerate(spots, 1):
            color = HOT if spot["status"] == "ambiguous" else (GOLD if spot["status"] == "bounded" else MINT)
            guides.append(("closed", Region(*spot["box"]).corners(), color, f"#{i}"))
            guides.append(("point", [spot["center"]], color, f"{spot['azimuth']:.0f}/{spot['elevation']:.0f}"))
        drawings.append((frame.title, overlay(base, guides)))
    ambiguous_count = sum(1 for entry in entries if entry["status"] == "ambiguous")
    bounded_count = sum(entry["status"] == "bounded" for entry in entries)
    result = Result(
        "mirror-ball", "Lamp direction from mirror ball highlights",
        "Turn marked mirror reflections into lamp azimuth and elevation, keeping the angular range each rough box allows.",
        {"photographs": len(frames), "ambiguous_photographs": ambiguous_count,
         "bounded_photographs": bounded_count,
         "confident_photographs": len(frames) - ambiguous_count - bounded_count,
         "angle_convention": "azimuth clockwise from image right, elevation toward the camera"}, rows)
    result.image("mirror-marks", contact_sheet(drawings, columns=min(3, len(drawings))))
    result.image("inferred-lamp-plate", _direction_plate(entries))
    result.table("mirror-lamp-directions")
    result.warnings.append("The reconstruction assumes a near-orthographic view of a round mirror ball centred at the relief. Tilted or off-axis balls bias the angles.")
    result.warnings.append("Ambiguous reflections keep the recorded lamp values. Confirm each inferred direction before writing it into the photograph settings.")
    return result


def mark_ellipse(mark):
    return [[mark.center.x - mark.radius, mark.center.y - mark.radius],
            [mark.center.x + mark.radius, mark.center.y - mark.radius],
            [mark.center.x + mark.radius, mark.center.y + mark.radius],
            [mark.center.x - mark.radius, mark.center.y + mark.radius]]
