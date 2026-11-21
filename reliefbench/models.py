"""Operators enter physical quantities in millimetres and normalized image coordinates."""
from dataclasses import dataclass
import math
from pathlib import PurePath
from typing import Any

import numpy as np


class Problem(Exception):
    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


def mapping(value, label="Document"):
    if not isinstance(value, dict):
        raise Problem(f"{label} needs named fields.")
    return value


def fields(body, *names):
    unexpected = sorted(set(body) - set(names))
    if unexpected:
        raise Problem(f"Unrecognized value: {', '.join(unexpected)}.")


def number(value, name, lo, hi):
    if isinstance(value, bool):
        raise Problem(f"{name} needs a number.")
    try:
        amount = float(value)
    except (TypeError, ValueError, OverflowError):
        raise Problem(f"{name} needs a number.") from None
    if not math.isfinite(amount) or amount < lo or amount > hi:
        raise Problem(f"{name} must be from {lo} to {hi}.")
    return amount


def whole(value, name, lo, hi):
    amount = number(value, name, lo, hi)
    if not amount.is_integer():
        raise Problem(f"{name} needs an integer.")
    return int(amount)


def choice(value, name, allowed):
    if value not in allowed:
        raise Problem(f"{name} is unsupported.")
    return value


def text(value, name, limit=120, required=True):
    if not isinstance(value, str):
        raise Problem(f"{name} needs text.")
    clean = value.strip()
    if len(clean) > limit or (required and not clean):
        raise Problem(f"Enter {name.lower()} with at most {limit} characters.")
    if any(ord(c) < 32 and c not in "\n\t" for c in clean):
        raise Problem(f"{name} has a control character.")
    return clean


def flag(value, name):
    if type(value) is not bool:
        raise Problem(f"{name} needs an on or off value.")
    return value


def list_of(value, name, lo, hi):
    if not isinstance(value, list) or not lo <= len(value) <= hi:
        raise Problem(f"{name} needs between {lo} and {hi} values.")
    return value


def identifier(value, name="Identifier"):
    token = text(value, name, 42)
    if not token.isascii() or not all(c.isalnum() or c in "-_" for c in token):
        raise Problem(f"{name} uses unsupported characters.")
    return token


def filename(name):
    allowed = ".-_ "
    return "".join(c if c.isascii() and (c.isalnum() or c in allowed) else "_" for c in PurePath(name).name)[-100:] or "capture"


def light_vector(azimuth, elevation):
    phi = math.radians(azimuth)
    theta = math.radians(elevation)
    return [math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi), math.sin(theta)]


def rounded(value, digits=6):
    if value is None:
        return None
    f = float(value)
    return round(f, digits) if math.isfinite(f) else None


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    @classmethod
    def parse(cls, value, name="Point", low=0, high=1):
        a, b = list_of(value, name, 2, 2)
        return cls(number(a, name, low, high), number(b, name, low, high))

    def public(self):
        return [self.x, self.y]

    def pixel(self, width, height):
        return self.x * (width - 1), self.y * (height - 1)


@dataclass(frozen=True)
class Region:
    left: float
    top: float
    right: float
    bottom: float

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls(0., 0., 1., 1.)
        left, top, right, bottom = [number(x, "Region", 0, 1) for x in list_of(value, "Region", 4, 4)]
        if right - left < .012 or bottom - top < .012:
            raise Problem("Set a larger rectangular region.")
        return cls(left, top, right, bottom)

    def public(self):
        return [self.left, self.top, self.right, self.bottom]

    def pixels(self, width, height):
        left = min(width - 1, int(math.floor(self.left * width)))
        top = min(height - 1, int(math.floor(self.top * height)))
        right = min(width, max(left + 1, int(math.ceil(self.right * width))))
        bottom = min(height, max(top + 1, int(math.ceil(self.bottom * height))))
        return left, top, right, bottom

    def window(self, width, height):
        left, top, right, bottom = self.pixels(width, height)
        return np.s_[top:bottom, left:right]

    def contains(self, point):
        return self.left <= point.x <= self.right and self.top <= point.y <= self.bottom

    def corners(self):
        return [[self.left, self.top], [self.right, self.top], [self.right, self.bottom], [self.left, self.bottom]]


@dataclass(frozen=True)
class SurfaceSpec:
    title: str
    width_mm: float
    height_mm: float
    accession: str
    note: str
    synthetic: bool

    @classmethod
    def parse(cls, raw):
        body = mapping(raw, "Surface")
        fields(body, "title", "width_mm", "height_mm", "accession", "note", "synthetic")
        return cls(
            text(body.get("title"), "Surface title"),
            number(body.get("width_mm"), "Width in mm", 2, 1500),
            number(body.get("height_mm"), "Height in mm", 2, 1500),
            text(body.get("accession", ""), "Accession", 72, False),
            text(body.get("note", ""), "Note", 800, False),
            flag(body.get("synthetic", False), "Synthetic sample"),
        )

    def public(self):
        return dict(title=self.title, width_mm=self.width_mm, height_mm=self.height_mm,
                    accession=self.accession, note=self.note, synthetic=self.synthetic)


INPUT_ROLES = ("raking", "diffuse", "sensor-dark", "reference")


@dataclass(frozen=True)
class CaptureSpec:
    title: str
    role: str
    azimuth: float
    elevation: float
    lamp_power: float
    exposure: float
    black_code: float
    camera: str
    note: str

    @classmethod
    def parse(cls, raw):
        body = mapping(raw, "Capture")
        fields(body, "title", "role", "azimuth", "elevation", "lamp_power", "exposure",
               "black_code", "camera", "note")
        return cls(
            text(body.get("title"), "Capture title"),
            choice(body.get("role", "raking"), "Role", INPUT_ROLES),
            number(body.get("azimuth", 0.), "Azimuth", -180, 360) % 360,
            number(body.get("elevation", 45.), "Elevation", 4., 89.9),
            number(body.get("lamp_power", 1.), "Relative lamp strength", .05, 12.),
            number(body.get("exposure", 0.), "Exposure EV", -6., 6.),
            number(body.get("black_code", 0.), "Black code", 0., .45),
            text(body.get("camera", ""), "Camera label", 80, False),
            text(body.get("note", ""), "Capture notes", 700, False),
        )

    @classmethod
    def defaults(cls, title):
        return cls.parse({"title": title}).public()

    def direction(self):
        return light_vector(self.azimuth, self.elevation)

    def public(self):
        return dict(title=self.title, role=self.role, azimuth=self.azimuth, elevation=self.elevation,
                    lamp_power=self.lamp_power, exposure=self.exposure, black_code=self.black_code,
                    camera=self.camera, note=self.note)


@dataclass(frozen=True)
class CropSpec:
    corners: tuple[Point, ...]
    note: str

    @classmethod
    def parse(cls, raw):
        body = mapping(raw, "Registration")
        fields(body, "corners", "note")
        values = list_of(body.get("corners"), "Corners", 4, 4)
        corners = tuple(Point.parse(p, "Registration corner", -.12, 1.12) for p in values)
        return cls(corners, text(body.get("note", ""), "Registration note", 480, False))

    def public(self):
        return {"corners": [x.public() for x in self.corners], "note": self.note}


@dataclass(frozen=True)
class JobRequest:
    kind: str
    captures: tuple[str, ...]
    corrections: dict[str, str]
    region: Region
    options: dict[str, Any]

    @classmethod
    def parse(cls, kind, raw):
        body = mapping(raw, "Inspection inputs")
        fields(body, "captures", "corrections", "region", "options", "title", "retain")
        ids = tuple(identifier(x, "Capture") for x in list_of(body.get("captures"), "Captures", 1, 16))
        if len(set(ids)) != len(ids):
            raise Problem("Use each source photo once.")
        keys = mapping(body.get("corrections", {}), "Source registrations")
        registrations = {identifier(k): identifier(v, "Registration") for k, v in keys.items()}
        if set(registrations) - set(ids):
            raise Problem("A registration refers to a photograph outside this inspection.")
        return cls(kind, ids, registrations, Region.parse(body.get("region")),
                   mapping(body.get("options", {}), "Inspection settings"))

    def public(self):
        return {"kind": self.kind, "captures": list(self.captures), "corrections": self.corrections,
                "region": self.region.public(), "options": self.options}


@dataclass(frozen=True)
class Scale:
    width_mm: float
    height_mm: float
    width: int
    height: int

    @property
    def dx(self):
        return self.width_mm / self.width

    @property
    def dy(self):
        return self.height_mm / self.height

    def spacing(self):
        return self.dy, self.dx

    def distance(self, a: Point, b: Point):
        return math.hypot((a.x - b.x) * self.width_mm, (a.y - b.y) * self.height_mm)

    def public(self):
        return {"size_px": [self.width, self.height], "size_mm": [self.width_mm, self.height_mm],
                "sampling_mm": [rounded(self.dx), rounded(self.dy)]}
