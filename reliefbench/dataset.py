"""Bind selected camera files, source settings and physical working grids."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .models import CaptureSpec, Problem, Region, rounded
from .raster import decode, intensity, normalized_grid, sample_camera, srgb_to_linear, statistics
from .registration import Registration, dimensions


MAX_WORKSIDE = 690


@dataclass(frozen=True)
class StackFrame:
    id: str
    record: dict
    spec: CaptureSpec
    raw: object
    registration: Registration
    code: np.ndarray
    alpha_valid: np.ndarray
    linear: np.ndarray
    clipped: np.ndarray

    @property
    def signal(self):
        return intensity(self.linear)

    @property
    def code_y(self):
        return intensity(self.code)

    @property
    def light(self):
        return self.spec.direction()

    @property
    def title(self):
        return self.spec.title

    def present(self, region: Region):
        image = self.raw.info
        return {
            "id": self.id, "title": self.title, "hash": self.record["hash"], "role": self.spec.role,
            "mime": image["mime"], "encoding": image["encoding"], "size": image["size"],
            "settings": self.spec.public(), "capture_version": self.record["version"],
            "registration_id": self.registration.id,
            "registration": self.record.get("registration"), "filename": self.record["filename"],
            "source_window": self.registration.observed_frame(self.raw.shape, region),
            "observed_fraction": rounded(self.alpha_valid.mean()),
        }


def make_frame(record, raw, grid):
    registration = Registration.from_data(record.get("registration"))
    spec = CaptureSpec.parse(record["metadata"])
    code, valid = sample_camera(raw, registration.camera(grid))
    encoded_black = srgb_to_linear(np.float32(spec.black_code))
    rgb = np.maximum(srgb_to_linear(code) - encoded_black, 0)
    linear = rgb / (2 ** spec.exposure * spec.lamp_power)
    clipped = (code >= .992).any(axis=-1) & valid
    return StackFrame(record["id"], record, spec, raw, registration, code, valid, linear, clipped)


@dataclass(frozen=True)
class DataSet:
    surface: dict
    frames: tuple[StackFrame, ...]
    scale: object
    region: Region
    selection: np.ndarray

    @classmethod
    def load(cls, library, surface_id, job):
        surface, entries = library.acquire(surface_id, job.captures, job.corrections)
        total_pixels = sum(rec["image"]["size"][0] * rec["image"]["size"][1] for rec, _ in entries)
        if total_pixels > 35_000_000:
            raise Problem("The selected batch exceeds the 35 million source-pixel read budget.", 413)
        sources = [(rec, decode(b, rec["filename"])) for rec, b in entries]
        scale = dimensions(surface, [frame.shape for _, frame in sources], MAX_WORKSIDE)
        grid = normalized_grid(scale.width, scale.height)
        frames = tuple(make_frame(rec, raw, grid) for rec, raw in sources)
        selection = np.zeros((scale.height, scale.width), dtype=bool)
        selection[job.region.window(scale.width, scale.height)] = True
        return cls(surface, frames, scale, job.region, selection)

    def one(self):
        if len(self.frames) != 1:
            raise Problem("Select one source photograph for this inspection.")
        return self.frames[0]

    def pair(self):
        if len(self.frames) != 2:
            raise Problem("Select two registered photographs for this inspection.")
        return self.frames

    def lamps(self):
        if len(self.frames) < 3:
            raise Problem("Select at least three directional photographs.")
        if any(frame.spec.role != "raking" for frame in self.frames):
            raise Problem("A surface solve expects only directional (raking) photographs.")
        return self.frames

    def describe(self, job):
        names = ("id", "title", "accession", "width_mm", "height_mm", "synthetic", "version")
        return {"surface": {key: self.surface[key] for key in names}, "scale": self.scale.public(),
                "parameters": job.public(), "sources": [frame.present(self.region) for frame in self.frames]}

    def trim(self, array):
        return array[self.region.window(self.scale.width, self.scale.height)]

    def require(self, mask, name, floor=.08, min_samples=48):
        covered = self.selection & mask
        if covered.sum() < max(min_samples, round(floor * self.selection.sum())):
            raise Problem(f"{name} has too little usable image support.")
        return covered


def region_row(frame, selection):
    observed = frame.alpha_valid & selection
    unclipped = observed & ~frame.clipped
    return {
        "source": frame.title, "azimuth": rounded(frame.spec.azimuth, 2),
        "elevation": rounded(frame.spec.elevation, 2), "exposure_ev": frame.spec.exposure,
        "visible_fraction": rounded(observed.sum() / max(selection.sum(), 1)),
        "clipped_fraction": rounded((frame.clipped & observed).sum() / max(observed.sum(), 1)),
        "quantile_code": statistics(frame.code_y, observed),
        "unclipped_linear": statistics(frame.signal, unclipped),
        "registration_version": frame.registration.id,
    }


def flatten_numbers(row):
    values = {}
    for key, value in row.items():
        if isinstance(value, dict):
            values.update({f"{key}_{suffix}": item for suffix, item in value.items()})
        else:
            values[key] = value
    return values


def sample_mask(selection, radius=1):
    if radius <= 0:
        return selection
    ring = ndimage.generate_binary_structure(2, 2)
    return ndimage.binary_erosion(selection, structure=ring, iterations=radius)
