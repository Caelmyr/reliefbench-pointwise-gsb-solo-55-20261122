"""Bind selected camera files, source settings and physical working grids."""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from .models import CaptureSpec, Problem, Region, rounded
from .raster import decode, intensity, normalized_grid, sample_camera, srgb_to_linear, statistics
from .registration import Registration, dimensions


MAX_WORKSIDE = 690
GRID_INK = (249, 197, 108)


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
        summary = self.registration.grid_summary(self.raw.shape)
        return {
            "id": self.id, "title": self.title, "hash": self.record["hash"], "role": self.spec.role,
            "mime": image["mime"], "encoding": image["encoding"], "size": image["size"],
            "settings": self.spec.public(), "capture_version": self.record["version"],
            "registration_id": self.registration.id,
            "registration": self.record.get("registration"), "filename": self.record["filename"],
            "source_window": self.registration.observed_frame(self.raw.shape, region),
            "observed_fraction": rounded(self.alpha_valid.mean()),
            "interior_grid": None if summary is None else {
                "control_points": summary.get("control_points"),
                "support_fraction": summary.get("support_fraction"),
                "support_extent_mm": summary.get("support_extent_mm"),
                "edge_correction_px": summary.get("edge_correction_px"),
                "max_correction_px": summary.get("max_correction_px"),
                "record": summary.get("record"),
            },
        }


def make_frame(record, raw, grid, physical):
    registration = Registration.from_data(record.get("registration"))
    if registration.has_grid:
        plate = registration.grid
        if (abs(plate.width_mm - float(physical["width_mm"])) > .01 or
                abs(plate.height_mm - float(physical["height_mm"])) > .01):
            raise Problem("This registration's interior cells use different plate millimetres than the current "
                          "surface size. Re-register the photograph or restore the surface dimensions.", 409)
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
        frames = tuple(make_frame(rec, raw, grid, surface) for rec, raw in sources)
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
                "parameters": job.public(), "sources": [frame.present(self.region) for frame in self.frames],
                "interior_grid": self.grid_readout()}

    def grid_report(self):
        """Which part of the working area was actually placed by interior plate cells."""
        summaries = []
        for frame in self.frames:
            summary = frame.registration.grid_summary(frame.raw.shape)
            if summary is None:
                continue
            summaries.append({"photo": frame.title, "registration": frame.registration.id,
                              "control_points": summary.get("control_points"),
                              "support_fraction": summary.get("support_fraction"),
                              "support_extent_mm": summary.get("support_extent_mm"),
                              "support_cells_mm": summary.get("support_cells_mm"),
                              "corner_correction_px": summary.get("corner_correction_px"),
                              "edge_correction_px": summary.get("edge_correction_px"),
                              "max_correction_px": summary.get("max_correction_px"),
                              "record": summary.get("record")})
        return {"used": bool(summaries), "sources": summaries}

    def grid_readout(self):
        """A short statement of which pixel span used the interior cells, or none."""
        report = self.grid_report()
        if not report["used"]:
            return {"used": False,
                    "statement": "Four-corner registration only; no interior plate cells were used."}
        spans = []
        for item in report["sources"]:
            extent = item.get("support_extent_mm")
            largest = item.get("max_correction_px") or 0.
            if not extent and largest < .5:
                spans.append(f'{item["photo"]}: marks matched the four corners, so nothing moved')
            elif not extent:
                spans.append(f'{item["photo"]}: marks sit too close to the border, so nothing moved')
            else:
                spans.append(f'{item["photo"]}: cells {extent[0]:g}-{extent[2]:g} mm across, '
                             f'{extent[1]:g}-{extent[3]:g} mm down')
        return {"used": True, "sources": report["sources"],
                "statement": ("Interior plate cells refined " + "; ".join(spans) +
                              f'. Corner and edge movement {report["sources"][0].get("corner_correction_px")} px.')}

    def surface_polygons(self):
        outlines = []
        for frame in self.frames:
            summary = frame.registration.grid_summary(frame.raw.shape)
            if summary is None:
                continue
            for outline in summary.get("support_polygons", []):
                if outline not in outlines:
                    outlines.append(outline)
        return outlines

    def grid_guides(self, space="surface"):
        """Drawable support outlines on the working area or on each camera view."""
        guides = []
        polygons = self.surface_polygons()
        if space == "surface":
            for outline in polygons:
                guides.append(("closed", outline, GRID_INK, "Interior cells"))
            return guides
        for frame in self.frames:
            summary = frame.registration.grid_summary(frame.raw.shape)
            if summary is None:
                continue
            for outline in summary.get("support_outlines_px", []):
                camera = (np.asarray(outline) / [frame.raw.shape[1] - 1, frame.raw.shape[0] - 1]).tolist()
                guides.append((frame.id, "closed", camera, GRID_INK, "Interior cells"))
        return guides

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
