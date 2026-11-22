"""Map explicit workbench choices to the data they consume."""
from typing import NamedTuple

from .angular_readout import DEFAULT_REGIONS, angles_report
from .camera_audit import camera_compensation, dark_report, flat_report
from .color_reference import DEFAULT_CHART, chart_report
from .dataset import DataSet
from .lamp_plan import DEFAULT_AZIMUTHS, DEFAULT_ELEVATIONS, DEFAULT_KEEPOUT, plan_lamps
from .layout_measure import DEFAULT_FIGURES, mark_geometry
from .physical_scale import ruler_readout
from .models import JobRequest, Problem
from .quality import focus_scout, held_out_light, physical_probe, stack_support
from .sampling import camera_trace, frames_review, glare_census, light_patch, overlap_review
from .shapetools import coverage, gradient_check, height_report, normal_report, physical_section, relighting
from .surface_export import contour_export, mesh_export
from .texture import texture_report, tile_material, topographic_census
from .grain import grain_report


class Tool(NamedTuple):
    title: str
    group: str
    input_kind: str
    minimum: int
    maximum: int
    description: str
    handler: object
    controls: tuple = ()

    def expose(self, key):
        return {"key": key, "title": self.title, "group": self.group, "input_kind": self.input_kind,
                "min": self.minimum, "max": self.maximum, "description": self.description, "controls": list(self.controls)}


def num(key, label, initial, lo, hi, step=None):
    return {"key": key, "label": label, "type": "number", "default": initial,
            "min": lo, "max": hi, "step": step or ("1" if type(initial) is int else "any")}


def tick(key, label, initial=False):
    return {"key": key, "label": label, "type": "checkbox", "default": initial}


def menu(key, label, initial, *items):
    return {"key": key, "label": label, "type": "select", "default": initial, "choices": items}


def points(key, label, initial):
    return {"key": key, "label": label, "type": "json", "default": initial}


def line_controls(a, b, step):
    return (points("start", "Line start (normalized x,y)", a), points("end", "Line end (normalized x,y)", b),
            num("samples", "Line samples", step, 24, 600))


def tile_controls(rows, columns):
    return (num("rows", "Rows", rows, 2, 12), num("columns", "Columns", columns, 2, 12))


TOOLS = {
    "intake": Tool("Signal & clipping", "Acquisition", "any", 1, 12,
                   "Read exposure levels and clipping from source photographs.", frames_review,
                   (num("bins", "Histogram bins", 44, 16, 100),)),
    "overlap": Tool("Alignment overlay", "Acquisition", "any", 2, 2,
                     "Check fixed landmarks in two registered photographs.", overlap_review,
                     (menu("mode", "Display", "checker", "checker", "stereo", "contours"),
                      num("edge_weight", "Edge contour level", .55, .12, .95, .01))),
    "tone-target": Tool("Neutral reference", "Acquisition", "any", 1, 1,
                        "Review known matte patches and their reliability before changing exposure.", light_patch,
                        (points("patches", "Neutral patch boxes", [[.25,.25,.34,.34],[.45,.25,.54,.34],[.65,.25,.74,.34]]),
                         points("levels", "Known linear reflectance", [.18, .40, .72]))),
    "camera-trace": Tool("Source signal line", "Acquisition", "any", 1, 4,
                         "Inspect camera counts along a physical line.", camera_trace,
                         (*line_controls([.18,.42],[.82,.55],180),
                          menu("mode", "Signal scale", "encoded", "encoded", "normalized"))),
    "glare": Tool("Bright candidates", "Acquisition", "any", 1, 1,
                  "Locate connected bright codes and identify which samples clip.", glare_census,
                  (num("code_limit", "Bright code threshold", .92, .60, 1., .005),
                   num("min_area_mm2", "Minimum area, mm²", .08, .01, 500))),
    "focus": Tool("Local detail scout", "Acquisition", "any", 1, 1,
                  "Inspect focus/detail cues in physical-scale tiles.", focus_scout,
                  (*tile_controls(5, 7), num("smoothness", "Smooth in pixels", .65, .3, 2.8),
                   tick("gain_normalized", "Normalize contrast"))),
    "support": Tool("Stack support", "Acquisition", "any", 1, 12,
                    "Review how many actual photographs support the result at each site.", stack_support,
                    (num("minimum", "Minimum photos covering one site", 3, 1, 12),
                     tick("discard_clips", "Exclude clipped sites", True))),
    "probe": Tool("Native point readout", "Acquisition", "any", 1, 12,
                  "Revisit marked positions in registered originals and their physical coordinates.", physical_probe,
                  (points("points", "Markers (normalized x,y)", [[.24,.35],[.43,.52],[.76,.61]]),)),
    "marks": Tool("Physical marks", "Acquisition", "any", 1, 1,
                   "Read planar length, angle and area with original source coordinates.", mark_geometry,
                   (points("figures", "Profiles and physical marks", DEFAULT_FIGURES),
                    num("segment_sample_density", "Samples per span", 120, 20, 400))),
    "ruler": Tool("Rule readback", "Acquisition", "any", 1, 1,
                  "Check a photographed ruler against the registered physical dimensions.", ruler_readout,
                  (*line_controls([.13,.26],[.87,.28],620)[:2],
                   num("samples", "Readings along the rule", 620, 100, 1900),
                   num("known_step_mm", "Measured ticks, mm", 5., .1, 100.),
                   num("band_mm", "Read band, mm", 2.6, .3, 25.),
                   tick("dark_ticks", "Ticks are darker", True))),
    "dark-check": Tool("Camera dark check", "Capture calibration", "any", 1, 8,
                       "Compare sensor darkness, spatial and temporal variation.", dark_report,
                       (num("tiles", "Dark tiles per side", 8, 3, 16),
                        num("hot_factor", "Hot point threshold factor", 10, 4, 35))),
    "flat-check": Tool("Flat reference falloff", "Capture calibration", "any", 1, 1,
                       "Read field falloff from a known uniform matte reference.", flat_report,
                       (num("rings", "Radial measurement rings", 12, 6, 32),
                        num("aspect", "Ellipse ratio", 1., .5, 2.),
                        points("reference", "Reference point (blank array for automatic)", []))),
    "camera-correction": Tool("Camera reference correction", "Capture calibration", "any", 3, 3,
                              "Preview camera correction from the dark and flat reference exposures.", camera_compensation,
                              (num("minimum_flat_signal", "Minimum reference signal", .05, .01, .65),
                               num("max_gain", "Maximum gain", 2.1, 1.1, 5.))),
    "color-chart": Tool("Color chart evaluation", "Capture calibration", "any", 1, 1,
                        "Read standards in known chart colors and validate on check patches.", chart_report,
                        (points("swatches", "Physical color patch references", DEFAULT_CHART),
                         menu("transform", "Response transform", "channels", "channels", "cross-channel"),
                         num("neutral_priority", "Neutral priority", .65, 0, 4))),
    "lamp-plan": Tool("Further lamp placements", "Light & surface", "raking", 3, 12,
                       "Review proposed mount positions before photographing the same setup again.", plan_lamps,
                       (points("azimuths", "Usable azimuth options", DEFAULT_AZIMUTHS),
                        points("elevations", "Available lamp elevations", DEFAULT_ELEVATIONS),
                        points("keepout", "Blocked azimuth intervals", DEFAULT_KEEPOUT),
                        num("surface_threshold", "Preferred surface response", .34, .1, .85),
                        num("count", "New mount proposals", 3, 1, 5), tick("prefer_support", "Prefer low-response coverage", True))),
    "lights": Tool("Lamp geometry", "Light & surface", "raking", 3, 12,
                   "Review whether your physical lamp directions span the normal components.", coverage,
                   (tick("inspect_all", "List all light pairs", True),)),
    "angular-readout": Tool("Response by lamp angle", "Light & surface", "raking", 3, 12,
                            "Compare marked response across observed angles.", angles_report,
                            (points("reference_regions", "Response regions", DEFAULT_REGIONS),
                             num("min_coverage", "Minimum room", .72, .38, 1.))),
    "normals": Tool("Normals & albedo", "Light & surface", "raking", 3, 12,
                    "Estimate surface orientations and reflectance from selected lighting.", normal_report,
                    (tick("show_residual", "Show fit residual", True),)),
    "holdout": Tool("Reserved lamp check", "Light & surface", "raking", 4, 12,
                     "Predict an omitted lamp photograph from the other inputs.", held_out_light,
                     (num("photo_index", "Reserved capture in selected order (1 based)", 1, 1, 12),
                      num("gain_range", "Residual color limit", .08, .008, .75))),
    "gradients": Tool("Slope consistency", "Light & surface", "raking", 3, 12,
                      "Look for disagreement in gradients before height integration.", gradient_check,
                      (num("slope_max", "Slope range", 2., .2, 8.),
                       tick("show_aspect", "Show slope orientation", True))),
    "height": Tool("Relative height", "Light & surface", "raking", 3, 12,
                   "Measure a physical relative height surface with independent connected datums.", height_report,
                   (num("max_slope", "Allowed gradient", 3., .3, 8.),
                    num("contours_mm", "Contour interval in mm", .22, .015, 25))),
    "relight": Tool("Virtual lamp", "Light & surface", "raking", 3, 12,
                    "Check measured surface under a chosen direction using local response.", relighting,
                    (num("azimuth", "Azimuth °", 42, -180, 360), num("elevation", "Elevation °", 32, 5, 89),
                     num("ambient", "Ambient", .16, 0, .7, .01), num("contrast", "Shading contrast", 1., .5, 2.5, .05),
                     tick("colored", "Use measured reflectance", True),
                     tick("opposite", "Show opposite azimuth", True))),
    "section": Tool("Physical section", "Metrology", "raking", 3, 12,
                    "Read a measured height section along a selected line.", physical_section,
                    (*line_controls([.16,.48],[.84,.52],210), tick("show_angles", "Show inclinations", True))),
    "texture": Tool("Texture spectrum", "Metrology", "raking", 3, 12,
                    "Inspect local RMS detail and overall direction/frequency at mm scale.", texture_report,
                    (menu("kind", "Texture input", "height", "height", "slope", "albedo"),
                     num("bands", "Frequency bands", 18, 8, 30), num("directions", "Directional groups", 12, 6, 24),
                     num("window_mm", "Local RMS radius, mm", 2.3, .25, 45.),
                     num("smoothing_mm", "Detrend radius, mm", 1.4, .2, 28.))),
    "tiles": Tool("Material cells", "Metrology", "raking", 3, 12,
                  "Compare physical tiles by albedo or estimated orientation.", tile_material,
                  (*tile_controls(6, 8), menu("kind", "Metric", "albedo", "albedo", "redness", "incline"))),
    "grain": Tool("Brush grain direction", "Metrology", "raking", 3, 12,
                  "Confirm brushed-line direction tile by tile at one chosen millimetre line width.",
                  grain_report,
                  (menu("kind", "Grain input", "slope", "height", "slope", "albedo"),
                   num("line_mm", "Brushed line width, mm", .8, .05, 12., .01),
                   *tile_controls(8, 10),
                   num("detrend_mm", "Detrend radius, mm", 1.6, .2, 40., .1),
                   num("boundary_margin_mm", "Boundary margin, mm", .5, 0., 12., .1))),
    "features": Tool("Local relief features", "Metrology", "raking", 3, 12,
                     "Locate and size relative bumps and dents on one measured area.", topographic_census,
                     (menu("sense", "Feature", "ridge", "ridge", "dent"),
                      num("min_area", "Minimum area, mm²", .25, .02, 500.),
                      num("displacement_mm", "Minimum signal, mm", .12, .015, 18.),
                      num("local_scale_mm", "Comparison radius, mm", 4.5, .6, 40.))),
    "mesh": Tool("Open surface mesh", "Export", "raking", 3, 12,
                 "Extract an open physical-scale geometry from observed heights only.", mesh_export,
                 (num("density", "Mesh divisions", 105, 25, 225),
                  num("vertical_scale", "Vertical scaling", 1., .2, 8.),
                  menu("format", "Open geometry", "obj", "obj", "ply"))),
    "contours": Tool("Physical contour file", "Export", "raking", 3, 12,
                     "Extract physical SVG paths from observed relative-height contours.", contour_export,
                     (num("spacing_mm", "Contour spacing, mm", .25, .015, 50.),
                      num("slope_limit", "Maximum interpreted slope", 3., .3, 8.))),
}


def catalog():
    return [tool.expose(key) for key, tool in TOOLS.items()]


def run(library, surface_id, kind, body):
    if kind not in TOOLS:
        raise Problem("Unknown inspection tool.", 404)
    job = JobRequest.parse(kind, body)
    tool = TOOLS[kind]
    if not tool.minimum <= len(job.captures) <= tool.maximum:
        raise Problem(f"Select {tool.minimum} to {tool.maximum} photographs for this tool.")
    dataset = DataSet.load(library, surface_id, job)
    if tool.input_kind == "raking":
        dataset.lamps()
    result = tool.handler(dataset, job.options)
    return result, dataset.describe(job)
