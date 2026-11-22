"""Normal, height, contour and virtual illumination readouts."""
import numpy as np
from PIL import Image
from scipy import ndimage

from .heightfield import contour_levels, derivative_consistency, integrate
from .lightmodel import (color_relit, compare_positions, directions, diversity, exposure_settings,
                         solve, spherical_coverage)
from .models import Point, Problem, fields, flag, light_vector, list_of, number, rounded, whole
from .raster import (line_samples, safe_fraction, sample_field, statistics, tiff_float)
from .results import Result
from .shadows import LEGEND, cast_shadows, classification_image, probe_shadow, tinted_image, with_legend
from .visual import (COOL, DIVERGE, GOLD, HOT, MINT, Extent, Trace, contact_sheet, flatten_rgb,
                     heatmap, normals_image, overlay, plot, scale_bar, split_visual, support_image)


def _physical_map(dataset, photo):
    return scale_bar(photo, dataset.scale.width_mm)


def coverage(dataset, options):
    fields(options, "inspect_all")
    frames = dataset.lamps()
    detail = flag(options.get("inspect_all", True), "List all light pairs")
    matrix = directions(frames)
    quality = diversity(matrix)
    rows = compare_positions(frames)
    use = rows if detail else rows[:min(8, len(rows))]
    gaps, description = spherical_coverage(frames)
    state = "usable leading directions" if quality["rank"] == 3 and quality["condition"] and quality["condition"] <= 38 else "poorly constrained"
    output = Result("lights", "Lamp directions and leverage",
                    "Review direction separation and where a mounted light cluster lacks coverage.",
                    {"direction_status": state, "photos": len(frames), **quality, **description}, use)
    picture = heatmap(gaps, np.ones_like(gaps, dtype=bool), Extent(0, 90)).resize((700, 350), Image.Resampling.BILINEAR)
    annotated = overlay(picture, [
        ("point", [[frame.spec.azimuth / 360, (frame.spec.elevation - 7) / 76]], MINT, str(i))
        for i, frame in enumerate(frames, 1) if 7 <= frame.spec.elevation <= 83
    ])
    output.image("hemisphere-coverage", annotated)
    output.table("lamp-pairs")
    output.table("selected-lamps", [{"photo": frame.title, "light": i + 1, "azimuth": frame.spec.azimuth,
                                    "elevation": frame.spec.elevation, "power": frame.spec.lamp_power,
                                    "exposure_ev": frame.spec.exposure}
                                   for i, frame in enumerate(frames)])
    output.warnings.append("Map runs from 0° to 360° azimuth and from 7° at the top to 83° elevation at the bottom.")
    return output


def _field_row(frame, field, selected):
    predicted = field.predicted(frame)
    error = (predicted - frame.linear) @ np.array([.2126, .7152, .0722])
    valid = field.valid & frame.alpha_valid & ~frame.clipped & selected
    loss = np.abs(error)
    return {"photo": frame.title, "view_used_in_input": frame.id,
            "visible_fit_samples": int(valid.sum()), "absolute_median_error": statistics(loss, valid)["median"],
            "absolute_p95_error": statistics(loss, valid)["p95"], "mean_signed_error": statistics(error, valid)["mean"]}


def _normal_span(field):
    return statistics(field.angular_incline(), field.valid)


def _fit_metrics(dataset, field):
    frames = dataset.frames
    return {
        "registered_scale": dataset.scale.public(), "photos": len(frames),
        "normal_fit_fraction": rounded(safe_fraction(field.valid, dataset.selection.sum())),
        "unknown_pixels_in_selection": int((dataset.selection & ~field.valid).sum()),
        "lamp_matrix_condition": rounded(field.global_condition),
        "orientations": _normal_span(field), "visible_albedo": statistics(field.reflectance, field.valid),
        "fit_linear_rmse": statistics(field.error, field.valid),
        "support_used": statistics(field.support, field.valid),
    }


def normal_report(dataset, options):
    fields(options, "show_residual")
    residual = flag(options.get("show_residual", True), "Fit residual view")
    field = solve(dataset)
    fit_summary = _fit_metrics(dataset, field)
    rows = [_field_row(frame, field, dataset.selection) for frame in dataset.frames]
    result = Result("normals", "Normal field and measured reflectance",
                    "Estimate which direction each visible surface element faces under the selected lamps.",
                    fit_summary, rows)
    result.image("surface-normals", _physical_map(dataset, normals_image(field.normals, field.valid)))
    result.image("surface-albedo", _physical_map(dataset, flatten_rgb(field.rgb, field.valid)))
    result.image("measurement-support", support_image(field.input_count, dataset.selection, len(dataset.frames)))
    result.add("normal-x", "normal-x.tiff", "image/tiff", tiff_float(field.normals[..., 0], field.valid))
    result.add("normal-y", "normal-y.tiff", "image/tiff", tiff_float(field.normals[..., 1], field.valid))
    result.add("normal-z", "normal-z.tiff", "image/tiff", tiff_float(field.normals[..., 2], field.valid))
    if residual:
        extent = Extent.fit(field.error, field.valid, zero="minimum")
        result.image("model-residual", heatmap(field.error, field.valid, extent, grid=True))
        result.metrics["residual_color_range"] = extent.public()
    result.warnings.append("Computed from unclipped captures above the minimum signal and a diffuse reflection model. Bright directional sheen can bias the fit.")
    return result.table("source-fit")


def gradient_check(dataset, options):
    fields(options, "slope_max", "show_aspect")
    maximum = number(options.get("slope_max", 2), "Slope axis range", .2, 8.)
    aspect = flag(options.get("show_aspect", True), "Aspect map")
    field = solve(dataset)
    dx, dy, usable = field.gradients(maximum)
    curls, raw = derivative_consistency(field, dataset.scale)
    valid = dataset.require(usable, "Slope range")
    supported_curl = raw & ndimage.binary_erosion(valid)
    extent = Extent.fit(curls, supported_curl, zero="symmetric")
    metrics = {"slope_limit": maximum, "x_slope": statistics(dx, valid), "y_slope": statistics(dy, valid),
               "rotational_mismatch_mm_inverse": statistics(curls, supported_curl),
               "common_valid_fraction": rounded(safe_fraction(valid, dataset.selection.sum())),
               "mismatch_color_range": extent.public()}
    tool = Result("gradients", "Slope and integrability",
                  "Check gradient directions and local disagreement before depth integration.", metrics)
    tool.image("slope-x", heatmap(dx, valid, Extent(-maximum, maximum), DIVERGE, True))
    tool.image("slope-y", heatmap(dy, valid, Extent(-maximum, maximum), DIVERGE, True))
    tool.image("slope-curl", heatmap(curls, supported_curl, extent, DIVERGE, True))
    if aspect:
        direction = field.angular_aspect()
        tool.image("slope-aspect", heatmap(direction, valid, Extent(0, 360)))
    tool.warnings.append("Large curl shows a disagreement in the fitted slopes. It is not a direct estimate of a crack or a physical discontinuity.")
    return tool


def height_report(dataset, options):
    fields(options, "max_slope", "contours_mm")
    max_slope = number(options.get("max_slope", 3), "Allowed gradient", .3, 8)
    interval = number(options.get("contours_mm", .22), "Contour interval", .015, 25)
    field = solve(dataset)
    height = integrate(field, dataset.scale, max_slope=max_slope)
    extent = Extent.fit(height.height, height.visible, zero="symmetric")
    image = heatmap(height.height, height.visible, extent, DIVERGE, True)
    result = Result("height", "Relative height, contours and island coverage",
                    "Integrate slopes on physically spaced pixels with a separate relative zero for each continuous island.",
                    {**height.report(), "display_mm_range": extent.public()}, list(height.islands))
    result.image("relative-height", scale_bar(image, dataset.scale.width_mm), high_resolution=True)
    try:
        lines, levels = contour_levels(height, interval)
    except Problem as issue:
        result.warnings.append(str(issue))
    else:
        output = np.array(image)
        output[lines] = (247, 236, 193)
        result.metrics["contour_levels_mm"] = levels
        result.image("height-contours", scale_bar(Image.fromarray(output), dataset.scale.width_mm))
    result.add("height-samples", "relative-height-mm.tiff", "image/tiff", tiff_float(height.height, height.visible))
    result.add("height-support", "height-island-id.tiff", "image/tiff",
               tiff_float(height.island_labels, height.visible))
    result.warnings.append("Each connected island has its own zero. This report does not measure absolute distance to a mount.")
    return result.table("height-islands")


def relighting(dataset, options):
    fields(options, "azimuth", "elevation", "ambient", "contrast", "colored", "opposite",
           "cast_shadows", "probe")
    azimuth = number(options.get("azimuth", 42), "Light azimuth", -180, 360)
    elevation = number(options.get("elevation", 32), "Light elevation", 5, 89.9)
    colored = flag(options.get("colored", True), "Use measured reflectance")
    opposite = flag(options.get("opposite", True), "Show opposite light")
    cast_mode = flag(options.get("cast_shadows", False), "Project measured cast shadows")
    contrast, ambient = exposure_settings(options)
    field = solve(dataset)
    direction = light_vector(azimuth, elevation)
    rgb = color_relit(field, direction, ambient, contrast, colored)
    angle_bins = _lighting_readout(field, direction)
    rendered = _physical_map(dataset, flatten_rgb(rgb, field.valid))
    metrics = {"shadow_model": "measured-height cast projection" if cast_mode else "local normal only",
               "virtual_light_vector": direction, "azimuth": azimuth % 360, "elevation": elevation,
               "ambient": ambient, "contrast": contrast, "use_measured_reflectance": colored,
               **angle_bins}
    result = Result("relight", "Virtual lamp inspection",
                    "Move a local light direction across measured normals while retaining the chosen reflectance view.",
                    metrics)
    result.image("selected-light", rendered)
    if opposite:
        back = color_relit(field, light_vector((azimuth + 180) % 360, elevation), ambient, contrast, colored)
        result.image("opposite-light", split_visual(rendered, _physical_map(dataset, flatten_rgb(back, field.valid)),
                                                   "Chosen lamp", "Opposite azimuth"))
    if not cast_mode:
        result.warnings.append("Light response is local to each normal. Relief casting a shadow onto another region is not simulated; switch on cast-shadow projection to include it.")
        return result
    height = integrate(field, dataset.scale)
    shadow = cast_shadows(height, direction)
    quantized = flatten_rgb(rgb, field.valid)
    base_view = tinted_image(np.asarray(quantized), shadow)
    result.image("cast-shadow-light", _physical_map(dataset, base_view))
    result.image("shadow-classification", _physical_map(dataset, classification_image(shadow)))
    result.add("shadow-classes", "shadow-classes.tiff", "image/tiff",
               tiff_float(shadow.state.astype(np.float32), shadow.visible))
    basis = {"basis": "Ray march on the integrated relative-height grid; LIT 0, self-backface 1, "
                      "measured cast shadow 2, uncertain shadow 3.",
             "shadow_bias": {"slope_degrees": 1.0, "minimum_mm": 0.003},
             "height_grid": height.scale.public(), "integration_stride": height.stride,
             "ray_step_mm": {"y": shadow.step_mm[0], "x": shadow.step_mm[1]},
             "maximum_ray_steps": shadow.ray_steps,
             "island_datum_note": "Rays crossing between independent height islands are treated as an unmeasured gap.",
             **shadow.fractions()}
    result.metrics.update(basis)
    result.document("shadow-basis", basis)
    result.warnings.append("Self backface (the element faces away from the lamp) is shown separately from measured cast shadows (another measured rise blocks the ray).")
    result.warnings.append("Uncertain shadows mean the ray crossed a cell with no measured height, or an island datum break, before reaching a blocker. They are not confirmed shadows and are marked with hatching.")
    probe = _probe_point(options)
    if probe is not None:
        row = probe_shadow(shadow, probe, height.height)
        result.metrics["probe"] = row
        result.table("shadow-probe", [row])
        result.image("probe-occluder", _probe_image(dataset, np.asarray(quantized), shadow, row))
        result.warnings.append("Probe geometry uses the coarse integrated height grid; inspect the height-grid sampling before treating the occluder point as exact.")
    else:
        result.warnings.append("Click a point in the cast-shadow image to identify the measured rise blocking its light.")
    return result


def _probe_point(options):
    raw = options.get("probe", [])
    if not isinstance(raw, list):
        raise Problem("Probe needs a coordinate pair or an empty list.")
    if not raw:
        return None
    values = list_of(raw, "Probe", 1, 1)
    return Point.parse(values[0], "Probe point")


def _probe_image(dataset, base_rgb, shadow, probe_row):
    plain = tinted_image(base_rgb, shadow, legend=False)
    items = []
    if probe_row["occluder"]:
        items.append(("line", [probe_row["point"], probe_row["occluder"]], GOLD, ""))
        items.append(("point", [probe_row["occluder"]], HOT, ""))
    if probe_row["first_gap_point"]:
        items.append(("point", [probe_row["first_gap_point"]], GOLD, ""))
    items.append(("point", [probe_row["point"]], MINT, ""))
    marked = overlay(plain, items) if items else plain
    extra = ((HOT, "Blocking rise marker"), (GOLD, "First unmeasured gap")) if probe_row["first_gap_point"] else (
        (HOT, "Blocking rise marker"),)
    return _physical_map(dataset, with_legend(marked, (*LEGEND, *extra)))


def _lighting_readout(field, direction):
    normal_dot = field.normals @ np.asarray(direction)
    seen = field.valid
    returns = {"normals_facing_light": rounded(safe_fraction((normal_dot > 0) & seen, seen.sum())),
               "normals_in_grazing_band": rounded(safe_fraction((np.abs(normal_dot) < .12) & seen, seen.sum())),
               "positive_response": statistics(np.maximum(normal_dot, 0.), seen),
               "observed_pixels": int(seen.sum())}
    return returns


def _profile_points(dataset, options):
    a = Point.parse(options.get("start", [.16, .48]), "Section start")
    b = Point.parse(options.get("end", [.84, .52]), "Section end")
    if not (dataset.region.contains(a) and dataset.region.contains(b)):
        raise Problem("Keep both section endpoints in the analysis rectangle.")
    n = whole(options.get("samples", 210), "Section samples", 24, 720)
    dist = dataset.scale.distance(a, b)
    if dist < .2:
        raise Problem("Place the section endpoints further apart.")
    points = line_samples(a, b, n)
    distance = np.linspace(0, dist, n)
    return a, b, points, distance


def _section_extent(normal_profile, heights, measured):
    if measured.sum() < 8:
        raise Problem("The chosen line misses most reconstructed surface support.")
    rows = []
    for i in range(len(heights)):
        rows.append({
            "height_mm": rounded(heights[i]) if measured[i] else None,
            "incline_deg": rounded(normal_profile[i]) if measured[i] else None,
            "usable": bool(measured[i]),
        })
    return rows


def physical_section(dataset, options):
    fields(options, "start", "end", "samples", "show_angles")
    show_angles = flag(options.get("show_angles", True), "Show line inclination")
    a, b, points, distance = _profile_points(dataset, options)
    field = solve(dataset)
    height = integrate(field, dataset.scale)
    values, vis = height.at(points)
    incline, angle_ok = sample_field(field.angular_incline(), field.valid, points)
    _, sample_islands = sample_field(height.island_labels, height.visible, points)
    ids_y = np.minimum(height.visible.shape[0] - 1, (points[:, 1] * height.visible.shape[0]).astype(int))
    ids_x = np.minimum(height.visible.shape[1] - 1, (points[:, 0] * height.visible.shape[1]).astype(int))
    island = height.island_labels[ids_y, ids_x]
    baseline = _section_extent(incline, values, vis & angle_ok & sample_islands)
    rows = [{"distance_mm": rounded(s), "surface_point": [rounded(x) for x in p],
             "connected_island": int(i) if base["usable"] else None, **base}
            for p, s, i, base in zip(points, distance, island, baseline, strict=True)]
    extent_rows = statistics(values, vis)
    validated = [r for r in rows if r["usable"]]
    result = Result("section", "Physical cross-section",
                    "Read a line of reconstructed height with unsupported spans shown as gaps.",
                    {"section_length_mm": rounded(distance[-1]), "tested_samples": len(points),
                     "visible_samples": len(validated), "connected_islands_along_line": sorted({r["connected_island"] for r in validated}),
                     "relative_height_range_mm": [extent_rows["min"], extent_rows["max"]]}, rows)
    shown = heatmap(height.height, height.visible, Extent.fit(height.height, height.visible, zero="symmetric"),
                    DIVERGE, True)
    line = [("line", [a.public(), b.public()], GOLD, "A-B section"),
            ("point", [a.public()], MINT, "A"), ("point", [b.public()], HOT, "B")]
    result.image("section-location", _physical_map(dataset, overlay(shown, line)))
    height_values = np.where(vis, values, np.nan)
    result.image("section-height", plot([Trace("Relative height", distance, height_values)], "Distance from A (mm)", "Height (mm)"))
    if show_angles:
        result.image("section-slope", plot([Trace("Inclination", distance, np.where(angle_ok & vis, incline, np.nan), COOL)],
                                         "Distance from A (mm)", "Angle (degrees)", y_window=(0, max(10, float(incline[angle_ok].max()) + 2))))
    if len({r["connected_island"] for r in validated}) > 1:
        result.warnings.append("The line crosses multiple independent height datums.")
    return result.table("physical-cross-section")
