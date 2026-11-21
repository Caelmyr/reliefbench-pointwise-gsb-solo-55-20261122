"""Consistent visible displays for measured rasters, unsupported areas and units."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage

from .raster import composite, quantize, thumb


INK = (227, 234, 230)
PAPER = (21, 33, 38)
GRID = (64, 84, 88)
MINT = (106, 230, 196)
GOLD = (249, 197, 108)
HOT = (239, 119, 101)
GAP = (43, 55, 62)
COOL = (128, 170, 247)
FONT = ImageFont.load_default()
THERMAL = np.array([(28, 36, 61), (45, 88, 137), (50, 161, 152), (160, 209, 126), (251, 225, 127)])
DIVERGE = np.array([(45, 81, 157), (94, 158, 183), (223, 229, 215), (244, 166, 100), (158, 53, 61)])


@dataclass(frozen=True)
class Extent:
    low: float
    high: float

    @classmethod
    def fit(cls, values, observed, opts=(3, 97), floor=1e-4, zero=None):
        picks = values[observed & np.isfinite(values)]
        if picks.size < 2:
            return cls(0., float(floor))
        least, most = np.percentile(picks, opts)
        if zero == "symmetric":
            maximum = max(abs(least), abs(most), floor)
            return cls(-float(maximum), float(maximum))
        if zero == "minimum":
            least = 0.
        return cls(float(least), float(max(most, least + floor)))

    def public(self):
        return [round(self.low, 7), round(self.high, 7)]


def _lookup(fractions, palette):
    scaled = np.clip(fractions, 0., 1.) * (len(palette) - 1)
    first = np.minimum(scaled.astype(np.int32), len(palette) - 2)
    remainder = (scaled - first)[..., None]
    return np.uint8(np.clip(palette[first] * (1. - remainder) + palette[first + 1] * remainder, 0, 255))


def colors(values, extent, palette=THERMAL):
    values = np.nan_to_num((values - extent.low) / max(extent.high - extent.low, 1e-12))
    return _lookup(values, palette)


def heatmap(values, valid, extent=None, palette=THERMAL, grid=False):
    extent = extent or Extent.fit(values, valid)
    c = colors(values, extent, palette)
    c[~valid] = GAP
    if grid:
        y, x = np.indices(valid.shape)
        check = ((x + y) // 5) % 2 == 0
        c[~valid & check] = (53, 64, 71)
    return Image.fromarray(c, "RGB")


def flatten_rgb(rgb, support, linear=True):
    return composite(quantize(rgb, linear=linear), support, GAP)


def support_image(values, support, maximum):
    extent = Extent(0., max(maximum, 1.))
    return heatmap(values, support, extent, grid=True)


def normals_image(normal, valid):
    return composite(quantize((normal + 1.) * .5, linear=False), valid, GAP)


def _nice_distance(amount):
    magnitude = 10 ** math.floor(math.log10(max(amount, .0001)))
    return max(p for p in (.5, 1, 2, 5) if p * magnitude <= amount) * magnitude


def scale_bar(base, width_mm, unit="mm"):
    image = base.convert("RGB").copy()
    width, height = image.size
    if width < 85 or height < 68:
        return image
    step = _nice_distance(width_mm * min(.26, 105 / width))
    length = min(round(width * .34), round(step / width_mm * width))
    x1, y1 = 10, height - 14
    draw = ImageDraw.Draw(image)
    label = f"{step:g} {unit}"
    draw.rounded_rectangle((x1 - 5, y1 - 21, x1 + max(length + 5, len(label) * 6 + 5), y1 + 6),
                           radius=4, fill=PAPER)
    draw.line((x1, y1, x1 + length, y1), fill=INK, width=2)
    for x in (x1, x1 + length):
        draw.line((x, y1 - 3, x, y1 + 2), fill=INK, width=1)
    draw.text((x1, y1 - 16), label, fill=INK, font=FONT)
    return image


def overlay(base, items):
    result = base.convert("RGB").copy()
    d = ImageDraw.Draw(result)
    width, height = result.size
    for shape, points, line, label in items:
        vertices = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in points]
        if shape == "point":
            x, y = vertices[0]
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=line, outline=PAPER, width=2)
        elif shape == "closed":
            d.line([*vertices, vertices[0]], fill=line, width=2)
        else:
            d.line(vertices, fill=line, width=2)
        if label:
            x = min(width - 60, max(4, vertices[0][0] + 6))
            y = min(height - 18, max(4, vertices[0][1] + 6))
            wide = d.textlength(label, font=FONT)
            d.rectangle((x - 3, y - 2, x + wide + 3, y + 12), fill=PAPER)
            d.text((x, y), label, fill=line, font=FONT)
    return result


def split_visual(first, second, label_a, label_b, max_side=1120):
    left = thumb(first, max_side // 2 - 10)
    right = thumb(second, max_side // 2 - 10)
    width, height = left.width + right.width + 10, max(left.height, right.height) + 26
    plate = Image.new("RGB", (width, height), PAPER)
    plate.paste(left, (0, 26))
    plate.paste(right, (left.width + 10, 26))
    d = ImageDraw.Draw(plate)
    d.text((6, 6), label_a, font=FONT, fill=INK)
    d.text((left.width + 16, 6), label_b, font=FONT, fill=INK)
    return plate


def contact_sheet(entries, columns=3, tile=(230, 177)):
    lines = math.ceil(len(entries) / columns)
    gutter = 8
    w, h = tile
    result = Image.new("RGB", (columns * (w + gutter) + gutter, lines * (h + gutter) + gutter), PAPER)
    draw = ImageDraw.Draw(result)
    for i, (label, photo) in enumerate(entries):
        x = gutter + (i % columns) * (w + gutter)
        y = gutter + (i // columns) * (h + gutter)
        draw.rounded_rectangle((x, y, x + w, y + h), radius=6, fill=(31, 46, 51))
        draw.text((x + 8, y + 6), label[:38], font=FONT, fill=INK)
        view = photo.copy()
        view.thumbnail((w - 16, h - 34), Image.Resampling.LANCZOS)
        result.paste(view, (x + (w - view.width) // 2, y + 26))
    return result


@dataclass(frozen=True)
class Trace:
    label: str
    x: np.ndarray
    y: np.ndarray
    color: tuple[int, int, int] = MINT


def _limits(series, dims):
    extrema = []
    for which, fixed in zip(("x", "y"), dims, strict=True):
        if fixed is not None:
            extrema.append(fixed)
            continue
        pieces = [getattr(trace, which) for trace in series]
        finite = np.concatenate([p[np.isfinite(p)] for p in pieces])
        low = float(np.min(finite)) if finite.size else 0.
        high = float(np.max(finite)) if finite.size else 1.
        extra = max(1e-4, (high - low) * .045, .001 * abs(low))
        extrema.append((low - extra, high + extra))
    return extrema


def _draw_axes(d, image, x_area, y_area, x0, y0, x1, y1, x_label, y_label):
    for i in range(6):
        fraction = i / 5
        x = round(x0 + (x1 - x0) * fraction)
        y = round(y1 - (y1 - y0) * fraction)
        d.line((x, y0, x, y1), fill=GRID, width=1)
        d.line((x0, y, x1, y), fill=GRID, width=1)
        tx = f"{x_area[0] + (x_area[1]-x_area[0])*fraction:.3g}"
        ty = f"{y_area[0] + (y_area[1]-y_area[0])*fraction:.3g}"
        d.text((x - d.textlength(tx, font=FONT) / 2, y1 + 6), tx, fill=INK, font=FONT)
        d.text((x0 - 8 - d.textlength(ty, font=FONT), y - 5), ty, fill=INK, font=FONT)
    d.text((x0, y0 - 17), y_label, fill=MINT, font=FONT)
    d.text((x0 + (x1 - x0) / 2 - d.textlength(x_label, font=FONT) / 2, image.height - 19),
           x_label, fill=MINT, font=FONT)


def plot(series, x_label, y_label, x_window=None, y_window=None, title=None):
    image = Image.new("RGB", (820, 333), PAPER)
    draw = ImageDraw.Draw(image)
    xa, ya = _limits(series, (x_window, y_window))
    x0, y0, x1, y1 = 65, 51, image.width - 23, image.height - 53
    if title:
        draw.text((65, 9), title, font=FONT, fill=GOLD)
    _draw_axes(draw, image, xa, ya, x0, y0, x1, y1, x_label, y_label)
    for trace in series:
        finite = np.isfinite(trace.x) & np.isfinite(trace.y)
        idx = np.flatnonzero(np.diff(finite.astype(np.int8), prepend=0, append=0))
        px = x0 + np.minimum(1, np.maximum(0, (trace.x - xa[0]) / (xa[1] - xa[0]))) * (x1 - x0)
        py = y1 - np.minimum(1, np.maximum(0, (trace.y - ya[0]) / (ya[1] - ya[0]))) * (y1 - y0)
        for first, last in zip(idx[::2], idx[1::2], strict=True):
            coordinates = [(float(x), float(y)) for x, y in zip(px[first:last], py[first:last], strict=True)]
            if len(coordinates) == 1:
                x, y = coordinates[0]
                draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=trace.color)
            elif coordinates:
                draw.line(coordinates, fill=trace.color, width=2)
    origin = x1
    for trace in reversed(series):
        advance = draw.textlength(trace.label, font=FONT) + 27
        origin -= advance
        draw.line((origin, 14, origin + 14, 14), fill=trace.color, width=2)
        draw.text((origin + 17, 8), trace.label, fill=INK, font=FONT)
    return image


def histogram(data, limits, bins=48):
    counts, edges = np.histogram(data, int(bins), range=limits)
    return (edges[:-1] + edges[1:]) * .5, counts.astype(float)


def support_boundary(valid):
    return valid & ~ndimage.binary_erosion(valid)
