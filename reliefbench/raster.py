"""Decode measured codes explicitly, carry alpha through sampling and rendering."""
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

import numpy as np
from PIL import Image, ImageCms, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .models import Point, Problem, Region, rounded


MAX_PIXELS = 9_000_000
FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "TIFF": "image/tiff", "WEBP": "image/webp"}
LUMA = np.array([.2126, .7152, .0722], dtype=np.float32)
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


def srgb_to_linear(code):
    return np.where(code <= .04045, code / 12.92, np.power((code + .055) / 1.055, 2.4))


def linear_to_srgb(data):
    values = np.clip(data, 0., 1.)
    return np.where(values <= .0031308, values * 12.92, 1.055 * values ** (1 / 2.4) - .055)


def quantize(value, linear=True):
    code = linear_to_srgb(value) if linear else np.clip(value, 0., 1.)
    return np.round(code * 255).astype(np.uint8)


def intensity(rgb):
    return rgb @ LUMA


def _icc_as_srgb(image, profile):
    if not profile:
        return image.convert("RGB"), "assumed sRGB"
    try:
        output = ImageCms.profileToProfile(image.convert("RGB"), BytesIO(profile),
                                          ImageCms.createProfile("sRGB"), outputMode="RGB")
    except (OSError, ValueError, ImageCms.PyCMSError):
        raise Problem("This embedded color profile could not be decoded.") from None
    return output, "embedded profile converted to sRGB"


@dataclass(frozen=True)
class RawCapture:
    rgb: np.ndarray
    alpha: np.ndarray
    encoding: str
    icc: str
    mime: str
    bits: int
    filename: str

    @property
    def shape(self):
        return self.alpha.shape

    @property
    def info(self):
        height, width = self.shape
        return {"size": [width, height], "bits": self.bits, "encoding": self.encoding,
                "profile": self.icc, "mime": self.mime, "visible_fraction": rounded(np.mean(self.alpha >= .98))}

    def preview(self, limit=960):
        rgb = self.rgb
        image = composite(quantize(rgb, linear=False), self.alpha >= .98)
        return png(thumb(image, limit))


def decode(raw, filename):
    try:
        image = Image.open(BytesIO(raw))
        image_format = image.format
        if image_format not in FORMATS:
            raise Problem("Use PNG, JPEG, TIFF or WebP photographs.")
        if image.width * image.height > MAX_PIXELS or min(image.size) < 28:
            raise Problem("Photograph must be at least 28 pixels per side and at most 9 megapixels.")
        if getattr(image, "n_frames", 1) != 1:
            raise Problem("The photograph should contain one frame.")
        image = ImageOps.exif_transpose(image)
        bits = 16 if image.mode.startswith("I;16") or image.mode == "I" else 8
        if bits == 16 and image.mode == "I":
            low, high = image.getextrema()
            if low < 0 or high > 65535:
                raise Problem("Integer monochrome needs unsigned 16-bit codes.")
        if bits == 16:
            gray = np.asarray(image, dtype=np.float32) / 65535.
            rgb = np.repeat(gray[..., None], 3, axis=2)
            alpha = np.ones(gray.shape, dtype=np.float32)
            encoding = "monochrome unsigned code, sRGB curve"
            icc = "assumed sRGB"
        else:
            rgba = image.convert("RGBA")
            converted, icc = _icc_as_srgb(image, image.info.get("icc_profile"))
            rgb = np.asarray(converted, dtype=np.float32) / 255.
            alpha = np.asarray(rgba.getchannel("A"), dtype=np.float32) / 255.
            encoding = "8-bit RGB code"
        if np.count_nonzero(alpha >= .98) < 32:
            raise Problem("The photograph has too little visible coverage.")
    except (UnidentifiedImageError, Image.DecompressionBombError, OSError) as exc:
        raise Problem(f"Cannot decode {PurePath(filename).name}.") from exc
    return RawCapture(rgb, alpha, encoding, icc, FORMATS[image_format], bits, filename)


def output_size(width, height, maximum):
    scale = min(1., maximum / max(width, height))
    return max(2, round(width * scale)), max(2, round(height * scale))


def thumb(image, maximum=940):
    size = output_size(*image.size, maximum)
    return image.resize(size, Image.Resampling.LANCZOS) if image.size != size else image


def png(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=False)
    return buffer.getvalue()


def tiff_float(values, valid):
    buffer = BytesIO()
    encoded = np.where(valid, values, np.nan).astype(np.float32)
    Image.fromarray(encoded).save(buffer, format="TIFF", compression="tiff_deflate")
    return buffer.getvalue()


def composite(rgb, valid, gap=(30, 42, 47)):
    return Image.fromarray(np.where(valid[..., None], rgb, gap).astype(np.uint8), "RGB")


def masked_rgba(rgb, valid):
    return Image.fromarray(np.dstack((rgb, np.where(valid, 255, 0).astype(np.uint8))), "RGBA")


def normalized_grid(width, height, region=None):
    if region is None:
        region = Region(0., 0., 1., 1.)
    x = region.left + (np.arange(width) + .5) * (region.right - region.left) / width
    y = region.top + (np.arange(height) + .5) * (region.bottom - region.top) / height
    u, v = np.meshgrid(x, y)
    return np.stack((u, v), axis=-1)


def sample_camera(raw, points):
    height, width = raw.shape
    inside = np.isfinite(points).all(axis=-1) & (np.min(points, axis=-1) >= 0) & (np.max(points, axis=-1) <= 1)
    points = np.clip(np.nan_to_num(points), 0., 1.)
    coords = [points[..., 1] * (height - 1), points[..., 0] * (width - 1)]
    alpha = ndimage.map_coordinates(raw.alpha, coords, order=1, mode="nearest")
    colors = np.stack([ndimage.map_coordinates(channel, coords, order=1, mode="nearest")
                       for channel in raw.rgb.transpose(2, 0, 1)], axis=-1)
    return colors, inside & (alpha >= .98)


def at_coordinates(values, region):
    window = region.window(values.shape[1], values.shape[0])
    return values[window]


def sample_field(values, valid, points):
    height, width = valid.shape
    coords = [points[:, 1] * height - .5, points[:, 0] * width - .5]
    weights = ndimage.map_coordinates(valid.astype(np.float32), coords, order=1, mode="nearest")
    result = ndimage.map_coordinates(np.where(valid, values, 0.), coords, order=1, mode="nearest")
    return np.divide(result, weights, out=np.zeros_like(result), where=weights >= .1), weights >= .98


def safe_fraction(mask, total=None):
    divisor = mask.size if total is None else int(total)
    return float(np.count_nonzero(mask)) / divisor if divisor else 0.


def statistics(values, valid=None):
    selected = values[np.isfinite(values)] if valid is None else values[valid & np.isfinite(values)]
    if not selected.size:
        return {"count": 0, "median": None, "p05": None, "p95": None, "mean": None, "mad": None,
                "min": None, "max": None}
    low, median, high = np.percentile(selected, [5, 50, 95])
    return {"count": int(selected.size), "median": rounded(median), "p05": rounded(low), "p95": rounded(high),
            "mean": rounded(selected.mean()), "mad": rounded(np.median(np.abs(selected - median))),
            "min": rounded(selected.min()), "max": rounded(selected.max())}


def line_samples(a: Point, b: Point, n: int):
    return np.linspace(a.public(), b.public(), n, dtype=np.float64)


def robust_local(data, support, sigma):
    stable = ndimage.gaussian_filter(support.astype(np.float32), sigma, mode="nearest")
    moment = ndimage.gaussian_filter(np.where(support, data, 0.), sigma, mode="nearest")
    return np.divide(moment, stable, out=np.zeros_like(moment), where=stable >= .01), stable
