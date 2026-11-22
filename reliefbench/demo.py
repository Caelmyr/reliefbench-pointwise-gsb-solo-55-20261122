"""Optically plausible, explicitly generated training scenes."""
from dataclasses import dataclass
import math

import numpy as np
from PIL import Image, ImageDraw

from .color_reference import DEFAULT_CHART
from .models import CaptureSpec, Problem, SurfaceSpec, light_vector
from .raster import png, quantize


@dataclass(frozen=True)
class Demo:
    key: str
    title: str
    accession: str
    about: str
    width_mm: float
    height_mm: float
    base_color: tuple[float, float, float]
    seed: int

    def surface(self):
        return SurfaceSpec.parse({"title": self.title, "width_mm": self.width_mm,
                                  "height_mm": self.height_mm, "accession": self.accession,
                                  "note": self.about, "synthetic": True})

    def public(self):
        return {"key": self.key, **self.surface().public(), "example_type": "Generated training scenes"}


DEMOS = {
    "seal": Demo("seal", "Ochre seal fragment", "TRAINING · RB-A",
                 "Synthetic practice seal with ridges, abrasions and directional shadows.", 108., 78., (.55, .34, .23), 15),
    "slate": Demo("slate", "Slate carving study", "TRAINING · RB-B",
                  "Synthetic study plate with broad engraved lines and a narrow unobserved margin.", 144., 104., (.29, .36, .44), 75),
    "brushplate": Demo("brushplate", "Brushed plate with cross scratch", "TRAINING · RB-C",
                       "Synthetic plate with two brush fields of different grain and one deep transverse scratch.",
                       110., 80., (.46, .47, .45), 41),
}
LIGHTS = (
    (16, 38, .85, 0.), (66, 43, 1.0, -.25), (118, 52, 1.02, 0.),
    (168, 39, .94, .12), (217, 45, 1.02, -.16), (269, 42, .98, .16),
    (318, 60, 1.1, -.10),
)


@dataclass(frozen=True)
class SyntheticSurface:
    surface: Demo
    heights: np.ndarray
    normals: np.ndarray
    material: np.ndarray
    alpha: np.ndarray
    highlight: np.ndarray
    grid: tuple


def _grid(spec, target):
    width = target
    height = round(width * spec.height_mm / spec.width_mm)
    x = (np.arange(width) + .5) / width
    y = (np.arange(height) + .5) / height
    return np.meshgrid(x, y)


def _ridge(t, center, width):
    return np.exp(-.5 * ((t - center) / width) ** 2)


def _build_seal(x, y):
    y0, x0 = y - .52, x - .47
    radius = np.hypot(x0 / .94, y0 * .97)
    angle = np.arctan2(y0, x0)
    mound = .45 * np.exp(-2.8 * radius ** 2)
    halo = 1.17 * _ridge(radius, .289, .016) * (1 + .16 * np.cos(11 * angle))
    core = 1.5 * _ridge(radius, .163, .025) * (.68 + .3 * np.sin(5 * angle + .24))
    notch = -.55 * _ridge(radius, .230, .024) * (1 + .36 * np.cos(6 * angle + 1.3))
    glyph = .57 * _ridge(y - (.39 + .31 * x), .0, .014) * _ridge(x, .48, .12)
    return mound + halo + core + notch + glyph, radius


def _build_slate(x, y):
    rows = np.zeros_like(x)
    for i in range(5):
        center = .25 + i * .104 + .047 * np.sin(2 * np.pi * (x + .11 * i))
        rows -= (.55 + .06 * i) * _ridge(y, center, .011 + .001 * i) * _ridge(x, .45 + i * .018, .24)
    fold = .78 * _ridge(x + .44 * y, .60, .056) * _ridge(y, .57, .33)
    crater = -.40 * _ridge(x, .27, .039) * _ridge(y, .67, .029)
    return rows + fold + crater, np.hypot(x - .56, y - .44)


def _seeded_micro(x, y, rng):
    texture = np.zeros_like(x)
    for _ in range(9):
        ax, ay = rng.uniform(17, 108, size=2)
        phase = rng.uniform(0, math.tau)
        texture += rng.uniform(.002, .005) * np.sin(math.tau * (x * ax + y * ay) + phase)
    return texture


# Two abutting brush fields run at these angles (degrees along the brushed
# lines, clockwise from image right) with this repeating line width in mm.
BRUSH_ANGLES = (28., 112.)
BRUSH_SPLIT_X = .53
BRUSH_LINE_MM = .8
BRUSH_SCRATCH_MM = 3.2


def _brush_field(x, y, width_mm, height_mm, rng):
    """Fine brushed ridges, a deep wide transverse scratch, and slow waviness."""
    relief = np.zeros_like(x)
    along = np.where(x < BRUSH_SPLIT_X, math.radians(BRUSH_ANGLES[0]), math.radians(BRUSH_ANGLES[1]))
    # The phase advances across the ridges, i.e. along their normal, which is
    # rotated 90 degrees from the direction the brushed lines run along.
    normal = along + math.pi / 2.
    # Build the pattern in physical millimetres so its wavevector has the
    # requested angle and magnitude.  The working grid has nearly square
    # pixels, so a physical wavevector at `normal` projects onto the grid with
    # components (cos n / lambda, sin n / lambda) in cycles per millimetre.
    # The second field is deliberately 18 percent wider.
    left_field = x < BRUSH_SPLIT_X
    wavelength = np.where(left_field, BRUSH_LINE_MM, BRUSH_LINE_MM * 1.18)
    x_mm, y_mm = x * width_mm, y * height_mm
    kx = np.cos(normal) / wavelength
    ky = np.sin(normal) / wavelength
    phase = math.tau * (kx * x_mm + ky * y_mm)
    wobble = .004 * np.sin(math.tau * (2.1 * y) + rng.uniform(0, math.tau))
    relief += .022 * np.sin(phase + wobble)
    relief += .007 * np.sin(2 * phase + rng.uniform(0, math.tau))
    # One deep, wide horizontal scratch: its wavelength is far outside the
    # fine-line band, so the grain reading must not follow it.
    scratch_y = .61 + .018 * np.sin(math.tau * 1.4 * x)
    relief -= .16 * np.exp(-.5 * ((y - scratch_y) / (BRUSH_SCRATCH_MM / height_mm / 2.4)) ** 2)
    return relief


def form(spec, target=510):
    x, y = _grid(spec, target)
    if spec.key == "brushplate":
        rng = np.random.default_rng(spec.seed)
        heights = _brush_field(x, y, spec.width_mm, spec.height_mm, rng)
        region = np.hypot(x - .5, y - .5)
    else:
        heights, region = _build_seal(x, y) if spec.key == "seal" else _build_slate(x, y)
        rng = np.random.default_rng(spec.seed)
        heights = heights + _seeded_micro(x, y, rng)
    step_x = spec.width_mm / x.shape[1]
    step_y = spec.height_mm / x.shape[0]
    dy, dx = np.gradient(heights, step_y, step_x)
    gradient = np.stack((-dx, -dy, np.ones_like(heights)), axis=-1)
    normals = gradient / np.linalg.norm(gradient, axis=2, keepdims=True)
    tint = .98 + .09 * np.sin(10 * x + 8 * y) + .07 * np.cos(19 * y - 13 * x)
    stain = .22 * _ridge(x, .69, .11) * _ridge(y, .29, .15)
    material = np.clip(np.asarray(spec.base_color) * (tint - stain)[..., None], 0, 1)
    alpha = np.ones_like(x, dtype=np.uint8) * 255
    if spec.key == "slate":
        alpha[(x > .94) & (y < .31)] = 0
        alpha[(x - .19) ** 2 + (y - .76) ** 2 < .016 ** 2] = 0
    if spec.key == "brushplate":
        # A narrow unphotographed margin exercises the uncertain-tile rules.
        alpha[(x < .035) & (y > .55)] = 0
        alpha[(x > .965) | (y > .97)] = 0
    reflect = .06 + .065 * _ridge(x, .67, .075) * _ridge(y, .36, .08)
    reflect *= region < .5
    return SyntheticSurface(spec, heights, normals, material, alpha, reflect, (x, y))


def _illuminate(geometry, azimuth, elevation, power, exposure, index):
    light = np.asarray(light_vector(azimuth, elevation))
    diffuse = np.maximum(geometry.normals @ light, 0.)
    halfway = light + np.array([0, 0, 1], dtype=float)
    halfway /= np.linalg.norm(halfway)
    brightness = np.maximum(geometry.normals @ halfway, 0.) ** 29 * geometry.highlight
    signal = geometry.material * diffuse[..., None] + brightness[..., None] * np.asarray([1., .89, .74])
    color_scale = np.asarray([1.0, 1.0 + .014 * math.sin(index), 1.0 - .01 * math.sin(index)])
    rng = np.random.default_rng(geometry.surface.seed * 100 + index)
    signal = signal * power * (2 ** exposure) * color_scale + rng.normal(0., .0019, size=signal.shape)
    code = quantize(signal)
    return np.dstack((code, geometry.alpha))


def _neutral_card(geometry):
    x, y = geometry.grid
    background = np.full((*x.shape, 3), .15, dtype=np.float32)
    for i, target in enumerate((.18, .40, .72)):
        left, right = (.25 + .2 * i, .34 + .2 * i)
        patch = (x >= left - .008) & (x <= right + .008) & (y >= .242) & (y <= .348)
        background[patch] = target * .91
    return Image.fromarray(quantize(background))


def _dark_sample(geometry, repeat):
    x, y = geometry.grid
    rng = np.random.default_rng(geometry.surface.seed * 83 + repeat)
    base = .009 + .0026 * np.sin(18 * x) + .0033 * y + rng.normal(0, .0016, size=x.shape)
    for row, col in [(47, 67), (80, 146), (197, 22)]:
        base[row, col] += .13
    picture = np.repeat(quantize(base, linear=False)[..., None], 3, axis=2)
    return png(picture)


def _flat_field(geometry):
    x, y = geometry.grid
    brighten = 1 - .64 * ((x - .46)**2 + (y - .5)**2)
    derivative = np.array([.71, .74, .69])
    rgb = brighten[..., None] * derivative
    return png(quantize(rgb))


def _color_sample(geometry):
    x, y = geometry.grid
    img = np.full((*x.shape, 3), .34, dtype=float)
    for item in DEFAULT_CHART:
        a, b, c, d = item["region"]
        patch = (x >= a - .01) & (x <= c + .01) & (y >= b - .01) & (y <= d + .01)
        palette = np.asarray(item["target_srgb"]) * np.array([1.04, .89, .83])
        img[patch] = palette
    return png(quantize(img, linear=False))


def _rule_sample(geometry):
    x, y = geometry.grid
    img = np.full((*x.shape, 3), .27, dtype=float)
    angular = np.mod(x * geometry.surface.width_mm + .011, 5)
    fine = ((angular < .30) | (angular > 4.70)) & (y >= .238) & (y <= .305)
    region = (y >= .23) & (y <= .34)
    img[region] = np.array([.77, .74, .65])
    img[fine] = np.array([.13, .12, .10])
    return png(quantize(img, linear=False))


def _instruments(prefix, geometry):
    definitions = [
        ("dark-a", "Dark reference A", "sensor-dark", _dark_sample(geometry, 1)),
        ("dark-b", "Dark reference B", "sensor-dark", _dark_sample(geometry, 2)),
        ("flat", "Uniform flat field", "reference", _flat_field(geometry)),
        ("color", "Six color standards", "reference", _color_sample(geometry)),
        ("rule", "Scale rule · 5 mm", "reference", _rule_sample(geometry)),
    ]
    return [(f"{prefix}-{key}.png", data, _metadata(f"{title} · generated", role=role,
                                                  note="Generated calibration exercise.")) for key,title,role,data in definitions]


def _metadata(title, **items):
    return CaptureSpec.parse({"title": title, "camera": "Generated, not a physical camera", **items})


def sample(key, store):
    if key not in DEMOS:
        raise Problem("Unknown demonstration scene.", 404)
    spec = DEMOS[key]
    geometry = form(spec)
    photos = []
    for i, (azimuth, elevation, power, exposure) in enumerate(LIGHTS, 1):
        captured = _illuminate(geometry, azimuth, elevation, power, exposure, i)
        photo_title = f"L{i} · {azimuth}° / {elevation}°"
        name = f"{spec.key}-lamp-{i}.png"
        photos.append((name, png(captured), _metadata(photo_title, azimuth=azimuth, elevation=elevation,
                                                      lamp_power=power, exposure=exposure, note="Synthetic demonstration.")))
    photos.append((f"{spec.key}-neutral-strip.png", png(_neutral_card(geometry)),
                   _metadata("Three neutral swatches · generated", role="reference",
                             note="Patch values .18, .40 and .72. Uniform .91 scene gain used for practice.")))
    photos.extend(_instruments(key, geometry))
    return store.ensure_sample(key, spec.surface(), photos)


def cover(key):
    if key not in DEMOS:
        raise Problem("Unknown demonstration scene.", 404)
    form_data = form(DEMOS[key], 280)
    rgba = _illuminate(form_data, 45, 48, 1, .45, 10)
    image = Image.fromarray(rgba).convert("RGB")
    panel = ImageDraw.Draw(image)
    panel.rectangle((7, 7, 130, 26), fill=(43, 39, 32))
    panel.text((13, 10), "SYNTHETIC STUDY", fill=(246, 228, 206))
    return png(image)
