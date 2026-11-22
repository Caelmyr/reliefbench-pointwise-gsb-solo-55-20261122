"""Print an inspection handout with frozen sources, scale and review state."""
from dataclasses import dataclass, field
from io import BytesIO
import math
import textwrap

from PIL import Image, ImageOps

from .models import Problem, choice, number, rounded
from .raster import output_size


POINTS_MM = 72. / 25.4
PAPERS = {"a4": (210, 297), "letter": (215.9, 279.4)}
INK = (.13, .19, .20)
MUTED = (.29, .37, .39)
BORDER = (.78, .82, .82)
GOLD = (.64, .38, .15)
BAND = (.95, .96, .94)
MAX_PAGES = 18


def pt(mm):
    return mm * POINTS_MM


def _latin(value):
    return str(value).encode("cp1252", errors="replace").decode("cp1252")


def _literal(text):
    return _latin(text).replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)").replace("\r", " ").replace("\n", " ")


def _orientation(name, horizontal):
    w, h = PAPERS[name]
    return (h, w) if horizontal else (w, h)


def _number(amount):
    return f"{float(amount):.4f}"


def _color(rgb):
    return " ".join(_number(v) for v in rgb)


def _jpeg(raw, maximum=1240):
    try:
        image = Image.open(BytesIO(raw))
        image.load()
    except OSError as exc:
        raise Problem("A retained report image could not be printed.", 409) from exc
    image = image.convert("RGB")
    size = output_size(*image.size, maximum)
    if size != image.size:
        image = image.resize(size, Image.Resampling.LANCZOS)
    result = BytesIO()
    image.save(result, format="JPEG", quality=90, subsampling=0)
    return result.getvalue(), image.width, image.height


@dataclass
class Page:
    width: float
    height: float
    rows: list = field(default_factory=list)
    pictures: list = field(default_factory=list)

    def rectangle(self, x, y, width, height, fill=None, stroke=None):
        box = " ".join(_number(pt(v)) for v in (x, self.height - y - height, width, height))
        if stroke is not None:
            self.rows.append(f"{_color(stroke)} RG .52 w")
        if fill is not None:
            self.rows.append(f"{_color(fill)} rg")
        action = "B" if stroke is not None and fill is not None else ("f" if fill is not None else "S")
        self.rows.append(f"{box} re {action}")

    def line(self, x1, y1, x2, y2, stroke=BORDER, width=.5):
        first = (_number(pt(x1)), _number(pt(self.height - y1)))
        second = (_number(pt(x2)), _number(pt(self.height - y2)))
        self.rows.append(f"{_color(stroke)} RG {width:.2f} w {first[0]} {first[1]} m {second[0]} {second[1]} l S")

    def label(self, x, y, value, size=10, bold=False, ink=INK):
        point = (_number(pt(x)), _number(pt(self.height - y)))
        font = "FB" if bold else "FR"
        self.rows.append(f"BT /{font} {size:g} Tf {_color(ink)} rg 1 0 0 1 {point[0]} {point[1]} Tm ({_literal(value)}) Tj ET")

    def paragraph(self, x, y, value, width, size=9, bold=False, color=MUTED, max_lines=3):
        columns = max(8, round(pt(width) / (size * .53)))
        rows = textwrap.wrap(" ".join(_latin(value).split()), width=columns)
        if len(rows) > max_lines:
            rows = rows[:max_lines]
            rows[-1] = rows[-1][:-2] + "..."
        for idx, line in enumerate(rows):
            self.label(x, y + idx * size / POINTS_MM * 1.35, line, size, bold, color)
        return y + len(rows) * size / POINTS_MM * 1.35

    def image(self, data, x, y, width, height):
        jpeg, iw, ih = _jpeg(data)
        ratio = min(width / iw, height / ih)
        span_w, span_h = iw * ratio, ih * ratio
        left = x + (width - span_w) / 2
        top = y + (height - span_h) / 2
        name = f"Im{len(self.pictures) + 1}"
        self.pictures.append((name, jpeg, iw, ih))
        self.rows.append(f"q {_number(pt(span_w))} 0 0 {_number(pt(span_h))} "
                         f"{_number(pt(left))} {_number(pt(self.height - top - span_h))} cm /{name} Do Q")
        return float(span_w), float(span_h)


class PDF:
    def __init__(self):
        self.objects = [None]

    def allocate(self, content=None):
        self.objects.append(content)
        return len(self.objects) - 1

    def stream(self, attributes, raw):
        header = f"<<{attributes} /Length {len(raw)}>>\nstream\n".encode("ascii")
        return self.allocate(header + raw + b"\nendstream")

    def assemble(self, pages):
        catalog = self.allocate()
        tree = self.allocate()
        regular = self.allocate(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>")
        bold = self.allocate(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>")
        kids = []
        for page in pages:
            resources = []
            for name, jpeg, width, height in page.pictures:
                attributes = f"/Type /XObject /Subtype /Image /Width {width} /Height {height} " \
                             "/BitsPerComponent 8 /ColorSpace /DeviceRGB /Filter /DCTDecode"
                ref = self.stream(attributes, jpeg)
                resources.append(f"/{name} {ref} 0 R")
            page_stream = self.stream("", ("\n".join(page.rows)).encode("cp1252"))
            drawables = " ".join(resources)
            body = f"<< /Type /Page /Parent {tree} 0 R /MediaBox [0 0 {pt(page.width):.4f} {pt(page.height):.4f}] " \
                   f"/Contents {page_stream} 0 R /Resources << /Font << /FR {regular} 0 R /FB {bold} 0 R >> " \
                   f"/XObject << {drawables} >> >> >>"
            kids.append(self.allocate(body.encode("ascii")))
        references = " ".join(f"{ref} 0 R" for ref in kids)
        self.objects[tree] = f"<< /Type /Pages /Count {len(kids)} /Kids [ {references} ] >>".encode()
        self.objects[catalog] = f"<< /Type /Catalog /Pages {tree} 0 R >>".encode()
        return self.output(catalog)

    def output(self, catalog):
        out = BytesIO()
        out.write(b"%PDF-1.5\n%\xac\xee\xec\xfb\n")
        xref = [0]
        for pos, block in enumerate(self.objects[1:], 1):
            xref.append(out.tell())
            out.write(f"{pos} 0 obj\n".encode())
            out.write(block)
            out.write(b"\nendobj\n")
        start = out.tell()
        out.write(f"xref\n0 {len(xref)}\n0000000000 65535 f \n".encode())
        for at in xref[1:]:
            out.write(f"{at:010d} 00000 n \n".encode())
        out.write(f"trailer\n<< /Size {len(xref)} /Root {catalog} 0 R >>\nstartxref\n{start}\n%%EOF\n".encode())
        return out.getvalue()


def _page(paper, orientation, record, page_number):
    width, height = _orientation(paper, orientation == "landscape")
    page = Page(width, height)
    page.label(14, 15, "RELIEFBENCH / FIELD EVIDENCE", 10, True, GOLD)
    page.label(width - 46, 15, f'P.{page_number:02d} · rev {record["revision"]}', 8, False, MUTED)
    page.line(14, 19, width - 14, 19)
    page.line(14, height - 18, width - 14, height - 18)
    page.label(14, height - 12, f'Record {record["id"][:16]}    {record["state"].upper()}    Local inspection copy', 8, False, MUTED)
    return page


def _summary_value(value):
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, list):
        return ", ".join(_summary_value(x) for x in value)
    if isinstance(value, dict):
        return " / ".join(f"{a}: {_summary_value(b)}" for a, b in value.items())
    return str(value)


def _format_label(value):
    return value.replace("_", " ").replace("-", " ").capitalize()


def _metadata(page, record):
    obj = record["evidence"]["surface"]
    width = page.width - 28
    name = f'{record["title"]}'
    y = page.paragraph(14, 32, name, width, 18, True, INK, max_lines=2)
    reveal = f'{obj["title"]} / {obj["width_mm"]:g} x {obj["height_mm"]:g} mm'
    if obj["accession"]:
        reveal += " / " + obj["accession"]
    if obj["synthetic"]:
        reveal += " / synthetic training"
    y = page.paragraph(14, y + 3, reveal, width, 9.5)
    y = page.paragraph(14, y + 4, record["report"]["summary"], width, 9.5, max_lines=3)
    zone = record["evidence"]["parameters"]["region"]
    fields = [
        ("Review status", record["state"]), ("Working region", [_number(f) for f in zone]),
        ("Photos at save", len(record["evidence"]["sources"])), ("Saved at UTC", record["created_at"]),
        ("Tool", record["kind"]),
    ]
    for key, value in fields:
        page.label(14, y + 5, key, 8, True)
        y = page.paragraph(55, y + 5, _summary_value(value), width - 45, 8.5, max_lines=2)
    grid = record["evidence"].get("interior_grid")
    if grid and grid.get("sources"):
        cells = " / ".join(
            f'{item["photo"]}: {item.get("support_extent_mm") or "none"} mm, edge {item.get("edge_correction_px")} px'
            for item in grid["sources"])
        page.label(14, y + 5, "Interior cells", 8, True)
        y = page.paragraph(55, y + 5, cells, width - 45, 8.5, max_lines=3)
    return y + 7


def _metrics(page, y, report):
    page.label(14, y, "Inspection readings", 11, True, GOLD)
    width = page.width - 28
    y += 4
    for name, value in report.get("metrics", {}).items():
        if y + 12 > page.height - 48:
            page.label(14, y + 5, "More detail is available in the retained record.", 8, False, MUTED)
            return y + 12
        page.line(14, y, page.width - 14, y)
        page.paragraph(14, y + 5, _format_label(name), 54, 8, True, INK, 2)
        y = max(y + 9, page.paragraph(73, y + 5, _summary_value(value), width - 59, 8, max_lines=2))
    return y + 4


def _line_calibrator(page, length=50):
    left, y = page.width - length - 15, page.height - 29
    page.line(left, y, left + length, y, INK, .9)
    for step in range(0, length + 1, 10):
        page.line(left + step, y - 3, left + step, y + 1, INK, .7)
    page.label(left, y + 5, f"{length} mm printing check", 7.5, False, MUTED)


def _source_sheet(page, sources, start):
    y = 33
    page.label(14, y, f"Source photographs · {start+1}–{start+len(sources)}", 13, True)
    y += 9
    for offset, source in enumerate(sources, start + 1):
        y += 4
        page.line(14, y - 4, page.width - 14, y - 4)
        page.paragraph(14, y + 4, f'{offset:02d}   {source["title"]}', page.width - 30, 10, True, INK)
        settings = source["settings"]
        label = f'{source["role"]} / {source["size"][0]} × {source["size"][1]} px / ' \
                f'{settings["azimuth"]:g}° azimuth / {settings["elevation"]:g}° elevation / {settings["exposure"]:g} EV'
        page.paragraph(14, y + 11, label, page.width - 28, 8)
        page.label(14, y + 16, f'Original SHA-256 {source["hash"]}', 7.3, False, MUTED)
        page.label(14, y + 21, f'Registration {source["registration_id"]}', 8, False, MUTED)
        grid = source.get("interior_grid")
        if grid:
            page.label(14, y + 26, (f'Interior cells: {grid.get("control_points", 0)} marks, '
                                    f'{_summary_value(grid.get("support_extent_mm"))} mm, '
                                    f'edge {grid.get("edge_correction_px")} px'), 7.5, False, MUTED)
            y += 5
        y += 22


def _plate(page, name, picture, record):
    width = page.width - 28
    room = page.height - 78
    page.label(14, 32, _format_label(name), 13, True)
    page.rectangle(14, 39, width, room, fill=BAND, stroke=BORDER)
    measured = page.image(picture, 15, 40, width - 2, room - 2)
    title = record["report"]["title"]
    page.paragraph(14, 45 + room, title, width * .55, 7.5)
    _line_calibrator(page)
    return measured


def _page_limits(page):
    return max(4, math.floor((page.height - 65) / 27))


def report_pdf(library, record_id, paper, orientation, shown):
    paper = choice(paper, "Paper", PAPERS)
    orientation = choice(orientation, "Page orientation", ("portrait", "landscape"))
    report = library.get_inspection(record_id)
    media = [file for file in report["files"] if file["mime"] == "image/png"]
    allowed = {part["key"] for part in media}
    if shown and (len(shown) > len(allowed) or not set(shown).issubset(allowed)):
        raise Problem("Choose existing retained picture files for printing.")
    if shown:
        media = [item for item in media if item["key"] in shown]
    intro = _page(paper, orientation, report, 1)
    pages = [intro]
    _metrics(intro, _metadata(intro, report), report["report"])
    _line_calibrator(intro)
    sources = report["evidence"]["sources"]
    limit = _page_limits(intro)
    for start in range(0, len(sources), limit):
        page = _page(paper, orientation, report, len(pages) + 1)
        _source_sheet(page, sources[start:start + limit], start)
        pages.append(page)
    for file in media:
        page = _page(paper, orientation, report, len(pages) + 1)
        data, _mime, _name = library.inspection_asset(record_id, file["key"])
        _plate(page, file["key"], data, report)
        pages.append(page)
    if len(pages) > MAX_PAGES:
        raise Problem("Choose a smaller set of evidence pictures for printing.", 413)
    return f"inspection-{record_id[:10]}-r{report['revision']}.pdf", PDF().assemble(pages)
