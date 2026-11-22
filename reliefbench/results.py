"""A single report envelope for live and saved pages."""
from dataclasses import dataclass, field
from io import StringIO
import base64
import csv
import json

from .raster import png, thumb


@dataclass(frozen=True)
class Asset:
    key: str
    name: str
    mime: str
    content: bytes

    def preview(self):
        view = {"key": self.key, "name": self.name, "mime": self.mime, "bytes": len(self.content)}
        inline = self.mime in ("text/csv", "application/json", "image/png") and len(self.content) < 1_400_000
        if inline:
            view["data"] = f"data:{self.mime};base64," + base64.b64encode(self.content).decode("ascii")
        return view


def _columns(rows):
    return list(dict.fromkeys(k for row in rows for k in row))


def _cell(value):
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, str) and value[:1] in ("=", "+", "-", "@", "\t"):
        return "'" + value
    return value


def as_csv(rows):
    buffer = StringIO(newline="")
    columns = _columns(rows)
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _cell(v) for k, v in row.items()})
    return buffer.getvalue().encode("utf-8")


@dataclass
class Result:
    kind: str
    title: str
    summary: str
    metrics: dict
    rows: list = field(default_factory=list)
    warnings: list = field(default_factory=list)
    assets: dict = field(default_factory=dict)
    payload: dict = field(default_factory=dict)

    @property
    def report(self):
        return {"kind": self.kind, "title": self.title, "summary": self.summary,
                "metrics": self.metrics, "rows": self.rows, "warnings": self.warnings}

    def add(self, key, name, mime, content):
        if key in self.assets:
            raise ValueError(f"Duplicate result asset {key}")
        self.assets[key] = Asset(key, name, mime, content)
        return self

    def image(self, key, picture, high_resolution=False):
        image = picture if high_resolution else thumb(picture)
        return self.add(key, f"{key}.png", "image/png", png(image))

    def table(self, key, rows=None):
        values = self.rows if rows is None else rows
        return self.add(key, f"{key}.csv", "text/csv", as_csv(values))

    def document(self, key, blob):
        payload = json.dumps(blob, ensure_ascii=False, indent=2, allow_nan=False).encode()
        return self.add(key, f"{key}.json", "application/json", payload)

    def public(self, evidence):
        return {"report": self.report, "evidence": evidence,
                "files": [item.preview() for item in self.assets.values()]}
