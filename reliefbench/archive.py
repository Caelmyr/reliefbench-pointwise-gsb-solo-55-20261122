"""Review packet with checkable identity and source evidence."""
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
from pathlib import PurePosixPath
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile, ZipInfo
import json
import re

from .models import Problem, filename


FORMAT = "reliefbench.evidence.v1"
CONTROL = "manifest.json"
RECORD = "inspection.json"
LEDGER = "reviews.json"
LIMIT_ENTRIES = 95
LIMIT_BYTES = 72 * 1024 * 1024
MAX_CONTROL_BYTES = 1_500_000
HASH = re.compile(r"^[0-9a-f]{64}$")


def _dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False).encode("utf-8") + b"\n"


def _hash(data):
    return sha256(data).hexdigest()


def _stable_media(data, mime, purpose, name, **extra):
    return {"path": name, "sha256": _hash(data), "bytes": len(data), "mime": mime, "purpose": purpose, **extra}


def _zip_write(arc, name, content):
    info = ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
    info.compress_type = ZIP_DEFLATED
    info.external_attr = (0o100644 << 16)
    arc.writestr(info, content)


def _metadata_entries(record):
    fact = {k: v for k, v in record.items() if k not in ("files", "events")}
    return {
        RECORD: (_dump(fact), "application/json", "inspection snapshot", {}),
        LEDGER: (_dump(record["events"]), "application/json", "review history", {}),
    }


def _source_name(source, index):
    return f"originals/{index:02d}-{source['id'][:10]}-{filename(source['filename'])}"


def _result_name(media):
    return f"results/{media['key']}/{filename(media['name'])}"


def _list_sources(record, library):
    src = record["evidence"]["sources"]
    for i, entry in enumerate(src, 1):
        blob, mime = library.read_asset(entry["hash"])
        yield _source_name(entry, i), (blob, mime, "original photograph",
                                       {"original_hash": entry["hash"], "capture": entry["id"]})


def _list_compared(record, library):
    epochs = record["evidence"].get("epochs")
    if not isinstance(epochs, list):
        return
    for index, epoch in enumerate(epochs, 1):
        if not isinstance(epoch, dict):
            continue
        try:
            snapshot = library.get_inspection(epoch["record"])
        except Problem:
            continue
        name = f"compared/{index:02d}-{snapshot['id'][:10]}.json"
        yield name, (_dump(snapshot), "application/json", "compared report snapshot",
                     {"compared_record": snapshot["id"]})


def _list_results(record, library):
    for item in record["files"]:
        blob, mime, _ = library.inspection_asset(record["id"], item["key"])
        yield _result_name(item), (blob, mime, "computed report asset", {"report_file": item["key"]})


def build_packet(library, item, include_sources):
    envelope = library.get_inspection(item)
    components = _metadata_entries(envelope)
    components.update(_list_results(envelope, library))
    components.update(_list_compared(envelope, library))
    sources = list(_list_sources(envelope, library)) if include_sources else []
    components.update(sources)
    payload_size = sum(len(raw) for raw, _, _, _ in components.values())
    if len(components) >= LIMIT_ENTRIES or payload_size > LIMIT_BYTES:
        raise Problem("This evidence packet exceeds the local inspection size limit.", 413)
    metadata = {name: _stable_media(raw, mime, purpose, name, **extra)
                for name, (raw, mime, purpose, extra) in components.items()}
    latest = max((event["decided_at"] for event in envelope["events"]), default=envelope["created_at"])
    manifest = {"format": FORMAT, "record_id": item, "review_status": envelope["state"],
                "revision": envelope["revision"], "snapshot_time": latest,
                "source_bytes_included": bool(include_sources),
                "source_count": len(envelope["evidence"]["sources"]),
                "files": [metadata[key] for key in sorted(metadata.keys())]}
    cache = BytesIO()
    with ZipFile(cache, "w", compression=ZIP_DEFLATED, compresslevel=4) as out:
        _zip_write(out, CONTROL, _dump(manifest))
        for name in sorted(components):
            _zip_write(out, name, components[name][0])
    return f"relief-{item[:12]}-v{envelope['revision']}.zip", cache.getvalue()


def _safe_name(key):
    if not isinstance(key, str) or len(key) > 241 or not key.isascii():
        return False
    path = PurePosixPath(key)
    return (not path.is_absolute() and path.as_posix() == key and len(path.parts) <= 3
            and ".." not in path.parts and all(re.fullmatch(r"[A-Za-z0-9_. -]+", p) for p in path.parts))


def _manifest(arc, names):
    if CONTROL not in names or names[CONTROL].file_size > MAX_CONTROL_BYTES:
        raise Problem("This packet has no readable ReliefBench manifest.")
    try:
        data = json.loads(arc.read(CONTROL))
    except (UnicodeError, json.JSONDecodeError, RuntimeError) as exc:
        raise Problem("This packet manifest could not be parsed.") from exc
    if not isinstance(data, dict) or data.get("format") != FORMAT or not isinstance(data.get("files"), list):
        raise Problem("The uploaded ZIP is not a ReliefBench evidence packet.")
    if len(data["files"]) > LIMIT_ENTRIES:
        raise Problem("This packet lists too many parts.")
    return data


def _descriptors(item):
    result = {}
    for row in item["files"]:
        if not isinstance(row, dict) or not _safe_name(row.get("path")):
            raise Problem("The manifest has an invalid relative path.")
        key = row["path"]
        if key == CONTROL or key in result:
            raise Problem("The manifest lists a duplicate path.")
        if not isinstance(row.get("sha256"), str) or not HASH.fullmatch(row["sha256"]):
            raise Problem("The manifest contains a malformed fingerprint.")
        if type(row.get("bytes")) is not int or not 0 <= row["bytes"] <= LIMIT_BYTES:
            raise Problem("The manifest contains an unsupported file size.")
        result[key] = row
    return result


def _stream_digest(arc, entry, expect):
    if entry.file_size != expect:
        return None
    digest = sha256()
    count = 0
    try:
        with arc.open(entry) as fp:
            while True:
                fragment = fp.read(262144)
                if not fragment:
                    break
                count += len(fragment)
                if count > expect:
                    return None
                digest.update(fragment)
    except (OSError, RuntimeError, NotImplementedError, BadZipFile):
        return None
    return digest.hexdigest() if count == expect else None


def _read_restricted(arc, key, available, unexpected, findings):
    if key not in available or key in unexpected:
        findings.append(f"Missing valid {key}.")
        return None
    target = arc.getinfo(key)
    if target.file_size > MAX_CONTROL_BYTES:
        findings.append(f"Too much data in {key}.")
        return None
    try:
        return json.loads(arc.read(target))
    except (UnicodeError, json.JSONDecodeError, RuntimeError):
        findings.append(f"Cannot read {key} as JSON.")
        return None


@dataclass(frozen=True)
class ArchiveReader:
    arc: ZipFile
    manifest: dict
    verified: tuple[str, ...]
    findings: list[str]

    def _get(self, name):
        return _read_restricted(self.arc, name, self.verified, (), self.findings)

    def inspection(self):
        value = self._get(RECORD)
        if not isinstance(value, dict):
            self.findings.append("Missing structured inspection data.")
            return None
        for key, other in (("id", "record_id"), ("revision", "revision"), ("state", "review_status")):
            if value.get(key) != self.manifest.get(other):
                self.findings.append(f"Inspection {key} disagrees with the manifest.")
        evidence = value.get("evidence")
        if not isinstance(evidence, dict) or not isinstance(evidence.get("sources"), list):
            self.findings.append("The inspection has no usable sources.")
            return None
        return value

    def check_sources(self, record, rows):
        evidence = record["evidence"]["sources"]
        if len(evidence) != self.manifest.get("source_count"):
            self.findings.append("The inspection source count disagrees with the manifest.")
        source_assets = {meta.get("capture"): meta for meta in rows.values()
                         if meta.get("purpose") == "original photograph" and isinstance(meta.get("capture"), str)}
        if not self.manifest.get("source_bytes_included"):
            if source_assets:
                self.findings.append("The packet lists source bytes despite its omitted-source flag.")
            return
        for row in evidence:
            if not isinstance(row, dict) or not isinstance(row.get("id"), str):
                self.findings.append("The inspection lists a malformed source entry.")
                continue
            take = source_assets.get(row["id"])
            if take is None or take["path"] not in self.verified:
                self.findings.append(f'Photograph {row["id"][:20]} is missing or does not match the manifest.')
            elif row.get("hash") != take["sha256"] or row.get("hash") != take.get("original_hash"):
                self.findings.append(f'Photograph {row["id"][:20]} differs from the saved source.')

    def check_history(self, record):
        log = self._get(LEDGER)
        if not isinstance(log, list) or len(log) >= LIMIT_ENTRIES:
            self.findings.append("Review history is not a supported ordered list.")
            return
        prior = "draft"
        index = 1
        for event in log:
            index += 1
            if not isinstance(event, dict) or event.get("revision") != index or event.get("decision") == prior:
                self.findings.append("Review revisions are out of order or repeated.")
                return
            prior = event.get("decision")
            if prior not in ("draft", "accepted", "revisit"):
                self.findings.append("Review history contains an unknown state.")
                return
        if record.get("revision") != index or record.get("state") != prior:
            self.findings.append("Review history does not end in the saved state.")


def verify_packet(payload):
    findings = []
    try:
        arc = ZipFile(BytesIO(payload))
    except (BadZipFile, OSError) as exc:
        raise Problem("Open a complete ReliefBench ZIP packet.") from exc
    with arc:
        entries = arc.infolist()
        sizes = sum(x.file_size for x in entries)
        if sizes > LIMIT_BYTES + MAX_CONTROL_BYTES or len(entries) > LIMIT_ENTRIES:
            raise Problem("The packet exceeds the read limit.", 413)
        counts = Counter(entry.filename for entry in entries)
        if any(count != 1 for count in counts.values()) or any(not _safe_name(name) for name in counts):
            raise Problem("The ZIP has unsafe or conflicting member names.")
        names = {entry.filename: entry for entry in entries}
        manifest = _manifest(arc, names)
        rows = _descriptors(manifest)
        unknown = set(names) - {CONTROL, *rows}
        absent = set(rows) - set(names)
        findings.extend(f"Unlisted member {name}." for name in sorted(unknown))
        findings.extend(f"Missing member {name}." for name in sorted(absent))
        valid = []
        for name in sorted(set(names) & set(rows)):
            row = rows[name]
            digest = _stream_digest(arc, names[name], row["bytes"])
            if digest != row["sha256"]:
                findings.append(f"Fingerprint mismatch in {name}.")
            else:
                valid.append(name)
        check = ArchiveReader(arc, manifest, tuple(valid), findings)
        record = check.inspection()
        if record is not None:
            check.check_sources(record, rows)
            check.check_history(record)
    return {"valid": not findings, "files_verified": valid, "checked_member_count": len(rows),
            "source_bytes_included": manifest.get("source_bytes_included"), "findings": findings,
            "review_state": manifest.get("review_status"), "record_id": manifest.get("record_id"),
            "rechecked_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
