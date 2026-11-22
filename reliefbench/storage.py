"""Local projects and content-addressed source photographs."""
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4
import json
import os
import sqlite3

from .models import Problem
from .raster import decode


DB_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS asset (
    hash TEXT PRIMARY KEY,
    size INTEGER NOT NULL,
    mime TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS surface (
    id TEXT PRIMARY KEY,
    body TEXT NOT NULL,
    version INTEGER NOT NULL,
    sample_key TEXT UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS capture (
    id TEXT PRIMARY KEY,
    surface_id TEXT NOT NULL REFERENCES surface(id),
    image_hash TEXT NOT NULL REFERENCES asset(hash),
    slug TEXT NOT NULL,
    body TEXT NOT NULL,
    version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    UNIQUE(surface_id, image_hash)
);
CREATE TABLE IF NOT EXISTS registration (
    id TEXT PRIMARY KEY,
    capture_id TEXT NOT NULL REFERENCES capture(id),
    revision INTEGER NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(capture_id, revision)
);
CREATE TABLE IF NOT EXISTS inspection (
    id TEXT PRIMARY KEY,
    surface_id TEXT NOT NULL REFERENCES surface(id),
    title TEXT NOT NULL,
    kind TEXT NOT NULL,
    evidence TEXT NOT NULL,
    report TEXT NOT NULL,
    state TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS inspection_file (
    inspection_id TEXT NOT NULL REFERENCES inspection(id),
    key TEXT NOT NULL,
    name TEXT NOT NULL,
    hash TEXT NOT NULL REFERENCES asset(hash),
    PRIMARY KEY(inspection_id, key)
);
CREATE TABLE IF NOT EXISTS procedure (
    id TEXT PRIMARY KEY,
    surface_id TEXT NOT NULL REFERENCES surface(id),
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(surface_id, title)
);
CREATE TABLE IF NOT EXISTS review_event (
    id INTEGER PRIMARY KEY,
    inspection_id TEXT NOT NULL REFERENCES inspection(id),
    reviewer TEXT NOT NULL,
    decision TEXT NOT NULL,
    revision INTEGER NOT NULL,
    note TEXT NOT NULL,
    decided_at TEXT NOT NULL,
    UNIQUE(inspection_id, revision)
);
CREATE TABLE IF NOT EXISTS height_join (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    left_id TEXT NOT NULL REFERENCES inspection(id),
    right_id TEXT NOT NULL REFERENCES inspection(id),
    body TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS join_file (
    join_id TEXT NOT NULL REFERENCES height_join(id),
    key TEXT NOT NULL,
    name TEXT NOT NULL,
    hash TEXT NOT NULL REFERENCES asset(hash),
    PRIMARY KEY(join_id, key)
);
CREATE INDEX IF NOT EXISTS capture_surface ON capture(surface_id, created_at);
CREATE INDEX IF NOT EXISTS inspection_surface ON inspection(surface_id, created_at);
CREATE INDEX IF NOT EXISTS join_created ON height_join(created_at);
"""


def utc():
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def key():
    return uuid4().hex


def dumps(data):
    return json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def guarded(row, subject):
    if row is None:
        raise Problem(f"Unknown {subject}.", 404)
    return row


def expose_surface(row):
    return {"id": row["id"], "version": row["version"], "created_at": row["created_at"],
            "updated_at": row["updated_at"], **json.loads(row["body"]),
            **({"sample": row["sample_key"]} if row["sample_key"] else {})}


def expose_capture(row):
    return {"id": row["id"], "surface_id": row["surface_id"], "hash": row["image_hash"],
            "filename": row["slug"], "version": row["version"], "created_at": row["created_at"],
            "updated_at": row["updated_at"], **json.loads(row["body"])}


def expose_registration(row):
    return {"id": row["id"], "capture_id": row["capture_id"], "revision": row["revision"],
            "created_at": row["created_at"], **json.loads(row["body"])}


def expose_inspection(row, include_report=False):
    values = {k: row[k] for k in ("id", "surface_id", "kind", "title", "state", "revision", "created_at", "updated_at")}
    if include_report:
        values.update(report=json.loads(row["report"]), evidence=json.loads(row["evidence"]))
    return values


class Library:
    def __init__(self, data):
        self.dir = Path(data).absolute()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.blobs = self.dir / "objects"
        self.blobs.mkdir(exist_ok=True)
        self.db = self.dir / "relief.sqlite3"
        with self.connect() as conn:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.executescript(DB_SCHEMA)

    @contextmanager
    def connect(self, writing=False):
        conn = sqlite3.connect(self.db, timeout=15, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            if writing:
                conn.execute("BEGIN IMMEDIATE")
            yield conn
            if writing:
                conn.commit()
        except Exception:
            if writing:
                conn.rollback()
            raise
        finally:
            conn.close()

    def _blob_path(self, digest):
        return self.blobs / digest[:2] / digest

    def _store_asset(self, conn, data, mime):
        digest = sha256(data).hexdigest()
        path = self._blob_path(digest)
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            tmp = path.with_name(digest + "." + key() + ".part")
            try:
                with open(tmp, "xb") as handle:
                    handle.write(data)
                os.replace(tmp, path)
            finally:
                tmp.unlink(missing_ok=True)
        row = conn.execute("SELECT * FROM asset WHERE hash=?", (digest,)).fetchone()
        if row is not None and row["size"] != len(data):
            raise Problem("The library found a conflicting asset digest.", 409)
        conn.execute("INSERT OR IGNORE INTO asset VALUES (?,?,?,?)", (digest, len(data), mime, utc()))
        return digest

    @staticmethod
    def _surface(conn, surface_id):
        return guarded(conn.execute("SELECT * FROM surface WHERE id=?", (surface_id,)).fetchone(), "surface")

    @staticmethod
    def _capture(conn, capture_id):
        return guarded(conn.execute("SELECT * FROM capture WHERE id=?", (capture_id,)).fetchone(), "photograph")

    @staticmethod
    def _registration(conn, capture_id, registration_id):
        if registration_id == "identity":
            return None
        if registration_id is None:
            row = conn.execute("SELECT * FROM registration WHERE capture_id=? ORDER BY revision DESC LIMIT 1",
                               (capture_id,)).fetchone()
            return expose_registration(row) if row is not None else None
        row = conn.execute("SELECT * FROM registration WHERE capture_id=? AND id=?",
                           (capture_id, registration_id)).fetchone()
        return expose_registration(guarded(row, "registration for that photograph"))

    @staticmethod
    def _inspection(conn, inspection_id):
        return guarded(conn.execute("SELECT * FROM inspection WHERE id=?", (inspection_id,)).fetchone(), "inspection")

    def surfaces(self):
        with self.connect() as conn:
            query = """SELECT s.*, (SELECT COUNT(*) FROM capture AS c WHERE c.surface_id=s.id) AS captures,
                     (SELECT COUNT(*) FROM inspection AS i WHERE i.surface_id=s.id) AS inspections
                     FROM surface AS s ORDER BY s.updated_at DESC"""
            return [{**expose_surface(row), "captures": row["captures"], "inspections": row["inspections"]}
                    for row in conn.execute(query)]

    def _insert_surface(self, conn, spec, sample_key):
        sid = key()
        now = utc()
        conn.execute("INSERT INTO surface VALUES (?,?,?,?,?,?)",
                     (sid, dumps(spec.public()), 1, sample_key, now, now))
        return expose_surface(self._surface(conn, sid))

    def create_surface(self, spec):
        with self.connect(writing=True) as conn:
            return self._insert_surface(conn, spec, None)

    def ensure_sample(self, sample_key, spec, materials):
        decoded = [(name, raw, meta, decode(raw, name)) for name, raw, meta in materials]
        with self.connect(writing=True) as conn:
            existing = conn.execute("SELECT * FROM surface WHERE sample_key=?", (sample_key,)).fetchone()
            if existing is not None:
                return expose_surface(existing), False
            surface = self._insert_surface(conn, spec, sample_key)
            for name, raw, meta, capture in decoded:
                self._insert_capture(conn, surface["id"], name, raw, meta, capture)
            return surface, True

    def get_surface(self, surface_id):
        with self.connect() as conn:
            surface = expose_surface(self._surface(conn, surface_id))
            rows = conn.execute("SELECT * FROM capture WHERE surface_id=? ORDER BY created_at,id", (surface_id,))
            captures = [expose_capture(row) for row in rows]
            for cap in captures:
                registrations = conn.execute("SELECT * FROM registration WHERE capture_id=? ORDER BY revision DESC",
                                             (cap["id"],))
                cap["registrations"] = [expose_registration(entry) for entry in registrations]
            inspections = conn.execute("SELECT * FROM inspection WHERE surface_id=? ORDER BY created_at DESC",
                                       (surface_id,))
            return {"surface": surface, "captures": captures, "inspections": [expose_inspection(i) for i in inspections],
                    "procedures": self._procedures(conn, surface_id)}

    def edit_surface(self, surface_id, expected, spec):
        with self.connect(writing=True) as conn:
            row = self._surface(conn, surface_id)
            if row["version"] != expected:
                raise Problem("The surface was edited in another window. Reload before saving.", 409)
            conn.execute("UPDATE surface SET body=?, version=version+1, updated_at=? WHERE id=?",
                         (dumps(spec.public()), utc(), surface_id))
            return expose_surface(self._surface(conn, surface_id))

    def _insert_capture(self, conn, surface_id, name, data, spec, raw):
        identity = self._store_asset(conn, data, raw.mime)
        observed = conn.execute("SELECT id FROM capture WHERE surface_id=? AND image_hash=?",
                                (surface_id, identity)).fetchone()
        if observed:
            raise Problem("This surface already contains these original bytes.", 409)
        cid, stamp = key(), utc()
        body = {"metadata": spec.public(), "image": raw.info}
        conn.execute("INSERT INTO capture VALUES (?,?,?,?,?,?,?,?)",
                     (cid, surface_id, identity, name, dumps(body), 1, stamp, stamp))
        conn.execute("UPDATE surface SET updated_at=? WHERE id=?", (stamp, surface_id))
        return expose_capture(self._capture(conn, cid))

    def add_capture(self, surface_id, name, data, spec):
        raw = decode(data, name)
        with self.connect(writing=True) as conn:
            self._surface(conn, surface_id)
            count = conn.execute("SELECT COUNT(*) FROM capture WHERE surface_id=?", (surface_id,)).fetchone()[0]
            if count >= 48:
                raise Problem("A surface can retain up to 48 photographs.", 413)
            return self._insert_capture(conn, surface_id, name, data, spec, raw)

    def edit_capture(self, capture_id, expected, spec):
        with self.connect(writing=True) as conn:
            row = self._capture(conn, capture_id)
            if expected != row["version"]:
                raise Problem("The photograph settings changed. Reload before saving.", 409)
            body = json.loads(row["body"])
            body["metadata"] = spec.public()
            conn.execute("UPDATE capture SET body=?, version=version+1, updated_at=? WHERE id=?",
                         (dumps(body), utc(), capture_id))
            return expose_capture(self._capture(conn, capture_id))

    def capture_asset(self, capture_id):
        with self.connect() as conn:
            row = expose_capture(self._capture(conn, capture_id))
        return row, self.read_asset(row["hash"])

    def read_asset(self, digest):
        with self.connect() as conn:
            row = guarded(conn.execute("SELECT * FROM asset WHERE hash=?", (digest,)).fetchone(), "stored image")
        try:
            data = self._blob_path(row["hash"]).read_bytes()
        except OSError as exc:
            raise Problem("The stored image is missing from the local volume.", 409) from exc
        if len(data) != row["size"] or sha256(data).hexdigest() != row["hash"]:
            raise Problem("The stored image did not match its content fingerprint.", 409)
        return data, row["mime"]

    def registrations(self, capture_id):
        with self.connect() as conn:
            self._capture(conn, capture_id)
            rows = conn.execute("SELECT * FROM registration WHERE capture_id=? ORDER BY revision DESC",
                                (capture_id,))
            return [expose_registration(row) for row in rows]

    def add_registration(self, capture_id, spec, metrics):
        # A reference keeps chronological versions. Excessive repeats usually
        # come from clicking Save instead of Preview during corner adjustment.
        with self.connect(writing=True) as conn:
            self._capture(conn, capture_id)
            latest = self._registration(conn, capture_id, None)
            if latest and latest["corners"] == spec.public()["corners"]:
                raise Problem("These are already the latest registration corners.", 409)
            rev = latest["revision"] + 1 if latest else 1
            if rev > 36:
                raise Problem("Keep at most 36 registrations per photograph.", 413)
            rid = key()
            body = {**spec.public(), "metrics": metrics}
            conn.execute("INSERT INTO registration VALUES (?,?,?,?,?)",
                         (rid, capture_id, rev, dumps(body), utc()))
            row = conn.execute("SELECT * FROM registration WHERE id=?", (rid,)).fetchone()
            return expose_registration(row)

    def acquire(self, surface_id, capture_ids, corrections):
        with self.connect() as conn:
            conn.execute("BEGIN")
            surface = expose_surface(self._surface(conn, surface_id))
            rows = []
            for cid in capture_ids:
                row = expose_capture(self._capture(conn, cid))
                if row["surface_id"] != surface_id:
                    raise Problem("All selected photographs must belong to this surface.")
                row["registration"] = self._registration(conn, cid, corrections.get(cid))
                rows.append(row)
            conn.commit()
        return surface, [(row, self.read_asset(row["hash"])[0]) for row in rows]

    def save_inspection(self, surface, title, result, evidence):
        rid, stamp = key(), utc()
        with self.connect(writing=True) as conn:
            self._surface(conn, surface)
            conn.execute("INSERT INTO inspection VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (rid, surface, title, result.kind, dumps(evidence), dumps(result.report),
                          "draft", 1, stamp, stamp))
            for item in result.assets.values():
                digest = self._store_asset(conn, item.content, item.mime)
                conn.execute("INSERT INTO inspection_file VALUES (?,?,?,?)",
                             (rid, item.key, item.name, digest))
            conn.execute("UPDATE surface SET updated_at=? WHERE id=?", (stamp, surface))
            return expose_inspection(self._inspection(conn, rid))

    def get_inspection(self, inspection_id):
        with self.connect() as conn:
            summary = expose_inspection(self._inspection(conn, inspection_id), True)
            media = conn.execute(
                """SELECT f.key,f.name,a.mime,a.size,a.hash FROM inspection_file f INNER JOIN asset a ON a.hash=f.hash
                WHERE f.inspection_id=? ORDER BY f.key""", (inspection_id,))
            summary["files"] = [dict(row) for row in media]
            events = conn.execute("""SELECT reviewer,decision,revision,note,decided_at FROM review_event
                                   WHERE inspection_id=? ORDER BY revision""", (inspection_id,))
            summary["events"] = [dict(event) for event in events]
            return summary

    def mark_inspection(self, inspection_id, expected, reviewer, decision, note):
        with self.connect(writing=True) as conn:
            row = self._inspection(conn, inspection_id)
            if row["revision"] != expected:
                raise Problem("Another reviewer already edited this record.", 409)
            if row["state"] == decision:
                raise Problem("Select a review state different from the current state.", 409)
            clock = utc()
            revision = 1 + expected
            conn.execute("UPDATE inspection SET state=?, revision=?, updated_at=? WHERE id=?",
                         (decision, revision, clock, inspection_id))
            conn.execute("""INSERT INTO review_event(inspection_id,reviewer,decision,revision,note,decided_at)
                          VALUES (?,?,?,?,?,?)""", (inspection_id, reviewer, decision, revision, note, clock))
            return expose_inspection(self._inspection(conn, inspection_id))

    def inspection_asset(self, inspection_id, name):
        with self.connect() as conn:
            self._inspection(conn, inspection_id)
            row = guarded(conn.execute("SELECT name,hash FROM inspection_file WHERE inspection_id=? AND key=?",
                                       (inspection_id, name)).fetchone(), "inspection media")
        data, mime = self.read_asset(row["hash"])
        return data, mime, row["name"]

    def import_capture_set(self, spec, document):
        with self.connect(writing=True) as conn:
            sid = self._insert_surface(conn, spec, None)["id"]
            imported = []
            for photo in document.photos:
                row = self._insert_capture(conn, sid, photo.name, photo.raw, photo.metadata, photo.image)
                for rev, (registration, metrics) in enumerate(photo.registrations, 1):
                    body = {**registration.public(), "metrics": metrics}
                    conn.execute("INSERT INTO registration VALUES (?,?,?,?,?)",
                                 (key(), row["id"], rev, dumps(body), utc()))
                imported.append({"previous": photo.source_id, "current": row["id"], "source_hash": photo.digest,
                                 "registrations": len(photo.registrations)})
            return {"surface": expose_surface(self._surface(conn, sid)), "imported": imported}

    def _procedures(self, conn, surface_id):
        rows = conn.execute("SELECT * FROM procedure WHERE surface_id=? ORDER BY created_at DESC", (surface_id,))
        return [{"id": row["id"], "created_at": row["created_at"], **json.loads(row["body"])} for row in rows]

    def height_records(self):
        with self.connect() as conn:
            rows = conn.execute(
                """SELECT i.id, i.title, i.surface_id, i.created_at, s.body AS surface_body
                   FROM inspection AS i INNER JOIN surface AS s ON s.id=i.surface_id
                   WHERE i.kind='height' ORDER BY i.created_at DESC""")
            return [{"id": row["id"], "title": row["title"], "surface_id": row["surface_id"],
                     "surface_title": json.loads(row["surface_body"])["title"],
                     "created_at": row["created_at"]} for row in rows]

    def save_join(self, title, left_id, right_id, body, assets):
        join_id, stamp = key(), utc()
        with self.connect(writing=True) as conn:
            self._inspection(conn, left_id)
            self._inspection(conn, right_id)
            conn.execute("INSERT INTO height_join VALUES (?,?,?,?,?,?)",
                         (join_id, title, left_id, right_id, dumps(body), stamp))
            self._store_join_assets(conn, join_id, assets)
            return {"id": join_id, "title": title, "left_id": left_id, "right_id": right_id,
                    "created_at": stamp, **body}

    def _store_join_assets(self, conn, join_id, assets):
        for item in assets.values():
            digest = self._store_asset(conn, item.content, item.mime)
            conn.execute("INSERT INTO join_file VALUES (?,?,?,?)", (join_id, item.key, item.name, digest))

    def joins(self):
        with self.connect() as conn:
            rows = conn.execute("""SELECT j.id, j.title, j.left_id, j.right_id, j.body, j.created_at
                                   FROM height_join AS j ORDER BY j.created_at DESC""")
            return [{"id": row["id"], "title": row["title"], "left_id": row["left_id"],
                     "right_id": row["right_id"], "created_at": row["created_at"], **json.loads(row["body"])}
                    for row in rows]

    def _join(self, conn, join_id):
        row = guarded(conn.execute("SELECT * FROM height_join WHERE id=?", (join_id,)).fetchone(), "joined height")
        media = conn.execute("""SELECT f.key,f.name,a.mime,a.size,a.hash FROM join_file f
                                INNER JOIN asset a ON a.hash=f.hash WHERE f.join_id=? ORDER BY f.key""",
                             (join_id,))
        return {**{k: row[k] for k in ("id", "title", "left_id", "right_id", "created_at")},
                **json.loads(row["body"]), "files": [dict(item) for item in media]}

    def get_join(self, join_id):
        with self.connect() as conn:
            return self._join(conn, join_id)

    def join_asset(self, join_id, key):
        with self.connect() as conn:
            self._join(conn, join_id)
            row = guarded(conn.execute("SELECT name,hash FROM join_file WHERE join_id=? AND key=?",
                                       (join_id, key)).fetchone(), "joined height file")
        data, mime = self.read_asset(row["hash"])
        return data, mime, row["name"]

    def procedures(self, surface_id):
        with self.connect() as conn:
            self._surface(conn, surface_id)
            return self._procedures(conn, surface_id)

    def _add_procedure(self, conn, surface_id, body):
        row = conn.execute("SELECT id FROM procedure WHERE surface_id=? AND title=?",
                           (surface_id, body["name"])).fetchone()
        if row:
            raise Problem("Choose a different procedure name for this surface.", 409)
        identifier, stamp = key(), utc()
        conn.execute("INSERT INTO procedure VALUES (?,?,?,?,?)", (identifier, surface_id, body["name"], dumps(body), stamp))
        return {"id": identifier, "created_at": stamp, **body}

    def get_procedure(self, surface_id, item):
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM procedure WHERE surface_id=? AND id=?", (surface_id, item)).fetchone()
            guarded(row, "saved procedure")
            return {"id": row["id"], "created_at": row["created_at"], **json.loads(row["body"])}
