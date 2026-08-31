"""SQLite catalog for document provenance, versions, chunks and CDC idempotency."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from agents.doc_parser_agent import DocumentChunk
from config import settings


@dataclass(slots=True)
class DocumentRecord:
    doc_id: str
    file_name: str
    file_path: str
    mime_type: str
    content_hash: str
    version: int
    status: str
    error: str
    created_at: float
    updated_at: float


@dataclass(slots=True)
class EventRecord:
    event_id: str
    operation: str
    doc_id: str
    file_path: str
    observed_hash: str
    status: str = "PENDING"
    attempts: int = 0
    error: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0


class CatalogService:
    """Thread-safe, local catalog used by the async application."""

    def __init__(self, path: str | None = None) -> None:
        self.path = path or settings.catalog_path
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    async def init(self) -> None:
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                mime_type TEXT NOT NULL DEFAULT '',
                content_hash TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 0,
                status TEXT NOT NULL,
                error TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            CREATE UNIQUE INDEX IF NOT EXISTS idx_documents_path ON documents(file_path);
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL REFERENCES documents(doc_id) ON DELETE CASCADE,
                content TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                doc_type TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                version INTEGER NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks(doc_id);
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                operation TEXT NOT NULL,
                doc_id TEXT NOT NULL,
                file_path TEXT NOT NULL,
                observed_hash TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                error TEXT NOT NULL DEFAULT '',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            );
            """
        )
        self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    def _db(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("catalog is not initialized")
        return self._conn

    def begin_document(
        self,
        *,
        doc_id: str,
        file_name: str,
        file_path: str,
        mime_type: str,
        content_hash: str,
    ) -> int:
        now = time.time()
        with self._lock:
            db = self._db()
            current = db.execute(
                "SELECT version,created_at FROM documents WHERE doc_id=?", (doc_id,)
            ).fetchone()
            next_version = (int(current["version"]) if current else 0) + 1
            created_at = float(current["created_at"]) if current else now
            db.execute(
                """
                INSERT INTO documents(
                  doc_id,file_name,file_path,mime_type,content_hash,version,
                  status,error,created_at,updated_at
                )
                VALUES(?,?,?,?,?,?,'PROCESSING','',?,?)
                ON CONFLICT(doc_id) DO UPDATE SET
                  file_name=excluded.file_name,file_path=excluded.file_path,mime_type=excluded.mime_type,
                  content_hash=excluded.content_hash,status='PROCESSING',error='',updated_at=excluded.updated_at
                """,
                (doc_id, file_name, file_path, mime_type, content_hash, next_version, created_at, now),
            )
            db.commit()
            return next_version

    def commit_document(
        self,
        doc_id: str,
        content_hash: str,
        version: int,
        chunks: Iterable[DocumentChunk],
    ) -> None:
        now = time.time()
        chunks = list(chunks)
        with self._lock:
            db = self._db()
            db.execute("BEGIN")
            db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
            db.executemany(
                """
                INSERT INTO chunks(
                  chunk_id,doc_id,content,content_hash,ordinal,doc_type,
                  metadata_json,version,created_at
                )
                VALUES(?,?,?,?,?,?,?,?,?)
                """,
                [
                    (
                        chunk.chunk_id,
                        doc_id,
                        chunk.content,
                        chunk.content_hash,
                        chunk.chunk_index,
                        chunk.doc_type.value,
                        json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                        version,
                        now,
                    )
                    for chunk in chunks
                ],
            )
            db.execute(
                """
                UPDATE documents
                SET content_hash=?,version=?,status='READY',error='',updated_at=?
                WHERE doc_id=?
                """,
                (content_hash, version, now, doc_id),
            )
            db.commit()

    def fail_document(self, doc_id: str, error: str) -> None:
        with self._lock:
            db = self._db()
            db.execute(
                "UPDATE documents SET status='FAILED',error=?,updated_at=? WHERE doc_id=?",
                (error[:2000], time.time(), doc_id),
            )
            db.commit()

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        row = self._db().execute("SELECT * FROM documents WHERE doc_id=?", (doc_id,)).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def get_document_by_path(self, file_path: str) -> DocumentRecord | None:
        row = self._db().execute("SELECT * FROM documents WHERE file_path=?", (file_path,)).fetchone()
        return DocumentRecord(**dict(row)) if row else None

    def get_chunks(self, doc_id: str) -> list[dict[str, Any]]:
        rows = (
            self._db().execute("SELECT * FROM chunks WHERE doc_id=? ORDER BY ordinal", (doc_id,)).fetchall()
        )
        result: list[dict[str, Any]] = []
        for row in rows:
            item = dict(row)
            item["metadata"] = json.loads(item.pop("metadata_json"))
            result.append(item)
        return result

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            db = self._db()
            count = db.execute("DELETE FROM documents WHERE doc_id=?", (doc_id,)).rowcount
            db.commit()
            return count

    def list_documents(self, *, include_failed: bool = False) -> list[DocumentRecord]:
        query = "SELECT * FROM documents"
        params: tuple[str, ...] = ()
        if not include_failed:
            query += " WHERE status='READY'"
        query += " ORDER BY created_at,doc_id"
        rows = self._db().execute(query, params).fetchall()
        return [DocumentRecord(**dict(row)) for row in rows]

    def enqueue_event(self, event: EventRecord) -> None:
        """Persist an event as PENDING before it is published to Kafka."""
        now = time.time()
        with self._lock:
            db = self._db()
            values = asdict(event)
            values.update(status="PENDING", attempts=0, created_at=now, updated_at=now)
            db.execute(
                """
                INSERT OR IGNORE INTO events(
                    event_id,operation,doc_id,file_path,observed_hash,status,
                    attempts,error,created_at,updated_at
                ) VALUES(
                    :event_id,:operation,:doc_id,:file_path,:observed_hash,:status,
                    :attempts,:error,:created_at,:updated_at
                )
                """,
                values,
            )
            db.commit()

    def claim_event(self, event: EventRecord) -> bool:
        """Claim an event for one attempt using an atomic status transition."""
        now = time.time()
        with self._lock:
            db = self._db()
            existing = db.execute(
                "SELECT status,attempts FROM events WHERE event_id=?", (event.event_id,)
            ).fetchone()
            if existing and existing["status"] in {"COMMITTED", "APPLYING"}:
                return False
            if existing and int(existing["attempts"]) >= settings.cdc_max_retries:
                return False
            if existing:
                db.execute(
                    """
                    UPDATE events
                    SET status='APPLYING',attempts=attempts+1,error='',updated_at=?
                    WHERE event_id=?
                    """,
                    (now, event.event_id),
                )
            else:
                values = asdict(event)
                values.update(status="APPLYING", attempts=1, created_at=now, updated_at=now)
                db.execute(
                    """
                    INSERT INTO events(
                      event_id,operation,doc_id,file_path,observed_hash,
                      status,attempts,error,created_at,updated_at
                    ) VALUES(
                      :event_id,:operation,:doc_id,:file_path,:observed_hash,
                      :status,:attempts,:error,:created_at,:updated_at
                    )
                    """,
                    values,
                )
            db.commit()
            return True

    def recover_interrupted_events(self, *, older_than_seconds: float = 300.0) -> int:
        """Move stale APPLYING events back to FAILED so they may be retried."""
        cutoff = time.time() - older_than_seconds
        with self._lock:
            db = self._db()
            count = db.execute(
                """
                UPDATE events
                SET status='FAILED',error='interrupted while applying',updated_at=?
                WHERE status='APPLYING' AND updated_at < ?
                """,
                (time.time(), cutoff),
            ).rowcount
            db.commit()
            return count

    def finish_event(self, event_id: str, success: bool, error: str = "") -> None:
        with self._lock:
            db = self._db()
            db.execute(
                "UPDATE events SET status=?,error=?,updated_at=? WHERE event_id=?",
                ("COMMITTED" if success else "FAILED", error[:2000], time.time(), event_id),
            )
            db.commit()

    def get_event(self, event_id: str) -> EventRecord | None:
        row = self._db().execute("SELECT * FROM events WHERE event_id=?", (event_id,)).fetchone()
        return EventRecord(**dict(row)) if row else None

    def delete_events(self, event_ids: Iterable[str]) -> int:
        identifiers = list(dict.fromkeys(event_ids))
        if not identifiers:
            return 0
        with self._lock:
            db = self._db()
            deleted = 0
            for start in range(0, len(identifiers), 500):
                batch = identifiers[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                deleted += db.execute(
                    f"DELETE FROM events WHERE event_id IN ({placeholders})",
                    batch,
                ).rowcount
            db.commit()
            return deleted

    def get_stats(self) -> dict[str, int]:
        db = self._db()
        return {
            "documents": int(db.execute("SELECT count(*) FROM documents").fetchone()[0]),
            "chunks": int(db.execute("SELECT count(*) FROM chunks").fetchone()[0]),
            "events": int(db.execute("SELECT count(*) FROM events").fetchone()[0]),
            "failed_documents": int(
                db.execute("SELECT count(*) FROM documents WHERE status='FAILED'").fetchone()[0]
            ),
        }
