"""SQLite catalog for document provenance, versions, chunks and CDC idempotency."""

from __future__ import annotations

import json
import re
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
        self._fts_enabled = False

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
        try:
            self._conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                    chunk_id UNINDEXED,
                    doc_id UNINDEXED,
                    content,
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
            # Keep the index self-healing when an existing catalog is opened
            # after this feature is introduced.
            self._conn.execute("DELETE FROM chunks_fts")
            rows = self._conn.execute(
                "SELECT chunk_id,doc_id,content,metadata_json FROM chunks"
            ).fetchall()
            self._conn.executemany(
                "INSERT INTO chunks_fts(chunk_id,doc_id,content) VALUES(?,?,?)",
                [
                    (
                        str(row["chunk_id"]),
                        str(row["doc_id"]),
                        self._searchable_content(str(row["content"]), str(row["metadata_json"])),
                    )
                    for row in rows
                ],
            )
            self._fts_enabled = True
        except sqlite3.OperationalError:
            # Some minimal SQLite builds omit FTS5. The public search method
            # still provides a deterministic in-process lexical fallback.
            self._fts_enabled = False
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
            try:
                db.execute("BEGIN")
                if self._fts_enabled:
                    db.execute("DELETE FROM chunks_fts WHERE doc_id=?", (doc_id,))
                db.execute("DELETE FROM chunks WHERE doc_id=?", (doc_id,))
                values = [
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
                ]
                db.executemany(
                    """
                    INSERT INTO chunks(
                      chunk_id,doc_id,content,content_hash,ordinal,doc_type,
                      metadata_json,version,created_at
                    )
                    VALUES(?,?,?,?,?,?,?,?,?)
                    """,
                    values,
                )
                if self._fts_enabled:
                    db.executemany(
                        "INSERT INTO chunks_fts(chunk_id,doc_id,content) VALUES(?,?,?)",
                        [
                            (
                                chunk.chunk_id,
                                doc_id,
                                self._searchable_content(
                                    chunk.content,
                                    json.dumps(chunk.metadata, ensure_ascii=False, sort_keys=True),
                                ),
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
            except Exception:
                db.rollback()
                raise

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

    @staticmethod
    def _searchable_content(content: str, metadata_json: str) -> str:
        try:
            metadata = json.loads(metadata_json)
        except (TypeError, json.JSONDecodeError):
            metadata = {}
        prefix = " ".join(
            str(metadata.get(key, ""))
            for key in ("file_name", "section", "title", "sheet", "page", "slide")
            if metadata.get(key) not in (None, "")
        )
        plain = f"{prefix}\n{content}".strip()
        cjk_ngrams = [
            sequence[index : index + 2]
            for sequence in re.findall(r"[\u3400-\u9fff]+", plain)
            for index in range(max(1, len(sequence) - 1))
        ]
        return f"{plain}\n{' '.join(cjk_ngrams)}".strip()

    @staticmethod
    def _lexical_tokens(query: str) -> list[str]:
        values = re.findall(r"[A-Za-z0-9_][A-Za-z0-9_.-]*|[\u3400-\u9fff]+", query.casefold())
        tokens: list[str] = []
        for value in values:
            tokens.append(value)
            if re.fullmatch(r"[\u3400-\u9fff]+", value) and len(value) > 2:
                tokens.extend(value[index : index + 2] for index in range(len(value) - 1))
        return list(dict.fromkeys(value for value in tokens if value))[:32]

    def search_chunks(
        self,
        query: str,
        *,
        limit: int = 40,
        document_ids: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Search catalog chunks with FTS5 BM25 and a portable fallback."""
        tokens = self._lexical_tokens(query)
        if not tokens or limit <= 0:
            return []
        with self._lock:
            db = self._db()
            if self._fts_enabled:
                match_query = " OR ".join(f'"{value.replace(chr(34), chr(34) * 2)}"' for value in tokens)
                clauses = ["chunks_fts MATCH ?", "documents.status='READY'"]
                params: list[Any] = [match_query]
                if document_ids:
                    placeholders = ",".join("?" for _ in document_ids)
                    clauses.append(f"chunks.doc_id IN ({placeholders})")
                    params.extend(document_ids)
                params.append(limit)
                rows = db.execute(
                    f"""
                    SELECT chunks.*, bm25(chunks_fts) AS bm25_score
                    FROM chunks_fts
                    JOIN chunks ON chunks.chunk_id=chunks_fts.chunk_id
                    JOIN documents ON documents.doc_id=chunks.doc_id
                    WHERE {' AND '.join(clauses)}
                    ORDER BY bm25_score ASC, chunks.chunk_id ASC
                    LIMIT ?
                    """,
                    params,
                ).fetchall()
                output: list[dict[str, Any]] = []
                for rank, row in enumerate(rows, start=1):
                    item = dict(row)
                    item["metadata"] = json.loads(item.pop("metadata_json"))
                    item["rank"] = rank
                    item["score"] = 1.0 / (1.0 + abs(float(item.pop("bm25_score"))))
                    output.append(item)
                return output

            clauses = ["documents.status='READY'"]
            params = []
            if document_ids:
                placeholders = ",".join("?" for _ in document_ids)
                clauses.append(f"chunks.doc_id IN ({placeholders})")
                params.extend(document_ids)
            rows = db.execute(
                f"""
                SELECT chunks.* FROM chunks
                JOIN documents ON documents.doc_id=chunks.doc_id
                WHERE {' AND '.join(clauses)}
                """,
                params,
            ).fetchall()
            scored: list[tuple[float, sqlite3.Row]] = []
            for row in rows:
                haystack = self._searchable_content(
                    str(row["content"]), str(row["metadata_json"])
                ).casefold()
                matched = sum(token in haystack for token in tokens)
                frequency = sum(haystack.count(token) for token in tokens)
                if matched:
                    scored.append((matched / len(tokens) + min(frequency, 10) / 100, row))
            scored.sort(key=lambda value: (-value[0], str(value[1]["chunk_id"])))
            output = []
            for rank, (score, row) in enumerate(scored[:limit], start=1):
                item = dict(row)
                item["metadata"] = json.loads(item.pop("metadata_json"))
                item.update(rank=rank, score=score)
                output.append(item)
            return output

    def get_context_windows(
        self,
        chunk_ids: Iterable[str],
        *,
        neighbor_count: int = 1,
        max_chars: int = 8000,
    ) -> dict[str, str]:
        """Return parent text plus nearby child chunks for reranking only."""
        identifiers = list(dict.fromkeys(value for value in chunk_ids if value))
        if not identifiers:
            return {}
        with self._lock:
            db = self._db()
            placeholders = ",".join("?" for _ in identifiers)
            targets = db.execute(
                "SELECT chunk_id,doc_id,ordinal,content,metadata_json "
                f"FROM chunks WHERE chunk_id IN ({placeholders})",
                identifiers,
            ).fetchall()
            by_doc: dict[str, list[sqlite3.Row]] = {}
            for target in targets:
                doc_id = str(target["doc_id"])
                if doc_id not in by_doc:
                    by_doc[doc_id] = db.execute(
                        "SELECT chunk_id,ordinal,content,metadata_json "
                        "FROM chunks WHERE doc_id=? ORDER BY ordinal",
                        (doc_id,),
                    ).fetchall()
            output: dict[str, str] = {}
            for target in targets:
                rows = by_doc[str(target["doc_id"])]
                index = next(
                    (position for position, row in enumerate(rows) if row["chunk_id"] == target["chunk_id"]),
                    0,
                )
                metadata = json.loads(str(target["metadata_json"]))
                parts = [str(metadata.get("parent_content", "")).strip()]
                start = max(0, index - neighbor_count)
                end = min(len(rows), index + neighbor_count + 1)
                unit_id = str(metadata.get("unit_id", ""))
                for row in rows[start:end]:
                    row_metadata = json.loads(str(row["metadata_json"]))
                    if unit_id and str(row_metadata.get("unit_id", "")) != unit_id:
                        continue
                    parts.append(str(row["content"]).strip())
                unique = list(dict.fromkeys(part for part in parts if part))
                output[str(target["chunk_id"])] = "\n\n".join(unique)[:max_chars]
            return output

    def delete_document(self, doc_id: str) -> int:
        with self._lock:
            db = self._db()
            if self._fts_enabled:
                db.execute("DELETE FROM chunks_fts WHERE doc_id=?", (doc_id,))
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
        stats = {
            "documents": int(db.execute("SELECT count(*) FROM documents").fetchone()[0]),
            "chunks": int(db.execute("SELECT count(*) FROM chunks").fetchone()[0]),
            "events": int(db.execute("SELECT count(*) FROM events").fetchone()[0]),
            "failed_documents": int(
                db.execute("SELECT count(*) FROM documents WHERE status='FAILED'").fetchone()[0]
            ),
        }
        stats["lexical_chunks"] = (
            int(db.execute("SELECT count(*) FROM chunks_fts").fetchone()[0])
            if self._fts_enabled
            else 0
        )
        return stats
