from __future__ import annotations

from pathlib import Path

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from services.catalog import CatalogService, EventRecord


@pytest.mark.asyncio
async def test_catalog_versions_chunks_and_event_idempotency(tmp_path: Path) -> None:
    catalog = CatalogService(str(tmp_path / "catalog.sqlite3"))
    await catalog.init()
    version = catalog.begin_document(
        doc_id="doc",
        file_name="a.txt",
        file_path=str(tmp_path / "a.txt"),
        mime_type="text/plain",
        content_hash="hash-1",
    )
    chunk = DocumentChunk(
        "hello",
        "doc",
        0,
        DocType.TEXT,
        {"source": "a.txt"},
        identifier="chunk-1",
        content_hash="content-hash",
    )
    catalog.commit_document("doc", "hash-1", version, [chunk])
    assert catalog.get_document("doc").status == "READY"
    assert catalog.get_chunks("doc")[0]["chunk_id"] == "chunk-1"
    assert catalog.list_documents()[0].doc_id == "doc"

    event = EventRecord("event-1", "UPDATE", "doc", str(tmp_path / "a.txt"), "hash-2")
    catalog.enqueue_event(event)
    assert catalog.get_event("event-1").status == "PENDING"
    assert catalog.claim_event(event)
    assert not catalog.claim_event(event)
    recovered = catalog.recover_interrupted_events(older_than_seconds=-1)
    assert recovered == 1
    assert catalog.claim_event(event)
    catalog.finish_event("event-1", True)
    assert not catalog.claim_event(event)
    assert catalog.get_event("event-1").attempts == 2
    assert catalog.delete_events([]) == 0
    assert catalog.delete_events(["event-1", "event-1"]) == 1
    assert catalog.get_event("event-1") is None
    await catalog.close()


@pytest.mark.asyncio
async def test_catalog_failure_and_cleanup(tmp_path: Path) -> None:
    catalog = CatalogService(str(tmp_path / "catalog.sqlite3"))
    await catalog.init()
    catalog.begin_document(
        doc_id="bad",
        file_name="bad.txt",
        file_path=str(tmp_path / "bad.txt"),
        mime_type="text/plain",
        content_hash="x",
    )
    catalog.fail_document("bad", "failure")
    assert catalog.get_stats()["failed_documents"] == 1
    assert catalog.list_documents() == []
    assert catalog.delete_document("bad") == 1
    await catalog.close()
