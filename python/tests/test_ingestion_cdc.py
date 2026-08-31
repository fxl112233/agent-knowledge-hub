from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import ExtractionResult
from agents.knowledge_update_agent import (
    CDCEvent,
    ChangeType,
    DocumentChange,
    KnowledgeUpdateAgent,
)
from services.catalog import CatalogService
from services.ingestion import IngestionService


class LineParser:
    async def parse(self, file_path: str, *, doc_id: str, version: int):
        chunks = []
        occurrences: dict[str, int] = {}
        for index, line in enumerate(Path(file_path).read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            digest = hashlib.sha256(line.strip().encode()).hexdigest()
            occurrence = occurrences.get(digest, 0)
            occurrences[digest] = occurrence + 1
            identifier = hashlib.sha256(f"{doc_id}:line:{digest}:{occurrence}".encode()).hexdigest()[:24]
            chunks.append(
                DocumentChunk(
                    line,
                    doc_id,
                    index,
                    DocType.TEXT,
                    {"source": file_path, "file_name": Path(file_path).name, "version": version},
                    identifier=identifier,
                    content_hash=digest,
                )
            )
        return chunks


class EmptyExtractor:
    async def extract(self, chunks: list[DocumentChunk]):
        return [ExtractionResult(source_chunk_id=chunk.chunk_id) for chunk in chunks]


class MemoryVector:
    def __init__(self) -> None:
        self.ids: dict[str, str] = {}
        self.added: list[list[str]] = []

    async def add_chunks(self, chunks: list[DocumentChunk]):
        self.added.append([chunk.chunk_id for chunk in chunks])
        self.ids.update({chunk.chunk_id: chunk.doc_id for chunk in chunks})
        return len(chunks)

    async def delete_chunks(self, ids: list[str]):
        for identifier in ids:
            self.ids.pop(identifier, None)
        return len(ids)

    async def delete_by_doc_id(self, doc_id: str):
        ids = [identifier for identifier, value in self.ids.items() if value == doc_id]
        await self.delete_chunks(ids)
        return len(ids)

    async def get_document_chunks(self, doc_id: str):
        return [identifier for identifier, value in self.ids.items() if value == doc_id]


class MemoryGraph(MemoryVector):
    async def upsert_extraction(self, _extraction: Any, chunk: DocumentChunk, *, version: int):
        self.ids[chunk.chunk_id] = chunk.doc_id
        return 0, 0


@pytest.mark.asyncio
async def test_delta_ingestion_preserves_unchanged_chunks(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("stable-a\nstable-b\nold-c\n", encoding="utf-8")
    catalog = CatalogService(str(tmp_path / "catalog.sqlite3"))
    await catalog.init()
    vector, graph = MemoryVector(), MemoryGraph()
    ingestion = IngestionService(LineParser(), EmptyExtractor(), vector, graph, catalog)
    first = await ingestion.ingest(str(path), doc_id="doc")
    first_ids = set(vector.ids)
    assert first.chunks_added == 3

    path.write_text("stable-a\nstable-b\nnew-c\n", encoding="utf-8")
    second = await ingestion.ingest(str(path), doc_id="doc")
    assert second.chunks_added == 1
    assert second.chunks_deleted == 1
    assert second.chunks_unchanged == 2
    assert len(first_ids & set(vector.ids)) == 2
    assert set(vector.ids) == set(graph.ids)

    unchanged = await ingestion.ingest(str(path), doc_id="doc")
    assert unchanged.status == "UNCHANGED"
    assert unchanged.chunks_unchanged == 3

    rebuilt = await ingestion.rebuild("doc")
    assert rebuilt.version == 3
    graph.ids.clear()
    repaired = await ingestion.repair_consistency("doc")
    assert repaired["repaired"] == ["doc"]
    assert await ingestion.delete("doc") == 1
    assert await ingestion.delete("missing") == 0
    with pytest.raises(KeyError):
        await ingestion.rebuild("missing")
    await catalog.close()


@pytest.mark.asyncio
async def test_cdc_event_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("one\ntwo\n", encoding="utf-8")
    catalog = CatalogService(str(tmp_path / "catalog.sqlite3"))
    await catalog.init()
    ingestion = IngestionService(LineParser(), EmptyExtractor(), MemoryVector(), MemoryGraph(), catalog)
    agent = KnowledgeUpdateAgent(ingestion, catalog)
    event = CDCEvent("event-1", "INSERT", "doc", str(path))
    first = await agent.process_event(event)
    second = await agent.process_event(event)
    assert first.success
    assert second.success and second.duplicate
    assert catalog.get_event("event-1").attempts == 1
    await catalog.close()


class AlwaysFails:
    attempts = 0

    async def ingest(self, *_args, **_kwargs):
        self.attempts += 1
        raise RuntimeError("planned failure")


@pytest.mark.asyncio
async def test_cdc_retries_at_most_three_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async def no_wait(_seconds: float) -> None:
        return None

    monkeypatch.setattr("agents.knowledge_update_agent.asyncio.sleep", no_wait)
    catalog = CatalogService(str(tmp_path / "catalog.sqlite3"))
    await catalog.init()
    ingestion = AlwaysFails()
    agent = KnowledgeUpdateAgent(ingestion, catalog)  # type: ignore[arg-type]
    result = await agent.process_event(CDCEvent("event-fail", "UPDATE", "doc", str(tmp_path / "x")))
    assert not result.success
    assert ingestion.attempts == 3
    assert catalog.get_event("event-fail").attempts == 3
    await catalog.close()


@pytest.mark.asyncio
async def test_change_detection_batch_and_delete_event(tmp_path: Path) -> None:
    known = tmp_path / "known.txt"
    created = tmp_path / "created.txt"
    known.write_text("old", encoding="utf-8")
    created.write_text("new", encoding="utf-8")
    catalog = CatalogService(str(tmp_path / "catalog.sqlite3"))
    await catalog.init()
    vector, graph = MemoryVector(), MemoryGraph()
    ingestion = IngestionService(LineParser(), EmptyExtractor(), vector, graph, catalog)
    await ingestion.ingest(str(known), doc_id="known")
    known.write_text("changed", encoding="utf-8")
    agent = KnowledgeUpdateAgent(ingestion, catalog)
    changes = agent.detect_changes([str(known), str(created)])
    assert {change.change_type for change in changes} == {ChangeType.CREATED, ChangeType.MODIFIED}
    results = await agent.process_batch([DocumentChange(str(created), ChangeType.CREATED, doc_id="created")])
    assert results[0].success
    deleted = await agent.process_event(CDCEvent("delete", "DELETE", "created", str(created)))
    assert deleted.success and deleted.vectors_deleted == 1
    assert agent._file_hash(str(tmp_path / "missing")) == ""
    await catalog.close()


class FailingGraph(MemoryGraph):
    async def upsert_extraction(self, _extraction: Any, chunk: DocumentChunk, *, version: int):
        raise RuntimeError("graph failed")


@pytest.mark.asyncio
async def test_ingestion_failure_rolls_back_new_vectors(tmp_path: Path) -> None:
    path = tmp_path / "document.txt"
    path.write_text("one", encoding="utf-8")
    catalog = CatalogService(str(tmp_path / "catalog.sqlite3"))
    await catalog.init()
    vector = MemoryVector()
    ingestion = IngestionService(LineParser(), EmptyExtractor(), vector, FailingGraph(), catalog)
    with pytest.raises(RuntimeError, match="graph failed"):
        await ingestion.ingest(str(path), doc_id="doc")
    assert vector.ids == {}
    assert catalog.get_document("doc").status == "FAILED"
    await catalog.close()
