from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi import HTTPException
from starlette.datastructures import Headers, UploadFile

from agents.knowledge_update_agent import CDCEvent
from agents.qa_agent import Citation, QAResult, QueryIntent
from api import main as api
from services.catalog import DocumentRecord, EventRecord
from services.ingestion import IngestResult


def upload(name: str, value: bytes, content_type: str) -> UploadFile:
    return UploadFile(
        file=io.BytesIO(value),
        filename=name,
        headers=Headers({"content-type": content_type}),
    )


class FakeWorkflow:
    async def ainvoke(self, state: dict[str, Any], config: dict[str, Any] | None = None):
        assert config and config["configurable"]["thread_id"]
        if "file_paths" in state:
            return {
                "results": [
                    IngestResult(
                        state["doc_ids"][0],
                        Path(state["file_paths"][0]).name,
                        1,
                        "READY",
                        1,
                        1,
                        0,
                        0,
                        0,
                        0,
                        1.5,
                    )
                ]
            }
        result = QAResult(
            state["question"],
            "Atlas belongs to Alice.",
            True,
            0.9,
            QueryIntent.FACTOID,
            [Citation("doc", "chunk", "a.txt", 1, None, "Atlas", 0.9, "hybrid")],
            [],
            10,
            20,
            30,
            {"total_tokens": 5},
            ["answer:generated"],
        )
        return {"result": result}


class FakeCatalog:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.document = DocumentRecord(
            "doc",
            "a.txt",
            str(root / "doc" / "a.txt"),
            "text/plain",
            "hash",
            1,
            "READY",
            "",
            1.0,
            2.0,
        )
        self.event: EventRecord | None = None

    def get_document(self, doc_id: str):
        return self.document if doc_id == "doc" else None

    def get_chunks(self, _doc_id: str):
        return [{"chunk_id": "chunk"}]

    def get_stats(self):
        return {"documents": 1}

    def enqueue_event(self, event: EventRecord):
        self.event = event

    def finish_event(self, event_id: str, success: bool, error: str):
        assert self.event and self.event.event_id == event_id
        self.event.status = "COMMITTED" if success else "FAILED"
        self.event.error = error

    def get_event(self, event_id: str):
        return self.event if self.event and self.event.event_id == event_id else None


class FakeHealth:
    async def health(self):
        return True

    async def get_stats(self):
        return {"ok": 1}


class FakeIngestion:
    async def delete(self, _doc_id: str):
        return 1


class FakeUpdater:
    def publish_event(self, event: CDCEvent):
        assert event.doc_id == "doc"


@pytest.fixture
def fake_container(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    catalog = FakeCatalog(tmp_path)
    value = SimpleNamespace(
        workflows={"ingest": FakeWorkflow(), "qa": FakeWorkflow()},
        catalog=catalog,
        vector_store=FakeHealth(),
        knowledge_graph=FakeHealth(),
        ingestion=FakeIngestion(),
        update_agent=FakeUpdater(),
        extractor=SimpleNamespace(llm=None),
    )
    monkeypatch.setattr(api, "container", value)
    monkeypatch.setattr(api.settings, "upload_dir", str(tmp_path))
    return value


@pytest.mark.asyncio
async def test_upload_batch_and_question_contract(fake_container: Any) -> None:
    response = await api.upload_document(upload("note.txt", b"Atlas documentation", "text/plain"))
    assert response.status == "READY"
    batch = await api.upload_batch(
        [
            upload("one.md", b"# One", "text/markdown"),
            upload("bad.pdf", b"not pdf", "application/pdf"),
        ]
    )
    assert [item.success for item in batch] == [True, False]
    answer = await api.ask_question(api.QuestionRequest(question="Who owns Atlas?"))
    assert answer.answerable
    assert answer.citations[0].chunk_id == "chunk"
    assert answer.usage["total_tokens"] == 5


@pytest.mark.asyncio
async def test_documents_update_events_stats_and_health(
    fake_container: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    record_path = Path(fake_container.catalog.document.file_path)
    record_path.parent.mkdir(parents=True)
    record_path.write_text("content", encoding="utf-8")
    document = await api.get_document("doc")
    assert document["chunks"] == 1
    with pytest.raises(HTTPException):
        await api.get_document("missing")

    event = await api.trigger_update(api.UpdateRequest(doc_id="doc", change_type="modified"))
    assert event.status == "PENDING"
    stored = await api.get_event(event.event_id)
    assert stored.event_id == event.event_id
    with pytest.raises(HTTPException):
        await api.get_event("missing")

    stats = await api.get_stats()
    assert stats["catalog"]["documents"] == 1
    assert (await api.live())["status"] == "ok"

    async def healthy_kafka() -> bool:
        return True

    monkeypatch.setattr(api, "_kafka_health", healthy_kafka)
    monkeypatch.setattr(api.settings, "llm_api_key", "test")
    monkeypatch.setattr(api.settings, "llm_model", "fake")
    monkeypatch.setattr(api.settings, "siliconflow_api_key", "test")
    assert (await api.ready())["status"] == "ready"
    deleted = await api.delete_document("doc")
    assert deleted["deleted"]
    assert not record_path.parent.exists()


@pytest.mark.asyncio
async def test_readiness_failure_and_missing_document(
    fake_container: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def unhealthy_kafka() -> bool:
        return False

    monkeypatch.setattr(api, "_kafka_health", unhealthy_kafka)
    with pytest.raises(HTTPException) as exc:
        await api.ready()
    assert exc.value.status_code == 503
    with pytest.raises(HTTPException):
        await api.delete_document("missing")
    with pytest.raises(HTTPException):
        await api.trigger_update(api.UpdateRequest(doc_id="missing"))


class LifecycleDependency:
    def __init__(self) -> None:
        self.initialized = False
        self.closed = False

    async def init(self):
        self.initialized = True

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_container_initializes_checkpointer_and_closes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    container = api.AppContainer()
    catalog, vector, graph = LifecycleDependency(), LifecycleDependency(), LifecycleDependency()
    container.catalog = catalog  # type: ignore[assignment]
    container.vector_store = vector  # type: ignore[assignment]
    container.knowledge_graph = graph  # type: ignore[assignment]
    monkeypatch.setattr(api.settings, "enable_kafka_consumer", False)
    monkeypatch.setattr(api.settings, "checkpoint_path", str(tmp_path / "checkpoints.sqlite3"))
    monkeypatch.setattr(api.settings, "upload_dir", str(tmp_path / "uploads"))
    monkeypatch.setattr(api.settings, "catalog_path", str(tmp_path / "catalog.sqlite3"))
    await container.init()
    assert container.checkpointer_context is not None
    assert set(container.workflows) == {"ingest", "qa", "update"}
    await container.close()
    assert catalog.closed and graph.closed
