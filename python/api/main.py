"""FastAPI entry point for AgentKnowledgeHub."""

from __future__ import annotations

import asyncio
import contextlib
import shutil
import uuid
import zipfile
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from agents.doc_parser_agent import DocParserAgent, DocumentParseError
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from agents.knowledge_update_agent import CDCEvent, KnowledgeUpdateAgent
from agents.qa_agent import QAAgent
from config import settings
from orchestrator.graph import build_supervisor_workflow, build_workflows
from services.catalog import CatalogService, EventRecord
from services.ingestion import IngestionService
from services.knowledge_graph import KnowledgeGraphService
from services.reranker import build_reranker
from services.vector_store import VectorStoreService
from services.vision import build_vision_client


class AppContainer:
    def __init__(self) -> None:
        self.catalog = CatalogService()
        self.vector_store = VectorStoreService()
        self.knowledge_graph = KnowledgeGraphService()
        self.extractor = KnowledgeExtractAgent()
        self.reranker = build_reranker()
        self.parser = DocParserAgent(vision_model=build_vision_client())
        self.ingestion = IngestionService(
            self.parser,
            self.extractor,
            self.vector_store,
            self.knowledge_graph,
            self.catalog,
        )
        self.qa_agent = QAAgent(
            vector_store=self.vector_store,
            knowledge_graph=self.knowledge_graph,
            llm=self.extractor.llm,
            reranker=self.reranker,
        )
        self.update_agent = KnowledgeUpdateAgent(self.ingestion, self.catalog)
        self.workflows: dict[str, Any] = {}
        self.supervisor_workflow: Any | None = None
        self.kafka_task: asyncio.Task[Any] | None = None
        self.checkpointer_context: Any | None = None

    async def init(self) -> None:
        settings.ensure_runtime_dirs()
        await self.catalog.init()
        await self.vector_store.init()
        await self.knowledge_graph.init()
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        self.checkpointer_context = AsyncSqliteSaver.from_conn_string(settings.checkpoint_path)
        checkpointer = await self.checkpointer_context.__aenter__()
        self.workflows = build_workflows(
            self.ingestion,
            self.qa_agent,
            self.update_agent,
            checkpointer=checkpointer,
        )
        self.supervisor_workflow = build_supervisor_workflow(
            self.workflows,
            checkpointer=checkpointer,
        )
        self.update_agent.workflow = self.workflows["update"]
        if settings.enable_kafka_consumer:
            self.kafka_task = asyncio.create_task(
                self.update_agent.start_kafka_consumer(),
                name="cdc-kafka-consumer",
            )

    async def close(self) -> None:
        self.update_agent.stop_watching()
        if self.kafka_task:
            self.kafka_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.kafka_task
        if self.checkpointer_context:
            await self.checkpointer_context.__aexit__(None, None, None)
            self.checkpointer_context = None
        if self.reranker is not None:
            await self.reranker.close()
        await self.knowledge_graph.close()
        await self.catalog.close()


container = AppContainer()


@asynccontextmanager
async def lifespan(app: FastAPI):
    await container.init()
    app.state.container = container
    try:
        yield
    finally:
        await container.close()


app = FastAPI(
    title="AgentKnowledgeHub",
    description="多格式文档解析、GraphRAG 问答与 CDC 增量更新 API",
    version="0.2.0",
    lifespan=lifespan,
)


class CitationResponse(BaseModel):
    doc_id: str
    chunk_id: str
    file_name: str
    page: int | None = None
    sheet: str | None = None
    quote: str
    score: float
    retrieval_type: str
    modality: str = "text"


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    top_k: int = Field(default=settings.default_top_k, ge=1, le=settings.max_top_k)
    mode: Literal["vector", "graph", "hybrid"] = "hybrid"
    document_ids: list[str] | None = None


class QuestionResponse(BaseModel):
    question: str
    answer: str
    answerable: bool
    evidence_score: float
    intent: str
    citations: list[CitationResponse]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    usage: dict[str, int]
    trace: list[str]


class IngestResponse(BaseModel):
    doc_id: str
    file_name: str
    version: int
    status: str
    chunks_total: int
    chunks_added: int
    chunks_deleted: int
    chunks_unchanged: int
    entities_added: int
    relations_added: int
    processing_time_ms: float
    partial_failures: list[str]
    modality_counts: dict[str, int] = Field(default_factory=dict)
    vision_fallbacks: list[str] = Field(default_factory=list)


class UpdateRequest(BaseModel):
    doc_id: str = Field(min_length=1)
    change_type: Literal["modified", "deleted"] = "modified"


class EventResponse(BaseModel):
    event_id: str
    doc_id: str
    status: str
    attempts: int = 0
    error: str = ""


class BatchIngestItem(BaseModel):
    file_name: str
    success: bool
    result: IngestResponse | None = None
    error: str = ""


ALLOWED_CONTENT_TYPES = {
    ".pdf": {"application/pdf", "application/octet-stream"},
    ".docx": {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/zip",
        "application/octet-stream",
    },
    ".xlsx": {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/zip",
        "application/octet-stream",
    },
    ".csv": {"text/csv", "application/csv", "text/plain", "application/octet-stream"},
    ".png": {"image/png", "application/octet-stream"},
    ".jpg": {"image/jpeg", "application/octet-stream"},
    ".jpeg": {"image/jpeg", "application/octet-stream"},
    ".txt": {"text/plain", "application/octet-stream"},
    ".md": {"text/markdown", "text/plain", "application/octet-stream"},
    ".pptx": {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/zip",
        "application/octet-stream",
    },
    ".html": {"text/html", "text/plain", "application/octet-stream"},
    ".htm": {"text/html", "text/plain", "application/octet-stream"},
    ".json": {"application/json", "text/json", "text/plain", "application/octet-stream"},
    ".jsonl": {
        "application/jsonl",
        "application/x-ndjson",
        "application/json",
        "text/plain",
        "application/octet-stream",
    },
    ".xml": {"application/xml", "text/xml", "text/plain", "application/octet-stream"},
}


def _safe_file_name(raw: str | None) -> str:
    if not raw:
        raise HTTPException(status_code=400, detail="file name is required")
    normalized = raw.replace("\\", "/")
    name = Path(normalized).name
    if name != normalized or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="unsafe file name")
    extension = Path(name).suffix.lower()
    if extension not in DocParserAgent.SUPPORTED_EXTENSIONS:
        raise HTTPException(status_code=415, detail=f"unsupported extension: {extension or '<none>'}")
    return name


async def _save_upload(file: UploadFile, doc_id: str, file_name: str) -> str:
    extension = Path(file_name).suffix.lower()
    content_type = (file.content_type or "application/octet-stream").split(";", 1)[0].lower()
    if content_type not in ALLOWED_CONTENT_TYPES[extension]:
        raise HTTPException(status_code=415, detail=f"content type {content_type} does not match {extension}")
    directory = Path(settings.upload_dir).resolve() / doc_id
    directory.mkdir(parents=True, exist_ok=False)
    destination = directory / file_name
    limit = settings.max_upload_mb * 1024 * 1024
    total = 0
    try:
        with destination.open("wb") as handle:
            while block := await file.read(1024 * 1024):
                total += len(block)
                if total > limit:
                    raise HTTPException(status_code=413, detail=f"file exceeds {settings.max_upload_mb}MB")
                handle.write(block)
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    finally:
        await file.close()
    if total == 0:
        shutil.rmtree(directory, ignore_errors=True)
        raise HTTPException(status_code=400, detail="empty file")
    try:
        _validate_file_signature(destination, extension)
    except HTTPException:
        shutil.rmtree(directory, ignore_errors=True)
        raise
    return str(destination)


def _validate_file_signature(path: Path, extension: str) -> None:
    """Reject obvious extension spoofing without trusting the Content-Type header."""
    head = path.read_bytes()[:16]
    if extension == ".pdf" and not head.startswith(b"%PDF-"):
        raise HTTPException(status_code=415, detail="file signature does not match PDF")
    if extension == ".png" and not head.startswith(b"\x89PNG\r\n\x1a\n"):
        raise HTTPException(status_code=415, detail="file signature does not match PNG")
    if extension in {".jpg", ".jpeg"} and not head.startswith(b"\xff\xd8\xff"):
        raise HTTPException(status_code=415, detail="file signature does not match JPEG")
    if extension in {".docx", ".xlsx", ".pptx"}:
        if not zipfile.is_zipfile(path):
            raise HTTPException(status_code=415, detail=f"file signature does not match {extension}")
        with zipfile.ZipFile(path) as archive:
            _validate_archive_safety(archive)
            names = set(archive.namelist())
        required_prefix = {".docx": "word/", ".xlsx": "xl/", ".pptx": "ppt/"}[extension]
        if not any(name.startswith(required_prefix) for name in names):
            raise HTTPException(status_code=415, detail=f"archive content does not match {extension}")
    if extension in {".txt", ".md", ".csv", ".html", ".htm", ".json", ".jsonl", ".xml"} and b"\x00" in head:
        raise HTTPException(status_code=415, detail="text-like file contains binary data")


def _validate_archive_safety(archive: zipfile.ZipFile) -> None:
    infos = archive.infolist()
    if len(infos) > settings.max_archive_files:
        raise HTTPException(status_code=413, detail="archive file-count limit exceeded")
    total = 0
    for info in infos:
        normalized = info.filename.replace("\\", "/")
        if normalized.startswith("/") or ".." in Path(normalized).parts:
            raise HTTPException(status_code=415, detail="unsafe archive member path")
        if info.flag_bits & 0x1:
            raise HTTPException(status_code=415, detail="encrypted archives are not supported")
        total += info.file_size
        if total > settings.max_archive_uncompressed_mb * 1024 * 1024:
            raise HTTPException(status_code=413, detail="archive uncompressed-size limit exceeded")


def _remove_managed_upload(doc_id: str, file_path: str) -> None:
    """Delete only the exact per-document directory managed by this API."""
    upload_root = Path(settings.upload_dir).resolve()
    expected = (upload_root / doc_id).resolve()
    actual = Path(file_path).resolve().parent
    if actual == expected and expected.parent == upload_root:
        shutil.rmtree(expected, ignore_errors=True)


async def _ingest_upload(file: UploadFile) -> IngestResponse:
    file_name = _safe_file_name(file.filename)
    doc_id = uuid.uuid4().hex
    path = await _save_upload(file, doc_id, file_name)
    try:
        result = await container.workflows["ingest"].ainvoke(
            {
                "file_paths": [path],
                "doc_ids": [doc_id],
                "mime_types": [file.content_type or ""],
            },
            config={"configurable": {"thread_id": f"ingest:{doc_id}"}},
        )
        if result.get("error"):
            if result.get("error_type") in {
                "DocumentParseError",
                "UnsupportedDocumentError",
                "ValueError",
                "FileNotFoundError",
            }:
                raise DocumentParseError(str(result["error"]))
            raise RuntimeError(str(result["error"]))
        return IngestResponse(**asdict(result["results"][0]))
    except DocumentParseError as exc:
        shutil.rmtree(Path(path).parent, ignore_errors=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail="ingestion failed; inspect server diagnostics") from exc


@app.post("/api/ingest/upload", response_model=IngestResponse, tags=["文档入库"])
async def upload_document(file: UploadFile = File(...)) -> IngestResponse:
    return await _ingest_upload(file)


@app.post("/api/ingest/batch", response_model=list[BatchIngestItem], tags=["文档入库"])
async def upload_batch(files: list[UploadFile] = File(...)) -> list[BatchIngestItem]:
    semaphore = asyncio.Semaphore(settings.batch_ingest_concurrency)

    async def limited(file: UploadFile) -> BatchIngestItem:
        file_name = file.filename or ""
        async with semaphore:
            try:
                result = await _ingest_upload(file)
                return BatchIngestItem(file_name=file_name, success=True, result=result)
            except HTTPException as exc:
                return BatchIngestItem(file_name=file_name, success=False, error=str(exc.detail))

    return list(await asyncio.gather(*(limited(file) for file in files)))


@app.post("/api/qa/ask", response_model=QuestionResponse, tags=["智能问答"])
async def ask_question(request: QuestionRequest) -> QuestionResponse:
    try:
        state = await container.workflows["qa"].ainvoke(
            request.model_dump(),
            config={"configurable": {"thread_id": f"qa:{uuid.uuid4().hex}"}},
        )
        result = state["result"]
        return QuestionResponse(
            question=result.question,
            answer=result.answer,
            answerable=result.answerable,
            evidence_score=result.evidence_score,
            intent=result.intent.value,
            citations=[CitationResponse(**asdict(citation)) for citation in result.citations],
            retrieval_ms=result.retrieval_ms,
            generation_ms=result.generation_ms,
            total_ms=result.total_ms,
            usage=result.usage,
            trace=result.trace,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail="question answering failed; inspect server diagnostics",
        ) from exc


@app.post(
    "/api/admin/update",
    response_model=EventResponse,
    status_code=status.HTTP_202_ACCEPTED,
    tags=["系统管理"],
)
async def trigger_update(request: UpdateRequest) -> EventResponse:
    record = container.catalog.get_document(request.doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    operation = "DELETE" if request.change_type == "deleted" else "UPDATE"
    event = CDCEvent.create(operation, record.file_path, doc_id=record.doc_id)
    pending = EventRecord(
        event_id=event.event_id,
        operation=event.operation,
        doc_id=event.doc_id,
        file_path=event.file_path,
        observed_hash=event.observed_hash,
    )
    container.catalog.enqueue_event(pending)
    try:
        await asyncio.to_thread(container.update_agent.publish_event, event)
    except Exception as exc:
        container.catalog.finish_event(event.event_id, False, str(exc))
        raise HTTPException(status_code=503, detail="unable to publish update event") from exc
    return EventResponse(event_id=event.event_id, doc_id=event.doc_id, status="PENDING")


@app.get("/api/admin/events/{event_id}", response_model=EventResponse, tags=["系统管理"])
async def get_event(event_id: str) -> EventResponse:
    event = container.catalog.get_event(event_id)
    if event is None:
        raise HTTPException(status_code=404, detail="event not found")
    return EventResponse(
        event_id=event.event_id,
        doc_id=event.doc_id,
        status=event.status,
        attempts=event.attempts,
        error=event.error,
    )


@app.get("/api/documents/{doc_id}", tags=["文档管理"])
async def get_document(doc_id: str) -> dict[str, Any]:
    record = container.catalog.get_document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    return {**asdict(record), "chunks": len(container.catalog.get_chunks(doc_id))}


@app.delete("/api/documents/{doc_id}", tags=["文档管理"])
async def delete_document(doc_id: str) -> dict[str, Any]:
    record = container.catalog.get_document(doc_id)
    if record is None:
        raise HTTPException(status_code=404, detail="document not found")
    deleted = await container.ingestion.delete(doc_id)
    _remove_managed_upload(doc_id, record.file_path)
    return {"doc_id": doc_id, "deleted": bool(deleted)}


@app.get("/api/admin/stats", tags=["系统管理"])
async def get_stats() -> dict[str, Any]:
    vector, graph = await asyncio.gather(
        container.vector_store.get_stats(),
        container.knowledge_graph.get_stats(),
    )
    model_usage = getattr(container.extractor.llm, "usage_snapshot", lambda: {})()
    reranker_usage = getattr(getattr(container, "reranker", None), "usage_snapshot", lambda: {})()
    return {
        "catalog": container.catalog.get_stats(),
        "vector_store": vector,
        "knowledge_graph": graph,
        "model_usage": model_usage,
        "reranker": reranker_usage,
    }


@app.get("/api/health/live", tags=["系统管理"])
async def live() -> dict[str, str]:
    return {"status": "ok", "service": "AgentKnowledgeHub"}


async def _kafka_health() -> bool:
    try:
        from confluent_kafka.admin import AdminClient

        client = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
        await asyncio.to_thread(client.list_topics, timeout=2)
        return True
    except Exception:
        return False


@app.get("/api/health/ready", tags=["系统管理"])
async def ready() -> dict[str, Any]:
    vector_ok, graph_ok, kafka_ok = await asyncio.gather(
        container.vector_store.health(),
        container.knowledge_graph.health(),
        _kafka_health(),
    )
    checks = {
        "catalog": True,
        "chroma": vector_ok,
        "neo4j": graph_ok,
        "kafka": kafka_ok,
        "llm_configured": settings.llm_configured,
        "embedding_configured": settings.embedding_configured,
        "vision_configured": (not settings.vision_enabled) or settings.vision_configured,
        "vl_embedding_configured": settings.vl_embedding_configured,
        "reranker_configured": settings.reranker_configured,
    }
    if not all(checks.values()):
        raise HTTPException(status_code=503, detail={"status": "not_ready", "checks": checks})
    return {"status": "ready", "checks": checks}


@app.get("/api/health", tags=["系统管理"])
async def health() -> dict[str, Any]:
    return await ready()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:app", host=settings.api_host, port=settings.api_port, reload=True)
