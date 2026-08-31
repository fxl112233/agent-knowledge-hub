"""Idempotent CDC agent shared by API, Watchdog and Kafka consumers."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any

from config import settings
from services.catalog import CatalogService, EventRecord
from services.ingestion import IngestionService


class ChangeType(StrEnum):
    CREATED = "created"
    MODIFIED = "modified"
    DELETED = "deleted"


@dataclass(slots=True)
class DocumentChange:
    file_path: str
    change_type: ChangeType
    doc_id: str = ""
    timestamp: float = field(default_factory=time.time)
    old_hash: str = ""
    new_hash: str = ""


@dataclass(slots=True)
class CDCEvent:
    event_id: str
    operation: str
    doc_id: str
    file_path: str
    observed_hash: str = ""
    timestamp: float = field(default_factory=time.time)

    @classmethod
    def create(
        cls,
        operation: str,
        file_path: str,
        *,
        doc_id: str = "",
        observed_hash: str = "",
    ) -> CDCEvent:
        return cls(
            event_id=uuid.uuid4().hex,
            operation=operation.upper(),
            doc_id=doc_id,
            file_path=os.path.abspath(file_path),
            observed_hash=observed_hash,
        )


@dataclass(slots=True)
class UpdateResult:
    event_id: str
    doc_id: str
    file_path: str
    status: str
    version: int = 0
    vectors_added: int = 0
    vectors_deleted: int = 0
    chunks_unchanged: int = 0
    entities_added: int = 0
    relations_added: int = 0
    success: bool = True
    error: str = ""
    processing_time_ms: float = 0.0
    duplicate: bool = False


class KnowledgeUpdateAgent:
    def __init__(self, ingestion: IngestionService, catalog: CatalogService) -> None:
        self.ingestion = ingestion
        self.catalog = catalog
        self._observer: Any | None = None
        self._last_seen: dict[str, float] = {}
        self.workflow: Any | None = None

    async def process_event(self, event: CDCEvent) -> UpdateResult:
        started = time.perf_counter()
        record = EventRecord(
            event_id=event.event_id,
            operation=event.operation,
            doc_id=event.doc_id,
            file_path=event.file_path,
            observed_hash=event.observed_hash,
        )
        existing = self.catalog.get_event(event.event_id)
        if existing and existing.status == "COMMITTED":
            return UpdateResult(
                event_id=event.event_id,
                doc_id=existing.doc_id or event.doc_id,
                file_path=event.file_path,
                status="COMMITTED",
                duplicate=True,
                processing_time_ms=(time.perf_counter() - started) * 1000,
            )
        result: UpdateResult | None = None
        while self.catalog.claim_event(record):
            try:
                if event.operation == "DELETE":
                    doc_id = event.doc_id
                    if not doc_id:
                        current = self.catalog.get_document_by_path(os.path.abspath(event.file_path))
                        doc_id = current.doc_id if current else ""
                    chunk_count = len(self.catalog.get_chunks(doc_id)) if doc_id else 0
                    if doc_id:
                        await self.ingestion.delete(doc_id)
                    result = UpdateResult(
                        event_id=event.event_id,
                        doc_id=doc_id,
                        file_path=event.file_path,
                        status="COMMITTED",
                        vectors_deleted=chunk_count,
                    )
                else:
                    ingested = await self.ingestion.ingest(event.file_path, doc_id=event.doc_id or None)
                    result = UpdateResult(
                        event_id=event.event_id,
                        doc_id=ingested.doc_id,
                        file_path=event.file_path,
                        status="COMMITTED",
                        version=ingested.version,
                        vectors_added=ingested.chunks_added,
                        vectors_deleted=ingested.chunks_deleted,
                        chunks_unchanged=ingested.chunks_unchanged,
                        entities_added=ingested.entities_added,
                        relations_added=ingested.relations_added,
                    )
                self.catalog.finish_event(event.event_id, True)
                break
            except Exception as exc:
                self.catalog.finish_event(event.event_id, False, str(exc))
                result = UpdateResult(
                    event_id=event.event_id,
                    doc_id=event.doc_id,
                    file_path=event.file_path,
                    status="FAILED",
                    success=False,
                    error=str(exc),
                )
                current_event = self.catalog.get_event(event.event_id)
                if current_event and current_event.attempts < settings.cdc_max_retries:
                    await asyncio.sleep(min(2 ** (current_event.attempts - 1), 4))
                    continue
                break
        if result is None:
            current_event = self.catalog.get_event(event.event_id)
            result = UpdateResult(
                event_id=event.event_id,
                doc_id=event.doc_id,
                file_path=event.file_path,
                status=current_event.status if current_event else "FAILED",
                success=False,
                error=(current_event.error if current_event else "event could not be claimed"),
                duplicate=bool(current_event and current_event.status == "APPLYING"),
            )
        result.processing_time_ms = (time.perf_counter() - started) * 1000
        return result

    async def process_change(self, change: DocumentChange) -> UpdateResult:
        operation = {
            ChangeType.CREATED: "INSERT",
            ChangeType.MODIFIED: "UPDATE",
            ChangeType.DELETED: "DELETE",
        }[change.change_type]
        return await self.process_event(
            CDCEvent.create(
                operation,
                change.file_path,
                doc_id=change.doc_id,
                observed_hash=change.new_hash,
            )
        )

    async def process_batch(self, changes: list[DocumentChange]) -> list[UpdateResult]:
        return [await self.process_change(change) for change in changes]

    def detect_changes(self, file_paths: list[str]) -> list[DocumentChange]:
        changes: list[DocumentChange] = []
        for file_path in file_paths:
            absolute = os.path.abspath(file_path)
            current = self.catalog.get_document_by_path(absolute)
            new_hash = self._file_hash(absolute)
            if current is None:
                changes.append(DocumentChange(absolute, ChangeType.CREATED, new_hash=new_hash))
            elif current.content_hash != new_hash:
                changes.append(
                    DocumentChange(
                        absolute,
                        ChangeType.MODIFIED,
                        doc_id=current.doc_id,
                        old_hash=current.content_hash,
                        new_hash=new_hash,
                    )
                )
        return changes

    def publish_event(self, event: CDCEvent) -> None:
        from confluent_kafka import Producer

        producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
        producer.produce(
            settings.kafka_topic_doc_changes,
            key=event.event_id,
            value=json.dumps(asdict(event), ensure_ascii=False).encode("utf-8"),
        )
        producer.flush(10)

    def start_watching(self, directory: str) -> None:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        agent = self

        class Handler(FileSystemEventHandler):
            def _publish(self, path: str, operation: str) -> None:
                now = time.monotonic()
                absolute = os.path.abspath(path)
                if now - agent._last_seen.get(absolute, 0.0) < settings.cdc_debounce_seconds:
                    return
                agent._last_seen[absolute] = now
                current = agent.catalog.get_document_by_path(absolute)
                doc_id = current.doc_id if current else ""
                observed_hash = agent._file_hash(absolute) if operation != "DELETE" else ""
                agent.publish_event(
                    CDCEvent.create(operation, absolute, doc_id=doc_id, observed_hash=observed_hash)
                )

            def on_created(self, event: Any) -> None:
                if not event.is_directory:
                    self._publish(event.src_path, "INSERT")

            def on_modified(self, event: Any) -> None:
                if not event.is_directory:
                    self._publish(event.src_path, "UPDATE")

            def on_deleted(self, event: Any) -> None:
                if not event.is_directory:
                    self._publish(event.src_path, "DELETE")

        observer = Observer()
        observer.schedule(Handler(), directory, recursive=True)
        observer.start()
        self._observer = observer

    def stop_watching(self) -> None:
        if self._observer:
            self._observer.stop()
            self._observer.join(timeout=5)
            self._observer = None

    async def start_kafka_consumer(self) -> None:
        from confluent_kafka import Consumer

        consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap_servers,
                "group.id": settings.kafka_consumer_group,
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )
        consumer.subscribe([settings.kafka_topic_doc_changes])
        try:
            while True:
                message = await asyncio.to_thread(consumer.poll, 1.0)
                if message is None or message.error():
                    await asyncio.sleep(0)
                    continue
                data = json.loads(message.value().decode("utf-8"))
                event = CDCEvent(**data)
                if self.workflow is not None:
                    state = await self.workflow.ainvoke(
                        {"events": [event]},
                        config={"configurable": {"thread_id": f"cdc:{event.event_id}"}},
                    )
                    results = state.get("results", [])
                    result = (
                        results[0]
                        if results
                        else UpdateResult(
                            event.event_id,
                            event.doc_id,
                            event.file_path,
                            "FAILED",
                            success=False,
                            error=str(state.get("error", "workflow returned no result")),
                        )
                    )
                else:
                    result = await self.process_event(event)
                if result.success:
                    consumer.commit(message=message, asynchronous=False)
        finally:
            consumer.close()

    @staticmethod
    def _file_hash(file_path: str) -> str:
        try:
            with open(file_path, "rb") as handle:
                return hashlib.sha256(handle.read()).hexdigest()
        except FileNotFoundError:
            return ""
