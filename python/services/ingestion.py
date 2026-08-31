"""Transactional-ish ingestion core shared by upload, rebuild and CDC paths."""

from __future__ import annotations

import asyncio
import hashlib
import mimetypes
import os
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field

from agents.doc_parser_agent import DocParserAgent, DocumentChunk
from agents.knowledge_extract_agent import ExtractionResult, KnowledgeExtractAgent
from services.catalog import CatalogService, DocumentRecord
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService


@dataclass(slots=True)
class IngestResult:
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
    partial_failures: list[str] = field(default_factory=list)
    modality_counts: dict[str, int] = field(default_factory=dict)
    vision_fallbacks: list[str] = field(default_factory=list)


@dataclass(slots=True)
class IngestTransaction:
    file_path: str
    doc_id: str
    mime_type: str
    force: bool
    started: float
    content_hash: str
    file_name: str
    version: int
    old_chunks: list[dict[str, object]] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    added_ids: set[str] = field(default_factory=set)
    removed_ids: set[str] = field(default_factory=set)
    unchanged_ids: set[str] = field(default_factory=set)
    added_chunks: list[DocumentChunk] = field(default_factory=list)
    extractions: list[ExtractionResult] = field(default_factory=list)
    entities_added: int = 0
    relations_added: int = 0
    unchanged_result: IngestResult | None = None
    result: IngestResult | None = None
    error: str = ""


class IngestionService:
    def __init__(
        self,
        parser: DocParserAgent,
        extractor: KnowledgeExtractAgent,
        vector_store: VectorStoreService,
        knowledge_graph: KnowledgeGraphService,
        catalog: CatalogService,
    ) -> None:
        self.parser = parser
        self.extractor = extractor
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.catalog = catalog

    def validate_stage(
        self,
        file_path: str,
        *,
        doc_id: str | None = None,
        mime_type: str = "",
        force: bool = False,
    ) -> IngestTransaction:
        started = time.perf_counter()
        absolute_path = os.path.abspath(file_path)
        content_hash = self.file_hash(absolute_path)
        current = (
            self.catalog.get_document(doc_id) if doc_id else self.catalog.get_document_by_path(absolute_path)
        )
        if current is not None:
            doc_id = current.doc_id
            if current.content_hash == content_hash and current.status == "READY" and not force:
                rows = self.catalog.get_chunks(current.doc_id)
                counts = Counter(
                    str(row["metadata"].get("modality") or row["metadata"].get("kind") or "text")
                    for row in rows
                )
                result = IngestResult(
                    doc_id=current.doc_id,
                    file_name=current.file_name,
                    version=current.version,
                    status="UNCHANGED",
                    chunks_total=len(rows),
                    chunks_added=0,
                    chunks_deleted=0,
                    chunks_unchanged=len(rows),
                    entities_added=0,
                    relations_added=0,
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    modality_counts=dict(counts),
                )
                return IngestTransaction(
                    absolute_path,
                    current.doc_id,
                    current.mime_type,
                    force,
                    started,
                    content_hash,
                    current.file_name,
                    current.version,
                    old_chunks=rows,
                    unchanged_result=result,
                    result=result,
                )
        doc_id = doc_id or uuid.uuid4().hex
        file_name = os.path.basename(absolute_path)
        mime_type = mime_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        old_chunks = self.catalog.get_chunks(doc_id) if current else []
        version = self.catalog.begin_document(
            doc_id=doc_id,
            file_name=file_name,
            file_path=absolute_path,
            mime_type=mime_type,
            content_hash=content_hash,
        )
        return IngestTransaction(
            absolute_path,
            doc_id,
            mime_type,
            force,
            started,
            content_hash,
            file_name,
            version,
            old_chunks=old_chunks,
        )

    async def parse_stage(self, transaction: IngestTransaction) -> None:
        if transaction.unchanged_result:
            return
        transaction.chunks = await self.parser.parse(
            transaction.file_path,
            doc_id=transaction.doc_id,
            version=transaction.version,
        )

    def diff_stage(self, transaction: IngestTransaction) -> None:
        if transaction.unchanged_result:
            return
        old_ids = {str(item["chunk_id"]) for item in transaction.old_chunks}
        new_by_id = {chunk.chunk_id: chunk for chunk in transaction.chunks}
        new_ids = set(new_by_id)
        if transaction.force:
            transaction.added_ids = new_ids
            transaction.removed_ids = old_ids
        else:
            transaction.added_ids = new_ids - old_ids
            transaction.removed_ids = old_ids - new_ids
            transaction.unchanged_ids = old_ids & new_ids
        transaction.added_chunks = [new_by_id[value] for value in sorted(transaction.added_ids)]

    async def embed_stage(self, transaction: IngestTransaction) -> None:
        if transaction.unchanged_result:
            return
        if transaction.force:
            await self.vector_store.delete_by_doc_id(transaction.doc_id)
        await self.vector_store.add_chunks(transaction.added_chunks)

    async def extract_stage(self, transaction: IngestTransaction) -> None:
        if transaction.unchanged_result:
            return
        transaction.extractions = await self.extractor.extract(transaction.added_chunks)

    async def graph_upsert_stage(self, transaction: IngestTransaction) -> None:
        if transaction.unchanged_result:
            return
        if transaction.force:
            await self.knowledge_graph.delete_by_doc_id(transaction.doc_id)
        for chunk, extraction in zip(transaction.added_chunks, transaction.extractions, strict=True):
            entities, relations = await self.knowledge_graph.upsert_extraction(
                extraction,
                chunk,
                version=transaction.version,
            )
            transaction.entities_added += entities
            transaction.relations_added += relations

    async def delete_removed_stage(self, transaction: IngestTransaction) -> None:
        if transaction.unchanged_result or transaction.force or not transaction.removed_ids:
            return
        removed = sorted(transaction.removed_ids)
        await self.vector_store.delete_chunks(removed)
        await self.knowledge_graph.delete_chunks(removed)

    def commit_stage(self, transaction: IngestTransaction) -> IngestResult:
        if transaction.unchanged_result:
            return transaction.unchanged_result
        self.catalog.commit_document(
            transaction.doc_id,
            transaction.content_hash,
            transaction.version,
            transaction.chunks,
        )
        asset_paths = {
            str(chunk.metadata.get("asset_path", ""))
            for chunk in transaction.chunks
            if chunk.metadata.get("asset_path")
        }
        if asset_store := getattr(self.parser, "asset_store", None):
            asset_store.prune_document(transaction.doc_id, asset_paths)
        modality_counts = Counter(
            str(chunk.metadata.get("modality") or chunk.metadata.get("kind") or "text")
            for chunk in transaction.chunks
        )
        fallbacks = sorted(
            {
                str(chunk.metadata["vision_fallback"])
                for chunk in transaction.chunks
                if chunk.metadata.get("vision_fallback") not in {None, "", "disabled_or_unconfigured"}
            }
        )
        transaction.result = IngestResult(
            doc_id=transaction.doc_id,
            file_name=transaction.file_name,
            version=transaction.version,
            status="READY",
            chunks_total=len(transaction.chunks),
            chunks_added=len(transaction.added_ids),
            chunks_deleted=len(transaction.removed_ids),
            chunks_unchanged=len(transaction.unchanged_ids),
            entities_added=transaction.entities_added,
            relations_added=transaction.relations_added,
            processing_time_ms=(time.perf_counter() - transaction.started) * 1000,
            partial_failures=[f"vision:{value}" for value in fallbacks],
            modality_counts=dict(modality_counts),
            vision_fallbacks=fallbacks,
        )
        return transaction.result

    async def verify_stage(self, transaction: IngestTransaction) -> bool:
        expected = {chunk.chunk_id for chunk in transaction.chunks}
        if transaction.unchanged_result:
            expected = {str(row["chunk_id"]) for row in transaction.old_chunks}
        vector_ids, graph_ids = await asyncio.gather(
            self.vector_store.get_document_chunks(transaction.doc_id),
            self.knowledge_graph.get_document_chunks(transaction.doc_id),
        )
        return expected == set(vector_ids) == set(graph_ids)

    async def rollback_stage(self, transaction: IngestTransaction, error: BaseException | str) -> None:
        if transaction.added_ids:
            try:
                await self.vector_store.delete_chunks(sorted(transaction.added_ids))
                await self.knowledge_graph.delete_chunks(sorted(transaction.added_ids))
            except BaseException:
                pass
        old_assets = {
            str(row["metadata"].get("asset_path", ""))
            for row in transaction.old_chunks
            if row["metadata"].get("asset_path")
        }
        if asset_store := getattr(self.parser, "asset_store", None):
            asset_store.prune_document(transaction.doc_id, old_assets)
        transaction.error = str(error) or type(error).__name__
        self.catalog.fail_document(transaction.doc_id, transaction.error)

    async def ingest(
        self,
        file_path: str,
        *,
        doc_id: str | None = None,
        mime_type: str = "",
        force: bool = False,
    ) -> IngestResult:
        started = time.perf_counter()
        absolute_path = os.path.abspath(file_path)
        content_hash = self.file_hash(absolute_path)
        current = (
            self.catalog.get_document(doc_id) if doc_id else self.catalog.get_document_by_path(absolute_path)
        )
        if current is not None:
            doc_id = current.doc_id
            if current.content_hash == content_hash and current.status == "READY" and not force:
                chunks = self.catalog.get_chunks(current.doc_id)
                modality_counts = Counter(
                    str(item["metadata"].get("modality") or item["metadata"].get("kind") or "text")
                    for item in chunks
                )
                return IngestResult(
                    doc_id=current.doc_id,
                    file_name=current.file_name,
                    version=current.version,
                    status="UNCHANGED",
                    chunks_total=len(chunks),
                    chunks_added=0,
                    chunks_deleted=0,
                    chunks_unchanged=len(chunks),
                    entities_added=0,
                    relations_added=0,
                    processing_time_ms=(time.perf_counter() - started) * 1000,
                    modality_counts=dict(modality_counts),
                )
        doc_id = doc_id or uuid.uuid4().hex
        file_name = os.path.basename(absolute_path)
        mime_type = mime_type or mimetypes.guess_type(file_name)[0] or "application/octet-stream"
        old_chunks = self.catalog.get_chunks(doc_id) if current else []
        old_asset_paths = {
            str(item["metadata"].get("asset_path", ""))
            for item in old_chunks
            if item["metadata"].get("asset_path")
        }
        version = self.catalog.begin_document(
            doc_id=doc_id,
            file_name=file_name,
            file_path=absolute_path,
            mime_type=mime_type,
            content_hash=content_hash,
        )

        added_to_stores: list[str] = []
        try:
            chunks = await self.parser.parse(absolute_path, doc_id=doc_id, version=version)
            old_ids = {str(item["chunk_id"]) for item in old_chunks}
            new_by_id = {chunk.chunk_id: chunk for chunk in chunks}
            new_ids = set(new_by_id)
            if force:
                await self.vector_store.delete_by_doc_id(doc_id)
                await self.knowledge_graph.delete_by_doc_id(doc_id)
                added_ids, removed_ids, unchanged_ids = new_ids, old_ids, set()
            else:
                added_ids = new_ids - old_ids
                removed_ids = old_ids - new_ids
                unchanged_ids = old_ids & new_ids

            added_chunks = [new_by_id[identifier] for identifier in added_ids]
            if added_chunks:
                await self.vector_store.add_chunks(added_chunks)
                added_to_stores = list(added_ids)
            extractions = await self.extractor.extract(added_chunks)
            entities_added = 0
            relations_added = 0
            for chunk, extraction in zip(added_chunks, extractions, strict=True):
                entities, relations = await self.knowledge_graph.upsert_extraction(
                    extraction,
                    chunk,
                    version=version,
                )
                entities_added += entities
                relations_added += relations

            if removed_ids and not force:
                removed = sorted(removed_ids)
                await self.vector_store.delete_chunks(removed)
                await self.knowledge_graph.delete_chunks(removed)

            self.catalog.commit_document(doc_id, content_hash, version, chunks)
            asset_paths = {
                str(chunk.metadata.get("asset_path", ""))
                for chunk in chunks
                if chunk.metadata.get("asset_path")
            }
            if asset_store := getattr(self.parser, "asset_store", None):
                asset_store.prune_document(doc_id, asset_paths)
            modality_counts = Counter(
                str(chunk.metadata.get("modality") or chunk.metadata.get("kind") or "text")
                for chunk in chunks
            )
            fallbacks = sorted(
                {
                    str(chunk.metadata["vision_fallback"])
                    for chunk in chunks
                    if chunk.metadata.get("vision_fallback") not in {None, "", "disabled_or_unconfigured"}
                }
            )
            return IngestResult(
                doc_id=doc_id,
                file_name=file_name,
                version=version,
                status="READY",
                chunks_total=len(chunks),
                chunks_added=len(added_ids),
                chunks_deleted=len(removed_ids),
                chunks_unchanged=len(unchanged_ids),
                entities_added=entities_added,
                relations_added=relations_added,
                processing_time_ms=(time.perf_counter() - started) * 1000,
                partial_failures=[f"vision:{value}" for value in fallbacks],
                modality_counts=dict(modality_counts),
                vision_fallbacks=fallbacks,
            )
        except BaseException as exc:
            # Newly added records are safe to remove; old catalog records remain until commit.
            if added_to_stores:
                try:
                    await self.vector_store.delete_chunks(added_to_stores)
                    await self.knowledge_graph.delete_chunks(added_to_stores)
                except BaseException:
                    pass
            error = str(exc) or exc.__class__.__name__
            if asset_store := getattr(self.parser, "asset_store", None):
                asset_store.prune_document(doc_id, old_asset_paths)
            self.catalog.fail_document(doc_id, error)
            raise

    async def delete(self, doc_id: str) -> int:
        record = self.catalog.get_document(doc_id)
        if record is None:
            return 0
        await self.vector_store.delete_by_doc_id(doc_id)
        await self.knowledge_graph.delete_by_doc_id(doc_id)
        deleted = self.catalog.delete_document(doc_id)
        if asset_store := getattr(self.parser, "asset_store", None):
            asset_store.delete_document(doc_id)
        return deleted

    async def rebuild(self, doc_id: str) -> IngestResult:
        record = self._require_document(doc_id)
        return await self.ingest(record.file_path, doc_id=doc_id, mime_type=record.mime_type, force=True)

    async def repair_consistency(self, doc_id: str | None = None) -> dict[str, object]:
        """Rebuild documents whose catalog, Chroma and Neo4j chunk IDs diverge."""
        documents = (
            [self._require_document(doc_id)] if doc_id else self.catalog.list_documents(include_failed=False)
        )
        checked = 0
        repaired: list[str] = []
        failures: dict[str, str] = {}
        for document in documents:
            checked += 1
            expected = {str(row["chunk_id"]) for row in self.catalog.get_chunks(document.doc_id)}
            vector_ids, graph_ids = await asyncio.gather(
                self.vector_store.get_document_chunks(document.doc_id),
                self.knowledge_graph.get_document_chunks(document.doc_id),
            )
            if expected == set(vector_ids) == set(graph_ids):
                continue
            try:
                await self.rebuild(document.doc_id)
                repaired.append(document.doc_id)
            except Exception as exc:
                failures[document.doc_id] = str(exc)
        return {
            "checked": checked,
            "repaired": repaired,
            "failures": failures,
            "consistent": not failures,
        }

    def _require_document(self, doc_id: str) -> DocumentRecord:
        record = self.catalog.get_document(doc_id)
        if record is None:
            raise KeyError(f"unknown document: {doc_id}")
        return record

    @staticmethod
    def file_hash(file_path: str) -> str:
        digest = hashlib.sha256()
        with open(file_path, "rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()
