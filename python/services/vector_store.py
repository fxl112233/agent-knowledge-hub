"""Modality-separated Chroma storage with legacy collection compatibility."""

from __future__ import annotations

import asyncio
import base64
import mimetypes
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import httpx

from agents.doc_parser_agent import DocumentChunk
from config import settings


class EmbeddingProvider(Protocol):
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...


class VLEmbeddingProvider(Protocol):
    async def aembed_images(self, paths: list[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...


@dataclass(slots=True)
class EmbeddingUsage:
    calls: int = 0
    input_tokens: int = 0
    inputs: int = 0


@dataclass(slots=True)
class MultimodalQueryVector:
    text: list[float]
    vl: list[float] | None = None


class OpenAICompatibleEmbeddingProvider:
    """Metered OpenAI-compatible text embedding client."""

    def __init__(self) -> None:
        if not settings.embedding_configured:
            raise RuntimeError("embedding provider is not configured")
        from openai import AsyncOpenAI

        self.client = AsyncOpenAI(
            api_key=settings.active_embedding_api_key,
            base_url=settings.active_embedding_base_url,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )
        self.model = settings.active_embedding_model
        self.usage = EmbeddingUsage()
        self._usage_lock = asyncio.Lock()

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        kwargs: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "float"}
        if "Qwen3-Embedding" in self.model:
            kwargs["dimensions"] = settings.embedding_dimensions
        response = await self.client.embeddings.create(**kwargs)
        vectors = [list(item.embedding) for item in sorted(response.data, key=lambda item: item.index)]
        self._validate_dimensions(vectors)
        returned_usage = getattr(response, "usage", None)
        tokens = int(
            getattr(returned_usage, "prompt_tokens", 0) or getattr(returned_usage, "total_tokens", 0) or 0
        )
        await self._record_usage(tokens, len(texts))
        return vectors

    async def aembed_query(self, text: str) -> list[float]:
        vectors = await self.aembed_documents([text])
        return vectors[0]

    async def _record_usage(self, tokens: int, inputs: int) -> None:
        async with self._usage_lock:
            self.usage.calls += 1
            self.usage.input_tokens += tokens
            self.usage.inputs += inputs

    @staticmethod
    def _validate_dimensions(vectors: list[list[float]]) -> None:
        if any(len(vector) != settings.embedding_dimensions for vector in vectors):
            dimensions = sorted({len(vector) for vector in vectors})
            raise RuntimeError(
                f"embedding dimension mismatch: expected {settings.embedding_dimensions}, got {dimensions}"
            )

    def usage_snapshot(self) -> dict[str, int]:
        return {
            "calls": self.usage.calls,
            "input_tokens": self.usage.input_tokens,
            "texts": self.usage.inputs,
        }


class SiliconFlowVLEmbeddingProvider:
    """Raw image and text-query embeddings in the same Qwen3-VL vector space."""

    def __init__(self) -> None:
        if not settings.vl_embedding_configured:
            raise RuntimeError("SiliconFlow VL embedding is not configured")
        self.model = settings.active_vl_embedding_model
        self.usage = EmbeddingUsage()
        self._usage_lock = asyncio.Lock()

    async def aembed_images(self, paths: list[str]) -> list[list[float]]:
        if not paths:
            return []
        inputs: list[dict[str, str]] = []
        for value in paths:
            path = Path(value)
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            inputs.append({"image": f"data:{mime};base64,{encoded}"})
        return await self._embed(inputs)

    async def aembed_query(self, text: str) -> list[float]:
        vectors = await self._embed([{"text": text}])
        return vectors[0]

    async def _embed(self, inputs: list[dict[str, str]]) -> list[list[float]]:
        payload = {
            "model": self.model,
            "input": inputs,
            "encoding_format": "float",
            "dimensions": settings.embedding_dimensions,
            "truncate": "right",
        }
        headers = {"Authorization": f"Bearer {settings.siliconflow_api_key}"}
        async with httpx.AsyncClient(timeout=httpx.Timeout(settings.vision_timeout_seconds)) as client:
            for attempt in range(settings.llm_max_retries + 1):
                try:
                    response = await client.post(
                        f"{settings.siliconflow_base_url.rstrip('/')}/embeddings",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()
                    data = response.json()
                    break
                except httpx.HTTPStatusError as exc:
                    status = exc.response.status_code
                    retryable = status in {408, 429} or status >= 500
                    if not retryable or attempt >= settings.llm_max_retries:
                        raise
                except httpx.TransportError:
                    if attempt >= settings.llm_max_retries:
                        raise
                await asyncio.sleep(2**attempt)
        items = sorted(data.get("data", []), key=lambda item: int(item.get("index", 0)))
        vectors = [[float(value) for value in item["embedding"]] for item in items]
        OpenAICompatibleEmbeddingProvider._validate_dimensions(vectors)
        usage = data.get("usage") or {}
        tokens = int(usage.get("prompt_tokens") or usage.get("total_tokens") or 0)
        async with self._usage_lock:
            self.usage.calls += 1
            self.usage.input_tokens += tokens
            self.usage.inputs += len(inputs)
        return vectors

    def usage_snapshot(self) -> dict[str, int]:
        return {
            "calls": self.usage.calls,
            "input_tokens": self.usage.input_tokens,
            "inputs": self.usage.inputs,
        }


def build_embedding_provider() -> EmbeddingProvider:
    return OpenAICompatibleEmbeddingProvider()


def build_vl_embedding_provider() -> VLEmbeddingProvider:
    return SiliconFlowVLEmbeddingProvider()


class VectorStoreService:
    LEGACY_COLLECTION = "knowledge_chunks"
    COLLECTION_NAME = LEGACY_COLLECTION
    COLLECTIONS = {
        "text": "knowledge_text_v2",
        "table": "knowledge_table_v2",
        "image": "knowledge_image_v2",
    }

    def __init__(
        self,
        embeddings: EmbeddingProvider | None = None,
        vl_embeddings: VLEmbeddingProvider | None = None,
    ) -> None:
        self.embeddings = embeddings
        self.vl_embeddings = vl_embeddings
        self._client: Any = None
        self._store: Any = None
        self._stores: dict[str, Any] = {}

    async def init(self) -> None:
        if self.embeddings is None:
            self.embeddings = build_embedding_provider()
        if self.vl_embeddings is None:
            self.vl_embeddings = build_vl_embedding_provider()
        import chromadb

        self._client = await asyncio.to_thread(
            chromadb.HttpClient,
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        await asyncio.to_thread(self._client.heartbeat)
        self._store = await self._get_or_create(self.LEGACY_COLLECTION)
        self._stores = {
            modality: await self._get_or_create(name) for modality, name in self.COLLECTIONS.items()
        }

    async def _get_or_create(self, name: str) -> Any:
        return await asyncio.to_thread(
            self._client.get_or_create_collection,
            name=name,
            metadata={"hnsw:space": "cosine"},
        )

    def _collection(self, modality: str | None = None) -> Any:
        if modality is not None and modality in self._stores:
            return self._stores[modality]
        if self._store is None:
            raise RuntimeError("vector store is not initialized")
        return self._store

    def _embedding(self) -> EmbeddingProvider:
        if self.embeddings is None:
            raise RuntimeError("embedding provider is not initialized")
        return self.embeddings

    def _vl_embedding(self) -> VLEmbeddingProvider:
        if self.vl_embeddings is None:
            raise RuntimeError("VL embedding provider is not initialized")
        return self.vl_embeddings

    async def health(self) -> bool:
        try:
            self._collection()
            collections = list(self._stores.values()) or [self._collection()]
            await asyncio.gather(*(asyncio.to_thread(collection.count) for collection in collections))
            return True
        except Exception:
            return False

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        if not chunks:
            return 0
        if not self._stores:
            await self._upsert_text_collection(self._collection(), chunks)
            return len(chunks)

        grouped: dict[str, list[DocumentChunk]] = {key: [] for key in self.COLLECTIONS}
        for chunk in chunks:
            modality = str(chunk.metadata.get("modality") or chunk.metadata.get("kind") or "text")
            grouped[modality if modality in grouped else "text"].append(chunk)
        if grouped["text"]:
            await self._upsert_text_collection(self._stores["text"], grouped["text"])
        if grouped["table"]:
            await self._upsert_text_collection(self._stores["table"], grouped["table"])
        if grouped["image"]:
            await self._upsert_image_collection(self._stores["image"], grouped["image"])
        return len(chunks)

    async def _upsert_text_collection(self, collection: Any, chunks: list[DocumentChunk]) -> None:
        texts = [self._embedding_text(chunk) for chunk in chunks]
        vectors: list[list[float]] = []
        for start in range(0, len(texts), settings.embedding_batch_size):
            vectors.extend(
                await self._embedding().aembed_documents(texts[start : start + settings.embedding_batch_size])
            )
        # Contextual prefixes improve retrieval but citations must keep the
        # original child text instead of exposing the synthetic prefix.
        await self._upsert(collection, chunks, vectors, [chunk.content for chunk in chunks])

    async def _upsert_image_collection(self, collection: Any, chunks: list[DocumentChunk]) -> None:
        paths = [str(chunk.metadata.get("asset_path", "")) for chunk in chunks]
        missing = [path for path in paths if not path or not Path(path).is_file()]
        if missing:
            raise RuntimeError("image chunk is missing its managed asset")
        vectors: list[list[float]] = []
        for start in range(0, len(paths), settings.vision_batch_size):
            vectors.extend(
                await self._vl_embedding().aembed_images(paths[start : start + settings.vision_batch_size])
            )
        await self._upsert(collection, chunks, vectors, [chunk.content for chunk in chunks])

    async def _upsert(
        self,
        collection: Any,
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
        documents: list[str],
    ) -> None:
        await asyncio.to_thread(
            collection.upsert,
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=documents,
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )

    @staticmethod
    def _embedding_text(chunk: DocumentChunk) -> str:
        metadata = chunk.metadata
        location = " ".join(
            value
            for value in (
                f"file={metadata.get('file_name')}" if metadata.get("file_name") else "",
                f"section={metadata.get('section')}" if metadata.get("section") else "",
                f"title={metadata.get('title')}" if metadata.get("title") else "",
                f"page={metadata.get('page')}" if metadata.get("page") else "",
                f"slide={metadata.get('slide')}" if metadata.get("slide") else "",
                f"sheet={metadata.get('sheet')}" if metadata.get("sheet") else "",
                f"rows={metadata.get('row_start')}-{metadata.get('row_end')}"
                if metadata.get("row_start")
                else "",
                f"json_path={metadata.get('json_path')}" if metadata.get("json_path") else "",
                f"xpath={metadata.get('xpath')}" if metadata.get("xpath") else "",
            )
            if value
        )
        return f"{location}\n{chunk.content}".strip()

    @staticmethod
    def _metadata(chunk: DocumentChunk) -> dict[str, str | int | float | bool]:
        metadata: dict[str, str | int | float | bool] = {
            "doc_id": chunk.doc_id,
            "chunk_id": chunk.chunk_id,
            "doc_type": chunk.doc_type.value,
            "source": str(chunk.metadata.get("source", "")),
            "file_name": str(chunk.metadata.get("file_name", "")),
            "unit_id": str(chunk.metadata.get("unit_id", "")),
            "parent_id": str(chunk.metadata.get("parent_id", "")),
            "chunk_index": chunk.chunk_index,
            "version": int(chunk.metadata.get("version", 1)),
            "modality": str(chunk.metadata.get("modality") or chunk.metadata.get("kind") or "text"),
        }
        for key in (
            "page",
            "sheet",
            "slide",
            "row_start",
            "row_end",
            "kind",
            "asset_path",
            "json_path",
            "xpath",
            "vision_fallback",
            "section",
            "title",
            "chunk_level",
        ):
            value = chunk.metadata.get(key)
            if isinstance(value, str | int | float | bool) and value != "":
                metadata[key] = value
        return metadata

    async def search(
        self,
        query: str,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        results, _ = await self.search_multimodal(query, top_k=top_k, document_ids=document_ids)
        return results

    async def search_multimodal(
        self,
        query: str,
        *,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> tuple[list[tuple[dict[str, Any], float]], dict[str, int]]:
        vectors = await self.embed_multimodal_query(query)
        return await self.search_multimodal_by_vectors(
            vectors,
            top_k=top_k,
            document_ids=document_ids,
        )

    async def embed_multimodal_query(self, query: str) -> MultimodalQueryVector:
        if not self._stores:
            return MultimodalQueryVector(text=await self.embed_query(query))
        text_vector, vl_vector = await asyncio.gather(
            self.embed_query(query),
            self._vl_embedding().aembed_query(query),
        )
        return MultimodalQueryVector(text=text_vector, vl=vl_vector)

    async def search_multimodal_by_vectors(
        self,
        vectors: MultimodalQueryVector,
        *,
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> tuple[list[tuple[dict[str, Any], float]], dict[str, int]]:
        if not self._stores:
            legacy = await self._query_collection(self._collection(), vectors.text, top_k, document_ids)
            return legacy, {"legacy": len(legacy), "text": 0, "table": 0, "image": 0}
        vl_vector = vectors.vl or vectors.text
        per_collection = max(top_k * 2, 10)
        text, table, image, legacy = await asyncio.gather(
            self._query_collection(self._stores["text"], vectors.text, per_collection, document_ids),
            self._query_collection(self._stores["table"], vectors.text, per_collection, document_ids),
            self._query_collection(self._stores["image"], vl_vector, per_collection, document_ids),
            self._query_collection(self._collection(), vectors.text, per_collection, document_ids),
        )
        v2_chunk_ids = {
            str(document.get("chunk_id") or "")
            for results in (text, table, image)
            for document, _score in results
        }
        # A migrated chunk can temporarily exist in both the legacy and v2
        # collections. Keep legacy as a read fallback without awarding the same
        # evidence a second RRF contribution.
        legacy = [item for item in legacy if str(item[0].get("chunk_id") or "") not in v2_chunk_ids]
        ranked = self._weighted_rrf(
            {
                "text": (text, settings.modality_text_weight),
                "table": (table, settings.modality_table_weight),
                "image": (image, settings.modality_image_weight),
                "legacy": (legacy, settings.modality_text_weight),
            }
        )[:top_k]
        return ranked, {
            "text": len(text),
            "table": len(table),
            "image": len(image),
            "legacy": len(legacy),
        }

    async def embed_query(self, query: str) -> list[float]:
        return await self._embedding().aembed_query(query)

    async def search_by_vector(
        self,
        vector: list[float],
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        result_sets = await self.search_many_by_vectors([vector], top_k=top_k, document_ids=document_ids)
        return result_sets[0] if result_sets else []

    async def search_many_by_vectors(
        self,
        vectors: list[list[float]],
        top_k: int = 5,
        document_ids: list[str] | None = None,
    ) -> list[list[tuple[dict[str, Any], float]]]:
        if not vectors:
            return []
        collection = self._stores.get("text") if self._stores else self._collection()
        return [await self._query_collection(collection, vector, top_k, document_ids) for vector in vectors]

    async def _query_collection(
        self,
        collection: Any,
        vector: list[float],
        top_k: int,
        document_ids: list[str] | None,
    ) -> list[tuple[dict[str, Any], float]]:
        if await asyncio.to_thread(collection.count) == 0:
            return []
        where: dict[str, Any] | None = None
        if document_ids:
            where = (
                {"doc_id": document_ids[0]} if len(document_ids) == 1 else {"doc_id": {"$in": document_ids}}
            )
        kwargs: dict[str, Any] = {
            "query_embeddings": [vector],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where
        results = await asyncio.to_thread(collection.query, **kwargs)
        documents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        output: list[tuple[dict[str, Any], float]] = []
        for document, metadata, distance in zip(documents, metadatas, distances, strict=False):
            metadata = metadata or {}
            score = max(0.0, min(1.0, 1.0 - float(distance)))
            output.append(
                (
                    {
                        "content": document or "",
                        "source": metadata.get("source", ""),
                        "metadata": metadata,
                        "chunk_id": metadata.get("chunk_id", ""),
                        "doc_id": metadata.get("doc_id", ""),
                    },
                    score,
                )
            )
        return output

    @staticmethod
    def _weighted_rrf(
        groups: dict[str, tuple[list[tuple[dict[str, Any], float]], float]],
        constant: int | None = None,
    ) -> list[tuple[dict[str, Any], float]]:
        constant = constant or settings.rrf_constant
        fused: dict[str, dict[str, Any]] = {}
        max_score = 0.0
        for modality, (results, weight) in groups.items():
            seen: set[str] = set()
            for rank, (document, _native_score) in enumerate(results, start=1):
                key = str(document.get("chunk_id") or f"{document.get('doc_id')}:{document.get('content')}")
                if key in seen:
                    continue
                seen.add(key)
                entry = fused.setdefault(
                    key,
                    {"document": document, "rrf": 0.0, "modalities": set()},
                )
                entry["rrf"] += weight / (constant + rank)
                entry["modalities"].add(modality)
                max_score = max(max_score, entry["rrf"])
        output: list[tuple[dict[str, Any], float]] = []
        for entry in fused.values():
            document = dict(entry["document"])
            metadata = dict(document.get("metadata") or {})
            metadata["modality_sources"] = ",".join(sorted(entry["modalities"]))
            document["metadata"] = metadata
            normalized_rrf = entry["rrf"] / max_score if max_score else 0.0
            output.append((document, min(1.0, normalized_rrf)))
        return sorted(output, key=lambda item: (-item[1], str(item[0].get("chunk_id", ""))))

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        identifiers = list(dict.fromkeys(chunk_ids))
        if not identifiers:
            return 0
        collections = [self._collection(), *self._stores.values()]
        for collection in collections:
            await asyncio.to_thread(collection.delete, ids=identifiers)
        return len(identifiers)

    async def delete_by_doc_id(self, doc_id: str) -> int:
        identifiers = await self.get_document_chunks(doc_id)
        if identifiers:
            await self.delete_chunks(identifiers)
        return len(identifiers)

    async def get_document_chunks(self, doc_id: str) -> list[str]:
        identifiers: set[str] = set()
        for collection in [self._collection(), *self._stores.values()]:
            existing = await asyncio.to_thread(collection.get, where={"doc_id": doc_id}, include=[])
            identifiers.update(str(value) for value in (existing.get("ids") or []))
        return sorted(identifiers)

    async def get_v2_document_chunks(self, doc_id: str) -> list[str]:
        identifiers: set[str] = set()
        for collection in self._stores.values():
            existing = await asyncio.to_thread(collection.get, where={"doc_id": doc_id}, include=[])
            identifiers.update(str(value) for value in (existing.get("ids") or []))
        return sorted(identifiers)

    async def get_stats(self) -> dict[str, Any]:
        collections = {"legacy": self._collection(), **self._stores}
        counts = {
            modality: await asyncio.to_thread(collection.count)
            for modality, collection in collections.items()
        }
        text_usage = getattr(self.embeddings, "usage_snapshot", lambda: {})()
        vl_usage = getattr(self.vl_embeddings, "usage_snapshot", lambda: {})()
        return {
            "backend": "chroma",
            "total_vectors": sum(counts.values()),
            "collections": {
                modality: {
                    "name": self.LEGACY_COLLECTION if modality == "legacy" else self.COLLECTIONS[modality],
                    "vectors": count,
                    "model": settings.active_embedding_model
                    if modality != "image"
                    else settings.active_vl_embedding_model,
                    "dimensions": settings.embedding_dimensions,
                }
                for modality, count in counts.items()
            },
            "usage": {"text": text_usage, "vl": vl_usage},
        }
