from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from services.vector_store import OpenAICompatibleEmbeddingProvider, VectorStoreService
from tests.conftest import FakeEmbedding


class FakeCollection:
    def __init__(self) -> None:
        self.values: dict[str, dict[str, Any]] = {}

    def count(self) -> int:
        return len(self.values)

    def upsert(self, *, ids, embeddings, documents, metadatas) -> None:
        for identifier, vector, document, metadata in zip(ids, embeddings, documents, metadatas, strict=True):
            self.values[identifier] = {"embedding": vector, "document": document, "metadata": metadata}

    def query(self, **_kwargs):
        values = list(self.values.values())
        count = len(_kwargs.get("query_embeddings") or [[]])
        return {
            "documents": [[value["document"] for value in values] for _ in range(count)],
            "metadatas": [[value["metadata"] for value in values] for _ in range(count)],
            "distances": [[0.1 + index * 0.1 for index in range(len(values))] for _ in range(count)],
        }

    def get(self, *, where, include):
        del include
        return {
            "ids": [
                key for key, value in self.values.items() if value["metadata"]["doc_id"] == where["doc_id"]
            ]
        }

    def delete(self, *, ids) -> None:
        for identifier in ids:
            self.values.pop(identifier, None)


class FakeEmbeddingsEndpoint:
    async def create(self, **kwargs: Any):
        data = [
            SimpleNamespace(index=index, embedding=[0.1, 0.2, 0.3]) for index, _ in enumerate(kwargs["input"])
        ]
        return SimpleNamespace(data=data, usage=SimpleNamespace(prompt_tokens=7, total_tokens=7))


class FakeVLEmbedding:
    def __init__(self) -> None:
        self.images: list[str] = []
        self.queries: list[str] = []

    async def aembed_images(self, paths: list[str]) -> list[list[float]]:
        self.images.extend(paths)
        return [[0.2] * 8 for _path in paths]

    async def aembed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return [0.2] * 8

    def usage_snapshot(self) -> dict[str, int]:
        return {"calls": len(self.images) + len(self.queries)}


class FakeOpenAI:
    def __init__(self, **_kwargs: Any) -> None:
        self.embeddings = FakeEmbeddingsEndpoint()


@pytest.mark.asyncio
async def test_metered_embedding_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("openai.AsyncOpenAI", FakeOpenAI)
    monkeypatch.setattr("services.vector_store.settings.embedding_dimensions", 3)
    monkeypatch.setattr("services.vector_store.settings.siliconflow_api_key", "test")
    provider = OpenAICompatibleEmbeddingProvider()
    assert len((await provider.aembed_documents(["a", "b"]))[0]) == 3
    assert await provider.aembed_documents([]) == []
    assert len(await provider.aembed_query("q")) == 3
    assert provider.usage_snapshot() == {"calls": 2, "input_tokens": 14, "texts": 3}


@pytest.mark.asyncio
async def test_vector_store_crud_and_filter_metadata() -> None:
    embeddings = FakeEmbedding()
    store = VectorStoreService(embeddings)
    store._store = FakeCollection()
    chunks = [
        DocumentChunk(
            "alpha",
            "doc",
            0,
            DocType.TEXT,
            {"source": "a.txt", "file_name": "a.txt", "version": 1, "page": 2},
            identifier="c1",
        ),
        DocumentChunk(
            "beta",
            "doc",
            1,
            DocType.TEXT,
            {"source": "a.txt", "file_name": "a.txt", "version": 1},
            identifier="c2",
        ),
    ]
    assert await store.add_chunks(chunks) == 2
    results = await store.search("alpha", top_k=2, document_ids=["doc"])
    assert results[0][0]["chunk_id"] == "c1"
    assert results[0][1] == pytest.approx(0.9)
    embedded_results = await store.search_by_vector([0.1] * 8, top_k=2, document_ids=["doc"])
    assert embedded_results[0][0]["chunk_id"] == "c1"
    batches = await store.search_many_by_vectors([[0.1] * 8, [0.2] * 8], top_k=2)
    assert len(batches) == 2
    assert await store.search_many_by_vectors([]) == []
    assert await store.get_document_chunks("doc") == ["c1", "c2"]
    stats = await store.get_stats()
    assert stats["total_vectors"] == 2
    assert await store.delete_by_doc_id("doc") == 2
    assert await store.health()


@pytest.mark.asyncio
async def test_vector_init_delete_chunks_and_unhealthy(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = FakeCollection()

    class Client:
        def heartbeat(self):
            return 1

        def get_or_create_collection(self, **_kwargs: Any):
            return collection

    monkeypatch.setattr("chromadb.HttpClient", lambda **_kwargs: Client())
    store = VectorStoreService(FakeEmbedding())
    await store.init()
    assert await store.add_chunks([]) == 0
    assert await store.delete_chunks([]) == 0
    collection.values["c"] = {"document": "x", "embedding": [], "metadata": {"doc_id": "d"}}
    assert await store.delete_chunks(["c"]) == 1
    store._store = None
    assert not await store.health()
    with pytest.raises(RuntimeError):
        store._collection()


@pytest.mark.asyncio
async def test_three_collections_raw_image_query_weights_and_cross_collection_delete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    image_path = tmp_path / "asset.png"
    Image.new("RGB", (30, 30), "blue").save(image_path)
    vl = FakeVLEmbedding()
    store = VectorStoreService(FakeEmbedding(), vl)
    store._store = FakeCollection()
    store._stores = {key: FakeCollection() for key in ("text", "table", "image")}
    chunks = [
        DocumentChunk(
            "Atlas owner Alice",
            "doc",
            0,
            DocType.TEXT,
            {"modality": "text", "source": "a.txt", "version": 1},
            identifier="text-1",
        ),
        DocumentChunk(
            "Metric | Value\nRevenue | 100",
            "doc",
            1,
            DocType.TABLE,
            {"modality": "table", "source": "a.xlsx", "sheet": "Sheet1", "version": 1},
            identifier="table-1",
        ),
        DocumentChunk(
            "blue revenue chart",
            "doc",
            2,
            DocType.IMAGE,
            {
                "modality": "image",
                "source": "chart.png",
                "asset_path": str(image_path),
                "version": 1,
            },
            identifier="image-1",
        ),
    ]
    assert await store.add_chunks(chunks) == 3
    assert {key: value.count() for key, value in store._stores.items()} == {
        "text": 1,
        "table": 1,
        "image": 1,
    }
    assert await store.add_chunks(chunks) == 3
    assert await store.get_v2_document_chunks("doc") == ["image-1", "table-1", "text-1"]
    assert all(collection.count() == 1 for collection in store._stores.values())
    assert vl.images == [str(image_path), str(image_path)]

    # During migration the same chunk can exist in legacy and v2. Legacy stays
    # readable, but the duplicate must not receive another RRF contribution.
    store._store.values["text-1"] = dict(store._stores["text"].values["text-1"])
    results, counts = await store.search_multimodal("revenue chart", top_k=3)
    assert counts == {"text": 1, "table": 1, "image": 1, "legacy": 0}
    assert {item[0]["metadata"]["modality"] for item in results} == {
        "text",
        "table",
        "image",
    }
    assert vl.queries == ["revenue chart"]

    monkeypatch.setattr("services.vector_store.settings.modality_image_weight", 2.0)
    boosted, _ = await store.search_multimodal("revenue chart", top_k=1)
    assert boosted[0][0]["chunk_id"] == "image-1"
    assert await store.delete_chunks(["text-1", "table-1", "image-1"]) == 3
    assert all(collection.count() == 0 for collection in store._stores.values())


def test_modality_weighted_rrf_ignores_native_score_scale() -> None:
    text = ({"chunk_id": "text", "metadata": {}}, 0.01)
    image = ({"chunk_id": "image", "metadata": {}}, 999.0)

    output = VectorStoreService._weighted_rrf(
        {"text": ([text], 1.0), "image": ([image], 0.5)},
        constant=60,
    )

    assert [document["chunk_id"] for document, _score in output] == ["text", "image"]
