from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from config import settings
from services.graph_rag import (
    EvidenceSlot,
    GraphRAGContext,
    GraphRAGPipeline,
    QueryPlan,
    RetrievalResult,
)
from services.reranker import RerankScore
from services.vector_store import MultimodalQueryVector
from tests.conftest import FakeChatModel


class FakeVectorStore:
    async def embed_query(self, _query: str):
        return [0.1, 0.2]

    async def search_by_vector(self, _vector, top_k: int, document_ids: list[str] | None = None):
        return await self.search("embedded", top_k, document_ids)

    async def search_many_by_vectors(self, vectors, top_k: int, document_ids=None):
        return [await self.search("embedded", top_k, document_ids) for _vector in vectors]

    async def search(self, query: str, top_k: int, document_ids: list[str] | None = None):
        assert "MATCH" not in query
        return [
            (
                {
                    "content": "Alice owns Atlas",
                    "source": "a.md",
                    "chunk_id": "c1",
                    "doc_id": "d1",
                    "metadata": {"file_name": "a.md", "doc_id": "d1", "chunk_id": "c1"},
                },
                0.8,
            ),
            (
                {
                    "content": "Revenue is 100",
                    "source": "b.md",
                    "chunk_id": "c2",
                    "doc_id": "d2",
                    "metadata": {"file_name": "b.md", "doc_id": "d2", "chunk_id": "c2"},
                },
                0.7,
            ),
        ]


class FakeGraph:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def graph_search(self, entities: list[str], **kwargs: Any):
        self.calls.append({"entities": entities, **kwargs})
        return [
            {
                "chunk_id": "c1",
                "doc_id": "d1",
                "content": "Alice owns Atlas",
                "source": "a.md",
                "file_name": "a.md",
                "page": 1,
                "hops": 1,
            }
        ]


@pytest.mark.asyncio
async def test_query_plan_and_rrf_hybrid_retrieval() -> None:
    llm = FakeChatModel(
        [json.dumps({"intent": "factoid", "queries": ["Atlas owner"], "entities": ["Alice"]})]
    )
    graph = FakeGraph()
    pipeline = GraphRAGPipeline(FakeVectorStore(), graph, llm=llm)
    result = await pipeline.retrieve("Who owns Atlas?", top_k=2, mode="hybrid")
    assert result.contexts[0].chunk_id == "c1"
    assert result.contexts[0].source_type == "hybrid"
    assert graph.calls[0]["hops"] == 1
    assert result.trace[-1] == "selected_contexts:2"
    assert result.planning_ms >= 0
    assert result.embedding_ms >= 0
    assert result.retrieval_ms >= 0


@pytest.mark.asyncio
async def test_hybrid_retrieval_falls_back_when_graph_backend_fails() -> None:
    class FailingGraph:
        async def graph_search(self, *_args: Any, **_kwargs: Any):
            raise RuntimeError("temporary graph failure")

    pipeline = GraphRAGPipeline(FakeVectorStore(), FailingGraph(), llm=FakeChatModel([]))
    result = await pipeline.retrieve(
        "Who owns Atlas?",
        top_k=2,
        mode="hybrid",
        plan=QueryPlan(queries=["Atlas owner"], entities=["Alice"]),
    )

    assert [context.chunk_id for context in result.contexts] == ["c1", "c2"]
    assert "graph_search_failed:RuntimeError" in result.trace


def test_rrf_deduplicates_chunks() -> None:
    context = GraphRAGContext("c", "d", "text", "x", "vector", 0.5)
    output = GraphRAGPipeline._rrf_fuse([context, context], [])
    assert len(output) == 1


def test_weighted_rrf_does_not_mix_cross_source_native_scores() -> None:
    vector = GraphRAGContext("vector", "dv", "vector", "v", "vector", 0.01)
    graph = GraphRAGContext("graph", "dg", "graph", "g", "graph", 999.0)

    output = GraphRAGPipeline._rrf_fuse(
        [vector],
        [graph],
        constant=60,
        vector_weight=1.0,
        graph_weight=0.5,
    )

    assert [context.chunk_id for context in output] == ["vector", "graph"]


def test_hybrid_selection_prefers_document_diversity_then_backfills() -> None:
    contexts = [
        GraphRAGContext("a1", "a", "first", "a", "vector", 1.0),
        GraphRAGContext("a2", "a", "second", "a", "vector", 0.9),
        GraphRAGContext("b1", "b", "third", "b", "graph", 0.8),
    ]

    diversified = GraphRAGPipeline._select_diverse(contexts, 2, max_per_document=1)
    backfilled = GraphRAGPipeline._select_diverse(contexts, 3, max_per_document=1)

    assert [context.chunk_id for context in diversified] == ["a1", "b1"]
    assert [context.chunk_id for context in backfilled] == ["a1", "b1", "a2"]


@pytest.mark.asyncio
async def test_plan_falls_back_on_invalid_json() -> None:
    pipeline = GraphRAGPipeline(FakeVectorStore(), FakeGraph(), llm=FakeChatModel(["bad"]))
    plan = await pipeline.plan_query("plain question")
    assert plan == QueryPlan(queries=["plain question"])


@pytest.mark.asyncio
async def test_query_planning_retries_transient_provider_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FlakyPlanner(FakeChatModel):
        attempts = 0

        async def ainvoke(self, messages, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise TimeoutError("temporary")
            return await super().ainvoke(messages, **kwargs)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("services.graph_rag.asyncio.sleep", no_sleep)
    model = FlakyPlanner([json.dumps({"queries": ["stable"]})])
    plan = await GraphRAGPipeline(FakeVectorStore(), FakeGraph(), llm=model).plan_query("question")

    assert plan.queries == ["stable"]
    assert model.attempts == 3


@pytest.mark.asyncio
async def test_document_refinement_reuses_query_vectors_and_filters_selected_docs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class RefiningStore:
        def __init__(self) -> None:
            self.embed_calls: list[str] = []
            self.search_filters: list[list[str] | None] = []

        async def embed_multimodal_query(self, query: str) -> MultimodalQueryVector:
            self.embed_calls.append(query)
            return MultimodalQueryVector(text=[float(len(self.embed_calls))])

        async def search_multimodal_by_vectors(
            self,
            _vector: MultimodalQueryVector,
            *,
            top_k: int,
            document_ids: list[str] | None = None,
        ):
            self.search_filters.append(document_ids)
            if document_ids is None:
                return (
                    [
                        (
                            {
                                "content": "weak document",
                                "source": "d1",
                                "chunk_id": "d1-a",
                                "doc_id": "d1",
                                "metadata": {"doc_id": "d1", "chunk_id": "d1-a"},
                            },
                            0.9,
                        ),
                        (
                            {
                                "content": "selected document",
                                "source": "d2",
                                "chunk_id": "d2-a",
                                "doc_id": "d2",
                                "metadata": {"doc_id": "d2", "chunk_id": "d2-a"},
                            },
                            0.8,
                        ),
                    ],
                    {"text": 2},
                )
            assert document_ids == ["d2"]
            return (
                [
                    (
                        {
                            "content": "selected document left",
                            "source": "d2",
                            "chunk_id": "d2-a",
                            "doc_id": "d2",
                            "metadata": {"doc_id": "d2", "chunk_id": "d2-a"},
                        },
                        0.9,
                    ),
                    (
                        {
                            "content": "selected document right",
                            "source": "d2",
                            "chunk_id": "d2-b",
                            "doc_id": "d2",
                            "metadata": {"doc_id": "d2", "chunk_id": "d2-b"},
                        },
                        0.8,
                    ),
                ][:top_k],
                {"text": 2},
            )

    class DeterministicReranker:
        async def rerank(self, _query: str, documents: list[str], *, top_n: int):
            scores = [0.99 if "selected document" in value else 0.1 for value in documents]
            return sorted(
                [RerankScore(index, score) for index, score in enumerate(scores)],
                key=lambda item: -item.score,
            )[:top_n]

        async def record_fallback(self) -> None:
            raise AssertionError("fallback should not run")

    monkeypatch.setattr(settings, "rerank_top_documents", 1)
    store = RefiningStore()
    pipeline = GraphRAGPipeline(store, FakeGraph(), llm=FakeChatModel([]), reranker=DeterministicReranker())
    retrieval_plan = QueryPlan(
        queries=["left query", "right query"],
        evidence_slots=[
            EvidenceSlot(slot_id="left", query="left query"),
            EvidenceSlot(slot_id="right", query="right query"),
        ],
    )
    result = await pipeline.retrieve("question", top_k=2, mode="vector", plan=retrieval_plan)
    assert store.embed_calls == ["left query", "right query"]
    assert store.search_filters == [None, None, ["d2"], ["d2"]]
    assert [item.doc_id for item in result.contexts] == ["d2", "d2"]
    assert len(result.contexts) == 2
    assert "selected_documents:1" in result.trace


@pytest.mark.asyncio
async def test_reranker_failure_falls_back_without_interrupting_answer_candidates() -> None:
    class FailingReranker:
        def __init__(self) -> None:
            self.fallbacks = 0

        async def rerank(self, *_args: Any, **_kwargs: Any):
            raise TimeoutError("unavailable")

        async def record_fallback(self) -> None:
            self.fallbacks += 1

    reranker = FailingReranker()
    pipeline = GraphRAGPipeline(FakeVectorStore(), FakeGraph(), llm=FakeChatModel([]), reranker=reranker)
    result = await pipeline.retrieve(
        "Who owns Atlas?",
        top_k=2,
        mode="hybrid",
        plan=QueryPlan(queries=["Atlas owner"], entities=["Alice"]),
    )
    assert result.contexts
    assert reranker.fallbacks >= 1
    assert any("rerank_docs_fallback:TimeoutError" == item for item in result.trace)


@pytest.mark.asyncio
async def test_graph_only_never_invokes_vector_search() -> None:
    class ExplodingVectorStore:
        def __getattr__(self, _name: str):
            raise AssertionError("graph mode must not touch vector store")

    pipeline = GraphRAGPipeline(ExplodingVectorStore(), FakeGraph(), llm=FakeChatModel([]))
    result = await pipeline.retrieve(
        "Who owns Atlas?",
        top_k=2,
        mode="graph",
        plan=QueryPlan(queries=["Atlas owner"], entities=["Alice"]),
    )
    assert [item.chunk_id for item in result.contexts] == ["c1"]
    assert "document_refine:skipped_graph_mode" in result.trace


@pytest.mark.asyncio
async def test_slot_reranking_runs_in_parallel_and_preserves_base_rrf() -> None:
    class ParallelReranker:
        def __init__(self) -> None:
            self.active = 0
            self.max_active = 0

        async def rerank(self, query: str, _documents: list[str], *, top_n: int):
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            await asyncio.sleep(0.01)
            self.active -= 1
            if query == "left":
                order = [(2, 0.99), (1, 0.5), (0, 0.01)]
            else:
                order = [(0, 0.99), (1, 0.5), (2, 0.01)]
            return [RerankScore(index, score) for index, score in order[:top_n]]

        async def record_fallback(self) -> None:
            raise AssertionError("fallback should not run")

    reranker = ParallelReranker()
    pipeline = GraphRAGPipeline(FakeVectorStore(), FakeGraph(), reranker=reranker)
    plan = QueryPlan(
        queries=["left", "right"],
        evidence_slots=[
            EvidenceSlot(slot_id="left", query="left"),
            EvidenceSlot(slot_id="right", query="right"),
        ],
    )
    contexts = [
        GraphRAGContext("a", "a", "A", "a", "hybrid", 1.0, {"rrf_score": 1.0}),
        GraphRAGContext("b", "b", "B", "b", "hybrid", 0.8, {"rrf_score": 0.8}),
        GraphRAGContext("c", "c", "C", "c", "hybrid", 0.6, {"rrf_score": 0.6}),
    ]

    ranked, _elapsed, error = await pipeline._rerank_contexts("question", plan, contexts)

    assert error == ""
    assert reranker.max_active == 2
    assert [item.chunk_id for item in ranked] == ["a", "b", "c"]
    assert ranked[0].metadata["rrf_score"] == 1.0
    assert ranked[0].metadata["rerank_slot_ids"] == ["left", "right"]
    assert "rerank_fused_score" in ranked[0].metadata


def test_evidence_packing_is_adaptive_instead_of_forcing_every_document() -> None:
    contexts = [
        GraphRAGContext("a1", "a", "A1", "a", "vector", 1.0),
        GraphRAGContext("a2", "a", "A2", "a", "vector", 0.9),
        GraphRAGContext("b1", "b", "B", "b", "vector", 0.8),
        GraphRAGContext("c1", "c", "C", "c", "vector", 0.7),
        GraphRAGContext("d1", "d", "D", "d", "vector", 0.6),
    ]

    selected = GraphRAGPipeline._pack_contexts(contexts, QueryPlan(), 4)

    assert [item.chunk_id for item in selected] == ["a1", "a2", "b1", "c1"]
    assert {item.doc_id for item in selected} == {"a", "b", "c"}


@pytest.mark.asyncio
async def test_temporal_supplement_reuses_vectors_and_preserves_document_filter() -> None:
    class SupplementalStore:
        def __init__(self) -> None:
            self.filters: list[list[str] | None] = []

        async def search_multimodal_by_vectors(self, _vector, *, top_k: int, document_ids=None):
            self.filters.append(document_ids)
            return (
                [
                    (
                        {
                            "content": "later evidence",
                            "source": "later.pdf",
                            "chunk_id": "later",
                            "doc_id": "allowed",
                            "metadata": {"doc_id": "allowed", "chunk_id": "later"},
                        },
                        0.9,
                    )
                ][:top_k],
                {"text": 1},
            )

    plan = QueryPlan(
        queries=["old", "new"],
        evidence_slots=[
            EvidenceSlot(slot_id="left", query="old"),
            EvidenceSlot(slot_id="right", query="new"),
        ],
    )
    store = SupplementalStore()
    pipeline = GraphRAGPipeline(store, FakeGraph())
    retrieval = RetrievalResult(
        plan,
        [GraphRAGContext("old", "allowed", "old evidence", "old.pdf", "vector", 1.0)],
        5,
        [],
        query_vectors=[MultimodalQueryVector(text=[0.1]), MultimodalQueryVector(text=[0.2])],
        document_ids=["allowed"],
    )

    result = await pipeline.supplement_evidence("Did it change?", retrieval, top_k=2)

    assert store.filters == [["allowed"], ["allowed"]]
    assert "temporal_supplement:applied" in result.trace
    assert {item.chunk_id for item in result.contexts} == {"old", "later"}
