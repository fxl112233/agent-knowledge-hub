"""Unified, safe GraphRAG retrieval with reranking and document refinement."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError, model_validator

from config import settings
from services.knowledge_graph import KnowledgeGraphService
from services.reranker import SiliconFlowReranker
from services.vector_store import MultimodalQueryVector, VectorStoreService


class RetrievalMode(StrEnum):
    VECTOR = "vector"
    GRAPH = "graph"
    HYBRID = "hybrid"


class AnswerType(StrEnum):
    BOOLEAN = "boolean"
    ENTITY = "entity"
    NUMBER = "number"
    TEXT = "text"


class TargetRelation(StrEnum):
    SAME_CLAIM = "same_claim"
    CHANGED_CLAIM = "changed_claim"
    BEFORE = "before"
    AFTER = "after"
    SAME_TIME = "same_time"
    NONE = "none"


class EvidenceSlot(BaseModel):
    slot_id: str = Field(min_length=1, max_length=40)
    query: str = Field(min_length=1, max_length=1000)
    role: str = Field(default="fact", max_length=80)
    required: bool = True


class QueryPlan(BaseModel):
    intent: str = "factoid"
    queries: list[str] = Field(default_factory=list, max_length=3)
    entities: list[str] = Field(default_factory=list, max_length=10)
    needs_calculation: bool = False
    answer_type: AnswerType = AnswerType.TEXT
    evidence_slots: list[EvidenceSlot] = Field(default_factory=list, max_length=3)
    target_relation: TargetRelation = TargetRelation.NONE

    @model_validator(mode="after")
    def synchronize_queries_and_slots(self) -> QueryPlan:
        self.queries = [value.strip() for value in self.queries if value.strip()][:3]
        self.entities = list(dict.fromkeys(value.strip() for value in self.entities if value.strip()))[:10]
        if not self.evidence_slots:
            self.evidence_slots = [
                EvidenceSlot(slot_id=f"evidence_{index}", query=query)
                for index, query in enumerate(self.queries, start=1)
            ]
        if not self.queries:
            self.queries = [slot.query for slot in self.evidence_slots][:3]
        return self


@dataclass(slots=True)
class GraphRAGContext:
    chunk_id: str
    doc_id: str
    content: str
    source: str
    source_type: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    plan: QueryPlan
    contexts: list[GraphRAGContext]
    retrieval_ms: float
    trace: list[str]
    planning_ms: float = 0.0
    embedding_ms: float = 0.0
    rerank_ms: float = 0.0
    candidate_count: int = 0
    query_vectors: list[MultimodalQueryVector] = field(default_factory=list, repr=False)
    document_ids: list[str] = field(default_factory=list, repr=False)


QUERY_PLAN_PROMPT = """把用户问题转换为结构化检索计划，只返回 JSON：
{
  "intent":"factoid|comparative|temporal|analytical|procedural|exploratory",
  "queries":["最多3个互不重复的检索查询"],
  "entities":["问题中明确出现的实体"],
  "needs_calculation":false,
  "answer_type":"boolean|entity|number|text",
  "evidence_slots":[
    {"slot_id":"left","query":"要查找的具体证据","role":"left|right|fact|date","required":true}
  ],
  "target_relation":"same_claim|changed_claim|before|after|same_time|none"
}
规则：
1. 实体事实题只生成一个槽位；比较或时间题生成左右两个槽位；确有桥接事实时才生成第三个。
2. 问题询问是否一致、是否没有变化时 target_relation=same_claim；询问是否变化、是否不一致时为 changed_claim。
3. 明确比较日期先后时使用 before/after/same_time。
4. 一致性、变化和日期先后问题的 intent 统一使用 temporal；普通属性比较才使用 comparative。
5. 是非题 answer_type=boolean；不要补充问题里没有出现的实体。
"""


class GraphRAGPipeline:
    def __init__(
        self,
        vector_store: VectorStoreService,
        knowledge_graph: KnowledgeGraphService,
        llm: Any | None = None,
        reranker: SiliconFlowReranker | None = None,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm or self._build_llm()
        self.reranker = reranker

    @staticmethod
    def _build_llm() -> Any | None:
        if not settings.llm_configured:
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def plan_query(self, question: str) -> QueryPlan:
        if self.llm is None:
            return self._fallback_plan(question)
        response: Any | None = None
        for attempt in range(settings.llm_max_retries + 1):
            try:
                response = await self.llm.ainvoke(
                    [
                        {"role": "system", "content": QUERY_PLAN_PROMPT},
                        {"role": "user", "content": question},
                    ]
                )
                break
            except Exception:
                if attempt >= settings.llm_max_retries:
                    raise
                await asyncio.sleep(2**attempt)
        if response is None:  # pragma: no cover - defensive guard
            raise RuntimeError("query planning returned no response")
        try:
            plan = QueryPlan.model_validate(self._json_object(self._response_text(response)))
        except (ValueError, json.JSONDecodeError, ValidationError):
            plan = self._fallback_plan(question)
        if not plan.queries:
            plan.queries = [question]
        if not plan.evidence_slots:
            plan.evidence_slots = [EvidenceSlot(slot_id="evidence_1", query=plan.queries[0])]
        return plan

    @staticmethod
    def _fallback_plan(question: str) -> QueryPlan:
        lowered = question.casefold()
        temporal_markers = (
            "between",
            "before",
            "after",
            "changed",
            "change",
            "consistent",
            "inconsistent",
            "remained",
            "subsequent",
            "一致",
            "变化",
            "之前",
            "之后",
        )
        boolean_markers = (
            "is ",
            "was ",
            "are ",
            "were ",
            "did ",
            "does ",
            "has ",
            "have ",
            "是否",
        )
        intent = "temporal" if any(marker in lowered for marker in temporal_markers) else "factoid"
        answer_type = (
            AnswerType.BOOLEAN
            if lowered.startswith(boolean_markers) or "是否" in lowered
            else AnswerType.TEXT
        )
        relation = TargetRelation.NONE
        if intent == "temporal":
            if any(value in lowered for value in ("not inconsistent", "没有不一致")):
                relation = TargetRelation.SAME_CLAIM
            elif any(value in lowered for value in ("not consistent", "并非一致")):
                relation = TargetRelation.CHANGED_CLAIM
            elif "before" in lowered or "之前" in lowered:
                relation = TargetRelation.BEFORE
            elif "after" in lowered or "之后" in lowered:
                relation = TargetRelation.AFTER
            elif any(value in lowered for value in ("same time", "simultaneous", "同时")):
                relation = TargetRelation.SAME_TIME
            elif any(
                value in lowered
                for value in ("consistent", "no change", "remained", "same", "一致", "没有变化")
            ):
                relation = TargetRelation.SAME_CLAIM
            else:
                relation = TargetRelation.CHANGED_CLAIM
            return QueryPlan(
                intent=intent,
                queries=[f"{question} first source", f"{question} second source"],
                answer_type=AnswerType.BOOLEAN,
                evidence_slots=[
                    EvidenceSlot(slot_id="left", query=f"{question} first source", role="left"),
                    EvidenceSlot(slot_id="right", query=f"{question} second source", role="right"),
                ],
                target_relation=relation,
            )
        return QueryPlan(
            intent=intent,
            queries=[question],
            answer_type=answer_type,
            evidence_slots=[EvidenceSlot(slot_id="evidence_1", query=question)],
            target_relation=relation,
        )

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int = 8,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        document_ids: list[str] | None = None,
        plan: QueryPlan | None = None,
    ) -> RetrievalResult:
        mode = RetrievalMode(mode)
        planning_started = time.perf_counter()
        if plan is None:
            plan = await self.plan_query(query)
            planning_ms = (time.perf_counter() - planning_started) * 1000
        else:
            planning_ms = 0.0
        candidate_k = max(settings.rerank_candidate_k, top_k * 4)
        vector_task = None
        graph_task = None
        if mode in (RetrievalMode.VECTOR, RetrievalMode.HYBRID):
            vector_task = asyncio.create_task(
                self.retrieve_candidates(
                    query,
                    plan=plan,
                    mode=RetrievalMode.VECTOR,
                    candidate_k=candidate_k,
                    document_ids=document_ids,
                )
            )
        if mode in (RetrievalMode.GRAPH, RetrievalMode.HYBRID):
            graph_task = asyncio.create_task(
                self.retrieve_candidates(
                    query,
                    plan=plan,
                    mode=RetrievalMode.GRAPH,
                    candidate_k=candidate_k,
                    document_ids=document_ids,
                )
            )
        empty = RetrievalResult(plan, [], 0.0, [])
        vector = await vector_task if vector_task else empty
        graph = await graph_task if graph_task else empty
        result = await self.finalize_candidates(
            query,
            plan=plan,
            mode=mode,
            vector=vector,
            graph=graph,
            top_k=top_k,
            document_ids=document_ids,
        )
        result.planning_ms = planning_ms
        return result

    async def retrieve_candidates(
        self,
        query: str,
        *,
        plan: QueryPlan,
        mode: RetrievalMode | str,
        candidate_k: int,
        document_ids: list[str] | None = None,
    ) -> RetrievalResult:
        del query
        mode = RetrievalMode(mode)
        started = time.perf_counter()
        if mode == RetrievalMode.VECTOR:
            contexts, modality_counts, embedding_ms, query_vectors = await self._vector_search_queries(
                plan,
                candidate_k,
                document_ids,
            )
            trace = [
                *(f"{name}_candidates:{count}" for name, count in sorted(modality_counts.items())),
                f"vector_candidates:{len(contexts)}",
            ]
            return RetrievalResult(
                plan,
                contexts,
                (time.perf_counter() - started) * 1000,
                trace,
                embedding_ms=embedding_ms,
                candidate_count=len(contexts),
                query_vectors=query_vectors,
            )
        if mode != RetrievalMode.GRAPH:
            raise ValueError("retrieve_candidates accepts vector or graph mode")
        hops = {
            "factoid": 1,
            "procedural": 1,
            "analytical": 2,
            "comparative": 2,
            "temporal": 2,
            "exploratory": 3,
        }.get(plan.intent, 1)
        try:
            contexts = await self._graph_search(
                plan,
                candidate_k,
                document_ids,
                hops=hops,
            )
            trace = [f"graph_candidates:{len(contexts)}"]
        except Exception as exc:
            contexts = []
            trace = [f"graph_search_failed:{type(exc).__name__}", "graph_candidates:0"]
        return RetrievalResult(
            plan,
            contexts,
            (time.perf_counter() - started) * 1000,
            trace,
            candidate_count=len(contexts),
        )

    async def finalize_candidates(
        self,
        question: str,
        *,
        plan: QueryPlan,
        mode: RetrievalMode | str,
        vector: RetrievalResult,
        graph: RetrievalResult,
        top_k: int,
        document_ids: list[str] | None = None,
    ) -> RetrievalResult:
        mode = RetrievalMode(mode)
        started = time.perf_counter()
        if mode == RetrievalMode.VECTOR:
            candidates = self._deduplicate(vector.contexts)
        elif mode == RetrievalMode.GRAPH:
            candidates = self._deduplicate(graph.contexts)
        else:
            graph_weight = self._graph_weight(plan, graph.contexts)
            candidates = self._rrf_fuse(
                vector.contexts,
                graph.contexts,
                graph_weight=graph_weight,
            )
        candidate_count = len(candidates)
        trace = [*vector.trace, *graph.trace]
        if mode == RetrievalMode.HYBRID:
            trace.append(f"graph_weight:{self._graph_weight(plan, graph.contexts):.2f}")
        trace.append(f"rrf_candidates:{candidate_count}")
        if not candidates:
            return RetrievalResult(
                plan,
                [],
                max(vector.retrieval_ms, graph.retrieval_ms),
                [*trace, "selected_contexts:0"],
                embedding_ms=vector.embedding_ms,
                candidate_count=0,
                query_vectors=vector.query_vectors,
                document_ids=list(document_ids or []),
            )
        doc_ranked, doc_rerank_ms, doc_error = await self._rerank_contexts(
            question,
            plan,
            candidates[: settings.rerank_candidate_k],
        )
        trace.extend(
            [
                f"rerank_docs:{len(doc_ranked)}",
                f"rerank_docs_slots:{self._rerank_slot_count(plan)}",
            ]
        )
        if doc_error:
            trace.append(f"rerank_docs_fallback:{doc_error}")
        selected_docs = self._select_document_ids(doc_ranked, settings.rerank_top_documents)
        trace.append(f"selected_documents:{len(selected_docs)}")
        if mode == RetrievalMode.GRAPH:
            contexts = self._pack_contexts(doc_ranked, plan, top_k)
            total_ms = max(vector.retrieval_ms, graph.retrieval_ms) + (time.perf_counter() - started) * 1000
            return RetrievalResult(
                plan,
                contexts,
                total_ms,
                [*trace, "document_refine:skipped_graph_mode", f"selected_contexts:{len(contexts)}"],
                embedding_ms=vector.embedding_ms,
                rerank_ms=doc_rerank_ms,
                candidate_count=candidate_count,
                query_vectors=vector.query_vectors,
                document_ids=list(document_ids or []),
            )
        local_contexts = await self._refine_documents(
            plan,
            vector.query_vectors,
            selected_docs,
        )
        selected_initial = [context for context in candidates if context.doc_id in selected_docs]
        local_pool = self._deduplicate([*selected_initial, *local_contexts])[
            : settings.rerank_max_local_candidates
        ]
        trace.append(f"document_refine_candidates:{len(local_pool)}")
        chunk_ranked, chunk_rerank_ms, chunk_error = await self._rerank_contexts(
            question,
            plan,
            local_pool,
        )
        trace.extend(
            [
                f"rerank_chunks:{len(chunk_ranked)}",
                f"rerank_chunks_slots:{self._rerank_slot_count(plan)}",
            ]
        )
        if chunk_error:
            trace.append(f"rerank_chunks_fallback:{chunk_error}")
        contexts = self._pack_contexts(chunk_ranked, plan, top_k)
        total_ms = max(vector.retrieval_ms, graph.retrieval_ms) + (time.perf_counter() - started) * 1000
        return RetrievalResult(
            plan,
            contexts,
            total_ms,
            [*trace, f"selected_contexts:{len(contexts)}"],
            embedding_ms=vector.embedding_ms,
            rerank_ms=doc_rerank_ms + chunk_rerank_ms,
            candidate_count=candidate_count,
            query_vectors=vector.query_vectors,
            document_ids=list(document_ids or []),
        )

    async def supplement_evidence(
        self,
        question: str,
        retrieval: RetrievalResult,
        *,
        top_k: int = 10,
    ) -> RetrievalResult:
        """Run one deeper vector lookup for an incomplete temporal comparison.

        Existing query vectors are reused, so this recovery path never makes an
        additional embedding request and keeps the original document filter.
        """
        if any(item.startswith("temporal_supplement:") for item in retrieval.trace):
            return retrieval
        search_vectors = getattr(self.vector_store, "search_multimodal_by_vectors", None)
        if search_vectors is None or not retrieval.query_vectors:
            retrieval.trace.append("temporal_supplement:skipped_no_vectors")
            return retrieval

        started = time.perf_counter()
        values = await asyncio.gather(
            *(
                search_vectors(
                    vector,
                    top_k=settings.rerank_max_local_candidates,
                    document_ids=retrieval.document_ids or None,
                )
                for vector in retrieval.query_vectors
            )
        )
        queries = retrieval.plan.queries or [slot.query for slot in retrieval.plan.evidence_slots]
        slot_ids = self._query_slot_ids(retrieval.plan, queries)
        supplemental = self._merge_query_results([value[0] for value in values], slot_ids)
        pool = self._deduplicate([*retrieval.contexts, *supplemental])[: settings.rerank_max_local_candidates]
        ranked, rerank_ms, error = await self._rerank_contexts(
            question,
            retrieval.plan,
            pool,
        )
        contexts = self._pack_contexts(ranked, retrieval.plan, top_k)
        elapsed_ms = (time.perf_counter() - started) * 1000
        trace = [
            *retrieval.trace,
            "temporal_supplement:applied",
            f"temporal_supplement_candidates:{len(pool)}",
            f"temporal_supplement_selected:{len(contexts)}",
        ]
        if error:
            trace.append(f"temporal_supplement_rerank_fallback:{error}")
        return RetrievalResult(
            retrieval.plan,
            contexts,
            retrieval.retrieval_ms + elapsed_ms,
            trace,
            planning_ms=retrieval.planning_ms,
            embedding_ms=retrieval.embedding_ms,
            rerank_ms=retrieval.rerank_ms + rerank_ms,
            candidate_count=max(retrieval.candidate_count, len(pool)),
            query_vectors=retrieval.query_vectors,
            document_ids=retrieval.document_ids,
        )

    async def _vector_search_queries(
        self,
        plan: QueryPlan,
        top_k: int,
        document_ids: list[str] | None,
    ) -> tuple[list[GraphRAGContext], dict[str, int], float, list[MultimodalQueryVector]]:
        started = time.perf_counter()
        queries = plan.queries or [slot.query for slot in plan.evidence_slots]
        slot_ids = self._query_slot_ids(plan, queries)
        embed = getattr(self.vector_store, "embed_multimodal_query", None)
        search_vectors = getattr(self.vector_store, "search_multimodal_by_vectors", None)
        result_sets: list[list[tuple[dict[str, Any], float]]]
        counts: dict[str, int] = {}
        query_vectors: list[MultimodalQueryVector] = []
        if embed is not None and search_vectors is not None:
            query_vectors = list(await asyncio.gather(*(embed(query) for query in queries)))
            values = await asyncio.gather(
                *(search_vectors(vector, top_k=top_k, document_ids=document_ids) for vector in query_vectors)
            )
            result_sets = [value[0] for value in values]
            for _results, batch_counts in values:
                for modality, count in batch_counts.items():
                    counts[modality] = counts.get(modality, 0) + count
        else:
            search_multimodal = getattr(self.vector_store, "search_multimodal", None)
            if search_multimodal is not None:
                values = await asyncio.gather(
                    *(search_multimodal(query, top_k=top_k, document_ids=document_ids) for query in queries)
                )
                result_sets = [value[0] for value in values]
                for _results, batch_counts in values:
                    for modality, count in batch_counts.items():
                        counts[modality] = counts.get(modality, 0) + count
            else:
                raw_vectors = await asyncio.gather(
                    *(self.vector_store.embed_query(query) for query in queries)
                )
                raw_sets = await self.vector_store.search_many_by_vectors(
                    list(raw_vectors),
                    top_k=top_k,
                    document_ids=document_ids,
                )
                result_sets = list(raw_sets)
                counts = {"legacy": sum(len(value) for value in result_sets)}
        contexts = self._merge_query_results(result_sets, slot_ids)
        return contexts, counts, (time.perf_counter() - started) * 1000, query_vectors

    async def _refine_documents(
        self,
        plan: QueryPlan,
        query_vectors: list[MultimodalQueryVector],
        document_ids: list[str],
    ) -> list[GraphRAGContext]:
        if not document_ids or not query_vectors:
            return []
        search_vectors = getattr(self.vector_store, "search_multimodal_by_vectors", None)
        if search_vectors is None:
            return []
        per_query = min(
            settings.rerank_max_local_candidates,
            settings.rerank_local_candidates_per_query * len(document_ids),
        )
        values = await asyncio.gather(
            *(search_vectors(vector, top_k=per_query, document_ids=document_ids) for vector in query_vectors)
        )
        slot_ids = self._query_slot_ids(plan, plan.queries)
        return self._merge_query_results([value[0] for value in values], slot_ids)

    @staticmethod
    def _query_slot_ids(plan: QueryPlan, queries: list[str]) -> list[str]:
        slots_by_query = {slot.query.casefold(): slot.slot_id for slot in plan.evidence_slots}
        return [slots_by_query.get(query.casefold(), f"query_{index}") for index, query in enumerate(queries)]

    @classmethod
    def _merge_query_results(
        cls,
        result_sets: list[list[tuple[dict[str, Any], float]]],
        slot_ids: list[str],
    ) -> list[GraphRAGContext]:
        combined: dict[str, dict[str, Any]] = {}
        for query_index, results in enumerate(result_sets):
            seen: set[str] = set()
            for rank, (document, _native_score) in enumerate(results, start=1):
                metadata = dict(document.get("metadata") or {})
                key = str(
                    document.get("chunk_id")
                    or metadata.get("chunk_id")
                    or f"{document.get('doc_id')}:{document.get('content')}"
                )
                if key in seen:
                    continue
                seen.add(key)
                context = cls._document_to_context(document, 0.0)
                entry = combined.setdefault(
                    key,
                    {"context": context, "rrf": 0.0, "slots": set()},
                )
                entry["rrf"] += 1.0 / (settings.rrf_constant + rank)
                if query_index < len(slot_ids):
                    entry["slots"].add(slot_ids[query_index])
        maximum = max((float(value["rrf"]) for value in combined.values()), default=0.0)
        output: list[GraphRAGContext] = []
        for value in combined.values():
            context = value["context"]
            score = float(value["rrf"]) / maximum if maximum else 0.0
            output.append(
                GraphRAGContext(
                    context.chunk_id,
                    context.doc_id,
                    context.content,
                    context.source,
                    "vector",
                    score,
                    {
                        **context.metadata,
                        "matched_slot_ids": sorted(value["slots"]),
                        "rrf_score": score,
                        "retrieval_sources": ["vector"],
                    },
                )
            )
        return sorted(output, key=lambda item: (-item.score, item.chunk_id))

    @staticmethod
    def _document_to_context(document: dict[str, Any], score: float) -> GraphRAGContext:
        metadata = dict(document.get("metadata") or {})
        return GraphRAGContext(
            chunk_id=str(document.get("chunk_id") or metadata.get("chunk_id", "")),
            doc_id=str(document.get("doc_id") or metadata.get("doc_id", "")),
            content=str(document.get("content", "")),
            source=str(document.get("source", "")),
            source_type="vector",
            score=float(score),
            metadata=metadata,
        )

    async def _graph_search(
        self,
        plan: QueryPlan,
        top_k: int,
        document_ids: list[str] | None,
        *,
        hops: int,
    ) -> list[GraphRAGContext]:
        records = await self.knowledge_graph.graph_search(
            plan.entities,
            hops=hops,
            document_ids=document_ids,
            limit=top_k,
        )
        contexts: list[GraphRAGContext] = []
        for record in records:
            record_hops = int(record.get("hops", 0))
            matched = max(1, int(record.get("matched_entity_count", 1)))
            context = GraphRAGContext(
                chunk_id=str(record.get("chunk_id", "")),
                doc_id=str(record.get("doc_id", "")),
                content=str(record.get("content", "")),
                source=str(record.get("source", "")),
                source_type="graph",
                score=max(0.0, min(1.0, 0.5 + matched * 0.1 - record_hops * 0.1)),
                metadata={
                    "file_name": record.get("file_name", ""),
                    "page": record.get("page"),
                    "sheet": record.get("sheet"),
                    "entities": record.get("entities", []),
                    "predicates": record.get("predicates", []),
                    "hops": record_hops,
                    "matched_entity_count": matched,
                    "modality": record.get("modality", "text"),
                    "retrieval_sources": ["graph"],
                },
            )
            relevance = self._graph_lexical_relevance(context, plan)
            contexts.append(
                GraphRAGContext(
                    context.chunk_id,
                    context.doc_id,
                    context.content,
                    context.source,
                    context.source_type,
                    min(1.0, context.score * 0.7 + relevance * 0.3),
                    {**context.metadata, "graph_lexical_relevance": relevance},
                )
            )
        return sorted(contexts, key=lambda item: (-item.score, item.chunk_id))

    @staticmethod
    def _graph_lexical_relevance(context: GraphRAGContext, plan: QueryPlan) -> float:
        query = " ".join([*plan.queries, *plan.entities]).casefold()
        haystack = " ".join(
            [
                context.content,
                " ".join(str(value) for value in context.metadata.get("predicates", [])),
                " ".join(str(value) for value in context.metadata.get("entities", [])),
            ]
        ).casefold()
        query_tokens = {value for value in re.findall(r"[\w-]+", query) if len(value) > 2}
        if not query_tokens:
            return 0.0
        haystack_tokens = set(re.findall(r"[\w-]+", haystack))
        return len(query_tokens & haystack_tokens) / len(query_tokens)

    @staticmethod
    def _graph_weight(plan: QueryPlan, contexts: list[GraphRAGContext]) -> float:
        if not contexts or not plan.entities:
            return 0.35
        if plan.intent in {"comparative", "temporal", "analytical"} and len(plan.entities) >= 2:
            return 1.0
        return settings.hybrid_graph_weight

    async def _rerank_contexts(
        self,
        question: str,
        plan: QueryPlan,
        contexts: list[GraphRAGContext],
    ) -> tuple[list[GraphRAGContext], float, str]:
        if not contexts:
            return [], 0.0, ""
        if self.reranker is None or not settings.rerank_enabled:
            return sorted(contexts, key=lambda item: (-item.score, item.chunk_id)), 0.0, "disabled"
        started = time.perf_counter()
        documents = [self._rerank_document(context) for context in contexts]
        slots = [(slot.slot_id, slot.query) for slot in plan.evidence_slots if slot.required]
        if not slots:
            slots = [("question", question)]
        results = await asyncio.gather(
            *(self.reranker.rerank(query, documents, top_n=len(documents)) for _slot_id, query in slots),
            return_exceptions=True,
        )
        failures = [value for value in results if isinstance(value, BaseException)]
        for _failure in failures:
            await self.reranker.record_fallback()
        successful = [
            (slot_id, value)
            for (slot_id, _query), value in zip(slots, results, strict=True)
            if not isinstance(value, BaseException)
        ]
        errors = ",".join(sorted({type(value).__name__ for value in failures}))
        if not successful:
            return (
                sorted(contexts, key=lambda item: (-item.score, item.chunk_id)),
                (time.perf_counter() - started) * 1000,
                errors or "RerankError",
            )

        fused: dict[int, float] = {
            index: settings.rerank_base_rrf_weight / (settings.rrf_constant + rank)
            for rank, index in enumerate(range(len(contexts)), start=1)
        }
        raw_scores: dict[int, dict[str, float]] = {index: {} for index in range(len(contexts))}
        matched_slots: dict[int, set[str]] = {index: set() for index in range(len(contexts))}
        for slot_id, scores in successful:
            for rank, value in enumerate(scores, start=1):
                if not 0 <= value.index < len(contexts):
                    continue
                fused[value.index] += settings.rerank_slot_rrf_weight / (settings.rrf_constant + rank)
                raw_scores[value.index][slot_id] = value.score
                if rank <= settings.rerank_slot_match_top_n:
                    matched_slots[value.index].add(slot_id)

        maximum = max(fused.values(), default=0.0)
        output: list[GraphRAGContext] = []
        for index, original in enumerate(contexts):
            score = fused[index] / maximum if maximum else 0.0
            slot_scores = raw_scores[index]
            output.append(
                GraphRAGContext(
                    original.chunk_id,
                    original.doc_id,
                    original.content,
                    original.source,
                    original.source_type,
                    score,
                    {
                        **original.metadata,
                        "rrf_score": original.metadata.get("rrf_score", original.score),
                        "rerank_score": max(slot_scores.values(), default=0.0),
                        "rerank_slot_scores": slot_scores,
                        "rerank_slot_ids": sorted(matched_slots[index]),
                        "rerank_fused_score": score,
                    },
                )
            )
        output.sort(key=lambda item: (-item.score, item.chunk_id))
        return output, (time.perf_counter() - started) * 1000, errors

    @staticmethod
    def _rerank_document(context: GraphRAGContext) -> str:
        file_name = context.metadata.get("file_name") or context.source
        page = context.metadata.get("page") or ""
        sheet = context.metadata.get("sheet") or ""
        return f"file={file_name} page={page} sheet={sheet}\n{context.content}"[:8000]

    @staticmethod
    def _rerank_slot_count(plan: QueryPlan) -> int:
        return max(1, sum(slot.required for slot in plan.evidence_slots))

    @staticmethod
    def _select_document_ids(contexts: list[GraphRAGContext], limit: int) -> list[str]:
        output: list[str] = []
        for context in contexts:
            key = context.doc_id or context.chunk_id
            if key and key not in output:
                output.append(key)
            if len(output) >= limit:
                break
        return output

    @staticmethod
    def _deduplicate(contexts: list[GraphRAGContext]) -> list[GraphRAGContext]:
        values: dict[str, GraphRAGContext] = {}
        for context in contexts:
            key = context.chunk_id or f"{context.doc_id}:{context.content[:80]}"
            current = values.get(key)
            if current is None:
                values[key] = context
                continue
            metadata = {**current.metadata, **context.metadata}
            slots = set(current.metadata.get("matched_slot_ids", [])) | set(
                context.metadata.get("matched_slot_ids", [])
            )
            sources = set(current.metadata.get("retrieval_sources", [current.source_type])) | set(
                context.metadata.get("retrieval_sources", [context.source_type])
            )
            metadata["matched_slot_ids"] = sorted(slots)
            metadata["retrieval_sources"] = sorted(sources)
            winner = context if context.score > current.score else current
            values[key] = GraphRAGContext(
                winner.chunk_id,
                winner.doc_id,
                winner.content,
                winner.source,
                "hybrid" if len(sources) > 1 else next(iter(sources)),
                max(current.score, context.score),
                metadata,
            )
        return sorted(values.values(), key=lambda item: (-item.score, item.chunk_id))

    @staticmethod
    def _rrf_fuse(
        vector_results: list[GraphRAGContext],
        graph_results: list[GraphRAGContext],
        constant: int | None = None,
        vector_weight: float | None = None,
        graph_weight: float | None = None,
    ) -> list[GraphRAGContext]:
        constant = settings.rrf_constant if constant is None else constant
        vector_weight = settings.hybrid_vector_weight if vector_weight is None else vector_weight
        graph_weight = settings.hybrid_graph_weight if graph_weight is None else graph_weight
        combined: dict[str, dict[str, Any]] = {}
        for source_name, results, weight in (
            ("vector", vector_results, vector_weight),
            ("graph", graph_results, graph_weight),
        ):
            deduplicated: dict[str, GraphRAGContext] = {}
            for context in sorted(results, key=lambda item: item.score, reverse=True):
                key = context.chunk_id or f"{context.doc_id}:{context.content[:80]}"
                if key not in deduplicated or context.score > deduplicated[key].score:
                    deduplicated[key] = context
            for rank, (key, context) in enumerate(deduplicated.items(), start=1):
                entry = combined.setdefault(
                    key,
                    {"context": context, "rrf": 0.0, "sources": set(), "slots": set()},
                )
                entry["rrf"] += weight / (constant + rank)
                entry["sources"].add(source_name)
                entry["slots"].update(context.metadata.get("matched_slot_ids", []))
                if len(context.content) > len(entry["context"].content):
                    entry["context"] = context
        maximum = max((float(entry["rrf"]) for entry in combined.values()), default=0.0)
        output: list[GraphRAGContext] = []
        for entry in combined.values():
            original = entry["context"]
            score = float(entry["rrf"]) / maximum if maximum else 0.0
            sources = entry["sources"]
            output.append(
                GraphRAGContext(
                    original.chunk_id,
                    original.doc_id,
                    original.content,
                    original.source,
                    "hybrid" if len(sources) > 1 else next(iter(sources)),
                    min(1.0, score),
                    {
                        **original.metadata,
                        "retrieval_sources": sorted(sources),
                        "matched_slot_ids": sorted(entry["slots"]),
                        "rrf_score": min(1.0, score),
                    },
                )
            )
        return sorted(
            output,
            key=lambda item: (
                -item.score,
                -len(item.metadata.get("retrieval_sources", [])),
                0 if "vector" in item.metadata.get("retrieval_sources", []) else 1,
                item.chunk_id,
            ),
        )

    @staticmethod
    def _pack_contexts(
        contexts: list[GraphRAGContext],
        plan: QueryPlan,
        top_k: int,
    ) -> list[GraphRAGContext]:
        selected: list[GraphRAGContext] = []
        selected_ids: set[str] = set()
        doc_counts: dict[str, int] = {}

        def add(context: GraphRAGContext) -> bool:
            key = context.chunk_id or f"{context.doc_id}:{context.content[:80]}"
            doc_key = context.doc_id or key
            if key in selected_ids or doc_counts.get(doc_key, 0) >= settings.rerank_max_chunks_per_document:
                return False
            selected.append(context)
            selected_ids.add(key)
            doc_counts[doc_key] = doc_counts.get(doc_key, 0) + 1
            return True

        for slot in (value for value in plan.evidence_slots if value.required):
            matching = []
            for context in contexts:
                matched = context.metadata.get("rerank_slot_ids") or context.metadata.get(
                    "matched_slot_ids", []
                )
                if slot.slot_id in matched:
                    matching.append(context)
            unseen = [
                context for context in matching if (context.doc_id or context.chunk_id) not in doc_counts
            ]
            for context in [*unseen, *matching]:
                if add(context):
                    break
            if len(selected) >= top_k:
                return selected
        for context in contexts:
            add(context)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _select_diverse(
        contexts: list[GraphRAGContext],
        top_k: int,
        *,
        max_per_document: int,
    ) -> list[GraphRAGContext]:
        selected: list[GraphRAGContext] = []
        deferred: list[GraphRAGContext] = []
        document_counts: dict[str, int] = {}
        for context in contexts:
            key = context.doc_id or context.chunk_id
            if document_counts.get(key, 0) >= max_per_document:
                deferred.append(context)
                continue
            selected.append(context)
            document_counts[key] = document_counts.get(key, 0) + 1
            if len(selected) >= top_k:
                return selected
        for context in deferred:
            selected.append(context)
            if len(selected) >= top_k:
                break
        return selected

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
            )
        return str(content)

    @staticmethod
    def _json_object(raw: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("missing JSON object")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
