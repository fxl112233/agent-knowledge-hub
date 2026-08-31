"""Question-answering agent backed by the unified GraphRAG pipeline."""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from config import settings
from services.comparison import ComparisonVerificationTool
from services.graph_rag import (
    AnswerType,
    GraphRAGContext,
    GraphRAGPipeline,
    RetrievalMode,
    TargetRelation,
)
from services.model_usage import ensure_metered
from services.temporal import TemporalComparisonTool
from utils.calculator import safe_calculate


class QueryIntent(StrEnum):
    FACTOID = "factoid"
    ANALYTICAL = "analytical"
    COMPARATIVE = "comparative"
    PROCEDURAL = "procedural"
    EXPLORATORY = "exploratory"
    TEMPORAL = "temporal"


@dataclass(slots=True)
class Citation:
    doc_id: str
    chunk_id: str
    file_name: str
    page: int | None
    sheet: str | None
    quote: str
    score: float
    retrieval_type: str
    modality: str = "text"


@dataclass(slots=True)
class QAResult:
    question: str
    answer: str
    answerable: bool
    evidence_score: float
    intent: QueryIntent
    citations: list[Citation]
    contexts: list[GraphRAGContext]
    retrieval_ms: float
    generation_ms: float
    total_ms: float
    usage: dict[str, int] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)

    @property
    def confidence(self) -> float:
        """Backward-compatible alias; this is an evidence score, not a probability."""
        return self.evidence_score

    @property
    def reasoning_steps(self) -> list[str]:
        return self.trace


class AnswerPayload(BaseModel):
    answer: str
    answerable: bool
    citation_ids: list[str] = Field(default_factory=list)
    calculation: str = ""


ANSWER_PROMPT = """你是企业文档问答助手。只能根据提供的证据回答。
规则：
1. 每个事实必须引用证据块 ID；禁止使用模型记忆补充事实。
2. 证据不足时 answerable=false，明确说明无法从文档确定。
3. 数值推导时在 calculation 中给出只含数字和 + - * / ( ) 的表达式。
4. answer 必须是能直接回答问题的最短答案：所有是非题（包括一致、变化、先后判断）只写 Yes/No；
   数值题只写按题意舍入后的数值；实体题只写名称；
   不要添加“约为”“根据文档”“计算结果”等解释。
5. 只返回 JSON：
{"answer":"简洁答案","answerable":true,"citation_ids":["chunk-id"],"calculation":""}
"""


class QAAgent:
    def __init__(
        self,
        vector_store: Any | None = None,
        knowledge_graph: Any | None = None,
        *,
        pipeline: GraphRAGPipeline | None = None,
        llm: Any | None = None,
        reranker: Any | None = None,
        temporal_tool: TemporalComparisonTool | None = None,
        comparison_tool: ComparisonVerificationTool | None = None,
    ) -> None:
        self.llm = ensure_metered(llm or self._build_llm())
        self.pipeline = pipeline or GraphRAGPipeline(
            vector_store,
            knowledge_graph,
            llm=self.llm,
            reranker=reranker,
        )
        self.temporal_tool = temporal_tool or (
            TemporalComparisonTool(self.llm) if self.llm is not None else None
        )
        self.comparison_tool = comparison_tool or (
            ComparisonVerificationTool(self.llm) if self.llm is not None else None
        )

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

    async def answer(
        self,
        question: str,
        *,
        top_k: int | None = None,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        document_ids: list[str] | None = None,
    ) -> QAResult:
        started = time.perf_counter()
        top_k = max(1, min(top_k or settings.default_top_k, settings.max_top_k))
        retrieval = await self.pipeline.retrieve(
            question,
            top_k=top_k,
            mode=mode,
            document_ids=document_ids,
        )
        return await self.answer_from_retrieval(question, retrieval, started=started)

    async def answer_from_retrieval(
        self,
        question: str,
        retrieval: Any,
        *,
        started: float | None = None,
        use_temporal: bool = True,
    ) -> QAResult:
        started = started or time.perf_counter()
        intent = self._intent(retrieval.plan.intent)
        if not retrieval.contexts:
            total_ms = (time.perf_counter() - started) * 1000
            return QAResult(
                question=question,
                answer="现有知识库中没有足够证据回答该问题。",
                answerable=False,
                evidence_score=0.0,
                intent=intent,
                citations=[],
                contexts=[],
                retrieval_ms=retrieval.retrieval_ms,
                generation_ms=0.0,
                total_ms=total_ms,
                trace=retrieval.trace + ["answer:abstained_no_context"],
            )
        if self.llm is None:
            raise RuntimeError("LLM_API_KEY and LLM_MODEL are required for answer generation")
        if (
            use_temporal
            and settings.comparison_tool_enabled
            and self._should_use_comparison(retrieval.plan)
        ):
            try:
                return await self.answer_comparison_from_retrieval(
                    question,
                    retrieval,
                    started=started,
                )
            except Exception as exc:
                retrieval.trace.append(f"comparison:fallback:{type(exc).__name__}")
        if use_temporal and self._should_use_temporal(retrieval.plan):
            try:
                return await self.answer_temporal_from_retrieval(
                    question,
                    retrieval,
                    started=started,
                )
            except Exception as exc:
                retrieval.trace.append(f"temporal:fallback:{type(exc).__name__}")

        generation_started = time.perf_counter()
        generation_contexts = self._select_generation_contexts(
            retrieval.contexts,
            retrieval.plan,
        )
        payload, usage = await self._generate(question, generation_contexts)
        generation_ms = (time.perf_counter() - generation_started) * 1000
        bound_citations = self._bind_required_slot_citations(
            payload,
            generation_contexts,
            retrieval.plan,
        )
        valid_contexts = {context.chunk_id: context for context in generation_contexts}
        citation_ids = list(
            dict.fromkeys(identifier for identifier in payload.citation_ids if identifier in valid_contexts)
        )
        trace = [*retrieval.trace, f"generation_contexts:{len(generation_contexts)}"]
        if bound_citations:
            trace.append(f"citations:slot_bound:{bound_citations}")

        if (
            payload.answerable
            and retrieval.plan.answer_type == AnswerType.BOOLEAN
            and payload.answer.strip().casefold() not in {"yes", "no"}
        ):
            payload.answerable = False
            payload.answer = "模型没有按 Yes/No 契约回答，因此无法确认答案。"
            trace.append("answer:abstained_boolean_contract")

        if payload.answerable and not citation_ids:
            payload.answerable = False
            payload.answer = "模型未能提供可验证的文档引用，因此无法确认答案。"
            trace.append("answer:abstained_invalid_citations")

        if payload.calculation:
            try:
                safe_calculate(payload.calculation)
                trace.append("calculator:verified")
            except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
                trace.append("calculator:rejected_expression")

        citations = [self._citation(valid_contexts[identifier]) for identifier in citation_ids]
        evidence_score = 0.0
        if citations:
            evidence_score = sum(item.score for item in citations) / len(citations)
        trace.append("answer:generated" if payload.answerable else "answer:abstained")
        return QAResult(
            question=question,
            answer=payload.answer,
            answerable=payload.answerable,
            evidence_score=round(min(1.0, evidence_score), 4),
            intent=intent,
            citations=citations,
            contexts=retrieval.contexts,
            retrieval_ms=round(retrieval.retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            total_ms=round((time.perf_counter() - started) * 1000, 3),
            usage=usage,
            trace=trace,
        )

    async def answer_temporal_from_retrieval(
        self,
        question: str,
        retrieval: Any,
        *,
        started: float | None = None,
    ) -> QAResult:
        started = started or time.perf_counter()
        if self.temporal_tool is None:
            raise RuntimeError("temporal comparison tool is unavailable")
        generation_started = time.perf_counter()
        temporal = await self.temporal_tool.compare(question, retrieval.plan, retrieval.contexts)
        usage = dict(temporal.usage)
        recoverable = {
            "temporal:missing_slot_evidence",
            "temporal:unknown",
            "temporal:invalid_citation",
        }
        supplement = getattr(self.pipeline, "supplement_evidence", None)
        if not temporal.answerable and recoverable.intersection(temporal.trace) and supplement:
            retrieval = await supplement(
                question,
                retrieval,
                top_k=max(settings.default_top_k, len(retrieval.contexts)),
            )
            if "temporal_supplement:applied" in retrieval.trace:
                retried = await self.temporal_tool.compare(
                    question,
                    retrieval.plan,
                    retrieval.contexts,
                )
                usage = self._merge_usage(usage, retried.usage)
                temporal = retried
        generation_ms = (time.perf_counter() - generation_started) * 1000
        valid_contexts = {context.chunk_id: context for context in retrieval.contexts}
        citation_ids = list(
            dict.fromkeys(identifier for identifier in temporal.citation_ids if identifier in valid_contexts)
        )
        answerable = temporal.answerable and bool(citation_ids)
        answer = temporal.answer if answerable or not temporal.answerable else "无法从文档确定"
        citations = [self._citation(valid_contexts[identifier]) for identifier in citation_ids]
        evidence_score = sum(item.score for item in citations) / len(citations) if citations else 0.0
        trace = [*retrieval.trace, "temporal:selected", *temporal.trace]
        trace.append("answer:generated" if answerable else "answer:abstained")
        return QAResult(
            question=question,
            answer=answer,
            answerable=answerable,
            evidence_score=round(min(1.0, evidence_score), 4),
            intent=QueryIntent.TEMPORAL,
            citations=citations,
            contexts=retrieval.contexts,
            retrieval_ms=round(retrieval.retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            total_ms=round((time.perf_counter() - started) * 1000, 3),
            usage=usage,
            trace=trace,
        )

    async def answer_comparison_from_retrieval(
        self,
        question: str,
        retrieval: Any,
        *,
        started: float | None = None,
    ) -> QAResult:
        started = started or time.perf_counter()
        if self.comparison_tool is None:
            raise RuntimeError("comparison verification tool is unavailable")
        generation_started = time.perf_counter()
        contexts = self._select_generation_contexts(retrieval.contexts, retrieval.plan)
        comparison = await self.comparison_tool.compare(question, retrieval.plan, contexts)
        usage = dict(comparison.usage)
        recoverable = {
            "comparison:missing_slot_evidence",
            "comparison:unknown_slot",
            "comparison:unknown",
            "comparison:invalid_citation",
        }
        supplement = getattr(self.pipeline, "supplement_evidence", None)
        if not comparison.answerable and recoverable.intersection(comparison.trace) and supplement:
            retrieval = await supplement(
                question,
                retrieval,
                top_k=max(settings.default_top_k, len(retrieval.contexts)),
            )
            if "temporal_supplement:applied" in retrieval.trace:
                contexts = self._select_generation_contexts(retrieval.contexts, retrieval.plan)
                retried = await self.comparison_tool.compare(question, retrieval.plan, contexts)
                usage = self._merge_usage(usage, retried.usage)
                comparison = retried
        generation_ms = (time.perf_counter() - generation_started) * 1000
        valid_contexts = {context.chunk_id: context for context in contexts}
        citation_ids = list(
            dict.fromkeys(
                identifier for identifier in comparison.citation_ids if identifier in valid_contexts
            )
        )
        answerable = comparison.answerable and bool(citation_ids)
        citations = [self._citation(valid_contexts[identifier]) for identifier in citation_ids]
        evidence_score = sum(item.score for item in citations) / len(citations) if citations else 0.0
        trace = [
            *retrieval.trace,
            f"generation_contexts:{len(contexts)}",
            "comparison:selected",
            *comparison.trace,
            "answer:generated" if answerable else "answer:abstained",
        ]
        return QAResult(
            question=question,
            answer=comparison.answer,
            answerable=answerable,
            evidence_score=round(min(1.0, evidence_score), 4),
            intent=self._intent(retrieval.plan.intent),
            citations=citations,
            contexts=retrieval.contexts,
            retrieval_ms=round(retrieval.retrieval_ms, 3),
            generation_ms=round(generation_ms, 3),
            total_ms=round((time.perf_counter() - started) * 1000, 3),
            usage=usage,
            trace=trace,
        )

    @staticmethod
    def _should_use_temporal(plan: Any) -> bool:
        return bool(plan.target_relation != TargetRelation.NONE)

    @staticmethod
    def _should_use_comparison(plan: Any) -> bool:
        required = [slot for slot in plan.evidence_slots if slot.required]
        return bool(
            plan.answer_type == AnswerType.BOOLEAN
            and len(required) >= 2
            and plan.target_relation
            not in {TargetRelation.BEFORE, TargetRelation.AFTER, TargetRelation.SAME_TIME}
        )

    @staticmethod
    def _merge_usage(left: dict[str, int], right: dict[str, int]) -> dict[str, int]:
        return {key: int(left.get(key, 0)) + int(right.get(key, 0)) for key in set(left) | set(right)}

    @staticmethod
    def _context_slot_ids(context: GraphRAGContext) -> set[str]:
        return set(context.metadata.get("rerank_slot_ids") or context.metadata.get("matched_slot_ids", []))

    @classmethod
    def _select_generation_contexts(
        cls,
        contexts: list[GraphRAGContext],
        plan: Any,
    ) -> list[GraphRAGContext]:
        selected: list[GraphRAGContext] = []
        selected_ids: set[str] = set()

        def add(context: GraphRAGContext) -> None:
            key = context.chunk_id or f"{context.doc_id}:{context.content[:80]}"
            if key not in selected_ids and len(selected) < settings.answer_max_context_chunks:
                selected.append(context)
                selected_ids.add(key)

        for slot in (value for value in plan.evidence_slots if value.required):
            matches = [context for context in contexts if slot.slot_id in cls._context_slot_ids(context)]
            selected_docs = {context.doc_id for context in selected if context.doc_id}
            match = next(
                (context for context in matches if context.doc_id not in selected_docs),
                matches[0] if matches else None,
            )
            if match is not None:
                add(match)
        for context in contexts:
            add(context)

        output: list[GraphRAGContext] = []
        remaining = settings.answer_max_context_chars
        for context in selected:
            if remaining <= 0:
                break
            content = context.content[:remaining]
            if not content:
                continue
            output.append(
                GraphRAGContext(
                    context.chunk_id,
                    context.doc_id,
                    content,
                    context.source,
                    context.source_type,
                    context.score,
                    context.metadata,
                )
            )
            remaining -= len(content)
        return output

    @classmethod
    def _bind_required_slot_citations(
        cls,
        payload: AnswerPayload,
        contexts: list[GraphRAGContext],
        plan: Any,
    ) -> int:
        if not payload.answerable:
            return 0
        by_id = {context.chunk_id: context for context in contexts}
        citation_ids = list(dict.fromkeys(value for value in payload.citation_ids if value in by_id))
        added = 0
        assigned_ids: set[str] = set()
        for slot in (value for value in plan.evidence_slots if value.required):
            existing = next(
                (
                    value
                    for value in citation_ids
                    if value not in assigned_ids and slot.slot_id in cls._context_slot_ids(by_id[value])
                ),
                None,
            )
            if existing is not None:
                assigned_ids.add(existing)
                continue
            match = next(
                (
                    context
                    for context in contexts
                    if context.chunk_id not in assigned_ids and slot.slot_id in cls._context_slot_ids(context)
                ),
                None,
            )
            if match is not None:
                assigned_ids.add(match.chunk_id)
                if match.chunk_id not in citation_ids:
                    citation_ids.append(match.chunk_id)
                    added += 1
        payload.citation_ids = citation_ids
        return added

    async def _generate(
        self,
        question: str,
        contexts: list[GraphRAGContext],
    ) -> tuple[AnswerPayload, dict[str, int]]:
        evidence = "\n\n".join(
            f"[{context.chunk_id}] 文件={context.metadata.get('file_name') or context.source} "
            f"页={context.metadata.get('page', '')} 表={context.metadata.get('sheet', '')}\n{context.content}"
            for context in contexts
        )
        last_error = ""
        for attempt in range(settings.llm_max_retries + 1):
            correction = f"\n上次 JSON 无效：{last_error}" if last_error else ""
            try:
                response = await self.llm.ainvoke(
                    [
                        {"role": "system", "content": ANSWER_PROMPT},
                        {"role": "user", "content": f"证据：\n{evidence}\n\n问题：{question}{correction}"},
                    ]
                )
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"[:300]
                if attempt >= settings.llm_max_retries:
                    raise
                await asyncio.sleep(2**attempt)
                continue
            try:
                payload = AnswerPayload.model_validate(self._json_object(self._response_text(response)))
                return payload, self._usage(response)
            except (ValueError, json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)[:300]
                if attempt >= settings.llm_max_retries:
                    break
        return (
            AnswerPayload(
                answer="模型输出无法通过结构化校验，暂时无法确认答案。",
                answerable=False,
            ),
            {},
        )

    @staticmethod
    def _citation(context: GraphRAGContext) -> Citation:
        page = context.metadata.get("page")
        return Citation(
            doc_id=context.doc_id,
            chunk_id=context.chunk_id,
            file_name=str(context.metadata.get("file_name") or context.source),
            page=int(page) if page is not None else None,
            sheet=str(context.metadata["sheet"]) if context.metadata.get("sheet") is not None else None,
            quote=context.content[:300],
            score=round(context.score, 4),
            retrieval_type=context.source_type,
            modality=str(context.metadata.get("modality") or "text"),
        )

    @staticmethod
    def _intent(raw: str) -> QueryIntent:
        try:
            return QueryIntent(raw)
        except ValueError:
            return QueryIntent.FACTOID

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

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None) or {}
        if not usage:
            usage = (getattr(response, "response_metadata", None) or {}).get("token_usage", {})
        return {
            "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
