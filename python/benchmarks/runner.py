"""Resumable live benchmark runner.

The runner intentionally requires an explicit confirmation flag. Dry-run cost
estimation is available without starting infrastructure or calling a model.
"""

from __future__ import annotations

import asyncio
import json
import random
import shutil
import sys
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict
from functools import partial
from pathlib import Path
from typing import Any

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_update_agent import CDCEvent
from benchmarks.data import BENCHMARK_ROOT, LOCK_PATH, PREPARED_ROOT
from benchmarks.metrics import (
    binary_f1,
    character_error_rate,
    citation_precision_recall,
    evidence_metrics,
    exact_match,
    multihop_official_accuracy,
    normalize_answer,
    numeric_equal,
    paired_bootstrap_difference,
    percentile,
    token_f1,
)
from config import settings
from services.graph_rag import QueryPlan

RESULT_ROOT = BENCHMARK_ROOT / "results" / "reference"


async def _retry_async[T](
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
) -> tuple[T, int]:
    """Retry a complete paid operation while preserving resumable record boundaries."""
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            return await operation(), attempt
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= attempts:
                raise
            await asyncio.sleep(2**attempt)
    raise RuntimeError("retry loop exhausted") from last_error  # pragma: no cover


async def _run_answer_case(
    qa_agent: Any,
    question: str,
    mode: str,
    document_ids: list[str] | None,
    plan: QueryPlan,
    started: float,
) -> tuple[Any, Any]:
    retrieval = await qa_agent.pipeline.retrieve(
        question,
        top_k=10,
        mode=mode,
        document_ids=document_ids,
        plan=plan,
    )
    result = await qa_agent.answer_from_retrieval(question, retrieval, started=started)
    return retrieval, result


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class ResultCache:
    def __init__(self, path: Path, *, resume: bool = True) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not resume and path.exists():
            path.unlink()
        records = read_jsonl(path) if resume else []
        self.records = {
            str(record["cache_key"]): record for record in records if record.get("cache_key") is not None
        }
        self.keys = set(self.records)

    def has(self, key: str) -> bool:
        return key in self.keys

    def get(self, key: str) -> dict[str, Any] | None:
        return self.records.get(key)

    def append(self, key: str, record: dict[str, Any]) -> None:
        value = {"cache_key": key, **record}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")
            handle.flush()
        self.keys.add(key)
        self.records[key] = value


def _table_cell_f1(content: str, expected: list[str]) -> float:
    predicted = {
        normalize_answer(value)
        for value in __import__("re").split(r"[|\n]", content)
        if normalize_answer(value)
    }
    references = {normalize_answer(value) for value in expected if normalize_answer(value)}
    if not references:
        return 1.0
    overlap = len(predicted & references)
    if not predicted or not overlap:
        return 0.0
    precision, recall = overlap / len(predicted), overlap / len(references)
    return 2 * precision * recall / (precision + recall)


async def run_parse(*, limit: int | None = None, resume: bool = True) -> None:
    root = PREPARED_ROOT / "akh_parse_70"
    records = read_jsonl(root / "references.jsonl")
    if limit:
        records = records[:limit]
    cache = ResultCache(RESULT_ROOT / "parse.jsonl", resume=resume)
    parser = DocParserAgent()
    for record in records:
        key = record["sample_id"]
        if cache.has(key):
            continue
        started = time.perf_counter()
        try:
            chunks = await parser.parse(str(root / record["path"]), doc_id=key)
            content = "\n".join(chunk.content for chunk in chunks)
            cache.append(
                key,
                {
                    "sample_id": key,
                    "category": record["category"],
                    "condition": record["condition"],
                    "success": True,
                    "text_f1": token_f1(content, record["expected_text"]),
                    "cer": character_error_rate(content, record["expected_text"]),
                    "table_cell_f1": _table_cell_f1(content, record["expected_cells"]),
                    "chunks": len(chunks),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )
        except Exception as exc:
            cache.append(
                key,
                {
                    "sample_id": key,
                    "category": record["category"],
                    "condition": record["condition"],
                    "success": False,
                    "error": str(exc),
                    "latency_ms": (time.perf_counter() - started) * 1000,
                },
            )


class LiveRuntime:
    def __init__(self) -> None:
        self.container: Any = None
        self._consumer_setting = settings.enable_kafka_consumer

    async def __aenter__(self) -> LiveRuntime:
        from api.main import AppContainer

        settings.enable_kafka_consumer = False
        self.container = AppContainer()
        await self.container.init()
        return self

    async def __aexit__(self, *_args: Any) -> None:
        await self.container.close()
        settings.enable_kafka_consumer = self._consumer_setting

    async def ingest_documents(
        self,
        dataset_root: Path,
        *,
        limit: int | None = None,
        selected_ids: set[str] | None = None,
    ) -> dict[str, str]:
        records = read_jsonl(dataset_root / "documents.jsonl")
        if selected_ids is not None:
            records = [record for record in records if str(record["doc_id"]) in selected_ids]
        if limit:
            records = records[:limit]
        semaphore = asyncio.Semaphore(settings.batch_ingest_concurrency)

        async def ingest(record: dict[str, Any]) -> tuple[str, str]:
            source_id = str(record["doc_id"])
            path = dataset_root / str(record.get("path") or record.get("initial"))
            async with semaphore:
                for attempt in range(3):
                    try:
                        result = await self.container.ingestion.ingest(str(path), doc_id=source_id)
                        break
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2**attempt)
            return source_id, result.doc_id

        tasks = [asyncio.create_task(ingest(record)) for record in records]
        try:
            return dict(await asyncio.gather(*tasks))
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise


async def reset_benchmark_state(runtime: LiveRuntime, suites: set[str]) -> dict[str, int]:
    """Delete only prepared benchmark documents/events before a clean formal run."""
    dataset_names: set[str] = set()
    if suites & {"retrieval", "multihop"}:
        dataset_names.add("multihop_rag")
    if "tatqa" in suites:
        dataset_names.add("tat_qa")
    if "rgb" in suites:
        dataset_names.add("rgb")
    if "cdc" in suites:
        dataset_names.add("akh_cdc_300")

    document_ids = {
        str(record["doc_id"])
        for dataset in dataset_names
        for record in read_jsonl(PREPARED_ROOT / dataset / "documents.jsonl")
    }
    deleted_documents = 0
    for doc_id in sorted(document_ids):
        if runtime.container.catalog.get_document(doc_id) is not None:
            deleted_documents += await runtime.container.ingestion.delete(doc_id)

    event_ids: list[str] = []
    if "cdc" in suites:
        event_ids = [
            str(record["event_id"])
            for record in read_jsonl(PREPARED_ROOT / "akh_cdc_300" / "events-300.jsonl")
        ]
    deleted_events = runtime.container.catalog.delete_events(event_ids)
    return {"documents": deleted_documents, "events": deleted_events}


async def run_retrieval(
    runtime: LiveRuntime,
    *,
    limit: int | None = None,
    resume: bool = True,
    modes: tuple[str, ...] = ("vector", "graph", "hybrid"),
    output: Path | None = None,
) -> Path:
    invalid_modes = set(modes) - {"vector", "graph", "hybrid"}
    if invalid_modes:
        raise ValueError(f"unsupported retrieval modes: {sorted(invalid_modes)}")
    root = PREPARED_ROOT / "multihop_rag"
    cases = read_jsonl(root / "retrieval.jsonl")
    if limit:
        cases = cases[:limit]
    selected_ids = (
        {str(doc_id) for case in cases for doc_id in case.get("evidence_doc_ids", [])} if limit else None
    )
    await runtime.ingest_documents(root, selected_ids=selected_ids)
    cache = ResultCache(output or RESULT_ROOT / "retrieval.jsonl", resume=resume)
    for case in cases:
        pending_modes = [mode for mode in modes if not cache.has(f"{case['sample_id']}:{mode}")]
        if not pending_modes:
            continue
        plan = await runtime.container.qa_agent.pipeline.plan_query(case["question"])
        for mode in pending_modes:
            key = f"{case['sample_id']}:{mode}"
            result = await runtime.container.qa_agent.pipeline.retrieve(
                case["question"], top_k=10, mode=mode, plan=plan
            )
            retrieved_docs = list(dict.fromkeys(context.doc_id for context in result.contexts))
            scores = evidence_metrics(retrieved_docs, case.get("evidence_doc_ids", []))
            cache.append(
                key,
                {
                    "sample_id": case["sample_id"],
                    "mode": mode,
                    "expected_doc_ids": case.get("evidence_doc_ids", []),
                    "retrieved_doc_ids": retrieved_docs,
                    "retrieved_chunk_ids": [context.chunk_id for context in result.contexts],
                    "retrieval_ms": result.retrieval_ms,
                    **scores,
                },
            )
    return cache.path


async def _run_answer_cases(
    runtime: LiveRuntime,
    dataset: str,
    case_file: str,
    modes: tuple[str, ...],
    *,
    limit: int | None,
    resume: bool,
    output: Path | None = None,
    plan_output: Path | None = None,
    case_ids: set[str] | None = None,
) -> Path:
    root = PREPARED_ROOT / dataset
    cases = read_jsonl(root / case_file)
    if case_ids is not None:
        cases = [case for case in cases if str(case["sample_id"]) in case_ids]
    if limit:
        cases = cases[:limit]
    selected_ids: set[str] | None = None
    if limit or case_ids is not None:
        selected_ids = {
            str(doc_id)
            for case in cases
            for doc_id in ([case["doc_id"]] if case.get("doc_id") else case.get("evidence_doc_ids", []))
        }
    await runtime.ingest_documents(root, selected_ids=selected_ids)
    result_path = output or RESULT_ROOT / f"answers-{dataset}.jsonl"
    cache = ResultCache(result_path, resume=resume)
    plans = ResultCache(
        plan_output or result_path.with_name(f"{result_path.stem}-query-plans.jsonl"),
        resume=resume,
    )
    target_records = len(cases) * len(modes)
    completed_records = sum(cache.has(f"{case['sample_id']}:{mode}") for case in cases for mode in modes)
    print(
        json.dumps(
            {
                "event": "answer_experiment_started",
                "completed": completed_records,
                "total": target_records,
                "modes": list(modes),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    for case in cases:
        pending_modes = [mode for mode in modes if not cache.has(f"{case['sample_id']}:{mode}")]
        if not pending_modes:
            continue
        plan_record = plans.get(str(case["sample_id"]))
        if plan_record and isinstance(plan_record.get("plan"), dict):
            plan = QueryPlan.model_validate(plan_record["plan"])
        else:
            model = getattr(runtime.container.qa_agent, "llm", None)
            before_usage = getattr(model, "usage_snapshot", lambda: {})()
            planning_started = time.perf_counter()
            try:
                plan, plan_retries = await _retry_async(
                    partial(runtime.container.qa_agent.pipeline.plan_query, case["question"])
                )
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "answer_case_failed",
                            "sample_id": case["sample_id"],
                            "stage": "plan",
                            "error_type": type(exc).__name__,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            after_usage = getattr(model, "usage_snapshot", lambda: {})()
            plan_usage = {
                field: after_usage.get(field, 0) - before_usage.get(field, 0)
                for field in ("calls", "errors", "input_tokens", "output_tokens", "total_tokens")
            }
            plans.append(
                str(case["sample_id"]),
                {
                    "question": case["question"],
                    "plan": plan.model_dump(),
                    "planning_ms": round((time.perf_counter() - planning_started) * 1000, 3),
                    "retries": plan_retries,
                    "usage": plan_usage,
                },
            )
        for mode in pending_modes:
            key = f"{case['sample_id']}:{mode}"
            document_ids = [str(case["doc_id"])] if case.get("doc_id") else None
            started = time.perf_counter()
            try:
                operation = partial(
                    _run_answer_case,
                    runtime.container.qa_agent,
                    case["question"],
                    mode,
                    document_ids,
                    plan,
                    started,
                )
                (retrieval, result), operation_retries = await _retry_async(operation)
            except Exception as exc:
                print(
                    json.dumps(
                        {
                            "event": "answer_case_failed",
                            "sample_id": case["sample_id"],
                            "mode": mode,
                            "stage": "retrieve_or_answer",
                            "error_type": type(exc).__name__,
                        }
                    ),
                    file=sys.stderr,
                    flush=True,
                )
                continue
            expected_docs = case.get("evidence_doc_ids") or document_ids or []
            retrieved_docs = list(dict.fromkeys(context.doc_id for context in retrieval.contexts))
            retrieval_scores = evidence_metrics(retrieved_docs, expected_docs)
            retrieval_hits = len(set(retrieved_docs) & set(expected_docs))
            retrieval_precision = retrieval_hits / len(set(retrieved_docs)) if retrieved_docs else 0.0
            retrieval_recall = float(retrieval_scores["evidence_recall"])
            retrieval_denominator = retrieval_precision + retrieval_recall
            retrieval_f1 = (
                2 * retrieval_precision * retrieval_recall / retrieval_denominator
                if retrieval_denominator
                else 0.0
            )
            cited_docs = list(dict.fromkeys(citation.doc_id for citation in result.citations))
            citation_precision, citation_recall = citation_precision_recall(cited_docs, expected_docs)
            citation_f1 = None
            if citation_precision is not None and citation_recall is not None:
                denominator = citation_precision + citation_recall
                citation_f1 = 2 * citation_precision * citation_recall / denominator if denominator else 0.0
            cache.append(
                key,
                {
                    "sample_id": case["sample_id"],
                    "question": case["question"],
                    "mode": mode,
                    "category": case.get("category") or case.get("answer_type") or "unknown",
                    "reference": case.get("answer", ""),
                    "prediction": result.answer,
                    "answerable": result.answerable,
                    "expected_unanswerable": bool(case.get("unanswerable", False)),
                    "exact_match": exact_match(result.answer, case.get("answer", "")),
                    "token_f1": token_f1(result.answer, case.get("answer", "")),
                    "official_accuracy": multihop_official_accuracy(result.answer, case.get("answer", "")),
                    "numeric_accuracy": numeric_equal(result.answer, case.get("answer", "")),
                    "citation_precision": citation_precision,
                    "citation_recall": citation_recall,
                    "citation_f1": citation_f1,
                    "expected_doc_ids": expected_docs,
                    "retrieved_doc_ids": retrieved_docs,
                    "retrieved_chunk_ids": [context.chunk_id for context in retrieval.contexts],
                    "retrieval_precision": retrieval_precision,
                    "retrieval_recall": retrieval_recall,
                    "retrieval_f1": retrieval_f1,
                    "all_evidence_hit": retrieval_scores["all_evidence_hit"],
                    "citations": [asdict(value) for value in result.citations],
                    "query_plan": plan.model_dump(),
                    "retrieval_ms": result.retrieval_ms,
                    "rerank_ms": getattr(retrieval, "rerank_ms", 0.0),
                    "generation_ms": result.generation_ms,
                    "total_ms": result.total_ms,
                    "usage": result.usage,
                    "trace": getattr(result, "trace", retrieval.trace),
                    "operation_retries": operation_retries,
                    "rerank_fallback": any(
                        "rerank_" in value and "fallback" in value for value in retrieval.trace
                    ),
                },
            )
            completed_records += 1
            print(
                json.dumps(
                    {
                        "event": "answer_progress",
                        "completed": completed_records,
                        "total": target_records,
                        "sample_id": case["sample_id"],
                        "mode": mode,
                    }
                ),
                flush=True,
            )
    return cache.path


def _embedding_usage(runtime: LiveRuntime) -> dict[str, int]:
    provider = runtime.container.vector_store.embeddings
    return getattr(provider, "usage_snapshot", lambda: {"calls": 0, "input_tokens": 0, "texts": 0})()


async def run_cdc(runtime: LiveRuntime, *, limit: int | None = None, resume: bool = True) -> None:
    root = PREPARED_ROOT / "akh_cdc_300"
    documents = read_jsonl(root / "documents.jsonl")
    events = read_jsonl(root / "events-300.jsonl")
    if limit:
        documents = documents[:limit]
    selected_ids = {record["doc_id"] for record in documents}
    events = [event for event in events if event["doc_id"] in selected_ids]
    active_root = Path(settings.upload_dir).resolve() / "benchmark-cdc"
    active_root.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(RESULT_ROOT / "cdc.jsonl", resume=resume)
    events_by_doc: dict[str, list[dict[str, Any]]] = {}
    for event in events:
        events_by_doc.setdefault(event["doc_id"], []).append(event)
    for document in documents:
        doc_id = document["doc_id"]
        if cache.has(f"{doc_id}:summary"):
            continue
        active = active_root / f"{doc_id}.md"
        shutil.copy2(root / document["initial"], active)
        await runtime.container.ingestion.ingest(str(active), doc_id=doc_id)
        incremental_ms: list[float] = []
        usage_before = _embedding_usage(runtime)
        unchanged_retained = 0
        expected_unchanged = 0
        duplicate_success = 0
        for event_record in sorted(events_by_doc.get(doc_id, []), key=lambda value: value["sequence"]):
            key = event_record["event_id"]
            if cache.has(key):
                continue
            shutil.copy2(root / event_record["snapshot"], active)
            event = CDCEvent(
                event_id=key,
                operation=event_record["operation"],
                doc_id=doc_id,
                file_path=str(active),
                observed_hash=event_record["sha256"],
            )
            result = await runtime.container.update_agent.process_event(event)
            replay = await runtime.container.update_agent.process_event(event)
            duplicate_success += int(replay.duplicate and replay.success)
            incremental_ms.append(result.processing_time_ms)
            expected = int(event_record.get("expected_unchanged_chunks", 0))
            expected_unchanged += expected
            unchanged_retained += min(result.chunks_unchanged, expected)
            cache.append(
                key,
                {
                    "event_id": key,
                    "doc_id": doc_id,
                    "change": event_record["change"],
                    "success": result.success,
                    "duplicate_replay": replay.duplicate and replay.success,
                    "vectors_added": result.vectors_added,
                    "vectors_deleted": result.vectors_deleted,
                    "chunks_unchanged": result.chunks_unchanged,
                    "expected_unchanged_chunks": expected,
                    "incremental_ms": result.processing_time_ms,
                },
            )
        incremental_ids = set(await runtime.container.vector_store.get_document_chunks(doc_id))
        usage_after_incremental = _embedding_usage(runtime)
        rebuild_started = time.perf_counter()
        usage_before_rebuild = _embedding_usage(runtime)
        rebuilt = await runtime.container.ingestion.rebuild(doc_id)
        rebuild_ms = (time.perf_counter() - rebuild_started) * 1000
        rebuild_ids = set(await runtime.container.vector_store.get_document_chunks(doc_id))
        usage_after_rebuild = _embedding_usage(runtime)
        incremental_tokens = usage_after_incremental["input_tokens"] - usage_before["input_tokens"]
        rebuild_tokens = usage_after_rebuild["input_tokens"] - usage_before_rebuild["input_tokens"]
        cache.append(
            f"{doc_id}:summary",
            {
                "record_type": "document_summary",
                "doc_id": doc_id,
                "consistent_with_rebuild": incremental_ids == rebuild_ids,
                "duplicate_success": duplicate_success,
                "events": len(events_by_doc.get(doc_id, [])),
                "unchanged_chunk_retention": (
                    unchanged_retained / expected_unchanged if expected_unchanged else 1.0
                ),
                "incremental_embedding_tokens": incremental_tokens,
                "full_rebuild_embedding_tokens": rebuild_tokens * len(events_by_doc.get(doc_id, [])),
                "median_incremental_ms": percentile(incremental_ms, 0.5),
                "full_rebuild_ms": rebuild_ms,
                "speedup": rebuild_ms / percentile(incremental_ms, 0.5)
                if incremental_ms and percentile(incremental_ms, 0.5)
                else 0,
                "chunks": rebuilt.chunks_total,
            },
        )


async def run_api_load(
    *,
    limit: int = 100,
    base_url: str = "http://127.0.0.1:8080",
) -> dict[str, dict[str, int | float]]:
    import httpx

    cases = read_jsonl(PREPARED_ROOT / "multihop_rag" / "answer-300.jsonl")[:limit]
    cache = ResultCache(RESULT_ROOT / "api-load.jsonl", resume=False)
    semaphore = asyncio.Semaphore(5)

    async with httpx.AsyncClient(base_url=base_url, timeout=180) as client:
        before = (await client.get("/api/admin/stats")).json()

        async def request(index: int, case: dict[str, Any]) -> None:
            async with semaphore:
                started = time.perf_counter()
                response = await client.post(
                    "/api/qa/ask",
                    json={"question": case["question"], "top_k": 10, "mode": "hybrid"},
                )
                elapsed = (time.perf_counter() - started) * 1000
                body = (
                    response.json()
                    if response.headers.get("content-type", "").startswith("application/json")
                    else {}
                )
                cache.append(
                    f"api-{index:03d}",
                    {
                        "sample_id": case["sample_id"],
                        "status_code": response.status_code,
                        "total_ms": elapsed,
                        "retrieval_ms": body.get("retrieval_ms", 0),
                        "generation_ms": body.get("generation_ms", 0),
                        "usage": body.get("usage", {}),
                    },
                )

        await asyncio.gather(*(request(index, case) for index, case in enumerate(cases)))
        after = (await client.get("/api/admin/stats")).json()

    def delta(current: dict[str, Any], previous: dict[str, Any]) -> dict[str, int | float]:
        values: dict[str, int | float] = {}
        for key, value in current.items():
            if isinstance(value, (int, float)):
                values[key] = value - previous.get(key, 0)
        return values

    return {
        "chat": delta(after.get("model_usage", {}), before.get("model_usage", {})),
        "reranker": delta(after.get("reranker", {}), before.get("reranker", {})),
        "embedding": delta(
            after.get("vector_store", {}).get("usage", {}),
            before.get("vector_store", {}).get("usage", {}),
        ),
    }


def _merge_usage(
    base: dict[str, Any],
    addition: dict[str, dict[str, int | float]],
) -> dict[str, Any]:
    for family, counters in addition.items():
        target = base.setdefault(family, {})
        for key, value in counters.items():
            target[key] = target.get(key, 0) + value
    return base


def _mean(records: list[dict[str, Any]], field: str) -> float:
    values = [float(record[field]) for record in records if record.get(field) is not None]
    return sum(values) / len(values) if values else 0.0


def _usage_total(records: list[dict[str, Any]]) -> dict[str, int]:
    totals = {
        "calls": 0,
        "errors": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "total_tokens": 0,
    }
    for record in records:
        usage = record.get("usage") or {}
        if usage:
            totals["calls"] += int(usage.get("calls", 1) or 0)
        for field in ("errors", "input_tokens", "output_tokens", "total_tokens"):
            totals[field] += int(usage.get(field, 0) or 0)
    return totals


def summarize_multihop_answers(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a paired Vector/Hybrid MultiHop-RAG answer experiment."""
    by_mode = {
        mode: [record for record in records if record.get("mode") == mode] for mode in ("vector", "hybrid")
    }
    modes: dict[str, Any] = {}
    for mode, mode_records in by_mode.items():
        answerable = [record for record in mode_records if not record["expected_unanswerable"]]
        modes[mode] = {
            "cases": len(mode_records),
            "answerable_cases": len(answerable),
            "answer_em": _mean(answerable, "exact_match"),
            "token_f1": _mean(answerable, "token_f1"),
            "official_overlap_accuracy": _mean(answerable, "official_accuracy"),
            "citation_precision": _mean(answerable, "citation_precision"),
            "citation_recall": _mean(answerable, "citation_recall"),
            "citation_f1": _mean(answerable, "citation_f1"),
            "retrieval_precision": _mean(answerable, "retrieval_precision"),
            "retrieval_recall": _mean(answerable, "retrieval_recall"),
            "retrieval_f1": _mean(answerable, "retrieval_f1"),
            "all_evidence_hit": _mean(answerable, "all_evidence_hit"),
            "rerank_fallback_rate": _mean(mode_records, "rerank_fallback"),
            "rejection_f1": binary_f1(
                [not record["answerable"] for record in mode_records],
                [record["expected_unanswerable"] for record in mode_records],
            )
            if mode_records
            else 0.0,
            "answerability_accuracy": _mean(
                [
                    {"correct": float(bool(record["answerable"]) != bool(record["expected_unanswerable"]))}
                    for record in mode_records
                ],
                "correct",
            ),
            "retrieval_p50_ms": percentile([float(record["retrieval_ms"]) for record in mode_records], 0.5),
            "retrieval_p95_ms": percentile([float(record["retrieval_ms"]) for record in mode_records], 0.95),
            "generation_p50_ms": percentile([float(record["generation_ms"]) for record in mode_records], 0.5),
            "generation_p95_ms": percentile(
                [float(record["generation_ms"]) for record in mode_records], 0.95
            ),
            "rerank_p50_ms": percentile([float(record.get("rerank_ms", 0)) for record in mode_records], 0.5),
            "rerank_p95_ms": percentile([float(record.get("rerank_ms", 0)) for record in mode_records], 0.95),
            "total_p50_ms": percentile([float(record["total_ms"]) for record in mode_records], 0.5),
            "total_p95_ms": percentile([float(record["total_ms"]) for record in mode_records], 0.95),
            "answer_generation_usage": _usage_total(mode_records),
        }

    vector_by_id = {record["sample_id"]: record for record in by_mode["vector"]}
    pairs = [
        record
        for record in by_mode["hybrid"]
        if record["sample_id"] in vector_by_id and not record["expected_unanswerable"]
    ]
    paired: dict[str, Any] = {
        "all_cases": len(set(vector_by_id) & {record["sample_id"] for record in by_mode["hybrid"]}),
        "answerable_cases": len(pairs),
    }
    if pairs:
        for field in ("exact_match", "token_f1", "official_accuracy", "citation_recall"):
            baseline = [float(vector_by_id[record["sample_id"]][field]) for record in pairs]
            candidate = [float(record[field]) for record in pairs]
            paired[f"{field}_gain"] = asdict(paired_bootstrap_difference(baseline, candidate))
        vector_f1 = [float(vector_by_id[record["sample_id"]]["token_f1"]) for record in pairs]
        hybrid_f1 = [float(record["token_f1"]) for record in pairs]
        paired["token_f1_comparison"] = {
            "improved": sum(right > left for left, right in zip(vector_f1, hybrid_f1, strict=True)),
            "tied": sum(right == left for left, right in zip(vector_f1, hybrid_f1, strict=True)),
            "worse": sum(right < left for left, right in zip(vector_f1, hybrid_f1, strict=True)),
        }

    categories: dict[str, Any] = {}
    for category in sorted(
        {str(record.get("category", "unknown")) for record in records if not record["expected_unanswerable"]}
    ):
        categories[category] = {}
        for mode, mode_records in by_mode.items():
            selected = [
                record
                for record in mode_records
                if record.get("category", "unknown") == category and not record["expected_unanswerable"]
            ]
            categories[category][mode] = {
                "cases": len(selected),
                "answer_em": _mean(selected, "exact_match"),
                "token_f1": _mean(selected, "token_f1"),
            }
    return {"modes": modes, "paired": paired, "categories": categories}


def write_answer_experiment_summary(
    result_path: Path,
    *,
    plan_path: Path | None = None,
    run_usage: dict[str, Any] | None = None,
) -> Path:
    records = read_jsonl(result_path)
    plan_records = read_jsonl(plan_path) if plan_path else []
    if "v5" in result_path.stem:
        experiment = "multihop_rag_paired_answer_vector_vs_hybrid_comparison_grounded_v5"
    elif "v4" in result_path.stem:
        experiment = "multihop_rag_paired_answer_vector_vs_hybrid_slot_rrf_temporal_v4"
    else:
        experiment = "multihop_rag_paired_answer_vector_vs_hybrid_rerank_temporal_v3"
    summary = {
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": experiment,
        "complete": len(records) >= 600 and len({record["sample_id"] for record in records}) >= 300,
        "raw_result": str(result_path),
        "query_plan_cache": str(plan_path) if plan_path else None,
        "config": {
            "documents": 100,
            "answer_cases": 300,
            "answerable_cases": 225,
            "unanswerable_cases": 75,
            "seed": 42,
            "top_k": 10,
            "temperature": 0,
            "llm_model": settings.llm_model,
            "embedding_model": settings.active_embedding_model,
            "chunk_size_tokens": settings.chunk_size_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "rrf_constant": settings.rrf_constant,
            "hybrid_vector_weight": settings.hybrid_vector_weight,
            "hybrid_graph_weight": settings.hybrid_graph_weight,
            "hybrid_max_chunks_per_document": settings.hybrid_max_chunks_per_document,
            "rerank_enabled": settings.rerank_enabled,
            "rerank_model": settings.rerank_model,
            "rerank_candidate_k": settings.rerank_candidate_k,
            "rerank_top_documents": settings.rerank_top_documents,
            "rerank_local_candidates_per_query": settings.rerank_local_candidates_per_query,
            "rerank_max_local_candidates": settings.rerank_max_local_candidates,
            "rerank_max_chunks_per_document": settings.rerank_max_chunks_per_document,
            "rerank_base_rrf_weight": settings.rerank_base_rrf_weight,
            "rerank_slot_rrf_weight": settings.rerank_slot_rrf_weight,
            "rerank_slot_match_top_n": settings.rerank_slot_match_top_n,
            "answer_max_context_chunks": settings.answer_max_context_chunks,
            "answer_max_context_chars": settings.answer_max_context_chars,
        },
        "results": summarize_multihop_answers(records),
        "query_plan_usage": _usage_total(plan_records),
        "current_command_usage": run_usage or {},
        "notes": [
            "EM/Token F1/official overlap are computed only on the 225 answerable cases.",
            "Rejection F1 is computed on all 300 cases and treats unanswerable as positive.",
            "Official overlap accuracy is lenient and is reported only for dataset compatibility.",
            "Paired gain values are fractions; multiply by 100 for percentage points.",
        ],
    }
    output = result_path.with_name(f"{result_path.stem}-summary.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output


def select_stratified_multihop_ids(
    counts: dict[str, int],
    *,
    seed: int = 42,
) -> set[str]:
    """Select fixed, category-balanced IDs without depending on source-file order."""
    cases = read_jsonl(PREPARED_ROOT / "multihop_rag" / "answer-300.jsonl")
    generator = random.Random(seed)
    selected: set[str] = set()
    for category, count in sorted(counts.items()):
        group = sorted(
            (case for case in cases if case.get("category") == category),
            key=lambda case: str(case["sample_id"]),
        )
        if len(group) < count:
            raise ValueError(f"category {category!r} has only {len(group)} cases, requested {count}")
        selected.update(str(case["sample_id"]) for case in generator.sample(group, count))
    return selected


def evaluate_multihop_gate(
    result_path: Path,
    expected_ids: set[str],
    *,
    experiment: str = "rerank-temporal-v3-gate",
) -> dict[str, Any]:
    records = [record for record in read_jsonl(result_path) if str(record.get("sample_id")) in expected_ids]
    expected_records = len(expected_ids) * 2
    expected_keys = {f"{sample_id}:{mode}" for sample_id in expected_ids for mode in ("vector", "hybrid")}
    completed_keys = {
        f"{record.get('sample_id')}:{record.get('mode')}"
        for record in records
        if record.get("sample_id") and record.get("mode")
    }
    missing_keys = sorted(expected_keys - completed_keys)
    summary = summarize_multihop_answers(records)
    hybrid = summary["modes"]["hybrid"]
    gain = summary.get("paired", {}).get("exact_match_gain", {}).get("estimate", 0.0)
    temporal = summary.get("categories", {}).get("temporal_query", {}).get("hybrid", {})
    metrics = {
        "completed_records": len(records),
        "expected_records": expected_records,
        "missing_records": len(missing_keys),
        "business_errors": sum(bool(record.get("error")) for record in records),
        "hybrid_evidence_recall": hybrid.get("retrieval_recall", 0.0),
        "hybrid_document_precision": hybrid.get("retrieval_precision", 0.0),
        "hybrid_all_evidence_hit": hybrid.get("all_evidence_hit", 0.0),
        "hybrid_answer_em": hybrid.get("answer_em", 0.0),
        "hybrid_vector_paired_em_gain": gain,
        "temporal_answer_em": temporal.get("answer_em", 0.0),
        "hybrid_citation_recall": hybrid.get("citation_recall", 0.0),
        "hybrid_rejection_f1": hybrid.get("rejection_f1", 0.0),
        "rerank_fallback_rate": hybrid.get("rerank_fallback_rate", 0.0),
        "hybrid_total_p95_ms": hybrid.get("total_p95_ms", 0.0),
    }
    gates = {
        "complete": metrics["completed_records"] == expected_records,
        "business_errors": metrics["business_errors"] == 0,
        "hybrid_evidence_recall": metrics["hybrid_evidence_recall"] >= 0.85,
        "hybrid_document_precision": metrics["hybrid_document_precision"] >= 0.32,
        "hybrid_all_evidence_hit": metrics["hybrid_all_evidence_hit"] >= 0.65,
        "hybrid_answer_em": metrics["hybrid_answer_em"] >= 0.60,
        "hybrid_vector_paired_em_gain": metrics["hybrid_vector_paired_em_gain"] >= 0.02,
        "temporal_answer_em": metrics["temporal_answer_em"] >= 0.40,
        "hybrid_citation_recall": metrics["hybrid_citation_recall"] >= 0.70,
        "hybrid_rejection_f1": metrics["hybrid_rejection_f1"] >= 0.80,
        "rerank_fallback_rate": metrics["rerank_fallback_rate"] <= 0.02,
        "hybrid_total_p95_ms": metrics["hybrid_total_p95_ms"] <= 14_000,
    }
    return {
        "schema_version": 1,
        "experiment": experiment,
        "seed": 42,
        "case_ids": sorted(expected_ids),
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "diagnostics": summary,
        "missing_keys": missing_keys,
    }


async def run_reranker_preflight() -> dict[str, Any]:
    """Make one real, cheap request and verify score direction and response shape."""
    from services.reranker import build_reranker

    reranker = build_reranker()
    if reranker is None:
        raise RuntimeError("Reranker is enabled but SiliconFlow is not configured")
    try:
        scores = await reranker.rerank(
            "What is the capital of France?",
            [
                "Paris is the capital and most populous city of France.",
                "Python is a general-purpose programming language.",
                "Bananas are cultivated in tropical regions.",
            ],
            top_n=3,
        )
        if not scores or scores[0].index != 0:
            raise RuntimeError("reranker preflight returned an unexpected score direction")
        return {
            "passed": True,
            "model": reranker.model,
            "top_index": scores[0].index,
            "result_count": len(scores),
            "usage": reranker.usage_snapshot(),
        }
    finally:
        await reranker.close()


async def run_multihop_gated_experiment(
    *,
    resume: bool = True,
    version: str = "v3",
) -> Path:
    """Run preflight -> fixed 10 smoke -> 100 gate -> conditional remaining 200."""
    if version not in {"v3", "v4", "v5"}:
        raise ValueError(f"unsupported experiment version: {version}")
    if not settings.llm_configured or not settings.embedding_configured:
        raise RuntimeError("live benchmark requires configured LLM and embedding providers")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    preflight = await run_reranker_preflight()
    stems = {
        "v3": "rerank-temporal-v3",
        "v4": "slot-rrf-temporal-v4",
        "v5": "comparison-grounded-v5",
    }
    stem = stems[version]

    smoke_ids = select_stratified_multihop_ids(
        {
            "comparison_query": 3,
            "inference_query": 2,
            "null_query": 2,
            "temporal_query": 3,
        }
    )
    smoke_output = RESULT_ROOT / f"answers-multihop-{stem}-smoke.jsonl"
    smoke_plans = RESULT_ROOT / f"answers-multihop-{stem}-smoke-query-plans.jsonl"
    await run_suites(
        {"multihop"},
        resume=resume,
        answer_modes=("vector", "hybrid"),
        answer_output=smoke_output,
        answer_plan_output=smoke_plans,
        answer_case_ids=smoke_ids,
    )
    smoke_records = [
        record for record in read_jsonl(smoke_output) if str(record.get("sample_id")) in smoke_ids
    ]
    smoke = {
        "case_ids": sorted(smoke_ids),
        "completed_records": len(smoke_records),
        "expected_records": 20,
        "rerank_fallbacks": sum(bool(record.get("rerank_fallback")) for record in smoke_records),
        "structured_contract": all(
            isinstance(record.get("prediction"), str)
            and isinstance(record.get("citations"), list)
            and isinstance(record.get("query_plan"), dict)
            for record in smoke_records
        ),
    }
    smoke["passed"] = bool(
        smoke["completed_records"] == smoke["expected_records"]
        and smoke["rerank_fallbacks"] == 0
        and smoke["structured_contract"]
    )

    gate_output = RESULT_ROOT / f"answers-multihop-{stem}.jsonl"
    gate_plans = RESULT_ROOT / f"answers-multihop-{stem}-query-plans.jsonl"
    report_path = RESULT_ROOT / f"answers-multihop-{stem}-gate.json"
    if not smoke["passed"]:
        report = {"preflight": preflight, "smoke": smoke, "gate": None, "full_run_started": False}
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return report_path

    gate_ids = select_stratified_multihop_ids(
        {
            "comparison_query": 25,
            "inference_query": 25,
            "null_query": 25,
            "temporal_query": 25,
        }
    )
    await run_suites(
        {"multihop"},
        resume=resume,
        answer_modes=("vector", "hybrid"),
        answer_output=gate_output,
        answer_plan_output=gate_plans,
        answer_case_ids=gate_ids,
    )
    gate = evaluate_multihop_gate(
        gate_output,
        gate_ids,
        experiment=f"{stem}-gate",
    )
    report: dict[str, Any] = {
        "preflight": preflight,
        "smoke": smoke,
        "gate": gate,
        "full_run_started": gate["passed"],
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not gate["passed"]:
        return report_path

    await run_suites(
        {"multihop"},
        resume=True,
        answer_modes=("vector", "hybrid"),
        answer_output=gate_output,
        answer_plan_output=gate_plans,
    )
    all_records = read_jsonl(gate_output)
    heldout_records = [record for record in all_records if str(record.get("sample_id")) not in gate_ids]
    final = summarize_multihop_answers(all_records)
    heldout = summarize_multihop_answers(heldout_records)
    hybrid = final["modes"]["hybrid"]
    gain = final.get("paired", {}).get("exact_match_gain", {})
    success_gates = {
        "hybrid_answer_em": hybrid.get("answer_em", 0.0) >= 0.65,
        "hybrid_citation_recall": hybrid.get("citation_recall", 0.0) >= 0.75,
        "hybrid_rejection_f1": hybrid.get("rejection_f1", 0.0) >= 0.80,
        "paired_em_gain": gain.get("estimate", 0.0) >= 0.05,
        "paired_em_ci_lower": gain.get("lower", 0.0) > 0,
    }
    report.update(
        {
            "full_run_completed": len(all_records) >= 600,
            "full_results": final,
            "heldout_200_results": heldout,
            "success_gates": success_gates,
            "success": all(success_gates.values()),
        }
    )
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


async def run_multihop_v4_experiment(*, resume: bool = True) -> Path:
    """Run the slot-level RRF and temporal-supplement experiment in isolated files."""
    return await run_multihop_gated_experiment(resume=resume, version="v4")


async def run_multihop_v5_experiment(*, resume: bool = True) -> Path:
    """Run comparison grounding, citation binding and adaptive graph fusion in isolated files."""
    return await run_multihop_gated_experiment(resume=resume, version="v5")


async def run_release_smoke(*, resume: bool = True) -> Path:
    """Validate the frozen default strategy on the fixed 10-case paired smoke set.

    This is an engineering contract check, not a replacement for the 100-case
    benchmark. It intentionally reuses the v5 QueryPlan cache so the check only
    varies the final routing/retrieval implementation and avoids planner cost.
    """
    if not settings.llm_configured or not settings.embedding_configured:
        raise RuntimeError("live smoke requires configured LLM and embedding providers")
    if settings.comparison_tool_enabled:
        raise RuntimeError("release smoke requires COMPARISON_TOOL_ENABLED=false")

    reference_root = RESULT_ROOT
    reference_root.mkdir(parents=True, exist_ok=True)
    preflight = await run_reranker_preflight()
    case_ids = select_stratified_multihop_ids(
        {
            "comparison_query": 3,
            "inference_query": 2,
            "null_query": 2,
            "temporal_query": 3,
        }
    )
    output = reference_root / "answers-multihop-release-smoke-v1.jsonl"
    plan_cache = reference_root / "answers-multihop-comparison-grounded-v5-query-plans.jsonl"
    if not plan_cache.exists():
        raise FileNotFoundError(f"required frozen QueryPlan cache is missing: {plan_cache}")

    await run_suites(
        {"multihop"},
        resume=resume,
        answer_modes=("vector", "hybrid"),
        answer_output=output,
        answer_plan_output=plan_cache,
        answer_case_ids=case_ids,
    )
    records = [record for record in read_jsonl(output) if str(record.get("sample_id")) in case_ids]
    expected_keys = {f"{sample_id}:{mode}" for sample_id in case_ids for mode in ("vector", "hybrid")}
    completed_keys = {
        f"{record.get('sample_id')}:{record.get('mode')}"
        for record in records
        if record.get("sample_id") and record.get("mode")
    }
    citations_valid = all(
        all(
            str(citation.get("chunk_id")) in set(record.get("retrieved_chunk_ids", []))
            for citation in record.get("citations", [])
        )
        for record in records
    )
    checks = {
        "complete": completed_keys == expected_keys,
        "business_errors": not any(record.get("error") for record in records),
        "rerank_fallbacks": not any(record.get("rerank_fallback") for record in records),
        "structured_contract": all(
            isinstance(record.get("prediction"), str)
            and isinstance(record.get("citations"), list)
            and isinstance(record.get("query_plan"), dict)
            for record in records
        ),
        "citations_map_to_retrieved_chunks": citations_valid,
    }
    report = {
        "schema_version": 1,
        "experiment": "release-smoke-v1",
        "purpose": "engineering contract check; not a formal quality estimate",
        "seed": 42,
        "case_ids": sorted(case_ids),
        "preflight": preflight,
        "configuration": {
            "comparison_tool_enabled": settings.comparison_tool_enabled,
            "answer_max_context_chunks": settings.answer_max_context_chunks,
            "answer_max_context_chars": settings.answer_max_context_chars,
            "rerank_top_documents": settings.rerank_top_documents,
        },
        "completed_records": len(records),
        "expected_records": len(expected_keys),
        "missing_keys": sorted(expected_keys - completed_keys),
        "checks": checks,
        "passed": all(checks.values()),
        "diagnostics": summarize_multihop_answers(records),
        "raw_results": str(output),
        "query_plan_cache": str(plan_cache),
    }
    report_path = reference_root / "answers-multihop-release-smoke-v1-report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report_path


def _aggregate() -> dict[str, Any]:
    parse = read_jsonl(RESULT_ROOT / "parse.jsonl")
    retrieval = read_jsonl(RESULT_ROOT / "retrieval.jsonl")
    multihop = read_jsonl(RESULT_ROOT / "answers-multihop_rag.jsonl")
    tatqa = read_jsonl(RESULT_ROOT / "answers-tat_qa.jsonl")
    rgb = read_jsonl(RESULT_ROOT / "answers-rgb.jsonl")
    cdc = read_jsonl(RESULT_ROOT / "cdc.jsonl")
    api = read_jsonl(RESULT_ROOT / "api-load.jsonl")
    metrics: dict[str, float] = {}
    if parse:
        metrics["parse_success_rate"] = _mean(parse, "success")
        digital = [record for record in parse if record.get("condition") == "digital"]
        clear_ocr = [record for record in parse if record.get("condition") == "clear_ocr"]
        disturbed_ocr = [record for record in parse if record.get("condition") == "disturbed_ocr"]
        tables = [record for record in parse if record.get("category") in {"pdf", "docx", "xlsx", "csv"}]
        if digital:
            metrics["digital_text_f1"] = _mean(digital, "text_f1")
        if clear_ocr:
            metrics["clear_ocr_cer"] = _mean(clear_ocr, "cer")
        if disturbed_ocr:
            metrics["disturbed_ocr_cer"] = _mean(disturbed_ocr, "cer")
        if tables:
            metrics["table_cell_f1"] = _mean(tables, "table_cell_f1")
    confidence: dict[str, Any] = {}
    if retrieval:
        by_mode = {
            mode: [record for record in retrieval if record["mode"] == mode]
            for mode in ("vector", "graph", "hybrid")
        }
        for mode, records in by_mode.items():
            metrics[f"{mode}_evidence_recall_at_10"] = _mean(records, "evidence_recall")
            metrics[f"{mode}_all_evidence_hit_at_10"] = _mean(records, "all_evidence_hit")
        metrics["evidence_recall_at_10"] = metrics["hybrid_evidence_recall_at_10"]
        metrics["all_evidence_hit_at_10"] = metrics["hybrid_all_evidence_hit_at_10"]
        vector_by_id = {record["sample_id"]: record for record in by_mode["vector"]}
        hybrid_pairs = [record for record in by_mode["hybrid"] if record["sample_id"] in vector_by_id]
        vector_scores = [vector_by_id[record["sample_id"]]["evidence_recall"] for record in hybrid_pairs]
        hybrid_scores = [record["evidence_recall"] for record in hybrid_pairs]
        if vector_scores:
            interval = paired_bootstrap_difference(vector_scores, hybrid_scores)
            metrics["hybrid_gain_points"] = interval.estimate
            confidence["hybrid_gain_points"] = asdict(interval)
    if multihop:
        by_mode = {
            mode: [record for record in multihop if record["mode"] == mode] for mode in ("vector", "hybrid")
        }
        for mode, records in by_mode.items():
            answerable = [record for record in records if not record["expected_unanswerable"]]
            metrics[f"{mode}_multihop_answer_em"] = _mean(answerable, "exact_match")
            metrics[f"{mode}_multihop_token_f1"] = _mean(answerable, "token_f1")
            metrics[f"{mode}_multihop_official_accuracy"] = _mean(answerable, "official_accuracy")
            metrics[f"{mode}_citation_precision"] = _mean(answerable, "citation_precision")
            metrics[f"{mode}_citation_recall"] = _mean(answerable, "citation_recall")
            metrics[f"{mode}_citation_f1"] = _mean(answerable, "citation_f1")
        hybrid = by_mode["hybrid"]
        hybrid_answerable = [record for record in hybrid if not record["expected_unanswerable"]]
        metrics["multihop_answer_em"] = _mean(hybrid_answerable, "exact_match")
        metrics["multihop_token_f1"] = _mean(hybrid_answerable, "token_f1")
        metrics["citation_recall"] = _mean(hybrid_answerable, "citation_recall")
        metrics["rejection_f1"] = binary_f1(
            [not record["answerable"] for record in hybrid],
            [record["expected_unanswerable"] for record in hybrid],
        )
        vector_by_id = {record["sample_id"]: record for record in by_mode["vector"]}
        hybrid_pairs = [record for record in hybrid_answerable if record["sample_id"] in vector_by_id]
        if hybrid_pairs:
            vector_f1 = [vector_by_id[record["sample_id"]]["token_f1"] for record in hybrid_pairs]
            hybrid_f1 = [record["token_f1"] for record in hybrid_pairs]
            interval = paired_bootstrap_difference(vector_f1, hybrid_f1)
            metrics["hybrid_answer_f1_gain_points"] = interval.estimate
            confidence["hybrid_answer_f1_gain_points"] = asdict(interval)
    if tatqa:
        metrics["tatqa_numeric_accuracy"] = _mean(tatqa, "numeric_accuracy")
        metrics["tatqa_token_f1"] = _mean(tatqa, "token_f1")
        metrics["tatqa_citation_recall"] = _mean(tatqa, "citation_recall")
    if rgb:
        noise = [record for record in rgb if "noise" in str(record.get("category", "")).lower()]
        integration = [record for record in rgb if "integrat" in str(record.get("category", "")).lower()]
        metrics["rgb_noise_accuracy"] = _mean(noise or rgb, "exact_match")
        metrics["rgb_integration_accuracy"] = _mean(integration or rgb, "exact_match")
        metrics["rgb_rejection_f1"] = binary_f1(
            [not record["answerable"] for record in rgb],
            [record["expected_unanswerable"] for record in rgb],
        )
    summaries = [record for record in cdc if record.get("record_type") == "document_summary"]
    events = [record for record in cdc if record.get("event_id")]
    if summaries:
        metrics["cdc_consistency"] = _mean(summaries, "consistent_with_rebuild")
        metrics["cdc_idempotency"] = _mean(events, "duplicate_replay")
        metrics["unchanged_chunk_retention"] = _mean(summaries, "unchanged_chunk_retention")
        total_incremental = sum(record["incremental_embedding_tokens"] for record in summaries)
        total_full = sum(record["full_rebuild_embedding_tokens"] for record in summaries)
        metrics["embedding_token_reduction"] = 1 - total_incremental / total_full if total_full else 0
        metrics["median_update_speedup"] = percentile([record["speedup"] for record in summaries], 0.5)
    if api:
        metrics["api_5xx_count"] = float(sum(record["status_code"] >= 500 for record in api))
        metrics["retrieval_p95_ms"] = percentile([record["retrieval_ms"] for record in api], 0.95)
        metrics["answer_p50_ms"] = percentile([record["total_ms"] for record in api], 0.5)
        metrics["answer_p95_ms"] = percentile([record["total_ms"] for record in api], 0.95)
    sample_counts = {
        "parse": len(parse),
        "retrieval_cases": len({record.get("sample_id") for record in retrieval}),
        "multihop_answer_cases": len({record.get("sample_id") for record in multihop}),
        "tatqa_answer_cases": len({record.get("sample_id") for record in tatqa}),
        "rgb_answer_cases": len({record.get("sample_id") for record in rgb}),
        "cdc_events": len([record for record in cdc if record.get("event_id")]),
        "cdc_documents": len(summaries),
        "api_requests": len(api),
    }
    return {
        "metrics": metrics,
        "confidence_intervals": confidence,
        "sample_counts": sample_counts,
    }


def _thresholds() -> dict[str, float]:
    path = BENCHMARK_ROOT / "config" / "reference.json"
    return json.loads(path.read_text(encoding="utf-8"))["thresholds"]


GATE_MAPPINGS = {
    "parse_success_rate": ("parse_success_rate", "min"),
    "digital_text_f1": ("digital_text_f1", "min"),
    "clear_ocr_cer": ("clear_ocr_cer_max", "max"),
    "disturbed_ocr_cer": ("disturbed_ocr_cer_max", "max"),
    "table_cell_f1": ("table_cell_f1", "min"),
    "evidence_recall_at_10": ("evidence_recall_at_10", "min"),
    "all_evidence_hit_at_10": ("all_evidence_hit_at_10", "min"),
    "hybrid_gain_points": ("hybrid_gain_points", "min"),
    "multihop_answer_em": ("multihop_answer_em", "min"),
    "multihop_token_f1": ("multihop_token_f1", "min"),
    "citation_recall": ("citation_recall", "min"),
    "rejection_f1": ("rejection_f1", "min"),
    "tatqa_numeric_accuracy": ("tatqa_numeric_accuracy", "min"),
    "tatqa_token_f1": ("tatqa_token_f1", "min"),
    "tatqa_citation_recall": ("citation_recall", "min"),
    "rgb_noise_accuracy": ("rgb_noise_accuracy", "min"),
    "rgb_integration_accuracy": ("rgb_integration_accuracy", "min"),
    "rgb_rejection_f1": ("rejection_f1", "min"),
    "cdc_consistency": ("cdc_consistency", "min"),
    "cdc_idempotency": ("cdc_idempotency", "min"),
    "unchanged_chunk_retention": ("unchanged_chunk_retention", "min"),
    "embedding_token_reduction": ("embedding_token_reduction", "min"),
    "median_update_speedup": ("median_update_speedup", "min"),
    "retrieval_p95_ms": ("retrieval_p95_ms", "max"),
    "api_5xx_count": ("api_5xx_count", "max"),
    "core_line_coverage": ("core_line_coverage", "min"),
}


REFERENCE_SAMPLE_COUNTS = {
    "parse": 70,
    "retrieval_cases": 1358,
    "multihop_answer_cases": 300,
    "tatqa_answer_cases": 300,
    "rgb_answer_cases": 200,
    "cdc_events": 300,
    "cdc_documents": 100,
    "api_requests": 100,
}

METRIC_SUITES = {
    "parse_success_rate": "parse",
    "digital_text_f1": "parse",
    "clear_ocr_cer": "parse",
    "disturbed_ocr_cer": "parse",
    "table_cell_f1": "parse",
    "evidence_recall_at_10": "retrieval",
    "all_evidence_hit_at_10": "retrieval",
    "hybrid_gain_points": "retrieval",
    "multihop_answer_em": "multihop",
    "multihop_token_f1": "multihop",
    "citation_recall": "multihop",
    "rejection_f1": "multihop",
    "tatqa_numeric_accuracy": "tatqa",
    "tatqa_token_f1": "tatqa",
    "tatqa_citation_recall": "tatqa",
    "rgb_noise_accuracy": "rgb",
    "rgb_integration_accuracy": "rgb",
    "rgb_rejection_f1": "rgb",
    "cdc_consistency": "cdc",
    "cdc_idempotency": "cdc",
    "unchanged_chunk_retention": "cdc",
    "embedding_token_reduction": "cdc",
    "median_update_speedup": "cdc",
    "retrieval_p95_ms": "api",
    "api_5xx_count": "api",
}


def _suite_completeness(sample_counts: dict[str, int]) -> dict[str, bool]:
    return {
        "parse": sample_counts.get("parse", 0) >= REFERENCE_SAMPLE_COUNTS["parse"],
        "retrieval": sample_counts.get("retrieval_cases", 0) >= REFERENCE_SAMPLE_COUNTS["retrieval_cases"],
        "multihop": sample_counts.get("multihop_answer_cases", 0)
        >= REFERENCE_SAMPLE_COUNTS["multihop_answer_cases"],
        "tatqa": sample_counts.get("tatqa_answer_cases", 0) >= REFERENCE_SAMPLE_COUNTS["tatqa_answer_cases"],
        "rgb": sample_counts.get("rgb_answer_cases", 0) >= REFERENCE_SAMPLE_COUNTS["rgb_answer_cases"],
        "cdc": (
            sample_counts.get("cdc_events", 0) >= REFERENCE_SAMPLE_COUNTS["cdc_events"]
            and sample_counts.get("cdc_documents", 0) >= REFERENCE_SAMPLE_COUNTS["cdc_documents"]
        ),
        "api": sample_counts.get("api_requests", 0) >= REFERENCE_SAMPLE_COUNTS["api_requests"],
    }


def _evaluate_gates(
    metrics: dict[str, float],
    thresholds: dict[str, float],
    suite_completeness: dict[str, bool],
) -> dict[str, bool | None]:
    gates: dict[str, bool | None] = {}
    for metric, value in metrics.items():
        if metric not in GATE_MAPPINGS:
            continue
        suite = METRIC_SUITES.get(metric)
        if suite and not suite_completeness.get(suite, False):
            gates[metric] = None
            continue
        threshold_name, direction = GATE_MAPPINGS[metric]
        threshold = thresholds.get(threshold_name, 0.0)
        gates[metric] = value <= threshold if direction == "max" else value >= threshold
    return gates


def _data_version() -> str:
    if not LOCK_PATH.exists():
        return "unlocked"
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    revisions = [value["revision"][:12] for _, value in sorted(lock["datasets"].items())]
    return "+".join(revisions)


def write_summary(usage: dict[str, Any] | None = None) -> Path:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    previous_summary = RESULT_ROOT / "summary.json"
    if usage is None and previous_summary.exists():
        usage = json.loads(previous_summary.read_text(encoding="utf-8")).get("usage", {})
    aggregated = _aggregate()
    thresholds = _thresholds()
    coverage_path = Path.cwd() / "coverage.json"
    if coverage_path.exists():
        coverage = json.loads(coverage_path.read_text(encoding="utf-8"))
        aggregated["metrics"]["core_line_coverage"] = coverage["totals"]["percent_covered"] / 100
    suite_completeness = _suite_completeness(aggregated["sample_counts"])
    formal_complete = all(suite_completeness.values())
    errors = []
    for path in RESULT_ROOT.glob("*.jsonl"):
        for record in read_jsonl(path):
            if record.get("success") is False:
                identifier = record.get("sample_id") or record.get("event_id")
                errors.append(f"{path.name}:{identifier}: {record.get('error', 'failed')}")
    summary = {
        "schema_version": 2,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_kind": "reference" if formal_complete else "partial",
        "formal_complete": formal_complete,
        "suite_completeness": suite_completeness,
        "reference_sample_counts": REFERENCE_SAMPLE_COUNTS,
        "data_version": _data_version(),
        "config": {
            "llm_model": settings.llm_model,
            "embedding_model": settings.active_embedding_model,
            "temperature": 0,
            "chunk_size_tokens": settings.chunk_size_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "top_k": 10,
            "seed": 42,
            "vision_enabled": settings.llm_supports_vision,
        },
        **aggregated,
        "thresholds": thresholds,
        "metric_thresholds": {
            metric: thresholds[threshold_name]
            for metric, (threshold_name, _direction) in GATE_MAPPINGS.items()
            if metric in aggregated["metrics"] and threshold_name in thresholds
        },
        "gates": _evaluate_gates(aggregated["metrics"], thresholds, suite_completeness),
        "usage": usage or {},
        "usage_scope": "最近一次命令；正式 all-suite 单次运行会汇总该命令的全部调用",
        "errors": errors[:100],
    }
    path = RESULT_ROOT / "summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


async def run_suites(
    suites: set[str],
    *,
    limit: int | None = None,
    resume: bool = True,
    fresh_state: bool = False,
    api_base_url: str = "http://127.0.0.1:8080",
    retrieval_modes: tuple[str, ...] = ("vector", "graph", "hybrid"),
    retrieval_output: Path | None = None,
    answer_modes: tuple[str, ...] = ("vector", "hybrid"),
    answer_output: Path | None = None,
    answer_plan_output: Path | None = None,
    answer_case_ids: set[str] | None = None,
) -> Path:
    if not settings.llm_configured or not settings.embedding_configured:
        raise RuntimeError("live benchmark requires configured LLM and embedding providers")
    requested = (
        {"parse", "retrieval", "multihop", "tatqa", "rgb", "cdc", "api"} if "all" in suites else suites
    )
    if "parse" in requested:
        await run_parse(limit=limit, resume=resume)
    usage: dict[str, Any] = {}
    retrieval_result: Path | None = None
    answer_result: Path | None = None
    service_suites = requested - {"parse", "api"}
    if service_suites:
        async with LiveRuntime() as runtime:
            if fresh_state:
                await reset_benchmark_state(runtime, service_suites)
            if "retrieval" in service_suites:
                retrieval_result = await run_retrieval(
                    runtime,
                    limit=limit,
                    resume=resume,
                    modes=retrieval_modes,
                    output=retrieval_output,
                )
            if "multihop" in service_suites:
                answer_result = await _run_answer_cases(
                    runtime,
                    "multihop_rag",
                    "answer-300.jsonl",
                    answer_modes,
                    limit=limit,
                    resume=resume,
                    output=answer_output,
                    plan_output=answer_plan_output,
                    case_ids=answer_case_ids,
                )
            if "tatqa" in service_suites:
                await _run_answer_cases(
                    runtime, "tat_qa", "answer-300.jsonl", ("hybrid",), limit=limit, resume=resume
                )
            if "rgb" in service_suites:
                await _run_answer_cases(
                    runtime, "rgb", "answer-200.jsonl", ("hybrid",), limit=limit, resume=resume
                )
            if "cdc" in service_suites:
                await run_cdc(runtime, limit=limit, resume=resume)
            usage = {
                "chat": getattr(runtime.container.extractor.llm, "usage_snapshot", lambda: {})(),
                "embedding": _embedding_usage(runtime),
                "reranker": getattr(
                    runtime.container.reranker,
                    "usage_snapshot",
                    lambda: {},
                )(),
            }
            embedding_tokens = int(usage["embedding"].get("input_tokens", 0))
            usage["embedding"]["cost_cny"] = round(
                embedding_tokens / 1000 * settings.embedding_price_per_1k_cny, 6
            )
    if "api" in requested:
        api_usage = await run_api_load(limit=limit or 100, base_url=api_base_url)
        _merge_usage(usage, api_usage)
    if usage.get("embedding"):
        embedding_tokens = int(usage["embedding"].get("input_tokens", 0))
        usage["embedding"]["cost_cny"] = round(
            embedding_tokens / 1000 * settings.embedding_price_per_1k_cny,
            6,
        )
    if retrieval_output is not None and requested == {"retrieval"}:
        return retrieval_result or retrieval_output
    if answer_output is not None and requested == {"multihop"}:
        resolved_result = answer_result or answer_output
        resolved_plans = answer_plan_output or resolved_result.with_name(
            f"{resolved_result.stem}-query-plans.jsonl"
        )
        return write_answer_experiment_summary(
            resolved_result,
            plan_path=resolved_plans,
            run_usage=usage,
        )
    return write_summary(usage)
