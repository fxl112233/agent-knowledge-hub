from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.prepare import _select_evidence_documents
from benchmarks.runner import (
    LiveRuntime,
    _retry_async,
    _run_answer_cases,
    evaluate_multihop_gate,
    read_jsonl,
    select_stratified_multihop_ids,
    summarize_multihop_answers,
)
from services.graph_rag import QueryPlan, RetrievalResult


def test_multihop_document_selection_is_frequency_ranked_and_deterministic() -> None:
    cases = [
        {"evidence_doc_ids": ["b", "a"]},
        {"evidence_doc_ids": ["b", "c"]},
        {"evidence_doc_ids": ["a", "d"]},
        {"evidence_doc_ids": []},
    ]

    assert _select_evidence_documents(cases, limit=2) == {"a", "b"}
    assert _select_evidence_documents(list(reversed(cases)), limit=2) == {"a", "b"}


@pytest.mark.asyncio
async def test_benchmark_ingestion_waits_for_child_cleanup_when_cancelled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    records = [{"doc_id": f"doc-{index}", "path": f"doc-{index}.md"} for index in range(3)]
    (tmp_path / "documents.jsonl").write_text(
        "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
    )
    for record in records:
        (tmp_path / record["path"]).write_text("test", encoding="utf-8")

    started = asyncio.Event()
    active = 0

    class FakeIngestion:
        async def ingest(self, _path: str, *, doc_id: str) -> SimpleNamespace:
            nonlocal active
            active += 1
            if active == 2:
                started.set()
            try:
                await asyncio.sleep(30)
            finally:
                active -= 1
            return SimpleNamespace(doc_id=doc_id)

    runtime = LiveRuntime()
    runtime.container = SimpleNamespace(ingestion=FakeIngestion())
    monkeypatch.setattr("benchmarks.runner.settings.batch_ingest_concurrency", 2)
    task = asyncio.create_task(runtime.ingest_documents(tmp_path))
    await asyncio.wait_for(started.wait(), timeout=1)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert active == 0


@pytest.mark.asyncio
async def test_benchmark_ingestion_retries_transient_document_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {"doc_id": "retry-doc", "path": "retry-doc.md"}
    (tmp_path / "documents.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    (tmp_path / record["path"]).write_text("test", encoding="utf-8")
    attempts = 0

    class FlakyIngestion:
        async def ingest(self, _path: str, *, doc_id: str) -> SimpleNamespace:
            nonlocal attempts
            attempts += 1
            if attempts < 3:
                raise RuntimeError("temporary provider failure")
            return SimpleNamespace(doc_id=doc_id)

    runtime = LiveRuntime()
    runtime.container = SimpleNamespace(ingestion=FlakyIngestion())
    monkeypatch.setattr("benchmarks.runner.settings.batch_ingest_concurrency", 2)

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("benchmarks.runner.asyncio.sleep", no_sleep)

    assert await runtime.ingest_documents(tmp_path) == {"retry-doc": "retry-doc"}
    assert attempts == 3


@pytest.mark.asyncio
async def test_paired_answer_run_reuses_one_cached_query_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "paired"
    dataset.mkdir()
    (dataset / "documents.jsonl").write_text("", encoding="utf-8")
    (dataset / "answers.jsonl").write_text(
        json.dumps(
            {
                "sample_id": "q1",
                "question": "question",
                "answer": "answer",
                "evidence_doc_ids": ["doc"],
                "unanswerable": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr("benchmarks.runner.PREPARED_ROOT", tmp_path)

    class Pipeline:
        plan_calls = 0
        plans: list[QueryPlan] = []

        async def plan_query(self, _question: str) -> QueryPlan:
            self.plan_calls += 1
            return QueryPlan(queries=["shared"], entities=["entity"])

        async def retrieve(self, _question: str, **kwargs) -> RetrievalResult:
            self.plans.append(kwargs["plan"])
            return RetrievalResult(kwargs["plan"], [], 1.0, [kwargs["mode"]])

    pipeline = Pipeline()

    class QA:
        def __init__(self) -> None:
            self.pipeline = pipeline

        async def answer_from_retrieval(self, _question: str, retrieval, **_kwargs):
            return SimpleNamespace(
                answer="answer",
                answerable=True,
                citations=[],
                retrieval_ms=retrieval.retrieval_ms,
                generation_ms=2.0,
                total_ms=3.0,
                usage={},
            )

    class Runtime:
        container = SimpleNamespace(qa_agent=QA())

        async def ingest_documents(self, *_args, **_kwargs):
            return {}

    output = tmp_path / "answers.jsonl"
    plans = tmp_path / "plans.jsonl"
    await _run_answer_cases(
        Runtime(),
        "paired",
        "answers.jsonl",
        ("vector", "hybrid"),
        limit=None,
        resume=True,
        output=output,
        plan_output=plans,
    )

    assert pipeline.plan_calls == 1
    assert pipeline.plans[0] is pipeline.plans[1]
    records = read_jsonl(output)
    assert len(records) == 2
    assert len(read_jsonl(plans)) == 1
    assert records[0]["question"] == "question"
    assert records[0]["expected_doc_ids"] == ["doc"]
    summary = summarize_multihop_answers(records)
    assert summary["paired"]["all_cases"] == 1
    assert summary["paired"]["token_f1_gain"]["estimate"] == 0


def test_stratified_selection_is_fixed_and_balanced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "multihop_rag"
    root.mkdir()
    cases = [
        {"sample_id": f"{category}-{index}", "category": category}
        for category in ("comparison_query", "temporal_query")
        for index in range(5)
    ]
    (root / "answer-300.jsonl").write_text(
        "".join(json.dumps(case) + "\n" for case in cases),
        encoding="utf-8",
    )
    monkeypatch.setattr("benchmarks.runner.PREPARED_ROOT", tmp_path)
    first = select_stratified_multihop_ids({"comparison_query": 2, "temporal_query": 2}, seed=42)
    second = select_stratified_multihop_ids({"comparison_query": 2, "temporal_query": 2}, seed=42)
    assert first == second
    assert sum(value.startswith("comparison_query") for value in first) == 2
    assert sum(value.startswith("temporal_query") for value in first) == 2


def test_gate_requires_all_thresholds_and_complete_pairs(tmp_path: Path) -> None:
    records = []
    for mode in ("vector", "hybrid"):
        records.extend(
            [
                {
                    "sample_id": "answerable",
                    "mode": mode,
                    "category": "temporal_query",
                    "expected_unanswerable": False,
                    "answerable": True,
                    "exact_match": float(mode == "hybrid"),
                    "token_f1": float(mode == "hybrid"),
                    "official_accuracy": float(mode == "hybrid"),
                    "citation_precision": 1.0,
                    "citation_recall": 1.0,
                    "citation_f1": 1.0,
                    "retrieval_precision": 0.5,
                    "retrieval_recall": 1.0,
                    "retrieval_f1": 2 / 3,
                    "all_evidence_hit": 1.0,
                    "rerank_fallback": False,
                    "retrieval_ms": 10,
                    "rerank_ms": 2,
                    "generation_ms": 10,
                    "total_ms": 20,
                    "usage": {},
                },
                {
                    "sample_id": "null",
                    "mode": mode,
                    "category": "null_query",
                    "expected_unanswerable": True,
                    "answerable": False,
                    "exact_match": 0.0,
                    "token_f1": 0.0,
                    "official_accuracy": 0.0,
                    "citation_precision": None,
                    "citation_recall": None,
                    "citation_f1": None,
                    "retrieval_precision": 0.0,
                    "retrieval_recall": 1.0,
                    "retrieval_f1": 0.0,
                    "all_evidence_hit": 1.0,
                    "rerank_fallback": False,
                    "retrieval_ms": 10,
                    "rerank_ms": 2,
                    "generation_ms": 10,
                    "total_ms": 20,
                    "usage": {},
                },
            ]
        )
    path = tmp_path / "gate.jsonl"
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )
    report = evaluate_multihop_gate(path, {"answerable", "null"})
    assert report["passed"]
    assert report["metrics"]["hybrid_vector_paired_em_gain"] == 1.0
    assert report["metrics"]["missing_records"] == 0


@pytest.mark.asyncio
async def test_complete_benchmark_operation_retries_before_record_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attempts = 0

    async def flaky() -> str:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise TimeoutError("temporary")
        return "ok"

    async def no_sleep(_delay: float) -> None:
        return None

    monkeypatch.setattr("benchmarks.runner.asyncio.sleep", no_sleep)
    result, retries = await _retry_async(flaky)

    assert result == "ok"
    assert retries == 2
