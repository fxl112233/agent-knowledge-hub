from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agents.knowledge_update_agent import CDCEvent, UpdateResult
from agents.qa_agent import QAResult, QueryIntent
from orchestrator.graph import build_supervisor_workflow, build_workflows
from services.graph_rag import (
    AnswerType,
    EvidenceSlot,
    GraphRAGContext,
    QueryPlan,
    RetrievalResult,
    TargetRelation,
)
from services.ingestion import IngestResult


class FakeIngestion:
    async def ingest(self, path: str, *, doc_id: str | None = None, mime_type: str = ""):
        return IngestResult(doc_id or "doc", path, 1, "READY", 1, 1, 0, 0, 0, 0, 1.0)


class FakeQA:
    async def answer(self, question: str, **_kwargs: Any):
        return QAResult(question, "answer", True, 0.8, QueryIntent.FACTOID, [], [], 1, 2, 3)


class FakeUpdater:
    async def process_event(self, event: CDCEvent):
        return UpdateResult(event.event_id, event.doc_id, event.file_path, "COMMITTED")


@pytest.mark.asyncio
async def test_all_typed_workflows_execute() -> None:
    workflows = build_workflows(FakeIngestion(), FakeQA(), FakeUpdater())  # type: ignore[arg-type]
    ingest = await workflows["ingest"].ainvoke(
        {"file_paths": ["a.txt"], "doc_ids": ["doc"], "mime_types": ["text/plain"]}
    )
    assert ingest["results"][0].doc_id == "doc"
    qa = await workflows["qa"].ainvoke({"question": "q", "top_k": 8, "mode": "hybrid"})
    assert qa["result"].answer == "answer"
    event = CDCEvent("event", "UPDATE", "doc", "a.txt")
    update = await workflows["update"].ainvoke({"events": [event]})
    assert update["results"][0].status == "COMMITTED"

    assert {
        "validate",
        "parse",
        "diff",
        "embed",
        "extract",
        "graph_upsert",
        "delete_removed",
        "commit",
        "verify",
        "rollback",
    } <= set(workflows["ingest"].get_graph().nodes)
    assert {
        "validate",
        "plan",
        "vector_retrieve",
        "graph_retrieve",
        "rrf_fuse",
        "rerank_docs",
        "document_refine",
        "rerank_chunks",
        "evidence_route",
        "comparison_tool",
        "temporal_tool",
        "calculator",
        "answer",
        "abstain",
        "citation_validate",
        "finalize",
        "fail",
    } <= set(workflows["qa"].get_graph().nodes)
    assert {
        "validate_event",
        "idempotency",
        "delete",
        "upsert",
        "diff",
        "apply",
        "verify",
        "repair",
        "commit",
        "fail",
    } <= set(workflows["update"].get_graph().nodes)

    supervisor = build_supervisor_workflow(workflows)
    supervised = await supervisor.ainvoke(
        {"workflow": "qa", "payload": {"question": "q", "mode": "hybrid"}, "thread_id": "t"}
    )
    assert supervised["result"]["result"].answer == "answer"
    assert {"validate", "ingest", "qa", "update", "fail"} <= set(supervisor.get_graph().nodes)


@pytest.mark.asyncio
async def test_ingest_stage_retries_once_and_records_high_level_trace() -> None:
    class RetryIngestion:
        def __init__(self) -> None:
            self.parse_calls = 0

        def validate_stage(self, *_args: Any, **_kwargs: Any):
            return SimpleNamespace(unchanged_result=None)

        async def parse_stage(self, _transaction: Any):
            self.parse_calls += 1
            if self.parse_calls == 1:
                raise RuntimeError("temporary")

        def diff_stage(self, _transaction: Any):
            return None

        async def embed_stage(self, _transaction: Any):
            return None

        async def extract_stage(self, _transaction: Any):
            return None

        async def graph_upsert_stage(self, _transaction: Any):
            return None

        async def delete_removed_stage(self, _transaction: Any):
            return None

        def commit_stage(self, _transaction: Any):
            return IngestResult("doc", "a.txt", 1, "READY", 1, 1, 0, 0, 0, 0, 1.0)

        async def verify_stage(self, _transaction: Any):
            return True

        async def rollback_stage(self, _transaction: Any, _error: Any):
            return None

    ingestion = RetryIngestion()
    graph = build_workflows(ingestion, FakeQA(), FakeUpdater())["ingest"]  # type: ignore[arg-type]
    result = await graph.ainvoke({"file_paths": ["a.txt"], "doc_ids": ["doc"], "mime_types": ["text/plain"]})
    assert ingestion.parse_calls == 2
    assert "ingest:parse:retry:1:RuntimeError" in result["trace"]
    assert result["results"][0].status == "READY"


@pytest.mark.asyncio
async def test_ingest_exhausted_retry_routes_to_rollback() -> None:
    class RollbackIngestion:
        def __init__(self) -> None:
            self.embed_calls = 0
            self.rollback_calls = 0

        def validate_stage(self, *_args: Any, **_kwargs: Any):
            return SimpleNamespace(unchanged_result=None)

        async def parse_stage(self, _transaction: Any):
            return None

        def diff_stage(self, _transaction: Any):
            return None

        async def embed_stage(self, _transaction: Any):
            self.embed_calls += 1
            raise RuntimeError("persistent")

        async def rollback_stage(self, _transaction: Any, _error: Any):
            self.rollback_calls += 1

    ingestion = RollbackIngestion()
    graph = build_workflows(ingestion, FakeQA(), FakeUpdater())["ingest"]  # type: ignore[arg-type]
    state = await graph.ainvoke({"file_paths": ["a.txt"], "doc_ids": ["doc"]})
    assert ingestion.embed_calls == 2
    assert ingestion.rollback_calls == 1
    assert state["error"] == "persistent"
    assert "ingest:rollback:RuntimeError" in state["trace"]


@pytest.mark.asyncio
async def test_qa_exhausted_retry_routes_to_fail() -> None:
    class FailingPipeline:
        def __init__(self) -> None:
            self.calls = 0

        async def plan_query(self, _question: str):
            self.calls += 1
            raise TimeoutError("planner unavailable")

    failing_qa = SimpleNamespace(pipeline=FailingPipeline())
    graph = build_workflows(FakeIngestion(), failing_qa, FakeUpdater())["qa"]  # type: ignore[arg-type]
    state = await graph.ainvoke({"question": "q", "top_k": 8, "mode": "hybrid"})
    assert failing_qa.pipeline.calls == 2
    assert state["error"] == "planner unavailable"
    assert "qa:plan:retry:2:TimeoutError" in state["trace"]
    assert any(item.startswith("qa:failed") for item in state["trace"])


@pytest.mark.asyncio
async def test_qa_graph_routes_enabled_comparison_tool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("orchestrator.graph.settings.comparison_tool_enabled", True)
    plan = QueryPlan(
        intent="temporal",
        queries=["left", "right"],
        answer_type=AnswerType.BOOLEAN,
        target_relation=TargetRelation.SAME_CLAIM,
        evidence_slots=[
            EvidenceSlot(slot_id="left", query="left"),
            EvidenceSlot(slot_id="right", query="right"),
        ],
    )
    context = GraphRAGContext("c1", "d1", "evidence", "doc.txt", "vector", 0.9)

    class Pipeline:
        async def plan_query(self, _question: str):
            return plan

        async def retrieve_candidates(self, *_args: Any, **_kwargs: Any):
            return RetrievalResult(plan, [context], 1.0, [])

        @staticmethod
        def _rrf_fuse(left: list[Any], right: list[Any]):
            return [*left, *right]

        async def finalize_candidates(self, *_args: Any, **_kwargs: Any):
            return RetrievalResult(plan, [context], 1.0, [])

    class ComparisonQA:
        def __init__(self) -> None:
            self.pipeline = Pipeline()
            self.calls = 0

        @staticmethod
        def _should_use_comparison(_plan: QueryPlan) -> bool:
            return True

        @staticmethod
        def _should_use_temporal(_plan: QueryPlan) -> bool:
            return True

        async def answer_comparison_from_retrieval(self, question: str, retrieval: Any, **_kwargs: Any):
            self.calls += 1
            return QAResult(
                question,
                "No",
                False,
                0.0,
                QueryIntent.TEMPORAL,
                [],
                retrieval.contexts,
                1.0,
                1.0,
                2.0,
            )

    qa = ComparisonQA()
    graph = build_workflows(FakeIngestion(), qa, FakeUpdater())["qa"]  # type: ignore[arg-type]
    state = await graph.ainvoke({"question": "same?", "top_k": 8, "mode": "hybrid"})
    assert qa.calls == 1
    assert state["result"].answer == "No"
    assert "qa:comparison_tool:complete" in state["trace"]


@pytest.mark.asyncio
async def test_update_consistency_failure_runs_repair_then_reverifies() -> None:
    class Catalog:
        @staticmethod
        def get_event(_event_id: str):
            return None

        @staticmethod
        def get_chunks(_doc_id: str):
            return [{"chunk_id": "c1"}]

    class Store:
        def __init__(self) -> None:
            self.ids: list[str] = []

        async def get_document_chunks(self, _doc_id: str):
            return self.ids

    class RepairIngestion:
        def __init__(self) -> None:
            self.vector_store = Store()
            self.knowledge_graph = Store()
            self.calls = 0

        async def repair_consistency(self, _doc_id: str):
            self.calls += 1
            self.vector_store.ids = ["c1"]
            self.knowledge_graph.ids = ["c1"]
            return {"consistent": True}

    class RepairUpdater:
        def __init__(self) -> None:
            self.catalog = Catalog()
            self.ingestion = RepairIngestion()

        async def process_event(self, event: CDCEvent):
            return UpdateResult(event.event_id, event.doc_id, event.file_path, "COMMITTED")

    updater = RepairUpdater()
    graph = build_workflows(FakeIngestion(), FakeQA(), updater)["update"]  # type: ignore[arg-type]
    state = await graph.ainvoke({"events": [CDCEvent("event", "UPDATE", "doc", "a.txt")]})
    assert updater.ingestion.calls == 1
    assert state["results"][0].status == "COMMITTED"
    assert "cdc:repair:complete:1" in state["trace"]
    assert state["trace"].count("cdc:verify:complete") == 1


@pytest.mark.asyncio
async def test_sqlite_checkpointer_persists_completed_state(tmp_path: Path) -> None:
    from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

    checkpoint_path = str(tmp_path / "workflow.sqlite3")
    config = {"configurable": {"thread_id": "resume-ingest"}}
    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        graph = build_workflows(
            FakeIngestion(),
            FakeQA(),
            FakeUpdater(),
            checkpointer=saver,  # type: ignore[arg-type]
        )["ingest"]
        await graph.ainvoke({"file_paths": ["a.txt"], "doc_ids": ["doc"]}, config=config)

    async with AsyncSqliteSaver.from_conn_string(checkpoint_path) as saver:
        restored = build_workflows(
            FakeIngestion(),
            FakeQA(),
            FakeUpdater(),
            checkpointer=saver,  # type: ignore[arg-type]
        )["ingest"]
        snapshot = await restored.aget_state(config)
        assert snapshot.values["results"][0].doc_id == "doc"
        assert "ingest:verify:complete" in snapshot.values["trace"]
