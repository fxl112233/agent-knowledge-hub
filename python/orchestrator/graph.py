"""Typed, conditional LangGraph workflows for the four-agent system."""

from __future__ import annotations

import asyncio
import operator
import time
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph

from agents.knowledge_update_agent import CDCEvent, KnowledgeUpdateAgent, UpdateResult
from agents.qa_agent import QAAgent, QAResult
from config import settings
from services.graph_rag import QueryPlan, RetrievalMode, RetrievalResult
from services.ingestion import IngestionService, IngestResult, IngestTransaction

Trace = Annotated[list[str], operator.add]


class IngestState(TypedDict, total=False):
    file_paths: list[str]
    doc_ids: list[str]
    mime_types: list[str]
    transactions: list[IngestTransaction]
    results: list[IngestResult]
    error: str
    error_type: str
    legacy: bool
    trace: Trace


class QAState(TypedDict, total=False):
    question: str
    top_k: int
    mode: str
    document_ids: list[str]
    plan: Any
    vector_retrieval: RetrievalResult
    graph_retrieval: RetrievalResult
    retrieval: RetrievalResult
    result: QAResult
    error: str
    vector_error: str
    graph_error: str
    trace: Trace


class UpdateState(TypedDict, total=False):
    events: list[CDCEvent]
    active_events: list[CDCEvent]
    duplicate_results: list[UpdateResult]
    active_results: list[UpdateResult]
    results: list[UpdateResult]
    error: str
    repair_attempted: bool
    trace: Trace


class SupervisorState(TypedDict, total=False):
    workflow: str
    payload: dict[str, Any]
    thread_id: str
    result: dict[str, Any]
    error: str
    trace: Trace


async def _retry_stage(
    name: str,
    action: Callable[[], Awaitable[None]],
    *,
    attempts: int = 2,
) -> tuple[str, str, list[str]]:
    trace: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            await action()
            trace.append(f"{name}:complete")
            return "", "", trace
        except Exception as exc:
            trace.append(f"{name}:retry:{attempt}:{type(exc).__name__}")
            if attempt == attempts:
                return str(exc) or type(exc).__name__, type(exc).__name__, trace
    return "unreachable retry state", "RuntimeError", trace


async def _retry_value_stage(
    name: str,
    action: Callable[[], Awaitable[Any]],
    *,
    attempts: int = 2,
) -> tuple[Any | None, str, list[str]]:
    trace: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            value = await action()
            trace.append(f"{name}:complete")
            return value, "", trace
        except Exception as exc:
            trace.append(f"{name}:retry:{attempt}:{type(exc).__name__}")
            if attempt == attempts:
                return None, str(exc) or type(exc).__name__, trace
    return None, "unreachable retry state", trace


def build_workflows(
    ingestion: IngestionService,
    qa_agent: QAAgent,
    update_agent: KnowledgeUpdateAgent,
    *,
    checkpointer: Any | None = None,
) -> dict[str, Any]:
    """Build the three stable public subgraphs."""
    return {
        "ingest": _build_ingest_graph(ingestion, checkpointer),
        "qa": _build_qa_graph(qa_agent, checkpointer),
        "update": _build_update_graph(update_agent, checkpointer),
    }


def build_supervisor_workflow(
    workflows: dict[str, Any],
    *,
    checkpointer: Any | None = None,
) -> Any:
    """Build the fourth, supervisor agent that dispatches to one of the public subgraphs."""

    async def validate(state: SupervisorState) -> dict[str, Any]:
        workflow = state.get("workflow", "")
        if workflow not in workflows:
            return {
                "error": f"unsupported workflow: {workflow}",
                "trace": ["supervisor:invalid_route"],
            }
        return {"trace": [f"supervisor:route:{workflow}"]}

    def route(state: SupervisorState) -> str:
        return "fail" if state.get("error") else state["workflow"]

    def runner(name: str) -> Callable[[SupervisorState], Awaitable[dict[str, Any]]]:
        async def run(state: SupervisorState) -> dict[str, Any]:
            result = await workflows[name].ainvoke(
                state.get("payload", {}),
                config={"configurable": {"thread_id": f"{state.get('thread_id', 'supervisor')}:{name}"}},
            )
            return {"result": result, "trace": [f"supervisor:{name}:complete"]}

        return run

    graph = StateGraph(SupervisorState)
    graph.add_node("validate", validate)
    for name in ("ingest", "qa", "update"):
        graph.add_node(name, runner(name))
        graph.add_edge(name, END)
    graph.add_node("fail", lambda state: {"trace": ["supervisor:failed"]})
    graph.set_entry_point("validate")
    graph.add_conditional_edges(
        "validate",
        route,
        {"ingest": "ingest", "qa": "qa", "update": "update", "fail": "fail"},
    )
    graph.add_edge("fail", END)
    return graph.compile(checkpointer=checkpointer)


def _build_ingest_graph(ingestion: IngestionService, checkpointer: Any | None) -> Any:
    async def validate(state: IngestState) -> dict[str, Any]:
        paths = state.get("file_paths", [])
        doc_ids = state.get("doc_ids", [])
        mime_types = state.get("mime_types", [])
        if not paths:
            return {
                "error": "at least one file is required",
                "error_type": "ValueError",
                "trace": ["ingest:validate:failed"],
            }
        if not hasattr(ingestion, "validate_stage"):
            return {"legacy": True, "trace": ["ingest:validate:complete"]}
        try:
            transactions = [
                ingestion.validate_stage(
                    path,
                    doc_id=doc_ids[index] if index < len(doc_ids) else None,
                    mime_type=mime_types[index] if index < len(mime_types) else "",
                )
                for index, path in enumerate(paths)
            ]
            return {"transactions": transactions, "trace": ["ingest:validate:complete"]}
        except Exception as exc:
            return {
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
                "trace": [f"ingest:validate:failed:{type(exc).__name__}"],
            }

    def stage_node(
        stage_name: str,
        operation: Callable[[IngestTransaction], Any],
        *,
        synchronous: bool = False,
    ) -> Callable[[IngestState], Awaitable[dict[str, Any]]]:
        async def run(state: IngestState) -> dict[str, Any]:
            if state.get("legacy"):
                return {"trace": [f"ingest:{stage_name}:complete"]}

            async def apply() -> None:
                for transaction in state.get("transactions", []):
                    value = operation(transaction)
                    if not synchronous:
                        await value

            error, error_type, trace = await _retry_stage(f"ingest:{stage_name}", apply)
            return {"error": error, "error_type": error_type, "trace": trace}

        return run

    async def commit(state: IngestState) -> dict[str, Any]:
        try:
            if state.get("legacy"):
                paths = state.get("file_paths", [])
                doc_ids = state.get("doc_ids", [])
                mime_types = state.get("mime_types", [])
                results = [
                    await ingestion.ingest(
                        path,
                        doc_id=doc_ids[index] if index < len(doc_ids) else None,
                        mime_type=mime_types[index] if index < len(mime_types) else "",
                    )
                    for index, path in enumerate(paths)
                ]
            else:
                results = [ingestion.commit_stage(tx) for tx in state.get("transactions", [])]
            return {"results": results, "trace": ["ingest:commit:complete"]}
        except Exception as exc:
            return {
                "error": str(exc) or type(exc).__name__,
                "error_type": type(exc).__name__,
                "trace": [f"ingest:commit:failed:{type(exc).__name__}"],
            }

    async def verify(state: IngestState) -> dict[str, Any]:
        if state.get("legacy"):
            return {"trace": ["ingest:verify:complete"]}
        checks = [await ingestion.verify_stage(transaction) for transaction in state.get("transactions", [])]
        if not all(checks):
            return {
                "error": "cross-store consistency verification failed",
                "error_type": "ConsistencyError",
                "trace": ["ingest:verify:failed"],
            }
        return {"trace": ["ingest:verify:complete"]}

    async def rollback(state: IngestState) -> dict[str, Any]:
        for transaction in state.get("transactions", []):
            await ingestion.rollback_stage(transaction, state.get("error", "ingestion failed"))
        return {"trace": [f"ingest:rollback:{state.get('error_type', 'Error')}"]}

    def failed_or(next_node: str) -> Callable[[IngestState], str]:
        return lambda state: "rollback" if state.get("error") else next_node

    def after_diff(state: IngestState) -> str:
        if state.get("error"):
            return "rollback"
        transactions = state.get("transactions", [])
        if state.get("legacy"):
            return "unchanged"
        return "unchanged" if all(tx.unchanged_result is not None for tx in transactions) else "embed"

    graph = StateGraph(IngestState)
    graph.add_node("validate", validate)
    graph.add_node("parse", stage_node("parse", getattr(ingestion, "parse_stage", lambda _tx: None)))
    graph.add_node(
        "diff",
        stage_node("diff", getattr(ingestion, "diff_stage", lambda _tx: None), synchronous=True),
    )
    graph.add_node("embed", stage_node("embed", getattr(ingestion, "embed_stage", lambda _tx: None)))
    graph.add_node("extract", stage_node("extract", getattr(ingestion, "extract_stage", lambda _tx: None)))
    graph.add_node(
        "graph_upsert",
        stage_node("graph_upsert", getattr(ingestion, "graph_upsert_stage", lambda _tx: None)),
    )
    graph.add_node(
        "delete_removed",
        stage_node("delete_removed", getattr(ingestion, "delete_removed_stage", lambda _tx: None)),
    )
    graph.add_node("commit", commit)
    graph.add_node("verify", verify)
    graph.add_node("rollback", rollback)
    graph.set_entry_point("validate")
    graph.add_conditional_edges("validate", failed_or("parse"), {"parse": "parse", "rollback": "rollback"})
    graph.add_conditional_edges("parse", failed_or("diff"), {"diff": "diff", "rollback": "rollback"})
    graph.add_conditional_edges(
        "diff", after_diff, {"embed": "embed", "unchanged": "commit", "rollback": "rollback"}
    )
    graph.add_conditional_edges("embed", failed_or("extract"), {"extract": "extract", "rollback": "rollback"})
    graph.add_conditional_edges(
        "extract", failed_or("graph_upsert"), {"graph_upsert": "graph_upsert", "rollback": "rollback"}
    )
    graph.add_conditional_edges(
        "graph_upsert",
        failed_or("delete_removed"),
        {"delete_removed": "delete_removed", "rollback": "rollback"},
    )
    graph.add_conditional_edges(
        "delete_removed", failed_or("commit"), {"commit": "commit", "rollback": "rollback"}
    )
    graph.add_conditional_edges("commit", failed_or("verify"), {"verify": "verify", "rollback": "rollback"})
    graph.add_conditional_edges("verify", failed_or("done"), {"done": END, "rollback": "rollback"})
    graph.add_edge("rollback", END)
    return graph.compile(checkpointer=checkpointer)


def _build_qa_graph(qa_agent: QAAgent, checkpointer: Any | None) -> Any:
    async def validate(state: QAState) -> dict[str, Any]:
        question = state.get("question", "").strip()
        if not question:
            return {"error": "question is required", "trace": ["qa:validate:failed"]}
        return {"question": question, "trace": ["qa:validate:complete"]}

    async def plan(state: QAState) -> dict[str, Any]:
        async def execute() -> QueryPlan:
            return (
                await qa_agent.pipeline.plan_query(state["question"])
                if hasattr(qa_agent, "pipeline")
                else QueryPlan(queries=[state["question"]])
            )

        plan_value, error, trace = await _retry_value_stage("qa:plan", execute)
        if error:
            return {"error": error, "trace": trace}
        return {"plan": plan_value, "trace": [*trace, f"qa:plan:{plan_value.intent}"]}

    async def vector_retrieve(state: QAState) -> dict[str, Any]:
        if state.get("error"):
            return {"trace": ["qa:vector_retrieve:skipped_error"]}
        if not hasattr(qa_agent, "pipeline") or state.get("mode", "hybrid") == RetrievalMode.GRAPH.value:
            result = RetrievalResult(state["plan"], [], 0.0, ["vector:skipped"])
        else:

            async def execute() -> RetrievalResult:
                return await qa_agent.pipeline.retrieve_candidates(
                    state["question"],
                    plan=state["plan"],
                    mode=RetrievalMode.VECTOR,
                    candidate_k=max(settings.rerank_candidate_k, state.get("top_k", 8) * 4),
                    document_ids=state.get("document_ids") or None,
                )

            result, error, trace = await _retry_value_stage("qa:vector_retrieve", execute)
            if error:
                return {"vector_error": error, "trace": trace}
            return {"vector_retrieval": result, "trace": trace}
        return {"vector_retrieval": result, "trace": ["qa:vector_retrieve:complete"]}

    async def graph_retrieve(state: QAState) -> dict[str, Any]:
        if state.get("error"):
            return {"trace": ["qa:graph_retrieve:skipped_error"]}
        if not hasattr(qa_agent, "pipeline") or state.get("mode", "hybrid") == RetrievalMode.VECTOR.value:
            result = RetrievalResult(state["plan"], [], 0.0, ["graph:skipped"])
        else:

            async def execute() -> RetrievalResult:
                return await qa_agent.pipeline.retrieve_candidates(
                    state["question"],
                    plan=state["plan"],
                    mode=RetrievalMode.GRAPH,
                    candidate_k=max(settings.rerank_candidate_k, state.get("top_k", 8) * 4),
                    document_ids=state.get("document_ids") or None,
                )

            result, error, trace = await _retry_value_stage("qa:graph_retrieve", execute)
            if error:
                return {"graph_error": error, "trace": trace}
            return {"graph_retrieval": result, "trace": trace}
        return {"graph_retrieval": result, "trace": ["qa:graph_retrieve:complete"]}

    async def rrf_fuse(state: QAState) -> dict[str, Any]:
        mode = RetrievalMode(state.get("mode", "hybrid"))
        vector_error = state.get("vector_error", "")
        graph_error = state.get("graph_error", "")
        fatal = state.get("error", "")
        if mode == RetrievalMode.VECTOR and vector_error:
            fatal = vector_error
        elif mode == RetrievalMode.GRAPH and graph_error:
            fatal = graph_error
        elif mode == RetrievalMode.HYBRID and vector_error and graph_error:
            fatal = f"vector={vector_error}; graph={graph_error}"
        if fatal:
            return {"error": fatal, "trace": ["qa:rrf_fuse:failed"]}
        vector = state.get("vector_retrieval") or RetrievalResult(state["plan"], [], 0.0, [])
        graph_result = state.get("graph_retrieval") or RetrievalResult(state["plan"], [], 0.0, [])
        if mode == RetrievalMode.VECTOR:
            count = len(vector.contexts)
        elif mode == RetrievalMode.GRAPH:
            count = len(graph_result.contexts)
        else:
            count = (
                len(qa_agent.pipeline._rrf_fuse(vector.contexts, graph_result.contexts))
                if hasattr(qa_agent, "pipeline")
                else 0
            )
        trace = [f"qa:rrf_fuse:{count}"]
        if vector_error or graph_error:
            trace.append("qa:rrf_fuse:single_retriever_fallback")
        return {"trace": trace}

    async def rerank_docs(state: QAState) -> dict[str, Any]:
        return {"trace": ["qa:rerank_docs:selected"]}

    async def document_refine(state: QAState) -> dict[str, Any]:
        return {"trace": ["qa:document_refine:selected"]}

    async def rerank_chunks(state: QAState) -> dict[str, Any]:
        vector = state.get("vector_retrieval") or RetrievalResult(state["plan"], [], 0.0, [])
        graph_result = state.get("graph_retrieval") or RetrievalResult(state["plan"], [], 0.0, [])

        async def execute() -> RetrievalResult:
            if hasattr(qa_agent, "pipeline"):
                return await qa_agent.pipeline.finalize_candidates(
                    state["question"],
                    plan=state["plan"],
                    mode=state.get("mode", "hybrid"),
                    vector=vector,
                    graph=graph_result,
                    top_k=state.get("top_k", 8),
                    document_ids=state.get("document_ids") or None,
                )
            return RetrievalResult(state["plan"], [], 0.0, ["selected_contexts:0"])

        retrieval, error, trace = await _retry_value_stage("qa:rerank_chunks", execute)
        if error:
            return {"error": error, "trace": trace}
        return {"retrieval": retrieval, "trace": trace}

    async def evidence_route(state: QAState) -> dict[str, Any]:
        retrieval = state.get("retrieval")
        count = len(retrieval.contexts) if retrieval else 0
        return {"trace": [f"qa:evidence_route:{count}"]}

    def answer_route(state: QAState) -> str:
        if state.get("error"):
            return "fail"
        retrieval = state["retrieval"]
        if not retrieval.contexts:
            return "abstain"
        if (
            settings.comparison_tool_enabled
            and getattr(qa_agent, "_should_use_comparison", lambda _plan: False)(retrieval.plan)
        ):
            return "comparison_tool"
        if getattr(qa_agent, "_should_use_temporal", lambda _plan: False)(retrieval.plan):
            return "temporal_tool"
        if retrieval.plan.needs_calculation:
            return "calculator"
        return "answer"

    async def calculator(_state: QAState) -> dict[str, Any]:
        return {"trace": ["qa:calculator:selected"]}

    async def temporal_tool(state: QAState) -> dict[str, Any]:
        async def execute() -> QAResult:
            return await qa_agent.answer_temporal_from_retrieval(
                state["question"], state["retrieval"], started=time.perf_counter()
            )

        result, error, trace = await _retry_value_stage("qa:temporal_tool", execute)
        if not error:
            return {"result": result, "trace": trace}

        async def fallback() -> QAResult:
            return await qa_agent.answer_from_retrieval(
                state["question"],
                state["retrieval"],
                started=time.perf_counter(),
                use_temporal=False,
            )

        result, fallback_error, fallback_trace = await _retry_value_stage(
            "qa:temporal_fallback",
            fallback,
        )
        if fallback_error:
            return {"error": fallback_error, "trace": [*trace, *fallback_trace]}
        return {
            "result": result,
            "trace": [*trace, "qa:temporal_tool:fallback", *fallback_trace],
        }

    async def comparison_tool(state: QAState) -> dict[str, Any]:
        async def execute() -> QAResult:
            return await qa_agent.answer_comparison_from_retrieval(
                state["question"], state["retrieval"], started=time.perf_counter()
            )

        result, error, trace = await _retry_value_stage("qa:comparison_tool", execute)
        if not error:
            return {"result": result, "trace": trace}

        async def fallback() -> QAResult:
            if getattr(qa_agent, "_should_use_temporal", lambda _plan: False)(
                state["retrieval"].plan
            ):
                return await qa_agent.answer_temporal_from_retrieval(
                    state["question"], state["retrieval"], started=time.perf_counter()
                )
            return await qa_agent.answer_from_retrieval(
                state["question"],
                state["retrieval"],
                started=time.perf_counter(),
                use_temporal=False,
            )

        result, fallback_error, fallback_trace = await _retry_value_stage(
            "qa:comparison_fallback",
            fallback,
        )
        if fallback_error:
            return {"error": fallback_error, "trace": [*trace, *fallback_trace]}
        return {
            "result": result,
            "trace": [*trace, "qa:comparison_tool:fallback", *fallback_trace],
        }

    async def answer(state: QAState) -> dict[str, Any]:
        async def execute() -> QAResult:
            if hasattr(qa_agent, "answer_from_retrieval"):
                return await qa_agent.answer_from_retrieval(
                    state["question"],
                    state["retrieval"],
                    started=time.perf_counter(),
                    use_temporal=False,
                )
            return await qa_agent.answer(
                state["question"],
                top_k=state.get("top_k"),
                mode=state.get("mode", "hybrid"),
                document_ids=state.get("document_ids") or None,
            )

        result, error, trace = await _retry_value_stage("qa:answer", execute)
        if error:
            return {"error": error, "trace": trace}
        return {"result": result, "trace": trace}

    async def abstain(state: QAState) -> dict[str, Any]:
        async def execute() -> QAResult:
            if hasattr(qa_agent, "answer_from_retrieval"):
                return await qa_agent.answer_from_retrieval(
                    state["question"], state["retrieval"], started=time.perf_counter()
                )
            return await qa_agent.answer(
                state["question"],
                top_k=state.get("top_k"),
                mode=state.get("mode", "hybrid"),
                document_ids=state.get("document_ids") or None,
            )

        result, error, trace = await _retry_value_stage("qa:abstain", execute)
        if error:
            return {"error": error, "trace": trace}
        return {"result": result, "trace": [*trace, "qa:abstain:no_context"]}

    async def citation_validate(state: QAState) -> dict[str, Any]:
        result = state["result"]
        if result.answerable and not result.citations:
            result.answerable = False
            return {"result": result, "trace": ["qa:citation_validate:failed"]}
        return {"trace": ["qa:citation_validate:complete"]}

    async def finalize(state: QAState) -> dict[str, Any]:
        result = state["result"]
        result.trace = [*state.get("trace", []), *result.trace, "qa:finalize"]
        return {"result": result}

    async def fail(state: QAState) -> dict[str, Any]:
        del state
        return {"trace": ["qa:failed"]}

    def failed_or(next_node: str) -> Callable[[QAState], str]:
        return lambda state: "fail" if state.get("error") else next_node

    graph = StateGraph(QAState)
    graph.add_node("validate", validate)
    graph.add_node("plan", plan)
    graph.add_node("vector_retrieve", vector_retrieve)
    graph.add_node("graph_retrieve", graph_retrieve)
    graph.add_node("rrf_fuse", rrf_fuse)
    graph.add_node("rerank_docs", rerank_docs)
    graph.add_node("document_refine", document_refine)
    graph.add_node("rerank_chunks", rerank_chunks)
    graph.add_node("evidence_route", evidence_route)
    graph.add_node("comparison_tool", comparison_tool)
    graph.add_node("temporal_tool", temporal_tool)
    graph.add_node("calculator", calculator)
    graph.add_node("answer", answer)
    graph.add_node("abstain", abstain)
    graph.add_node("citation_validate", citation_validate)
    graph.add_node("finalize", finalize)
    graph.add_node("fail", fail)
    graph.set_entry_point("validate")
    graph.add_conditional_edges("validate", failed_or("plan"), {"plan": "plan", "fail": "fail"})
    graph.add_edge("plan", "vector_retrieve")
    graph.add_edge("plan", "graph_retrieve")
    graph.add_edge("vector_retrieve", "rrf_fuse")
    graph.add_edge("graph_retrieve", "rrf_fuse")
    graph.add_conditional_edges(
        "rrf_fuse", failed_or("rerank_docs"), {"rerank_docs": "rerank_docs", "fail": "fail"}
    )
    graph.add_edge("rerank_docs", "document_refine")
    graph.add_edge("document_refine", "rerank_chunks")
    graph.add_conditional_edges(
        "rerank_chunks",
        failed_or("evidence_route"),
        {"evidence_route": "evidence_route", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "evidence_route",
        answer_route,
        {
            "comparison_tool": "comparison_tool",
            "temporal_tool": "temporal_tool",
            "calculator": "calculator",
            "answer": "answer",
            "abstain": "abstain",
            "fail": "fail",
        },
    )
    graph.add_edge("calculator", "answer")
    graph.add_conditional_edges(
        "comparison_tool",
        failed_or("citation_validate"),
        {"citation_validate": "citation_validate", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "temporal_tool",
        failed_or("citation_validate"),
        {"citation_validate": "citation_validate", "fail": "fail"},
    )
    graph.add_conditional_edges(
        "answer", failed_or("citation_validate"), {"citation_validate": "citation_validate", "fail": "fail"}
    )
    graph.add_conditional_edges(
        "abstain", failed_or("citation_validate"), {"citation_validate": "citation_validate", "fail": "fail"}
    )
    graph.add_edge("citation_validate", "finalize")
    graph.add_edge("finalize", END)
    graph.add_edge("fail", END)
    return graph.compile(checkpointer=checkpointer)


def _build_update_graph(update_agent: KnowledgeUpdateAgent, checkpointer: Any | None) -> Any:
    async def validate_event(state: UpdateState) -> dict[str, Any]:
        events = state.get("events", [])
        if not events or any(event.operation not in {"INSERT", "UPDATE", "DELETE"} for event in events):
            return {"error": "invalid CDC event", "trace": ["cdc:validate_event:failed"]}
        return {"trace": [f"cdc:validate_event:{len(events)}"]}

    async def idempotency(state: UpdateState) -> dict[str, Any]:
        active: list[CDCEvent] = []
        duplicates: list[UpdateResult] = []
        for event in state.get("events", []):
            existing = (
                update_agent.catalog.get_event(event.event_id) if hasattr(update_agent, "catalog") else None
            )
            if existing and existing.status == "COMMITTED":
                duplicates.append(
                    UpdateResult(
                        event.event_id,
                        existing.doc_id or event.doc_id,
                        event.file_path,
                        "COMMITTED",
                        duplicate=True,
                    )
                )
            else:
                active.append(event)
        return {
            "active_events": active,
            "duplicate_results": duplicates,
            "trace": [f"cdc:idempotency:duplicates:{len(duplicates)}"],
        }

    def operation_route(state: UpdateState) -> str:
        if state.get("error"):
            return "fail"
        active = state.get("active_events", [])
        if not active:
            return "commit"
        return "delete" if all(event.operation == "DELETE" for event in active) else "upsert"

    async def delete(_state: UpdateState) -> dict[str, Any]:
        return {"trace": ["cdc:route:delete"]}

    async def upsert(_state: UpdateState) -> dict[str, Any]:
        return {"trace": ["cdc:route:upsert"]}

    async def diff(state: UpdateState) -> dict[str, Any]:
        changed = sum(event.operation != "DELETE" for event in state.get("active_events", []))
        return {"trace": [f"cdc:diff:upserts:{changed}"]}

    async def apply(state: UpdateState) -> dict[str, Any]:
        results = [await update_agent.process_event(event) for event in state.get("active_events", [])]
        return {"active_results": results, "trace": [f"cdc:apply:{len(results)}"]}

    async def verify(state: UpdateState) -> dict[str, Any]:
        if state.get("error") and state.get("repair_attempted"):
            return {"trace": ["cdc:verify:repair_failed"]}
        failed = [result for result in state.get("active_results", []) if not result.success]
        if failed:
            return {"error": failed[0].error or "CDC apply failed", "trace": ["cdc:verify:failed"]}
        ingestion = getattr(update_agent, "ingestion", None)
        if ingestion is not None:
            events = {event.event_id: event for event in state.get("active_events", [])}
            for result in state.get("active_results", []):
                event = events.get(result.event_id)
                doc_id = result.doc_id or (event.doc_id if event else "")
                if not doc_id:
                    continue
                expected = (
                    set()
                    if event and event.operation == "DELETE"
                    else {str(row["chunk_id"]) for row in update_agent.catalog.get_chunks(doc_id)}
                )
                vector_ids, graph_ids = await asyncio.gather(
                    ingestion.vector_store.get_document_chunks(doc_id),
                    ingestion.knowledge_graph.get_document_chunks(doc_id),
                )
                if expected != set(vector_ids) or expected != set(graph_ids):
                    return {
                        "error": f"cross-store consistency verification failed for {doc_id}",
                        "trace": ["cdc:verify:failed:ConsistencyError"],
                    }
        return {"trace": ["cdc:verify:complete"]}

    def verify_route(state: UpdateState) -> str:
        if not state.get("error"):
            return "commit"
        return "fail" if state.get("repair_attempted") else "repair"

    async def repair(state: UpdateState) -> dict[str, Any]:
        try:
            ingestion = update_agent.ingestion
            repaired: set[str] = set()
            for event in state.get("active_events", []):
                doc_id = event.doc_id
                if not doc_id or doc_id in repaired:
                    continue
                if event.operation == "DELETE":
                    await asyncio.gather(
                        ingestion.vector_store.delete_by_doc_id(doc_id),
                        ingestion.knowledge_graph.delete_by_doc_id(doc_id),
                    )
                else:
                    await ingestion.repair_consistency(doc_id)
                repaired.add(doc_id)
            return {
                "error": "",
                "repair_attempted": True,
                "trace": [f"cdc:repair:complete:{len(repaired)}"],
            }
        except Exception as exc:
            return {
                "error": str(exc) or type(exc).__name__,
                "repair_attempted": True,
                "trace": [f"cdc:repair:failed:{type(exc).__name__}"],
            }

    async def commit(state: UpdateState) -> dict[str, Any]:
        results = [*state.get("duplicate_results", []), *state.get("active_results", [])]
        return {"results": results, "trace": [f"cdc:commit:{len(results)}"]}

    async def fail(state: UpdateState) -> dict[str, Any]:
        results = [*state.get("duplicate_results", []), *state.get("active_results", [])]
        return {"results": results, "trace": ["cdc:fail"]}

    graph = StateGraph(UpdateState)
    graph.add_node("validate_event", validate_event)
    graph.add_node("idempotency", idempotency)
    graph.add_node("delete", delete)
    graph.add_node("upsert", upsert)
    graph.add_node("diff", diff)
    graph.add_node("apply", apply)
    graph.add_node("verify", verify)
    graph.add_node("repair", repair)
    graph.add_node("commit", commit)
    graph.add_node("fail", fail)
    graph.set_entry_point("validate_event")
    graph.add_edge("validate_event", "idempotency")
    graph.add_conditional_edges(
        "idempotency",
        operation_route,
        {"delete": "delete", "upsert": "upsert", "commit": "commit", "fail": "fail"},
    )
    graph.add_edge("delete", "diff")
    graph.add_edge("upsert", "diff")
    graph.add_edge("diff", "apply")
    graph.add_edge("apply", "verify")
    graph.add_conditional_edges(
        "verify", verify_route, {"commit": "commit", "repair": "repair", "fail": "fail"}
    )
    graph.add_edge("repair", "verify")
    graph.add_edge("commit", END)
    graph.add_edge("fail", END)
    return graph.compile(checkpointer=checkpointer)


build_knowledge_graph_workflow = build_workflows
