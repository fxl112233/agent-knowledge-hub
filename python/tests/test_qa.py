from __future__ import annotations

import json

import pytest

from agents.qa_agent import QAAgent
from services.graph_rag import (
    AnswerType,
    EvidenceSlot,
    GraphRAGContext,
    QueryPlan,
    RetrievalResult,
    TargetRelation,
)
from tests.conftest import FakeChatModel


class FixedPipeline:
    def __init__(self, contexts: list[GraphRAGContext]) -> None:
        self.contexts = contexts

    async def retrieve(self, *_args, **_kwargs) -> RetrievalResult:
        return RetrievalResult(
            QueryPlan(intent="analytical", queries=["q"]), self.contexts, 12.5, ["retrieved"]
        )


def context() -> GraphRAGContext:
    return GraphRAGContext(
        chunk_id="chunk-1",
        doc_id="doc-1",
        content="Revenue was 100 and cost was 60.",
        source="report.pdf",
        source_type="hybrid",
        score=0.9,
        metadata={"file_name": "report.pdf", "page": 2},
    )


@pytest.mark.asyncio
async def test_answer_validates_citation_and_calculation() -> None:
    response = {
        "answer": "Profit was 40.",
        "answerable": True,
        "citation_ids": ["chunk-1", "invented"],
        "calculation": "100 - 60",
    }
    agent = QAAgent(pipeline=FixedPipeline([context()]), llm=FakeChatModel([json.dumps(response)]))
    result = await agent.answer("What was profit?")
    assert result.answerable
    assert [citation.chunk_id for citation in result.citations] == ["chunk-1"]
    assert result.citations[0].page == 2
    assert result.evidence_score == 0.9
    assert "calculator:verified" in result.trace
    assert result.usage["total_tokens"] == 13


@pytest.mark.asyncio
async def test_answer_abstains_without_context_or_valid_citations() -> None:
    no_context = QAAgent(pipeline=FixedPipeline([]), llm=FakeChatModel())
    result = await no_context.answer("missing")
    assert not result.answerable
    assert result.citations == []

    invalid = QAAgent(
        pipeline=FixedPipeline([context()]),
        llm=FakeChatModel([json.dumps({"answer": "guess", "answerable": True, "citation_ids": ["x"]})]),
    )
    result = await invalid.answer("guess?")
    assert not result.answerable
    assert "answer:abstained_invalid_citations" in result.trace


@pytest.mark.asyncio
async def test_answer_invalid_json_exhaustion_abstains() -> None:
    agent = QAAgent(pipeline=FixedPipeline([context()]), llm=FakeChatModel(["bad", "bad", "bad"]))
    result = await agent.answer("question")
    assert not result.answerable


@pytest.mark.asyncio
async def test_answer_retries_model_invocation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class FlakyChatModel(FakeChatModel):
        attempts = 0

        async def ainvoke(self, messages, **kwargs):
            self.attempts += 1
            if self.attempts < 3:
                raise TimeoutError("temporary upstream failure")
            return await super().ainvoke(messages, **kwargs)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("agents.qa_agent.asyncio.sleep", no_sleep)
    model = FlakyChatModel(
        [json.dumps({"answer": "supported", "answerable": True, "citation_ids": ["chunk-1"]})]
    )
    agent = QAAgent(pipeline=FixedPipeline([context()]), llm=model)

    result = await agent.answer("question")

    assert result.answerable
    assert model.attempts == 3


@pytest.mark.asyncio
async def test_comparison_answer_uses_one_structured_call_and_yes_no_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.qa_agent.settings.comparison_tool_enabled", True)
    contexts = [
        context(),
        GraphRAGContext(
            "chunk-2",
            "doc-2",
            "Revenue later changed to 120.",
            "later.pdf",
            "hybrid",
            0.8,
        ),
    ]
    retrieval = RetrievalResult(
        QueryPlan(
            intent="temporal",
            queries=["old revenue", "new revenue"],
            answer_type=AnswerType.BOOLEAN,
            evidence_slots=[
                EvidenceSlot(slot_id="left", query="old revenue"),
                EvidenceSlot(slot_id="right", query="new revenue"),
            ],
            target_relation=TargetRelation.CHANGED_CLAIM,
        ),
        contexts,
        10,
        ["retrieved"],
    )
    response = {
        "evidence": [
            {
                "slot_id": "left",
                "claim": "100",
                "supported": True,
                "citation_id": "chunk-1",
            },
            {
                "slot_id": "right",
                "claim": "120",
                "supported": True,
                "citation_id": "chunk-2",
            },
        ],
        "verdict": "yes",
    }
    model = FakeChatModel([json.dumps(response)])
    agent = QAAgent(pipeline=FixedPipeline(contexts), llm=model)
    result = await agent.answer_from_retrieval("Did revenue change?", retrieval)
    assert result.answer == "Yes"
    assert result.answerable
    assert len(model.calls) == 1
    assert {value.chunk_id for value in result.citations} == {"chunk-1", "chunk-2"}


@pytest.mark.asyncio
async def test_generic_boolean_answer_rejects_same_different_output() -> None:
    retrieval = RetrievalResult(
        QueryPlan(queries=["q"], answer_type=AnswerType.BOOLEAN),
        [context()],
        1,
        [],
    )
    model = FakeChatModel([json.dumps({"answer": "Same", "answerable": True, "citation_ids": ["chunk-1"]})])
    agent = QAAgent(pipeline=FixedPipeline([context()]), llm=model)
    result = await agent.answer_from_retrieval("Is it the same?", retrieval)
    assert not result.answerable
    assert "answer:abstained_boolean_contract" in result.trace


@pytest.mark.asyncio
async def test_generic_answer_binds_one_real_citation_per_required_slot() -> None:
    left = context()
    left.metadata["rerank_slot_ids"] = ["left"]
    right = GraphRAGContext(
        "chunk-2",
        "doc-2",
        "Second required fact.",
        "second.pdf",
        "hybrid",
        0.8,
        {"rerank_slot_ids": ["right"]},
    )
    plan = QueryPlan(
        queries=["left", "right"],
        evidence_slots=[
            EvidenceSlot(slot_id="left", query="left"),
            EvidenceSlot(slot_id="right", query="right"),
        ],
    )
    retrieval = RetrievalResult(plan, [left, right], 1, [])
    model = FakeChatModel(
        [json.dumps({"answer": "supported", "answerable": True, "citation_ids": ["chunk-1"]})]
    )

    result = await QAAgent(pipeline=FixedPipeline([left, right]), llm=model).answer_from_retrieval(
        "question",
        retrieval,
    )

    assert {citation.chunk_id for citation in result.citations} == {"chunk-1", "chunk-2"}
    assert "citations:slot_bound:1" in result.trace


def test_generation_contexts_obey_chunk_and_character_budgets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.qa_agent.settings.answer_max_context_chunks", 2)
    monkeypatch.setattr("agents.qa_agent.settings.answer_max_context_chars", 10)
    values = [GraphRAGContext(str(index), str(index), "abcdefgh", "x", "vector", 1.0) for index in range(3)]

    selected = QAAgent._select_generation_contexts(values, QueryPlan())

    assert len(selected) == 2
    assert sum(len(value.content) for value in selected) == 10


def test_semantic_relation_routes_to_comparison_tool() -> None:
    plan = QueryPlan(
        intent="comparative",
        queries=["left", "right"],
        answer_type=AnswerType.BOOLEAN,
        target_relation=TargetRelation.SAME_CLAIM,
    )
    assert QAAgent._should_use_comparison(plan)
    assert QAAgent._should_use_temporal(plan)


@pytest.mark.asyncio
async def test_temporal_missing_slot_runs_exactly_one_supplement_then_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("agents.qa_agent.settings.comparison_tool_enabled", True)
    first = context()
    second = GraphRAGContext(
        "chunk-2",
        "doc-2",
        "Revenue later changed to 120.",
        "later.pdf",
        "hybrid",
        0.8,
    )
    plan = QueryPlan(
        intent="temporal",
        queries=["old revenue", "new revenue"],
        answer_type=AnswerType.BOOLEAN,
        evidence_slots=[
            EvidenceSlot(slot_id="left", query="old revenue"),
            EvidenceSlot(slot_id="right", query="new revenue"),
        ],
        target_relation=TargetRelation.CHANGED_CLAIM,
    )

    class SupplementalPipeline(FixedPipeline):
        def __init__(self) -> None:
            super().__init__([first])
            self.supplements = 0

        async def supplement_evidence(self, _question, retrieval, *, top_k):
            self.supplements += 1
            return RetrievalResult(
                retrieval.plan,
                [first, second],
                retrieval.retrieval_ms + 1,
                [*retrieval.trace, "temporal_supplement:applied"],
            )

    model = FakeChatModel(
        [
            json.dumps(
                {
                    "evidence": [
                        {
                            "slot_id": "left",
                            "claim": "100",
                            "supported": True,
                            "citation_id": "chunk-1",
                        }
                    ],
                    "verdict": "unknown",
                }
            ),
            json.dumps(
                {
                    "evidence": [
                        {
                            "slot_id": "left",
                            "claim": "100",
                            "supported": True,
                            "citation_id": "chunk-1",
                        },
                        {
                            "slot_id": "right",
                            "claim": "120",
                            "supported": True,
                            "citation_id": "chunk-2",
                        },
                    ],
                    "verdict": "yes",
                }
            ),
        ]
    )
    pipeline = SupplementalPipeline()
    retrieval = RetrievalResult(plan, [first], 1, ["retrieved"])
    result = await QAAgent(pipeline=pipeline, llm=model).answer_from_retrieval(
        "Did revenue change?",
        retrieval,
    )

    assert result.answer == "Yes"
    assert result.answerable
    assert pipeline.supplements == 1
    assert len(model.calls) == 2
    assert "temporal_supplement:applied" in result.trace
