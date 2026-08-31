from __future__ import annotations

import json

import pytest

from services.graph_rag import (
    AnswerType,
    EvidenceSlot,
    GraphRAGContext,
    GraphRAGPipeline,
    QueryPlan,
    TargetRelation,
)
from services.temporal import ObservedRelation, TemporalComparisonTool, TemporalEvidence
from tests.conftest import FakeChatModel


def context(chunk_id: str) -> GraphRAGContext:
    return GraphRAGContext(chunk_id, f"doc-{chunk_id}", "evidence", "file", "hybrid", 0.9)


def plan(target: TargetRelation) -> QueryPlan:
    return QueryPlan(
        intent="temporal",
        queries=["left", "right"],
        answer_type=AnswerType.BOOLEAN,
        evidence_slots=[
            EvidenceSlot(slot_id="left", query="left", role="left"),
            EvidenceSlot(slot_id="right", query="right", role="right"),
        ],
        target_relation=target,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("target", "observed", "expected"),
    [
        (TargetRelation.CHANGED_CLAIM, "changed_claim", "Yes"),
        (TargetRelation.SAME_CLAIM, "changed_claim", "No"),
        (TargetRelation.SAME_CLAIM, "same_claim", "Yes"),
    ],
)
async def test_temporal_claim_comparison_renders_yes_no(
    target: TargetRelation,
    observed: str,
    expected: str,
) -> None:
    response = {
        "evidence": [
            {"slot_id": "left", "claim": "A", "citation_id": "c1"},
            {"slot_id": "right", "claim": "B", "citation_id": "c2"},
        ],
        "observed_relation": observed,
    }
    tool = TemporalComparisonTool(FakeChatModel([json.dumps(response)]))
    result = await tool.compare("Did it change?", plan(target), [context("c1"), context("c2")])
    assert result.answer == expected
    assert result.answerable
    assert result.citation_ids == ["c1", "c2"]


@pytest.mark.asyncio
async def test_temporal_date_comparison_is_deterministic() -> None:
    response = {
        "evidence": [
            {"slot_id": "left", "date": "2024-02-01", "citation_id": "c1"},
            {"slot_id": "right", "date": "2024-03-01", "citation_id": "c2"},
        ],
        "observed_relation": "after",
    }
    tool = TemporalComparisonTool(FakeChatModel([json.dumps(response)]))
    result = await tool.compare(
        "Was the first event before the second?",
        plan(TargetRelation.BEFORE),
        [context("c1"), context("c2")],
    )
    assert result.answer == "Yes"
    assert result.observed_relation == ObservedRelation.BEFORE


@pytest.mark.asyncio
async def test_temporal_abstains_on_missing_slot_conflict_and_invalid_citation() -> None:
    responses = [
        {
            "evidence": [{"slot_id": "left", "claim": "A", "citation_id": "c1"}],
            "observed_relation": "same_claim",
        },
        {
            "evidence": [
                {"slot_id": "left", "date": "2024-01-01", "citation_id": "c1"},
                {"slot_id": "left", "date": "2025-01-01", "citation_id": "c1"},
                {"slot_id": "right", "date": "2024-06-01", "citation_id": "c2"},
            ],
            "observed_relation": "before",
        },
        {
            "evidence": [
                {"slot_id": "left", "claim": "A", "citation_id": "invented"},
                {"slot_id": "right", "claim": "B", "citation_id": "c2"},
            ],
            "observed_relation": "changed_claim",
        },
    ]
    model = FakeChatModel([json.dumps(value) for value in responses])
    tool = TemporalComparisonTool(model)
    contexts = [context("c1"), context("c2")]
    missing = await tool.compare("q", plan(TargetRelation.SAME_CLAIM), contexts)
    conflict = await tool.compare("q", plan(TargetRelation.BEFORE), contexts)
    invalid = await tool.compare("q", plan(TargetRelation.CHANGED_CLAIM), contexts)
    assert not missing.answerable
    assert not conflict.answerable
    assert not invalid.answerable


def test_temporal_date_parser_and_fallback_negations() -> None:
    evidence = [TemporalEvidence(slot_id="left", date="2024", citation_id="c1")]
    assert TemporalComparisonTool._single_date(evidence).isoformat() == "2024-01-01"
    assert TemporalComparisonTool._parse_date("2024-13-99") is None
    assert GraphRAGPipeline._fallback_plan("Was the statement not inconsistent?").target_relation == (
        TargetRelation.SAME_CLAIM
    )
    assert GraphRAGPipeline._fallback_plan("Was the statement not consistent?").target_relation == (
        TargetRelation.CHANGED_CLAIM
    )
