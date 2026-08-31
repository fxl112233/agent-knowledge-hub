from __future__ import annotations

import json

import pytest

from services.comparison import ComparisonVerificationTool
from services.graph_rag import EvidenceSlot, GraphRAGContext, QueryPlan
from tests.conftest import FakeChatModel


def _plan() -> QueryPlan:
    return QueryPlan(
        queries=["claim A", "claim B"],
        evidence_slots=[
            EvidenceSlot(slot_id="left", query="claim A"),
            EvidenceSlot(slot_id="right", query="claim B"),
        ],
    )


def _contexts() -> list[GraphRAGContext]:
    return [
        GraphRAGContext("left-id", "left-doc", "claim A", "a", "hybrid", 1.0),
        GraphRAGContext("right-id", "right-doc", "claim B", "b", "hybrid", 0.9),
    ]


@pytest.mark.asyncio
async def test_comparison_verifies_complete_question_with_grounded_slots() -> None:
    response = {
        "evidence": [
            {
                "slot_id": "left",
                "claim": "A",
                "supported": True,
                "citation_id": "left-id",
            },
            {
                "slot_id": "right",
                "claim": "B",
                "supported": True,
                "citation_id": "right-id",
            },
        ],
        "verdict": "yes",
    }
    result = await ComparisonVerificationTool(FakeChatModel([json.dumps(response)])).compare(
        "Does A hold while B holds?", _plan(), _contexts()
    )

    assert result.answer == "Yes"
    assert result.answerable
    assert result.citation_ids == ["left-id", "right-id"]


@pytest.mark.asyncio
async def test_comparison_abstains_on_missing_slot_or_invented_citation() -> None:
    missing = {
        "evidence": [
            {
                "slot_id": "left",
                "claim": "A",
                "supported": True,
                "citation_id": "left-id",
            }
        ],
        "verdict": "unknown",
    }
    result = await ComparisonVerificationTool(FakeChatModel([json.dumps(missing)])).compare(
        "question", _plan(), _contexts()
    )
    assert not result.answerable
    assert result.trace == ["comparison:missing_slot_evidence"]

    invented = {
        "evidence": [
            {
                "slot_id": "left",
                "claim": "A",
                "supported": True,
                "citation_id": "invented",
            },
            {
                "slot_id": "right",
                "claim": "B",
                "supported": False,
                "citation_id": "right-id",
            },
        ],
        "verdict": "no",
    }
    result = await ComparisonVerificationTool(FakeChatModel([json.dumps(invented)])).compare(
        "question", _plan(), _contexts()
    )
    assert not result.answerable
    assert result.trace == ["comparison:invalid_citation"]
