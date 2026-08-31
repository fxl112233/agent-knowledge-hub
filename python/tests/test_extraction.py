from __future__ import annotations

import json

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from tests.conftest import FakeChatModel


@pytest.mark.asyncio
async def test_structured_extraction_retries_deduplicates_and_adds_provenance() -> None:
    payload = {
        "entities": [
            {"name": "Alice", "type": "Person"},
            {"name": "alice", "type": "Person"},
            {"name": "Atlas", "type": "Product"},
        ],
        "relations": [
            {"head": "Alice", "relation": "owns", "tail": "Atlas", "confidence": 0.9},
            {"head": "Ghost", "relation": "owns", "tail": "Atlas", "confidence": 1.0},
        ],
        "events": [],
    }
    llm = FakeChatModel(["not-json", json.dumps(payload)])
    agent = KnowledgeExtractAgent(llm=llm)
    chunk = DocumentChunk("Alice owns Atlas", "doc", 0, DocType.TEXT, identifier="chunk-1")
    result = (await agent.extract([chunk]))[0]
    assert len(llm.calls) == 2
    assert len(result.entities) == 2
    assert len(result.relations) == 1
    assert result.relations[0].source_chunk_id == "chunk-1"


def test_parser_rejects_non_object() -> None:
    with pytest.raises(ValueError):
        KnowledgeExtractAgent._parse_response("[]", "chunk")
