from __future__ import annotations

from typing import Any

import pytest

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from services.knowledge_graph import KnowledgeGraphService


class RecordingGraph(KnowledgeGraphService):
    def __init__(self) -> None:
        super().__init__(driver=object())
        self.queries: list[tuple[str, dict[str, Any]]] = []

    async def execute_cypher(self, cypher: str, params: dict[str, Any] | None = None):
        self.queries.append((cypher, params or {}))
        if "collect(c.chunk_id)" in cypher:
            return [{"ids": ["c1", "c2"]}]
        if "RETURN count(c) AS deleted" in cypher:
            return [{"deleted": 2}]
        return []


@pytest.mark.asyncio
async def test_graph_search_uses_parameters_and_caps_hops() -> None:
    graph = RecordingGraph()
    malicious = "Alice') MATCH (n) DETACH DELETE n //"
    await graph.graph_search([malicious], hops=99, document_ids=["doc"], limit=7)
    assert len(graph.queries) == 2
    cypher, params = graph.queries[0]
    assert all(malicious not in query for query, _params in graph.queries)
    assert params["keys"] == [malicious.casefold()]
    assert "{key:requested_key}" in cypher
    assert "RELATION*" not in "\n".join(query for query, _params in graph.queries)
    assert params["doc_ids"] == ["doc"]
    assert graph.queries[1][1]["limit"] == 200


@pytest.mark.asyncio
async def test_delete_document_removes_sources_then_orphans() -> None:
    graph = RecordingGraph()
    assert await graph.delete_by_doc_id("doc") == 2
    assert len(graph.queries) == 3
    assert "source_chunk_id" in graph.queries[1][0]
    assert "NOT (e)-[:MENTIONED_IN]" in graph.queries[2][0]


class FakeResult:
    def __init__(self, query: str) -> None:
        self.query = query

    async def data(self):
        if "RETURN entities,relations" in self.query:
            return [{"entities": 2, "relations": 1, "chunks": 1}]
        if "RETURN c.chunk_id AS chunk_id" in self.query:
            return [{"chunk_id": "c1"}]
        if "RETURN e.name AS name" in self.query:
            return [{"name": "Alice", "type": "Person", "description": "Owner"}]
        return []


class FakeSession:
    def __init__(self, queries: list[str]) -> None:
        self.queries = queries

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: Any):
        return None

    async def run(self, query: str, *_args: Any, **_kwargs: Any):
        self.queries.append(query)
        return FakeResult(query)


class FakeDriver:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.closed = False

    async def verify_connectivity(self):
        return None

    def session(self):
        return FakeSession(self.queries)

    async def close(self):
        self.closed = True


@pytest.mark.asyncio
async def test_graph_lifecycle_upsert_and_read_helpers() -> None:
    driver = FakeDriver()
    graph = KnowledgeGraphService(driver)
    await graph.init()
    chunk = DocumentChunk(
        "Alice owns Atlas",
        "doc",
        0,
        DocType.TEXT,
        {"source": "a.txt", "file_name": "a.txt", "page": 1},
        identifier="c1",
    )
    extraction = ExtractionResult(
        entities=[Entity(name="Alice", type="Person"), Entity(name="Atlas", type="Product")],
        relations=[Relation(head="Alice", relation="owns", tail="Atlas", confidence=0.9)],
        source_chunk_id="c1",
    )
    assert await graph.upsert_extraction(extraction, chunk, version=1) == (2, 1)
    await graph.upsert_entity(Entity(name="Bob"), source="legacy")
    await graph.add_relation(Relation(head="Alice", relation="knows", tail="Bob"))
    assert (await graph.search_entities("ali"))[0]["name"] == "Alice"
    assert await graph.get_document_chunks("doc") == ["c1"]
    assert (await graph.get_stats())["entities"] == 2
    assert await graph.health()
    assert any("CREATE CONSTRAINT" in query for query in driver.queries)
    await graph.close()
    assert driver.closed
