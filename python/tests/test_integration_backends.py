from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import pytest

from agents.doc_parser_agent import DocParserAgent
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from services.catalog import CatalogService
from services.ingestion import IngestionService
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import VectorStoreService
from tests.conftest import FakeChatModel, FakeEmbedding, FakeVLEmbedding

pytestmark = pytest.mark.integration


@pytest.mark.asyncio
async def test_real_sqlite_chroma_and_neo4j_round_trip(tmp_path: Path) -> None:
    """Exercise actual persistence engines without calling a paid model API."""
    doc_id = f"integration-{uuid.uuid4().hex}"
    path = tmp_path / "knowledge.txt"
    path.write_text("阿尔法项目由星河公司研发，用于验证跨存储一致性。", encoding="utf-8")
    llm = FakeChatModel(
        [
            json.dumps(
                {
                    "entities": [
                        {"name": "阿尔法项目", "type": "Product", "description": "测试项目"},
                        {"name": "星河公司", "type": "Organization", "description": "研发方"},
                    ],
                    "relations": [
                        {
                            "head": "星河公司",
                            "relation": "研发",
                            "tail": "阿尔法项目",
                            "confidence": 1.0,
                        }
                    ],
                    "events": [],
                },
                ensure_ascii=False,
            )
        ]
    )
    catalog = CatalogService(str(tmp_path / "catalog.sqlite3"))
    vector = VectorStoreService(
        FakeEmbedding(dimensions=1024),
        FakeVLEmbedding(dimensions=1024),
    )
    graph = KnowledgeGraphService()
    await catalog.init()
    await vector.init()
    await graph.init()
    ingestion = IngestionService(
        DocParserAgent(),
        KnowledgeExtractAgent(llm),
        vector,
        graph,
        catalog,
    )
    try:
        result = await ingestion.ingest(str(path), doc_id=doc_id)
        assert result.status == "READY"
        assert result.chunks_total == 1
        assert result.entities_added == 2
        assert result.relations_added == 1

        expected_ids = {str(row["chunk_id"]) for row in catalog.get_chunks(doc_id)}
        assert set(await vector.get_document_chunks(doc_id)) == expected_ids
        assert set(await graph.get_document_chunks(doc_id)) == expected_ids

        vector_hits = await vector.search("谁研发了阿尔法项目？", document_ids=[doc_id])
        graph_hits = await graph.graph_search(["星河公司"], document_ids=[doc_id])
        assert vector_hits[0][0]["doc_id"] == doc_id
        assert graph_hits[0]["doc_id"] == doc_id

        assert await ingestion.delete(doc_id) == 1
        assert await vector.get_document_chunks(doc_id) == []
        assert await graph.get_document_chunks(doc_id) == []
    finally:
        await vector.delete_by_doc_id(doc_id)
        await graph.delete_by_doc_id(doc_id)
        await graph.close()
        await catalog.close()


def test_real_kafka_produce_consume_round_trip() -> None:
    from confluent_kafka import Consumer, Producer
    from confluent_kafka.admin import AdminClient, NewTopic

    from config import settings

    topic = f"akh-integration-{uuid.uuid4().hex}"
    value = json.dumps({"event_id": uuid.uuid4().hex, "operation": "UPDATE"}).encode()
    admin = AdminClient({"bootstrap.servers": settings.kafka_bootstrap_servers})
    admin.create_topics([NewTopic(topic, num_partitions=1, replication_factor=1)])[topic].result(10)
    producer = Producer({"bootstrap.servers": settings.kafka_bootstrap_servers})
    consumer = Consumer(
        {
            "bootstrap.servers": settings.kafka_bootstrap_servers,
            "group.id": f"akh-integration-{uuid.uuid4().hex}",
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([topic])
    try:
        producer.produce(topic, value=value)
        producer.flush(10)
        deadline = time.monotonic() + 15
        message = None
        while time.monotonic() < deadline:
            candidate = consumer.poll(1)
            if candidate is not None and candidate.error() is None:
                message = candidate
                break
        assert message is not None
        assert message.value() == value
    finally:
        consumer.close()
        admin.delete_topics([topic], operation_timeout=10)[topic].result(15)
