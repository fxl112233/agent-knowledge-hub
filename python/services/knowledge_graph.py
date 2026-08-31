"""Neo4j service with chunk-level provenance and parameterized graph retrieval."""

from __future__ import annotations

import time
from typing import Any

from agents.doc_parser_agent import DocType, DocumentChunk
from agents.knowledge_extract_agent import Entity, ExtractionResult, Relation
from config import settings


class KnowledgeGraphService:
    def __init__(self, driver: Any | None = None) -> None:
        self._driver = driver

    async def init(self) -> None:
        if self._driver is None:
            from neo4j import AsyncGraphDatabase

            self._driver = AsyncGraphDatabase.driver(
                settings.neo4j_uri,
                auth=(settings.neo4j_user, settings.neo4j_password),
            )
        await self._driver.verify_connectivity()
        await self._ensure_indexes()

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()

    def _get_driver(self) -> Any:
        if self._driver is None:
            raise RuntimeError("knowledge graph is not initialized")
        return self._driver

    async def health(self) -> bool:
        try:
            await self._get_driver().verify_connectivity()
            return True
        except Exception:
            return False

    async def _ensure_indexes(self) -> None:
        queries = [
            "CREATE CONSTRAINT entity_key IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
            "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE",
            "CREATE INDEX chunk_doc_id IF NOT EXISTS FOR (c:Chunk) ON (c.doc_id)",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
        ]
        async with self._get_driver().session() as session:
            for query in queries:
                await session.run(query)
            # Sentinels keep the property schema uniform and avoid Neo4j warning
            # notifications when a source format has no page or sheet.
            await session.run("MATCH (c:Chunk) SET c.page=coalesce(c.page,0),c.sheet=coalesce(c.sheet,'')")

    async def upsert_extraction(
        self,
        extraction: ExtractionResult,
        chunk: DocumentChunk,
        *,
        version: int,
    ) -> tuple[int, int]:
        entities = [entity.model_dump() for entity in extraction.entities]
        relations = [relation.model_dump() for relation in extraction.relations]
        params = {
            "chunk_id": chunk.chunk_id,
            "doc_id": chunk.doc_id,
            "content": chunk.content,
            "source": str(chunk.metadata.get("source", "")),
            "file_name": str(chunk.metadata.get("file_name", "")),
            "page": chunk.metadata.get("page") or 0,
            "sheet": chunk.metadata.get("sheet") or "",
            "modality": str(chunk.metadata.get("modality") or chunk.metadata.get("kind") or "text"),
            "asset_path": str(chunk.metadata.get("asset_path") or ""),
            "version": version,
            "now": int(time.time()),
            "entities": entities,
            "relations": relations,
        }
        async with self._get_driver().session() as session:
            # Store the chunk even when no entities were extracted.
            await session.run(
                """
                MERGE (c:Chunk {chunk_id:$chunk_id})
                SET c.doc_id=$doc_id,c.content=$content,c.source=$source,c.file_name=$file_name,
                    c.page=$page,c.sheet=$sheet,c.modality=$modality,c.asset_path=$asset_path,
                    c.version=$version,c.updated_at=$now
                """,
                params,
            )
            if entities:
                await session.run(
                    """
                    MATCH (c:Chunk {chunk_id:$chunk_id})
                    UNWIND $entities AS entity
                    MERGE (e:Entity {key:toLower(trim(entity.name))})
                    ON CREATE SET e.created_at=$now
                    SET e.name=entity.name,e.type=entity.type,
                        e.description=CASE
                          WHEN entity.description <> '' THEN entity.description
                          ELSE e.description
                        END,
                        e.updated_at=$now
                    MERGE (e)-[m:MENTIONED_IN {source_chunk_id:$chunk_id}]->(c)
                    SET m.version=$version
                    """,
                    params,
                )
            if relations:
                await session.run(
                    """
                    UNWIND $relations AS relation
                    MATCH (h:Entity {key:toLower(trim(relation.head))})
                    MATCH (t:Entity {key:toLower(trim(relation.tail))})
                    MERGE (h)-[r:RELATION {predicate:relation.relation,source_chunk_id:$chunk_id}]->(t)
                    SET r.confidence=relation.confidence,r.version=$version,r.updated_at=$now
                    """,
                    params,
                )
        return len(entities), len(relations)

    async def upsert_entity(
        self,
        entity: Entity,
        version: int = 1,
        source: str = "",
        source_chunk_id: str = "legacy",
    ) -> None:
        """Compatibility helper for callers migrating from the original service."""
        chunk = DocumentChunk(
            content="",
            doc_id=source or "legacy",
            chunk_index=0,
            doc_type=DocType.TEXT,
            metadata={"source": source, "version": version},
            identifier=source_chunk_id,
        )
        await self.upsert_extraction(
            ExtractionResult(entities=[entity], relations=[], events=[], source_chunk_id=source_chunk_id),
            chunk,
            version=version,
        )

    async def add_relation(
        self,
        relation: Relation,
        source: str = "",
        source_chunk_id: str = "legacy",
        version: int = 1,
    ) -> None:
        async with self._get_driver().session() as session:
            await session.run(
                """
                MATCH (h:Entity {key:toLower(trim($head))})
                MATCH (t:Entity {key:toLower(trim($tail))})
                MERGE (h)-[r:RELATION {predicate:$predicate,source_chunk_id:$chunk_id}]->(t)
                SET r.confidence=$confidence,r.source=$source,r.version=$version,r.updated_at=$now
                """,
                head=relation.head,
                tail=relation.tail,
                predicate=relation.relation,
                chunk_id=source_chunk_id,
                confidence=relation.confidence,
                source=source,
                version=version,
                now=int(time.time()),
            )

    async def graph_search(
        self,
        entity_names: list[str],
        *,
        hops: int = 2,
        document_ids: list[str] | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        if not entity_names:
            return []
        hops = max(1, min(int(hops), 3))
        frontier = list(dict.fromkeys(name.strip().casefold() for name in entity_names if name.strip()))
        visited = set(frontier)
        contexts: dict[str, dict[str, Any]] = {}
        frontier_limit = max(limit * 20, 200)
        for depth in range(hops + 1):
            if not frontier or len(contexts) >= limit:
                break
            records = await self.execute_cypher(
                """
                UNWIND $keys AS requested_key
                MATCH (entity:Entity {key:requested_key})-[:MENTIONED_IN]->(chunk:Chunk)
                WHERE size($doc_ids)=0 OR chunk.doc_id IN $doc_ids
                WITH chunk,collect(DISTINCT entity.name) AS entities
                RETURN chunk.chunk_id AS chunk_id,chunk.doc_id AS doc_id,
                       chunk.content AS content,chunk.source AS source,
                       chunk.file_name AS file_name,
                       CASE WHEN chunk.page=0 THEN null ELSE chunk.page END AS page,
                       CASE WHEN chunk.sheet='' THEN null ELSE chunk.sheet END AS sheet,
                       coalesce(chunk.modality,'text') AS modality,
                       entities,size(entities) AS matched_entity_count,
                       [] AS predicates,$hops AS hops
                ORDER BY matched_entity_count DESC,chunk.doc_id,chunk.chunk_id
                LIMIT $limit
                """,
                {
                    "keys": frontier,
                    "doc_ids": document_ids or [],
                    "hops": depth,
                    "limit": limit - len(contexts),
                },
            )
            for record in records:
                contexts.setdefault(str(record.get("chunk_id", "")), record)
            if depth == hops or len(contexts) >= limit:
                break
            neighbors = await self.execute_cypher(
                """
                UNWIND $keys AS requested_key
                MATCH (current:Entity {key:requested_key})-[relation:RELATION]-(neighbor:Entity)
                WHERE NOT neighbor.key IN $visited
                WITH neighbor,max(coalesce(relation.confidence,0.0)) AS confidence,
                     collect(DISTINCT relation.predicate) AS predicates
                RETURN neighbor.key AS key,neighbor.name AS name,predicates,confidence
                ORDER BY confidence DESC,key ASC
                LIMIT $limit
                """,
                {
                    "keys": frontier,
                    "visited": sorted(visited),
                    "limit": frontier_limit,
                },
            )
            frontier = [
                str(record["key"])
                for record in neighbors
                if record.get("key") and str(record["key"]) not in visited
            ]
            visited.update(frontier)
        return list(contexts.values())[:limit]

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict[str, Any]]:
        return await self.execute_cypher(
            """
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower($keyword)
               OR toLower(coalesce(e.description,'')) CONTAINS toLower($keyword)
            RETURN e.name AS name,e.type AS type,e.description AS description
            LIMIT $limit
            """,
            {"keyword": keyword, "limit": limit},
        )

    async def execute_cypher(self, cypher: str, params: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        async with self._get_driver().session() as session:
            result = await session.run(cypher, params or {})
            return await result.data()

    async def delete_chunks(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        records = await self.execute_cypher(
            """
            UNWIND $chunk_ids AS chunk_id
            OPTIONAL MATCH ()-[r:RELATION {source_chunk_id:chunk_id}]->()
            DELETE r
            WITH DISTINCT chunk_id
            OPTIONAL MATCH (c:Chunk {chunk_id:chunk_id})
            WITH c
            DETACH DELETE c
            RETURN count(c) AS deleted
            """,
            {"chunk_ids": chunk_ids},
        )
        deleted = int(records[0].get("deleted", 0)) if records else 0
        await self.execute_cypher(
            """
            MATCH (e:Entity)
            WHERE NOT (e)-[:MENTIONED_IN]->(:Chunk)
              AND NOT (e)-[:RELATION]-(:Entity)
            DELETE e
            RETURN count(e) AS deleted
            """
        )
        return deleted

    async def delete_by_doc_id(self, doc_id: str) -> int:
        records = await self.execute_cypher(
            "MATCH (c:Chunk {doc_id:$doc_id}) RETURN collect(c.chunk_id) AS ids",
            {"doc_id": doc_id},
        )
        ids = records[0].get("ids", []) if records else []
        return await self.delete_chunks(ids)

    async def delete_by_source(self, source: str) -> int:
        records = await self.execute_cypher(
            "MATCH (c:Chunk {source:$source}) RETURN collect(c.chunk_id) AS ids",
            {"source": source},
        )
        ids = records[0].get("ids", []) if records else []
        return await self.delete_chunks(ids)

    async def get_document_chunks(self, doc_id: str) -> list[str]:
        records = await self.execute_cypher(
            "MATCH (c:Chunk {doc_id:$doc_id}) RETURN c.chunk_id AS chunk_id",
            {"doc_id": doc_id},
        )
        return [str(record["chunk_id"]) for record in records]

    async def get_stats(self) -> dict[str, Any]:
        records = await self.execute_cypher(
            """
            OPTIONAL MATCH (e:Entity)
            WITH count(e) AS entities
            OPTIONAL MATCH ()-[r:RELATION]->()
            WITH entities,count(r) AS relations
            OPTIONAL MATCH (c:Chunk)
            RETURN entities,relations,count(c) AS chunks
            """
        )
        return records[0] if records else {"entities": 0, "relations": 0, "chunks": 0}
