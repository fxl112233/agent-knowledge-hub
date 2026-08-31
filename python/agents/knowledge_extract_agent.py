"""Structured entity and relationship extraction with chunk-level provenance."""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from agents.doc_parser_agent import DocumentChunk
from config import settings
from services.model_usage import ensure_metered


class ModelNotConfiguredError(RuntimeError):
    pass


class Entity(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    type: str = Field(default="Concept", max_length=80)
    description: str = Field(default="", max_length=1000)

    @field_validator("name", "type", "description", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()


class Relation(BaseModel):
    head: str = Field(min_length=1, max_length=200)
    relation: str = Field(default="related_to", min_length=1, max_length=100)
    tail: str = Field(min_length=1, max_length=200)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_chunk_id: str = ""

    @field_validator("head", "relation", "tail", mode="before")
    @classmethod
    def clean_text(cls, value: Any) -> str:
        return re.sub(r"\s+", " ", str(value or "")).strip()


class KnowledgeEvent(BaseModel):
    trigger: str = ""
    type: str = ""
    participants: list[str] = Field(default_factory=list)
    source_chunk_id: str = ""


class ExtractionResult(BaseModel):
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    events: list[KnowledgeEvent] = Field(default_factory=list)
    source_chunk_id: str = ""


EXTRACTION_PROMPT = """你是企业知识抽取器。根据给定文本提取实体、关系和事件。
只返回一个 JSON 对象，格式如下：
{
  "entities": [
    {"name":"名称","type":"Person|Organization|Location|Product|Technology|Concept|Event|Time",
     "description":"描述"}
  ],
  "relations": [{"head":"实体名","relation":"简短关系谓词","tail":"实体名","confidence":0.0}],
  "events": [{"trigger":"触发词","type":"事件类型","participants":["实体名"]}]
}
要求：关系两端必须出现在 entities 中；不要猜测文本未提供的事实；没有结果时返回空数组。"""


class KnowledgeExtractAgent:
    def __init__(self, llm: Any | None = None, concurrency: int | None = None) -> None:
        self.llm = ensure_metered(llm or self._build_llm())
        concurrency = concurrency or settings.knowledge_extraction_concurrency
        self._semaphore = asyncio.Semaphore(max(1, concurrency))

    @staticmethod
    def _build_llm() -> Any | None:
        if not settings.llm_configured:
            return None
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0,
            timeout=settings.llm_timeout_seconds,
            max_retries=settings.llm_max_retries,
        )

    async def extract(self, chunks: list[DocumentChunk]) -> list[ExtractionResult]:
        if not chunks:
            return []
        return await asyncio.gather(*(self._extract_limited(chunk) for chunk in chunks))

    async def _extract_limited(self, chunk: DocumentChunk) -> ExtractionResult:
        async with self._semaphore:
            return await self.extract_single(chunk.content, chunk.chunk_id)

    async def extract_single(self, text: str, chunk_id: str = "") -> ExtractionResult:
        if self.llm is None:
            raise ModelNotConfiguredError("LLM_API_KEY and LLM_MODEL are required for knowledge extraction")
        last_error = ""
        for attempt in range(settings.llm_max_retries + 1):
            suffix = f"\n上次输出无效：{last_error}。请只输出合法 JSON。" if last_error else ""
            response = await self.llm.ainvoke(
                [
                    {"role": "system", "content": EXTRACTION_PROMPT},
                    {"role": "user", "content": f"来源块：{chunk_id}\n文本：\n{text}{suffix}"},
                ]
            )
            try:
                raw = self._response_text(response)
                result = self._parse_response(raw, chunk_id)
                return self._deduplicate(result)
            except (json.JSONDecodeError, ValidationError, ValueError) as exc:
                last_error = str(exc)[:300]
                if attempt >= settings.llm_max_retries:
                    break
        raise ValueError(f"knowledge extraction failed for {chunk_id}: {last_error}")

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
            )
        return str(content)

    @staticmethod
    def _parse_response(raw: str, chunk_id: str) -> ExtractionResult:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.IGNORECASE)
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("response does not contain a JSON object")
        data = json.loads(cleaned[start : end + 1])
        result = ExtractionResult.model_validate(data)
        result.source_chunk_id = chunk_id
        for relation in result.relations:
            relation.source_chunk_id = chunk_id
        for event in result.events:
            event.source_chunk_id = chunk_id
        return result

    @staticmethod
    def _deduplicate(result: ExtractionResult) -> ExtractionResult:
        entity_map: dict[tuple[str, str], Entity] = {}
        for entity in result.entities:
            entity_map.setdefault((entity.name.casefold(), entity.type.casefold()), entity)
        valid_names = {entity.name.casefold() for entity in entity_map.values()}
        relation_map: dict[tuple[str, str, str], Relation] = {}
        for relation in result.relations:
            if relation.head.casefold() not in valid_names or relation.tail.casefold() not in valid_names:
                continue
            key = (relation.head.casefold(), relation.relation.casefold(), relation.tail.casefold())
            current = relation_map.get(key)
            if current is None or relation.confidence > current.confidence:
                relation_map[key] = relation
        return ExtractionResult(
            entities=list(entity_map.values()),
            relations=list(relation_map.values()),
            events=result.events,
            source_chunk_id=result.source_chunk_id,
        )
