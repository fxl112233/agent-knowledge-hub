"""Citation-grounded temporal and claim-change comparison."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from services.graph_rag import GraphRAGContext, QueryPlan, TargetRelation


class ObservedRelation(StrEnum):
    SAME_CLAIM = "same_claim"
    CHANGED_CLAIM = "changed_claim"
    BEFORE = "before"
    AFTER = "after"
    SAME_TIME = "same_time"
    UNKNOWN = "unknown"


class TemporalEvidence(BaseModel):
    slot_id: str = Field(min_length=1, max_length=40)
    subject: str = Field(default="", max_length=500)
    claim: str = Field(default="", max_length=3000)
    date: str = Field(default="", max_length=40)
    citation_id: str = Field(min_length=1, max_length=200)


class TemporalPayload(BaseModel):
    evidence: list[TemporalEvidence] = Field(default_factory=list, max_length=6)
    observed_relation: ObservedRelation = ObservedRelation.UNKNOWN


@dataclass(slots=True)
class TemporalResult:
    answer: str
    answerable: bool
    citation_ids: list[str]
    observed_relation: ObservedRelation
    usage: dict[str, int] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


TEMPORAL_PROMPT = """你是企业文档证据比较器。只能使用给定证据，并只返回 JSON。
把问题中要比较的左右来源或时间点分别映射到 evidence slot。每条事实必须引用真实 chunk ID。

输出格式：
{
  "evidence":[
    {"slot_id":"left","subject":"主体","claim":"核心主张",
     "date":"YYYY-MM-DD 或空字符串","citation_id":"chunk-id"}
  ],
  "observed_relation":"same_claim|changed_claim|before|after|same_time|unknown"
}

规则：
1. same_claim 表示两侧核心主张语义一致；changed_claim 表示两侧核心主张发生实质变化。
2. before/after/same_time 比较第一个必需槽位相对第二个必需槽位的日期。
3. 证据缺少一侧、日期不明确、来源互相冲突或无法可靠判断时必须输出 unknown。
4. 不要回答 Yes/No；不要输出解释；不要编造日期、事实或 chunk ID。
"""


class TemporalComparisonTool:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def compare(
        self,
        question: str,
        plan: QueryPlan,
        contexts: list[GraphRAGContext],
    ) -> TemporalResult:
        if self.llm is None:
            raise RuntimeError("temporal comparison requires an LLM")
        valid_ids = {context.chunk_id for context in contexts}
        required_slots = [slot for slot in plan.evidence_slots if slot.required]
        if len(required_slots) < 2:
            return self._abstain("temporal:missing_required_slots")
        response = await self.llm.ainvoke(
            [
                {"role": "system", "content": TEMPORAL_PROMPT},
                {
                    "role": "user",
                    "content": self._request_content(question, plan, contexts),
                },
            ]
        )
        try:
            payload = TemporalPayload.model_validate(self._json_object(self._response_text(response)))
        except (ValueError, json.JSONDecodeError, ValidationError):
            raise ValueError("invalid temporal tool response") from None

        evidence = [item for item in payload.evidence if item.citation_id in valid_ids]
        if len(evidence) != len(payload.evidence):
            return self._abstain("temporal:invalid_citation")
        by_slot: dict[str, list[TemporalEvidence]] = {}
        for item in evidence:
            by_slot.setdefault(item.slot_id, []).append(item)
        if any(slot.slot_id not in by_slot for slot in required_slots):
            return self._abstain("temporal:missing_slot_evidence")

        observed = self._determine_relation(plan, required_slots, by_slot, payload.observed_relation)
        if observed == ObservedRelation.UNKNOWN:
            return self._abstain("temporal:unknown")
        target = self._target_relation(plan.target_relation)
        if target is None:
            return self._abstain("temporal:missing_target_relation")
        citation_ids = list(
            dict.fromkeys(
                item.citation_id for slot in required_slots for item in by_slot.get(slot.slot_id, [])
            )
        )
        return TemporalResult(
            answer="Yes" if observed == target else "No",
            answerable=True,
            citation_ids=citation_ids,
            observed_relation=observed,
            usage=self._usage(response),
            trace=[f"temporal:observed:{observed.value}", f"temporal:target:{target.value}"],
        )

    @classmethod
    def _determine_relation(
        cls,
        plan: QueryPlan,
        required_slots: list[Any],
        by_slot: dict[str, list[TemporalEvidence]],
        model_relation: ObservedRelation,
    ) -> ObservedRelation:
        if plan.target_relation in {
            TargetRelation.BEFORE,
            TargetRelation.AFTER,
            TargetRelation.SAME_TIME,
        }:
            left = cls._single_date(by_slot[required_slots[0].slot_id])
            right = cls._single_date(by_slot[required_slots[1].slot_id])
            if left is None or right is None:
                return ObservedRelation.UNKNOWN
            if left < right:
                return ObservedRelation.BEFORE
            if left > right:
                return ObservedRelation.AFTER
            return ObservedRelation.SAME_TIME
        if model_relation in {ObservedRelation.SAME_CLAIM, ObservedRelation.CHANGED_CLAIM}:
            return model_relation
        return ObservedRelation.UNKNOWN

    @classmethod
    def _single_date(cls, evidence: list[TemporalEvidence]) -> date | None:
        values = {value for item in evidence if (value := cls._parse_date(item.date)) is not None}
        return next(iter(values)) if len(values) == 1 else None

    @staticmethod
    def _parse_date(raw: str) -> date | None:
        value = raw.strip()
        if re.fullmatch(r"\d{4}", value):
            return date(int(value), 1, 1)
        match = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/](\d{1,2}))?", value)
        if not match:
            return None
        try:
            return date(int(match.group(1)), int(match.group(2)), int(match.group(3) or 1))
        except ValueError:
            return None

    @staticmethod
    def _target_relation(value: TargetRelation) -> ObservedRelation | None:
        try:
            return ObservedRelation(value.value)
        except ValueError:
            return None

    @staticmethod
    def _abstain(trace: str) -> TemporalResult:
        return TemporalResult(
            answer="无法从文档确定",
            answerable=False,
            citation_ids=[],
            observed_relation=ObservedRelation.UNKNOWN,
            trace=[trace],
        )

    @staticmethod
    def _request_content(
        question: str,
        plan: QueryPlan,
        contexts: list[GraphRAGContext],
    ) -> str:
        slots = "\n".join(
            f"- {slot.slot_id}: role={slot.role}; required={slot.required}; query={slot.query}"
            for slot in plan.evidence_slots
        )
        evidence = "\n\n".join(
            f"[{context.chunk_id}] file={context.metadata.get('file_name') or context.source} "
            f"page={context.metadata.get('page', '')}\n{context.content}"
            for context in contexts
        )
        return (
            f"Question: {question}\nTarget relation: {plan.target_relation.value}\n"
            f"Evidence slots:\n{slots}\n\nEvidence:\n{evidence}"
        )

    @staticmethod
    def _response_text(response: Any) -> str:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            return "".join(
                str(item.get("text", "")) if isinstance(item, dict) else str(item) for item in content
            )
        return str(content)

    @staticmethod
    def _json_object(raw: str) -> dict[str, Any]:
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError("missing JSON object")
        value = json.loads(cleaned[start : end + 1])
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value

    @staticmethod
    def _usage(response: Any) -> dict[str, int]:
        usage = getattr(response, "usage_metadata", None) or {}
        if not usage:
            usage = (getattr(response, "response_metadata", None) or {}).get("token_usage", {})
        return {
            "input_tokens": int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0),
            "output_tokens": int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        }
