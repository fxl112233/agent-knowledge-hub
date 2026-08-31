"""Grounded verification for multi-source boolean comparison questions."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from services.graph_rag import GraphRAGContext, QueryPlan


class ComparisonVerdict(StrEnum):
    YES = "yes"
    NO = "no"
    UNKNOWN = "unknown"


class ComparisonEvidence(BaseModel):
    slot_id: str = Field(min_length=1, max_length=40)
    claim: str = Field(default="", max_length=3000)
    supported: bool | None = None
    citation_id: str = Field(min_length=1, max_length=200)


class ComparisonPayload(BaseModel):
    evidence: list[ComparisonEvidence] = Field(default_factory=list, max_length=8)
    verdict: ComparisonVerdict = ComparisonVerdict.UNKNOWN


@dataclass(slots=True)
class ComparisonResult:
    answer: str
    answerable: bool
    citation_ids: list[str]
    verdict: ComparisonVerdict
    usage: dict[str, int] = field(default_factory=dict)
    trace: list[str] = field(default_factory=list)


COMPARISON_PROMPT = """You verify a multi-source enterprise-document question using only the
supplied evidence.
Return JSON only:
{
  "evidence": [
    {"slot_id":"left", "claim":"fact found in this source", "supported":true,
     "citation_id":"real-chunk-id"}
  ],
  "verdict":"yes|no|unknown"
}

Rules:
1. Map every required evidence slot to a real chunk ID. `supported` means that the cited text
   supports the exact proposition requested by that slot, including words such as not/never.
2. Judge the complete original question, not merely whether the two extracted claims sound similar.
   Questions of the form "Does A ..., while B ...?" ask whether the combined proposition is true.
3. Handle same/different, agreement/disagreement, change/no-change, and double negation literally.
4. Return yes only when all facts needed by the complete proposition are supported. Return no when
   the evidence contradicts the proposition. Return unknown if either side is missing or ambiguous.
5. Never use model memory and never invent a fact, date, source, or chunk ID.
"""


class ComparisonVerificationTool:
    def __init__(self, llm: Any) -> None:
        self.llm = llm

    async def compare(
        self,
        question: str,
        plan: QueryPlan,
        contexts: list[GraphRAGContext],
    ) -> ComparisonResult:
        if self.llm is None:
            raise RuntimeError("comparison verification requires an LLM")
        required_slots = [slot for slot in plan.evidence_slots if slot.required]
        if len(required_slots) < 2:
            return self._abstain("comparison:missing_required_slots")
        valid_ids = {context.chunk_id for context in contexts}
        response = await self.llm.ainvoke(
            [
                {"role": "system", "content": COMPARISON_PROMPT},
                {"role": "user", "content": self._request_content(question, plan, contexts)},
            ]
        )
        try:
            payload = ComparisonPayload.model_validate(self._json_object(self._response_text(response)))
        except (ValueError, json.JSONDecodeError, ValidationError):
            raise ValueError("invalid comparison tool response") from None

        if any(item.citation_id not in valid_ids for item in payload.evidence):
            return self._abstain("comparison:invalid_citation")
        by_slot: dict[str, list[ComparisonEvidence]] = {}
        for item in payload.evidence:
            by_slot.setdefault(item.slot_id, []).append(item)
        if any(slot.slot_id not in by_slot for slot in required_slots):
            return self._abstain("comparison:missing_slot_evidence")
        required_evidence = [item for slot in required_slots for item in by_slot[slot.slot_id]]
        if any(item.supported is None for item in required_evidence):
            return self._abstain("comparison:unknown_slot")
        if payload.verdict == ComparisonVerdict.UNKNOWN:
            return self._abstain("comparison:unknown")
        if payload.verdict == ComparisonVerdict.YES and not all(item.supported for item in required_evidence):
            return self._abstain("comparison:inconsistent_verdict")

        citation_ids = list(dict.fromkeys(item.citation_id for item in required_evidence))
        return ComparisonResult(
            answer="Yes" if payload.verdict == ComparisonVerdict.YES else "No",
            answerable=True,
            citation_ids=citation_ids,
            verdict=payload.verdict,
            usage=self._usage(response),
            trace=[f"comparison:verdict:{payload.verdict.value}"],
        )

    @staticmethod
    def _abstain(trace: str) -> ComparisonResult:
        return ComparisonResult(
            answer="无法从文档确定",
            answerable=False,
            citation_ids=[],
            verdict=ComparisonVerdict.UNKNOWN,
            trace=[trace],
        )

    @staticmethod
    def _request_content(
        question: str,
        plan: QueryPlan,
        contexts: list[GraphRAGContext],
    ) -> str:
        slots = "\n".join(
            f"- {slot.slot_id}: role={slot.role}; required={slot.required}; proposition={slot.query}"
            for slot in plan.evidence_slots
        )
        evidence = "\n\n".join(
            f"[{context.chunk_id}] file={context.metadata.get('file_name') or context.source} "
            f"page={context.metadata.get('page', '')}\n{context.content}"
            for context in contexts
        )
        return f"Original question: {question}\nEvidence slots:\n{slots}\n\nEvidence:\n{evidence}"

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
