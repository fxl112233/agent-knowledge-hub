"""Estimate live model calls and embedding cost before a benchmark run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from benchmarks.data import PREPARED_ROOT
from config import settings


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _tokens(text: str) -> int:
    try:
        import tiktoken

        return len(tiktoken.get_encoding("cl100k_base").encode(text))
    except Exception:
        return max(1, math.ceil(len(text) / 3))


def _document_tokens(root: Path) -> int:
    records = _jsonl(root / "documents.jsonl")
    total = 0
    for record in records:
        path = root / str(record.get("path") or record.get("initial") or "")
        if path.suffix.lower() in {".txt", ".md", ".csv"} and path.exists():
            total += _tokens(path.read_text(encoding="utf-8", errors="ignore"))
        elif path.exists():
            total += max(1, math.ceil(path.stat().st_size / 12))
    return total


def estimate_run() -> dict[str, Any]:
    multihop_retrieval = _jsonl(PREPARED_ROOT / "multihop_rag" / "retrieval.jsonl")
    multihop_answers = _jsonl(PREPARED_ROOT / "multihop_rag" / "answer-300.jsonl")
    tat_answers = _jsonl(PREPARED_ROOT / "tat_qa" / "answer-300.jsonl")
    rgb_answers = _jsonl(PREPARED_ROOT / "rgb" / "answer-200.jsonl")
    prepared = bool(multihop_retrieval and multihop_answers and tat_answers and rgb_answers)
    retrieval_cases = len(multihop_retrieval) if prepared else 1358
    answer_cases = (
        len(multihop_answers) * 2 + len(tat_answers) + len(rgb_answers) if prepared else 300 * 2 + 300 + 200
    )
    document_tokens = sum(
        _document_tokens(PREPARED_ROOT / name) for name in ("multihop_rag", "tat_qa", "rgb")
    )
    if not document_tokens:
        document_tokens = 900_000
    effective_step = max(1, settings.chunk_size_tokens - settings.chunk_overlap_tokens)
    estimated_chunks = math.ceil(document_tokens / effective_step)
    question_tokens = sum(
        _tokens(str(record.get("question", "")))
        for record in [*multihop_retrieval, *multihop_answers, *tat_answers, *rgb_answers]
    )
    if not question_tokens:
        question_tokens = retrieval_cases * 30 + answer_cases * 35
    embedding_query_calls = retrieval_cases * 2 + answer_cases
    embedding_calls = math.ceil(estimated_chunks / settings.embedding_batch_size) + embedding_query_calls
    embedding_tokens = document_tokens + question_tokens * 2
    extraction_calls = estimated_chunks
    query_planning_calls = retrieval_cases + answer_cases
    answer_generation_calls = answer_cases
    llm_calls = extraction_calls + query_planning_calls + answer_generation_calls
    return {
        "status": "estimate_only",
        "prepared_data_available": prepared,
        "models": {
            "llm_model": settings.llm_model or "<not configured>",
            "embedding_model": settings.active_embedding_model,
        },
        "samples": {
            "retrieval": retrieval_cases,
            "answers": answer_cases,
            "estimated_chunks": estimated_chunks,
        },
        "estimated_calls": {
            "llm_total": llm_calls,
            "knowledge_extraction": extraction_calls,
            "query_planning": query_planning_calls,
            "answer_generation": answer_generation_calls,
            "embedding": embedding_calls,
        },
        "estimated_tokens": {
            "embedding_input": embedding_tokens,
            "llm_input_lower_bound": document_tokens + question_tokens,
            "llm_output_not_priced": True,
        },
        "estimated_embedding_cost_cny": round(
            embedding_tokens / 1000 * settings.embedding_price_per_1k_cny, 4
        ),
        "warning": "LLM cost is not estimated because no trusted runtime price was configured.",
    }
