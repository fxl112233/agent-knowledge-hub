"""SiliconFlow reranking with bounded retries and secret-free accounting."""

from __future__ import annotations

import asyncio
import math
import time
from dataclasses import dataclass
from typing import Any

import httpx

from config import settings


@dataclass(frozen=True, slots=True)
class RerankScore:
    index: int
    score: float


class SiliconFlowReranker:
    """Thin client for the OpenAI-style SiliconFlow rerank endpoint."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        if not settings.reranker_configured:
            raise RuntimeError("SiliconFlow reranker is not configured")
        self.model = settings.rerank_model
        self._client = client
        self._owns_client = client is None
        self._lock = asyncio.Lock()
        self._calls = 0
        self._errors = 0
        self._retries = 0
        self._fallbacks = 0
        self._documents = 0
        self._input_tokens = 0
        self._latencies_ms: list[float] = []

    async def close(self) -> None:
        if self._client is not None and self._owns_client:
            await self._client.aclose()
        self._client = None

    async def rerank(
        self,
        query: str,
        documents: list[str],
        *,
        top_n: int | None = None,
    ) -> list[RerankScore]:
        if not documents:
            return []
        top_n = max(1, min(top_n or len(documents), len(documents)))
        payload = {
            "model": self.model,
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": False,
        }
        started = time.perf_counter()
        async with self._lock:
            self._calls += 1
            self._documents += len(documents)
        client = self._client
        if client is None:
            client = httpx.AsyncClient(timeout=httpx.Timeout(settings.rerank_timeout_seconds))
            self._client = client
        data: dict[str, Any] | None = None
        for attempt in range(settings.rerank_max_retries + 1):
            try:
                response = await client.post(
                    f"{settings.siliconflow_base_url.rstrip('/')}/rerank",
                    headers={"Authorization": f"Bearer {settings.siliconflow_api_key}"},
                    json=payload,
                )
                response.raise_for_status()
                value = response.json()
                if not isinstance(value, dict):
                    raise ValueError("rerank response must be an object")
                data = value
                break
            except httpx.HTTPStatusError as exc:
                retryable = exc.response.status_code in {408, 429} or exc.response.status_code >= 500
                if not retryable or attempt >= settings.rerank_max_retries:
                    await self._record_error(started)
                    raise
            except (httpx.TransportError, ValueError):
                if attempt >= settings.rerank_max_retries:
                    await self._record_error(started)
                    raise
            async with self._lock:
                self._retries += 1
            await asyncio.sleep(2**attempt)
        if data is None:
            await self._record_error(started)
            raise RuntimeError("rerank request did not produce a response")
        try:
            scores = self._parse_scores(data, len(documents), top_n)
        except (KeyError, TypeError, ValueError):
            await self._record_error(started)
            raise
        usage = data.get("meta") or data.get("usage") or {}
        tokens = usage.get("tokens") if isinstance(usage, dict) else {}
        if not isinstance(tokens, dict):
            tokens = {}
        input_tokens = int(
            tokens.get("input_tokens") or usage.get("input_tokens", 0) or usage.get("prompt_tokens", 0) or 0
        )
        async with self._lock:
            self._input_tokens += input_tokens
            self._latencies_ms.append((time.perf_counter() - started) * 1000)
        return scores

    async def record_fallback(self) -> None:
        async with self._lock:
            self._fallbacks += 1

    async def _record_error(self, started: float) -> None:
        async with self._lock:
            self._errors += 1
            self._latencies_ms.append((time.perf_counter() - started) * 1000)

    @staticmethod
    def _parse_scores(data: dict[str, Any], document_count: int, top_n: int) -> list[RerankScore]:
        results = data.get("results")
        if not isinstance(results, list):
            raise ValueError("rerank response is missing results")
        output: list[RerankScore] = []
        seen: set[int] = set()
        for item in results:
            if not isinstance(item, dict):
                raise ValueError("rerank result must be an object")
            index = int(item["index"])
            score = float(item.get("relevance_score", item.get("score")))
            if index < 0 or index >= document_count or index in seen or not math.isfinite(score):
                raise ValueError("rerank result contains an invalid index or score")
            seen.add(index)
            output.append(RerankScore(index, max(0.0, min(1.0, score))))
        if len(output) < top_n:
            raise ValueError("rerank response returned fewer results than requested")
        return sorted(output, key=lambda item: (-item.score, item.index))[:top_n]

    def usage_snapshot(self) -> dict[str, Any]:
        values = sorted(self._latencies_ms)

        def percentile(quantile: float) -> float:
            if not values:
                return 0.0
            position = (len(values) - 1) * quantile
            lower, upper = math.floor(position), math.ceil(position)
            if lower == upper:
                return values[lower]
            return values[lower] * (upper - position) + values[upper] * (position - lower)

        return {
            "model": self.model,
            "calls": self._calls,
            "errors": self._errors,
            "retries": self._retries,
            "fallbacks": self._fallbacks,
            "documents": self._documents,
            "input_tokens": self._input_tokens,
            "p50_ms": round(percentile(0.5), 3),
            "p95_ms": round(percentile(0.95), 3),
        }


def build_reranker() -> SiliconFlowReranker | None:
    if not settings.rerank_enabled or not settings.reranker_configured:
        return None
    return SiliconFlowReranker()
