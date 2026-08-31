"""Process-local, secret-free accounting for chat model calls and token usage."""

from __future__ import annotations

import asyncio
import time
from typing import Any


class MeteredChatModel:
    def __init__(self, model: Any) -> None:
        self.model = model
        self._lock = asyncio.Lock()
        self._totals = {
            "calls": 0,
            "errors": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
        }

    async def ainvoke(self, *args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            response = await self.model.ainvoke(*args, **kwargs)
        except Exception:
            async with self._lock:
                self._totals["calls"] += 1
                self._totals["errors"] += 1
                self._totals["latency_ms"] += (time.perf_counter() - started) * 1000
            raise
        usage = getattr(response, "usage_metadata", None) or {}
        if not usage:
            usage = (getattr(response, "response_metadata", None) or {}).get("token_usage", {})
        input_tokens = int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
        output_tokens = int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0)
        total_tokens = int(usage.get("total_tokens", input_tokens + output_tokens) or 0)
        async with self._lock:
            self._totals["calls"] += 1
            self._totals["input_tokens"] += input_tokens
            self._totals["output_tokens"] += output_tokens
            self._totals["total_tokens"] += total_tokens
            self._totals["latency_ms"] += (time.perf_counter() - started) * 1000
        return response

    def usage_snapshot(self) -> dict[str, int | float]:
        return {
            key: round(value, 3) if isinstance(value, float) else value for key, value in self._totals.items()
        }


def ensure_metered(model: Any | None) -> MeteredChatModel | None:
    if model is None or isinstance(model, MeteredChatModel):
        return model
    return MeteredChatModel(model)
