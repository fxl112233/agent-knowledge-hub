from __future__ import annotations

import json

import httpx
import pytest

from config import settings
from services.reranker import SiliconFlowReranker


@pytest.mark.asyncio
async def test_reranker_orders_scores_and_records_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "siliconflow_api_key", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["documents"] == ["low", "high", "middle"]
        assert "test-key" not in request.content.decode()
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 1, "relevance_score": 0.91},
                    {"index": 2, "relevance_score": 0.52},
                    {"index": 0, "relevance_score": 0.11},
                ],
                "usage": {"input_tokens": 17},
            },
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = SiliconFlowReranker(client)
    result = await reranker.rerank("query", ["low", "high", "middle"])
    assert [item.index for item in result] == [1, 2, 0]
    stats = reranker.usage_snapshot()
    assert stats["calls"] == 1
    assert stats["errors"] == 0
    assert stats["documents"] == 3
    assert stats["input_tokens"] == 17
    await client.aclose()


@pytest.mark.asyncio
async def test_reranker_retries_429_and_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "siliconflow_api_key", "test-key")
    statuses = [429, 503, 200]

    def handler(_request: httpx.Request) -> httpx.Response:
        status = statuses.pop(0)
        if status != 200:
            return httpx.Response(status, json={"message": "retry"})
        return httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]})

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("services.reranker.asyncio.sleep", no_sleep)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = SiliconFlowReranker(client)
    result = await reranker.rerank("q", ["d"])
    assert result[0].score == 0.8
    assert reranker.usage_snapshot()["retries"] == 2
    await client.aclose()


@pytest.mark.asyncio
async def test_reranker_rejects_invalid_indices(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "siliconflow_api_key", "test-key")

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"results": [{"index": 8, "relevance_score": 0.8}]})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = SiliconFlowReranker(client)
    with pytest.raises(ValueError, match="invalid index"):
        await reranker.rerank("q", ["d"])
    assert reranker.usage_snapshot()["errors"] == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_reranker_retries_transport_errors_then_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(settings, "siliconflow_api_key", "test-key")

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    async def no_sleep(_seconds: float) -> None:
        return None

    monkeypatch.setattr("services.reranker.asyncio.sleep", no_sleep)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    reranker = SiliconFlowReranker(client)
    with pytest.raises(httpx.ReadTimeout):
        await reranker.rerank("q", ["d"])
    stats = reranker.usage_snapshot()
    assert stats["retries"] == settings.rerank_max_retries
    assert stats["errors"] == 1
    await client.aclose()
