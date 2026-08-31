from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import pytest

from services.vector_store import SiliconFlowVLEmbeddingProvider
from services.vision import SiliconFlowVisionClient


class FakeHTTPResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeHTTPClient:
    calls: list[dict[str, Any]] = []

    def __init__(self, **_kwargs: Any) -> None:
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args: Any) -> None:
        return None

    async def post(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
        self.calls.append({"url": url, **kwargs})
        if url.endswith("/chat/completions"):
            return FakeHTTPResponse({"choices": [{"message": {"content": "a bar chart"}}]})
        count = len(kwargs["json"]["input"])
        return FakeHTTPResponse(
            {
                "data": [{"index": index, "embedding": [float(index + 1)] * 8} for index in range(count)],
                "usage": {"prompt_tokens": 9},
            }
        )


class FlakyEmbeddingHTTPClient(FakeHTTPClient):
    attempts = 0

    async def post(self, url: str, **kwargs: Any) -> FakeHTTPResponse:
        self.__class__.attempts += 1
        if self.__class__.attempts < 3:
            raise httpx.ReadTimeout("temporary SiliconFlow timeout")
        return await super().post(url, **kwargs)


@pytest.mark.asyncio
async def test_vision_and_vl_clients_use_low_detail_and_raw_base64(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    FakeHTTPClient.calls.clear()
    monkeypatch.setattr("services.vision.httpx.AsyncClient", FakeHTTPClient)
    monkeypatch.setattr("services.vector_store.httpx.AsyncClient", FakeHTTPClient)
    monkeypatch.setattr("services.vision.settings.siliconflow_api_key", "secret-test-key")
    monkeypatch.setattr("services.vector_store.settings.siliconflow_api_key", "secret-test-key")
    monkeypatch.setattr("services.vector_store.settings.embedding_dimensions", 8)

    image = Image.new("RGB", (32, 32), "blue")
    image_path = tmp_path / "image.png"
    image.save(image_path)
    assert await SiliconFlowVisionClient().describe(image) == "a bar chart"
    provider = SiliconFlowVLEmbeddingProvider()
    image_vectors = await provider.aembed_images([str(image_path)])
    query_vector = await provider.aembed_query("a blue square")

    vision_payload = FakeHTTPClient.calls[0]["json"]
    image_part = vision_payload["messages"][0]["content"][1]
    assert image_part["image_url"]["detail"] == "low"
    assert image_part["image_url"]["url"].startswith("data:image/jpeg;base64,")
    embedding_payload = FakeHTTPClient.calls[1]["json"]
    assert embedding_payload["input"][0]["image"].startswith("data:image/png;base64,")
    assert len(image_vectors[0]) == len(query_vector) == 8
    assert provider.usage_snapshot() == {"calls": 2, "input_tokens": 18, "inputs": 2}
    assert all("secret-test-key" not in str(call["json"]) for call in FakeHTTPClient.calls)


@pytest.mark.asyncio
async def test_vl_embedding_retries_transient_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def no_sleep(_seconds: float) -> None:
        return None

    FakeHTTPClient.calls.clear()
    FlakyEmbeddingHTTPClient.attempts = 0
    monkeypatch.setattr("services.vector_store.httpx.AsyncClient", FlakyEmbeddingHTTPClient)
    monkeypatch.setattr("services.vector_store.asyncio.sleep", no_sleep)
    monkeypatch.setattr("services.vector_store.settings.siliconflow_api_key", "secret-test-key")
    monkeypatch.setattr("services.vector_store.settings.embedding_dimensions", 8)
    monkeypatch.setattr("services.vector_store.settings.llm_max_retries", 2)

    vector = await SiliconFlowVLEmbeddingProvider().aembed_query("retry me")

    assert len(vector) == 8
    assert FlakyEmbeddingHTTPClient.attempts == 3
