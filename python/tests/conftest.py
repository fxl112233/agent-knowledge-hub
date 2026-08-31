from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import pytest


@dataclass
class FakeResponse:
    content: str
    usage_metadata: dict[str, int] | None = None


class FakeChatModel:
    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = list(responses or [])
        self.calls: list[Any] = []

    async def ainvoke(self, messages: Any, **_: Any) -> FakeResponse:
        self.calls.append(messages)
        content = (
            self.responses.pop(0)
            if self.responses
            else json.dumps({"answer": "ok", "answerable": True, "citation_ids": []})
        )
        return FakeResponse(content, {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13})


class FakeEmbedding:
    def __init__(self, dimensions: int = 8) -> None:
        self.dimensions = dimensions
        self.document_batches: list[list[str]] = []
        self.queries: list[str] = []

    def _vector(self, text: str) -> list[float]:
        seed = sum(text.encode("utf-8"))
        return [((seed + index * 17) % 101) / 100 for index in range(self.dimensions)]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_batches.append(texts)
        return [self._vector(text) for text in texts]

    async def aembed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        return self._vector(text)


class FakeVLEmbedding(FakeEmbedding):
    def __init__(self, dimensions: int = 8) -> None:
        super().__init__(dimensions)
        self.images: list[str] = []

    async def aembed_images(self, paths: list[str]) -> list[list[float]]:
        self.images.extend(paths)
        return [self._vector(path) for path in paths]

    def usage_snapshot(self) -> dict[str, int]:
        return {
            "calls": len(self.document_batches) + len(self.queries) + len(self.images),
            "input_tokens": 0,
            "inputs": sum(map(len, self.document_batches)) + len(self.queries) + len(self.images),
        }


@pytest.fixture
def fake_llm() -> FakeChatModel:
    return FakeChatModel()
