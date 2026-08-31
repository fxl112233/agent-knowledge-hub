from __future__ import annotations

import pytest

from services.model_usage import MeteredChatModel, ensure_metered
from tests.conftest import FakeChatModel


@pytest.mark.asyncio
async def test_model_usage_accumulates_without_secrets() -> None:
    wrapped = MeteredChatModel(FakeChatModel(["ok"]))
    await wrapped.ainvoke([])
    stats = wrapped.usage_snapshot()
    assert stats["calls"] == 1
    assert stats["total_tokens"] == 13
    assert "key" not in stats
    assert ensure_metered(wrapped) is wrapped
