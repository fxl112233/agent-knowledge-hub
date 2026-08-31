"""SiliconFlow vision description client with a small, explicit response contract."""

from __future__ import annotations

import base64
import io
from typing import Any

import httpx

from config import settings


class SiliconFlowVisionClient:
    def __init__(self) -> None:
        if not settings.vision_configured:
            raise RuntimeError("SiliconFlow vision is not configured")

    async def describe(self, image: Any) -> str:
        buffer = io.BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=85, optimize=True)
        encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
        payload = {
            "model": settings.active_vision_model,
            "temperature": 0,
            "max_tokens": 512,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "请客观描述文档图片中的文字、表格、图表、数值及其关系。不要猜测看不清的内容。"
                            ),
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{encoded}",
                                "detail": "low",
                            },
                        },
                    ],
                }
            ],
        }
        headers = {"Authorization": f"Bearer {settings.siliconflow_api_key}"}
        timeout = httpx.Timeout(settings.vision_timeout_seconds)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(
                f"{settings.siliconflow_base_url.rstrip('/')}/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
            data = response.json()
        return str(data["choices"][0]["message"]["content"]).strip()


def build_vision_client() -> SiliconFlowVisionClient | None:
    return SiliconFlowVisionClient() if settings.vision_configured else None
