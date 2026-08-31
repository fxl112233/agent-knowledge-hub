from __future__ import annotations

import os

import httpx
import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.live_api]


@pytest.mark.asyncio
async def test_live_api_upload_ask_get_and_delete() -> None:
    base_url = os.getenv("AKH_E2E_BASE_URL")
    if not base_url:
        pytest.skip("set AKH_E2E_BASE_URL to run against a live Compose API")
    async with httpx.AsyncClient(base_url=base_url, timeout=180) as client:
        ready = await client.get("/api/health/ready")
        assert ready.status_code == 200, ready.text

        uploaded = await client.post(
            "/api/ingest/upload",
            files={
                "file": (
                    "e2e-knowledge.txt",
                    "北辰计划的负责人是林海，预算为120万元。".encode(),
                    "text/plain",
                )
            },
        )
        assert uploaded.status_code == 200, uploaded.text
        doc_id = uploaded.json()["doc_id"]
        try:
            document = await client.get(f"/api/documents/{doc_id}")
            assert document.status_code == 200
            assert document.json()["chunks"] >= 1

            answered = await client.post(
                "/api/qa/ask",
                json={
                    "question": "北辰计划的负责人是谁？",
                    "top_k": 5,
                    "mode": "vector",
                    "document_ids": [doc_id],
                },
            )
            assert answered.status_code == 200, answered.text
            payload = answered.json()
            assert payload["answerable"] is True
            assert payload["citations"]
            assert {item["doc_id"] for item in payload["citations"]} == {doc_id}
            assert payload["total_ms"] >= payload["retrieval_ms"]
            assert payload["trace"]
        finally:
            deleted = await client.delete(f"/api/documents/{doc_id}")
            assert deleted.status_code == 200, deleted.text
            assert deleted.json()["deleted"] is True
