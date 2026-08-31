"""Operational commands for consistency repair and full document rebuilds."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import tempfile
import time
from pathlib import Path
from typing import Any

from agents.doc_parser_agent import DocParserAgent, DocType, DocumentChunk
from agents.knowledge_extract_agent import KnowledgeExtractAgent
from config import settings
from services.catalog import CatalogService
from services.ingestion import IngestionService
from services.knowledge_graph import KnowledgeGraphService
from services.vector_store import SiliconFlowVLEmbeddingProvider, VectorStoreService
from services.vision import build_vision_client


async def _preflight() -> dict[str, Any]:
    """Validate both independent model providers without exposing credentials or URLs."""
    if not settings.llm_configured:
        raise RuntimeError("LLM_API_KEY, LLM_MODEL and LLM_BASE_URL are required")
    if not settings.embedding_configured:
        raise RuntimeError("embedding API key and base URL are required")
    from openai import AsyncOpenAI

    llm = AsyncOpenAI(
        api_key=settings.llm_api_key,
        base_url=settings.llm_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )
    embedding = AsyncOpenAI(
        api_key=settings.active_embedding_api_key,
        base_url=settings.active_embedding_base_url,
        timeout=settings.llm_timeout_seconds,
        max_retries=settings.llm_max_retries,
    )

    async def check_llm() -> dict[str, Any]:
        started = time.perf_counter()
        try:
            chat = await llm.chat.completions.create(
                model=settings.llm_model,
                temperature=0,
                max_tokens=8,
                messages=[{"role": "user", "content": "Reply with exactly OK."}],
            )
            return {
                "ok": bool(chat.choices),
                "model": settings.llm_model,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        except Exception as exc:
            return {
                "ok": False,
                "model": settings.llm_model,
                "error_type": type(exc).__name__,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }

    async def check_embedding() -> dict[str, Any]:
        started = time.perf_counter()
        try:
            vectors = await embedding.embeddings.create(
                model=settings.active_embedding_model,
                input=["AgentKnowledgeHub preflight"],
                encoding_format="float",
            )
            dimensions = len(vectors.data[0].embedding)
            usage = getattr(vectors, "usage", None)
            return {
                "ok": dimensions == settings.embedding_dimensions,
                "model": settings.active_embedding_model,
                "dimensions": dimensions,
                "expected_dimensions": settings.embedding_dimensions,
                "input_tokens": int(getattr(usage, "prompt_tokens", 0) or 0),
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }
        except Exception as exc:
            return {
                "ok": False,
                "model": settings.active_embedding_model,
                "error_type": type(exc).__name__,
                "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            }

    llm_result, embedding_result = await asyncio.gather(check_llm(), check_embedding())
    return {
        "ok": bool(llm_result["ok"] and embedding_result["ok"]),
        "llm": llm_result,
        "embedding": embedding_result,
    }


async def _multimodal_preflight() -> dict[str, Any]:
    """Run one generated image through vision and the shared VL vector space."""
    if not settings.vision_configured or not settings.vl_embedding_configured:
        raise RuntimeError("SiliconFlow vision and VL embedding configuration are required")
    from PIL import Image, ImageDraw

    vision = build_vision_client()
    if vision is None:
        raise RuntimeError("vision client is disabled")
    vl = SiliconFlowVLEmbeddingProvider()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix="akh-vl-preflight-") as directory:
        image_path = Path(directory) / "chart.png"
        image = Image.new("RGB", (256, 160), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((30, 60, 80, 130), fill="blue")
        draw.rectangle((105, 35, 155, 130), fill="green")
        draw.rectangle((180, 15, 230, 130), fill="red")
        draw.text((30, 8), "Revenue 100 150 200", fill="black")
        image.save(image_path)
        description, image_vectors, query_vector = await asyncio.gather(
            vision.describe(image),
            vl.aembed_images([str(image_path)]),
            vl.aembed_query("a revenue bar chart with three increasing bars"),
        )
    image_vector = image_vectors[0]
    denominator = math.sqrt(sum(value * value for value in image_vector)) * math.sqrt(
        sum(value * value for value in query_vector)
    )
    similarity = (
        sum(left * right for left, right in zip(image_vector, query_vector, strict=True)) / denominator
        if denominator
        else 0.0
    )
    return {
        "ok": bool(description and len(image_vector) == settings.embedding_dimensions),
        "vision": {
            "model": settings.active_vision_model,
            "description_chars": len(description),
        },
        "vl_embedding": {
            "model": settings.active_vl_embedding_model,
            "dimensions": len(image_vector),
            "text_image_cosine": round(similarity, 6),
            "usage": vl.usage_snapshot(),
        },
        "latency_ms": round((time.perf_counter() - started) * 1000, 3),
    }


async def _services() -> tuple[CatalogService, VectorStoreService, KnowledgeGraphService, IngestionService]:
    settings.ensure_runtime_dirs()
    catalog = CatalogService()
    vector = VectorStoreService()
    graph = KnowledgeGraphService()
    extractor = KnowledgeExtractAgent()
    ingestion = IngestionService(
        DocParserAgent(vision_model=build_vision_client()), extractor, vector, graph, catalog
    )
    await catalog.init()
    await vector.init()
    await graph.init()
    return catalog, vector, graph, ingestion


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    if args.command == "preflight":
        return await _preflight()
    if args.command == "preflight-multimodal":
        return await _multimodal_preflight()
    catalog, vector, graph, ingestion = await _services()
    try:
        if args.command == "repair":
            recovered = catalog.recover_interrupted_events(older_than_seconds=args.stale_seconds)
            result = await ingestion.repair_consistency(args.doc_id)
            return {"recovered_events": recovered, **result}
        if args.command == "rebuild":
            result = await ingestion.rebuild(args.doc_id)
            return {"doc_id": result.doc_id, "version": result.version, "chunks": result.chunks_total}
        if args.command == "migrate-multimodal":
            documents = (
                [catalog.get_document(args.doc_id)]
                if args.doc_id
                else catalog.list_documents(include_failed=False)
            )
            documents = [document for document in documents if document is not None]
            migrated: list[dict[str, Any]] = []
            for document in documents:
                try:
                    rows = catalog.get_chunks(document.doc_id)
                    expected = {str(row["chunk_id"]) for row in rows}
                    existing = set(await vector.get_v2_document_chunks(document.doc_id))
                    missing_rows = [row for row in rows if str(row["chunk_id"]) not in existing]
                    if not args.dry_run and missing_rows:
                        chunks = [
                            DocumentChunk(
                                content=str(row["content"]),
                                doc_id=document.doc_id,
                                chunk_index=int(row["ordinal"]),
                                doc_type=DocType(str(row["doc_type"])),
                                metadata=dict(row["metadata"]),
                                identifier=str(row["chunk_id"]),
                                content_hash=str(row["content_hash"]),
                            )
                            for row in missing_rows
                        ]
                        await vector.add_chunks(chunks)
                        existing = set(await vector.get_v2_document_chunks(document.doc_id))
                    migrated.append(
                        {
                            "doc_id": document.doc_id,
                            "expected": len(expected),
                            "already_present": len(expected & existing),
                            "pending": len(expected - existing),
                            "verified": expected <= existing,
                        }
                    )
                except Exception as exc:  # isolate one bad legacy document
                    migrated.append(
                        {
                            "doc_id": document.doc_id,
                            "expected": len(catalog.get_chunks(document.doc_id)),
                            "already_present": 0,
                            "pending": len(catalog.get_chunks(document.doc_id)),
                            "verified": False,
                            "error_type": type(exc).__name__,
                        }
                    )
            return {
                "dry_run": args.dry_run,
                "documents": migrated,
                "verified": all(item["verified"] for item in migrated) if not args.dry_run else None,
                "legacy_collection_retained": True,
            }
        raise ValueError(f"unknown command: {args.command}")
    finally:
        await graph.close()
        await catalog.close()


def main() -> None:
    parser = argparse.ArgumentParser(prog="akh-admin")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="validate chat and embedding provider configuration")
    subparsers.add_parser("preflight-multimodal", help="run one real vision and cross-modal embedding probe")
    repair = subparsers.add_parser("repair", help="repair cross-store chunk inconsistencies")
    repair.add_argument("--doc-id")
    repair.add_argument("--stale-seconds", type=float, default=300.0)
    rebuild = subparsers.add_parser("rebuild", help="fully rebuild one document")
    rebuild.add_argument("doc_id")
    migrate = subparsers.add_parser(
        "migrate-multimodal", help="idempotently copy catalog chunks into v2 modality collections"
    )
    migrate.add_argument("--doc-id")
    migrate.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(json.dumps(asyncio.run(_run(args)), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
