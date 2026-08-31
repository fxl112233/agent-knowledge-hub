"""Compatibility exports for the unified CDC implementation."""

from agents.knowledge_update_agent import CDCEvent, KnowledgeUpdateAgent, UpdateResult

CDCProcessResult = UpdateResult
CDCProcessor = KnowledgeUpdateAgent

__all__ = ["CDCEvent", "CDCProcessResult", "CDCProcessor"]
