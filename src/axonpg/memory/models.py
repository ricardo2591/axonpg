"""Validated Pydantic schemas for AxonPG's four cognitive memory layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

EpisodeOutcome = Literal["success", "failure", "partial"]
MemoryCategory = Literal["fact", "scene", "procedure", "lesson"]
ForesightStatus = Literal["active", "done", "expired"]


class EntityInput(BaseModel):
    """Entity node input for the Knowledge Graph."""

    namespace: str = "default"
    entity_type: str = Field(min_length=1, max_length=64)
    name: str = Field(min_length=1, max_length=255)
    aliases: list[str] = Field(default_factory=list)
    summary: str | None = None
    properties: dict[str, Any] = Field(default_factory=dict)
    source_lang: str | None = None


class RelationInput(BaseModel):
    """Directed graph edge between two entities."""

    namespace: str = "default"
    source_name: str = Field(min_length=1, max_length=255)
    source_type: str = "concept"
    relation_type: str = Field(min_length=1, max_length=64)
    target_name: str = Field(min_length=1, max_length=255)
    target_type: str = "concept"
    weight: float = 1.0
    properties: dict[str, Any] = Field(default_factory=dict)


class EpisodeInput(BaseModel):
    """Atomic episodic memory trace (MemCell) preserving exact hard facts."""

    namespace: str = "default"
    task_name: str = Field(min_length=1, max_length=128)
    outcome: EpisodeOutcome = "success"
    what_happened: str = Field(min_length=1)
    lessons_learned: str | None = None
    hard_facts: dict[str, Any] = Field(default_factory=dict)
    entity_names: list[str] = Field(default_factory=list)
    source_lang: str | None = None


class FactInput(BaseModel):
    """Semantic fact or scene memory with optional supersession key (`fact_key`)."""

    namespace: str = "default"
    category: MemoryCategory = "fact"
    fact_key: str | None = Field(default=None, max_length=255)
    content: str = Field(min_length=1)
    hard_facts: dict[str, Any] = Field(default_factory=dict)
    source_lang: str | None = None
    supersede_existing: bool = True


class ForesightInput(BaseModel):
    """Prospective time-bounded commitment or reminder that expires automatically."""

    namespace: str = "default"
    subject: str = Field(min_length=1, max_length=255)
    reminder: str = Field(min_length=1)
    hard_facts: dict[str, Any] = Field(default_factory=dict)
    source_lang: str | None = None
    due_at: datetime | None = None
    expires_at: datetime


class RecallItem(BaseModel):
    """Single scored item returned by hybrid recall."""

    id: str
    layer: Literal["entity", "memory", "episode", "foresight"]
    score: float
    content: str
    hard_facts: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallResult(BaseModel):
    """Token-budgeted reconstructive recall payload for agent prompt injection."""

    query: str
    namespace: str
    entities: list[dict[str, Any]] = Field(default_factory=list)
    relations: list[dict[str, Any]] = Field(default_factory=list)
    memories: list[RecallItem] = Field(default_factory=list)
    episodes: list[dict[str, Any]] = Field(default_factory=list)
    foresights: list[dict[str, Any]] = Field(default_factory=list)
    expired_archived: int = 0
    estimated_tokens: int = 0
    prompt_context: str = ""


class ConsolidationReport(BaseModel):
    """Summary of a sleep-time consolidation run."""

    namespace: str
    episodes_processed: int = 0
    scenes_created: int = 0
    facts_superseded: int = 0
    foresights_expired: int = 0
    used_llm: bool = False
    details: list[str] = Field(default_factory=list)


class MemoryStats(BaseModel):
    """Counts across all cognitive layers."""

    backend: str
    namespace: str
    entities: int = 0
    relations: int = 0
    episodes: int = 0
    unconsolidated_episodes: int = 0
    active_memories: int = 0
    superseded_memories: int = 0
    active_foresights: int = 0
    expired_foresights: int = 0
