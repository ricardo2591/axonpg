"""Abstract MemoryBackend contract, shared RRF hybrid ranking, and prompt context assembly."""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from axonpg.i18n import extract_search_tokens, normalize_lookup_key, normalize_unicode
from axonpg.memory.models import (
    EntityInput,
    EpisodeInput,
    FactInput,
    ForesightInput,
    MemoryStats,
    RecallResult,
    RelationInput,
)


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimate (~3.6 chars/token across multilingual text)."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 3.6))


def reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, float]]],
    k: int = 60,
) -> dict[str, float]:
    """Combine keyword/FTS, trigram, and vector rankings via Reciprocal Rank Fusion."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank_idx, (item_id, _raw_score) in enumerate(ranked, start=1):
            scores[item_id] = scores.get(item_id, 0.0) + 1.0 / (k + rank_idx)
    return scores


def format_recall_context(result: RecallResult, max_tokens: int = 800) -> str:
    """
    Build a concise, language-neutral reconstructive prompt block from recalled memory layers
    while strictly preserving exact `hard_facts`.
    """
    sections: list[str] = []

    if result.foresights:
        lines = ["[ACTIVE_FORESIGHTS]"]
        for fs in result.foresights:
            due = f" (due: {fs['due_at']})" if fs.get("due_at") else ""
            facts = f" | hard_facts={json.dumps(fs['hard_facts'], ensure_ascii=False)}" if fs.get("hard_facts") else ""
            lines.append(f"- {fs['subject']}: {fs['reminder']}{due}{facts}")
        sections.append("\n".join(lines))

    if result.entities or result.relations:
        lines = ["[KNOWLEDGE_GRAPH]"]
        for ent in result.entities:
            props = f" props={json.dumps(ent['properties'], ensure_ascii=False)}" if ent.get("properties") else ""
            summary = f" — {ent['summary']}" if ent.get("summary") else ""
            lines.append(f"- ({ent['entity_type']}) {ent['name']}{summary}{props}")
        for rel in result.relations:
            lines.append(f"- {rel['source']} --[{rel['relation_type']}]--> {rel['target']}")
        sections.append("\n".join(lines))

    if result.memories:
        lines = ["[SEMANTIC_MEMORIES]"]
        for mem in result.memories:
            facts = f" | hard_facts={json.dumps(mem.hard_facts, ensure_ascii=False)}" if mem.hard_facts else ""
            lines.append(f"- [{mem.metadata.get('category', 'fact')}] {mem.content}{facts}")
        sections.append("\n".join(lines))

    if result.episodes:
        lines = ["[EPISODIC_LESSONS]"]
        for ep in result.episodes:
            lesson = f" | lesson: {ep['lessons_learned']}" if ep.get("lessons_learned") else ""
            facts = f" | hard_facts={json.dumps(ep['hard_facts'], ensure_ascii=False)}" if ep.get("hard_facts") else ""
            lines.append(f"- [{ep['outcome']}] {ep['task_name']}: {ep['what_happened']}{lesson}{facts}")
        sections.append("\n".join(lines))

    if not sections:
        return ""

    header = f"[AXONPG_MEMORY namespace={result.namespace}]"
    assembled = [header]
    char_budget = int(max_tokens * 3.6)
    used_chars = len(header)

    for sec in sections:
        if used_chars + len(sec) + 2 <= char_budget:
            assembled.append(sec)
            used_chars += len(sec) + 2
        else:
            remaining = char_budget - used_chars - 20
            if remaining > 60:
                assembled.append(sec[:remaining].rstrip() + "\n... (truncated to token budget)")
            break

    return "\n\n".join(assembled)


class MemoryBackend(ABC):
    """Unified interface implemented by both PostgresMemoryStore and SQLiteMemoryStore."""

    backend_name: str

    @abstractmethod
    def ensure_schema(self) -> None:
        """Initialize tables, indexes, and extensions idempotently."""

    @abstractmethod
    def upsert_entity(self, item: EntityInput) -> dict[str, Any]:
        """Insert or update an entity node in the knowledge graph."""

    @abstractmethod
    def add_relation(self, item: RelationInput) -> dict[str, Any]:
        """Create or update a directed relation between two entities."""

    @abstractmethod
    def record_episode(self, item: EpisodeInput) -> dict[str, Any]:
        """Record an atomic episodic trace (MemCell) preserving hard facts."""

    @abstractmethod
    def upsert_fact(self, item: FactInput) -> dict[str, Any]:
        """Store a semantic fact/scene, optionally superseding an older fact with the same key."""

    @abstractmethod
    def add_foresight(self, item: ForesightInput) -> dict[str, Any]:
        """Store a time-bounded foresight commitment/reminder."""

    @abstractmethod
    def complete_foresight(self, foresight_id: str, namespace: str = "default") -> dict[str, Any]:
        """Mark an active foresight as done."""

    @abstractmethod
    def expire_stale_foresights(self, namespace: str = "default") -> int:
        """Mark past-expiration foresights as expired and return count archived."""

    @abstractmethod
    def recall(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 5,
        max_tokens: int = 800,
        include_episodes: bool = True,
    ) -> RecallResult:
        """Perform hybrid graph + FTS + vector + prospective reconstructive recall."""

    @abstractmethod
    def fetch_unconsolidated_episodes(self, namespace: str = "default", limit: int = 50) -> list[dict[str, Any]]:
        """Fetch unconsolidated episodes for sleep-time consolidation."""

    @abstractmethod
    def mark_episodes_consolidated(self, episode_ids: list[str], namespace: str = "default") -> int:
        """Mark processed episodes as consolidated."""

    @abstractmethod
    def stats(self, namespace: str = "default") -> MemoryStats:
        """Return memory layer counts for diagnostics."""

    @abstractmethod
    def healthcheck(self) -> tuple[bool, str]:
        """Verify database reachability and schema readiness."""

    @abstractmethod
    def list_foresights(
        self,
        namespace: str = "default",
        status: str = "active",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List prospective foresights filtered by status ('active', 'done', 'expired', or 'all')."""

    @abstractmethod
    def export_namespace(self, namespace: str = "default") -> dict[str, Any]:
        """Export all cognitive layers of a namespace as a portable JSON-serializable snapshot."""

    @abstractmethod
    def reset_namespace(self, namespace: str) -> dict[str, int]:
        """Delete all stored rows for a namespace and return deleted counts per layer."""

    def import_namespace(
        self,
        payload: dict[str, Any],
        target_namespace: str | None = None,
    ) -> dict[str, int]:
        """Import a portable namespace snapshot into this backend cleanly and re-embed facts."""
        ns = target_namespace or str(payload.get("namespace") or "default")
        counts = {"entities": 0, "relations": 0, "episodes": 0, "facts": 0, "foresights": 0}

        for ent in payload.get("entities") or []:
            self.upsert_entity(
                EntityInput(
                    namespace=ns,
                    entity_type=str(ent.get("entity_type") or "concept"),
                    name=str(ent["name"]),
                    aliases=list(ent.get("aliases") or []),
                    summary=ent.get("summary"),
                    properties=dict(ent.get("properties") or {}),
                    source_lang=ent.get("source_lang"),
                )
            )
            counts["entities"] += 1

        for rel in payload.get("relations") or []:
            self.add_relation(
                RelationInput(
                    namespace=ns,
                    source_name=str(rel["source_name"]),
                    source_type=str(rel.get("source_type") or "concept"),
                    relation_type=str(rel["relation_type"]),
                    target_name=str(rel["target_name"]),
                    target_type=str(rel.get("target_type") or "concept"),
                    weight=float(rel.get("weight", 1.0)),
                    properties=dict(rel.get("properties") or {}),
                )
            )
            counts["relations"] += 1

        consolidated_ids: list[str] = []
        for ep in payload.get("episodes") or []:
            res = self.record_episode(
                EpisodeInput(
                    namespace=ns,
                    task_name=str(ep["task_name"]),
                    outcome=ep.get("outcome") or "success",
                    what_happened=str(ep["what_happened"]),
                    lessons_learned=ep.get("lessons_learned"),
                    hard_facts=dict(ep.get("hard_facts") or {}),
                    entity_names=list(ep.get("entity_names") or []),
                    source_lang=ep.get("source_lang"),
                )
            )
            if ep.get("consolidated") and res.get("id"):
                consolidated_ids.append(str(res["id"]))
            counts["episodes"] += 1

        if consolidated_ids:
            self.mark_episodes_consolidated(consolidated_ids, namespace=ns)

        for fact in payload.get("facts") or []:
            self.upsert_fact(
                FactInput(
                    namespace=ns,
                    category=fact.get("category") or "fact",
                    fact_key=fact.get("fact_key"),
                    content=str(fact["content"]),
                    hard_facts=dict(fact.get("hard_facts") or {}),
                    source_lang=fact.get("source_lang"),
                    supersede_existing=bool(fact.get("fact_key")),
                )
            )
            counts["facts"] += 1

        for fs in payload.get("foresights") or []:
            expires_raw = str(fs["expires_at"]).replace("Z", "+00:00")
            due_raw = fs.get("due_at")
            due_dt = datetime.fromisoformat(str(due_raw).replace("Z", "+00:00")) if due_raw else None
            fs_res = self.add_foresight(
                ForesightInput(
                    namespace=ns,
                    subject=str(fs["subject"]),
                    reminder=str(fs["reminder"]),
                    hard_facts=dict(fs.get("hard_facts") or {}),
                    source_lang=fs.get("source_lang"),
                    due_at=due_dt,
                    expires_at=datetime.fromisoformat(expires_raw),
                )
            )
            if fs.get("status") == "done" and fs_res.get("id"):
                self.complete_foresight(str(fs_res["id"]), namespace=ns)
            counts["foresights"] += 1

        self.expire_stale_foresights(namespace=ns)
        return counts

    @staticmethod
    def _utc_iso_now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def close(self) -> None:  # noqa: B027
        """Release underlying connections cleanly."""

    @staticmethod
    def _clean_text(value: str | None) -> str:
        return normalize_unicode(value)

    @staticmethod
    def _lookup_key(value: str | None) -> str:
        return normalize_lookup_key(value)

    @staticmethod
    def _search_tokens(value: str | None) -> list[str]:
        return extract_search_tokens(value)
