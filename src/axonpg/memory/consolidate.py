"""Sleep-time episodic-to-semantic consolidation preserving exact hard_facts across languages."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from axonpg.i18n import normalize_lookup_key, normalize_unicode
from axonpg.memory.backend import MemoryBackend
from axonpg.memory.models import (
    ConsolidationReport,
    EntityInput,
    FactInput,
    RelationInput,
)
from axonpg.providers.llm import LLMProvider

CONSOLIDATION_SYSTEM_PROMPT = """You are AxonPG's sleep-time cognitive memory consolidator.
Synthesize raw episodic traces into durable semantic scenes and lessons.
CRITICAL RULES:
1. Preserve the user/source language of the episodes; NEVER translate non-English memories into English unless mixed.
2. NEVER invent or alter numbers, dates, IDs, currencies, URLs, or error codes.
3. Return ONLY a strict JSON object matching this schema:
{
  "scenes": [
    {
      "fact_key": "stable lowercase identifier",
      "category": "scene",
      "summary": "concise consolidated knowledge in original language",
      "source_lang": "es|en|pt|fr|...",
      "entities": [{"name": "Entity", "entity_type": "concept"}],
      "relations": [{"source": "A", "relation_type": "depends_on", "target": "B"}]
    }
  ]
}
"""


def _merge_hard_facts(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    """Merge exact hard_facts dictionaries chronologically without losing precision."""
    merged: dict[str, Any] = {}
    for ep in episodes:
        facts = ep.get("hard_facts") or {}
        for k, v in facts.items():
            merged[str(k)] = v
    return merged


def _deterministic_scene_text(task_name: str, episodes: list[dict[str, Any]]) -> str:
    """Build a language-neutral deterministic scene summary from grouped episodes."""
    outcomes = [ep["outcome"] for ep in episodes]
    events = [normalize_unicode(ep["what_happened"]) for ep in episodes if ep.get("what_happened")]
    lessons = [
        normalize_unicode(ep["lessons_learned"])
        for ep in episodes
        if ep.get("lessons_learned")
    ]
    parts = [f"{task_name} ({'/'.join(dict.fromkeys(outcomes))}, n={len(episodes)}):"]
    if events:
        parts.append(" | ".join(dict.fromkeys(events[-3:])))
    if lessons:
        parts.append(f"lessons: {' ; '.join(dict.fromkeys(lessons))}")
    return " ".join(parts)


class SleepConsolidator:
    """Consolidates unconsolidated episodic MemCells into active semantic scenes & graph links."""

    def __init__(self, store: MemoryBackend, llm: LLMProvider | None = None):
        self.store = store
        self.llm = llm

    def run(self, namespace: str = "default", batch_limit: int = 50) -> ConsolidationReport:
        """Expire stale foresights and compress unconsolidated episodes into semantic memories."""
        expired = self.store.expire_stale_foresights(namespace=namespace)
        episodes = self.store.fetch_unconsolidated_episodes(namespace=namespace, limit=batch_limit)
        if not episodes:
            return ConsolidationReport(
                namespace=namespace,
                foresights_expired=expired,
                details=["No unconsolidated episodes pending."],
            )

        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for ep in episodes:
            key = normalize_lookup_key(ep.get("task_name")) or "general"
            grouped[key].append(ep)

        used_llm = False
        scenes_created = 0
        facts_superseded = 0
        details: list[str] = []

        for group_key, group_eps in grouped.items():
            hard_facts = _merge_hard_facts(group_eps)
            task_label = normalize_unicode(group_eps[-1]["task_name"])
            source_lang = next((e.get("source_lang") for e in reversed(group_eps) if e.get("source_lang")), None)

            llm_scenes: list[dict[str, Any]] | None = None
            if self.llm is not None:
                try:
                    payload = json.dumps(group_eps, ensure_ascii=False)
                    raw = self.llm.complete_json(
                        CONSOLIDATION_SYSTEM_PROMPT,
                        f"Consolidate these {len(group_eps)} episodes (preserve exact language and facts):\n{payload}",
                    )
                    candidate = raw.get("scenes")
                    if isinstance(candidate, list) and candidate:
                        llm_scenes = candidate
                        used_llm = True
                except Exception as exc:
                    details.append(f"LLM fallback for '{group_key}': {exc}")

            if llm_scenes:
                for sc in llm_scenes:
                    content = normalize_unicode(str(sc.get("summary") or ""))
                    if not content:
                        continue
                    fact_key = normalize_lookup_key(str(sc.get("fact_key") or f"scene:{group_key}"))
                    cat = sc.get("category") if sc.get("category") in {"fact", "scene", "procedure", "lesson"} else "scene"
                    res = self.store.upsert_fact(
                        FactInput(
                            namespace=namespace,
                            category=cat,
                            fact_key=fact_key,
                            content=content,
                            hard_facts=hard_facts,
                            source_lang=sc.get("source_lang") or source_lang,
                            supersede_existing=True,
                        )
                    )
                    scenes_created += 1
                    facts_superseded += int(res.get("superseded_previous", 0))

                    for ent in sc.get("entities") or []:
                        if isinstance(ent, dict) and ent.get("name"):
                            self.store.upsert_entity(
                                EntityInput(
                                    namespace=namespace,
                                    entity_type=str(ent.get("entity_type") or "concept"),
                                    name=str(ent["name"]),
                                    source_lang=source_lang,
                                )
                            )
                    for rel in sc.get("relations") or []:
                        if isinstance(rel, dict) and rel.get("source") and rel.get("target"):
                            self.store.add_relation(
                                RelationInput(
                                    namespace=namespace,
                                    source_name=str(rel["source"]),
                                    relation_type=str(rel.get("relation_type") or "related_to"),
                                    target_name=str(rel["target"]),
                                )
                            )
            else:
                summary = _deterministic_scene_text(task_label, group_eps)
                category = "lesson" if any(ep.get("lessons_learned") for ep in group_eps) else "scene"
                res = self.store.upsert_fact(
                    FactInput(
                        namespace=namespace,
                        category=category,
                        fact_key=f"scene:{group_key}",
                        content=summary,
                        hard_facts=hard_facts,
                        source_lang=source_lang,
                        supersede_existing=True,
                    )
                )
                scenes_created += 1
                facts_superseded += int(res.get("superseded_previous", 0))

                seen_entities: list[str] = []
                for ep in group_eps:
                    for ename in ep.get("entity_names") or []:
                        cleaned = normalize_unicode(str(ename))
                        if cleaned and cleaned not in seen_entities:
                            seen_entities.append(cleaned)
                            self.store.upsert_entity(
                                EntityInput(
                                    namespace=namespace,
                                    entity_type="concept",
                                    name=cleaned,
                                    source_lang=source_lang,
                                )
                            )

        ep_ids = [ep["id"] for ep in episodes]
        processed = self.store.mark_episodes_consolidated(ep_ids, namespace=namespace)
        details.append(f"Consolidated {processed} episodes into {scenes_created} active scene(s).")

        return ConsolidationReport(
            namespace=namespace,
            episodes_processed=processed,
            scenes_created=scenes_created,
            facts_superseded=facts_superseded,
            foresights_expired=expired,
            used_llm=used_llm,
            details=details,
        )


def consolidate_namespace(
    store: MemoryBackend,
    llm: LLMProvider | None = None,
    namespace: str = "default",
    batch_limit: int = 50,
) -> ConsolidationReport:
    """Convenience functional wrapper around SleepConsolidator."""
    return SleepConsolidator(store=store, llm=llm).run(namespace=namespace, batch_limit=batch_limit)
