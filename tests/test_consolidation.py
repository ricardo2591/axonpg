"""Tests for sleep-time consolidation (deterministic and LLM-assisted with fallback)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from axonpg.memory.consolidate import consolidate_namespace
from axonpg.memory.models import EpisodeInput, ForesightInput
from axonpg.memory.sqlite_store import SQLiteMemoryStore
from axonpg.providers.embeddings import NullEmbeddingProvider
from axonpg.providers.llm import LLMProvider


class FakeMultilingualLLM(LLMProvider):
    model_name = "fake-llm"

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        return {
            "scenes": [
                {
                    "fact_key": "incident:router-bogota",
                    "category": "lesson",
                    "summary": "La rotación automática en Bogotá falló por timeout DNS y se resolvió aumentando el TTL.",
                    "source_lang": "es",
                    "entities": [{"name": "Router Bogotá", "entity_type": "hardware"}],
                    "relations": [
                        {"source": "Router Bogotá", "relation_type": "fixed_by", "target": "DNS TTL Cache"}
                    ],
                }
            ]
        }


class FailingLLM(LLMProvider):
    model_name = "broken-llm"

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        raise RuntimeError("Simulated upstream outage")


def test_deterministic_consolidation_preserves_hard_facts_and_supersedes():
    store = SQLiteMemoryStore(":memory:", NullEmbeddingProvider())
    try:
        store.add_foresight(
            ForesightInput(
                namespace="ops",
                subject="Old alert",
                reminder="Already stale",
                expires_at=datetime.now(timezone.utc) - timedelta(minutes=5),
            )
        )
        store.record_episode(
            EpisodeInput(
                namespace="ops",
                task_name="Rotación Token",
                outcome="failure",
                what_happened="Primer intento devolvió HTTP 429",
                lessons_learned="Esperar backoff exponencial",
                hard_facts={"ticket": "INC-901", "status_code": 429},
                entity_names=["Gateway Auth"],
                source_lang="es",
            )
        )
        store.record_episode(
            EpisodeInput(
                namespace="ops",
                task_name="Rotación Token",
                outcome="success",
                what_happened="Segundo intento completado tras 15s",
                hard_facts={"retry_wait_s": 15.5},
                source_lang="es",
            )
        )

        report = consolidate_namespace(store=store, llm=None, namespace="ops")
        assert report.episodes_processed == 2
        assert report.scenes_created == 1
        assert report.foresights_expired == 1
        assert report.used_llm is False

        recalled = store.recall("Rotacion Token Gateway Auth", namespace="ops")
        assert recalled.memories
        assert recalled.memories[0].hard_facts == {"ticket": "INC-901", "status_code": 429, "retry_wait_s": 15.5}
        assert any(e["name"] == "Gateway Auth" for e in recalled.entities)

        # Adding another episode with same task_name consolidates and supersedes the prior scene cleanly
        store.record_episode(
            EpisodeInput(
                namespace="ops",
                task_name="Rotación Token",
                outcome="success",
                what_happened="Tercer intento sin demoras",
                hard_facts={"retry_wait_s": 0},
                source_lang="es",
            )
        )
        report2 = consolidate_namespace(store=store, llm=None, namespace="ops")
        assert report2.facts_superseded == 1
        assert store.stats("ops").unconsolidated_episodes == 0
    finally:
        store.close()


def test_llm_consolidation_and_graceful_fallback():
    store = SQLiteMemoryStore(":memory:", NullEmbeddingProvider())
    try:
        store.record_episode(
            EpisodeInput(
                namespace="net",
                task_name="Router Bogotá",
                outcome="failure",
                what_happened="Timeout DNS en enlace primario",
                hard_facts={"ttl": 300, "latency_ms": 420.5},
                source_lang="es",
            )
        )
        rep = consolidate_namespace(store=store, llm=FakeMultilingualLLM(), namespace="net")
        assert rep.used_llm is True
        assert rep.scenes_created == 1

        res = store.recall("Bogota DNS TTL", namespace="net")
        assert any("Bogotá" in m.content for m in res.memories)
        assert res.memories[0].hard_facts["latency_ms"] == 420.5
        assert any(r["relation_type"] == "fixed_by" for r in res.relations)

        # Now ensure FailingLLM cleanly falls back to deterministic synthesis without data loss
        store.record_episode(
            EpisodeInput(
                namespace="net",
                task_name="Backup DB",
                outcome="success",
                what_happened="Dump complete",
                hard_facts={"size_mb": 128},
            )
        )
        rep_fb = consolidate_namespace(store=store, llm=FailingLLM(), namespace="net")
        assert rep_fb.episodes_processed == 1
        assert rep_fb.used_llm is False
        assert any("LLM fallback" in d for d in rep_fb.details)
    finally:
        store.close()
