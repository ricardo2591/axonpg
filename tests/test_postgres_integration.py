"""Live PostgreSQL + pgvector + pg_trgm + unaccent integration tests."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest

from axonpg.config import AxonConfig
from axonpg.memory.consolidate import consolidate_namespace
from axonpg.memory.models import (
    EntityInput,
    EpisodeInput,
    FactInput,
    ForesightInput,
    RelationInput,
)
from axonpg.memory.postgres_store import PostgresMemoryStore
from axonpg.providers.embeddings import EmbeddingProvider


class DeterministicPGEmbedder(EmbeddingProvider):
    """Fixed-384d deterministic semantic test vectors (orthogonal across topics)."""

    model_name = "test-pg-384d"
    dim = 384

    def _encode(self, text: str) -> list[float]:
        low = text.casefold()
        vec = [0.0] * self.dim
        if any(w in low for w in ("postgres", "5435", "database", "データベース")):
            vec[0] = 1.0
        if any(w in low for w in ("bogota", "bogotá", "camión", "camion")):
            vec[1] = 1.0
        if any(w in low for w in ("são paulo", "sao paulo", "pasarela")):
            vec[2] = 1.0
        if sum(vec) == 0.0:
            vec[3] = 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode(query)


pytestmark = pytest.mark.integration


def _require_pg_dsn() -> str:
    if os.getenv("AXONPG_RUN_PG_TESTS") != "1":
        pytest.skip("Set AXONPG_RUN_PG_TESTS=1 with a running pgvector instance to execute integration tests.")
    cfg = AxonConfig.from_env()
    return cfg.postgres_dsn()


def test_postgres_pgvector_end_to_end():
    dsn = _require_pg_dsn()
    store = PostgresMemoryStore(dsn=dsn, embedder=DeterministicPGEmbedder())
    ns = "itest_axonpg"
    try:
        # Clean test namespace before run so repeated runs stay idempotent
        with store._cursor(commit=True) as cur:
            for tbl in ("axon_relations", "axon_entities", "axon_episodes", "axon_memories", "axon_foresights"):
                cur.execute(f"DELETE FROM {tbl} WHERE namespace = %s;", (ns,))

        ok, msg = store.healthcheck()
        assert ok, msg

        store.upsert_entity(
            EntityInput(
                namespace=ns,
                entity_type="service",
                name="Pasarela São Paulo",
                aliases=["Gateway SP"],
                summary="Nodo regional de ingesta",
                source_lang="pt",
            )
        )
        store.add_relation(
            RelationInput(
                namespace=ns,
                source_name="Pasarela São Paulo",
                relation_type="connects_to",
                target_name="PostgreSQL Core",
            )
        )

        store.upsert_fact(
            FactInput(
                namespace=ns,
                fact_key="port:pg",
                content="Puerto inicial era 5432 en São Paulo",
                hard_facts={"port": 5432},
                source_lang="es",
            )
        )
        r2 = store.upsert_fact(
            FactInput(
                namespace=ns,
                fact_key="port:pg",
                content="Migración completada: PostgreSQL escucha en 5435 para camión Bogotá",
                hard_facts={"port": 5435, "city": "Bogotá"},
                source_lang="es",
            )
        )
        assert r2["superseded_previous"] == 1

        store.upsert_fact(
            FactInput(
                namespace=ns,
                fact_key="jp:note",
                content="東京データベースの接続テスト成功",
                source_lang="ja",
            )
        )

        now = datetime.now(timezone.utc)
        store.add_foresight(
            ForesightInput(
                namespace=ns,
                subject="Expired PG item",
                reminder="Old item",
                expires_at=now - timedelta(minutes=10),
            )
        )
        active_fs = store.add_foresight(
            ForesightInput(
                namespace=ns,
                subject="Rotate backup key",
                reminder="Verificar respaldo semanal",
                expires_at=now + timedelta(days=5),
            )
        )

        store.record_episode(
            EpisodeInput(
                namespace=ns,
                task_name="Validación Migración",
                outcome="success",
                what_happened="Pool Postgres estable tras cambio a 5435",
                lessons_learned="Confirmar pg_trgm y unaccent antes de tráfico",
                hard_facts={"port": 5435, "pool_max": 8},
                entity_names=["PostgreSQL Core"],
                source_lang="es",
            )
        )

        # Unaccented Spanish/Portuguese query must match accented Postgres records + vector + graph
        recalled = store.recall("camion Bogota Sao Paulo 5435", namespace=ns, limit=5)
        ports = [m.hard_facts.get("port") for m in recalled.memories if "port" in m.hard_facts]
        assert ports == [5435]
        assert recalled.expired_archived == 1
        assert any(e["name"] == "Pasarela São Paulo" for e in recalled.entities)
        assert any(r["relation_type"] == "connects_to" for r in recalled.relations)
        assert len(recalled.foresights) == 1
        assert recalled.foresights[0]["id"] == active_fs["id"]

        # Japanese recall via Postgres CJK-safe substring/vector fallback
        recalled_jp = store.recall("東京データベース", namespace=ns, limit=3)
        assert any("東京" in m.content for m in recalled_jp.memories)

        # Deterministic sleep consolidation in Postgres
        rep = consolidate_namespace(store=store, llm=None, namespace=ns)
        assert rep.episodes_processed == 1
        assert rep.scenes_created == 1

        stats = store.stats(ns)
        assert stats.unconsolidated_episodes == 0
        assert stats.active_memories >= 3
        assert stats.superseded_memories == 1

        # Export -> Reset -> Import -> Redundant Import parity in PostgreSQL
        snap = store.export_namespace(namespace=ns)
        assert snap["schema_version"] == 1
        assert len(snap["entities"]) >= 2
        assert len(snap["facts"]) >= 3

        deleted = store.reset_namespace(namespace=ns)
        assert deleted["memories"] >= 3
        assert store.stats(ns).active_memories == 0

        imported = store.import_namespace(snap, target_namespace=ns)
        assert imported["facts"] == len(snap["facts"])
        assert store.stats(ns).active_memories >= 3
    finally:
        store.close()
