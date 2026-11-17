"""End-to-end multilingual CRUD, supersession, foresight expiry, and hybrid recall in SQLite --lite."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from axonpg.memory.models import (
    EntityInput,
    EpisodeInput,
    FactInput,
    ForesightInput,
    RelationInput,
)
from axonpg.memory.sqlite_store import SQLiteMemoryStore
from axonpg.providers.embeddings import EmbeddingProvider


class DeterministicMultilingualEmbedder(EmbeddingProvider):
    """Offline test embedder mapping topics across languages to orthogonal unit vectors."""

    model_name = "test-multilingual-vectors"

    def __init__(self, dim: int = 4):
        self.dim = dim

    def _encode(self, text: str) -> list[float]:
        low = text.casefold()
        vec = [0.0] * self.dim
        if any(w in low for w in ("postgres", "5435", "database", "データベース", "base de datos")):
            vec[0] = 1.0
        if any(w in low for w in ("são paulo", "sao paulo", "gateway", "camion", "camión")):
            vec[1] = 1.0
        if any(w in low for w in ("llave", "rotation", "rotación", "secret")):
            vec[2] = 1.0
        if sum(vec) == 0.0:
            vec[3] = 1.0
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._encode(t) for t in texts]

    def embed_query(self, query: str) -> list[float]:
        return self._encode(query)


def test_multilingual_hybrid_recall_across_all_four_layers(tmp_path):
    db_file = tmp_path / "memory.db"
    store = SQLiteMemoryStore(db_path=db_file, embedder=DeterministicMultilingualEmbedder())
    try:
        # 1. Knowledge Graph with accented aliases + relation
        store.upsert_entity(
            EntityInput(
                namespace="prod",
                entity_type="service",
                name="Pasarela São Paulo",
                aliases=["Gateway SP", "Pasarela Sao Paulo"],
                summary="Nodo principal de ingesta",
                properties={"region": "sa-east-1"},
                source_lang="pt",
            )
        )
        store.add_relation(
            RelationInput(
                namespace="prod",
                source_name="Pasarela São Paulo",
                source_type="service",
                relation_type="writes_to",
                target_name="PostgreSQL Core",
                target_type="database",
            )
        )

        # 2. Facts in Spanish, French, Japanese, and Arabic
        store.upsert_fact(
            FactInput(
                namespace="prod",
                fact_key="deploy:sao-paulo-camion",
                content="La configuración del camión en São Paulo usa PostgreSQL en puerto 5435",
                hard_facts={"port": 5435, "city": "São Paulo"},
                source_lang="es",
            )
        )
        store.upsert_fact(
            FactInput(
                namespace="prod",
                fact_key="net:reseau",
                content="Réseau principal configuré avec tolérance de panne",
                source_lang="fr",
            )
        )
        store.upsert_fact(
            FactInput(
                namespace="prod",
                fact_key="jp:db",
                content="東京データベースの設定は正常に完了しました",
                source_lang="ja",
            )
        )
        store.upsert_fact(
            FactInput(
                namespace="prod",
                fact_key="ar:server",
                content="تم تشغيل خادم الذاكرة بنجاح",
                source_lang="ar",
            )
        )

        # Query WITHOUT accents in Spanish/Portuguese -> retrieves entity, relation, and fact!
        res_es = store.recall("configuracion camion Sao Paulo", namespace="prod", limit=5)
        assert any(e["name"] == "Pasarela São Paulo" for e in res_es.entities)
        assert any(r["relation_type"] == "writes_to" for r in res_es.relations)
        assert any(m.hard_facts.get("port") == 5435 for m in res_es.memories)
        assert "[KNOWLEDGE_GRAPH]" in res_es.prompt_context
        assert "[SEMANTIC_MEMORIES]" in res_es.prompt_context

        # Query in French without accents
        res_fr = store.recall("reseau configure", namespace="prod", limit=3)
        assert any("Réseau" in m.content for m in res_fr.memories)

        # Query in Japanese (no whitespace between tokens!)
        res_ja = store.recall("東京データベース", namespace="prod", limit=3)
        assert any("東京" in m.content for m in res_ja.memories)

        # Query in Arabic
        res_ar = store.recall("خادم الذاكرة", namespace="prod", limit=3)
        assert any("خادم" in m.content for m in res_ar.memories)
    finally:
        store.close()


def test_contradiction_supersession_and_foresight_auto_expiry():
    store = SQLiteMemoryStore(db_path=":memory:", embedder=DeterministicMultilingualEmbedder())
    try:
        r1 = store.upsert_fact(
            FactInput(
                namespace="app",
                fact_key="db:port",
                content="PostgreSQL port is 5432",
                hard_facts={"port": 5432},
            )
        )
        assert r1["superseded_previous"] == 0

        r2 = store.upsert_fact(
            FactInput(
                namespace="app",
                fact_key="db:port",
                content="PostgreSQL port migrated to 5435",
                hard_facts={"port": 5435},
            )
        )
        assert r2["superseded_previous"] == 1

        now = datetime.now(timezone.utc)
        store.add_foresight(
            ForesightInput(
                namespace="app",
                subject="Expired certificate check",
                reminder="Renew staging TLS",
                expires_at=now - timedelta(hours=1),
            )
        )
        active_fs = store.add_foresight(
            ForesightInput(
                namespace="app",
                subject="Audit backups",
                reminder="Verify nightly pg_dump checksum",
                hard_facts={"bucket": "bkp-01"},
                expires_at=now + timedelta(days=2),
            )
        )

        store.record_episode(
            EpisodeInput(
                namespace="app",
                task_name="migrate-port",
                outcome="success",
                what_happened="Changed database listener to 5435",
                lessons_learned="Update client .env before restart",
                hard_facts={"old_port": 5432, "new_port": 5435},
            )
        )

        res = store.recall("PostgreSQL port 5435", namespace="app", limit=5)
        ports = [m.hard_facts.get("port") for m in res.memories if "port" in m.hard_facts]
        assert ports == [5435]
        assert res.expired_archived == 1
        assert len(res.foresights) == 1
        assert res.foresights[0]["id"] == active_fs["id"]
        assert len(res.episodes) == 1

        # Complete foresight cleanly
        comp = store.complete_foresight(active_fs["id"], namespace="app")
        assert comp["updated"] is True
        stats = store.stats("app")
        assert stats.active_memories == 1
        assert stats.superseded_memories == 1
        assert stats.active_foresights == 0
        assert stats.expired_foresights == 1
    finally:
        store.close()


def test_recall_token_budget_truncation():
    store = SQLiteMemoryStore(db_path=":memory:", embedder=DeterministicMultilingualEmbedder())
    try:
        for i in range(6):
            store.upsert_fact(
                FactInput(
                    namespace="budget",
                    content=f"PostgreSQL memory item #{i}: " + ("detalles críticos del sistema " * 12),
                )
            )
        res = store.recall("PostgreSQL", namespace="budget", limit=6, max_tokens=120)
        assert res.estimated_tokens <= 150  # truncation indicator adds small overhead
        assert "truncated to token budget" in res.prompt_context
    finally:
        store.close()
