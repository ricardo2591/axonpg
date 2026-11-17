"""PostgreSQL + pgvector + pg_trgm + unaccent + tsvector hybrid cognitive memory backend."""

from __future__ import annotations

import json
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any

from psycopg2.extras import RealDictCursor
from psycopg2.pool import ThreadedConnectionPool

from axonpg.i18n import fold_accents_preserving_scripts
from axonpg.memory.backend import (
    MemoryBackend,
    estimate_tokens,
    format_recall_context,
    reciprocal_rank_fusion,
)
from axonpg.memory.models import (
    EntityInput,
    EpisodeInput,
    FactInput,
    ForesightInput,
    MemoryStats,
    RecallItem,
    RecallResult,
    RelationInput,
)
from axonpg.providers.embeddings import EmbeddingProvider


def _utc_dt(dt: datetime | None = None) -> datetime:
    target = dt or datetime.now(timezone.utc)
    if target.tzinfo is None:
        return target.replace(tzinfo=timezone.utc)
    return target.astimezone(timezone.utc)


def _vector_literal(vec: list[float] | None) -> str | None:
    if not vec:
        return None
    return "[" + ",".join(f"{float(v):.8f}" for v in vec) + "]"


def _safe_tsquery_expr(tokens: list[str]) -> str:
    """Build a Postgres `to_tsquery('simple', ...)` OR expression safe against syntax chars."""
    clean_parts: list[str] = []
    for tok in tokens[:10]:
        words = re.findall(r"[^\W_]+", tok, re.UNICODE)
        for word in words:
            folded = fold_accents_preserving_scripts(word)
            if folded:
                clean_parts.append(folded)
    return " | ".join(dict.fromkeys(clean_parts))


def _iso_or_none(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc).isoformat()
    return str(value)


class PostgresMemoryStore(MemoryBackend):
    """Production single-PostgreSQL cognitive memory backend."""

    backend_name = "postgres-pgvector"

    def __init__(self, dsn: str, embedder: EmbeddingProvider, minconn: int = 1, maxconn: int = 8):
        self.dsn = dsn
        self.embedder = embedder
        self._pool = ThreadedConnectionPool(minconn=minconn, maxconn=maxconn, dsn=dsn)
        self.ensure_schema()

    @contextmanager
    def _cursor(self, commit: bool = False):
        conn = self._pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                yield cur
            if commit:
                conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            self._pool.putconn(conn)

    def ensure_schema(self) -> None:
        dim = int(self.embedder.dim)
        if not 32 <= dim <= 4096:
            raise ValueError(f"Unsupported embedding dimension for pgvector: {dim}")

        with self._cursor(commit=True) as cur:
            cur.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp";')
            cur.execute('CREATE EXTENSION IF NOT EXISTS "pg_trgm";')
            cur.execute('CREATE EXTENSION IF NOT EXISTS "unaccent";')
            cur.execute('CREATE EXTENSION IF NOT EXISTS "vector";')

            cur.execute(
                """
                CREATE OR REPLACE FUNCTION axon_unaccent(text)
                RETURNS text AS $$
                    SELECT public.unaccent('public.unaccent', $1);
                $$ LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE;
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS axon_entities (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
                    entity_type VARCHAR(64) NOT NULL,
                    name VARCHAR(255) NOT NULL,
                    name_norm VARCHAR(255) NOT NULL,
                    aliases TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                    summary TEXT,
                    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source_lang VARCHAR(16),
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_axon_entity UNIQUE (namespace, entity_type, name_norm)
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS axon_relations (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
                    source_id UUID NOT NULL REFERENCES axon_entities(id) ON DELETE CASCADE,
                    relation_type VARCHAR(64) NOT NULL,
                    target_id UUID NOT NULL REFERENCES axon_entities(id) ON DELETE CASCADE,
                    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
                    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
                    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    valid_until TIMESTAMPTZ,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    CONSTRAINT uq_axon_relation UNIQUE (namespace, source_id, relation_type, target_id)
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS axon_episodes (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
                    task_name VARCHAR(128) NOT NULL,
                    outcome VARCHAR(24) NOT NULL DEFAULT 'success',
                    what_happened TEXT NOT NULL,
                    lessons_learned TEXT,
                    hard_facts JSONB NOT NULL DEFAULT '{}'::jsonb,
                    entity_names TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
                    source_lang VARCHAR(16),
                    consolidated BOOLEAN NOT NULL DEFAULT FALSE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS axon_memories (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
                    category VARCHAR(48) NOT NULL DEFAULT 'fact',
                    fact_key VARCHAR(255),
                    content TEXT NOT NULL,
                    hard_facts JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('simple', axon_unaccent(coalesce(content, '')))
                    ) STORED,
                    embedding vector({dim}),
                    embed_model VARCHAR(128),
                    source_lang VARCHAR(16),
                    superseded_by UUID REFERENCES axon_memories(id) ON DELETE SET NULL,
                    active BOOLEAN NOT NULL DEFAULT TRUE,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS axon_foresights (
                    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
                    subject VARCHAR(255) NOT NULL,
                    reminder TEXT NOT NULL,
                    hard_facts JSONB NOT NULL DEFAULT '{}'::jsonb,
                    source_lang VARCHAR(16),
                    due_at TIMESTAMPTZ,
                    expires_at TIMESTAMPTZ NOT NULL,
                    status VARCHAR(24) NOT NULL DEFAULT 'active',
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )

            cur.execute("CREATE INDEX IF NOT EXISTS idx_axon_entities_ns_norm ON axon_entities (namespace, name_norm);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_axon_entities_trgm ON axon_entities USING gin (name_norm gin_trgm_ops);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_axon_relations_src ON axon_relations (namespace, source_id);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_axon_relations_tgt ON axon_relations (namespace, target_id);")
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_axon_episodes_unconsolidated ON axon_episodes (namespace, consolidated) WHERE consolidated = FALSE;"
            )
            cur.execute(
                "CREATE INDEX IF NOT EXISTS idx_axon_memories_active_key ON axon_memories (namespace, fact_key) WHERE active = TRUE;"
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_axon_memories_tsv ON axon_memories USING gin (tsv);")

            cur.execute(
                """
                SELECT format_type(a.atttypid, a.atttypmod) AS vec_type
                FROM pg_attribute a
                JOIN pg_class c ON c.oid = a.attrelid
                JOIN pg_namespace n ON n.oid = c.relnamespace
                WHERE n.nspname = current_schema()
                  AND c.relname = 'axon_memories'
                  AND a.attname = 'embedding'
                  AND NOT a.attisdropped;
                """
            )
            col = cur.fetchone()
            expected_type = f"vector({dim})"
            if col and col["vec_type"] != expected_type:
                cur.execute(
                    "SELECT EXISTS (SELECT 1 FROM axon_memories WHERE embedding IS NOT NULL LIMIT 1) AS has_vecs;"
                )
                if cur.fetchone()["has_vecs"]:
                    raise RuntimeError(
                        f"Embedding dimension mismatch: axon_memories.embedding is {col['vec_type']} "
                        f"with stored vectors, but configured model '{self.embedder.model_name}' "
                        f"requires {expected_type}."
                    )
                cur.execute("DROP INDEX IF EXISTS idx_axon_memories_hnsw;")
                cur.execute(f"ALTER TABLE axon_memories ALTER COLUMN embedding TYPE {expected_type};")

            if dim <= 2000:
                cur.execute(
                    "CREATE INDEX IF NOT EXISTS idx_axon_memories_hnsw ON axon_memories USING hnsw (embedding vector_cosine_ops);"
                )

    def upsert_entity(self, item: EntityInput) -> dict[str, Any]:
        name = self._clean_text(item.name)
        name_norm = self._lookup_key(name)
        aliases = [self._clean_text(a) for a in item.aliases if self._clean_text(a)]
        summary = self._clean_text(item.summary) if item.summary else None

        with self._cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO axon_entities (namespace, entity_type, name, name_norm, aliases, summary, properties, source_lang)
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                ON CONFLICT (namespace, entity_type, name_norm) DO UPDATE SET
                    name = EXCLUDED.name,
                    aliases = ARRAY(
                        SELECT DISTINCT elem FROM unnest(axon_entities.aliases || EXCLUDED.aliases) AS elem
                    ),
                    summary = COALESCE(EXCLUDED.summary, axon_entities.summary),
                    properties = axon_entities.properties || EXCLUDED.properties,
                    source_lang = COALESCE(EXCLUDED.source_lang, axon_entities.source_lang),
                    updated_at = NOW()
                RETURNING id::text;
                """,
                (
                    item.namespace,
                    item.entity_type,
                    name,
                    name_norm,
                    aliases,
                    summary,
                    json.dumps(item.properties, ensure_ascii=False),
                    item.source_lang,
                ),
            )
            row = cur.fetchone()
        return {"id": row["id"], "namespace": item.namespace, "entity_type": item.entity_type, "name": name}

    def _ensure_entity_id(self, namespace: str, name: str, entity_type: str) -> str:
        res = self.upsert_entity(EntityInput(namespace=namespace, entity_type=entity_type, name=name))
        return res["id"]

    def add_relation(self, item: RelationInput) -> dict[str, Any]:
        src_id = self._ensure_entity_id(item.namespace, item.source_name, item.source_type)
        tgt_id = self._ensure_entity_id(item.namespace, item.target_name, item.target_type)
        with self._cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO axon_relations (namespace, source_id, relation_type, target_id, weight, properties)
                VALUES (%s, %s::uuid, %s, %s::uuid, %s, %s::jsonb)
                ON CONFLICT (namespace, source_id, relation_type, target_id) DO UPDATE SET
                    weight = EXCLUDED.weight,
                    properties = axon_relations.properties || EXCLUDED.properties,
                    valid_until = NULL;
                """,
                (
                    item.namespace,
                    src_id,
                    item.relation_type,
                    tgt_id,
                    item.weight,
                    json.dumps(item.properties, ensure_ascii=False),
                ),
            )
        return {
            "source": item.source_name,
            "relation_type": item.relation_type,
            "target": item.target_name,
            "namespace": item.namespace,
        }

    def record_episode(self, item: EpisodeInput) -> dict[str, Any]:
        entities = [self._clean_text(e) for e in item.entity_names if self._clean_text(e)]
        with self._cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO axon_episodes (
                    namespace, task_name, outcome, what_happened, lessons_learned,
                    hard_facts, entity_names, source_lang
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s, %s)
                RETURNING id::text;
                """,
                (
                    item.namespace,
                    self._clean_text(item.task_name),
                    item.outcome,
                    self._clean_text(item.what_happened),
                    self._clean_text(item.lessons_learned) if item.lessons_learned else None,
                    json.dumps(item.hard_facts, ensure_ascii=False),
                    entities,
                    item.source_lang,
                ),
            )
            row = cur.fetchone()
        return {"id": row["id"], "task_name": item.task_name, "outcome": item.outcome, "namespace": item.namespace}

    def upsert_fact(self, item: FactInput) -> dict[str, Any]:
        content = self._clean_text(item.content)
        fact_key = self._lookup_key(item.fact_key) if item.fact_key else None
        vectors = self.embedder.embed_documents([content])
        vec_lit = _vector_literal(vectors[0] if vectors else None)

        with self._cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO axon_memories (
                    namespace, category, fact_key, content, hard_facts,
                    embedding, embed_model, source_lang, active
                )
                VALUES (%s, %s, %s, %s, %s::jsonb, %s::vector, %s, %s, TRUE)
                RETURNING id::text;
                """,
                (
                    item.namespace,
                    item.category,
                    fact_key,
                    content,
                    json.dumps(item.hard_facts, ensure_ascii=False),
                    vec_lit,
                    self.embedder.model_name if vec_lit else None,
                    item.source_lang,
                ),
            )
            mem_id = cur.fetchone()["id"]

            superseded_count = 0
            if item.supersede_existing and fact_key:
                cur.execute(
                    """
                    UPDATE axon_memories
                    SET active = FALSE, superseded_by = %s::uuid, updated_at = NOW()
                    WHERE namespace = %s AND fact_key = %s AND id <> %s::uuid AND active = TRUE;
                    """,
                    (mem_id, item.namespace, fact_key, mem_id),
                )
                superseded_count = cur.rowcount

        return {
            "id": mem_id,
            "namespace": item.namespace,
            "fact_key": fact_key,
            "superseded_previous": superseded_count,
        }

    def add_foresight(self, item: ForesightInput) -> dict[str, Any]:
        due_dt = _utc_dt(item.due_at) if item.due_at else None
        exp_dt = _utc_dt(item.expires_at)
        with self._cursor(commit=True) as cur:
            cur.execute(
                """
                INSERT INTO axon_foresights (
                    namespace, subject, reminder, hard_facts, source_lang, due_at, expires_at, status
                )
                VALUES (%s, %s, %s, %s::jsonb, %s, %s, %s, 'active')
                RETURNING id::text;
                """,
                (
                    item.namespace,
                    self._clean_text(item.subject),
                    self._clean_text(item.reminder),
                    json.dumps(item.hard_facts, ensure_ascii=False),
                    item.source_lang,
                    due_dt,
                    exp_dt,
                ),
            )
            fs_id = cur.fetchone()["id"]
        return {"id": fs_id, "subject": item.subject, "status": "active", "expires_at": exp_dt.isoformat()}

    def complete_foresight(self, foresight_id: str, namespace: str = "default") -> dict[str, Any]:
        with self._cursor(commit=True) as cur:
            cur.execute(
                "UPDATE axon_foresights SET status = 'done' WHERE id = %s::uuid AND namespace = %s;",
                (foresight_id, namespace),
            )
            updated = cur.rowcount > 0
        return {"id": foresight_id, "updated": updated, "status": "done"}

    def expire_stale_foresights(self, namespace: str = "default") -> int:
        with self._cursor(commit=True) as cur:
            cur.execute(
                """
                UPDATE axon_foresights
                SET status = 'expired'
                WHERE namespace = %s AND status = 'active' AND expires_at <= NOW();
                """,
                (namespace,),
            )
            return cur.rowcount

    def recall(
        self,
        query: str,
        namespace: str = "default",
        limit: int = 5,
        max_tokens: int = 800,
        include_episodes: bool = True,
    ) -> RecallResult:
        expired_count = self.expire_stale_foresights(namespace)
        clean_q = self._clean_text(query)
        q_tokens = self._search_tokens(clean_q)
        q_norm = self._lookup_key(clean_q)
        ts_expr = _safe_tsquery_expr(q_tokens)

        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        memories: list[RecallItem] = []
        episodes: list[dict[str, Any]] = []
        foresights: list[dict[str, Any]] = []

        q_vec = self.embedder.embed_query(clean_q)
        vec_lit = _vector_literal(q_vec)

        with self._cursor(commit=True) as cur:
            # 1. Knowledge Graph entities & 1-hop relations
            cur.execute(
                """
                SELECT id::text, entity_type, name, name_norm, aliases, summary, properties
                FROM axon_entities
                WHERE namespace = %s;
                """,
                (namespace,),
            )
            matched_ent_ids: list[str] = []
            for row in cur.fetchall():
                haystack = " ".join([row["name_norm"], row["name"], *(row["aliases"] or []), row["summary"] or ""]).casefold()
                folded_haystack = fold_accents_preserving_scripts(haystack)
                if (q_norm and q_norm in row["name_norm"]) or any(tok in folded_haystack for tok in q_tokens):
                    matched_ent_ids.append(row["id"])
                    entities.append({
                        "id": row["id"],
                        "entity_type": row["entity_type"],
                        "name": row["name"],
                        "summary": row["summary"],
                        "properties": row["properties"] or {},
                    })
            entities = entities[:limit]

            if matched_ent_ids:
                cur.execute(
                    """
                    SELECT s.name AS source, r.relation_type, t.name AS target
                    FROM axon_relations r
                    JOIN axon_entities s ON r.source_id = s.id
                    JOIN axon_entities t ON r.target_id = t.id
                    WHERE r.namespace = %s AND r.valid_until IS NULL
                      AND (r.source_id = ANY(%s::uuid[]) OR r.target_id = ANY(%s::uuid[]))
                    LIMIT %s;
                    """,
                    (namespace, matched_ent_ids, matched_ent_ids, limit * 2),
                )
                relations = [dict(r) for r in cur.fetchall()]

            # 2. Hybrid Memory Ranking (pg tsvector + pg_trgm/token substring + pgvector HNSW)
            fts_ranked: list[tuple[str, float]] = []
            if ts_expr:
                cur.execute(
                    """
                    SELECT id::text, ts_rank_cd(tsv, to_tsquery('simple', %s)) AS rank_score
                    FROM axon_memories
                    WHERE namespace = %s AND active = TRUE AND superseded_by IS NULL
                      AND tsv @@ to_tsquery('simple', %s)
                    ORDER BY rank_score DESC
                    LIMIT %s;
                    """,
                    (ts_expr, namespace, ts_expr, limit * 3),
                )
                fts_ranked = [(r["id"], float(r["rank_score"])) for r in cur.fetchall()]

            kw_ranked: list[tuple[str, float]] = []
            if q_tokens or q_norm:
                like_patterns = [f"%{tok}%" for tok in q_tokens[:8]]
                search_probe = q_norm or clean_q.casefold()
                if like_patterns:
                    cur.execute(
                        """
                        SELECT id::text,
                               similarity(axon_unaccent(lower(content)), %s) AS trgm_score
                        FROM axon_memories
                        WHERE namespace = %s AND active = TRUE AND superseded_by IS NULL
                          AND (
                              similarity(axon_unaccent(lower(content)), %s) > 0.10
                              OR axon_unaccent(lower(content)) LIKE ANY(%s)
                          )
                        ORDER BY trgm_score DESC
                        LIMIT %s;
                        """,
                        (search_probe, namespace, search_probe, like_patterns, limit * 3),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id::text,
                               similarity(axon_unaccent(lower(content)), %s) AS trgm_score
                        FROM axon_memories
                        WHERE namespace = %s AND active = TRUE AND superseded_by IS NULL
                          AND similarity(axon_unaccent(lower(content)), %s) > 0.10
                        ORDER BY trgm_score DESC
                        LIMIT %s;
                        """,
                        (search_probe, namespace, search_probe, limit * 3),
                    )
                kw_ranked = [(r["id"], float(r["trgm_score"])) for r in cur.fetchall()]

            vec_ranked: list[tuple[str, float]] = []
            if vec_lit:
                cur.execute(
                    """
                    SELECT id::text, 1.0 - (embedding <=> %s::vector) AS cosine_sim
                    FROM axon_memories
                    WHERE namespace = %s AND active = TRUE AND superseded_by IS NULL AND embedding IS NOT NULL
                    ORDER BY embedding <=> %s::vector ASC
                    LIMIT %s;
                    """,
                    (vec_lit, namespace, vec_lit, limit * 3),
                )
                vec_ranked = [(r["id"], float(r["cosine_sim"])) for r in cur.fetchall() if float(r["cosine_sim"]) > 0.15]

            fused = reciprocal_rank_fusion([fts_ranked, kw_ranked, vec_ranked])
            top_ids = [mid for mid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:limit]]

            if top_ids:
                cur.execute(
                    """
                    SELECT id::text, category, fact_key, content, hard_facts
                    FROM axon_memories
                    WHERE namespace = %s AND id = ANY(%s::uuid[]);
                    """,
                    (namespace, top_ids),
                )
                by_id = {r["id"]: r for r in cur.fetchall()}
                for mid in top_ids:
                    row = by_id.get(mid)
                    if row:
                        memories.append(
                            RecallItem(
                                id=row["id"],
                                layer="memory",
                                score=round(fused[mid], 5),
                                content=row["content"],
                                hard_facts=row["hard_facts"] or {},
                                metadata={"category": row["category"], "fact_key": row["fact_key"]},
                            )
                        )
                cur.execute(
                    "UPDATE axon_memories SET hit_count = hit_count + 1 WHERE namespace = %s AND id = ANY(%s::uuid[]);",
                    (namespace, top_ids),
                )

            # 3. Episodic lessons matching query tokens
            if include_episodes and q_tokens:
                cur.execute(
                    """
                    SELECT id::text, task_name, outcome, what_happened, lessons_learned, hard_facts
                    FROM axon_episodes
                    WHERE namespace = %s
                    ORDER BY created_at DESC
                    LIMIT %s;
                    """,
                    (namespace, limit * 5),
                )
                for ep in cur.fetchall():
                    text_blob = fold_accents_preserving_scripts(
                        f"{ep['task_name']} {ep['what_happened']} {ep['lessons_learned'] or ''} {json.dumps(ep['hard_facts'], ensure_ascii=False)}"
                    ).casefold()
                    if any(tok in text_blob for tok in q_tokens):
                        episodes.append({
                            "id": ep["id"],
                            "task_name": ep["task_name"],
                            "outcome": ep["outcome"],
                            "what_happened": ep["what_happened"],
                            "lessons_learned": ep["lessons_learned"],
                            "hard_facts": ep["hard_facts"] or {},
                        })
                        if len(episodes) >= limit:
                            break

            # 4. Active foresights
            cur.execute(
                """
                SELECT id::text, subject, reminder, due_at, expires_at, hard_facts
                FROM axon_foresights
                WHERE namespace = %s AND status = 'active'
                ORDER BY COALESCE(due_at, expires_at) ASC
                LIMIT %s;
                """,
                (namespace, limit),
            )
            foresights = [
                {
                    "id": r["id"],
                    "subject": r["subject"],
                    "reminder": r["reminder"],
                    "due_at": _iso_or_none(r["due_at"]),
                    "expires_at": _iso_or_none(r["expires_at"]),
                    "hard_facts": r["hard_facts"] or {},
                }
                for r in cur.fetchall()
            ]

        result = RecallResult(
            query=clean_q,
            namespace=namespace,
            entities=entities,
            relations=relations,
            memories=memories,
            episodes=episodes,
            foresights=foresights,
            expired_archived=expired_count,
        )
        result.prompt_context = format_recall_context(result, max_tokens=max_tokens)
        result.estimated_tokens = estimate_tokens(result.prompt_context)
        return result

    def fetch_unconsolidated_episodes(self, namespace: str = "default", limit: int = 50) -> list[dict[str, Any]]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT id::text, namespace, task_name, outcome, what_happened, lessons_learned,
                       hard_facts, entity_names, source_lang
                FROM axon_episodes
                WHERE namespace = %s AND consolidated = FALSE
                ORDER BY created_at ASC
                LIMIT %s;
                """,
                (namespace, limit),
            )
            return [
                {
                    "id": r["id"],
                    "namespace": r["namespace"],
                    "task_name": r["task_name"],
                    "outcome": r["outcome"],
                    "what_happened": r["what_happened"],
                    "lessons_learned": r["lessons_learned"],
                    "hard_facts": r["hard_facts"] or {},
                    "entity_names": r["entity_names"] or [],
                    "source_lang": r["source_lang"],
                }
                for r in cur.fetchall()
            ]

    def mark_episodes_consolidated(self, episode_ids: list[str], namespace: str = "default") -> int:
        if not episode_ids:
            return 0
        with self._cursor(commit=True) as cur:
            cur.execute(
                "UPDATE axon_episodes SET consolidated = TRUE WHERE namespace = %s AND id = ANY(%s::uuid[]);",
                (namespace, episode_ids),
            )
            return cur.rowcount

    def stats(self, namespace: str = "default") -> MemoryStats:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM axon_entities WHERE namespace = %s) AS entities,
                    (SELECT COUNT(*) FROM axon_relations WHERE namespace = %s) AS relations,
                    (SELECT COUNT(*) FROM axon_episodes WHERE namespace = %s) AS episodes,
                    (SELECT COUNT(*) FROM axon_episodes WHERE namespace = %s AND consolidated = FALSE) AS unconsolidated_episodes,
                    (SELECT COUNT(*) FROM axon_memories WHERE namespace = %s AND active = TRUE) AS active_memories,
                    (SELECT COUNT(*) FROM axon_memories WHERE namespace = %s AND active = FALSE) AS superseded_memories,
                    (SELECT COUNT(*) FROM axon_foresights WHERE namespace = %s AND status = 'active') AS active_foresights,
                    (SELECT COUNT(*) FROM axon_foresights WHERE namespace = %s AND status = 'expired') AS expired_foresights;
                """,
                (namespace,) * 8,
            )
            r = cur.fetchone()
        return MemoryStats(
            backend=self.backend_name,
            namespace=namespace,
            entities=int(r["entities"]),
            relations=int(r["relations"]),
            episodes=int(r["episodes"]),
            unconsolidated_episodes=int(r["unconsolidated_episodes"]),
            active_memories=int(r["active_memories"]),
            superseded_memories=int(r["superseded_memories"]),
            active_foresights=int(r["active_foresights"]),
            expired_foresights=int(r["expired_foresights"]),
        )

    def healthcheck(self) -> tuple[bool, str]:
        try:
            with self._cursor() as cur:
                cur.execute("SELECT extname FROM pg_extension WHERE extname IN ('vector', 'pg_trgm', 'unaccent');")
                exts = {r["extname"] for r in cur.fetchall()}
                missing = {"vector", "pg_trgm", "unaccent"} - exts
                if missing:
                    return False, f"missing extensions: {sorted(missing)}"

                cur.execute(
                    """
                    SELECT format_type(a.atttypid, a.atttypmod) AS vec_type
                    FROM pg_attribute a
                    JOIN pg_class c ON c.oid = a.attrelid
                    JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE n.nspname = current_schema()
                      AND c.relname = 'axon_memories'
                      AND a.attname = 'embedding'
                      AND NOT a.attisdropped;
                    """
                )
                col = cur.fetchone()
            expected_type = f"vector({int(self.embedder.dim)})"
            if col and col["vec_type"] != expected_type:
                return False, f"dimension mismatch: DB has {col['vec_type']}, config expects {expected_type}"
            return True, f"ok (extensions={sorted(exts)}, embedding={expected_type})"
        except Exception as exc:
            return False, str(exc)

    def list_foresights(
        self,
        namespace: str = "default",
        status: str = "active",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self.expire_stale_foresights(namespace=namespace)
        with self._cursor() as cur:
            if status == "all":
                cur.execute(
                    """
                    SELECT id::text, subject, reminder, hard_facts, source_lang, due_at, expires_at, status
                    FROM axon_foresights
                    WHERE namespace = %s
                    ORDER BY COALESCE(due_at, expires_at) ASC
                    LIMIT %s;
                    """,
                    (namespace, limit),
                )
            else:
                cur.execute(
                    """
                    SELECT id::text, subject, reminder, hard_facts, source_lang, due_at, expires_at, status
                    FROM axon_foresights
                    WHERE namespace = %s AND status = %s
                    ORDER BY COALESCE(due_at, expires_at) ASC
                    LIMIT %s;
                    """,
                    (namespace, status, limit),
                )
            rows = cur.fetchall()
        return [
            {
                "id": r["id"],
                "subject": r["subject"],
                "reminder": r["reminder"],
                "hard_facts": r["hard_facts"] or {},
                "source_lang": r["source_lang"],
                "due_at": _iso_or_none(r["due_at"]),
                "expires_at": _iso_or_none(r["expires_at"]),
                "status": r["status"],
            }
            for r in rows
        ]

    def export_namespace(self, namespace: str = "default") -> dict[str, Any]:
        with self._cursor() as cur:
            cur.execute(
                """
                SELECT entity_type, name, aliases, summary, properties, source_lang
                FROM axon_entities
                WHERE namespace = %s
                ORDER BY created_at ASC;
                """,
                (namespace,),
            )
            entities = [
                {
                    "entity_type": r["entity_type"],
                    "name": r["name"],
                    "aliases": list(r["aliases"] or []),
                    "summary": r["summary"],
                    "properties": r["properties"] or {},
                    "source_lang": r["source_lang"],
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT r.relation_type, r.weight, r.properties,
                       es.name AS source_name, es.entity_type AS source_type,
                       et.name AS target_name, et.entity_type AS target_type
                FROM axon_relations r
                JOIN axon_entities es ON es.id = r.source_id
                JOIN axon_entities et ON et.id = r.target_id
                WHERE r.namespace = %s
                ORDER BY r.created_at ASC;
                """,
                (namespace,),
            )
            relations = [
                {
                    "source_name": r["source_name"],
                    "source_type": r["source_type"],
                    "relation_type": r["relation_type"],
                    "target_name": r["target_name"],
                    "target_type": r["target_type"],
                    "weight": float(r["weight"]),
                    "properties": r["properties"] or {},
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT task_name, outcome, what_happened, lessons_learned, hard_facts,
                       entity_names, source_lang, consolidated
                FROM axon_episodes
                WHERE namespace = %s
                ORDER BY created_at ASC;
                """,
                (namespace,),
            )
            episodes = [
                {
                    "task_name": r["task_name"],
                    "outcome": r["outcome"],
                    "what_happened": r["what_happened"],
                    "lessons_learned": r["lessons_learned"],
                    "hard_facts": r["hard_facts"] or {},
                    "entity_names": list(r["entity_names"] or []),
                    "source_lang": r["source_lang"],
                    "consolidated": bool(r["consolidated"]),
                }
                for r in cur.fetchall()
            ]
            cur.execute(
                """
                SELECT category, fact_key, content, hard_facts, source_lang
                FROM axon_memories
                WHERE namespace = %s AND active = TRUE AND superseded_by IS NULL
                ORDER BY created_at ASC;
                """,
                (namespace,),
            )
            facts = [
                {
                    "category": r["category"],
                    "fact_key": r["fact_key"],
                    "content": r["content"],
                    "hard_facts": r["hard_facts"] or {},
                    "source_lang": r["source_lang"],
                }
                for r in cur.fetchall()
            ]

        foresights = self.list_foresights(namespace=namespace, status="all", limit=1000)
        return {
            "schema_version": 1,
            "namespace": namespace,
            "exported_at": self._utc_iso_now(),
            "entities": entities,
            "relations": relations,
            "episodes": episodes,
            "facts": facts,
            "foresights": foresights,
        }

    def reset_namespace(self, namespace: str) -> dict[str, int]:
        deleted: dict[str, int] = {}
        with self._cursor(commit=True) as cur:
            for key, tbl in (
                ("relations", "axon_relations"),
                ("entities", "axon_entities"),
                ("episodes", "axon_episodes"),
                ("memories", "axon_memories"),
                ("foresights", "axon_foresights"),
            ):
                cur.execute(f"DELETE FROM {tbl} WHERE namespace = %s;", (namespace,))
                deleted[key] = cur.rowcount
        return deleted

    def close(self) -> None:
        self._pool.closeall()
