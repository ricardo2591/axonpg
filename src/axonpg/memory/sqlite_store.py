"""Zero-Docker SQLite + FTS5 + Python cosine vector backend (--lite mode and fast CI)."""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
from axonpg.providers.embeddings import EmbeddingProvider, cosine_similarity


def _utc_iso(dt: datetime | None = None) -> str:
    target = dt or datetime.now(timezone.utc)
    if target.tzinfo is None:
        target = target.replace(tzinfo=timezone.utc)
    return target.astimezone(timezone.utc).isoformat()


class SQLiteMemoryStore(MemoryBackend):
    """Lightweight single-file SQLite WAL + FTS5 implementation with full API parity."""

    backend_name = "sqlite-lite"

    def __init__(self, db_path: str | Path, embedder: EmbeddingProvider):
        self.db_path = Path(db_path).expanduser()
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.embedder = embedder
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON;")
        if str(self.db_path) != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL;")
        self.ensure_schema()

    @contextmanager
    def _tx(self):
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    def ensure_schema(self) -> None:
        with self._tx() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS axon_entities (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    entity_type TEXT NOT NULL,
                    name TEXT NOT NULL,
                    name_norm TEXT NOT NULL,
                    aliases_json TEXT NOT NULL DEFAULT '[]',
                    summary TEXT,
                    properties_json TEXT NOT NULL DEFAULT '{}',
                    source_lang TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    UNIQUE (namespace, entity_type, name_norm)
                );

                CREATE TABLE IF NOT EXISTS axon_relations (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    source_id TEXT NOT NULL REFERENCES axon_entities(id) ON DELETE CASCADE,
                    relation_type TEXT NOT NULL,
                    target_id TEXT NOT NULL REFERENCES axon_entities(id) ON DELETE CASCADE,
                    weight REAL NOT NULL DEFAULT 1.0,
                    properties_json TEXT NOT NULL DEFAULT '{}',
                    valid_from TEXT NOT NULL,
                    valid_until TEXT,
                    created_at TEXT NOT NULL,
                    UNIQUE (namespace, source_id, relation_type, target_id)
                );

                CREATE TABLE IF NOT EXISTS axon_episodes (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    task_name TEXT NOT NULL,
                    outcome TEXT NOT NULL DEFAULT 'success',
                    what_happened TEXT NOT NULL,
                    lessons_learned TEXT,
                    hard_facts_json TEXT NOT NULL DEFAULT '{}',
                    entity_names_json TEXT NOT NULL DEFAULT '[]',
                    source_lang TEXT,
                    consolidated INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS axon_memories (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    category TEXT NOT NULL DEFAULT 'fact',
                    fact_key TEXT,
                    content TEXT NOT NULL,
                    folded_content TEXT NOT NULL,
                    hard_facts_json TEXT NOT NULL DEFAULT '{}',
                    embedding_json TEXT,
                    embed_model TEXT,
                    source_lang TEXT,
                    superseded_by TEXT,
                    active INTEGER NOT NULL DEFAULT 1,
                    hit_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS axon_memories_fts USING fts5(
                    memory_id UNINDEXED,
                    namespace UNINDEXED,
                    folded_content,
                    tokenize = 'unicode61 remove_diacritics 2'
                );

                CREATE TABLE IF NOT EXISTS axon_foresights (
                    id TEXT PRIMARY KEY,
                    namespace TEXT NOT NULL DEFAULT 'default',
                    subject TEXT NOT NULL,
                    reminder TEXT NOT NULL,
                    hard_facts_json TEXT NOT NULL DEFAULT '{}',
                    source_lang TEXT,
                    due_at TEXT,
                    expires_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TEXT NOT NULL
                );
                """
            )

    def upsert_entity(self, item: EntityInput) -> dict[str, Any]:
        now = _utc_iso()
        name = self._clean_text(item.name)
        name_norm = self._lookup_key(name)
        aliases = [self._clean_text(a) for a in item.aliases if self._clean_text(a)]
        summary = self._clean_text(item.summary) if item.summary else None

        with self._tx() as conn:
            existing = conn.execute(
                "SELECT id, aliases_json, properties_json FROM axon_entities WHERE namespace=? AND entity_type=? AND name_norm=?",
                (item.namespace, item.entity_type, name_norm),
            ).fetchone()

            if existing:
                old_aliases = json.loads(existing["aliases_json"])
                merged_aliases = list(dict.fromkeys([*old_aliases, *aliases]))
                old_props = json.loads(existing["properties_json"])
                merged_props = {**old_props, **item.properties}
                conn.execute(
                    """
                    UPDATE axon_entities
                    SET name=?, aliases_json=?, summary=COALESCE(?, summary),
                        properties_json=?, source_lang=COALESCE(?, source_lang), updated_at=?
                    WHERE id=?
                    """,
                    (
                        name,
                        json.dumps(merged_aliases, ensure_ascii=False),
                        summary,
                        json.dumps(merged_props, ensure_ascii=False),
                        item.source_lang,
                        now,
                        existing["id"],
                    ),
                )
                ent_id = existing["id"]
            else:
                ent_id = str(uuid.uuid4())
                conn.execute(
                    """
                    INSERT INTO axon_entities (id, namespace, entity_type, name, name_norm, aliases_json, summary, properties_json, source_lang, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        ent_id,
                        item.namespace,
                        item.entity_type,
                        name,
                        name_norm,
                        json.dumps(aliases, ensure_ascii=False),
                        summary,
                        json.dumps(item.properties, ensure_ascii=False),
                        item.source_lang,
                        now,
                        now,
                    ),
                )

        return {"id": ent_id, "namespace": item.namespace, "entity_type": item.entity_type, "name": name}

    def _ensure_entity_id(self, namespace: str, name: str, entity_type: str) -> str:
        res = self.upsert_entity(EntityInput(namespace=namespace, entity_type=entity_type, name=name))
        return res["id"]

    def add_relation(self, item: RelationInput) -> dict[str, Any]:
        src_id = self._ensure_entity_id(item.namespace, item.source_name, item.source_type)
        tgt_id = self._ensure_entity_id(item.namespace, item.target_name, item.target_type)
        now = _utc_iso()
        rel_id = str(uuid.uuid4())

        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO axon_relations (id, namespace, source_id, relation_type, target_id, weight, properties_json, valid_from, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(namespace, source_id, relation_type, target_id) DO UPDATE SET
                    weight=excluded.weight,
                    properties_json=excluded.properties_json,
                    valid_until=NULL
                """,
                (
                    rel_id,
                    item.namespace,
                    src_id,
                    item.relation_type,
                    tgt_id,
                    item.weight,
                    json.dumps(item.properties, ensure_ascii=False),
                    now,
                    now,
                ),
            )
        return {
            "source": item.source_name,
            "relation_type": item.relation_type,
            "target": item.target_name,
            "namespace": item.namespace,
        }

    def record_episode(self, item: EpisodeInput) -> dict[str, Any]:
        ep_id = str(uuid.uuid4())
        now = _utc_iso()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO axon_episodes (id, namespace, task_name, outcome, what_happened, lessons_learned, hard_facts_json, entity_names_json, source_lang, consolidated, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0, ?)
                """,
                (
                    ep_id,
                    item.namespace,
                    self._clean_text(item.task_name),
                    item.outcome,
                    self._clean_text(item.what_happened),
                    self._clean_text(item.lessons_learned) if item.lessons_learned else None,
                    json.dumps(item.hard_facts, ensure_ascii=False),
                    json.dumps([self._clean_text(e) for e in item.entity_names], ensure_ascii=False),
                    item.source_lang,
                    now,
                ),
            )
        return {"id": ep_id, "task_name": item.task_name, "outcome": item.outcome, "namespace": item.namespace}

    def upsert_fact(self, item: FactInput) -> dict[str, Any]:
        mem_id = str(uuid.uuid4())
        now = _utc_iso()
        content = self._clean_text(item.content)
        folded = fold_accents_preserving_scripts(content)
        fact_key = self._lookup_key(item.fact_key) if item.fact_key else None

        vectors = self.embedder.embed_documents([content])
        vec = vectors[0] if vectors else []
        superseded_count = 0

        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO axon_memories (id, namespace, category, fact_key, content, folded_content, hard_facts_json, embedding_json, embed_model, source_lang, active, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)
                """,
                (
                    mem_id,
                    item.namespace,
                    item.category,
                    fact_key,
                    content,
                    folded,
                    json.dumps(item.hard_facts, ensure_ascii=False),
                    json.dumps(vec) if vec else None,
                    self.embedder.model_name if vec else None,
                    item.source_lang,
                    now,
                    now,
                ),
            )
            conn.execute(
                "INSERT INTO axon_memories_fts (memory_id, namespace, folded_content) VALUES (?, ?, ?)",
                (mem_id, item.namespace, folded),
            )

            if item.supersede_existing and fact_key:
                cur = conn.execute(
                    """
                    UPDATE axon_memories
                    SET active=0, superseded_by=?, updated_at=?
                    WHERE namespace=? AND fact_key=? AND id!=? AND active=1
                    """,
                    (mem_id, now, item.namespace, fact_key, mem_id),
                )
                superseded_count = cur.rowcount

        return {
            "id": mem_id,
            "namespace": item.namespace,
            "fact_key": fact_key,
            "superseded_previous": superseded_count,
        }

    def add_foresight(self, item: ForesightInput) -> dict[str, Any]:
        fs_id = str(uuid.uuid4())
        now = _utc_iso()
        with self._tx() as conn:
            conn.execute(
                """
                INSERT INTO axon_foresights (id, namespace, subject, reminder, hard_facts_json, source_lang, due_at, expires_at, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'active', ?)
                """,
                (
                    fs_id,
                    item.namespace,
                    self._clean_text(item.subject),
                    self._clean_text(item.reminder),
                    json.dumps(item.hard_facts, ensure_ascii=False),
                    item.source_lang,
                    _utc_iso(item.due_at) if item.due_at else None,
                    _utc_iso(item.expires_at),
                    now,
                ),
            )
        return {"id": fs_id, "subject": item.subject, "status": "active", "expires_at": _utc_iso(item.expires_at)}

    def complete_foresight(self, foresight_id: str, namespace: str = "default") -> dict[str, Any]:
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE axon_foresights SET status='done' WHERE id=? AND namespace=?",
                (foresight_id, namespace),
            )
        return {"id": foresight_id, "updated": cur.rowcount > 0, "status": "done"}

    def expire_stale_foresights(self, namespace: str = "default") -> int:
        now = _utc_iso()
        with self._tx() as conn:
            cur = conn.execute(
                "UPDATE axon_foresights SET status='expired' WHERE namespace=? AND status='active' AND expires_at <= ?",
                (namespace, now),
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

        # 1. Knowledge Graph entities matching query tokens or normalized query
        entities: list[dict[str, Any]] = []
        relations: list[dict[str, Any]] = []
        ent_rows = self._conn.execute(
            "SELECT * FROM axon_entities WHERE namespace=?",
            (namespace,),
        ).fetchall()

        matched_ent_ids: list[str] = []
        for row in ent_rows:
            aliases = json.loads(row["aliases_json"])
            haystack = " ".join([row["name_norm"], row["name"], *aliases, row["summary"] or ""]).casefold()
            folded_haystack = fold_accents_preserving_scripts(haystack)
            if (q_norm and q_norm in row["name_norm"]) or any(tok in folded_haystack for tok in q_tokens):
                matched_ent_ids.append(row["id"])
                entities.append({
                    "id": row["id"],
                    "entity_type": row["entity_type"],
                    "name": row["name"],
                    "summary": row["summary"],
                    "properties": json.loads(row["properties_json"]),
                })
        entities = entities[:limit]

        if matched_ent_ids:
            placeholders = ",".join("?" for _ in matched_ent_ids)
            rel_rows = self._conn.execute(
                f"""
                SELECT s.name AS source, r.relation_type, t.name AS target
                FROM axon_relations r
                JOIN axon_entities s ON r.source_id = s.id
                JOIN axon_entities t ON r.target_id = t.id
                WHERE r.namespace=? AND r.valid_until IS NULL
                  AND (r.source_id IN ({placeholders}) OR r.target_id IN ({placeholders}))
                LIMIT ?
                """,
                (namespace, *matched_ent_ids, *matched_ent_ids, limit * 2),
            ).fetchall()
            relations = [dict(r) for r in rel_rows]

        # 2. Hybrid ranking on active memories (FTS5 + Substring fallback + Cosine similarity)
        active_mems = self._conn.execute(
            "SELECT * FROM axon_memories WHERE namespace=? AND active=1 AND superseded_by IS NULL",
            (namespace,),
        ).fetchall()
        mem_by_id = {row["id"]: row for row in active_mems}

        fts_ranked: list[tuple[str, float]] = []
        if q_tokens:
            safe_terms = [f'"{tok}"' for tok in q_tokens[:8] if tok]
            if safe_terms:
                fts_query = " OR ".join(safe_terms)
                try:
                    fts_rows = self._conn.execute(
                        """
                        SELECT f.memory_id, bm25(axon_memories_fts) AS rank_score
                        FROM axon_memories_fts f
                        JOIN axon_memories m ON m.id = f.memory_id
                        WHERE f.namespace=? AND m.active=1 AND axon_memories_fts MATCH ?
                        ORDER BY rank_score ASC
                        LIMIT ?
                        """,
                        (namespace, fts_query, limit * 3),
                    ).fetchall()
                    fts_ranked = [(r["memory_id"], -float(r["rank_score"])) for r in fts_rows]
                except sqlite3.OperationalError:
                    fts_ranked = []

        # Substring/CJK fallback ranker
        kw_ranked: list[tuple[str, float]] = []
        if q_tokens:
            scored_kw: list[tuple[str, float]] = []
            for row in active_mems:
                folded = row["folded_content"].casefold()
                hits = sum(1 for tok in q_tokens if tok in folded)
                if hits > 0:
                    scored_kw.append((row["id"], float(hits)))
            scored_kw.sort(key=lambda x: x[1], reverse=True)
            kw_ranked = scored_kw[: limit * 3]

        vec_ranked: list[tuple[str, float]] = []
        q_vec = self.embedder.embed_query(clean_q)
        if q_vec:
            scored_vec: list[tuple[str, float]] = []
            for row in active_mems:
                if row["embedding_json"]:
                    vec = json.loads(row["embedding_json"])
                    sim = cosine_similarity(q_vec, vec)
                    if sim > 0.15:
                        scored_vec.append((row["id"], sim))
            scored_vec.sort(key=lambda x: x[1], reverse=True)
            vec_ranked = scored_vec[: limit * 3]

        fused = reciprocal_rank_fusion([fts_ranked, kw_ranked, vec_ranked])
        top_ids = [mid for mid, _ in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:limit]]

        memories: list[RecallItem] = []
        for mid in top_ids:
            row = mem_by_id.get(mid)
            if row:
                memories.append(
                    RecallItem(
                        id=row["id"],
                        layer="memory",
                        score=round(fused[mid], 5),
                        content=row["content"],
                        hard_facts=json.loads(row["hard_facts_json"]),
                        metadata={"category": row["category"], "fact_key": row["fact_key"]},
                    )
                )

        # 3. Episodic lessons matching query tokens
        episodes: list[dict[str, Any]] = []
        if include_episodes:
            ep_rows = self._conn.execute(
                "SELECT * FROM axon_episodes WHERE namespace=? ORDER BY created_at DESC LIMIT 50",
                (namespace,),
            ).fetchall()
            for ep in ep_rows:
                text_blob = fold_accents_preserving_scripts(
                    " ".join([ep["task_name"], ep["what_happened"], ep["lessons_learned"] or "", ep["entity_names_json"]])
                ).casefold()
                if not q_tokens or any(tok in text_blob for tok in q_tokens):
                    episodes.append({
                        "id": ep["id"],
                        "task_name": ep["task_name"],
                        "outcome": ep["outcome"],
                        "what_happened": ep["what_happened"],
                        "lessons_learned": ep["lessons_learned"],
                        "hard_facts": json.loads(ep["hard_facts_json"]),
                    })
                    if len(episodes) >= limit:
                        break

        # 4. Active (non-expired) prospective foresights
        fs_rows = self._conn.execute(
            "SELECT * FROM axon_foresights WHERE namespace=? AND status='active' ORDER BY COALESCE(due_at, expires_at) ASC LIMIT ?",
            (namespace, limit),
        ).fetchall()
        foresights = [
            {
                "id": r["id"],
                "subject": r["subject"],
                "reminder": r["reminder"],
                "due_at": r["due_at"],
                "expires_at": r["expires_at"],
                "hard_facts": json.loads(r["hard_facts_json"]),
            }
            for r in fs_rows
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
        rows = self._conn.execute(
            "SELECT * FROM axon_episodes WHERE namespace=? AND consolidated=0 ORDER BY created_at ASC LIMIT ?",
            (namespace, limit),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "namespace": r["namespace"],
                "task_name": r["task_name"],
                "outcome": r["outcome"],
                "what_happened": r["what_happened"],
                "lessons_learned": r["lessons_learned"],
                "hard_facts": json.loads(r["hard_facts_json"]),
                "entity_names": json.loads(r["entity_names_json"]),
                "source_lang": r["source_lang"],
            }
            for r in rows
        ]

    def mark_episodes_consolidated(self, episode_ids: list[str], namespace: str = "default") -> int:
        if not episode_ids:
            return 0
        placeholders = ",".join("?" for _ in episode_ids)
        with self._tx() as conn:
            cur = conn.execute(
                f"UPDATE axon_episodes SET consolidated=1 WHERE namespace=? AND id IN ({placeholders})",
                (namespace, *episode_ids),
            )
        return cur.rowcount

    def stats(self, namespace: str = "default") -> MemoryStats:
        c = self._conn
        return MemoryStats(
            backend=self.backend_name,
            namespace=namespace,
            entities=c.execute("SELECT COUNT(*) FROM axon_entities WHERE namespace=?", (namespace,)).fetchone()[0],
            relations=c.execute("SELECT COUNT(*) FROM axon_relations WHERE namespace=?", (namespace,)).fetchone()[0],
            episodes=c.execute("SELECT COUNT(*) FROM axon_episodes WHERE namespace=?", (namespace,)).fetchone()[0],
            unconsolidated_episodes=c.execute(
                "SELECT COUNT(*) FROM axon_episodes WHERE namespace=? AND consolidated=0", (namespace,)
            ).fetchone()[0],
            active_memories=c.execute(
                "SELECT COUNT(*) FROM axon_memories WHERE namespace=? AND active=1", (namespace,)
            ).fetchone()[0],
            superseded_memories=c.execute(
                "SELECT COUNT(*) FROM axon_memories WHERE namespace=? AND active=0", (namespace,)
            ).fetchone()[0],
            active_foresights=c.execute(
                "SELECT COUNT(*) FROM axon_foresights WHERE namespace=? AND status='active'", (namespace,)
            ).fetchone()[0],
            expired_foresights=c.execute(
                "SELECT COUNT(*) FROM axon_foresights WHERE namespace=? AND status='expired'", (namespace,)
            ).fetchone()[0],
        )

    def healthcheck(self) -> tuple[bool, str]:
        try:
            self._conn.execute("SELECT 1;").fetchone()
            self._conn.execute("SELECT COUNT(*) FROM axon_memories_fts;").fetchone()
            return True, f"ok (sqlite+fts5, {self.db_path})"
        except Exception as exc:
            return False, str(exc)

    def list_foresights(
        self,
        namespace: str = "default",
        status: str = "active",
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        self.expire_stale_foresights(namespace=namespace)
        if status == "all":
            rows = self._conn.execute(
                "SELECT * FROM axon_foresights WHERE namespace=? ORDER BY COALESCE(due_at, expires_at) ASC LIMIT ?",
                (namespace, limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM axon_foresights WHERE namespace=? AND status=? ORDER BY COALESCE(due_at, expires_at) ASC LIMIT ?",
                (namespace, status, limit),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "subject": r["subject"],
                "reminder": r["reminder"],
                "hard_facts": json.loads(r["hard_facts_json"]),
                "source_lang": r["source_lang"],
                "due_at": r["due_at"],
                "expires_at": r["expires_at"],
                "status": r["status"],
            }
            for r in rows
        ]

    def export_namespace(self, namespace: str = "default") -> dict[str, Any]:
        c = self._conn
        entities = [
            {
                "entity_type": r["entity_type"],
                "name": r["name"],
                "aliases": json.loads(r["aliases_json"]),
                "summary": r["summary"],
                "properties": json.loads(r["properties_json"]),
                "source_lang": r["source_lang"],
            }
            for r in c.execute(
                "SELECT * FROM axon_entities WHERE namespace=? ORDER BY created_at ASC",
                (namespace,),
            ).fetchall()
        ]
        relations = [
            {
                "source_name": r["source_name"],
                "source_type": r["source_type"],
                "relation_type": r["relation_type"],
                "target_name": r["target_name"],
                "target_type": r["target_type"],
                "weight": float(r["weight"]),
                "properties": json.loads(r["properties_json"]),
            }
            for r in c.execute(
                """
                SELECT r.relation_type, r.weight, r.properties_json,
                       es.name AS source_name, es.entity_type AS source_type,
                       et.name AS target_name, et.entity_type AS target_type
                FROM axon_relations r
                JOIN axon_entities es ON es.id = r.source_id
                JOIN axon_entities et ON et.id = r.target_id
                WHERE r.namespace=?
                ORDER BY r.created_at ASC
                """,
                (namespace,),
            ).fetchall()
        ]
        episodes = [
            {
                "task_name": r["task_name"],
                "outcome": r["outcome"],
                "what_happened": r["what_happened"],
                "lessons_learned": r["lessons_learned"],
                "hard_facts": json.loads(r["hard_facts_json"]),
                "entity_names": json.loads(r["entity_names_json"]),
                "source_lang": r["source_lang"],
                "consolidated": bool(r["consolidated"]),
            }
            for r in c.execute(
                "SELECT * FROM axon_episodes WHERE namespace=? ORDER BY created_at ASC",
                (namespace,),
            ).fetchall()
        ]
        facts = [
            {
                "category": r["category"],
                "fact_key": r["fact_key"],
                "content": r["content"],
                "hard_facts": json.loads(r["hard_facts_json"]),
                "source_lang": r["source_lang"],
            }
            for r in c.execute(
                "SELECT * FROM axon_memories WHERE namespace=? AND active=1 AND superseded_by IS NULL ORDER BY created_at ASC",
                (namespace,),
            ).fetchall()
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
        with self._tx() as conn:
            deleted["relations"] = conn.execute(
                "DELETE FROM axon_relations WHERE namespace=?", (namespace,)
            ).rowcount
            deleted["entities"] = conn.execute(
                "DELETE FROM axon_entities WHERE namespace=?", (namespace,)
            ).rowcount
            deleted["episodes"] = conn.execute(
                "DELETE FROM axon_episodes WHERE namespace=?", (namespace,)
            ).rowcount
            conn.execute("DELETE FROM axon_memories_fts WHERE namespace=?", (namespace,))
            deleted["memories"] = conn.execute(
                "DELETE FROM axon_memories WHERE namespace=?", (namespace,)
            ).rowcount
            deleted["foresights"] = conn.execute(
                "DELETE FROM axon_foresights WHERE namespace=?", (namespace,)
            ).rowcount
        return deleted

    def close(self) -> None:
        self._conn.close()
