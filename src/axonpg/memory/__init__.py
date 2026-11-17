"""Cognitive memory storage factory selecting Postgres (docker) or SQLite (--lite)."""

from __future__ import annotations

from axonpg.config import AxonConfig
from axonpg.memory.backend import MemoryBackend
from axonpg.memory.postgres_store import PostgresMemoryStore
from axonpg.memory.sqlite_store import SQLiteMemoryStore
from axonpg.providers.embeddings import EmbeddingProvider, build_embedding_provider


def build_memory_store(
    config: AxonConfig,
    embedder: EmbeddingProvider | None = None,
) -> MemoryBackend:
    """Instantiate the configured MemoryBackend (`docker` PostgreSQL or `lite` SQLite)."""
    emb = embedder or build_embedding_provider(config)
    if config.mode == "lite":
        return SQLiteMemoryStore(db_path=config.resolved_sqlite_path(), embedder=emb)
    return PostgresMemoryStore(dsn=config.postgres_dsn(), embedder=emb)


__all__ = [
    "MemoryBackend",
    "PostgresMemoryStore",
    "SQLiteMemoryStore",
    "build_memory_store",
]
