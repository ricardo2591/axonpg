"""Model Context Protocol (MCP) server exposing AxonPG's cognitive memory tools."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from mcp.server.mcpserver import MCPServer

from axonpg import __version__
from axonpg.config import AxonConfig
from axonpg.memory import MemoryBackend, build_memory_store
from axonpg.memory.consolidate import consolidate_namespace
from axonpg.memory.models import (
    EntityInput,
    EpisodeInput,
    FactInput,
    ForesightInput,
    RelationInput,
)
from axonpg.providers import LLMProvider, build_llm_provider


def _parse_iso_datetime(value: str) -> datetime:
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    return datetime.fromisoformat(normalized)


def create_mcp_server(
    store: MemoryBackend | None = None,
    config: AxonConfig | None = None,
    llm: LLMProvider | None = None,
) -> MCPServer:
    """Build an MCPServer wired to the configured or injected MemoryBackend."""
    cfg = config or AxonConfig.from_env()
    backend = store or build_memory_store(cfg)
    consolidator_llm = llm if llm is not None else build_llm_provider(cfg)

    mcp = MCPServer(
        name="axonpg",
        title="AxonPG Cognitive Memory",
        description=(
            "Multilingual PostgreSQL-backed cognitive memory engine with Knowledge Graph, "
            "Episodic MemCells, Hybrid Semantic Recall, and Prospective Foresights."
        ),
        instructions=(
            "Call memory_recall before complex tasks to retrieve relevant entities, active "
            "facts, past lessons, and active foresights. Preserve exact values inside hard_facts."
        ),
        version=__version__,
    )

    @mcp.tool(
        description=(
            "Retrieve relevant cognitive memories across Knowledge Graph entities/relations, "
            "hybrid semantic memories, episodic lessons, and active foresights within a token budget."
        )
    )
    def memory_recall(
        query: str,
        namespace: str = cfg.namespace,
        limit: int = 5,
        max_tokens: int = 800,
        include_episodes: bool = True,
    ) -> dict[str, Any]:
        res = backend.recall(
            query=query,
            namespace=namespace,
            limit=limit,
            max_tokens=max_tokens,
            include_episodes=include_episodes,
        )
        return res.model_dump()

    @mcp.tool(
        description="Create or update an entity node (with optional multilingual aliases and properties) in the knowledge graph."
    )
    def memory_upsert_entity(
        name: str,
        entity_type: str = "concept",
        aliases: list[str] | None = None,
        summary: str | None = None,
        properties: dict[str, Any] | None = None,
        source_lang: str | None = None,
        namespace: str = cfg.namespace,
    ) -> dict[str, Any]:
        item = EntityInput(
            namespace=namespace,
            entity_type=entity_type,
            name=name,
            aliases=aliases or [],
            summary=summary,
            properties=properties or {},
            source_lang=source_lang,
        )
        return backend.upsert_entity(item)

    @mcp.tool(
        description="Create or update a directed relation between two entities in the knowledge graph."
    )
    def memory_add_relation(
        source_name: str,
        relation_type: str,
        target_name: str,
        source_type: str = "concept",
        target_type: str = "concept",
        weight: float = 1.0,
        properties: dict[str, Any] | None = None,
        namespace: str = cfg.namespace,
    ) -> dict[str, Any]:
        item = RelationInput(
            namespace=namespace,
            source_name=source_name,
            source_type=source_type,
            relation_type=relation_type,
            target_name=target_name,
            target_type=target_type,
            weight=weight,
            properties=properties or {},
        )
        return backend.add_relation(item)

    @mcp.tool(
        description=(
            "Record an atomic episodic trace (MemCell) of an action/task outcome while preserving exact hard_facts."
        )
    )
    def memory_record_episode(
        task_name: str,
        what_happened: str,
        outcome: Literal["success", "failure", "partial"] = "success",
        lessons_learned: str | None = None,
        hard_facts: dict[str, Any] | None = None,
        entity_names: list[str] | None = None,
        source_lang: str | None = None,
        namespace: str = cfg.namespace,
    ) -> dict[str, Any]:
        item = EpisodeInput(
            namespace=namespace,
            task_name=task_name,
            outcome=outcome,
            what_happened=what_happened,
            lessons_learned=lessons_learned,
            hard_facts=hard_facts or {},
            entity_names=entity_names or [],
            source_lang=source_lang,
        )
        return backend.record_episode(item)

    @mcp.tool(
        description=(
            "Store a semantic fact or procedure; optionally supersedes older active facts sharing the same fact_key."
        )
    )
    def memory_upsert_fact(
        content: str,
        fact_key: str | None = None,
        category: Literal["fact", "scene", "procedure", "lesson"] = "fact",
        hard_facts: dict[str, Any] | None = None,
        source_lang: str | None = None,
        supersede_existing: bool = True,
        namespace: str = cfg.namespace,
    ) -> dict[str, Any]:
        item = FactInput(
            namespace=namespace,
            category=category,
            fact_key=fact_key,
            content=content,
            hard_facts=hard_facts or {},
            source_lang=source_lang,
            supersede_existing=supersede_existing,
        )
        return backend.upsert_fact(item)

    @mcp.tool(
        description=(
            "Store a prospective reminder or time-bounded foresight that automatically expires after expires_at (ISO-8601)."
        )
    )
    def memory_add_foresight(
        subject: str,
        reminder: str,
        expires_at: str,
        due_at: str | None = None,
        hard_facts: dict[str, Any] | None = None,
        source_lang: str | None = None,
        namespace: str = cfg.namespace,
    ) -> dict[str, Any]:
        item = ForesightInput(
            namespace=namespace,
            subject=subject,
            reminder=reminder,
            hard_facts=hard_facts or {},
            source_lang=source_lang,
            due_at=_parse_iso_datetime(due_at) if due_at else None,
            expires_at=_parse_iso_datetime(expires_at),
        )
        return backend.add_foresight(item)

    @mcp.tool(description="Mark an active prospective foresight reminder as completed.")
    def memory_complete_foresight(
        foresight_id: str,
        namespace: str = cfg.namespace,
    ) -> dict[str, Any]:
        return backend.complete_foresight(foresight_id=foresight_id, namespace=namespace)

    @mcp.tool(
        description="Run sleep-time consolidation to compress pending episodes into semantic scenes & expire stale foresights."
    )
    def memory_consolidate(
        namespace: str = cfg.namespace,
        batch_limit: int = 50,
    ) -> dict[str, Any]:
        report = consolidate_namespace(
            store=backend,
            llm=consolidator_llm,
            namespace=namespace,
            batch_limit=batch_limit,
        )
        return report.model_dump()

    @mcp.tool(description="Return memory statistics across all cognitive layers for diagnostics.")
    def memory_stats(namespace: str = cfg.namespace) -> dict[str, Any]:
        return backend.stats(namespace=namespace).model_dump()

    return mcp


def run_mcp_server(transport: Literal["stdio", "streamable-http"] = "stdio") -> None:
    """Entrypoint to launch AxonPG MCP server over stdio or streamable-http."""
    cfg = AxonConfig.from_env()
    store = build_memory_store(cfg)
    try:
        server = create_mcp_server(store=store, config=cfg)
        server.run(transport=transport)
    finally:
        store.close()
