---
name: axonpg-memory
description: Multilingual PostgreSQL-native cognitive memory skill for AI agents via AxonPG MCP. Use before complex work to recall active facts, entities, lessons, and foresights, and afterward to record outcomes without losing exact values.
---

# AxonPG Cognitive Memory Protocol

Use the `axonpg` MCP tools consistently to maintain durable, cross-session, multilingual memory without bloating prompts.

## When to Recall (`memory_recall`)

- Before debugging, deploying, migrating, answering historical project questions, or continuing previous work:
  Call `memory_recall(query=..., namespace=..., limit=5, max_tokens=800)`.
- Read `[ACTIVE_FORESIGHTS]`, `[KNOWLEDGE_GRAPH]`, `[SEMANTIC_MEMORIES]`, and `[EPISODIC_LESSONS]` carefully.
- Treat any field inside `hard_facts` (IDs, ports, timestamps, currency amounts, URLs, error codes) as **exact ground truth**.

## When to Write

1. **Completed or Failed Action -> `memory_record_episode`**:
   Record `task_name`, `outcome` (`"success" | "failure" | "partial"`), `what_happened`, `lessons_learned`, `entity_names`, and `hard_facts`.
   Preserve the user's original language (`source_lang`) — do not force-translate non-English facts.

2. **Verified Current Fact or Changed Rule -> `memory_upsert_fact`**:
   Provide a stable normalized `fact_key` (e.g., `"deploy:postgres-port"`) with `supersede_existing=True` so newer facts automatically supersede outdated values instead of leaving contradictions.

3. **Entities & Graph Relationships -> `memory_upsert_entity` / `memory_add_relation`**:
   Track services, people, projects, and aliases (`aliases=["..."]`) plus causal edges (`depends_on`, `fixed_by`, `owns`, `supersedes`).

4. **Time-Bounded Reminders / Follow-ups -> `memory_add_foresight` / `memory_complete_foresight`**:
   Include an explicit ISO-8601 `expires_at` so stale alerts auto-expire without manual cleanup.

## Periodic Hygiene

- Run `memory_consolidate(namespace=...)` when pending unconsolidated episodes accumulate (`memory_stats`).
