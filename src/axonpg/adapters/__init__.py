"""Agent runtime adapters (Hermes Agent, OpenClaw, and Generic MCP clients)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from axonpg.config import AxonConfig

DEFAULT_SKILL_MARKDOWN = """---
name: axonpg-memory
description: Multilingual PostgreSQL-native cognitive memory via AxonPG MCP. Recall before complex work and record episodes, entities, facts, and foresights with exact hard_facts.
---

# AxonPG Cognitive Memory

- Recall prior state: `memory_recall(query=..., namespace=...)`
- Record outcome + exact values: `memory_record_episode(...)`
- Upsert evolving facts (auto-supersedes `fact_key`): `memory_upsert_fact(...)`
- Track relations & entities: `memory_upsert_entity(...)`, `memory_add_relation(...)`
- Set time-bounded reminders: `memory_add_foresight(...)`
"""


def find_skill_markdown(repo_root: Path | None = None) -> str:
    """Load canonical SKILL.md if present in package workspace, else use embedded copy."""
    candidates = []
    if repo_root is not None:
        candidates.append(repo_root / "skills" / "axonpg-memory" / "SKILL.md")
    candidates.append(Path(__file__).resolve().parents[3] / "skills" / "axonpg-memory" / "SKILL.md")
    for path in candidates:
        if path.is_file():
            return path.read_text(encoding="utf-8")
    return DEFAULT_SKILL_MARKDOWN


def build_mcp_entry(
    config: AxonConfig,
    env_file: Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Build a secret-free stdio MCP server entry referencing `.env` via AXONPG_ENV_FILE."""
    py = python_executable or sys.executable
    args = ["-m", "axonpg", "serve"]
    env: dict[str, str] = {
        "AXONPG_MODE": config.mode,
        "AXONPG_NAMESPACE": config.namespace,
    }
    if env_file is not None:
        resolved = str(env_file.expanduser().resolve())
        args.extend(["--env-file", resolved])
        env["AXONPG_ENV_FILE"] = resolved

    return {
        "command": py,
        "args": args,
        "env": env,
    }


def merge_json_mcp_file(
    target_path: Path,
    server_name: str,
    entry: dict[str, Any],
    root_key: str = "mcpServers",
) -> Path:
    """Idempotently merge an MCP server entry into a JSON file without clobbering existing keys."""
    target_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if target_path.is_file():
        raw = target_path.read_text(encoding="utf-8").strip()
        if raw:
            loaded = json.loads(raw)
            if isinstance(loaded, dict):
                payload = loaded

    section = payload.get(root_key)
    if not isinstance(section, dict):
        section = {}
    section[server_name] = entry
    payload[root_key] = section
    target_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return target_path


def install_skill_file(skill_dir: Path, repo_root: Path | None = None) -> Path:
    """Install `axonpg-memory/SKILL.md` into an agent's skill directory."""
    skill_dir.mkdir(parents=True, exist_ok=True)
    target = skill_dir / "SKILL.md"
    target.write_text(find_skill_markdown(repo_root), encoding="utf-8")
    return target
