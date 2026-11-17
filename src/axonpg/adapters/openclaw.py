"""OpenClaw adapter: generates strict-schema-safe standalone MCP config & workspace skill."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axonpg.adapters import build_mcp_entry, install_skill_file, merge_json_mcp_file
from axonpg.config import AxonConfig


def configure_openclaw(
    config: AxonConfig,
    env_file: Path | None = None,
    openclaw_home: Path | None = None,
    python_executable: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Configure OpenClaw without mutating strict-validated `openclaw.json` root schema:
    1. Merge `mcpServers.axonpg` into `~/.openclaw/mcp/axonpg.json`.
    2. Install `axonpg-memory/SKILL.md` into `~/.openclaw/skills/axonpg-memory/SKILL.md`.
    """
    home = (openclaw_home or (Path.home() / ".openclaw")).expanduser()
    mcp_config_path = merge_json_mcp_file(
        target_path=home / "mcp" / "axonpg.json",
        server_name="axonpg",
        entry=build_mcp_entry(config=config, env_file=env_file, python_executable=python_executable),
        root_key="mcpServers",
    )
    skill_path = install_skill_file(home / "skills" / "axonpg-memory", repo_root=repo_root)
    return {
        "agent": "openclaw",
        "mcp_config_path": str(mcp_config_path),
        "skill_path": str(skill_path),
        "note": (
            f"Generated {mcp_config_path} without mutating openclaw.json schema. "
            "Register via your OpenClaw MCP plugin / mcporter config."
        ),
    }
