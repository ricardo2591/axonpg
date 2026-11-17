"""Hermes Agent adapter (default auto-configured runtime: MCP YAML + evolving skill)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from axonpg.adapters import build_mcp_entry, install_skill_file
from axonpg.config import AxonConfig


def configure_hermes(
    config: AxonConfig,
    env_file: Path | None = None,
    hermes_home: Path | None = None,
    python_executable: str | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """
    Configure Hermes Agent non-destructively:
    1. Merge `mcp_servers.axonpg` into `~/.hermes/config.yaml`.
    2. Install `axonpg-memory/SKILL.md` into `~/.hermes/skills/axonpg-memory/SKILL.md`.
    """
    home = (hermes_home or (Path.home() / ".hermes")).expanduser()
    home.mkdir(parents=True, exist_ok=True)

    config_path = home / "config.yaml"
    data: dict[str, Any] = {}
    if config_path.is_file():
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded

    mcp_servers = data.get("mcp_servers")
    if not isinstance(mcp_servers, dict):
        mcp_servers = {}

    entry = build_mcp_entry(config=config, env_file=env_file, python_executable=python_executable)
    mcp_servers["axonpg"] = entry
    data["mcp_servers"] = mcp_servers

    config_path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    skill_path = install_skill_file(home / "skills" / "axonpg-memory", repo_root=repo_root)
    return {
        "agent": "hermes",
        "config_path": str(config_path),
        "skill_path": str(skill_path),
        "mcp_entry": entry,
    }
