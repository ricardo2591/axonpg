"""Generic MCP adapter for Claude Desktop, Cursor, Windsurf, and Antigravity."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from axonpg.adapters import build_mcp_entry, merge_json_mcp_file
from axonpg.config import AxonConfig


def configure_generic_mcp(
    config: AxonConfig,
    output_path: Path,
    env_file: Path | None = None,
    python_executable: str | None = None,
) -> dict[str, Any]:
    """Write standard `mcpServers` JSON snippet for any MCP-compatible host."""
    entry = build_mcp_entry(config=config, env_file=env_file, python_executable=python_executable)
    written = merge_json_mcp_file(
        target_path=output_path,
        server_name="axonpg",
        entry=entry,
        root_key="mcpServers",
    )
    return {
        "agent": "generic",
        "mcp_config_path": str(written),
        "mcpServers": {"axonpg": entry},
    }
