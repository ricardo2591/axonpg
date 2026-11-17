"""End-to-end tests for AxonPG MCP tools, agent adapters (Hermes/OpenClaw/Generic), and CLI."""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timedelta, timezone

import yaml

from axonpg.adapters.generic_mcp import configure_generic_mcp
from axonpg.adapters.hermes import configure_hermes
from axonpg.adapters.openclaw import configure_openclaw
from axonpg.cli import main as cli_main
from axonpg.config import AxonConfig
from axonpg.mcp_server import create_mcp_server
from axonpg.memory.sqlite_store import SQLiteMemoryStore
from axonpg.providers.embeddings import NullEmbeddingProvider


def test_mcp_server_exposes_and_executes_all_nine_tools():
    store = SQLiteMemoryStore(":memory:", NullEmbeddingProvider())
    try:
        server = create_mcp_server(store=store)
        tools = asyncio.run(server.list_tools())
        names = {tool.name for tool in tools}
        assert names == {
            "memory_recall",
            "memory_upsert_entity",
            "memory_add_relation",
            "memory_record_episode",
            "memory_upsert_fact",
            "memory_add_foresight",
            "memory_complete_foresight",
            "memory_consolidate",
            "memory_stats",
        }

        asyncio.run(
            server.call_tool(
                "memory_upsert_entity",
                {"name": "Gateway Bogotá", "entity_type": "service", "aliases": ["GW-BOG"], "namespace": "mcp"},
            )
        )
        asyncio.run(
            server.call_tool(
                "memory_add_relation",
                {
                    "source_name": "Gateway Bogotá",
                    "relation_type": "depends_on",
                    "target_name": "AxonPG DB",
                    "namespace": "mcp",
                },
            )
        )
        asyncio.run(
            server.call_tool(
                "memory_record_episode",
                {
                    "task_name": "Despliegue MCP",
                    "what_happened": "Servicio activo en puerto 5435",
                    "outcome": "success",
                    "hard_facts": {"port": 5435},
                    "namespace": "mcp",
                },
            )
        )
        asyncio.run(
            server.call_tool(
                "memory_upsert_fact",
                {
                    "fact_key": "mcp:lang",
                    "content": "El servidor MCP responde en múltiples idiomas",
                    "namespace": "mcp",
                },
            )
        )
        fs_res = asyncio.run(
            server.call_tool(
                "memory_add_foresight",
                {
                    "subject": "Verificar métricas",
                    "reminder": "Revisar consumo tras 24h",
                    "expires_at": (datetime.now(timezone.utc) + timedelta(days=2)).isoformat(),
                    "namespace": "mcp",
                },
            )
        )
        fs_id = fs_res.structured_content["id"]

        recall_res = asyncio.run(
            server.call_tool("memory_recall", {"query": "Bogota servidor MCP 5435", "namespace": "mcp"})
        )
        assert recall_res.is_error is False
        payload = recall_res.structured_content
        assert payload["memories"]
        assert payload["entities"]
        assert payload["relations"]
        assert payload["foresights"]

        comp_res = asyncio.run(
            server.call_tool("memory_complete_foresight", {"foresight_id": fs_id, "namespace": "mcp"})
        )
        assert comp_res.structured_content["updated"] is True

        con_res = asyncio.run(server.call_tool("memory_consolidate", {"namespace": "mcp"}))
        assert con_res.structured_content["episodes_processed"] == 1

        stats_res = asyncio.run(server.call_tool("memory_stats", {"namespace": "mcp"}))
        assert stats_res.structured_content["unconsolidated_episodes"] == 0
    finally:
        store.close()


def test_agent_adapters_preserve_existing_configs_and_never_leak_secrets(tmp_path):
    cfg = AxonConfig(
        mode="lite",
        namespace="demo",
        db_password="secret_db_value_should_not_leak",
        embed_api_key="secret_embed_key_should_not_leak",
        llm_api_key="secret_llm_key_should_not_leak",
    )
    env_file = tmp_path / ".env"
    env_file.write_text(cfg.to_env_text(), encoding="utf-8")

    # Hermes preserves existing custom YAML settings
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    (hermes_home / "config.yaml").write_text("model: custom/hermes-default\nmcp_servers:\n  other:\n    command: echo\n", encoding="utf-8")
    configure_hermes(config=cfg, env_file=env_file, hermes_home=hermes_home)

    hermes_yaml = yaml.safe_load((hermes_home / "config.yaml").read_text(encoding="utf-8"))
    assert hermes_yaml["model"] == "custom/hermes-default"
    assert "other" in hermes_yaml["mcp_servers"]
    assert "axonpg" in hermes_yaml["mcp_servers"]
    assert (hermes_home / "skills" / "axonpg-memory" / "SKILL.md").is_file()
    raw_yaml = (hermes_home / "config.yaml").read_text(encoding="utf-8")
    assert "secret_embed_key_should_not_leak" not in raw_yaml
    assert "secret_db_value_should_not_leak" not in raw_yaml

    # OpenClaw preserves openclaw.json and writes standalone mcp/axonpg.json
    oc_home = tmp_path / ".openclaw"
    oc_home.mkdir()
    (oc_home / "openclaw.json").write_text('{"gateway": {"port": 18789}}', encoding="utf-8")
    oc_res = configure_openclaw(config=cfg, env_file=env_file, openclaw_home=oc_home)
    assert (oc_home / "openclaw.json").read_text(encoding="utf-8") == '{"gateway": {"port": 18789}}'
    oc_mcp = json.loads((oc_home / "mcp" / "axonpg.json").read_text(encoding="utf-8"))
    assert "axonpg" in oc_mcp["mcpServers"]
    assert oc_res["skill_path"].endswith("SKILL.md")

    # Generic MCP
    gen_path = tmp_path / "axonpg.mcp.json"
    configure_generic_mcp(config=cfg, output_path=gen_path, env_file=env_file)
    gen_data = json.loads(gen_path.read_text(encoding="utf-8"))
    assert gen_data["mcpServers"]["axonpg"]["env"]["AXONPG_ENV_FILE"] == str(env_file.resolve())


def test_cli_init_doctor_consolidate_and_stats_lite_mode(tmp_path):
    hermes_home = tmp_path / "hermes_cli"
    rc_init = cli_main(
        [
            "init",
            "--lang",
            "es",
            "--mode",
            "lite",
            "--agent",
            "hermes",
            "--embed-provider",
            "none",
            "--llm-provider",
            "none",
            "--namespace",
            "cli_ns",
            "--dir",
            str(tmp_path),
            "--hermes-home",
            str(hermes_home),
            "--yes",
        ]
    )
    assert rc_init == 0
    env_file = tmp_path / ".env"
    assert env_file.is_file()
    assert (tmp_path / "axonpg_lite.db").is_file()
    assert (hermes_home / "config.yaml").is_file()

    assert cli_main(["doctor", "--env-file", str(env_file)]) == 0
    assert cli_main(["stats", "--env-file", str(env_file), "--namespace", "cli_ns"]) == 0
    assert cli_main(["consolidate", "--env-file", str(env_file), "--namespace", "cli_ns"]) == 0


def test_cli_json_flags_and_visual_ui_rendering(tmp_path, capsys):
    hermes_home = tmp_path / "hermes_json"
    assert (
        cli_main(
            [
                "init",
                "--lang",
                "en",
                "--mode",
                "lite",
                "--agent",
                "generic",
                "--embed-provider",
                "none",
                "--llm-provider",
                "none",
                "--namespace",
                "json_ns",
                "--dir",
                str(tmp_path),
                "--hermes-home",
                str(hermes_home),
                "--yes",
            ]
        )
        == 0
    )
    capsys.readouterr()
    env_file = tmp_path / ".env"

    assert cli_main(["doctor", "--env-file", str(env_file), "--json"]) == 0
    doc_payload = json.loads(capsys.readouterr().out)
    assert doc_payload["ok"] is True
    assert doc_payload["config"]["namespace"] == "json_ns"

    assert cli_main(["stats", "--env-file", str(env_file), "--namespace", "json_ns", "--json"]) == 0
    stats_payload = json.loads(capsys.readouterr().out)
    assert stats_payload["backend"] == "sqlite-lite"

    assert cli_main(["consolidate", "--env-file", str(env_file), "--namespace", "json_ns", "--json"]) == 0
    con_payload = json.loads(capsys.readouterr().out)
    assert con_payload["namespace"] == "json_ns"


def test_cli_interactive_plain_fallback_and_cancel(tmp_path, monkeypatch):
    from axonpg.ui import detect_preflight

    preflight = detect_preflight(db_port=59999, hermes_home=tmp_path / ".h", openclaw_home=tmp_path / ".o")
    assert preflight.hermes_exists is False
    assert preflight.openclaw_exists is False

    answers = iter(["2", "2", "3", "8", "1", "fallback_ns", "y"])
    monkeypatch.setattr("axonpg.ui.is_interactive_tty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))

    rc = cli_main(["init", "--dir", str(tmp_path), "--openclaw-home", str(tmp_path / ".oc")])
    assert rc == 0
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "AXONPG_LANG=es" in env_text
    assert "AXONPG_MODE=lite" in env_text
    assert "AXONPG_NAMESPACE=fallback_ns" in env_text
    assert (tmp_path / "axonpg.mcp.json").is_file()

    def _raise_interrupt(_prompt=""):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", _raise_interrupt)
    assert cli_main(["init", "--dir", str(tmp_path)]) == 130


def test_doctor_catches_missing_keys_and_unreachable_ollama(tmp_path, capsys, monkeypatch):
    # Prevent host ~/.env or process env from bleeding into isolated test runs
    for env_key in (
        "AXONPG_MODE",
        "AXONPG_EMBED_PROVIDER",
        "AXONPG_EMBED_API_KEY",
        "AXONPG_LLM_PROVIDER",
        "AXONPG_LLM_API_KEY",
    ):
        monkeypatch.delenv(env_key, raising=False)

    # 1. Missing API keys for gemini embeddings + openrouter LLM must fail doctor explicitly
    bad_env = tmp_path / "missing_keys.env"
    bad_cfg = AxonConfig(
        mode="lite",
        sqlite_path=str(tmp_path / "doc.db"),
        embed_provider="gemini",
        embed_api_key="",
        llm_provider="openrouter",
        llm_api_key="",
    )
    bad_env.write_text(bad_cfg.to_env_text(), encoding="utf-8")
    assert cli_main(["doctor", "--env-file", str(bad_env), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    checks_by_name = {c["component"]: c for c in payload["checks"]}
    assert checks_by_name["Embeddings"]["ok"] is False
    assert "AXONPG_EMBED_API_KEY missing" in checks_by_name["Embeddings"]["detail"]
    assert checks_by_name["Consolidation LLM"]["ok"] is False
    assert "AXONPG_LLM_API_KEY missing" in checks_by_name["Consolidation LLM"]["detail"]

    # 2. Unreachable Ollama LLM must fail doctor cleanly
    ol_env = tmp_path / "ollama_down.env"
    ol_cfg = AxonConfig(
        mode="lite",
        sqlite_path=str(tmp_path / "doc_ol.db"),
        embed_provider="none",
        llm_provider="ollama",
        llm_base_url="http://127.0.0.1:59998",
    )
    ol_env.write_text(ol_cfg.to_env_text(), encoding="utf-8")
    assert cli_main(["doctor", "--env-file", str(ol_env), "--json"]) == 1
    ol_payload = json.loads(capsys.readouterr().out)
    ol_checks = {c["component"]: c for c in ol_payload["checks"]}
    assert ol_checks["Consolidation LLM"]["ok"] is False
    assert "unreachable" in ol_checks["Consolidation LLM"]["detail"].lower()
