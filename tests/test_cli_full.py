"""Comprehensive end-to-end, redundancy, and failure-mode tests for the full AxonPG CLI."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone

import pytest

from axonpg.cli import HELP_GROUPS
from axonpg.cli import main as cli_main
from axonpg.config import AxonConfig
from axonpg.i18n import SUPPORTED_LANGUAGES
from axonpg.memory.models import (
    EntityInput,
    EpisodeInput,
    FactInput,
    ForesightInput,
    RelationInput,
)
from axonpg.memory.sqlite_store import SQLiteMemoryStore
from axonpg.providers.embeddings import NullEmbeddingProvider


def test_help_displays_all_groups_and_supports_all_languages(capsys):
    expected_commands = {item.name for grp in HELP_GROUPS for item in grp.items}
    assert expected_commands == {
        "init",
        "doctor",
        "up",
        "down",
        "config",
        "recall",
        "stats",
        "consolidate",
        "foresights",
        "serve",
        "connect",
        "agents",
        "export",
        "import",
        "reset",
    }

    # Empty invocation displays help and exits cleanly
    assert cli_main([]) == 0
    out_default = capsys.readouterr().out
    for cmd_name in expected_commands:
        assert cmd_name in out_default

    for lang in SUPPORTED_LANGUAGES:
        with pytest.raises(SystemExit) as exc:
            cli_main(["--lang", lang, "--help"])
        assert exc.value.code == 0
        out = capsys.readouterr().out
        assert "AxonPG" in out
        assert "recall" in out
        assert "[auto|en|es|pt|fr]" in out

    with pytest.raises(SystemExit) as sub_exc:
        cli_main(["recall", "--help"])
    assert sub_exc.value.code == 0
    sub_out = capsys.readouterr().out
    assert "--max-tokens" in sub_out
    assert "arguments:" in sub_out

    with pytest.raises(SystemExit):
        cli_main(["--lang", "es", "--help"])
    es_out = capsys.readouterr().out
    assert "Asistente guiado multilingüe paso a paso" in es_out


def test_global_flags_work_before_and_after_subcommands(tmp_path, capsys):
    hermes_home = tmp_path / "hermes"
    rc = cli_main(
        [
            "setup",  # alias for init
            "--mode",
            "lite",
            "--agent",
            "hermes",
            "--embed-provider",
            "none",
            "--llm-provider",
            "none",
            "--namespace",
            "ns_flags",
            "--dir",
            str(tmp_path),
            "--hermes-home",
            str(hermes_home),
            "--yes",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    env_file = tmp_path / ".env"

    # Global flags BEFORE subcommand
    assert cli_main(["--env-file", str(env_file), "--namespace", "ns_flags", "--json", "stats"]) == 0
    before_payload = json.loads(capsys.readouterr().out)

    # Global flags AFTER subcommand
    assert cli_main(["stats", "--env-file", str(env_file), "--namespace", "ns_flags", "--json"]) == 0
    after_payload = json.loads(capsys.readouterr().out)

    assert before_payload == after_payload
    assert before_payload["namespace"] == "ns_flags"


def test_full_lifecycle_recall_foresights_export_reset_and_redundant_import(tmp_path, capsys, monkeypatch):
    env_file = tmp_path / ".env"
    db_path = tmp_path / "lifecycle.db"
    cfg = AxonConfig(
        mode="lite",
        namespace="prod_ns",
        sqlite_path=str(db_path),
        embed_provider="none",
        llm_provider="none",
    )
    env_file.write_text(cfg.to_env_text(), encoding="utf-8")

    # Seed rich multilingual cognitive data across all 4 layers
    store = SQLiteMemoryStore(db_path=db_path, embedder=NullEmbeddingProvider())
    try:
        store.upsert_entity(
            EntityInput(
                namespace="prod_ns",
                entity_type="service",
                name="Pasarela Bogotá",
                aliases=["GW-BOG"],
                summary="Nodo principal de ingesta",
                source_lang="es",
            )
        )
        store.add_relation(
            RelationInput(
                namespace="prod_ns",
                source_name="Pasarela Bogotá",
                source_type="service",
                relation_type="connects_to",
                target_name="PostgreSQL Core",
            )
        )
        store.record_episode(
            EpisodeInput(
                namespace="prod_ns",
                task_name="Deploy DB",
                outcome="success",
                what_happened="Migrated service to port 5435",
                lessons_learned="Verify unaccent extension first",
                hard_facts={"port": 5435, "node": "GW-BOG"},
                entity_names=["Pasarela Bogotá"],
                source_lang="en",
            )
        )
        store.upsert_fact(
            FactInput(
                namespace="prod_ns",
                fact_key="service:port",
                content="Pasarela Bogotá listens on port 5435",
                hard_facts={"port": 5435},
                source_lang="es",
            )
        )
        fs_res = store.add_foresight(
            ForesightInput(
                namespace="prod_ns",
                subject="Audit TLS cert",
                reminder="Renew certificate before expiry",
                hard_facts={"port": 5435},
                expires_at=datetime.now(timezone.utc) + timedelta(days=7),
            )
        )
        fs_id = fs_res["id"]
    finally:
        store.close()

    # 1. Recall JSON + Rich output
    assert cli_main(["recall", "Bogota 5435", "--env-file", str(env_file), "--json"]) == 0
    rec_payload = json.loads(capsys.readouterr().out)
    assert rec_payload["estimated_tokens"] > 0
    assert any(m["hard_facts"].get("port") == 5435 for m in rec_payload["memories"])
    assert any(e["name"] == "Pasarela Bogotá" for e in rec_payload["entities"])
    assert cli_main(["recall", "Bogota 5435", "--env-file", str(env_file)]) == 0
    capsys.readouterr()

    # 2. Foresights list -> complete -> verify done -> nonexistent returns 1
    assert cli_main(["foresights", "--env-file", str(env_file), "--json"]) == 0
    fs_list = json.loads(capsys.readouterr().out)["foresights"]
    assert len(fs_list) == 1 and fs_list[0]["id"] == fs_id

    assert cli_main(["foresights", "--complete", fs_id, "--env-file", str(env_file), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["updated"] is True

    assert cli_main(["foresights", "--status", "done", "--env-file", str(env_file), "--json"]) == 0
    done_list = json.loads(capsys.readouterr().out)["foresights"]
    assert len(done_list) == 1 and done_list[0]["status"] == "done"

    assert cli_main(["foresights", "--complete", "nonexistent-id", "--env-file", str(env_file), "--json"]) == 1
    capsys.readouterr()

    # 3. Export snapshot
    snap_path = tmp_path / "snapshot.json"
    assert cli_main(["export", str(snap_path), "--env-file", str(env_file), "--json"]) == 0
    exp_out = json.loads(capsys.readouterr().out)
    assert exp_out["counts"] == {"entities": 2, "relations": 1, "episodes": 1, "facts": 1, "foresights": 1}

    # 4. Reset aborted without --yes must preserve data and return 130
    monkeypatch.setattr("axonpg.ui.is_interactive_tty", lambda: False)
    monkeypatch.setattr("builtins.input", lambda _p="": "n")
    assert cli_main(["reset", "--env-file", str(env_file)]) == 130
    capsys.readouterr()

    assert cli_main(["stats", "--env-file", str(env_file), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["active_memories"] == 1

    # 5. Reset with --yes clears namespace completely
    assert cli_main(["reset", "--yes", "--env-file", str(env_file), "--json"]) == 0
    capsys.readouterr()
    assert cli_main(["stats", "--env-file", str(env_file), "--json"]) == 0
    cleared = json.loads(capsys.readouterr().out)
    assert cleared["entities"] == 0 and cleared["active_memories"] == 0 and cleared["episodes"] == 0

    # 6. Import into target namespace + REDUNDANT second import checks deduplication
    assert cli_main(["import", str(snap_path), "--env-file", str(env_file), "--namespace", "restored_ns", "--json"]) == 0
    capsys.readouterr()
    assert cli_main(["import", str(snap_path), "--env-file", str(env_file), "--namespace", "restored_ns", "--json"]) == 0
    capsys.readouterr()

    assert cli_main(["stats", "--env-file", str(env_file), "--namespace", "restored_ns", "--json"]) == 0
    restored_stats = json.loads(capsys.readouterr().out)
    assert restored_stats["entities"] == 2
    assert restored_stats["relations"] == 1
    assert restored_stats["active_memories"] == 1
    assert restored_stats["superseded_memories"] == 1

    # 7. Invalid import paths & bad snapshot fail cleanly
    assert cli_main(["import", str(tmp_path / "missing.json"), "--env-file", str(env_file), "--json"]) == 1
    capsys.readouterr()
    bad_snap = tmp_path / "bad.json"
    bad_snap.write_text('{"schema_version": 99}', encoding="utf-8")
    assert cli_main(["import", str(bad_snap), "--env-file", str(env_file), "--json"]) == 1
    capsys.readouterr()


def test_config_redaction_connect_and_agents_commands(tmp_path, capsys):
    env_file = tmp_path / ".env"
    cfg = AxonConfig(
        mode="lite",
        namespace="multi_agent",
        sqlite_path=str(tmp_path / "ag.db"),
        db_password="RAW_DB_PASSWORD_999",
        embed_provider="none",
        embed_api_key="RAW_EMBED_KEY_999",
        llm_provider="none",
        llm_api_key="RAW_LLM_KEY_999",
    )
    env_file.write_text(cfg.to_env_text(), encoding="utf-8")

    # Config command never leaks secrets
    assert cli_main(["config", "--env-file", str(env_file), "--json"]) == 0
    raw_cfg_json = capsys.readouterr().out
    assert "RAW_DB_PASSWORD_999" not in raw_cfg_json
    assert "RAW_EMBED_KEY_999" not in raw_cfg_json
    assert "RAW_LLM_KEY_999" not in raw_cfg_json
    assert json.loads(raw_cfg_json)["config"]["db_password"] == "***"

    hermes_home = tmp_path / ".h"
    openclaw_home = tmp_path / ".o"
    assert cli_main(["connect", "hermes", "--dir", str(tmp_path), "--hermes-home", str(hermes_home), "--env-file", str(env_file), "--json"]) == 0
    capsys.readouterr()
    assert cli_main(["connect", "openclaw", "--dir", str(tmp_path), "--openclaw-home", str(openclaw_home), "--env-file", str(env_file), "--json"]) == 0
    capsys.readouterr()
    assert cli_main(["connect", "generic", "--dir", str(tmp_path), "--env-file", str(env_file), "--json"]) == 0
    capsys.readouterr()

    assert (
        cli_main(
            [
                "agents",
                "--dir",
                str(tmp_path),
                "--hermes-home",
                str(hermes_home),
                "--openclaw-home",
                str(openclaw_home),
                "--json",
            ]
        )
        == 0
    )
    agents_out = json.loads(capsys.readouterr().out)["agents"]
    assert len(agents_out) == 3
    assert all(a["configured"] is True for a in agents_out)


def test_up_and_down_docker_lifecycle_and_error_handling(tmp_path, capsys, monkeypatch):
    # Missing docker-compose.yml fails cleanly
    assert cli_main(["up", "--dir", str(tmp_path), "--json"]) == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False

    # Create dummy compose file and test missing docker CLI
    (tmp_path / "docker-compose.yml").write_text("services: {}\n", encoding="utf-8")
    monkeypatch.setattr("axonpg.cli.shutil.which", lambda _name: None)
    assert cli_main(["up", "--dir", str(tmp_path), "--json"]) == 1
    capsys.readouterr()

    # Mock working docker CLI + pg_isready
    monkeypatch.setattr("axonpg.cli.shutil.which", lambda _name: "/usr/bin/docker")
    seen_cmds: list[list[str]] = []

    def _fake_run(cmd, **_kwargs):
        seen_cmds.append(list(cmd))
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("axonpg.cli.subprocess.run", _fake_run)
    assert cli_main(["up", "--dir", str(tmp_path), "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["ok"] is True
    assert ["docker", "compose", "up", "-d"] in seen_cmds
    assert any("pg_isready" in cmd for cmd in seen_cmds)

    assert cli_main(["down", "--dir", str(tmp_path), "-v", "--json"]) == 0
    down_payload = json.loads(capsys.readouterr().out)
    assert down_payload["ok"] is True and down_payload["volumes_removed"] is True
    assert ["docker", "compose", "down", "-v"] in seen_cmds


def test_port_scanning_detects_busy_port_and_reassigns_cleanly(tmp_path, monkeypatch):
    import socket

    from axonpg.cli import _resolve_db_port
    from axonpg.ui import create_console, find_available_port, is_port_available

    # 1. Real bound OS socket must be detected as occupied
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as occupied:
        occupied.bind(("127.0.0.1", 0))
        occupied.listen(1)
        busy_port = occupied.getsockname()[1]

        assert is_port_available(busy_port) is False
        scanned = find_available_port(busy_port, max_scan=20)
        assert scanned != busy_port
        assert is_port_available(scanned) is True

        # Interactive _resolve_db_port warns and accepts the suggested free port (even after retrying a busy input)
        answers = iter([str(busy_port), str(scanned)])
        monkeypatch.setattr("axonpg.ui.is_interactive_tty", lambda: False)
        monkeypatch.setattr("builtins.input", lambda _p="": next(answers))
        chosen = _resolve_db_port(create_console(), requested_port=busy_port, lang="es", interactive=True)
        assert chosen == scanned

    # 2. Non-interactive `axonpg init --mode docker --yes` avoids colliding when default 5435 is occupied
    monkeypatch.setattr("axonpg.cli._is_axonpg_container_on_port", lambda _p: False)
    monkeypatch.setattr("axonpg.cli.is_port_available", lambda p: p != 5435)
    monkeypatch.setattr("axonpg.cli.find_available_port", lambda p: 5436 if p == 5435 else p)

    rc = cli_main(
        [
            "init",
            "--mode",
            "docker",
            "--agent",
            "generic",
            "--embed-provider",
            "none",
            "--llm-provider",
            "none",
            "--dir",
            str(tmp_path),
            "--yes",
        ]
    )
    assert rc == 0
    env_text = (tmp_path / ".env").read_text(encoding="utf-8")
    assert "AXONPG_DB_PORT=5436" in env_text
