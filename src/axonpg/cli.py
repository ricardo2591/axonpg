"""Interactive & scriptable CLI for AxonPG (`init`, `doctor`, `up`, `down`, `recall`, `export`, etc.)."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import yaml
from dotenv import load_dotenv

from axonpg import __version__
from axonpg.adapters.generic_mcp import configure_generic_mcp
from axonpg.adapters.hermes import configure_hermes
from axonpg.adapters.openclaw import configure_openclaw
from axonpg.config import EMBEDDING_PRESETS, LLM_PRESETS, AxonConfig
from axonpg.i18n import SUPPORTED_LANGUAGES, resolve_language, t
from axonpg.memory import build_memory_store
from axonpg.memory.consolidate import consolidate_namespace
from axonpg.providers import build_embedding_provider, build_llm_provider
from axonpg.ui import (
    CommandHelpGroup,
    CommandHelpItem,
    PreflightStatus,
    SelectOption,
    create_console,
    detect_preflight,
    find_available_port,
    is_port_available,
    prompt_confirm,
    prompt_text,
    render_agent_comparison,
    render_agents_table,
    render_banner,
    render_config_table,
    render_consolidation_report,
    render_doctor_report,
    render_foresights_table,
    render_init_summary,
    render_preflight,
    render_recall_result,
    render_root_help,
    render_stats_report,
    render_step_header,
    render_subcommand_help,
    render_transfer_counts,
    select_option,
    status_spinner,
)

HELP_GROUPS: list[CommandHelpGroup] = [
    CommandHelpGroup(
        title_key="grp_setup",
        items=(
            CommandHelpItem("init", "cmd_init", aliases=("setup",)),
            CommandHelpItem("doctor", "cmd_doctor"),
            CommandHelpItem("up", "cmd_up"),
            CommandHelpItem("down", "cmd_down"),
            CommandHelpItem("config", "cmd_config"),
        ),
    ),
    CommandHelpGroup(
        title_key="grp_memory",
        items=(
            CommandHelpItem("recall", "cmd_recall"),
            CommandHelpItem("stats", "cmd_stats"),
            CommandHelpItem("consolidate", "cmd_consolidate"),
            CommandHelpItem("foresights", "cmd_foresights"),
        ),
    ),
    CommandHelpGroup(
        title_key="grp_agents",
        items=(
            CommandHelpItem("serve", "cmd_serve"),
            CommandHelpItem("connect", "cmd_connect"),
            CommandHelpItem("agents", "cmd_agents"),
        ),
    ),
    CommandHelpGroup(
        title_key="grp_data",
        items=(
            CommandHelpItem("export", "cmd_export"),
            CommandHelpItem("import", "cmd_import"),
            CommandHelpItem("reset", "cmd_reset"),
        ),
    ),
]

_LANGUAGE_OPTIONS: list[SelectOption] = [
    SelectOption(value="en", label="English", description="Default global interface"),
    SelectOption(value="es", label="Español", description="Interfaz en español"),
    SelectOption(value="pt", label="Português", description="Interface em português"),
    SelectOption(value="fr", label="Français", description="Interface en français"),
]


def _resolve_env_path(env_file: str | None = None, base_dir: Path | None = None) -> Path | None:
    candidate = env_file or os.getenv("AXONPG_ENV_FILE")
    if candidate:
        p = Path(candidate).expanduser()
        return p if p.is_file() else None
    local_env = (base_dir or Path.cwd()) / ".env"
    return local_env if local_env.is_file() else None


def _load_env_if_present(env_file: str | None = None) -> str:
    candidate = env_file or os.getenv("AXONPG_ENV_FILE")
    if candidate:
        p = Path(candidate).expanduser()
        if p.is_file():
            load_dotenv(p, override=True)
            return str(p.resolve())
    local_env = Path.cwd() / ".env"
    if local_env.is_file():
        load_dotenv(local_env, override=False)
        return str(local_env.resolve())
    return "process environment / defaults"


def _write_env_file(target_dir: Path, config: AxonConfig) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    env_path = target_dir / ".env"
    env_path.write_text(config.to_env_text(), encoding="utf-8")
    with contextlib.suppress(OSError):
        os.chmod(env_path, 0o600)
    return env_path


def _wait_for_docker_postgres(target_dir: Path, config: AxonConfig, timeout_seconds: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        proc = subprocess.run(
            [
                "docker",
                "compose",
                "exec",
                "-T",
                "axonpg-db",
                "pg_isready",
                "-U",
                config.db_user,
                "-d",
                config.db_name,
            ],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            return True
        time.sleep(1.0)
    return False


def _choose_storage_mode(lang: str, preflight: PreflightStatus, default_mode: str) -> str:
    options = [
        SelectOption(
            value="docker",
            label="Docker",
            description=t("mode_docker_detail", lang),
            recommended=preflight.docker_running,
        ),
        SelectOption(
            value="lite",
            label="Lite",
            description=t("mode_lite_detail", lang),
            recommended=not preflight.docker_running,
        ),
    ]
    preferred = default_mode if preflight.docker_running else "lite"
    return select_option(t("choose_mode", lang), options, default=preferred, lang=lang)


def _choose_agent_runtime(lang: str, preflight: PreflightStatus, default_agent: str) -> str:
    preferred = default_agent
    if preflight.openclaw_exists and not preflight.hermes_exists:
        preferred = "openclaw"
    options = [
        SelectOption(
            value="hermes",
            label="Hermes",
            description=t("agent_hermes_integration", lang),
            recommended=True,
        ),
        SelectOption(
            value="openclaw",
            label="OpenClaw",
            description=t("agent_openclaw_integration", lang),
        ),
        SelectOption(
            value="generic",
            label="Generic MCP",
            description=t("agent_generic_integration", lang),
        ),
    ]
    return select_option(t("choose_agent", lang), options, default=preferred, lang=lang)


def _choose_embed_provider(lang: str, preflight: PreflightStatus, default_provider: str) -> str:
    recommended = "ollama" if (preflight.ollama_reachable and not preflight.fastembed_installed) else "fastembed"
    options = [
        SelectOption(
            value=name,
            label=name,
            description=t(f"embed_{name}_desc", lang),
            recommended=(name == recommended),
        )
        for name in EMBEDDING_PRESETS
    ]
    chosen_default = recommended if default_provider == "fastembed" else default_provider
    return select_option(t("choose_embed", lang), options, default=chosen_default, lang=lang)


def _choose_llm_provider(lang: str, default_provider: str) -> str:
    options = [
        SelectOption(
            value=name,
            label=name,
            description=t(f"llm_{name}_desc", lang),
            recommended=(name == "none"),
        )
        for name in LLM_PRESETS
    ]
    return select_option(t("choose_llm", lang), options, default=default_provider, lang=lang)


def _apply_agent_adapter(
    *,
    agent: str,
    config: AxonConfig,
    env_file: Path | None,
    target_dir: Path,
    hermes_home: Path | None = None,
    openclaw_home: Path | None = None,
    generic_output: Path | None = None,
) -> dict[str, Any]:
    if agent == "hermes":
        return configure_hermes(
            config=config,
            env_file=env_file,
            hermes_home=hermes_home,
            repo_root=target_dir,
        )
    if agent == "openclaw":
        return configure_openclaw(
            config=config,
            env_file=env_file,
            openclaw_home=openclaw_home,
            repo_root=target_dir,
        )
    return configure_generic_mcp(
        config=config,
        output_path=generic_output or (target_dir / "axonpg.mcp.json"),
        env_file=env_file,
    )


def _is_axonpg_container_on_port(port: int) -> bool:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False
    with contextlib.suppress(Exception):
        proc = subprocess.run(
            [docker_bin, "ps", "--filter", "name=^axonpg-db$", "--format", "{{.Ports}}"],
            capture_output=True,
            text=True,
            timeout=1.5,
            check=False,
        )
        if proc.returncode == 0 and f":{port}->" in proc.stdout:
            return True
    return False


def _resolve_db_port(
    console: Any,
    *,
    requested_port: int | None,
    lang: str,
    interactive: bool,
) -> int:
    base = requested_port if requested_port is not None else 5435
    if is_port_available(base) or _is_axonpg_container_on_port(base):
        return base

    suggested = find_available_port(base)
    if not interactive:
        if requested_port is None:
            console.print(f"[yellow]! {t('port_reassigned_warn', lang, old=base, new=suggested)}[/yellow]")
            return suggested
        console.print(f"[yellow]! {t('preflight_port_suggest', lang, port=base, suggested=suggested)}[/yellow]")
        return base

    console.print(f"[yellow]! {t('port_reassigned_warn', lang, old=base, new=suggested)}[/yellow]")
    while True:
        raw = prompt_text(t("prompt_db_port", lang), default=str(suggested))
        try:
            candidate = int(raw)
        except ValueError:
            console.print(f"[yellow]! {t('port_invalid_warn', lang)}[/yellow]")
            continue
        if not (1 <= candidate <= 65535) or (
            not is_port_available(candidate) and not _is_axonpg_container_on_port(candidate)
        ):
            console.print(f"[yellow]! {t('port_invalid_warn', lang)}[/yellow]")
            continue
        return candidate


def _cmd_init(args: argparse.Namespace) -> int:
    console = create_console()
    lang_arg = args.lang or "auto"
    lang = resolve_language(lang_arg if lang_arg != "auto" else None)
    initial_port = args.db_port if args.db_port is not None else 5435

    hermes_home = Path(args.hermes_home).expanduser() if args.hermes_home else None
    openclaw_home = Path(args.openclaw_home).expanduser() if args.openclaw_home else None
    preflight = detect_preflight(
        db_port=initial_port,
        hermes_home=hermes_home,
        openclaw_home=openclaw_home,
    )

    mode = args.mode or "docker"
    agent = args.agent or "hermes"
    embed_prov = args.embed_provider or "fastembed"
    llm_prov = args.llm_provider or "none"
    namespace = args.namespace or "default"
    db_port = initial_port
    embed_api_key = ""
    llm_api_key = ""
    start_docker = bool(args.start_docker)

    try:
        if not args.yes:
            render_banner(console, version=__version__, lang=lang)
            render_preflight(console, preflight, db_port=initial_port, lang=lang)

            total_steps = 6
            step_idx = 1

            if lang_arg == "auto":
                render_step_header(console, step_idx, total_steps, "step_lang", lang)
                lang = select_option(t("choose_lang", lang), _LANGUAGE_OPTIONS, default=lang, lang=lang)
            step_idx += 1

            if not args.mode:
                render_step_header(console, step_idx, total_steps, "step_storage", lang)
                mode = _choose_storage_mode(lang, preflight, default_mode=mode)
            step_idx += 1

            if not args.agent:
                render_step_header(console, step_idx, total_steps, "step_agent", lang)
                render_agent_comparison(console, lang)
                agent = _choose_agent_runtime(lang, preflight, default_agent=agent)
            step_idx += 1

            if not args.embed_provider:
                render_step_header(console, step_idx, total_steps, "step_embed", lang)
                embed_prov = _choose_embed_provider(lang, preflight, default_provider=embed_prov)
            if EMBEDDING_PRESETS[embed_prov].requires_api_key:
                embed_api_key = prompt_text(
                    f"[embed:{embed_prov}] {t('prompt_api_key', lang)}",
                    secret=True,
                )
            step_idx += 1

            if not args.llm_provider:
                render_step_header(console, step_idx, total_steps, "step_llm", lang)
                llm_prov = _choose_llm_provider(lang, default_provider=llm_prov)
            if LLM_PRESETS[llm_prov].requires_api_key:
                if llm_prov == embed_prov and embed_api_key:
                    llm_api_key = embed_api_key
                else:
                    llm_api_key = prompt_text(
                        f"[llm:{llm_prov}] {t('prompt_api_key', lang)}",
                        secret=True,
                    )
            step_idx += 1

            render_step_header(console, step_idx, total_steps, "step_namespace", lang)
            namespace = prompt_text(t("prompt_namespace", lang), default=namespace)
            if mode == "docker":
                db_port = _resolve_db_port(
                    console,
                    requested_port=args.db_port,
                    lang=lang,
                    interactive=True,
                )
            target_dir_probe = Path(args.dir).expanduser().resolve()
            if (
                mode == "docker"
                and not start_docker
                and preflight.docker_running
                and (target_dir_probe / "docker-compose.yml").is_file()
            ):
                start_docker = prompt_confirm(t("prompt_start_docker", lang), default=True)
        elif mode == "docker":
            db_port = _resolve_db_port(
                console,
                requested_port=args.db_port,
                lang=lang,
                interactive=False,
            )
    except KeyboardInterrupt:
        console.print(f"\n[yellow]! {t('setup_cancelled', lang)}[/yellow]")
        return 130

    embed_preset = EMBEDDING_PRESETS[embed_prov]
    llm_preset = LLM_PRESETS[llm_prov]

    target_dir = Path(args.dir).expanduser().resolve()
    sqlite_default = str(target_dir / "axonpg_lite.db") if mode == "lite" else "~/.axonpg/axonpg.db"

    config = AxonConfig(
        mode=mode,
        namespace=namespace,
        lang=lang,
        sqlite_path=args.sqlite_path or sqlite_default,
        db_port=db_port,
        embed_provider=embed_prov,
        embed_model=args.embed_model or embed_preset.default_model,
        embed_dim=args.embed_dim or embed_preset.default_dim,
        embed_api_key=embed_api_key,
        embed_base_url=embed_preset.base_url,
        llm_provider=llm_prov,
        llm_model=args.llm_model or llm_preset.default_model,
        llm_api_key=llm_api_key,
        llm_base_url=llm_preset.base_url,
    )

    with status_spinner(console, t("spin_write_env", lang)):
        env_path = _write_env_file(target_dir, config)

    # Pre-initialize SQLite file when using lite mode so MCP & doctor work immediately
    if config.mode == "lite":
        from axonpg.memory.sqlite_store import SQLiteMemoryStore
        from axonpg.providers.embeddings import NullEmbeddingProvider

        with status_spinner(console, t("spin_init_sqlite", lang)):
            lite_store = SQLiteMemoryStore(
                db_path=config.resolved_sqlite_path(),
                embedder=NullEmbeddingProvider(dim=config.embed_dim),
            )
            lite_store.close()
    elif start_docker and (target_dir / "docker-compose.yml").is_file():
        with status_spinner(console, t("spin_start_docker", lang)):
            subprocess.run(["docker", "compose", "up", "-d"], cwd=target_dir, capture_output=True, check=False)
            ready = _wait_for_docker_postgres(target_dir, config)
        if not ready:
            console.print(f"[yellow]! {t('docker_wait_warn', lang)}[/yellow]")

    with status_spinner(console, t("spin_config_agent", lang)):
        adapter_res = _apply_agent_adapter(
            agent=agent,
            config=config,
            env_file=env_path,
            target_dir=target_dir,
            hermes_home=hermes_home,
            openclaw_home=openclaw_home,
        )

    render_init_summary(
        console,
        lang=lang,
        config=config,
        env_path=env_path,
        adapter_res=adapter_res,
        hermes_existed_before=preflight.hermes_exists,
    )

    if not args.yes and (config.mode == "lite" or start_docker):
        with contextlib.suppress(KeyboardInterrupt):
            if prompt_confirm(t("prompt_run_doctor", lang), default=True):
                return _run_doctor(config, lang=lang, as_json=False, console=console)
    return 0


def _check_docker_status() -> tuple[bool, str]:
    docker_bin = shutil.which("docker")
    if not docker_bin:
        return False, "Docker CLI missing (or use AXONPG_MODE=lite)"
    try:
        proc = subprocess.run(
            [docker_bin, "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
    except Exception as exc:
        return False, f"Docker check failed: {exc}"
    if proc.returncode != 0:
        return False, "Docker CLI found, but daemon is stopped (start Docker Desktop / dockerd)"
    ver = proc.stdout.strip()
    return True, f"CLI + daemon ready (v{ver})" if ver else f"ok ({docker_bin})"


def _check_embeddings_status(cfg: AxonConfig) -> tuple[bool, str, Any]:
    preset = EMBEDDING_PRESETS.get(cfg.embed_provider)
    if preset and preset.requires_api_key and not cfg.embed_api_key:
        return (
            False,
            f"AXONPG_EMBED_API_KEY missing for '{cfg.embed_provider}' (set it in .env)",
            None,
        )
    embedder = build_embedding_provider(cfg)
    emb_ok, emb_msg = embedder.healthcheck()
    if not emb_ok and cfg.embed_provider == "fastembed" and "fastembed" in emb_msg.lower():
        emb_msg = f'{emb_msg} — install via: pip install "axonpg[local]"'
    return emb_ok, emb_msg, embedder


def _check_llm_status(cfg: AxonConfig) -> tuple[bool, str]:
    if cfg.llm_provider == "none":
        return True, "deterministic offline synthesis ($0)"
    preset = LLM_PRESETS.get(cfg.llm_provider)
    if preset and preset.requires_api_key and not cfg.llm_api_key:
        return False, f"AXONPG_LLM_API_KEY missing for '{cfg.llm_provider}' (set it in .env)"
    if cfg.llm_provider == "ollama":
        base = (cfg.llm_base_url or "http://127.0.0.1:11434").rstrip("/")
        try:
            resp = httpx.get(f"{base}/api/tags", timeout=1.5)
            if resp.status_code >= 400:
                return False, f"Ollama returned HTTP {resp.status_code} at {base}"
            return True, f"ok (ollama · {cfg.llm_model})"
        except Exception:
            return False, f"Ollama unreachable at {base} (run: ollama serve)"
    return True, f"configured ({cfg.llm_provider} · {cfg.llm_model})"


def _check_database_status(cfg: AxonConfig, embedder: Any) -> tuple[bool, str, dict[str, Any] | None]:
    active_embedder = embedder or build_embedding_provider(cfg)
    try:
        store = build_memory_store(cfg, embedder=active_embedder)
        try:
            db_ok, db_msg = store.healthcheck()
            stats_payload = store.stats(cfg.namespace).model_dump() if db_ok else None
            return db_ok, db_msg, stats_payload
        finally:
            store.close()
    except Exception as exc:
        detail = str(exc).strip()
        if cfg.mode == "docker":
            import socket

            port_open = False
            with contextlib.suppress(OSError), socket.create_connection((cfg.db_host, cfg.db_port), timeout=0.5):
                port_open = True
            if not port_open:
                detail = f"No PostgreSQL listening on {cfg.db_host}:{cfg.db_port} — run: axonpg up"
            else:
                detail = f"{detail} — port {cfg.db_port} is open; verify DB name/user/password or container logs"
        return False, detail, None


def _check_agent_wiring() -> tuple[str, bool, str]:
    rows = _inspect_agent_rows(
        target_dir=Path.cwd(),
        hermes_home=None,
        openclaw_home=None,
    )
    wired = [r for r in rows if r["configured"]]
    if not wired:
        return ("Agent MCP", True, "none in default paths (run `axonpg init` or `axonpg connect` when ready)")

    names = ", ".join(r["agent"] for r in wired)
    broken: list[str] = []
    for r in wired:
        cmd = r.get("command")
        if cmd and not Path(cmd).is_file() and not shutil.which(str(cmd)):
            broken.append(f"{r['agent']} -> missing binary '{cmd}'")
    if broken:
        return ("Agent MCP", False, "; ".join(broken))
    return ("Agent MCP", True, f"registered for {names}")


def _run_doctor(cfg: AxonConfig, *, lang: str, as_json: bool = False, console: Any = None) -> int:
    active_console = console or create_console()
    checks: list[tuple[str, bool, str]] = []
    stats_payload: dict[str, Any] | None = None
    all_ok = True

    def _collect() -> None:
        nonlocal stats_payload, all_ok
        if cfg.mode == "docker":
            d_ok, d_msg = _check_docker_status()
            checks.append(("Docker Engine", d_ok, d_msg))
            all_ok = all_ok and d_ok

        emb_ok, emb_msg, embedder = _check_embeddings_status(cfg)
        checks.append(("Embeddings", emb_ok, emb_msg))
        all_ok = all_ok and emb_ok

        llm_ok, llm_msg = _check_llm_status(cfg)
        checks.append(("Consolidation LLM", llm_ok, llm_msg))
        all_ok = all_ok and llm_ok

        db_ok, db_msg, stats_payload = _check_database_status(cfg, embedder)
        checks.append(("Database", db_ok, db_msg))
        all_ok = all_ok and db_ok

        ag_name, ag_ok, ag_msg = _check_agent_wiring()
        checks.append((ag_name, ag_ok, ag_msg))
        all_ok = all_ok and ag_ok

    if as_json:
        _collect()
        payload = {
            "version": __version__,
            "ok": all_ok,
            "config": cfg.redacted_dict(),
            "checks": [{"component": c, "ok": ok, "detail": d} for c, ok, d in checks],
            "stats": stats_payload,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        with status_spinner(active_console, t("spin_healthcheck", lang)):
            _collect()
        render_doctor_report(
            active_console,
            version=__version__,
            lang=lang,
            cfg=cfg,
            checks=checks,
            stats_payload=stats_payload,
            all_ok=all_ok,
        )
    return 0 if all_ok else 1


def _load_runtime(args: argparse.Namespace) -> tuple[AxonConfig, str, str]:
    env_source = _load_env_if_present(getattr(args, "env_file", None))
    cfg = AxonConfig.from_env()
    if getattr(args, "namespace", None):
        cfg = cfg.model_copy(update={"namespace": args.namespace})
    lang = resolve_language(getattr(args, "lang", None) or cfg.lang)
    return cfg, lang, env_source


def _cmd_doctor(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    return _run_doctor(cfg, lang=lang, as_json=bool(args.json))


def _cmd_up(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    console = create_console()
    target_dir = Path(args.dir).expanduser().resolve()
    compose_file = target_dir / "docker-compose.yml"

    if not compose_file.is_file():
        msg = f"docker-compose.yml not found in {target_dir}"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, indent=2))
        else:
            console.print(f"[red]✗[/red] {msg}")
        return 1

    if not shutil.which("docker"):
        msg = "Docker CLI not found on PATH."
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, indent=2))
        else:
            console.print(f"[red]✗[/red] {msg}")
        return 1

    spinner_ctx = contextlib.nullcontext() if args.json else status_spinner(console, t("spin_start_docker", lang))
    with spinner_ctx:
        up_proc = subprocess.run(
            ["docker", "compose", "up", "-d"],
            cwd=target_dir,
            capture_output=True,
            text=True,
            check=False,
        )
        ready = up_proc.returncode == 0 and _wait_for_docker_postgres(target_dir, cfg)

    if args.json:
        print(json.dumps({"ok": ready, "returncode": up_proc.returncode, "dir": str(target_dir)}, indent=2))
    elif ready:
        console.print(f"[bold green]✓[/bold green] {t('up_ok', lang)}")
    else:
        err = up_proc.stderr.strip() or t("docker_wait_warn", lang)
        console.print(f"[yellow]![/yellow] {err}")
    return 0 if ready else 1


def _cmd_down(args: argparse.Namespace) -> int:
    _, lang, _ = _load_runtime(args)
    console = create_console()
    target_dir = Path(args.dir).expanduser().resolve()
    if not (target_dir / "docker-compose.yml").is_file():
        msg = f"docker-compose.yml not found in {target_dir}"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, indent=2))
        else:
            console.print(f"[red]✗[/red] {msg}")
        return 1

    cmd = ["docker", "compose", "down"]
    if args.volumes:
        cmd.append("-v")
    proc = subprocess.run(cmd, cwd=target_dir, capture_output=True, text=True, check=False)
    ok = proc.returncode == 0
    if args.json:
        print(json.dumps({"ok": ok, "volumes_removed": bool(args.volumes), "dir": str(target_dir)}, indent=2))
    elif ok:
        console.print(f"[bold green]✓[/bold green] {t('down_ok', lang)}")
    else:
        console.print(f"[red]✗[/red] {proc.stderr.strip() or 'docker compose down failed'}")
    return 0 if ok else 1


def _cmd_config(args: argparse.Namespace) -> int:
    cfg, lang, env_source = _load_runtime(args)
    if args.json:
        print(json.dumps({"env_source": env_source, "config": cfg.redacted_dict()}, indent=2, ensure_ascii=False))
    else:
        render_config_table(create_console(), lang=lang, cfg=cfg, env_source=env_source)
    return 0


def _cmd_recall(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    store = build_memory_store(cfg)
    try:
        res = store.recall(
            query=args.query,
            namespace=cfg.namespace,
            limit=args.limit,
            max_tokens=args.max_tokens,
            include_episodes=not args.no_episodes,
        )
        payload = res.model_dump()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            render_recall_result(create_console(), lang=lang, payload=payload)
    finally:
        store.close()
    return 0


def _cmd_foresights(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    store = build_memory_store(cfg)
    try:
        if args.complete:
            res = store.complete_foresight(foresight_id=args.complete, namespace=cfg.namespace)
            if args.json:
                print(json.dumps(res, indent=2, ensure_ascii=False))
            else:
                console = create_console()
                if res.get("updated"):
                    console.print(f"[green]✓[/green] Foresight {args.complete} marked done.")
                else:
                    console.print(f"[yellow]![/yellow] Foresight {args.complete} not found in '{cfg.namespace}'.")
                    return 1
            return 0 if res.get("updated") else 1

        items = store.list_foresights(namespace=cfg.namespace, status=args.status, limit=args.limit)
        if args.json:
            print(json.dumps({"namespace": cfg.namespace, "status": args.status, "foresights": items}, indent=2, ensure_ascii=False))
        else:
            render_foresights_table(create_console(), lang=lang, items=items)
    finally:
        store.close()
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    _load_env_if_present(args.env_file)
    from axonpg.mcp_server import run_mcp_server

    run_mcp_server(transport=args.transport)
    return 0


def _cmd_connect(args: argparse.Namespace) -> int:
    cfg, _, _ = _load_runtime(args)
    target_dir = Path(args.dir).expanduser().resolve()
    env_path = _resolve_env_path(args.env_file, base_dir=target_dir)
    hermes_home = Path(args.hermes_home).expanduser() if args.hermes_home else None
    openclaw_home = Path(args.openclaw_home).expanduser() if args.openclaw_home else None
    generic_out = Path(args.output).expanduser().resolve() if args.output else None

    res = _apply_agent_adapter(
        agent=args.agent,
        config=cfg,
        env_file=env_path,
        target_dir=target_dir,
        hermes_home=hermes_home,
        openclaw_home=openclaw_home,
        generic_output=generic_out,
    )
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        console = create_console()
        cfg_file = res.get("config_path") or res.get("mcp_config_path") or "-"
        console.print(f"[bold green]✓[/bold green] Connected [cyan]{args.agent}[/cyan] → {cfg_file}")
        if res.get("skill_path"):
            console.print(f"  [dim]SKILL.md:[/dim] {res['skill_path']}")
    return 0


def _read_mcp_command(entry: Any) -> str | None:
    if isinstance(entry, dict) and entry.get("command"):
        return str(entry["command"])
    return None


def _inspect_agent_rows(
    *,
    target_dir: Path,
    hermes_home: Path | None,
    openclaw_home: Path | None,
) -> list[dict[str, Any]]:
    h_dir = (hermes_home or (Path.home() / ".hermes")).expanduser()
    h_cfg = h_dir / "config.yaml"
    h_configured = False
    h_cmd: str | None = None
    if h_cfg.is_file():
        with contextlib.suppress(Exception):
            loaded = yaml.safe_load(h_cfg.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("mcp_servers"), dict) and "axonpg" in loaded["mcp_servers"]:
                h_configured = True
                h_cmd = _read_mcp_command(loaded["mcp_servers"]["axonpg"])

    oc_dir = (openclaw_home or (Path.home() / ".openclaw")).expanduser()
    oc_cfg = oc_dir / "mcp" / "axonpg.json"
    oc_configured = False
    oc_cmd: str | None = None
    if oc_cfg.is_file():
        with contextlib.suppress(Exception):
            loaded = json.loads(oc_cfg.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("mcpServers"), dict) and "axonpg" in loaded["mcpServers"]:
                oc_configured = True
                oc_cmd = _read_mcp_command(loaded["mcpServers"]["axonpg"])

    gen_cfg = target_dir / "axonpg.mcp.json"
    gen_configured = False
    gen_cmd: str | None = None
    if gen_cfg.is_file():
        with contextlib.suppress(Exception):
            loaded = json.loads(gen_cfg.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and isinstance(loaded.get("mcpServers"), dict) and "axonpg" in loaded["mcpServers"]:
                gen_configured = True
                gen_cmd = _read_mcp_command(loaded["mcpServers"]["axonpg"])

    return [
        {"agent": "Hermes Agent", "detected": h_dir.is_dir(), "configured": h_configured, "command": h_cmd, "path": str(h_cfg)},
        {"agent": "OpenClaw", "detected": oc_dir.is_dir(), "configured": oc_configured, "command": oc_cmd, "path": str(oc_cfg)},
        {"agent": "Generic MCP", "detected": gen_cfg.is_file(), "configured": gen_configured, "command": gen_cmd, "path": str(gen_cfg)},
    ]


def _cmd_agents(args: argparse.Namespace) -> int:
    _, lang, _ = _load_runtime(args)
    target_dir = Path(args.dir).expanduser().resolve()
    hermes_home = Path(args.hermes_home).expanduser() if args.hermes_home else None
    openclaw_home = Path(args.openclaw_home).expanduser() if args.openclaw_home else None
    rows = _inspect_agent_rows(target_dir=target_dir, hermes_home=hermes_home, openclaw_home=openclaw_home)
    if args.json:
        print(json.dumps({"agents": rows}, indent=2, ensure_ascii=False))
    else:
        render_agents_table(create_console(), lang=lang, rows=rows)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    out_path = Path(args.output).expanduser().resolve()
    store = build_memory_store(cfg)
    try:
        snapshot = store.export_namespace(namespace=cfg.namespace)
    finally:
        store.close()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(snapshot, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    counts = {k: len(snapshot.get(k, [])) for k in ("entities", "relations", "episodes", "facts", "foresights")}

    if args.json:
        print(json.dumps({"namespace": cfg.namespace, "output": str(out_path), "counts": counts}, indent=2))
    else:
        render_transfer_counts(
            create_console(),
            title=t("export_done", lang, ns=cfg.namespace, path=str(out_path)),
            counts=counts,
        )
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    console = create_console()
    in_path = Path(args.input).expanduser().resolve()
    if not in_path.is_file():
        msg = f"Snapshot file not found: {in_path}"
        if args.json:
            print(json.dumps({"ok": False, "error": msg}, indent=2))
        else:
            console.print(f"[red]✗[/red] {msg}")
        return 1

    try:
        payload = json.loads(in_path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
            raise ValueError("Invalid AxonPG snapshot format (expected JSON object with schema_version=1).")
    except Exception as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            console.print(f"[red]✗[/red] {exc}")
        return 1

    target_ns = args.namespace or cfg.namespace
    store = build_memory_store(cfg)
    try:
        counts = store.import_namespace(payload, target_namespace=target_ns)
    finally:
        store.close()

    if args.json:
        print(json.dumps({"ok": True, "namespace": target_ns, "counts": counts}, indent=2))
    else:
        render_transfer_counts(create_console(), title=t("import_done", lang, ns=target_ns), counts=counts)
    return 0


def _cmd_reset(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    console = create_console()
    if not args.yes:
        try:
            confirmed = prompt_confirm(t("confirm_reset", lang, ns=cfg.namespace), default=False)
        except KeyboardInterrupt:
            console.print(f"\n[yellow]! {t('setup_cancelled', lang)}[/yellow]")
            return 130
        if not confirmed:
            console.print(f"[yellow]! {t('setup_cancelled', lang)}[/yellow]")
            return 130

    store = build_memory_store(cfg)
    try:
        counts = store.reset_namespace(namespace=cfg.namespace)
    finally:
        store.close()

    if args.json:
        print(json.dumps({"ok": True, "namespace": cfg.namespace, "deleted": counts}, indent=2))
    else:
        render_transfer_counts(console, title=t("reset_done", lang, ns=cfg.namespace), counts=counts)
    return 0


def _cmd_consolidate(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    store = build_memory_store(cfg)
    llm = build_llm_provider(cfg)
    try:
        report = consolidate_namespace(store=store, llm=llm, namespace=cfg.namespace, batch_limit=args.limit)
        payload = report.model_dump()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            render_consolidation_report(create_console(), lang=lang, report_payload=payload)
    finally:
        store.close()
    return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    cfg, lang, _ = _load_runtime(args)
    store = build_memory_store(cfg)
    try:
        payload = store.stats(cfg.namespace).model_dump()
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            render_stats_report(create_console(), lang=lang, stats_payload=payload)
    finally:
        store.close()
    return 0


class _RootHelpAction(argparse.Action):
    """Display Rich grouped CLI help honoring `--lang` when `-h`/`--help` is requested on root."""

    def __init__(self, option_strings: Sequence[str], dest: str = argparse.SUPPRESS, **kwargs: Any):
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        lang_hint = getattr(namespace, "lang", None)
        render_root_help(
            create_console(),
            version=__version__,
            lang=resolve_language(lang_hint),
            groups=HELP_GROUPS,
        )
        parser.exit(0)


class _SubcommandHelpAction(argparse.Action):
    """Display Rich styled subcommand help on `-h`/`--help`."""

    def __init__(self, option_strings: Sequence[str], dest: str = argparse.SUPPRESS, **kwargs: Any):
        super().__init__(option_strings=option_strings, dest=dest, nargs=0, **kwargs)

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        lang_hint = getattr(namespace, "lang", None)
        render_subcommand_help(
            create_console(),
            parser=parser,
            lang=resolve_language(lang_hint),
        )
        parser.exit(0)


def _add_global_flags(target: argparse.ArgumentParser, *, suppress_defaults: bool, include_sub_help: bool = False) -> None:
    d_none: Any = argparse.SUPPRESS if suppress_defaults else None
    d_false: Any = argparse.SUPPRESS if suppress_defaults else False
    if include_sub_help:
        target.add_argument("-h", "--help", action=_SubcommandHelpAction, help="Show help for this command and exit")
    target.add_argument("--env-file", default=d_none, help="Load settings from a specific .env path")
    target.add_argument("--namespace", default=d_none, help="Target memory namespace")
    target.add_argument("--lang", choices=["auto", *SUPPORTED_LANGUAGES], default=d_none, help="UI language")
    target.add_argument("--json", action="store_true", default=d_false, help="Emit machine-readable JSON")


def build_parser() -> argparse.ArgumentParser:
    """Create the root AxonPG CLI parser with grouped Rich help and uniform global flags."""
    shared_parent = argparse.ArgumentParser(add_help=False)
    _add_global_flags(shared_parent, suppress_defaults=True, include_sub_help=True)

    parser = argparse.ArgumentParser(
        prog="axonpg",
        add_help=False,
        description="AxonPG — Postgres-native multilingual cognitive memory engine for AI agents",
    )
    parser.add_argument("-h", "--help", action=_RootHelpAction, help="Show grouped help message and exit")
    parser.add_argument("-V", "--version", action="version", version=f"%(prog)s {__version__}")
    _add_global_flags(parser, suppress_defaults=False)

    sub = parser.add_subparsers(dest="command", required=False)

    p_init = sub.add_parser("init", aliases=["setup"], add_help=False, parents=[shared_parent], description="Guided setup assistant (.env, DB schema & agent wiring)")
    p_init.add_argument("--mode", choices=["docker", "lite"], default=None, help="Storage backend mode")
    p_init.add_argument("--agent", choices=["hermes", "openclaw", "generic"], default=None, help="Target AI agent runtime")
    p_init.add_argument("--embed-provider", choices=list(EMBEDDING_PRESETS), default=None, help="Embedding engine preset")
    p_init.add_argument("--embed-model", default=None, help="Override embedding model name")
    p_init.add_argument("--embed-dim", type=int, default=None, help="Override vector dimension")
    p_init.add_argument("--llm-provider", choices=list(LLM_PRESETS), default=None, help="Sleep-time consolidation LLM provider")
    p_init.add_argument("--llm-model", default=None, help="Override consolidation LLM model")
    p_init.add_argument("--db-port", type=int, default=None, help="PostgreSQL host port (auto-scans from 5435 if busy)")
    p_init.add_argument("--sqlite-path", default=None, help="SQLite DB file path when using --mode lite")
    p_init.add_argument("--dir", default=".", help="Project/config directory (default: current folder)")
    p_init.add_argument("--hermes-home", default=None, help="Custom Hermes config home (default: ~/.hermes)")
    p_init.add_argument("--openclaw-home", default=None, help="Custom OpenClaw config home (default: ~/.openclaw)")
    p_init.add_argument("--start-docker", action="store_true", help="Start PostgreSQL Docker container immediately")
    p_init.add_argument("-y", "--yes", action="store_true", help="Non-interactive mode using flags/defaults")
    p_init.set_defaults(func=_cmd_init)

    p_doc = sub.add_parser("doctor", add_help=False, parents=[shared_parent], description="Diagnose Docker, API keys, embeddings, LLM & database health")
    p_doc.set_defaults(func=_cmd_doctor)

    p_up = sub.add_parser("up", add_help=False, parents=[shared_parent], description="Start PostgreSQL + pgvector container in the background")
    p_up.add_argument("--dir", default=".", help="Directory containing docker-compose.yml")
    p_up.set_defaults(func=_cmd_up)

    p_down = sub.add_parser("down", add_help=False, parents=[shared_parent], description="Stop the PostgreSQL Docker Compose stack")
    p_down.add_argument("--dir", default=".", help="Directory containing docker-compose.yml")
    p_down.add_argument("-v", "--volumes", action="store_true", help="Also remove persistent Docker data volumes")
    p_down.set_defaults(func=_cmd_down)

    p_cfg = sub.add_parser("config", add_help=False, parents=[shared_parent], description="View current settings with secrets safely hidden")
    p_cfg.set_defaults(func=_cmd_config)

    p_rec = sub.add_parser("recall", add_help=False, parents=[shared_parent], description="Search what the agent remembers (Graph + Semantic + Lessons)")
    p_rec.add_argument("query", help="Search query text in any language")
    p_rec.add_argument("--limit", type=int, default=5, help="Max items per layer (default: 5)")
    p_rec.add_argument("--max-tokens", type=int, default=800, help="Token budget for assembled prompt context (default: 800)")
    p_rec.add_argument("--no-episodes", action="store_true", help="Exclude episodic traces from recall")
    p_rec.set_defaults(func=_cmd_recall)

    p_st = sub.add_parser("stats", add_help=False, parents=[shared_parent], description="Show stored memory counts across all layers")
    p_st.set_defaults(func=_cmd_stats)

    p_con = sub.add_parser("consolidate", add_help=False, parents=[shared_parent], description="Compress & organize recent episodes into durable knowledge")
    p_con.add_argument("--limit", type=int, default=50, help="Max pending episodes to process per batch (default: 50)")
    p_con.set_defaults(func=_cmd_consolidate)

    p_fs = sub.add_parser("foresights", add_help=False, parents=[shared_parent], description="List upcoming reminders or mark one completed")
    p_fs.add_argument("--status", choices=["active", "done", "expired", "all"], default="active", help="Filter reminders by status")
    p_fs.add_argument("--limit", type=int, default=50, help="Max reminders to return (default: 50)")
    p_fs.add_argument("--complete", default=None, help="Mark a foresight ID as done")
    p_fs.set_defaults(func=_cmd_foresights)

    p_srv = sub.add_parser("serve", add_help=False, parents=[shared_parent], description="Start the MCP server so AI agents can connect")
    p_srv.add_argument("--transport", choices=["stdio", "streamable-http"], default="stdio", help="MCP transport protocol")
    p_srv.set_defaults(func=_cmd_serve)

    p_conn = sub.add_parser("connect", add_help=False, parents=[shared_parent], description="Connect another agent using active .env")
    p_conn.add_argument("agent", choices=["hermes", "openclaw", "generic"], help="Agent runtime to configure")
    p_conn.add_argument("--dir", default=".", help="Workspace directory (default: current folder)")
    p_conn.add_argument("--hermes-home", default=None, help="Custom Hermes home directory")
    p_conn.add_argument("--openclaw-home", default=None, help="Custom OpenClaw home directory")
    p_conn.add_argument("--output", default=None, help="Output file path for generic MCP JSON")
    p_conn.set_defaults(func=_cmd_connect)

    p_ag = sub.add_parser("agents", add_help=False, parents=[shared_parent], description="Check which AI agents are detected and connected to AxonPG")
    p_ag.add_argument("--dir", default=".", help="Directory to check for axonpg.mcp.json")
    p_ag.add_argument("--hermes-home", default=None, help="Custom Hermes home directory")
    p_ag.add_argument("--openclaw-home", default=None, help="Custom OpenClaw home directory")
    p_ag.set_defaults(func=_cmd_agents)

    p_exp = sub.add_parser("export", add_help=False, parents=[shared_parent], description="Back up a memory namespace to a portable file")
    p_exp.add_argument("output", help="Destination JSON file path")
    p_exp.set_defaults(func=_cmd_export)

    p_imp = sub.add_parser("import", add_help=False, parents=[shared_parent], description="Restore a backup file into a namespace (re-indexes automatically)")
    p_imp.add_argument("input", help="Source JSON snapshot path")
    p_imp.set_defaults(func=_cmd_import)

    p_rst = sub.add_parser("reset", add_help=False, parents=[shared_parent], description="Erase all memories inside a target namespace")
    p_rst.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompt")
    p_rst.set_defaults(func=_cmd_reset)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for `axonpg`."""
    args_list = list(argv) if argv is not None else sys.argv[1:]
    parser = build_parser()
    if not args_list:
        render_root_help(
            create_console(),
            version=__version__,
            lang=resolve_language(),
            groups=HELP_GROUPS,
        )
        return 0
    args = parser.parse_args(args_list)
    if not getattr(args, "func", None):
        render_root_help(
            create_console(),
            version=__version__,
            lang=resolve_language(getattr(args, "lang", None)),
            groups=HELP_GROUPS,
        )
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
