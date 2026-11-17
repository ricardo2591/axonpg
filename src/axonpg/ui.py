"""Modern Rich + Questionary terminal UI helpers for the AxonPG CLI."""

from __future__ import annotations

import contextlib
import getpass
import importlib.util
import os
import shutil
import socket
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeVar

import questionary
from prompt_toolkit.styles import Style as PTStyle
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from axonpg.config import AxonConfig
from axonpg.i18n import t

AXON_LOGO = r"""    ___                       ____  ______
   /   |  _  ______  ____    / __ \/ ____/
  / /| | | |/_/ __ \/ __ \  / /_/ / / __
 / ___ |_>  </ /_/ / / / / / ____/ /_/ /
/_/  |_/_/|_|\____/_/ /_/ /_/    \____/"""

AXON_Q_STYLE = PTStyle.from_dict(
    {
        "qmark": "fg:#22d3ee bold",
        "question": "fg:#f8fafc bold",
        "answer": "fg:#10b981 bold",
        "pointer": "fg:#22d3ee bold",
        "highlighted": "fg:#22d3ee bold",
        "selected": "fg:#10b981",
        "instruction": "fg:#94a3b8 italic",
        "text": "fg:#e2e8f0",
        "disabled": "fg:#64748b italic",
    }
)


@dataclass(frozen=True)
class SelectOption:
    """Single interactive menu option with value, label, and metadata badge."""

    value: str
    label: str
    description: str = ""
    recommended: bool = False


@dataclass(frozen=True)
class PreflightStatus:
    """Snapshot of local runtime capabilities detected before setup begins."""

    docker_cli: bool
    docker_running: bool
    docker_version: str
    port_free: bool
    suggested_port: int
    fastembed_installed: bool
    hermes_exists: bool
    openclaw_exists: bool
    ollama_reachable: bool


def configure_stdio_utf8() -> None:
    """Ensure UTF-8 output on Windows consoles so Unicode UI borders render cleanly."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            with contextlib.suppress(Exception):
                reconfigure(encoding="utf-8", errors="replace")


def is_interactive_tty() -> bool:
    """Return True only when running inside an interactive terminal (never in CI pipes)."""
    if os.getenv("CI"):
        return False
    return bool(getattr(sys.stdin, "isatty", lambda: False)() and getattr(sys.stdout, "isatty", lambda: False)())


def create_console() -> Console:
    """Create a Rich Console respecting NO_COLOR and Windows console capabilities."""
    configure_stdio_utf8()
    return Console(highlight=False)


def _probe_tcp_open(host: str, port: int, timeout: float = 0.25) -> bool:
    with contextlib.suppress(OSError), socket.create_connection((host, port), timeout=timeout):
        return True
    return False


def is_port_available(port: int) -> bool:
    """Return True if a local TCP port accepts a fresh exclusive bind on IPv4/IPv6."""
    if not (1 <= port <= 65535):
        return False
    if _probe_tcp_open("127.0.0.1", port, timeout=0.15) or _probe_tcp_open("::1", port, timeout=0.15):
        return False
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            sock.bind(("0.0.0.0", port))
        return True
    except OSError:
        return False


def find_available_port(preferred: int = 5435, max_scan: int = 50) -> int:
    """Scan upward from `preferred` and return the first free local TCP port."""
    for candidate in range(preferred, min(65536, preferred + max_scan + 1)):
        if is_port_available(candidate):
            return candidate
    return preferred


def detect_preflight(
    db_port: int = 5435,
    hermes_home: Path | None = None,
    openclaw_home: Path | None = None,
) -> PreflightStatus:
    """Inspect Docker, target port, local embedders, and existing agent homes quickly."""
    docker_bin = shutil.which("docker")
    docker_running = False
    docker_version = ""
    if docker_bin:
        with contextlib.suppress(Exception):
            proc = subprocess.run(
                [docker_bin, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if proc.returncode == 0:
                docker_running = True
                docker_version = proc.stdout.strip()

    hermes_dir = (hermes_home or (Path.home() / ".hermes")).expanduser()
    openclaw_dir = (openclaw_home or (Path.home() / ".openclaw")).expanduser()
    port_free = is_port_available(db_port)
    suggested_port = db_port if port_free else find_available_port(db_port)

    return PreflightStatus(
        docker_cli=bool(docker_bin),
        docker_running=docker_running,
        docker_version=docker_version,
        port_free=port_free,
        suggested_port=suggested_port,
        fastembed_installed=importlib.util.find_spec("fastembed") is not None,
        hermes_exists=hermes_dir.is_dir(),
        openclaw_exists=openclaw_dir.is_dir(),
        ollama_reachable=_probe_tcp_open("127.0.0.1", 11434),
    )


def render_banner(console: Console, version: str, lang: str) -> None:
    """Render the branded AxonPG header card."""
    body = Text()
    body.append(AXON_LOGO, style="bold cyan")
    body.append(f"   v{version}\n", style="bold bright_green")
    body.append(f"  {t('tagline', lang)}", style="bright_white")
    console.print()
    console.print(
        Panel(
            body,
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 2),
        )
    )


def render_preflight(console: Console, preflight: PreflightStatus, db_port: int, lang: str) -> None:
    """Display detected environment capabilities before interactive prompts."""
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False, expand=True)
    table.add_column("k1", style="bold white", no_wrap=True)
    table.add_column("v1")
    table.add_column("k2", style="bold white", no_wrap=True)
    table.add_column("v2")

    if preflight.docker_running:
        ver = f" ({preflight.docker_version})" if preflight.docker_version else ""
        docker_cell = f"[green]●[/green] {t('preflight_docker_ready', lang)}{ver}"
    elif preflight.docker_cli:
        docker_cell = f"[yellow]●[/yellow] {t('preflight_docker_stopped', lang)}"
    else:
        docker_cell = f"[dim]○[/dim] {t('preflight_docker_missing', lang)}"

    port_cell = (
        f"[green]●[/green] {t('preflight_port_free', lang, port=db_port)}"
        if preflight.port_free
        else f"[yellow]●[/yellow] {t('preflight_port_suggest', lang, port=db_port, suggested=preflight.suggested_port)}"
    )

    embed_cell = (
        f"[green]●[/green] {t('preflight_ready', lang)} (fastembed)"
        if preflight.fastembed_installed
        else f"[dim]○[/dim] {t('preflight_optional', lang)}"
    )

    agents_parts = [
        f"Hermes {'[green]●[/green]' if preflight.hermes_exists else '[dim]○[/dim]'}",
        f"OpenClaw {'[green]●[/green]' if preflight.openclaw_exists else '[dim]○[/dim]'}",
    ]
    agents_cell = "  ".join(agents_parts)

    table.add_row(t("preflight_docker", lang), docker_cell, t("preflight_port", lang), port_cell)
    table.add_row(t("preflight_fastembed", lang), embed_cell, "Agents", agents_cell)

    console.print(
        Panel(
            table,
            title=f"[bold cyan]{t('preflight_title', lang)}[/bold cyan]",
            title_align="left",
            box=box.ROUNDED,
            border_style="bright_black",
            padding=(0, 1),
        )
    )


def render_step_header(console: Console, step: int, total: int, title_key: str, lang: str) -> None:
    """Render a crisp section divider indicating wizard progress."""
    label = f"[bold cyan]{t('step_of', lang, step=step, total=total)}[/bold cyan] [dim]·[/dim] [bold white]{t(title_key, lang)}[/bold white]"
    console.print()
    console.print(Rule(title=label, align="left", style="bright_black"))


def render_agent_comparison(console: Console, lang: str) -> None:
    """Display a side-by-side comparison table of supported MCP agent runtimes."""
    table = Table(
        title=f"[bold white]{t('agent_compare_title', lang)}[/bold white]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold cyan",
        expand=True,
    )
    table.add_column(t("col_runtime", lang), style="bold bright_white", no_wrap=True)
    table.add_column(t("col_stack", lang), style="cyan", no_wrap=True)
    table.add_column(t("col_strength", lang), style="white")
    table.add_column(t("col_integration", lang), style="dim", overflow="fold")

    table.add_row(
        f"Hermes Agent [green]({t('recommended_tag', lang)})[/green]",
        t("agent_hermes_stack", lang),
        t("agent_hermes_strength", lang),
        t("agent_hermes_integration", lang),
    )
    table.add_row(
        "OpenClaw",
        t("agent_openclaw_stack", lang),
        t("agent_openclaw_strength", lang),
        t("agent_openclaw_integration", lang),
    )
    table.add_row(
        "Generic MCP",
        t("agent_generic_stack", lang),
        t("agent_generic_strength", lang),
        t("agent_generic_integration", lang),
    )
    console.print(table)


def _format_choice_title(opt: SelectOption, lang: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = [("class:text bold", f"{opt.label:<12}")]
    if opt.recommended:
        tokens.append(("class:answer", f" [{t('recommended_tag', lang)}]"))
    if opt.description:
        tokens.append(("class:instruction", f"  {opt.description}"))
    return tokens


def _plain_select_fallback(message: str, options: list[SelectOption], default: str) -> str:
    print(f"\n{message}")
    default_idx = 1
    for idx, opt in enumerate(options, start=1):
        marker = "*" if opt.value == default else " "
        desc = f" — {opt.description}" if opt.description else ""
        print(f"  {marker} {idx}) {opt.label}{desc}")
        if opt.value == default:
            default_idx = idx
    raw = input(f"> [{default_idx}]: ").strip()
    if not raw:
        return default
    if raw.isdigit():
        pos = int(raw) - 1
        if 0 <= pos < len(options):
            return options[pos].value
    lowered = raw.lower()
    for opt in options:
        if opt.value.lower() == lowered:
            return opt.value
    return default


_T = TypeVar("_T")


def _ask_with_fallback(prompt_fn: Callable[[], _T | None], fallback_fn: Callable[[], _T]) -> _T:
    """Execute an interactive questionary prompt with non-TTY / non-console fallback."""
    if not is_interactive_tty():
        return fallback_fn()
    try:
        answer = prompt_fn()
    except Exception as exc:
        if type(exc).__name__ == "NoConsoleScreenBufferError":
            return fallback_fn()
        raise
    if answer is None:
        raise KeyboardInterrupt
    return answer


def select_option(
    message: str,
    options: list[SelectOption],
    default: str,
    lang: str,
) -> str:
    """Prompt the user with arrow-key selection in TTY, or numbered fallback otherwise."""
    choices = [
        questionary.Choice(title=_format_choice_title(opt, lang), value=opt.value)
        for opt in options
    ]

    def _ask() -> str | None:
        return questionary.select(
            message=message,
            choices=choices,
            default=default,
            qmark="◆",
            pointer="❯",
            style=AXON_Q_STYLE,
            use_shortcuts=False,
            use_arrow_keys=True,
            use_indicator=False,
            instruction=f"({t('nav_hint', lang)})",
        ).ask()

    return str(_ask_with_fallback(_ask, lambda: _plain_select_fallback(message, options, default)))


def _plain_text_fallback(message: str, default: str = "", secret: bool = False) -> str:
    suffix = f" [{default}]" if default and not secret else ""
    raw = getpass.getpass(f"{message}{suffix}: ") if secret else input(f"{message}{suffix}: ")
    cleaned = raw.strip()
    return cleaned if cleaned else default


def prompt_text(message: str, default: str = "", secret: bool = False) -> str:
    """Prompt for freeform or masked secret input with interactive styling and TTY fallback."""

    def _ask() -> str | None:
        if secret:
            return questionary.password(message=message, qmark="◆", style=AXON_Q_STYLE).ask()
        return questionary.text(message=message, default=default, qmark="◆", style=AXON_Q_STYLE).ask()

    raw = _ask_with_fallback(_ask, lambda: _plain_text_fallback(message, default=default, secret=secret))
    cleaned = str(raw).strip()
    return cleaned if cleaned else default


def _plain_confirm_fallback(message: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    raw = input(f"{message} [{hint}]: ").strip().lower()
    if not raw:
        return default
    return raw in {"y", "yes", "s", "si", "sí", "o", "oui"}


def prompt_confirm(message: str, default: bool = True) -> bool:
    """Ask a yes/no confirmation question cleanly in TTY or plain mode."""
    return bool(
        _ask_with_fallback(
            lambda: questionary.confirm(message=message, default=default, qmark="◆", style=AXON_Q_STYLE).ask(),
            lambda: _plain_confirm_fallback(message, default=default),
        )
    )


@contextlib.contextmanager
def status_spinner(console: Console, message: str) -> Iterator[None]:
    """Show an animated spinner in TTY terminals or clean progress line in non-TTY logs."""
    if is_interactive_tty():
        with console.status(f"[cyan]{message}[/cyan]", spinner="dots"):
            yield
    else:
        console.print(f"[cyan]•[/cyan] {message}")
        yield


def render_init_summary(
    console: Console,
    *,
    lang: str,
    config: AxonConfig,
    env_path: Path,
    adapter_res: dict[str, Any],
    hermes_existed_before: bool = True,
) -> None:
    """Render the final setup summary and actionable next-step commands."""
    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False, expand=True)
    table.add_column("key", style="bold cyan", no_wrap=True, ratio=1)
    table.add_column("val", style="white", ratio=3)

    storage_desc = (
        f"Docker PostgreSQL + pgvector ({config.db_host}:{config.db_port}/{config.db_name})"
        if config.mode == "docker"
        else f"SQLite Lite + FTS5 ({config.resolved_sqlite_path()})"
    )
    agent_cfg_path = adapter_res.get("config_path") or adapter_res.get("mcp_config_path") or "-"

    table.add_row(t("summary_env_file", lang), str(env_path))
    table.add_row(t("summary_storage", lang), storage_desc)
    table.add_row(t("summary_namespace", lang), config.namespace)
    table.add_row(
        t("summary_embeddings", lang),
        f"{config.embed_provider} · {config.embed_model} ({config.embed_dim}d)",
    )
    table.add_row(t("summary_llm", lang), f"{config.llm_provider} · {config.llm_model}")
    table.add_row(t("summary_agent_cfg", lang), str(agent_cfg_path))
    if adapter_res.get("skill_path"):
        table.add_row(t("summary_skill", lang), str(adapter_res["skill_path"]))
    table.add_row(t("summary_tools", lang), "9 cognitive MCP tools active")

    console.print()
    console.print(
        Panel(
            table,
            title=f"[bold green]✓ {t('setup_complete', lang)}[/bold green]",
            title_align="left",
            box=box.ROUNDED,
            border_style="green",
            padding=(0, 1),
        )
    )

    next_lines = [
        f"[cyan]$[/cyan] {t('next_step_doctor', lang)}",
        f"[cyan]$[/cyan] {t('next_step_serve', lang)}",
    ]
    if adapter_res.get("agent") == "openclaw":
        next_lines.append(f"[yellow]![/yellow] {t('next_step_openclaw', lang)}")
    elif adapter_res.get("agent") == "hermes" and not hermes_existed_before:
        next_lines.append(f"[yellow]![/yellow] {t('next_step_hermes_hint', lang)}")

    console.print(
        Panel(
            "\n".join(next_lines),
            title=f"[bold cyan]{t('next_steps_title', lang)}[/bold cyan]",
            title_align="left",
            box=box.ROUNDED,
            border_style="cyan",
            padding=(0, 1),
        )
    )


def render_doctor_report(
    console: Console,
    *,
    version: str,
    lang: str,
    cfg: AxonConfig,
    checks: list[tuple[str, bool, str]],
    stats_payload: dict[str, Any] | None,
    all_ok: bool,
) -> None:
    """Render styled health checks and cognitive memory layer metrics for `axonpg doctor`."""
    table = Table(
        title=f"[bold cyan]{t('doctor_title', lang)}[/bold cyan] [dim](v{version})[/dim]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="green" if all_ok else "yellow",
        header_style="bold white",
        expand=True,
    )
    table.add_column(t("doctor_check_col", lang), style="bold cyan", no_wrap=True)
    table.add_column(t("doctor_status_col", lang), no_wrap=True)
    table.add_column(t("doctor_detail_col", lang), style="white")

    table.add_row(
        "Config",
        "[green]OK[/green]",
        f"mode={cfg.mode} · namespace={cfg.namespace} · embed={cfg.embed_provider} ({cfg.embed_dim}d) · llm={cfg.llm_provider}",
    )
    for component, ok, detail in checks:
        badge = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
        table.add_row(component, badge, detail)

    console.print()
    console.print(table)

    if stats_payload:
        render_stats_report(console, lang=lang, stats_payload=stats_payload)

    footer = (
        f"[bold green]✓ {t('doctor_ok', lang)}[/bold green]"
        if all_ok
        else f"[bold yellow]! {t('doctor_warn', lang)}[/bold yellow]"
    )
    console.print(footer)


def render_stats_report(console: Console, *, lang: str, stats_payload: dict[str, Any]) -> None:
    """Render cognitive memory layer counts as a compact table."""
    table = Table(
        title=f"[bold white]{t('stats_title', lang)}[/bold white]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Backend", style="cyan", no_wrap=True)
    table.add_column("Namespace", style="white", no_wrap=True)
    table.add_column("Entities", justify="right")
    table.add_column("Relations", justify="right")
    table.add_column("Episodes (Pending)", justify="right")
    table.add_column("Memories (Superseded)", justify="right")
    table.add_column("Foresights (Expired)", justify="right")

    table.add_row(
        str(stats_payload.get("backend", "-")),
        str(stats_payload.get("namespace", "-")),
        str(stats_payload.get("entities", 0)),
        str(stats_payload.get("relations", 0)),
        f"{stats_payload.get('episodes', 0)} ({stats_payload.get('unconsolidated_episodes', 0)})",
        f"{stats_payload.get('active_memories', 0)} ({stats_payload.get('superseded_memories', 0)})",
        f"{stats_payload.get('active_foresights', 0)} ({stats_payload.get('expired_foresights', 0)})",
    )
    console.print(table)


def render_consolidation_report(console: Console, *, lang: str, report_payload: dict[str, Any]) -> None:
    """Render sleep-time consolidation metrics cleanly."""
    table = Table(
        title=f"[bold cyan]{t('consolidate_title', lang)}[/bold cyan]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=False,
        expand=True,
    )
    table.add_column("Metric", style="bold white", ratio=1)
    table.add_column("Value", style="cyan", ratio=2)

    table.add_row("Namespace", str(report_payload.get("namespace", "default")))
    table.add_row("Episodes Processed", str(report_payload.get("episodes_processed", 0)))
    table.add_row("Scenes Created", str(report_payload.get("scenes_created", 0)))
    table.add_row("Facts Superseded", str(report_payload.get("facts_superseded", 0)))
    table.add_row("Foresights Expired", str(report_payload.get("foresights_expired", 0)))
    table.add_row("Used LLM", "yes" if report_payload.get("used_llm") else "no (deterministic)")
    details = report_payload.get("details") or []
    if details:
        table.add_row("Details", "\n".join(str(d) for d in details))

    console.print(table)


@dataclass(frozen=True)
class CommandHelpItem:
    """Single command entry displayed in AxonPG grouped CLI help."""

    name: str
    desc_key: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class CommandHelpGroup:
    """Logical category grouping commands in `axonpg --help`."""

    title_key: str
    items: tuple[CommandHelpItem, ...]


def render_root_help(
    console: Console,
    *,
    version: str,
    lang: str,
    groups: list[CommandHelpGroup],
) -> None:
    """Render grouped, colorized Hermes-style CLI help with Rich."""
    console.print()
    console.print(
        f"[bold cyan]AxonPG[/bold cyan] [bold bright_green]v{version}[/bold bright_green] "
        f"[dim]—[/dim] [white]{t('tagline', lang)}[/white]"
    )
    console.print(
        f"[bold blue]{t('help_usage', lang)}:[/bold blue] "
        "[white]axonpg[/white] [cyan][-h] [-V] [--env-file PATH] [--namespace NAME] "
        "[--lang LANG] [--json][/cyan] [green]<command>[/green] [dim]...[/dim]"
    )

    for grp in groups:
        console.print()
        console.print(f"[bold blue]{t(grp.title_key, lang)}:[/bold blue]")
        tbl = Table(box=None, show_header=False, pad_edge=False, expand=False)
        tbl.add_column("cmd", style="bold green", width=18, no_wrap=True)
        tbl.add_column("desc", style="white")
        for item in grp.items:
            alias_txt = f" ({', '.join(item.aliases)})" if item.aliases else ""
            tbl.add_row(f"  {item.name}{alias_txt}", Text(t(item.desc_key, lang)))
        console.print(tbl)

    console.print()
    console.print(f"[bold blue]{t('help_options', lang)}:[/bold blue]")
    opt_tbl = Table(box=None, show_header=False, pad_edge=False, expand=False)
    opt_tbl.add_column("opt", style="bold cyan", width=22, no_wrap=True)
    opt_tbl.add_column("desc", style="white")
    for flag, key in (
        ("  -h, --help", "opt_help"),
        ("  -V, --version", "opt_version"),
        ("  --env-file PATH", "opt_env_file"),
        ("  --namespace NAME", "opt_namespace"),
        ("  --lang LANG", "opt_lang"),
        ("  --json", "opt_json"),
    ):
        opt_tbl.add_row(flag, Text(t(key, lang)))
    console.print(opt_tbl)

    console.print()
    console.print(f"[bold blue]{t('help_examples', lang)}:[/bold blue]")
    examples = (
        ("axonpg init", "ex_init"),
        ("axonpg doctor", "ex_doctor"),
        ('axonpg recall "port 5435"', "ex_recall"),
        ("axonpg connect openclaw", "ex_connect"),
        ("axonpg export backup.json", "ex_export"),
    )
    ex_tbl = Table(box=None, show_header=False, pad_edge=False, expand=False)
    ex_tbl.add_column("cmd", style="bold cyan", width=29, no_wrap=True)
    ex_tbl.add_column("note", style="dim")
    for cmd_txt, key in examples:
        ex_tbl.add_row(f"  {cmd_txt}", Text(f"# {t(key, lang)}"))
    console.print(ex_tbl)
    console.print(f"\n[dim]{t('help_cmd_hint', lang)}[/dim]")


def render_subcommand_help(console: Console, *, parser: Any, lang: str) -> None:
    """Render styled Rich help for an individual subcommand."""
    usage_txt = parser.format_usage().replace("usage: ", "").strip()
    console.print()
    console.print(f"[bold blue]{t('help_usage', lang)}:[/bold blue] [white]{usage_txt}[/white]")
    if parser.description:
        console.print(f"\n[white]{parser.description}[/white]")

    pos_actions = [a for a in parser._actions if not a.option_strings]
    opt_actions = [a for a in parser._actions if a.option_strings]

    if pos_actions:
        console.print("\n[bold blue]arguments:[/bold blue]")
        pos_tbl = Table(box=None, show_header=False, pad_edge=False, expand=False)
        pos_tbl.add_column("arg", style="bold green", width=24, no_wrap=True)
        pos_tbl.add_column("desc", style="white")
        for act in pos_actions:
            label = act.metavar or act.dest
            choices = f" ({'/'.join(map(str, act.choices))})" if act.choices else ""
            pos_tbl.add_row(f"  {label}{choices}", Text(act.help or ""))
        console.print(pos_tbl)

    if opt_actions:
        console.print("\n[bold blue]options:[/bold blue]")
        opt_tbl = Table(box=None, show_header=False, pad_edge=False, expand=False)
        opt_tbl.add_column("opt", style="bold cyan", width=28, no_wrap=True)
        opt_tbl.add_column("desc", style="white")
        for act in opt_actions:
            flags = ", ".join(act.option_strings)
            if act.nargs != 0 and act.option_strings:
                metavar = act.metavar or (
                    f"[{'|'.join(map(str, act.choices))}]" if act.choices else act.dest.upper()
                )
                flags = f"{flags} {metavar}"
            opt_tbl.add_row(Text(f"  {flags}"), Text(act.help or ""))
        console.print(opt_tbl)
    console.print()


def render_config_table(console: Console, *, lang: str, cfg: AxonConfig, env_source: str) -> None:
    """Render effective masked configuration cleanly."""
    redacted = cfg.redacted_dict()
    table = Table(
        title=f"[bold cyan]{t('config_title', lang)}[/bold cyan]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=False,
        expand=True,
    )
    table.add_column("Key", style="bold cyan", no_wrap=True, ratio=1)
    table.add_column("Value", style="white", ratio=3)
    table.add_row("env_source", env_source)
    for k, v in redacted.items():
        display = str(v) if v != "" else "[dim](empty)[/dim]"
        table.add_row(k, display)
    console.print(table)


def render_recall_result(console: Console, *, lang: str, payload: dict[str, Any]) -> None:
    """Render hybrid cognitive recall counts and assembled context block."""
    summary = (
        f"namespace=[cyan]{payload.get('namespace')}[/cyan] · "
        f"tokens≈[green]{payload.get('estimated_tokens', 0)}[/green] · "
        f"entities=[white]{len(payload.get('entities', []))}[/white] · "
        f"relations=[white]{len(payload.get('relations', []))}[/white] · "
        f"memories=[white]{len(payload.get('memories', []))}[/white] · "
        f"episodes=[white]{len(payload.get('episodes', []))}[/white] · "
        f"foresights=[white]{len(payload.get('foresights', []))}[/white]"
    )
    ctx = str(payload.get("prompt_context") or "").strip() or "[dim](no matching cognitive memories)[/dim]"
    console.print()
    console.print(f"[bold cyan]{t('recall_title', lang)}:[/bold cyan] {summary}")
    console.print(Panel(ctx, box=box.ROUNDED, border_style="cyan", padding=(0, 1)))


def render_foresights_table(console: Console, *, lang: str, items: list[dict[str, Any]]) -> None:
    """Render prospective foresights table."""
    table = Table(
        title=f"[bold cyan]{t('foresights_title', lang)}[/bold cyan]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Subject", style="bold white")
    table.add_column("Reminder", style="white")
    table.add_column("Expires", style="cyan", no_wrap=True)

    for fs in items:
        st = str(fs.get("status", "active"))
        st_cell = (
            "[green]active[/green]"
            if st == "active"
            else ("[cyan]done[/cyan]" if st == "done" else "[dim]expired[/dim]")
        )
        table.add_row(
            str(fs.get("id", ""))[:12],
            st_cell,
            str(fs.get("subject", "")),
            str(fs.get("reminder", "")),
            str(fs.get("expires_at", ""))[:19],
        )
    console.print(table)


def render_agents_table(console: Console, *, lang: str, rows: list[dict[str, Any]]) -> None:
    """Render detected agent integrations and AxonPG MCP registration status."""
    table = Table(
        title=f"[bold cyan]{t('agents_title', lang)}[/bold cyan]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="bright_black",
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Agent", style="bold white", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Config Path", style="white")

    for row in rows:
        status_cell = (
            "[green]configured[/green]"
            if row.get("configured")
            else ("[yellow]detected (not wired)[/yellow]" if row.get("detected") else "[dim]not detected[/dim]")
        )
        table.add_row(str(row["agent"]), status_cell, str(row["path"]))
    console.print(table)


def render_transfer_counts(console: Console, *, title: str, counts: dict[str, int]) -> None:
    """Render portable export/import/reset counts cleanly."""
    table = Table(
        title=f"[bold cyan]{title}[/bold cyan]",
        title_justify="left",
        box=box.ROUNDED,
        border_style="green",
        show_header=False,
        expand=True,
    )
    table.add_column("Layer", style="bold white", ratio=1)
    table.add_column("Count", style="green", justify="right", ratio=1)
    for k, v in counts.items():
        table.add_row(k, str(v))
    console.print(table)
