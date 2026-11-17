"""Security and hygiene tests verifying zero sensitive leaks and secret-safe config rules."""

from __future__ import annotations

from pathlib import Path

FORBIDDEN_MARKERS = (
    "hybrico",
    "newsunroad",
    "boy7014",
    "178.156.",
    "10.0.0.",
    "tigo-vpn",
    "hoja de vida",
)


def test_repository_contains_no_forbidden_private_markers():
    repo_root = Path(__file__).resolve().parents[1]
    scan_roots = [
        repo_root / "src",
        repo_root / "sql",
        repo_root / "installers",
        repo_root / "skills",
        repo_root / "docs",
        repo_root / ".github",
        repo_root / "README.md",
        repo_root / "docker-compose.yml",
        repo_root / "pyproject.toml",
        repo_root / ".env.example",
    ]
    files_to_scan: list[Path] = []
    for target in scan_roots:
        if target.is_file():
            files_to_scan.append(target)
        elif target.is_dir():
            for path in target.rglob("*"):
                if path.is_file() and "__pycache__" not in path.parts:
                    files_to_scan.append(path)

    for path in files_to_scan:
        content = path.read_text(encoding="utf-8", errors="ignore").casefold()
        for marker in FORBIDDEN_MARKERS:
            assert marker not in content, f"Forbidden marker '{marker}' found in {path}"


def test_gitignore_blocks_env_pdfs_and_sqlite_files():
    repo_root = Path(__file__).resolve().parents[1]
    gi = (repo_root / ".gitignore").read_text(encoding="utf-8")
    for pattern in (".env", "*.pdf", ".venv/"):
        assert pattern in gi
