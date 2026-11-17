# AxonPG (Français)

[English](../../README.md) • [Español](README.es.md) • [Português](README.pt.md) • **Français**

**AxonPG** est un moteur de **mémoire cognitive multilingue natif sur PostgreSQL (`pgvector` + `pg_trgm` + `unaccent`)** pour les agents IA (**Hermes Agent**, **OpenClaw**, **Claude**, **Cursor**, **Antigravity**), incluant un mode local sans Docker (**SQLite + FTS5 `--lite`**).

## Démarrage Rapide

```bash
git clone https://github.com/ricardo2591/axonpg.git
cd axonpg
python -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"

# Mode production (PostgreSQL + pgvector)
docker compose up -d
axonpg init --lang fr --mode docker --agent hermes

# Ou mode léger sans Docker (SQLite + FTS5)
axonpg init --lang fr --mode lite --agent hermes

axonpg doctor
```

Consultez [README.md](../../README.md) pour la liste complète des outils MCP et des fournisseurs (`fastembed`, `ollama`, `nvidia`, `openrouter`, `gemini`, `openai`).
