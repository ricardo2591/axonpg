# AxonPG (Português)

[English](../../README.md) • [Español](README.es.md) • **Português** • [Français](README.fr.md)

**AxonPG** é um motor de **memória cognitiva multilíngue nativo em PostgreSQL (`pgvector` + `pg_trgm` + `unaccent`)** para agentes de IA (**Hermes Agent**, **OpenClaw**, **Claude**, **Cursor**, **Antigravity**), com modo local sem Docker (**SQLite + FTS5 `--lite`**).

## Início Rápido

```bash
git clone https://github.com/ricardo2591/axonpg.git
cd axonpg
python -m venv .venv
source .venv/bin/activate
pip install -e ".[local]"

# Modo Docker (PostgreSQL + pgvector)
docker compose up -d
axonpg init --lang pt --mode docker --agent hermes

# Ou modo local leve (SQLite + FTS5)
axonpg init --lang pt --mode lite --agent hermes

axonpg doctor
```

Consulte [README.md](../../README.md) para detalhes das ferramentas MCP e provedores (`fastembed`, `ollama`, `nvidia`, `openrouter`, `gemini`, `openai`).
