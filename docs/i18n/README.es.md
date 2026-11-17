# AxonPG (Español)

[English](../../README.md) • **Español** • [Português](README.pt.md) • [Français](README.fr.md)

**AxonPG** es un motor de **memoria cognitiva multilingüe nativa en PostgreSQL (`pgvector` + `pg_trgm` + `unaccent`)** diseñado para agentes de IA (**Hermes Agent**, **OpenClaw**, **Claude**, **Cursor**, **Antigravity**), con modo ligero sin Docker (**SQLite + FTS5 `--lite`**).

## Capas Cognitivas

1. **Grafo de Conocimiento (`axon_entities`, `axon_relations`)**: entidades con alias multilingües y relaciones dirigidas.
2. **Memoria Episódica (`axon_episodes`)**: trazas atómicas de tareas preservando valores exactos en `hard_facts` (IDs, fechas, montos, puertos, códigos de error).
3. **Memoria Semántica Híbrida (`axon_memories`)**: ranking híbrido RRF (búsqueda léxica sin acentos + trigramas/CJK + similitud vectorial HNSW) con reemplazo automático de contradicciones (`fact_key`).
4. **Prospectiva (`axon_foresights`)**: recordatorios con expiración automática (`expires_at`).

## Inicio Rápido

```bash
git clone https://github.com/ricardo2591/axonpg.git
cd axonpg
python -m venv .venv
source .venv/bin/activate  # En Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -e ".[local]"

# Modo producción (PostgreSQL + pgvector)
docker compose up -d
axonpg init --lang es --mode docker --agent hermes

# O modo portátil sin Docker (SQLite + FTS5)
axonpg init --lang es --mode lite --agent hermes

axonpg doctor
```

## ¿Hermes Agent vs OpenClaw vs MCP Genérico?

- **Hermes Agent (por defecto)**: integra automáticamente `mcp_servers.axonpg` en `~/.hermes/config.yaml` e instala la skill `~/.hermes/skills/axonpg-memory/SKILL.md`.
- **OpenClaw**: genera `~/.openclaw/mcp/axonpg.json` y `~/.openclaw/skills/axonpg-memory/SKILL.md` sin modificar ni romper el esquema estricto de `openclaw.json`.
- **MCP Genérico**: genera `axonpg.mcp.json` estándar compatible con Claude Desktop, Cursor, Windsurf y Antigravity.

Todas las credenciales viven únicamente en `.env`, nunca dentro de los archivos de configuración de los agentes.
