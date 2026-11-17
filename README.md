# AxonPG

**One PostgreSQL. Four cognitive memory layers. Zero Neo4j/Qdrant bloat.**

AxonPG is an open-source, Postgres-native (`pgvector` + `tsvector` + Relational Graph) cognitive memory engine built for **Hermes Agent**, **OpenClaw**, and any **MCP-compatible** AI assistant.

## Why AxonPG?
Most agent memory stacks force you to run 3–4 separate databases or lose critical hard facts through destructive summarization. AxonPG keeps everything inside **one PostgreSQL instance** with four structured memory layers:

1. **Knowledge Graph (`axon_entities` + `axon_relations`)**: Deterministic entity states and causal links.
2. **Episodic Memory (`axon_episodes`)**: Atomic outcome traces (`success` / `failure`) preserving hard numbers, IDs, and commands.
3. **Hybrid Semantic Scenes (`axon_memories`)**: Combined BM25 (`tsvector`) + Cosine (`pgvector` HNSW) retrieval with contradiction supersession (`superseded_by`).
4. **Prospective Memory (`axon_foresights`)**: Time-bounded reminders and commitments that auto-expire to save tokens.

## Multi-Provider Embeddings (Zero Lock-in)
* **100% Local / Zero API Key:** FastEmbed (ONNX CPU), Ollama (`bge-m3`, `nomic-embed-text`)
* **Free / Low-Cost Cloud:** NVIDIA NIM, OpenRouter
* **Enterprise APIs:** Gemini, OpenAI, or any OpenAI-compatible `BASE_URL`

## Roadmap
- [x] Initial clean project structure & SQL schema (`sql/init.sql`)
- [ ] Sprint 1: Multi-provider embeddings, FastMCP server & `axonpg init` (Hermes + OpenClaw adapter)
- [ ] Sprint 2: Background sleep-time consolidation & contradiction resolver
- [ ] Sprint 3: Token/latency benchmark suite & PyPI release (`uvx axonpg init`)
