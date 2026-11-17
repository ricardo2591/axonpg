-- AxonPG: Cognitive Memory Schema (Single-PostgreSQL Architecture)
-- Requires: pgvector, pg_trgm, uuid-ossp

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "vector";

-- 1. Knowledge Graph: Entities (Current Truth + Aliases)
CREATE TABLE IF NOT EXISTS axon_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    entity_type VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    aliases TEXT[] DEFAULT ARRAY[]::TEXT[],
    summary TEXT,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_axon_entity UNIQUE (namespace, entity_type, name)
);

-- 2. Knowledge Graph: Causal & Operational Relations
CREATE TABLE IF NOT EXISTS axon_relations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    source_id UUID NOT NULL REFERENCES axon_entities(id) ON DELETE CASCADE,
    relation_type VARCHAR(64) NOT NULL,
    target_id UUID NOT NULL REFERENCES axon_entities(id) ON DELETE CASCADE,
    weight DOUBLE PRECISION NOT NULL DEFAULT 1.0,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    valid_from TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    valid_until TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 3. Episodic Memory (MemCell-style atomic event traces with outcome)
CREATE TABLE IF NOT EXISTS axon_episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    task_name VARCHAR(128) NOT NULL,
    outcome VARCHAR(24) NOT NULL DEFAULT 'success', -- success | failure | partial
    what_happened TEXT NOT NULL,
    lessons_learned TEXT,
    hard_facts JSONB NOT NULL DEFAULT '{}'::jsonb, -- Exact numbers, IDs, commands never lost
    entity_names TEXT[] DEFAULT ARRAY[]::TEXT[],
    consolidated BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Semantic Scenes & Hybrid Embeddings (BM25 tsvector + HNSW Vector + Supersession)
-- Note: Adjust vector dimension before production init if using 768/1024/1536 models.
CREATE TABLE IF NOT EXISTS axon_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    category VARCHAR(48) NOT NULL DEFAULT 'scene',  -- fact | scene | procedure | lesson
    content TEXT NOT NULL,
    hard_facts JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED,
    embedding vector(384),
    embed_model VARCHAR(128),
    superseded_by UUID REFERENCES axon_memories(id) ON DELETE SET NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Prospective Memory (Foresights: time-bounded commitments that auto-expire)
CREATE TABLE IF NOT EXISTS axon_foresights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    subject VARCHAR(255) NOT NULL,
    reminder TEXT NOT NULL,
    due_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(24) NOT NULL DEFAULT 'active', -- active | done | expired
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_axon_entities_ns_type ON axon_entities (namespace, entity_type);
CREATE INDEX IF NOT EXISTS idx_axon_episodes_unconsolidated ON axon_episodes (namespace, consolidated) WHERE consolidated = FALSE;
CREATE INDEX IF NOT EXISTS idx_axon_memories_tsv ON axon_memories USING gin (tsv);
CREATE INDEX IF NOT EXISTS idx_axon_memories_hnsw ON axon_memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_axon_foresights_active ON axon_foresights (namespace, expires_at) WHERE status = 'active';
