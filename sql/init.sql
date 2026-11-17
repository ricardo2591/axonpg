-- AxonPG: Multilingual Cognitive Memory Schema (Single-PostgreSQL Architecture)
-- Requires: pgvector, pg_trgm, unaccent, uuid-ossp

CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "unaccent";
CREATE EXTENSION IF NOT EXISTS "vector";

-- Immutable accent-folding wrapper so generated tsvector columns work cleanly across Latin accents
CREATE OR REPLACE FUNCTION axon_unaccent(text)
RETURNS text AS $$
    SELECT public.unaccent('public.unaccent', $1);
$$ LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE;

-- 1. Knowledge Graph: Entities (Current Truth + Multilingual Aliases)
CREATE TABLE IF NOT EXISTS axon_entities (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    entity_type VARCHAR(64) NOT NULL,
    name VARCHAR(255) NOT NULL,
    name_norm VARCHAR(255) NOT NULL,
    aliases TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    summary TEXT,
    properties JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_lang VARCHAR(16),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_axon_entity UNIQUE (namespace, entity_type, name_norm)
);

-- 2. Knowledge Graph: Causal & Operational Relations (Temporally Bounded)
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
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT uq_axon_relation UNIQUE (namespace, source_id, relation_type, target_id)
);

-- 3. Episodic Memory (MemCell-style atomic event traces with hard_facts preservation)
CREATE TABLE IF NOT EXISTS axon_episodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    task_name VARCHAR(128) NOT NULL,
    outcome VARCHAR(24) NOT NULL DEFAULT 'success',
    what_happened TEXT NOT NULL,
    lessons_learned TEXT,
    hard_facts JSONB NOT NULL DEFAULT '{}'::jsonb,
    entity_names TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    source_lang VARCHAR(16),
    consolidated BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 4. Hybrid Semantic Memory (Multilingual tsvector + HNSW Vector + Contradiction Supersession)
CREATE TABLE IF NOT EXISTS axon_memories (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    category VARCHAR(48) NOT NULL DEFAULT 'fact',
    fact_key VARCHAR(255),
    content TEXT NOT NULL,
    hard_facts JSONB NOT NULL DEFAULT '{}'::jsonb,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('simple', axon_unaccent(coalesce(content, '')))) STORED,
    embedding vector(384),
    embed_model VARCHAR(128),
    source_lang VARCHAR(16),
    superseded_by UUID REFERENCES axon_memories(id) ON DELETE SET NULL,
    active BOOLEAN NOT NULL DEFAULT TRUE,
    hit_count INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- 5. Prospective Memory (Foresights: time-bounded commitments that auto-expire)
CREATE TABLE IF NOT EXISTS axon_foresights (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    namespace VARCHAR(64) NOT NULL DEFAULT 'default',
    subject VARCHAR(255) NOT NULL,
    reminder TEXT NOT NULL,
    hard_facts JSONB NOT NULL DEFAULT '{}'::jsonb,
    source_lang VARCHAR(16),
    due_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ NOT NULL,
    status VARCHAR(24) NOT NULL DEFAULT 'active',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_axon_entities_ns_norm ON axon_entities (namespace, name_norm);
CREATE INDEX IF NOT EXISTS idx_axon_entities_trgm ON axon_entities USING gin (name_norm gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_axon_relations_src ON axon_relations (namespace, source_id);
CREATE INDEX IF NOT EXISTS idx_axon_relations_tgt ON axon_relations (namespace, target_id);
CREATE INDEX IF NOT EXISTS idx_axon_episodes_unconsolidated ON axon_episodes (namespace, consolidated) WHERE consolidated = FALSE;
CREATE INDEX IF NOT EXISTS idx_axon_memories_active_key ON axon_memories (namespace, fact_key) WHERE active = TRUE;
CREATE INDEX IF NOT EXISTS idx_axon_memories_tsv ON axon_memories USING gin (tsv);
CREATE INDEX IF NOT EXISTS idx_axon_memories_hnsw ON axon_memories USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_axon_foresights_active ON axon_foresights (namespace, expires_at) WHERE status = 'active';
