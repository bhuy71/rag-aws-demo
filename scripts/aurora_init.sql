-- Enable pgvector extension (requires Aurora PostgreSQL version >= 15.2)
CREATE EXTENSION IF NOT EXISTS vector;

-- LangChain will create these tables automatically, but you can pre-create
-- them to manage permissions explicitly.
CREATE TABLE IF NOT EXISTS langchain_pg_collection (
    id UUID PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    cmetadata JSONB
);

CREATE TABLE IF NOT EXISTS langchain_pg_embedding (
    id UUID PRIMARY KEY,
    collection_id UUID REFERENCES langchain_pg_collection(id) ON DELETE CASCADE,
    document TEXT,
    embedding VECTOR(1536),
    cmetadata JSONB,
    custom_id TEXT,
    version SMALLINT DEFAULT 1
);

CREATE INDEX IF NOT EXISTS idx_langchain_pg_embedding_collection
    ON langchain_pg_embedding (collection_id);
