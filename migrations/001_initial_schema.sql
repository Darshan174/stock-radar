-- Initial schema for Research Radar
-- Run these migrations in your Supabase project

-- Enable pgvector extension for semantic search
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- Competitors table
CREATE TABLE IF NOT EXISTS competitors (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL UNIQUE,
  url TEXT NOT NULL UNIQUE,
  created_at TIMESTAMPTZ DEFAULT now(),
  updated_at TIMESTAMPTZ DEFAULT now()
);

-- Crawls table (web scrape results)
CREATE TABLE IF NOT EXISTS crawls (
  id BIGSERIAL PRIMARY KEY,
  competitor_id BIGINT NOT NULL REFERENCES competitors(id) ON DELETE CASCADE,
  markdown TEXT,
  html TEXT,
  url TEXT NOT NULL,
  embedding vector(768),
  crawl_date TIMESTAMPTZ DEFAULT now(),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Changes table (detected differences)
CREATE TABLE IF NOT EXISTS changes (
  id BIGSERIAL PRIMARY KEY,
  crawl_id BIGINT NOT NULL REFERENCES crawls(id) ON DELETE CASCADE,
  type TEXT CHECK (type IN ('pricing', 'feature', 'hiring', 'partnership', 'product', 'announcement', 'content', 'other')),
  summary TEXT NOT NULL,
  importance TEXT CHECK (importance IN ('high', 'medium', 'low')) DEFAULT 'low',
  detected_at TIMESTAMPTZ DEFAULT now(),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Alerts table (sent notifications)
CREATE TABLE IF NOT EXISTS alerts (
  id BIGSERIAL PRIMARY KEY,
  change_id BIGINT NOT NULL REFERENCES changes(id) ON DELETE CASCADE UNIQUE,
  slack_ts TEXT NOT NULL,
  alerted_at TIMESTAMPTZ DEFAULT now(),
  created_at TIMESTAMPTZ DEFAULT now()
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_crawls_competitor_date ON crawls(competitor_id, crawl_date DESC);
CREATE INDEX IF NOT EXISTS idx_crawls_competitor_id ON crawls(competitor_id);
CREATE INDEX IF NOT EXISTS idx_changes_crawl_id ON changes(crawl_id);
CREATE INDEX IF NOT EXISTS idx_changes_detected_at ON changes(detected_at DESC);
CREATE INDEX IF NOT EXISTS idx_changes_importance ON changes(importance);
CREATE INDEX IF NOT EXISTS idx_alerts_change_id ON alerts(change_id);
CREATE INDEX IF NOT EXISTS idx_crawls_embedding ON crawls USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- RPC function for semantic search
CREATE OR REPLACE FUNCTION search_crawls(
    query_embedding vector(768),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 5,
    filter_competitor_id int DEFAULT NULL
)
RETURNS TABLE (
    id bigint,
    competitor_id bigint,
    url text,
    markdown text,
    crawl_date timestamptz,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.competitor_id,
        c.url,
        c.markdown,
        c.crawl_date,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM crawls c
    WHERE
        c.embedding IS NOT NULL
        AND (filter_competitor_id IS NULL OR c.competitor_id = filter_competitor_id)
        AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
