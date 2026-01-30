-- =============================================================================
-- STOCK RADAR - RAG (Retrieval-Augmented Generation) Enhancements
-- Run this in your Supabase SQL Editor
-- =============================================================================

-- Enable pgvector extension if not already enabled
CREATE EXTENSION IF NOT EXISTS vector;

-- =============================================================================
-- 1. Add embedding columns to analysis table for semantic search
-- =============================================================================

-- Add embedding column to analysis table
ALTER TABLE analysis
ADD COLUMN IF NOT EXISTS embedding vector(1024);

-- Add embedding column to signals table for context retrieval
ALTER TABLE signals
ADD COLUMN IF NOT EXISTS context_embedding vector(1024);

-- Add text content column to analysis for embedding source
ALTER TABLE analysis
ADD COLUMN IF NOT EXISTS embedding_text TEXT;

-- =============================================================================
-- 2. Create indexes for vector similarity search
-- =============================================================================

-- Index for analysis embeddings (semantic search over past analyses)
CREATE INDEX IF NOT EXISTS idx_analysis_embedding
ON analysis USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Index for signal context embeddings
CREATE INDEX IF NOT EXISTS idx_signals_context_embedding
ON signals USING ivfflat (context_embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- 3. Create chat_history table for conversational RAG
-- =============================================================================

CREATE TABLE IF NOT EXISTS chat_history (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    context_used JSONB,  -- Store which documents were retrieved for context
    stock_symbols TEXT[],  -- Stocks mentioned in the message
    embedding vector(1024),  -- For finding similar past conversations
    tokens_used INTEGER,
    model_used VARCHAR(100),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Index for chat history retrieval
CREATE INDEX IF NOT EXISTS idx_chat_history_session ON chat_history(session_id, created_at);
CREATE INDEX IF NOT EXISTS idx_chat_history_user ON chat_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_history_embedding
ON chat_history USING ivfflat (embedding vector_cosine_ops) WITH (lists = 50);

-- =============================================================================
-- 4. Create knowledge_base table for custom documents/notes
-- =============================================================================

CREATE TABLE IF NOT EXISTS knowledge_base (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    title VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    category VARCHAR(100),  -- 'strategy', 'research', 'notes', 'earnings', etc.
    stock_symbols TEXT[],  -- Related stock symbols
    tags TEXT[],
    embedding vector(1024),
    source_url TEXT,
    is_public BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for knowledge base
CREATE INDEX IF NOT EXISTS idx_knowledge_base_user ON knowledge_base(user_id);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_category ON knowledge_base(category);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_symbols ON knowledge_base USING gin(stock_symbols);
CREATE INDEX IF NOT EXISTS idx_knowledge_base_embedding
ON knowledge_base USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- =============================================================================
-- 5. RPC Functions for RAG retrieval
-- =============================================================================

-- Function to search news articles semantically
CREATE OR REPLACE FUNCTION search_news(
    query_embedding vector(1024),
    filter_stock_id INT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id INT,
    stock_id INT,
    headline TEXT,
    summary TEXT,
    source VARCHAR,
    url TEXT,
    published_at TIMESTAMPTZ,
    sentiment_label VARCHAR,
    sentiment_score FLOAT,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        n.id,
        n.stock_id,
        n.headline,
        n.summary,
        n.source,
        n.url,
        n.published_at,
        n.sentiment_label,
        n.sentiment_score::FLOAT,
        (1 - (n.embedding <=> query_embedding))::FLOAT as similarity
    FROM news n
    WHERE n.embedding IS NOT NULL
      AND (filter_stock_id IS NULL OR n.stock_id = filter_stock_id)
      AND (1 - (n.embedding <=> query_embedding)) > match_threshold
    ORDER BY n.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to search similar analyses
CREATE OR REPLACE FUNCTION search_similar_analyses(
    query_embedding vector(1024),
    filter_stock_symbol TEXT DEFAULT NULL,
    filter_mode TEXT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id UUID,
    stock_id INT,
    symbol TEXT,
    mode VARCHAR,
    signal VARCHAR,
    confidence FLOAT,
    reasoning TEXT,
    technical_summary TEXT,
    sentiment_summary TEXT,
    target_price FLOAT,
    stop_loss FLOAT,
    created_at TIMESTAMPTZ,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        a.id,
        a.stock_id,
        s.symbol,
        a.mode,
        a.signal,
        a.confidence::FLOAT,
        a.reasoning,
        a.technical_summary,
        a.sentiment_summary,
        a.target_price::FLOAT,
        a.stop_loss::FLOAT,
        a.created_at,
        (1 - (a.embedding <=> query_embedding))::FLOAT as similarity
    FROM analysis a
    JOIN stocks s ON a.stock_id = s.id
    WHERE a.embedding IS NOT NULL
      AND (filter_stock_symbol IS NULL OR s.symbol = filter_stock_symbol)
      AND (filter_mode IS NULL OR a.mode = filter_mode)
      AND (1 - (a.embedding <=> query_embedding)) > match_threshold
    ORDER BY a.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to search signals with similar context
CREATE OR REPLACE FUNCTION search_similar_signals(
    query_embedding vector(1024),
    filter_stock_symbol TEXT DEFAULT NULL,
    filter_signal_type TEXT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id INT,
    stock_id INT,
    symbol TEXT,
    signal_type VARCHAR,
    signal VARCHAR,
    price_at_signal FLOAT,
    reason TEXT,
    importance VARCHAR,
    created_at TIMESTAMPTZ,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        sig.id,
        sig.stock_id,
        s.symbol,
        sig.signal_type,
        sig.signal,
        sig.price_at_signal::FLOAT,
        sig.reason,
        sig.importance,
        sig.created_at,
        (1 - (sig.context_embedding <=> query_embedding))::FLOAT as similarity
    FROM signals sig
    JOIN stocks s ON sig.stock_id = s.id
    WHERE sig.context_embedding IS NOT NULL
      AND (filter_stock_symbol IS NULL OR s.symbol = filter_stock_symbol)
      AND (filter_signal_type IS NULL OR sig.signal_type = filter_signal_type)
      AND (1 - (sig.context_embedding <=> query_embedding)) > match_threshold
    ORDER BY sig.context_embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to search knowledge base
CREATE OR REPLACE FUNCTION search_knowledge_base(
    query_embedding vector(1024),
    filter_user_id UUID DEFAULT NULL,
    filter_category TEXT DEFAULT NULL,
    filter_symbols TEXT[] DEFAULT NULL,
    include_public BOOLEAN DEFAULT TRUE,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    title VARCHAR,
    content TEXT,
    category VARCHAR,
    stock_symbols TEXT[],
    tags TEXT[],
    source_url TEXT,
    created_at TIMESTAMPTZ,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        kb.id,
        kb.title,
        kb.content,
        kb.category,
        kb.stock_symbols,
        kb.tags,
        kb.source_url,
        kb.created_at,
        (1 - (kb.embedding <=> query_embedding))::FLOAT as similarity
    FROM knowledge_base kb
    WHERE kb.embedding IS NOT NULL
      AND (filter_user_id IS NULL OR kb.user_id = filter_user_id OR (include_public AND kb.is_public))
      AND (filter_category IS NULL OR kb.category = filter_category)
      AND (filter_symbols IS NULL OR kb.stock_symbols && filter_symbols)
      AND (1 - (kb.embedding <=> query_embedding)) > match_threshold
    ORDER BY kb.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- Function to search chat history for similar conversations
CREATE OR REPLACE FUNCTION search_chat_history(
    query_embedding vector(1024),
    filter_user_id UUID DEFAULT NULL,
    filter_session_id UUID DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id UUID,
    session_id UUID,
    role VARCHAR,
    content TEXT,
    stock_symbols TEXT[],
    context_used JSONB,
    created_at TIMESTAMPTZ,
    similarity FLOAT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        ch.id,
        ch.session_id,
        ch.role,
        ch.content,
        ch.stock_symbols,
        ch.context_used,
        ch.created_at,
        (1 - (ch.embedding <=> query_embedding))::FLOAT as similarity
    FROM chat_history ch
    WHERE ch.embedding IS NOT NULL
      AND (filter_user_id IS NULL OR ch.user_id = filter_user_id)
      AND (filter_session_id IS NULL OR ch.session_id = filter_session_id)
      AND (1 - (ch.embedding <=> query_embedding)) > match_threshold
    ORDER BY ch.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 6. Comprehensive RAG search function (searches all sources)
-- =============================================================================

CREATE OR REPLACE FUNCTION rag_search(
    query_embedding vector(1024),
    filter_stock_symbol TEXT DEFAULT NULL,
    filter_user_id UUID DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.4,
    max_results_per_source INT DEFAULT 3
)
RETURNS TABLE (
    source_type TEXT,
    source_id TEXT,
    title TEXT,
    content TEXT,
    metadata JSONB,
    similarity FLOAT,
    created_at TIMESTAMPTZ
) AS $$
BEGIN
    RETURN QUERY
    WITH news_results AS (
        SELECT
            'news'::TEXT as source_type,
            n.id::TEXT as source_id,
            n.headline::TEXT as title,
            COALESCE(n.summary, n.headline)::TEXT as content,
            jsonb_build_object(
                'source', n.source,
                'url', n.url,
                'sentiment_label', n.sentiment_label,
                'sentiment_score', n.sentiment_score
            ) as metadata,
            (1 - (n.embedding <=> query_embedding))::FLOAT as similarity,
            n.created_at
        FROM news n
        LEFT JOIN stocks s ON n.stock_id = s.id
        WHERE n.embedding IS NOT NULL
          AND (filter_stock_symbol IS NULL OR s.symbol = filter_stock_symbol)
          AND (1 - (n.embedding <=> query_embedding)) > match_threshold
        ORDER BY n.embedding <=> query_embedding
        LIMIT max_results_per_source
    ),
    analysis_results AS (
        SELECT
            'analysis'::TEXT as source_type,
            a.id::TEXT as source_id,
            CONCAT(s.symbol, ' ', a.mode, ' analysis - ', a.signal)::TEXT as title,
            CONCAT(a.reasoning, ' ', COALESCE(a.technical_summary, ''), ' ', COALESCE(a.sentiment_summary, ''))::TEXT as content,
            jsonb_build_object(
                'symbol', s.symbol,
                'mode', a.mode,
                'signal', a.signal,
                'confidence', a.confidence,
                'target_price', a.target_price,
                'stop_loss', a.stop_loss
            ) as metadata,
            (1 - (a.embedding <=> query_embedding))::FLOAT as similarity,
            a.created_at
        FROM analysis a
        JOIN stocks s ON a.stock_id = s.id
        WHERE a.embedding IS NOT NULL
          AND (filter_stock_symbol IS NULL OR s.symbol = filter_stock_symbol)
          AND (1 - (a.embedding <=> query_embedding)) > match_threshold
        ORDER BY a.embedding <=> query_embedding
        LIMIT max_results_per_source
    ),
    knowledge_results AS (
        SELECT
            'knowledge'::TEXT as source_type,
            kb.id::TEXT as source_id,
            kb.title::TEXT as title,
            kb.content::TEXT as content,
            jsonb_build_object(
                'category', kb.category,
                'tags', kb.tags,
                'stock_symbols', kb.stock_symbols,
                'source_url', kb.source_url
            ) as metadata,
            (1 - (kb.embedding <=> query_embedding))::FLOAT as similarity,
            kb.created_at
        FROM knowledge_base kb
        WHERE kb.embedding IS NOT NULL
          AND (filter_user_id IS NULL OR kb.user_id = filter_user_id OR kb.is_public)
          AND (filter_stock_symbol IS NULL OR filter_stock_symbol = ANY(kb.stock_symbols))
          AND (1 - (kb.embedding <=> query_embedding)) > match_threshold
        ORDER BY kb.embedding <=> query_embedding
        LIMIT max_results_per_source
    )
    SELECT * FROM news_results
    UNION ALL
    SELECT * FROM analysis_results
    UNION ALL
    SELECT * FROM knowledge_results
    ORDER BY similarity DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- 7. Comments for documentation
-- =============================================================================

COMMENT ON TABLE chat_history IS 'Stores conversation history for RAG-powered chat assistant';
COMMENT ON TABLE knowledge_base IS 'User-uploadable documents and research notes for RAG retrieval';
COMMENT ON FUNCTION search_news IS 'Search news articles with semantic similarity';
COMMENT ON FUNCTION search_similar_analyses IS 'Find past analyses semantically similar to a query';
COMMENT ON FUNCTION search_similar_signals IS 'Find trading signals with similar context';
COMMENT ON FUNCTION search_knowledge_base IS 'Search user knowledge base with semantic similarity';
COMMENT ON FUNCTION rag_search IS 'Comprehensive RAG search across all data sources';
COMMENT ON COLUMN analysis.embedding IS '1024-dim Cohere embedding for semantic search over analyses';
COMMENT ON COLUMN analysis.embedding_text IS 'Text used to generate the embedding for this analysis';
