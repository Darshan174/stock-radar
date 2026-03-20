-- =============================================================================
-- STOCK RADAR - Embedding Runtime + RAG RPC Fixes
-- Run this after 001_stock_schema.sql and 003_rag_enhancements.sql
-- =============================================================================

-- -----------------------------------------------------------------------------
-- 1. Add embedding metadata columns for safer provider/model migrations
-- -----------------------------------------------------------------------------

ALTER TABLE IF EXISTS news
ADD COLUMN IF NOT EXISTS embedding_provider TEXT,
ADD COLUMN IF NOT EXISTS embedding_model_name TEXT,
ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER;

ALTER TABLE IF EXISTS analysis
ADD COLUMN IF NOT EXISTS embedding_provider TEXT,
ADD COLUMN IF NOT EXISTS embedding_model_name TEXT,
ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER;

ALTER TABLE IF EXISTS signals
ADD COLUMN IF NOT EXISTS context_embedding_provider TEXT,
ADD COLUMN IF NOT EXISTS context_embedding_model_name TEXT,
ADD COLUMN IF NOT EXISTS context_embedding_dimension INTEGER;

ALTER TABLE IF EXISTS chat_history
ADD COLUMN IF NOT EXISTS embedding_provider TEXT,
ADD COLUMN IF NOT EXISTS embedding_model_name TEXT,
ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER;

ALTER TABLE IF EXISTS knowledge_base
ADD COLUMN IF NOT EXISTS embedding_provider TEXT,
ADD COLUMN IF NOT EXISTS embedding_model_name TEXT,
ADD COLUMN IF NOT EXISTS embedding_dimension INTEGER;

-- Backfill existing rows that were produced before metadata existed.
UPDATE news
SET embedding_provider = COALESCE(embedding_provider, 'cohere'),
    embedding_model_name = COALESCE(embedding_model_name, 'embed-english-v3.0'),
    embedding_dimension = COALESCE(embedding_dimension, 1024)
WHERE embedding IS NOT NULL;

UPDATE analysis
SET embedding_provider = COALESCE(embedding_provider, 'cohere'),
    embedding_model_name = COALESCE(embedding_model_name, 'embed-english-v3.0'),
    embedding_dimension = COALESCE(embedding_dimension, 1024)
WHERE embedding IS NOT NULL;

UPDATE signals
SET context_embedding_provider = COALESCE(context_embedding_provider, 'cohere'),
    context_embedding_model_name = COALESCE(context_embedding_model_name, 'embed-english-v3.0'),
    context_embedding_dimension = COALESCE(context_embedding_dimension, 1024)
WHERE context_embedding IS NOT NULL;

UPDATE chat_history
SET embedding_provider = COALESCE(embedding_provider, 'cohere'),
    embedding_model_name = COALESCE(embedding_model_name, 'embed-english-v3.0'),
    embedding_dimension = COALESCE(embedding_dimension, 1024)
WHERE embedding IS NOT NULL;

UPDATE knowledge_base
SET embedding_provider = COALESCE(embedding_provider, 'cohere'),
    embedding_model_name = COALESCE(embedding_model_name, 'embed-english-v3.0'),
    embedding_dimension = COALESCE(embedding_dimension, 1024)
WHERE embedding IS NOT NULL;

-- -----------------------------------------------------------------------------
-- 2. Recreate RAG RPC functions with schema-aligned BIGINT return types
-- -----------------------------------------------------------------------------

-- 001_stock_schema.sql and 003_rag_enhancements.sql can leave behind two
-- overloaded search_news signatures. Drop both before recreating the canonical
-- function.
DROP FUNCTION IF EXISTS search_news(vector(1024), double precision, integer, bigint);
DROP FUNCTION IF EXISTS search_news(vector(1024), integer, double precision, integer);
DROP FUNCTION IF EXISTS search_similar_analyses(vector(1024), text, text, double precision, integer);
DROP FUNCTION IF EXISTS search_similar_signals(vector(1024), text, text, double precision, integer);
DROP FUNCTION IF EXISTS search_knowledge_base(vector(1024), uuid, text, text[], boolean, double precision, integer);
DROP FUNCTION IF EXISTS search_chat_history(vector(1024), uuid, uuid, double precision, integer);
DROP FUNCTION IF EXISTS rag_search(vector(1024), text, uuid, double precision, integer);

CREATE FUNCTION search_news(
    query_embedding vector(1024),
    filter_stock_id BIGINT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 10
)
RETURNS TABLE (
    id BIGINT,
    stock_id BIGINT,
    headline TEXT,
    summary TEXT,
    source TEXT,
    url TEXT,
    published_at TIMESTAMPTZ,
    sentiment_label TEXT,
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
        (1 - (n.embedding <=> query_embedding))::FLOAT AS similarity
    FROM news n
    WHERE n.embedding IS NOT NULL
      AND (filter_stock_id IS NULL OR n.stock_id = filter_stock_id)
      AND (1 - (n.embedding <=> query_embedding)) > match_threshold
    ORDER BY n.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

CREATE FUNCTION search_similar_analyses(
    query_embedding vector(1024),
    filter_stock_symbol TEXT DEFAULT NULL,
    filter_mode TEXT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id BIGINT,
    stock_id BIGINT,
    symbol TEXT,
    mode TEXT,
    signal TEXT,
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
        (1 - (a.embedding <=> query_embedding))::FLOAT AS similarity
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

CREATE FUNCTION search_similar_signals(
    query_embedding vector(1024),
    filter_stock_symbol TEXT DEFAULT NULL,
    filter_signal_type TEXT DEFAULT NULL,
    match_threshold FLOAT DEFAULT 0.5,
    match_count INT DEFAULT 5
)
RETURNS TABLE (
    id BIGINT,
    stock_id BIGINT,
    symbol TEXT,
    signal_type TEXT,
    signal TEXT,
    price_at_signal FLOAT,
    reason TEXT,
    importance TEXT,
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
        (1 - (sig.context_embedding <=> query_embedding))::FLOAT AS similarity
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

CREATE FUNCTION search_knowledge_base(
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
        (1 - (kb.embedding <=> query_embedding))::FLOAT AS similarity
    FROM knowledge_base kb
    WHERE kb.embedding IS NOT NULL
      AND (
        filter_user_id IS NULL
        OR kb.user_id = filter_user_id
        OR (include_public AND kb.is_public)
      )
      AND (filter_category IS NULL OR kb.category = filter_category)
      AND (filter_symbols IS NULL OR kb.stock_symbols && filter_symbols)
      AND (1 - (kb.embedding <=> query_embedding)) > match_threshold
    ORDER BY kb.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

CREATE FUNCTION search_chat_history(
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
        (1 - (ch.embedding <=> query_embedding))::FLOAT AS similarity
    FROM chat_history ch
    WHERE ch.embedding IS NOT NULL
      AND (filter_user_id IS NULL OR ch.user_id = filter_user_id)
      AND (filter_session_id IS NULL OR ch.session_id = filter_session_id)
      AND (1 - (ch.embedding <=> query_embedding)) > match_threshold
    ORDER BY ch.embedding <=> query_embedding
    LIMIT match_count;
END;
$$ LANGUAGE plpgsql;

CREATE FUNCTION rag_search(
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
            'news'::TEXT AS source_type,
            n.id::TEXT AS source_id,
            n.headline::TEXT AS title,
            COALESCE(n.summary, n.headline)::TEXT AS content,
            jsonb_build_object(
                'source', n.source,
                'url', n.url,
                'sentiment_label', n.sentiment_label,
                'sentiment_score', n.sentiment_score
            ) AS metadata,
            (1 - (n.embedding <=> query_embedding))::FLOAT AS similarity,
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
            'analysis'::TEXT AS source_type,
            a.id::TEXT AS source_id,
            CONCAT(s.symbol, ' ', a.mode, ' analysis - ', a.signal)::TEXT AS title,
            CONCAT(
                a.reasoning,
                ' ',
                COALESCE(a.technical_summary, ''),
                ' ',
                COALESCE(a.sentiment_summary, '')
            )::TEXT AS content,
            jsonb_build_object(
                'symbol', s.symbol,
                'mode', a.mode,
                'signal', a.signal,
                'confidence', a.confidence,
                'target_price', a.target_price,
                'stop_loss', a.stop_loss
            ) AS metadata,
            (1 - (a.embedding <=> query_embedding))::FLOAT AS similarity,
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
            'knowledge'::TEXT AS source_type,
            kb.id::TEXT AS source_id,
            kb.title::TEXT AS title,
            kb.content::TEXT AS content,
            jsonb_build_object(
                'category', kb.category,
                'tags', kb.tags,
                'stock_symbols', kb.stock_symbols,
                'source_url', kb.source_url
            ) AS metadata,
            (1 - (kb.embedding <=> query_embedding))::FLOAT AS similarity,
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

-- -----------------------------------------------------------------------------
-- 3. Documentation comments
-- -----------------------------------------------------------------------------

COMMENT ON COLUMN news.embedding_provider IS 'Embedding provider used for the stored news vector';
COMMENT ON COLUMN news.embedding_model_name IS 'Embedding model/version used for the stored news vector';
COMMENT ON COLUMN news.embedding_dimension IS 'Vector length for the stored news embedding';
COMMENT ON COLUMN analysis.embedding_provider IS 'Embedding provider used for the stored analysis vector';
COMMENT ON COLUMN analysis.embedding_model_name IS 'Embedding model/version used for the stored analysis vector';
COMMENT ON COLUMN analysis.embedding_dimension IS 'Vector length for the stored analysis embedding';
COMMENT ON COLUMN signals.context_embedding_provider IS 'Embedding provider used for the signal context vector';
COMMENT ON COLUMN signals.context_embedding_model_name IS 'Embedding model/version used for the signal context vector';
COMMENT ON COLUMN signals.context_embedding_dimension IS 'Vector length for the signal context embedding';
COMMENT ON COLUMN chat_history.embedding_provider IS 'Embedding provider used for the stored chat vector';
COMMENT ON COLUMN chat_history.embedding_model_name IS 'Embedding model/version used for the stored chat vector';
COMMENT ON COLUMN chat_history.embedding_dimension IS 'Vector length for the stored chat embedding';
COMMENT ON COLUMN knowledge_base.embedding_provider IS 'Embedding provider used for the stored knowledge-base vector';
COMMENT ON COLUMN knowledge_base.embedding_model_name IS 'Embedding model/version used for the stored knowledge-base vector';
COMMENT ON COLUMN knowledge_base.embedding_dimension IS 'Vector length for the stored knowledge-base embedding';
