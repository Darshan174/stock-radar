-- =============================================================================
-- STOCK RADAR - Database Schema
-- Run this in your Supabase SQL Editor
-- =============================================================================

-- Enable pgvector extension for embeddings
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- -----------------------------------------------------------------------------
-- USERS & SUBSCRIPTIONS
-- -----------------------------------------------------------------------------

-- Users table (for freemium model)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    plan TEXT DEFAULT 'free' CHECK (plan IN ('free', 'basic', 'pro', 'enterprise')),
    stocks_limit INTEGER DEFAULT 3,
    trading_mode TEXT DEFAULT 'intraday' CHECK (trading_mode IN ('intraday', 'longterm')),
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- STOCKS & WATCHLIST
-- -----------------------------------------------------------------------------

-- Stocks master table
CREATE TABLE IF NOT EXISTS stocks (
    id BIGSERIAL PRIMARY KEY,
    symbol TEXT NOT NULL,                    -- e.g., 'RELIANCE.NS', 'AAPL'
    name TEXT NOT NULL,                       -- e.g., 'Reliance Industries'
    exchange TEXT NOT NULL,                   -- e.g., 'NSE', 'BSE', 'NASDAQ'
    sector TEXT,                              -- e.g., 'Technology', 'Banking'
    industry TEXT,                            -- e.g., 'IT Services'
    market_cap BIGINT,                        -- Market capitalization
    currency TEXT DEFAULT 'INR',              -- INR, USD
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT now(),
    updated_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(symbol, exchange)
);

-- User watchlist
CREATE TABLE IF NOT EXISTS watchlist (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    stock_id BIGINT REFERENCES stocks(id) ON DELETE CASCADE,
    mode TEXT DEFAULT 'intraday' CHECK (mode IN ('intraday', 'longterm')),
    alerts_enabled BOOLEAN DEFAULT true,
    support_level DECIMAL(12,2),              -- User-defined support
    resistance_level DECIMAL(12,2),           -- User-defined resistance
    target_price DECIMAL(12,2),               -- User's target
    stop_loss DECIMAL(12,2),                  -- User's stop loss
    notes TEXT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(user_id, stock_id)
);

-- -----------------------------------------------------------------------------
-- PRICE DATA
-- -----------------------------------------------------------------------------

-- Price history (OHLCV data)
CREATE TABLE IF NOT EXISTS price_history (
    id BIGSERIAL PRIMARY KEY,
    stock_id BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe TEXT DEFAULT '1d' CHECK (timeframe IN ('1m', '5m', '15m', '1h', '1d', '1w')),
    open DECIMAL(12,2) NOT NULL,
    high DECIMAL(12,2) NOT NULL,
    low DECIMAL(12,2) NOT NULL,
    close DECIMAL(12,2) NOT NULL,
    volume BIGINT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(stock_id, timestamp, timeframe)
);

-- Technical indicators
CREATE TABLE IF NOT EXISTS technical_indicators (
    id BIGSERIAL PRIMARY KEY,
    stock_id BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    timestamp TIMESTAMPTZ NOT NULL,
    timeframe TEXT DEFAULT '1d',
    -- Moving Averages
    sma_20 DECIMAL(12,2),
    sma_50 DECIMAL(12,2),
    sma_200 DECIMAL(12,2),
    ema_12 DECIMAL(12,2),
    ema_26 DECIMAL(12,2),
    -- Momentum
    rsi_14 DECIMAL(5,2),
    macd DECIMAL(12,4),
    macd_signal DECIMAL(12,4),
    macd_histogram DECIMAL(12,4),
    -- Volatility
    bollinger_upper DECIMAL(12,2),
    bollinger_middle DECIMAL(12,2),
    bollinger_lower DECIMAL(12,2),
    atr_14 DECIMAL(12,2),
    -- Volume
    volume_sma_20 BIGINT,
    obv BIGINT,
    created_at TIMESTAMPTZ DEFAULT now(),
    UNIQUE(stock_id, timestamp, timeframe)
);

-- -----------------------------------------------------------------------------
-- NEWS & SENTIMENT
-- -----------------------------------------------------------------------------

-- News articles
CREATE TABLE IF NOT EXISTS news (
    id BIGSERIAL PRIMARY KEY,
    stock_id BIGINT REFERENCES stocks(id) ON DELETE SET NULL,
    headline TEXT NOT NULL,
    summary TEXT,
    source TEXT,                              -- e.g., 'Economic Times', 'Reuters'
    url TEXT,
    published_at TIMESTAMPTZ,
    sentiment_score DECIMAL(3,2),             -- -1.0 to +1.0
    sentiment_label TEXT CHECK (sentiment_label IN ('positive', 'negative', 'neutral')),
    relevance_score DECIMAL(3,2),             -- 0.0 to 1.0
    embedding vector(1024),                   -- Cohere embeddings
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Social sentiment (Reddit, Twitter, etc.)
CREATE TABLE IF NOT EXISTS social_sentiment (
    id BIGSERIAL PRIMARY KEY,
    stock_id BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    source TEXT NOT NULL,                     -- 'reddit', 'twitter', 'stocktwits'
    mentions_count INTEGER DEFAULT 0,
    sentiment_score DECIMAL(3,2),             -- -1.0 to +1.0
    sentiment_label TEXT,
    sample_posts JSONB,                       -- Sample of posts
    measured_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- AI ANALYSIS & SIGNALS
-- -----------------------------------------------------------------------------

-- AI analysis results
CREATE TABLE IF NOT EXISTS analysis (
    id BIGSERIAL PRIMARY KEY,
    stock_id BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    mode TEXT NOT NULL CHECK (mode IN ('intraday', 'longterm')),
    signal TEXT NOT NULL CHECK (signal IN ('strong_buy', 'buy', 'hold', 'sell', 'strong_sell')),
    confidence DECIMAL(3,2),                  -- 0.0 to 1.0
    reasoning TEXT NOT NULL,                  -- AI explanation
    -- Technical summary
    technical_score DECIMAL(3,2),
    technical_summary TEXT,
    -- Fundamental summary (for longterm)
    fundamental_score DECIMAL(3,2),
    fundamental_summary TEXT,
    -- Sentiment summary
    sentiment_score DECIMAL(3,2),
    sentiment_summary TEXT,
    -- Price targets
    support_level DECIMAL(12,2),
    resistance_level DECIMAL(12,2),
    target_price DECIMAL(12,2),
    stop_loss DECIMAL(12,2),
    -- Metadata
    llm_model TEXT,                           -- Which model generated this
    llm_tokens_used INTEGER,
    analysis_duration_ms INTEGER,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- Trading signals history
CREATE TABLE IF NOT EXISTS signals (
    id BIGSERIAL PRIMARY KEY,
    analysis_id BIGINT REFERENCES analysis(id) ON DELETE CASCADE,
    stock_id BIGINT NOT NULL REFERENCES stocks(id) ON DELETE CASCADE,
    signal_type TEXT NOT NULL,                -- 'entry', 'exit', 'stop_loss', 'target_hit'
    signal TEXT NOT NULL,                     -- 'buy', 'sell', 'hold'
    price_at_signal DECIMAL(12,2),
    reason TEXT,
    importance TEXT DEFAULT 'medium' CHECK (importance IN ('high', 'medium', 'low')),
    is_triggered BOOLEAN DEFAULT false,
    triggered_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- ALERTS & NOTIFICATIONS
-- -----------------------------------------------------------------------------

-- Alerts sent to users
CREATE TABLE IF NOT EXISTS alerts (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    signal_id BIGINT REFERENCES signals(id) ON DELETE CASCADE,
    stock_id BIGINT REFERENCES stocks(id) ON DELETE CASCADE,
    channel TEXT NOT NULL,                    -- 'slack', 'telegram', 'email'
    message TEXT NOT NULL,
    status TEXT DEFAULT 'sent' CHECK (status IN ('pending', 'sent', 'failed')),
    external_id TEXT,                         -- Slack ts, Telegram message_id
    sent_at TIMESTAMPTZ DEFAULT now(),
    created_at TIMESTAMPTZ DEFAULT now()
);

-- -----------------------------------------------------------------------------
-- INDEXES FOR PERFORMANCE
-- -----------------------------------------------------------------------------

-- Price data indexes
CREATE INDEX IF NOT EXISTS idx_price_history_stock_time
    ON price_history(stock_id, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_price_history_timeframe
    ON price_history(timeframe, timestamp DESC);

-- Technical indicators index
CREATE INDEX IF NOT EXISTS idx_technical_stock_time
    ON technical_indicators(stock_id, timestamp DESC);

-- News indexes
CREATE INDEX IF NOT EXISTS idx_news_stock_published
    ON news(stock_id, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_sentiment
    ON news(sentiment_label, published_at DESC);
CREATE INDEX IF NOT EXISTS idx_news_embedding
    ON news USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Analysis indexes
CREATE INDEX IF NOT EXISTS idx_analysis_stock_created
    ON analysis(stock_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_analysis_signal
    ON analysis(signal, mode);

-- Signals indexes
CREATE INDEX IF NOT EXISTS idx_signals_stock_created
    ON signals(stock_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_signals_triggered
    ON signals(is_triggered, importance);

-- Watchlist indexes
CREATE INDEX IF NOT EXISTS idx_watchlist_user
    ON watchlist(user_id);

-- -----------------------------------------------------------------------------
-- RPC FUNCTIONS
-- -----------------------------------------------------------------------------

-- Semantic search for news
CREATE OR REPLACE FUNCTION search_news(
    query_embedding vector(1024),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 10,
    filter_stock_id bigint DEFAULT NULL
)
RETURNS TABLE (
    id bigint,
    stock_id bigint,
    headline text,
    summary text,
    source text,
    published_at timestamptz,
    sentiment_label text,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        n.id,
        n.stock_id,
        n.headline,
        n.summary,
        n.source,
        n.published_at,
        n.sentiment_label,
        1 - (n.embedding <=> query_embedding) AS similarity
    FROM news n
    WHERE
        n.embedding IS NOT NULL
        AND (filter_stock_id IS NULL OR n.stock_id = filter_stock_id)
        AND 1 - (n.embedding <=> query_embedding) > match_threshold
    ORDER BY n.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

-- Get latest analysis for stocks
CREATE OR REPLACE FUNCTION get_latest_analysis(
    p_stock_ids bigint[],
    p_mode text DEFAULT 'intraday'
)
RETURNS TABLE (
    stock_id bigint,
    symbol text,
    signal text,
    confidence decimal,
    reasoning text,
    support_level decimal,
    resistance_level decimal,
    created_at timestamptz
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT DISTINCT ON (a.stock_id)
        a.stock_id,
        s.symbol,
        a.signal,
        a.confidence,
        a.reasoning,
        a.support_level,
        a.resistance_level,
        a.created_at
    FROM analysis a
    JOIN stocks s ON s.id = a.stock_id
    WHERE
        a.stock_id = ANY(p_stock_ids)
        AND a.mode = p_mode
    ORDER BY a.stock_id, a.created_at DESC;
END;
$$;

-- -----------------------------------------------------------------------------
-- SEED DATA (Sample stocks)
-- -----------------------------------------------------------------------------

-- Insert some popular stocks
INSERT INTO stocks (symbol, name, exchange, sector, currency) VALUES
    -- Indian Stocks (NSE)
    ('RELIANCE.NS', 'Reliance Industries', 'NSE', 'Energy', 'INR'),
    ('TCS.NS', 'Tata Consultancy Services', 'NSE', 'Technology', 'INR'),
    ('HDFCBANK.NS', 'HDFC Bank', 'NSE', 'Banking', 'INR'),
    ('INFY.NS', 'Infosys', 'NSE', 'Technology', 'INR'),
    ('ICICIBANK.NS', 'ICICI Bank', 'NSE', 'Banking', 'INR'),
    -- US Stocks
    ('AAPL', 'Apple Inc', 'NASDAQ', 'Technology', 'USD'),
    ('MSFT', 'Microsoft Corporation', 'NASDAQ', 'Technology', 'USD'),
    ('GOOGL', 'Alphabet Inc', 'NASDAQ', 'Technology', 'USD'),
    ('AMZN', 'Amazon.com Inc', 'NASDAQ', 'Consumer Cyclical', 'USD'),
    ('TSLA', 'Tesla Inc', 'NASDAQ', 'Automotive', 'USD')
ON CONFLICT (symbol, exchange) DO NOTHING;

-- Create a default user for testing
INSERT INTO users (email, name, plan, stocks_limit) VALUES
    ('test@stockradar.com', 'Test User', 'free', 3)
ON CONFLICT (email) DO NOTHING;
