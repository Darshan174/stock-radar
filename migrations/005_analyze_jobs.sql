-- =============================================================================
-- STOCK RADAR - Analyze Jobs Table
-- Persistent job state for async analysis (replaces in-memory store)
-- =============================================================================

CREATE TABLE IF NOT EXISTS analyze_jobs (
    job_id      TEXT PRIMARY KEY,
    status      TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'succeeded', 'failed')),
    symbol      TEXT,
    mode        TEXT,
    period      TEXT,
    result      JSONB,
    error       TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_analyze_jobs_status ON analyze_jobs (status);
CREATE INDEX IF NOT EXISTS idx_analyze_jobs_created ON analyze_jobs (created_at);
