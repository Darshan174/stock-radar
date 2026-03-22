-- =============================================================================
-- STOCK RADAR - Analyze Jobs: worker claim & crash recovery columns
-- Run after 005_analyze_jobs.sql
-- =============================================================================

ALTER TABLE analyze_jobs
ADD COLUMN IF NOT EXISTS worker_id    TEXT,
ADD COLUMN IF NOT EXISTS locked_until TIMESTAMPTZ;

-- Partial index: fast lookup of claimable rows (queued, or stale running)
CREATE INDEX IF NOT EXISTS idx_analyze_jobs_claimable
    ON analyze_jobs (status, locked_until)
    WHERE status IN ('queued', 'running');
