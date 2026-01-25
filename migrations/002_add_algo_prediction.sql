-- =============================================================================
-- STOCK RADAR - Add algo_prediction column
-- Run this in your Supabase SQL Editor
-- =============================================================================

-- Add algo_prediction column to analysis table for AI algo trading predictions
ALTER TABLE analysis 
ADD COLUMN IF NOT EXISTS algo_prediction JSONB;

-- Create index for querying algo predictions
CREATE INDEX IF NOT EXISTS idx_analysis_algo_prediction 
ON analysis USING gin (algo_prediction);

-- Comment
COMMENT ON COLUMN analysis.algo_prediction IS 'AI algo trading prediction with momentum, value, quality scores and predicted returns';
