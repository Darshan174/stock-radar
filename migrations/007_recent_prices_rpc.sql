-- =============================================================================
-- STOCK RADAR - RPC: recent prices per stock (avoids fetching all rows)
-- Returns the most recent `limit_per_stock` price rows for each given stock_id.
-- =============================================================================

CREATE OR REPLACE FUNCTION recent_prices_for_stocks(
    stock_ids BIGINT[],
    limit_per_stock INT DEFAULT 30
)
RETURNS TABLE (
    stock_id BIGINT,
    close    DOUBLE PRECISION,
    "timestamp" TIMESTAMPTZ
)
LANGUAGE sql STABLE
AS $$
    SELECT ph.stock_id, ph.close, ph.timestamp
    FROM unnest(stock_ids) AS sid(id)
    CROSS JOIN LATERAL (
        SELECT ph2.stock_id, ph2.close, ph2.timestamp
        FROM price_history ph2
        WHERE ph2.stock_id = sid.id
        ORDER BY ph2.timestamp DESC
        LIMIT limit_per_stock
    ) ph
    ORDER BY ph.stock_id, ph.timestamp DESC;
$$;
