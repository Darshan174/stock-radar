"""
Dataset builder for ML training.

Primary mode (`future`) creates labels from future returns, not rule-engine output.
Legacy mode (`rule`) keeps the previous one-row-per-symbol behavior.

Phase-5 upgrade: adds cross-sectional, factor, and microstructure features.

Usage:
    python -m training.dataset_builder --symbols AAPL,MSFT --output data/training_data.csv
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.feature_engineering import (
    FEATURE_NAMES,
    extract_features,
    future_return_to_signal,
)
from training.cross_sectional import (
    ALL_NEW_FEATURE_NAMES,
    compute_factor_features,
    compute_microstructure_features,
    _returns_from_closes,
    _safe_float,
)
from training.sentiment import compute_sentiment_features

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

DEFAULT_SYMBOLS_FILE = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "training_symbols.txt"
)

# Sector mapping — provider-based with a minimal fallback.
# The fetcher's get_fundamentals() returns a "sector" field from Yahoo Finance.
# We cache resolved sectors to avoid repeated API calls within a single build run.

_SECTOR_CACHE: dict[str, str] = {}
# Per-symbol headline cache: symbol → list of (headline, published_at datetime)
_HEADLINE_CACHE: dict[str, list[tuple[str, datetime]]] = {}

# Minimal fallback map used ONLY when the provider API fails.  This is NOT
# the source of truth — prefer provider metadata via _resolve_sector().
_FALLBACK_SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "GOOGL": "Technology",
    "AMZN": "Technology", "META": "Technology", "NVDA": "Technology",
    "TSLA": "Technology", "AMD": "Technology", "CRM": "Technology", "ADBE": "Technology",
    # Finance
    "JPM": "Finance", "BAC": "Finance", "GS": "Finance",
    "V": "Finance", "MA": "Finance",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "PFE": "Healthcare",
    "ABBV": "Healthcare", "MRK": "Healthcare",
    # Consumer
    "WMT": "Consumer", "KO": "Consumer", "PEP": "Consumer",
    "MCD": "Consumer", "NKE": "Consumer",
    # Energy & Industrials
    "XOM": "Energy", "CVX": "Energy", "CAT": "Industrials",
    "BA": "Industrials", "GE": "Industrials",
}


def _resolve_sector(symbol: str, fundamentals: dict | None = None) -> str:
    """
    Resolve sector for *symbol* using provider data, with fallback.

    Priority:
      1. Cache (from a previous call in this run)
      2. ``fundamentals["sector"]`` (from Yahoo Finance)
      3. ``_FALLBACK_SECTOR_MAP``
      4. ``"Unknown"``
    """
    if symbol in _SECTOR_CACHE:
        return _SECTOR_CACHE[symbol]

    sector = None
    if fundamentals:
        sector = fundamentals.get("sector")

    if not sector:
        sector = _FALLBACK_SECTOR_MAP.get(symbol, "Unknown")

    _SECTOR_CACHE[symbol] = sector
    return sector

# SPY as benchmark
BENCHMARK_SYMBOL = "SPY"


def load_symbols(symbols_str: str | None = None, symbols_file: str | None = None) -> list[str]:
    """Load symbol list from comma-separated string or file."""
    if symbols_str:
        return [s.strip().upper() for s in symbols_str.split(",") if s.strip()]

    path = symbols_file or DEFAULT_SYMBOLS_FILE
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return [line.strip().upper() for line in f if line.strip() and not line.startswith("#")]

    return []


def _timestamp_to_iso(ts) -> str:
    try:
        return ts.astimezone().isoformat()
    except Exception:
        return str(ts)


# ---------------------------------------------------------------------------
# Phase-5: Pre-compute per-symbol feature data
# ---------------------------------------------------------------------------

def _compute_per_symbol_daily_features(
    *,
    symbol: str,
    history,
    fundamentals: dict | None,
    horizon_days: int,
    min_history: int,
    use_static_fundamentals: bool,
    thresholds: dict[str, float],
    fetcher,
    enable_sentiment: bool = False,
) -> list[dict]:
    """
    Build one row per valid day for a single symbol.

    Each row contains:
      - metadata (symbol, timestamp, prices, label)
      - base features (20 original)
      - factor features (6 new)
      - microstructure features (5 new)
      - sentiment features (8 Phase-6, populated when enable_sentiment=True)
      - helper fields for cross-sectional computation (trailing_return, etc.)

    Cross-sectional features are NOT computed here; they require the full
    universe and are filled later in a second pass.
    """
    rows: list[dict] = []
    if len(history) < (min_history + horizon_days):
        return rows

    volumes = [p.volume for p in history]
    closes = [p.close for p in history]
    highs = [p.high for p in history]
    lows = [p.low for p in history]
    static_fund = fundamentals if use_static_fundamentals else None

    # Pre-fetch headlines once per symbol covering the full history window.
    # Finnhub's company-news endpoint supports date ranges (true historical);
    # Yahoo only returns recent news, so we use it as a fallback.
    all_headlines: list[tuple[str, datetime]] = []
    if enable_sentiment:
        if symbol not in _HEADLINE_CACHE:
            try:
                first_date = history[0].timestamp
                last_date = history[-1].timestamp
                # Try Finnhub first (date-parameterized → real historical coverage)
                news_items = fetcher.get_news_finnhub(
                    symbol, from_date=first_date, to_date=last_date,
                )
                if not news_items:
                    # Fallback: Yahoo (recent headlines only)
                    news_items = fetcher.get_news_yahoo(symbol)
                _HEADLINE_CACHE[symbol] = [
                    (ni.headline, ni.published_at)
                    for ni in news_items
                    if hasattr(ni, "published_at") and ni.published_at
                ]
                time.sleep(0.5)  # Rate-limit: one call per symbol
            except Exception as e:
                logger.debug(f"News fetch failed for {symbol}: {e}")
                _HEADLINE_CACHE[symbol] = []
        all_headlines = _HEADLINE_CACHE[symbol]

    for i in range(min_history - 1, len(history) - horizon_days):
        current = history[i]
        future = history[i + horizon_days]
        window = history[: i + 1]
        indicators = fetcher.calculate_indicators(window)
        if not indicators:
            continue

        avg_start = max(0, i - 19)
        avg_volume = float(np.mean(volumes[avg_start : i + 1])) if i >= avg_start else current.volume
        quote_dict = {
            "price": current.close,
            "volume": current.volume,
            "avg_volume": avg_volume,
        }

        # ------ Factor features (need full close history up to this point) ------
        closes_window = closes[: i + 1]
        factor_feats = compute_factor_features(closes_window, static_fund)

        # ------ Microstructure features ------
        highs_window = highs[: i + 1]
        lows_window = lows[: i + 1]
        volumes_window = volumes[: i + 1]
        micro_feats = compute_microstructure_features(
            closes_window, highs_window, lows_window, volumes_window
        )

        # ------ Sentiment features (Phase-6, opt-in) ------
        sentiment_feats = None
        if enable_sentiment and all_headlines:
            try:
                current_date = current.timestamp
                start_dt = current_date - timedelta(days=7)
                # Filter pre-fetched headlines to 7-day window ending at current_date
                headlines = [h for h, ts in all_headlines if start_dt <= ts <= current_date]
                timestamps = [ts for h, ts in all_headlines if start_dt <= ts <= current_date]
                sentiment_feats = compute_sentiment_features(
                    headlines=headlines or None,
                    headline_timestamps=timestamps or None,
                    fundamentals=static_fund,
                    reference_date=current_date,
                )
            except Exception as e:
                logger.debug(f"Sentiment compute failed for {symbol} {str(current.timestamp)[:10]}: {e}")

        # Base features (original 20)
        base_features = extract_features(
            indicators, static_fund, quote_dict,
            factor=factor_feats,
            microstructure=micro_feats,
            sentiment=sentiment_feats,
        )

        # Labels
        future_return_pct = ((future.close / current.close) - 1.0) * 100.0
        signal = future_return_to_signal(
            future_return_pct,
            sell_threshold=thresholds["sell_threshold"],
            strong_sell_threshold=thresholds["strong_sell_threshold"],
            buy_threshold=thresholds["buy_threshold"],
            strong_buy_threshold=thresholds["strong_buy_threshold"],
        )

        # Trailing 20-day return (for cross-sectional z-score)
        if i >= 20:
            trailing_return = (closes[i] / closes[i - 20] - 1.0) * 100.0
        else:
            trailing_return = np.nan

        # Realised vol (60-day annualised)
        if i >= 60:
            rets_60 = _returns_from_closes(closes[max(0, i - 59) : i + 1])
            realised_vol = float(np.std(rets_60, ddof=1)) * np.sqrt(252) if len(rets_60) > 1 else np.nan
        else:
            realised_vol = np.nan

        # Avg volume ratio
        vol_ratio = current.volume / avg_volume if avg_volume > 0 else np.nan

        row = {
            "symbol": symbol,
            "timestamp": _timestamp_to_iso(current.timestamp),
            "_timestamp_raw": current.timestamp,
            "signal": signal,
            "label_source": "future_return",
            "future_horizon_days": horizon_days,
            "future_return_pct": round(float(future_return_pct), 6),
            "current_price": round(float(current.close), 6),
            "future_price": round(float(future.close), 6),
            "composite_score": "",
            # Helper fields for cross-sectional pass (removed before writing)
            "_trailing_return": trailing_return,
            "_momentum_12_1": _safe_float(factor_feats.get("momentum_12_1"), np.nan),
            "_realised_vol": realised_vol,
            "_vol_ratio": vol_ratio,
            "_sector": _resolve_sector(symbol, fundamentals),
            # Store 45-feature vector temporarily
            "_base_features": base_features,
            # Store factor + micro dicts for later merging
            "_factor_feats": factor_feats,
            "_micro_feats": micro_feats,
        }
        rows.append(row)

    return rows


def _add_cross_sectional_features(
    all_rows: list[dict],
    benchmark_returns_by_date: dict[str, float],
) -> None:
    """
    Second pass: compute cross-sectional (universe-relative) features.

    Groups rows by date, computes z-scores and relative-strength across the
    universe for each date, and patches the feature vectors in-place.
    """
    from training.cross_sectional import (
        CROSS_SECTIONAL_FEATURE_NAMES,
        compute_cross_sectional_features,
    )

    # Group rows by date string (ISO date portion)
    by_date: dict[str, list[dict]] = defaultdict(list)
    for row in all_rows:
        ts = row["timestamp"]
        # Extract date portion (first 10 chars of ISO timestamp)
        date_key = str(ts)[:10]
        by_date[date_key].append(row)

    for date_key, date_rows in by_date.items():
        n = len(date_rows)
        if n < 2:
            # Single stock on this date: no cross-sectional context
            for row in date_rows:
                xs = {name: np.nan for name in CROSS_SECTIONAL_FEATURE_NAMES}
                row["_xs_feats"] = xs
            continue

        # Collect universe arrays for this date
        trailing_returns = np.array([row["_trailing_return"] for row in date_rows], dtype=np.float64)
        momentums = np.array([row["_momentum_12_1"] for row in date_rows], dtype=np.float64)
        vols = np.array([row["_realised_vol"] for row in date_rows], dtype=np.float64)
        vol_ratios = np.array([row["_vol_ratio"] for row in date_rows], dtype=np.float64)

        # Group by sector for this date
        sector_returns_map: dict[str, list[float]] = defaultdict(list)
        for row in date_rows:
            if not np.isnan(row["_trailing_return"]):
                sector_returns_map[row["_sector"]].append(row["_trailing_return"])

        benchmark_ret = benchmark_returns_by_date.get(date_key, 0.0)

        for row in date_rows:
            sector = row["_sector"]
            sector_rets = np.array(sector_returns_map.get(sector, []), dtype=np.float64)

            xs = compute_cross_sectional_features(
                symbol=row["symbol"],
                trailing_return_pct=row["_trailing_return"],
                momentum_12_1=row["_momentum_12_1"],
                realised_vol=row["_realised_vol"],
                avg_volume_ratio=row["_vol_ratio"],
                sector=sector,
                universe_trailing_returns=trailing_returns,
                universe_momentums=momentums,
                universe_vols=vols,
                universe_volume_ratios=vol_ratios,
                sector_returns=sector_rets,
                benchmark_return_pct=benchmark_ret,
            )
            row["_xs_feats"] = xs


def _finalize_rows(all_rows: list[dict]) -> list[dict]:
    """
    Convert internal row dicts into final CSV-writable dicts.

    Merges base, cross-sectional, factor, and microstructure feature values,
    removes internal helper fields.
    """
    from training.feature_engineering import FEATURE_NAMES

    csv_rows: list[dict] = []
    for row in all_rows:
        base = row["_base_features"]
        xs = row.get("_xs_feats", {})

        # Rebuild the complete 45-element feature vector:
        # extract_features already placed factor + micro features into base[20:37],
        # and sentiment slots [37:45] are NaN for historical data.
        # Cross-sectional slots (indices 20:26) are NaN. Overwrite them.
        from training.cross_sectional import CROSS_SECTIONAL_FEATURE_NAMES
        for j, name in enumerate(CROSS_SECTIONAL_FEATURE_NAMES):
            idx = 20 + j  # cross-sectional starts at index 20
            val = xs.get(name, np.nan)
            base[idx] = val if (val is not None and val == val) else np.nan

        csv_row = {
            "symbol": row["symbol"],
            "timestamp": row["timestamp"],
            "signal": row["signal"],
            "label_source": row["label_source"],
            "future_horizon_days": row["future_horizon_days"],
            "future_return_pct": row["future_return_pct"],
            "current_price": row["current_price"],
            "future_price": row["future_price"],
            "composite_score": row["composite_score"],
        }
        for name, val in zip(FEATURE_NAMES, base):
            csv_row[name] = "" if (val != val) else round(float(val), 6)  # NaN check

        csv_rows.append(csv_row)

    return csv_rows


# ---------------------------------------------------------------------------
#  Main dataset builder
# ---------------------------------------------------------------------------

def build_dataset(
    symbols: list[str],
    period: str = "3y",
    output_path: str = "data/training_data.csv",
    delay: float = 1.0,
    label_mode: str = "future",
    horizon_days: int = 5,
    min_history: int = 60,
    use_static_fundamentals: bool = False,
    sell_threshold: float = -1.0,
    strong_sell_threshold: float = -3.0,
    buy_threshold: float = 1.0,
    strong_buy_threshold: float = 3.0,
    enable_sentiment: bool = False,
) -> int:
    """
    Build training dataset in CSV format.

    Phase-5 upgrade: two-pass approach.
      Pass 1: per-symbol base + factor + micro features
      Pass 2: cross-sectional (universe-relative) features

    Returns:
        Number of rows written.
    """
    from agents.fetcher import StockFetcher
    from agents.scorer import StockScorer

    fetcher = StockFetcher()
    scorer = StockScorer()

    # Clear headline cache for fresh run
    _HEADLINE_CACHE.clear()

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "symbol",
        "timestamp",
        "signal",
        "label_source",
        "future_horizon_days",
        "future_return_pct",
        "current_price",
        "future_price",
        "composite_score",
    ] + FEATURE_NAMES

    thresholds = {
        "sell_threshold": sell_threshold,
        "strong_sell_threshold": strong_sell_threshold,
        "buy_threshold": buy_threshold,
        "strong_buy_threshold": strong_buy_threshold,
    }

    # ====== PASS 1: Per-symbol features ======
    all_rows: list[dict] = []
    benchmark_history = None

    if label_mode == "future":
        # Fetch benchmark (SPY) for excess-return calculation
        logger.info(f"Fetching benchmark ({BENCHMARK_SYMBOL}) history...")
        try:
            benchmark_history = fetcher.get_price_history(BENCHMARK_SYMBOL, period=period, interval="1d")
            logger.info(f"Benchmark: {len(benchmark_history)} bars")
        except Exception as e:
            logger.warning(f"Could not fetch benchmark: {e}")
            benchmark_history = []

        time.sleep(delay)

        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"[{idx}/{len(symbols)}] Processing {symbol}...")
            try:
                history = fetcher.get_price_history(symbol, period=period, interval="1d")
                if len(history) < 30:
                    logger.warning(f"  Insufficient history for {symbol} ({len(history)} days), skipping")
                    continue

                fundamentals = fetcher.get_fundamentals(symbol)

                symbol_rows = _compute_per_symbol_daily_features(
                    symbol=symbol,
                    history=history,
                    fundamentals=fundamentals,
                    horizon_days=horizon_days,
                    min_history=min_history,
                    use_static_fundamentals=use_static_fundamentals,
                    thresholds=thresholds,
                    fetcher=fetcher,
                    enable_sentiment=enable_sentiment,
                )
                all_rows.extend(symbol_rows)
                logger.info(f"  Generated {len(symbol_rows)} rows for {symbol}")

            except Exception as e:
                logger.error(f"  Error processing {symbol}: {e}")

            if idx < len(symbols):
                time.sleep(delay)

        # ====== PASS 2: Cross-sectional features ======
        logger.info(f"Pass 2: Computing cross-sectional features across {len(all_rows)} rows...")

        # Build benchmark trailing-return lookup by date
        benchmark_returns_by_date: dict[str, float] = {}
        if benchmark_history and len(benchmark_history) >= 21:
            bm_closes = [p.close for p in benchmark_history]
            for i in range(20, len(benchmark_history)):
                date_key = str(_timestamp_to_iso(benchmark_history[i].timestamp))[:10]
                ret = (bm_closes[i] / bm_closes[i - 20] - 1.0) * 100.0
                benchmark_returns_by_date[date_key] = round(ret, 6)

        _add_cross_sectional_features(all_rows, benchmark_returns_by_date)

        # Convert to CSV-writable dicts
        csv_rows = _finalize_rows(all_rows)
        logger.info(f"Finalized {len(csv_rows)} rows with {len(FEATURE_NAMES)} features each")

    else:
        # Legacy rule-label mode (no Phase-5 features, single snapshot)
        csv_rows = []
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"[{idx}/{len(symbols)}] Processing {symbol} (rule mode)...")
            try:
                history = fetcher.get_price_history(symbol, period=period, interval="1d")
                if len(history) < 30:
                    logger.warning(f"  Insufficient history for {symbol} ({len(history)} days), skipping")
                    continue

                fundamentals = fetcher.get_fundamentals(symbol)
                quote = fetcher.get_quote(symbol)
                if not quote:
                    logger.warning(f"  No quote for {symbol}, skipping")
                    continue
                indicators = fetcher.calculate_indicators(history)
                if not indicators:
                    logger.warning(f"  No indicators for {symbol}, skipping")
                    continue

                quote_dict = {
                    "price": quote.price,
                    "volume": quote.volume,
                    "avg_volume": quote.avg_volume,
                }
                scores = scorer.calculate_all_scores(
                    quote=quote_dict,
                    indicators=indicators,
                    fundamentals=fundamentals,
                    price_history_days=len(history),
                    has_news=False,
                )
                features = extract_features(indicators, fundamentals, quote_dict)
                row = {
                    "symbol": symbol,
                    "timestamp": _timestamp_to_iso(quote.timestamp),
                    "signal": scores.overall_signal,
                    "label_source": "rule_score",
                    "future_horizon_days": "",
                    "future_return_pct": "",
                    "current_price": round(float(quote.price), 6),
                    "future_price": "",
                    "composite_score": round(float(scores.composite_score), 6),
                }
                for name, val in zip(FEATURE_NAMES, features):
                    row[name] = "" if (val != val) else round(float(val), 6)
                csv_rows.append(row)

            except Exception as e:
                logger.error(f"  Error processing {symbol}: {e}")

            if idx < len(symbols):
                time.sleep(delay)

    # ====== Write CSV ======
    rows_written = 0
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        writer.writerows(csv_rows)
        rows_written = len(csv_rows)

    logger.info(f"Dataset written to {output_path} ({rows_written} rows, {len(FEATURE_NAMES)} features)")
    return rows_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ML training dataset from stock data")
    parser.add_argument("--symbols", type=str, help="Comma-separated symbols (e.g., AAPL,MSFT)")
    parser.add_argument("--symbols-file", type=str, help="Path to symbols file (one per line)")
    parser.add_argument("--period", default="3y", help="Historical data period (default: 3y)")
    parser.add_argument("--output", "-o", default="data/training_data.csv", help="Output CSV path")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument(
        "--label-mode",
        choices=["future", "rule"],
        default="future",
        help="Labeling mode: future-return labels (recommended) or legacy rule labels",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=5,
        help="Forward return horizon in trading days for future labels (default: 5)",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=60,
        help="Minimum trailing days before first sample (default: 60)",
    )
    parser.add_argument(
        "--use-static-fundamentals",
        action="store_true",
        help="Use latest fundamentals for all historical rows (introduces point-in-time leakage)",
    )
    parser.add_argument("--sell-threshold", type=float, default=-1.0, help="Sell threshold (%%)")
    parser.add_argument("--strong-sell-threshold", type=float, default=-3.0, help="Strong sell threshold (%%)")
    parser.add_argument("--buy-threshold", type=float, default=1.0, help="Buy threshold (%%)")
    parser.add_argument("--strong-buy-threshold", type=float, default=3.0, help="Strong buy threshold (%%)")
    parser.add_argument(
        "--sentiment",
        action="store_true",
        help="Enable historical news sentiment backfill (slower, requires API calls)",
    )
    args = parser.parse_args()

    symbols = load_symbols(args.symbols, args.symbols_file)
    if not symbols:
        print("No symbols provided. Use --symbols or --symbols-file, or create data/training_symbols.txt")
        sys.exit(1)

    print(
        "Building dataset for "
        f"{len(symbols)} symbols (period={args.period}, mode={args.label_mode}, horizon={args.horizon_days})..."
    )
    print(f"Feature vector: {len(FEATURE_NAMES)} features (20 base + {len(ALL_NEW_FEATURE_NAMES)} Phase-5 + 8 sentiment)")
    if args.sentiment:
        print("Sentiment backfill: ENABLED (will fetch news headlines per symbol)")
    count = build_dataset(
        symbols,
        period=args.period,
        output_path=args.output,
        delay=args.delay,
        label_mode=args.label_mode,
        horizon_days=args.horizon_days,
        min_history=args.min_history,
        use_static_fundamentals=args.use_static_fundamentals,
        sell_threshold=args.sell_threshold,
        strong_sell_threshold=args.strong_sell_threshold,
        buy_threshold=args.buy_threshold,
        strong_buy_threshold=args.strong_buy_threshold,
        enable_sentiment=args.sentiment,
    )
    print(f"Done: {count} rows written to {args.output}")


if __name__ == "__main__":
    main()
