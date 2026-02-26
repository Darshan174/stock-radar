"""
Cross-sectional, factor-style, and microstructure-lite feature engine.

Phase 5 of the Quant Upgrade Roadmap.

Three feature families:
  1. Cross-sectional: universe-relative z-scores, relative strength, excess return
  2. Factor-style: momentum (12-1), quality proxy, low-vol proxy
  3. Microstructure-lite: VWAP deviation, volume imbalance, volatility compression

All helpers are pure numpy, no pandas dependency needed.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from training.sentiment import SENTIMENT_FEATURE_NAMES  # Phase-6


# ---------------------------------------------------------------------------
#  Utility helpers
# ---------------------------------------------------------------------------

def _safe_float(v: Any, default: float = np.nan) -> float:
    """Best-effort float conversion; returns *default* on failure."""
    if v is None or v == "" or (isinstance(v, float) and math.isnan(v)):
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _returns_from_closes(closes: Sequence[float]) -> np.ndarray:
    """Simple arithmetic returns from a close-price series (oldest-first)."""
    c = np.asarray(closes, dtype=np.float64)
    if len(c) < 2:
        return np.array([], dtype=np.float64)
    return c[1:] / c[:-1] - 1.0


# ---------------------------------------------------------------------------
#  1. Cross-sectional features  (universe-relative)
# ---------------------------------------------------------------------------

# Feature names exported for CSV header / feature vector ordering
CROSS_SECTIONAL_FEATURE_NAMES: List[str] = [
    "xs_return_zscore",        # z-score of trailing return vs universe
    "xs_momentum_zscore",      # z-score of 12-1 momentum vs universe
    "xs_volatility_zscore",    # z-score of realised vol vs universe
    "xs_volume_zscore",        # z-score of avg volume ratio vs universe
    "relative_strength_sector",  # return rank vs sector peers  (0-1)
    "excess_return_vs_benchmark",  # stock return - benchmark return (%)
]


def _zscore(value: float, values: np.ndarray) -> float:
    """Compute z-score of *value* within array *values*."""
    if len(values) < 2:
        return 0.0
    mu = float(np.nanmean(values))
    sigma = float(np.nanstd(values, ddof=1))
    if sigma < 1e-12:
        return 0.0
    return (value - mu) / sigma


def compute_cross_sectional_features(
    *,
    symbol: str,
    trailing_return_pct: float,
    momentum_12_1: float,
    realised_vol: float,
    avg_volume_ratio: float,
    sector: Optional[str],
    # Universe-level arrays (one entry per symbol in universe, same order)
    universe_trailing_returns: np.ndarray,
    universe_momentums: np.ndarray,
    universe_vols: np.ndarray,
    universe_volume_ratios: np.ndarray,
    # Sector-peer returns (only same-sector symbols)
    sector_returns: Optional[np.ndarray] = None,
    # Benchmark return for excess-return calc
    benchmark_return_pct: float = 0.0,
) -> Dict[str, float]:
    """Return a dict keyed by ``CROSS_SECTIONAL_FEATURE_NAMES``."""

    xs_return_z = _zscore(trailing_return_pct, universe_trailing_returns)
    xs_mom_z = _zscore(momentum_12_1, universe_momentums)
    xs_vol_z = _zscore(realised_vol, universe_vols)
    xs_volr_z = _zscore(avg_volume_ratio, universe_volume_ratios)

    # Relative strength: percentile rank within sector peers
    if sector_returns is not None and len(sector_returns) > 0:
        rank = float(np.nanmean(sector_returns <= trailing_return_pct))
    else:
        rank = 0.5  # no peers → neutral

    excess = trailing_return_pct - benchmark_return_pct

    return {
        "xs_return_zscore": round(xs_return_z, 6),
        "xs_momentum_zscore": round(xs_mom_z, 6),
        "xs_volatility_zscore": round(xs_vol_z, 6),
        "xs_volume_zscore": round(xs_volr_z, 6),
        "relative_strength_sector": round(rank, 6),
        "excess_return_vs_benchmark": round(excess, 6),
    }


# ---------------------------------------------------------------------------
#  2. Factor-style features  (per-stock, from price + fundamentals)
# ---------------------------------------------------------------------------

FACTOR_FEATURE_NAMES: List[str] = [
    "momentum_12_1",         # 12-month return ex last month (%)
    "momentum_1m",           # last 1-month return (%)
    "quality_score",         # composite quality proxy  (0-1)
    "low_vol_factor",        # inverse realised vol, normalised
    "earnings_yield",        # E/P (inverse of P/E)
    "book_to_price",         # B/P (inverse of P/B)
]


def compute_factor_features(
    closes: Sequence[float],
    fundamentals: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Compute per-stock factor features from close prices and optionally fundamentals.

    ``closes`` must be oldest-first, daily, with at least ~252 bars for full
    momentum calculation.
    """
    c = np.asarray(closes, dtype=np.float64)
    fund = fundamentals or {}

    # --- Momentum 12-1  (skip last 21 days ≈ 1 month) --
    if len(c) >= 252:
        mom_12_1 = (c[-22] / c[-252] - 1.0) * 100.0
    elif len(c) >= 126:
        # fallback: 6-month ex-1-month
        skip = min(21, len(c) - 2)
        mom_12_1 = (c[-1 - skip] / c[0] - 1.0) * 100.0
    else:
        mom_12_1 = np.nan

    # --- Momentum 1M --
    if len(c) >= 22:
        mom_1m = (c[-1] / c[-22] - 1.0) * 100.0
    else:
        mom_1m = np.nan

    # --- Quality proxy (ROE × gross-margin × (1/leverage)) --
    roe = _safe_float(fund.get("roe"), np.nan)
    profit_margin = _safe_float(fund.get("profit_margin"), np.nan)
    d2e = _safe_float(fund.get("debt_to_equity"), np.nan)
    current_ratio = _safe_float(fund.get("current_ratio"), np.nan)

    quality_components: list[float] = []
    if not np.isnan(roe):
        # ROE contribution: clip to [0,0.5] and normalise to [0,1]
        quality_components.append(min(max(roe, 0.0), 0.5) / 0.5)
    if not np.isnan(profit_margin):
        quality_components.append(min(max(profit_margin, 0.0), 0.5) / 0.5)
    if not np.isnan(d2e) and d2e > 0:
        # Lower leverage is better: 1/(1+D/E), mapped so D/E=0 → 1.0
        quality_components.append(1.0 / (1.0 + d2e / 100.0))
    if not np.isnan(current_ratio):
        # current_ratio > 1.5 is good, cap at 3
        quality_components.append(min(current_ratio, 3.0) / 3.0)

    quality_score = float(np.mean(quality_components)) if quality_components else np.nan

    # --- Low-vol factor (inverse of 60-day realised vol, normalised) --
    if len(c) >= 60:
        rets_60 = _returns_from_closes(c[-60:])
        vol_60 = float(np.std(rets_60, ddof=1)) * np.sqrt(252)
        low_vol_factor = 1.0 / (1.0 + vol_60) if vol_60 > 0 else 1.0
    else:
        low_vol_factor = np.nan

    # --- Value factors --
    pe = _safe_float(fund.get("pe_ratio"), np.nan)
    pb = _safe_float(fund.get("pb_ratio"), np.nan)
    earnings_yield = (1.0 / pe) if (not np.isnan(pe) and pe > 0) else np.nan
    book_to_price = (1.0 / pb) if (not np.isnan(pb) and pb > 0) else np.nan

    return {
        "momentum_12_1": round(mom_12_1, 6) if not np.isnan(mom_12_1) else np.nan,
        "momentum_1m": round(mom_1m, 6) if not np.isnan(mom_1m) else np.nan,
        "quality_score": round(quality_score, 6) if not np.isnan(quality_score) else np.nan,
        "low_vol_factor": round(low_vol_factor, 6) if not np.isnan(low_vol_factor) else np.nan,
        "earnings_yield": round(earnings_yield, 6) if not np.isnan(earnings_yield) else np.nan,
        "book_to_price": round(book_to_price, 6) if not np.isnan(book_to_price) else np.nan,
    }


# ---------------------------------------------------------------------------
#  3. Microstructure-lite features
# ---------------------------------------------------------------------------

MICROSTRUCTURE_FEATURE_NAMES: List[str] = [
    "vwap_deviation_pct",    # distance from rolling VWAP (%)
    "volume_imbalance",      # (up-vol − down-vol) / total-vol
    "volatility_compression", # ATR(5) / ATR(20)  – breakout detector
    "high_low_range_pct",    # (H−L)/C averaged over 5 bars vs 20 bars
    "close_location_value",  # (C−L)/(H−L) last bar, measures buying pressure
]


def _atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int) -> float:
    """Average True Range over last *period* bars."""
    if len(highs) < period + 1:
        return np.nan
    tr_list: list[float] = []
    for i in range(-period, 0):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_list.append(max(hl, hc, lc))
    return float(np.mean(tr_list))


def compute_microstructure_features(
    closes: Sequence[float],
    highs: Sequence[float],
    lows: Sequence[float],
    volumes: Sequence[float],
) -> Dict[str, float]:
    """
    Compute microstructure-lite features.

    All arrays oldest-first, daily bars.
    """
    c = np.asarray(closes, dtype=np.float64)
    h = np.asarray(highs, dtype=np.float64)
    lo = np.asarray(lows, dtype=np.float64)
    v = np.asarray(volumes, dtype=np.float64)

    # --- VWAP deviation (20-bar rolling VWAP vs close) ---
    if len(c) >= 20 and len(v) >= 20:
        typical_price = (h[-20:] + lo[-20:] + c[-20:]) / 3.0
        cum_tp_vol = float(np.sum(typical_price * v[-20:]))
        cum_vol = float(np.sum(v[-20:]))
        if cum_vol > 0:
            vwap_20 = cum_tp_vol / cum_vol
            vwap_dev = (c[-1] - vwap_20) / vwap_20 * 100.0
        else:
            vwap_dev = np.nan
    else:
        vwap_dev = np.nan

    # --- Volume imbalance  (up-vol − down-vol) / total over 10 bars ---
    if len(c) >= 11 and len(v) >= 10:
        up_vol = 0.0
        down_vol = 0.0
        for i in range(-10, 0):
            if c[i] > c[i - 1]:
                up_vol += v[i]
            elif c[i] < c[i - 1]:
                down_vol += v[i]
        total_vol = up_vol + down_vol
        vol_imb = (up_vol - down_vol) / total_vol if total_vol > 0 else 0.0
    else:
        vol_imb = np.nan

    # --- Volatility compression: ATR(5) / ATR(20) ---
    atr5 = _atr(h, lo, c, 5)
    atr20 = _atr(h, lo, c, 20)
    if not np.isnan(atr5) and not np.isnan(atr20) and atr20 > 1e-12:
        vol_compress = atr5 / atr20
    else:
        vol_compress = np.nan

    # --- High-Low range compression: avg(H-L)/C over 5 bars vs 20 bars ---
    if len(c) >= 20:
        range_5 = float(np.mean((h[-5:] - lo[-5:]) / c[-5:])) * 100.0
        range_20 = float(np.mean((h[-20:] - lo[-20:]) / c[-20:])) * 100.0
        hl_range_ratio = range_5 / range_20 if range_20 > 1e-12 else np.nan
    else:
        hl_range_ratio = np.nan

    # --- Close Location Value:  (C - L) / (H - L) of last bar ---
    if len(c) >= 1:
        spread = h[-1] - lo[-1]
        clv = (c[-1] - lo[-1]) / spread if spread > 1e-12 else 0.5
    else:
        clv = np.nan

    return {
        "vwap_deviation_pct": round(vwap_dev, 6) if not np.isnan(vwap_dev) else np.nan,
        "volume_imbalance": round(vol_imb, 6) if not np.isnan(vol_imb) else np.nan,
        "volatility_compression": round(vol_compress, 6) if not np.isnan(vol_compress) else np.nan,
        "high_low_range_pct": round(hl_range_ratio, 6) if not np.isnan(hl_range_ratio) else np.nan,
        "close_location_value": round(clv, 6) if not np.isnan(clv) else np.nan,
    }


# ---------------------------------------------------------------------------
#  Combined feature names list (for CSV header / feature vector ordering)
# ---------------------------------------------------------------------------



ALL_NEW_FEATURE_NAMES: List[str] = (
    CROSS_SECTIONAL_FEATURE_NAMES
    + FACTOR_FEATURE_NAMES
    + MICROSTRUCTURE_FEATURE_NAMES
    + SENTIMENT_FEATURE_NAMES
)
"""25 new features: 17 Phase-5 + 8 Phase-6 sentiment."""
