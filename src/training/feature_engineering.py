"""
Feature engineering + label utilities for ML signal prediction.

Original:   ~20 numerical features from indicator/fundamental/quote dicts.
Phase-5:  +17 cross-sectional, factor-style, and microstructure features.
Phase-6:  +8 sentiment features (news/FinBERT, Finnhub, earnings proximity).

Combined feature vector has 45 columns (see FEATURE_NAMES).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from training.cross_sectional import (
    ALL_NEW_FEATURE_NAMES,
    CROSS_SECTIONAL_FEATURE_NAMES,
    FACTOR_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_NAMES,
)

# Signal label encoding
SIGNAL_LABELS = ["strong_sell", "sell", "hold", "buy", "strong_buy"]
SIGNAL_TO_INT = {s: i for i, s in enumerate(SIGNAL_LABELS)}
INT_TO_SIGNAL = {i: s for i, s in enumerate(SIGNAL_LABELS)}

# --- Original features (20) ---
BASE_FEATURE_NAMES = [
    # Technical / Momentum (8)
    "rsi_14",
    "macd",
    "macd_signal",
    "macd_histogram",
    "price_vs_sma20_pct",
    "price_vs_sma50_pct",
    "bollinger_width_pct",
    "volume_ratio",
    # Volatility / Risk (4)
    "atr_pct",
    "adx",
    "plus_di",
    "minus_di",
    # Valuation (4)
    "pe_ratio",
    "pb_ratio",
    "peg_ratio",
    "dividend_yield",
    # Quality (4)
    "roe",
    "profit_margin",
    "debt_to_equity",
    "current_ratio",
]

# --- Combined feature list  (20 + 17 + 8 = 45) ---
FEATURE_NAMES: List[str] = BASE_FEATURE_NAMES + ALL_NEW_FEATURE_NAMES
"""
45 feature names in canonical order:
  [0:20]  – original base features
  [20:26] – cross-sectional (6)
  [26:32] – factor-style (6)
  [32:37] – microstructure-lite (5)
  [37:45] – sentiment (8)  [Phase-6]
"""


def encode_signal(signal: str) -> int:
    """Encode a signal string to an integer label."""
    return SIGNAL_TO_INT.get(signal.lower().strip(), 2)  # default hold


def decode_signal(label: int) -> str:
    """Decode an integer label back to a signal string."""
    return INT_TO_SIGNAL.get(label, "hold")


def future_return_to_signal(
    future_return_pct: float,
    sell_threshold: float = -1.0,
    strong_sell_threshold: float = -3.0,
    buy_threshold: float = 1.0,
    strong_buy_threshold: float = 3.0,
) -> str:
    """
    Convert forward return (%) into a 5-class trading label.

    Example:
        -4.2 -> strong_sell
        -1.7 -> sell
         0.3 -> hold
         1.4 -> buy
         5.1 -> strong_buy
    """
    if future_return_pct <= strong_sell_threshold:
        return "strong_sell"
    if future_return_pct <= sell_threshold:
        return "sell"
    if future_return_pct >= strong_buy_threshold:
        return "strong_buy"
    if future_return_pct >= buy_threshold:
        return "buy"
    return "hold"


def extract_features(
    indicators: Dict[str, Any],
    fundamentals: Optional[Dict[str, Any]],
    quote: Optional[Dict[str, Any]],
    *,
    cross_sectional: Optional[Dict[str, float]] = None,
    factor: Optional[Dict[str, float]] = None,
    microstructure: Optional[Dict[str, float]] = None,
    sentiment: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    """
    Extract a fixed-size feature vector from indicator/fundamental/quote dicts.

    New optional keyword arguments supply Phase-5/6 features; when absent, the
    corresponding slots are filled with NaN (graceful degradation for
    inference without universe context).

    Returns:
        1-D numpy array of shape (len(FEATURE_NAMES),). Missing values are NaN.
    """
    fund = fundamentals or {}
    q = quote or {}

    # Bollinger width as % of price
    bb_upper = indicators.get("bollinger_upper")
    bb_lower = indicators.get("bollinger_lower")
    price = q.get("price") or indicators.get("sma_20")
    if bb_upper and bb_lower and price and price > 0:
        bollinger_width_pct = (bb_upper - bb_lower) / price * 100
    else:
        bollinger_width_pct = np.nan

    # Volume ratio
    vol = q.get("volume")
    avg_vol = q.get("avg_volume")
    if vol and avg_vol and avg_vol > 0:
        volume_ratio = vol / avg_vol
    else:
        volume_ratio = np.nan

    def _get(d: dict, key: str) -> float:
        v = d.get(key)
        return float(v) if v is not None else np.nan

    # --- Base features (20) ---
    base = [
        _get(indicators, "rsi_14"),
        _get(indicators, "macd"),
        _get(indicators, "macd_signal"),
        _get(indicators, "macd_histogram"),
        _get(indicators, "price_vs_sma20_pct"),
        _get(indicators, "price_vs_sma50_pct"),
        bollinger_width_pct,
        volume_ratio,
        _get(indicators, "atr_pct"),
        _get(indicators, "adx"),
        _get(indicators, "plus_di"),
        _get(indicators, "minus_di"),
        _get(fund, "pe_ratio"),
        _get(fund, "pb_ratio"),
        _get(fund, "peg_ratio"),
        _get(fund, "dividend_yield"),
        _get(fund, "roe"),
        _get(fund, "profit_margin"),
        _get(fund, "debt_to_equity"),
        _get(fund, "current_ratio"),
    ]

    # --- Phase-5 cross-sectional features (6) ---
    xs = cross_sectional or {}
    xs_vals = [xs.get(name, np.nan) for name in CROSS_SECTIONAL_FEATURE_NAMES]

    # --- Phase-5 factor features (6) ---
    fac = factor or {}
    fac_vals = [fac.get(name, np.nan) for name in FACTOR_FEATURE_NAMES]

    # --- Phase-5 microstructure features (5) ---
    mic = microstructure or {}
    mic_vals = [mic.get(name, np.nan) for name in MICROSTRUCTURE_FEATURE_NAMES]

    # --- Phase-6 sentiment features (8) ---
    from training.sentiment import SENTIMENT_FEATURE_NAMES as _sfn
    sent = sentiment or {}
    sent_vals = [sent.get(name, np.nan) for name in _sfn]

    features = np.array(
        base + xs_vals + fac_vals + mic_vals + sent_vals,
        dtype=np.float64,
    )

    return features


def extract_features_batch(
    rows: List[Dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract features and labels from a list of row dicts (as loaded from CSV).

    Each row must have keys matching FEATURE_NAMES plus a 'signal' column.
    Missing Phase-5 columns gracefully default to NaN.

    Returns:
        (X, y) where X is (n_samples, n_features) and y is (n_samples,) int labels.
    """
    X_rows = []
    y_rows = []

    for row in rows:
        feat = np.array(
            [float(row[name]) if row.get(name) not in (None, "", "nan") else np.nan for name in FEATURE_NAMES],
            dtype=np.float64,
        )
        X_rows.append(feat)
        y_rows.append(encode_signal(row["signal"]))

    return np.array(X_rows), np.array(y_rows)
