"""Market regime detection utilities for trading models."""

from __future__ import annotations

from typing import Any, Mapping


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert values to float with safe fallback."""
    try:
        if value in (None, "", "nan"):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _clip01(value: float) -> float:
    """Clamp value to [0, 1]."""
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def classify_market_regime(indicators: Mapping[str, Any] | None) -> dict[str, float | str]:
    """
    Classify market regime from common technical indicators.

    Regimes:
      - trending
      - mean_reverting
      - high_volatility
      - neutral
    """
    ind = indicators or {}

    adx = _safe_float(ind.get("adx"), default=20.0)
    atr_pct = _safe_float(ind.get("atr_pct"), default=1.5)
    rsi = _safe_float(ind.get("rsi_14"), default=50.0)
    p20 = _safe_float(ind.get("price_vs_sma20_pct"), default=0.0)
    p50 = _safe_float(ind.get("price_vs_sma50_pct"), default=0.0)
    macd_hist = _safe_float(ind.get("macd_histogram"), default=0.0)

    adx_norm = _clip01(adx / 50.0)
    vol_norm = _clip01(atr_pct / 4.0)

    # Are short and medium trend directions aligned?
    trend_alignment = 1.0 if (p20 == 0.0 or p50 == 0.0 or (p20 * p50) > 0) else 0.0
    trend_magnitude = _clip01((abs(p20) + abs(p50)) / 12.0)
    macd_trend = _clip01(abs(macd_hist) / 2.0)

    # RSI extreme is useful for mean-reversion and high-vol states.
    rsi_extreme = _clip01(abs(rsi - 50.0) / 25.0)

    trend_score = 0.45 * adx_norm + 0.25 * trend_alignment + 0.2 * trend_magnitude + 0.1 * macd_trend
    mean_reversion_score = 0.5 * (1.0 - adx_norm) + 0.3 * (1.0 - vol_norm) + 0.2 * rsi_extreme
    high_vol_score = 0.7 * vol_norm + 0.2 * rsi_extreme + 0.1 * adx_norm

    score_map = {
        "trending": trend_score,
        "mean_reverting": mean_reversion_score,
        "high_volatility": high_vol_score,
        "neutral": 0.45,
    }

    # Guardrails to reduce overfitting-style regime flips.
    if high_vol_score >= 0.62:
        regime = "high_volatility"
        confidence = high_vol_score
    elif trend_score >= 0.58:
        regime = "trending"
        confidence = trend_score
    elif mean_reversion_score >= 0.55:
        regime = "mean_reverting"
        confidence = mean_reversion_score
    else:
        regime = "neutral"
        confidence = max(0.35, min(0.55, max(score_map.values()) - 0.05))

    return {
        "regime": regime,
        "confidence": round(float(_clip01(confidence)), 4),
        "trend_score": round(float(trend_score), 4),
        "mean_reversion_score": round(float(mean_reversion_score), 4),
        "high_volatility_score": round(float(high_vol_score), 4),
        "volatility_pct": round(float(atr_pct), 4),
        "trend_strength": round(float(adx), 4),
    }


def regime_risk_factor(regime: str | None) -> float:
    """Map regime to a conservative risk multiplier."""
    key = str(regime or "neutral").strip().lower()
    if key == "trending":
        return 1.0
    if key == "mean_reverting":
        return 0.8
    if key == "high_volatility":
        return 0.55
    return 0.7
