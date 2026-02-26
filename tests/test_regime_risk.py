"""Tests for market regime and risk-based position sizing."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.regime import classify_market_regime, regime_risk_factor
from training.risk import calculate_position_size


def test_regime_detects_trending_market():
    regime = classify_market_regime(
        {
            "adx": 38,
            "atr_pct": 1.4,
            "rsi_14": 63,
            "price_vs_sma20_pct": 3.0,
            "price_vs_sma50_pct": 5.2,
            "macd_histogram": 0.9,
        }
    )
    assert regime["regime"] == "trending"
    assert regime["confidence"] > 0.5


def test_regime_detects_high_volatility_market():
    regime = classify_market_regime(
        {
            "adx": 24,
            "atr_pct": 5.8,
            "rsi_14": 42,
            "price_vs_sma20_pct": -0.7,
            "price_vs_sma50_pct": -1.1,
            "macd_histogram": -0.2,
        }
    )
    assert regime["regime"] == "high_volatility"


def test_position_size_scales_with_confidence():
    low_conf = calculate_position_size(signal="buy", confidence=0.38, regime="trending")
    high_conf = calculate_position_size(signal="buy", confidence=0.86, regime="trending")
    assert abs(high_conf["position_size"]) > abs(low_conf["position_size"])


def test_position_size_reduced_in_high_volatility_regime():
    trending = calculate_position_size(
        signal="strong_buy",
        confidence=0.9,
        regime="trending",
        volatility_pct=1.2,
    )
    high_vol = calculate_position_size(
        signal="strong_buy",
        confidence=0.9,
        regime="high_volatility",
        volatility_pct=5.0,
    )
    assert abs(high_vol["position_size"]) < abs(trending["position_size"])


def test_regime_risk_factor_mapping():
    assert regime_risk_factor("trending") > regime_risk_factor("high_volatility")
    assert regime_risk_factor("neutral") > 0
