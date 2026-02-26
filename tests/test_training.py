"""Unit tests for the training pipeline."""

import os
import sys
import json
import tempfile

import numpy as np
import pytest

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.feature_engineering import (
    FEATURE_NAMES,
    SIGNAL_LABELS,
    encode_signal,
    decode_signal,
    future_return_to_signal,
    extract_features,
    extract_features_batch,
)


class TestEncoding:
    """Test signal encoding/decoding."""

    def test_encode_known_signals(self):
        assert encode_signal("strong_sell") == 0
        assert encode_signal("sell") == 1
        assert encode_signal("hold") == 2
        assert encode_signal("buy") == 3
        assert encode_signal("strong_buy") == 4

    def test_decode_known_labels(self):
        assert decode_signal(0) == "strong_sell"
        assert decode_signal(2) == "hold"
        assert decode_signal(4) == "strong_buy"

    def test_encode_decode_roundtrip(self):
        for signal in SIGNAL_LABELS:
            assert decode_signal(encode_signal(signal)) == signal

    def test_encode_unknown_defaults_to_hold(self):
        assert encode_signal("unknown") == 2  # hold

    def test_encode_case_insensitive(self):
        assert encode_signal("BUY") == 3
        assert encode_signal("Strong_Buy") == 4

    def test_future_return_to_signal_thresholds(self):
        assert future_return_to_signal(-4.0) == "strong_sell"
        assert future_return_to_signal(-1.5) == "sell"
        assert future_return_to_signal(0.0) == "hold"
        assert future_return_to_signal(1.5) == "buy"
        assert future_return_to_signal(4.0) == "strong_buy"


class TestFeatureExtraction:
    """Test feature extraction from financial data."""

    def test_extract_features_shape(self):
        indicators = {"rsi_14": 55.0, "macd": 2.5, "macd_signal": 1.8}
        features = extract_features(indicators, None, None)
        assert features.shape == (len(FEATURE_NAMES),)

    def test_extract_features_known_values(self):
        indicators = {
            "rsi_14": 62.5,
            "macd": 3.2,
            "macd_signal": 2.1,
            "macd_histogram": 1.1,
            "price_vs_sma20_pct": 1.5,
            "price_vs_sma50_pct": 3.2,
            "atr_pct": 1.8,
            "adx": 32.0,
            "plus_di": 28.0,
            "minus_di": 18.0,
        }
        fundamentals = {
            "pe_ratio": 18.5,
            "pb_ratio": 1.8,
            "roe": 0.18,
            "profit_margin": 0.12,
        }
        quote = {"price": 150.0, "volume": 1000000, "avg_volume": 800000}

        features = extract_features(indicators, fundamentals, quote)

        # Check RSI is at the right index
        assert features[0] == 62.5  # rsi_14
        assert features[1] == 3.2  # macd
        assert features[12] == 18.5  # pe_ratio

    def test_missing_values_are_nan(self):
        features = extract_features({}, None, None)
        assert np.isnan(features).sum() == len(FEATURE_NAMES)

    def test_volume_ratio_calculation(self):
        quote = {"price": 100, "volume": 2000, "avg_volume": 1000}
        features = extract_features({}, None, quote)
        # volume_ratio is at index 7
        assert features[7] == 2.0

    def test_bollinger_width_calculation(self):
        indicators = {"bollinger_upper": 110, "bollinger_lower": 90}
        quote = {"price": 100}
        features = extract_features(indicators, None, quote)
        # bollinger_width_pct at index 6: (110-90)/100*100 = 20.0
        assert features[6] == 20.0


class TestFeatureBatch:
    """Test batch feature extraction from CSV-like rows."""

    def test_batch_extraction(self):
        rows = [
            {name: "1.0" for name in FEATURE_NAMES} | {"signal": "buy"},
            {name: "2.0" for name in FEATURE_NAMES} | {"signal": "sell"},
        ]
        X, y = extract_features_batch(rows)
        assert X.shape == (2, len(FEATURE_NAMES))
        assert y.shape == (2,)
        assert y[0] == encode_signal("buy")
        assert y[1] == encode_signal("sell")

    def test_batch_handles_missing(self):
        rows = [
            {name: "" for name in FEATURE_NAMES} | {"signal": "hold"},
        ]
        X, y = extract_features_batch(rows)
        assert np.isnan(X).sum() == len(FEATURE_NAMES)


class TestPredictor:
    """Test predictor with a mock model."""

    def test_signal_predictor_with_mock(self, tmp_path):
        """Test SignalPredictor with a simple mock pipeline."""
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        import joblib

        # Create a tiny training set
        rng = np.random.RandomState(42)
        X_train = rng.randn(50, len(FEATURE_NAMES))
        y_train = rng.randint(0, 5, size=50)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        model_path = str(tmp_path / "test_model.joblib")
        joblib.dump(pipeline, model_path)

        from training.predictor import SignalPredictor

        predictor = SignalPredictor(model_path)
        result = predictor.predict(
            indicators={"rsi_14": 55, "macd": 1.0},
            fundamentals={"pe_ratio": 20},
            quote={"price": 100, "volume": 1000, "avg_volume": 800},
        )

        assert "signal" in result
        assert result["signal"] in SIGNAL_LABELS
        assert 0 <= result["confidence"] <= 1
        assert "probabilities" in result
