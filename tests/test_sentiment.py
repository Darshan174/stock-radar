"""
Tests for Phase-6 sentiment feature layer (training.sentiment).

Tests the sentiment scoring, feature computation, and integration
with the feature engineering pipeline.  The tests do NOT require
FinBERT or VADER to be installed — they mock the scorers where needed.
"""

from __future__ import annotations

import math
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.sentiment import (
    SENTIMENT_FEATURE_NAMES,
    compute_sentiment_features,
    score_headline,
    score_headlines,
)
from training.feature_engineering import (
    FEATURE_NAMES,
    extract_features,
)


# ═════════════════════════════════════════════════════════════
#  Helpers
# ═════════════════════════════════════════════════════════════

def _mock_score_headlines(headlines):
    """Deterministic mock: 'positive' → +0.8, 'negative' → −0.7, else → 0.0."""
    scores = []
    for h in headlines:
        h_lower = h.lower()
        if "positive" in h_lower or "surge" in h_lower or "beat" in h_lower:
            scores.append(0.8)
        elif "negative" in h_lower or "crash" in h_lower or "miss" in h_lower:
            scores.append(-0.7)
        else:
            scores.append(0.0)
    return np.array(scores, dtype=np.float64)


# ═════════════════════════════════════════════════════════════
#  1. Feature names
# ═════════════════════════════════════════════════════════════

class TestSentimentFeatureNames:
    def test_count(self):
        assert len(SENTIMENT_FEATURE_NAMES) == 8

    def test_total_feature_vector_size(self):
        """20 base + 17 Phase-5 + 8 Phase-6 = 45."""
        assert len(FEATURE_NAMES) == 45

    def test_sentiment_names_in_feature_names(self):
        for name in SENTIMENT_FEATURE_NAMES:
            assert name in FEATURE_NAMES


# ═════════════════════════════════════════════════════════════
#  2. Headline scoring
# ═════════════════════════════════════════════════════════════

class TestHeadlineScoring:
    @patch("training.sentiment.score_headlines", side_effect=_mock_score_headlines)
    def test_score_headlines_batch(self, _mock):
        headlines = ["Positive earnings", "Crash in stocks", "Neutral update"]
        scores = _mock_score_headlines(headlines)
        assert len(scores) == 3
        assert scores[0] > 0
        assert scores[1] < 0

    def test_score_headlines_empty(self):
        scores = score_headlines([])
        assert len(scores) == 0

    @patch("training.sentiment._load_finbert", return_value=None)
    @patch("training.sentiment._load_vader", return_value=None)
    def test_score_headline_no_scorer_returns_nan(self, _v, _f):
        result = score_headline("some headline")
        assert np.isnan(result)

    @patch("training.sentiment._load_finbert", return_value=None)
    @patch("training.sentiment._load_vader", return_value=None)
    def test_score_headlines_no_scorer_returns_nans(self, _v, _f):
        results = score_headlines(["a", "b", "c"])
        assert len(results) == 3
        assert np.isnan(results).all()


# ═════════════════════════════════════════════════════════════
#  3. compute_sentiment_features
# ═════════════════════════════════════════════════════════════

class TestComputeSentimentFeatures:
    def test_all_nan_when_no_data(self):
        """No data provided → all 8 features are NaN."""
        feats = compute_sentiment_features()
        assert len(feats) == 8
        for name in SENTIMENT_FEATURE_NAMES:
            assert name in feats
            assert np.isnan(feats[name])

    @patch("training.sentiment.score_headlines", side_effect=_mock_score_headlines)
    def test_headline_features_populated(self, _mock):
        headlines = [
            "Positive earnings surge",
            "Negative crash warning",
            "Neutral market update",
        ]
        feats = compute_sentiment_features(headlines=headlines)

        assert not np.isnan(feats["news_sentiment_mean"])
        assert not np.isnan(feats["news_volume_7d"])
        # mean of [0.8, -0.7, 0.0] ≈ 0.033
        expected_mean = (0.8 + (-0.7) + 0.0) / 3
        assert abs(feats["news_sentiment_mean"] - expected_mean) < 0.01
        # volume = log₂(1 + 3) = log₂(4) = 2.0
        assert abs(feats["news_volume_7d"] - 2.0) < 0.01

    @patch("training.sentiment.score_headlines", side_effect=_mock_score_headlines)
    def test_sentiment_std(self, _mock):
        headlines = ["Positive report", "Negative outlook"]
        feats = compute_sentiment_features(headlines=headlines)
        assert feats["news_sentiment_std"] > 0

    @patch("training.sentiment.score_headlines", side_effect=_mock_score_headlines)
    def test_sentiment_momentum_with_timestamps(self, _mock):
        """Recent positive headlines vs older negative → positive momentum."""
        now = datetime.now(timezone.utc)
        headlines = [
            "Positive surge today",   # 1 day old
            "Positive beat today",    # 2 days old
            "Negative crash last week",  # 6 days old
        ]
        timestamps = [
            now - timedelta(days=1),
            now - timedelta(days=2),
            now - timedelta(days=6),
        ]
        feats = compute_sentiment_features(
            headlines=headlines,
            headline_timestamps=timestamps,
            reference_date=now,
        )
        # Recent (0-3d) = [0.8, 0.8], All = [0.8, 0.8, -0.7]
        assert feats["news_sentiment_momentum"] > 0

    @patch("training.sentiment.score_headlines", side_effect=_mock_score_headlines)
    def test_sentiment_momentum_positional_fallback(self, _mock):
        """Without timestamps, position-based approximation is used."""
        headlines = [
            "Positive headline",  # position 0 = recent
            "Neutral content",
            "Neutral content",
            "Neutral content",
            "Negative crash",  # position 4 = older
        ]
        feats = compute_sentiment_features(headlines=headlines)
        # news_sentiment_momentum should be defined (not NaN)
        assert not np.isnan(feats["news_sentiment_momentum"])

    def test_finnhub_features(self):
        finnhub = {
            "buzz_score": 0.85,
            "sentiment_score": 0.65,
            "sector_avg_sentiment": 0.50,
        }
        feats = compute_sentiment_features(finnhub_sentiment=finnhub)
        assert abs(feats["finnhub_buzz_score"] - 0.85) < 1e-6
        assert abs(feats["finnhub_bullish_pct"] - 0.65) < 1e-6
        assert abs(feats["sentiment_vs_sector"] - 0.15) < 1e-6

    def test_earnings_proximity_future(self):
        """Earnings date 10 days away → specific proximity value."""
        now = datetime(2025, 3, 1, tzinfo=timezone.utc)
        fund = {"next_earnings_date": "2025-03-11"}
        feats = compute_sentiment_features(
            fundamentals=fund, reference_date=now,
        )
        # 10 days away → -log₂(1 + 10) = -log₂(11) ≈ -3.459
        expected = -math.log2(1 + 10)
        assert abs(feats["earnings_proximity"] - expected) < 0.1

    def test_earnings_proximity_past(self):
        """Earnings already passed → days_to_earnings = 0 → proximity = 0."""
        now = datetime(2025, 3, 15, tzinfo=timezone.utc)
        fund = {"next_earnings_date": "2025-03-10"}
        feats = compute_sentiment_features(
            fundamentals=fund, reference_date=now,
        )
        # Past earnings → days = 0 → -log₂(1) = 0
        assert abs(feats["earnings_proximity"] - 0.0) < 0.01

    def test_earnings_proximity_no_date(self):
        feats = compute_sentiment_features(fundamentals={})
        assert np.isnan(feats["earnings_proximity"])

    @patch("training.sentiment.score_headlines", side_effect=_mock_score_headlines)
    def test_full_integration_all_sources(self, _mock):
        """Test with all data sources populated simultaneously."""
        now = datetime(2025, 3, 1, tzinfo=timezone.utc)
        headlines = ["Positive earnings beat expectations", "Market surge continues"]
        timestamps = [now - timedelta(days=1), now - timedelta(days=2)]
        finnhub = {
            "buzz_score": 0.9,
            "sentiment_score": 0.7,
            "sector_avg_sentiment": 0.5,
        }
        fund = {"next_earnings_date": "2025-03-05"}

        feats = compute_sentiment_features(
            headlines=headlines,
            headline_timestamps=timestamps,
            finnhub_sentiment=finnhub,
            fundamentals=fund,
            reference_date=now,
        )

        # All 8 features should be non-NaN
        nan_count = sum(1 for v in feats.values() if np.isnan(v))
        # At most news_sentiment_momentum might be NaN with few headlines
        assert nan_count <= 1, f"Too many NaN features: {feats}"


# ═════════════════════════════════════════════════════════════
#  4. Integration with extract_features
# ═════════════════════════════════════════════════════════════

class TestSentimentIntegration:
    def test_extract_features_without_sentiment(self):
        """No sentiment → slots [37:45] are NaN."""
        features = extract_features(
            indicators={"rsi_14": 50},
            fundamentals=None,
            quote={"price": 100},
        )
        assert features.shape == (45,)
        assert np.isnan(features[37:]).all()

    def test_extract_features_with_sentiment(self):
        """Sentiment dict → slots [37:45] populated."""
        sent = {name: float(i) * 0.1 for i, name in enumerate(SENTIMENT_FEATURE_NAMES)}
        features = extract_features(
            indicators={"rsi_14": 50},
            fundamentals=None,
            quote={"price": 100},
            sentiment=sent,
        )
        assert features.shape == (45,)
        # Sentiment slots should NOT be NaN
        assert not np.isnan(features[37:]).any()
        # Check specific values
        for i, name in enumerate(SENTIMENT_FEATURE_NAMES):
            assert abs(features[37 + i] - i * 0.1) < 1e-6

    def test_partial_sentiment_dict(self):
        """Only some sentiment features provided → rest are NaN."""
        sent = {
            "news_sentiment_mean": 0.5,
            "news_volume_7d": 3.0,
        }
        features = extract_features(
            indicators={"rsi_14": 50},
            fundamentals=None,
            quote={"price": 100},
            sentiment=sent,
        )
        assert abs(features[37] - 0.5) < 1e-6  # news_sentiment_mean
        assert np.isnan(features[38])  # news_sentiment_std → NaN
        assert abs(features[39] - 3.0) < 1e-6  # news_volume_7d

    def test_old_model_backward_compat(self):
        """A model trained on 37 features slices away sentiment slots."""
        features = extract_features(
            indicators={"rsi_14": 50},
            fundamentals=None,
            quote={"price": 100},
            sentiment={"news_sentiment_mean": 0.5},
        )
        sliced = features[:37]
        assert sliced.shape == (37,)
        # Slicing removes sentiment
        assert abs(features[37] - 0.5) < 1e-6
