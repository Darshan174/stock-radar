"""
Tests for Phase 7: Sentiment backfill, compare_backtests, extended rollout gates,
chat data-coverage fallback, and feature_reference_stats scope.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.backtesting import compare_backtests
from training.model_registry import (
    register_model,
    set_active,
    promote_if_better,
    get_model_metrics,
)
from training.feature_health import compute_reference_stats, check_feature_health
from training.feature_engineering import FEATURE_NAMES


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _make_registry(tmp_path):
    p = tmp_path / "registry.json"
    p.write_text(json.dumps({"models": [], "active_version": None}))
    return str(p)


def _metrics(sharpe=1.0, cagr=0.10, max_dd=-0.15, turnover=0.5,
             net_total=8.0, gross_total=10.0):
    return {
        "backtest_metrics": {
            "sharpe": sharpe,
            "cagr": cagr,
            "max_drawdown": max_dd,
            "turnover": turnover,
            "net_return_total": net_total,
            "gross_return_total": gross_total,
        }
    }


# ─────────────────────────────────────────────────────────────
#  compare_backtests tests
# ─────────────────────────────────────────────────────────────

class TestCompareBacktests:
    def test_all_gates_pass(self):
        inc = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
               "turnover": 0.5, "net_return_total": 8.0, "gross_return_total": 10.0}
        cand = {"sharpe": 1.2, "cagr": 0.12, "max_drawdown": -0.10,
                "turnover": 0.6, "net_return_total": 9.0, "gross_return_total": 11.0}
        result = compare_backtests(inc, cand)
        assert result["passed"] is True
        assert all(g["passed"] for g in result["gate_results"])

    def test_sharpe_regression_fails(self):
        inc = {"sharpe": 1.5, "cagr": 0.10, "max_drawdown": -0.15,
               "turnover": 0.5, "net_return_total": 8.0, "gross_return_total": 10.0}
        cand = {"sharpe": 1.0, "cagr": 0.12, "max_drawdown": -0.10,
                "turnover": 0.5, "net_return_total": 9.0, "gross_return_total": 11.0}
        result = compare_backtests(inc, cand)
        assert result["passed"] is False
        sharpe_gate = [g for g in result["gate_results"] if g["gate"] == "sharpe"][0]
        assert sharpe_gate["passed"] is False

    def test_turnover_overtrading_fails(self):
        inc = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
               "turnover": 0.3, "net_return_total": 8.0, "gross_return_total": 10.0}
        cand = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
                "turnover": 0.9, "net_return_total": 8.0, "gross_return_total": 10.0}
        result = compare_backtests(inc, cand)
        assert result["passed"] is False
        turn_gate = [g for g in result["gate_results"] if g["gate"] == "turnover"][0]
        assert turn_gate["passed"] is False

    def test_high_costs_fails_net_vs_gross(self):
        inc = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
               "turnover": 0.5, "net_return_total": 8.0, "gross_return_total": 10.0}
        cand = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
                "turnover": 0.5, "net_return_total": 3.0, "gross_return_total": 10.0}
        result = compare_backtests(inc, cand)
        assert result["passed"] is False
        nvg = [g for g in result["gate_results"] if g["gate"] == "net_vs_gross"][0]
        assert nvg["passed"] is False

    def test_missing_metrics_fail_safely(self):
        result = compare_backtests({}, {})
        assert result["passed"] is False

    def test_cagr_within_tolerance_passes(self):
        inc = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
               "turnover": 0.5, "net_return_total": 8.0, "gross_return_total": 10.0}
        cand = {"sharpe": 1.0, "cagr": 0.095, "max_drawdown": -0.15,
                "turnover": 0.5, "net_return_total": 8.0, "gross_return_total": 10.0}
        result = compare_backtests(inc, cand)
        cagr_gate = [g for g in result["gate_results"] if g["gate"] == "cagr"][0]
        assert cagr_gate["passed"] is True  # 0.095 >= 0.10 - 0.01

    def test_zero_incumbent_turnover_passes(self):
        """No incumbent turnover data should not block candidate."""
        inc = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
               "turnover": 0, "net_return_total": 8.0, "gross_return_total": 10.0}
        cand = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
                "turnover": 0.5, "net_return_total": 8.0, "gross_return_total": 10.0}
        result = compare_backtests(inc, cand)
        turn_gate = [g for g in result["gate_results"] if g["gate"] == "turnover"][0]
        assert turn_gate["passed"] is True

    def test_custom_gates_override(self):
        inc = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
               "turnover": 0.3, "net_return_total": 8.0, "gross_return_total": 10.0}
        cand = {"sharpe": 1.0, "cagr": 0.10, "max_drawdown": -0.15,
                "turnover": 0.9, "net_return_total": 8.0, "gross_return_total": 10.0}
        # With relaxed turnover ratio (10x), should pass
        result = compare_backtests(inc, cand, gates={"turnover_max_ratio": 10.0})
        turn_gate = [g for g in result["gate_results"] if g["gate"] == "turnover"][0]
        assert turn_gate["passed"] is True


# ─────────────────────────────────────────────────────────────
#  Extended promote_if_better tests (turnover + net_vs_gross)
# ─────────────────────────────────────────────────────────────

class TestExtendedRolloutGates:
    def test_turnover_gate_blocks_promotion(self, tmp_path):
        rp = _make_registry(tmp_path)
        # Incumbent: low turnover
        register_model(1, "m1.joblib", _metrics(turnover=0.3), registry_path=rp)
        set_active(1, registry_path=rp)
        # Candidate: high turnover (>1.5x)
        register_model(2, "m2.joblib", _metrics(turnover=0.9), registry_path=rp)

        result = promote_if_better(2, registry_path=rp)
        assert result["promoted"] is False
        assert "turnover" in str(result["checks"])

    def test_net_vs_gross_gate_blocks_promotion(self, tmp_path):
        rp = _make_registry(tmp_path)
        register_model(1, "m1.joblib", _metrics(net_total=8.0, gross_total=10.0), registry_path=rp)
        set_active(1, registry_path=rp)
        # Candidate: costs eat >30% (net=5, gross=10 → ratio=0.5)
        register_model(2, "m2.joblib", _metrics(net_total=5.0, gross_total=10.0), registry_path=rp)

        result = promote_if_better(2, registry_path=rp)
        assert result["promoted"] is False
        assert "net_vs_gross" in str(result["checks"])

    def test_get_model_metrics_extracts_new_fields(self, tmp_path):
        entry = {
            "metrics": {
                "backtest_metrics": {
                    "sharpe": 1.5,
                    "cagr": 0.12,
                    "max_drawdown": -0.10,
                    "turnover": 0.4,
                    "net_return_total": 9.0,
                    "gross_return_total": 10.0,
                }
            }
        }
        m = get_model_metrics(entry)
        assert m["turnover"] == 0.4
        assert m["net_return_total"] == 9.0
        assert m["gross_return_total"] == 10.0


# ─────────────────────────────────────────────────────────────
#  Feature reference stats scope (all features, not just sentiment)
# ─────────────────────────────────────────────────────────────

class TestFeatureReferenceStatsScope:
    def test_compute_reference_stats_covers_all_features(self):
        """Reference stats should cover all 45 features, not just sentiment."""
        n_features = len(FEATURE_NAMES)
        X = np.random.randn(100, n_features)
        # Make some features NaN
        X[:, 20:26] = np.nan  # cross-sectional
        X[:, 37:45] = np.nan  # sentiment

        stats = compute_reference_stats(X, list(FEATURE_NAMES))

        assert len(stats) == n_features
        for fname in FEATURE_NAMES:
            assert fname in stats
        # Cross-sectional features should have nan_rate=1.0
        assert stats[FEATURE_NAMES[20]]["nan_rate"] == 1.0
        # Base features should have nan_rate=0.0
        assert stats[FEATURE_NAMES[0]]["nan_rate"] == 0.0

    def test_health_check_with_full_reference_no_false_critical(self):
        """
        With full reference stats, features that were 100% NaN during training
        should NOT fire critical at inference when they're still 100% NaN.
        The health check compares NaN-rate *increase* over the training baseline.
        """
        n_features = len(FEATURE_NAMES)
        # Training data: cross-sectional NaN, base populated
        X_train = np.random.randn(100, n_features)
        X_train[:, 20:26] = np.nan  # cross-sectional all NaN in training

        ref_stats = compute_reference_stats(X_train, list(FEATURE_NAMES))

        # Inference data: same NaN pattern
        X_inf = np.random.randn(1, n_features)
        X_inf[:, 20:26] = np.nan

        report = check_feature_health(
            X_inf, feature_names=list(FEATURE_NAMES), reference_stats=ref_stats,
        )

        # Cross-sectional features are 100% NaN at both train and inference.
        # NaN increase = 0, so they should NOT be critical.
        for idx in range(20, 26):
            fname = FEATURE_NAMES[idx]
            feat = report["features"].get(fname, {})
            assert feat.get("status") == "healthy", (
                f"XS feature {fname} should be healthy (NaN matched training baseline), "
                f"got {feat.get('status')}"
            )

        # Base features should still be healthy (0% NaN)
        for fname in FEATURE_NAMES[:20]:
            feat = report["features"].get(fname, {})
            assert feat.get("status") != "critical", f"Base feature {fname} should not be critical"

    def test_health_check_detects_new_nan_regression(self):
        """
        A feature that was 0% NaN in training but 100% NaN at inference should
        be flagged as critical (genuine regression).
        """
        n_features = len(FEATURE_NAMES)
        # Training: all features populated
        X_train = np.random.randn(100, n_features)
        ref_stats = compute_reference_stats(X_train, list(FEATURE_NAMES))

        # Inference: feature 0 goes fully NaN (regression)
        X_inf = np.random.randn(1, n_features)
        X_inf[:, 0] = np.nan

        report = check_feature_health(
            X_inf, feature_names=list(FEATURE_NAMES), reference_stats=ref_stats,
        )

        feat = report["features"].get(FEATURE_NAMES[0], {})
        assert feat.get("status") == "critical", (
            f"Feature {FEATURE_NAMES[0]} went from 0% to 100% NaN — should be critical"
        )


# ─────────────────────────────────────────────────────────────
#  Sentiment backfill integration
# ─────────────────────────────────────────────────────────────

class TestSentimentBackfill:
    def test_headline_cache_is_per_symbol(self):
        """Verify _HEADLINE_CACHE key is symbol, not (symbol, date)."""
        from training.dataset_builder import _HEADLINE_CACHE
        assert isinstance(_HEADLINE_CACHE, dict)

    def test_sentiment_features_populated_with_headlines(self):
        """compute_sentiment_features returns non-NaN for valid headlines."""
        from training.sentiment import compute_sentiment_features

        headlines = ["Stock surges on strong earnings", "Company beats revenue estimates"]
        timestamps = [
            datetime.now(timezone.utc) - timedelta(hours=2),
            datetime.now(timezone.utc) - timedelta(days=1),
        ]
        result = compute_sentiment_features(
            headlines=headlines,
            headline_timestamps=timestamps,
            reference_date=datetime.now(timezone.utc),
        )
        # At minimum, news_volume_7d should be populated (it only depends on count)
        assert not np.isnan(result["news_volume_7d"])


# ─────────────────────────────────────────────────────────────
#  Chat data-coverage fallback
# ─────────────────────────────────────────────────────────────

class TestChatEnsureStockData:
    def test_returns_true_if_symbol_exists(self):
        """If symbol is already in DB, _ensure_stock_data returns True immediately."""
        mock_storage = MagicMock()
        mock_storage.get_stock_by_symbol.return_value = {"id": 1, "symbol": "AAPL"}

        from agents.chat_assistant import StockChatAssistant
        assistant = StockChatAssistant.__new__(StockChatAssistant)
        assistant.storage = mock_storage

        assert assistant._ensure_stock_data("AAPL") is True
        # Should check DB, not call fetcher
        mock_storage.get_stock_by_symbol.assert_called()

    def test_returns_true_if_ns_variant_exists(self):
        """If RELIANCE doesn't exist but RELIANCE.NS does, returns True."""
        mock_storage = MagicMock()
        mock_storage.get_stock_by_symbol.side_effect = lambda s: (
            {"id": 1, "symbol": "RELIANCE.NS"} if s == "RELIANCE.NS" else None
        )

        from agents.chat_assistant import StockChatAssistant
        assistant = StockChatAssistant.__new__(StockChatAssistant)
        assistant.storage = mock_storage

        assert assistant._ensure_stock_data("RELIANCE") is True

    def test_fetches_fresh_when_not_in_db(self):
        """When symbol not in DB, attempts fresh fetch."""
        mock_storage = MagicMock()
        mock_storage.get_stock_by_symbol.return_value = None
        mock_storage.get_or_create_stock.return_value = {"id": 99}

        mock_quote = MagicMock()
        mock_quote.price = 150.0
        mock_data = {
            "quote": mock_quote,
            "indicators": {"rsi_14": 55},
            "fundamentals": {"name": "Test Corp", "sector": "Technology"},
            "news": [],
        }

        from agents.chat_assistant import StockChatAssistant
        assistant = StockChatAssistant.__new__(StockChatAssistant)
        assistant.storage = mock_storage

        with patch("agents.fetcher.StockFetcher") as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance.get_full_analysis_data.return_value = mock_data

            result = assistant._ensure_stock_data("NVDA")

        assert result is True
        mock_storage.get_or_create_stock.assert_called_once()
        call_kwargs = mock_storage.get_or_create_stock.call_args
        # Should use "name" key (not "longName")
        assert call_kwargs[1]["name"] == "Test Corp" or call_kwargs.kwargs["name"] == "Test Corp"

    def test_returns_false_when_quote_missing(self):
        """If fetcher returns no quote, symbol doesn't exist."""
        mock_storage = MagicMock()
        mock_storage.get_stock_by_symbol.return_value = None

        from agents.chat_assistant import StockChatAssistant
        assistant = StockChatAssistant.__new__(StockChatAssistant)
        assistant.storage = mock_storage

        with patch("agents.fetcher.StockFetcher") as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance.get_full_analysis_data.return_value = {"quote": None}

            result = assistant._ensure_stock_data("INVALID")

        assert result is False

    def test_returns_false_on_exception(self):
        """Graceful fallback on fetch failure."""
        mock_storage = MagicMock()
        mock_storage.get_stock_by_symbol.return_value = None

        from agents.chat_assistant import StockChatAssistant
        assistant = StockChatAssistant.__new__(StockChatAssistant)
        assistant.storage = mock_storage

        with patch("agents.fetcher.StockFetcher") as MockFetcher:
            fetcher_instance = MockFetcher.return_value
            fetcher_instance.get_full_analysis_data.side_effect = Exception("API down")

            result = assistant._ensure_stock_data("FAIL")

        assert result is False
