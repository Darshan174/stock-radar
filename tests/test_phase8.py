"""
Tests for Phase 8: Portfolio + Risk Engine.

Covers: stop-loss/take-profit, per-trade risk, portfolio constraints,
regime router, portfolio backtest, and paper trading.
"""

from __future__ import annotations

import os
import sys
from unittest.mock import patch

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.risk import (
    calculate_stop_take_profit,
    calculate_per_trade_risk,
)
from training.portfolio_constraints import (
    enforce_portfolio_constraints,
    compute_portfolio_risk_summary,
)
from training.regime_router import discover_regime_models, RegimeAwarePredictor
from training.portfolio_backtest import PortfolioBacktester
from training.paper_trading import PaperPortfolio


# ═══════════════════════════════════════════════════════════════
#  TestStopTakeProfit
# ═══════════════════════════════════════════════════════════════

class TestStopTakeProfit:
    def test_buy_signal_stop_below_entry(self):
        result = calculate_stop_take_profit(
            signal="buy", entry_price=100.0, atr=2.0,
        )
        assert result["stop_loss"] is not None
        assert result["take_profit"] is not None
        assert result["stop_loss"] < 100.0
        assert result["take_profit"] > 100.0
        assert result["risk_reward_ratio"] >= 1.5

    def test_sell_signal_reversed(self):
        result = calculate_stop_take_profit(
            signal="sell", entry_price=100.0, atr=2.0,
        )
        assert result["stop_loss"] > 100.0  # stop above for short
        assert result["take_profit"] < 100.0  # target below for short

    def test_hold_returns_none(self):
        result = calculate_stop_take_profit(
            signal="hold", entry_price=100.0, atr=2.0,
        )
        assert result["stop_loss"] is None
        assert result["take_profit"] is None
        assert result["risk_reward_ratio"] is None

    def test_stop_capped_at_max_pct(self):
        # ATR=20 on a $100 stock would be 40% stop without capping
        result = calculate_stop_take_profit(
            signal="buy",
            entry_price=100.0,
            atr=20.0,
            atr_multiplier_stop=2.0,
            max_stop_pct=5.0,
        )
        # Stop should be at most 5% from entry
        stop_distance_pct = abs(100.0 - result["stop_loss"]) / 100.0 * 100
        assert stop_distance_pct <= 5.01  # small float tolerance


# ═══════════════════════════════════════════════════════════════
#  TestPerTradeRisk
# ═══════════════════════════════════════════════════════════════

class TestPerTradeRisk:
    def test_within_limits(self):
        result = calculate_per_trade_risk(
            position_size=0.5, stop_loss_pct=3.0,
            portfolio_value=100_000.0, max_risk_per_trade_pct=2.0,
        )
        # 0.5 * 100k * 3% = 1500 = 1.5% of portfolio < 2%
        assert result["within_limits"] is True
        assert result["risk_pct"] == pytest.approx(1.5, abs=0.01)

    def test_exceeds_limits_scales_down(self):
        result = calculate_per_trade_risk(
            position_size=1.0, stop_loss_pct=3.0,
            portfolio_value=100_000.0, max_risk_per_trade_pct=2.0,
        )
        # 1.0 * 100k * 3% = 3000 = 3% > 2%
        assert result["within_limits"] is False
        assert result["adjusted_position_size"] < 1.0
        # Adjusted should bring risk to exactly 2%
        adj_risk = result["adjusted_position_size"] * 3.0
        assert adj_risk == pytest.approx(2.0, abs=0.01)


# ═══════════════════════════════════════════════════════════════
#  TestPortfolioConstraints
# ═══════════════════════════════════════════════════════════════

class TestPortfolioConstraints:
    def test_single_stock_cap(self):
        positions = [{"symbol": "AAPL", "weight": 0.50, "sector": "Tech"}]
        result = enforce_portfolio_constraints(
            positions, max_single_weight=0.20,
        )
        assert abs(result[0]["weight"]) <= 0.20 + 1e-6
        assert "single_stock_cap" in result[0]["constraint_applied"]

    def test_sector_cap(self):
        positions = [
            {"symbol": "AAPL", "weight": 0.20, "sector": "Tech"},
            {"symbol": "MSFT", "weight": 0.20, "sector": "Tech"},
        ]
        result = enforce_portfolio_constraints(
            positions, max_sector_weight=0.35,
        )
        tech_total = sum(abs(p["weight"]) for p in result if p["sector"] == "Tech")
        assert tech_total <= 0.35 + 1e-6

    def test_total_exposure_cap(self):
        positions = [
            {"symbol": "A", "weight": 0.40, "sector": "S1"},
            {"symbol": "B", "weight": 0.40, "sector": "S2"},
            {"symbol": "C", "weight": 0.40, "sector": "S3"},
        ]
        result = enforce_portfolio_constraints(
            positions, max_total_exposure=1.0, max_single_weight=0.50,
        )
        total = sum(abs(p["weight"]) for p in result)
        assert total <= 1.0 + 1e-6

    def test_risk_summary(self):
        positions = [
            {"symbol": "A", "weight": 0.30, "sector": "Tech"},
            {"symbol": "B", "weight": -0.10, "sector": "Fin"},
        ]
        summary = compute_portfolio_risk_summary(positions)
        assert summary["long_exposure"] == pytest.approx(0.30, abs=1e-4)
        assert summary["short_exposure"] == pytest.approx(0.10, abs=1e-4)
        assert summary["total_exposure"] == pytest.approx(0.40, abs=1e-4)
        assert summary["num_positions"] == 2
        assert "Tech" in summary["sector_concentrations"]
        assert summary["herfindahl_index"] > 0


# ═══════════════════════════════════════════════════════════════
#  TestRegimeRouter
# ═══════════════════════════════════════════════════════════════

class TestRegimeRouter:
    def test_discover_empty_dir(self, tmp_path):
        result = discover_regime_models(str(tmp_path))
        assert result == {}

    def test_discover_finds_regime_model(self, tmp_path):
        # Create a fake regime model file
        (tmp_path / "signal_classifier_v3_trending.joblib").touch()
        result = discover_regime_models(str(tmp_path))
        assert "trending" in result
        assert "v3_trending" in result["trending"]

    def test_fallback_to_general(self, tmp_path):
        # Create a fake general model
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42)),
        ])

        # Train on trivial data
        X = np.random.randn(50, 20)
        y = np.random.randint(0, 5, 50)
        pipeline.fit(X, y)

        model_path = tmp_path / "signal_classifier_v1.joblib"
        joblib.dump(pipeline, model_path)

        rap = RegimeAwarePredictor(
            general_model_path=str(model_path),
            model_dir=str(tmp_path),
        )

        # Predict — should fall back to general (no regime models)
        indicators = {"rsi_14": 50, "adx": 20, "atr_pct": 1.5}
        result = rap.predict(indicators)
        assert "signal" in result
        assert result["regime_model_used"] is False

    def test_routes_to_regime_model(self, tmp_path):
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier

        X = np.random.randn(50, 20)
        y = np.random.randint(0, 5, 50)

        # General model
        pipeline_gen = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42)),
        ])
        pipeline_gen.fit(X, y)
        gen_path = tmp_path / "signal_classifier_v1.joblib"
        joblib.dump(pipeline_gen, gen_path)

        # Trending regime model
        pipeline_trend = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=99)),
        ])
        pipeline_trend.fit(X, y)
        trend_path = tmp_path / "signal_classifier_v1_trending.joblib"
        joblib.dump(pipeline_trend, trend_path)

        rap = RegimeAwarePredictor(
            general_model_path=str(gen_path),
            model_dir=str(tmp_path),
        )

        # Force trending regime
        indicators = {"rsi_14": 60, "adx": 40, "atr_pct": 1.0,
                       "price_vs_sma20_pct": 3.0, "price_vs_sma50_pct": 5.0,
                       "macd_histogram": 2.0}

        with patch("training.regime_router.classify_market_regime") as mock_regime:
            mock_regime.return_value = {"regime": "trending", "confidence": 0.8}
            result = rap.predict(indicators)

        assert result["regime_model_used"] is True
        assert "trending" in result["regime_model_name"]


# ═══════════════════════════════════════════════════════════════
#  TestPortfolioBacktest
# ═══════════════════════════════════════════════════════════════

class TestPortfolioBacktest:
    def test_basic_backtest(self):
        bt = PortfolioBacktester(initial_capital=100_000.0)
        from datetime import datetime

        signals = [
            {"symbol": "AAPL", "signal": "buy", "confidence": 0.8,
             "sector": "Tech", "future_return_pct": 2.0},
            {"symbol": "MSFT", "signal": "buy", "confidence": 0.7,
             "sector": "Tech", "future_return_pct": 1.0},
        ]
        bt.step(datetime(2025, 1, 1), signals)

        report = bt.report()
        assert "sharpe" in report
        assert "equity_curve" in report
        assert report["num_periods"] == 1
        assert len(report["equity_curve"]) == 2  # initial + 1 step

    def test_portfolio_constraints_enforced(self):
        bt = PortfolioBacktester(
            initial_capital=100_000.0,
            max_single_weight=0.10,
            use_position_sizing=False,
        )
        from datetime import datetime

        # Strong buy gives weight=1.0, should be capped
        signals = [
            {"symbol": "AAPL", "signal": "strong_buy", "confidence": 1.0,
             "sector": "Tech", "future_return_pct": 5.0},
        ]
        bt.step(datetime(2025, 1, 1), signals)

        # Position should be capped to max_single_weight
        assert abs(bt.positions.get("AAPL", 0)) <= 0.10 + 1e-6

    def test_turnover_constraint(self):
        bt = PortfolioBacktester(
            initial_capital=100_000.0,
            max_turnover_per_step=0.10,
            use_position_sizing=False,
        )
        from datetime import datetime

        # Step 1: buy
        bt.step(datetime(2025, 1, 1), [
            {"symbol": "AAPL", "signal": "strong_buy", "confidence": 0.9,
             "sector": "Tech", "future_return_pct": 1.0},
        ])

        # Step 2: sell — large turnover should be capped
        bt.step(datetime(2025, 1, 2), [
            {"symbol": "AAPL", "signal": "strong_sell", "confidence": 0.9,
             "sector": "Tech", "future_return_pct": -1.0},
        ])

        report = bt.report()
        avg_turnover = report["avg_turnover_per_step"]
        assert avg_turnover <= 0.10 + 1e-3


# ═══════════════════════════════════════════════════════════════
#  TestPaperTrading
# ═══════════════════════════════════════════════════════════════

class TestPaperTrading:
    def test_record_signal_creates_files(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        pp.record_signal(
            symbol="AAPL", signal="buy", confidence=0.8,
            price=150.0, position_size=0.1,
        )
        assert (tmp_path / "signals.jsonl").exists()
        assert (tmp_path / "positions.json").exists()

    def test_buy_then_sell_closes_position(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        pp.record_signal(
            symbol="AAPL", signal="buy", confidence=0.8, price=100.0,
        )
        assert "AAPL" in pp.get_open_positions()

        pp.record_signal(
            symbol="AAPL", signal="sell", confidence=0.7, price=110.0,
        )
        assert "AAPL" not in pp.get_open_positions()

        trades = pp.get_closed_trades()
        assert len(trades) == 1
        assert trades[0]["pnl_pct"] == pytest.approx(10.0, abs=0.1)

    def test_performance_summary(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))

        # Trade 1: +10%
        pp.record_signal(symbol="AAPL", signal="buy", confidence=0.8, price=100.0)
        pp.record_signal(symbol="AAPL", signal="sell", confidence=0.7, price=110.0)

        # Trade 2: -5%
        pp.record_signal(symbol="MSFT", signal="buy", confidence=0.6, price=200.0)
        pp.record_signal(symbol="MSFT", signal="sell", confidence=0.5, price=190.0)

        summary = pp.get_performance_summary()
        assert summary["total_trades"] == 2
        assert summary["win_rate"] == 0.5
        assert summary["best_trade_pct"] == pytest.approx(10.0, abs=0.1)
        assert summary["worst_trade_pct"] == pytest.approx(-5.0, abs=0.1)

    def test_stop_loss_auto_close(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        pp.record_signal(
            symbol="AAPL", signal="buy", confidence=0.8,
            price=100.0, stop_loss=95.0, take_profit=115.0,
        )
        assert "AAPL" in pp.get_open_positions()

        # Price drops to stop loss
        pp.update_prices({"AAPL": 94.0})
        assert "AAPL" not in pp.get_open_positions()

        trades = pp.get_closed_trades()
        assert len(trades) == 1
        assert trades[0]["pnl_pct"] < 0

    def test_empty_portfolio_summary(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        summary = pp.get_performance_summary()
        assert summary["total_trades"] == 0
        assert summary["open_positions"] == 0


# ═══════════════════════════════════════════════════════════════
#  TestPredictorStopLoss
# ═══════════════════════════════════════════════════════════════

class TestPredictorStopLoss:
    def test_predict_includes_stop_take_profit(self, tmp_path):
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import GradientBoostingClassifier
        from training.predictor import SignalPredictor

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42)),
        ])

        X = np.random.randn(50, 20)
        y = np.random.randint(0, 5, 50)
        pipeline.fit(X, y)

        model_path = tmp_path / "test_model.joblib"
        joblib.dump(pipeline, model_path)

        predictor = SignalPredictor(str(model_path))
        indicators = {
            "rsi_14": 65, "macd": 1.0, "macd_signal": 0.5,
            "sma_20": 100, "sma_50": 98, "adx": 25,
            "atr_pct": 2.0, "atr_14": 3.0,
            "price_vs_sma20_pct": 2.0, "price_vs_sma50_pct": 3.0,
        }
        quote = {"price": 150.0}

        result = predictor.predict(indicators, quote=quote)
        assert "stop_loss" in result
        assert "take_profit" in result
        assert "risk_reward" in result
