"""
Tests for Phase 9: Live Validation + Execution Hardening.

Covers: calibration metrics, rolling performance, paper-trading gates,
kill switches, execution realism, ops dashboard, calibrated predictor.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.calibration import (
    compute_brier_score,
    compute_calibration_metrics,
    compute_ece,
)
from training.kill_switches import (
    check_abnormal_slippage,
    check_all_kill_switches,
    check_max_daily_loss,
    check_stale_data_feed,
)
from training.paper_trading import PaperPortfolio, check_paper_trading_gates
from training.portfolio_backtest import PortfolioBacktester


# ═══════════════════════════════════════════════════════════════
#  TestCalibration
# ═══════════════════════════════════════════════════════════════


class TestCalibration:
    def test_perfect_ece_near_zero(self):
        """Perfect predictions should have ECE near 0."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        # Perfect one-hot probabilities
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        ece = compute_ece(y_true, y_prob)
        assert ece < 0.01, f"Perfect predictions should have ECE near 0, got {ece}"

    def test_overconfident_ece_high(self):
        """Overconfident wrong predictions should have high ECE."""
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # All predict class 1 with very high confidence, but true is class 0
        y_prob = np.zeros((10, 3))
        y_prob[:, 1] = 0.95
        y_prob[:, 0] = 0.025
        y_prob[:, 2] = 0.025
        ece = compute_ece(y_true, y_prob)
        assert ece > 0.5, f"Overconfident wrong predictions should have high ECE, got {ece}"

    def test_brier_perfect_zero(self):
        """Perfect predictions should have Brier score of 0."""
        y_true = np.array([0, 1, 2])
        y_prob = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ])
        brier = compute_brier_score(y_true, y_prob)
        assert brier == pytest.approx(0.0, abs=1e-6)

    def test_brier_random_positive(self):
        """Random predictions should have Brier > 0."""
        y_true = np.array([0, 1, 2, 0, 1])
        rng = np.random.RandomState(42)
        y_prob = rng.dirichlet([1, 1, 1], size=5)
        brier = compute_brier_score(y_true, y_prob)
        assert brier > 0.0

    def test_compute_calibration_metrics_all_fields(self):
        """compute_calibration_metrics returns all expected keys."""
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([
            [0.7, 0.3],
            [0.4, 0.6],
            [0.8, 0.2],
            [0.3, 0.7],
        ])
        metrics = compute_calibration_metrics(y_true, y_prob)
        assert "ece" in metrics
        assert "brier_score" in metrics
        assert "mean_confidence" in metrics
        assert "mean_accuracy" in metrics
        assert "n_bins" in metrics
        assert metrics["n_bins"] == 10


# ═══════════════════════════════════════════════════════════════
#  TestRollingPerformance
# ═══════════════════════════════════════════════════════════════


class TestRollingPerformance:
    def _make_pp(self, tmpdir):
        return PaperPortfolio(paper_dir=str(tmpdir), initial_capital=100_000)

    def test_empty_returns_zeros(self, tmp_path):
        pp = self._make_pp(tmp_path)
        rolling = pp.get_rolling_performance()
        assert rolling["total_trades"] == 0
        assert rolling["sharpe"] == 0.0
        assert rolling["max_drawdown"] == 0.0

    def test_last_n_trades_window(self, tmp_path):
        pp = self._make_pp(tmp_path)
        now = datetime.now(timezone.utc)
        # Write 20 closed trades
        for i in range(20):
            trade = {
                "symbol": f"SYM{i}",
                "direction": "long",
                "entry_price": 100.0,
                "exit_price": 100.0 + (i % 5),
                "exit_time": (now - timedelta(hours=20 - i)).isoformat(),
                "pnl_pct": float(i % 5),
                "position_size": 0.1,
            }
            pp._append_jsonl(pp.closed_path, trade)

        rolling = pp.get_rolling_performance(last_n_trades=10)
        assert rolling["window_trades"] == 10
        assert rolling["total_trades"] == 10

    def test_last_n_days_window(self, tmp_path):
        pp = self._make_pp(tmp_path)
        now = datetime.now(timezone.utc)
        # Write trades: 5 from today, 5 from 10 days ago
        for i in range(5):
            pp._append_jsonl(pp.closed_path, {
                "symbol": f"TODAY{i}", "pnl_pct": 1.0,
                "exit_time": now.isoformat(), "position_size": 0.1,
            })
        for i in range(5):
            pp._append_jsonl(pp.closed_path, {
                "symbol": f"OLD{i}", "pnl_pct": -1.0,
                "exit_time": (now - timedelta(days=10)).isoformat(), "position_size": 0.1,
            })

        rolling = pp.get_rolling_performance(last_n_days=3)
        assert rolling["window_trades"] == 5  # Only today's trades

    def test_sharpe_and_dd_computation(self, tmp_path):
        pp = self._make_pp(tmp_path)
        now = datetime.now(timezone.utc)
        # Win, win, big loss
        pnls = [5.0, 3.0, -10.0, 2.0, 1.0]
        for i, pnl in enumerate(pnls):
            pp._append_jsonl(pp.closed_path, {
                "symbol": f"S{i}", "pnl_pct": pnl,
                "exit_time": (now - timedelta(hours=len(pnls) - i)).isoformat(),
                "position_size": 0.1,
            })

        rolling = pp.get_rolling_performance()
        assert rolling["total_trades"] == 5
        # max_drawdown should be negative (from the -10 loss)
        assert rolling["max_drawdown"] < 0
        # Sharpe should be computed (non-zero std)
        assert isinstance(rolling["sharpe"], float)


# ═══════════════════════════════════════════════════════════════
#  TestPaperTradingGates
# ═══════════════════════════════════════════════════════════════


class TestPaperTradingGates:
    def _seed_trades(self, paper_dir, n=20, pnl=2.0):
        pp = PaperPortfolio(paper_dir=str(paper_dir))
        now = datetime.now(timezone.utc)
        for i in range(n):
            pp._append_jsonl(pp.closed_path, {
                "symbol": f"S{i}", "pnl_pct": pnl,
                "exit_time": (now - timedelta(hours=n - i)).isoformat(),
                "position_size": 0.1,
            })

    def test_all_pass(self, tmp_path):
        self._seed_trades(tmp_path, n=20, pnl=2.0)
        result = check_paper_trading_gates(
            paper_dir=str(tmp_path), min_trades=10,
            sharpe_min=-999, max_drawdown_floor=-999, win_rate_min=0.0,
            turnover_max=999,
        )
        assert result["passed"] is True

    def test_insufficient_trades(self, tmp_path):
        self._seed_trades(tmp_path, n=3)
        result = check_paper_trading_gates(
            paper_dir=str(tmp_path), min_trades=10,
        )
        assert result["passed"] is False
        assert "Insufficient" in result["reason"]

    def test_bad_sharpe(self, tmp_path):
        # All losses → negative Sharpe
        self._seed_trades(tmp_path, n=20, pnl=-5.0)
        result = check_paper_trading_gates(
            paper_dir=str(tmp_path), min_trades=10,
            sharpe_min=0.5,  # strict threshold
        )
        assert result["passed"] is False
        assert "sharpe" in str(result.get("checks", {}).keys()) or "sharpe" in result["reason"]

    def test_bad_drawdown(self, tmp_path):
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        now = datetime.now(timezone.utc)
        # Create trades with large cumulative loss
        for i in range(20):
            pp._append_jsonl(pp.closed_path, {
                "symbol": f"S{i}", "pnl_pct": -5.0,
                "exit_time": (now - timedelta(hours=20 - i)).isoformat(),
                "position_size": 0.1,
            })

        result = check_paper_trading_gates(
            paper_dir=str(tmp_path), min_trades=10,
            max_drawdown_floor=-10.0,  # Strict floor
        )
        assert result["passed"] is False


# ═══════════════════════════════════════════════════════════════
#  TestKillSwitches
# ═══════════════════════════════════════════════════════════════


class TestKillSwitches:
    def test_daily_loss_ok(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades = [
            {"exit_time": f"{today}T10:00:00+00:00", "pnl_pct": -1.0},
            {"exit_time": f"{today}T11:00:00+00:00", "pnl_pct": -1.0},
        ]
        result = check_max_daily_loss(trades, max_daily_loss_pct=5.0)
        assert result["triggered"] is False
        assert result["daily_loss_pct"] == pytest.approx(-2.0)

    def test_daily_loss_triggered(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades = [
            {"exit_time": f"{today}T10:00:00+00:00", "pnl_pct": -3.0},
            {"exit_time": f"{today}T11:00:00+00:00", "pnl_pct": -3.0},
        ]
        result = check_max_daily_loss(trades, max_daily_loss_pct=5.0)
        assert result["triggered"] is True

    def test_fresh_data_ok(self):
        result = check_stale_data_feed({"age_ms": 5000}, max_stale_ms=60_000)
        assert result["triggered"] is False

    def test_stale_data_triggered(self):
        result = check_stale_data_feed({"age_ms": 120_000}, max_stale_ms=60_000)
        assert result["triggered"] is True

    def test_no_data_triggers(self):
        result = check_stale_data_feed(None, max_stale_ms=60_000)
        assert result["triggered"] is True

    def test_combined_check(self):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        trades = [
            {"exit_time": f"{today}T10:00:00+00:00", "pnl_pct": -6.0},
        ]
        result = check_all_kill_switches(
            closed_trades=trades,
            latest_realtime_data={"age_ms": 5000},
            max_daily_loss_pct=5.0,
            max_stale_ms=60_000,
        )
        assert result["halted"] is True
        assert len(result["reasons"]) >= 1
        assert result["checks"]["daily_loss"]["triggered"] is True
        assert result["checks"]["stale_data"]["triggered"] is False


# ═══════════════════════════════════════════════════════════════
#  TestAbnormalSlippage
# ═══════════════════════════════════════════════════════════════


class TestAbnormalSlippage:
    def test_no_slippage(self):
        """Trades exiting exactly at SL or TP have zero slippage."""
        trades = [
            {"entry_price": 100, "exit_price": 95, "stop_loss": 95, "take_profit": 110},
            {"entry_price": 100, "exit_price": 110, "stop_loss": 95, "take_profit": 110},
        ]
        result = check_abnormal_slippage(trades, slippage_threshold_pct=2.0)
        assert result["triggered"] is False
        assert result["avg_slippage_pct"] == pytest.approx(0.0)

    def test_high_slippage(self):
        """Trades exiting far from SL/TP should trigger."""
        trades = [
            {"entry_price": 100, "exit_price": 85, "stop_loss": 95, "take_profit": 110},
        ]
        result = check_abnormal_slippage(trades, slippage_threshold_pct=2.0)
        assert result["triggered"] is True
        assert result["avg_slippage_pct"] > 2.0


# ═══════════════════════════════════════════════════════════════
#  TestExecutionRealism
# ═══════════════════════════════════════════════════════════════


class TestExecutionRealism:
    def _run_backtest(self, **kwargs):
        bt = PortfolioBacktester(initial_capital=100_000, **kwargs)
        signals = [
            {"symbol": "AAPL", "signal": "buy", "confidence": 0.8,
             "volatility_pct": 1.5, "sector": "tech", "future_return_pct": 5.0},
        ]
        bt.step(datetime(2024, 1, 1), signals)
        return bt.report()

    def test_spread_reduces_equity(self):
        """Higher spread should result in lower final equity."""
        report_low = self._run_backtest(spread_bps=0.0, latency_penalty_bps=0.0)
        report_high = self._run_backtest(spread_bps=50.0, latency_penalty_bps=0.0)
        assert report_high["equity_curve"][-1] < report_low["equity_curve"][-1]
        assert report_high["spread_cost_total"] > report_low["spread_cost_total"]

    def test_latency_reduces_equity(self):
        """Higher latency penalty should result in lower final equity."""
        report_low = self._run_backtest(spread_bps=0.0, latency_penalty_bps=0.0)
        report_high = self._run_backtest(spread_bps=0.0, latency_penalty_bps=50.0)
        assert report_high["equity_curve"][-1] < report_low["equity_curve"][-1]
        assert report_high["latency_cost_total"] > report_low["latency_cost_total"]

    def test_partial_fill_caps_position(self):
        """Partial fill should trigger when order exceeds ADV limit."""
        bt = PortfolioBacktester(
            initial_capital=100_000,
            partial_fill_adv_limit=0.01,  # 1% ADV limit
        )
        signals = [
            {"symbol": "SMALL", "signal": "buy", "confidence": 0.8,
             "volatility_pct": 1.5, "sector": "tech", "future_return_pct": 5.0,
             "avg_daily_volume": 100},  # Very small ADV
        ]
        bt.step(datetime(2024, 1, 1), signals)
        report = bt.report()
        assert report["partial_fill_count"] >= 1


# ═══════════════════════════════════════════════════════════════
#  TestOpsDashboard
# ═══════════════════════════════════════════════════════════════


class TestOpsDashboard:
    def test_runs_without_error(self, tmp_path, capsys):
        """Dashboard should print without crashing on populated data."""
        pp = PaperPortfolio(paper_dir=str(tmp_path))
        now = datetime.now(timezone.utc)
        # Seed some data
        pp.record_signal(
            symbol="AAPL", signal="buy", confidence=0.8,
            price=150.0, regime="trending", position_size=0.1,
        )
        pp._append_jsonl(pp.closed_path, {
            "symbol": "MSFT", "direction": "long",
            "entry_price": 300.0, "exit_price": 310.0,
            "exit_time": now.isoformat(), "pnl_pct": 3.33,
            "position_size": 0.1,
        })

        settings = MagicMock()
        settings.paper_trading_dir = str(tmp_path)
        settings.paper_trading_capital = 100_000.0
        settings.kill_switch_max_daily_loss_pct = 5.0
        settings.kill_switch_max_stale_ms = 60_000
        settings.kill_switch_slippage_threshold_pct = 2.0

        # Import and call
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from main import _print_paper_dashboard
        _print_paper_dashboard(pp, settings)

        captured = capsys.readouterr()
        assert "PAPER TRADING DASHBOARD" in captured.out
        assert "PnL Summary" in captured.out

    def test_empty_portfolio_no_crash(self, tmp_path, capsys):
        """Dashboard should not crash with empty portfolio."""
        pp = PaperPortfolio(paper_dir=str(tmp_path))

        settings = MagicMock()
        settings.paper_trading_dir = str(tmp_path)
        settings.paper_trading_capital = 100_000.0
        settings.kill_switch_max_daily_loss_pct = 5.0
        settings.kill_switch_max_stale_ms = 60_000
        settings.kill_switch_slippage_threshold_pct = 2.0

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from main import _print_paper_dashboard
        _print_paper_dashboard(pp, settings)

        captured = capsys.readouterr()
        assert "PAPER TRADING DASHBOARD" in captured.out


# ═══════════════════════════════════════════════════════════════
#  TestCalibratedPredictor
# ═══════════════════════════════════════════════════════════════


class TestCalibratedPredictor:
    def test_predictor_uses_calibrated_pipeline(self, tmp_path):
        """Predictor should use calibrated pipeline when available."""
        import joblib
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

        # Create a minimal mock model that works with 20 features
        from sklearn.ensemble import GradientBoostingClassifier
        rng = np.random.RandomState(42)
        X = rng.randn(50, 20)
        y = rng.randint(0, 5, size=50)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=5, random_state=42)),
        ])
        pipeline.fit(X, y)

        # Save as both raw and calibrated
        model_path = tmp_path / "model.joblib"
        cal_path = tmp_path / "model_calibrated.joblib"
        meta_path = tmp_path / "model_meta.json"

        joblib.dump(pipeline, model_path)
        joblib.dump(pipeline, cal_path)  # Same for simplicity

        meta = {
            "version": 99,
            "feature_count": 20,
            "calibrated_model_path": str(cal_path),
        }
        meta_path.write_text(json.dumps(meta))

        from training.predictor import SignalPredictor
        predictor = SignalPredictor(
            model_path=str(model_path),
            meta_path=str(meta_path),
        )

        assert predictor.calibrated_pipeline is not None

        # Run prediction with minimal indicators
        indicators = {
            "rsi_14": 50.0, "macd_line": 0.1, "macd_signal": 0.05,
            "macd_histogram": 0.05, "sma_20": 100.0, "sma_50": 99.0,
            "ema_12": 100.5, "bb_percent": 0.5,
            "atr_14": 2.0, "atr_pct": 2.0, "obv_slope": 0.01,
            "vwap_ratio": 1.0, "pe_ratio": 15.0, "pb_ratio": 3.0,
            "dividend_yield": 2.0, "market_cap_log": 25.0,
            "roe": 0.15, "debt_to_equity": 0.5, "current_ratio": 1.5,
            "revenue_growth": 0.1,
        }
        quote = {"price": 100.0}

        result = predictor.predict(indicators, quote=quote)
        assert result["calibrated"] is True
        assert "signal" in result
        assert "confidence" in result
