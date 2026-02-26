"""
Focused tests for:
  1. Rollout gates  (model_registry.promote_if_better)
  2. Feature health  (feature_health.check_feature_health / compute_reference_stats)
  3. Slippage math   (backtesting.evaluate_predictions with slippage_bps)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.model_registry import (
    register_model,
    set_active,
    get_active_model,
    promote_if_better,
)
from training.feature_health import (
    compute_reference_stats,
    check_feature_health,
    log_feature_health,
)
from training.backtesting import evaluate_predictions


# ─────────────────────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────────────────────

def _make_registry(tmp_path):
    """Create an empty registry file and return its path."""
    p = tmp_path / "registry.json"
    p.write_text(json.dumps({"models": [], "active_version": None}))
    return str(p)


# ═════════════════════════════════════════════════════════════
#  1.  Rollout Gates
# ═════════════════════════════════════════════════════════════

class TestRolloutGates:
    """promote_if_better must block promotion when metrics are missing or worse."""

    def test_auto_promote_when_no_incumbent(self, tmp_path):
        """If no active model exists, candidate is auto-promoted."""
        reg = _make_registry(tmp_path)
        register_model(1, "m1.joblib", {"sharpe": 1.0, "cagr": 0.1, "max_drawdown": -0.1}, registry_path=reg)

        result = promote_if_better(1, registry_path=reg)
        assert result["promoted"] is True
        assert "auto-promoted" in result["reason"].lower() or "no incumbent" in result["reason"].lower()
        assert get_active_model(reg)["version"] == 1

    def test_better_candidate_promoted(self, tmp_path):
        """Candidate with strictly better Sharpe, CAGR, MaxDD is promoted."""
        reg = _make_registry(tmp_path)
        register_model(1, "m1.joblib",
                       {"sharpe": 0.8, "cagr": 0.05, "max_drawdown": -0.15},
                       registry_path=reg)
        set_active(1, reg)
        register_model(2, "m2.joblib",
                       {"sharpe": 1.2, "cagr": 0.08, "max_drawdown": -0.10},
                       registry_path=reg)

        result = promote_if_better(2, registry_path=reg)
        assert result["promoted"] is True
        assert get_active_model(reg)["version"] == 2
        for metric in ("sharpe", "cagr", "max_drawdown"):
            assert result["checks"][metric]["passed"] is True

    def test_worse_candidate_blocked(self, tmp_path):
        """Candidate with worse metrics is NOT promoted."""
        reg = _make_registry(tmp_path)
        register_model(1, "m1.joblib",
                       {"sharpe": 1.2, "cagr": 0.08, "max_drawdown": -0.10},
                       registry_path=reg)
        set_active(1, reg)
        register_model(2, "m2.joblib",
                       {"sharpe": 0.5, "cagr": 0.02, "max_drawdown": -0.25},
                       registry_path=reg)

        result = promote_if_better(2, registry_path=reg)
        assert result["promoted"] is False
        # All three should fail
        for metric in ("sharpe", "cagr", "max_drawdown"):
            assert result["checks"][metric]["passed"] is False
        # v1 remains active
        assert get_active_model(reg)["version"] == 1

    def test_missing_candidate_metrics_blocks_promotion(self, tmp_path):
        """If the CANDIDATE is missing metrics, promotion must be BLOCKED."""
        reg = _make_registry(tmp_path)
        register_model(1, "m1.joblib",
                       {"sharpe": 1.0, "cagr": 0.05, "max_drawdown": -0.10},
                       registry_path=reg)
        set_active(1, reg)
        # Candidate has NO metrics
        register_model(2, "m2.joblib", {}, registry_path=reg)

        result = promote_if_better(2, registry_path=reg)
        assert result["promoted"] is False
        for metric in ("sharpe", "cagr", "max_drawdown"):
            assert result["checks"][metric]["passed"] is False
            assert "missing" in result["checks"][metric].get("note", "").lower()
        assert get_active_model(reg)["version"] == 1

    def test_missing_incumbent_metrics_blocks_promotion(self, tmp_path):
        """If the INCUMBENT is missing metrics, promotion must be BLOCKED."""
        reg = _make_registry(tmp_path)
        register_model(1, "m1.joblib", {}, registry_path=reg)
        set_active(1, reg)
        register_model(2, "m2.joblib",
                       {"sharpe": 1.0, "cagr": 0.05, "max_drawdown": -0.10},
                       registry_path=reg)

        result = promote_if_better(2, registry_path=reg)
        assert result["promoted"] is False
        for metric in ("sharpe", "cagr", "max_drawdown"):
            assert result["checks"][metric]["passed"] is False
        assert get_active_model(reg)["version"] == 1

    def test_partial_missing_metrics_blocks(self, tmp_path):
        """Even one missing metric blocks promotion."""
        reg = _make_registry(tmp_path)
        register_model(1, "m1.joblib",
                       {"sharpe": 0.8, "cagr": 0.05, "max_drawdown": -0.15},
                       registry_path=reg)
        set_active(1, reg)
        # Candidate has sharpe & cagr but no max_drawdown
        register_model(2, "m2.joblib",
                       {"sharpe": 1.5, "cagr": 0.20},
                       registry_path=reg)

        result = promote_if_better(2, registry_path=reg)
        assert result["promoted"] is False
        assert result["checks"]["sharpe"]["passed"] is True
        assert result["checks"]["cagr"]["passed"] is True
        assert result["checks"]["max_drawdown"]["passed"] is False
        assert get_active_model(reg)["version"] == 1

    def test_custom_thresholds_raise_bar(self, tmp_path):
        """Custom improvement thresholds can block even a slightly-better model."""
        reg = _make_registry(tmp_path)
        register_model(1, "m1.joblib",
                       {"sharpe": 1.0, "cagr": 0.05, "max_drawdown": -0.15},
                       registry_path=reg)
        set_active(1, reg)
        # Candidate is slightly better
        register_model(2, "m2.joblib",
                       {"sharpe": 1.01, "cagr": 0.051, "max_drawdown": -0.14},
                       registry_path=reg)

        # With high thresholds, this small improvement isn't enough
        result = promote_if_better(
            2, registry_path=reg,
            thresholds={"sharpe_min_improvement": 0.1, "cagr_min_improvement": 0.02, "max_dd_tolerance": 0.0},
        )
        assert result["promoted"] is False

    def test_candidate_not_found(self, tmp_path):
        """Requesting a non-existent version returns promoted=False."""
        reg = _make_registry(tmp_path)
        result = promote_if_better(99, registry_path=reg)
        assert result["promoted"] is False
        assert "not found" in result["reason"].lower()

    def test_nested_backtest_metrics(self, tmp_path):
        """Metrics stored under backtest_metrics sub-dict are found."""
        reg = _make_registry(tmp_path)
        register_model(1, "m1.joblib",
                       {"backtest_metrics": {"sharpe": 0.8, "cagr": 0.05, "max_drawdown": -0.15}},
                       registry_path=reg)
        set_active(1, reg)
        register_model(2, "m2.joblib",
                       {"backtest_metrics": {"sharpe": 1.2, "cagr": 0.08, "max_drawdown": -0.10}},
                       registry_path=reg)

        result = promote_if_better(2, registry_path=reg)
        assert result["promoted"] is True


# ═════════════════════════════════════════════════════════════
#  2.  Feature Health
# ═════════════════════════════════════════════════════════════

class TestFeatureHealth:
    """Tests for compute_reference_stats and check_feature_health."""

    NAMES = [f"feat_{i}" for i in range(5)]

    def test_reference_stats_shape(self):
        """Reference stats contain correct keys for each feature."""
        X = np.random.randn(100, 5)
        stats = compute_reference_stats(X, self.NAMES)
        assert len(stats) == 5
        for name in self.NAMES:
            assert set(stats[name].keys()) == {"mean", "std", "nan_rate", "p5", "p95", "count"}
            assert stats[name]["nan_rate"] == 0.0
            assert stats[name]["count"] == 100

    def test_reference_stats_with_nans(self):
        """NaN rate is correctly computed."""
        X = np.random.randn(100, 5)
        X[:40, 0] = np.nan  # 40% NaN in feat_0
        stats = compute_reference_stats(X, self.NAMES)
        assert abs(stats["feat_0"]["nan_rate"] - 0.4) < 1e-6
        assert stats["feat_0"]["count"] == 60

    def test_healthy_features(self):
        """Clean features with no shift produce healthy status."""
        X = np.random.randn(200, 5)
        ref = compute_reference_stats(X, self.NAMES)
        # Same distribution → should be healthy
        X_live = np.random.randn(50, 5)
        report = check_feature_health(X_live, self.NAMES, ref)
        assert report["overall_status"] == "healthy"
        assert report["summary"]["critical"] == 0
        assert report["summary"]["warn"] == 0

    def test_high_nan_rate_triggers_warn(self):
        """Feature with >25% NaN should trigger warn."""
        X_ref = np.random.randn(100, 5)
        ref = compute_reference_stats(X_ref, self.NAMES)

        X_live = np.random.randn(100, 5)
        X_live[:30, 2] = np.nan  # 30% NaN in feat_2
        report = check_feature_health(X_live, self.NAMES, ref, nan_rate_warn=0.25)
        assert report["features"]["feat_2"]["status"] in ("warn", "critical")
        assert report["overall_status"] in ("warn", "critical")

    def test_very_high_nan_rate_triggers_critical(self):
        """Feature with >50% NaN should trigger critical."""
        X_ref = np.random.randn(100, 5)
        ref = compute_reference_stats(X_ref, self.NAMES)

        X_live = np.random.randn(100, 5)
        X_live[:60, 0] = np.nan  # 60% NaN
        report = check_feature_health(X_live, self.NAMES, ref)
        assert report["features"]["feat_0"]["status"] == "critical"
        assert report["overall_status"] == "critical"

    def test_drift_detection(self):
        """Large mean shift triggers drift warning."""
        X_ref = np.random.randn(500, 5)  # mean ≈ 0, std ≈ 1
        ref = compute_reference_stats(X_ref, self.NAMES)

        # Shift feat_1 mean by +5σ
        X_live = np.random.randn(100, 5) + np.array([0, 5, 0, 0, 0])
        report = check_feature_health(X_live, self.NAMES, ref, drift_sigma=2.0)
        assert report["features"]["feat_1"]["drift_status"] in ("warn", "critical")
        assert report["features"]["feat_1"]["drift_z"] > 2.0

    def test_phase5_only_flag(self):
        """phase5_only=True should only check Phase-5 feature names."""
        from training.cross_sectional import ALL_NEW_FEATURE_NAMES
        from training.feature_engineering import FEATURE_NAMES

        X = np.random.randn(50, len(FEATURE_NAMES))
        ref = compute_reference_stats(X, list(FEATURE_NAMES))
        report = check_feature_health(X, list(FEATURE_NAMES), ref, phase5_only=True)
        # Should only contain Phase-5 features
        for name in report["features"]:
            assert name in ALL_NEW_FEATURE_NAMES

    def test_single_vector_input(self):
        """1-D input (single sample) should work without error."""
        X_ref = np.random.randn(100, 5)
        ref = compute_reference_stats(X_ref, self.NAMES)
        x_single = np.random.randn(5)  # 1-D
        report = check_feature_health(x_single, self.NAMES, ref)
        assert report["overall_status"] in ("healthy", "warn", "critical")

    def test_log_feature_health_no_error(self):
        """log_feature_health() runs without raising on all statuses."""
        for status in ("healthy", "warn", "critical"):
            report = {
                "overall_status": status,
                "summary": {"healthy": 3, "warn": 1, "critical": 1},
                "features": {
                    "feat_0": {"nan_rate": 0.0, "status": "healthy"},
                    "feat_1": {"nan_rate": 0.30, "status": "warn", "drift_z": 2.5},
                    "feat_2": {"nan_rate": 0.60, "status": "critical", "drift_z": 5.0},
                },
            }
            log_feature_health(report)  # should not raise


# ═════════════════════════════════════════════════════════════
#  3.  Slippage & Liquidity
# ═════════════════════════════════════════════════════════════

class TestSlippageAndLiquidity:
    """Tests for the slippage and liquidity constraint logic in evaluate_predictions."""

    def _make_rows(self, n=10, *, price=100.0, avg_volume=1_000_000, future_ret=1.0):
        """Create simple test rows."""
        return [
            {
                "symbol": "TEST",
                "timestamp": f"2025-01-{(i + 1):02d}T00:00:00+00:00",
                "future_return_pct": future_ret,
                "current_price": price,
                "avg_volume": avg_volume,
                "atr_pct": 2.0,
                "rsi_14": 50.0,
            }
            for i in range(n)
        ]

    def test_zero_slippage_matches_baseline(self):
        """With slippage_bps=0, results match the no-slippage baseline."""
        rows = self._make_rows()
        preds = [1] * len(rows)  # buy signal

        r_base = evaluate_predictions(rows, preds, transaction_cost_bps=10)
        r_slip = evaluate_predictions(rows, preds, transaction_cost_bps=10, slippage_bps=0)

        assert abs(r_base["slippage_total"] - 0.0) < 1e-12
        assert abs(r_slip["slippage_total"] - 0.0) < 1e-12
        assert abs(r_base["net_return_total"] - r_slip["net_return_total"]) < 1e-12

    def test_positive_slippage_increases_cost(self):
        """Non-zero slippage should increase total cost, reducing net return."""
        rows = self._make_rows()
        preds = [1] * len(rows)

        r_no_slip = evaluate_predictions(rows, preds, transaction_cost_bps=10, slippage_bps=0)
        r_with_slip = evaluate_predictions(rows, preds, transaction_cost_bps=10, slippage_bps=20)

        assert r_with_slip["slippage_total"] > 0
        assert r_with_slip["net_return_total"] < r_no_slip["net_return_total"]

    def test_slippage_scales_with_participation(self):
        """Higher participation (smaller ADV) → higher slippage."""
        rows_big_adv = self._make_rows(avg_volume=10_000_000)
        rows_small_adv = self._make_rows(avg_volume=100_000)
        preds = [1] * 10

        r_big = evaluate_predictions(rows_big_adv, preds, slippage_bps=20, transaction_cost_bps=0)
        r_small = evaluate_predictions(rows_small_adv, preds, slippage_bps=20, transaction_cost_bps=0)

        assert r_small["slippage_total"] > r_big["slippage_total"]

    def test_slippage_is_material_for_realistic_volumes(self):
        """
        Slippage should be meaningfully non-zero even for liquid names.
        This is the regression test for the under-scaled participation bug.
        A $100K position in a stock with $100M ADV (0.1% participation) and
        20bps slippage_bps should produce visible slippage, not near-zero.
        """
        rows = self._make_rows(n=1, price=100.0, avg_volume=1_000_000)
        preds = [1]  # buy

        result = evaluate_predictions(
            rows, preds,
            transaction_cost_bps=0,
            slippage_bps=20,
            notional_per_unit=100_000,
        )
        # participation = 100_000 / (1_000_000 * 100) = 0.001
        # impact = 20/10000 * sqrt(0.001) ≈ 0.002 * 0.0316 ≈ 6.3e-5
        # slippage = 1.0 * 6.3e-5 ≈ 6.3e-5  (non-negligible)
        assert result["slippage_total"] > 1e-6, (
            f"Slippage is too small ({result['slippage_total']:.2e}); "
            "participation math may be under-scaled"
        )

    def test_liquidity_cap_limits_position(self):
        """Position should be capped when it exceeds max_adv_participation."""
        # Very illiquid: 1000 shares avg volume at $100 → $100K ADV
        # 1% of ADV = $1000.   At $100K notional_per_unit, max_position = 0.01
        rows = self._make_rows(n=1, price=100.0, avg_volume=1000)
        preds = [1]

        result = evaluate_predictions(
            rows, preds,
            transaction_cost_bps=0,
            max_adv_participation=0.01,
            notional_per_unit=100_000,
        )
        assert result["liquidity_limited_pct"] > 0
        # Position should be tiny due to illiquidity
        assert result["avg_abs_position"] < 0.02

    def test_no_liquidity_cap_when_volume_large(self):
        """Highly liquid names should not be constrained."""
        rows = self._make_rows(n=1, price=100.0, avg_volume=50_000_000)
        preds = [1]

        result = evaluate_predictions(
            rows, preds,
            transaction_cost_bps=0,
            max_adv_participation=0.01,
            notional_per_unit=100_000,
        )
        assert result["liquidity_limited_pct"] == 0.0

    def test_flat_slippage_fallback_no_volume(self):
        """When avg_volume is missing, slippage falls back to flat bps."""
        rows = [
            {
                "symbol": "TEST",
                "timestamp": "2025-01-01T00:00:00+00:00",
                "future_return_pct": 1.0,
                "current_price": 100.0,
                # no avg_volume
            }
        ]
        preds = [1]

        result = evaluate_predictions(rows, preds, transaction_cost_bps=0, slippage_bps=20)
        # Should still have non-zero slippage (flat fallback)
        assert result["slippage_total"] > 0
