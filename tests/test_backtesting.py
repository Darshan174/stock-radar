"""Unit tests for cost-aware backtesting metrics."""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.backtesting import evaluate_predictions


def test_backtesting_metrics_keys_present():
    rows = [
        {"symbol": "AAA", "timestamp": "2024-01-01", "future_return_pct": "2.0"},
        {"symbol": "AAA", "timestamp": "2024-01-02", "future_return_pct": "-1.0"},
        {"symbol": "AAA", "timestamp": "2024-01-03", "future_return_pct": "1.5"},
    ]
    preds = [4, 0, 3]  # strong_buy, strong_sell, buy

    metrics = evaluate_predictions(rows, preds, horizon_days=1, transaction_cost_bps=10.0)
    assert metrics["num_samples"] == 3
    assert metrics["num_periods"] == 3
    assert "sharpe" in metrics
    assert "max_drawdown" in metrics
    assert "cagr" in metrics
    assert "turnover" in metrics
    assert "transaction_cost_total" in metrics
    assert "avg_abs_position" in metrics


def test_transaction_cost_reduces_net():
    rows = [
        {"symbol": "AAA", "timestamp": "2024-01-01", "future_return_pct": "1.0"},
        {"symbol": "AAA", "timestamp": "2024-01-02", "future_return_pct": "1.0"},
        {"symbol": "AAA", "timestamp": "2024-01-03", "future_return_pct": "1.0"},
    ]
    preds = [4, 0, 4]  # force high turnover

    no_cost = evaluate_predictions(rows, preds, horizon_days=1, transaction_cost_bps=0.0)
    with_cost = evaluate_predictions(rows, preds, horizon_days=1, transaction_cost_bps=25.0)

    assert with_cost["transaction_cost_total"] > 0
    assert with_cost["net_return_total"] < no_cost["net_return_total"]


def test_allow_short_flag_changes_exposure():
    rows = [
        {"symbol": "AAA", "timestamp": "2024-01-01", "future_return_pct": "-2.0"},
        {"symbol": "AAA", "timestamp": "2024-01-02", "future_return_pct": "-2.0"},
    ]
    preds = [0, 0]  # strong_sell

    short_enabled = evaluate_predictions(rows, preds, horizon_days=1, allow_short=True)
    short_disabled = evaluate_predictions(rows, preds, horizon_days=1, allow_short=False)

    # If shorts are disabled, we should not profit from negative returns via sell signals.
    assert short_enabled["net_return_total"] > short_disabled["net_return_total"]


def test_position_sizing_uses_prediction_confidence():
    rows = [
        {
            "symbol": "AAA",
            "timestamp": "2024-01-01",
            "future_return_pct": "1.5",
            "atr_pct": "1.2",
            "adx": "30",
            "rsi_14": "58",
            "price_vs_sma20_pct": "2.0",
            "price_vs_sma50_pct": "3.0",
            "macd_histogram": "0.5",
        },
        {
            "symbol": "AAA",
            "timestamp": "2024-01-02",
            "future_return_pct": "1.5",
            "atr_pct": "1.2",
            "adx": "30",
            "rsi_14": "58",
            "price_vs_sma20_pct": "2.0",
            "price_vs_sma50_pct": "3.0",
            "macd_histogram": "0.5",
        },
    ]
    preds = [4, 4]  # strong_buy, strong_buy

    low_conf = evaluate_predictions(
        rows,
        preds,
        horizon_days=1,
        transaction_cost_bps=0.0,
        use_position_sizing=True,
        prediction_confidences=[0.40, 0.40],
    )
    high_conf = evaluate_predictions(
        rows,
        preds,
        horizon_days=1,
        transaction_cost_bps=0.0,
        use_position_sizing=True,
        prediction_confidences=[0.90, 0.90],
    )

    assert high_conf["avg_abs_position"] > low_conf["avg_abs_position"]
    assert high_conf["net_return_total"] > low_conf["net_return_total"]
