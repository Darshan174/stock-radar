"""
Lightweight, cost-aware backtest metrics for model predictions.

This is intentionally simple and designed for offline model evaluation where
rows contain a forward return target (`future_return_pct`).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from training.feature_engineering import decode_signal
from training.regime import classify_market_regime
from training.risk import calculate_position_size


SIGNAL_TO_POSITION = {
    "strong_sell": -1.0,
    "sell": -0.5,
    "hold": 0.0,
    "buy": 0.5,
    "strong_buy": 1.0,
}


def _to_float(value: Any) -> float | None:
    try:
        if value in (None, "", "nan"):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_timestamp(value: Any) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value), tz=timezone.utc)
        except (OSError, ValueError):
            return None
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return datetime.fromisoformat(raw)
        except ValueError:
            try:
                return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return None
    return None


def _signal_from_prediction(pred: Any) -> str:
    if isinstance(pred, (int, np.integer)):
        return decode_signal(int(pred))
    if isinstance(pred, (float, np.floating)) and float(pred).is_integer():
        return decode_signal(int(pred))
    return str(pred).strip().lower() or "hold"


def evaluate_predictions(
    rows: Iterable[Mapping[str, Any]],
    predictions: Iterable[Any],
    *,
    horizon_days: int = 5,
    transaction_cost_bps: float = 10.0,
    allow_short: bool = True,
    prediction_confidences: Sequence[float] | None = None,
    use_position_sizing: bool = False,
    risk_factor: float = 1.0,
    target_volatility_pct: float = 2.0,
    min_confidence: float = 0.35,
    # ---------- Slippage / liquidity ----------
    slippage_bps: float = 0.0,
    max_adv_participation: float = 0.01,
    notional_per_unit: float = 100_000.0,
) -> dict[str, float]:
    """
    Evaluate model predictions as a tradable strategy.

    Required row fields:
        - symbol
        - future_return_pct
        - timestamp (optional but recommended)

    Slippage / liquidity parameters:
        slippage_bps: market-impact coefficient (bps).  Realised slippage
            is ``slippage_bps * sqrt(trade_notional / adv_notional)`` per
            trade.  Set to 0 to disable (default).
        max_adv_participation: maximum fraction of average daily volume
            the strategy may consume per trade (default 1 %).  Positions
            exceeding this are scaled down.
        notional_per_unit: dollar notional per unit of position.  Used to
            convert the abstract position size into dollar volume for the
            market-impact model (default $100 000).
    """
    records: list[dict[str, Any]] = []
    for idx, (row, pred) in enumerate(zip(rows, predictions)):
        future_return_pct = _to_float(row.get("future_return_pct"))
        if future_return_pct is None:
            continue

        signal = _signal_from_prediction(pred)
        base_position = SIGNAL_TO_POSITION.get(signal, 0.0)
        if not allow_short:
            base_position = max(0.0, base_position)

        confidence = None
        if prediction_confidences is not None and idx < len(prediction_confidences):
            confidence = _to_float(prediction_confidences[idx])
        if confidence is None:
            confidence = _to_float(row.get("prediction_confidence"))

        regime = None
        if use_position_sizing:
            regime_info = classify_market_regime(
                {
                    "adx": _to_float(row.get("adx")),
                    "atr_pct": _to_float(row.get("atr_pct")),
                    "rsi_14": _to_float(row.get("rsi_14")),
                    "price_vs_sma20_pct": _to_float(row.get("price_vs_sma20_pct")),
                    "price_vs_sma50_pct": _to_float(row.get("price_vs_sma50_pct")),
                    "macd_histogram": _to_float(row.get("macd_histogram")),
                }
            )
            regime = str(regime_info.get("regime", "neutral"))
            sizing = calculate_position_size(
                signal=signal,
                confidence=(confidence if confidence is not None else 0.5),
                volatility_pct=_to_float(row.get("atr_pct")),
                regime=regime,
                risk_factor=risk_factor,
                target_volatility_pct=target_volatility_pct,
                min_confidence=min_confidence,
                allow_short=allow_short,
            )
            position = float(sizing["position_size"])
        else:
            position = base_position

        # --- Liquidity constraint: cap position at max_adv_participation ---
        avg_vol = _to_float(row.get("avg_volume"))
        liquidity_limited = False
        if avg_vol and avg_vol > 0 and notional_per_unit > 0:
            # max trade notional = avg_vol * price * max_adv_participation
            price = _to_float(row.get("current_price")) or _to_float(row.get("price"))
            if price and price > 0:
                adv_notional = avg_vol * price
                max_position = (adv_notional * max_adv_participation) / notional_per_unit
                if abs(position) > max_position and max_position > 0:
                    sign = 1.0 if position >= 0 else -1.0
                    position = sign * max_position
                    liquidity_limited = True

        records.append(
            {
                "index": idx,
                "symbol": str(row.get("symbol", "")).upper() or "UNKNOWN",
                "timestamp": _parse_timestamp(row.get("timestamp")),
                "future_return_pct": future_return_pct,
                "position": position,
                "confidence": confidence if confidence is not None else 0.0,
                "regime": regime or "neutral",
                "avg_volume": avg_vol,
                "current_price": _to_float(row.get("current_price")) or _to_float(row.get("price")),
                "liquidity_limited": liquidity_limited,
            }
        )

    if not records:
        return {
            "num_samples": 0,
            "num_periods": 0,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "annualized_volatility": 0.0,
            "gross_return_total": 0.0,
            "transaction_cost_total": 0.0,
            "slippage_total": 0.0,
            "net_return_total": 0.0,
            "avg_abs_position": 0.0,
            "avg_confidence": 0.0,
            "sizing_enabled": float(1 if use_position_sizing else 0),
            "liquidity_limited_pct": 0.0,
        }

    records.sort(
        key=lambda x: (
            x["symbol"],
            x["timestamp"] if x["timestamp"] is not None else datetime.min.replace(tzinfo=timezone.utc),
            x["index"],
        )
    )

    prev_pos_by_symbol: dict[str, float] = {}
    for rec in records:
        symbol = rec["symbol"]
        prev_pos = prev_pos_by_symbol.get(symbol, 0.0)
        turnover = abs(rec["position"] - prev_pos)
        # Fixed transaction cost (commissions)
        fixed_cost = turnover * (transaction_cost_bps / 10_000.0)

        # Volume-dependent slippage (square-root market impact)
        slip = 0.0
        if slippage_bps > 0 and turnover > 0:
            adv = rec.get("avg_volume")
            price = rec.get("current_price")
            if adv and adv > 0 and price and price > 0:
                # trade_dollar_volume = notional traded
                trade_dollar_vol = turnover * notional_per_unit
                # adv_dollar_volume = average daily dollar volume
                adv_dollar_vol = adv * price
                # participation rate: fraction of daily volume consumed
                participation = trade_dollar_vol / adv_dollar_vol
                impact = (slippage_bps / 10_000.0) * (participation ** 0.5)
                slip = turnover * impact
            else:
                # No volume/price info → apply flat slippage
                slip = turnover * (slippage_bps / 10_000.0)

        cost = fixed_cost + slip
        gross = rec["position"] * (rec["future_return_pct"] / 100.0)
        net = gross - cost

        rec["turnover"] = turnover
        rec["cost"] = cost
        rec["slippage"] = slip
        rec["gross"] = gross
        rec["net"] = net
        prev_pos_by_symbol[symbol] = rec["position"]

    # Aggregate by timestamp if available; else treat rows as sequential periods.
    if all(rec["timestamp"] is not None for rec in records):
        grouped: dict[datetime, list[float]] = {}
        grouped_turnover: dict[datetime, list[float]] = {}
        for rec in records:
            ts = rec["timestamp"]
            grouped.setdefault(ts, []).append(rec["net"])
            grouped_turnover.setdefault(ts, []).append(rec["turnover"])
        ordered_timestamps = sorted(grouped.keys())
        period_returns = np.array([float(np.mean(grouped[ts])) for ts in ordered_timestamps], dtype=np.float64)
        period_turnover = np.array(
            [float(np.mean(grouped_turnover[ts])) for ts in ordered_timestamps], dtype=np.float64
        )
    else:
        period_returns = np.array([rec["net"] for rec in records], dtype=np.float64)
        period_turnover = np.array([rec["turnover"] for rec in records], dtype=np.float64)

    # Basic portfolio statistics
    equity_curve = np.cumprod(1.0 + period_returns)
    total_return = float(equity_curve[-1] - 1.0)
    periods_per_year = max(1.0, 252.0 / max(1, horizon_days))
    years = len(period_returns) / periods_per_year
    cagr = float((equity_curve[-1] ** (1.0 / years) - 1.0)) if years > 0 else total_return

    mean_r = float(np.mean(period_returns))
    std_r = float(np.std(period_returns, ddof=1)) if len(period_returns) > 1 else 0.0
    sharpe = float((mean_r / std_r) * np.sqrt(periods_per_year)) if std_r > 1e-12 else 0.0
    annualized_vol = float(std_r * np.sqrt(periods_per_year)) if std_r > 0 else 0.0

    running_max = np.maximum.accumulate(equity_curve)
    drawdowns = (equity_curve / running_max) - 1.0
    max_drawdown = float(np.min(drawdowns)) if len(drawdowns) else 0.0

    win_rate = float(np.mean(period_returns > 0.0)) if len(period_returns) else 0.0
    turnover = float(np.mean(period_turnover)) if len(period_turnover) else 0.0

    gross_total = float(np.sum([rec["gross"] for rec in records]))
    cost_total = float(np.sum([rec["cost"] for rec in records]))
    slippage_total = float(np.sum([rec["slippage"] for rec in records]))
    net_total = float(np.sum([rec["net"] for rec in records]))
    avg_abs_position = float(np.mean([abs(rec["position"]) for rec in records]))
    avg_confidence = float(np.mean([rec["confidence"] for rec in records]))
    liq_limited = float(np.mean([1.0 if rec["liquidity_limited"] else 0.0 for rec in records]))

    return {
        "num_samples": int(len(records)),
        "num_periods": int(len(period_returns)),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "win_rate": win_rate,
        "turnover": turnover,
        "annualized_volatility": annualized_vol,
        "gross_return_total": gross_total,
        "transaction_cost_total": cost_total,
        "slippage_total": slippage_total,
        "net_return_total": net_total,
        "avg_abs_position": avg_abs_position,
        "avg_confidence": avg_confidence,
        "sizing_enabled": float(1 if use_position_sizing else 0),
        "liquidity_limited_pct": liq_limited,
    }


# ---------------------------------------------------------------------------
#  A/B backtest comparison with cost-aware gates
# ---------------------------------------------------------------------------

DEFAULT_COMPARISON_GATES = {
    "sharpe_no_regress": True,         # candidate.sharpe >= incumbent.sharpe
    "cagr_tolerance": 0.01,            # candidate.cagr >= incumbent.cagr - tolerance
    "max_dd_tolerance": 0.02,          # candidate.max_dd >= incumbent.max_dd - tolerance
    "turnover_max_ratio": 1.5,         # candidate.turnover <= incumbent.turnover * ratio
    "net_vs_gross_min": 0.70,          # candidate.net >= candidate.gross * min (costs < 30%)
}


def compare_backtests(
    incumbent_metrics: dict,
    candidate_metrics: dict,
    gates: dict | None = None,
) -> dict:
    """
    Compare two backtest results. Returns {passed: bool, gate_results: [...]}.

    Default gates:
    - sharpe_ratio: candidate >= incumbent (no regression)
    - cagr: candidate >= incumbent - 0.01
    - max_drawdown: candidate >= incumbent - 0.02 (less negative = better)
    - turnover: candidate <= incumbent * 1.5 (don't overtrade)
    - net_return: candidate.net >= candidate.gross * 0.7 (costs < 30% of gross)
    """
    g = {**DEFAULT_COMPARISON_GATES, **(gates or {})}
    results = []
    all_passed = True

    def _get(d, key):
        v = d.get(key)
        if v is None:
            bt = d.get("backtest_metrics", {})
            v = bt.get(key)
        return float(v) if v is not None else None

    # Gate 1: Sharpe (no regression)
    c_sharpe = _get(candidate_metrics, "sharpe")
    i_sharpe = _get(incumbent_metrics, "sharpe")
    if c_sharpe is not None and i_sharpe is not None:
        passed = c_sharpe >= i_sharpe
        results.append({"gate": "sharpe", "passed": passed,
                        "candidate": c_sharpe, "incumbent": i_sharpe})
        if not passed:
            all_passed = False
    else:
        results.append({"gate": "sharpe", "passed": False,
                        "candidate": c_sharpe, "incumbent": i_sharpe, "note": "missing"})
        all_passed = False

    # Gate 2: CAGR (with tolerance)
    c_cagr = _get(candidate_metrics, "cagr")
    i_cagr = _get(incumbent_metrics, "cagr")
    if c_cagr is not None and i_cagr is not None:
        passed = c_cagr >= i_cagr - g["cagr_tolerance"]
        results.append({"gate": "cagr", "passed": passed,
                        "candidate": c_cagr, "incumbent": i_cagr})
        if not passed:
            all_passed = False
    else:
        results.append({"gate": "cagr", "passed": False,
                        "candidate": c_cagr, "incumbent": i_cagr, "note": "missing"})
        all_passed = False

    # Gate 3: Max drawdown (less negative = better)
    c_dd = _get(candidate_metrics, "max_drawdown")
    i_dd = _get(incumbent_metrics, "max_drawdown")
    if c_dd is not None and i_dd is not None:
        passed = c_dd >= i_dd - g["max_dd_tolerance"]
        results.append({"gate": "max_drawdown", "passed": passed,
                        "candidate": c_dd, "incumbent": i_dd})
        if not passed:
            all_passed = False
    else:
        results.append({"gate": "max_drawdown", "passed": False,
                        "candidate": c_dd, "incumbent": i_dd, "note": "missing"})
        all_passed = False

    # Gate 4: Turnover (don't overtrade)
    c_turn = _get(candidate_metrics, "turnover")
    i_turn = _get(incumbent_metrics, "turnover")
    if c_turn is not None and i_turn is not None and i_turn > 0:
        passed = c_turn <= i_turn * g["turnover_max_ratio"]
        results.append({"gate": "turnover", "passed": passed,
                        "candidate": c_turn, "incumbent": i_turn})
        if not passed:
            all_passed = False
    elif c_turn is not None and (i_turn is None or i_turn == 0):
        # No incumbent turnover to compare — pass by default
        results.append({"gate": "turnover", "passed": True,
                        "candidate": c_turn, "incumbent": i_turn, "note": "no incumbent baseline"})

    # Gate 5: Net vs gross (cost efficiency)
    c_net = _get(candidate_metrics, "net_return_total")
    c_gross = _get(candidate_metrics, "gross_return_total")
    if c_net is not None and c_gross is not None and c_gross > 0:
        ratio = c_net / c_gross
        passed = ratio >= g["net_vs_gross_min"]
        results.append({"gate": "net_vs_gross", "passed": passed,
                        "candidate_ratio": round(ratio, 4),
                        "threshold": g["net_vs_gross_min"]})
        if not passed:
            all_passed = False
    elif c_gross is not None and c_gross <= 0:
        # Gross return <= 0 — can't meaningfully evaluate cost ratio
        results.append({"gate": "net_vs_gross", "passed": True,
                        "note": "gross return non-positive, gate skipped"})

    return {"passed": all_passed, "gate_results": results}
