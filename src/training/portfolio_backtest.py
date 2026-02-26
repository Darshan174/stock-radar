"""Portfolio-level walk-forward backtester with state."""

from __future__ import annotations

import logging
from collections import defaultdict
from datetime import datetime
from typing import Any

from training.feature_engineering import decode_signal
from training.portfolio_constraints import enforce_portfolio_constraints
from training.risk import calculate_position_size

logger = logging.getLogger(__name__)


class PortfolioBacktester:
    """
    Walk-forward portfolio backtester with state.

    Maintains positions, cash, equity curve. At each step:
    1. Compute target weights from signals (position sizing)
    2. Apply portfolio constraints
    3. Cap turnover per step
    4. Execute rebalance (deduct costs/slippage)
    5. Mark-to-market using future_return_pct
    """

    def __init__(
        self,
        *,
        initial_capital: float = 100_000.0,
        transaction_cost_bps: float = 10.0,
        slippage_bps: float = 5.0,
        max_total_exposure: float = 1.0,
        max_single_weight: float = 0.20,
        max_sector_weight: float = 0.35,
        max_turnover_per_step: float = 0.30,
        use_position_sizing: bool = True,
        risk_factor: float = 1.0,
        target_volatility_pct: float = 2.0,
        min_confidence: float = 0.35,
        spread_bps: float = 3.0,
        latency_penalty_bps: float = 1.0,
        partial_fill_adv_limit: float = 0.02,
    ):
        self.initial_capital = initial_capital
        self.transaction_cost_bps = transaction_cost_bps
        self.slippage_bps = slippage_bps
        self.max_total_exposure = max_total_exposure
        self.max_single_weight = max_single_weight
        self.max_sector_weight = max_sector_weight
        self.max_turnover_per_step = max_turnover_per_step
        self.use_position_sizing = use_position_sizing
        self.risk_factor = risk_factor
        self.target_volatility_pct = target_volatility_pct
        self.min_confidence = min_confidence
        self.spread_bps = spread_bps
        self.latency_penalty_bps = latency_penalty_bps
        self.partial_fill_adv_limit = partial_fill_adv_limit

        # State
        self.cash = initial_capital
        self.positions: dict[str, float] = {}  # symbol -> weight
        self.equity_curve: list[float] = [initial_capital]
        self.step_returns: list[float] = []
        self.total_transaction_cost = 0.0
        self.total_slippage = 0.0
        self.total_spread_cost = 0.0
        self.total_latency_cost = 0.0
        self.partial_fill_count = 0
        self.total_turnover = 0.0
        self.constraint_violations = 0
        self.num_steps = 0

    def step(self, date: datetime, signals: list[dict]) -> dict:
        """
        Process one time step.

        signals = [{symbol, signal, confidence, volatility_pct, sector,
                    future_return_pct, ...}]
        """
        self.num_steps += 1

        # 1. Compute target weights from signals
        target_positions: list[dict[str, Any]] = []
        for sig in signals:
            signal_str = sig.get("signal", "hold")
            confidence = sig.get("confidence", 0.5)

            if self.use_position_sizing:
                pos_info = calculate_position_size(
                    signal=signal_str,
                    confidence=confidence,
                    volatility_pct=sig.get("volatility_pct"),
                    regime=sig.get("regime"),
                    risk_factor=self.risk_factor,
                    target_volatility_pct=self.target_volatility_pct,
                    min_confidence=self.min_confidence,
                )
                weight = pos_info["position_size"]
            else:
                from training.risk import SIGNAL_TO_DIRECTION
                weight = SIGNAL_TO_DIRECTION.get(signal_str, 0.0)

            target_positions.append({
                "symbol": sig.get("symbol", ""),
                "weight": weight,
                "sector": sig.get("sector", "unknown"),
                "future_return_pct": sig.get("future_return_pct", 0.0),
            })

        # 2. Apply portfolio constraints
        pre_constraint_exposure = sum(abs(p["weight"]) for p in target_positions)
        constrained = enforce_portfolio_constraints(
            target_positions,
            max_total_exposure=self.max_total_exposure,
            max_single_weight=self.max_single_weight,
            max_sector_weight=self.max_sector_weight,
        )
        post_constraint_exposure = sum(abs(p["weight"]) for p in constrained)
        if post_constraint_exposure < pre_constraint_exposure - 1e-6:
            self.constraint_violations += 1

        # Build target weight map
        target_weights: dict[str, float] = {}
        future_returns: dict[str, float] = {}
        for p in constrained:
            sym = p["symbol"]
            target_weights[sym] = p["weight"]
            future_returns[sym] = p.get("future_return_pct", 0.0)

        # 3. Cap turnover
        all_symbols = set(list(self.positions.keys()) + list(target_weights.keys()))
        raw_turnover = sum(
            abs(target_weights.get(s, 0.0) - self.positions.get(s, 0.0))
            for s in all_symbols
        )
        if raw_turnover > self.max_turnover_per_step and raw_turnover > 0:
            scale = self.max_turnover_per_step / raw_turnover
            for s in all_symbols:
                old_w = self.positions.get(s, 0.0)
                new_w = target_weights.get(s, 0.0)
                target_weights[s] = old_w + (new_w - old_w) * scale
            turnover = self.max_turnover_per_step
        else:
            turnover = raw_turnover

        self.total_turnover += turnover

        # 4. Deduct costs/slippage + execution realism
        cost_rate = self.transaction_cost_bps / 10_000.0
        slip_rate = self.slippage_bps / 10_000.0

        equity = self.equity_curve[-1]
        cost = turnover * equity * cost_rate
        slippage = turnover * equity * slip_rate
        self.total_transaction_cost += cost
        self.total_slippage += slippage

        # Spread cost (half-spread per unit turnover)
        spread_cost = turnover * equity * (self.spread_bps / 10_000 / 2)
        self.total_spread_cost += spread_cost

        # Latency penalty
        latency_cost = turnover * equity * (self.latency_penalty_bps / 10_000)
        self.total_latency_cost += latency_cost

        # Partial fills: cap order size to partial_fill_adv_limit of ADV
        for sig in signals:
            adv = sig.get("avg_daily_volume")
            if adv is not None and adv > 0:
                sym = sig.get("symbol", "")
                w = abs(target_weights.get(sym, 0.0))
                if w > 0:
                    order_fraction = w  # weight as proxy for order fraction
                    if order_fraction > self.partial_fill_adv_limit:
                        fill_ratio = self.partial_fill_adv_limit / order_fraction
                        old_w = target_weights[sym]
                        target_weights[sym] = old_w * fill_ratio
                        self.partial_fill_count += 1

        # 5. Mark-to-market using future_return_pct
        step_pnl = 0.0
        for sym, w in target_weights.items():
            ret = future_returns.get(sym, 0.0)
            try:
                ret = float(ret) if ret not in (None, "", "nan") else 0.0
            except (TypeError, ValueError):
                ret = 0.0
            step_pnl += w * (ret / 100.0) * equity

        new_equity = equity + step_pnl - cost - slippage - spread_cost - latency_cost
        self.equity_curve.append(new_equity)

        step_return = (new_equity - equity) / equity if equity > 0 else 0.0
        self.step_returns.append(step_return)

        # Update positions
        self.positions = {s: w for s, w in target_weights.items() if abs(w) > 1e-8}

        return {
            "date": str(date),
            "equity": round(new_equity, 2),
            "step_return": round(step_return, 6),
            "turnover": round(turnover, 6),
            "num_positions": len(self.positions),
        }

    def report(self) -> dict:
        """
        Final report: sharpe, cagr, max_drawdown, win_rate, turnover,
        avg_positions_held, avg_total_exposure, equity_curve,
        gross_return_total, net_return_total, transaction_cost_total,
        slippage_total, portfolio_constraint_violations.
        """
        import numpy as np

        returns = np.array(self.step_returns) if self.step_returns else np.array([0.0])
        equity = np.array(self.equity_curve)

        # Sharpe (annualized, assuming ~252/horizon steps per year)
        if len(returns) > 1 and returns.std() > 0:
            sharpe = float(returns.mean() / returns.std() * np.sqrt(min(252, len(returns))))
        else:
            sharpe = 0.0

        # CAGR
        total_return = equity[-1] / equity[0] if equity[0] > 0 else 1.0
        n_periods = max(1, len(returns))
        periods_per_year = min(252, n_periods)
        cagr = float(total_return ** (periods_per_year / n_periods) - 1.0)

        # Max drawdown
        peak = np.maximum.accumulate(equity)
        drawdown = (equity - peak) / np.where(peak > 0, peak, 1.0)
        max_drawdown = float(drawdown.min())

        # Win rate
        wins = (returns > 0).sum()
        win_rate = float(wins / len(returns)) if len(returns) > 0 else 0.0

        gross_return = float(equity[-1] / equity[0] - 1.0) if equity[0] > 0 else 0.0
        net_return = gross_return  # costs already deducted in equity curve

        return {
            "sharpe": round(sharpe, 4),
            "cagr": round(cagr, 6),
            "max_drawdown": round(max_drawdown, 6),
            "win_rate": round(win_rate, 4),
            "turnover": round(self.total_turnover, 4),
            "avg_turnover_per_step": round(
                self.total_turnover / max(1, self.num_steps), 4
            ),
            "num_periods": self.num_steps,
            "gross_return_total": round(gross_return, 6),
            "net_return_total": round(net_return, 6),
            "transaction_cost_total": round(self.total_transaction_cost, 2),
            "slippage_total": round(self.total_slippage, 2),
            "spread_cost_total": round(self.total_spread_cost, 2),
            "latency_cost_total": round(self.total_latency_cost, 2),
            "partial_fill_count": self.partial_fill_count,
            "portfolio_constraint_violations": self.constraint_violations,
            "equity_curve": [round(e, 2) for e in self.equity_curve],
        }


def run_portfolio_backtest(
    rows: list[dict],
    predictions: list,
    *,
    prediction_confidences=None,
    sectors=None,
    horizon_days: int = 5,
    **kwargs,
) -> dict:
    """
    Convenience function matching evaluate_predictions() API.
    Groups rows by timestamp, runs PortfolioBacktester, returns report.
    """
    if not rows or len(rows) == 0:
        return {"error": "no data", "num_periods": 0}

    bt = PortfolioBacktester(**kwargs)

    # Group by timestamp
    groups: dict[str, list[dict]] = defaultdict(list)
    for i, row in enumerate(rows):
        ts = row.get("timestamp", str(i))
        pred_label = predictions[i] if i < len(predictions) else 2  # hold
        signal = decode_signal(int(pred_label)) if isinstance(pred_label, (int, float)) else str(pred_label)

        conf = 0.5
        if prediction_confidences is not None and i < len(prediction_confidences):
            conf = float(prediction_confidences[i])

        sector = row.get("sector", "unknown") or "unknown"
        if sectors is not None and i < len(sectors):
            sector = sectors[i] or sector

        future_ret = row.get("future_return_pct", 0.0)
        try:
            future_ret = float(future_ret) if future_ret not in (None, "", "nan") else 0.0
        except (TypeError, ValueError):
            future_ret = 0.0

        groups[ts].append({
            "symbol": row.get("symbol", f"sym_{i}"),
            "signal": signal,
            "confidence": conf,
            "volatility_pct": row.get("atr_pct"),
            "sector": sector,
            "future_return_pct": future_ret,
        })

    # Process in sorted order
    for ts in sorted(groups.keys()):
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except (ValueError, TypeError):
            dt = datetime.now()
        bt.step(dt, groups[ts])

    return bt.report()
