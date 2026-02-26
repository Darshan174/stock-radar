"""Paper trading portfolio with JSON-file persistence."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class PaperPortfolio:
    """
    JSON-file-backed paper trading portfolio.

    Files in {paper_dir}/:
        signals.jsonl        - append-only signal log
        positions.json       - current open positions
        closed_trades.jsonl  - completed trades with P&L
    """

    def __init__(
        self,
        paper_dir: str = "data/paper_trading",
        initial_capital: float = 100_000.0,
    ):
        self.paper_dir = Path(paper_dir)
        self.paper_dir.mkdir(parents=True, exist_ok=True)
        self.initial_capital = initial_capital

        self.signals_path = self.paper_dir / "signals.jsonl"
        self.positions_path = self.paper_dir / "positions.json"
        self.closed_path = self.paper_dir / "closed_trades.jsonl"

    def _load_positions(self) -> dict[str, dict]:
        if not self.positions_path.exists():
            return {}
        try:
            return json.loads(self.positions_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}

    def _save_positions(self, positions: dict[str, dict]) -> None:
        self.positions_path.write_text(
            json.dumps(positions, indent=2, default=str), encoding="utf-8"
        )

    def _append_jsonl(self, path: Path, record: dict) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    def _read_jsonl(self, path: Path) -> list[dict]:
        if not path.exists():
            return []
        records = []
        for line in path.read_text(encoding="utf-8").strip().split("\n"):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records

    def record_signal(
        self,
        *,
        symbol: str,
        signal: str,
        confidence: float,
        price: float,
        regime: str = "neutral",
        position_size: float = 0.0,
        sector: str | None = None,
        stop_loss: float | None = None,
        take_profit: float | None = None,
        model_version: str | None = None,
        allow_open: bool = True,
        metadata: dict | None = None,
    ) -> dict:
        """Log signal to signals.jsonl, open/close/adjust paper positions."""
        now = datetime.now(timezone.utc).isoformat()
        sig = str(signal).strip().lower()

        record = {
            "timestamp": now,
            "symbol": symbol,
            "signal": sig,
            "confidence": confidence,
            "price": price,
            "regime": regime,
            "position_size": position_size,
            "sector": sector,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "model_version": model_version,
            "allow_open": allow_open,
            "metadata": metadata,
        }

        # Log signal
        self._append_jsonl(self.signals_path, record)

        # Manage positions
        positions = self._load_positions()
        existing = positions.get(symbol)

        action = "none"

        if sig in ("buy", "strong_buy"):
            if existing and existing.get("direction") == "short":
                # Close short
                pnl_pct = (existing["entry_price"] - price) / existing["entry_price"] * 100
                self._close_position(existing, price, pnl_pct, now)
                del positions[symbol]
                action = "close_short"
            elif not existing and allow_open:
                # Open long
                positions[symbol] = {
                    "symbol": symbol,
                    "direction": "long",
                    "entry_price": price,
                    "entry_time": now,
                    "sector": sector or "unknown",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": position_size,
                    "signal": sig,
                    "confidence": confidence,
                }
                action = "open_long"
            elif not existing:
                action = "blocked_open_long"

        elif sig in ("sell", "strong_sell"):
            if existing and existing.get("direction") == "long":
                # Close long
                pnl_pct = (price - existing["entry_price"]) / existing["entry_price"] * 100
                self._close_position(existing, price, pnl_pct, now)
                del positions[symbol]
                action = "close_long"
            elif not existing and allow_open:
                # Open short
                positions[symbol] = {
                    "symbol": symbol,
                    "direction": "short",
                    "entry_price": price,
                    "entry_time": now,
                    "sector": sector or "unknown",
                    "stop_loss": stop_loss,
                    "take_profit": take_profit,
                    "position_size": position_size,
                    "signal": sig,
                    "confidence": confidence,
                }
                action = "open_short"
            elif not existing:
                action = "blocked_open_short"

        self._save_positions(positions)
        record["action"] = action
        return record

    def _close_position(
        self, position: dict, exit_price: float, pnl_pct: float, exit_time: str
    ) -> None:
        trade = {
            **position,
            "exit_price": exit_price,
            "exit_time": exit_time,
            "pnl_pct": round(pnl_pct, 4),
            "holding_period": exit_time,  # simplified
        }
        self._append_jsonl(self.closed_path, trade)

    def update_prices(self, current_prices: dict[str, float]) -> dict:
        """Mark-to-market open positions. Check stop-loss/take-profit hits."""
        positions = self._load_positions()
        now = datetime.now(timezone.utc).isoformat()
        closed_symbols = []

        for symbol, pos in list(positions.items()):
            price = current_prices.get(symbol)
            if price is None:
                continue

            sl = pos.get("stop_loss")
            tp = pos.get("take_profit")
            direction = pos.get("direction", "long")

            triggered = None

            if direction == "long":
                if sl is not None and price <= sl:
                    triggered = "stop_loss"
                elif tp is not None and price >= tp:
                    triggered = "take_profit"
            else:  # short
                if sl is not None and price >= sl:
                    triggered = "stop_loss"
                elif tp is not None and price <= tp:
                    triggered = "take_profit"

            if triggered:
                entry = pos["entry_price"]
                if direction == "long":
                    pnl_pct = (price - entry) / entry * 100
                else:
                    pnl_pct = (entry - price) / entry * 100

                self._close_position(pos, price, pnl_pct, now)
                closed_symbols.append(symbol)
                logger.info(
                    f"Paper {triggered} triggered for {symbol}: "
                    f"entry={entry}, exit={price}, pnl={pnl_pct:.2f}%"
                )

        for sym in closed_symbols:
            del positions[sym]

        self._save_positions(positions)

        return {
            "updated": len(current_prices),
            "closed": closed_symbols,
        }

    def get_open_positions(self) -> dict[str, dict]:
        return self._load_positions()

    def get_closed_trades(self) -> list[dict]:
        return self._read_jsonl(self.closed_path)

    def get_performance_summary(self) -> dict:
        """
        total_trades, win_rate, avg_pnl_pct, total_pnl_pct,
        best_trade_pct, worst_trade_pct, open_positions.
        """
        trades = self.get_closed_trades()
        positions = self.get_open_positions()

        if not trades:
            return {
                "total_trades": 0,
                "win_rate": 0.0,
                "avg_pnl_pct": 0.0,
                "total_pnl_pct": 0.0,
                "best_trade_pct": 0.0,
                "worst_trade_pct": 0.0,
                "open_positions": len(positions),
            }

        pnls = [t.get("pnl_pct", 0.0) for t in trades]
        wins = sum(1 for p in pnls if p > 0)

        return {
            "total_trades": len(trades),
            "win_rate": round(wins / len(trades), 4) if trades else 0.0,
            "avg_pnl_pct": round(sum(pnls) / len(pnls), 4),
            "total_pnl_pct": round(sum(pnls), 4),
            "best_trade_pct": round(max(pnls), 4),
            "worst_trade_pct": round(min(pnls), 4),
            "open_positions": len(positions),
        }

    def compare_predictions(self) -> list[dict]:
        """
        Match signals.jsonl entries to closed_trades, report
        predicted direction vs realized direction accuracy.
        """
        signals = self._read_jsonl(self.signals_path)
        trades = self.get_closed_trades()

        # Index trades by symbol for matching
        trade_map: dict[str, list[dict]] = {}
        for t in trades:
            sym = t.get("symbol", "")
            trade_map.setdefault(sym, []).append(t)

        comparisons = []
        for sig in signals:
            sym = sig.get("symbol", "")
            sig_signal = sig.get("signal", "hold")
            if sig_signal == "hold":
                continue

            predicted_up = sig_signal in ("buy", "strong_buy")

            # Find matching closed trade
            matched_trades = trade_map.get(sym, [])
            for trade in matched_trades:
                pnl = trade.get("pnl_pct", 0.0)
                realized_up = pnl > 0

                comparisons.append({
                    "symbol": sym,
                    "predicted_signal": sig_signal,
                    "predicted_direction": "up" if predicted_up else "down",
                    "realized_pnl_pct": pnl,
                    "realized_direction": "up" if realized_up else "down",
                    "correct": predicted_up == realized_up,
                })

        return comparisons

    def get_rolling_performance(
        self,
        *,
        last_n_trades: int | None = None,
        last_n_days: int | None = None,
    ) -> dict:
        """Rolling-window performance metrics.

        Filters closed trades by window (last N trades or last N days).

        Returns:
            {sharpe, max_drawdown, win_rate, turnover, total_trades,
             avg_pnl_pct, total_pnl_pct, window_trades, window_days}
        """
        trades = self.get_closed_trades()

        # Filter by window
        if last_n_days is not None and trades:
            cutoff = (datetime.now(timezone.utc) - timedelta(days=last_n_days)).isoformat()
            trades = [t for t in trades if t.get("exit_time", "") >= cutoff]

        if last_n_trades is not None and trades:
            trades = trades[-last_n_trades:]

        empty = {
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "turnover": 0.0,
            "total_trades": 0,
            "avg_pnl_pct": 0.0,
            "total_pnl_pct": 0.0,
            "window_trades": 0,
            "window_days": last_n_days,
        }

        if not trades:
            return empty

        pnls = []
        position_sizes = []
        for t in trades:
            try:
                pnls.append(float(t.get("pnl_pct", 0.0)))
            except (TypeError, ValueError):
                pnls.append(0.0)
            try:
                position_sizes.append(abs(float(t.get("position_size", 0.0))))
            except (TypeError, ValueError):
                position_sizes.append(0.0)

        pnl_arr = np.array(pnls, dtype=float)
        wins = int((pnl_arr > 0).sum())
        total = len(pnl_arr)

        # Sharpe from trade-level PnL
        if total > 1 and pnl_arr.std() > 0:
            sharpe = float(pnl_arr.mean() / pnl_arr.std() * np.sqrt(min(252, total)))
        else:
            sharpe = 0.0

        # Max drawdown from cumulative PnL curve
        cum_pnl = np.cumsum(pnl_arr)
        peak = np.maximum.accumulate(cum_pnl)
        dd = cum_pnl - peak
        max_dd = float(dd.min()) if len(dd) > 0 else 0.0

        # Turnover: sum of absolute position sizes (approximation)
        turnover = float(sum(position_sizes))

        return {
            "sharpe": round(sharpe, 4),
            "max_drawdown": round(max_dd, 4),
            "win_rate": round(wins / total, 4) if total > 0 else 0.0,
            "turnover": round(turnover, 4),
            "total_trades": total,
            "avg_pnl_pct": round(float(pnl_arr.mean()), 4) if total > 0 else 0.0,
            "total_pnl_pct": round(float(pnl_arr.sum()), 4),
            "window_trades": total,
            "window_days": last_n_days,
        }

    def reset(self) -> None:
        """Clear all paper trading data."""
        for path in [self.signals_path, self.positions_path, self.closed_path]:
            if path.exists():
                path.unlink()


def check_paper_trading_gates(
    paper_dir: str = "data/paper_trading",
    initial_capital: float = 100_000.0,
    *,
    last_n_trades: int = 50,
    min_trades: int = 10,
    sharpe_min: float = 0.0,
    max_drawdown_floor: float = -20.0,
    win_rate_min: float = 0.40,
    turnover_max: float = 5.0,
) -> dict:
    """Check paper-trading promotion gates.

    Returns:
        {passed, reason, checks: {sharpe, max_drawdown, win_rate, turnover}, metrics}
    """
    pp = PaperPortfolio(paper_dir=paper_dir, initial_capital=initial_capital)
    metrics = pp.get_rolling_performance(last_n_trades=last_n_trades)

    checks: dict[str, dict] = {}

    # Insufficient trades
    if metrics["total_trades"] < min_trades:
        return {
            "passed": False,
            "reason": f"Insufficient trades: {metrics['total_trades']} < {min_trades}",
            "checks": {},
            "metrics": metrics,
        }

    # Sharpe gate
    passed_sharpe = metrics["sharpe"] >= sharpe_min
    checks["sharpe"] = {
        "passed": passed_sharpe,
        "value": metrics["sharpe"],
        "threshold": sharpe_min,
    }

    # Max drawdown gate (max_drawdown is negative or zero, floor is negative)
    passed_dd = metrics["max_drawdown"] >= max_drawdown_floor
    checks["max_drawdown"] = {
        "passed": passed_dd,
        "value": metrics["max_drawdown"],
        "threshold": max_drawdown_floor,
    }

    # Win rate gate
    passed_wr = metrics["win_rate"] >= win_rate_min
    checks["win_rate"] = {
        "passed": passed_wr,
        "value": metrics["win_rate"],
        "threshold": win_rate_min,
    }

    # Turnover gate
    passed_turnover = metrics["turnover"] <= turnover_max
    checks["turnover"] = {
        "passed": passed_turnover,
        "value": metrics["turnover"],
        "threshold": turnover_max,
    }

    all_passed = all(c["passed"] for c in checks.values())
    if all_passed:
        reason = "All paper-trading gates passed"
    else:
        failed = [k for k, v in checks.items() if not v["passed"]]
        reason = f"Paper-trading gate(s) failed: {failed}"

    return {
        "passed": all_passed,
        "reason": reason,
        "checks": checks,
        "metrics": metrics,
    }
