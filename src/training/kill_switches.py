"""Risk kill-switches for paper trading.

Circuit breakers that halt paper trading when conditions are dangerous:
- Max daily loss exceeded
- Stale data feed
- Abnormal slippage between expected and realized exits
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


def check_max_daily_loss(
    closed_trades: list[dict],
    max_daily_loss_pct: float = 5.0,
) -> dict:
    """Sum today's realized losses.

    Args:
        closed_trades: list of closed trade dicts with exit_time and pnl_pct
        max_daily_loss_pct: threshold for triggering (positive number, e.g. 5.0)

    Returns:
        {triggered, daily_loss_pct, threshold}
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    daily_pnl = 0.0

    for trade in closed_trades:
        exit_time = trade.get("exit_time", "")
        if isinstance(exit_time, str) and exit_time[:10] == today_str:
            pnl = trade.get("pnl_pct", 0.0)
            try:
                daily_pnl += float(pnl)
            except (TypeError, ValueError):
                continue

    triggered = daily_pnl < -max_daily_loss_pct

    return {
        "triggered": triggered,
        "daily_loss_pct": round(daily_pnl, 4),
        "threshold": -max_daily_loss_pct,
    }


def check_stale_data_feed(
    latest_data: dict | None,
    max_stale_ms: int = 60_000,
) -> dict:
    """Check age of realtime feed data.

    Args:
        latest_data: dict with 'age_ms' or 'timestamp' key from realtime feed.
                     If None, treats data as stale.
        max_stale_ms: max acceptable age in milliseconds

    Returns:
        {triggered, age_ms, threshold_ms}
    """
    if latest_data is None:
        return {
            "triggered": True,
            "age_ms": None,
            "threshold_ms": max_stale_ms,
        }

    age_ms = latest_data.get("age_ms")
    if age_ms is None:
        # Try to compute from timestamp
        ts = latest_data.get("timestamp")
        if ts is not None:
            try:
                if isinstance(ts, str):
                    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                elif isinstance(ts, (int, float)):
                    dt = datetime.fromtimestamp(ts / 1000 if ts > 1e12 else ts, tz=timezone.utc)
                else:
                    dt = ts
                now = datetime.now(timezone.utc)
                age_ms = int((now - dt).total_seconds() * 1000)
            except Exception:
                age_ms = None

    if age_ms is None:
        return {
            "triggered": True,
            "age_ms": None,
            "threshold_ms": max_stale_ms,
        }

    triggered = age_ms > max_stale_ms

    return {
        "triggered": triggered,
        "age_ms": age_ms,
        "threshold_ms": max_stale_ms,
    }


def check_abnormal_slippage(
    closed_trades: list[dict],
    positions: dict[str, dict] | None = None,
    slippage_threshold_pct: float = 2.0,
    lookback_trades: int = 20,
) -> dict:
    """Compare exit prices to expected SL/TP targets.

    For recent closed trades, measure how far the actual exit price diverged
    from the closest SL or TP target. High average slippage indicates
    execution issues.

    Args:
        closed_trades: list of closed trade dicts
        positions: current open positions (unused, for future extension)
        slippage_threshold_pct: threshold for triggering
        lookback_trades: how many recent trades to check

    Returns:
        {triggered, avg_slippage_pct, threshold, trades_checked}
    """
    recent = closed_trades[-lookback_trades:] if closed_trades else []

    slippages = []
    for trade in recent:
        exit_price = trade.get("exit_price")
        entry_price = trade.get("entry_price")
        sl = trade.get("stop_loss")
        tp = trade.get("take_profit")

        if exit_price is None or entry_price is None:
            continue

        try:
            exit_price = float(exit_price)
            entry_price = float(entry_price)
        except (TypeError, ValueError):
            continue

        if entry_price == 0:
            continue

        # Find the expected exit (closest of SL or TP to actual exit)
        expected_exits = []
        if sl is not None:
            try:
                expected_exits.append(float(sl))
            except (TypeError, ValueError):
                pass
        if tp is not None:
            try:
                expected_exits.append(float(tp))
            except (TypeError, ValueError):
                pass

        if not expected_exits:
            continue

        # Slippage = distance from closest expected exit, as % of entry price
        distances = [abs(exit_price - exp) / entry_price * 100 for exp in expected_exits]
        min_slippage = min(distances)
        slippages.append(min_slippage)

    if not slippages:
        return {
            "triggered": False,
            "avg_slippage_pct": 0.0,
            "threshold": slippage_threshold_pct,
            "trades_checked": 0,
        }

    avg_slippage = sum(slippages) / len(slippages)
    triggered = avg_slippage > slippage_threshold_pct

    return {
        "triggered": triggered,
        "avg_slippage_pct": round(avg_slippage, 4),
        "threshold": slippage_threshold_pct,
        "trades_checked": len(slippages),
    }


def check_all_kill_switches(
    *,
    closed_trades: list[dict] | None = None,
    positions: dict[str, dict] | None = None,
    latest_realtime_data: dict | None = None,
    max_daily_loss_pct: float = 5.0,
    max_stale_ms: int = 60_000,
    slippage_threshold_pct: float = 2.0,
) -> dict:
    """Run all kill-switch checks.

    Returns:
        {halted, checks: {daily_loss, stale_data, slippage}, reasons: [...]}
    """
    closed_trades = closed_trades or []
    positions = positions or {}

    checks = {}
    reasons = []

    # Daily loss check
    daily_loss = check_max_daily_loss(closed_trades, max_daily_loss_pct=max_daily_loss_pct)
    checks["daily_loss"] = daily_loss
    if daily_loss["triggered"]:
        reasons.append(
            f"Daily loss {daily_loss['daily_loss_pct']:.2f}% exceeds "
            f"threshold {daily_loss['threshold']:.1f}%"
        )

    # Stale data check
    stale = check_stale_data_feed(latest_realtime_data, max_stale_ms=max_stale_ms)
    checks["stale_data"] = stale
    if stale["triggered"]:
        age_str = f"{stale['age_ms']}ms" if stale["age_ms"] is not None else "unknown"
        reasons.append(
            f"Data feed stale: age={age_str}, threshold={stale['threshold_ms']}ms"
        )

    # Slippage check
    slippage = check_abnormal_slippage(
        closed_trades, positions,
        slippage_threshold_pct=slippage_threshold_pct,
    )
    checks["slippage"] = slippage
    if slippage["triggered"]:
        reasons.append(
            f"Abnormal slippage: avg={slippage['avg_slippage_pct']:.2f}%, "
            f"threshold={slippage['threshold']:.1f}%"
        )

    halted = len(reasons) > 0

    return {
        "halted": halted,
        "checks": checks,
        "reasons": reasons,
    }
