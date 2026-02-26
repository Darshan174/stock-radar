"""Canary rollout mode for controlled go-live.

State persisted in {canary_dir}/canary_state.json.
Only allowed symbols can trade, and trades are capped so that
a breach automatically disables canary mode.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_STATE: dict = {
    "enabled": False,
    "total_trades": 0,
    "total_pnl_pct": 0.0,
    "breach_count": 0,
    "history": [],
    "disabled_reason": None,
    "updated_at": None,
}


def load_canary_state(canary_dir: str = "data/canary") -> dict:
    """Load canary state from disk, returning defaults if missing."""
    path = Path(canary_dir) / "canary_state.json"
    if not path.exists():
        return dict(_DEFAULT_STATE)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        # Merge with defaults for missing keys
        merged = dict(_DEFAULT_STATE)
        merged.update(data)
        return merged
    except (json.JSONDecodeError, OSError):
        return dict(_DEFAULT_STATE)


def save_canary_state(state: dict, canary_dir: str = "data/canary") -> None:
    """Persist canary state to disk."""
    d = Path(canary_dir)
    d.mkdir(parents=True, exist_ok=True)
    state["updated_at"] = datetime.now(timezone.utc).isoformat()
    (d / "canary_state.json").write_text(
        json.dumps(state, indent=2, default=str), encoding="utf-8"
    )


def is_canary_eligible(symbol: str, allowed_symbols: list[str]) -> bool:
    """Return True if symbol is in the canary allow-list (or list is empty)."""
    if not allowed_symbols:
        return True
    return symbol.upper() in [s.upper() for s in allowed_symbols]


def check_canary_breach(
    state: dict,
    *,
    max_trades: int = 50,
    max_loss_pct: float = 3.0,
    max_breach_count: int = 2,
) -> dict:
    """Check whether canary limits are breached.

    Returns:
        {breached, reasons, state}
    Mutates state.enabled = False on breach.
    """
    reasons: list[str] = []

    if state["total_trades"] >= max_trades:
        reasons.append(
            f"Canary trade limit reached: {state['total_trades']} >= {max_trades}"
        )

    if state["total_pnl_pct"] < -max_loss_pct:
        reasons.append(
            f"Canary loss limit breached: {state['total_pnl_pct']:.2f}% < -{max_loss_pct:.1f}%"
        )

    breached = len(reasons) > 0
    if breached:
        state["breach_count"] = state.get("breach_count", 0) + 1
        if state["breach_count"] >= max_breach_count:
            state["enabled"] = False
            state["disabled_reason"] = f"Auto-disabled after {state['breach_count']} breaches"
            reasons.append(state["disabled_reason"])

    return {"breached": breached, "reasons": reasons, "state": state}


def record_canary_trade(state: dict, pnl_pct: float) -> dict:
    """Record a canary trade and update running totals.

    Returns the mutated state.
    """
    state["total_trades"] = state.get("total_trades", 0) + 1
    state["total_pnl_pct"] = round(
        state.get("total_pnl_pct", 0.0) + pnl_pct, 4
    )
    state["history"] = state.get("history", [])
    state["history"].append(
        {
            "pnl_pct": pnl_pct,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    )
    return state


def enable_canary(canary_dir: str = "data/canary") -> dict:
    """Enable canary mode, resetting counters."""
    state = {
        "enabled": True,
        "total_trades": 0,
        "total_pnl_pct": 0.0,
        "breach_count": 0,
        "history": [],
        "disabled_reason": None,
        "updated_at": None,
    }
    save_canary_state(state, canary_dir)
    logger.info("Canary mode ENABLED")
    return state


def disable_canary(canary_dir: str = "data/canary", reason: str = "manual") -> dict:
    """Disable canary mode."""
    state = load_canary_state(canary_dir)
    state["enabled"] = False
    state["disabled_reason"] = reason
    save_canary_state(state, canary_dir)
    logger.info("Canary mode DISABLED: %s", reason)
    return state
