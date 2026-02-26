"""Pre-trade risk checks.

Pure functions that gate order submission.  Each returns
{blocked, check, reason, ...} — same pattern as kill_switches.py.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def check_max_position_size(
    proposed_size: float,
    max_single_position: float = 0.20,
) -> dict:
    """Block if a single position exceeds max allocation."""
    blocked = proposed_size > max_single_position
    return {
        "blocked": blocked,
        "check": "max_position_size",
        "reason": (
            f"Position size {proposed_size:.2%} exceeds max {max_single_position:.2%}"
            if blocked
            else ""
        ),
        "proposed_size": proposed_size,
        "threshold": max_single_position,
    }


def check_sector_concentration(
    symbol: str,
    proposed_size: float,
    positions: dict[str, dict],
    sector_map: dict[str, str],
    max_sector_weight: float = 0.35,
) -> dict:
    """Block if adding this trade pushes sector weight past threshold."""
    target_sector = sector_map.get(symbol, "unknown")
    sector_weight = proposed_size
    for sym, pos in positions.items():
        if sector_map.get(sym, "unknown") == target_sector:
            sector_weight += abs(pos.get("position_size", 0.0))

    blocked = sector_weight > max_sector_weight
    return {
        "blocked": blocked,
        "check": "sector_concentration",
        "reason": (
            f"Sector '{target_sector}' weight {sector_weight:.2%} exceeds max {max_sector_weight:.2%}"
            if blocked
            else ""
        ),
        "sector": target_sector,
        "sector_weight": round(sector_weight, 4),
        "threshold": max_sector_weight,
    }


def check_daily_loss_cap(
    closed_trades_today: list[dict],
    max_daily_loss_pct: float = 5.0,
) -> dict:
    """Block if realised daily losses already exceed cap."""
    daily_pnl = 0.0
    for trade in closed_trades_today:
        try:
            daily_pnl += float(trade.get("pnl_pct", 0.0))
        except (TypeError, ValueError):
            continue

    blocked = daily_pnl < -max_daily_loss_pct
    return {
        "blocked": blocked,
        "check": "daily_loss_cap",
        "reason": (
            f"Daily loss {daily_pnl:.2f}% exceeds cap -{max_daily_loss_pct:.1f}%"
            if blocked
            else ""
        ),
        "daily_pnl_pct": round(daily_pnl, 4),
        "threshold": -max_daily_loss_pct,
    }


def check_total_exposure(
    proposed_size: float,
    positions: dict[str, dict],
    max_total_exposure: float = 1.0,
) -> dict:
    """Block if gross portfolio exposure would exceed limit."""
    current_exposure = sum(
        abs(pos.get("position_size", 0.0)) for pos in positions.values()
    )
    new_exposure = current_exposure + abs(proposed_size)
    blocked = new_exposure > max_total_exposure
    return {
        "blocked": blocked,
        "check": "total_exposure",
        "reason": (
            f"Total exposure {new_exposure:.2%} exceeds max {max_total_exposure:.2%}"
            if blocked
            else ""
        ),
        "current_exposure": round(current_exposure, 4),
        "new_exposure": round(new_exposure, 4),
        "threshold": max_total_exposure,
    }


def check_all_pre_trade_risk(
    *,
    symbol: str,
    proposed_size: float,
    positions: dict[str, dict] | None = None,
    closed_trades_today: list[dict] | None = None,
    sector_map: dict[str, str] | None = None,
    max_single_position: float = 0.20,
    max_sector_weight: float = 0.35,
    max_daily_loss_pct: float = 5.0,
    max_total_exposure: float = 1.0,
) -> dict:
    """Run all pre-trade risk checks.

    Returns:
        {blocked, checks: {...}, reasons: [...]}
    """
    positions = positions or {}
    closed_trades_today = closed_trades_today or []
    sector_map = sector_map or {}

    checks: dict[str, dict] = {}
    reasons: list[str] = []

    pos = check_max_position_size(proposed_size, max_single_position)
    checks["max_position_size"] = pos
    if pos["blocked"]:
        reasons.append(pos["reason"])

    sector = check_sector_concentration(
        symbol, proposed_size, positions, sector_map, max_sector_weight
    )
    checks["sector_concentration"] = sector
    if sector["blocked"]:
        reasons.append(sector["reason"])

    daily = check_daily_loss_cap(closed_trades_today, max_daily_loss_pct)
    checks["daily_loss_cap"] = daily
    if daily["blocked"]:
        reasons.append(daily["reason"])

    exposure = check_total_exposure(proposed_size, positions, max_total_exposure)
    checks["total_exposure"] = exposure
    if exposure["blocked"]:
        reasons.append(exposure["reason"])

    return {
        "blocked": len(reasons) > 0,
        "checks": checks,
        "reasons": reasons,
    }
