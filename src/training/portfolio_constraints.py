"""Portfolio-level constraint enforcement and risk summary."""

from __future__ import annotations

from collections import defaultdict
from typing import Any


def enforce_portfolio_constraints(
    positions: list[dict[str, Any]],
    *,
    max_total_exposure: float = 1.0,
    max_single_weight: float = 0.20,
    max_sector_weight: float = 0.35,
) -> list[dict[str, Any]]:
    """
    Apply caps to proposed positions. Returns same list with adjusted weights
    and added 'constraint_applied' field.

    Steps: 1) cap single-stock, 2) cap sector (proportional within sector),
    3) cap total exposure (proportional across all).
    """
    if not positions:
        return []

    result = [dict(p) for p in positions]

    # Step 1: Cap single-stock weight
    for pos in result:
        w = pos.get("weight", 0.0)
        if abs(w) > max_single_weight:
            sign = 1.0 if w >= 0 else -1.0
            pos["weight"] = sign * max_single_weight
            pos["constraint_applied"] = pos.get("constraint_applied", [])
            pos["constraint_applied"].append("single_stock_cap")

    # Step 2: Cap sector weight (proportional scale-down within sector)
    sector_weights: dict[str, float] = defaultdict(float)
    sector_positions: dict[str, list[dict]] = defaultdict(list)
    for pos in result:
        sector = pos.get("sector", "unknown")
        sector_weights[sector] += abs(pos.get("weight", 0.0))
        sector_positions[sector].append(pos)

    for sector, total in sector_weights.items():
        if total > max_sector_weight and total > 0:
            scale = max_sector_weight / total
            for pos in sector_positions[sector]:
                pos["weight"] = pos["weight"] * scale
                pos["constraint_applied"] = pos.get("constraint_applied", [])
                pos["constraint_applied"].append("sector_cap")

    # Step 3: Cap total exposure (proportional across all)
    total_exposure = sum(abs(p.get("weight", 0.0)) for p in result)
    if total_exposure > max_total_exposure and total_exposure > 0:
        scale = max_total_exposure / total_exposure
        for pos in result:
            pos["weight"] = pos["weight"] * scale
            pos["constraint_applied"] = pos.get("constraint_applied", [])
            pos["constraint_applied"].append("total_exposure_cap")

    # Round weights
    for pos in result:
        pos["weight"] = round(pos.get("weight", 0.0), 6)
        if "constraint_applied" not in pos:
            pos["constraint_applied"] = []

    return result


def compute_portfolio_risk_summary(
    positions: list[dict[str, Any]],
) -> dict[str, float]:
    """
    Compute portfolio risk metrics.

    Returns {total_exposure, long_exposure, short_exposure, num_positions,
     max_single_weight, sector_concentrations, herfindahl_index}.
    """
    if not positions:
        return {
            "total_exposure": 0.0,
            "long_exposure": 0.0,
            "short_exposure": 0.0,
            "num_positions": 0,
            "max_single_weight": 0.0,
            "sector_concentrations": {},
            "herfindahl_index": 0.0,
        }

    weights = [p.get("weight", 0.0) for p in positions]
    abs_weights = [abs(w) for w in weights]

    long_exposure = sum(w for w in weights if w > 0)
    short_exposure = sum(abs(w) for w in weights if w < 0)
    total_exposure = sum(abs_weights)

    # Sector concentrations
    sector_weights: dict[str, float] = defaultdict(float)
    for p in positions:
        sector = p.get("sector", "unknown")
        sector_weights[sector] += abs(p.get("weight", 0.0))

    # Herfindahl-Hirschman Index (sum of squared weight proportions)
    hhi = 0.0
    if total_exposure > 0:
        hhi = sum((w / total_exposure) ** 2 for w in abs_weights)

    return {
        "total_exposure": round(total_exposure, 6),
        "long_exposure": round(long_exposure, 6),
        "short_exposure": round(short_exposure, 6),
        "num_positions": len(positions),
        "max_single_weight": round(max(abs_weights) if abs_weights else 0.0, 6),
        "sector_concentrations": {k: round(v, 6) for k, v in sector_weights.items()},
        "herfindahl_index": round(hhi, 6),
    }
