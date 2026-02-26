"""
File-based model registry.

Tracks model versions, metrics, and active/inactive status in
models/registry.json.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

DEFAULT_REGISTRY_PATH = "models/registry.json"


def _load_registry(path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    p = Path(path)
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"models": [], "active_version": None}


def _save_registry(data: Dict[str, Any], path: str = DEFAULT_REGISTRY_PATH) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(data, indent=2), encoding="utf-8")


def register_model(
    version: int,
    model_path: str,
    metrics: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
    registry_path: str = DEFAULT_REGISTRY_PATH,
) -> Dict[str, Any]:
    """Register a new model version in the registry."""
    registry = _load_registry(registry_path)

    entry = {
        "version": version,
        "model_path": model_path,
        "metrics": metrics,
        "metadata": metadata or {},
        "status": "inactive",
        "registered_at": datetime.now(timezone.utc).isoformat(),
    }

    # Remove existing entry for same version
    registry["models"] = [m for m in registry["models"] if m["version"] != version]
    registry["models"].append(entry)
    registry["models"].sort(key=lambda m: m["version"])

    _save_registry(registry, registry_path)
    logger.info(f"Registered model v{version} at {model_path}")
    return entry


def get_active_model(registry_path: str = DEFAULT_REGISTRY_PATH) -> Optional[Dict[str, Any]]:
    """Get the currently active model entry."""
    registry = _load_registry(registry_path)
    active_version = registry.get("active_version")
    if active_version is None:
        return None
    for m in registry["models"]:
        if m["version"] == active_version:
            return m
    return None


def set_active(version: int, registry_path: str = DEFAULT_REGISTRY_PATH) -> bool:
    """Set a model version as the active model."""
    registry = _load_registry(registry_path)
    found = False
    for m in registry["models"]:
        if m["version"] == version:
            m["status"] = "active"
            found = True
        else:
            m["status"] = "inactive"

    if found:
        registry["active_version"] = version
        _save_registry(registry, registry_path)
        logger.info(f"Model v{version} set as active")
    else:
        logger.warning(f"Model v{version} not found in registry")

    return found


def list_models(registry_path: str = DEFAULT_REGISTRY_PATH) -> List[Dict[str, Any]]:
    """List all registered models."""
    registry = _load_registry(registry_path)
    return registry.get("models", [])


# ---------------------------------------------------------------------------
#  Rollout gates — promote only if the candidate beats the incumbent
# ---------------------------------------------------------------------------

# Default thresholds: candidate must be >= these relative to incumbent.
# A value of 0.0 means "at least as good"; positive values raise the bar.
DEFAULT_GATE_THRESHOLDS = {
    "sharpe_min_improvement": 0.0,   # candidate.sharpe >= incumbent.sharpe + this
    "cagr_min_improvement": 0.0,     # candidate.cagr >= incumbent.cagr + this
    "max_dd_tolerance": 0.0,         # candidate.max_dd >= incumbent.max_dd - this
                                     # (max_dd is negative, so "better" = less negative)
    "turnover_max_ratio": 1.5,       # candidate.turnover <= incumbent.turnover * this
    "net_vs_gross_min": 0.70,        # candidate.net >= candidate.gross * this
}


def get_model_metrics(entry: Dict[str, Any]) -> Dict[str, float]:
    """
    Extract backtest metrics from a registry entry.

    Looks in entry["metrics"] first (flat), then in
    entry["metrics"]["backtest_metrics"] for nested storage.
    """
    metrics = entry.get("metrics", {})

    def _get(key: str) -> Optional[float]:
        # Flat
        if key in metrics:
            return float(metrics[key])
        # Nested under backtest_metrics
        bt = metrics.get("backtest_metrics", {})
        if key in bt:
            return float(bt[key])
        # Nested under position_sized_backtest_metrics (preferred if present)
        pbt = metrics.get("position_sized_backtest_metrics", {})
        if key in pbt:
            return float(pbt[key])
        return None

    return {
        "sharpe": _get("sharpe"),
        "cagr": _get("cagr"),
        "max_drawdown": _get("max_drawdown"),
        "turnover": _get("turnover"),
        "net_return_total": _get("net_return_total"),
        "gross_return_total": _get("gross_return_total"),
    }


def promote_if_better(
    candidate_version: int,
    registry_path: str = DEFAULT_REGISTRY_PATH,
    thresholds: Optional[Dict[str, float]] = None,
    paper_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Promote *candidate_version* only if its backtest Sharpe, CAGR, and MaxDD
    beat the current active model.

    If no active model exists, the candidate is promoted unconditionally.

    Args:
        candidate_version: version number of the candidate model.
        registry_path: path to registry JSON.
        thresholds: override default gate thresholds.

    Returns:
        {
            "promoted": bool,
            "reason": str,
            "candidate_version": int,
            "incumbent_version": int | None,
            "checks": {metric: {"passed": bool, "candidate": ..., "incumbent": ...}},
        }
    """
    gates = {**DEFAULT_GATE_THRESHOLDS, **(thresholds or {})}
    registry = _load_registry(registry_path)

    # Find candidate entry
    candidate = None
    for m in registry["models"]:
        if m["version"] == candidate_version:
            candidate = m
            break
    if candidate is None:
        return {
            "promoted": False,
            "reason": f"Candidate v{candidate_version} not found in registry",
            "candidate_version": candidate_version,
            "incumbent_version": None,
            "checks": {},
        }

    # Find incumbent (current active)
    incumbent = get_active_model(registry_path)
    if incumbent is None:
        # No active model → auto-promote
        set_active(candidate_version, registry_path)
        logger.info(f"No active model found — auto-promoting v{candidate_version}")
        return {
            "promoted": True,
            "reason": "No incumbent; auto-promoted",
            "candidate_version": candidate_version,
            "incumbent_version": None,
            "checks": {},
        }

    # Compare metrics
    c_metrics = get_model_metrics(candidate)
    i_metrics = get_model_metrics(incumbent)

    checks: Dict[str, Dict[str, Any]] = {}
    all_passed = True

    # Sharpe: higher is better
    c_sharpe = c_metrics.get("sharpe")
    i_sharpe = i_metrics.get("sharpe")
    if c_sharpe is not None and i_sharpe is not None:
        passed = c_sharpe >= i_sharpe + gates["sharpe_min_improvement"]
        checks["sharpe"] = {"passed": passed, "candidate": c_sharpe, "incumbent": i_sharpe}
        if not passed:
            all_passed = False
    else:
        checks["sharpe"] = {"passed": False, "candidate": c_sharpe, "incumbent": i_sharpe,
                            "note": "metric missing — gate failed"}
        all_passed = False

    # CAGR: higher is better
    c_cagr = c_metrics.get("cagr")
    i_cagr = i_metrics.get("cagr")
    if c_cagr is not None and i_cagr is not None:
        passed = c_cagr >= i_cagr + gates["cagr_min_improvement"]
        checks["cagr"] = {"passed": passed, "candidate": c_cagr, "incumbent": i_cagr}
        if not passed:
            all_passed = False
    else:
        checks["cagr"] = {"passed": False, "candidate": c_cagr, "incumbent": i_cagr,
                          "note": "metric missing — gate failed"}
        all_passed = False

    # MaxDD: less negative is better (closer to 0)
    c_dd = c_metrics.get("max_drawdown")
    i_dd = i_metrics.get("max_drawdown")
    if c_dd is not None and i_dd is not None:
        passed = c_dd >= i_dd - gates["max_dd_tolerance"]
        checks["max_drawdown"] = {"passed": passed, "candidate": c_dd, "incumbent": i_dd}
        if not passed:
            all_passed = False
    else:
        checks["max_drawdown"] = {"passed": False, "candidate": c_dd, "incumbent": i_dd,
                                  "note": "metric missing — gate failed"}
        all_passed = False

    # Turnover: candidate should not overtrade (<=1.5x incumbent)
    c_turn = c_metrics.get("turnover")
    i_turn = i_metrics.get("turnover")
    turnover_max_ratio = gates.get("turnover_max_ratio", 1.5)
    if c_turn is not None and i_turn is not None and i_turn > 0:
        passed = c_turn <= i_turn * turnover_max_ratio
        checks["turnover"] = {"passed": passed, "candidate": c_turn, "incumbent": i_turn}
        if not passed:
            all_passed = False
    elif c_turn is not None:
        checks["turnover"] = {"passed": True, "candidate": c_turn, "incumbent": i_turn,
                               "note": "no incumbent baseline — gate skipped"}

    # Net vs Gross: costs should not eat more than 30% of gross returns
    c_net = c_metrics.get("net_return_total")
    c_gross = c_metrics.get("gross_return_total")
    net_vs_gross_min = gates.get("net_vs_gross_min", 0.70)
    if c_net is not None and c_gross is not None and c_gross > 0:
        ratio = c_net / c_gross
        passed = ratio >= net_vs_gross_min
        checks["net_vs_gross"] = {"passed": passed, "candidate_ratio": round(ratio, 4),
                                   "threshold": net_vs_gross_min}
        if not passed:
            all_passed = False
    elif c_gross is not None and c_gross <= 0:
        checks["net_vs_gross"] = {"passed": True, "note": "gross non-positive — gate skipped"}

    # Paper-trading gate (non-blocking if no paper data exists)
    try:
        from pathlib import Path as _Path
        if paper_dir is None:
            try:
                from config import settings as _cfg
                _paper_dir = _cfg.paper_trading_dir
            except Exception:
                _paper_dir = "data/paper_trading"
        else:
            _paper_dir = paper_dir
        paper_closed = _Path(_paper_dir) / "closed_trades.jsonl"
        if paper_closed.exists():
            from training.paper_trading import check_paper_trading_gates
            paper_result = check_paper_trading_gates(paper_dir=_paper_dir)
            checks["paper_trading"] = {
                "passed": paper_result["passed"],
                "reason": paper_result["reason"],
                "metrics": paper_result.get("metrics"),
            }
            if not paper_result["passed"]:
                all_passed = False
    except Exception as e:
        logger.debug(f"Paper-trading gate skipped: {e}")

    if all_passed:
        set_active(candidate_version, registry_path)
        reason = (
            f"v{candidate_version} beats v{incumbent['version']} on all gates — promoted"
        )
        logger.info(reason)
    else:
        failed = [k for k, v in checks.items() if not v["passed"]]
        reason = (
            f"v{candidate_version} failed gate(s) {failed} vs v{incumbent['version']} — NOT promoted"
        )
        logger.warning(reason)

    return {
        "promoted": all_passed,
        "reason": reason,
        "candidate_version": candidate_version,
        "incumbent_version": incumbent["version"],
        "checks": checks,
    }
