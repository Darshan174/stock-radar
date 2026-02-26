"""
Feature health monitoring for Phase-5 features.

Tracks NaN rates, distribution drift (mean/std shift), and
value-range violations for the 17 new cross-sectional, factor,
and microstructure features in production.

Usage:
    from training.feature_health import check_feature_health, log_feature_health

    report = check_feature_health(feature_vectors, reference_stats=ref)
    log_feature_health(report)
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional

import numpy as np

from training.cross_sectional import ALL_NEW_FEATURE_NAMES
from training.feature_engineering import FEATURE_NAMES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Reference statistics (populated from training set at model-train time)
# ---------------------------------------------------------------------------

# Default thresholds
DEFAULT_NAN_RATE_WARN = 0.25      # warn if >25 % NaN for any feature
DEFAULT_NAN_RATE_CRITICAL = 0.50  # critical if >50 %
DEFAULT_DRIFT_SIGMA = 2.0        # warn if mean shifted >2σ from reference


def compute_reference_stats(
    feature_matrix: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Dict[str, float]]:
    """
    Compute per-feature reference statistics from a training matrix.

    Save the result alongside the model metadata so it can be loaded
    at inference time for drift comparison.

    Returns:
        Dict mapping feature name → {mean, std, nan_rate, p5, p95}.
    """
    n_samples, n_features = feature_matrix.shape
    stats: Dict[str, Dict[str, float]] = {}

    for idx, name in enumerate(feature_names):
        col = feature_matrix[:, idx].astype(np.float64)
        nan_mask = np.isnan(col)
        nan_rate = float(np.mean(nan_mask))

        valid = col[~nan_mask]
        if len(valid) > 0:
            stats[name] = {
                "mean": float(np.mean(valid)),
                "std": float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0,
                "nan_rate": nan_rate,
                "p5": float(np.percentile(valid, 5)),
                "p95": float(np.percentile(valid, 95)),
                "count": int(len(valid)),
            }
        else:
            stats[name] = {
                "mean": 0.0,
                "std": 0.0,
                "nan_rate": 1.0,
                "p5": 0.0,
                "p95": 0.0,
                "count": 0,
            }

    return stats


# ---------------------------------------------------------------------------
#  Live health check
# ---------------------------------------------------------------------------

def check_feature_health(
    feature_vectors: np.ndarray,
    feature_names: Optional[List[str]] = None,
    reference_stats: Optional[Dict[str, Dict[str, float]]] = None,
    *,
    nan_rate_warn: float = DEFAULT_NAN_RATE_WARN,
    nan_rate_critical: float = DEFAULT_NAN_RATE_CRITICAL,
    drift_sigma: float = DEFAULT_DRIFT_SIGMA,
    phase5_only: bool = False,
) -> Dict[str, Any]:
    """
    Run feature health diagnostics on a batch of feature vectors.

    Args:
        feature_vectors: (N, F) array of feature values.
        feature_names: names matching columns; defaults to FEATURE_NAMES.
        reference_stats: per-feature reference from training (optional).
        nan_rate_warn: NaN rate threshold for WARN level.
        nan_rate_critical: NaN rate threshold for CRITICAL level.
        drift_sigma: number of reference σ for drift detection.
        phase5_only: if True, only check the 17 Phase-5 features.

    Returns:
        {
            "overall_status": "healthy" | "warn" | "critical",
            "features": {name: {nan_rate, status, drift_z, ...}, ...},
            "summary": {healthy: N, warn: N, critical: N},
        }
    """
    if feature_names is None:
        feature_names = list(FEATURE_NAMES)

    if feature_vectors.ndim == 1:
        feature_vectors = feature_vectors.reshape(1, -1)

    n_samples, n_features = feature_vectors.shape

    # Determine which features to check
    if phase5_only:
        target_names = set(ALL_NEW_FEATURE_NAMES)
    else:
        target_names = set(feature_names)

    feature_reports: Dict[str, Dict[str, Any]] = {}
    counts = {"healthy": 0, "warn": 0, "critical": 0}

    for idx, name in enumerate(feature_names):
        if idx >= n_features:
            break
        if name not in target_names:
            continue

        col = feature_vectors[:, idx].astype(np.float64)
        nan_mask = np.isnan(col)
        nan_rate = float(np.mean(nan_mask))

        valid = col[~nan_mask]
        live_mean = float(np.mean(valid)) if len(valid) > 0 else float("nan")
        live_std = float(np.std(valid, ddof=1)) if len(valid) > 1 else 0.0

        # Determine NaN-rate status.
        # When reference stats exist, compare the NaN-rate *increase* over
        # the training baseline rather than the absolute rate.  A feature
        # that was 100 % NaN in training is expected to stay NaN at
        # inference and should not trigger a false critical.
        ref_nan_rate = 0.0
        if reference_stats and name in reference_stats:
            ref_nan_rate = reference_stats[name].get("nan_rate", 0.0)

        nan_increase = max(0.0, nan_rate - ref_nan_rate)
        if nan_increase >= nan_rate_critical:
            status = "critical"
        elif nan_increase >= nan_rate_warn:
            status = "warn"
        else:
            status = "healthy"

        # Drift detection (if reference available)
        drift_z = None
        drift_status = "unknown"
        if reference_stats and name in reference_stats:
            ref = reference_stats[name]
            ref_std = ref.get("std", 0.0)
            if ref_std > 1e-12 and not math.isnan(live_mean):
                drift_z = abs(live_mean - ref["mean"]) / ref_std
                if drift_z > drift_sigma * 2:
                    drift_status = "critical"
                    status = "critical"  # override
                elif drift_z > drift_sigma:
                    drift_status = "warn"
                    if status == "healthy":
                        status = "warn"
                else:
                    drift_status = "healthy"

        # Out-of-range check
        oor_pct = None
        if reference_stats and name in reference_stats and len(valid) > 0:
            ref = reference_stats[name]
            below = float(np.mean(valid < ref.get("p5", -np.inf)))
            above = float(np.mean(valid > ref.get("p95", np.inf)))
            oor_pct = below + above

        report = {
            "nan_rate": round(nan_rate, 4),
            "live_mean": round(live_mean, 6) if not math.isnan(live_mean) else None,
            "live_std": round(live_std, 6),
            "n_samples": n_samples,
            "status": status,
        }
        if drift_z is not None:
            report["drift_z"] = round(drift_z, 4)
            report["drift_status"] = drift_status
        if oor_pct is not None:
            report["out_of_range_pct"] = round(oor_pct, 4)

        feature_reports[name] = report
        counts[status] += 1

    # Overall status
    if counts["critical"] > 0:
        overall = "critical"
    elif counts["warn"] > 0:
        overall = "warn"
    else:
        overall = "healthy"

    return {
        "overall_status": overall,
        "features": feature_reports,
        "summary": counts,
    }


def log_feature_health(report: Dict[str, Any]) -> None:
    """Emit structured log messages for feature health issues."""
    overall = report["overall_status"]
    summary = report["summary"]

    if overall == "healthy":
        logger.info(
            "Feature health: OK — all %d features within thresholds",
            summary["healthy"],
        )
        return

    log_fn = logger.critical if overall == "critical" else logger.warning
    log_fn(
        "Feature health: %s — %d healthy, %d warn, %d critical",
        overall.upper(),
        summary["healthy"],
        summary["warn"],
        summary["critical"],
    )

    for name, detail in report["features"].items():
        if detail["status"] == "critical":
            logger.critical(
                "  CRITICAL  %-30s  NaN=%.1f%%  drift_z=%s",
                name,
                detail["nan_rate"] * 100,
                detail.get("drift_z", "n/a"),
            )
        elif detail["status"] == "warn":
            logger.warning(
                "  WARN      %-30s  NaN=%.1f%%  drift_z=%s",
                name,
                detail["nan_rate"] * 100,
                detail.get("drift_z", "n/a"),
            )
