"""Confidence calibration metrics for signal classifier.

Computes Expected Calibration Error (ECE) and Brier score to assess
whether predicted probabilities match observed frequencies.
"""

from __future__ import annotations

import numpy as np


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error — weighted |accuracy - confidence| per bin.

    For multi-class, uses the predicted class probability as the confidence.

    Args:
        y_true: shape (N,) integer labels
        y_prob: shape (N, C) predicted probabilities
        n_bins: number of equal-width bins

    Returns:
        ECE in [0, 1].
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    if y_prob.ndim == 1:
        confidences = y_prob
        predicted = (y_prob >= 0.5).astype(int)
    else:
        confidences = np.max(y_prob, axis=1)
        predicted = np.argmax(y_prob, axis=1)

    accuracies = (predicted == y_true).astype(float)
    n = len(y_true)
    if n == 0:
        return 0.0

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > lo) & (confidences <= hi) if i > 0 else (confidences >= lo) & (confidences <= hi)
        count = mask.sum()
        if count == 0:
            continue
        avg_confidence = confidences[mask].mean()
        avg_accuracy = accuracies[mask].mean()
        ece += (count / n) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Multi-class Brier score — (1/N) * sum (p_ic - y_ic)^2.

    Args:
        y_true: shape (N,) integer labels
        y_prob: shape (N, C) predicted probabilities

    Returns:
        Brier score >= 0. Lower is better. 0 = perfect.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    n = len(y_true)
    if n == 0:
        return 0.0

    if y_prob.ndim == 1:
        # Binary case: treat as 2-class
        y_prob = np.column_stack([1.0 - y_prob, y_prob])

    n_classes = y_prob.shape[1]
    y_onehot = np.zeros((n, n_classes), dtype=float)
    for i in range(n):
        if 0 <= int(y_true[i]) < n_classes:
            y_onehot[i, int(y_true[i])] = 1.0

    return float(np.mean(np.sum((y_prob - y_onehot) ** 2, axis=1)))


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """Compute calibration metrics.

    Returns:
        {ece, brier_score, mean_confidence, mean_accuracy, n_bins}
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    ece = compute_ece(y_true, y_prob, n_bins=n_bins)
    brier = compute_brier_score(y_true, y_prob)

    if y_prob.ndim == 1:
        mean_conf = float(y_prob.mean()) if len(y_prob) > 0 else 0.0
        predicted = (y_prob >= 0.5).astype(int)
    else:
        mean_conf = float(np.max(y_prob, axis=1).mean()) if len(y_prob) > 0 else 0.0
        predicted = np.argmax(y_prob, axis=1)

    mean_acc = float((predicted == y_true).mean()) if len(y_true) > 0 else 0.0

    return {
        "ece": round(ece, 6),
        "brier_score": round(brier, 6),
        "mean_confidence": round(mean_conf, 6),
        "mean_accuracy": round(mean_acc, 6),
        "n_bins": n_bins,
    }
