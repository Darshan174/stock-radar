"""
Stock Radar - Evaluation Metrics.

Quantitative metrics to measure the quality of AI analysis outputs.

WHY THIS MATTERS (AI Engineering):
- "How do you know your AI system is working correctly?" is THE interview question.
- Without metrics, you're flying blind - you can't improve what you can't measure.
- These metrics compare your system's outputs against known good answers.

METRICS:
    signal_accuracy     - Did the AI predict the right signal? (exact match)
    signal_direction    - Did the AI get the direction right? (buy vs sell)
    confidence_calibration - Is confidence correlated with accuracy?
    score_deviation     - How far off are algorithmic scores?
    latency_p50/p95     - Performance percentiles
"""

from __future__ import annotations

from collections import Counter


def signal_accuracy(predicted: str, expected: str) -> float:
    """
    Exact match of trading signals.

    Args:
        predicted: The signal from the AI (e.g., "buy")
        expected: The correct signal (e.g., "buy")

    Returns:
        1.0 if exact match, 0.0 otherwise
    """
    return 1.0 if predicted.lower().strip() == expected.lower().strip() else 0.0


def signal_direction_accuracy(predicted: str, expected: str) -> float:
    """
    Check if the AI got the direction right (bullish vs bearish).

    "strong_buy" and "buy" are both bullish.
    "strong_sell" and "sell" are both bearish.
    "hold" matches "hold".

    Returns:
        1.0 if direction matches, 0.0 otherwise.
    """
    bullish = {"strong_buy", "buy"}
    bearish = {"strong_sell", "sell"}
    neutral = {"hold"}

    pred = predicted.lower().strip()
    exp = expected.lower().strip()

    if pred in bullish and exp in bullish:
        return 1.0
    if pred in bearish and exp in bearish:
        return 1.0
    if pred in neutral and exp in neutral:
        return 1.0

    return 0.0


def score_deviation(predicted_score: float, expected_score: float) -> float:
    """
    Absolute deviation between predicted and expected scores.

    Lower is better. 0 = perfect match.

    Args:
        predicted_score: Score from the system (0-100)
        expected_score: Expected score (0-100)

    Returns:
        Absolute deviation (0-100)
    """
    return abs(predicted_score - expected_score)


def confidence_calibration(
    predictions: list[dict],
) -> dict[str, float]:
    """
    Check if confidence scores are well-calibrated.

    Groups predictions by confidence bucket and checks actual accuracy.
    A well-calibrated system has 80% accuracy on 80%-confidence predictions.

    Args:
        predictions: List of dicts with "confidence", "predicted", "expected"

    Returns:
        Dict mapping confidence bucket to actual accuracy
    """
    buckets: dict[str, list[bool]] = {
        "0.0-0.3": [],
        "0.3-0.5": [],
        "0.5-0.7": [],
        "0.7-0.9": [],
        "0.9-1.0": [],
    }

    for pred in predictions:
        conf = pred.get("confidence", 0.5)
        correct = pred.get("predicted", "").lower() == pred.get("expected", "").lower()

        if conf < 0.3:
            buckets["0.0-0.3"].append(correct)
        elif conf < 0.5:
            buckets["0.3-0.5"].append(correct)
        elif conf < 0.7:
            buckets["0.5-0.7"].append(correct)
        elif conf < 0.9:
            buckets["0.7-0.9"].append(correct)
        else:
            buckets["0.9-1.0"].append(correct)

    result = {}
    for bucket, outcomes in buckets.items():
        if outcomes:
            result[bucket] = sum(outcomes) / len(outcomes)
        else:
            result[bucket] = None

    return result


def compute_percentiles(values: list[float]) -> dict[str, float]:
    """
    Compute p50, p95, p99 for a list of values (e.g., latencies).

    Args:
        values: List of numeric values

    Returns:
        Dict with min, p50, p95, p99, max
    """
    if not values:
        return {"min": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}

    sorted_vals = sorted(values)
    n = len(sorted_vals)

    return {
        "min": sorted_vals[0],
        "p50": sorted_vals[int(n * 0.50)],
        "p95": sorted_vals[min(int(n * 0.95), n - 1)],
        "p99": sorted_vals[min(int(n * 0.99), n - 1)],
        "max": sorted_vals[-1],
        "count": n,
    }
