"""
Stock Radar - Evaluation Runner.

Runs your analysis pipeline against a test dataset and measures quality.

WHY THIS MATTERS (AI Engineering):
- This is your automated test suite for AI quality.
- Run it after every prompt change, model swap, or code update.
- If signal accuracy drops, you know something broke.

DATASET FORMAT (JSONL - one JSON per line):
    {"symbol":"AAPL","expected_signal":"buy","expected_score_range":[60,80],"mode":"intraday"}
    {"symbol":"TSLA","expected_signal":"hold","expected_score_range":[40,60],"mode":"intraday"}
    {"symbol":"MSFT","expected_signal":"strong_buy","expected_score_range":[75,95],"mode":"longterm"}

USAGE:
    python -m eval.runner data/eval_signals.jsonl --output eval_results.json

    # Or from code:
    from eval.runner import run_eval
    results = run_eval("data/eval_signals.jsonl")
"""

from __future__ import annotations

import json
import time
import argparse
import sys
import os
from pathlib import Path
from typing import Any

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eval.metrics import (
    signal_accuracy,
    signal_direction_accuracy,
    score_deviation,
    confidence_calibration,
    compute_percentiles,
)


def run_eval(
    dataset_path: str,
    output_path: str | None = None,
    mode_override: str | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """
    Run evaluation against a test dataset.

    This imports the analyzer lazily so the eval module can be used
    independently for testing metrics.

    Args:
        dataset_path: Path to JSONL dataset file.
        output_path: Optional path to write results JSON.
        mode_override: Override mode for all test cases.
        verbose: Print progress to stdout.

    Returns:
        Dict with summary metrics and per-example results.
    """
    # Lazy imports to avoid circular dependencies
    from agents.scorer import StockScorer
    from agents.fetcher import StockFetcher

    # Load dataset
    dataset: list[dict] = []
    with Path(dataset_path).open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                dataset.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"  Warning: Skipping line {line_num}: {e}")

    if not dataset:
        return {"error": "Empty dataset", "count": 0}

    if verbose:
        print(f"Loaded {len(dataset)} test cases from {dataset_path}")
        print("-" * 60)

    # Initialize components
    scorer = StockScorer()
    fetcher = StockFetcher()

    results: list[dict] = []
    start_time = time.time()

    for idx, test_case in enumerate(dataset, 1):
        symbol = test_case["symbol"]
        expected_signal = test_case.get("expected_signal", "hold")
        expected_score_range = test_case.get("expected_score_range", [0, 100])
        mode = mode_override or test_case.get("mode", "intraday")

        if verbose:
            print(f"[{idx}/{len(dataset)}] {symbol} (expected: {expected_signal})...", end=" ")

        case_start = time.time()

        try:
            # Fetch live data
            quote = fetcher.get_quote(symbol)
            if not quote:
                result = {
                    "symbol": symbol,
                    "status": "error",
                    "error": "No quote data",
                }
                results.append(result)
                if verbose:
                    print("SKIP (no data)")
                continue

            history = fetcher.get_price_history(symbol, period="3mo")
            indicators = fetcher.calculate_indicators(history)
            fundamentals = fetcher.get_fundamentals(symbol) if mode == "longterm" else {}

            # Run algorithmic scoring
            scores = scorer.calculate_all_scores(
                quote={
                    "price": quote.price,
                    "volume": quote.volume,
                    "avg_volume": quote.avg_volume,
                },
                indicators=indicators,
                fundamentals=fundamentals,
                price_history_days=len(history),
                has_news=False,
            )

            case_latency = time.time() - case_start

            # Compare results
            result = {
                "symbol": symbol,
                "status": "ok",
                "mode": mode,
                "expected_signal": expected_signal,
                "predicted_signal": scores.overall_signal,
                "signal_exact_match": signal_accuracy(scores.overall_signal, expected_signal),
                "signal_direction_match": signal_direction_accuracy(
                    scores.overall_signal, expected_signal
                ),
                "composite_score": scores.composite_score,
                "expected_score_range": expected_score_range,
                "score_in_range": (
                    1.0
                    if expected_score_range[0] <= scores.composite_score <= expected_score_range[1]
                    else 0.0
                ),
                "momentum_score": scores.momentum_score,
                "value_score": scores.value_score,
                "quality_score": scores.quality_score,
                "risk_score": scores.risk_score,
                "confidence_score": scores.confidence_score,
                "latency_sec": round(case_latency, 2),
            }
            results.append(result)

            status = "PASS" if result["signal_direction_match"] == 1.0 else "FAIL"
            if verbose:
                print(
                    f"{status} (predicted={scores.overall_signal}, "
                    f"score={scores.composite_score:.0f}, {case_latency:.1f}s)"
                )

        except Exception as e:
            case_latency = time.time() - case_start
            result = {
                "symbol": symbol,
                "status": "error",
                "error": str(e),
                "latency_sec": round(case_latency, 2),
            }
            results.append(result)
            if verbose:
                print(f"ERROR: {e}")

    total_elapsed = time.time() - start_time

    # Compute summary
    ok_results = [r for r in results if r["status"] == "ok"]
    error_results = [r for r in results if r["status"] == "error"]

    summary: dict[str, Any] = {
        "total_cases": len(dataset),
        "successful": len(ok_results),
        "errors": len(error_results),
    }

    if ok_results:
        summary.update({
            "signal_exact_accuracy": sum(r["signal_exact_match"] for r in ok_results) / len(ok_results),
            "signal_direction_accuracy": sum(r["signal_direction_match"] for r in ok_results) / len(ok_results),
            "score_in_range_pct": sum(r["score_in_range"] for r in ok_results) / len(ok_results),
            "avg_composite_score": sum(r["composite_score"] for r in ok_results) / len(ok_results),
            "latency": compute_percentiles([r["latency_sec"] for r in ok_results]),
            "confidence_calibration": confidence_calibration([
                {
                    "confidence": r["confidence_score"] / 100,
                    "predicted": r["predicted_signal"],
                    "expected": r["expected_signal"],
                }
                for r in ok_results
            ]),
        })

    summary["total_runtime_sec"] = round(total_elapsed, 2)

    payload = {"summary": summary, "results": results}

    # Write output
    if output_path:
        Path(output_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if verbose:
            print(f"\nResults written to {output_path}")

    # Print summary
    if verbose:
        print(f"\n{'=' * 60}")
        print("EVALUATION SUMMARY")
        print(f"{'=' * 60}")
        print(f"Total: {summary['total_cases']} | Success: {summary['successful']} | Errors: {summary['errors']}")
        if ok_results:
            print(f"Signal Exact Accuracy:     {summary['signal_exact_accuracy']:.1%}")
            print(f"Signal Direction Accuracy:  {summary['signal_direction_accuracy']:.1%}")
            print(f"Score In Expected Range:    {summary['score_in_range_pct']:.1%}")
            print(f"Avg Composite Score:        {summary['avg_composite_score']:.1f}")
            lat = summary["latency"]
            print(f"Latency: p50={lat['p50']:.1f}s, p95={lat['p95']:.1f}s")
        print(f"Total Runtime: {summary['total_runtime_sec']:.1f}s")

    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stock Radar evaluation")
    parser.add_argument("dataset", help="Path to JSONL evaluation dataset")
    parser.add_argument("--output", "-o", help="Output JSON file path")
    parser.add_argument("--mode", choices=["intraday", "longterm"], help="Override mode")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress output")
    args = parser.parse_args()

    run_eval(
        dataset_path=args.dataset,
        output_path=args.output,
        mode_override=args.mode,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
