"""
Training script for signal classifier.

Loads CSV dataset, performs purged walk-forward split, runs Optuna hyperparameter
search over GradientBoostingClassifier, and saves the best model + metadata.

Phase-5 upgrade:
  - Supports 37-feature vector (20 base + 17 cross-sectional/factor/micro)
  - Adds Information Coefficient (IC) and rank IC diagnostics
  - Feature stability report
  - Feature group importance breakdown

Usage:
    python -m training.train --dataset data/training_data.csv --n-trials 30 --output models/
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.feature_engineering import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    SIGNAL_LABELS,
    extract_features_batch,
)
from training.cross_sectional import (
    CROSS_SECTIONAL_FEATURE_NAMES,
    FACTOR_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_NAMES,
)
from training.sentiment import SENTIMENT_FEATURE_NAMES
from training.backtesting import evaluate_predictions

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# ---------------------------------------------------------------------------
#  Feature group boundaries for importance breakdown
# ---------------------------------------------------------------------------
FEATURE_GROUPS = {
    "base_technical": BASE_FEATURE_NAMES[:8],
    "base_volatility": BASE_FEATURE_NAMES[8:12],
    "base_valuation": BASE_FEATURE_NAMES[12:16],
    "base_quality": BASE_FEATURE_NAMES[16:20],
    "cross_sectional": list(CROSS_SECTIONAL_FEATURE_NAMES),
    "factor_style": list(FACTOR_FEATURE_NAMES),
    "microstructure": list(MICROSTRUCTURE_FEATURE_NAMES),
    "sentiment": list(SENTIMENT_FEATURE_NAMES),
}


def load_csv_dataset(path: str) -> list[dict]:
    """Load training CSV into list of row dicts."""
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def _compute_ic_metrics(
    X: np.ndarray,
    y_true_returns: np.ndarray,
    feature_names: list[str],
) -> dict[str, dict[str, float]]:
    """
    Compute Information Coefficient (IC) and rank IC for each feature
    against forward returns.

    IC = Pearson correlation  (feature_i, forward_return)
    Rank IC = Spearman correlation
    """
    from scipy import stats

    results: dict[str, dict[str, float]] = {}
    n = len(y_true_returns)
    if n < 10:
        return results

    valid_mask = ~np.isnan(y_true_returns)

    for j, fname in enumerate(feature_names):
        col = X[:, j]
        # Rows where both feature and return are valid
        mask = valid_mask & ~np.isnan(col)
        if mask.sum() < 10:
            results[fname] = {"ic": 0.0, "rank_ic": 0.0, "coverage": 0.0}
            continue

        feat_vals = col[mask]
        ret_vals = y_true_returns[mask]

        # Pearson IC
        ic, _ = stats.pearsonr(feat_vals, ret_vals)
        # Spearman Rank IC
        rank_ic, _ = stats.spearmanr(feat_vals, ret_vals)

        results[fname] = {
            "ic": round(float(ic), 6),
            "rank_ic": round(float(rank_ic), 6),
            "coverage": round(float(mask.sum()) / n, 4),
        }

    return results


def _feature_group_importance(
    importances: dict[str, float],
) -> dict[str, float]:
    """Aggregate feature importances by group."""
    group_imp: dict[str, float] = {}
    for group_name, group_features in FEATURE_GROUPS.items():
        total = sum(importances.get(f, 0.0) for f in group_features)
        group_imp[group_name] = round(total, 4)
    return group_imp


def _purged_walk_forward_split(
    rows: list[dict],
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    embargo_days: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[dict]]:
    """
    Purged walk-forward split: train on earlier data, test on later data,
    with an embargo gap to prevent leakage from overlapping labels.

    The embargo is measured in **calendar days**, not row count.
    This matters when the dataset has multiple symbols per date:
    30 symbols × 1 date = 30 rows, so a row-offset of 5 would only
    skip 1/6th of a single day rather than 5 calendar days.

    Returns:
        X_train, X_test, y_train, y_test, test_rows
    """
    from datetime import datetime, timedelta

    n = len(rows)
    # Sort indices by timestamp
    sorted_indices = sorted(range(n), key=lambda i: rows[i].get("timestamp", ""))

    # ---- Determine split date and embargo cutoff date ----
    split_point = int(n * (1.0 - test_size))
    # last index that goes into training
    last_train_idx = sorted_indices[min(split_point - 1, n - 1)]
    last_train_ts = rows[last_train_idx].get("timestamp", "")

    # Parse the split-boundary date
    try:
        split_date = datetime.strptime(last_train_ts[:10], "%Y-%m-%d")
    except (ValueError, TypeError):
        # Fallback: if dates are unparseable, use row-based embargo
        split_date = None

    if split_date is not None:
        embargo_cutoff = split_date + timedelta(days=embargo_days)
        embargo_cutoff_str = embargo_cutoff.strftime("%Y-%m-%d")

        train_idx = []
        test_idx = []

        for i in sorted_indices:
            ts = rows[i].get("timestamp", "")[:10]
            if ts <= last_train_ts[:10]:
                train_idx.append(i)
            elif ts > embargo_cutoff_str:
                test_idx.append(i)
            # else: row falls in the embargo zone → discard

        train_idx = np.array(train_idx) if train_idx else np.array([], dtype=int)
        test_idx = np.array(test_idx) if test_idx else np.array([], dtype=int)
    else:
        # Unparseable timestamps: fall back to row-based (legacy)
        embargo_end = min(split_point + embargo_days, n)
        train_idx = np.array(sorted_indices[:split_point])
        test_idx = np.array(sorted_indices[embargo_end:])

    if len(test_idx) == 0:
        # Fallback if embargo eats all test data
        test_idx = np.array(sorted_indices[split_point:])

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = y[train_idx]
    y_test = y[test_idx]
    test_rows = [rows[i] for i in test_idx]

    return X_train, X_test, y_train, y_test, test_rows


def train_model(
    dataset_path: str,
    output_dir: str = "models/",
    n_trials: int = 30,
    test_size: float = 0.2,
    random_state: int = 42,
    transaction_cost_bps: float = 10.0,
    use_walk_forward: bool = True,
    regime_filter: str | None = None,
) -> dict:
    """
    Train signal classifier with Optuna hyperparameter optimization.

    Phase-5 additions:
      - Purged walk-forward split (default) instead of random split
      - IC and rank-IC diagnostics
      - Feature group importance breakdown
      - Extended metadata

    Returns:
        Dict with model path, metrics, and metadata.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import cross_val_score, train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    import joblib
    import optuna

    # Suppress Optuna info logs
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Load data
    rows = load_csv_dataset(dataset_path)
    if len(rows) < 10:
        raise ValueError(f"Dataset too small ({len(rows)} rows). Need at least 10.")

    logger.info(f"Loaded {len(rows)} samples from {dataset_path}")

    # Regime filtering: classify each row and keep only matching regime
    if regime_filter:
        from training.regime import classify_market_regime

        filtered = []
        for row in rows:
            row_indicators = {k: row.get(k) for k in row}
            regime_info = classify_market_regime(row_indicators)
            if regime_info["regime"] == regime_filter:
                filtered.append(row)
        logger.info(f"Regime filter '{regime_filter}': {len(filtered)}/{len(rows)} rows match")
        rows = filtered
        if len(rows) < 10:
            raise ValueError(f"Too few rows ({len(rows)}) after regime filter '{regime_filter}'.")
    logger.info(f"Feature vector size: {len(FEATURE_NAMES)} ({len(BASE_FEATURE_NAMES)} base + "
                f"{len(CROSS_SECTIONAL_FEATURE_NAMES)} XS + {len(FACTOR_FEATURE_NAMES)} factor + "
                f"{len(MICROSTRUCTURE_FEATURE_NAMES)} micro)")

    X, y = extract_features_batch(rows)
    logger.info(f"Features shape: {X.shape}, Labels distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    # --- NaN coverage report ---
    nan_counts = np.isnan(X).sum(axis=0)
    nan_report = {
        fname: f"{nan_counts[i]}/{len(X)} ({nan_counts[i]/len(X)*100:.1f}%)"
        for i, fname in enumerate(FEATURE_NAMES) if nan_counts[i] > 0
    }
    if nan_report:
        logger.info(f"Features with NaN: {len(nan_report)} of {len(FEATURE_NAMES)}")
        for name, count_str in sorted(nan_report.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.info(f"  {name}: {count_str}")

    # --- Split ---
    if use_walk_forward:
        logger.info("Using purged walk-forward split with embargo")
        X_train, X_test, y_train, y_test, test_rows = _purged_walk_forward_split(
            rows, X, y, test_size=test_size, embargo_days=5,
        )
    else:
        indices = np.arange(len(rows))
        train_idx, test_idx = train_test_split(
            indices, test_size=test_size, random_state=random_state, stratify=y
        )
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_rows = [rows[i] for i in test_idx]

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Optuna objective
    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        }

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(random_state=random_state, **params)),
        ])

        # TimeSeriesSplit preserves temporal order inside the tuning loop,
        # preventing look-ahead bias that StratifiedKFold(shuffle=True) caused.
        from sklearn.model_selection import TimeSeriesSplit
        n_splits = min(5, max(2, len(X_train) // 50))
        cv = TimeSeriesSplit(n_splits=n_splits)
        scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy")
        return scores.mean()

    # Run optimization
    logger.info(f"Running Optuna optimization with {n_trials} trials...")
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    logger.info(f"Best params: {best_params}")
    logger.info(f"Best CV accuracy: {study.best_value:.4f}")

    # Train final model with best params
    final_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(random_state=random_state, **best_params)),
    ])

    final_pipeline.fit(X_train, y_train)

    # --- Phase-9: Confidence calibration ---
    from sklearn.calibration import CalibratedClassifierCV
    from training.calibration import compute_calibration_metrics

    calibrated_pipeline = None
    calibration_metrics = {}
    try:
        calibrated_pipeline = CalibratedClassifierCV(
            final_pipeline, method="isotonic", cv=5,
        )
        calibrated_pipeline.fit(X_train, y_train)
        cal_prob = calibrated_pipeline.predict_proba(X_test)
        calibration_metrics = compute_calibration_metrics(y_test, cal_prob)
        logger.info(
            f"Calibration: ECE={calibration_metrics['ece']:.4f}, "
            f"Brier={calibration_metrics['brier_score']:.4f}"
        )
    except Exception as e:
        logger.warning(f"Calibration failed (model will use raw proba): {e}")
        calibrated_pipeline = None

    # Evaluate on test set — use calibrated model when available
    eval_pipeline = calibrated_pipeline if calibrated_pipeline is not None else final_pipeline
    y_pred = eval_pipeline.predict(X_test)
    y_prob = eval_pipeline.predict_proba(X_test)
    y_conf = np.max(y_prob, axis=1)
    test_accuracy = (y_pred == y_test).mean()

    report = classification_report(
        y_test,
        y_pred,
        labels=list(range(len(SIGNAL_LABELS))),
        target_names=SIGNAL_LABELS,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(SIGNAL_LABELS)))).tolist()

    test_horizon = 5
    try:
        horizon_values = [int(float(r.get("future_horizon_days"))) for r in rows if r.get("future_horizon_days")]
        if horizon_values:
            test_horizon = int(np.median(horizon_values))
    except (TypeError, ValueError):
        test_horizon = 5

    backtest_metrics = evaluate_predictions(
        test_rows,
        y_pred,
        horizon_days=test_horizon,
        transaction_cost_bps=transaction_cost_bps,
        allow_short=True,
    )
    position_sized_backtest = evaluate_predictions(
        test_rows,
        y_pred,
        horizon_days=test_horizon,
        transaction_cost_bps=transaction_cost_bps,
        allow_short=True,
        prediction_confidences=y_conf.tolist(),
        use_position_sizing=True,
    )

    # Portfolio-level backtest
    portfolio_backtest_metrics = {}
    try:
        from training.portfolio_backtest import run_portfolio_backtest
        portfolio_backtest_metrics = run_portfolio_backtest(
            test_rows,
            y_pred,
            prediction_confidences=y_conf.tolist(),
        )
        if portfolio_backtest_metrics.get("num_periods", 0) > 0:
            logger.info(
                "Portfolio backtest: "
                f"Sharpe={portfolio_backtest_metrics['sharpe']:.3f}, "
                f"CAGR={portfolio_backtest_metrics['cagr']:.3%}, "
                f"MaxDD={portfolio_backtest_metrics['max_drawdown']:.3%}, "
                f"Violations={portfolio_backtest_metrics['portfolio_constraint_violations']}"
            )
    except Exception as e:
        logger.warning(f"Portfolio backtest skipped: {e}")

    # Feature importances from the GradientBoosting estimator
    clf = final_pipeline.named_steps["clf"]
    importances = dict(zip(FEATURE_NAMES, [round(float(x), 4) for x in clf.feature_importances_]))

    # --- Phase-5: feature group importance breakdown ---
    group_importance = _feature_group_importance(importances)

    # --- Phase-5: IC and rank-IC diagnostics ---
    forward_returns = np.array([
        float(r.get("future_return_pct", "nan")) if r.get("future_return_pct") not in (None, "", "nan") else np.nan
        for r in test_rows
    ], dtype=np.float64)

    ic_metrics = {}
    try:
        ic_metrics = _compute_ic_metrics(X_test, forward_returns, FEATURE_NAMES)
    except Exception as e:
        logger.warning(f"IC computation failed: {e}")

    logger.info(f"Test accuracy: {test_accuracy:.4f}")
    logger.info(
        "Classification report:\n"
        + classification_report(
            y_test,
            y_pred,
            labels=list(range(len(SIGNAL_LABELS))),
            target_names=SIGNAL_LABELS,
            zero_division=0,
        )
    )

    # --- Feature group importance report ---
    logger.info("Feature Group Importance:")
    for grp, imp in sorted(group_importance.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {grp}: {imp:.4f}")

    # --- Top IC features ---
    if ic_metrics:
        sorted_ic = sorted(ic_metrics.items(), key=lambda x: abs(x[1].get("rank_ic", 0)), reverse=True)
        logger.info("Top features by |Rank IC|:")
        for name, metrics in sorted_ic[:10]:
            logger.info(f"  {name}: IC={metrics['ic']:.4f}, Rank IC={metrics['rank_ic']:.4f}, "
                       f"Coverage={metrics['coverage']:.1%}")

    # --- Reference stats for ALL features (used by predictor health check) ---
    from training.feature_health import compute_reference_stats
    feature_reference_stats = compute_reference_stats(X_train, list(FEATURE_NAMES))

    # --- Sentiment feature coverage (specific reporting) ---
    sentiment_coverage = {}
    for fname in SENTIMENT_FEATURE_NAMES:
        idx = FEATURE_NAMES.index(fname)
        col = X_test[:, idx]
        cov = float(np.mean(~np.isnan(col)))
        sentiment_coverage[fname] = round(cov, 4)

    avg_sentiment_cov = float(np.mean(list(sentiment_coverage.values())))
    logger.info(f"Sentiment feature coverage (avg): {avg_sentiment_cov:.1%}")
    for fname, cov in sentiment_coverage.items():
        logger.info(f"  {fname}: {cov:.1%}")
    if avg_sentiment_cov < 0.10:
        logger.warning("All sentiment features have <10% coverage — consider using --sentiment in dataset build")

    if backtest_metrics.get("num_samples", 0) > 0:
        logger.info(
            "Backtest (cost-aware): "
            f"Sharpe={backtest_metrics['sharpe']:.3f}, "
            f"CAGR={backtest_metrics['cagr']:.3%}, "
            f"MaxDD={backtest_metrics['max_drawdown']:.3%}, "
            f"WinRate={backtest_metrics['win_rate']:.2%}, "
            f"Turnover={backtest_metrics['turnover']:.3f}"
        )
    if position_sized_backtest.get("num_samples", 0) > 0:
        logger.info(
            "Backtest (position-sized): "
            f"Sharpe={position_sized_backtest['sharpe']:.3f}, "
            f"CAGR={position_sized_backtest['cagr']:.3%}, "
            f"MaxDD={position_sized_backtest['max_drawdown']:.3%}, "
            f"WinRate={position_sized_backtest['win_rate']:.2%}, "
            f"AvgPos={position_sized_backtest['avg_abs_position']:.3f}"
        )

    # Determine version
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    regime_suffix = f"_{regime_filter}" if regime_filter else ""
    existing = list(output.glob(f"signal_classifier_v*{regime_suffix}.joblib"))
    version = len(existing) + 1

    model_path = output / f"signal_classifier_v{version}{regime_suffix}.joblib"
    meta_path = output / f"signal_classifier_v{version}{regime_suffix}_meta.json"

    # Save model
    joblib.dump(final_pipeline, model_path)
    logger.info(f"Model saved to {model_path}")

    # Save calibrated model
    calibrated_model_path = None
    if calibrated_pipeline is not None:
        cal_path = output / f"signal_classifier_v{version}{regime_suffix}_calibrated.joblib"
        joblib.dump(calibrated_pipeline, cal_path)
        calibrated_model_path = str(cal_path)
        logger.info(f"Calibrated model saved to {cal_path}")

    # Save metadata
    metadata = {
        "version": version,
        "model_path": str(model_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset_path": dataset_path,
        "dataset_size": len(rows),
        "train_size": len(X_train),
        "test_size": len(X_test),
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "feature_groups": {k: len(v) for k, v in FEATURE_GROUPS.items()},
        "signal_labels": SIGNAL_LABELS,
        "best_params": best_params,
        "cv_accuracy": round(study.best_value, 4),
        "test_accuracy": round(test_accuracy, 4),
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "feature_group_importance": group_importance,
        "ic_metrics": ic_metrics,
        "n_trials": n_trials,
        "transaction_cost_bps": transaction_cost_bps,
        "regime_filter": regime_filter,
        "split_method": "purged_walk_forward" if use_walk_forward else "stratified_random",
        "backtest_metrics": backtest_metrics,
        "position_sized_backtest_metrics": position_sized_backtest,
        "portfolio_backtest_metrics": portfolio_backtest_metrics,
        "sentiment_feature_coverage": sentiment_coverage,
        "feature_reference_stats": feature_reference_stats,
        "calibrated_model_path": calibrated_model_path,
        "calibration_metrics": calibration_metrics,
    }

    meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    logger.info(f"Metadata saved to {meta_path}")

    # Print top features
    sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
    logger.info("Top features:")
    for name, imp in sorted_imp[:10]:
        logger.info(f"  {name}: {imp:.4f}")

    # --- A/B backtest comparison with incumbent ---
    try:
        from training.model_registry import (
            register_model,
            promote_if_better,
        )

        # Register candidate model
        register_model(
            version=version,
            model_path=str(model_path),
            metrics=metadata,
        )

        # promote_if_better is the single authority: it compares against the
        # incumbent on all gates (Sharpe, CAGR, MaxDD, turnover, net-vs-gross)
        # and promotes only if all pass.  No separate compare_backtests call.
        promo = promote_if_better(version)

        logger.info("Promotion gate results:")
        for gate_name, gate_detail in promo.get("checks", {}).items():
            status = "PASS" if gate_detail.get("passed") else "FAIL"
            logger.info(f"  [{status}] {gate_name}: {gate_detail}")
        logger.info(f"Promotion result: {promo['reason']}")

        metadata["promotion"] = promo

        # Re-save metadata with promotion info
        meta_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    except Exception as e:
        logger.warning(f"Model registry A/B comparison skipped: {e}")

    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Train signal classifier")
    parser.add_argument("--dataset", required=True, help="Path to training CSV")
    parser.add_argument("--output", "-o", default="models/", help="Output directory for model")
    parser.add_argument("--n-trials", type=int, default=30, help="Optuna trials (default: 30)")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split ratio (default: 0.2)")
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Transaction cost in basis points per unit turnover (default: 10)",
    )
    parser.add_argument(
        "--no-walk-forward",
        action="store_true",
        help="Use stratified random split instead of purged walk-forward",
    )
    parser.add_argument(
        "--regime",
        choices=["trending", "mean_reverting", "high_volatility"],
        default=None,
        help="Train regime-specific model (filters dataset to matching regime)",
    )
    args = parser.parse_args()

    metadata = train_model(
        dataset_path=args.dataset,
        output_dir=args.output,
        n_trials=args.n_trials,
        test_size=args.test_size,
        transaction_cost_bps=args.cost_bps,
        use_walk_forward=not args.no_walk_forward,
        regime_filter=args.regime,
    )

    print("\nTraining complete!")
    print(f"  Model: {metadata['model_path']}")
    print(f"  Features: {metadata['feature_count']} ({len(BASE_FEATURE_NAMES)} base + "
          f"{len(CROSS_SECTIONAL_FEATURE_NAMES) + len(FACTOR_FEATURE_NAMES) + len(MICROSTRUCTURE_FEATURE_NAMES)} Phase-5 + "
          f"{len(SENTIMENT_FEATURE_NAMES)} sentiment)")
    print(f"  Split: {metadata['split_method']}")
    print(f"  CV Accuracy: {metadata['cv_accuracy']:.1%}")
    print(f"  Test Accuracy: {metadata['test_accuracy']:.1%}")

    # Feature group importance
    if metadata.get("feature_group_importance"):
        print("\n  Feature Group Importance:")
        for grp, imp in sorted(metadata["feature_group_importance"].items(), key=lambda x: x[1], reverse=True):
            print(f"    {grp}: {imp:.4f}")

    if metadata.get("backtest_metrics", {}).get("num_samples", 0) > 0:
        bt = metadata["backtest_metrics"]
        print(
            "  Backtest: "
            f"Sharpe={bt['sharpe']:.3f}, "
            f"CAGR={bt['cagr']:.2%}, "
            f"MaxDD={bt['max_drawdown']:.2%}, "
            f"WinRate={bt['win_rate']:.2%}"
        )
    if metadata.get("position_sized_backtest_metrics", {}).get("num_samples", 0) > 0:
        bt = metadata["position_sized_backtest_metrics"]
        print(
            "  Backtest (sized): "
            f"Sharpe={bt['sharpe']:.3f}, "
            f"CAGR={bt['cagr']:.2%}, "
            f"MaxDD={bt['max_drawdown']:.2%}, "
            f"AvgPos={bt['avg_abs_position']:.2f}"
        )

    # Top IC features
    ic = metadata.get("ic_metrics", {})
    if ic:
        sorted_ic = sorted(ic.items(), key=lambda x: abs(x[1].get("rank_ic", 0)), reverse=True)
        print("\n  Top 5 features by |Rank IC|:")
        for name, m in sorted_ic[:5]:
            print(f"    {name}: IC={m['ic']:.4f}, Rank IC={m['rank_ic']:.4f}")


if __name__ == "__main__":
    main()
