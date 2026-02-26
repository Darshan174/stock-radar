"""
Phase-5 integration tests – fills three identified gaps:

1. Loading an old 20-feature model in SignalPredictor still works.
2. End-to-end analyzer → predictor path ensures Phase-5 features
   are present (or gracefully NaN) at inference time.
3. _purged_walk_forward_split embargo is date-based, not row-based.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.feature_engineering import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    SIGNAL_LABELS,
    extract_features,
)



# ─────────────────────────────────────────────────────────────
#  Gap 1 – Old 20-feature model backward compatibility
# ─────────────────────────────────────────────────────────────

class TestOldModelBackwardCompat:
    """
    A model trained on 20 features must still load and predict
    through SignalPredictor even though extract_features now emits
    a 37-element vector.  The pipeline's SimpleImputer fills the
    17 extra NaN slots with their training-median (which is arbitrary
    for the old model, but must not crash).
    """

    @pytest.fixture
    def old_model_path(self, tmp_path):
        """Build and persist a 20-feature model (pre-Phase-5)."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib

        rng = np.random.RandomState(99)
        X_train = rng.randn(80, 20)                      # 20 features only
        y_train = rng.randint(0, 5, size=80)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=99)),
        ])
        pipeline.fit(X_train, y_train)

        model_file = tmp_path / "old_v1_model.joblib"
        joblib.dump(pipeline, model_file)

        # Write a v1-style metadata file (feature_count=20)
        meta = {
            "version": 1,
            "feature_count": 20,
            "feature_names": BASE_FEATURE_NAMES,
        }
        meta_file = tmp_path / "old_v1_model_meta.json"
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        return str(model_file)

    def test_old_model_loads_without_error(self, old_model_path):
        from training.predictor import SignalPredictor

        predictor = SignalPredictor(old_model_path)
        assert predictor.feature_count == 20
        assert predictor.model_version == 1

    def test_old_model_predict_with_37_feature_vector(self, old_model_path):
        """
        extract_features returns 37 values.  The old model was
        trained on 20, so the extra 17 columns must be imputed
        (not crash) when passed to pipeline.predict().
        """
        from training.predictor import SignalPredictor

        predictor = SignalPredictor(old_model_path)

        # NOTE: SignalPredictor.predict() calls extract_features internally,
        # which ALWAYS returns 37 features now.  The old pipeline has an
        # imputer → scaler → clf that only saw 20 columns at fit-time.
        # sklearn will raise ValueError: X has 37 features, but was fitted
        # with 20 features.
        #
        # This test intentionally CAPTURES that gap.
        # If the code already handles this (e.g. by slicing to
        # metadata['feature_count']), the predict should succeed.
        # If it doesn't handle it, the test exposes the missing logic.

        result = predictor.predict(
            indicators={"rsi_14": 50, "macd": 0.5, "adx": 20},
            fundamentals={"pe_ratio": 15},
            quote={"price": 100, "volume": 1000, "avg_volume": 900},
        )

        assert result["signal"] in SIGNAL_LABELS
        assert 0 <= result["confidence"] <= 1
        assert "probabilities" in result

    def test_old_model_predict_with_price_series(self, old_model_path):
        """Even with Phase-5 price series supplied, old model survives."""
        from training.predictor import SignalPredictor

        predictor = SignalPredictor(old_model_path)
        closes = list(np.linspace(95, 105, 300))
        highs = [c + 1.0 for c in closes]
        lows = [c - 1.0 for c in closes]
        volumes = [1_000_000.0] * 300

        result = predictor.predict(
            indicators={"rsi_14": 55, "macd": 1.0},
            fundamentals={"pe_ratio": 18, "roe": 0.15},
            quote={"price": 105, "volume": 1_200_000, "avg_volume": 1_000_000},
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
        )

        assert result["signal"] in SIGNAL_LABELS
        # Phase-5 factor features should be present in the result
        assert "factor_features" in result
        assert "momentum_12_1" in result["factor_features"]


# ─────────────────────────────────────────────────────────────
#  Gap 2 – End-to-end analyzer → predictor Phase-5 path
# ─────────────────────────────────────────────────────────────

class TestAnalyzerPredictorE2E:
    """
    The analyzer's generate_algo_prediction() calls
    predictor.predict(indicators, fundamentals, quote) WITHOUT
    passing closes/highs/lows/volumes.

    This means Phase-5 factor and microstructure features are always
    NaN at inference through the analyzer path.
    """

    @pytest.fixture
    def v2_model_path(self, tmp_path):
        """37-feature model (Phase-5)."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib

        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 37)
        y_train = rng.randint(0, 5, size=100)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        model_file = tmp_path / "signal_classifier_v2.joblib"
        joblib.dump(pipeline, model_file)

        meta = {
            "version": 2,
            "feature_count": 37,
            "feature_names": FEATURE_NAMES,
        }
        meta_file = tmp_path / "signal_classifier_v2_meta.json"
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
        return str(model_file)

    def test_predictor_without_price_series_fills_phase5_with_nan(self, v2_model_path):
        """
        When called the way analyzer.py calls it (no closes/highs/lows/volumes),
        factor and microstructure slots should be NaN, and the pipeline's
        imputer should fill them without error.
        """
        from training.predictor import SignalPredictor

        predictor = SignalPredictor(v2_model_path)

        result = predictor.predict(
            indicators={"rsi_14": 60, "macd": 2.0, "macd_signal": 1.5},
            fundamentals={"pe_ratio": 22, "pb_ratio": 3.0},
            quote={"price": 150, "volume": 2_000_000, "avg_volume": 1_500_000},
        )

        assert result["signal"] in SIGNAL_LABELS
        assert 0 <= result["confidence"] <= 1
        # Without price series, no factor_features key
        assert "factor_features" not in result

    def test_feature_vector_has_nan_for_phase5_when_no_series(self):
        """
        Directly verify extract_features output: slots 20-44 should
        all be NaN when no Phase-5/6 dicts are supplied.
        """
        features = extract_features(
            indicators={"rsi_14": 50, "macd": 0.5},
            fundamentals={"pe_ratio": 20},
            quote={"price": 100, "volume": 1000, "avg_volume": 800},
        )
        assert features.shape == (45,)
        # Base features should NOT all be NaN
        base_nan_count = np.isnan(features[:20]).sum()
        assert base_nan_count < 20  # at least rsi_14, macd, pe_ratio are set

        # Phase-5/6 features SHOULD all be NaN
        assert np.isnan(features[20:]).all(), (
            f"Expected all Phase-5/6 slots (20-44) to be NaN, "
            f"but got non-NaN at positions: "
            f"{[i + 20 for i, v in enumerate(features[20:]) if not np.isnan(v)]}"
        )

    def test_predictor_with_price_series_populates_phase5(self, v2_model_path):
        """
        When closes/highs/lows/volumes ARE supplied, factor and micro
        features should be computed and present in the result.
        """
        from training.predictor import SignalPredictor

        predictor = SignalPredictor(v2_model_path)
        np.random.seed(42)
        closes = list(np.cumsum(np.random.randn(300) * 0.02) + 100)
        highs = [c + np.random.rand() * 2 for c in closes]
        lows = [c - np.random.rand() * 2 for c in closes]
        volumes = [float(np.random.randint(1_000_000, 5_000_000)) for _ in closes]

        result = predictor.predict(
            indicators={"rsi_14": 55, "macd": 1.0, "adx": 25},
            fundamentals={"pe_ratio": 20, "roe": 0.15, "profit_margin": 0.12},
            quote={"price": 100, "volume": 2_000_000, "avg_volume": 1_800_000},
            closes=closes,
            highs=highs,
            lows=lows,
            volumes=volumes,
        )

        assert "factor_features" in result
        assert "momentum_12_1" in result["factor_features"]
        assert "signal" in result
        assert result["signal"] in SIGNAL_LABELS


# ─────────────────────────────────────────────────────────────
#  Gap 2b – Analyzer.generate_algo_prediction wiring
# ─────────────────────────────────────────────────────────────

class TestAnalyzerAlgoPredictionWiring:
    """
    Verify that generate_algo_prediction() correctly forwards
    price_history to predictor.predict() as closes/highs/lows/volumes.
    """

    @pytest.fixture
    def v2_model_path(self, tmp_path):
        """37-feature model (Phase-5)."""
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        import joblib

        rng = np.random.RandomState(42)
        X_train = rng.randn(100, 37)
        y_train = rng.randint(0, 5, size=100)

        pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(n_estimators=10, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        model_file = tmp_path / "signal_classifier_v2.joblib"
        joblib.dump(pipeline, model_file)

        meta = {
            "version": 2,
            "feature_count": 37,
            "feature_names": FEATURE_NAMES,
        }
        meta_file = tmp_path / "signal_classifier_v2_meta.json"
        meta_file.write_text(json.dumps(meta), encoding="utf-8")
        return str(model_file)

    def test_price_history_forwarded_to_predictor(self, v2_model_path):
        """
        Mock SignalPredictor so we can inspect what kwargs
        generate_algo_prediction actually passes to predict().
        """
        from unittest.mock import patch, MagicMock

        captured_kwargs = {}

        def mock_predict(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "signal": "buy",
                "confidence": 0.75,
                "probabilities": {"strong_sell": 0.05, "sell": 0.05,
                                  "hold": 0.15, "buy": 0.6, "strong_buy": 0.15},
                "model_version": 2,
                "feature_count": 37,
                "market_regime": "normal",
                "regime_confidence": 0.8,
                "position_size": 0.5,
                "position_rationale": "normal sizing",
            }

        mock_predictor_instance = MagicMock()
        mock_predictor_instance.predict = mock_predict

        price_history = [
            {"open": 99 + i * 0.1, "high": 100 + i * 0.1,
             "low": 98 + i * 0.1, "close": 99.5 + i * 0.1, "volume": 1_000_000}
            for i in range(100)
        ]

        with patch("agents.analyzer.SignalPredictor", return_value=mock_predictor_instance):
            with patch("agents.analyzer._cfg") as mock_cfg:
                mock_cfg.ml_model_enabled = True
                mock_cfg.ml_model_path = v2_model_path

                with patch("agents.analyzer.ML_AVAILABLE", True):
                    from agents.analyzer import StockAnalyzer
                    analyzer = StockAnalyzer.__new__(StockAnalyzer)
                    # Minimal init
                    analyzer.available_models = []

                    analyzer.generate_algo_prediction(
                        symbol="TEST",
                        quote={"price": 105, "volume": 2_000_000, "avg_volume": 1_500_000},
                        indicators={"rsi_14": 55, "adx": 25},
                        fundamentals={"pe_ratio": 20},
                        price_history=price_history,
                    )

        # The key assertion: closes/highs/lows/volumes were forwarded
        assert "closes" in captured_kwargs, (
            "generate_algo_prediction did not forward 'closes' to predictor.predict()"
        )
        assert captured_kwargs["closes"] is not None
        assert len(captured_kwargs["closes"]) == 100
        assert "highs" in captured_kwargs and captured_kwargs["highs"] is not None
        assert "lows" in captured_kwargs and captured_kwargs["lows"] is not None
        assert "volumes" in captured_kwargs and captured_kwargs["volumes"] is not None

    def test_no_price_history_passes_none(self, v2_model_path):
        """When no price_history is provided, closes/highs/lows/volumes should be None."""
        from unittest.mock import patch, MagicMock

        captured_kwargs = {}

        def mock_predict(**kwargs):
            captured_kwargs.update(kwargs)
            return {
                "signal": "hold",
                "confidence": 0.5,
                "probabilities": {"strong_sell": 0.1, "sell": 0.1,
                                  "hold": 0.6, "buy": 0.1, "strong_buy": 0.1},
                "model_version": 2,
                "feature_count": 37,
                "market_regime": "normal",
                "regime_confidence": 0.8,
                "position_size": 0.5,
                "position_rationale": "normal sizing",
            }

        mock_predictor_instance = MagicMock()
        mock_predictor_instance.predict = mock_predict

        with patch("agents.analyzer.SignalPredictor", return_value=mock_predictor_instance):
            with patch("agents.analyzer._cfg") as mock_cfg:
                mock_cfg.ml_model_enabled = True
                mock_cfg.ml_model_path = v2_model_path

                with patch("agents.analyzer.ML_AVAILABLE", True):
                    from agents.analyzer import StockAnalyzer
                    analyzer = StockAnalyzer.__new__(StockAnalyzer)
                    analyzer.available_models = []

                    analyzer.generate_algo_prediction(
                        symbol="TEST",
                        quote={"price": 100, "volume": 1_000_000, "avg_volume": 1_000_000},
                        indicators={"rsi_14": 50},
                    )

        # Without price_history, all series should be None
        assert captured_kwargs.get("closes") is None
        assert captured_kwargs.get("highs") is None
        assert captured_kwargs.get("lows") is None
        assert captured_kwargs.get("volumes") is None


# ─────────────────────────────────────────────────────────────
#  Gap 3 – Embargo is date-based, not row-based
# ─────────────────────────────────────────────────────────────

class TestEmbargoDateBased:
    """
    _purged_walk_forward_split should embargo by calendar dates,
    not by row count.

    Consider a dataset with 30 symbols × 20 dates = 600 rows.
      - embargo_days=5  →  should exclude 5 *calendar days*
                            (i.e. 5 × 30 = 150 rows), not just 5 rows.

    The original implementation uses `split_point + embargo_days`
    which treats embargo_days as a ROW offset.  This test catches
    that bug and validates the fix.
    """

    def _make_rows(
        self,
        n_symbols: int = 10,
        n_dates: int = 20,
        start_date: str = "2025-01-01",
    ) -> tuple[list[dict], np.ndarray, np.ndarray]:
        """
        Create a synthetic dataset with n_symbols × n_dates rows,
        each having a 'timestamp' field formatted as YYYY-MM-DD.
        """


        base = datetime.strptime(start_date, "%Y-%m-%d")
        rows = []
        for d in range(n_dates):
            date_str = (base + timedelta(days=d)).strftime("%Y-%m-%d")
            for s in range(n_symbols):
                rows.append({
                    "timestamp": date_str,
                    "symbol": f"SYM{s}",
                    "signal": "hold",
                })

        n = len(rows)
        rng = np.random.RandomState(42)
        X = rng.randn(n, 37)
        y = rng.randint(0, 5, size=n)
        return rows, X, y

    def test_embargo_excludes_calendar_days_not_rows(self):
        """
        With 10 symbols × 40 dates and embargo_days=5:
          - 80/20 split → ~32 dates train, ~8 dates test
          - embargo eats 5 calendar days after the last train date
          - net: ~3 dates remaining in test
          - gap between last train date and first test date ≥ 5

        Uses enough dates so embargo doesn't eat all test data.
        """
        from training.train import _purged_walk_forward_split

        rows, X, y = self._make_rows(n_symbols=10, n_dates=40)
        embargo_days = 5

        X_train, X_test, y_train, y_test, test_rows = _purged_walk_forward_split(
            rows, X, y, test_size=0.2, embargo_days=embargo_days,
        )

        if len(test_rows) == 0:
            pytest.skip("No test rows produced (embargo ate everything)")

        test_dates = sorted({r["timestamp"] for r in test_rows})

        # Reconstruct actual train rows using X_train size:
        # the first X_train.shape[0] sorted indices are train
        sorted_indices = sorted(range(len(rows)), key=lambda i: rows[i].get("timestamp", ""))
        train_indices_set = set(sorted_indices[:X_train.shape[0]])
        actual_train_rows = [rows[i] for i in sorted_indices if i in train_indices_set]
        train_dates = sorted({r["timestamp"] for r in actual_train_rows})

        last_train_date = train_dates[-1]
        first_test_date = test_dates[0]

        gap = (datetime.strptime(first_test_date, "%Y-%m-%d")
               - datetime.strptime(last_train_date, "%Y-%m-%d")).days

        # CORE ASSERTION: gap must be > embargo_days
        # (embargo zone is (split_date, split_date + embargo_days],
        #  so test starts strictly after, meaning > not >=)
        assert gap > embargo_days, (
            f"Embargo gap is only {gap} calendar days, but embargo_days={embargo_days}. "
            f"Last train date={last_train_date}, first test date={first_test_date}. "
            f"This means embargo is row-based, not date-based."
        )

    def test_no_date_overlap_between_train_and_test(self):
        """Train and test must never share a calendar date."""
        from training.train import _purged_walk_forward_split

        rows, X, y = self._make_rows(n_symbols=10, n_dates=40)

        X_train, X_test, y_train, y_test, test_rows = _purged_walk_forward_split(
            rows, X, y, test_size=0.2, embargo_days=5,
        )

        test_dates = {r["timestamp"] for r in test_rows}

        # Reconstruct train from X_train shape
        sorted_indices = sorted(range(len(rows)), key=lambda i: rows[i].get("timestamp", ""))
        train_indices_set = set(sorted_indices[:X_train.shape[0]])
        train_dates = {rows[i]["timestamp"] for i in train_indices_set}

        overlap = train_dates & test_dates
        assert len(overlap) == 0, (
            f"Train and test share {len(overlap)} dates: {sorted(overlap)}"
        )

    def test_embargo_single_symbol_still_works(self):
        """
        With only 1 symbol per date and enough dates,
        the embargo should correctly gap train from test.
        """
        from training.train import _purged_walk_forward_split

        rows, X, y = self._make_rows(n_symbols=1, n_dates=50)

        X_train, X_test, y_train, y_test, test_rows = _purged_walk_forward_split(
            rows, X, y, test_size=0.2, embargo_days=5,
        )

        assert len(X_train) > 0
        assert len(X_test) > 0

        test_dates = sorted({r["timestamp"] for r in test_rows})

        # Reconstruct train
        sorted_indices = sorted(range(len(rows)), key=lambda i: rows[i].get("timestamp", ""))
        train_indices_set = set(sorted_indices[:X_train.shape[0]])
        train_dates = sorted({rows[i]["timestamp"] for i in train_indices_set})

        if train_dates and test_dates:
            gap = (datetime.strptime(test_dates[0], "%Y-%m-%d")
                   - datetime.strptime(train_dates[-1], "%Y-%m-%d")).days
            assert gap > 5, (
                f"Embargo gap is only {gap} days. "
                f"Last train={train_dates[-1]}, first test={test_dates[0]}"
            )
