"""
ML signal predictor.

SignalPredictor loads a trained joblib model and returns predictions
with signal, confidence, and per-class probabilities.

Phase-5 upgrade: supports 37-feature vector with optional factor and
microstructure features passed through at inference time.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from training.feature_engineering import (
    SIGNAL_LABELS,
    decode_signal,
    extract_features,
)
from training.cross_sectional import (
    compute_factor_features,
    compute_microstructure_features,
)
from training.feature_health import check_feature_health, log_feature_health
from training.regime import classify_market_regime
from training.risk import (
    calculate_position_size,
    calculate_stop_take_profit,
    calculate_per_trade_risk,
)

logger = logging.getLogger(__name__)


class SignalPredictor:
    """Load a trained model and predict signals from raw financial data."""

    def __init__(
        self,
        model_path: str,
        meta_path: str | None = None,
        risk_factor: float = 1.0,
        target_volatility_pct: float = 2.0,
        min_confidence: float = 0.35,
    ):
        import joblib

        self.model_path = model_path
        self.pipeline = joblib.load(model_path)
        self.risk_factor = risk_factor
        self.target_volatility_pct = target_volatility_pct
        self.min_confidence = min_confidence

        # Load metadata if available
        self.metadata: dict = {}
        mp = Path(meta_path) if meta_path else Path(model_path).with_name(
            Path(model_path).stem + "_meta.json"
        )
        if mp.exists():
            import json
            self.metadata = json.loads(mp.read_text(encoding="utf-8"))

        self.model_version = self.metadata.get("version", 0)
        self.feature_reference_stats = self.metadata.get("feature_reference_stats")

        # Phase-9: load calibrated model if available
        self.calibrated_pipeline = None
        cal_path_str = self.metadata.get("calibrated_model_path")
        if cal_path_str:
            cal_path = Path(cal_path_str)
            if cal_path.exists():
                import joblib as _jl
                self.calibrated_pipeline = _jl.load(cal_path)
                logger.info(f"Calibrated pipeline loaded from {cal_path}")

        logger.info(f"SignalPredictor loaded v{self.model_version} from {model_path}")

    @property
    def feature_count(self) -> int:
        """Resolved feature count: metadata → pipeline introspection → 20."""
        return self._expected_feature_count() or 20

    def _expected_feature_count(self) -> int | None:
        """
        How many features the loaded model expects.

        Checks the metadata first; falls back to introspecting
        ``n_features_in_`` on the pipeline's first step.  Returns
        ``None`` if the count cannot be determined.
        """
        # Metadata is the most reliable source
        n = self.metadata.get("feature_count")
        if n:
            return int(n)

        # Introspect the fitted pipeline
        try:
            first_step = self.pipeline[0]  # imputer
            if hasattr(first_step, "n_features_in_"):
                return int(first_step.n_features_in_)
        except Exception:
            pass
        return None

    def predict(
        self,
        indicators: Dict[str, Any],
        fundamentals: Optional[Dict[str, Any]] = None,
        quote: Optional[Dict[str, Any]] = None,
        *,
        closes: Optional[Sequence[float]] = None,
        highs: Optional[Sequence[float]] = None,
        lows: Optional[Sequence[float]] = None,
        volumes: Optional[Sequence[float]] = None,
        headlines: Optional[Sequence[str]] = None,
        headline_timestamps: Optional[Sequence[Any]] = None,
        finnhub_sentiment: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Predict signal from raw financial data.

        Phase-5: if close/high/low/volume price series are provided, factor
        and microstructure features are computed on-the-fly.  Otherwise
        those feature slots are NaN (backward compatible with v1 models
        that only used the first 20 features—imputer fills NaN with median).

        Phase-6: if headlines / finnhub_sentiment are provided, 8 sentiment
        features are computed on-the-fly.

        Returns:
            Dict with signal, confidence, probabilities, model_version, etc.
        """
        # Compute Phase-5 features if price series are available
        factor_feats = None
        micro_feats = None

        if closes is not None and len(closes) >= 20:
            factor_feats = compute_factor_features(closes, fundamentals)

            if highs is not None and lows is not None and volumes is not None:
                micro_feats = compute_microstructure_features(
                    closes, highs, lows, volumes,
                )

        # Compute Phase-6 sentiment features if data available
        sentiment_feats = None
        if headlines or finnhub_sentiment:
            from training.sentiment import compute_sentiment_features

            sentiment_feats = compute_sentiment_features(
                headlines=headlines,
                headline_timestamps=headline_timestamps,
                finnhub_sentiment=finnhub_sentiment,
                fundamentals=fundamentals,
            )

        features = extract_features(
            indicators, fundamentals, quote,
            factor=factor_feats,
            microstructure=micro_feats,
            sentiment=sentiment_feats,
        )

        # Backward compatibility: if the model was trained on fewer
        # features (e.g. 20 in v1 or 37 in v2), slice the vector to match.
        # This lets old models work with the new 45-feature extraction.
        model_n_features = self._expected_feature_count()
        if model_n_features and model_n_features < len(features):
            features = features[:model_n_features]

        X = features.reshape(1, -1)

        # ── Feature health check (Phase-5) ──
        feature_health_report = None
        if self.feature_reference_stats:
            from training.feature_engineering import FEATURE_NAMES as _fn
            fnames = list(_fn)
            if model_n_features and model_n_features < len(fnames):
                fnames = fnames[:model_n_features]
            feature_health_report = check_feature_health(
                X,
                feature_names=fnames,
                reference_stats=self.feature_reference_stats,
            )
            if feature_health_report["overall_status"] != "healthy":
                log_feature_health(feature_health_report)

        # Phase-9: use calibrated pipeline if available
        use_calibrated = self.calibrated_pipeline is not None
        active_pipeline = self.calibrated_pipeline if use_calibrated else self.pipeline

        predicted_label = int(active_pipeline.predict(X)[0])
        probabilities = active_pipeline.predict_proba(X)[0]

        signal = decode_signal(predicted_label)
        confidence = float(probabilities[predicted_label])
        regime_info = classify_market_regime(indicators)
        position_info = calculate_position_size(
            signal=signal,
            confidence=confidence,
            volatility_pct=indicators.get("atr_pct") if indicators else None,
            regime=regime_info["regime"],
            risk_factor=self.risk_factor,
            target_volatility_pct=self.target_volatility_pct,
            min_confidence=self.min_confidence,
        )

        stop_tp = calculate_stop_take_profit(
            signal=signal,
            entry_price=float(quote.get("price", 0)) if quote else 0.0,
            atr=indicators.get("atr_14") if indicators else None,
        )

        # Per-trade risk budgeting: scale down position if risk exceeds budget
        final_position = position_info["position_size"]
        per_trade_risk = None
        if stop_tp["risk_pct"] is not None:
            per_trade_risk = calculate_per_trade_risk(
                position_size=abs(final_position),
                stop_loss_pct=stop_tp["risk_pct"],
            )
            if not per_trade_risk["within_limits"]:
                sign = 1.0 if final_position >= 0 else -1.0
                final_position = sign * per_trade_risk["adjusted_position_size"]
                position_info = {
                    **position_info,
                    "position_size": round(float(final_position), 6),
                    "position_size_pct": round(abs(final_position) * 100.0, 3),
                    "risk_adjusted": True,
                }

        result = {
            "signal": signal,
            "confidence": round(confidence, 4),
            "probabilities": {
                SIGNAL_LABELS[i]: round(float(p), 4)
                for i, p in enumerate(probabilities)
            },
            "model_version": self.model_version,
            "feature_count": self._expected_feature_count() or self.feature_count,
            "market_regime": regime_info["regime"],
            "regime_confidence": regime_info["confidence"],
            "position_size": final_position,
            "position_size_pct": round(abs(final_position) * 100.0, 3),
            "position_sizing": position_info,
            "stop_loss": stop_tp["stop_loss"],
            "take_profit": stop_tp["take_profit"],
            "risk_reward": stop_tp,
            "per_trade_risk": per_trade_risk,
            "calibrated": use_calibrated,
        }

        # Add Phase-5 factor summary if available
        if factor_feats:
            result["factor_features"] = {
                k: v for k, v in factor_feats.items()
                if v is not None and v == v  # exclude NaN
            }

        # Add feature health report if computed
        if feature_health_report:
            result["feature_health"] = {
                "status": feature_health_report["overall_status"],
                "summary": feature_health_report["summary"],
            }

        return result
