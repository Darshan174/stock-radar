"""Regime-aware model routing for signal prediction."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict

from training.regime import classify_market_regime

logger = logging.getLogger(__name__)

REGIME_SUFFIXES = ["trending", "mean_reverting", "high_volatility"]


def discover_regime_models(model_dir: str = "models/") -> dict[str, str]:
    """
    Scan for files matching signal_classifier_v*_{regime}.joblib.
    Returns {regime_name: model_path} for highest version per regime.
    """
    model_path = Path(model_dir)
    if not model_path.exists():
        return {}

    pattern = re.compile(
        r"signal_classifier_v(\d+)_(" + "|".join(REGIME_SUFFIXES) + r")\.joblib$"
    )

    # Track highest version per regime
    best: dict[str, tuple[int, str]] = {}

    for f in model_path.glob("signal_classifier_v*_*.joblib"):
        match = pattern.match(f.name)
        if not match:
            continue
        version = int(match.group(1))
        regime = match.group(2)
        if regime not in best or version > best[regime][0]:
            best[regime] = (version, str(f))

    return {regime: path for regime, (_, path) in best.items()}


class RegimeAwarePredictor:
    """
    Wraps SignalPredictor (composition, not inheritance).
    Lazy-loads regime models on first use.
    Falls back to general model if regime model missing or load fails.
    """

    def __init__(
        self,
        general_model_path: str,
        general_meta_path: str | None = None,
        model_dir: str = "models/",
        risk_factor: float = 1.0,
        target_volatility_pct: float = 2.0,
        min_confidence: float = 0.35,
    ):
        from training.predictor import SignalPredictor

        self.general_predictor = SignalPredictor(
            model_path=general_model_path,
            meta_path=general_meta_path,
            risk_factor=risk_factor,
            target_volatility_pct=target_volatility_pct,
            min_confidence=min_confidence,
        )
        self.model_dir = model_dir
        self.risk_factor = risk_factor
        self.target_volatility_pct = target_volatility_pct
        self.min_confidence = min_confidence

        self._regime_models: dict[str, Any] = {}
        self._regime_paths = discover_regime_models(model_dir)
        self._loaded = False

        if self._regime_paths:
            logger.info(
                f"Regime models discovered: {list(self._regime_paths.keys())}"
            )

    def _get_regime_predictor(self, regime: str):
        """Lazy-load a regime-specific predictor."""
        if regime in self._regime_models:
            return self._regime_models[regime]

        path = self._regime_paths.get(regime)
        if not path:
            return None

        try:
            from training.predictor import SignalPredictor

            predictor = SignalPredictor(
                model_path=path,
                risk_factor=self.risk_factor,
                target_volatility_pct=self.target_volatility_pct,
                min_confidence=self.min_confidence,
            )
            self._regime_models[regime] = predictor
            logger.info(f"Loaded regime model for '{regime}' from {path}")
            return predictor
        except Exception as e:
            logger.warning(f"Failed to load regime model for '{regime}': {e}")
            self._regime_models[regime] = None
            return None

    def predict(self, indicators: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        1. classify_market_regime(indicators)
        2. Pick regime-specific SignalPredictor (or general)
        3. predictor.predict(indicators, **kwargs)
        4. Add regime_model_used, regime_model_name to result
        """
        regime_info = classify_market_regime(indicators)
        regime = regime_info["regime"]

        # Neutral always uses the general model
        predictor = None
        regime_model_name = None
        if regime != "neutral" and regime in self._regime_paths:
            predictor = self._get_regime_predictor(regime)
            if predictor is not None:
                regime_model_name = Path(self._regime_paths[regime]).stem

        if predictor is None:
            predictor = self.general_predictor
            regime_model_name = None

        result = predictor.predict(indicators, **kwargs)
        result["regime_model_used"] = regime_model_name is not None
        result["regime_model_name"] = regime_model_name
        return result
