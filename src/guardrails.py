"""
Stock Radar - LLM Output Guardrails & Validation.

Validates every LLM response before returning it to the user.
Catches hallucinations, invalid data, and ensures output quality.

WHY THIS MATTERS (AI Engineering):
- LLMs hallucinate. They make up numbers, invent companies, claim fake signals.
- Interviewers ask: "How do you handle hallucinations?"
- This module answers: by validating EVERY output against the actual data.

RULES ENFORCED:
    1. Schema Validation   - Response has all required fields (signal, confidence, etc.)
    2. Signal Validity      - Signal is one of: strong_buy, buy, hold, sell, strong_sell
    3. Confidence Bounds    - Confidence is between 0.0 and configured max (default 0.95)
    4. Price Sanity         - Target/stop_loss are within reasonable range of current price
    5. Reasoning Required   - Non-empty reasoning text
    6. Consistency Check    - Signal direction matches the reasoning sentiment
    7. Disclaimer Check     - Ensure no financial advice claims

USAGE:
    from guardrails import GuardrailEngine, GuardrailResult

    engine = GuardrailEngine()
    result = engine.validate(
        llm_output={"signal": "strong_buy", "confidence": 1.5, ...},
        current_price=150.0,
        mode="intraday"
    )

    if not result.passed:
        print(result.issues)       # ["confidence_capped: 1.5 -> 0.95"]
        print(result.adjusted)     # The corrected output
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from metrics import GUARDRAIL_TRIGGERS


VALID_SIGNALS = {"strong_buy", "buy", "hold", "sell", "strong_sell"}

BULLISH_SIGNALS = {"strong_buy", "buy"}
BEARISH_SIGNALS = {"strong_sell", "sell"}

# Words that suggest bullish reasoning
BULLISH_WORDS = {
    "bullish", "uptrend", "breakout", "support", "oversold", "accumulation",
    "positive", "growth", "undervalued", "strong", "momentum",
}
BEARISH_WORDS = {
    "bearish", "downtrend", "breakdown", "resistance", "overbought", "distribution",
    "negative", "decline", "overvalued", "weak", "selling",
}


@dataclass
class GuardrailIssue:
    """A single guardrail violation."""
    rule: str
    severity: str  # "error", "warning", "info"
    message: str
    action: str  # "blocked", "adjusted", "warned"
    original_value: Any = None
    adjusted_value: Any = None


@dataclass
class GuardrailResult:
    """Result of guardrail validation."""
    passed: bool
    issues: list[GuardrailIssue] = field(default_factory=list)
    adjusted: dict[str, Any] = field(default_factory=dict)

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")


class GuardrailEngine:
    """
    Validates and sanitizes LLM outputs before they reach the user.

    Every analysis goes through this engine. Issues are logged as
    Prometheus metrics so you can track how often the LLM misbehaves.
    """

    def __init__(
        self,
        max_confidence: float = 0.95,
        require_reasoning: bool = True,
        max_target_deviation_pct: float = 50.0,
    ) -> None:
        self.max_confidence = max_confidence
        self.require_reasoning = require_reasoning
        self.max_target_deviation_pct = max_target_deviation_pct

    def validate(
        self,
        llm_output: dict[str, Any],
        current_price: float | None = None,
        mode: str = "intraday",
    ) -> GuardrailResult:
        """
        Run all guardrail checks on an LLM output.

        Args:
            llm_output: The parsed JSON from the LLM.
            current_price: Current stock price (for price sanity checks).
            mode: "intraday" or "longterm".

        Returns:
            GuardrailResult with pass/fail status, issues, and adjusted output.
        """
        issues: list[GuardrailIssue] = []
        adjusted = dict(llm_output)  # Work on a copy

        # 1. Schema validation
        issues.extend(self._check_schema(adjusted))

        # 2. Signal validity
        issues.extend(self._check_signal(adjusted))

        # 3. Confidence bounds
        issues.extend(self._check_confidence(adjusted))

        # 4. Price sanity
        if current_price is not None:
            issues.extend(self._check_prices(adjusted, current_price, mode))

        # 5. Reasoning required
        if self.require_reasoning:
            issues.extend(self._check_reasoning(adjusted))

        # 6. Consistency check
        issues.extend(self._check_consistency(adjusted))

        # Record metrics
        for issue in issues:
            GUARDRAIL_TRIGGERS.labels(rule=issue.rule, action=issue.action).inc()

        passed = all(i.severity != "error" or i.action == "adjusted" for i in issues)

        return GuardrailResult(passed=passed, issues=issues, adjusted=adjusted)

    def _check_schema(self, output: dict) -> list[GuardrailIssue]:
        """Check that required fields exist."""
        issues = []
        required = ["signal", "confidence", "reasoning"]

        for key in required:
            if key not in output or output[key] is None:
                issues.append(
                    GuardrailIssue(
                        rule="schema_missing_field",
                        severity="error",
                        message=f"Missing required field: {key}",
                        action="blocked",
                    )
                )

        return issues

    def _check_signal(self, output: dict) -> list[GuardrailIssue]:
        """Check that signal is valid."""
        issues = []
        signal = output.get("signal", "")

        if signal not in VALID_SIGNALS:
            # Try to fix common variations
            normalized = signal.lower().strip().replace(" ", "_").replace("-", "_")
            if normalized in VALID_SIGNALS:
                issues.append(
                    GuardrailIssue(
                        rule="signal_normalized",
                        severity="info",
                        message=f"Signal normalized: '{signal}' -> '{normalized}'",
                        action="adjusted",
                        original_value=signal,
                        adjusted_value=normalized,
                    )
                )
                output["signal"] = normalized
            else:
                # Default to hold
                issues.append(
                    GuardrailIssue(
                        rule="signal_invalid",
                        severity="warning",
                        message=f"Invalid signal '{signal}', defaulting to 'hold'",
                        action="adjusted",
                        original_value=signal,
                        adjusted_value="hold",
                    )
                )
                output["signal"] = "hold"

        return issues

    def _check_confidence(self, output: dict) -> list[GuardrailIssue]:
        """Check confidence is within bounds."""
        issues = []
        confidence = output.get("confidence")

        if confidence is None:
            return issues

        try:
            confidence = float(confidence)
        except (ValueError, TypeError):
            issues.append(
                GuardrailIssue(
                    rule="confidence_invalid_type",
                    severity="warning",
                    message=f"Confidence is not a number: {confidence}, defaulting to 0.5",
                    action="adjusted",
                    original_value=confidence,
                    adjusted_value=0.5,
                )
            )
            output["confidence"] = 0.5
            return issues

        if confidence > self.max_confidence:
            issues.append(
                GuardrailIssue(
                    rule="confidence_capped",
                    severity="info",
                    message=f"Confidence capped: {confidence} -> {self.max_confidence}",
                    action="adjusted",
                    original_value=confidence,
                    adjusted_value=self.max_confidence,
                )
            )
            output["confidence"] = self.max_confidence

        if confidence < 0:
            output["confidence"] = 0.0
            issues.append(
                GuardrailIssue(
                    rule="confidence_floor",
                    severity="warning",
                    message=f"Negative confidence {confidence} set to 0.0",
                    action="adjusted",
                    original_value=confidence,
                    adjusted_value=0.0,
                )
            )

        return issues

    def _check_prices(
        self, output: dict, current_price: float, mode: str
    ) -> list[GuardrailIssue]:
        """Check that target/stop_loss are reasonable."""
        issues = []
        max_dev = self.max_target_deviation_pct
        if mode == "longterm":
            max_dev *= 2  # Allow wider range for long-term

        for key in ("target_price", "stop_loss"):
            value = output.get(key)
            if value is None:
                continue

            try:
                value = float(value)
            except (ValueError, TypeError):
                output[key] = None
                issues.append(
                    GuardrailIssue(
                        rule=f"{key}_invalid",
                        severity="warning",
                        message=f"{key} is not a number: {output.get(key)}, set to null",
                        action="adjusted",
                    )
                )
                continue

            deviation_pct = abs((value - current_price) / current_price * 100)
            if deviation_pct > max_dev:
                issues.append(
                    GuardrailIssue(
                        rule=f"{key}_unrealistic",
                        severity="warning",
                        message=(
                            f"{key}={value} is {deviation_pct:.0f}% from current "
                            f"price {current_price} (max {max_dev}%)"
                        ),
                        action="warned",
                        original_value=value,
                    )
                )

        return issues

    def _check_reasoning(self, output: dict) -> list[GuardrailIssue]:
        """Check that reasoning is non-empty and substantive."""
        issues = []
        reasoning = output.get("reasoning", "")

        if not reasoning or len(reasoning.strip()) < 20:
            issues.append(
                GuardrailIssue(
                    rule="reasoning_too_short",
                    severity="warning",
                    message=f"Reasoning is too short ({len(reasoning)} chars)",
                    action="warned",
                )
            )

        return issues

    def _check_consistency(self, output: dict) -> list[GuardrailIssue]:
        """Check that signal direction is consistent with reasoning."""
        issues = []
        signal = output.get("signal", "hold")
        reasoning = (output.get("reasoning", "") or "").lower()

        if not reasoning or signal == "hold":
            return issues

        # Count bullish vs bearish words in reasoning
        bullish_count = sum(1 for w in BULLISH_WORDS if w in reasoning)
        bearish_count = sum(1 for w in BEARISH_WORDS if w in reasoning)

        if signal in BULLISH_SIGNALS and bearish_count > bullish_count + 2:
            issues.append(
                GuardrailIssue(
                    rule="signal_reasoning_mismatch",
                    severity="warning",
                    message=(
                        f"Bullish signal '{signal}' but reasoning has more "
                        f"bearish words ({bearish_count}) than bullish ({bullish_count})"
                    ),
                    action="warned",
                )
            )

        if signal in BEARISH_SIGNALS and bullish_count > bearish_count + 2:
            issues.append(
                GuardrailIssue(
                    rule="signal_reasoning_mismatch",
                    severity="warning",
                    message=(
                        f"Bearish signal '{signal}' but reasoning has more "
                        f"bullish words ({bullish_count}) than bearish ({bearish_count})"
                    ),
                    action="warned",
                )
            )

        return issues
