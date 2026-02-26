"""
Alert rules for monitoring Stock Radar.

Defines alert conditions and checks in-process Prometheus metrics
against thresholds.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List

import structlog

logger = structlog.get_logger(__name__)


@dataclass
class AlertRule:
    """Definition of an alert condition."""
    name: str
    metric: str
    condition: str  # 'gt' (greater than) or 'lt' (less than)
    threshold: float
    severity: str  # 'critical', 'warning', 'info'
    message: str

    def evaluate(self, value: float) -> bool:
        """Check if the alert condition is triggered."""
        if self.condition == "gt":
            return value > self.threshold
        elif self.condition == "lt":
            return value < self.threshold
        return False


# Default alert rules
DEFAULT_RULES: List[AlertRule] = [
    AlertRule(
        name="llm_high_error_rate",
        metric="stockradar_llm_requests_total{status='error'}",
        condition="gt",
        threshold=0.20,
        severity="critical",
        message="LLM error rate exceeds 20%",
    ),
    AlertRule(
        name="llm_high_latency",
        metric="stockradar_llm_latency_seconds_p95",
        condition="gt",
        threshold=30.0,
        severity="warning",
        message="LLM p95 latency exceeds 30 seconds",
    ),
    AlertRule(
        name="data_fetch_failures",
        metric="stockradar_data_fetch_errors_total",
        condition="gt",
        threshold=5.0,
        severity="warning",
        message="Data fetch failures exceed 5",
    ),
    AlertRule(
        name="guardrail_block_rate",
        metric="stockradar_guardrail_triggers_total{action='blocked'}",
        condition="gt",
        threshold=0.10,
        severity="warning",
        message="Guardrail block rate exceeds 10%",
    ),
]


def check_alerts(
    rules: List[AlertRule] | None = None,
) -> List[Dict[str, Any]]:
    """
    Check all alert rules against current Prometheus metrics.

    This queries in-process metrics from prometheus_client and evaluates
    each rule.

    Returns:
        List of triggered alerts with details.
    """
    from prometheus_client import REGISTRY

    rules = rules or DEFAULT_RULES
    triggered = []

    # Collect current metric values from the in-process registry
    metric_values: Dict[str, float] = {}
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            key = sample.name
            if sample.labels:
                label_str = ",".join(f"{k}='{v}'" for k, v in sorted(sample.labels.items()))
                key = f"{sample.name}{{{label_str}}}"
            metric_values[key] = sample.value

    for rule in rules:
        # Try to find matching metric
        value = metric_values.get(rule.metric)
        if value is not None and rule.evaluate(value):
            alert = {
                "name": rule.name,
                "severity": rule.severity,
                "message": rule.message,
                "metric": rule.metric,
                "value": value,
                "threshold": rule.threshold,
            }
            triggered.append(alert)
            logger.warning(
                "alert_triggered",
                alert_name=rule.name,
                severity=rule.severity,
                value=value,
                threshold=rule.threshold,
            )

    return triggered
