"""Unit tests for the monitoring server."""

import json
import os
import sys
import threading
import time
import urllib.request

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestHealthEndpoint:
    """Test /health JSON structure."""

    def test_health_json_structure(self):
        from monitoring.server import start_server

        server = start_server(port=19090, background=True)
        time.sleep(0.5)  # Let server start

        try:
            req = urllib.request.Request("http://localhost:19090/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode())

            assert "status" in data
            assert data["status"] in ("healthy", "degraded")
            assert "uptime_seconds" in data
            assert isinstance(data["uptime_seconds"], (int, float))
            assert "timestamp" in data
            assert "components" in data
            assert "redis" in data["components"]
            assert "supabase" in data["components"]
            assert "ml_model" in data["components"]

            # Each component should have a status field
            for comp_name, comp in data["components"].items():
                assert "status" in comp, f"Component {comp_name} missing status"
        finally:
            server.shutdown()


class TestMetricsEndpoint:
    """Test /metrics returns valid Prometheus format."""

    def test_metrics_prometheus_format(self):
        from monitoring.server import start_server

        server = start_server(port=19091, background=True)
        time.sleep(0.5)

        try:
            req = urllib.request.Request("http://localhost:19091/metrics")
            with urllib.request.urlopen(req, timeout=5) as resp:
                body = resp.read().decode()
                content_type = resp.headers.get("Content-Type", "")

            # Prometheus format validation
            assert "text/plain" in content_type or "text/plain" in content_type.lower() or "openmetrics" in content_type.lower()
            # Should contain at least some stockradar metrics
            assert "stockradar" in body or "python" in body  # At least process metrics
            # Lines should be metric format: metric_name{labels} value
            lines = [l for l in body.split("\n") if l and not l.startswith("#")]
            assert len(lines) > 0
        finally:
            server.shutdown()


class TestAlertRules:
    """Test alert rule evaluation."""

    def test_alert_rule_gt(self):
        from monitoring.alerts import AlertRule

        rule = AlertRule(
            name="test",
            metric="test_metric",
            condition="gt",
            threshold=10.0,
            severity="warning",
            message="test alert",
        )

        assert rule.evaluate(15.0) is True
        assert rule.evaluate(5.0) is False
        assert rule.evaluate(10.0) is False

    def test_alert_rule_lt(self):
        from monitoring.alerts import AlertRule

        rule = AlertRule(
            name="test",
            metric="test_metric",
            condition="lt",
            threshold=5.0,
            severity="critical",
            message="test alert",
        )

        assert rule.evaluate(3.0) is True
        assert rule.evaluate(7.0) is False
