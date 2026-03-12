"""In-process tests for backend health and Prometheus metrics."""

from __future__ import annotations

import os

from fastapi.testclient import TestClient


os.environ.setdefault("BACKEND_API_KEY", "test-backend-key")

from backend.app import app  # noqa: E402


client = TestClient(app)


def test_health_route_returns_expected_shape():
    response = client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "stock-radar-backend"
    assert "timestamp" in data
    assert "uptimeSeconds" in data
    assert "dependencies" in data
    assert "supabase" in data["dependencies"]
    assert "llm" in data["dependencies"]


def test_metrics_route_returns_prometheus_payload():
    response = client.get("/metrics")

    assert response.status_code == 200
    content_type = response.headers.get("content-type", "").lower()
    assert "text/plain" in content_type or "openmetrics" in content_type

    body = response.text
    assert "stockradar_system_up" in body
    assert "stockradar_ml_model_loaded" in body
