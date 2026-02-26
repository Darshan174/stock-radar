"""
Lightweight HTTP server for Prometheus metrics and health checks.

Exposes:
    GET /metrics  - Prometheus metrics endpoint
    GET /health   - JSON health check with component status

Runs on port 9090 (configurable). Can be started standalone or as a
background thread from main.py.

Usage:
    python -m monitoring.server              # Standalone
    python -m monitoring.server --port 9091  # Custom port
"""

from __future__ import annotations

import json
import os
import sys
import time
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

logger = logging.getLogger(__name__)

_start_time = time.time()


def _check_redis() -> dict:
    """Check Redis connectivity."""
    try:
        redis_url = os.getenv("REDIS_URL")
        if not redis_url:
            return {"status": "not_configured"}
        import redis
        r = redis.from_url(redis_url, socket_timeout=2)
        r.ping()
        return {"status": "healthy"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def _check_supabase() -> dict:
    """Check Supabase connectivity."""
    try:
        url = os.getenv("SUPABASE_URL")
        key = os.getenv("SUPABASE_KEY")
        if not url or not key:
            return {"status": "not_configured"}
        return {"status": "configured"}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}


def _check_ml_model() -> dict:
    """Check if ML model is loaded."""
    try:
        from config import settings
        if not settings.ml_model_enabled:
            return {"status": "disabled"}
        if settings.ml_model_path and os.path.exists(settings.ml_model_path):
            return {"status": "loaded", "path": settings.ml_model_path}
        return {"status": "not_found"}
    except Exception:
        return {"status": "disabled"}


class MetricsHandler(BaseHTTPRequestHandler):
    """HTTP handler for /metrics and /health endpoints."""

    def do_GET(self):
        if self.path == "/metrics":
            self._serve_metrics()
        elif self.path == "/health":
            self._serve_health()
        else:
            self.send_error(404)

    def _serve_metrics(self):
        metrics_data = generate_latest()
        self.send_response(200)
        self.send_header("Content-Type", CONTENT_TYPE_LATEST)
        self.send_header("Content-Length", str(len(metrics_data)))
        self.end_headers()
        self.wfile.write(metrics_data)

    def _serve_health(self):
        uptime = time.time() - _start_time

        health = {
            "status": "healthy",
            "uptime_seconds": round(uptime, 1),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "redis": _check_redis(),
                "supabase": _check_supabase(),
                "ml_model": _check_ml_model(),
            },
        }

        # Mark unhealthy if critical components are down
        for comp in health["components"].values():
            if comp.get("status") == "unhealthy":
                health["status"] = "degraded"
                break

        payload = json.dumps(health, indent=2).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        """Suppress default request logging."""
        pass


def start_server(
    port: int = 9090,
    background: bool = False,
    host: str = "0.0.0.0",
) -> HTTPServer | None:
    """
    Start the metrics/health server.

    Args:
        port: Port to listen on.
        background: If True, runs in a daemon thread.
        host: Interface to bind to (default: 0.0.0.0).

    Returns:
        HTTPServer instance (or None if backgrounded).
    """
    bind_host = host
    try:
        server = HTTPServer((bind_host, port), MetricsHandler)
    except PermissionError:
        # Some restricted environments disallow wildcard binds.
        # Fall back to loopback so health/metrics tests can still run.
        bind_host = "127.0.0.1"
        server = HTTPServer((bind_host, port), MetricsHandler)

    logger.info(f"Metrics server started on http://{bind_host}:{port}")
    logger.info("  /metrics - Prometheus metrics")
    logger.info("  /health  - Health check")

    if background:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server
    else:
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
        return server


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Stock Radar metrics server")
    parser.add_argument("--port", type=int, default=9090, help="Port (default: 9090)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    print(f"Starting metrics server on port {args.port}...")
    start_server(port=args.port, background=False)


if __name__ == "__main__":
    main()
