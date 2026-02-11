"""
Stock Radar - Structured Logging Configuration.

Uses structlog to produce machine-readable JSON logs in production
and human-friendly colored logs in development.

WHY THIS MATTERS (AI Engineering):
- When your AI system is running in production and something goes wrong,
  you need structured logs to debug it.
- JSON logs can be ingested by tools like Datadog, ELK Stack, CloudWatch.
- Every log entry carries context (symbol, model_used, latency_sec, tokens).
- This is the difference between "print debugging" and production observability.

USAGE:
    from logging_config import get_logger
    logger = get_logger(__name__)

    logger.info("analysis_complete", symbol="AAPL", signal="buy", latency_sec=1.2)
    logger.warning("model_fallback", from_model="zai", to_model="gemini")
    logger.error("api_failed", provider="twelvedata", error="timeout")
"""

from __future__ import annotations

import logging
import sys

import structlog


def configure_logging(json_mode: bool = False) -> None:
    """
    Set up structured logging for the entire application.

    Args:
        json_mode: If True, output JSON logs (for production).
                   If False, output colored console logs (for development).
    """
    # Shared processors that run on every log message
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_mode:
        # Production: machine-readable JSON
        renderer = structlog.processors.JSONRenderer()
    else:
        # Development: colored, human-readable
        renderer = structlog.dev.ConsoleRenderer(colors=True)

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the standard library root logger
    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            renderer,
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "urllib3", "litellm", "yfinance"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Get a structured logger for a module.

    Usage:
        logger = get_logger(__name__)
        logger.info("something_happened", key="value")
    """
    return structlog.get_logger(name)


# Auto-configure on import (can be reconfigured later)
def _init() -> None:
    try:
        from config import settings
        configure_logging(json_mode=settings.log_json)
    except Exception:
        # Fallback if config not available yet
        configure_logging(json_mode=False)


_init()
