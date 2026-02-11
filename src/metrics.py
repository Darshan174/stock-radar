"""
Stock Radar - Prometheus Metrics.

Exposes quantitative metrics about your AI system's behavior.
Prometheus scrapes the /metrics endpoint, and Grafana visualizes it.

WHY THIS MATTERS (AI Engineering):
- "How do you monitor your AI system in production?" is a top interview question.
- These metrics answer: How fast is it? How often does it fail? How much does it cost?
- Prometheus is the industry standard for metrics collection.

METRICS EXPOSED:
    stockradar_llm_latency_seconds        - How long each LLM call takes
    stockradar_llm_requests_total          - Total LLM calls (by model, status)
    stockradar_llm_fallback_total          - How often fallbacks trigger
    stockradar_llm_tokens_total            - Total tokens used (input/output)
    stockradar_analysis_confidence         - Distribution of confidence scores
    stockradar_analysis_signal_total       - Count of each signal type
    stockradar_api_cost_usd_total          - Running cost in USD
    stockradar_data_fetch_latency_seconds  - Data provider latency
    stockradar_cache_hits_total            - Cache effectiveness
    stockradar_guardrail_triggers_total    - How often guardrails fire

USAGE:
    from metrics import (
        LLM_LATENCY, LLM_REQUESTS, LLM_TOKENS,
        ANALYSIS_CONFIDENCE, API_COST
    )

    # Record an LLM call
    LLM_LATENCY.labels(model="zai").observe(1.2)
    LLM_REQUESTS.labels(model="zai", status="success").inc()
    LLM_TOKENS.labels(direction="input", model="zai").inc(1200)
    LLM_TOKENS.labels(direction="output", model="zai").inc(350)
    API_COST.labels(provider="zai").inc(0.004)

    # Record analysis result
    ANALYSIS_CONFIDENCE.observe(0.85)
"""

from __future__ import annotations

from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST


# =============================================================================
# LLM Metrics
# =============================================================================

LLM_LATENCY = Histogram(
    "stockradar_llm_latency_seconds",
    "LLM API call latency in seconds",
    ["model"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)

LLM_REQUESTS = Counter(
    "stockradar_llm_requests_total",
    "Total LLM API requests",
    ["model", "status"],  # status: success, error
)

LLM_FALLBACK = Counter(
    "stockradar_llm_fallback_total",
    "Number of times a fallback model was used",
    ["from_model", "to_model"],
)

LLM_TOKENS = Counter(
    "stockradar_llm_tokens_total",
    "Total tokens used",
    ["direction", "model"],  # direction: input, output
)

# =============================================================================
# Analysis Metrics
# =============================================================================

ANALYSIS_CONFIDENCE = Histogram(
    "stockradar_analysis_confidence",
    "Distribution of analysis confidence scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

ANALYSIS_SIGNAL = Counter(
    "stockradar_analysis_signal_total",
    "Count of each trading signal",
    ["signal", "mode"],  # signal: strong_buy/buy/hold/sell/strong_sell
)

ANALYSIS_DURATION = Histogram(
    "stockradar_analysis_duration_seconds",
    "Total analysis pipeline duration",
    ["mode"],  # intraday, longterm
    buckets=(1, 2, 5, 10, 30, 60, 120),
)

# =============================================================================
# Cost Metrics
# =============================================================================

API_COST = Counter(
    "stockradar_api_cost_usd_total",
    "Running API cost in USD",
    ["provider"],  # zai, gemini, cohere, twelvedata, finnhub
)

# =============================================================================
# Data Fetch Metrics
# =============================================================================

DATA_FETCH_LATENCY = Histogram(
    "stockradar_data_fetch_latency_seconds",
    "Data provider fetch latency",
    ["provider", "operation"],  # provider: yfinance, twelvedata, finnhub
    buckets=(0.1, 0.5, 1, 2, 5, 10, 30),
)

DATA_FETCH_ERRORS = Counter(
    "stockradar_data_fetch_errors_total",
    "Data provider errors",
    ["provider", "error_type"],
)

# =============================================================================
# Cache Metrics
# =============================================================================

CACHE_HITS = Counter(
    "stockradar_cache_hits_total",
    "Cache hit count",
    ["cache_type"],  # quote, fundamentals, analysis, embedding
)

CACHE_MISSES = Counter(
    "stockradar_cache_misses_total",
    "Cache miss count",
    ["cache_type"],
)

# =============================================================================
# Guardrail Metrics
# =============================================================================

GUARDRAIL_TRIGGERS = Counter(
    "stockradar_guardrail_triggers_total",
    "Guardrail trigger count",
    ["rule", "action"],  # action: blocked, adjusted, warned
)

# =============================================================================
# Scoring Metrics
# =============================================================================

SCORING_DISTRIBUTION = Histogram(
    "stockradar_scoring_composite",
    "Distribution of composite algorithmic scores",
    buckets=(10, 20, 30, 40, 50, 60, 70, 80, 90, 100),
)


def get_metrics_response() -> tuple[bytes, str]:
    """Generate Prometheus metrics response."""
    return generate_latest(), CONTENT_TYPE_LATEST
