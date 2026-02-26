# Stock Radar - AI Engineering Features Guide

This document explains the 10 production-grade AI engineering features
added to Stock Radar. Each feature is something interviewers specifically
ask about when hiring AI Engineers

---

## Table of Contents

1. [Centralized Config (Pydantic)](#1-centralized-config)
2. [Structured Logging (structlog)](#2-structured-logging)
3. [Prometheus Metrics](#3-prometheus-metrics)
4. [Retry Logic (tenacity)](#4-retry-logic)
5. [Evaluation Framework](#5-evaluation-framework)
6. [Token Accounting & Cost Estimation](#6-token-accounting)
7. [Streaming LLM Responses (SSE)](#7-streaming)
8. [LLM Guardrails](#8-guardrails)
9. [Prompt Versioning & A/B Testing](#9-prompt-versioning)
10. [Caching Layer](#10-caching)

---

## 1. Centralized Config

**File:** `src/config.py`

**Interview Question:** "How do you manage configuration in a production AI system?"

### What It Does

Instead of scattered `os.getenv()` calls everywhere, ALL settings live in
one type-safe class using Pydantic BaseSettings:

```python
from config import settings

# Type-safe, validated, with defaults
print(settings.zai_api_key)         # str | None
print(settings.llm_temperature)     # float (default 0.3)
print(settings.retry_max_attempts)  # int (default 3)
print(settings.fallback_models)     # ["zai/glm-5", "gemini/...", "ollama/..."]
```

### How It Works

```
Priority order (highest to lowest):
1. Environment variables  →  export ZAI_API_KEY=xxx
2. .env file              →  ZAI_API_KEY=xxx
3. Default in code        →  Field(default=None)
```

Pydantic automatically:
- Validates types (string stays string, int stays int)
- Raises errors at startup if config is invalid
- Documents all settings in one place

### Key Settings

| Setting | Type | Default | What It Controls |
|---------|------|---------|------------------|
| `LLM_FALLBACK_ORDER` | str | zai,gemini,ollama | Which models to try, in order |
| `LLM_TEMPERATURE` | float | 0.3 | How creative the LLM is (0=deterministic) |
| `RETRY_MAX_ATTEMPTS` | int | 3 | How many times to retry failed API calls |
| `CACHE_QUOTE_TTL_SEC` | int | 60 | How long to cache stock quotes |
| `GUARDRAILS_ENABLED` | bool | True | Whether to validate LLM outputs |
| `PROMPT_VERSION` | str | v1 | Which prompt template to use |
| `REDIS_URL` | str | None | Redis URL (None = use in-memory cache) |

---

## 2. Structured Logging

**File:** `src/logging_config.py`

**Interview Question:** "How do you debug production AI issues?"

### What It Does

Replaces `print()` and basic `logging` with structured, machine-readable logs:

```python
from logging_config import get_logger
logger = get_logger(__name__)

# Instead of: print(f"Analysis done for AAPL in 1.2s")
# You write:
logger.info("analysis_complete",
    symbol="AAPL",
    signal="buy",
    confidence=0.85,
    model="zai/glm-5",
    latency_sec=1.2,
    tokens_used=1550,
    cost_usd=0.004
)
```

### Output Formats

**Development** (`LOG_JSON=false`):
```
2025-02-08 10:30:00 [info] analysis_complete  symbol=AAPL signal=buy confidence=0.85 latency_sec=1.2
```

**Production** (`LOG_JSON=true`):
```json
{
  "event": "analysis_complete",
  "level": "info",
  "timestamp": "2025-02-08T10:30:00Z",
  "symbol": "AAPL",
  "signal": "buy",
  "confidence": 0.85,
  "latency_sec": 1.2
}
```

### Why JSON Logs?

JSON logs can be ingested by:
- **Datadog** - Search `symbol:AAPL AND signal:buy`
- **ELK Stack** - Visualize latency over time
- **CloudWatch** - Alert on `level:error`
- **Grafana Loki** - Correlate with Prometheus metrics

---

## 3. Prometheus Metrics

**File:** `src/metrics.py`

**Interview Question:** "How do you monitor your AI system in production?"

### What It Does

Tracks quantitative metrics about every aspect of the system:

```python
from metrics import LLM_LATENCY, LLM_REQUESTS, ANALYSIS_CONFIDENCE, API_COST

# After each LLM call:
LLM_LATENCY.labels(model="zai").observe(1.2)          # Histogram
LLM_REQUESTS.labels(model="zai", status="success").inc()  # Counter
API_COST.labels(provider="zai").inc(0.004)             # Counter

# After each analysis:
ANALYSIS_CONFIDENCE.observe(0.85)                       # Histogram
```

### Metrics Available

| Metric | Type | Labels | What It Tracks |
|--------|------|--------|----------------|
| `stockradar_llm_latency_seconds` | Histogram | model | How fast each LLM responds |
| `stockradar_llm_requests_total` | Counter | model, status | Success vs failure rate |
| `stockradar_llm_fallback_total` | Counter | from_model, to_model | Fallback frequency |
| `stockradar_llm_tokens_total` | Counter | direction, model | Token consumption |
| `stockradar_analysis_confidence` | Histogram | - | Confidence distribution |
| `stockradar_analysis_signal_total` | Counter | signal, mode | Signal distribution |
| `stockradar_api_cost_usd_total` | Counter | provider | Running cost |
| `stockradar_cache_hits_total` | Counter | cache_type | Cache effectiveness |
| `stockradar_guardrail_triggers_total` | Counter | rule, action | How often LLM misbehaves |

### Visualization

Expose metrics at `/metrics` endpoint, then connect Grafana:

```
Prometheus (scrapes /metrics every 15s)
    ↓
Grafana (dashboards & alerts)
    ↓
Alert: "LLM latency p95 > 10s" → Slack notification
```

---

## 4. Retry Logic

**File:** Already in `src/agents/fetcher.py`, enhanced across the system

**Interview Question:** "How do you handle API failures?"

### What It Does

Wraps every external API call with automatic retry + exponential backoff:

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),        # Try 3 times max
    wait=wait_exponential(min=1, max=10),  # Wait 1s, 2s, 4s between retries
)
def fetch_quote(symbol: str):
    response = requests.get(f"https://api.example.com/quote/{symbol}")
    response.raise_for_status()  # Raises on 4xx/5xx
    return response.json()
```

### Retry Timeline

```
Attempt 1: Call API
    ↓ (fails with timeout)
Wait 1 second
Attempt 2: Call API
    ↓ (fails with 503)
Wait 2 seconds
Attempt 3: Call API
    ↓ (succeeds!)
Return result
```

### Where Retries Are Applied

| Component | What Retries | Max Attempts |
|-----------|-------------|--------------|
| Fetcher | Twelvedata, Finnhub, Yahoo Finance | 3 |
| Analyzer | LLM calls (Groq, Gemini, Ollama) | Fallback chain |
| Storage | Supabase writes | 3 |
| Embeddings | Cohere API | 3 |

---

## 5. Evaluation Framework

**Files:** `src/eval/runner.py`, `src/eval/metrics.py`, `data/eval_signals.jsonl`

**Interview Question:** "How do you measure AI quality?"

### What It Does

Runs your analysis pipeline against a test dataset and measures:

| Metric | What It Measures | Good Score |
|--------|-----------------|------------|
| Signal Exact Accuracy | Did it predict the exact signal? | > 50% |
| Signal Direction Accuracy | Did it get bullish/bearish right? | > 70% |
| Score In Range | Is composite score in expected range? | > 60% |
| Confidence Calibration | Is confidence correlated with accuracy? | Monotonic |
| Latency p50/p95 | How fast is the pipeline? | p50 < 5s |

### Dataset Format

Create `data/eval_signals.jsonl` (one JSON per line):

```json
{"symbol": "AAPL", "expected_signal": "buy", "expected_score_range": [55, 85], "mode": "intraday"}
{"symbol": "TSLA", "expected_signal": "hold", "expected_score_range": [30, 65], "mode": "intraday"}
```

### Running Evaluation

```bash
cd src
python -m eval.runner ../data/eval_signals.jsonl --output eval_results.json
```

### Sample Output

```
[1/8] AAPL (expected: buy)... PASS (predicted=buy, score=68, 2.1s)
[2/8] MSFT (expected: buy)... PASS (predicted=strong_buy, score=72, 1.8s)
[3/8] GOOGL (expected: hold)... FAIL (predicted=buy, score=62, 2.3s)

============================================================
EVALUATION SUMMARY
============================================================
Total: 8 | Success: 8 | Errors: 0
Signal Exact Accuracy:     62.5%
Signal Direction Accuracy:  87.5%
Score In Expected Range:    75.0%
Latency: p50=2.1s, p95=3.4s
```

### When to Run

- After changing prompts → Did quality improve?
- After swapping models → Is the new model better?
- After code changes → Did anything break?
- Weekly → Track trends over time

---

## 6. Token Accounting & Cost Estimation

**File:** `src/token_accounting.py`

**Interview Question:** "How do you manage AI costs at scale?"

### What It Does

Every API response includes a `meta` block with full cost transparency:

```json
{
  "analysis": { "signal": "buy", "confidence": 0.85, ... },
  "meta": {
    "tokens_in": 1200,
    "tokens_out": 350,
    "total_tokens": 1550,
    "cost_usd": 0.004,
    "model_used": "zai/glm-5",
    "models_tried": 1,
    "latency_sec": 1.2,
    "data_sources_used": ["yfinance", "finnhub"],
    "llm_calls": [
      {
        "model": "zai/glm-5",
        "tokens_in": 1200,
        "tokens_out": 350,
        "cost_usd": 0.004,
        "latency_sec": 1.2
      }
    ]
  }
}
```

### Usage

```python
from token_accounting import TokenAccountant

# Create per-request
accountant = TokenAccountant()
accountant.set_mode("intraday")

# Record LLM call
accountant.record_llm_call(
    model="zai/glm-5",
    tokens_in=1200,
    tokens_out=350,
    latency_sec=1.2,
)

# Get the meta block
meta = accountant.get_request_meta()
```

### Cost Rates

| Model | Input (per 1K) | Output (per 1K) |
|-------|---------------|-----------------|
| Groq Llama 3.3 70B | $0.00059 | $0.00079 |
| Gemini 2.0 Flash | Free | Free |
| Ollama (local) | Free | Free |

---

## 7. Streaming LLM Responses

**File:** `src/streaming.py`

**Interview Question:** "Have you built real-time AI features?"

### What It Does

Instead of waiting for the full LLM response, tokens stream to the frontend
in real-time using Server-Sent Events (SSE):

```
User clicks "Analyze AAPL"
    ↓ (instant)
"Fetching data for AAPL..."
    ↓ (1 second)
"Calculating indicators... Price: $185.50"
    ↓ (instant)
"Starting AI analysis..."
    ↓ (streaming tokens)
"Based on the current RSI of 62 and..."
"...bullish MACD crossover, the short-term..."
"...outlook is positive. The stock is trading..."
    ↓ (complete)
{type: "done", model: "zai", tokens: 350, latency: 1.2}
```

### Usage

```python
from streaming import stream_analysis_sse

# In your HTTP handler:
for event in stream_analysis_sse("AAPL", mode="intraday"):
    response.write(event)  # SSE format: "data: {...}\n\n"
```

### SSE Event Types

| Event Type | When | Example |
|-----------|------|---------|
| `status` | Data fetching phases | `{"phase": "fetching", "message": "..."}` |
| `start` | LLM begins generating | `{"model": "zai/glm-5"}` |
| `token` | Each token arrives | `{"content": "Based on"}` |
| `fallback` | Model failed, trying next | `{"failed_model": "zai", "error": "..."}` |
| `done` | Generation complete | `{"tokens": 350, "latency_sec": 1.2}` |
| `error` | All models failed | `{"message": "..."}` |

---

## 8. LLM Guardrails

**File:** `src/guardrails.py`

**Interview Question:** "How do you handle hallucinations?"

### What It Does

Validates EVERY LLM output before returning it. Six rules:

| Rule | What It Catches | Action |
|------|----------------|--------|
| Schema Validation | Missing required fields | Blocks response |
| Signal Validity | Invalid signal strings | Auto-corrects to "hold" |
| Confidence Bounds | Confidence > 0.95 or < 0 | Caps/floors the value |
| Price Sanity | Target price 500% away from current | Warns |
| Reasoning Required | Empty or too-short reasoning | Warns |
| Consistency Check | Bullish signal with bearish reasoning | Warns |

### Usage

```python
from guardrails import GuardrailEngine

engine = GuardrailEngine(max_confidence=0.95)

result = engine.validate(
    llm_output={"signal": "STRONG BUY", "confidence": 1.5, "reasoning": ""},
    current_price=150.0,
    mode="intraday",
)

print(result.passed)    # False
print(result.issues)    # [
                        #   {rule: "confidence_capped", action: "adjusted", 1.5 -> 0.95},
                        #   {rule: "signal_normalized", action: "adjusted", "STRONG BUY" -> "strong_buy"},
                        #   {rule: "reasoning_too_short", action: "warned"},
                        # ]
print(result.adjusted)  # The corrected output with confidence=0.95, signal="strong_buy"
```

### Why This Matters

Without guardrails, an LLM might:
- Return confidence of 1.5 (impossible)
- Say "STRONGLY_BUY" (not a valid signal)
- Set target price at $1000 for a $50 stock
- Give a "buy" signal while explaining why the stock is terrible
- Return empty reasoning

Guardrails catch ALL of these automatically.

---

## 9. Prompt Versioning & A/B Testing

**File:** `src/prompt_manager.py`

**Interview Question:** "How do you iterate on prompts?"

### What It Does

Manages versioned prompt templates. Instead of editing strings in code,
prompts are organized by name and version:

```python
from prompt_manager import prompt_manager

# Get active version (set by PROMPT_VERSION env var)
prompt = prompt_manager.get_prompt(
    "intraday_analysis",
    symbol="AAPL",
    price=185.50,
    rsi_14=62,
    # ... other values
)

# Returns: {"system": "...", "user": "..."}
```

### Versions Available

| Prompt | v1 | v2 |
|--------|----|----|
| `intraday_analysis` | Basic plain-text format | Markdown with step-by-step instructions |
| `longterm_analysis` | Basic format | Structured with weighting rules |
| `algo_explanation` | Simple | - |

### A/B Testing

```python
# Run same analysis with two prompt versions
prompt_v1 = prompt_manager.get_prompt("intraday_analysis", version="v1", **data)
prompt_v2 = prompt_manager.get_prompt("intraday_analysis", version="v2", **data)

# Compare results
result_v1 = analyzer.analyze(prompt_v1)
result_v2 = analyzer.analyze(prompt_v2)

# Track which produces better signals over time
```

### Adding a New Version

Add to the `PROMPTS` dict in `prompt_manager.py`:

```python
"intraday_analysis": {
    "v1": { ... },
    "v2": { ... },
    "v3": {  # Your new version
        "system": "New system prompt...",
        "user": "New user prompt with {symbol} and {price}...",
    },
},
```

Then set `PROMPT_VERSION=v3` in your `.env` file.

---

## 10. Caching Layer

**File:** `src/cache.py`

**Interview Question:** "How do you optimize performance and reduce costs?"

### What It Does

Caches expensive operations to avoid redundant API calls:

```python
from cache import cache

# Check cache first
quote = cache.get_quote("AAPL")
if quote is None:
    quote = fetch_from_api("AAPL")   # Expensive API call
    cache.set_quote("AAPL", quote)    # Cache for 60 seconds
```

### Cache Strategy

| Data Type | TTL | Why |
|-----------|-----|-----|
| Stock Quotes | 60 seconds | Prices change frequently |
| Fundamentals | 1 hour | P/E, ROE change slowly |
| Analysis Results | 15 minutes | Same prompt = same answer |
| Embeddings | 24 hours | Same text = same vector |
| Indicators | 5 minutes | Recalculated from prices |

### Backends

| Backend | When | Setup |
|---------|------|-------|
| In-Memory (default) | Development, testing | No setup needed |
| Redis | Production | Set `REDIS_URL=redis://localhost:6379` |

The system auto-selects: if `REDIS_URL` is set, uses Redis. Otherwise, in-memory.

### Cache Metrics

Prometheus tracks hit/miss rates:

```
stockradar_cache_hits_total{cache_type="quote"} 45
stockradar_cache_misses_total{cache_type="quote"} 12
# Hit rate = 45/(45+12) = 78.9%
```

---

## How These Features Connect

```
User Request: "Analyze AAPL"
    │
    ├── [Config] Load settings from .env
    ├── [Cache] Check if recent analysis exists
    │     ├── HIT  → Return cached result
    │     └── MISS → Continue pipeline
    │
    ├── [Logging] Log "analysis_started" with structured fields
    ├── [Metrics] Start latency timer
    │
    ├── [Data Fetch] Get quote, indicators, news
    │     ├── [Cache] Check quote cache (60s TTL)
    │     ├── [Retry] Auto-retry on timeout (3 attempts)
    │     └── [Metrics] Record fetch latency
    │
    ├── [Prompt Manager] Get versioned prompt template
    │
    ├── [LLM Call] Send to Groq → Gemini → Ollama (fallback)
    │     ├── [Streaming] Stream tokens to frontend via SSE
    │     ├── [Retry] Exponential backoff on transient errors
    │     ├── [Token Accounting] Record tokens in/out, calculate cost
    │     └── [Metrics] Record LLM latency, model used
    │
    ├── [Guardrails] Validate LLM output
    │     ├── Check schema, signal, confidence, prices
    │     ├── Auto-correct where possible
    │     └── [Metrics] Record guardrail triggers
    │
    ├── [Cache] Store analysis result (15 min TTL)
    ├── [Logging] Log "analysis_complete" with all details
    ├── [Metrics] Record total duration, signal type, confidence
    │
    └── Response to user:
          {
            "analysis": { signal, confidence, reasoning, ... },
            "meta": { tokens, cost, latency, model, sources }
          }
```

---

## Quick Start

### 1. Install new dependencies

```bash
pip install -r requirements.txt
```

### 2. Run evaluation

```bash
cd src
python -m eval.runner ../data/eval_signals.jsonl --output eval_results.json
```

### 3. Add to your .env

```bash
# Optional - these have sensible defaults
LOG_JSON=false
METRICS_ENABLED=true
GUARDRAILS_ENABLED=true
PROMPT_VERSION=v1
REDIS_URL=                    # Leave empty for in-memory cache
```

---

## File Summary

| File | Feature | Lines |
|------|---------|-------|
| `src/config.py` | Centralized Config | ~150 |
| `src/logging_config.py` | Structured Logging | ~90 |
| `src/metrics.py` | Prometheus Metrics | ~120 |
| `src/token_accounting.py` | Token & Cost Tracking | ~150 |
| `src/guardrails.py` | LLM Output Validation | ~280 |
| `src/cache.py` | Redis/In-Memory Cache | ~200 |
| `src/prompt_manager.py` | Prompt Versioning | ~300 |
| `src/streaming.py` | SSE Streaming | ~200 |
| `src/eval/metrics.py` | Quality Metrics | ~120 |
| `src/eval/runner.py` | Evaluation Runner | ~220 |
| `data/eval_signals.jsonl` | Test Dataset | 8 cases |
