# The Complete AI Engineering Guide

**For aspiring AI Engineers — from fundamentals to production systems**

*This guide covers every topic you need to know to land an AI Engineering role. Each section explains the concept in depth, why it matters, and uses the Stock-Radar project as a real-world reference wherever applicable.*

---

## Table of Contents

1. [Python Foundations for AI Engineering](#1-python-foundations-for-ai-engineering)
2. [Working with LLMs (Large Language Models)](#2-working-with-llms-large-language-models)
3. [Prompt Engineering](#3-prompt-engineering)
4. [RAG (Retrieval-Augmented Generation)](#4-rag-retrieval-augmented-generation)
5. [Embeddings & Vector Databases](#5-embeddings--vector-databases)
6. [LLM Output Guardrails & Validation](#6-llm-output-guardrails--validation)
7. [Streaming LLM Responses](#7-streaming-llm-responses)
8. [Token Accounting & Cost Management](#8-token-accounting--cost-management)
9. [Traditional ML in AI Systems](#9-traditional-ml-in-ai-systems)
10. [Feature Engineering](#10-feature-engineering)
11. [Model Training & Hyperparameter Tuning](#11-model-training--hyperparameter-tuning)
12. [Model Serving & Inference](#12-model-serving--inference)
13. [Evaluation Frameworks](#13-evaluation-frameworks)
14. [Agentic AI & Tool Use](#14-agentic-ai--tool-use)
15. [Fine-Tuning LLMs](#15-fine-tuning-llms)
16. [Production Configuration Management](#16-production-configuration-management)
17. [Structured Logging & Observability](#17-structured-logging--observability)
18. [Metrics & Monitoring (Prometheus + Grafana)](#18-metrics--monitoring-prometheus--grafana)
19. [Caching Strategies](#19-caching-strategies)
20. [Retry Logic & Resilience](#20-retry-logic--resilience)
21. [API Development (FastAPI)](#21-api-development-fastapi)
22. [Data Engineering for AI](#22-data-engineering-for-ai)
23. [Containerization & Docker](#23-containerization--docker)
24. [CI/CD for AI Projects](#24-cicd-for-ai-projects)
25. [Testing AI Systems](#25-testing-ai-systems)
26. [Experiment Tracking](#26-experiment-tracking)
27. [System Design for AI Applications](#27-system-design-for-ai-applications)
28. [Security & Safety in AI Systems](#28-security--safety-in-ai-systems)
29. [Common Interview Topics & Questions](#29-common-interview-topics--questions)
30. [Learning Roadmap & Resources](#30-learning-roadmap--resources)

---

## 1. Python Foundations for AI Engineering

### Why Python?

Python dominates AI/ML because of its ecosystem: NumPy, pandas, scikit-learn, PyTorch, TensorFlow, HuggingFace Transformers, LangChain, and virtually every LLM SDK is Python-first. As an AI engineer, Python is non-negotiable.

### Core Language Features You Must Know

#### Type Hints

Modern Python AI code is typed. Every serious AI codebase uses type hints for readability, IDE support, and catching bugs early.

```python
# Stock-Radar uses type hints extensively
from typing import Optional, Dict, Any, List

def analyze_stock(
    symbol: str,
    mode: str = "intraday",
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    ...
```

**Stock-Radar example** — from `src/agents/fetcher.py`:
```python
@dataclass
class StockQuote:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    avg_volume: int
    high: float
    low: float
    open: float
    prev_close: float
    market_cap: Optional[int]
    pe_ratio: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    timestamp: datetime
```

#### Dataclasses

Dataclasses replace verbose `__init__` methods. They are used everywhere in AI systems for structured data:

```python
from dataclasses import dataclass, field

@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    model: str
    tokens_in: int
    tokens_out: int
    latency_sec: float
    cost_usd: float
    timestamp: float = field(default_factory=time.time)
```

**Stock-Radar** uses dataclasses for: `StockQuote`, `PriceData`, `NewsItem`, `AlgorithmicScores`, `GuardrailIssue`, `GuardrailResult`, `RAGContext`, `ChatMessage`, `LLMCallRecord`, `EmbeddingResult`, and many more.

#### Enums

Enums enforce valid values and prevent magic strings:

```python
from enum import Enum

class TradingMode(str, Enum):
    INTRADAY = "intraday"
    LONGTERM = "longterm"

class Signal(str, Enum):
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
```

#### Async Python

Production AI systems are I/O heavy (API calls, database queries, LLM calls). Async Python lets you handle many concurrent requests without blocking:

```python
import asyncio
import aiohttp

async def fetch_multiple_quotes(symbols: list[str]) -> list[dict]:
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_one(session, s) for s in symbols]
        return await asyncio.gather(*tasks)
```

**In Stock-Radar:** The project uses some async patterns (SSE streaming) but is mostly synchronous. In a production system, you would want `async` endpoints throughout — this is a good area for improvement.

**What could be used:** `asyncio`, `aiohttp`, `httpx` (async HTTP client), `asyncpg` (async Postgres).

---

## 2. Working with LLMs (Large Language Models)

### What Is an LLM?

An LLM is a neural network trained on massive text corpora that can generate, summarize, classify, and reason about text. As an AI engineer, you don't train these models — you integrate them into applications.

### Calling LLMs via API

The fundamental operation: send a prompt, get a response. Every LLM provider (OpenAI, Anthropic, Google, Groq, etc.) exposes a chat completions API:

```python
from litellm import completion

response = completion(
    model="gemini/gemini-2.5-flash",
    messages=[
        {"role": "system", "content": "You are a stock analyst."},
        {"role": "user", "content": "Analyze AAPL for intraday trading."},
    ],
    temperature=0.3,
    max_tokens=2000,
)

text = response.choices[0].message.content
```

**Stock-Radar** uses `litellm` — a unified interface that lets you call any LLM provider with the same code. This is a key pattern: **abstract your LLM provider so you can swap models without changing application code.**

### Multi-Model Fallback Chains

In production, a single LLM provider will fail. Rate limits, outages, and timeouts are inevitable. The solution: a fallback chain.

**Stock-Radar implementation** (from `src/agents/analyzer.py`):
```python
# Fallback order: try ZAI first, then Gemini, then Ollama
for model in self.fallback_models:
    try:
        response = completion(
            model=model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response  # Success — stop trying
    except Exception as e:
        logger.warning(f"Model {model} failed: {e}")
        LLM_FALLBACK.labels(from_model=prev_model, to_model=model).inc()
        continue

raise RuntimeError("All models failed")
```

The fallback order is configurable via environment:
```
LLM_FALLBACK_ORDER=openai/glm-4.7,gemini/gemini-2.5-flash
```

### Task-Based Model Routing

Different tasks may benefit from different models. A cheap, fast model might be fine for sentiment analysis, while complex financial analysis needs a more capable model.

**Stock-Radar** supports this via `LLM_TASK_ROUTES`:
```
LLM_TASK_ROUTES=analysis=openai/glm-4.7,gemini/gemini-2.5-flash;chat=groq/llama-3.1-70b-versatile;sentiment=groq/llama-3.1-8b-instant
```

### Key Concepts to Know

| Concept | What It Is |
|---------|-----------|
| **Temperature** | Controls randomness. 0 = deterministic, 1 = creative. Stock-Radar uses 0.3 for consistency. |
| **Max Tokens** | Limits response length. Stock-Radar uses 2000. |
| **System Prompt** | Sets the LLM's persona and rules. Stays constant across requests. |
| **User Prompt** | The specific request with data. Changes per request. |
| **Token** | A word fragment (~4 chars in English). LLMs charge per token. |
| **Context Window** | The max tokens (input + output) a model can process. |
| **Stop Sequences** | Strings that tell the LLM to stop generating. |

---

## 3. Prompt Engineering

### Why It Matters

Prompts are the "source code" of AI applications. A single word change can dramatically alter output quality. As an AI engineer, prompt engineering is a daily skill.

### Prompt Versioning

Never edit prompts as inline strings. Use a versioned prompt management system.

**Stock-Radar implementation** (from `src/prompt_manager.py`):

```python
PROMPTS = {
    "intraday_analysis": {
        "v1": {
            "system": "You are an expert intraday stock trader...",
            "user": "Analyze this stock: {symbol}\nPrice: {price}\nRSI: {rsi_14}..."
        },
        "v2": {
            "system": "You are a senior quantitative trader. RULES:\n"
                      "1. Only use data provided - do not hallucinate.\n"
                      "2. If data insufficient, signal 'hold' with low confidence.\n"
                      "3. Cite specific indicator values in reasoning.",
            "user": "## Intraday Analysis Request\n**Stock:** {symbol}..."
        },
    },
}
```

**Key differences between v1 and v2:**
- v2 adds explicit anti-hallucination rules
- v2 uses markdown formatting (LLMs respond better to structured input)
- v2 requires step-by-step reasoning
- v2 demands citing specific data values

### A/B Testing Prompts

Run the same analysis with two prompt versions and compare results:

```python
from prompt_manager import prompt_manager

prompt_v1 = prompt_manager.get_prompt("intraday_analysis", version="v1", **data)
prompt_v2 = prompt_manager.get_prompt("intraday_analysis", version="v2", **data)

result_v1 = analyzer.analyze(prompt_v1)
result_v2 = analyzer.analyze(prompt_v2)

# Compare: which version produces more accurate signals?
```

The active version is controlled by an environment variable:
```
PROMPT_VERSION=v2
```

### Prompt Engineering Techniques

| Technique | Description | Example |
|-----------|------------|---------|
| **Few-Shot** | Give examples of desired output format | "Here's an example: {example}" |
| **Chain-of-Thought** | Ask the model to reason step by step | "Think step-by-step, then respond." |
| **Role Assignment** | Give the model a specific persona | "You are a senior quantitative trader." |
| **Output Formatting** | Specify exact JSON schema | "Respond with this exact JSON format: {...}" |
| **Constraint Setting** | Explicit rules to prevent errors | "Only use data provided. Do not hallucinate." |
| **Structured Input** | Use markdown tables, headers | Stock-Radar v2 prompts use markdown tables |

### What Could Be Used

- **LangSmith** or **Braintrust** for prompt experiment tracking
- **DSPy** for programmatic prompt optimization
- **Prompt templates with Jinja2** for more complex templating logic

---

## 4. RAG (Retrieval-Augmented Generation)

### What Is RAG?

RAG = Retrieval-Augmented Generation. Instead of relying solely on an LLM's training data (which is frozen at a cutoff date), you **retrieve relevant information** from your own data sources and inject it into the prompt as context.

```
Without RAG:
  User: "What's AAPL's recent performance?"
  LLM: "As of my last update..." (stale data)

With RAG:
  1. Retrieve AAPL's latest analyses from database
  2. Retrieve similar historical technical setups
  3. Inject this context into the prompt
  4. LLM: "Based on the last 5 analyses and similar RSI patterns..." (current + grounded)
```

### RAG Pipeline

```
User Query
    |
    v
[1. RETRIEVAL] --> Search vector DB, keyword search, SQL queries
    |
    v
[2. CONTEXT BUILDING] --> Format retrieved docs into prompt context
    |
    v
[3. GENERATION] --> LLM generates answer using retrieved context
    |
    v
[4. VALIDATION] --> Check if response is grounded in retrieved context
```

### Stock-Radar RAG Implementation

**`src/agents/rag_retriever.py`** retrieves context from multiple sources:

```python
@dataclass
class RAGContext:
    """Container for retrieved RAG context."""
    query: str
    stock_symbol: Optional[str] = None

    # Retrieved content by source
    similar_analyses: List[Dict[str, Any]] = field(default_factory=list)
    similar_signals: List[Dict[str, Any]] = field(default_factory=list)
    relevant_news: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_entries: List[Dict[str, Any]] = field(default_factory=list)
    similar_conversations: List[Dict[str, Any]] = field(default_factory=list)

    # Stock-specific context
    recent_analyses: List[Dict[str, Any]] = field(default_factory=list)
    recent_signals: List[Dict[str, Any]] = field(default_factory=list)
    technical_indicators: Optional[Dict[str, Any]] = None

    # Metadata
    total_results: int = 0
    retrieval_time_ms: int = 0
    sources_searched: List[str] = field(default_factory=list)
```

**`src/agents/rag_validator.py`** validates that the LLM's response is actually grounded in the retrieved context — it checks whether the LLM is using the data you gave it or making things up.

### RAG Configuration in Stock-Radar

```python
# From src/config.py
top_k_rag: int = Field(default=5, alias="TOP_K_RAG")
rag_match_threshold: float = Field(default=0.4, alias="RAG_MATCH_THRESHOLD")
```

- `TOP_K_RAG=5` — retrieve the 5 most similar documents
- `RAG_MATCH_THRESHOLD=0.4` — minimum similarity score to include a result

### Advanced RAG Techniques (What Could Be Used)

| Technique | Description |
|-----------|------------|
| **Hybrid Search** | Combine vector similarity with keyword (BM25) search for better recall |
| **Re-Ranking** | After initial retrieval, use a cross-encoder model to re-rank results by relevance |
| **Chunking Strategies** | Split documents into overlapping chunks (512 tokens with 50-token overlap is common) |
| **Query Expansion** | Rewrite the user's query into multiple variations to improve retrieval |
| **HyDE** | Generate a hypothetical answer, then use its embedding to search for real documents |
| **RAPTOR** | Hierarchical chunking — summarize chunks into clusters, enabling multi-level retrieval |
| **Contextual Compression** | After retrieval, compress/filter documents to keep only the most relevant parts |

---

## 5. Embeddings & Vector Databases

### What Are Embeddings?

Embeddings are numerical representations (vectors) of text. Similar texts produce similar vectors. This enables **semantic search** — finding content by meaning, not just keywords.

```
"AAPL is showing bullish momentum"  -->  [0.23, -0.14, 0.87, ...]  (1024 dimensions)
"Apple stock is trending upward"     -->  [0.21, -0.12, 0.85, ...]  (very similar vector!)
"The weather is nice today"          -->  [0.91, 0.33, -0.54, ...]  (very different vector)
```

### How Stock-Radar Uses Embeddings

**`src/agents/storage.py`** generates embeddings using Cohere and stores them in Supabase (PostgreSQL + pgvector):

```python
@dataclass
class EmbeddingResult:
    vector: list[float]
    provider: str
    model: str
    dimension: int

# Configuration
embedding_provider: str = Field(default="cohere", alias="EMBEDDING_PROVIDER")
cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
embedding_model: str = Field(default="embed-english-v3.0", alias="EMBEDDING_MODEL")
embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")
```

Every analysis, signal, and conversation is embedded and stored as a vector. When the RAG retriever searches for "similar analyses," it computes the cosine similarity between the query embedding and stored analysis embeddings.

### Vector Search: How It Works

```
1. Embed the query: "AAPL momentum analysis" --> [0.23, -0.14, 0.87, ...]

2. Find nearest neighbors in the vector database:
   SELECT * FROM analyses
   ORDER BY embedding <=> query_embedding  -- cosine distance
   LIMIT 5;

3. Return the 5 most similar analyses
```

### Vector Database Options

| Database | Type | When to Use |
|----------|------|------------|
| **pgvector** (Supabase) | Postgres extension | Already using Postgres; moderate scale. **Used in Stock-Radar.** |
| **Pinecone** | Managed SaaS | Serverless, auto-scaling; production RAG |
| **Weaviate** | Open source | Hybrid search (vector + keyword); self-hosted |
| **Qdrant** | Open source | High performance; filtering support |
| **ChromaDB** | Open source | Quick prototyping; in-memory option |
| **FAISS** | Library (Meta) | Billions of vectors; research / offline batch |
| **Milvus** | Open source | Enterprise scale; GPU acceleration |

### Embedding Models

| Model | Provider | Dimensions | Best For |
|-------|----------|-----------|----------|
| `embed-english-v3.0` | Cohere | 1024 | General purpose. **Used in Stock-Radar.** |
| `text-embedding-3-large` | OpenAI | 3072 | Highest quality (OpenAI ecosystem) |
| `voyage-3` | Voyage AI | 1024 | Code + text mixed retrieval |
| `bge-m3` | BAAI | 1024 | Open source; multilingual |
| `nomic-embed-text` | Nomic | 768 | Open source; runs locally |

---

## 6. LLM Output Guardrails & Validation

### The Problem

LLMs hallucinate. They make up numbers, invent data, return invalid formats, and contradict themselves. Every production AI system needs a validation layer between the LLM and the user.

**Interview question:** "How do you handle hallucinations?"

### Stock-Radar's GuardrailEngine

**`src/guardrails.py`** validates every LLM response with 7 rules:

```python
class GuardrailEngine:
    def validate(self, llm_output, current_price=None, mode="intraday"):
        issues = []
        adjusted = dict(llm_output)  # Work on a copy

        issues.extend(self._check_schema(adjusted))      # 1. Required fields exist?
        issues.extend(self._check_signal(adjusted))       # 2. Valid signal string?
        issues.extend(self._check_confidence(adjusted))   # 3. Confidence in [0, 0.95]?
        issues.extend(self._check_prices(adjusted, ...))  # 4. Target price sane?
        issues.extend(self._check_reasoning(adjusted))    # 5. Reasoning non-empty?
        issues.extend(self._check_consistency(adjusted))  # 6. Signal matches reasoning?

        # Record metrics for every guardrail trigger
        for issue in issues:
            GUARDRAIL_TRIGGERS.labels(rule=issue.rule, action=issue.action).inc()

        return GuardrailResult(passed=passed, issues=issues, adjusted=adjusted)
```

### What Each Rule Catches

**Rule 1: Schema Validation** — Does the response have `signal`, `confidence`, and `reasoning`?
```python
# LLM might return: {"recommendation": "buy"} (wrong key name)
# Guardrail: BLOCKED — missing required field "signal"
```

**Rule 2: Signal Validity** — Is the signal one of the 5 valid values?
```python
# LLM returns: {"signal": "STRONG BUY"}
# Guardrail: AUTO-CORRECTED to "strong_buy" (normalized)

# LLM returns: {"signal": "maybe buy"}
# Guardrail: AUTO-CORRECTED to "hold" (unrecognizable, default to safe)
```

**Rule 3: Confidence Bounds** — Is confidence realistic?
```python
# LLM returns: {"confidence": 1.5}
# Guardrail: CAPPED to 0.95 (max allowed)

# LLM returns: {"confidence": -0.3}
# Guardrail: FLOORED to 0.0
```

**Rule 4: Price Sanity** — Is the target price within 50% of current price?
```python
# Current price: $150, LLM says target: $800
# Guardrail: WARNING — target is 433% from current price
```

**Rule 5: Reasoning Required** — Is the reasoning at least 20 characters?
```python
# LLM returns: {"reasoning": "Buy it."}
# Guardrail: WARNING — reasoning too short (7 chars)
```

**Rule 6: Consistency Check** — Does the signal direction match the reasoning sentiment?
```python
# LLM returns signal="strong_buy" but reasoning mentions
# "bearish", "downtrend", "overvalued", "weak" (more bearish words than bullish)
# Guardrail: WARNING — signal-reasoning mismatch
```

### Guardrail Actions

| Action | Meaning |
|--------|---------|
| `blocked` | Response rejected entirely, cannot be served |
| `adjusted` | Response auto-corrected (e.g., confidence capped) |
| `warned` | Issue logged but response still served |

### Guardrail Metrics

Every trigger is recorded as a Prometheus metric:
```python
GUARDRAIL_TRIGGERS.labels(rule="confidence_capped", action="adjusted").inc()
```

This lets you track: "How often does our LLM return unrealistic confidence scores?" and answer: "It happened 47 times today, down from 120 yesterday after we improved the prompt."

### What Could Be Used

- **Guardrails AI** — open-source library with pre-built validators (PII detection, toxicity, etc.)
- **NeMo Guardrails** (NVIDIA) — programmable guardrail framework with dialog flows
- **Pydantic output parsers** — use Pydantic models to validate and parse LLM JSON output
- **LLM-as-judge** — use a second LLM to validate the first LLM's output

---

## 7. Streaming LLM Responses

### Why Streaming Matters

Without streaming: user waits 5-10 seconds staring at a loading spinner, then gets the full response at once.

With streaming: tokens appear in real-time (like ChatGPT/Claude), giving the user immediate feedback.

**Interview question:** "Have you built real-time AI features?"

### Server-Sent Events (SSE)

SSE is the standard protocol for streaming LLM responses. It's simpler than WebSockets — the server pushes events to the client over a long-lived HTTP connection.

```
Client                          Server
  |--- GET /stream --->            |
  |    Accept: text/event-stream   |
  |                                |
  |<-- data: {"type":"start"} -----|  (connection stays open)
  |<-- data: {"type":"token","content":"Based"} ---|
  |<-- data: {"type":"token","content":" on"} -----|
  |<-- data: {"type":"token","content":" the"} ----|
  |<-- data: {"type":"done"} ------|
  |<-- connection closed ----------|
```

### Stock-Radar Streaming Implementation

**`src/streaming.py`** implements a complete SSE streaming pipeline:

```python
def stream_llm_response(prompt, system_prompt="", models=None, ...):
    """Stream LLM response token by token with model fallback."""

    for model in models:
        try:
            # Tell litellm to stream
            response = litellm.completion(
                model=model, messages=messages, stream=True
            )

            yield {"type": "start", "model": model}

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield {"type": "token", "content": delta.content}

            yield {"type": "done", "model": model, "tokens": total_tokens}
            return  # Success

        except Exception as e:
            yield {"type": "fallback", "failed_model": model, "error": str(e)}
            continue

    yield {"type": "error", "message": "All models failed"}
```

### SSE Event Types

| Event | When | Data |
|-------|------|------|
| `status` | Data fetching phases | `{"phase": "fetching", "message": "Fetching data for AAPL..."}` |
| `start` | LLM begins generating | `{"model": "gemini/gemini-2.5-flash"}` |
| `token` | Each token arrives | `{"content": "Based on"}` |
| `fallback` | Model failed, trying next | `{"failed_model": "zai", "error": "timeout"}` |
| `done` | Generation complete | `{"tokens": 350, "latency_sec": 1.2}` |
| `error` | All models failed | `{"message": "All models failed"}` |

### Frontend Consumption (Next.js)

```javascript
const response = await fetch('/api/stream', {
    method: 'POST',
    body: JSON.stringify({ symbol: 'AAPL' }),
});

const reader = response.body.getReader();
while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    const text = new TextDecoder().decode(value);
    // Parse SSE events and update UI in real-time
}
```

---

## 8. Token Accounting & Cost Management

### The Problem

LLM APIs charge per token. A single GPT-4o request can cost $0.01-$0.10. At 10,000 requests/day, that's $100-$1000/day. Without tracking, costs spiral out of control.

**Interview question:** "How do you manage AI costs at scale?"

### Stock-Radar's TokenAccountant

**`src/token_accounting.py`** tracks every LLM call:

```python
class TokenAccountant:
    """Tracks all LLM calls for a single request/analysis."""

    def record_llm_call(self, model, tokens_in, tokens_out, latency_sec):
        # Calculate cost from rate card
        rates = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        cost = (tokens_in / 1000) * rates["input"] + \
               (tokens_out / 1000) * rates["output"]

        # Record to Prometheus metrics
        LLM_LATENCY.labels(model=provider).observe(latency_sec)
        LLM_REQUESTS.labels(model=provider, status="success").inc()
        LLM_TOKENS.labels(direction="input", model=provider).inc(tokens_in)
        LLM_TOKENS.labels(direction="output", model=provider).inc(tokens_out)
        API_COST.labels(provider=provider).inc(cost)

    def get_request_meta(self):
        """Build the transparency meta block for API responses."""
        return {
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "cost_usd": round(total_cost, 6),
            "model_used": model_used,
            "models_tried": len(self.llm_calls),
            "latency_sec": round(total_latency, 2),
            "data_sources_used": sources_used,
            "llm_calls": [...]  # Detailed per-call breakdown
        }
```

### Every API Response Includes Cost Transparency

```json
{
  "analysis": { "signal": "buy", "confidence": 0.85 },
  "meta": {
    "tokens_in": 1200,
    "tokens_out": 350,
    "total_tokens": 1550,
    "cost_usd": 0.004,
    "model_used": "openai/glm-4.7",
    "models_tried": 1,
    "latency_sec": 1.2,
    "data_sources_used": ["yfinance", "finnhub"],
    "llm_calls": [
      {
        "model": "openai/glm-4.7",
        "tokens_in": 1200,
        "tokens_out": 350,
        "cost_usd": 0.004,
        "latency_sec": 1.2
      }
    ]
  }
}
```

### Cost Optimization Strategies

| Strategy | How | Implemented? |
|----------|-----|:------------:|
| Use cheaper models for simple tasks | Route sentiment to Llama-8B, analysis to GLM-4.7 | Yes (task routing) |
| Cache LLM responses | Same prompt = same answer (for a time window) | Yes (cache layer) |
| Reduce prompt size | Compress context, remove redundant info | Partially |
| Use free-tier models | Gemini Flash, Groq free tier | Yes |
| Set token limits | `max_tokens=2000` prevents runaway responses | Yes |
| Monitor costs in real-time | Prometheus dashboards show running costs | Yes |

---

## 9. Traditional ML in AI Systems

### Why Traditional ML Still Matters

LLMs are powerful but expensive, slow, and non-deterministic. For structured data problems (classification, regression, time series), traditional ML models are:
- 100x-1000x faster
- Deterministic (same input = same output)
- Free (no API costs)
- More interpretable

### Stock-Radar's ML Pipeline

The project has a complete ML training + inference pipeline for signal prediction:

```
Raw Market Data
    |
    v
[Feature Engineering] --> 45 numerical features
    |
    v
[Model Training] --> GradientBoosting classifier (scikit-learn + Optuna)
    |
    v
[Model Registry] --> Versioned model artifacts (joblib)
    |
    v
[Inference] --> SignalPredictor loads model and predicts
    |
    v
[Backtesting] --> Evaluate against historical data
```

### The Hybrid Approach: ML + LLM

Stock-Radar combines both:
1. **ML model** predicts a signal from structured features (fast, cheap, deterministic)
2. **LLM** provides reasoning, explanation, and handles unstructured data (news, sentiment)
3. **Scorer** produces a formula-based composite score (no ML or LLM — pure math)

This is a common production pattern: **use traditional ML for the core prediction, LLM for explanation and unstructured data**.

---

## 10. Feature Engineering

### What It Is

Feature engineering = transforming raw data into numerical features that ML models can learn from. It's often the most impactful part of an ML pipeline.

### Stock-Radar's 45-Feature Vector

**`src/training/feature_engineering.py`** defines features in 4 groups:

**Base Technical Features (20):**
```python
BASE_FEATURE_NAMES = [
    # Technical / Momentum (8)
    "rsi_14", "macd", "macd_signal", "macd_histogram",
    "price_vs_sma20_pct", "price_vs_sma50_pct",
    "bollinger_width_pct", "volume_ratio",
    # Volatility / Risk (4)
    "atr_pct", "adx", "plus_di", "minus_di",
    # Valuation (4)
    "pe_ratio", "pb_ratio", "peg_ratio", "dividend_yield",
    # Quality (4)
    "roe", "current_ratio", "debt_to_equity", "profit_margin",
]
```

**Cross-Sectional Features (17)** — how this stock compares to peers:
- Relative strength vs sector
- Factor scores (momentum, value, quality)
- Microstructure features (bid-ask spread, volume patterns)

**Sentiment Features (8):**
```python
SENTIMENT_FEATURE_NAMES = [
    "news_sentiment_mean",       # Avg FinBERT/VADER score (-1 to +1)
    "news_sentiment_std",        # Std of per-headline scores
    "news_volume_7d",            # log2(1 + articles in last 7 days)
    "news_sentiment_momentum",   # Delta: last 3d avg - last 7d avg
    "finnhub_buzz_score",        # Finnhub buzz metric (0-1)
    "finnhub_bullish_pct",       # Finnhub bullish percent (0-1)
    "earnings_proximity",        # -log2(1 + days to next earnings)
    "sentiment_vs_sector",       # Stock sentiment - sector avg sentiment
]
```

### Feature Engineering Best Practices

| Practice | Example in Stock-Radar |
|----------|----------------------|
| **Normalize** | `price_vs_sma20_pct` instead of raw price (removes scale) |
| **Log transforms** | `news_volume_7d = log2(1 + count)` (compresses skewed data) |
| **Relative features** | `sentiment_vs_sector` (comparison to peers is more informative) |
| **Time-aware features** | `earnings_proximity` (captures calendar effects) |
| **Domain knowledge** | Using RSI, MACD, Bollinger Bands (financial domain expertise) |

---

## 11. Model Training & Hyperparameter Tuning

### Stock-Radar's Training Pipeline

**`src/training/train.py`** implements:

1. **Dataset Loading** — CSV with features + labels
2. **Purged Walk-Forward Split** — time-series aware train/test splitting (prevents look-ahead bias)
3. **Optuna Hyperparameter Search** — automated tuning of model hyperparameters
4. **GradientBoosting Classifier** — scikit-learn's ensemble model
5. **IC Diagnostics** — Information Coefficient and rank IC for feature quality
6. **Feature Importance Breakdown** — which feature groups matter most

```python
# Feature groups for importance breakdown
FEATURE_GROUPS = {
    "base_technical": BASE_FEATURE_NAMES[:8],
    "base_volatility": BASE_FEATURE_NAMES[8:12],
    "base_valuation": BASE_FEATURE_NAMES[12:16],
    "base_quality": BASE_FEATURE_NAMES[16:20],
    "cross_sectional": list(CROSS_SECTIONAL_FEATURE_NAMES),
    "factor_style": list(FACTOR_FEATURE_NAMES),
    "microstructure": list(MICROSTRUCTURE_FEATURE_NAMES),
}
```

### Hyperparameter Optimization with Optuna

```python
import optuna

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }
    model = GradientBoostingClassifier(**params)
    model.fit(X_train, y_train)
    return model.score(X_val, y_val)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=30)
```

### Key Concepts

| Concept | What It Is | Why It Matters |
|---------|-----------|----------------|
| **Walk-Forward Split** | Train on past data, test on future data | Prevents look-ahead bias in time series |
| **Purged Split** | Add gap between train/test sets | Prevents data leakage from autocorrelation |
| **Information Coefficient** | Correlation between predicted and actual returns | Measures prediction quality (IC > 0.05 is good) |
| **Feature Importance** | How much each feature contributes to predictions | Identifies which features to keep/remove |

### What Could Be Used

- **XGBoost / LightGBM** — faster, more feature-rich gradient boosting
- **Cross-validation** — k-fold for more robust evaluation
- **Feature selection** — SHAP values for interpretable feature ranking
- **MLflow** — experiment tracking with hyperparameters, metrics, and artifacts

---

## 12. Model Serving & Inference

### Stock-Radar's Inference Pipeline

**`src/training/predictor.py`** loads a trained model and runs inference:

```python
class SignalPredictor:
    def __init__(self, model_path, risk_factor=1.0, min_confidence=0.35):
        import joblib
        self.pipeline = joblib.load(model_path)  # Load trained model

    def predict(self, indicators, fundamentals=None, ...):
        # 1. Extract 45 features from raw data
        features = extract_features(indicators, fundamentals)

        # 2. Check feature health
        health = check_feature_health(features)

        # 3. Classify market regime (trending, mean-reverting, etc.)
        regime = classify_market_regime(indicators)

        # 4. Run ML prediction
        signal, confidence, probabilities = self._predict(features)

        # 5. Calculate position size based on risk
        position = calculate_position_size(
            signal=signal, confidence=confidence,
            volatility_pct=atr_pct, regime=regime["regime"]
        )

        # 6. Calculate stop-loss and take-profit
        levels = calculate_stop_take_profit(signal, price, atr_pct)

        return {signal, confidence, position, levels, regime}
```

### The Full Prediction Flow

```
Raw market data (price, volume, indicators, fundamentals)
    |
    v
Feature extraction (45 numeric features)
    |
    v
Feature health check (are features valid? NaN? outliers?)
    |
    v
Market regime classification (trending? mean-reverting? volatile?)
    |
    v
ML model prediction (signal + confidence + class probabilities)
    |
    v
Risk-adjusted position sizing (how much to trade?)
    |
    v
Stop-loss / take-profit calculation
    |
    v
Final prediction result
```

### What Could Be Used for Larger Scale

| Tool | What It Does |
|------|-------------|
| **TorchServe** | Serve PyTorch models with batching, auto-scaling |
| **Triton Inference Server** | NVIDIA's high-performance model server (GPU) |
| **BentoML** | Package models as API endpoints with versioning |
| **Ray Serve** | Scalable model serving with composition pipelines |
| **ONNX Runtime** | Convert models to ONNX for faster inference |

---

## 13. Evaluation Frameworks

### Why Evaluation Matters

You can't improve what you can't measure. Every AI system needs automated evaluation:
- After changing prompts — did quality improve?
- After swapping models — is the new model better?
- After code changes — did anything break?
- Continuously — are we regressing over time?

### Stock-Radar's Eval Framework

**`src/eval/runner.py`** runs the analysis pipeline against a test dataset:

```bash
python -m eval.runner ../data/eval_signals.jsonl --output eval_results.json
```

Test dataset format (`data/eval_signals.jsonl`):
```json
{"symbol": "AAPL", "expected_signal": "buy", "expected_score_range": [55, 85], "mode": "intraday"}
{"symbol": "TSLA", "expected_signal": "hold", "expected_score_range": [30, 65], "mode": "intraday"}
```

### Metrics Evaluated

| Metric | What It Measures | Good Score |
|--------|-----------------|------------|
| Signal Exact Accuracy | Predicted exact signal matches expected | > 50% |
| Signal Direction Accuracy | Got bullish/bearish direction right | > 70% |
| Score In Range | Composite score falls in expected range | > 60% |
| Confidence Calibration | Higher confidence = higher accuracy | Monotonic increase |
| Latency p50/p95 | Pipeline speed | p50 < 5s |

### Output Example

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

### Advanced Evaluation Techniques (What Could Be Used)

| Technique | Description |
|-----------|------------|
| **LLM-as-Judge** | Use a stronger LLM (e.g., Claude/GPT-4) to grade weaker model outputs |
| **RAGAS** | Framework for evaluating RAG pipelines (faithfulness, relevance, context recall) |
| **DeepEval** | Unit testing framework for LLM outputs |
| **Human Evaluation** | Have domain experts rate output quality on a rubric |
| **A/B Testing in Production** | Route 10% of traffic to new model, compare metrics |
| **Regression Testing** | Run golden set before every deployment to catch regressions |

---

## 14. Agentic AI & Tool Use

### What Is Agentic AI?

An "agent" is an LLM that can:
1. **Reason** about what to do next
2. **Use tools** (call APIs, search databases, run code)
3. **Loop** — observe results, decide next action, repeat until done

This is the hottest area in AI engineering right now.

### Stock-Radar's Agent Architecture

Stock-Radar uses the term "agents" loosely — its agents are **service modules**, not autonomous LLM agents. Each "agent" is a specialized class:

```
StockRadar (orchestrator)
    |
    +-- StockFetcher     (data collection)
    +-- StockAnalyzer    (LLM-powered analysis)
    +-- StockScorer      (formula-based scoring)
    +-- StockStorage     (database operations)
    +-- RAGRetriever     (context retrieval)
    +-- ChatAssistant    (conversational Q&A)
    +-- NotificationManager (alerts)
    +-- RealtimeManager  (WebSocket data)
    +-- UsageTracker     (cost tracking)
```

The orchestrator coordinates them in a fixed pipeline:
```python
class StockRadar:
    def analyze(self, symbol, mode="intraday"):
        data = self.fetcher.get_quote(symbol)         # Step 1: Fetch
        indicators = self.fetcher.calculate_indicators()  # Step 2: Calculate
        analysis = self.analyzer.analyze(data, indicators)  # Step 3: LLM
        self.storage.save_analysis(analysis)           # Step 4: Store
        self.notifications.send_alert(analysis)        # Step 5: Notify
        return analysis
```

### What's Missing: True Agentic Behavior

Stock-Radar does NOT implement:
- **LLM function calling** — the LLM can't call tools on its own
- **Planning loops** — no ReAct/plan-and-execute cycles
- **Dynamic decision-making** — the pipeline is fixed, not chosen by the LLM
- **Multi-step reasoning** — no chain-of-thought with tool use

### What You Should Learn

#### Function Calling / Tool Use

Modern LLMs can be given "tools" (function definitions) and decide when to call them:

```python
# Example: OpenAI function calling
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current price of a stock",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbol": {"type": "string", "description": "Stock ticker"}
                },
                "required": ["symbol"]
            }
        }
    }
]

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "What's AAPL trading at?"}],
    tools=tools,
)

# The LLM responds with a tool_call:
# {"function": {"name": "get_stock_price", "arguments": '{"symbol": "AAPL"}'}}

# You execute the function, feed the result back, and the LLM generates a final answer.
```

#### ReAct Pattern (Reason + Act)

```
User: "Compare AAPL and MSFT momentum and tell me which is stronger"

LLM thinks: "I need to get data for both stocks. Let me start with AAPL."
LLM acts:   tool_call(get_indicators, symbol="AAPL")
Observation: {rsi: 65, macd: 2.1, ...}

LLM thinks: "Got AAPL data. Now I need MSFT."
LLM acts:   tool_call(get_indicators, symbol="MSFT")
Observation: {rsi: 58, macd: 1.3, ...}

LLM thinks: "Now I can compare. AAPL has higher RSI and MACD."
LLM responds: "AAPL has stronger momentum: RSI 65 vs 58, MACD 2.1 vs 1.3."
```

#### Frameworks for Building Agents

| Framework | Description |
|-----------|------------|
| **LangGraph** | Build stateful, multi-step agent workflows as graphs |
| **CrewAI** | Multi-agent orchestration (agents collaborate on tasks) |
| **Anthropic Claude Agent SDK** | Build agents using Claude's tool use |
| **OpenAI Assistants API** | Managed agent runtime with tools and threads |
| **AutoGen** | Microsoft's multi-agent conversation framework |
| **Build your own** | Often better for production; full control |

---

## 15. Fine-Tuning LLMs

### What Is Fine-Tuning?

Fine-tuning = taking a pre-trained LLM and training it further on your domain-specific data. The result is a model that's better at your specific task.

```
Base model (general knowledge)
    |
    + Fine-tune on 1000 stock analysis examples
    |
    v
Fine-tuned model (expert at stock analysis format + reasoning)
```

### When to Fine-Tune vs. When to Prompt

| Use Prompting When | Use Fine-Tuning When |
|---------------------|----------------------|
| You have < 100 examples | You have 1000+ examples |
| Task format is simple | Task requires specialized format/style |
| You need flexibility | You need consistent, reliable output |
| Quick iteration needed | Performance is critical |
| General-purpose task | Domain-specific task |

### Fine-Tuning Techniques

| Technique | What It Is | Compute Needed |
|-----------|-----------|----------------|
| **Full Fine-Tuning** | Update all model weights | Very high (multiple GPUs) |
| **LoRA** | Train small adapter matrices; freeze base weights | Moderate (single GPU) |
| **QLoRA** | LoRA + 4-bit quantization | Low (consumer GPU: 24GB VRAM) |
| **Prefix Tuning** | Train virtual prompt tokens | Very low |
| **RLHF/DPO** | Align model with human preferences | High |

### Stock-Radar: Not Fine-Tuned (But Could Be)

Stock-Radar uses prompt engineering instead of fine-tuning. A fine-tuned version could be trained on:
- 1000+ historical analyses with expert ratings
- Domain-specific financial terminology
- Consistent output format (reducing guardrail triggers)

### Tools for Fine-Tuning

| Tool | Best For |
|------|---------|
| **Unsloth** | Fastest QLoRA fine-tuning (2x speed, 60% less memory) |
| **HuggingFace PEFT** | LoRA/QLoRA with any HuggingFace model |
| **Axolotl** | Easy config-driven fine-tuning |
| **OpenAI Fine-Tuning API** | Managed fine-tuning for GPT models |
| **Together AI** | Managed fine-tuning for open models |

---

## 16. Production Configuration Management

### The Problem

AI systems have dozens of configuration values: API keys, model names, thresholds, feature flags, retry counts. Scattering `os.getenv()` calls throughout the code leads to:
- No validation (typos in env var names fail silently)
- No documentation of what settings exist
- Type errors (getting a string when you need a float)
- No defaults management

### Stock-Radar's Solution: Pydantic Settings

**`src/config.py`** centralizes ALL configuration in one type-safe class:

```python
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # LLM Configuration
    llm_fallback_order: str = Field(
        default="openai/glm-4.7,gemini/gemini-2.5-flash",
        alias="LLM_FALLBACK_ORDER",
    )
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2000, alias="LLM_MAX_TOKENS")

    # Guardrails
    guardrails_enabled: bool = Field(default=True, alias="GUARDRAILS_ENABLED")
    guardrails_max_confidence: float = Field(default=0.95, alias="GUARDRAILS_MAX_CONFIDENCE")

    # Cache
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    cache_quote_ttl_sec: int = Field(default=60, alias="CACHE_QUOTE_TTL_SEC")

    # ... 50+ more settings

# Singleton - created once, imported everywhere
settings = Settings()
```

### Priority Order

```
1. Environment variables    (highest)   export LLM_TEMPERATURE=0.5
2. .env file                            LLM_TEMPERATURE=0.5
3. Default in code          (lowest)    Field(default=0.3)
```

### Benefits

| Benefit | How |
|---------|-----|
| **Type safety** | `llm_temperature: float` — Pydantic rejects non-float values |
| **Validation at startup** | Invalid config crashes immediately, not at 3am in production |
| **Documentation** | All settings visible in one file |
| **IDE autocomplete** | `settings.` gives you all options |
| **Secret management** | API keys in `.env` file, never in code |

### What Could Be Used

- **AWS Secrets Manager / GCP Secret Manager** for production secrets
- **Vault (HashiCorp)** for secret rotation
- **Dynaconf** for multi-environment configs (dev/staging/prod)
- **Hydra (Facebook)** for complex ML experiment configurations

---

## 17. Structured Logging & Observability

### The Problem

When your AI system is running in production and something goes wrong, you need to answer questions like:
- "Which LLM model was used for this request?"
- "How long did the analysis take?"
- "What was the confidence score?"
- "Why did the fallback trigger?"

`print()` statements can't answer these questions. You need **structured logging**.

### Stock-Radar's Structured Logging

**`src/logging_config.py`** uses `structlog` for machine-readable logs:

```python
from logging_config import get_logger
logger = get_logger(__name__)

# Instead of: print(f"Analysis done for AAPL in 1.2s")
logger.info("analysis_complete",
    symbol="AAPL",
    signal="buy",
    confidence=0.85,
    model="openai/glm-4.7",
    latency_sec=1.2,
    tokens_used=1550,
    cost_usd=0.004,
)
```

### Output Formats

**Development** (`LOG_JSON=false`) — human-readable:
```
2025-02-08 10:30:00 [info] analysis_complete  symbol=AAPL signal=buy confidence=0.85
```

**Production** (`LOG_JSON=true`) — machine-readable:
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

JSON logs can be ingested by observability platforms:

| Platform | What You Can Do |
|----------|----------------|
| **Datadog** | Search `symbol:AAPL AND signal:buy`, build dashboards |
| **ELK Stack** | Visualize latency over time, set up anomaly detection |
| **CloudWatch** | Alert on `level:error`, metric extraction |
| **Grafana Loki** | Correlate logs with Prometheus metrics |

### The Three Pillars of Observability

```
                    OBSERVABILITY
                    /     |      \
               Logs    Metrics    Traces
               |         |          |
           structlog  Prometheus  (OpenTelemetry)
           "What       "How        "How do requests
            happened?"  much?"      flow through
                                    services?"
```

Stock-Radar implements **Logs** (structlog) and **Metrics** (Prometheus). **Traces** (distributed tracing with OpenTelemetry) could be added for request flow visualization across the Next.js frontend, FastAPI backend, and external APIs.

---

## 18. Metrics & Monitoring (Prometheus + Grafana)

### Why Metrics Matter

Logs tell you *what happened*. Metrics tell you *how much* and *how fast*. In production, you need both.

**Interview question:** "How do you monitor your AI system in production?"

### Stock-Radar's Prometheus Metrics

**`src/metrics.py`** defines quantitative metrics:

```python
from prometheus_client import Counter, Histogram, Gauge

# LLM Performance
LLM_LATENCY = Histogram(
    "stockradar_llm_latency_seconds",
    "LLM API call latency in seconds",
    ["model"],
    buckets=(0.5, 1, 2, 5, 10, 30, 60),
)

LLM_REQUESTS = Counter(
    "stockradar_llm_requests_total",
    "Total LLM API requests",
    ["model", "status"],
)

LLM_FALLBACK = Counter(
    "stockradar_llm_fallback_total",
    "Number of times a fallback model was used",
    ["from_model", "to_model"],
)

# Cost Tracking
API_COST = Counter(
    "stockradar_api_cost_usd_total",
    "Running API cost in USD",
    ["provider"],
)

# Quality Tracking
ANALYSIS_CONFIDENCE = Histogram(
    "stockradar_analysis_confidence",
    "Distribution of analysis confidence scores",
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

GUARDRAIL_TRIGGERS = Counter(
    "stockradar_guardrail_triggers_total",
    "Guardrail trigger count",
    ["rule", "action"],
)
```

### Metric Types Explained

| Type | What It Tracks | Example |
|------|---------------|---------|
| **Counter** | Things that only go up | Total requests, total tokens, total cost |
| **Histogram** | Distribution of values | Latency distribution, confidence distribution |
| **Gauge** | Current value (can go up or down) | Is system up? Is ML model loaded? |

### The Monitoring Stack

```
Your App (Python)
    |
    | Exposes /metrics endpoint
    v
Prometheus (scrapes /metrics every 15s)
    |
    | Stores time-series data
    v
Grafana (visualizes dashboards)
    |
    | Sends alerts
    v
Slack/PagerDuty/Email
```

Stock-Radar includes the full stack in `docker-compose.yml`:
```yaml
services:
  prometheus:
    image: prom/prometheus:latest
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml

  grafana:
    image: grafana/grafana:latest
    volumes:
      - ./monitoring/grafana/dashboards:/var/lib/grafana/dashboards
```

### Key Dashboards to Build

| Dashboard | Metrics Used | What It Shows |
|-----------|-------------|---------------|
| LLM Health | latency, requests, fallbacks | Is the LLM fast? How often does it fail? |
| Cost Tracker | api_cost, tokens | How much are we spending per hour/day? |
| Quality | confidence, signal distribution, guardrails | Is output quality stable? |
| Cache Efficiency | cache hits, cache misses | Is caching saving us money? |

---

## 19. Caching Strategies

### Why Cache?

LLM calls are slow (1-10s) and expensive ($0.001-$0.10 each). If someone asks about AAPL twice in 5 minutes, the answer won't change. Cache it.

### Stock-Radar's Cache Strategy

| Data Type | TTL | Why |
|-----------|-----|-----|
| Stock Quotes | 60 seconds | Prices change frequently |
| Fundamentals | 1 hour | P/E, ROE change slowly |
| Analysis Results | 15 minutes | Same prompt = same answer |
| Embeddings | 24 hours | Same text = same vector |
| Indicators | 5 minutes | Recalculated from prices |

### Cache Backends

Stock-Radar auto-selects:

```python
# If REDIS_URL is set, use Redis (production)
# Otherwise, use in-memory dict (development)
redis_url: str | None = Field(default=None, alias="REDIS_URL")
```

| Backend | Pros | Cons | When |
|---------|------|------|------|
| **In-Memory (dict)** | Zero setup, fast | Lost on restart, not shared | Development |
| **Redis** | Shared across processes, persistent, TTL support | Needs infrastructure | Production |

### Cache Metrics

Prometheus tracks hit/miss rates:
```
stockradar_cache_hits_total{cache_type="quote"} 45
stockradar_cache_misses_total{cache_type="quote"} 12
# Hit rate = 45 / (45 + 12) = 78.9%
```

A high cache hit rate means you're saving money and latency.

### Advanced Caching (What Could Be Used)

- **Semantic caching** — cache based on prompt similarity, not exact match (e.g., GPTCache)
- **Redis Cluster** — distributed caching for high availability
- **CDN caching** — for static or slowly-changing API responses
- **Write-through vs. write-behind** — caching strategies for database writes

---

## 20. Retry Logic & Resilience

### The Problem

External APIs fail. Timeouts, rate limits, 503 errors, network blips. Your system must handle these gracefully.

### Stock-Radar uses Tenacity

```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10),
)
def fetch_quote(symbol: str):
    response = requests.get(f"https://api.example.com/quote/{symbol}")
    response.raise_for_status()
    return response.json()
```

### Retry Timeline

```
Attempt 1: Call API
    | (fails with timeout)
Wait 1 second (exponential backoff)
Attempt 2: Call API
    | (fails with 503)
Wait 2 seconds
Attempt 3: Call API
    | (succeeds!)
Return result
```

### Configurable via Settings

```python
retry_max_attempts: int = Field(default=3, alias="RETRY_MAX_ATTEMPTS")
retry_min_wait_sec: int = Field(default=1, alias="RETRY_MIN_WAIT_SEC")
retry_max_wait_sec: int = Field(default=10, alias="RETRY_MAX_WAIT_SEC")
```

### Resilience Patterns

| Pattern | What It Does | Used in Stock-Radar? |
|---------|-------------|:--------------------:|
| **Retry with backoff** | Retry failed calls with increasing delay | Yes |
| **Fallback chain** | Try alternative providers on failure | Yes (LLM fallback) |
| **Circuit breaker** | Stop calling a failing service temporarily | No (could add) |
| **Timeout** | Kill requests that take too long | Yes (`LLM_TIMEOUT_SEC`) |
| **Bulkhead** | Isolate failures in one component from others | No (could add) |
| **Rate limiting** | Limit outgoing request rate | No (could add) |

---

## 21. API Development (FastAPI)

### Why FastAPI?

FastAPI is the standard for Python AI backends:
- Automatic OpenAPI docs
- Type validation with Pydantic
- Async support
- Fast (built on Starlette/Uvicorn)

### Stock-Radar's Backend

**`backend/app.py`** is a FastAPI application:

```python
from fastapi import FastAPI, HTTPException, Depends
from backend.schemas import AnalyzeJobCreateRequest, AskRequest

app = FastAPI()

@app.post("/api/analyze")
async def create_analyze_job(req: AnalyzeJobCreateRequest, auth=Depends(verify_backend_auth)):
    """Submit async analysis job."""
    job_id = job_manager.create_job(req.symbol, req.mode)
    return {"job_id": job_id, "status": "pending"}

@app.get("/api/analyze/status")
async def get_analyze_status(job_id: str):
    """Poll analysis job status."""
    status = job_manager.get_status(job_id)
    return status

@app.post("/api/ask")
async def ask_question(req: AskRequest, auth=Depends(verify_backend_auth)):
    """RAG-powered Q&A about stocks."""
    response = chat_assistant.ask(req.question, symbol=req.symbol)
    return response
```

### Key FastAPI Patterns

| Pattern | Description |
|---------|------------|
| **Pydantic schemas** | Request/response validation |
| **Dependency injection** | `Depends(verify_backend_auth)` for auth |
| **Async jobs** | Long-running analysis submitted as background jobs |
| **Health checks** | `/health` endpoint for load balancers |
| **Metrics endpoint** | `/metrics` for Prometheus scraping |
| **Input validation** | Regex patterns for symbols: `^[A-Za-z0-9.\-^]{1,20}$` |

### API Security

Stock-Radar uses API key authentication:
```python
# backend/auth.py
def verify_backend_auth(request: Request):
    key = request.headers.get("X-Backend-Api-Key")
    if key != os.getenv("BACKEND_API_KEY"):
        raise HTTPException(status_code=401, detail="Unauthorized")
```

---

## 22. Data Engineering for AI

### Data Sources in Stock-Radar

| Source | Data Type | API Type |
|--------|----------|----------|
| **Yahoo Finance** (yfinance) | Price history, fundamentals | Python library (free) |
| **Twelvedata** | Real-time prices, 30+ year history | REST API |
| **Finnhub** | News, sentiment, earnings calendar | REST + WebSocket |
| **Supabase** (PostgreSQL) | Persistent storage, vector search | REST API |
| **Cohere** | Text embeddings | REST API |

### Data Fetching Architecture

**`src/agents/fetcher.py`** uses multiple providers with fallback:

```python
class StockFetcher:
    def get_quote(self, symbol):
        """Fetch from Twelvedata first, fallback to Yahoo Finance."""
        try:
            return self._fetch_twelvedata(symbol)
        except Exception:
            return self._fetch_yfinance(symbol)

    def get_news(self, symbol):
        """Fetch news from Finnhub."""
        return self._fetch_finnhub_news(symbol)

    def calculate_indicators(self, history):
        """Calculate technical indicators from price history."""
        return {
            "rsi_14": self._calculate_rsi(history),
            "macd": self._calculate_macd(history),
            "bollinger_upper": ...,
            # ... 20+ indicators
        }
```

### Real-Time Data with WebSockets

**`src/agents/realtime.py`** connects to Finnhub WebSocket for live prices:

```python
import websocket

class RealtimeManager:
    def connect(self, symbols):
        ws = websocket.WebSocketApp(
            f"wss://ws.finnhub.io?token={API_KEY}",
            on_message=self.on_message,
        )
        for symbol in symbols:
            ws.send(json.dumps({"type": "subscribe", "symbol": symbol}))
```

### Database Design

Stock-Radar uses Supabase (PostgreSQL + pgvector) for:
- Stock records and price history
- Analysis results with embeddings (for RAG)
- Signal history
- Conversation logs
- Knowledge base entries

### What Could Be Used

- **Apache Kafka / Redpanda** — for real-time data streaming pipelines
- **Apache Airflow** — for scheduled ETL (data pipeline orchestration)
- **dbt** — for SQL-based data transformations
- **Delta Lake / Apache Iceberg** — for versioned data lakes
- **Great Expectations** — for data quality validation

---

## 23. Containerization & Docker

### Why Docker?

Docker packages your application with all its dependencies into a portable container. "Works on my machine" becomes "works everywhere."

### Stock-Radar's Docker Setup

**`Dockerfile`** (backend):
```dockerfile
FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "backend.app:app", "--host", "0.0.0.0", "--port", "8000"]
```

**`docker-compose.yml`** orchestrates 5 services:

```yaml
services:
  backend:       # FastAPI Python backend
    build: .
    ports: ["9090:9090"]
    depends_on: [redis]

  web:           # Next.js frontend
    build: ./web
    ports: ["3000:3000"]
    depends_on: [backend]

  redis:         # Cache layer
    image: redis:7-alpine
    ports: ["6379:6379"]

  prometheus:    # Metrics collection
    image: prom/prometheus:latest
    ports: ["9091:9090"]

  grafana:       # Metrics visualization
    image: grafana/grafana:latest
    ports: ["3001:3000"]
    depends_on: [prometheus]
```

### Key Docker Concepts for AI Engineers

| Concept | What It Is |
|---------|-----------|
| **Multi-stage builds** | Reduce image size by separating build and runtime |
| **Volume mounts** | Persist data (models, databases) across container restarts |
| **Environment variables** | Configure containers without rebuilding |
| **Health checks** | Docker monitors if your service is healthy |
| **Networking** | Containers communicate via service names (`redis:6379`) |

### What Could Be Used

- **Kubernetes** — for production orchestration at scale
- **Docker BuildKit** — for faster, more efficient builds
- **Distroless images** — minimal container images for security
- **GPU containers** — `nvidia/cuda` base images for ML inference

---

## 24. CI/CD for AI Projects

### Stock-Radar's CI Pipeline

**`.github/workflows/ci.yml`**:

```yaml
name: CI
on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.13" }
      - run: pip install ruff
      - run: ruff check src/ tests/

  test:
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4
      - run: pip install -r requirements.txt
      - run: pytest tests/ -v --tb=short

  docker:
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - run: docker build -t stock-radar-backend .
      - run: docker run --rm stock-radar-backend python -c "import sys; ..."
```

### Pipeline Stages

```
Push to main
    |
    v
[1. LINT] --> ruff check (code style, common errors)
    |
    v
[2. TEST] --> pytest (unit + integration tests)
    |
    v
[3. DOCKER] --> Build image, smoke test
    |
    v
(Deploy to Railway / Vercel)
```

### What Should Be Added for Production AI CI/CD

| Stage | What It Does | Tool |
|-------|-------------|------|
| **Model evaluation** | Run eval suite on every PR | Custom eval runner |
| **Regression testing** | Compare against golden outputs | DeepEval, custom |
| **Security scanning** | Check for vulnerabilities in dependencies | Snyk, Trivy |
| **Performance benchmarks** | Track latency/throughput regressions | pytest-benchmark |
| **Model artifact publishing** | Upload trained models to registry | MLflow, DVC |

---

## 25. Testing AI Systems

### Stock-Radar's Test Suite

15+ test files covering different aspects:

```
tests/
    test_analysis_timeframes.py     # Different analysis modes
    test_analyzer_rag.py            # RAG-enhanced analysis
    test_backend_metrics.py         # Prometheus metrics
    test_backtesting.py             # ML backtesting
    test_cross_sectional.py         # Cross-sectional features
    test_llm_routing.py             # Model routing logic
    test_regime_risk.py             # Market regime detection
    test_sentiment.py               # Sentiment feature extraction
    test_storage_embeddings.py      # Vector storage
    test_training.py                # ML training pipeline
    ...
```

### Types of Tests for AI Systems

| Test Type | What It Validates | Example |
|-----------|------------------|---------|
| **Unit tests** | Individual functions work correctly | "Does RSI calculation produce correct values?" |
| **Integration tests** | Components work together | "Does fetcher → analyzer → storage pipeline complete?" |
| **Evaluation tests** | AI output quality meets standards | "Is signal accuracy > 50%?" |
| **Regression tests** | New changes don't break existing behavior | "Does AAPL still get a 'buy' signal?" |
| **Property-based tests** | Output satisfies invariants for any input | "Confidence is always in [0, 1]" |
| **Load tests** | System handles expected traffic | "Can we handle 100 concurrent analyses?" |
| **Snapshot tests** | Output doesn't change unexpectedly | "LLM output matches golden file" |

### Testing LLM Outputs

Testing LLM outputs is inherently non-deterministic. Strategies:

```python
def test_analysis_produces_valid_signal():
    """Test that analysis output has valid structure."""
    result = analyzer.analyze("AAPL", mode="intraday")

    # Don't test exact values — test structure and constraints
    assert result["signal"] in {"strong_buy", "buy", "hold", "sell", "strong_sell"}
    assert 0 <= result["confidence"] <= 1.0
    assert len(result["reasoning"]) > 20
    assert result.get("target_price") is None or isinstance(result["target_price"], (int, float))
```

---

## 26. Experiment Tracking

### What It Is

Experiment tracking = recording every ML experiment (hyperparameters, metrics, artifacts, code version) so you can compare and reproduce results.

### Stock-Radar: Not Implemented (Opportunity)

Stock-Radar stores model metadata in JSON:
```python
# From training/train.py - saves metadata alongside model
meta = {
    "accuracy": accuracy,
    "n_features": n_features,
    "best_params": best_params,
    "timestamp": datetime.now().isoformat(),
}
```

### What Should Be Used

| Tool | Description | Type |
|------|------------|------|
| **Weights & Biases (W&B)** | Most popular experiment tracker; great visualization | SaaS (free tier) |
| **MLflow** | Open-source experiment tracking + model registry | Self-hosted |
| **Comet ML** | Experiment tracking with comparison tools | SaaS |
| **DVC** | Data and model versioning (like Git for data) | Open source |
| **Neptune.ai** | Lightweight experiment tracking | SaaS |

### What W&B Tracking Looks Like

```python
import wandb

wandb.init(project="stock-radar", config={
    "model": "GradientBoosting",
    "n_estimators": 200,
    "max_depth": 5,
    "features": 45,
})

# Log metrics during training
wandb.log({"train_accuracy": 0.78, "val_accuracy": 0.72, "ic": 0.08})

# Log model artifact
wandb.log_artifact("models/best_model.joblib", type="model")

wandb.finish()
```

---

## 27. System Design for AI Applications

### Stock-Radar's Architecture

```
                            ┌─────────────────────┐
                            │   User / Client      │
                            └──────────┬───────────┘
                                       │
                          ┌────────────▼───────────┐
                          │  Vercel: Next.js Web    │
                          │  (Frontend + API Routes)│
                          └────────────┬───────────┘
                                       │ X-Backend-Api-Key
                          ┌────────────▼───────────┐
                          │  Railway: FastAPI       │
                          │  (Backend Compute)      │
                          └──┬────┬────┬────┬──────┘
                             │    │    │    │
              ┌──────────────┘    │    │    └──────────────┐
              │                   │    │                   │
    ┌─────────▼──────┐  ┌────────▼────▼───┐  ┌──────────▼──────┐
    │  LLM Providers  │  │   Supabase DB    │  │  Market Data    │
    │  (Gemini, Groq,  │  │  (PostgreSQL +   │  │  (Twelvedata,   │
    │   ZAI, Ollama)   │  │   pgvector)      │  │   Yahoo, Finnhub│
    └─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Key Design Decisions

| Decision | Why |
|----------|-----|
| **Split backend from frontend** | Python for AI compute, Next.js for UI; each scales independently |
| **API key auth between services** | Simple, sufficient for service-to-service communication |
| **Async job pattern** | Analysis takes 5-30s; async prevents timeouts |
| **Multi-provider LLM** | No single point of failure for AI |
| **Formula + ML + LLM** | Three approaches for different strengths |
| **Supabase for everything** | One database for relational data + vectors + auth |

### System Design Interview Prep

Common questions for AI engineering interviews:

1. **"Design a RAG system"** — Ingestion pipeline, chunking, vector DB, retrieval, generation, evaluation
2. **"Design an LLM gateway"** — Rate limiting, fallback, caching, cost tracking, streaming
3. **"Design a real-time AI assistant"** — WebSocket/SSE, context management, tool use, memory
4. **"How would you reduce LLM costs by 90%?"** — Caching, smaller models, batching, prompt compression
5. **"How would you handle 1000 concurrent AI requests?"** — Async processing, queue-based architecture, horizontal scaling

---

## 28. Security & Safety in AI Systems

### API Key Management

Stock-Radar keeps secrets in `.env` files, never in code:

```python
zai_api_key: str | None = Field(default=None, alias="ZAI_API_KEY")
gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
supabase_key: str | None = Field(default=None, alias="SUPABASE_KEY")
```

### Input Validation

The backend validates all user input:
```python
SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-^]{1,20}$")  # Only valid ticker chars
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")   # Alphanumeric session IDs
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}
```

### AI-Specific Security Concerns

| Concern | Description | Mitigation |
|---------|------------|------------|
| **Prompt injection** | User crafts input to override system prompt | Input sanitization, output validation |
| **Data leakage** | LLM reveals training data or internal prompts | Don't put secrets in prompts |
| **Hallucination** | LLM generates false information | Guardrails, RAG grounding |
| **Cost attacks** | Malicious user generates expensive LLM calls | Rate limiting, usage caps |
| **PII exposure** | LLM includes personal data in responses | Output filtering, PII detection |

### What Could Be Used

- **OWASP LLM Top 10** — security checklist for LLM applications
- **Rebuff** — prompt injection detection
- **LLM Guard** — content moderation and safety checks
- **Presidio** — PII detection and anonymization

---

## 29. Common Interview Topics & Questions

### Technical Questions You Will Be Asked

#### LLM Integration
- "How do you handle LLM API failures?" → Fallback chains, retry with backoff
- "How do you manage hallucinations?" → Guardrails, RAG grounding, output validation
- "How do you reduce LLM costs?" → Caching, cheaper models for simple tasks, prompt optimization
- "Explain streaming vs non-streaming" → SSE protocol, token-by-token delivery, user experience
- "What's the difference between system and user prompts?" → Persona/rules vs. specific request

#### RAG
- "Walk me through a RAG pipeline" → Embed query → vector search → build context → generate → validate
- "How do you evaluate RAG quality?" → Faithfulness, relevance, context recall (RAGAS metrics)
- "What chunking strategy would you use?" → Depends on document type; 512 tokens, 50-token overlap is a common starting point

#### Production
- "How do you monitor an AI system?" → Prometheus metrics, structured logging, dashboards, alerts
- "How do you iterate on prompts?" → Version control, A/B testing, evaluation framework
- "How do you handle configuration?" → Pydantic settings, env vars, secrets management
- "Design a system that handles 10K AI requests/minute" → Async queue, caching, horizontal scaling, batch inference

#### ML
- "When would you use traditional ML vs LLM?" → Structured data + speed + cost → ML; unstructured + reasoning → LLM
- "How do you prevent data leakage in time series?" → Purged walk-forward splits
- "What is feature importance?" → How much each input feature contributes to predictions

### Behavioral / Portfolio Questions

- "Walk me through a project you built" → Know Stock-Radar inside and out
- "What was the hardest technical decision?" → Multi-model fallback vs. single model, formula scoring vs. ML vs. LLM
- "How did you test your AI system?" → Evaluation framework, unit tests, guardrail validation
- "What would you do differently?" → Add agentic tool use, experiment tracking, async throughout

---

## 30. Learning Roadmap & Resources

### Phase 1: Foundations (Weeks 1-2)

| Topic | What to Do |
|-------|-----------|
| Python | Master dataclasses, type hints, async/await, decorators |
| LLM APIs | Call OpenAI/Anthropic/Gemini APIs directly, understand tokens/pricing |
| Prompt Engineering | Read Anthropic/OpenAI prompt engineering guides; practice structured prompts |
| Git | Branching, PRs, CI workflows |

### Phase 2: Core AI Engineering (Weeks 3-5)

| Topic | What to Build |
|-------|-------------|
| RAG | Build a RAG app from scratch: embed documents, store in vector DB, retrieve, generate |
| Streaming | Implement SSE streaming with FastAPI |
| Guardrails | Build an output validation layer for your RAG app |
| Evaluation | Create an eval framework that measures your RAG pipeline's quality |

### Phase 3: Production Skills (Weeks 6-8)

| Topic | What to Do |
|-------|-----------|
| Docker | Containerize your AI app with docker-compose |
| Monitoring | Add Prometheus metrics and Grafana dashboards |
| CI/CD | Set up GitHub Actions with linting, testing, eval |
| Caching | Add Redis caching with TTL management |
| Logging | Replace print statements with structlog |

### Phase 4: Advanced Topics (Weeks 9-12)

| Topic | What to Build |
|-------|-------------|
| Agents | Build a ReAct agent with tool use (function calling) |
| Fine-Tuning | Fine-tune an open model with QLoRA on your domain data |
| ML Pipeline | Train, evaluate, version, and serve an ML model |
| System Design | Practice designing AI systems on paper (whiteboard interviews) |

### Recommended Resources

**Courses:**
- Andrej Karpathy's "Neural Networks: Zero to Hero" (YouTube, free)
- DeepLearning.AI "LLMOps" specialization
- HuggingFace NLP Course (free)

**Documentation to Read:**
- Anthropic Prompt Engineering Guide
- OpenAI Cookbook
- LangChain / LangGraph documentation
- FastAPI documentation

**Books:**
- "Designing Machine Learning Systems" by Chip Huyen
- "Building LLM Apps" by Valentino Gagliardi

**Practice:**
- Build 3-5 small AI projects with different techniques
- Contribute to open-source AI tools
- Study Stock-Radar's codebase — it covers ~73% of what you need

---

## Appendix: Stock-Radar Coverage Summary

| Category | Topics Covered | Total Topics | Coverage |
|----------|:-------------:|:------------:|:--------:|
| LLM Integration & Prompting | 7 | 10 | 70% |
| RAG & Embeddings | 3 | 5 | 60% |
| Traditional ML | 6 | 8 | 75% |
| Production / MLOps | 9 | 14 | 64% |
| Data Engineering | 4 | 5 | 80% |
| Software Engineering | 8 | 9 | 89% |
| **Total** | **37** | **51** | **73%** |

**Strongest areas:** Software engineering practices, production observability, multi-provider LLM integration, formula + ML + LLM hybrid architecture.

**Key gaps to fill:** Agentic AI (tool use, planning loops), fine-tuning, deep learning fundamentals, experiment tracking, advanced RAG (re-ranking, hybrid search).

---

*This guide reflects patterns and practices as of 2026. The AI engineering field evolves rapidly — stay current with the latest tools and techniques.*
