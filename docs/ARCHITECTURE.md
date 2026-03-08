# Stock Radar — Complete Architecture & Engineering Documentation

> **"If you can't explain it simply, you don't understand it well enough."** — Albert Einstein
>
> This document explains every moving part of Stock Radar — from the 10,000-foot view down to the individual functions — so that anyone, from a junior developer to a seasoned architect, can understand exactly how this system works.

---

## Table of Contents

1. [What Is Stock Radar?](#1-what-is-stock-radar)
2. [The Big Picture — System Architecture](#2-the-big-picture--system-architecture)
3. [The Orchestrator — `StockRadar` (main.py)](#3-the-orchestrator--stockradar-mainpy)
4. [Agent Layer — The Specialist Workers](#4-agent-layer--the-specialist-workers)
   - [4.1 StockFetcher — The Data Collector](#41-stockfetcher--the-data-collector)
   - [4.2 StockAnalyzer — The AI Brain](#42-stockanalyzer--the-ai-brain)
   - [4.3 StockStorage — The Memory](#43-stockstorage--the-memory)
   - [4.4 NotificationManager — The Messenger](#44-notificationmanager--the-messenger)
   - [4.5 StockScorer — The Formula Engine](#45-stockscorer--the-formula-engine)
   - [4.6 ChatAssistant — The Conversationalist](#46-chatassistant--the-conversationalist)
   - [4.7 RAGRetriever & RAGValidator — The Knowledge System](#47-ragretriever--ragvalidator--the-knowledge-system)
   - [4.8 RealtimeManager — The Live Wire](#48-realtimemanager--the-live-wire)
   - [4.9 UsageTracker — The Accountant](#49-usagetracker--the-accountant)
5. [The Training & ML Subsystem](#5-the-training--ml-subsystem)
   - [5.1 SignalPredictor — The ML Model](#51-signalpredictor--the-ml-model)
   - [5.2 PaperPortfolio — The Simulator](#52-paperportfolio--the-simulator)
   - [5.3 BrokerAdapter — The Execution Layer](#53-brokeradapter--the-execution-layer)
   - [5.4 Risk Management — The Safety Net](#54-risk-management--the-safety-net)
6. [Infrastructure & Cross-Cutting Concerns](#6-infrastructure--cross-cutting-concerns)
   - [6.1 Configuration Management](#61-configuration-management)
   - [6.2 Caching Layer](#62-caching-layer)
   - [6.3 LLM Output Guardrails](#63-llm-output-guardrails)
   - [6.4 Prompt Versioning & A/B Testing](#64-prompt-versioning--ab-testing)
   - [6.5 Streaming Responses (SSE)](#65-streaming-responses-sse)
   - [6.6 Prometheus Metrics & Monitoring](#66-prometheus-metrics--monitoring)
   - [6.7 Token Accounting & Cost Estimation](#67-token-accounting--cost-estimation)
7. [The Backend API (FastAPI)](#7-the-backend-api-fastapi)
8. [The Frontend (Next.js)](#8-the-frontend-nextjs)
9. [Web3 & Blockchain Layer](#9-web3--blockchain-layer)
10. [Database Schema](#10-database-schema)
11. [The Complete Analysis Pipeline — Step by Step](#11-the-complete-analysis-pipeline--step-by-step)
12. [Data Flow Diagrams](#12-data-flow-diagrams)
13. [CLI Commands Reference](#13-cli-commands-reference)
14. [Deployment Architecture](#14-deployment-architecture)
15. [Design Decisions & Trade-offs](#15-design-decisions--trade-offs)

---

## 1. What Is Stock Radar?

Stock Radar is an **AI-powered stock analysis platform** that brings institutional-grade intelligence to individual traders. Think of it as having a team of expert analysts working 24/7 — one fetching the latest market data, one crunching numbers using technical indicators, one reading news and sentiment, one running machine learning models, and one sending you alerts when something important happens.

**What it does in plain English:**
- You give it a stock symbol (like `AAPL` or `RELIANCE.NS`)
- It fetches the current price, historical data, news, social media buzz, and company fundamentals — all in parallel
- It runs AI analysis using multiple LLM providers with automatic failover
- It calculates algorithmic scores for momentum, value, quality, and risk
- Optionally, it runs an ML model trained on historical data for predictions
- It stores everything in a database with vector embeddings for semantic search
- It sends smart alerts to your Slack and Telegram
- It displays all of this in a beautiful Next.js dashboard

**The engineering philosophy:**
Every component follows the **Single Responsibility Principle** — each class does exactly one thing, and does it well. This isn't just academic best practice; it means if the Yahoo Finance API goes down, only the fetcher knows about it. The analyzer, storage, and alerts keep working with whatever data is available.

---

## 2. The Big Picture — System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER INTERFACES                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐              │
│  │   Next.js    │  │   CLI        │  │  Slack/TG    │              │
│  │  Dashboard   │  │  Terminal    │  │  Bot Alerts  │              │
│  └──────┬───────┘  └──────┬───────┘  └──────▲───────┘              │
│         │                 │                 │                       │
│         ▼                 ▼                 │                       │
│  ┌──────────────┐  ┌──────────────┐         │                       │
│  │  FastAPI     │  │  main.py     │─────────┘                       │
│  │  Backend     │  │  (CLI)       │                                 │
│  └──────┬───────┘  └──────┬───────┘                                 │
│         │                 │                                         │
│         └────────┬────────┘                                         │
│                  ▼                                                   │
│  ┌────────────────────────────────────────────────────────┐         │
│  │              STOCK RADAR — ORCHESTRATOR                │         │
│  │                                                        │         │
│  │  Coordinates the 5-step analysis pipeline:             │         │
│  │  1. Fetch → 2. Analyze → 3. Verify → 4. Store → 5. Alert       │
│  └──────────┬──────────┬──────────┬──────────┬───────────┘         │
│             │          │          │          │                       │
│  ┌──────────▼──┐ ┌─────▼──────┐ ┌▼────────┐ ┌▼─────────────┐      │
│  │ StockFetcher│ │StockAnalyzer│ │ Storage │ │Notifications │      │
│  │             │ │             │ │         │ │              │      │
│  │ •TwelveData │ │ •LLM Chain  │ │•Supabase│ │ •Slack       │      │
│  │ •yFinance   │ │ •ML Model   │ │•Cohere  │ │ •Telegram    │      │
│  │ •Finnhub    │ │ •Scorer     │ │•pgvector│ │              │      │
│  │ •Reddit API │ │ •RAG        │ │         │ │              │      │
│  └─────────────┘ └─────────────┘ └─────────┘ └──────────────┘      │
│                                                                     │
│  ┌────────────────────  SUPPORTING SYSTEMS  ─────────────────────┐  │
│  │  Config │ Cache │ Guardrails │ Metrics │ Prompts │ Streaming  │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                     │
│  ┌────────────────────  TRAINING / ML  ──────────────────────────┐  │
│  │  Predictor │ Paper Trading │ Broker │ Risk │ Kill Switches    │  │
│  └─────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. The Orchestrator — `StockRadar` (main.py)

**File:** `main.py` (1,489 lines)
**Role:** The **conductor** of the entire system.

Think of `StockRadar` like the manager of a restaurant. The manager doesn't cook the food (that's the `StockFetcher`), doesn't design the menu (that's the `StockAnalyzer`), doesn't take payments (that's `StockStorage`), and doesn't seat guests (that's `NotificationManager`). But the manager coordinates all of them to deliver a great experience.

### What Happens When You Initialize StockRadar

```python
class StockRadar:
    def __init__(self):
        # 1. Start the metrics server (Prometheus monitoring)
        self._metrics_server = start_metrics_server(port=9090)

        # 2. Create each specialist agent
        self.fetcher = StockFetcher()           # The data collector
        self.analyzer = StockAnalyzer()         # The AI brain
        self.storage = StockStorage()           # The database layer
        self.notifications = NotificationManager()  # The alert system

        # 3. Start real-time WebSocket feed (Finnhub)
        self._realtime = get_realtime_manager()
        self._realtime.start()

        # 4. Verify the database schema is correct
        self.storage.ensure_schema()
```

**Why this matters:** Each agent is instantiated independently. If Finnhub is down, the realtime feed won't start — but the rest of the system works perfectly with historical data. This is **graceful degradation** in action.

### The 5-Step Analysis Pipeline

When you call `radar.analyze_stock("AAPL", mode="intraday")`, here's exactly what happens:

```
Step 1: FETCH DATA
    └─ Calls fetcher.get_full_analysis_data("AAPL")
    └─ Runs 5 API calls IN PARALLEL (ThreadPoolExecutor):
        • get_quote()           → Current price, volume, change
        • get_price_history()   → Historical OHLCV bars
        • get_fundamentals()    → P/E, ROE, revenue, etc.
        • get_news_yahoo()      → Recent news articles
        • get_news_finnhub()    → Finnhub news (if API key set)
    └─ Calculates technical indicators from price history:
        RSI, MACD, Bollinger Bands, ATR, VWAP, ADX
    └─ Also fetches social sentiment from Reddit (ApeWisdom API)

Step 2: AI ANALYSIS
    └─ Sends everything to analyzer.analyze_intraday() or analyze_longterm()
    └─ Builds a structured prompt with all the data
    └─ Calls the LLM with fallback chain: Groq → Z.AI GLM → Gemini
    └─ Parses the JSON response into a Signal (strong_buy/buy/hold/sell/strong_sell)
    └─ If RAG is enabled, retrieves historical context and validates the analysis

Step 3: VERIFICATION (Optional)
    └─ Uses a second LLM call to cross-check the analysis
    └─ Checks: Does the signal match the technicals? Are price levels reasonable?

Step 3.5: ALGO PREDICTION
    └─ Runs algorithmic scoring (StockScorer) — pure math, no AI
    └─ If ML model exists, runs SignalPredictor for a trained prediction
    └─ Classifies market regime (bull/bear/neutral/volatile)
    └─ Calculates position size based on risk parameters
    └─ Sets stop-loss and take-profit levels using ATR

Step 3.6: PAPER TRADING (If Enabled)
    └─ Checks kill switches (max daily loss, stale data, slippage)
    └─ Runs canary gate (symbol allow-list, breach checks)
    └─ Runs pre-trade risk checks (position limits, sector concentration)
    └─ Papers the trade through PaperBroker (simulated execution)
    └─ Records everything to an immutable audit trail

Step 4: STORE
    └─ Gets or creates the stock record in Supabase
    └─ Stores price data, indicators, and the analysis
    └─ Generates vector embeddings (Cohere) for semantic search
    └─ Stores actionable signals for tracking

Step 5: ALERT
    └─ If the signal is actionable (buy/sell), sends notifications
    └─ Formats rich messages for Slack (with blocks) and Telegram (with HTML)
    └─ Records alert delivery status in the database
    └─ Sends API usage summary to Slack
```

---

## 4. Agent Layer — The Specialist Workers

### 4.1 StockFetcher — The Data Collector

**File:** `src/agents/fetcher.py` (1,196 lines)
**Analogy:** The researcher who goes out, gathers all the raw information, and brings it back organized.

**What it fetches and from where:**

| Data Type | Primary Source | Fallback | API Calls |
|-----------|---------------|----------|-----------|
| Live Quotes | Twelve Data | Yahoo Finance (yfinance) | REST API |
| Price History (30+ years) | Twelve Data | Yahoo Finance | REST API |
| Fundamentals (P/E, ROE, etc.) | Yahoo Finance | — | REST API |
| News | Yahoo Finance + Finnhub | Yahoo alone | REST API |
| Social Sentiment | Finnhub | — | REST API |
| Reddit Buzz | ApeWisdom | — | REST API (free, no key) |

**Key Data Structures:**

```python
@dataclass
class StockQuote:
    """Everything you'd see on a stock ticker"""
    symbol: str              # "AAPL"
    price: float             # 185.42
    change: float            # +2.30
    change_percent: float    # +1.26%
    volume: int              # 52,000,000
    avg_volume: int          # 48,000,000
    high: float              # 186.10 (day's high)
    low: float               # 183.50 (day's low)
    open: float              # 184.00
    prev_close: float        # 183.12
    market_cap: int          # 2,850,000,000,000
    pe_ratio: float          # 31.2
    fifty_two_week_high: float
    fifty_two_week_low: float
    timestamp: datetime

@dataclass
class PriceData:
    """One candlestick bar (OHLCV)"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str   # "1d", "1h", "5m", etc.
```

**The Parallel Fetching Pattern (`get_full_analysis_data`):**

This is one of the most important engineering patterns in the project. Instead of calling 5 APIs one-by-one (which would take 5 × 3 seconds = 15 seconds), we call them all at the same time:

```python
with ThreadPoolExecutor(max_workers=5) as pool:
    futures = {
        "quote": pool.submit(self.get_quote, symbol),
        "history": pool.submit(self.get_price_history, symbol, period),
        "fundamentals": pool.submit(self.get_fundamentals, symbol),
        "news_yahoo": pool.submit(self.get_news_yahoo, symbol),
        "news_finnhub": pool.submit(self.get_news_finnhub, symbol),
    }
    # All 5 calls run simultaneously
    # Total time = max(single call) ≈ 3 seconds instead of 15
```

**Technical Indicator Calculations:**

The fetcher calculates all indicators from raw price data using standard financial math (no libraries like TA-Lib). This is intentional — it means you understand exactly what every number means:

- **RSI (Relative Strength Index):** Measures if a stock is overbought (>70) or oversold (<30). Uses Wilder's smoothing method.
- **MACD (Moving Average Convergence Divergence):** Shows momentum by comparing fast vs. slow moving averages.
- **Bollinger Bands:** Shows volatility — price touching upper band = potentially overbought.
- **ATR (Average True Range):** Shows how much a stock typically moves in a day. Used for stop-loss calculations.
- **VWAP (Volume Weighted Average Price):** Institutional benchmark — price above VWAP = institutional buying.
- **ADX (Average Directional Index):** Measures trend strength — ADX > 25 = strong trend.

---

### 4.2 StockAnalyzer — The AI Brain

**File:** `src/agents/analyzer.py` (1,202 lines)
**Analogy:** The senior analyst who reads all the data, applies expertise, and delivers a recommendation.

**Multi-Model LLM Architecture:**

The analyzer doesn't depend on a single AI provider. It uses a **fallback chain** — if one model fails (rate limit, timeout, error), it automatically tries the next one:

```
Task: "analysis"  →  Groq Llama 70B → Z.AI GLM-4.7 → Gemini Flash
Task: "algo"      →  Groq Llama 70B → Z.AI GLM-4.7 → Gemini Flash
Task: "news"      →  Groq Llama 8B  → Gemini Flash  → Z.AI GLM-4.7
Task: "sentiment" →  Groq Llama 8B  → Gemini Flash  → Z.AI GLM-4.7
Task: "chat"      →  Groq Llama 70B → Z.AI GLM-4.7 → Gemini Flash
```

**Why different models for different tasks?** News summarization doesn't need a 70B parameter model — a faster 8B model gives you the same quality at 5× the speed. But stock analysis needs deeper reasoning, so we use the 70B model. This is called **task-based model routing** and it's a real-world production pattern.

**How the LLM call works internally:**

```python
def _call_llm(self, prompt, system_prompt, task="default"):
    models = self._models_for_task(task)  # Get ordered model list

    for model in models:
        try:
            # Build API request with provider-specific config
            kwargs = {"model": model, "messages": [...], "temperature": 0.3}

            if model.startswith("openai/"):  # Z.AI uses OpenAI-compatible API
                kwargs["api_base"] = "https://open.bigmodel.cn/api/coding/paas/v4"
                kwargs["api_key"] = self.zai_key

            response = litellm.completion(**kwargs)  # LiteLLM handles all providers
            return response.content, model, tokens_used

        except Exception:
            continue  # Try next model in the chain

    raise Exception("All models failed")
```

**The Algo Prediction Pipeline (Hybrid AI + Math):**

This is the most sophisticated part. It combines **pure mathematical scoring** with **ML predictions** and **LLM reasoning**:

1. **StockScorer** runs formula-based calculations (no AI involved):
   - Momentum Score (0-100): Based on RSI, MACD, price vs SMA
   - Value Score (0-100): Based on P/E, P/B, dividend yield
   - Quality Score (0-100): Based on ROE, profit margins, debt/equity
   - Risk Score (1-10): Based on ATR volatility, ADX trend strength

2. **ML Model** (if trained and available): Runs the trained classifier for a signal prediction with probability distribution

3. **Market Regime Classification**: Is we in a bull, bear, neutral, or volatile market?

4. **Position Sizing**: Based on signal confidence, volatility, and regime, calculates how much to invest

5. **LLM Reasoning**: Only used to *explain* the scores — never to *generate* them

---

### 4.3 StockStorage — The Memory

**File:** `src/agents/storage.py` (1,724 lines)
**Analogy:** The filing cabinet that remembers everything, and can find related documents by meaning (not just keywords).

**Technology:** Supabase (PostgreSQL) + pgvector (vector database) + Cohere (embeddings)

**What gets stored:**

```
┌──────────────────────────────────────────────────────┐
│                   SUPABASE DATABASE                   │
│                                                       │
│  users         → User accounts & preferences         │
│  stocks        → Master stock list                   │
│  watchlist     → User's tracked stocks               │
│  price_history → Historical OHLCV data               │
│  indicators    → Calculated technical indicators      │
│  analyses      → AI analysis results + embeddings    │   ← pgvector
│  signals       → Actionable trading signals          │
│  alerts        → Notification delivery records       │
│  news          → Stored news articles + embeddings   │   ← pgvector
│  knowledge     → RAG knowledge base + embeddings     │   ← pgvector
└──────────────────────────────────────────────────────┘
```

**Vector Embeddings — The Cool Part:**

When we store an analysis, we don't just store the raw text. We also generate a **vector embedding** — a 1024-dimensional numerical representation of the meaning:

```python
class CohereEmbeddings:
    def embed_text(self, text):
        # "AAPL is showing bullish momentum with strong earnings"
        # → [0.023, -0.112, 0.445, ..., 0.087]  (1024 numbers)
        response = requests.post("https://api.cohere.ai/v1/embed", ...)
        return response.json()["embeddings"][0]
```

**Why?** Because later, when you ask "show me stocks with similar patterns to current AAPL", we can do a **semantic search** — finding analyses that are *meaningfully similar*, not just containing the same keywords. This is the foundation of the RAG (Retrieval-Augmented Generation) system.

---

### 4.4 NotificationManager — The Messenger

**File:** `src/agents/alerts.py` (973 lines)
**Analogy:** The assistant who taps you on the shoulder when something important happens, in the right format for each channel.

**Two channels, tailored formatting:**

**Slack:** Rich block-based messages with color-coded signals:
```
🟢🟢 STRONG BUY — Apple Inc. (AAPL)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Mode: Intraday | Confidence: 85%

RSI oversold at 28 with MACD bullish crossover.
Strong institutional buying above VWAP.

📊 Price: $185.42 → Target: $192.00 | Stop: $181.50
```

**Telegram:** HTML-formatted messages for mobile:
```html
🟢🟢 <b>STRONG BUY — AAPL</b>
Confidence: 85%
Price: $185.42 → Target: $192.00
```

**Retry logic:** If Slack's API returns a rate-limit error, the notification manager waits and retries automatically.

---

### 4.5 StockScorer — The Formula Engine

**File:** `src/agents/scorer.py` (852 lines)
**Analogy:** The math professor who grades stocks using nothing but formulas and data — no opinions, no AI, just numbers.

This is one of the most important engineering distinctions in the project: **scores are algorithmic, not AI-generated**. This means they're deterministic, reproducible, and auditable.

**Momentum Score (0-100):**
```
If RSI < 30 (oversold):       +30 points (bullish signal)
If RSI > 70 (overbought):     +0 points  (bearish signal)
If MACD > Signal line:         +25 points (upward momentum)
If Price > SMA(20):            +25 points (short-term uptrend)
If Price > SMA(50):            +20 points (medium-term uptrend)
```

**Value Score (0-100):**
```
P/E < 15:  +30 points  (undervalued by traditional standards)
P/E 15-25: +20 points  (fairly valued)
P/B < 1.5: +25 points  (trading below book value — classic value)
Dividend > 3%: +20 points
```

**Configurable Weights:**
The system supports different weight presets:
```python
WEIGHT_PRESETS = {
    "balanced":       Momentum=35%, Value=25%, Quality=25%, Risk=15%
    "momentum_focus": Momentum=50%, Value=15%, Quality=15%, Risk=20%
    "value_focus":    Momentum=20%, Value=40%, Quality=25%, Risk=15%
    "conservative":   Momentum=25%, Value=25%, Quality=25%, Risk=25%
}
```

---

### 4.6 ChatAssistant — The Conversationalist

**File:** `src/agents/chat_assistant.py` (1,052 lines)
**Analogy:** A knowledgeable financial advisor you can have a conversation with, who remembers context.

It uses **RAG (Retrieval-Augmented Generation)** — before answering your question, it searches the database for relevant historical analyses, signals, and news. This means when you ask "How has AAPL performed recently?", it doesn't guess — it retrieves actual past analyses from your database and uses them to ground its answer.

**Flow:**
```
User: "What's happening with AAPL?"
  │
  ├─ 1. Extract stock symbols from the question ("AAPL")
  ├─ 2. Ensure stock data exists in DB (fetch if needed)
  ├─ 3. Retrieve RAG context (past analyses, signals, news)
  ├─ 4. Fetch live data (latest quote, indicators)
  ├─ 5. Build prompt with context + live data + conversation history
  ├─ 6. Call LLM with fallback chain
  └─ 7. Return grounded answer with sources cited
```

---

### 4.7 RAGRetriever & RAGValidator — The Knowledge System

**Files:** `src/agents/rag_retriever.py` (618 lines), `src/agents/rag_validator.py` (603 lines)

**RAGRetriever** searches across multiple data sources to find relevant context:
- Similar past analyses (by vector similarity)
- Historical signals (by stock and timeframe)
- Related news articles (by semantic relevance)
- Signals from correlated stocks (by sector)

**RAGValidator** implements RAGAS-style validation metrics:
- **Faithfulness:** Is the answer grounded in retrieved context? (not hallucinated)
- **Context Relevancy:** Are the retrieved documents actually useful?
- **Groundedness:** Can specific claims be traced back to source data?
- **Temporal Validity:** Is the context fresh enough? (intraday needs recent data)

Each analysis gets a letter grade (A through F) based on these metrics.

---

### 4.8 RealtimeManager — The Live Wire

**File:** `src/agents/realtime.py` (254 lines)
**Analogy:** A TV tuned to the stock ticker, constantly updating prices in the background.

Uses **Finnhub WebSocket** for real-time trade data (<100ms latency):

```python
# WebSocket URL: wss://ws.finnhub.io?token=<API_KEY>
# Subscribe:   {"type": "subscribe", "symbol": "AAPL"}
# Receive:     {"data": [{"p": 185.42, "s": "AAPL", "t": 1707..., "v": 100}]}
```

Runs in a background daemon thread. Caches latest prices in-memory so `get_quote()` can return instantly for subscribed symbols instead of making an HTTP call.

---

### 4.9 UsageTracker — The Accountant

**File:** `src/agents/usage_tracker.py` (349 lines)

Tracks API usage across all providers with:
- Per-request breakdowns
- Threshold alerts at 50%, 75%, 90%, 95%, 100% of free tier limits
- Auto-reset based on daily/monthly periods

```
Service Limits:
  Twelve Data:  800 calls/day  (free tier)
  Z.AI GLM:     varies
  Gemini:       Unlimited (free tier)
  Groq:         14,400 requests/day
  Cohere:       1,000 calls/month
  Finnhub:      30,000 calls/day
```

---

## 5. The Training & ML Subsystem

### 5.1 SignalPredictor — The ML Model

**File:** `src/training/predictor.py` (270 lines)

Loads a trained scikit-learn model (saved as `.joblib`) and predicts signals from raw financial data.

**Feature Vector (37+ features):**
```
Base indicators:        RSI, MACD, SMA ratios, Bollinger position, etc.
Factor features:        Fama-French style momentum, value, quality factors
Microstructure features: Volume profile, bid-ask spread, order flow
Sentiment features:     FinBERT scores on headlines, Finnhub sentiment
```

**How it integrates with the main pipeline:**
```
Algo Prediction Pipeline:
  ├─ Try ML Model first (trained data > formulas)
  │   └─ RegimeAwarePredictor checks if regime-specific models exist
  │   └─ Falls back to general SignalPredictor
  ├─ If ML fails, use StockScorer formulas (always available)
  └─ ML signal takes priority when available; formulas are the safety net
```

---

### 5.2 PaperPortfolio — The Simulator

**File:** `src/training/paper_trading.py` (472 lines)

A complete simulated trading environment:
- Opens positions when buy/sell signals fire
- Tracks stop-loss and take-profit hits
- Calculates rolling performance: Sharpe ratio, max drawdown, win rate, turnover
- Determines if the system is "ready for promotion" to live trading

**Promotion Gates:**
```
Must pass ALL to go live:
  ✓ Minimum 10 trades completed
  ✓ Sharpe ratio > 0.0
  ✓ Max drawdown < -20%
  ✓ Win rate > 40%
  ✓ Turnover < 5.0
```

---

### 5.3 BrokerAdapter — The Execution Layer

**File:** `src/training/broker.py` (247 lines)

An **abstract interface** for order execution:

```python
class BrokerAdapter(ABC):
    def submit_order(self, order: Order) -> Fill: ...
    def get_order_status(self, order_id: str) -> dict: ...
```

Currently implements `PaperBroker` (simulated fills). The architecture supports adding a real broker (Alpaca, Interactive Brokers) by implementing the same interface.

**Key design patterns:**
- **Idempotent orders:** Submitting the same `order_id` twice returns the cached fill — no double-execution
- **Exponential backoff retry:** If the broker call fails, waits 1s, 2s, 4s before retrying

---

### 5.4 Risk Management — The Safety Net

Multiple layers of protection:

**Kill Switches** (`src/training/kill_switches.py`):
- Max daily loss exceeded → HALT all trading
- Stale data detected → HALT (don't trade on old prices)
- Excessive slippage → HALT

**Pre-Trade Risk** (`src/training/pre_trade_risk.py`):
- Max single position size (e.g., no more than 10% in one stock)
- Sector concentration limits (e.g., no more than 30% in tech)
- Daily loss limits
- Total exposure caps

**Canary Rollout** (`src/training/canary.py`):
- Trade only on approved symbols first
- Monitor P&L on the canary set before allowing broader trading
- Auto-disable if canary set shows losses

**Audit Trail** (`src/training/audit.py`):
- Every signal, order, fill, risk check, and kill switch activation is recorded
- Immutable append-only log
- Daily reports summarize all activity

---

## 6. Infrastructure & Cross-Cutting Concerns

### 6.1 Configuration Management

**File:** `src/config.py` (244 lines)

Uses **Pydantic BaseSettings** for type-safe configuration:

```python
class Settings(BaseSettings):
    # API Keys (loaded from .env)
    zai_api_key: str = Field(default="", alias="ZAI_API_KEY")
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    supabase_url: str = Field(alias="SUPABASE_URL")

    # Feature Flags
    paper_trading_enabled: bool = Field(default=False)
    kill_switch_enabled: bool = Field(default=True)
    canary_enabled: bool = Field(default=False)

    # Model Configuration
    llm_fallback_order: str = "openai/glm-4.7,gemini/gemini-2.5-flash"

    model_config = SettingsConfigDict(env_file=".env")
```

**Priority order:** Environment variables > `.env` file > Default values

---

### 6.2 Caching Layer

**File:** `src/cache.py` (247 lines)

Two backends with automatic fallback:

```
Cache Strategy:
  Quotes:       60s TTL    (prices change frequently)
  Fundamentals: 1 hour TTL (quarterly data, rarely changes)
  Analysis:     15 min TTL (avoid re-running expensive LLM calls)
  Embeddings:   24 hour TTL (same text always gets same embedding)
  Indicators:   5 min TTL  (technical indicators update continuously)
```

Redis in production, in-memory dict in development. All cache hits/misses are tracked as Prometheus metrics.

---

### 6.3 LLM Output Guardrails

**File:** `src/guardrails.py` (361 lines)

**Every LLM response passes through validation before reaching the user:**

1. **Schema check:** Does the JSON have all required fields?
2. **Signal check:** Is the signal one of the 5 valid values?
3. **Confidence check:** Cap at 0.95 max (LLMs shouldn't be 100% sure about stocks)
4. **Price check:** Is the target price within ±50% of current? (catches hallucinated prices)
5. **Reasoning check:** Is the explanation substantive (>20 words)?
6. **Consistency check:** Does a "buy" signal have bullish reasoning? (catches contradictions)

All violations are logged as Prometheus metrics so you can track how often the LLM misbehaves.

---

### 6.4 Prompt Versioning & A/B Testing

**File:** `src/prompt_manager.py` (313 lines)

Prompts are the "source code" of AI applications. This system versions them:

```python
PROMPTS = {
    "intraday_analysis": {
        "v1": {"system": "You are an expert intraday trader...", "user": "..."},
        "v2": {"system": "You are a senior quantitative trader...", "user": "..."},
    },
    "longterm_analysis": {
        "v1": {"system": "...", "user": "..."},
        "v2": {"system": "...", "user": "..."},
    },
}
```

This lets you A/B test different prompts and measure which version gets better results.

---

### 6.5 Streaming Responses (SSE)

**File:** `src/streaming.py` (261 lines)

Streams LLM tokens to the frontend in real-time using Server-Sent Events:

```
Client request → Server starts generating →
  "data: {"type":"token","content":"The"}\n\n"
  "data: {"type":"token","content":" stock"}\n\n"
  "data: {"type":"token","content":" shows"}\n\n"
  ...
  "data: {"type":"done","model":"groq/llama-3.1-70b","tokens":350}\n\n"
  "data: [DONE]\n\n"
```

Users see tokens appear instantly instead of waiting 5-10 seconds for the complete response.

---

### 6.6 Prometheus Metrics & Monitoring

**File:** `src/metrics.py` (185 lines)
**Config:** `monitoring/prometheus.yml`, `monitoring/grafana/`

Exposes quantitative metrics about the AI system's behavior:

```
stockradar_llm_latency_seconds        → How long each LLM call takes
stockradar_llm_requests_total          → Total LLM calls (by model, status)
stockradar_llm_fallback_total          → How often fallbacks trigger
stockradar_llm_tokens_total            → Total tokens consumed
stockradar_analysis_confidence         → Distribution of confidence scores
stockradar_analysis_signal_total       → Count of each signal type
stockradar_api_cost_usd_total          → Running cost in USD
stockradar_cache_hits_total            → Cache effectiveness
stockradar_guardrail_triggers_total    → How often guardrails fire
```

---

### 6.7 Token Accounting & Cost Estimation

**File:** `src/token_accounting.py` (195 lines)

Every API response includes a `meta` block with full cost transparency:

```json
{
  "meta": {
    "tokens_in": 1200,
    "tokens_out": 350,
    "total_tokens": 1550,
    "cost_usd": 0.004,
    "model_used": "groq/llama-3.1-70b-versatile",
    "models_tried": 1,
    "latency_sec": 1.2,
    "data_sources_used": ["twelvedata", "finnhub", "yfinance"]
  }
}
```

---

## 7. The Backend API (FastAPI)

**File:** `backend/app.py` (641 lines)

Exposes the Python analysis engine to the web frontend:

| Endpoint | Method | What It Does |
|----------|--------|-------------|
| `/v1/analyze` | POST | Start async analysis job |
| `/v1/analyze/{job_id}` | GET | Check analysis job status |
| `/v1/ask` | POST | Chat assistant Q&A |
| `/v1/fundamentals` | GET | Stock fundamentals |
| `/v1/agent/momentum` | GET | Momentum analysis |
| `/v1/agent/rsi-divergence` | GET | RSI divergence detection |
| `/v1/agent/social-sentiment` | GET | Reddit/social buzz |
| `/v1/agent/support-resistance` | GET | Key price levels |
| `/v1/agent/stock-score` | GET | Algorithmic scores |
| `/v1/agent/news-impact` | GET | News sentiment analysis |
| `/v1/health` | GET | System health check |

**Authentication:** Bearer token (`BACKEND_AUTH_TOKEN` env var)

**Job Management:** Long-running analyses run in a background thread pool. The frontend polls for status using the job ID.

---

## 8. The Frontend (Next.js)

**Directory:** `web/` (Next.js + TypeScript + TailwindCSS)

```
web/src/
├── app/           → Pages & routing (Next.js App Router)
│   ├── page.tsx   → Landing page
│   ├── dashboard/ → Main dashboard
│   └── stock/     → Individual stock pages
├── components/    → Reusable UI components
│   ├── Header.tsx
│   ├── StockChart.tsx
│   ├── AIChat.tsx
│   └── ...
├── lib/           → API clients, utilities
│   ├── api.ts     → Backend API integration
│   └── utils.ts
├── hooks/         → Custom React hooks
└── providers/     → Context providers
```

---

## 9. Web3 & Blockchain Layer

### Move Agent Registry (Aptos)
**Directory:** `move-agent-registry/`

A smart contract on Aptos blockchain that registers and tracks AI agents on-chain.

### XMTP Messaging Agent
**Directory:** `xmtp/`

An XMTP messaging bot that lets users interact with Stock Radar through decentralized messaging protocols.

### Blockchain Indexer
**Directory:** `indexer/`

Indexes on-chain events from the agent registry for off-chain analysis.

---

## 10. Database Schema

**Directory:** `migrations/`

Three migration files build the complete schema:

```sql
-- 001_stock_schema.sql: Core tables
CREATE TABLE stocks (id, symbol, name, exchange, sector, currency, ...);
CREATE TABLE price_history (stock_id, timeframe, timestamp, open, high, low, close, volume);
CREATE TABLE analyses (stock_id, mode, signal, confidence, reasoning, embedding vector(1024));
CREATE TABLE signals (stock_id, signal_type, signal, price_at_signal, reason);
CREATE TABLE alerts (user_id, stock_id, channel, message, status);

-- 002_add_algo_prediction.sql: Algo trading support
ALTER TABLE analyses ADD COLUMN algo_prediction jsonb;

-- 003_rag_enhancements.sql: RAG/vector search support
CREATE TABLE knowledge_base (topic, content, embedding vector(1024));
CREATE TABLE news (stock_id, headline, summary, embedding vector(1024));
-- Vector similarity search functions (pgvector)
```

---

## 11. The Complete Analysis Pipeline — Step by Step

Here's what happens when a user enters "AAPL" in the dashboard, from click to alert:

```
1. Frontend sends POST /v1/analyze {symbol: "AAPL", mode: "intraday"}
       │
2. Backend creates background job, returns job_id
       │
3. StockRadar.analyze_stock("AAPL") is called
       │
4. ┌── PARALLEL (ThreadPoolExecutor, 5 workers) ──────┐
   │ get_quote("AAPL")          → StockQuote          │
   │ get_price_history("AAPL")  → [PriceData × 500]   │
   │ get_fundamentals("AAPL")   → {pe_ratio, roe, ...} │
   │ get_news_yahoo("AAPL")     → [NewsItem × 10]      │
   │ get_news_finnhub("AAPL")   → [NewsItem × 15]      │
   └───────────────────────────────────────────────────┘
       │
5. calculate_indicators(price_history)
   → RSI=62, MACD=15.5, SMA20=2800, BB_upper=2900, ATR=45
       │
6. get_social_sentiment("AAPL")
   → {reddit_mentions: 342, rank: #5, sentiment: "bullish"}
       │
7. analyzer.analyze_intraday(symbol, quote, indicators, news, social)
   │
   ├── 7a. _get_rag_context() → Find similar past analyses
   ├── 7b. Build prompt with all data + RAG context
   ├── 7c. _call_llm(prompt, task="analysis")
   │        Try: groq/llama-3.1-70b → ✓ Success in 1.2s
   ├── 7d. Parse JSON response → Signal=BUY, Confidence=0.75
   └── 7e. RAG Validation → Faithfulness=0.85, Grade=B+
       │
8. generate_algo_prediction(symbol, quote, indicators, ...)
   │
   ├── 8a. ML Model → predict(features) → signal=buy, confidence=0.72
   ├── 8b. StockScorer → M=65, V=55, Q=70, R=4
   ├── 8c. classify_market_regime → "neutral" (confidence: 0.6)
   ├── 8d. calculate_position_size → 2.5% of portfolio
   └── 8e. calculate_stop_take_profit → SL=$178.50, TP=$195.00
       │
9. PAPER TRADING (if enabled)
   │
   ├── 9a. Kill switch check → All clear
   ├── 9b. Canary check → Symbol allowed
   ├── 9c. Pre-trade risk → Within limits
   ├── 9d. PaperBroker.submit_order() → Fill confirmed
   └── 9e. Audit trail → Signal logged
       │
10. storage.store_analysis_with_embedding(...)
    → Record saved + Cohere embedding generated
       │
11. storage.store_signal(...)
    → Actionable signal stored for tracking
       │
12. notifications.send_analysis_alert(...)
    │
    ├── Slack: Rich block message sent ✓
    └── Telegram: HTML message sent ✓
       │
13. Frontend polls GET /v1/analyze/{job_id}
    → Returns complete result with signal, scores, reasoning
       │
14. Dashboard renders: BUY signal, 75% confidence, charts, news
```

---

## 12. Data Flow Diagrams

### How Data Flows Through the System:

```
External APIs                    Stock Radar Core                  Users
─────────────              ──────────────────────           ───────────────

Twelve Data ─────┐
Yahoo Finance ───┤         ┌──────────────────┐
Finnhub WS ──────┤─────►   │   StockFetcher   │
Reddit API ──────┤         │  (parallel fetch) │
ApeWisdom ───────┘         └────────┬─────────┘
                                    │ raw data
                                    ▼
                           ┌──────────────────┐
LLM Providers:             │  StockAnalyzer   │
  Groq ──────────┐         │  (AI + formulas) │
  Z.AI GLM ──────┤─────►   │                  │────► CLI Terminal
  Gemini ────────┘         └────────┬─────────┘
                                    │ analysis results
                                    ▼
Cohere API ──────┐         ┌──────────────────┐
                 ├─────►   │  StockStorage    │────► FastAPI ────► Next.js
Supabase ────────┘         │  (DB + vectors)  │
                           └────────┬─────────┘
                                    │ notifications
                                    ▼
                           ┌──────────────────┐
                           │ NotificationMgr  │────► Slack
                           │  (alerts)        │────► Telegram
                           └──────────────────┘
```

---

## 13. CLI Commands Reference

```bash
# Analyze a single stock
python main.py analyze AAPL --mode intraday --period max
python main.py analyze RELIANCE.NS --mode longterm --no-alert

# Analyze a user's watchlist
python main.py watchlist user@email.com --mode intraday

# Explain why a stock is moving
python main.py explain TSLA

# Continuous monitoring (every 15 minutes)
python main.py continuous AAPL MSFT GOOGL --interval 15

# Interactive AI chat
python main.py chat --symbol AAPL

# Single question
python main.py ask "Is AAPL a good buy right now?"

# Paper trading management
python main.py paper status        # View positions & performance
python main.py paper trades        # View closed trades
python main.py paper dashboard     # Full terminal dashboard
python main.py paper reset         # Clear all paper trading data

# Canary rollout controls
python main.py canary status       # View canary state
python main.py canary enable       # Enable canary mode
python main.py canary disable      # Disable canary mode

# Audit trail
python main.py audit report --date 2026-03-07

# Backfill historical data
python main.py backfill AAPL MSFT --period max --clear

# API usage monitoring
python main.py usage               # View API usage
python main.py usage --reset       # Reset all counters

# Test all connections
python main.py test
```

---

## 14. Deployment Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     PRODUCTION                           │
│                                                         │
│  ┌─────────────┐     ┌──────────────┐                   │
│  │  Vercel      │     │  Railway      │                   │
│  │  (Frontend)  │────►│  (Backend)    │                   │
│  │  Next.js     │     │  FastAPI      │                   │
│  └─────────────┘     └──────┬───────┘                   │
│                             │                            │
│            ┌────────────────┼────────────────┐           │
│            ▼                ▼                ▼           │
│     ┌────────────┐  ┌────────────┐  ┌────────────┐      │
│     │  Supabase  │  │ Prometheus │  │   Redis    │      │
│     │  (Postgres │  │  + Grafana │  │  (Cache)   │      │
│     │  +pgvector)│  │            │  │            │      │
│     └────────────┘  └────────────┘  └────────────┘      │
│                                                         │
│  Docker Compose: docker-compose.yml                      │
│  Dockerfiles: Dockerfile, Dockerfile.backend             │
└─────────────────────────────────────────────────────────┘
```

---

## 15. Design Decisions & Trade-offs

### Why LiteLLM instead of direct API calls?
LiteLLM provides a single interface for all LLM providers. It means switching from Groq to Anthropic requires changing one string, not rewriting API integration code.

### Why calculate indicators manually instead of using TA-Lib?
1. **Understanding:** You know exactly what every number means
2. **Portability:** No native C library dependency
3. **Transparency:** Interview-ready — you can explain the math

### Why StockScorer exists alongside the ML model?
The ML model might not be trained yet, or might fail. The scorer provides a reliable, deterministic fallback that never fails. It's also more explainable — you can trace exactly why a stock scored 65 on momentum.

### Why Cohere for embeddings instead of OpenAI?
Cohere's embedding model (`embed-english-v3.0`) produces high-quality 1024-dimensional vectors, has a generous free tier (1,000 calls/month), and is specifically designed for search/retrieval use cases.

### Why paper trading before live trading?
Professional quant shops never deploy directly to production. Paper trading validates that the model's signals actually make money with properly sized positions and risk controls before real capital is at risk.

### Why kill switches and canary rollouts?
These are production-grade safety mechanisms borrowed from Facebook's and Google's deployment practices. If a model starts making bad trades, the kill switch halts everything automatically. The canary system lets you test with a small subset before going wide.

---

> **Built by Darshan** — An AI-powered stock analysis platform that brings institutional-grade intelligence to individual traders.
