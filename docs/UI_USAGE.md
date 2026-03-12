# Stock Radar — Web Dashboard Usage Guide

A complete walkthrough of every page in the Stock Radar web interface with explanations of what each element does and how to use it effectively.

---

## Quick Start

```bash
# Start the development server
cd web
npm run dev

# Open in browser
open http://localhost:3000
```

---

## Table of Contents

1. [Dashboard (Home)](#1-dashboard-home)
2. [Stocks (Watchlist)](#2-stocks-watchlist)
3. [Stock Detail Page](#3-stock-detail-page)
4. [AI Chat](#4-ai-chat)
5. [Signals](#5-signals)
6. [API Usage](#6-api-usage)
7. [Settings](#7-settings)
8. [Documentation](#8-documentation)
9. [x402 Demo](#9-x402-demo)

---

## Navigation

The app uses a **fixed top header** with horizontal navigation. Every page is one click away:

| Nav Item | Icon | Route | Purpose |
|----------|------|-------|---------|
| **Dashboard** | 🏠 LayoutDashboard | `/` | Market overview, intro, and recent signals |
| **Stocks** | 📈 TrendingUp | `/stocks` | Your watchlist — add, track, and analyze stocks |
| **Signals** | ⚡ Zap | `/signals` | Historical trading signals from all analyses |
| **Docs** | 📖 BookOpen | `/docs` | Architecture & engineering documentation |
| **API Usage** | 📊 BarChart3 | `/usage` | Monitor API requests and rate limits |
| **x402 Demo** | 💳 CreditCard | `/x402-demo` | Aptos blockchain integration demo |
| **Settings** | ⚙️ Settings | `/settings` | Preferences, notifications, analysis config |

A **theme toggle** (☀️/🌙) in the header lets you switch between light and dark mode.

---

## 1. Dashboard (Home)

**Route**: `/`

This is the landing page. It features the **Project Intro** — a visually rich, animated section that explains what Stock Radar is, its architecture, and the tech stack.

### What you see:

- **Hero section** with the Stock Radar logo, tagline, and a "Get Started" call-to-action
- **Architecture overview** — animated breakdown of the 5-step pipeline (Fetch → Analyze → Verify → Store → Alert)
- **Tech stack cards** — all technologies used, organized by category
- **Component deep dive** — interactive cards explaining each agent (Fetcher, Analyzer, Scorer, Storage, Alerts)
- **Live data flow visualization** — shows how data moves through the system

### What to do here:

1. **Read through the intro** if you're new to the project
2. Click **"Get Started"** or the **Stocks** nav link to begin analyzing stocks

---

## 2. Stocks (Watchlist)

**Route**: `/stocks`

This is your central hub for managing stocks and running analyses. Think of it as your personal watchlist + analysis launcher.

### How to Add a Stock

1. Type a stock symbol in the **search input** at the top (e.g., `AAPL`, `RELIANCE.NS`, `TCS.BO`)
2. Click the **"Add"** button (or press Enter)
3. The stock appears as a card in your watchlist
4. The system will automatically fetch a live quote snapshot (current price, change %)

### How to Analyze a Stock

1. Find the stock card in your watchlist
2. Select **Analysis Mode** at the top:
   - **Intraday** — Short-term trading signals (day trading, swing trading)
   - **Long-term** — Investment recommendations (weeks to months)
3. Click the **"Analyze"** button on the stock card
4. Watch the progress:
   - **Queued** — Waiting to start
   - **Running** — AI is crunching the data (typically 4-8 seconds)
   - **Succeeded** — Analysis complete! Click the stock to see full results
   - **Failed** — Something went wrong (check backend logs)

### Stock Card Information

Each stock card displays:

| Element | Description |
|---------|-------------|
| **Symbol** | Stock ticker (e.g., AAPL) |
| **Live Price** | Current market price with real-time updates when available |
| **Change %** | Day's price change in percentage (green = up, red = down) |
| **Sparkline** | Mini chart showing recent price movement at a glance |
| **Analysis Count** | Number of times this stock has been analyzed |
| **Analyze Button** | Triggers the full AI analysis pipeline |
| **Click to Open** | Clicking the card navigates to the Stock Detail page |

### Tips

- **Analysis mode persists** — the toggle remembers your preference
- **Mode affects history resolution** — Intraday fetches 5-day/15-min bars, Long-term fetches 5-year/weekly bars
- **Add Indian stocks** by appending `.NS` (NSE) or `.BO` (BSE) — e.g., `RELIANCE.NS`
- The watchlist data comes from Supabase, so it persists across sessions

---

## 3. Stock Detail Page

**Route**: `/stocks/[symbol]` (e.g., `/stocks/AAPL`)

This is the richest page in the app. After clicking a stock card, you land here — a Bloomberg-style dashboard with everything about the stock.

### Layout Overview

The page has **four main panels** plus a **chart area**:

```
┌──────────────────────────────────────────────┐
│  Stock Header (Name, Price, Change, Analyze) │
├──────────────────┬───────────────────────────│
│                  │                           │
│  Price Chart     │  Stock Info Panel          │
│  (Candlestick)   │  (Fundamentals, Quote,     │
│                  │   Indicators, News)         │
│                  │                           │
├──────────────────┴───────────────────────────│
│  AI Analysis Panel (Signal, Reasoning, Algo) │
├──────────────────────────────────────────────│
│  RAG Insights Panel (Context, Validation)    │
├──────────────────────────────────────────────│
│  Advanced Charts Panel (Technical Charts)    │
└──────────────────────────────────────────────┘
```

### The Chart

- **Candlestick chart** powered by TradingView's lightweight-charts library
- **Period selector**: 1D, 1W, 1M, 3M, 6M, 1Y, 3Y, 5Y, All
- **Volume bars** displayed below the candles
- **Live price ticker** shows real-time updates when the market is open (via Finnhub WebSocket)
- Intraday (1D) uses a **line chart** for cleaner visualization

### Stock Info Panel

This panel shows everything the `StockFetcher` gathered, organized into tabs:

#### Quote Tab
- Current Price, Open, High, Low, Close
- Volume vs Average Volume (bar comparison)
- Market Cap
- 52-Week High/Low
- Change from open and previous close

#### Fundamentals Tab
- **Valuation**: P/E, Forward P/E, PEG, P/B, P/S
- **Profitability**: ROE, ROA, Profit Margin, Operating Margin
- **Growth**: Revenue Growth, Earnings Growth
- **Health**: Current Ratio, D/E, Free Cash Flow
- **Dividends**: Yield, Payout Ratio
- Each metric has a visual meter showing how it compares to benchmarks

#### Indicators Tab
- RSI (with overbought/oversold zones highlighted)
- MACD (with signal line crossovers)
- Bollinger Bands (upper, middle, lower)
- ATR (Average True Range)
- VWAP (Volume Weighted Average Price)
- ADX (Average Directional Index)

#### News Tab
- Recent news headlines from Yahoo Finance and Finnhub
- Source attribution and published timestamps
- Click any headline to read the full article

### AI Analysis Panel

After running an analysis (clicking the **Analyze** button at the top), this panel lights up with:

#### Signal Badge
- Color-coded badge: **STRONG BUY** (bright green), **BUY** (green), **HOLD** (amber), **SELL** (red), **STRONG SELL** (dark red)
- **Confidence percentage** displayed as a circular gauge

#### Analysis Details
- **Reasoning** — The LLM's full written analysis explaining why it chose this signal
- **Technical Summary** — One-paragraph technical analysis
- **Sentiment Summary** — News and social sentiment assessment
- **Key Levels** displayed as cards:
  - **Support** — Where the LLM expects buying pressure
  - **Resistance** — Where the LLM expects selling pressure
  - **Target Price** — Expected price move in the signal direction
  - **Stop Loss** — Recommended exit point to limit losses
  - **Risk-Reward Ratio** — Automatically calculated from target and stop loss

#### Algo Intelligence Section
- **Market Regime**: Bull / Neutral / Bear / Volatile
- **Score Gauges**:
  - Momentum Score (0-100) — RSI, MACD, price vs SMA
  - Value Score (0-100) — P/E, P/B, dividend yield
  - Quality Score (0-100) — ROE, margins, debt
  - Risk Score (1-10) — ATR volatility, ADX, debt levels
- **Composite Score** — Weighted blend of all scores
- **Position Size** — Recommended portfolio allocation %
- **Scoring Method** — Whether it used the ML model, regime-aware predictor, or formula scorer
- **Model Version** — Exact model pipeline used

#### Model & Token Info
- LLM model used (e.g., `groq/llama-3.1-70b-versatile`)
- Token count (input + output) with cost estimate — powered by **TokenAccountant** (`token_accounting.py`)
- Analysis duration
- Verification status

#### Guardrails Section

Every LLM response passes through the **GuardrailEngine** before reaching the UI. The analysis payload includes a `guardrails` key showing:

- **Passed** — Whether all checks passed (true/false)
- **Error Count** — Critical issues (schema violations, invalid signals)
- **Warning Count** — Adjusted values (confidence capped, price sanity corrections)
- **Issues** — Details of each issue with rule name, severity, and what was adjusted

This ensures the AI never sends a nonsensical signal (e.g., 150% confidence, or a "buy" signal with bearish reasoning).

### RAG Insights Panel

Shows the **Retrieval Augmented Generation** context used during analysis:

- **Similar Past Analyses** — Previous analyses of this stock that were semantically similar to the current context
- **Historical Signals** — Past trading signals and their outcomes
- **Related News** — News articles retrieved via vector similarity
- **RAG Quality Badge** — Grades the analysis on:
  - **Faithfulness** — Does the analysis stay true to the data?
  - **Relevancy** — Is the retrieved context actually relevant?
  - **Groundedness** — Are claims supported by evidence?
  - **Temporal Validity** — Is the analysis using current (not stale) context?
  - Overall letter grade (A through F)

### Advanced Charts Panel

Extended technical charts:
- **RSI Chart** — Full RSI line with overbought (70) and oversold (30) zones
- **MACD Chart** — MACD line, signal line, and histogram
- **Volume Chart** — Volume bars with moving average overlay
- **Volatility Chart** — ATR and Bollinger Band width

### Running Analysis from the Detail Page

1. Select your **Analysis Mode** (Intraday / Long-term) using the toggle at the top
2. Click the **"Analyze"** button in the header
3. The button shows a spinner during analysis
4. Once complete, all panels (AI Analysis, RAG, Algo Intelligence) populate with fresh data

---

## 4. AI Chat

**Route**: `/stocks/[symbol]/chat` (per-stock chat)

The AI chat is **embedded within each stock's detail page**. To access it:

1. Navigate to a stock that has **at least one analysis** on record
2. Click the **Chat button** (speech bubble icon) on the stock detail page
3. This opens a dedicated chat page focused on that specific stock

### Chat Features

- **Context-aware** — The AI knows the stock's full analysis history, signals, indicators, and news
- **RAG-powered** — Retrieves relevant past analyses and signals via vector similarity search
- **Multi-turn conversation** — Ask follow-up questions; the AI maintains context across the conversation
- **Source attribution** — Each response shows which data sources were used (past analyses, signals, news)
- **Model transparency** — Shows which LLM model was used, token count, and response time
- **Token accounting & fallback metrics** — Every chat interaction is tracked via `TokenAccountant`, and LLM fallback events are recorded as Prometheus metrics (`chat_assistant.py:595, 784`)

### Example Conversations

```
You: What's the current sentiment on this stock?

AI: Based on 3 recent analyses and 12 news articles...
    The overall sentiment is moderately bullish (68% confidence).
    Reddit mentions have increased 45% this week, mostly positive.
    
    Sources: 2 analyses, 5 news articles, social sentiment data
    [groq/llama-3.1-70b | 842 tokens | 1.2s]

You: How does the current RSI compare to last month?

AI: The RSI has moved from 71 (slightly overbought last month)
    to 58 (neutral-bullish currently), suggesting the stock has
    cooled off from overbought territory and may have more room
    to run...
```

### Important Notes

- **No analysis = no chat** — Stocks without any past analysis will show a disabled (greyed-out) chat button with an info tooltip explaining why
- **Chat is per-stock** — Each stock has its own isolated conversation context
- The chat page redirects to the `/chat` info page if accessed without stock context

---

## 5. Signals

**Route**: `/signals`

A centralized feed of every trading signal generated by any analysis across all stocks. Think of it as your signal history dashboard.

### Table Columns

| Column | Description |
|--------|-------------|
| **Symbol** | Stock ticker that was analyzed |
| **Signal** | Color-coded badge: BUY (green), SELL (red), HOLD (amber) |
| **Confidence** | How confident the AI was (0-100%) |
| **Target** | LLM's predicted target price |
| **Stop Loss** | Recommended stop-loss level |
| **Model** | Which LLM produced this signal (abbreviated) |
| **Time** | When the analysis was run |

### Features

- **Real-time updates** — New signals appear automatically via Supabase real-time subscriptions (no manual refresh needed)
- **Last 50 signals** displayed by default, newest first
- Click any row's symbol to jump to that stock's detail page

---

## 6. API Usage

**Route**: `/usage`

Monitor your API consumption across all external services to ensure you stay within rate limits.

### Summary Cards (Top)

Three top-level cards give you the high-level picture:

| Card | What It Shows |
|------|---------------|
| **Total Requests** | Sum of all API calls across all services |
| **Total Tokens** | Combined LLM tokens consumed |
| **Last Reset** | When counters were last cleared |

### Service Cards

Each API provider gets its own card showing:

| Element | Description |
|---------|-------------|
| **Service Name** | The API provider (ZAI, Gemini, Cohere, Finnhub, Ollama) |
| **Request Count** | Number of API calls made vs limit |
| **Progress Bar** | Visual fill indicator (green → yellow → red as you approach limits) |
| **Token Count** | LLM tokens consumed (for LLM services) |
| **Period** | Rate limit window (per minute, per month, unlimited) |
| **Warnings** | ⚡ Yellow warning at 75% usage, ⚠️ Red critical at 90% |

### CLI Integration

The page also links to equivalent CLI commands:
- `python3 main.py usage` — Same data in the terminal
- `python3 main.py usage --reset` — Reset all counters

### Auto-Refresh

The page **automatically refreshes every 30 seconds** so you always see current data.

---

## 7. Settings

**Route**: `/settings`

Configure how Stock Radar behaves — notifications, analysis defaults, and environment readiness.

### Summary Cards (Top)

Three quick-glance cards at the top:

| Card | What It Shows |
|------|---------------|
| **Default Analysis** | Current analysis mode (Intraday or Long-term) |
| **Confidence Threshold** | Minimum confidence % for triggering alerts |
| **Notification Channels** | Number of active notification channels |

### Notifications Section

Toggle and configure notification delivery:

| Setting | Options | What It Does |
|---------|---------|--------------|
| **Slack Notifications** | On/Off + Configure | Enable/disable Slack alerts. Configure via `SLACK_CHANNEL_ID` in `.env` |
| **Email Notifications** | On/Off + Configure | Enable/disable email alerts (coming soon) |

Clicking **"Configure"** opens a dialog to set the channel ID or email address.

### Analysis Settings

| Setting | Options | What It Does |
|---------|---------|--------------|
| **Default Mode** | Intraday / Long-term | Which mode is selected by default on the Stocks page |
| **Confidence Threshold** | 50% / 60% / 70% / 80% / 90% | Minimum confidence for a signal to trigger notifications |

The confidence threshold slider gives visual feedback — a progress bar fills to show the selected threshold.

### Environment Status

A grid showing readiness checks:

| Check | Status | What It Means |
|-------|--------|---------------|
| **Supabase Client Env** | Configured / Missing | Whether `NEXT_PUBLIC_SUPABASE_URL` and `ANON_KEY` are set |
| **Slack Channel** | Enabled / Disabled | Whether Slack notifications are toggled on |
| **Email Channel** | Enabled / Disabled | Whether Email notifications are toggled on |
| **AI Mode** | Intraday / Long-term | Currently configured analysis mode |

### Persistence

All settings are saved to **localStorage**, so they persist across page reloads and browser sessions. No server-side configuration needed for the web UI.

---

## 8. Documentation

**Route**: `/docs`

A comprehensive, interactive documentation page covering the entire Stock Radar architecture:

- **Architecture overview** with a scrollspy sidebar for easy navigation
- **Sections covered**:
  1. High-Level Architecture (layer diagram)
  2. Data Models (Pydantic schemas)
  3. Analysis Pipeline (5-step walkthrough)
  4. Core Components (Fetcher, Analyzer, Storage, Alerts, Scorer, Predictor, RAG)
  5. ML & Training Subsystem
  6. Monitoring & Observability (Prometheus metrics)
  7. Backend API (FastAPI endpoints)
  8. Full Pipeline Walkthrough (step-by-step trace of a real analysis)
  9. CLI Commands Reference
  10. Design Decisions & Trade-offs

Each section uses **interactive components**: expandable code blocks, data tables, info cards, and subsection collapsibles.

---

## 9. x402 Demo

**Route**: `/x402-demo`

A reference implementation of the Aptos blockchain integration using the x402 payment protocol.

> **Note:** The `petra-wallet-provider.tsx` and `wallet-connect-button.tsx` components have been removed. The x402 demo page remains as a reference for the payment protocol concept, but wallet connection is no longer wired into the main application flow.

---

## Common Workflows

### "I want to analyze a stock for the first time"

1. Go to **Stocks** (`/stocks`)
2. Type the ticker in the search bar (e.g., `AAPL`)
3. Click **Add**
4. Select **Intraday** or **Long-term** mode
5. Click **Analyze** on the stock card
6. Wait for completion (~5 seconds)
7. Click the stock card to open the detail page
8. Explore: AI Analysis, Algo Scores, RAG Insights, Charts

### "I want to compare multiple stocks"

1. Add all stocks to your watchlist
2. Analyze each one (same mode for fair comparison)
3. Check the **Signals** page to see all results side-by-side in a table
4. Open each stock's detail page to compare fundamentals, indicators, and AI reasoning

### "I want to chat with the AI about a stock"

1. Make sure the stock has at least **one analysis** on record
2. Go to the stock's detail page (`/stocks/AAPL`)
3. Click the **Chat** button (speech bubble icon)
4. Ask questions — the AI has full context of the stock's analysis history

### "I want to monitor my API costs"

1. Go to **API Usage** (`/usage`)
2. Check each service card for usage percentages
3. If approaching limits, use `python3 main.py usage --reset` from the terminal

### "I want to change notification behavior"

1. Go to **Settings** (`/settings`)
2. Toggle Slack/Email on or off
3. Adjust the **Confidence Threshold** — higher = fewer but more confident alerts
4. Switch the **Default Mode** between Intraday and Long-term

---

> **Built by Darshan** — AI-powered stock analysis bringing institutional-grade intelligence to individual traders.
