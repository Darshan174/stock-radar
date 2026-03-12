# Stock Radar — CLI Usage Guide

Everything you can do from the terminal, explained with real examples.

---

## Quick Start

```bash
# Make sure you're in the project root and your .env file is configured
cd stock-radar

# Test that all connections are working
python main.py test

# Run your first analysis
python main.py analyze AAPL --mode intraday
```

---

## Table of Contents

1. [analyze](#1-analyze--analyze-a-single-stock)
2. [watchlist](#2-watchlist--analyze-your-entire-watchlist)
3. [explain](#3-explain--why-is-a-stock-moving)
4. [continuous](#4-continuous--auto-pilot-monitoring)
5. [chat](#5-chat--interactive-ai-assistant)
6. [ask](#6-ask--single-question-one-shot)
7. [paper](#7-paper--paper-trading-management)
8. [canary](#8-canary--canary-rollout-controls)
9. [audit](#9-audit--immutable-audit-trail)
10. [backfill](#10-backfill--historical-data-import)
11. [usage](#11-usage--api-consumption-tracking)
12. [test](#12-test--connection-health-check)

---

## 1. `analyze` — Analyze a Single Stock

This is the core command. It runs the **full 5-step pipeline**: fetch data → AI analysis → verification → store → alert.

### Syntax

```bash
python main.py analyze <SYMBOL> [--mode intraday|longterm] [--period PERIOD] [--no-alert] [--no-verify]
```

### Arguments & Flags

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `SYMBOL` | *(required)* | Stock ticker. US stocks: `AAPL`, `GOOGL`. Indian stocks: `RELIANCE.NS`, `TCS.BO` |
| `--mode`, `-m` | `intraday` | `intraday` = short-term trading signals. `longterm` = investment recommendations |
| `--period`, `-p` | `max` | History window: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `max` |
| `--no-alert` | `false` | Skip sending Slack/Telegram notifications |
| `--no-verify` | `false` | Skip the Ollama verification step |

### Examples

```bash
# Intraday analysis with default settings (sends alerts)
python main.py analyze AAPL

# Long-term investment analysis, no alerts
python main.py analyze RELIANCE.NS --mode longterm --no-alert

# Intraday with 1 month of history, skip verification
python main.py analyze TSLA -m intraday -p 1mo --no-verify

# US tech stock, full history
python main.py analyze MSFT --period max
```

### What You'll See

```
2026-03-09 23:00:01 - Starting intraday analysis for AAPL
[1/5] Fetching data for AAPL...
   Current price: 227.48
[2/5] Running intraday AI analysis...
[2.5/5] Fetching social media sentiment...
   Reddit mentions: 342
[3/5] Skipping verification (Ollama not available)
[3.5/5] Generating AI algo trading prediction...
   Algo Signal: buy (momentum=72, value=58, quality=81)
[4/5] Storing analysis in database...
[5/5] Sending notifications...
   Alerts sent to: ['slack', 'telegram']

==================================================
ANALYSIS RESULT: AAPL
==================================================
Company: Apple Inc.
Mode: intraday
Signal: BUY
Confidence: 78%
Current Price: 227.48
Target Price: 242.00
Stop Loss: 218.50

Reasoning:
Apple shows strong momentum with RSI at 58 (neutral-bullish zone)...

Model: groq/llama-3.1-70b-versatile
Duration: 4.2s

──────────────────────────────────────────────────
ALGO INTELLIGENCE
──────────────────────────────────────────────────
Market Regime:   neutral (formula_scorer)
Momentum Score:  72/100
Value Score:     58/100
Quality Score:   81/100
Risk Score:      3/10
Position Size:   2.5%
Algo SL / TP:    $218.50 / $242.00

──────────────────────────────────────────────────
GUARDRAILS
──────────────────────────────────────────────────
Passed:          True
Errors:          0
Warnings:        1  (confidence_capped: 0.82 → 0.78)
```

### What Happens Under the Hood

1. `StockFetcher.get_full_analysis_data()` runs up to 6 API calls **in parallel**
2. Mode-aware history config selects the right bar size (intraday → 15-min bars, longterm → weekly bars)
3. Technical indicators are calculated: RSI, MACD, Bollinger Bands, ATR, VWAP, ADX
4. Reddit/social sentiment is fetched from ApeWisdom
5. The LLM analyzes everything with RAG context from past analyses
6. **Guardrails** validate the LLM output — schema, signal, confidence cap, price sanity, reasoning quality, consistency (`guardrails.py`)
7. **Token accounting** tracks every LLM call's tokens, cost, and latency — results appear in the `meta` block (`token_accounting.py`)
8. `StockScorer` runs pure-math scoring (Momentum, Value, Quality, Risk)
9. ML model predicts if trained (RegimeAwarePredictor → SignalPredictor)
10. All metrics (latency, tokens, cost, guardrail triggers) are pushed to **Prometheus** (`metrics.py`)
11. Results are stored in Supabase with Cohere vector embedding
12. Alerts go to Slack (rich block messages) and Telegram (HTML formatting)
13. Guardrail results are exposed in the returned payload (`main.py:733`)

> **Note:** When running via CLI, a standalone Prometheus metrics server starts on port 9090. When running via the FastAPI backend, metrics are served at `GET /metrics` instead (the standalone server is disabled).

---

## 2. `watchlist` — Analyze Your Entire Watchlist

Runs the full analysis pipeline on every stock in a user's saved watchlist from the database.

### Syntax

```bash
python main.py watchlist <EMAIL> [--mode intraday|longterm] [--no-alert]
```

### Arguments & Flags

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `EMAIL` | *(required)* | User email linked to the watchlist in Supabase |
| `--mode`, `-m` | per-stock setting | Override all stocks to one mode |
| `--no-alert` | `false` | Skip notifications for all stocks |

### Examples

```bash
# Analyze all stocks in a user's watchlist
python main.py watchlist darshan@email.com

# Override everything to longterm mode, no alerts
python main.py watchlist darshan@email.com --mode longterm --no-alert
```

### What You'll See

```
Analyzing watchlist for darshan@email.com
Found 5 stocks in watchlist

Analyzing AAPL...
   Signal: BUY (78%)
Analyzing GOOGL...
   Signal: HOLD (65%)
Analyzing RELIANCE.NS...
   Signal: STRONG_BUY (82%)
...

==================================================
WATCHLIST ANALYSIS SUMMARY
==================================================
AAPL: BUY (78%)
GOOGL: HOLD (65%)
RELIANCE.NS: STRONG_BUY (82%)
MSFT: BUY (71%)
TCS.NS: HOLD (60%)
```

---

## 3. `explain` — Why Is a Stock Moving?

Fetches recent data and asks the AI to explain the stock's current price movement. Great for understanding sudden moves.

### Syntax

```bash
python main.py explain <SYMBOL>
```

### Examples

```bash
python main.py explain TSLA
python main.py explain RELIANCE.NS
```

### What You'll See

```
==================================================
WHY IS TSLA MOVING?
==================================================
Current Price: 178.32
Change: -4.58%

Tesla is experiencing significant selling pressure today driven by:
1. Deliveries miss: Q1 deliveries came in at 386,810 vs expected 425,000
2. Margin compression concerns following recent price cuts in China
3. Broader EV sector rotation as investors move to traditional automakers
...
```

---

## 4. `continuous` — Auto-Pilot Monitoring

Runs analysis on a list of stocks at regular intervals. Think of it as a cron job built into the app.

### Syntax

```bash
python main.py continuous <SYMBOL1> <SYMBOL2> ... [--mode intraday|longterm] [--interval MINUTES] [--iterations N]
```

### Arguments & Flags

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `SYMBOL1 SYMBOL2 ...` | *(required)* | One or more stock symbols |
| `--mode`, `-m` | `intraday` | Trading mode |
| `--interval`, `-i` | `15` | Minutes between analysis runs |
| `--iterations`, `-n` | infinite | Stop after N iterations |

### Examples

```bash
# Monitor FAANG stocks every 15 minutes forever
python main.py continuous AAPL GOOGL MSFT AMZN META

# Every 30 minutes, intraday, stop after 10 runs
python main.py continuous AAPL TSLA -m intraday -i 30 -n 10

# Long-term check every hour
python main.py continuous RELIANCE.NS TCS.NS -m longterm -i 60
```

### What You'll See

```
Starting continuous analysis for 5 stocks
Mode: intraday, Interval: 15 minutes

==================================================
ITERATION 1 - 2026-03-09 23:00:00
==================================================
Analyzing AAPL... Signal: BUY (78%)
Analyzing GOOGL... Signal: HOLD (65%)
...

Sleeping for 15 minutes...

==================================================
ITERATION 2 - 2026-03-09 23:15:00
==================================================
...
```

---

## 5. `chat` — Interactive AI Assistant

Opens a REPL-style chat session. The assistant has access to the full database — past analyses, signals, news, and price history — via RAG.

### Syntax

```bash
python main.py chat [--symbol SYMBOL]
```

### Examples

```bash
# General market chat
python main.py chat

# Chat focused on a specific stock
python main.py chat --symbol AAPL
```

### What You'll See

```
Stock Radar AI Chat
Type 'quit' to exit, 'clear' to reset conversation

You: What's your take on AAPL right now?

AI: Based on recent analysis data, AAPL is showing a BUY signal at 78%
    confidence. The RSI is at 58 (neutral-bullish), MACD is positive,
    and the stock is trading above its 50-day SMA. Reddit sentiment
    shows 342 mentions with bullish bias. Historical patterns suggest
    similar setups have yielded +3-5% in the following week.

    Sources: 3 past analyses, 2 signals, 5 news articles

You: Compare it with MSFT

AI: Comparing the two mega-cap tech stocks...

You: quit
Goodbye!
```

---

## 6. `ask` — Single Question (One-Shot)

Ask a single question without entering the interactive chat loop. Great for scripting or quick lookups.

### Syntax

```bash
python main.py ask "<QUESTION>" [--symbol SYMBOL]
```

### Examples

```bash
# General question
python main.py ask "Which tech stocks have the strongest momentum right now?"

# Question about a specific stock
python main.py ask "Is AAPL a good buy?" --symbol AAPL

# Fundamentals question
python main.py ask "What's the P/E ratio trend for RELIANCE.NS?" -s RELIANCE.NS
```

### What You'll See

```
Processing your question...

============================================================
ANSWER
============================================================

Based on recent analyses in the database, AAPL shows a BUY signal...

Stocks mentioned: AAPL, MSFT, GOOGL

Sources used: 3
  - analysis: AAPL
  - signal: AAPL
  - news: Apple reports record Q1 revenue...

[groq/llama-3.1-70b-versatile | 1,247 tokens | 1,823ms]
```

---

## 7. `paper` — Paper Trading Management

Manage the simulated trading environment. Paper trading records every signal as a virtual trade with position sizing, stop-loss, and take-profit tracking.

### Syntax

```bash
python main.py paper <status|trades|dashboard|reset>
```

### Sub-commands

| Action | Description |
|--------|-------------|
| `status` | View open positions and performance summary |
| `trades` | List all closed trades with P&L |
| `dashboard` | Full terminal dashboard (positions, PnL, regimes, kill switches, promotion gates) |
| `reset` | Clear all paper trading data (asks for confirmation) |

### Examples

```bash
# Quick position check
python main.py paper status

# See all closed trades
python main.py paper trades

# Full dashboard (the most useful view)
python main.py paper dashboard

# Reset everything (requires typing "yes")
python main.py paper reset
```

### Dashboard Output

```
============================================================
  PAPER TRADING DASHBOARD
============================================================

--- PnL Summary ---
  Total Trades:   47
  Win Rate:       63.8%
  Avg P&L:        +1.42%
  Total P&L:      +66.74%
  Best Trade:     +8.53%
  Worst Trade:    -4.21%

--- Rolling Window (last 50 trades) ---
  Sharpe:         1.4532
  Max Drawdown:   -6.82%
  Win Rate:       63.8%
  Turnover:       2.14

--- Open Exposure (3 positions) ---
  AAPL: long @ 222.50 size=0.0250 SL=215.0 TP=240.0
  GOOGL: long @ 171.30 size=0.0180 SL=165.0 TP=185.0
  TSLA: short @ 178.50 size=0.0150 SL=190.0 TP=160.0
  Long: 0.0430  Short: 0.0150  Net: +0.0280

--- Regime Distribution (127 signals) ---
  neutral               58 ##########################################################
  bull                   42 ##########################################
  volatile               18 ##################
  bear                    9 #########

--- Recent Signals (last 10) ---
  2026-03-09T22:15  AAPL      buy           conf=0.78
  2026-03-09T22:15  GOOGL     hold          conf=0.65
  ...

--- Kill-Switch Status ---
  max_daily_loss       [OK]
  stale_data           [OK]
  slippage             [OK]
  >> All clear

--- Promotion Gates ---
  min_trades           [PASS]  value=47  threshold=10
  sharpe_ratio         [PASS]  value=1.45  threshold=0.0
  max_drawdown         [PASS]  value=-6.82  threshold=-20
  win_rate             [PASS]  value=0.638  threshold=0.4
  turnover             [PASS]  value=2.14  threshold=5.0
  >> Overall: [PASS] All gates passed — ready for live promotion
```

---

## 8. `canary` — Canary Rollout Controls

Controls the canary deployment system. Think of it like a feature flag for trading — test with a small subset of symbols before going wide.

### Syntax

```bash
python main.py canary <status|enable|disable>
```

### Sub-commands

| Action | Description |
|--------|-------------|
| `status` | View current canary state (enabled, trades, P&L, breach count) |
| `enable` | Enable canary mode (resets counters) |
| `disable` | Disable canary mode (manual override) |

### Examples

```bash
# Check canary state
python main.py canary status

# Enable canary (only symbols in allow-list will trade)
python main.py canary enable

# Emergency disable
python main.py canary disable
```

### What You'll See

```
==================================================
CANARY STATUS
==================================================
Enabled: True
Total trades: 12
Total P&L: +3.45%
Breach count: 0
Allow-list symbols: AAPL, GOOGL, MSFT
```

---

## 9. `audit` — Immutable Audit Trail

Generate daily reports from the immutable append-only audit log. Every signal, order, fill, risk check, and kill switch event is recorded.

### Syntax

```bash
python main.py audit report [--date YYYY-MM-DD]
```

### Examples

```bash
# Today's audit report
python main.py audit report

# Specific date
python main.py audit report --date 2026-03-07
```

---

## 10. `backfill` — Historical Data Import

Fetches and stores full historical price data into Supabase. Useful for initial setup or rebuilding the database.

### Syntax

```bash
python main.py backfill [SYMBOL1 SYMBOL2 ...] [--period PERIOD] [--clear]
```

### Arguments & Flags

| Argument / Flag | Default | Description |
|-----------------|---------|-------------|
| `SYMBOL1 SYMBOL2 ...` | all stocks in DB | Specific stocks to backfill (leave empty for all) |
| `--period`, `-p` | `max` | How far back to fetch |
| `--clear` | `false` | Delete existing price history before importing |

### Examples

```bash
# Backfill specific stocks with full history
python main.py backfill AAPL MSFT GOOGL --period max

# Backfill all stocks in the database
python main.py backfill

# Clear and re-import (fresh start)
python main.py backfill AAPL --period max --clear
```

### What You'll See

```
Backfilling historical price data...
--------------------------------------------------
Backfilling 3 stock(s) with period=max

[AAPL] Fetching historical data...
  Stored 8,542 records (fetched 12,000 total)

[MSFT] Fetching historical data...
  Stored 9,100 records (fetched 11,500 total)

[GOOGL] Fetching historical data...
  Stored 5,200 records (fetched 5,200 total)

--------------------------------------------------
Backfill complete: 3/3 stocks, 22,842 total records
```

---

## 11. `usage` — API Consumption Tracking

View current API usage across all providers, with rate limit percentages and token counts.

### Syntax

```bash
python main.py usage [--reset]
```

### Examples

```bash
# View all API usage
python main.py usage

# Reset all counters (start fresh)
python main.py usage --reset
```

---

## 12. `test` — Connection Health Check

Tests all external connections: notifications (Slack, Telegram), database (Supabase), data fetcher (Yahoo Finance), and AI models.

### Syntax

```bash
python main.py test
```

### What You'll See

```
Testing Stock Radar connections...
--------------------------------------------------

Notification Channels:
  slack: CONNECTED
  telegram: CONNECTED

Database:
  Supabase: CONNECTED

Data Fetcher:
  Yahoo Finance: CONNECTED (AAPL = $227.48)

AI Analyzer:
  Available Models: ['groq/llama-3.1-70b', 'openai/glm-4.7', 'gemini/gemini-2.5-flash']
  Ollama Backup: NOT AVAILABLE

--------------------------------------------------
Test complete!
```

---

## Practical Workflows

### First-Time Setup

```bash
# 1. Test connections
python main.py test

# 2. Run a quick analysis to verify everything works
python main.py analyze AAPL --no-alert

# 3. Backfill historical data for your favorite stocks
python main.py backfill AAPL GOOGL MSFT TSLA --period max

# 4. Check API usage
python main.py usage
```

### Daily Trading Routine

```bash
# Morning: Start continuous monitoring
python main.py continuous AAPL GOOGL MSFT TSLA AMZN -m intraday -i 15

# During the day: Check paper trading performance
python main.py paper dashboard

# Quick question to the AI
python main.py ask "Which stock in my watchlist has the best risk-reward ratio?"

# End of day: Audit report
python main.py audit report
```

### Weekend Research

```bash
# Long-term analysis of your portfolio
python main.py analyze AAPL --mode longterm
python main.py analyze RELIANCE.NS --mode longterm
python main.py analyze MSFT --mode longterm

# Chat with the AI about findings
python main.py chat
```

---

> **Built by Darshan** — AI-powered stock analysis bringing institutional-grade intelligence to individual traders.
