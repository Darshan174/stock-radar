# Stock-Radar: Autonomous Financial Intelligence Agent on Aptos

> Pay-per-use AI stock analysis powered by x402 micropayments, on-chain identity, and agent-to-agent coordination.

No accounts. No subscriptions. Just micropayments.

PUBLIC URL OF THE PROJECT: https://reproductive-but-expression-anyway.trycloudflare.com/x402-demo

UI DEMO:

https://github.com/user-attachments/assets/0c9450fc-90d2-4067-9c45-1adf5709acf5

CLI DASHBOARD DEMO:

https://github.com/user-attachments/assets/dec2ede4-12a5-4142-9537-957fb0d22c67


---

## What Is This

Stock-Radar is a **verified autonomous agent** registered on the Aptos blockchain. It sells financial intelligence (momentum signals, stock scores, sentiment analysis, fundamentals) through HTTP APIs protected by the **x402 payment protocol** — the same way a web server returns `401 Unauthorized`, Stock-Radar returns `402 Payment Required` with exact payment instructions.

Clients pay per request in APT (as low as 50 octas per call). After payment verification on-chain, the agent serves data and records the interaction to its **on-chain reputation** — a tamper-proof track record of requests served, success rate, and revenue earned.

---

## Architecture

```
stock-radar/
├── web/                          Next.js 15 — dashboard, API routes, x402 enforcement
│   ├── src/app/api/agent/        10 protected endpoints + discovery + messaging
│   ├── src/lib/                  x402 protocol, agent registry, orchestrator, XMTP
│   └── src/app/x402-demo/        Interactive payment demo with on-chain identity
├── move-agent-registry/          Move smart contracts (Aptos testnet)
│   └── sources/
│       ├── minimal_registry.move Agent registration, capabilities, reputation
│       ├── agent_registry.move   Extended registry with ratings & tags
│       └── agent_marketplace.move Task lifecycle, escrow, disputes
├── demo-client/                  CLI demo — full x402 flow in terminal
├── scripts/                      Agent registration & automation
├── xmtp/                         Agent-to-agent messaging examples
└── src/                          Python analysis engine (LLM, technical indicators)
```

---

## Protected Endpoints & Pricing

All prices in **octas** (1 APT = 100,000,000 octas). Discovery and messaging are free.

| Endpoint | Price | Description |
|----------|-------|-------------|
| `/api/agent/discover` | Free | Agent capabilities, pricing, contact info |
| `/api/agent/message` | Free | XMTP-style agent messaging |
| `/api/agent/reputation` | Free | Live on-chain reputation data |
| `/api/agent/momentum` | 100 | RSI, MACD, volume momentum signals |
| `/api/agent/rsi-divergence` | 100 | Bullish/bearish divergence detection |
| `/api/agent/social-sentiment` | 100 | Social media sentiment |
| `/api/agent/support-resistance` | 100 | Pivot points, Bollinger Bands, ATR levels |
| `/api/fundamentals` | 100 | P/E, P/B, ROE, margins, analyst targets |
| `/api/agent/news-impact` | 150 | News sentiment & price impact |
| `/api/agent/stock-score` | 200 | Multi-factor algorithmic scoring |
| `/api/agent/orchestrate` | 400 | All analyses combined |
| `/api/live-price` | 50 | Real-time price, volume, OHLC |
| `/api/analyze` | 500 | Full AI analysis with LLM reasoning |

---

## x402 Payment Flow

1. **Request** — call any endpoint without payment
2. **402 Response** — server returns amount, recipient, and deadline
3. **Pay** — sign and submit an Aptos transfer for the exact amount
4. **Retry** — include the transaction hash in the `X-Payment-Tx` header
5. **Data** — server verifies on-chain, returns analysis, updates reputation

Supports three verification modes: **direct** (on-chain), **facilitator** (gasless), and **hybrid** (facilitator with direct fallback).

---

## Quick Start

### 1. Install & run

```bash
cd stock-radar/web
npm install
cp .env.example .env.local
# Fill in your environment variables (see .env.example for details)
npm run dev
```

### 2. Interactive demo

Open **http://localhost:3000/x402-demo** — select a wallet, pick an endpoint, enter a stock symbol, and watch the live x402 payment flow.

### 3. CLI demo

```bash
cd demo-client
npm install
cp .env.example .env
# Add your Aptos private key to .env
npm run demo
```

### 4. Agent discovery

```bash
curl http://localhost:3000/api/agent/discover | jq .
curl http://localhost:3000/api/agent/reputation | jq .
```

---

## On-Chain Agent Identity

Stock-Radar is registered on Aptos testnet with 10 declared capabilities and live reputation tracking.

**Move modules deployed:**

| Module | Purpose |
|--------|---------|
| `minimal_registry` | Agent registration, capability catalog, reputation counters |
| `agent_registry` | Extended identity — name, description, tags, star ratings |
| `agent_marketplace` | Task creation, assignment, escrow, dispute resolution |

After every paid API call, reputation is updated on-chain automatically — total requests, successes, and earnings are all recorded to the contract.

---

## Task Orchestration

The `/api/agent/orchestrate` endpoint runs sub-analyses in parallel:

```
POST /api/agent/orchestrate { symbol: "AAPL" }
              |
    ┌─────────┼─────────┐
    v         v         v
Momentum  Fundamentals  Sentiment
(40%)       (40%)       (20%)
    └────┬────┘────┬────┘
         v         v
   Weighted Aggregate → Combined Signal + Score
```

Priced at 400 octas (discount vs 450+ if called individually).

---

## Agent-to-Agent Messaging

Stock-Radar exposes an XMTP-compatible HTTP endpoint for agent communication — capability discovery, pricing negotiation, and task requests.

```bash
# Discover capabilities
curl -X POST http://localhost:3000/api/agent/message \
  -H "Content-Type: application/json" \
  -d '{"protocol":"agent-xmtp-v1","message_type":"capability_discovery","payload":{}}'
```

---

## Deploy Move Contracts

```bash
cd move-agent-registry
aptos move compile --named-addresses agent_registry=default
aptos move publish --named-addresses agent_registry=default --assume-yes
node scripts/register-agent.js
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, Tailwind CSS 4, Radix UI |
| Blockchain | Aptos testnet, Move language |
| Wallet | Petra (AIP-62), @aptos-labs/ts-sdk |
| AI/LLM | Groq (llama-3.3-70b), Gemini fallback |
| Database | Supabase (PostgreSQL) |
| Messaging | XMTP protocol over HTTP |

---

## What Makes This Different

- **x402 as a native protocol** — HTTP 402 was reserved for "future use" since 1997. This implements it for real: any HTTP client that can read a 402 response and submit an Aptos transaction can use the API. No SDK required, no registration.

- **On-chain agent identity** — Stock-Radar is a registered entity on Aptos with a verifiable track record. Other agents can check its reputation before paying.

- **Agent-to-agent economy** — x402 (payment) + XMTP (messaging) + on-chain registry (identity) + task orchestrator (coordination) = infrastructure for agents to discover, negotiate, pay, and rate each other without human intermediaries.

- **Micro-pricing** — 50 octas for a live quote. 100 for momentum analysis. Granular enough for agents to make thousands of decisions without expensive subscriptions.
