# Stock-Radar: Autonomous Financial Intelligence Agent on Aptos

> Pay-per-use AI stock analysis powered by x402 micropayments, on-chain identity, and agent-to-agent coordination.

```
Client                       Stock-Radar Agent                  Aptos Testnet
  |                                |                                |
  |  GET /api/agent/discover       |                                |
  |<-- capabilities + pricing      |                                |
  |                                |                                |
  |  GET /api/agent/momentum       |                                |
  |<-- 402 {amount, recipient}     |                                |
  |                                |                                |
  |  sign & submit payment  -------+------- transfer 100 octas ---->|
  |                                |                                |
  |  retry + X-Payment-Tx: hash    |                                |
  |<-- 200 {momentum_score, ...}   |------- update_reputation ----->|
```

No accounts. No subscriptions. No API keys. Just micropayments.

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
└── src/                          Python analysis engine (Groq LLM, technical indicators)
```

---

## On-Chain Agent Identity (Aptos Testnet)

Stock-Radar is formally registered on the Aptos testnet as an autonomous agent with 10 declared capabilities and live reputation tracking.

**Contract address:** [`0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240`](https://explorer.aptoslabs.com/account/0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240/modules?network=testnet)

**Modules deployed:**

| Module | Purpose |
|--------|---------|
| `minimal_registry` | Agent registration, capability catalog, reputation counters |
| `agent_registry` | Extended identity — name, description, tags, star ratings |
| `agent_marketplace` | Task creation, assignment, escrow, dispute resolution |

**On-chain data:**
- Agent address, endpoint URL, capabilities with prices
- `Reputation` struct: total requests, successes, failures, earnings, ratings
- All updated automatically after each verified x402 payment

```bash
# Verify on-chain
aptos move view \
  --function-id '0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240::minimal_registry::get_reputation' \
  --args 'address:0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240' \
  --url https://fullnode.testnet.aptoslabs.com
```

---

## Protected Endpoints & Pricing

All prices in **octas** (1 APT = 100,000,000 octas). Discovery and messaging are free.

| Endpoint | Price | Method | Description |
|----------|-------|--------|-------------|
| `/api/agent/discover` | Free | GET | Agent capabilities, pricing, contact info |
| `/api/agent/message` | Free | POST | XMTP-style agent messaging |
| `/api/agent/reputation` | Free | GET | Live on-chain reputation data |
| `/api/agent/momentum` | 100 | GET | RSI, MACD, volume momentum signals |
| `/api/agent/rsi-divergence` | 100 | GET | Bullish/bearish divergence detection |
| `/api/agent/social-sentiment` | 100 | GET | Reddit & social media sentiment |
| `/api/agent/support-resistance` | 100 | GET | Pivot points, Bollinger Bands, ATR levels |
| `/api/fundamentals` | 100 | GET | P/E, P/B, ROE, margins, analyst targets |
| `/api/agent/news-impact` | 150 | GET | News sentiment & price impact |
| `/api/agent/stock-score` | 200 | GET | Multi-factor algorithmic scoring |
| `/api/agent/orchestrate` | 400 | POST | All analyses combined (40% momentum, 40% fundamentals, 20% sentiment) |
| `/api/live-price` | 50 | GET | Real-time price, volume, OHLC |
| `/api/analyze` | 500 | POST | Full AI analysis with LLM reasoning |

---

## x402 Payment Protocol

Every protected endpoint follows the same flow:

**1. Request** — call any endpoint without payment headers
```bash
curl http://localhost:3000/api/agent/momentum?symbol=AAPL
```

**2. 402 Response** — server returns payment requirements
```json
{
  "error": "Payment Required",
  "code": 402,
  "payment": {
    "amount": "100",
    "recipient": "0xaaefee8...",
    "network": "testnet",
    "deadline": 1769940000,
    "nonce": "abc123"
  }
}
```

**3. Pay** — sign and submit an Aptos transfer for the exact amount

**4. Retry** — include the transaction hash in the `X-Payment-Tx` header
```bash
curl -H "X-Payment-Tx: 0x2a2c529f..." \
  http://localhost:3000/api/agent/momentum?symbol=AAPL
```

**5. Data** — server verifies on-chain, returns analysis, records reputation

**Verification modes:** direct (on-chain), facilitator (gasless), hybrid (facilitator with direct fallback)

---

## Quick Start

### 1. Install & run

```bash
# Clone and install
cd stock-radar/web
npm install

# Configure environment
cp .env.example .env.local
# Required: APTOS_RECIPIENT_ADDRESS, GROQ_API_KEY, SUPABASE_URL, SUPABASE_ANON_KEY

# Start development server
npm run dev
# → http://localhost:3000
```

### 2. Try the interactive demo

Open **http://localhost:3000/x402-demo** in your browser:

1. Select a test wallet (5 pre-loaded community wallets) or connect Petra
2. Pick an endpoint (Momentum Analysis, Stock Score, etc.)
3. Enter a stock symbol (AAPL, TSLA, MSFT)
4. Click "Call Protected API"
5. Watch the live x402 flow: 402 received → payment sent → verified → data returned
6. See on-chain reputation update in the Agent Identity card at the bottom

### 3. Try the CLI demo

```bash
cd demo-client
npm install

# Set your private key in .env
echo 'APTOS_PRIVATE_KEY=0x...' > .env

npm run demo
```

The CLI walks through all 7 steps: discovery → balance check → 402 → payment → verification → data → on-chain reputation update.

### 4. Agent discovery

```bash
# Discover capabilities and pricing
curl http://localhost:3000/api/agent/discover | jq .

# Check on-chain reputation
curl http://localhost:3000/api/agent/reputation | jq .

# Agent messaging (XMTP-style)
curl -X POST http://localhost:3000/api/agent/message \
  -H "Content-Type: application/json" \
  -d '{"protocol":"agent-xmtp-v1","message_type":"capability_discovery","payload":{}}'
```

---

## Key Components

### x402 Protocol Stack (`web/src/lib/`)

| File | Role |
|------|------|
| `x402-config.ts` | Single source of truth for endpoint pricing |
| `x402-middleware.ts` | Payment verification — checks transaction on-chain |
| `x402-enforcer.ts` | Route protection, 402 responses, reputation wiring |
| `x402-client.ts` | Client SDK — auto-handles 402 → pay → retry |
| `x402-facilitator.ts` | Gasless mode — facilitator sponsors gas fees |

### Agent Infrastructure

| File | Role |
|------|------|
| `agent-registry.ts` | TypeScript client for the Move registry contract |
| `task-orchestrator.ts` | Decomposes complex tasks into micro-tasks for specialist agents |
| `xmtp-client.ts` | Agent-to-agent messaging (negotiation, task requests, vouching) |

### Move Contracts (`move-agent-registry/sources/`)

| Contract | Key Functions |
|----------|---------------|
| `minimal_registry` | `register_agent`, `update_reputation`, `add_capability`, `get_agent`, `get_reputation` |
| `agent_registry` | Extended: `submit_rating`, `get_agent_info`, tags, descriptions |
| `agent_marketplace` | `create_task`, `assign_task`, `complete_task`, `dispute_task` |

### Python Analysis Engine (`src/`)

| Module | Role |
|--------|------|
| `analyzer.py` | LLM-powered analysis (Groq llama-3.3-70b, Gemini fallback) |
| `scorer.py` | Algorithmic scoring — momentum, value, quality, risk |
| `fetcher.py` | Market data via Yahoo Finance |
| `rag_retriever.py` | RAG context retrieval for grounded analysis |

---

## Reputation Flow

After every paid API call, reputation is updated on-chain automatically:

```
x402-enforcer.ts → withX402()
  ├── verify payment on-chain
  ├── call endpoint handler → return data
  └── fire-and-forget: recordReputationUpdate()
        ├── sign update_reputation() tx with agent's private key
        └── increment: total_requests, successful_requests, total_earned
```

The reputation API (`/api/agent/reputation`) reads directly from the contract:

```json
{
  "address": "0x7f10a07e...",
  "registered": true,
  "capabilities": 10,
  "totalRequests": 5,
  "successfulRequests": 5,
  "completionRate": "100.0",
  "totalEarnedAPT": "0.00000500",
  "explorerUrl": "https://explorer.aptoslabs.com/account/0x7f10a07e...?network=testnet"
}
```

The x402-demo UI shows this as a live "On-Chain Agent Identity" card with verified badge, capabilities count, request stats, success rate, and earnings — all sourced from the blockchain.

---

## Task Orchestration

The `/api/agent/orchestrate` endpoint demonstrates multi-agent coordination:

```
Client sends: POST /api/agent/orchestrate { symbol: "AAPL" }
                              |
                    ┌─────────┼─────────┐
                    v         v         v
              Momentum    Fundamentals  Sentiment
              (weight 40%) (weight 40%) (weight 20%)
                    |         |         |
                    └────┬────┘────┬────┘
                         v         v
                   Weighted Aggregate
                         |
                         v
                 Combined Signal + Score
```

Runs sub-analyses in parallel via `Promise.allSettled`, aggregates with configurable weights, returns a single comprehensive result. Priced at 400 octas (discount vs 450+ if called individually).

---

## Agent-to-Agent Messaging

Stock-Radar exposes an XMTP-compatible HTTP endpoint for agent communication:

```bash
# Discover capabilities via messaging
POST /api/agent/message
{
  "protocol": "agent-xmtp-v1",
  "message_type": "capability_discovery",
  "payload": {}
}

# Negotiate pricing
POST /api/agent/message
{
  "protocol": "agent-xmtp-v1",
  "message_type": "pricing_inquiry",
  "payload": { "capability": "momentum" }
}

# Request a task
POST /api/agent/message
{
  "protocol": "agent-xmtp-v1",
  "message_type": "task_request",
  "payload": {
    "task_type": "stock-analysis",
    "symbol": "AAPL",
    "budget": 100
  }
}
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `APTOS_RECIPIENT_ADDRESS` | Yes | Wallet that receives API payments |
| `APTOS_NETWORK` | No | Fullnode URL (default: testnet) |
| `AGENT_REGISTRY_ADDRESS` | No | Move contract address |
| `AGENT_REGISTRY_PRIVATE_KEY` | No | Signs reputation updates on-chain |
| `INTERNAL_API_KEY` | No | Bypasses payment for app's own pages |
| `GROQ_API_KEY` | Yes | Groq LLM for AI analysis |
| `NEXT_PUBLIC_SUPABASE_URL` | Yes | Supabase project URL |
| `NEXT_PUBLIC_SUPABASE_ANON_KEY` | Yes | Supabase anon key |

---

## Deploy Move Contracts

```bash
cd move-agent-registry

# Compile
aptos move compile --named-addresses agent_registry=default

# Deploy to testnet
aptos move publish --named-addresses agent_registry=default --assume-yes

# Register agent on-chain
node scripts/register-agent.js
```

---

## Test Wallets

Five community wallets are pre-loaded in the x402-demo UI for testing:

| Wallet | Address |
|--------|---------|
| Wallet 1 | `0xaaefee8ba1e5f24ef88a74a3f445e0d2b810b90c...` |
| Wallet 2 | `0xaaea48900c8f8045876505fe5fc5a623b1e423ef...` |
| Wallet 3 | `0x924c2e983753bb29b45ae9b4036d48861f204da0...` |
| Wallet 4 | `0xf1697d22257fd39653319eb3a2ee23fca2ca99b2...` |
| Wallet 5 | `0x6cd199bbbc8bb3c17de4d2aebc2e75b4e9d7e308...` |

Or connect a Petra browser wallet via the toggle in the demo UI.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Next.js 15, React 19, Tailwind CSS 4, Radix UI |
| API | Next.js API Routes (TypeScript) |
| Blockchain | Aptos testnet, Move language |
| Wallet | Petra (AIP-62), @aptos-labs/ts-sdk |
| AI/LLM | Groq (llama-3.3-70b), Gemini fallback |
| Database | Supabase (PostgreSQL) |
| Charts | Recharts, Lightweight Charts, uPlot |
| Messaging | XMTP protocol over HTTP |

---

## What Makes This Different

**x402 as a native protocol** — Not a wrapper around Stripe. The HTTP 402 status code was reserved for "future use" since 1997. This implements it for real: any HTTP client that can read a 402 response and submit an Aptos transaction can use the API. No SDK required, no registration, no API keys.

**On-chain agent identity** — Stock-Radar isn't just an API. It's a registered entity on the Aptos blockchain with a verifiable track record. Other agents can check its reputation before paying for services. Every successful interaction strengthens the on-chain record.

**Agent-to-agent economy** — The combination of x402 (payment), XMTP (messaging), on-chain registry (identity), and task orchestrator (coordination) creates infrastructure for agents to discover, negotiate with, pay, and rate each other — without human intermediaries.

**Micro-pricing** — 50 octas for a live price quote. 100 octas for momentum analysis. The pricing is granular enough for agents to make thousands of small decisions without budgeting for expensive subscriptions.
