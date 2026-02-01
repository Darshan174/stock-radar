# Stock-Radar x402: Uber for AI Agents

## 🏆 Hackathon Submission Summary

**Stock-Radar** is a production-ready **"Uber for AI Agents"** built on Aptos x402, enabling:
- **Pay-per-use financial intelligence APIs**
- **Gasless micropayments** via facilitator
- **Agent-to-agent coordination** and task decomposition
- **On-chain reputation and identity**
- **Trustless marketplace** for AI services

---

## 📦 What's Been Built

### 1. Core Payment Infrastructure ✅

**x402 Protocol Implementation:**
- ✅ HTTP 402 Payment Required enforcement
- ✅ 9 protected API endpoints
- ✅ Facilitator support for gasless transactions
- ✅ Three verification modes (direct/facilitator/hybrid)
- ✅ Replay protection & security

**Pricing:**
| Endpoint | Price (octas) | Description |
|----------|---------------|-------------|
| `/api/agent/momentum` | 100 | Momentum signals (RSI, MACD) |
| `/api/agent/stock-score` | 200 | Comprehensive scoring |
| `/api/agent/news-impact` | 150 | News sentiment |
| `/api/analyze` | 500 | Full AI analysis |
| ... | ... | 9 endpoints total |

### 2. Gasless Transactions ✅

**How it works:**
```
Without Facilitator:  Pay 100 octas + 500 octas gas = 600 octas
With Facilitator:      Pay 100 octas only = 100 octas
                       (Facilitator pays 500 octas gas!)
```

**Benefits:**
- Users pay only API fee
- No need for extra APT for gas
- Lower barrier to entry
- Faster transactions

### 3. On-Chain Agent Registry (Move Contract) ✅

**File:** `move-agent-registry/sources/agent_marketplace.move`

**Features:**
- ✅ Agent registration with on-chain identity
- ✅ Capability declaration (what agent can do)
- ✅ Reputation scoring (completion rate, ratings, disputes)
- ✅ Payment history tracking
- ✅ Task marketplace (create/assign/complete)
- ✅ Dispute resolution
- ✅ Behavior/preferences storage

**Trust Metrics:**
```move
struct AgentReputation {
    total_tasks_completed: u64,
    total_tasks_failed: u64,
    total_disputes: u64,
    average_rating: u64,
    successful_payments: u64,
    on_time_deliveries: u64,
}
```

**How to know an agent won't scam you:**
- View completion rate on-chain
- Check average rating (1-5 stars)
- See dispute history
- Verify payment success rate
- Check on-time delivery rate

### 4. Task Orchestrator ("Uber for AI Agents") ✅

**File:** `web/src/lib/task-orchestrator.ts`

**Break big tasks into micro-tasks:**
```typescript
// Complex task
const result = await orchestrator.executeComplexTask({
  type: 'comprehensive-analysis',
  target: 'AAPL',
  budget: 1000, // Total budget
  decomposition: [
    { type: 'momentum', weight: 0.4 },      // 400 octas
    { type: 'fundamentals', weight: 0.4 },  // 400 octas
    { type: 'sentiment', weight: 0.2 },     // 200 octas
  ]
})

// Automatically:
// 1. Finds best agent for each sub-task
// 2. Assigns based on reputation + price
// 3. Executes in parallel
// 4. Retries failed tasks with other agents
// 5. Aggregates results
```

**Agent Selection Criteria:**
- Capability match
- Reputation score (> 70/100)
- Price competitiveness
- Availability (online/busy/offline)

### 5. XMTP Agent Messaging ✅

**File:** `web/src/lib/xmtp-client.ts`

**Agent-to-agent communication:**
```typescript
// Send task request
await xmtp.sendTaskRequest(agentAddress, {
  taskType: 'stock-analysis',
  description: 'Analyze AAPL momentum',
  budget: 100,
  deadline: Date.now() + 3600000
})

// Negotiate terms
await xmtp.sendMessage(agentAddress, "I can do it for 80 octas")

// Reputation vouching
await xmtp.sendReputationVouch(agentAddress, {
  rating: 5,
  completedTasks: 50,
  comment: "Reliable agent, fast delivery"
})
```

**Use cases:**
- Task negotiation before accepting
- Multi-agent coordination
- Dispute resolution
- Reputation sharing between agents

### 6. Petra Wallet Integration ✅

**File:** `web/src/components/petra-wallet-provider.tsx`

**User-friendly payments:**
```typescript
const { payWithPetra } = usePetraPayment()

// One click - Petra popup opens
const txHash = await payWithPetra({
  recipient: "0x...",
  amount: 100,
  gasless: true  // Facilitator pays gas!
})
```

**Features:**
- One-click wallet connection
- Secure transaction signing
- Gasless option
- Transaction history

### 7. Demo Client ✅

**Location:** `demo-client/`

**Command-line demo:**
```bash
cd demo-client
npm install
npm run demo
```

**Shows:**
1. Agent discovery
2. 402 payment required
3. Gasless transaction via facilitator
4. API response

### 8. Interactive Web Demo ✅

**URL:** `http://localhost:3000/x402-demo`

**Features:**
- Connect wallet (paste private key)
- Toggle gasless mode
- Select API endpoint
- Real-time transaction logs
- View results

### 9. Usage Dashboard ✅

**URL:** `http://localhost:3000/usage-dashboard`

**Shows:**
- Total requests & revenue
- Gasless vs direct breakdown
- Endpoint usage
- Recent transactions
- Gas savings calculation

---

## 🔍 Key Innovations

### 1. Uber for AI Agents
Break complex tasks into micro-tasks handled by specialist agents:
- Momentum specialist
- Fundamentals specialist  
- Sentiment specialist
- Aggregator agent

### 2. Trust Through Transparency
On-chain reputation means:
- Can't fake completion history
- Ratings are verifiable
- Disputes are public
- Payment history is permanent

### 3. Gasless Micro-Payments
Users pay only the API fee (e.g., 100 octas = $0.0006):
- Facilitator sponsors gas
- Lower barrier than subscriptions
- Pay-per-use model
- No credit cards needed

### 4. Agent Marketplace
Like an API marketplace but for AI agents:
- Discovery via `/api/agent/discover`
- Reputation on-chain
- Pricing transparent
- XMTP for negotiation

---

## 🚀 Quick Start

### 1. Start the Server
```bash
cd web
npm run dev
```

### 2. Try the Demo
```bash
# Web UI
open http://localhost:3000/x402-demo

# Or CLI
cd demo-client
npm install
npm run demo
```

### 3. Test Wallets (from cheatsheet)
**Wallet 1:**
- Address: `0xaaefee8ba1e5f24ef88a74a3f445e0d2b810b90c1996466dae5ea9a0b85d42a0`
- Private key: available in the hackathon wallet cheatsheet

---

## 📊 Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         CLIENTS                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Web Demo    │  │  CLI Client  │  │  Petra Wallet│      │
│  │  (/x402-demo)│  │  (demo.js)   │  │  (UI)        │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
          ▼                 ▼                 ▼
┌─────────────────────────────────────────────────────────────┐
│                    X402 PROTOCOL                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  Payment Flow:                                      │    │
│  │  1. Client calls API → 402 Payment Required        │    │
│  │  2. SDK signs transaction                          │    │
│  │  3. Facilitator submits (gasless!)                 │    │
│  │  4. Retry with X-Payment-Tx header                 │    │
│  │  5. Server verifies → Returns data                 │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│  Facilitator │  │  Stock-Radar │  │  Aptos       │
│  (Gasless)   │  │  APIs (9)    │  │  Blockchain  │
└──────────────┘  └──────────────┘  └──────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                ON-CHAIN AGENT REGISTRY                       │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Agent       │  │  Reputation  │  │  Task        │      │
│  │  Identity    │  │  Scores      │  │  Marketplace │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

---

## 🎯 Demo Script for Judges

### Demo 1: Gasless Payment Flow
```bash
# 1. Open demo page
curl http://localhost:3000/api/agent/discover

# 2. Try without payment
curl http://localhost:3000/api/agent/momentum?symbol=AAPL
# → Returns 402 with payment requirements

# 3. Use demo client (handles everything automatically)
cd demo-client && npm run demo
```

### Demo 2: Agent Discovery
```bash
curl http://localhost:3000/api/agent/discover | jq .
```

### Demo 3: Interactive Web Demo
1. Open `http://localhost:3000/x402-demo`
2. Paste Wallet 1 private key
3. Toggle "Gasless" ON
4. Select "Momentum Analysis"
5. Enter symbol: `AAPL`
6. Watch automatic payment flow!

### Demo 4: Usage Dashboard
Open `http://localhost:3000/usage-dashboard`

### Demo 5: Move Contract (On-Chain)
```bash
cd move-agent-registry
aptos move compile
aptos move publish --profile testnet
```

---

## 📦 Files Added

### Smart Contracts
- `move-agent-registry/sources/agent_registry.move` - Basic registry
- `move-agent-registry/sources/agent_marketplace.move` - Full marketplace

### Web Application
- `web/src/lib/x402-facilitator.ts` - Facilitator client
- `web/src/lib/x402-client.ts` - Client SDK
- `web/src/lib/xmtp-client.ts` - XMTP messaging
- `web/src/lib/task-orchestrator.ts` - Task decomposition
- `web/src/lib/agent-registry.ts` - On-chain registry client
- `web/src/components/petra-wallet-provider.tsx` - Petra wallet
- `web/src/app/x402-demo/page.tsx` - Interactive demo
- `web/src/app/usage-dashboard/page.tsx` - Analytics dashboard

### Demo Client
- `demo-client/demo.js` - CLI demo
- `demo-client/README.md` - Documentation

---

## 🏅 Why This Wins

1. **Complete x402 Implementation** - Full protocol with facilitator
2. **Gasless Innovation** - Users pay only API fee
3. **On-Chain Trust** - Reputation & identity verifiable
4. **Uber for AI** - Task decomposition & coordination
5. **Production Ready** - 9 endpoints, real payments, working demos
6. **Agent Ecosystem** - Discovery, messaging, marketplace

---

## 🔗 Resources

- **Demo URL:** http://localhost:3000/x402-demo
- **Dashboard:** http://localhost:3000/usage-dashboard
- **Discovery:** http://localhost:3000/api/agent/discover
- **Facilitator:** https://x402-navy.vercel.app/facilitator/
- **Aptos Explorer:** https://explorer.aptoslabs.com/?network=testnet

---

**Ready to demo! 🚀**
