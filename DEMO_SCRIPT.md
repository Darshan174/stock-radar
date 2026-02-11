# 🎙️ Stock-Radar x402: Demo Presentation Script

**Project:** Stock-Radar (Uber for AI Agents & Financial Intelligence)  
**Theme:** "DeFi for AI Logic" / "Pay-per-use Intelligence"  
**Time Estimate:** 3-5 Minutes

---

## 1. The Hook (30s)
> "We are building the **Uber for AI Agents**. Right now, high-quality financial AI is locked behind expensive monthly subscriptions ($50-$200/mo). If you just want *one* answer—like 'Should I buy AAPL right not?'—you shouldn't have to subscribe. 
> 
> **Stock-Radar** solves this using the **x402 Protocol on Aptos**. It allows users or *other agents* to pay for exactly what they use—down to the micro-cent—instantly and gaslessly."

---

## 2. The Solution Overview (30s)
> "We've built a complete ecosystem:
> 1. **Financial Intelligence Agents**: AI models that analyze stocks (Momentum, Sentiment, Risk).
> 2. **x402 Protocol**: A standard for 'Payment Required' APIs using Aptos.
> 3. **Gasless Micropayments**: Users pay *only* the API fee (e.g., 100 octas). The facilitator handles the gas.
> 4. **On-Chain Reputation**: Every task completed is recorded on-chain, creating a verifiable trust score."

---

## 3. The Live Demo (2-3m)

### Step 1: The Visual Hook (Dashboard)
*   **Action:** Show the **Main Dashboard** (`http://localhost:3000`).
*   **Script:** "Here is the Stock Radar dashboard. It looks like a standard financial app, but under the hood, every chart and signal is powered by an independent, pay-per-use AI agent."

### Step 2: The Core Interaction (x402 Flow)
*   **Action:** Go to the **Demo Page** (`http://localhost:3000/x402-demo`).
*   **Action:** 
    1. Paste a Private Key (from `demo-client/.env`).
    2. Toggle **"Gasless Mode"** to ON.
    3. Select **"Momentum Analysis"** for `AAPL`.
    4. Click **"Run Analysis"**.
*   **Script:** "Let's ask the Momentum Agent for an analysis on Apple. Watch what happens:
    1. The API responds with `402 Payment Required`.
    2. Our SDK intercepts this, signs a micro-transaction (100 octas), and sends it to a Relayer.
    3. The Relayer pays the gas, executes the payment on Aptos, and gets a transaction hash.
    4. The API verifies the payment on-chain and returns the intelligence."
*   **Action:** Show the result (Momentum Score, Signal).
*   **Script:** "Done. We just bought high-quality AI inference for a fraction of a cent, with zero friction."

### Step 3: On-Chain Verification (The innovative part)
*   **Action:** Click the **Transaction Hash** link to open Aptos Explorer.
*   **Script:** "If we look at the blockchain, we see two things happened:
    1. The payment was settled.
    2. The Agent's **On-Chain Reputation** was updated. This creates a permanent, undeniable track record of this agent's performance."

### Step 4: Developer Experience (CLI)
*   **Action:** Switch to Terminal. Run: `cd demo-client && npm run demo`.
*   **Script:** "This isn't just for web apps. Any developer or *other AI agent* can consume these services. Here's a CLI agent discovering our service, negotiating a price, and paying for it strictly via code."

---

## 4. Closing (30s)
> "Stock-Radar proves that **AI Agents + Blockchain** is the perfect match. 
> - Blockchain handles the **Trust** (Reputation) and **Value** (Payments).
> - AI handles the **Logic**.
> 
> We are building the infrastructure for the autonomous economy, where agents hire agents to work for you. Thank you."

---

## ⚡️ Quick Prep Checklist
1. **Reset Demo:** Clear console, have `demo-client/.env` key ready to copy.
2. **Start Tunnel:** Ensure your `cloudflared` tunnel is running for remote access.
3. **Open Tabs:**
    - Dashboard: `http://localhost:3000`
    - Demo Page: `http://localhost:3000/x402-demo`
    - Aptos Explorer: `https://explorer.aptoslabs.com/?network=testnet`
