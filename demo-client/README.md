# Stock-Radar x402 Demo Client

A command-line demo client for interacting with the Stock-Radar x402-protected APIs.

## Features

- 🔥 **Automatic Payment Handling** - SDK handles 402 responses automatically
- ⚡ **Gasless Transactions** - Facilitator pays gas fees for you
- 📊 **Real-time Analysis** - Get stock momentum, scores, and more
- 🔑 **Testnet Ready** - Pre-configured with test wallets

## Quick Start

### 1. Install Dependencies

```bash
cd demo-client
npm install
```

### 2. Run the Demo

```bash
npm run demo
```

Or with custom options:

```bash
# Use different stock symbol
SYMBOL=TSLA npm run demo

# Disable gasless (you pay gas)
GASLESS=false npm run demo

# Use different API endpoint
API_URL=https://your-api.com npm run demo
```

### 3. Watch the Magic! ✨

The demo will:
1. Discover agent capabilities
2. Check your wallet balance
3. Call a protected API
4. Automatically handle the 402 payment required
5. Execute a gasless transaction (via facilitator)
6. Display the stock analysis results

## How It Works

```
Step 1: Call API without payment
        → GET /api/agent/momentum?symbol=AAPL
        ← 402 Payment Required

Step 2: SDK automatically:
        → Builds Aptos transaction
        → Signs with your private key
        → Sends to facilitator (gasless!)
        ← Gets transaction hash

Step 3: Retry with payment
        → GET /api/agent/momentum
        → Header: X-Payment-Tx: 0x...
        ← { momentum_score: 75, signal: "bullish" }
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `API_URL` | `http://localhost:3000` | Stock-Radar API URL |
| `APTOS_PRIVATE_KEY` | Wallet 1 | Your Aptos private key |
| `SYMBOL` | `AAPL` | Stock symbol to analyze |
| `GASLESS` | `true` | Use facilitator for gasless |
| `FACILITATOR_URL` | Public | Facilitator endpoint |

## Test Wallets

Use any funded Aptos testnet wallet. Set `APTOS_PRIVATE_KEY` in `.env`.

Fund a wallet at https://aptos.dev/network/faucet

## Example Output

```
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           📊 STOCK-RADAR x402 DEMO CLIENT                        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

   API Base URL         http://localhost:3000
   Wallet Address       0xaaef...42a0
   Gasless Mode         Enabled (facilitator pays gas)
   Symbol               AAPL

📝 Step 1: Discovering Agent Capabilities
────────────────────────────────────────────────────────────
✅ Agent discovered!
   Agent Name           Stock-Radar Financial Intelligence Agent
   Description          AI-powered stock analysis...
   Capabilities         9
   Pricing Model        pay-per-use

   Available Endpoints:
   • momentum           100 octas
   • stock-score        200 octas
   • news-impact        150 octas
   ...

📝 Step 2: Checking Wallet Balance
────────────────────────────────────────────────────────────
✅ Balance: 0.00100000 APT (100000 octas)

📝 Step 3: Calling Protected API: Momentum Analysis
────────────────────────────────────────────────────────────
   Endpoint: /api/agent/momentum
   URL: http://localhost:3000/api/agent/momentum?symbol=AAPL

   → Sending initial request (no payment)...
✅ Received 402 Payment Required
   Amount Required      100 octas
   Recipient            0xaaef...42a0

📝 Step 4: Executing Payment
────────────────────────────────────────────────────────────
   → Building transaction...
   → Signing transaction...
   → Using facilitator (gasless)...
✅ Gasless transaction submitted!
   Transaction Hash     0xabc...123
   Gas Paid By          Facilitator (you paid 0 gas!)

📝 Step 5: Retrying API Call with Payment
────────────────────────────────────────────────────────────
   → Sending request with X-Payment-Tx header...

📝 Step 6: API Response Received!
────────────────────────────────────────────────────────────
✅ Momentum Analysis Complete

   📈 Results:
   Symbol:          AAPL
   Momentum Score:  75/100
   Signal:          BULLISH
   Timestamp:       2026-01-31T10:30:00Z

   Score Breakdown:
   • rsi: 15 points
   • macd: 10 points
   • price_vs_sma: 12 points

✅ Demo completed successfully!

💡 Key Takeaways:
   • x402 enables pay-per-use API access
   • Gasless transactions = users pay only API fee
   • Automatic payment handling via SDK
   • No accounts, subscriptions, or UI needed

🔗 View transaction on Aptos Explorer:
   https://explorer.aptoslabs.com/txn/0xabc...123?network=testnet
```

## Troubleshooting

### "Transaction already used"
Each transaction hash can only be used once. The demo generates fresh transactions each run.

### "Insufficient balance"
Get more testnet APT:
```bash
# Visit: https://aptos.dev/network/faucet
# Or use Aptos CLI
aptos account fund-with-faucet --account <address>
```

### "Facilitator not available"
Check if the public facilitator is up:
```bash
curl https://x402-navy.vercel.app/facilitator/
```

Or disable gasless mode:
```bash
GASLESS=false npm run demo
```

## License

MIT
