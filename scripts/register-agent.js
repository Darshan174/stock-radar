#!/usr/bin/env node
/**
 * Stock-Radar Agent Registration Script
 * 
 * Registers the agent on-chain with full capabilities.
 * This transforms the narrative from "API" to "agent economy".
 * 
 * Usage: node scripts/register-agent.js
 */

import { Aptos, AptosConfig, Network, Account, Ed25519PrivateKey } from "@aptos-labs/ts-sdk"
import { config } from "dotenv"
import { fileURLToPath } from "url"
import { dirname, join } from "path"

// Load env from project root
const __dirname = dirname(fileURLToPath(import.meta.url))
config({ path: join(__dirname, "..", ".env") })

// Configuration
const REGISTRY_ADDRESS = process.env.AGENT_REGISTRY_ADDRESS || "0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240"
const PRIVATE_KEY = process.env.AGENT_REGISTRY_PRIVATE_KEY

if (!PRIVATE_KEY) {
  console.error("❌ AGENT_REGISTRY_PRIVATE_KEY is required in .env")
  process.exit(1)
}

// Agent capabilities (from x402-config.ts)
const AGENT_CONFIG = {
  endpointUrl: "https://stock-radar.vercel.app",
  capabilities: [
    { name: "momentum", price: 100, description: "Momentum analysis with RSI, MACD, volume signals" },
    { name: "rsi-divergence", price: 100, description: "RSI divergence detection for reversal signals" },
    { name: "news-impact", price: 150, description: "AI-powered news sentiment and impact analysis" },
    { name: "stock-score", price: 200, description: "Comprehensive stock scoring with multiple factors" },
    { name: "social-sentiment", price: 100, description: "Social media sentiment aggregation" },
    { name: "support-resistance", price: 100, description: "Key price level identification" },
    { name: "orchestrate", price: 400, description: "Multi-agent orchestration combining all analyses" },
    { name: "analyze", price: 500, description: "Full technical + fundamental analysis" },
    { name: "fundamentals", price: 100, description: "Company fundamentals and ratios" },
    { name: "live-price", price: 50, description: "Real-time price data with indicators" },
  ]
}

console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           🤖 STOCK-RADAR AGENT REGISTRATION                      ║
║                                                                  ║
║   Registering as an autonomous agent in the agent economy       ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
`)

// Initialize
const aptosConfig = new AptosConfig({ network: Network.TESTNET })
const aptos = new Aptos(aptosConfig)

const privateKeyHex = PRIVATE_KEY.startsWith("0x") ? PRIVATE_KEY.slice(2) : PRIVATE_KEY
const privateKey = new Ed25519PrivateKey(privateKeyHex)
const account = Account.fromPrivateKey({ privateKey })

console.log(`📝 Agent Address: ${account.accountAddress.toString()}`)
console.log(`📝 Registry:      ${REGISTRY_ADDRESS}`)
console.log(`📝 Capabilities:  ${AGENT_CONFIG.capabilities.length}`)
console.log("")

// Check if already registered
async function checkIfRegistered() {
  try {
    const [exists] = await aptos.view({
      payload: {
        function: `${REGISTRY_ADDRESS}::minimal_registry::agent_exists`,
        functionArguments: [account.accountAddress.toString()],
      },
    })
    return exists
  } catch {
    return false
  }
}

// Register the agent
async function registerAgent() {
  const isRegistered = await checkIfRegistered()
  
  if (isRegistered) {
    console.log("✅ Agent already registered on-chain!")
    console.log("")
    
    // Fetch and display current data
    const [agentData] = await aptos.view({
      payload: {
        function: `${REGISTRY_ADDRESS}::minimal_registry::get_agent`,
        functionArguments: [account.accountAddress.toString()],
      },
    })
    
    console.log("📊 Current Registration:")
    console.log(`   Endpoint: ${agentData.endpoint_url}`)
    console.log(`   Capabilities: ${agentData.capabilities.length}`)
    console.log(`   Total Requests: ${agentData.reputation.total_requests}`)
    console.log(`   Total Earned: ${agentData.reputation.total_earned} octas`)
    console.log("")
    console.log(`🔗 View on Explorer:`)
    console.log(`   https://explorer.aptoslabs.com/account/${account.accountAddress.toString()}?network=testnet`)
    return
  }
  
  console.log("🚀 Registering agent on-chain...")
  console.log("")
  
  // Prepare capability arrays
  const names = AGENT_CONFIG.capabilities.map(c => c.name)
  const prices = AGENT_CONFIG.capabilities.map(c => c.price)
  const descriptions = AGENT_CONFIG.capabilities.map(c => c.description)
  
  try {
    const transaction = await aptos.transaction.build.simple({
      sender: account.accountAddress,
      data: {
        function: `${REGISTRY_ADDRESS}::minimal_registry::register_agent`,
        functionArguments: [
          AGENT_CONFIG.endpointUrl,
          names,
          prices,
          descriptions,
        ],
      },
    })
    
    const pendingTxn = await aptos.signAndSubmitTransaction({
      signer: account,
      transaction,
    })
    
    console.log(`📤 Transaction submitted: ${pendingTxn.hash.slice(0, 16)}...`)
    console.log("⏳ Waiting for confirmation...")
    
    await aptos.waitForTransaction({ transactionHash: pendingTxn.hash })
    
    console.log("")
    console.log("═".repeat(66))
    console.log("✅ AGENT REGISTERED SUCCESSFULLY!")
    console.log("═".repeat(66))
    console.log("")
    console.log("📊 Registration Details:")
    console.log(`   Agent Address:  ${account.accountAddress.toString()}`)
    console.log(`   Endpoint URL:   ${AGENT_CONFIG.endpointUrl}`)
    console.log(`   Capabilities:   ${AGENT_CONFIG.capabilities.length}`)
    console.log(`   Transaction:    ${pendingTxn.hash}`)
    console.log("")
    console.log("💰 Capability Pricing:")
    AGENT_CONFIG.capabilities.forEach(c => {
      console.log(`   • ${c.name.padEnd(18)} ${c.price} octas`)
    })
    console.log("")
    console.log("🔗 View on Aptos Explorer:")
    console.log(`   https://explorer.aptoslabs.com/txn/${pendingTxn.hash}?network=testnet`)
    console.log("")
    console.log("🎉 Stock-Radar is now a verified agent in the agent economy!")
    
  } catch (error) {
    console.error("❌ Registration failed:", error.message)
    
    if (error.message.includes("E_AGENT_EXISTS")) {
      console.log("")
      console.log("ℹ️  Agent is already registered. Use update functions to modify.")
    }
    
    process.exit(1)
  }
}

registerAgent().catch(console.error)
