#!/usr/bin/env node

/**
 * Stock-Radar x402 Demo Client
 * 
 * This script demonstrates how to call x402-protected APIs with automatic
 * payment handling. It supports both gasless (via facilitator) and direct
 * payment modes.
 * 
 * Usage:
 *   npm install
 *   npm run demo
 * 
 * Or with custom parameters:
 *   API_URL=http://localhost:3000 SYMBOL=TSLA node demo.js
 */

import { Aptos, AptosConfig, Network, Account, Ed25519PrivateKey } from "@aptos-labs/ts-sdk"
import { config } from "dotenv"
import { readFileSync } from "fs"
import { fileURLToPath } from "url"
import { dirname, join } from "path"

// Load environment variables
const __dirname = dirname(fileURLToPath(import.meta.url))
config({ path: join(__dirname, ".env") })

// Configuration
const API_URL = process.env.API_URL || "http://localhost:3000"
const PRIVATE_KEY = process.env.APTOS_PRIVATE_KEY
if (!PRIVATE_KEY) {
  console.error("❌ APTOS_PRIVATE_KEY is required. Set it in demo-client/.env")
  process.exit(1)
}
const SYMBOL = process.env.SYMBOL || "AAPL"
const GASLESS = process.env.GASLESS === "true" // Default to false (facilitator doesn't support Aptos)
const FACILITATOR_URL = process.env.FACILITATOR_URL || "https://x402-navy.vercel.app/facilitator/"

// ASCII Art Banner
console.log(`
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║           📊 STOCK-RADAR x402 DEMO CLIENT                        ║
║                                                                  ║
║   Gasless Micropayments for Financial Intelligence APIs         ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
`)

// Utility functions
const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms))

const formatOctas = (octas) => {
  return `${(octas / 100000000).toFixed(8)} APT (${octas} octas)`
}

const printStep = (step, message) => {
  console.log(`\n📝 Step ${step}: ${message}`)
  console.log("─".repeat(60))
}

const printSuccess = (message) => {
  console.log(`✅ ${message}`)
}

const printInfo = (label, value) => {
  console.log(`   ${label.padEnd(20)} ${value}`)
}

const printError = (message) => {
  console.log(`❌ ${message}`)
}

// Initialize Aptos client
const aptosConfig = new AptosConfig({ network: Network.TESTNET })
const aptos = new Aptos(aptosConfig)

// Parse private key and create account
const privateKeyHex = PRIVATE_KEY.startsWith("0x") ? PRIVATE_KEY.slice(2) : PRIVATE_KEY
const privateKey = new Ed25519PrivateKey(privateKeyHex)
const account = Account.fromPrivateKey({ privateKey })

printInfo("API Base URL", API_URL)
printInfo("Wallet Address", account.accountAddress.toString())
printInfo("Gasless Mode", GASLESS ? "Enabled (facilitator pays gas)" : "Disabled (direct on-chain)")
printInfo("Symbol", SYMBOL)
console.log("")

/**
 * Step 1: Discover Agent
 */
printStep(1, "Discovering Agent Capabilities")

try {
  const response = await fetch(`${API_URL}/api/agent/discover`)
  const agentInfo = await response.json()
  
  printSuccess("Agent discovered!")
  printInfo("Agent Name", agentInfo.name)
  printInfo("Description", agentInfo.description)
  printInfo("Capabilities", agentInfo.capabilities.endpoints.length.toString())
  printInfo("Pricing Model", agentInfo.pricing.model)
  
  console.log("\n   Available Endpoints:")
  agentInfo.capabilities.endpoints.slice(0, 5).forEach(ep => {
    console.log(`   • ${ep.name.padEnd(20)} ${ep.price.amount} octas`)
  })
} catch (error) {
  printError(`Discovery failed: ${error.message}`)
  process.exit(1)
}

await sleep(1000)

/**
 * Step 2: Check Balance
 */
printStep(2, "Checking Wallet Balance")

let balance = 0
try {
  // Try new Fungible Asset format first (testnet migrated to this)
  const [faBalance] = await aptos.view({
    payload: {
      function: "0x1::primary_fungible_store::balance",
      typeArguments: ["0x1::fungible_asset::Metadata"],
      functionArguments: [account.accountAddress.toString(), "0xa"],
    },
  })
  balance = parseInt(faBalance)
} catch {
  // Fall back to legacy CoinStore format
  try {
    const resource = await aptos.account.getAccountResource({
      accountAddress: account.accountAddress,
      resourceType: "0x1::coin::CoinStore<0x1::aptos_coin::AptosCoin>",
    })
    balance = parseInt(resource.coin.value)
  } catch {
    balance = 0
  }
}

if (balance > 0) {
  printSuccess(`Balance: ${formatOctas(balance)}`)
} else {
  printError("Wallet has no funds on testnet.")
  console.log("")
  console.log("   To fund the wallet:")
  console.log("   1. Open https://aptos.dev/network/faucet")
  console.log("   2. Connect or paste your address")
  console.log(`   3. Address: ${account.accountAddress.toString()}`)
  console.log("")
  process.exit(1)
}

if (balance < 1000) {
  printError("Balance too low for API payments. Fund the wallet first.")
  process.exit(1)
}

await sleep(1000)

/**
 * Step 3: Call Protected API (Momentum Analysis)
 */
printStep(3, "Calling Protected API: Momentum Analysis")

const endpoint = "/api/agent/momentum"
const url = `${API_URL}${endpoint}?symbol=${SYMBOL}`

console.log(`   Endpoint: ${endpoint}`)
console.log(`   URL: ${url}`)
console.log("")

// Try calling without payment
console.log("   → Sending initial request (no payment)...")
let response = await fetch(url)

if (response.status === 402) {
  const errorData = await response.json()
  const paymentRequest = errorData.payment
  
  printSuccess("Received 402 Payment Required")
  printInfo("Amount Required", `${paymentRequest.amount} octas`)
  printInfo("Recipient", `${paymentRequest.recipient.slice(0, 6)}...${paymentRequest.recipient.slice(-4)}`)
  printInfo("Deadline", new Date(paymentRequest.deadline * 1000).toLocaleString())
  
  await sleep(1000)
  
  /**
   * Step 4: Construct and Execute Payment
   */
  printStep(4, "Executing Payment")
  
  console.log("   → Building transaction...")
  const transaction = await aptos.transferCoinTransaction({
    sender: account.accountAddress,
    recipient: paymentRequest.recipient,
    amount: BigInt(paymentRequest.amount),
  })
  
  console.log("   → Signing transaction...")
  const senderAuthenticator = aptos.transaction.sign({
    signer: account,
    transaction,
  })
  
  let txHash
  
  if (GASLESS) {
    /**
     * GASLESS MODE: Use Facilitator
     */
    console.log("   → Using facilitator (gasless)...")
    
    // Serialize transaction
    const rawTxBytes = transaction.rawTransaction.bcsToBytes()
    const authenticatorBytes = senderAuthenticator.bcsToBytes()
    
    const combined = new Uint8Array(4 + rawTxBytes.length + authenticatorBytes.length)
    const view = new DataView(combined.buffer)
    view.setUint32(0, rawTxBytes.length, true)
    combined.set(rawTxBytes, 4)
    combined.set(authenticatorBytes, 4 + rawTxBytes.length)
    
    const signedTxHex = "0x" + Array.from(combined)
      .map(b => b.toString(16).padStart(2, "0"))
      .join("")
    
    // Submit to facilitator
    const settleResponse = await fetch(`${FACILITATOR_URL}settle`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        signedTx: signedTxHex,
        paymentRequest,
      }),
    })
    
    const settleResult = await settleResponse.json()
    
    if (!settleResult.success) {
      throw new Error(`Facilitator failed: ${settleResult.error}`)
    }
    
    txHash = settleResult.txHash
    printSuccess(`Gasless transaction submitted!`)
    printInfo("Transaction Hash", `${txHash.slice(0, 10)}...${txHash.slice(-8)}`)
    printInfo("Gas Paid By", "Facilitator (you paid 0 gas!)")
    
  } else {
    /**
     * DIRECT MODE: Submit to blockchain
     */
    console.log("   → Submitting directly to blockchain...")
    
    const pendingTxn = await aptos.transaction.submit.simple({
      transaction,
      senderAuthenticator,
    })
    
    txHash = pendingTxn.hash
    printSuccess(`Transaction submitted!`)
    printInfo("Transaction Hash", `${txHash.slice(0, 10)}...${txHash.slice(-8)}`)
    
    console.log("   → Waiting for confirmation...")
    await aptos.waitForTransaction({ transactionHash: txHash })
    printSuccess("Transaction confirmed!")
  }
  
  await sleep(1000)
  
  /**
   * Step 5: Retry with Payment Header
   */
  printStep(5, "Retrying API Call with Payment")
  
  console.log(`   → Sending request with X-Payment-Tx header...`)
  response = await fetch(url, {
    headers: {
      "X-Payment-Tx": txHash,
    },
  })
}

if (!response.ok) {
  const error = await response.text()
  printError(`Request failed: ${response.status} - ${error}`)
  process.exit(1)
}

/**
 * Step 6: Display Results
 */
printStep(6, "API Response Received!")

const data = await response.json()

printSuccess("Momentum Analysis Complete")
console.log("")
console.log("   📈 Results:")
console.log(`   Symbol:          ${data.symbol}`)
console.log(`   Momentum Score:  ${data.momentum_score}/100`)
console.log(`   Signal:          ${data.signal.toUpperCase()}`)
console.log(`   Timestamp:       ${data.timestamp}`)

if (data.breakdown) {
  console.log("")
  console.log("   Score Breakdown:")
  Object.entries(data.breakdown).forEach(([key, value]) => {
    if (typeof value === "object") {
      console.log(`   • ${key}:`)
      Object.entries(value).forEach(([k, v]) => {
        console.log(`     - ${k}: ${v}`)
      })
    }
  })
}

/**
 * Step 7: Update On-Chain Reputation
 */
printStep(7, "Recording On-Chain Reputation")

const REGISTRY_ADDRESS = process.env.AGENT_REGISTRY_ADDRESS || "0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240"
const REGISTRY_PRIVATE_KEY = process.env.AGENT_REGISTRY_PRIVATE_KEY

if (REGISTRY_PRIVATE_KEY) {
  try {
    console.log("   → Building reputation update transaction...")
    const regKeyHex = REGISTRY_PRIVATE_KEY.startsWith("0x") ? REGISTRY_PRIVATE_KEY.slice(2) : REGISTRY_PRIVATE_KEY
    const regPrivateKey = new Ed25519PrivateKey(regKeyHex)
    const regAccount = Account.fromPrivateKey({ privateKey: regPrivateKey })
    const agentAddress = regAccount.accountAddress.toString()

    // Use a retry mechanism for transient sequence number issues
    let repPending
    let retries = 3
    while (retries > 0) {
      try {
        // Build fresh transaction (gets latest sequence number from chain)
        const repTx = await aptos.transaction.build.simple({
          sender: regAccount.accountAddress,
          data: {
            function: `${REGISTRY_ADDRESS}::minimal_registry::update_reputation`,
            functionArguments: [agentAddress, true, 100, 0],
          },
        })
        
        // Sign and submit
        repPending = await aptos.signAndSubmitTransaction({
          signer: regAccount,
          transaction: repTx,
        })
        break // Success, exit retry loop
      } catch (txErr) {
        retries--
        const isSeqError = txErr.message?.includes("SEQUENCE_NUMBER_TOO_OLD") || 
                           txErr.message?.includes("vm_error_code\":3")
        if (isSeqError && retries > 0) {
          console.log("   → Sequence number stale, retrying...")
          await sleep(1000) // Wait longer for chain state to settle
        } else {
          throw txErr
        }
      }
    }

    await aptos.waitForTransaction({ transactionHash: repPending.hash })
    printSuccess("Reputation updated on-chain!")
    printInfo("Tx Hash", `${repPending.hash.slice(0, 10)}...${repPending.hash.slice(-8)}`)

    // Fetch and display updated reputation
    console.log("")
    console.log("   → Fetching updated reputation...")
    const [repData] = await aptos.view({
      payload: {
        function: `${REGISTRY_ADDRESS}::minimal_registry::get_reputation`,
        functionArguments: [agentAddress],
      },
    })

    printInfo("Total Requests", repData.total_requests)
    printInfo("Successful", repData.successful_requests)
    printInfo("Total Earned", `${repData.total_earned} octas`)
  } catch (repErr) {
    printError(`Reputation update failed: ${repErr.message}`)
    console.log("   (This is non-critical — the API call still succeeded)")
  }
} else {
  console.log("   Skipped — AGENT_REGISTRY_PRIVATE_KEY not set in .env")
  console.log("   Set it to record API usage on-chain.")
}

console.log("")
console.log("─".repeat(66))
console.log("✅ Demo completed successfully!")
console.log("")
console.log("💡 Key Takeaways:")
console.log("   • x402 enables pay-per-use API access")
console.log("   • Gasless transactions = users pay only API fee")
console.log("   • Automatic payment handling via SDK")
console.log("   • No accounts, subscriptions, or UI needed")
console.log("   • On-chain reputation tracks agent reliability")
console.log("")
console.log("🔗 View transaction on Aptos Explorer:")
console.log(`   https://explorer.aptoslabs.com/txn/${data.txHash || "your_tx_hash"}?network=testnet`)
console.log("")
