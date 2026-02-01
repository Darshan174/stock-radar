/**
 * x402 Client SDK
 * 
 * This SDK makes it easy to consume x402-protected APIs.
 * It automatically handles:
 * 1. Making initial requests
 * 2. Detecting 402 Payment Required responses
 * 3. Constructing and signing payments
 * 4. Retrying with payment headers
 * 
 * ## Quick Start
 * 
 * ```typescript
 * import { x402Client } from "@/lib/x402-client"
 * 
 * // Create client with your private key
 * const client = x402Client({
 *   privateKey: "0x...",
 *   network: "testnet"
 * })
 * 
 * // Call protected API - payment is handled automatically!
 * const data = await client.get("/api/agent/momentum?symbol=AAPL")
 * console.log(data)
 * ```
 * 
 * ## How It Works
 * 
 * ```
 * Step 1: Client calls API
 *         GET /api/agent/momentum?symbol=AAPL
 *         
 * Step 2: Server returns 402 with payment requirements
 *         { amount: "100", recipient: "0x...", ... }
 *         
 * Step 3: SDK automatically:
 *         - Constructs Aptos transaction
 *         - Signs with your private key
 *         - Submits to blockchain
 *         - Retries with X-Payment-Tx header
 *         
 * Step 4: Server verifies and returns data
 *         { momentum_score: 75, signal: "bullish" }
 * ```
 * 
 * ## Gasless Transactions (via Facilitator)
 * 
 * When gasless mode is enabled:
 * 1. SDK constructs and signs transaction
 * 2. Sends signed tx to facilitator
 * 3. Facilitator submits to blockchain (pays gas!)
 * 4. Facilitator returns txHash
 * 5. SDK retries API call with txHash
 * 
 * Benefits:
 * - Users pay only API fee (no gas!)
 * - Faster submission
 * - Lower barrier to entry
 */

import { 
  Account, 
  Ed25519PrivateKey, 
  Aptos, 
  AptosConfig, 
  Network,
  SimpleTransaction
} from "@aptos-labs/ts-sdk"
import { PaymentRequest } from "./x402-middleware"
import { FacilitatorClient } from "./x402-facilitator"

export interface X402ClientConfig {
  /** Your Aptos private key (with or without 0x prefix) */
  privateKey: string
  /** Network to use */
  network?: "mainnet" | "testnet" | "devnet" | "local"
  /** Facilitator URL (optional, for gasless transactions) */
  facilitatorUrl?: string
  /** Enable gasless transactions via facilitator */
  gasless?: boolean
  /** Base URL for API calls */
  baseUrl?: string
}

export interface X402Client {
  /** Make a GET request to a protected endpoint */
  get(url: string, options?: RequestInit): Promise<any>
  /** Make a POST request to a protected endpoint */
  post(url: string, body: any, options?: RequestInit): Promise<any>
  /** Get the wallet address */
  getAddress(): string
  /** Check wallet balance */
  getBalance(): Promise<string>
}

/**
 * Create an x402 client
 * 
 * @param config Client configuration
 * @returns X402Client instance
 * 
 * @example
 * ```typescript
 * const client = x402Client({
 *   privateKey: process.env.APTOS_PRIVATE_KEY!,
 *   network: "testnet",
 *   gasless: true // Use facilitator for gasless transactions
 * })
 * 
 * // Call protected API
 * const result = await client.get("/api/agent/momentum?symbol=AAPL")
 * console.log(result.momentum_score)
 * ```
 */
export function x402Client(config: X402ClientConfig): X402Client {
  // Parse private key
  const privateKeyHex = config.privateKey.startsWith('0x') 
    ? config.privateKey.slice(2) 
    : config.privateKey
  
  const privateKey = new Ed25519PrivateKey(privateKeyHex)
  const account = Account.fromPrivateKey({ privateKey })
  const address = account.accountAddress.toString()

  // Setup Aptos client
  const network = config.network || "testnet"
  const aptosConfig = new AptosConfig({ 
    network: network as Network 
  })
  const aptos = new Aptos(aptosConfig)

  // Setup facilitator if gasless is enabled
  const facilitator = config.gasless
    ? new FacilitatorClient({
        ...(config.facilitatorUrl ? { url: config.facilitatorUrl } : {}),
        gasless: true,
        timeout: 30000
      })
    : null

  const baseUrl = config.baseUrl || ""

  /**
   * Execute a payment transaction
   * 
   * This either:
   * - Submits directly to blockchain (user pays gas)
   * - Uses facilitator for gasless submission (facilitator pays gas)
   */
  async function executePayment(
    paymentRequest: PaymentRequest
  ): Promise<{ txHash: string }> {
    const recipient = paymentRequest.recipient
    const amount = BigInt(paymentRequest.amount)

    console.log(`Executing payment: ${amount} octas to ${recipient}`)

    // Build simple transfer transaction
    const transaction = await aptos.transferCoinTransaction({
      sender: account.accountAddress,
      recipient: recipient,
      amount: amount,
    })

    // Sign transaction
    const senderAuthenticator = aptos.transaction.sign({
      signer: account,
      transaction,
    })

    // If gasless and facilitator is available, use it
    if (facilitator && config.gasless) {
      console.log("Using facilitator for gasless transaction...")
      
      // For facilitator, we need to serialize the signed transaction
      // The facilitator expects the raw transaction bytes
      const rawTransaction = transaction.rawTransaction
      
      // Serialize the raw transaction and authenticator
      // We'll use a hex encoding of both
      const rawTxBytes = rawTransaction.bcsToBytes()
      const authenticatorBytes = senderAuthenticator.bcsToBytes()
      
      // Combine: raw_tx_len (4 bytes) + raw_tx + authenticator
      const combined = new Uint8Array(
        4 + rawTxBytes.length + authenticatorBytes.length
      )
      
      // Write length of raw transaction as 4 bytes (little endian)
      const view = new DataView(combined.buffer)
      view.setUint32(0, rawTxBytes.length, true)
      
      // Copy raw transaction
      combined.set(rawTxBytes, 4)
      
      // Copy authenticator
      combined.set(authenticatorBytes, 4 + rawTxBytes.length)
      
      // Convert to hex
      const signedTxHex = '0x' + Array.from(combined)
        .map(b => b.toString(16).padStart(2, '0'))
        .join('')

      const result = await facilitator.settle({
        signedTx: signedTxHex,
        paymentRequest,
      })

      if (!result.success || !result.txHash) {
        throw new Error(`Facilitator settlement failed: ${result.error}`)
      }

      console.log("Gasless transaction submitted:", result.txHash)
      return { txHash: result.txHash }
    }

    // Otherwise, submit directly (user pays gas)
    console.log("Submitting transaction directly...")
    const pendingTxn = await aptos.transaction.submit.simple({
      transaction,
      senderAuthenticator,
    })

    console.log("Transaction submitted:", pendingTxn.hash)
    
    // Wait for confirmation
    await aptos.waitForTransaction({ transactionHash: pendingTxn.hash })
    console.log("Transaction confirmed!")

    return { txHash: pendingTxn.hash }
  }

  /**
   * Make a request with automatic payment handling
   */
  async function makeRequest(
    method: string,
    url: string,
    body?: any,
    options: RequestInit = {}
  ): Promise<any> {
    const fullUrl = url.startsWith('http') ? url : `${baseUrl}${url}`

    // Step 1: Try the request without payment
    console.log(`Step 1: Calling ${method} ${url}`)
    let response = await fetch(fullUrl, {
      method,
      headers: {
        "Content-Type": "application/json",
        ...options.headers,
      },
      body: body ? JSON.stringify(body) : undefined,
    })

    // Step 2: If we get 402, handle payment
    if (response.status === 402) {
      console.log("Step 2: Payment required, constructing transaction...")
      
      const errorData = await response.json()
      const paymentRequest: PaymentRequest = errorData.payment

      if (!paymentRequest) {
        throw new Error("Invalid 402 response: missing payment requirements")
      }

      console.log(`Payment required: ${paymentRequest.amount} octas to ${paymentRequest.recipient}`)
      console.log(`Gasless mode: ${config.gasless ? 'enabled (facilitator pays gas)' : 'disabled (you pay gas)'}`)

      // Execute payment
      const { txHash } = await executePayment(paymentRequest)

      // Step 3: Retry with payment header
      console.log("Step 3: Retrying with payment header...")
      response = await fetch(fullUrl, {
        method,
        headers: {
          "Content-Type": "application/json",
          "X-Payment-Tx": txHash,
          ...options.headers,
        },
        body: body ? JSON.stringify(body) : undefined,
      })
    }

    // Step 4: Return the result
    if (!response.ok) {
      const errorText = await response.text()
      throw new Error(`Request failed: ${response.status} - ${errorText}`)
    }

    return await response.json()
  }

  return {
    async get(url: string, options?: RequestInit): Promise<any> {
      return makeRequest("GET", url, undefined, options)
    },

    async post(url: string, body: any, options?: RequestInit): Promise<any> {
      return makeRequest("POST", url, body, options)
    },

    getAddress(): string {
      return address
    },

    async getBalance(): Promise<string> {
      // Try new Fungible Asset format first (testnet migrated to this)
      try {
        const [balance] = await aptos.view({
          payload: {
            function: "0x1::primary_fungible_store::balance",
            typeArguments: ["0x1::fungible_asset::Metadata"],
            functionArguments: [account.accountAddress.toString(), "0xa"],
          },
        })
        return String(balance)
      } catch {
        // Fall back to legacy CoinStore format
        try {
          const resource = await aptos.account.getAccountResource({
            accountAddress: account.accountAddress,
            resourceType: "0x1::coin::CoinStore<0x1::aptos_coin::AptosCoin>",
          })
          return (resource as any).coin.value.toString()
        } catch {
          return "0"
        }
      }
    },
  }
}

/**
 * Create a simple fetch wrapper that handles x402 payments
 * 
 * This is a lower-level API than x402Client - it wraps the native fetch
 * function to automatically handle 402 responses.
 * 
 * @example
 * ```typescript
 * const fetchWithPayment = wrapFetchWithX402({
 *   privateKey: "0x...",
 *   gasless: true
 * })
 * 
 * // Use just like regular fetch
 * const response = await fetchWithPayment("/api/agent/momentum?symbol=AAPL")
 * const data = await response.json()
 * ```
 */
export function wrapFetchWithX402(
  config: Omit<X402ClientConfig, 'baseUrl'>
): (input: RequestInfo | URL, init?: RequestInit) => Promise<Response> {
  const client = x402Client({ ...config, baseUrl: "" })

  return async (input: RequestInfo | URL, init?: RequestInit): Promise<Response> => {
    const url = input.toString()
    const method = init?.method || "GET"
    const body = init?.body

    try {
      const data = method === "GET" 
        ? await client.get(url, init)
        : await client.post(url, body, init)

      // Return a mock Response object
      return new Response(JSON.stringify(data), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    } catch (error) {
      return new Response(
        JSON.stringify({ error: (error as Error).message }), 
        { status: 500 }
      )
    }
  }
}

/**
 * Check if an API requires payment
 * 
 * @param url API endpoint to check
 * @returns Object with requiresPayment flag and payment details if applicable
 * 
 * @example
 * ```typescript
 * const check = await checkPaymentRequirement("/api/agent/momentum")
 * if (check.requiresPayment) {
 *   console.log(`Requires ${check.price} octas`)
 * }
 * ```
 */
export async function checkPaymentRequirement(url: string): Promise<{
  requiresPayment: boolean
  price?: string
  recipient?: string
  network?: string
}> {
  const response = await fetch(url, { method: "HEAD" })

  if (response.status === 402) {
    const data = await response.json().catch(() => ({}))
    return {
      requiresPayment: true,
      price: data.payment?.amount,
      recipient: data.payment?.recipient,
      network: data.payment?.network,
    }
  }

  return { requiresPayment: false }
}
