import { Aptos, AptosConfig, Network } from "@aptos-labs/ts-sdk"
import { FacilitatorClient, VerificationMode, defaultFacilitator } from "./x402-facilitator"
import { getPriceForEndpoint as getPrice } from "./x402-config"

const APTOS_NETWORK_URL = process.env.APTOS_NETWORK || "https://fullnode.testnet.aptoslabs.com/v1"

// Determine network from URL
const getNetworkFromUrl = (url: string): Network => {
  if (url.includes("mainnet")) return Network.MAINNET
  if (url.includes("testnet")) return Network.TESTNET
  if (url.includes("devnet")) return Network.DEVNET
  return Network.TESTNET // default
}

const APTOS_NETWORK = getNetworkFromUrl(APTOS_NETWORK_URL)

/**
 * X402 Payment Request
 * Sent to client when payment is required
 */
export interface PaymentRequest {
  amount: string
  recipient: string
  deadline: number
  nonce: string
  metadata: {
    endpoint: string
    symbol?: string
    params?: Record<string, any>
  }
}

/**
 * Payment Receipt
 * Created after successful payment verification
 */
export interface PaymentReceipt {
  txHash: string
  amount: string
  recipient: string
  sender: string
  endpoint: string
  timestamp: number
  nonce: string
  verificationMode: "direct" | "facilitator"
}

/**
 * In-memory store for used transaction hashes (prevents replay attacks)
 * In production, use Redis or a database
 */
class UsedTransactionsStore {
  private usedTxHashes: Map<string, PaymentReceipt> = new Map()
  private maxSize = 10000

  has(txHash: string): boolean {
    return this.usedTxHashes.has(txHash)
  }

  add(receipt: PaymentReceipt): void {
    if (this.usedTxHashes.size >= this.maxSize) {
      const oldestKey = this.usedTxHashes.keys().next().value
      if (oldestKey) {
        this.usedTxHashes.delete(oldestKey)
      }
    }
    this.usedTxHashes.set(receipt.txHash, receipt)
  }

  get(txHash: string): PaymentReceipt | undefined {
    return this.usedTxHashes.get(txHash)
  }

  cleanup(): void {
    const now = Date.now()
    const maxAge = 24 * 60 * 60 * 1000
    for (const [txHash, receipt] of this.usedTxHashes.entries()) {
      if (now - receipt.timestamp > maxAge) {
        this.usedTxHashes.delete(txHash)
      }
    }
  }
}

const usedTransactions = new UsedTransactionsStore()

/**
 * X402 Payment Middleware
 * 
 * Handles payment verification via:
 * 1. Direct on-chain verification (default, more decentralized)
 * 2. Facilitator verification (faster, enables gasless transactions)
 */
export class X402Middleware {
  private aptos: Aptos
  private recipientAddress: string
  private pricePerRequest: number
  private facilitator: FacilitatorClient | null
  private verificationMode: VerificationMode

  constructor(
    recipientAddress: string, 
    pricePerRequest: number = 100,
    options: {
      facilitator?: FacilitatorClient
      verificationMode?: VerificationMode
    } = {}
  ) {
    const config = new AptosConfig({ 
      network: APTOS_NETWORK,
      fullnode: APTOS_NETWORK_URL 
    })
    this.aptos = new Aptos(config)
    this.recipientAddress = recipientAddress
    this.pricePerRequest = pricePerRequest
    this.facilitator = options.facilitator || null
    this.verificationMode = options.verificationMode || VerificationMode.DIRECT
  }

  /**
   * Generate a payment request for a client
   */
  generatePaymentRequest(
    endpoint: string,
    symbol?: string,
    params?: Record<string, any>
  ): PaymentRequest {
    const nonce = Buffer.from(`${Date.now()}-${Math.random()}`).toString("base64")
    const deadline = Math.floor(Date.now() / 1000) + 300 // 5 minutes

    return {
      amount: this.pricePerRequest.toString(),
      recipient: this.recipientAddress,
      deadline,
      nonce,
      metadata: {
        endpoint,
        symbol,
        params,
      },
    }
  }

  /**
   * Verify a payment transaction
   * 
   * Supports three modes:
   * - DIRECT: Verify on-chain directly (default)
   * - FACILITATOR: Verify via facilitator service
   * - HYBRID: Try facilitator first, fall back to direct
   */
  async verifyPayment(
    txHash: string, 
    expectedPayment: { 
      amount: string
      recipient: string
      endpoint: string
      nonce?: string
    }
  ): Promise<{ 
    valid: boolean
    receipt?: PaymentReceipt
    error?: string
    mode?: "direct" | "facilitator"
  }> {
    // Check for replay attack
    if (usedTransactions.has(txHash)) {
      return { valid: false, error: "Transaction already used" }
    }

    // Try facilitator first if in FACILITATOR or HYBRID mode
    if (
      (this.verificationMode === VerificationMode.FACILITATOR ||
        this.verificationMode === VerificationMode.HYBRID) &&
      this.facilitator
    ) {
      const result = await this.verifyViaFacilitator(txHash, expectedPayment)
      if (result.valid) {
        return { ...result, mode: "facilitator" }
      }
      // In FACILITATOR-only mode, don't fall through to direct
      if (this.verificationMode === VerificationMode.FACILITATOR) {
        return { ...result, mode: "facilitator" }
      }
      // HYBRID: fall through to direct verification
    }

    // Direct on-chain verification (DIRECT mode, or HYBRID fallback)
    const result = await this.verifyDirect(txHash, expectedPayment)
    return { ...result, mode: "direct" }
  }

  /**
   * Verify payment via facilitator (faster, supports gasless)
   */
  private async verifyViaFacilitator(
    txHash: string,
    expectedPayment: { amount: string; recipient: string; endpoint: string; nonce?: string }
  ): Promise<{ valid: boolean; receipt?: PaymentReceipt; error?: string }> {
    if (!this.facilitator) {
      return { valid: false, error: "Facilitator not configured" }
    }

    // Get network identifier
    const network = this.getNetworkIdentifier()

    const result = await this.facilitator.verify({
      txHash,
      payment: {
        amount: expectedPayment.amount,
        recipient: expectedPayment.recipient,
        network,
      },
      paymentRequest: this.generatePaymentRequest(expectedPayment.endpoint),
    })

    if (!result.valid || !result.transaction) {
      return { 
        valid: false, 
        error: result.error || "Facilitator verification failed" 
      }
    }

    const receipt: PaymentReceipt = {
      txHash,
      amount: result.transaction.amount,
      recipient: result.transaction.recipient,
      sender: result.transaction.sender,
      endpoint: expectedPayment.endpoint,
      timestamp: Date.now(),
      nonce: expectedPayment.nonce || txHash,
      verificationMode: "facilitator",
    }

    usedTransactions.add(receipt)
    return { valid: true, receipt }
  }

  /**
   * Verify payment directly on-chain (more decentralized)
   */
  private async verifyDirect(
    txHash: string,
    expectedPayment: { amount: string; recipient: string; endpoint: string; nonce?: string }
  ): Promise<{ valid: boolean; receipt?: PaymentReceipt; error?: string }> {
    try {
      const tx = await this.aptos.transaction.getTransactionByHash({ transactionHash: txHash })
      
      if (!tx) {
        return { valid: false, error: "Transaction not found" }
      }

      const userTx = tx as any
      if (userTx.vm_status !== "Executed successfully") {
        return { valid: false, error: `Transaction failed: ${userTx.vm_status}` }
      }

      // Verify transaction age (max 1 hour)
      const txTimestamp = userTx.timestamp ? parseInt(userTx.timestamp) / 1000 : 0
      const now = Date.now()
      if (txTimestamp > 0 && (now - txTimestamp) > 60 * 60 * 1000) {
        return { valid: false, error: "Transaction too old" }
      }

      // Parse and verify transaction payload
      const payload = userTx.payload
      if (!payload) {
        return { valid: false, error: "Invalid transaction payload" }
      }

      if (payload.type === "entry_function_payload") {
        const functionName = payload.function
        
        const isTransfer = 
          functionName.includes("::aptos_account::transfer") ||
          functionName.includes("::coin::transfer") ||
          functionName.includes("0x1::aptos_account::transfer") ||
          functionName.includes("0x1::coin::transfer")

        if (!isTransfer) {
          return { valid: false, error: "Not a payment transaction" }
        }

        const args = payload.arguments || []
        if (args.length < 2) {
          return { valid: false, error: "Invalid payment arguments" }
        }

        const txRecipient = args[0]
        const txAmount = args[1]

        // Normalize and verify recipient
        const normalizedExpectedRecipient = expectedPayment.recipient.toLowerCase().replace(/^0x/, '')
        const normalizedTxRecipient = txRecipient.toLowerCase().replace(/^0x/, '')

        if (normalizedTxRecipient !== normalizedExpectedRecipient) {
          return { valid: false, error: "Recipient mismatch" }
        }

        // Verify amount (with 1% tolerance)
        const expectedAmount = BigInt(expectedPayment.amount)
        const actualAmount = BigInt(txAmount)
        const tolerance = expectedAmount / BigInt(100)
        
        if (actualAmount < expectedAmount - tolerance) {
          return { valid: false, error: "Insufficient payment amount" }
        }

        const receipt: PaymentReceipt = {
          txHash,
          amount: txAmount.toString(),
          recipient: txRecipient,
          sender: userTx.sender,
          endpoint: expectedPayment.endpoint,
          timestamp: Date.now(),
          nonce: expectedPayment.nonce || txHash,
          verificationMode: "direct",
        }

        usedTransactions.add(receipt)
        return { valid: true, receipt }
      }

      return { valid: false, error: "Unsupported transaction type" }

    } catch (error) {
      console.error("Direct payment verification failed:", error)
      return { valid: false, error: "Verification error" }
    }
  }

  /**
   * Get network identifier for facilitator
   */
  private getNetworkIdentifier(): string {
    switch (APTOS_NETWORK) {
      case Network.MAINNET: return "aptos:1"
      case Network.TESTNET: return "aptos:2"
      case Network.DEVNET: return "aptos:3"
      default: return "aptos:2" // default to testnet
    }
  }

  create402Response(paymentRequest: PaymentRequest): Response {
    return new Response(
      JSON.stringify({
        error: "Payment Required",
        code: 402,
        payment: paymentRequest,
        message: "Payment required to access this endpoint",
        gasless: this.facilitator?.isGaslessEnabled() || false,
      }),
      {
        status: 402,
        headers: {
          "Content-Type": "application/json",
        },
      }
    )
  }
}

const RECIPIENT_ADDRESS = process.env.APTOS_RECIPIENT_ADDRESS || "0x1"
const PRICE_PER_REQUEST = parseInt(process.env.APTOS_PRICE_PER_REQUEST || "100")

// Determine verification mode from environment
const getVerificationMode = (): VerificationMode => {
  const mode = process.env.X402_VERIFICATION_MODE?.toLowerCase()
  if (mode === "facilitator") return VerificationMode.FACILITATOR
  if (mode === "hybrid") return VerificationMode.HYBRID
  return VerificationMode.DIRECT
}

// Determine if facilitator should be used
const shouldUseFacilitator = (): boolean => {
  return process.env.X402_USE_FACILITATOR === "true" || 
         getVerificationMode() === VerificationMode.FACILITATOR
}

export const x402Middleware = new X402Middleware(
  RECIPIENT_ADDRESS, 
  PRICE_PER_REQUEST,
  {
    facilitator: shouldUseFacilitator() ? defaultFacilitator : undefined,
    verificationMode: getVerificationMode(),
  }
)

export async function requirePayment(
  request: Request,
  endpoint: string,
  symbol?: string,
  params?: Record<string, any>
): Promise<{ 
  authorized: boolean
  paymentRequest?: PaymentRequest
  txHash?: string
  receipt?: PaymentReceipt
  error?: string
}> {
  const authHeader = request.headers.get("X-Payment-Tx")

  const expectedAmount = getPriceForEndpoint(endpoint)

  if (!authHeader) {
    const paymentRequest = x402Middleware.generatePaymentRequest(endpoint, symbol, params)
    // Override amount with the endpoint-specific price
    paymentRequest.amount = expectedAmount.toString()
    return {
      authorized: false,
      paymentRequest,
      error: "Payment required. Include transaction hash in X-Payment-Tx header."
    }
  }

  const result = await x402Middleware.verifyPayment(authHeader, { 
    amount: expectedAmount.toString(),
    recipient: RECIPIENT_ADDRESS,
    endpoint,
    nonce: undefined
  })

  if (!result.valid) {
    const paymentRequest = x402Middleware.generatePaymentRequest(endpoint, symbol, params)
    paymentRequest.amount = expectedAmount.toString()
    return {
      authorized: false,
      paymentRequest,
      error: result.error || "Payment verification failed"
    }
  }

  return { 
    authorized: true, 
    txHash: authHeader,
    receipt: result.receipt
  }
}

function getPriceForEndpoint(endpoint: string): number {
  return getPrice(endpoint)
}

export { usedTransactions, VerificationMode }
