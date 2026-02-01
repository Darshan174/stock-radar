/**
 * x402 Facilitator Client
 * 
 * This module provides integration with the x402 facilitator service.
 * The facilitator handles blockchain interactions, allowing for gasless transactions.
 * 
 * Public Facilitator: https://x402-navy.vercel.app/facilitator/
 * 
 * What is a Facilitator?
 * ---------------------
 * A facilitator is a service that:
 * 1. Verifies payment transactions off-chain (fast)
 * 2. Submits transactions to the blockchain (sponsors gas)
 * 3. Provides settlement confirmations
 * 
 * Benefits:
 * - Gasless transactions (users don't need APT for gas)
 * - Faster verification (off-chain first)
 * - Simpler client code
 * 
 * Two Modes of Operation:
 * -----------------------
 * 1. DIRECT (default): Server verifies transactions directly on-chain
 *    - More decentralized
 *    - No dependency on external service
 *    - User pays gas
 * 
 * 2. FACILITATOR: Use facilitator for verification and settlement
 *    - Gasless transactions
 *    - Faster response
 *    - Depends on facilitator uptime
 */

import { PaymentRequest } from "./x402-middleware"

export interface FacilitatorConfig {
  /** Facilitator URL (default: public facilitator) */
  url: string
  /** Enable gasless transactions (facilitator pays gas) */
  gasless: boolean
  /** Timeout for facilitator requests (ms) */
  timeout: number
}

export interface VerifyRequest {
  /** Transaction hash to verify */
  txHash: string
  /** Expected payment details */
  payment: {
    amount: string
    recipient: string
    network: string
  }
  /** Payment request that was sent to client */
  paymentRequest: PaymentRequest
}

export interface VerifyResponse {
  /** Whether payment is valid */
  valid: boolean
  /** Transaction details if valid */
  transaction?: {
    hash: string
    sender: string
    amount: string
    recipient: string
    timestamp: string
  }
  /** Error message if invalid */
  error?: string
}

export interface SettleRequest {
  /** Signed transaction payload */
  signedTx: string
  /** Payment request */
  paymentRequest: PaymentRequest
}

export interface SettleResponse {
  /** Whether settlement succeeded */
  success: boolean
  /** Transaction hash on blockchain */
  txHash?: string
  /** Error message if failed */
  error?: string
}

// Default configuration uses public facilitator
const DEFAULT_CONFIG: FacilitatorConfig = {
  url: process.env.FACILITATOR_URL || "https://x402-navy.vercel.app/facilitator/",
  gasless: true,
  timeout: 30000, // 30 seconds
}

export class FacilitatorClient {
  private config: FacilitatorConfig

  constructor(config: Partial<FacilitatorConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config }
    
    // Ensure URL ends with /
    if (!this.config.url.endsWith('/')) {
      this.config.url += '/'
    }
  }

  /**
   * Verify a payment transaction via the facilitator
   * 
   * This is faster than on-chain verification because the facilitator
   * caches transaction data and validates off-chain first.
   * 
   * @param request Verification request
   * @returns Verification response
   * 
   * @example
   * ```typescript
   * const result = await facilitator.verify({
   *   txHash: "0xabc...",
   *   payment: {
   *     amount: "100",
   *     recipient: "0x123...",
   *     network: "aptos:2"
   *   },
   *   paymentRequest: originalPaymentRequest
   * })
   * 
   * if (result.valid) {
   *   console.log("Payment verified!", result.transaction)
   * }
   * ```
   */
  async verify(request: VerifyRequest): Promise<VerifyResponse> {
    try {
      const response = await fetch(`${this.config.url}verify`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          txHash: request.txHash,
          payment: request.payment,
          paymentRequest: request.paymentRequest,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Facilitator error: ${response.status} - ${errorText}`)
      }

      const result = await response.json()
      return result as VerifyResponse

    } catch (error) {
      console.error("Facilitator verification failed:", error)
      return {
        valid: false,
        error: error instanceof Error ? error.message : "Unknown error"
      }
    }
  }

  /**
   * Settle (submit) a signed transaction via the facilitator
   * 
   * This enables gasless transactions - the facilitator pays the gas fee
   * and submits the transaction to the blockchain.
   * 
   * @param request Settlement request with signed transaction
   * @returns Settlement response with txHash
   * 
   * @example
   * ```typescript
   * const result = await facilitator.settle({
   *   signedTx: signedTransactionHex,
   *   paymentRequest: originalPaymentRequest
   * })
   * 
   * if (result.success) {
   *   console.log("Transaction submitted!", result.txHash)
   * }
   * ```
   */
  async settle(request: SettleRequest): Promise<SettleResponse> {
    try {
      const response = await fetch(`${this.config.url}settle`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          signedTx: request.signedTx,
          paymentRequest: request.paymentRequest,
        }),
      })

      if (!response.ok) {
        const errorText = await response.text()
        throw new Error(`Facilitator error: ${response.status} - ${errorText}`)
      }

      const result = await response.json()
      return result as SettleResponse

    } catch (error) {
      console.error("Facilitator settlement failed:", error)
      return {
        success: false,
        error: error instanceof Error ? error.message : "Unknown error"
      }
    }
  }

  /**
   * Check if facilitator is available
   * 
   * @returns true if facilitator is reachable
   */
  async isAvailable(): Promise<boolean> {
    try {
      const response = await fetch(this.config.url, {
        method: "GET",
        signal: AbortSignal.timeout(5000), // 5 second timeout
      })
      return response.ok
    } catch {
      return false
    }
  }

  /**
   * Get current configuration
   */
  getConfig(): FacilitatorConfig {
    return { ...this.config }
  }

  /**
   * Check if gasless transactions are enabled
   */
  isGaslessEnabled(): boolean {
    return this.config.gasless
  }
}

// Singleton instance using default config
export const defaultFacilitator = new FacilitatorClient()

/**
 * Create a custom facilitator client
 * 
 * @example
 * ```typescript
 * const facilitator = createFacilitator({
 *   url: "https://my-facilitator.com/",
 *   gasless: true
 * })
 * ```
 */
export function createFacilitator(config: Partial<FacilitatorConfig>): FacilitatorClient {
  return new FacilitatorClient(config)
}

/**
 * Payment verification modes
 */
export enum VerificationMode {
  /** Verify directly on-chain (default, decentralized) */
  DIRECT = "direct",
  /** Verify via facilitator (faster, gasless) */
  FACILITATOR = "facilitator",
  /** Try facilitator first, fall back to direct */
  HYBRID = "hybrid",
}
