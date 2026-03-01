import { NextRequest, NextResponse } from "next/server"
import { X402Middleware, PaymentRequest, PaymentReceipt, requirePayment } from "./x402-middleware"
import { PROTECTED_ENDPOINTS, X402ProtectedRoute, getPriceForEndpoint } from "./x402-config"
import { recordReputationUpdate } from "./agent-registry"

export type { X402ProtectedRoute }

// Re-export so existing imports from x402-enforcer still work
export { PROTECTED_ENDPOINTS }

export class X402Enforcer {
  private static instance: X402Enforcer
  private middleware: X402Middleware
  private protectedRoutes: Map<string, X402ProtectedRoute & { params?: (request: NextRequest) => { symbol?: string; [key: string]: any } }> = new Map()
  private _isConfigured: boolean = false

  private constructor() {
    const recipient = process.env.APTOS_RECIPIENT_ADDRESS
    const price = parseInt(process.env.APTOS_PRICE_PER_REQUEST || "100")

    if (!recipient || recipient === "0x1") {
      console.warn(
        "⚠️  X402 WARNING: APTOS_RECIPIENT_ADDRESS not configured or using default. " +
        "Set a valid recipient address to enable payment enforcement."
      )
      this._isConfigured = false
    } else {
      this._isConfigured = true
      console.log(`✓ X402 Payment Enforcement Enabled`)
      console.log(`  Recipient: ${recipient}`)
      console.log(`  Price per request: ${price} octas`)
    }

    this.middleware = new X402Middleware(recipient || "0x1", price)
  }

  static getInstance(): X402Enforcer {
    if (!X402Enforcer.instance) {
      X402Enforcer.instance = new X402Enforcer()
    }
    return X402Enforcer.instance
  }

  registerRoute(route: X402ProtectedRoute & { params?: (request: NextRequest) => { symbol?: string; [key: string]: any } }): void {
    this.protectedRoutes.set(route.path, route)
    console.log(`✓ Protected route: ${route.path}`)
  }

  registerRoutes(routes: X402ProtectedRoute[]): void {
    routes.forEach(route => this.registerRoute(route))
  }

  isRouteProtected(path: string): boolean {
    return this.protectedRoutes.has(path)
  }

  async checkPayment(request: NextRequest, endpoint: string): Promise<{
    authorized: boolean
    paymentRequest?: PaymentRequest
    txHash?: string
    receipt?: PaymentReceipt
    error?: string
  }> {
    const route = this.protectedRoutes.get(endpoint)
    const params = route?.params ? route.params(request) : undefined

    return requirePayment(request, endpoint, params?.symbol, params)
  }

  create402Response(paymentRequest: PaymentRequest): NextResponse {
    return NextResponse.json(
      {
        error: "Payment Required",
        code: 402,
        payment: paymentRequest,
        message: "Payment required to access this endpoint",
      },
      { status: 402 }
    )
  }

  createErrorResponse(message: string, status: number = 402): NextResponse {
    return NextResponse.json(
      {
        error: message,
        code: status,
      },
      { status }
    )
  }

  enforce(
    endpoint: string,
    params?: { symbol?: string; [key: string]: any }
  ): (request: NextRequest) => Promise<NextResponse | null> {
    return async (request: NextRequest) => {
      if (!this._isConfigured) {
        console.warn(`⚠️  Payment enforcement disabled - route: ${endpoint}`)
        return null
      }

      const check = await this.checkPayment(request, endpoint)

      if (!check.authorized) {
        if (check.paymentRequest) {
          return this.create402Response(check.paymentRequest)
        }
        return this.createErrorResponse(check.error || "Payment Required", 402)
      }

      return null
    }
  }

  getMiddleware(): X402Middleware {
    return this.middleware
  }

  getRecipientAddress(): string {
    return process.env.APTOS_RECIPIENT_ADDRESS || "0x1"
  }

  getPricePerRequest(): number {
    return parseInt(process.env.APTOS_PRICE_PER_REQUEST || "100")
  }

  validateConfig(): { valid: boolean; errors: string[] } {
    const errors: string[] = []

    if (!process.env.APTOS_RECIPIENT_ADDRESS) {
      errors.push("APTOS_RECIPIENT_ADDRESS is not set")
    } else if (process.env.APTOS_RECIPIENT_ADDRESS === "0x1") {
      errors.push("APTOS_RECIPIENT_ADDRESS is set to default value (0x1)")
    }

    if (!process.env.APTOS_PRICE_PER_REQUEST) {
      errors.push("APTOS_PRICE_PER_REQUEST is not set")
    } else {
      const price = parseInt(process.env.APTOS_PRICE_PER_REQUEST)
      if (isNaN(price) || price <= 0) {
        errors.push("APTOS_PRICE_PER_REQUEST must be a positive number")
      }
    }

    // Validate network URL if set
    if (process.env.APTOS_NETWORK) {
      const validNetworks = [
        "https://fullnode.mainnet.aptoslabs.com/v1",
        "https://fullnode.testnet.aptoslabs.com/v1",
        "https://fullnode.devnet.aptoslabs.com/v1",
      ]
      if (!validNetworks.includes(process.env.APTOS_NETWORK) &&
          !process.env.APTOS_NETWORK.startsWith("http://localhost")) {
        errors.push("APTOS_NETWORK appears to be invalid")
      }
    }

    return {
      valid: errors.length === 0,
      errors,
    }
  }

  isConfigured(): boolean {
    return this._isConfigured
  }
}

export const x402Enforcer = X402Enforcer.getInstance()

export async function withX402(
  request: NextRequest,
  endpoint: string,
  handler: (request: NextRequest, txHash?: string, receipt?: PaymentReceipt) => Promise<NextResponse>
): Promise<NextResponse> {
  // Bypass payment for internal app requests (injected by middleware)
  const internalKey = request.headers.get("X-Internal-Key")
  if (internalKey && internalKey === process.env.INTERNAL_API_KEY) {
    console.info(`[x402] INTERNAL_API_KEY bypass for ${endpoint}`)
    return handler(request)
  }

  const enforcement = await x402Enforcer.checkPayment(request, endpoint)

  if (!enforcement.authorized) {
    if (enforcement.paymentRequest) {
      return x402Enforcer.create402Response(enforcement.paymentRequest)
    }
    return x402Enforcer.createErrorResponse(
      enforcement.error || "Payment Required",
      402
    )
  }

  const response = await handler(request, enforcement.txHash, enforcement.receipt)

  // Fire-and-forget reputation update on successful payment + response
  const success = response.status >= 200 && response.status < 400
  const amount = getPriceForEndpoint(endpoint)
  recordReputationUpdate(endpoint, amount, success).catch((err) => {
    console.error("[reputation] fire-and-forget failed:", err?.message || err)
  })

  return response
}

x402Enforcer.registerRoutes(PROTECTED_ENDPOINTS)
