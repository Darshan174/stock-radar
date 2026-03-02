/**
 * X402 endpoint pricing configuration - single source of truth.
 *
 * All endpoint prices are defined here and consumed by both
 * x402-enforcer.ts and x402-middleware.ts.
 */

export type X402ProtectedRoute = {
  path: string
  price?: number
}

export const PROTECTED_ENDPOINTS: X402ProtectedRoute[] = [
  { path: "/api/agent/momentum", price: 100 },
  { path: "/api/agent/rsi-divergence", price: 100 },
  { path: "/api/agent/news-impact", price: 150 },
  { path: "/api/agent/stock-score", price: 200 },
  { path: "/api/agent/social-sentiment", price: 100 },
  { path: "/api/agent/support-resistance", price: 100 },
  { path: "/api/agent/orchestrate", price: 400 },
  { path: "/api/fundamentals", price: 100 },
  { path: "/api/live-price", price: 50 },
]

export const DEFAULT_PRICE = parseInt(process.env.APTOS_PRICE_PER_REQUEST || "100")

export function getPriceForEndpoint(endpoint: string): number {
  const route = PROTECTED_ENDPOINTS.find(r => r.path === endpoint)
  return route?.price ?? DEFAULT_PRICE
}
