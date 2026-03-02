import { NextResponse } from "next/server"
import type { NextRequest } from "next/server"

/**
 * Next.js Middleware
 * 
 * Enforces payment configuration check for all protected API routes.
 * Returns 503 if the service is not properly configured for payments.
 */

export function middleware(request: NextRequest) {
  const url = request.nextUrl
  const path = url.pathname

  // Check if this is a protected API route
  const protectedPaths = [
    "/api/agent/",
    "/api/fundamentals",
    "/api/live-price",
  ]

  const isProtected = protectedPaths.some(prefix => path.startsWith(prefix))

  if (isProtected) {
    // Allow discovery and message endpoints without payment config
    if (path === "/api/agent/discover" || path === "/api/agent/message") {
      return NextResponse.next()
    }

    const recipient = process.env.APTOS_RECIPIENT_ADDRESS

    // Check if payment is properly configured
    if (!recipient || recipient === "0x1") {
      return NextResponse.json(
        {
          error: "Payment service not configured",
          code: 503,
          message: "This agent requires payment configuration. Please set APTOS_RECIPIENT_ADDRESS.",
          discovery: `${url.protocol}//${url.host}/api/agent/discover`,
        },
        { status: 503 }
      )
    }

    // Inject internal API key for same-origin requests (app's own pages)
    // so they bypass x402 payment enforcement.
    // EXCEPT /api/agent/* endpoints — those are payment-gated by design
    // and the x402-demo page needs to see real 402 responses.
    const isAgentEndpoint = path.startsWith("/api/agent/")
    if (!isAgentEndpoint) {
      const referer = request.headers.get("referer") || ""
      const isSameOrigin = referer.startsWith(`${url.protocol}//${url.host}`)
      const isExternalClient = request.headers.has("X-Payment-Tx")

      if (isSameOrigin && !isExternalClient) {
        const internalKey = process.env.INTERNAL_API_KEY
        if (internalKey) {
          const requestHeaders = new Headers(request.headers)
          requestHeaders.set("X-Internal-Key", internalKey)
          return NextResponse.next({
            request: { headers: requestHeaders },
          })
        }
      }
    }
  }

  // Add CORS headers for agent-to-agent communication
  if (path.startsWith("/api/agent/")) {
    const response = NextResponse.next()

    // Allow cross-origin requests for agent discovery
    response.headers.set("Access-Control-Allow-Origin", "*")
    response.headers.set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
    response.headers.set("Access-Control-Allow-Headers", "Content-Type, X-Payment-Tx")

    return response
  }

  return NextResponse.next()
}

export const config = {
  matcher: [
    "/api/:path*",
  ],
}
