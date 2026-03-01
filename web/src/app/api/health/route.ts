import { NextRequest, NextResponse } from "next/server"
import { getX402Status } from "@/lib/x402-startup"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"
import { backendRequest } from "@/lib/backend-client"

/**
 * Health Check Endpoint
 *
 * Returns the current status of the Stock-Radar agent including:
 * - Service health
 * - X402 payment configuration
 * - Available endpoints
 */

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  const x402Status = getX402Status()

  let backendHealth: Record<string, unknown> | null = null
  let backendStatus: "healthy" | "degraded" = "healthy"

  try {
    backendHealth = await backendRequest<Record<string, unknown>>("/health", {
      method: "GET",
      timeoutMs: 5000,
    })
  } catch (error) {
    backendStatus = "degraded"
    backendHealth = {
      status: "unreachable",
      detail: error instanceof Error ? error.message : "Unknown error",
    }
  }

  return NextResponse.json({
    status: backendStatus === "healthy" ? "healthy" : "degraded",
    service: {
      name: "Stock-Radar Financial Intelligence Agent",
      version: "1.0.0",
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
    },
    backend: {
      status: backendStatus,
      health: backendHealth,
    },
    x402: {
      ...x402Status,
      status: x402Status.configured ? "active" : "not_configured",
    },
    features: {
      aiAnalysis: true,
      algorithmicScoring: true,
      realTimeData: true,
      paymentRequired: x402Status.configured,
      asyncAnalyzeJobs: true,
    },
    documentation: {
      discovery: "/api/agent/discover",
      paymentInstructions: {
        header: "X-Payment-Tx",
        description: "Include Aptos transaction hash in this header",
        network: x402Status.network.includes("testnet") ? "testnet" :
                 x402Status.network.includes("mainnet") ? "mainnet" : "unknown",
        recipient: x402Status.recipient,
      },
    },
  }, {
    headers: {
      "Cache-Control": "no-cache, no-store, must-revalidate",
    },
  })
}
