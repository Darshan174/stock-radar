import { NextResponse } from "next/server"
import { getX402Status } from "@/lib/x402-startup"

/**
 * Health Check Endpoint
 * 
 * Returns the current status of the Stock-Radar agent including:
 * - Service health
 * - X402 payment configuration
 * - Available endpoints
 */

export async function GET() {
  const x402Status = getX402Status()

  return NextResponse.json({
    status: "healthy",
    service: {
      name: "Stock-Radar Financial Intelligence Agent",
      version: "1.0.0",
      uptime: process.uptime(),
      timestamp: new Date().toISOString(),
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
