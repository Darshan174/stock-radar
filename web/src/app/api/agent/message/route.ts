import { NextRequest, NextResponse } from "next/server"
import { PROTECTED_ENDPOINTS, getPriceForEndpoint } from "@/lib/x402-config"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

/**
 * Agent Messaging Endpoint (XMTP-style over HTTP)
 *
 * Public endpoint (no payment required) — serves as the negotiation channel.
 * Agents send messages following the agent-xmtp-v1 protocol to discover
 * capabilities, inquire about pricing, and request tasks.
 *
 * POST { protocol: "agent-xmtp-v1", message_type: "...", payload: {...} }
 */

const PROTOCOL_VERSION = "agent-xmtp-v1"

export async function POST(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  let body: any
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 })
  }

  if (body.protocol !== PROTOCOL_VERSION) {
    return NextResponse.json(
      {
        error: "Unsupported protocol",
        expected: PROTOCOL_VERSION,
        received: body.protocol,
      },
      { status: 400 }
    )
  }

  const host = request.headers.get("host") || "localhost:3000"
  const protocol = host.includes("localhost") ? "http" : "https"
  const baseUrl = `${protocol}://${host}`

  switch (body.message_type) {
    case "capability_discovery":
      return handleCapabilityDiscovery(baseUrl, body.payload)

    case "pricing_inquiry":
      return handlePricingInquiry(baseUrl, body.payload)

    case "task_request":
      return handleTaskRequest(baseUrl, body.payload)

    default:
      return NextResponse.json(
        {
          error: "Unknown message_type",
          supported: ["capability_discovery", "pricing_inquiry", "task_request"],
          received: body.message_type,
        },
        { status: 400 }
      )
  }
}

function handleCapabilityDiscovery(baseUrl: string, payload: any) {
  const capabilities = PROTECTED_ENDPOINTS.map((ep) => {
    const name = ep.path.split("/").pop() || "unknown"
    return {
      name,
      endpoint: `${baseUrl}${ep.path}`,
      price: ep.price ?? 100,
      currency: "APT",
    }
  })

  return NextResponse.json({
    protocol: PROTOCOL_VERSION,
    message_type: "offer",
    timestamp: Date.now(),
    payload: {
      capabilities,
      bulk_discount: {
        endpoint: `${baseUrl}/api/agent/orchestrate`,
        price: 400,
        note: "Comprehensive analysis (momentum + fundamentals + sentiment) at a discount vs individual calls",
      },
      valid_until: Date.now() + 86400000, // 24 hours
    },
  })
}

function handlePricingInquiry(baseUrl: string, payload: any) {
  const capability = payload?.capability
  if (!capability) {
    return NextResponse.json(
      { error: "payload.capability is required" },
      { status: 400 }
    )
  }

  // Find the matching endpoint
  const endpoint = PROTECTED_ENDPOINTS.find((ep) =>
    ep.path.endsWith(`/${capability}`)
  )

  if (!endpoint) {
    return NextResponse.json(
      {
        error: "Unknown capability",
        available: PROTECTED_ENDPOINTS.map((ep) => ep.path.split("/").pop()),
      },
      { status: 404 }
    )
  }

  return NextResponse.json({
    protocol: PROTOCOL_VERSION,
    message_type: "pricing_response",
    timestamp: Date.now(),
    payload: {
      capability,
      endpoint: `${baseUrl}${endpoint.path}`,
      price: endpoint.price ?? 100,
      currency: "APT",
      network: process.env.APTOS_NETWORK?.includes("mainnet")
        ? "mainnet"
        : "testnet",
      volume_discount:
        payload?.estimated_volume && payload.estimated_volume >= 10
          ? { discount_percent: 15, note: "Use /api/agent/orchestrate for bundled analysis" }
          : null,
    },
  })
}

function handleTaskRequest(baseUrl: string, payload: any) {
  const capability = payload?.capability
  const maxPrice = payload?.max_price

  if (!capability) {
    return NextResponse.json(
      { error: "payload.capability is required" },
      { status: 400 }
    )
  }

  // Find the matching endpoint
  const endpoint = PROTECTED_ENDPOINTS.find((ep) =>
    ep.path.endsWith(`/${capability}`)
  )

  if (!endpoint) {
    return NextResponse.json(
      { error: "Unknown capability", capability },
      { status: 404 }
    )
  }

  const price = endpoint.price ?? 100

  // Validate budget
  if (maxPrice !== undefined && maxPrice < price) {
    return NextResponse.json({
      protocol: PROTOCOL_VERSION,
      message_type: "task_rejection",
      timestamp: Date.now(),
      payload: {
        task_id: payload.task_id || null,
        reason: "budget_insufficient",
        required_price: price,
        offered_price: maxPrice,
        currency: "APT",
      },
    })
  }

  return NextResponse.json({
    protocol: PROTOCOL_VERSION,
    message_type: "task_acceptance",
    timestamp: Date.now(),
    payload: {
      task_id: payload.task_id || `task-${Date.now()}`,
      capability,
      agreed_price: price,
      currency: "APT",
      endpoint: `${baseUrl}${endpoint.path}`,
      payment_address: process.env.APTOS_RECIPIENT_ADDRESS || "0x1",
      payment_instructions: {
        step1: `Send ${price} octas APT to the payment_address`,
        step2: "Include transaction hash in X-Payment-Tx header",
        step3: `POST/GET to ${baseUrl}${endpoint.path} with your parameters`,
      },
      estimated_completion_ms: 5000,
    },
  })
}
