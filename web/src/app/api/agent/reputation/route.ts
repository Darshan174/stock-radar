import { NextRequest, NextResponse } from "next/server"
import { agentRegistry } from "@/lib/agent-registry"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

const AGENT_ADDRESS =
  process.env.AGENT_REGISTRY_ADDRESS ||
  "0x7f10a07e484263ee7f4debd27a8adac2b918b7f3969ee79d3b6da636c3666240"

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  try {
    // Fetch full agent info (includes identity + reputation)
    const agentInfo = await agentRegistry.getAgentInfo(AGENT_ADDRESS)

    if (!agentInfo) {
      // Fallback: try to get just reputation (legacy flow)
      const reputation = await agentRegistry.getReputation(AGENT_ADDRESS)

      if (!reputation) {
        return NextResponse.json(
          { error: "Agent not registered on-chain" },
          { status: 404 }
        )
      }

      const completionRate =
        reputation.totalRequests > 0
          ? ((reputation.successfulRequests / reputation.totalRequests) * 100).toFixed(1)
          : "0.0"

      return NextResponse.json({
        address: AGENT_ADDRESS,
        registered: false,
        totalRequests: reputation.totalRequests,
        successfulRequests: reputation.successfulRequests,
        failedRequests: reputation.failedRequests,
        totalEarned: reputation.totalEarned,
        totalEarnedAPT: (reputation.totalEarned / 100_000_000).toFixed(8),
        completionRate,
        updatedAt: reputation.updatedAt,
        explorerUrl: `https://explorer.aptoslabs.com/account/${AGENT_ADDRESS}?network=testnet`,
      })
    }

    // Full agent info available — agent is formally registered
    const completionRate =
      agentInfo.totalRequests > 0
        ? ((agentInfo.successfulRequests / agentInfo.totalRequests) * 100).toFixed(1)
        : "100.0"

    return NextResponse.json({
      // Agent Identity (the "agent economy" narrative)
      address: AGENT_ADDRESS,
      registered: true,
      endpointUrl: agentInfo.endpointUrl,
      capabilities: agentInfo.capabilities.length,
      capabilityList: agentInfo.capabilities.map(c => ({
        name: c.name,
        price: c.price,
        description: c.description,
      })),
      isActive: agentInfo.isActive,

      // Reputation stats
      totalRequests: agentInfo.totalRequests,
      successfulRequests: agentInfo.successfulRequests,
      failedRequests: agentInfo.failedRequests,
      totalEarned: agentInfo.totalRevenue,
      totalEarnedAPT: (agentInfo.totalRevenue / 100_000_000).toFixed(8),
      completionRate,

      // Timestamps
      createdAt: agentInfo.createdAt,
      updatedAt: agentInfo.updatedAt,

      // Explorer link
      explorerUrl: `https://explorer.aptoslabs.com/account/${AGENT_ADDRESS}?network=testnet`,
    })
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : "Unknown error"
    console.error("[reputation] Error:", message)
    return NextResponse.json(
      { error: "Failed to fetch on-chain agent data" },
      { status: 500 }
    )
  }
}
