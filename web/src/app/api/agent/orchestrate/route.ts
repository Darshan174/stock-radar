import { NextRequest, NextResponse } from "next/server"
import { withX402 } from "@/lib/x402-enforcer"
import { handleMomentumSignal } from "@/app/api/agent/momentum/route"
import { handleSocialSentiment } from "@/app/api/agent/social-sentiment/route"
import { handleFundamentals } from "@/app/api/fundamentals/route"

/**
 * Self-Orchestration Endpoint
 *
 * Runs momentum, fundamentals, and social-sentiment analyses in parallel
 * by calling handler functions directly (no HTTP loopback, no double-payment).
 *
 * Protected at 400 octas — a discount vs 100+100+100 = 300 individual,
 * plus you get weighted aggregation on top.
 *
 * POST { type?: "comprehensive-analysis", symbol: "AAPL" }
 */

async function handleOrchestrate(request: NextRequest): Promise<NextResponse> {
  let body: any
  try {
    body = await request.json()
  } catch {
    return NextResponse.json({ error: "Invalid JSON body" }, { status: 400 })
  }

  const symbol = body.symbol
  if (!symbol || typeof symbol !== "string") {
    return NextResponse.json(
      { error: "symbol is required (e.g. AAPL)" },
      { status: 400 }
    )
  }

  const host = request.headers.get("host") || "localhost:3000"
  const protocol = host.includes("localhost") ? "http" : "https"
  const baseUrl = `${protocol}://${host}`

  // Build fake NextRequest objects pointing at each sub-handler's expected URL
  const momentumUrl = `${baseUrl}/api/agent/momentum?symbol=${encodeURIComponent(symbol)}`
  const sentimentUrl = `${baseUrl}/api/agent/social-sentiment?symbol=${encodeURIComponent(symbol)}`
  const fundamentalsUrl = `${baseUrl}/api/fundamentals?symbol=${encodeURIComponent(symbol)}`

  const makeSubRequest = (url: string) =>
    new NextRequest(url, { method: "GET", headers: request.headers })

  // Run all three in parallel
  const [momentumResult, sentimentResult, fundamentalsResult] =
    await Promise.allSettled([
      handleMomentumSignal(makeSubRequest(momentumUrl)),
      handleSocialSentiment(makeSubRequest(sentimentUrl)),
      handleFundamentals(makeSubRequest(fundamentalsUrl)),
    ])

  // Extract JSON from each settled result
  const extractData = async (
    result: PromiseSettledResult<NextResponse>,
    label: string
  ) => {
    if (result.status === "fulfilled") {
      try {
        const data = await result.value.json()
        if (result.value.status >= 400) {
          return { status: "error" as const, error: data.error || "Unknown error", data: null }
        }
        return { status: "ok" as const, error: null, data }
      } catch {
        return { status: "error" as const, error: `Failed to parse ${label} response`, data: null }
      }
    }
    return {
      status: "error" as const,
      error: result.reason?.message || `${label} failed`,
      data: null,
    }
  }

  const momentum = await extractData(momentumResult, "momentum")
  const sentiment = await extractData(sentimentResult, "sentiment")
  const fundamentals = await extractData(fundamentalsResult, "fundamentals")

  // Weighted scoring: momentum 40%, fundamentals 40%, sentiment 20%
  const momentumScore = momentum.data?.momentum_score ?? 50
  const fundamentalsScore = fundamentals.data?.value_score ?? 50

  // Derive a numeric sentiment score from social sentiment data
  let sentimentScore = 50
  if (sentiment.data) {
    const overall = sentiment.data.social_sentiment?.overall
    if (overall === "bullish") sentimentScore = 75
    else if (overall === "bearish") sentimentScore = 25
    // mentions boost
    const mentions = sentiment.data.reddit?.mentions ?? 0
    if (mentions > 100) sentimentScore = Math.min(sentimentScore + 10, 100)
  }

  const overallScore = Math.round(
    momentumScore * 0.4 + fundamentalsScore * 0.4 + sentimentScore * 0.2
  )

  let signal = "hold"
  if (overallScore >= 70) signal = "strong_buy"
  else if (overallScore >= 55) signal = "buy"
  else if (overallScore <= 30) signal = "strong_sell"
  else if (overallScore <= 45) signal = "sell"

  return NextResponse.json({
    symbol,
    type: body.type || "comprehensive-analysis",
    overall_score: overallScore,
    signal,
    weights: { momentum: 0.4, fundamentals: 0.4, sentiment: 0.2 },
    components: {
      momentum: {
        score: momentumScore,
        status: momentum.status,
        data: momentum.data,
        error: momentum.error,
      },
      fundamentals: {
        score: fundamentalsScore,
        status: fundamentals.status,
        data: fundamentals.data,
        error: fundamentals.error,
      },
      sentiment: {
        score: sentimentScore,
        status: sentiment.status,
        data: sentiment.data,
        error: sentiment.error,
      },
    },
    aggregated_at: new Date().toISOString(),
  })
}

export async function POST(request: NextRequest) {
  return withX402(request, "/api/agent/orchestrate", handleOrchestrate)
}
