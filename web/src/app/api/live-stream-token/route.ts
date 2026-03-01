import { NextRequest, NextResponse } from "next/server"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  const token =
    process.env.FINNHUB_API_KEY ||
    process.env.NEXT_PUBLIC_FINNHUB_API_KEY ||
    process.env.NEXT_PUBLIC_FINNHUB_KEY

  if (!token) {
    return NextResponse.json(
      {
        enabled: false,
        provider: "none",
      },
      {
        status: 200,
        headers: {
          "Cache-Control": "no-store",
        },
      }
    )
  }

  return NextResponse.json(
    {
      enabled: true,
      provider: "finnhub",
      wsUrl: `wss://ws.finnhub.io?token=${token}`,
    },
    {
      headers: {
        "Cache-Control": "no-store",
      },
    }
  )
}
