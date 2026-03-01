import { NextRequest, NextResponse } from "next/server"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol, validateNumericParam } from "@/lib/input-validation"
import { backendErrorResponse, backendRequest } from "@/lib/backend-client"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

async function handleNewsImpact(request: NextRequest): Promise<NextResponse> {
  const searchParams = request.nextUrl.searchParams

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const daysCheck = validateNumericParam(searchParams.get("days"), "7", "days", 1, 90)
  if (!daysCheck.valid) {
    return NextResponse.json({ error: daysCheck.error }, { status: 400 })
  }

  const result = await backendRequest<Record<string, unknown>>("/v1/agent/news-impact", {
    method: "GET",
    timeoutMs: 30000,
    query: {
      symbol: symbolCheck.value,
      days: daysCheck.value,
    },
  })

  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.paid)
  if (limited) return limited

  try {
    return await withX402(request, "/api/agent/news-impact", handleNewsImpact)
  } catch (error) {
    return backendErrorResponse(error, "Failed to generate news impact summary")
  }
}
