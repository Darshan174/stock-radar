import { NextRequest, NextResponse } from "next/server"
import { backendErrorResponse, backendRequest } from "@/lib/backend-client"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol } from "@/lib/input-validation"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

export async function handleFundamentals(request: NextRequest): Promise<NextResponse> {
  const { searchParams } = new URL(request.url)

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const result = await backendRequest<Record<string, unknown>>("/v1/fundamentals", {
    method: "GET",
    timeoutMs: 30000,
    query: { symbol: symbolCheck.value },
  })

  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.paid)
  if (limited) return limited

  try {
    return await withX402(request, "/api/fundamentals", handleFundamentals)
  } catch (error) {
    return backendErrorResponse(error, "Failed to fetch fundamentals")
  }
}
