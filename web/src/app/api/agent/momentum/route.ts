import { NextRequest, NextResponse } from "next/server"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol, validateNumericParam } from "@/lib/input-validation"
import { backendErrorResponse, backendRequest } from "@/lib/backend-client"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

export async function handleMomentumSignal(request: NextRequest): Promise<NextResponse> {
  const searchParams = request.nextUrl.searchParams

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const periodCheck = validateNumericParam(searchParams.get("period"), "14", "period", 1, 200)
  if (!periodCheck.valid) {
    return NextResponse.json({ error: periodCheck.error }, { status: 400 })
  }

  const result = await backendRequest<Record<string, unknown>>("/v1/agent/momentum", {
    method: "GET",
    timeoutMs: 30000,
    query: {
      symbol: symbolCheck.value,
      period: periodCheck.value,
    },
  })

  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.paid)
  if (limited) return limited

  try {
    return await withX402(request, "/api/agent/momentum", handleMomentumSignal)
  } catch (error) {
    return backendErrorResponse(error, "Failed to calculate momentum signal")
  }
}
