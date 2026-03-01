import { NextRequest, NextResponse } from "next/server"
import { backendErrorResponse, backendRequest } from "@/lib/backend-client"
import type { AnalyzeJobCreated } from "@/lib/analyze-contracts"
import { withX402 } from "@/lib/x402-enforcer"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"
import { validateSymbol } from "@/lib/input-validation"

async function handleAnalyze(request: NextRequest): Promise<NextResponse> {
  const { symbol, mode = "intraday", period = "max" } = await request.json()

  const symbolCheck = validateSymbol(symbol)
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const allowedModes = new Set(["intraday", "longterm"])
  if (!allowedModes.has(mode)) {
    return NextResponse.json({ error: "Invalid mode" }, { status: 400 })
  }

  const validPeriods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
  const effectivePeriod = validPeriods.includes(period) ? period : "max"

  const created = await backendRequest<AnalyzeJobCreated>("/v1/analyze/jobs", {
    method: "POST",
    timeoutMs: 5000,
    body: {
      symbol: symbolCheck.value,
      mode,
      period: effectivePeriod,
    },
  })

  return NextResponse.json({
    jobId: created.jobId,
    statusUrl: `/api/analyze/status?jobId=${encodeURIComponent(created.jobId)}`,
    status: created.status,
  }, { status: 202 })
}

export async function POST(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.analyzeJobs)
  if (limited) return limited

  try {
    return await withX402(request, "/api/analyze", handleAnalyze)
  } catch (error) {
    console.error("Analyze route error:", error)
    return backendErrorResponse(error, "Failed to submit analysis job")
  }
}
