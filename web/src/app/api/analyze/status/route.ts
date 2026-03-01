import { NextRequest, NextResponse } from "next/server"
import { backendErrorResponse, backendRequest } from "@/lib/backend-client"
import type { AnalyzeJobStatus } from "@/lib/analyze-contracts"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  const jobId = request.nextUrl.searchParams.get("jobId")
  if (!jobId) {
    return NextResponse.json({ error: "jobId is required" }, { status: 400 })
  }

  try {
    const statusPayload = await backendRequest<AnalyzeJobStatus>(
      `/v1/analyze/jobs/${encodeURIComponent(jobId)}`,
      {
        method: "GET",
        timeoutMs: 10000,
      },
    )

    return NextResponse.json(statusPayload)
  } catch (error) {
    return backendErrorResponse(error, "Failed to fetch analysis job status")
  }
}
