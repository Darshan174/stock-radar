import { NextRequest, NextResponse } from "next/server"
import { backendErrorResponse, backendRequest, BackendProxyError } from "@/lib/backend-client"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

async function runChat(question: string, symbol?: string | null, sessionId?: string | null) {
  const payload = {
    question: String(question || "").trim(),
    symbol: symbol?.trim() || null,
    sessionId: sessionId?.trim() || null,
  }

  if (!payload.question) {
    throw new BackendProxyError("Question is required", 400, "Question is required")
  }

  return backendRequest<{
    answer: string
    stockSymbols: string[]
    sourcesUsed: Array<{
      type: string
      symbol?: string
      headline?: string
      similarity?: number
    }>
    modelUsed: string
    tokensUsed: number
    processingTimeMs: number
    sessionId: string
    contextRetrieved: {
      totalResults: number
      sourcesSearched: string[]
      retrievalTimeMs: number
    }
  }>("/v1/ask", {
    method: "POST",
    timeoutMs: 60000,
    body: payload,
  })
}

export async function POST(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  try {
    const { question, symbol, sessionId } = await request.json()
    const result = await runChat(question, symbol, sessionId)

    return NextResponse.json({
      success: true,
      ...result,
    })
  } catch (error) {
    console.error("Chat assistant error:", error)
    return backendErrorResponse(error, "Failed to process chat request")
  }
}

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  const searchParams = request.nextUrl.searchParams
  const question = searchParams.get("q")
  const symbol = searchParams.get("symbol")
  const sessionId = searchParams.get("sessionId")

  if (!question) {
    return NextResponse.json(
      { error: "Question (q) parameter is required" },
      { status: 400 }
    )
  }

  try {
    const result = await runChat(question, symbol, sessionId)
    return NextResponse.json({
      success: true,
      ...result,
    })
  } catch (error) {
    console.error("Chat assistant error:", error)
    return backendErrorResponse(error, "Failed to process chat request")
  }
}
