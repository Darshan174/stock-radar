import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"

const execAsync = promisify(exec)

export async function POST(request: NextRequest) {
  try {
    const { question, symbol, sessionId } = await request.json()

    if (!question) {
      return NextResponse.json(
        { error: "Question is required" },
        { status: 400 }
      )
    }

    // Sanitize inputs
    const sanitizedQuestion = question
      .replace(/['"\\]/g, "") // Remove quotes and backslashes
      .substring(0, 1000) // Limit length

    const sanitizedSymbol = symbol
      ? symbol.replace(/[^A-Za-z0-9.\-^]/g, "").toUpperCase()
      : ""

    // Path to the stock-radar project
    const projectPath = path.join(process.cwd(), "..")

    // Build Python command to run chat assistant
    const symbolArg = sanitizedSymbol ? `--symbol "${sanitizedSymbol}"` : ""
    const pythonCode = `
import sys
import json
sys.path.insert(0, "${projectPath}/src")
from agents.chat_assistant import StockChatAssistant

assistant = StockChatAssistant()
assistant.start_session()
response = assistant.ask(
    question="""${sanitizedQuestion}""",
    stock_symbol=${sanitizedSymbol ? `"${sanitizedSymbol}"` : "None"}
)

result = {
    "answer": response.answer,
    "stockSymbols": response.stock_symbols,
    "sourcesUsed": response.sources_used,
    "modelUsed": response.model_used,
    "tokensUsed": response.tokens_used,
    "processingTimeMs": response.processing_time_ms,
    "contextRetrieved": {
        "totalResults": response.context_retrieved.total_results,
        "sourcesSearched": response.context_retrieved.sources_searched,
        "retrievalTimeMs": response.context_retrieved.retrieval_time_ms
    }
}
print(json.dumps(result))
`.trim()

    const { stdout, stderr } = await execAsync(
      `cd "${projectPath}" && python3 -c '${pythonCode.replace(/'/g, "\\'")}'`,
      { timeout: 60000 } // 1 minute timeout
    )

    // Parse the JSON output
    try {
      const result = JSON.parse(stdout.trim())
      return NextResponse.json({
        success: true,
        ...result,
      })
    } catch {
      // If JSON parsing fails, return the raw output
      return NextResponse.json({
        success: true,
        answer: stdout.trim(),
        stockSymbols: [],
        sourcesUsed: [],
        modelUsed: "unknown",
        tokensUsed: 0,
        processingTimeMs: 0,
      })
    }
  } catch (error) {
    console.error("Chat assistant error:", error)
    const message = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json({ error: message }, { status: 500 })
  }
}

// Also support GET for simple questions
export async function GET(request: NextRequest) {
  const searchParams = request.nextUrl.searchParams
  const question = searchParams.get("q")
  const symbol = searchParams.get("symbol")

  if (!question) {
    return NextResponse.json(
      { error: "Question (q) parameter is required" },
      { status: 400 }
    )
  }

  // Convert to POST request
  const postRequest = new NextRequest(request.url, {
    method: "POST",
    body: JSON.stringify({ question, symbol }),
  })

  return POST(postRequest)
}
