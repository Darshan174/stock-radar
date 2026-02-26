import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import fs from "fs"

const execAsync = promisify(exec)

function loadDotEnvFile(filePath: string): Record<string, string> {
  try {
    const content = fs.readFileSync(filePath, "utf-8")
    const env: Record<string, string> = {}
    for (const rawLine of content.split(/\r?\n/)) {
      const line = rawLine.trim()
      if (!line || line.startsWith("#")) continue
      const eq = line.indexOf("=")
      if (eq <= 0) continue
      const key = line.slice(0, eq).trim()
      let value = line.slice(eq + 1).trim()
      if (
        (value.startsWith('"') && value.endsWith('"')) ||
        (value.startsWith("'") && value.endsWith("'"))
      ) {
        value = value.slice(1, -1)
      }
      if (key) env[key] = value
    }
    return env
  } catch {
    return {}
  }
}

async function runChat(
  question: string,
  symbol?: string | null,
  sessionId?: string | null
) {
  const sanitizedQuestion = String(question).trim().slice(0, 2000)
  const sanitizedSymbol = symbol
    ? symbol.replace(/[^A-Za-z0-9.\-^]/g, "").toUpperCase()
    : ""
  const sanitizedSessionId =
    sessionId && /^[A-Za-z0-9_-]{1,128}$/.test(sessionId) ? sessionId : ""

  const projectPath = path.join(process.cwd(), "..")
  const rootEnv = loadDotEnvFile(path.join(projectPath, ".env"))

  const getEnv = (...keys: string[]) => {
    for (const key of keys) {
      const value = process.env[key] || rootEnv[key]
      if (value) return value
    }
    return ""
  }

  const payload = Buffer.from(
    JSON.stringify({
      question: sanitizedQuestion,
      symbol: sanitizedSymbol || null,
      sessionId: sanitizedSessionId || null,
    }),
    "utf-8"
  ).toString("base64")

  const pythonCode = `
import os
import sys
import json
import base64
from datetime import datetime, timezone

sys.path.insert(0, "${projectPath}/src")
from agents.chat_assistant import StockChatAssistant, ChatMessage

payload = json.loads(base64.b64decode(os.environ["STOCKRADAR_CHAT_PAYLOAD_B64"]).decode("utf-8"))
question = payload.get("question", "")
stock_symbol = payload.get("symbol")
session_id = payload.get("sessionId")

assistant = StockChatAssistant()

if session_id:
    assistant.session_id = session_id
    try:
        history = assistant.storage.get_chat_history(session_id=session_id, limit=40)
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role not in ("user", "assistant", "system") or not content:
                continue
            assistant.conversation_history.append(
                ChatMessage(
                    role=role,
                    content=content,
                    stock_symbols=msg.get("stock_symbols") or [],
                    context_used=msg.get("context_used"),
                    created_at=msg.get("created_at") or datetime.now(timezone.utc).isoformat(),
                )
            )
    except Exception:
        pass
else:
    assistant.start_session()

response = assistant.ask(
    question=question,
    stock_symbol=stock_symbol
)

result = {
    "answer": response.answer,
    "stockSymbols": response.stock_symbols,
    "sourcesUsed": response.sources_used,
    "modelUsed": response.model_used,
    "tokensUsed": response.tokens_used,
    "processingTimeMs": response.processing_time_ms,
    "sessionId": assistant.session_id,
    "contextRetrieved": {
        "totalResults": response.context_retrieved.total_results,
        "sourcesSearched": response.context_retrieved.sources_searched,
        "retrievalTimeMs": response.context_retrieved.retrieval_time_ms
    }
}
print("JSON_RESULT:" + json.dumps(result))
`.trim()

  const pythonEnv: NodeJS.ProcessEnv = {
    ...process.env,
    PYTHONUNBUFFERED: "1",
    STOCKRADAR_CHAT_PAYLOAD_B64: payload,
    SUPABASE_URL: getEnv("SUPABASE_URL", "NEXT_PUBLIC_SUPABASE_URL"),
    SUPABASE_KEY: getEnv(
      "SUPABASE_KEY",
      "SUPABASE_SERVICE_ROLE_KEY",
      "NEXT_PUBLIC_SUPABASE_ANON_KEY"
    ),
    COHERE_API_KEY: getEnv("COHERE_API_KEY"),
    ZAI_API_KEY: getEnv("ZAI_API_KEY"),
    ZAI_API_BASE: getEnv("ZAI_API_BASE"),
    GEMINI_API_KEY: getEnv("GEMINI_API_KEY"),
  }

  const { stdout, stderr } = await execAsync(
    `cd "${projectPath}" && python3 -c '${pythonCode.replace(/'/g, "\\'")}'`,
    {
      timeout: 60000,
      maxBuffer: 5 * 1024 * 1024,
      env: pythonEnv,
    }
  )

  const lines = stdout
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean)
  const jsonLine = [...lines].reverse().find((line) => line.startsWith("JSON_RESULT:"))

  if (!jsonLine) {
    const errText = stderr?.trim() || "Chat assistant returned invalid response"
    throw new Error(errText)
  }

  const parsed = JSON.parse(jsonLine.slice("JSON_RESULT:".length))
  if (parsed?.modelUsed === "error") {
    const detail =
      typeof parsed?.answer === "string" && parsed.answer.trim()
        ? parsed.answer
        : "All chat models failed. Check ZAI or Gemini API keys."
    throw new Error(detail)
  }

  return parsed
}

export async function POST(request: NextRequest) {
  try {
    const { question, symbol, sessionId } = await request.json()

    if (!question) {
      return NextResponse.json({ error: "Question is required" }, { status: 400 })
    }

    const result = await runChat(question, symbol, sessionId)

    return NextResponse.json({
      success: true,
      ...result,
    })
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
    const message = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
