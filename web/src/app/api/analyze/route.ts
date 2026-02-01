import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { withX402 } from "@/lib/x402-enforcer"

const execAsync = promisify(exec)

async function handleAnalyze(request: NextRequest): Promise<NextResponse> {
  const { symbol, mode = "intraday", period = "max", fetchFullHistory = true } = await request.json()

  if (!symbol) {
    return NextResponse.json({ error: "Symbol is required" }, { status: 400 })
  }

  const symbolRegex = /^[A-Z0-9.\-^]+$/i
  if (!symbolRegex.test(symbol)) {
    return NextResponse.json({ error: "Invalid symbol format" }, { status: 400 })
  }

  const validPeriods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
  const effectivePeriod = validPeriods.includes(period) ? period : "max"

  const projectPath = path.join(process.cwd(), "..")

  const { stdout, stderr } = await execAsync(
    `cd "${projectPath}" && python3 main.py analyze ${symbol} --mode ${mode} --period ${effectivePeriod}`,
    { timeout: 180000 }
  )

  if (stderr && !stderr.includes("INFO") && !stderr.includes("WARNING")) {
    console.error("Analysis stderr:", stderr)
  }

  const fatalErrorPatterns = [
    /^Error:/im,
    /Analysis failed for/i,
    /Failed to fetch/i,
    /Invalid symbol/i,
    /No data found/i,
  ]

  const hasFatalError = fatalErrorPatterns.some(pattern => pattern.test(stdout))
  const hasSuccessIndicators =
    stdout.includes("Analysis complete") ||
    stdout.includes("stored successfully") ||
    stdout.includes("Saved analysis")

  if (hasFatalError && !hasSuccessIndicators) {
    const errorMatch = stdout.match(/Error[:\s]+(.+)/i)
    return NextResponse.json(
      { error: errorMatch ? errorMatch[1] : "Analysis failed" },
      { status: 500 }
    )
  }

  return NextResponse.json({
    success: true,
    symbol: symbol.toUpperCase(),
    mode,
    message: `Analysis completed for ${symbol}`,
    output: stdout,
  })
}

export async function POST(request: NextRequest) {
  return withX402(request, "/api/analyze", handleAnalyze)
}
