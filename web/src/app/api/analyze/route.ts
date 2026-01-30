import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"

const execAsync = promisify(exec)

export async function POST(request: NextRequest) {
  try {
    const { symbol, mode = "intraday", period = "max", fetchFullHistory = true } = await request.json()

    if (!symbol) {
      return NextResponse.json({ error: "Symbol is required" }, { status: 400 })
    }

    // Validate symbol format (allow letters, numbers, dots, dashes for crypto/futures)
    const symbolRegex = /^[A-Z0-9.\-^]+$/i
    if (!symbolRegex.test(symbol)) {
      return NextResponse.json({ error: "Invalid symbol format" }, { status: 400 })
    }

    // Validate period
    const validPeriods = ["1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"]
    const effectivePeriod = validPeriods.includes(period) ? period : "max"

    // Path to the stock-radar project
    const projectPath = path.join(process.cwd(), "..")

    // Run the analysis command with period parameter
    // Use "max" by default to fetch full historical data
    const { stdout, stderr } = await execAsync(
      `cd "${projectPath}" && python3 main.py analyze ${symbol} --mode ${mode} --period ${effectivePeriod}`,
      { timeout: 180000 } // 3 minute timeout for full history fetch
    )

    // Check for errors in output
    if (stderr && !stderr.includes("INFO") && !stderr.includes("WARNING")) {
      console.error("Analysis stderr:", stderr)
    }

    // Parse the output to check for ACTUAL failures (not just log warnings)
    // Look for specific fatal error patterns, not casual mentions of "error" in logs
    const fatalErrorPatterns = [
      /^Error:/im,                           // Lines starting with "Error:"
      /Analysis failed for/i,                // Explicit analysis failure message
      /Failed to fetch/i,                    // Network failures
      /Invalid symbol/i,                     // Invalid symbol
      /No data found/i,                      // No data available
    ]

    const hasFatalError = fatalErrorPatterns.some(pattern => pattern.test(stdout))

    // Also check if the process reported success - look for success indicators
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
  } catch (error) {
    console.error("Analysis error:", error)
    const message = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
