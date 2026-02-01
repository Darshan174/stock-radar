import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol, validateNumericParam } from "@/lib/input-validation"

const execAsync = promisify(exec)

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

  const symbol = symbolCheck.value
  const period = periodCheck.value

  const projectPath = path.join(process.cwd(), "..")
  const { stdout, stderr } = await execAsync(
    `cd "${projectPath}" && python3 -c "
import sys, os, json
sys.path.append('src')
from agents.fetcher import StockFetcher
from agents.storage import StockStorage

fetcher = StockFetcher()
storage = StockStorage()

symbol = os.environ['SR_SYMBOL']
quote = fetcher.get_quote(symbol)
if not quote:
    print(json.dumps({'error': 'Quote not found'}))
    sys.exit(1)

stock = storage.get_stock_by_symbol(symbol)
if not stock:
    print(json.dumps({'error': 'Stock not found in database'}))
    sys.exit(1)

stock_id = stock['id']
indicators = storage.get_latest_indicators(stock_id)
if not indicators:
    print(json.dumps({'error': 'Indicators not found'}))
    sys.exit(1)

from agents.scorer import StockScorer
scorer = StockScorer()
momentum_score, breakdown = scorer.calculate_momentum_score(indicators)

result = {
    'symbol': symbol,
    'momentum_score': momentum_score,
    'signal': 'bullish' if momentum_score > 60 else 'bearish' if momentum_score < 40 else 'neutral',
    'breakdown': breakdown,
    'timestamp': quote.timestamp.isoformat()
}
print(json.dumps(result))
"`,
    {
      timeout: 60000,
      env: { ...process.env, SR_SYMBOL: symbol, SR_PERIOD: String(period) },
    }
  )

  if (stderr && stderr.includes("Error")) {
    console.error("Momentum endpoint error:", stderr)
    return NextResponse.json({ error: "Failed to calculate momentum signal" }, { status: 500 })
  }

  const result = JSON.parse(stdout.trim())
  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  return withX402(request, "/api/agent/momentum", handleMomentumSignal)
}
