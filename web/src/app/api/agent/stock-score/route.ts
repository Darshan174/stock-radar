import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol } from "@/lib/input-validation"

const execAsync = promisify(exec)

async function handleStockScore(request: NextRequest): Promise<NextResponse> {
  const searchParams = request.nextUrl.searchParams

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const symbol = symbolCheck.value

  const projectPath = path.join(process.cwd(), "..")
  const { stdout, stderr } = await execAsync(
    `cd "${projectPath}" && python3 -c "
import sys, os, json
sys.path.append('src')
from agents.fetcher import StockFetcher
from agents.storage import StockStorage
from agents.scorer import StockScorer

fetcher = StockFetcher()
storage = StockStorage()
scorer = StockScorer()

symbol = os.environ['SR_SYMBOL']

quote = fetcher.get_quote(symbol)
if not quote:
    print(json.dumps({'error': 'Quote not found'}))
    sys.exit(1)

indicators = storage.get_latest_indicators(symbol)
if not indicators:
    print(json.dumps({'error': 'Indicators not found'}))
    sys.exit(1)

fundamentals = storage.get_fundamentals(symbol)
if not fundamentals:
    fundamentals = {}

all_scores = scorer.calculate_all_scores(indicators, fundamentals)

result = {
    'symbol': symbol,
    'overall_score': all_scores['overall_score'],
    'recommendation': all_scores['recommendation'],
    'confidence': all_scores['confidence'],
    'scores': {
        'momentum': {
            'score': all_scores['momentum_score'],
            'breakdown': all_scores['momentum_breakdown']
        },
        'value': {
            'score': all_scores['value_score'],
            'breakdown': all_scores['value_breakdown']
        },
        'quality': {
            'score': all_scores['quality_score'],
            'breakdown': all_scores['quality_breakdown']
        },
        'risk': {
            'score': all_scores['risk_score'],
            'breakdown': all_scores['risk_breakdown']
        }
    },
    'current_price': quote.price,
    'timestamp': quote.timestamp.isoformat()
}
print(json.dumps(result))
"`,
    {
      timeout: 60000,
      env: { ...process.env, SR_SYMBOL: symbol },
    }
  )

  if (stderr && stderr.includes("Error")) {
    console.error("Stock score endpoint error:", stderr)
    return NextResponse.json({ error: "Failed to calculate stock score" }, { status: 500 })
  }

  const result = JSON.parse(stdout.trim())
  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  return withX402(request, "/api/agent/stock-score", handleStockScore)
}
