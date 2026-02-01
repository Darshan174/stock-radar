import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol, validateNumericParam } from "@/lib/input-validation"

const execAsync = promisify(exec)

async function handleSupportResistance(request: NextRequest): Promise<NextResponse> {
  const searchParams = request.nextUrl.searchParams

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const periodCheck = validateNumericParam(searchParams.get("period"), "14", "period", 1, 365)
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
from datetime import datetime

fetcher = StockFetcher()
storage = StockStorage()

symbol = os.environ['SR_SYMBOL']
period = int(os.environ['SR_PERIOD'])

df = storage.get_historical_data(symbol, period + 10)
if df is None or len(df) < 2:
    print(json.dumps({'error': 'Insufficient historical data'}))
    sys.exit(1)

df = df.tail(period)

high = df['high'].iloc[-1]
low = df['low'].iloc[-1]
close = df['close'].iloc[-1]

pivot = (high + low + close) / 3
r1 = 2 * pivot - low
r2 = pivot + (high - low)
r3 = high + 2 * (pivot - low)
s1 = 2 * pivot - high
s2 = pivot - (high - low)
s3 = low - 2 * (high - pivot)

sma_period = 20
sma_20 = df['close'].rolling(sma_period).mean().iloc[-1] if len(df) >= sma_period else None

atr_period = 14
tr_values = []
for i in range(1, min(atr_period + 1, len(df))):
    h = df['high'].iloc[-i]
    l = df['low'].iloc[-i]
    c_prev = df['close'].iloc[-i-1]
    tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
    tr_values.append(tr)
atr = sum(tr_values) / len(tr_values) if tr_values else None

upper_band = close + (2 * atr) if atr else None
lower_band = close - (2 * atr) if atr else None

result = {
    'symbol': symbol,
    'support_resistance': {
        'pivot_points': {
            'pivot': round(pivot, 2),
            'resistance_1': round(r1, 2),
            'resistance_2': round(r2, 2),
            'resistance_3': round(r3, 2),
            'support_1': round(s1, 2),
            'support_2': round(s2, 2),
            'support_3': round(s3, 2)
        },
        'bollinger_bands': {
            'upper': round(upper_band, 2) if upper_band else None,
            'middle': round(sma_20, 2) if sma_20 else None,
            'lower': round(lower_band, 2) if lower_band else None
        },
        'atr': {
            'value': round(atr, 2) if atr else None,
            'period': atr_period
        }
    },
    'current_price': round(close, 2),
    'price_position': 'above_pivot' if close > pivot else 'below_pivot',
    'timestamp': datetime.utcnow().isoformat()
}
print(json.dumps(result))
"`,
    {
      timeout: 60000,
      env: { ...process.env, SR_SYMBOL: symbol, SR_PERIOD: String(period) },
    }
  )

  if (stderr && stderr.includes("Error")) {
    console.error("Support/Resistance endpoint error:", stderr)
    return NextResponse.json({ error: "Failed to calculate support/resistance levels" }, { status: 500 })
  }

  const result = JSON.parse(stdout.trim())
  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  return withX402(request, "/api/agent/support-resistance", handleSupportResistance)
}
