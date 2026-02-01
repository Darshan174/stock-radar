import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol, validateNumericParam } from "@/lib/input-validation"

const execAsync = promisify(exec)

async function handleRSIDivergence(request: NextRequest): Promise<NextResponse> {
  const searchParams = request.nextUrl.searchParams

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const periodCheck = validateNumericParam(searchParams.get("period"), "14", "period", 1, 200)
  if (!periodCheck.valid) {
    return NextResponse.json({ error: periodCheck.error }, { status: 400 })
  }

  const lookbackCheck = validateNumericParam(searchParams.get("lookback"), "5", "lookback", 1, 100)
  if (!lookbackCheck.valid) {
    return NextResponse.json({ error: lookbackCheck.error }, { status: 400 })
  }

  const symbol = symbolCheck.value
  const period = periodCheck.value
  const lookback = lookbackCheck.value

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
lookback = int(os.environ['SR_LOOKBACK'])
rsi_period = int(os.environ['SR_PERIOD'])

df = storage.get_historical_data(symbol, lookback + 10)
if df is None or len(df) < lookback + 2:
    print(json.dumps({'error': 'Insufficient historical data'}))
    sys.exit(1)

rsi_values = []
for i in range(len(df) - rsi_period + 1):
    window = df['close'].iloc[i:i+rsi_period]
    gains = []
    losses = []
    for j in range(1, len(window)):
        change = window.iloc[j] - window.iloc[j-1]
        gains.append(max(0, change))
        losses.append(abs(min(0, change)))
    avg_gain = sum(gains[-rsi_period:]) / rsi_period if gains else 0
    avg_loss = sum(losses[-rsi_period:]) / rsi_period if losses else 0
    if avg_loss == 0:
        rs = 100
    else:
        rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi_values.append(rsi)

prices = df['close'].values[-lookback:]
rsi_values = rsi_values[-lookback:]

price_high_idx = prices.argmax()
price_low_idx = prices.argmin()

rsi_at_price_high = rsi_values[price_high_idx]
rsi_at_price_low = rsi_values[price_low_idx]

rsi_high_idx = max(range(len(rsi_values)), key=lambda i: rsi_values[i])
rsi_low_idx = min(range(len(rsi_values)), key=lambda i: rsi_values[i])

price_at_rsi_high = prices[rsi_high_idx]
price_at_rsi_low = prices[rsi_low_idx]

bearish_divergence = False
bullish_divergence = False

if price_high_idx > 0:
    prev_high_idx = prices[:price_high_idx].argmax()
    if prev_high_idx >= 0:
        if prices[price_high_idx] > prices[prev_high_idx]:
            if rsi_values[price_high_idx] < rsi_values[prev_high_idx]:
                bearish_divergence = True

if price_low_idx < len(prices) - 1:
    prev_low_idx = len(prices) - 1 - prices[price_low_idx+1:][::-1].argmin()
    if prev_low_idx < len(prices):
        if prices[price_low_idx] < prices[prev_low_idx]:
            if rsi_values[price_low_idx] > rsi_values[prev_low_idx]:
                bullish_divergence = True

signal = 'neutral'
confidence = 0
if bearish_divergence:
    signal = 'bearish'
    confidence = 70
elif bullish_divergence:
    signal = 'bullish'
    confidence = 70

result = {
    'symbol': symbol,
    'rsi_divergence': {
        'signal': signal,
        'confidence': confidence,
        'bearish_divergence': bearish_divergence,
        'bullish_divergence': bullish_divergence,
        'current_rsi': round(rsi_values[-1], 2),
        'rsi_trend': 'up' if rsi_values[-1] > rsi_values[-2] else 'down',
        'price_trend': 'up' if prices[-1] > prices[-2] else 'down',
        'lookback_period': lookback,
        'data_points': len(prices)
    },
    'timestamp': df['timestamp'].iloc[-1].isoformat()
}
print(json.dumps(result))
"`,
    {
      timeout: 60000,
      env: {
        ...process.env,
        SR_SYMBOL: symbol,
        SR_PERIOD: String(period),
        SR_LOOKBACK: String(lookback),
      },
    }
  )

  if (stderr && stderr.includes("Error")) {
    console.error("RSI divergence endpoint error:", stderr)
    return NextResponse.json({ error: "Failed to calculate RSI divergence" }, { status: 500 })
  }

  const result = JSON.parse(stdout.trim())
  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  return withX402(request, "/api/agent/rsi-divergence", handleRSIDivergence)
}
