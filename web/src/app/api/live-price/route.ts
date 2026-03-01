import { NextRequest, NextResponse } from "next/server"
import { withX402 } from "@/lib/x402-enforcer"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

async function handleLivePrice(request: NextRequest): Promise<NextResponse> {
  const { searchParams } = new URL(request.url)
  const symbol = searchParams.get("symbol")

  if (!symbol) {
    return NextResponse.json({ error: "Symbol is required" }, { status: 400 })
  }

  const yahooUrl = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=1m&range=1d`

  const response = await fetch(yahooUrl, {
    headers: {
      "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    },
    cache: "no-store",
  })

  if (!response.ok) {
    throw new Error(`Yahoo Finance returned ${response.status}`)
  }

  const data = await response.json()
  const quote = data?.chart?.result?.[0]

  if (!quote) {
    return NextResponse.json({ error: "No data found" }, { status: 404 })
  }

  const meta = quote.meta
  const currentPrice = meta.regularMarketPrice
  const previousClose = meta.previousClose || meta.chartPreviousClose
  const change = currentPrice - previousClose
  const changePercent = previousClose > 0 ? (change / previousClose) * 100 : 0

  return NextResponse.json({
    symbol: meta.symbol,
    price: currentPrice,
    change: parseFloat(change.toFixed(2)),
    changePercent: parseFloat(changePercent.toFixed(2)),
    volume: meta.regularMarketVolume || 0,
    high: meta.regularMarketDayHigh,
    low: meta.regularMarketDayLow,
    open: meta.regularMarketOpen,
    previousClose: previousClose,
    timestamp: new Date().toISOString(),
  })
}

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.paid)
  if (limited) return limited

  return withX402(request, "/api/live-price", handleLivePrice)
}
