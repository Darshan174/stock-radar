import { NextRequest, NextResponse } from "next/server"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  const symbols = request.nextUrl.searchParams.get("symbols")
  if (!symbols) {
    return NextResponse.json({ error: "symbols query param is required" }, { status: 400 })
  }

  const list = symbols
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 50)

  if (list.length === 0) {
    return NextResponse.json({ quotes: {} })
  }

  const quotes: Record<string, { latest_price: number; price_change?: number } | null> = {}

  // Fetch all in parallel
  const results = await Promise.allSettled(
    list.map(async (symbol) => {
      const yahooUrl = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=5m&range=1d&includePrePost=false`
      const res = await fetch(yahooUrl, {
        headers: {
          "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        },
        next: { revalidate: 30 },
      })

      if (!res.ok) return { symbol, data: null }

      const json = await res.json()
      const result = json?.chart?.result?.[0]
      if (!result) return { symbol, data: null }

      const quote = result.indicators?.quote?.[0]
      const timestamps = result.timestamp || []

      if (!quote || timestamps.length === 0) return { symbol, data: null }

      // Find last valid close
      let latestPrice: number | undefined
      for (let i = timestamps.length - 1; i >= 0; i--) {
        const c = quote.close?.[i]
        if (typeof c === "number" && Number.isFinite(c)) {
          latestPrice = parseFloat(c.toFixed(2))
          break
        }
      }

      if (latestPrice === undefined) return { symbol, data: null }

      const previousClose = result.meta?.previousClose ?? result.meta?.chartPreviousClose
      let priceChange: number | undefined
      if (typeof previousClose === "number" && previousClose > 0) {
        priceChange = ((latestPrice - previousClose) / previousClose) * 100
      }

      return { symbol, data: { latest_price: latestPrice, price_change: priceChange } }
    }),
  )

  for (const r of results) {
    if (r.status === "fulfilled" && r.value) {
      quotes[r.value.symbol] = r.value.data
    }
  }

  return NextResponse.json({ quotes })
}
