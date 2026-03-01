import { NextRequest, NextResponse } from "next/server"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

const PERIOD_CONFIG: Record<
    string,
    { range?: string; interval: string; usePeriod?: boolean; days?: number }
> = {
    "1d": { range: "1d", interval: "1m" },
    "1w": { range: "5d", interval: "5m" },
    "1m": { range: "1mo", interval: "15m" },
    "3m": { range: "3mo", interval: "1h" },
    "6m": { range: "6mo", interval: "1d" },
    "1y": { range: "1y", interval: "1d" },
    "3y": { usePeriod: true, days: 1095, interval: "1wk" },
    "5y": { range: "5y", interval: "1wk" },
    all: { range: "max", interval: "1mo" },
}

export async function GET(request: NextRequest) {
    const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
    if (limited) return limited

    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get("symbol")
    const period = searchParams.get("period") || "3m"

    if (!symbol) {
        return NextResponse.json({ error: "Symbol is required" }, { status: 400 })
    }

    const config = PERIOD_CONFIG[period]
    if (!config) {
        return NextResponse.json({ error: "Invalid period" }, { status: 400 })
    }

    try {
        let yahooUrl: string
        if (config.usePeriod && config.days) {
            const now = Math.floor(Date.now() / 1000)
            const period1 = now - config.days * 86400
            yahooUrl = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${period1}&period2=${now}&interval=${config.interval}`
        } else {
            yahooUrl = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?range=${config.range}&interval=${config.interval}`
        }

        const response = await fetch(yahooUrl, {
            headers: {
                "User-Agent":
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
            next: { revalidate: period === "1d" || period === "1w" ? 5 : 120 },
        })

        if (!response.ok) {
            throw new Error(`Yahoo Finance returned ${response.status}`)
        }

        const data = await response.json()
        const result = data?.chart?.result?.[0]

        if (!result) {
            return NextResponse.json({ error: "No data found" }, { status: 404 })
        }

        const timestamps = result.timestamp || []
        const quote = result.indicators?.quote?.[0] || {}
        const isIntraday = !["1d", "1wk", "1mo"].includes(config.interval)

        const candles = timestamps
            .map((ts: number, i: number) => {
                const open = quote.open?.[i]
                const high = quote.high?.[i]
                const low = quote.low?.[i]
                const close = quote.close?.[i]
                const volume = quote.volume?.[i]

                if (open == null || high == null || low == null || close == null) {
                    return null
                }

                return {
                    // Always use unix seconds to keep axis/crosshair formatting consistent.
                    time: ts,
                    open: parseFloat(open.toFixed(2)),
                    high: parseFloat(high.toFixed(2)),
                    low: parseFloat(low.toFixed(2)),
                    close: parseFloat(close.toFixed(2)),
                    volume: volume || 0,
                }
            })
            .filter(Boolean)

        return NextResponse.json({
            symbol: result.meta?.symbol || symbol,
            period,
            interval: config.interval,
            isIntraday,
            candles,
            meta: {
                currency: result.meta?.currency || "USD",
                regularMarketPrice: result.meta?.regularMarketPrice,
                previousClose: result.meta?.previousClose || result.meta?.chartPreviousClose,
                exchangeTimezoneName: result.meta?.exchangeTimezoneName || "UTC",
            },
        })
    } catch (error) {
        console.error("Chart data fetch error:", error)
        return NextResponse.json(
            { error: "Failed to fetch chart data" },
            { status: 500 }
        )
    }
}
