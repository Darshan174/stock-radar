import { NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest) {
    const { searchParams } = new URL(request.url)
    const symbol = searchParams.get("symbol")
    const interval = searchParams.get("interval") || "5m" // 1m, 5m, 15m, 1h, 1d
    const range = searchParams.get("range") || "1d" // 1d, 5d, 1mo

    if (!symbol) {
        return NextResponse.json({ error: "Symbol is required" }, { status: 400 })
    }

    try {
        // Fetch intraday data for specified interval and range from Yahoo Finance
        const yahooUrl = `https://query2.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?interval=${interval}&range=${range}&includePrePost=false`

        const response = await fetch(yahooUrl, {
            headers: {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            },
            next: { revalidate: 30 }, // Cache for 30 seconds for intraday
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

        // Convert to OHLCV format with Unix timestamps for time axis
        const candles = timestamps.map((ts: number, i: number) => {
            const open = quote.open?.[i]
            const high = quote.high?.[i]
            const low = quote.low?.[i]
            const close = quote.close?.[i]
            const volume = quote.volume?.[i]

            // Skip invalid data points
            if (open == null || high == null || low == null || close == null) {
                return null
            }

            return {
                time: ts, // Unix timestamp for lightweight-charts
                open: parseFloat(open.toFixed(2)),
                high: parseFloat(high.toFixed(2)),
                low: parseFloat(low.toFixed(2)),
                close: parseFloat(close.toFixed(2)),
                volume: volume || 0,
            }
        }).filter(Boolean)

        // Get today's aggregated data for mixing with historical
        let todayCandle = null
        if (candles.length > 0 && (interval === "1m" || interval === "5m" || interval === "15m")) {
            const todayOpen = candles[0].open
            const todayHigh = Math.max(...candles.map((c: any) => c.high))
            const todayLow = Math.min(...candles.map((c: any) => c.low))
            const todayClose = candles[candles.length - 1].close
            const todayVolume = candles.reduce((acc: number, c: any) => acc + c.volume, 0)

            // Get date string for the daily candle
            const firstDate = new Date(candles[0].time * 1000)
            const dateStr = firstDate.toISOString().split("T")[0]

            todayCandle = {
                time: dateStr,
                open: todayOpen,
                high: todayHigh,
                low: todayLow,
                close: todayClose,
                volume: todayVolume,
            }
        }

        return NextResponse.json({
            symbol: result.meta?.symbol || symbol,
            interval,
            range,
            candles,
            todayCandle,
            meta: {
                currency: result.meta?.currency || "USD",
                exchangeTimezoneName: result.meta?.exchangeTimezoneName,
                regularMarketPrice: result.meta?.regularMarketPrice,
            },
            lastUpdate: new Date().toISOString(),
        })
    } catch (error) {
        console.error("Intraday fetch error:", error)
        return NextResponse.json(
            { error: "Failed to fetch intraday data" },
            { status: 500 }
        )
    }
}
