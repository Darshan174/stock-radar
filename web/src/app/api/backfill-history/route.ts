import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { createClient } from "@supabase/supabase-js"
import { enforceRateLimit, RATE_BUCKETS } from "@/lib/rate-limit"

const execAsync = promisify(exec)

// Lazy Supabase client initialization to avoid module load errors
function getSupabase() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseKey) {
    throw new Error("Missing Supabase environment variables")
  }

  return createClient(supabaseUrl, supabaseKey)
}

/**
 * POST /api/backfill-history
 *
 * Backfills historical price data for a stock or all stocks.
 * This fetches the maximum available history and stores it in the database.
 *
 * Body:
 *   - symbol: Stock symbol to backfill (optional, if not provided backfills all stocks)
 *   - period: Period to fetch ("max" by default for full history)
 *   - clearExisting: Whether to clear existing price data before backfilling (default: false)
 */
export async function POST(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.paid)
  if (limited) return limited

  try {
    const supabase = getSupabase()
    const { symbol, period = "max", clearExisting = false } = await request.json()

    const projectPath = path.join(process.cwd(), "..")
    const results: Array<{ symbol: string; success: boolean; message: string; recordsAdded?: number }> = []

    // Get list of stocks to backfill
    let stocksToBackfill: Array<{ id: number; symbol: string }> = []

    if (symbol) {
      // Single stock
      const { data: stock } = await supabase
        .from("stocks")
        .select("id, symbol")
        .eq("symbol", symbol.toUpperCase())
        .single()

      if (stock) {
        stocksToBackfill = [stock]
      } else {
        return NextResponse.json({ error: `Stock ${symbol} not found` }, { status: 404 })
      }
    } else {
      // All active stocks
      const { data: stocks } = await supabase
        .from("stocks")
        .select("id, symbol")
        .eq("is_active", true)

      stocksToBackfill = stocks || []
    }

    if (stocksToBackfill.length === 0) {
      return NextResponse.json({ error: "No stocks found to backfill" }, { status: 404 })
    }

    // Process each stock
    for (const stock of stocksToBackfill) {
      try {
        // Optionally clear existing price history
        if (clearExisting) {
          await supabase
            .from("price_history")
            .delete()
            .eq("stock_id", stock.id)
        }

        // Run Python script to fetch and store historical data
        const pythonCode = `
import sys
sys.path.insert(0, "${projectPath}/src")
from agents.fetcher import StockFetcher
from agents.storage import StockStorage

fetcher = StockFetcher()
storage = StockStorage()

# Fetch price history with max period
prices = fetcher.get_price_history("${stock.symbol}", period="${period}")

if prices:
    # Get stock record
    stock_record = storage.get_stock_by_symbol("${stock.symbol}")
    if stock_record:
        # Convert PriceData objects to dicts
        price_dicts = []
        for p in prices:
            price_dicts.append({
                "timestamp": p.timestamp,
                "open": p.open,
                "high": p.high,
                "low": p.low,
                "close": p.close,
                "volume": p.volume
            })

        # Store price data
        count = storage.store_price_data(
            stock_id=stock_record["id"],
            prices=price_dicts,
            timeframe="1d"
        )
        print(f"SUCCESS:{count}")
    else:
        print("ERROR:Stock not found in database")
else:
    print("ERROR:No price data returned")
`.trim()

        const { stdout, stderr } = await execAsync(
          `cd "${projectPath}" && python3 -c '${pythonCode.replace(/'/g, "\\'")}'`,
          { timeout: 120000 }
        )

        // Parse result
        const output = stdout.trim()
        if (output.startsWith("SUCCESS:")) {
          const recordsAdded = parseInt(output.split(":")[1]) || 0

          // Get new record count
          const { count: afterCount } = await supabase
            .from("price_history")
            .select("*", { count: "exact", head: true })
            .eq("stock_id", stock.id)

          results.push({
            symbol: stock.symbol,
            success: true,
            message: `Backfilled ${recordsAdded} records (total: ${afterCount})`,
            recordsAdded
          })
        } else if (output.startsWith("ERROR:")) {
          results.push({
            symbol: stock.symbol,
            success: false,
            message: output.split(":")[1] || "Unknown error"
          })
        } else {
          results.push({
            symbol: stock.symbol,
            success: false,
            message: stderr || "Unknown error"
          })
        }
      } catch (stockError) {
        results.push({
          symbol: stock.symbol,
          success: false,
          message: stockError instanceof Error ? stockError.message : "Unknown error"
        })
      }
    }

    const successCount = results.filter(r => r.success).length
    const totalRecords = results.reduce((sum, r) => sum + (r.recordsAdded || 0), 0)

    return NextResponse.json({
      success: true,
      message: `Backfilled ${successCount}/${stocksToBackfill.length} stocks with ${totalRecords} total records`,
      results
    })
  } catch (error) {
    console.error("Backfill error:", error)
    const message = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json({ error: message }, { status: 500 })
  }
}

/**
 * GET /api/backfill-history?symbol=SYMBOL
 *
 * Get the current price history status for a stock or all stocks.
 */
export async function GET(request: NextRequest) {
  const limited = await enforceRateLimit(request, RATE_BUCKETS.free)
  if (limited) return limited

  try {
    const supabase = getSupabase()
    const searchParams = request.nextUrl.searchParams
    const symbol = searchParams.get("symbol")

    if (symbol) {
      // Get status for single stock
      const { data: stock } = await supabase
        .from("stocks")
        .select("id, symbol, name")
        .eq("symbol", symbol.toUpperCase())
        .single()

      if (!stock) {
        return NextResponse.json({ error: `Stock ${symbol} not found` }, { status: 404 })
      }

      const { count } = await supabase
        .from("price_history")
        .select("*", { count: "exact", head: true })
        .eq("stock_id", stock.id)

      // Get date range
      const { data: dateRange } = await supabase
        .from("price_history")
        .select("timestamp")
        .eq("stock_id", stock.id)
        .order("timestamp", { ascending: true })
        .limit(1)

      const { data: latestDate } = await supabase
        .from("price_history")
        .select("timestamp")
        .eq("stock_id", stock.id)
        .order("timestamp", { ascending: false })
        .limit(1)

      return NextResponse.json({
        symbol: stock.symbol,
        name: stock.name,
        recordCount: count || 0,
        dateRange: {
          from: dateRange?.[0]?.timestamp || null,
          to: latestDate?.[0]?.timestamp || null
        }
      })
    } else {
      // Get status for all stocks
      const { data: stocks } = await supabase
        .from("stocks")
        .select("id, symbol, name")
        .eq("is_active", true)

      if (!stocks || stocks.length === 0) {
        return NextResponse.json({ stocks: [] })
      }

      const stockStatuses = await Promise.all(
        stocks.map(async (stock) => {
          const { count } = await supabase
            .from("price_history")
            .select("*", { count: "exact", head: true })
            .eq("stock_id", stock.id)

          return {
            symbol: stock.symbol,
            name: stock.name,
            recordCount: count || 0
          }
        })
      )

      return NextResponse.json({
        stocks: stockStatuses.sort((a, b) => b.recordCount - a.recordCount)
      })
    }
  } catch (error) {
    console.error("Status check error:", error)
    const message = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json({ error: message }, { status: 500 })
  }
}
