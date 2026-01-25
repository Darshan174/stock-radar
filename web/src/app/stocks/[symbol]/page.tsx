"use client"

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { CandlestickChart } from "@/components/charts"
import { AIAnalysisPanel } from "@/components/ai-analysis-panel"
import { StockInfoPanel } from "@/components/stock-info-panel"
import { AdvancedChartsPanel } from "@/components/advanced-charts-panel"
import { LivePriceTicker } from "@/components/live-price-ticker"
import { FloatingChat } from "@/components/chat-assistant"
import {
  ArrowLeft,
  TrendingUp,
  TrendingDown,
  Play,
  Loader2,
  Target,
  ShieldAlert,
  BarChart3,
  Cpu,
  Database,
} from "lucide-react"
import { supabase, Stock, Analysis, PriceHistory } from "@/lib/supabase"

interface StockDetail extends Stock {
  price_history: PriceHistory[]
  analyses: Analysis[]
  latest_price?: number
  price_change?: number
}

const PERIOD_OPTIONS = [
  { value: "21", label: "21 Days" },
  { value: "30", label: "1 Month" },
  { value: "90", label: "3 Months" },
  { value: "180", label: "6 Months" },
  { value: "365", label: "1 Year" },
  { value: "730", label: "2 Years" },
  { value: "all", label: "All Time" },
]

// Interval options for intraday charts
const INTERVAL_OPTIONS = [
  { value: "5m", label: "5M", range: "1d" },
  { value: "15m", label: "15M", range: "5d" },
  { value: "1h", label: "1H", range: "5d" },
  { value: "4h", label: "4H", range: "1mo" },
  { value: "1d", label: "1D", range: "1mo" },
]

function getSignalColor(signal: string) {
  switch (signal) {
    case "strong_buy":
    case "buy":
      return "bg-green-500/10 text-green-500 border-green-500/20"
    case "strong_sell":
    case "sell":
      return "bg-red-500/10 text-red-500 border-red-500/20"
    default:
      return "bg-yellow-500/10 text-yellow-500 border-yellow-500/20"
  }
}

export default function StockDetailPage() {
  const params = useParams()
  const router = useRouter()
  const symbol = params.symbol as string

  const [stock, setStock] = useState<StockDetail | null>(null)
  const [fundamentals, setFundamentals] = useState<Record<string, any> | null>(null)
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [period, setPeriod] = useState("90")
  const [chartMode, setChartMode] = useState<"daily" | "intraday">("daily")
  const [timeInterval, setTimeInterval] = useState("5m")
  const [intradayData, setIntradayData] = useState<any[]>([])

  useEffect(() => {
    async function fetchStockData() {
      try {
        // Fetch stock info
        const { data: stockData } = await supabase
          .from("stocks")
          .select("*")
          .eq("symbol", symbol.toUpperCase())
          .single()

        if (!stockData) {
          setLoading(false)
          return
        }

        // Calculate date range based on period
        let priceHistory: any[] = []

        if (period === "all") {
          // For "All Time", fetch newest 2000 records (ordered desc), then reverse for chart
          const { data } = await supabase
            .from("price_history")
            .select("*")
            .eq("stock_id", stockData.id)
            .order("timestamp", { ascending: false })
            .limit(2000)
          priceHistory = data ? [...data].reverse() : []
        } else {
          // For specific periods, use date filter
          const days = parseInt(period)
          const startDate = new Date()
          startDate.setDate(startDate.getDate() - days)

          const { data } = await supabase
            .from("price_history")
            .select("*")
            .eq("stock_id", stockData.id)
            .gte("timestamp", startDate.toISOString())
            .order("timestamp", { ascending: true })
            .limit(1000)
          priceHistory = data || []
        }

        // Fetch analyses
        const { data: analyses } = await supabase
          .from("analysis")
          .select("*")
          .eq("stock_id", stockData.id)
          .order("created_at", { ascending: false })
          .limit(10)

        // Fetch intraday data to add/update today's candle
        try {
          const intradayResponse = await fetch(`/api/intraday?symbol=${symbol}`)
          if (intradayResponse.ok) {
            const intradayData = await intradayResponse.json()
            if (intradayData.todayCandle) {
              const todayCandle = intradayData.todayCandle
              // Check if last historical candle is for today
              if (priceHistory.length > 0) {
                const lastDate = new Date(priceHistory[priceHistory.length - 1].timestamp).toISOString().split("T")[0]
                const todayDate = todayCandle.time

                if (lastDate === todayDate) {
                  // Update today's candle with latest data
                  priceHistory[priceHistory.length - 1] = {
                    ...priceHistory[priceHistory.length - 1],
                    open: todayCandle.open,
                    high: todayCandle.high,
                    low: todayCandle.low,
                    close: todayCandle.close,
                    volume: todayCandle.volume,
                  }
                } else {
                  // Add today's candle if not present
                  priceHistory.push({
                    id: -1,
                    stock_id: stockData.id,
                    timestamp: new Date(todayCandle.time).toISOString(),
                    open: todayCandle.open,
                    high: todayCandle.high,
                    low: todayCandle.low,
                    close: todayCandle.close,
                    volume: todayCandle.volume,
                  })
                }
              }
            }
          }
        } catch (intradayError) {
          console.error("Error fetching intraday data:", intradayError)
        }

        // Calculate price change
        let latest_price = 0
        let price_change = 0
        if (priceHistory && priceHistory.length > 0) {
          latest_price = priceHistory[priceHistory.length - 1].close
          if (priceHistory.length > 1) {
            const prev = priceHistory[priceHistory.length - 2].close
            price_change = prev > 0 ? ((latest_price - prev) / prev) * 100 : 0
          }
        }

        setStock({
          ...stockData,
          price_history: priceHistory || [],
          analyses: analyses || [],
          latest_price,
          price_change,
        })

        // Fetch fundamentals from API
        try {
          const fundResponse = await fetch(`/api/fundamentals?symbol=${symbol}`)
          if (fundResponse.ok) {
            const fundData = await fundResponse.json()
            if (fundData && !fundData.error) {
              setFundamentals({ ...fundData, symbol })
            }
          }
        } catch (fundError) {
          console.error("Error fetching fundamentals:", fundError)
        }
      } catch (error) {
        console.error("Error fetching stock:", error)
      } finally {
        setLoading(false)
      }
    }

    if (symbol) {
      fetchStockData()
    }
  }, [symbol, period])

  // Fetch intraday data when in intraday mode
  useEffect(() => {
    async function fetchIntradayData() {
      if (chartMode !== "intraday" || !symbol) return

      try {
        const intervalOption = INTERVAL_OPTIONS.find(i => i.value === timeInterval)
        const range = intervalOption?.range || "1d"

        const response = await fetch(`/api/intraday?symbol=${symbol}&interval=${timeInterval}&range=${range}`)
        if (response.ok) {
          const data = await response.json()
          if (data.candles && data.candles.length > 0) {
            setIntradayData(data.candles)
          }
        }
      } catch (error) {
        console.error("Error fetching intraday data:", error)
      }
    }

    fetchIntradayData()

    // Auto-refresh every 30 seconds when in intraday mode
    const refreshTimer = chartMode === "intraday" ? window.setInterval(fetchIntradayData, 30000) : null

    return () => {
      if (refreshTimer) window.clearInterval(refreshTimer)
    }
  }, [symbol, chartMode, timeInterval])

  async function handleAnalyze() {
    setAnalyzing(true)
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      })

      if (response.ok) {
        // Refresh data
        window.location.reload()
      }
    } catch (error) {
      console.error("Analysis error:", error)
    } finally {
      setAnalyzing(false)
    }
  }

  if (loading) {
    return (
      <div className="p-8">
        <Skeleton className="h-8 w-32 mb-6" />
        <Skeleton className="h-[400px] mb-6" />
        <div className="grid gap-4 md:grid-cols-2">
          <Skeleton className="h-48" />
          <Skeleton className="h-48" />
        </div>
      </div>
    )
  }

  if (!stock) {
    return (
      <div className="p-8">
        <Button variant="ghost" onClick={() => router.back()} className="mb-6">
          <ArrowLeft className="h-4 w-4 mr-2" />
          Back
        </Button>
        <Card className="p-8 text-center">
          <p className="text-muted-foreground">Stock not found: {symbol}</p>
          <Button className="mt-4" onClick={() => router.push("/stocks")}>
            Go to Watchlist
          </Button>
        </Card>
      </div>
    )
  }

  const latestAnalysis = stock.analyses[0]
  const chartData = stock.price_history.map((p) => ({
    time: p.timestamp.split("T")[0],
    open: p.open,
    high: p.high,
    low: p.low,
    close: p.close,
    volume: p.volume,
  }))

  // Get signals for chart markers
  const signals = stock.analyses
    .filter((a) => ["buy", "strong_buy", "sell", "strong_sell"].includes(a.signal))
    .map((a) => ({
      time: a.created_at.split("T")[0],
      type: a.signal.includes("buy") ? "buy" as const : "sell" as const,
      price: a.target_price || 0,
    }))

  return (
    <div className="p-8">
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.back()}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-3xl font-bold">{stock.symbol}</h1>
              {latestAnalysis && (
                <Badge variant="outline" className={getSignalColor(latestAnalysis.signal)}>
                  {latestAnalysis.signal.toUpperCase()}
                </Badge>
              )}
            </div>
            <p className="text-muted-foreground">{stock.name}</p>
          </div>
        </div>
        <div className="flex items-center gap-4">
          {/* Live Price Ticker */}
          <LivePriceTicker
            symbol={stock.symbol}
            currency={stock.currency === "INR" ? "₹" : "$"}
          />
          <Button onClick={handleAnalyze} disabled={analyzing}>
            {analyzing ? (
              <Loader2 className="h-4 w-4 mr-2 animate-spin" />
            ) : (
              <Play className="h-4 w-4 mr-2" />
            )}
            Analyze
          </Button>
        </div>
      </div>

      {/* AI Analysis Panel */}
      <div className="mb-6">
        <AIAnalysisPanel
          analysis={latestAnalysis}
          currentPrice={stock.latest_price}
          currency={stock.currency === "INR" ? "₹" : "$"}
          onRunAnalysis={handleAnalyze}
          isAnalyzing={analyzing}
        />
      </div>

      {/* Chart */}
      <Card className="mb-6">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="flex items-center gap-2">
                <BarChart3 className="h-5 w-5" />
                Price Chart
              </CardTitle>
              <CardDescription>
                {chartMode === "intraday"
                  ? `${intradayData.length} ${timeInterval} candles (auto-refreshing)`
                  : chartData.length > 0
                    ? `${chartData.length} days of price history`
                    : "No price data available"}
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              {/* Mode Toggle */}
              <div className="flex items-center rounded-md border">
                <Button
                  variant={chartMode === "daily" ? "default" : "ghost"}
                  size="sm"
                  className="rounded-r-none text-xs"
                  onClick={() => setChartMode("daily")}
                >
                  Daily
                </Button>
                <Button
                  variant={chartMode === "intraday" ? "default" : "ghost"}
                  size="sm"
                  className="rounded-l-none text-xs"
                  onClick={() => setChartMode("intraday")}
                >
                  Intraday
                </Button>
              </div>

              {/* Interval Selector (for intraday) */}
              {chartMode === "intraday" && (
                <div className="flex items-center gap-1">
                  {INTERVAL_OPTIONS.map((opt) => (
                    <Button
                      key={opt.value}
                      variant={timeInterval === opt.value ? "default" : "ghost"}
                      size="sm"
                      className="text-xs h-8 px-2"
                      onClick={() => setTimeInterval(opt.value)}
                    >
                      {opt.label}
                    </Button>
                  ))}
                </div>
              )}

              {/* Period Selector (for daily) */}
              {chartMode === "daily" && (
                <Select value={period} onValueChange={setPeriod}>
                  <SelectTrigger className="w-32">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {PERIOD_OPTIONS.map((opt) => (
                      <SelectItem key={opt.value} value={opt.value}>
                        {opt.label}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              )}
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {chartMode === "intraday" ? (
            intradayData.length > 0 ? (
              <CandlestickChart
                key={`${symbol}-intraday-${timeInterval}`}
                data={intradayData}
                signals={[]}
                height={500}
                currency={stock.currency === "INR" ? "₹" : "$"}
              />
            ) : (
              <div className="h-[500px] flex items-center justify-center text-muted-foreground bg-background">
                Loading intraday data...
              </div>
            )
          ) : chartData.length > 0 ? (
            <CandlestickChart
              key={`${symbol}-${period}`}
              data={chartData}
              signals={signals}
              height={500}
              currency={stock.currency === "INR" ? "₹" : "$"}
            />
          ) : (
            <div className="h-[500px] flex items-center justify-center text-muted-foreground bg-background">
              Run an analysis to fetch price data
            </div>
          )}
        </CardContent>
      </Card>

      {/* Fundamentals & Stats Panel */}
      {fundamentals && fundamentals.symbol && (
        <div className="mb-6">
          <StockInfoPanel
            fundamentals={fundamentals as any}
            currentPrice={stock.latest_price}
            currency={stock.currency === "INR" ? "₹" : "$"}
          />
        </div>
      )}

      {/* Advanced Charts & AI Algo */}
      <div className="mb-6">
        <AdvancedChartsPanel
          fundamentals={fundamentals ? (fundamentals as any) : { symbol: stock.symbol }}
          aiPrediction={(latestAnalysis as any)?.algo_prediction || null}
          currency={stock.currency === "INR" ? "₹" : "$"}
        />
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* Price Targets */}
        <Card>
          <CardHeader>
            <CardTitle>Price Targets</CardTitle>
            <CardDescription>AI-generated price levels</CardDescription>
          </CardHeader>
          <CardContent>
            {latestAnalysis ? (
              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 rounded-lg bg-green-500/10">
                  <div className="flex items-center gap-2">
                    <Target className="h-4 w-4 text-green-500" />
                    <span>Target Price</span>
                  </div>
                  <span className="font-bold text-green-500">
                    {latestAnalysis.target_price
                      ? `${stock.currency === "INR" ? "₹" : "$"}${latestAnalysis.target_price.toFixed(2)}`
                      : "N/A"}
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg bg-red-500/10">
                  <div className="flex items-center gap-2">
                    <ShieldAlert className="h-4 w-4 text-red-500" />
                    <span>Stop Loss</span>
                  </div>
                  <span className="font-bold text-red-500">
                    {latestAnalysis.stop_loss
                      ? `${stock.currency === "INR" ? "₹" : "$"}${latestAnalysis.stop_loss.toFixed(2)}`
                      : "N/A"}
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg bg-blue-500/10">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="h-4 w-4 text-blue-500" />
                    <span>Support</span>
                  </div>
                  <span className="font-bold text-blue-500">
                    {latestAnalysis.support_level
                      ? `${stock.currency === "INR" ? "₹" : "$"}${latestAnalysis.support_level.toFixed(2)}`
                      : "N/A"}
                  </span>
                </div>
                <div className="flex items-center justify-between p-3 rounded-lg bg-purple-500/10">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="h-4 w-4 text-purple-500" />
                    <span>Resistance</span>
                  </div>
                  <span className="font-bold text-purple-500">
                    {latestAnalysis.resistance_level
                      ? `${stock.currency === "INR" ? "₹" : "$"}${latestAnalysis.resistance_level.toFixed(2)}`
                      : "N/A"}
                  </span>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-4">
                No price targets available
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Analysis History */}
      {stock.analyses.length > 1 && (
        <Card className="mt-4">
          <CardHeader>
            <CardTitle>Analysis History</CardTitle>
            <CardDescription>Previous {stock.analyses.length} analyses</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {stock.analyses.slice(1).map((analysis: any) => (
                <div
                  key={analysis.id}
                  className="flex items-center justify-between p-3 rounded-lg border"
                >
                  <div className="flex items-center gap-3">
                    <Badge variant="outline" className={getSignalColor(analysis.signal)}>
                      {analysis.signal.toUpperCase()}
                    </Badge>
                    <span className="text-sm text-muted-foreground">
                      {Math.round(analysis.confidence * 100)}% confidence
                    </span>
                    {/* Model & RAG indicators */}
                    <div className="flex items-center gap-2">
                      {analysis.llm_model && (
                        <span className="flex items-center gap-1 text-xs text-blue-400">
                          <Cpu className="h-3 w-3" />
                          {analysis.llm_model.split("/").pop()}
                        </span>
                      )}
                      {analysis.embedding_text && (
                        <span className="flex items-center gap-1 text-xs text-purple-400">
                          <Database className="h-3 w-3" />
                          RAG
                        </span>
                      )}
                    </div>
                  </div>
                  <span className="text-xs text-muted-foreground">
                    {new Date(analysis.created_at).toLocaleDateString()}
                  </span>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Floating Chat Assistant - Bottom Left */}
      <FloatingChat defaultSymbol={stock.symbol} />
    </div>
  )
}
