"use client"

import { useEffect, useMemo, useState, type ComponentProps } from "react"
import { useParams, useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { CandlestickChart } from "@/components/charts"
import { AIAnalysisPanel } from "@/components/ai-analysis-panel"
import { StockInfoPanel } from "@/components/stock-info-panel"
import { AdvancedChartsPanel } from "@/components/advanced-charts-panel"
import { LivePriceTicker } from "@/components/live-price-ticker"
import { FloatingChat } from "@/components/chat-assistant"
import { RAGInsightsPanel } from "@/components/rag-insights-panel"
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
  ChevronDown,
  ChevronUp,
  Brain,
  Database,
} from "lucide-react"
import { RAGBadge } from "@/components/rag-badge"
import { supabase, Stock, Analysis } from "@/lib/supabase"
import { useLiveStockData } from "@/hooks/use-live-stock-data"

interface StockDetail extends Stock {
  analyses: Analysis[]
}

interface ChartCandle {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

interface ChartMeta {
  isIntraday: boolean
  interval: string
  exchangeTimezoneName: string
}

interface LiveLineTick {
  time: number
  value: number
}

type StockInfoFundamentals = ComponentProps<typeof StockInfoPanel>["fundamentals"]
type AdvancedPanelFundamentals = ComponentProps<typeof AdvancedChartsPanel>["fundamentals"]
type AdvancedPanelPrediction = ComponentProps<typeof AdvancedChartsPanel>["aiPrediction"]

interface AnalysisWithAlgo extends Analysis {
  algo_prediction?: AdvancedPanelPrediction
}

const PERIOD_OPTIONS = [
  { value: "1d", label: "1D" },
  { value: "1w", label: "1W" },
  { value: "1m", label: "1M" },
  { value: "3m", label: "3M" },
  { value: "6m", label: "6M" },
  { value: "1y", label: "1Y" },
  { value: "3y", label: "3Y" },
  { value: "5y", label: "5Y" },
  { value: "all", label: "All" },
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
  const [fundamentals, setFundamentals] = useState<StockInfoFundamentals | null>(null)
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [period, setPeriod] = useState("1d")
  const [chartData, setChartData] = useState<ChartCandle[]>([])
  const [chartMeta, setChartMeta] = useState<ChartMeta>({
    isIntraday: false,
    interval: "1d",
    exchangeTimezoneName: "UTC",
  })
  const [liveLineTicks, setLiveLineTicks] = useState<LiveLineTick[]>([])
  const [chartLoading, setChartLoading] = useState(true)
  const [aiPanelOpen, setAiPanelOpen] = useState(false)
  const [ragPanelOpen, setRagPanelOpen] = useState(false)

  // Fetch stock info + analyses from Supabase
  useEffect(() => {
    async function fetchStockInfo() {
      try {
        const { data: stockData } = await supabase
          .from("stocks")
          .select("*")
          .eq("symbol", symbol.toUpperCase())
          .single()

        if (!stockData) {
          setLoading(false)
          return
        }

        const { data: analyses } = await supabase
          .from("analysis")
          .select("*")
          .eq("stock_id", stockData.id)
          .order("created_at", { ascending: false })
          .limit(10)

        setStock({
          ...stockData,
          analyses: analyses || [],
        })

        // Fetch fundamentals
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
      fetchStockInfo()
    }
  }, [symbol])

  // Fetch chart data from Yahoo Finance (independent of analysis)
  useEffect(() => {
    async function fetchChartData(options?: { background?: boolean }) {
      if (!symbol) return
      const background = options?.background === true
      if (!background) {
        setChartLoading(true)
      }

      try {
        const response = await fetch(
          `/api/chart-data?symbol=${encodeURIComponent(symbol)}&period=${period}`
        )
        if (response.ok) {
          const data = await response.json()
          if (data.candles && data.candles.length > 0) {
            setChartData(data.candles)
            setChartMeta({
              isIntraday: !!data.isIntraday,
              interval: data.interval || "1d",
              exchangeTimezoneName: data.meta?.exchangeTimezoneName || "UTC",
            })
          } else if (!background) {
            setChartData([])
          }
        } else if (!background) {
          setChartData([])
        }
      } catch (error) {
        console.error("Error fetching chart data:", error)
        if (!background) {
          setChartData([])
        }
      } finally {
        if (!background) {
          setChartLoading(false)
        }
      }
    }

    fetchChartData()

    // Auto-refresh intraday periods
    const isIntraday = period === "1d" || period === "1w"
    const refreshTimer = isIntraday
      ? window.setInterval(() => fetchChartData({ background: true }), 30000)
      : null

    return () => {
      if (refreshTimer) window.clearInterval(refreshTimer)
    }
  }, [symbol, period])

  const { livePrice } = useLiveStockData({
    symbol,
    enabled: !!symbol,
    preferWebSocket: true,
    refreshInterval: 2000,
    updateThrottleMs: 0,
  })

  const intervalSeconds = useMemo(() => {
    const match = chartMeta.interval.match(/^(\d+)([mhdwk])$/)
    if (!match) return 60
    const value = Number(match[1])
    const unit = match[2]
    if (unit === "m") return value * 60
    if (unit === "h") return value * 3600
    if (unit === "d") return value * 86400
    if (unit === "w") return value * 7 * 86400
    return 60
  }, [chartMeta.interval])

  useEffect(() => {
    setLiveLineTicks([])
  }, [symbol, period])

  // Keep a short high-resolution tick trail for line mode.
  useEffect(() => {
    if (!livePrice) return
    setLiveLineTicks((prev) => {
      const last = prev[prev.length - 1]
      if (last && Math.abs(last.value - livePrice.price) < 0.0001) {
        return prev
      }

      let tickTime = livePrice.timestamp.getTime() / 1000
      if (last && tickTime <= last.time) {
        tickTime = last.time + 0.0001
      }

      const next = [...prev, { time: tickTime, value: livePrice.price }]
      const maxPoints = 3000
      return next.length > maxPoints ? next.slice(next.length - maxPoints) : next
    })
  }, [livePrice])

  // Patch the last intraday candle with live ticks from websocket.
  useEffect(() => {
    if (!chartMeta.isIntraday || !livePrice || chartData.length === 0) return

    const liveTs = Math.floor(livePrice.timestamp.getTime() / 1000)
    const bucketTs = Math.floor(liveTs / intervalSeconds) * intervalSeconds

    setChartData((prev) => {
      if (!prev.length) return prev
      const next = [...prev]
      const last = next[next.length - 1]

      if (bucketTs > last.time) {
        next.push({
          time: bucketTs,
          open: last.close,
          high: Math.max(last.close, livePrice.price),
          low: Math.min(last.close, livePrice.price),
          close: livePrice.price,
          volume: livePrice.volume || last.volume || 0,
        })
        return next
      }

      const updatedLast: ChartCandle = {
        ...last,
        high: Math.max(last.high, livePrice.price),
        low: Math.min(last.low, livePrice.price),
        close: livePrice.price,
        volume: livePrice.volume || last.volume || 0,
      }
      next[next.length - 1] = updatedLast
      return next
    })
  }, [livePrice, chartMeta.isIntraday, intervalSeconds, chartData.length])

  async function handleAnalyze() {
    setAnalyzing(true)
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      })

      if (response.ok) {
        window.location.reload()
      }
    } catch (error) {
      console.error("Analysis error:", error)
    } finally {
      setAnalyzing(false)
    }
  }

  const signals = useMemo(
    () =>
      (stock?.analyses || [])
        .filter((a) => ["buy", "strong_buy", "sell", "strong_sell"].includes(a.signal))
        .map((a) => ({
          time: a.created_at.split("T")[0],
          type: a.signal.includes("buy") ? ("buy" as const) : ("sell" as const),
          price: a.target_price || 0,
        })),
    [stock?.analyses]
  )

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

  const latestAnalysis = stock.analyses[0] as AnalysisWithAlgo | undefined

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
          <LivePriceTicker
            symbol={stock.symbol}
            currency={stock.currency === "INR" ? "\u20B9" : "$"}
            livePriceOverride={livePrice}
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

      {/* AI Analysis Panel - Collapsible */}
      <div className="mb-4">
        <button
          onClick={() => setAiPanelOpen(!aiPanelOpen)}
          className="w-full flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <Brain className="h-4 w-4 text-purple-500" />
            <span className="font-medium text-sm">AI Analysis</span>
            {latestAnalysis && (
              <>
                <Badge variant="outline" className={getSignalColor(latestAnalysis.signal)}>
                  {latestAnalysis.signal.toUpperCase()}
                </Badge>
                <span className="text-xs text-muted-foreground">
                  {Math.round(latestAnalysis.confidence * 100)}% confidence
                </span>
              </>
            )}
          </div>
          {aiPanelOpen ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </button>
        {aiPanelOpen && (
          <div className="mt-2">
            <AIAnalysisPanel
              analysis={latestAnalysis}
              currentPrice={chartData.length > 0 ? chartData[chartData.length - 1].close : undefined}
              currency={stock.currency === "INR" ? "\u20B9" : "$"}
              onRunAnalysis={handleAnalyze}
              isAnalyzing={analyzing}
            />
          </div>
        )}
      </div>

      {/* RAG Insights Panel - Collapsible */}
      <div className="mb-4">
        <button
          onClick={() => setRagPanelOpen(!ragPanelOpen)}
          className="w-full flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <Database className="h-4 w-4 text-blue-500" />
            <span className="font-medium text-sm">RAG Insights</span>
            <Badge variant="outline" className="text-xs">
              {stock.symbol}
            </Badge>
          </div>
          {ragPanelOpen ? (
            <ChevronUp className="h-4 w-4 text-muted-foreground" />
          ) : (
            <ChevronDown className="h-4 w-4 text-muted-foreground" />
          )}
        </button>
        {ragPanelOpen && (
          <div className="mt-2">
            <RAGInsightsPanel
              symbol={stock.symbol}
              defaultExpanded={true}
              autoRefresh={false}
            />
          </div>
        )}
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
                {chartData.length > 0
                  ? `${chartData.length} data points`
                  : chartLoading
                    ? "Loading..."
                    : "No data available"}
              </CardDescription>
            </div>
            {/* Period Selector - Inline Buttons */}
            <div className="flex items-center gap-1">
              {PERIOD_OPTIONS.map((opt) => (
                <Button
                  key={opt.value}
                  variant={period === opt.value ? "default" : "ghost"}
                  size="sm"
                  className="text-xs h-8 px-2.5"
                  onClick={() => setPeriod(opt.value)}
                >
                  {opt.label}
                </Button>
              ))}
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-0">
          {chartLoading ? (
            <div className="h-[700px] flex items-center justify-center text-muted-foreground bg-background">
              <Loader2 className="h-6 w-6 animate-spin mr-2" />
              Loading chart data...
            </div>
          ) : chartData.length > 0 ? (
            <CandlestickChart
              key={`${symbol}-${period}`}
              data={chartData}
              liveTicks={liveLineTicks}
              signals={period === "1d" || period === "1w" ? [] : signals}
              height={700}
              currency={stock.currency === "INR" ? "\u20B9" : "$"}
              isIntraday={chartMeta.isIntraday}
              timezone={chartMeta.exchangeTimezoneName}
              interval={chartMeta.interval}
            />
          ) : (
            <div className="h-[700px] flex items-center justify-center text-muted-foreground bg-background">
              No chart data available for this period
            </div>
          )}
        </CardContent>
      </Card>

      {/* Fundamentals & Stats Panel */}
      {fundamentals && fundamentals.symbol && (
        <div className="mb-6">
          <StockInfoPanel
            fundamentals={fundamentals}
            currentPrice={chartData.length > 0 ? chartData[chartData.length - 1].close : undefined}
            currency={stock.currency === "INR" ? "\u20B9" : "$"}
          />
        </div>
      )}

      {/* Advanced Charts & AI Algo */}
      <div className="mb-6">
        <AdvancedChartsPanel
          fundamentals={
            fundamentals
              ? (fundamentals as unknown as AdvancedPanelFundamentals)
              : ({ symbol: stock.symbol } as AdvancedPanelFundamentals)
          }
          aiPrediction={latestAnalysis?.algo_prediction || null}
          currency={stock.currency === "INR" ? "\u20B9" : "$"}
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
                      ? `${stock.currency === "INR" ? "\u20B9" : "$"}${latestAnalysis.target_price.toFixed(2)}`
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
                      ? `${stock.currency === "INR" ? "\u20B9" : "$"}${latestAnalysis.stop_loss.toFixed(2)}`
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
                      ? `${stock.currency === "INR" ? "\u20B9" : "$"}${latestAnalysis.support_level.toFixed(2)}`
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
                      ? `${stock.currency === "INR" ? "\u20B9" : "$"}${latestAnalysis.resistance_level.toFixed(2)}`
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
              {stock.analyses.slice(1).map((analysis) => (
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
                    <div className="flex items-center gap-2">
                      {analysis.llm_model && (
                        <span className="flex items-center gap-1 text-xs text-blue-400">
                          <Cpu className="h-3 w-3" />
                          {analysis.llm_model.split("/").pop()}
                        </span>
                      )}
                      <RAGBadge
                        isActive={!!analysis.embedding_text}
                        size="sm"
                        variant="inline"
                      />
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

      {/* Floating Chat Assistant */}
      <FloatingChat defaultSymbol={stock.symbol} />
    </div>
  )
}
