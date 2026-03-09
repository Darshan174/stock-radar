"use client"

import { useCallback, useEffect, useMemo, useState, type ComponentProps } from "react"
import { useParams, useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { AnalysisModeToggle } from "@/components/analysis-mode-toggle"
import { CandlestickChart } from "@/components/charts"
import { AIAnalysisPanel } from "@/components/ai-analysis-panel"
import { StockInfoPanel } from "@/components/stock-info-panel"
import { AdvancedChartsPanel } from "@/components/advanced-charts-panel"
import { LivePriceTicker } from "@/components/live-price-ticker"
import { RAGInsightsPanel } from "@/components/rag-insights-panel"
import { ChatContextBadge } from "@/components/chat-context-badge"
import { StockChatButton } from "@/components/stock-chat-button"
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
  type LucideIcon,
} from "lucide-react"
import { RAGBadge } from "@/components/rag-badge"
import {
  getAnalysisModeLabel,
  getDefaultAnalysisPeriod,
  loadStoredAnalysisMode,
  type AnalysisMode,
} from "@/lib/analysis-mode"
import { supabase, Stock, Analysis } from "@/lib/supabase"
import { useLiveStockData } from "@/hooks/use-live-stock-data"
import type { AnalyzeJobCreated, AnalyzeJobStatus } from "@/lib/analyze-contracts"

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

function getAnalyzeStatusClass(state: "idle" | "queued" | "running" | "succeeded" | "failed") {
  switch (state) {
    case "queued":
    case "running":
      return "text-blue-500"
    case "succeeded":
      return "text-green-500"
    case "failed":
      return "text-red-500"
    default:
      return "text-muted-foreground"
  }
}

function formatRelativeTime(dateStr: string) {
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffMinutes = Math.floor(diffMs / (1000 * 60))
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)

  if (diffMinutes < 1) return "just now"
  if (diffMinutes < 60) return `${diffMinutes}m ago`
  if (diffHours < 24) return `${diffHours}h ago`
  return `${diffDays}d ago`
}

function formatMoney(value: number | null | undefined, currency: string) {
  if (typeof value !== "number" || !Number.isFinite(value)) return "N/A"
  return `${currency}${value.toFixed(2)}`
}

function getPriceMovePercent(currentPrice: number | undefined, level: number | null | undefined) {
  if (typeof currentPrice !== "number" || typeof level !== "number" || currentPrice === 0) return null
  return ((level - currentPrice) / currentPrice) * 100
}

function formatPriceRelation(currentPrice: number | undefined, level: number | null | undefined) {
  const move = getPriceMovePercent(currentPrice, level)
  if (move === null) return "No live comparison"
  if (Math.abs(move) < 0.05) return "At current price"
  return `${Math.abs(move).toFixed(1)}% ${move > 0 ? "above" : "below"} current`
}

function getRiskRewardValue(analysis: Analysis, currentPrice: number | undefined) {
  if (typeof analysis.risk_reward_ratio === "number" && Number.isFinite(analysis.risk_reward_ratio)) {
    return analysis.risk_reward_ratio
  }

  if (
    typeof currentPrice !== "number" ||
    typeof analysis.target_price !== "number" ||
    typeof analysis.stop_loss !== "number"
  ) {
    return null
  }

  const reward = Math.abs(analysis.target_price - currentPrice)
  const risk = Math.abs(currentPrice - analysis.stop_loss)
  if (risk <= 0) return null
  return reward / risk
}

export default function StockDetailPage() {
  const params = useParams()
  const router = useRouter()
  const symbol = params.symbol as string

  const [stock, setStock] = useState<StockDetail | null>(null)
  const [fundamentals, setFundamentals] = useState<StockInfoFundamentals | null>(null)
  const [loading, setLoading] = useState(true)
  const [analyzing, setAnalyzing] = useState(false)
  const [analyzeState, setAnalyzeState] = useState<"idle" | "queued" | "running" | "succeeded" | "failed">("idle")
  const [analyzeMessage, setAnalyzeMessage] = useState("")
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
  const [selectedMode, setSelectedMode] = useState<AnalysisMode>(loadStoredAnalysisMode)

  const fetchStockInfo = useCallback(async () => {
    try {
      const { data: stockData } = await supabase
        .from("stocks")
        .select("*")
        .eq("symbol", symbol.toUpperCase())
        .single()

      if (!stockData) {
        setStock(null)
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
  }, [symbol])

  // Fetch stock info + analyses from Supabase
  useEffect(() => {
    if (symbol) {
      fetchStockInfo()
    }
  }, [fetchStockInfo, symbol])

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

  async function waitForAnalyzeJob(jobId: string, timeoutMs = 180000): Promise<AnalyzeJobStatus> {
    const startedAt = Date.now()

    while (Date.now() - startedAt < timeoutMs) {
      const statusResponse = await fetch(`/api/analyze/status?jobId=${encodeURIComponent(jobId)}`, {
        cache: "no-store",
      })

      const statusData = (await statusResponse.json()) as AnalyzeJobStatus & { error?: string }
      if (!statusResponse.ok) {
        throw new Error(statusData.error || "Failed to fetch analysis status")
      }

      if (statusData.status === "succeeded") {
        return statusData
      }

      if (statusData.status === "failed") {
        throw new Error(statusData.error || "Analysis failed")
      }

      setAnalyzeState(statusData.status === "queued" ? "queued" : "running")
      setAnalyzeMessage(
        statusData.status === "queued"
          ? `${getAnalysisModeLabel(selectedMode)} analysis queued...`
          : `${getAnalysisModeLabel(selectedMode)} analysis running...`
      )

      await new Promise((resolve) => setTimeout(resolve, 2000))
    }

    throw new Error("Analysis timed out. Please try again.")
  }

  async function handleAnalyze() {
    setAnalyzing(true)
    setAnalyzeState("queued")
    setAnalyzeMessage(`Submitting ${getAnalysisModeLabel(selectedMode)} analysis job...`)
    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          symbol,
          mode: selectedMode,
          period: getDefaultAnalysisPeriod(selectedMode),
        }),
      })

      const data = (await response.json()) as Partial<AnalyzeJobCreated> & { error?: string }
      if (!response.ok) {
        throw new Error(data.error || "Failed to submit analysis job")
      }

      if (!data.jobId) {
        throw new Error("Analysis job was not created")
      }

      setAnalyzeState("running")
      setAnalyzeMessage(`${getAnalysisModeLabel(selectedMode)} analysis running...`)
      await waitForAnalyzeJob(data.jobId)

      setAnalyzeState("succeeded")
      setAnalyzeMessage(`${getAnalysisModeLabel(selectedMode)} analysis complete. Refreshing...`)
      await fetchStockInfo()
      setAiPanelOpen(true)
    } catch (error) {
      console.error("Analysis error:", error)
      setAnalyzeState("failed")
      setAnalyzeMessage(error instanceof Error ? error.message : "Analysis failed")
    } finally {
      setAnalyzing(false)
    }
  }

  const signals = useMemo(
    () =>
      (stock?.analyses || [])
        .filter((a) => a.mode === selectedMode)
        .filter((a) => ["buy", "strong_buy", "sell", "strong_sell"].includes(a.signal))
        .map((a) => ({
          time: a.created_at.split("T")[0],
          type: a.signal.includes("buy") ? ("buy" as const) : ("sell" as const),
          price: a.target_price || 0,
        })),
    [selectedMode, stock?.analyses]
  )

  if (loading) {
    return (
      <div className="app-page">
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
      <div className="app-page">
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

  const intradayAnalyses = stock.analyses.filter((analysis) => analysis.mode === "intraday") as AnalysisWithAlgo[]
  const longtermAnalyses = stock.analyses.filter((analysis) => analysis.mode === "longterm") as AnalysisWithAlgo[]
  const selectedAnalyses = selectedMode === "intraday" ? intradayAnalyses : longtermAnalyses
  const latestIntradayAnalysis = intradayAnalyses[0]
  const latestLongtermAnalysis = longtermAnalyses[0]
  const latestAnalysis = selectedAnalyses[0]
  const hasAnalysis = stock.analyses.length > 0
  const analysisCount = stock.analyses.length
  const selectedModeCount = selectedAnalyses.length
  const priceDisplayCurrency = stock.currency === "INR" ? "\u20B9" : "$"
  const currentSetupPrice = livePrice?.price ?? (chartData.length > 0 ? chartData[chartData.length - 1].close : undefined)
  const tradeSetupLevels = latestAnalysis
    ? [
      {
        key: "target",
        label: "Target",
        value: latestAnalysis.target_price,
        description: "Primary take-profit level from the latest analysis.",
        color: "text-green-400",
        barClass: "bg-green-400",
        Icon: Target,
      },
      {
        key: "resistance",
        label: "Resistance",
        value: latestAnalysis.resistance_level,
        description: "Area where selling pressure may start to build.",
        color: "text-violet-300",
        barClass: "bg-violet-400",
        Icon: TrendingUp,
      },
      {
        key: "current",
        label: "Current",
        value: currentSetupPrice,
        description: livePrice ? "Streaming live price." : "Latest chart close.",
        color: "text-white",
        barClass: "bg-white",
        Icon: BarChart3,
      },
      {
        key: "support",
        label: "Support",
        value: latestAnalysis.support_level,
        description: "Area where buyers may start defending the price.",
        color: "text-sky-300",
        barClass: "bg-sky-400",
        Icon: TrendingDown,
      },
      {
        key: "stop",
        label: "Stop",
        value: latestAnalysis.stop_loss,
        description: "Suggested invalidation level if the setup fails.",
        color: "text-red-400",
        barClass: "bg-red-400",
        Icon: ShieldAlert,
      },
    ].filter((level): level is {
      key: string
      label: string
      value: number
      description: string
      color: string
      barClass: string
      Icon: LucideIcon
    } => typeof level.value === "number" && Number.isFinite(level.value))
    : []
  const tradeRange = tradeSetupLevels.reduce(
    (range, level) => ({
      min: Math.min(range.min, level.value),
      max: Math.max(range.max, level.value),
    }),
    { min: Number.POSITIVE_INFINITY, max: Number.NEGATIVE_INFINITY }
  )
  const tradeRangeSpan =
    Number.isFinite(tradeRange.max) && Number.isFinite(tradeRange.min)
      ? Math.max(tradeRange.max - tradeRange.min, 1)
      : 1
  const riskRewardValue = latestAnalysis ? getRiskRewardValue(latestAnalysis, currentSetupPrice) : null

  return (
    <div className="app-page">
      {/* Header */}
      <div className="mb-6 flex flex-col gap-4 lg:flex-row lg:items-start lg:justify-between">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.back()}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="app-page-title">{stock.symbol}</h1>
              {latestAnalysis && (
                <Badge variant="outline" className={getSignalColor(latestAnalysis.signal)}>
                  {latestAnalysis.signal.toUpperCase()}
                </Badge>
              )}
              <Badge variant="outline" className="border-cyan-200 bg-cyan-50 text-cyan-700 dark:border-cyan-500/20 dark:bg-cyan-500/10 dark:text-cyan-200">
                {getAnalysisModeLabel(selectedMode)}
              </Badge>
            </div>
            <p className="app-page-subtitle">{stock.name}</p>
          </div>
        </div>
        <div className="flex flex-col gap-3 lg:items-end">
          <AnalysisModeToggle mode={selectedMode} onModeChange={setSelectedMode} compact />
          <div className="flex flex-wrap items-center gap-4 lg:justify-end">
            <LivePriceTicker
              symbol={stock.symbol}
              currency={priceDisplayCurrency}
              livePriceOverride={livePrice}
            />
            <StockChatButton symbol={stock.symbol} hasAnalysis={hasAnalysis} />
            <ChatContextBadge analysisCount={analysisCount} showCount={true} />
            <Button onClick={handleAnalyze} disabled={analyzing}>
              {analyzing ? (
                <Loader2 className="h-4 w-4 mr-2 animate-spin" />
              ) : (
                <Play className="h-4 w-4 mr-2" />
              )}
              Run {getAnalysisModeLabel(selectedMode)}
            </Button>
          </div>
          {analyzeState !== "idle" && (
            <p className={`text-xs ${getAnalyzeStatusClass(analyzeState)}`}>{analyzeMessage}</p>
          )}
        </div>
      </div>

      <div className="mb-4 grid gap-3 md:grid-cols-2">
        {([
          { mode: "intraday" as const, latest: latestIntradayAnalysis, count: intradayAnalyses.length },
          { mode: "longterm" as const, latest: latestLongtermAnalysis, count: longtermAnalyses.length },
        ]).map(({ mode, latest, count }) => {
          const isActive = selectedMode === mode

          return (
            <button
              key={mode}
              type="button"
              onClick={() => setSelectedMode(mode)}
              className={`rounded-2xl border p-4 text-left transition-colors ${
                isActive
                  ? "border-cyan-400/50 bg-cyan-500/10 shadow-[0_0_0_1px_rgba(34,211,238,0.18)]"
                  : "border-border/70 bg-card/70 hover:bg-accent/35"
              }`}
            >
              <div className="flex items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium">Latest {getAnalysisModeLabel(mode)}</p>
                  <p className="text-xs text-muted-foreground">
                    {count === 0 ? "No saved analysis yet" : `${count} saved ${count === 1 ? "analysis" : "analyses"}`}
                  </p>
                </div>
                {latest ? (
                  <Badge variant="outline" className={getSignalColor(latest.signal)}>
                    {latest.signal.toUpperCase()}
                  </Badge>
                ) : (
                  <Badge variant="outline" className="border-dashed text-muted-foreground">
                    Empty
                  </Badge>
                )}
              </div>
              <div className="mt-4">
                {latest ? (
                  <>
                    <p className="text-sm font-medium text-foreground">
                      {Math.round(latest.confidence * 100)}% confidence
                    </p>
                    <p className="mt-1 text-sm text-muted-foreground">
                      {formatRelativeTime(latest.created_at)} · {latest.technical_summary || latest.reasoning}
                    </p>
                  </>
                ) : (
                  <p className="text-sm text-muted-foreground">
                    Run a {getAnalysisModeLabel(mode).toLowerCase()} analysis to create a mode-specific setup and history.
                  </p>
                )}
              </div>
            </button>
          )
        })}
      </div>

      {/* AI Analysis Panel - Collapsible */}
      <div className="mb-4">
        <button
          onClick={() => setAiPanelOpen(!aiPanelOpen)}
          className="w-full flex items-center justify-between p-3 rounded-lg border bg-card hover:bg-accent/50 transition-colors"
        >
          <div className="flex items-center gap-3">
            <Brain className="h-4 w-4 text-purple-500" />
            <span className="font-medium text-sm">AI Analysis · {getAnalysisModeLabel(selectedMode)}</span>
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
              analysis={latestAnalysis ?? null}
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
      <div className="grid gap-4">
        <Card className="border-cyan-500/20 bg-gradient-to-br from-cyan-500/10 via-background to-background">
          <CardHeader>
            <div className="flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
              <div>
                <CardTitle>Trade Setup</CardTitle>
                <CardDescription>
                  {latestAnalysis
                    ? `${getAnalysisModeLabel(latestAnalysis.mode)} setup from ${formatRelativeTime(latestAnalysis.created_at)} with ${Math.round(latestAnalysis.confidence * 100)}% confidence`
                    : "AI-generated trade levels and context"}
                </CardDescription>
              </div>
              {latestAnalysis && (
                <div className="flex flex-wrap items-center gap-2">
                  <Badge variant="outline" className={getSignalColor(latestAnalysis.signal)}>
                    {latestAnalysis.signal.toUpperCase()}
                  </Badge>
                  <Badge variant="outline" className="border-white/10 bg-white/5 text-white/80">
                    {getAnalysisModeLabel(latestAnalysis.mode)}
                  </Badge>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent>
            {latestAnalysis ? (
              <div className="space-y-5">
                <div className="grid gap-3 md:grid-cols-2 xl:grid-cols-4">
                  <div className="rounded-xl border border-white/10 bg-black/30 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-muted-foreground">Current Price</div>
                    <div className="mt-2 text-2xl font-semibold text-white">
                      {formatMoney(currentSetupPrice, priceDisplayCurrency)}
                    </div>
                    <div className="mt-1 text-xs text-muted-foreground">
                      {livePrice ? "Live market price" : "Using latest chart close"}
                    </div>
                  </div>
                  <div className="rounded-xl border border-green-500/20 bg-green-500/10 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-green-200/80">Target Move</div>
                    <div className="mt-2 text-2xl font-semibold text-green-300">
                      {formatMoney(latestAnalysis.target_price, priceDisplayCurrency)}
                    </div>
                    <div className="mt-1 text-xs text-green-100/80">
                      {formatPriceRelation(currentSetupPrice, latestAnalysis.target_price)}
                    </div>
                  </div>
                  <div className="rounded-xl border border-red-500/20 bg-red-500/10 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-red-200/80">Stop Buffer</div>
                    <div className="mt-2 text-2xl font-semibold text-red-300">
                      {formatMoney(latestAnalysis.stop_loss, priceDisplayCurrency)}
                    </div>
                    <div className="mt-1 text-xs text-red-100/80">
                      {formatPriceRelation(currentSetupPrice, latestAnalysis.stop_loss)}
                    </div>
                  </div>
                  <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-4">
                    <div className="text-xs uppercase tracking-[0.18em] text-cyan-100/80">Risk / Reward</div>
                    <div className="mt-2 text-2xl font-semibold text-cyan-200">
                      {riskRewardValue !== null ? `${riskRewardValue.toFixed(1)}:1` : "N/A"}
                    </div>
                    <div className="mt-1 text-xs text-cyan-50/80">
                      {riskRewardValue !== null
                        ? riskRewardValue >= 2
                          ? "Favorable setup"
                          : "Moderate setup"
                        : "Needs target and stop"}
                    </div>
                  </div>
                </div>

                <div className="rounded-xl border border-white/10 bg-black/25 p-4">
                  <div className="mb-4 flex items-center justify-between">
                    <div>
                      <div className="text-sm font-medium text-white">Level Ladder</div>
                      <div className="text-xs text-muted-foreground">
                        Higher rows sit above lower rows in the current setup.
                      </div>
                    </div>
                    <div className="text-right text-xs text-muted-foreground">
                      <div>Range</div>
                      <div className="text-white/80">
                        {formatMoney(tradeRange.min, priceDisplayCurrency)} to {formatMoney(tradeRange.max, priceDisplayCurrency)}
                      </div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {[...tradeSetupLevels]
                      .sort((a, b) => b.value - a.value)
                      .map((level) => {
                        const position = ((level.value - tradeRange.min) / tradeRangeSpan) * 100

                        return (
                          <div key={level.key} className="grid gap-2 rounded-lg border border-white/5 bg-white/[0.03] p-3 lg:grid-cols-[120px_minmax(0,1fr)_180px] lg:items-center">
                            <div className={`flex items-center gap-2 text-sm font-medium ${level.color}`}>
                              <level.Icon className="h-4 w-4" />
                              {level.label}
                            </div>
                            <div className="space-y-2">
                              <div className="relative h-2 rounded-full bg-white/10">
                                <div
                                  className={`absolute top-0 h-2 w-4 -translate-x-1/2 rounded-full shadow-[0_0_12px_-2px_rgba(255,255,255,0.45)] ${level.barClass}`}
                                  style={{ left: `${Math.min(Math.max(position, 2), 98)}%` }}
                                />
                              </div>
                              <p className="text-xs text-muted-foreground">{level.description}</p>
                            </div>
                            <div className="text-left lg:text-right">
                              <div className={`text-lg font-semibold ${level.color}`}>
                                {formatMoney(level.value, priceDisplayCurrency)}
                              </div>
                              <div className="text-xs text-muted-foreground">
                                {formatPriceRelation(currentSetupPrice, level.value)}
                              </div>
                            </div>
                          </div>
                        )
                      })}
                  </div>
                </div>

                <div className="grid gap-3 lg:grid-cols-2">
                  <div className="rounded-xl border border-white/10 bg-muted/20 p-4">
                    <div className="text-sm font-medium text-white">Setup Summary</div>
                    <p className="mt-2 text-sm leading-relaxed text-muted-foreground">
                      {latestAnalysis.technical_summary || latestAnalysis.reasoning || "No detailed summary available yet."}
                    </p>
                  </div>
                  <div className="rounded-xl border border-white/10 bg-muted/20 p-4">
                    <div className="text-sm font-medium text-white">How to Read It</div>
                    <div className="mt-2 space-y-2 text-sm text-muted-foreground">
                      <p>
                        Target and resistance show where upside may slow or profits may be taken.
                      </p>
                      <p>
                        Support and stop show where the setup starts weakening and risk should be reassessed.
                      </p>
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-muted-foreground text-center py-4">
                No {getAnalysisModeLabel(selectedMode).toLowerCase()} trade setup available yet.
              </p>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Analysis History */}
      {selectedModeCount > 1 && (
        <Card className="mt-4">
          <CardHeader>
            <CardTitle>{getAnalysisModeLabel(selectedMode)} History</CardTitle>
            <CardDescription>
              Previous {selectedModeCount - 1} {selectedModeCount - 1 === 1 ? "analysis" : "analyses"} for this mode
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {selectedAnalyses.slice(1).map((analysis) => (
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
                    <Badge variant="outline" className="border-cyan-200 bg-cyan-50 text-cyan-700 dark:border-cyan-500/20 dark:bg-cyan-500/10 dark:text-cyan-200">
                      {getAnalysisModeLabel(analysis.mode)}
                    </Badge>
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

    </div>
  )
}
