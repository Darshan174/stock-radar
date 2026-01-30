"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
} from "@/components/ui/dialog"
import {
  Database,
  TrendingUp,
  TrendingDown,
  Minus,
  Newspaper,
  FileText,
  AlertTriangle,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Clock,
  Zap,
  BarChart3,
  BookOpen,
  Loader2,
  ExternalLink,
  Info,
} from "lucide-react"
import { cn } from "@/lib/utils"
import { RAGBadge, RAGSourcesIndicator } from "./rag-badge"


interface RAGContextResponse {
  success: boolean
  symbol: string
  context: {
    similarAnalyses: Array<{
      id: number
      symbol: string
      signal: string
      confidence: number
      reasoning: string
      createdAt: string
      similarity: number
    }>
    relatedSignals: Array<{
      id: number
      symbol: string
      signal: string
      priceAtSignal: number
      reason: string
      importance: string
      similarity: number
    }>
    relevantNews: Array<{
      id: number
      headline: string
      summary: string
      source: string
      sentimentLabel: string
      publishedAt: string
      similarity: number
    }>
    knowledgeBase: Array<{
      id: number
      title: string
      content: string
      category: string
      similarity: number
    }>
    sectorSentiment?: {
      sector: string
      sentimentSummary: string
      positiveCount: number
      negativeCount: number
    } | null
  }
  metadata: {
    totalResults: number
    sourcesSearched: string[]
    retrievalTimeMs: number
  }
  error?: string
}

interface RAGInsightsPanelProps {
  symbol: string
  className?: string
  defaultExpanded?: boolean
  autoRefresh?: boolean
  refreshInterval?: number // in milliseconds
}

function getSignalColor(signal: string) {
  switch (signal?.toLowerCase()) {
    case "strong_buy":
      return "text-emerald-400 bg-emerald-500/10 border-emerald-500/30"
    case "buy":
      return "text-green-400 bg-green-500/10 border-green-500/30"
    case "hold":
      return "text-yellow-400 bg-yellow-500/10 border-yellow-500/30"
    case "sell":
      return "text-orange-400 bg-orange-500/10 border-orange-500/30"
    case "strong_sell":
      return "text-red-400 bg-red-500/10 border-red-500/30"
    default:
      return "text-muted-foreground bg-muted border-border"
  }
}

function getSignalIcon(signal: string) {
  switch (signal?.toLowerCase()) {
    case "strong_buy":
    case "buy":
      return <TrendingUp className="h-3 w-3" />
    case "strong_sell":
    case "sell":
      return <TrendingDown className="h-3 w-3" />
    default:
      return <Minus className="h-3 w-3" />
  }
}

function getSentimentColor(sentiment: string) {
  switch (sentiment?.toLowerCase()) {
    case "positive":
      return "text-green-400 bg-green-500/10 border-green-500/30"
    case "negative":
      return "text-red-400 bg-red-500/10 border-red-500/30"
    default:
      return "text-yellow-400 bg-yellow-500/10 border-yellow-500/30"
  }
}

function getImportanceColor(importance: string) {
  switch (importance?.toLowerCase()) {
    case "high":
      return "text-red-400"
    case "medium":
      return "text-yellow-400"
    default:
      return "text-muted-foreground"
  }
}

function SimilarityBadge({ similarity }: { similarity: number }) {
  const percent = Math.round(similarity * 100)
  const color =
    percent >= 80
      ? "text-green-400"
      : percent >= 60
        ? "text-yellow-400"
        : "text-muted-foreground"

  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span className={cn("text-xs font-medium", color)}>
            {percent}% match
          </span>
        </TooltipTrigger>
        <TooltipContent side="top" className="text-xs">
          Semantic similarity to your current context
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

// Info tooltip helper
function InfoTooltip({ text }: { text: string }) {
  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <Info className="h-3.5 w-3.5 text-muted-foreground/50 hover:text-muted-foreground cursor-help ml-1" />
        </TooltipTrigger>
        <TooltipContent side="top" className="max-w-xs text-xs">
          {text}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

// Expandable text component with click-to-expand dialog
function ExpandableText({
  text,
  title = "Full Details",
  maxLines = 2,
  className
}: {
  text: string
  title?: string
  maxLines?: number
  className?: string
}) {
  const [showDialog, setShowDialog] = useState(false)

  // Rough estimate if text is likely to be truncated
  const isLongText = text.length > 120

  return (
    <>
      <p
        className={cn(
          "text-sm text-muted-foreground",
          isLongText && "cursor-pointer hover:text-foreground transition-colors",
          maxLines === 2 && "line-clamp-2",
          maxLines === 3 && "line-clamp-3",
          className
        )}
        onClick={() => isLongText && setShowDialog(true)}
      >
        {text}
        {isLongText && <span className="text-primary/70 ml-1">...</span>}
      </p>

      {/* Dialog for full text */}
      <Dialog open={showDialog} onOpenChange={setShowDialog}>
        <DialogContent className="max-w-2xl max-h-[80vh] overflow-auto">
          <DialogHeader>
            <DialogTitle>{title}</DialogTitle>
            <DialogDescription>Click outside or press escape to close</DialogDescription>
          </DialogHeader>
          <p className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
            {text}
          </p>
        </DialogContent>
      </Dialog>
    </>
  )
}

export function RAGInsightsPanel({
  symbol,
  className,
  defaultExpanded = true,
  autoRefresh = false,
  refreshInterval = 300000, // 5 minutes
}: RAGInsightsPanelProps) {
  const [data, setData] = useState<RAGContextResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const [isExpanded, setIsExpanded] = useState(defaultExpanded)
  const [activeTab, setActiveTab] = useState<"analyses" | "signals" | "news" | "knowledge">("analyses")

  const fetchRAGContext = async () => {
    if (!symbol) return

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(
        `/api/stock-context?symbol=${encodeURIComponent(symbol)}&sector=true`
      )
      const result: RAGContextResponse = await response.json()

      if (!result.success) {
        setError(result.error || "Failed to retrieve RAG context")
      } else {
        setData(result)
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Failed to fetch RAG context")
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    fetchRAGContext()

    if (autoRefresh && refreshInterval > 0) {
      const interval = setInterval(fetchRAGContext, refreshInterval)
      return () => clearInterval(interval)
    }
  }, [symbol, autoRefresh, refreshInterval])

  const hasData = data && data.metadata.totalResults > 0
  const tabs = [
    {
      id: "analyses" as const,
      label: "Analyses",
      icon: BarChart3,
      count: data?.context.similarAnalyses.length || 0,
      info: "Similar past analyses based on current technical setup and market conditions.",
    },
    {
      id: "signals" as const,
      label: "Signals",
      icon: Zap,
      count: data?.context.relatedSignals.length || 0,
      info: "Historical trading signals from similar market patterns and technical setups.",
    },
    {
      id: "news" as const,
      label: "News",
      icon: Newspaper,
      count: data?.context.relevantNews.length || 0,
      info: "Relevant news articles collected from Yahoo Finance and Finnhub during past analyses.",
    },
    {
      id: "knowledge" as const,
      label: "Knowledge",
      icon: BookOpen,
      count: data?.context.knowledgeBase.length || 0,
      info: "Custom knowledge entries and market insights stored in the knowledge base.",
    },
  ]


  if (loading) {
    return (
      <Card className={cn("border-purple-500/20", className)}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <Database className="h-5 w-5 text-purple-400 animate-pulse" />
              </div>
              <div>
                <CardTitle className="text-lg flex items-center gap-2">
                  RAG Insights
                  <Loader2 className="h-4 w-4 animate-spin text-purple-400" />
                </CardTitle>
                <CardDescription>Retrieving historical context...</CardDescription>
              </div>
            </div>
          </div>
        </CardHeader>
        <CardContent className="space-y-3">
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
          <Skeleton className="h-20 w-full" />
        </CardContent>
      </Card>
    )
  }

  if (error) {
    return (
      <Card className={cn("border-red-500/20", className)}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-red-500/10">
                <AlertTriangle className="h-5 w-5 text-red-400" />
              </div>
              <div>
                <CardTitle className="text-lg">RAG Insights</CardTitle>
                <CardDescription className="text-red-400">{error}</CardDescription>
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={fetchRAGContext}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Retry
            </Button>
          </div>
        </CardHeader>
      </Card>
    )
  }

  if (!hasData) {
    return (
      <Card className={cn("border-dashed border-purple-500/20", className)}>
        <CardHeader className="pb-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <Database className="h-5 w-5 text-purple-400" />
              </div>
              <div>
                <CardTitle className="text-lg flex items-center gap-2">
                  RAG Insights
                  <RAGBadge isActive={false} size="sm" />
                </CardTitle>
                <CardDescription>
                  No historical context available yet
                </CardDescription>
              </div>
            </div>
            <Button variant="outline" size="sm" onClick={fetchRAGContext}>
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh
            </Button>
          </div>
        </CardHeader>
        <CardContent className="text-center py-6">
          <p className="text-muted-foreground text-sm">
            Run analyses for this stock to build up RAG context that can provide
            historical insights and pattern matching.
          </p>
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className={cn("border-purple-500/30 transition-all", className)}>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="p-2 rounded-lg bg-purple-500/10">
              <Database className="h-5 w-5 text-purple-400" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <CardTitle className="text-lg">RAG Insights</CardTitle>
                <RAGBadge
                  isActive={true}
                  size="sm"
                  totalResults={data?.metadata.totalResults}
                  retrievalTimeMs={data?.metadata.retrievalTimeMs}
                  sourcesSearched={data?.metadata.sourcesSearched}
                />
              </div>
              <CardDescription className="flex items-center gap-2">
                <span>
                  {data?.metadata.totalResults} results from historical context
                </span>
                <span className="text-muted-foreground/50">|</span>
                <Clock className="h-3 w-3" />
                <span>{data?.metadata.retrievalTimeMs}ms</span>
              </CardDescription>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Button variant="ghost" size="sm" onClick={fetchRAGContext}>
              <RefreshCw className="h-4 w-4" />
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsExpanded(!isExpanded)}
            >
              {isExpanded ? (
                <ChevronUp className="h-4 w-4" />
              ) : (
                <ChevronDown className="h-4 w-4" />
              )}
            </Button>
          </div>
        </div>

        {/* Sector Sentiment Summary */}
        {data?.context.sectorSentiment && isExpanded && (
          <div className="mt-3 p-2 rounded-lg bg-muted/50 flex items-center justify-between">
            <div className="flex items-center gap-2 text-sm">
              <Info className="h-4 w-4 text-blue-400" />
              <span className="text-muted-foreground">Sector:</span>
              <span className="font-medium">{data.context.sectorSentiment.sector}</span>
            </div>
            <span className="text-sm text-muted-foreground">
              {data.context.sectorSentiment.sentimentSummary}
            </span>
          </div>
        )}
      </CardHeader>

      {isExpanded && (
        <CardContent className="space-y-4">
          {/* Tab Navigation */}
          <div className="flex gap-1 p-1 bg-muted/50 rounded-lg">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={cn(
                  "flex-1 flex items-center justify-center gap-2 px-3 py-2 rounded-md text-sm font-medium transition-colors",
                  activeTab === tab.id
                    ? "bg-background shadow-sm text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                <tab.icon className="h-4 w-4" />
                {tab.label}
                <InfoTooltip text={tab.info} />
                {tab.count > 0 && (
                  <Badge variant="secondary" className="text-xs h-5 px-1.5">
                    {tab.count}
                  </Badge>
                )}
              </button>
            ))}
          </div>

          {/* Tab Content */}
          <div className="min-h-[200px]">
            {/* Similar Analyses */}
            {activeTab === "analyses" && (
              <div className="space-y-3">
                {data?.context.similarAnalyses.length === 0 ? (
                  <div className="text-center py-8">
                    <BarChart3 className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-muted-foreground">No similar analyses found</p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      Run more analyses to build up the historical context database.
                    </p>
                  </div>
                ) : (
                  data?.context.similarAnalyses.map((analysis) => (
                    <div
                      key={analysis.id}
                      className="p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Badge
                            variant="outline"
                            className={cn("text-xs", getSignalColor(analysis.signal))}
                          >
                            {getSignalIcon(analysis.signal)}
                            <span className="ml-1">{analysis.signal.toUpperCase()}</span>
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {Math.round(analysis.confidence * 100)}% confidence
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <SimilarityBadge similarity={analysis.similarity} />
                          <span className="text-xs text-muted-foreground">
                            {analysis.createdAt}
                          </span>
                        </div>
                      </div>
                      <ExpandableText
                        text={analysis.reasoning}
                        title="Analysis Reasoning"
                        maxLines={2}
                      />
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Related Signals */}
            {activeTab === "signals" && (
              <div className="space-y-3">
                {data?.context.relatedSignals.length === 0 ? (
                  <div className="text-center py-8">
                    <Zap className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-muted-foreground">No related signals found</p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      Signals are generated during stock analyses based on technical patterns.
                    </p>
                  </div>
                ) : (
                  data?.context.relatedSignals.map((signal) => (
                    <div
                      key={signal.id}
                      className="p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Badge
                            variant="outline"
                            className={cn("text-xs", getSignalColor(signal.signal))}
                          >
                            {getSignalIcon(signal.signal)}
                            <span className="ml-1">{signal.signal.toUpperCase()}</span>
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            @ {signal.priceAtSignal?.toFixed(2) || "N/A"}
                          </span>
                          <span
                            className={cn(
                              "text-xs",
                              getImportanceColor(signal.importance)
                            )}
                          >
                            {signal.importance}
                          </span>
                        </div>
                        <SimilarityBadge similarity={signal.similarity} />
                      </div>
                      <ExpandableText
                        text={signal.reason}
                        title="Signal Reason"
                        maxLines={2}
                      />
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Relevant News */}
            {activeTab === "news" && (
              <div className="space-y-3">
                {data?.context.relevantNews.length === 0 ? (
                  <div className="text-center py-8">
                    <Newspaper className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-muted-foreground">No relevant news found</p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      News is collected from Yahoo Finance and Finnhub during analyses.
                      Run more analyses to build up the news database.
                    </p>
                  </div>
                ) : (
                  data?.context.relevantNews.map((news) => (
                    <div
                      key={news.id}
                      className="p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <Badge
                            variant="outline"
                            className={cn(
                              "text-xs",
                              getSentimentColor(news.sentimentLabel)
                            )}
                          >
                            {news.sentimentLabel}
                          </Badge>
                          <span className="text-xs text-muted-foreground">
                            {news.source}
                          </span>
                        </div>
                        <div className="flex items-center gap-2">
                          <SimilarityBadge similarity={news.similarity} />
                          <span className="text-xs text-muted-foreground">
                            {news.publishedAt}
                          </span>
                        </div>
                      </div>
                      <p className="text-sm font-medium mb-1">{news.headline}</p>
                      {news.summary && (
                        <ExpandableText
                          text={news.summary}
                          title="News Summary"
                          maxLines={2}
                        />
                      )}
                    </div>
                  ))
                )}
              </div>
            )}

            {/* Knowledge Base */}
            {activeTab === "knowledge" && (
              <div className="space-y-3">
                {data?.context.knowledgeBase.length === 0 ? (
                  <div className="text-center py-8">
                    <BookOpen className="h-8 w-8 mx-auto text-muted-foreground mb-2" />
                    <p className="text-muted-foreground">No knowledge base entries found</p>
                    <p className="text-xs text-muted-foreground/70 mt-1">
                      Add custom market insights and notes to the knowledge base
                      to enhance future analyses.
                    </p>
                  </div>
                ) : (
                  data?.context.knowledgeBase.map((kb) => (
                    <div
                      key={kb.id}
                      className="p-3 rounded-lg border bg-card hover:bg-muted/50 transition-colors"
                    >
                      <div className="flex items-center justify-between mb-2">
                        <div className="flex items-center gap-2">
                          <FileText className="h-4 w-4 text-blue-400" />
                          <span className="text-sm font-medium">{kb.title}</span>
                        </div>
                        <div className="flex items-center gap-2">
                          <Badge variant="outline" className="text-xs">
                            {kb.category}
                          </Badge>
                          <SimilarityBadge similarity={kb.similarity} />
                        </div>
                      </div>
                      <ExpandableText
                        text={kb.content}
                        title={kb.title}
                        maxLines={3}
                      />
                    </div>
                  ))
                )}
              </div>
            )}
          </div>

          {/* Sources Footer */}
          <div className="pt-3 border-t">
            <RAGSourcesIndicator sources={data?.metadata.sourcesSearched || []} />
          </div>
        </CardContent>
      )}
    </Card>
  )
}
