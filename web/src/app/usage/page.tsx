"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Skeleton } from "@/components/ui/skeleton"
import { RefreshCw, Zap, Clock, TrendingUp } from "lucide-react"

interface ServiceData {
  count: number
  tokens: number
  limit: number | null
  period: string
  unit: string
  percentage: number
  last_reset: string | null
}

interface UsageResponse {
  services: Record<string, ServiceData>
  created_at: string | null
}

const serviceInfo: Record<string, { name: string; description: string; icon: string }> = {
  zai: { name: "ZAI (GLM-4.7)", description: "Primary LLM (GLM-4.7 200K)", icon: "🚀" },
  gemini: { name: "Gemini", description: "Backup LLM (Gemini 2.5 Pro)", icon: "✨" },
  cohere: { name: "Cohere", description: "News embeddings", icon: "📰" },
  finnhub: { name: "Finnhub", description: "Stock news & data", icon: "📊" },
  ollama: { name: "Ollama", description: "Local LLM (unlimited)", icon: "🏠" },
}

function formatNumber(num: number): string {
  if (num >= 1000000) return `${(num / 1000000).toFixed(1)}M`
  if (num >= 1000) return `${(num / 1000).toFixed(1)}K`
  return num.toLocaleString()
}

function formatTimeAgo(dateStr: string | null): string {
  if (!dateStr) return "Never"
  const date = new Date(dateStr)
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60))
  const diffDays = Math.floor(diffHours / 24)

  if (diffDays > 0) return `${diffDays}d ago`
  if (diffHours > 0) return `${diffHours}h ago`
  return "Recently"
}

export default function UsagePage() {
  const [usage, setUsage] = useState<UsageResponse | null>(null)
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)

  async function fetchUsage() {
    try {
      const response = await fetch("/api/usage")
      const data = await response.json()
      setUsage(data)
    } catch (error) {
      console.error("Error fetching usage:", error)
    } finally {
      setLoading(false)
      setRefreshing(false)
    }
  }

  useEffect(() => {
    fetchUsage()
    const interval = setInterval(fetchUsage, 30000)
    return () => clearInterval(interval)
  }, [])

  const handleRefresh = () => {
    setRefreshing(true)
    fetchUsage()
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="mb-8">
          <Skeleton className="h-9 w-32 mb-2" />
          <Skeleton className="h-5 w-48" />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-48" />
          ))}
        </div>
      </div>
    )
  }

  // Calculate totals
  const totalRequests = Object.values(usage?.services || {}).reduce((sum, s) => sum + s.count, 0)
  const totalTokens = Object.values(usage?.services || {}).reduce((sum, s) => sum + s.tokens, 0)

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">API Usage</h1>
          <p className="text-muted-foreground">Track your API consumption and token usage</p>
        </div>
        <Button variant="outline" onClick={handleRefresh} disabled={refreshing}>
          <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? "animate-spin" : ""}`} />
          Refresh
        </Button>
      </div>

      {/* Summary Cards */}
      <div className="grid gap-4 md:grid-cols-3 mb-6">
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-blue-500/10">
                <TrendingUp className="h-5 w-5 text-blue-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Requests</p>
                <p className="text-2xl font-bold">{formatNumber(totalRequests)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-purple-500/10">
                <Zap className="h-5 w-5 text-purple-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Total Tokens</p>
                <p className="text-2xl font-bold">{formatNumber(totalTokens)}</p>
              </div>
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="pt-6">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-lg bg-green-500/10">
                <Clock className="h-5 w-5 text-green-500" />
              </div>
              <div>
                <p className="text-sm text-muted-foreground">Last Reset</p>
                <p className="text-2xl font-bold">
                  {formatTimeAgo(usage?.services?.zai?.last_reset || null)}
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Service Cards */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {Object.entries(usage?.services || {}).map(([key, service]) => {
          const info = serviceInfo[key] || { name: key, description: "", icon: "📦" }
          const isUnlimited = service.limit === null

          return (
            <Card key={key}>
              <CardHeader className="pb-2">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="text-xl">{info.icon}</span>
                    <CardTitle className="text-lg">{info.name}</CardTitle>
                  </div>
                  <span className="text-xs text-muted-foreground capitalize px-2 py-1 bg-muted rounded">
                    {service.period}
                  </span>
                </div>
                <CardDescription>{info.description}</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {/* Request count */}
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-muted-foreground">Requests</span>
                      <span className="font-medium">
                        {service.count.toLocaleString()}
                        {!isUnlimited && ` / ${service.limit?.toLocaleString()}`}
                      </span>
                    </div>
                    {!isUnlimited && (
                      <div className="h-2 rounded-full bg-muted overflow-hidden">
                        <div
                          className={`h-full transition-all duration-500 ${service.percentage > 90
                              ? "bg-red-500"
                              : service.percentage > 75
                                ? "bg-yellow-500"
                                : "bg-green-500"
                            }`}
                          style={{ width: `${Math.min(service.percentage, 100)}%` }}
                        />
                      </div>
                    )}
                  </div>

                  {/* Token count */}
                  <div className="flex justify-between text-sm">
                    <span className="text-muted-foreground flex items-center gap-1">
                      <Zap className="h-3 w-3" /> Tokens
                    </span>
                    <span className="font-medium text-purple-500">
                      {formatNumber(service.tokens)}
                    </span>
                  </div>

                  {/* Status */}
                  {!isUnlimited && service.percentage >= 75 && (
                    <p className={`text-xs ${service.percentage >= 90 ? "text-red-500" : "text-yellow-500"}`}>
                      {service.percentage >= 90 ? "⚠️ Critical: " : "⚡ Warning: "}
                      {service.limit && Math.round(service.limit - service.count).toLocaleString()} remaining
                    </p>
                  )}
                </div>
              </CardContent>
            </Card>
          )
        })}
      </div>

      {/* CLI Commands */}
      <Card className="mt-6">
        <CardHeader>
          <CardTitle className="text-lg">CLI Commands</CardTitle>
          <CardDescription>Manage usage from the terminal</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2">
          <div className="flex items-center gap-2">
            <code className="bg-muted px-3 py-1.5 rounded text-sm font-mono">
              python3 main.py usage
            </code>
            <span className="text-muted-foreground text-sm">View usage in terminal</span>
          </div>
          <div className="flex items-center gap-2">
            <code className="bg-muted px-3 py-1.5 rounded text-sm font-mono">
              python3 main.py usage --reset
            </code>
            <span className="text-muted-foreground text-sm">Reset all counters</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
