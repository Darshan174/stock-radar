"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { TrendingUp, TrendingDown, Minus, Zap, BarChart3, Activity, MessageSquare, CreditCard } from "lucide-react"
import { supabase, Analysis, Stock, hasSupabaseEnv } from "@/lib/supabase"
import { ProjectIntro } from "@/components/project-intro"
import { useSidebar } from "@/providers/sidebar-provider"
import Link from "next/link"
import { cn } from "@/lib/utils"

interface SignalWithStock extends Analysis {
  stocks: Stock
}

const INTRO_SEEN_KEY = "stock_radar_intro_seen_v1"

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

function getSignalIcon(signal: string) {
  switch (signal) {
    case "strong_buy":
    case "buy":
      return <TrendingUp className="h-4 w-4" />
    case "strong_sell":
    case "sell":
      return <TrendingDown className="h-4 w-4" />
    default:
      return <Minus className="h-4 w-4" />
  }
}

function timeAgo(date: string) {
  const seconds = Math.floor((new Date().getTime() - new Date(date).getTime()) / 1000)
  if (seconds < 60) return `${seconds}s ago`
  const minutes = Math.floor(seconds / 60)
  if (minutes < 60) return `${minutes}m ago`
  const hours = Math.floor(minutes / 60)
  if (hours < 24) return `${hours}h ago`
  return `${Math.floor(hours / 24)}d ago`
}

export default function DashboardPage() {
  const [recentSignals, setRecentSignals] = useState<SignalWithStock[]>([])
  const [stockCount, setStockCount] = useState(0)
  const [signalsToday, setSignalsToday] = useState(0)
  const [dataError, setDataError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [introLoaded, setIntroLoaded] = useState(false)
  const [showIntro, setShowIntro] = useState(false)
  const { setHidden } = useSidebar()

  useEffect(() => {
    try {
      const seen = window.sessionStorage.getItem(INTRO_SEEN_KEY) === "1"
      setShowIntro(!seen)
      setHidden(!seen)
    } catch {
      setShowIntro(false)
      setHidden(false)
    } finally {
      setIntroLoaded(true)
    }
  }, [setHidden])

  useEffect(() => {
    async function fetchData() {
      try {
        if (!hasSupabaseEnv) {
          throw new Error("NEXT_PUBLIC_SUPABASE_URL / NEXT_PUBLIC_SUPABASE_ANON_KEY not configured")
        }
        setDataError(null)
        // Fetch recent analyses with stock info
        const { data: analyses, error: analysesError } = await supabase
          .from("analysis")
          .select("*, stocks(*)")
          .order("created_at", { ascending: false })
          .limit(5)

        if (analysesError) {
          throw analysesError
        }

        if (analyses) {
          setRecentSignals(analyses as SignalWithStock[])
        }

        // Count stocks
        const { count: stocks, error: stocksError } = await supabase
          .from("stocks")
          .select("*", { count: "exact", head: true })

        if (stocksError) {
          throw stocksError
        }

        setStockCount(stocks || 0)

        // Count today's signals
        const today = new Date()
        today.setHours(0, 0, 0, 0)
        const { count: todaySignals, error: todaySignalsError } = await supabase
          .from("analysis")
          .select("*", { count: "exact", head: true })
          .gte("created_at", today.toISOString())

        if (todaySignalsError) {
          throw todaySignalsError
        }

        setSignalsToday(todaySignals || 0)

      } catch (error) {
        console.error("Error fetching data:", error)
        const message = error instanceof Error ? error.message : "Unknown data loading error"
        setDataError(message)
      } finally {
        setLoading(false)
      }
    }

    fetchData()

    // Subscribe to real-time updates
    const channel = supabase
      .channel("dashboard")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "analysis" }, () => {
        // Refetch when new analysis is added
        fetchData()
      })
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [])

  function handleEnterDashboard() {
    try {
      window.sessionStorage.setItem(INTRO_SEEN_KEY, "1")
    } catch {
      // No-op if storage is unavailable.
    }
    setShowIntro(false)
    setHidden(false)
  }

  const stats = [
    { label: "Active Stocks", value: stockCount.toString(), icon: Activity, href: "/stocks" },
    { label: "Signals Today", value: signalsToday.toString(), icon: Zap, href: "/signals" },
    { label: "AI Chat", value: "Ask Assistant", icon: MessageSquare, href: "/chat" },
    { label: "Aptos Payment", value: "x402 Demo", icon: CreditCard, href: "/x402-demo" },
    { label: "API Usage", value: "View Details", icon: BarChart3, href: "/usage" },
  ]

  if (!introLoaded) {
    return <div className="app-page" />
  }

  if (showIntro) {
    return <ProjectIntro onEnter={handleEnterDashboard} />
  }

  if (loading) {
    return (
      <div className="app-page">
        <div className="mb-8">
          <Skeleton className="h-9 w-48 mb-2" />
          <Skeleton className="h-5 w-64" />
        </div>
        <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-5 mb-8">
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-[132px]" />
          ))}
        </div>
        <Skeleton className="h-96" />
      </div>
    )
  }

  return (
    <div className="app-page">
      {/* Header */}
      <div className="app-page-header">
        <h1 className="app-page-title">Dashboard</h1>
        <p className="app-page-subtitle">Overview of your stock analysis</p>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-5 mb-8">
        {stats.map((stat) => (
          <Link key={stat.label} href={stat.href}>
            <Card className="h-[132px] hover:bg-muted/50 transition-colors">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {stat.label}
                </CardTitle>
                <stat.icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold font-mono tabular-nums tracking-tight">{stat.value}</div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

      {dataError && (
        <Card className="mb-8 border-red-500/35 bg-red-500/8">
          <CardHeader className="pb-3">
            <CardTitle className="text-base text-red-200">Data connection issue</CardTitle>
            <CardDescription className="text-red-100/80">
              Dashboard could not read from Supabase. Check Vercel env vars:
              {" "}
              <code>NEXT_PUBLIC_SUPABASE_URL</code>
              {" "}
              and
              {" "}
              <code>NEXT_PUBLIC_SUPABASE_ANON_KEY</code>.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <p className="text-sm text-red-100/90">
              Technical error: {dataError}
            </p>
          </CardContent>
        </Card>
      )}

      {/* Recent Signals */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Zap className="h-5 w-5" />
            Recent Signals
          </CardTitle>
          <CardDescription>Latest trading signals from AI analysis</CardDescription>
        </CardHeader>
        <CardContent>
          {recentSignals.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">
              No signals yet. Run an analysis to see signals here.
            </p>
          ) : (
            <div className="space-y-4">
              {recentSignals.map((signal) => (
                <Link
                  key={signal.id}
                  href={`/stocks/${signal.stocks?.symbol || ""}`}
                  className="flex items-center justify-between rounded-lg border p-4 transition-colors hover:bg-muted/50 block w-full"
                >
                  <div className="flex items-center gap-4">
                    <div className={`rounded-full p-2 ${getSignalColor(signal.signal)}`}>
                      {getSignalIcon(signal.signal)}
                    </div>
                    <div>
                      <p className="font-semibold">{signal.stocks?.symbol || "Unknown"}</p>
                      <p className="text-sm text-muted-foreground font-mono tabular-nums mt-0.5">
                        {signal.target_price ? `Target: $${signal.target_price.toFixed(2)}` : signal.mode}
                      </p>
                    </div>
                  </div>
                  <div className="text-right flex items-center gap-4">
                    <div>
                      <Badge variant="outline" className={cn(getSignalColor(signal.signal), "uppercase text-[10px] tracking-wider")}>
                        {signal.signal.replace('_', ' ')}
                      </Badge>
                      <p className="text-xs text-muted-foreground mt-1.5 font-mono tabular-nums">
                        {Math.round(signal.confidence * 100)}% conf
                      </p>
                    </div>
                    <span className="text-xs text-muted-foreground w-16 text-right font-mono tabular-nums">
                      {timeAgo(signal.created_at)}
                    </span>
                  </div>
                </Link>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
