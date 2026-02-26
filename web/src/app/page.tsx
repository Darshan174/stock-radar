"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { TrendingUp, TrendingDown, Minus, Zap, BarChart3, Activity, MessageSquare, CreditCard } from "lucide-react"
import { supabase, Analysis, Stock } from "@/lib/supabase"
import { ProjectIntro } from "@/components/project-intro"
import { useSidebar } from "@/providers/sidebar-provider"
import Link from "next/link"

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
        // Fetch recent analyses with stock info
        const { data: analyses } = await supabase
          .from("analysis")
          .select("*, stocks(*)")
          .order("created_at", { ascending: false })
          .limit(5)

        if (analyses) {
          setRecentSignals(analyses as SignalWithStock[])
        }

        // Count stocks
        const { count: stocks } = await supabase
          .from("stocks")
          .select("*", { count: "exact", head: true })
        setStockCount(stocks || 0)

        // Count today's signals
        const today = new Date()
        today.setHours(0, 0, 0, 0)
        const { count: todaySignals } = await supabase
          .from("analysis")
          .select("*", { count: "exact", head: true })
          .gte("created_at", today.toISOString())
        setSignalsToday(todaySignals || 0)

      } catch (error) {
        console.error("Error fetching data:", error)
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
    { label: "Active Stocks", value: stockCount.toString(), icon: Activity, change: "", href: "/stocks" },
    { label: "Signals Today", value: signalsToday.toString(), icon: Zap, change: "", href: "/signals" },
    { label: "AI Chat", value: "Ask Assistant", icon: MessageSquare, change: "", href: "/chat" },
    { label: "Aptos Payment", value: "x402 Demo", icon: CreditCard, change: "", href: "/x402-demo" },
    { label: "API Usage", value: "View Details", icon: BarChart3, change: "", href: "/usage" },
  ]

  if (!introLoaded) {
    return <div className="p-8" />
  }

  if (showIntro) {
    return <ProjectIntro onEnter={handleEnterDashboard} />
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="mb-8">
          <Skeleton className="h-9 w-48 mb-2" />
          <Skeleton className="h-5 w-64" />
        </div>
        <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-5 mb-8">
          {[1, 2, 3, 4, 5].map((i) => (
            <Skeleton key={i} className="h-32" />
          ))}
        </div>
        <Skeleton className="h-96" />
      </div>
    )
  }

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground">Overview of your stock analysis</p>
      </div>

      {/* Stats Cards */}
      <div className="grid gap-4 md:grid-cols-3 lg:grid-cols-5 mb-8">
        {stats.map((stat) => (
          <Link key={stat.label} href={stat.href}>
            <Card className="hover:bg-muted/50 transition-colors h-full">
              <CardHeader className="flex flex-row items-center justify-between pb-2">
                <CardTitle className="text-sm font-medium text-muted-foreground">
                  {stat.label}
                </CardTitle>
                <stat.icon className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{stat.value}</div>
              </CardContent>
            </Card>
          </Link>
        ))}
      </div>

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
                      <p className="text-sm text-muted-foreground">
                        {signal.target_price ? `Target: $${signal.target_price.toFixed(2)}` : signal.mode}
                      </p>
                    </div>
                  </div>
                  <div className="text-right flex items-center gap-4">
                    <div>
                      <Badge variant="outline" className={getSignalColor(signal.signal)}>
                        {signal.signal.toUpperCase()}
                      </Badge>
                      <p className="text-sm text-muted-foreground mt-1">
                        {Math.round(signal.confidence * 100)}% confidence
                      </p>
                    </div>
                    <span className="text-xs text-muted-foreground w-16 text-right">
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
