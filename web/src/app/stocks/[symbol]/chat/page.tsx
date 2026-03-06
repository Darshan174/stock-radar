"use client"

import { useEffect, useState } from "react"
import { useParams, useRouter } from "next/navigation"
import { ArrowLeft, Lock, MessageSquare } from "lucide-react"
import { ChatContextBadge } from "@/components/chat-context-badge"
import { ChatAssistant } from "@/components/chat-assistant"
import { StockChatButton } from "@/components/stock-chat-button"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { supabase, Stock } from "@/lib/supabase"

interface StockChatState {
  stock: Stock | null
  analysisCount: number
}

export default function StockChatPage() {
  const params = useParams()
  const router = useRouter()
  const symbol = decodeURIComponent(params.symbol as string).toUpperCase()

  const [state, setState] = useState<StockChatState>({ stock: null, analysisCount: 0 })
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchStockChatState() {
      try {
        const { data: stockData } = await supabase
          .from("stocks")
          .select("*")
          .eq("symbol", symbol)
          .single()

        if (!stockData) {
          setState({ stock: null, analysisCount: 0 })
          return
        }

        const { data: analyses } = await supabase
          .from("analysis")
          .select("id")
          .eq("stock_id", stockData.id)

        setState({
          stock: stockData,
          analysisCount: analyses?.length || 0,
        })
      } catch (error) {
        console.error("Error fetching stock chat state:", error)
        setState({ stock: null, analysisCount: 0 })
      } finally {
        setLoading(false)
      }
    }

    if (symbol) {
      fetchStockChatState()
    }
  }, [symbol])

  if (loading) {
    return (
      <div className="app-page">
        <Skeleton className="mb-6 h-8 w-48" />
        <Skeleton className="h-[70vh]" />
      </div>
    )
  }

  if (!state.stock) {
    return (
      <div className="app-page">
        <Button variant="ghost" onClick={() => router.push("/stocks")} className="mb-6">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to Watchlist
        </Button>
        <Card className="mx-auto max-w-2xl border-dashed">
          <CardContent className="py-10 text-center">
            <p className="text-lg font-semibold">Stock not found</p>
            <p className="mt-2 text-sm text-muted-foreground">
              We couldn&apos;t find a tracked stock for {symbol}.
            </p>
          </CardContent>
        </Card>
      </div>
    )
  }

  if (state.analysisCount < 1) {
    return (
      <div className="app-page">
        <Button variant="ghost" onClick={() => router.push(`/stocks/${state.stock?.symbol}`)} className="mb-6">
          <ArrowLeft className="mr-2 h-4 w-4" />
          Back to {state.stock.symbol}
        </Button>

        <Card className="mx-auto max-w-2xl border-dashed">
          <CardHeader className="text-center">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full border border-white/15 bg-black text-white/70 grayscale saturate-0">
              <Lock className="h-6 w-6" />
            </div>
            <CardTitle>Chat unavailable for {state.stock.symbol}</CardTitle>
            <CardDescription>
              AI chat only opens after at least one saved analysis exists for this stock.
            </CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col items-center gap-4 pb-8">
            <ChatContextBadge analysisCount={0} showLocked={true} showCount={true} />
            <StockChatButton symbol={state.stock?.symbol ?? ''} hasAnalysis={false} />
            <Button onClick={() => router.push(`/stocks/${state.stock?.symbol ?? ''}`)}>
              Run or view analysis
            </Button>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="app-page flex h-[calc(100vh-1rem)] flex-col">
      <div className="mb-4 flex items-center justify-between gap-4">
        <div className="flex items-center gap-4">
          <Button variant="ghost" size="icon" onClick={() => router.push(`/stocks/${state.stock?.symbol}`)}>
            <ArrowLeft className="h-5 w-5" />
          </Button>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="app-page-title">{state.stock.symbol} Chat</h1>
              <Badge variant="outline" className="border-cyan-500/30 bg-cyan-500/10 text-cyan-300">
                Analysis Ready
              </Badge>
              <ChatContextBadge analysisCount={state.analysisCount} showCount={true} />
            </div>
            <p className="app-page-subtitle">
              Ask about saved analyses, signals, and news for {state.stock.name}.
            </p>
          </div>
        </div>
        <div className="hidden items-center gap-2 rounded-lg border border-cyan-500/20 bg-cyan-500/10 px-3 py-2 text-sm text-cyan-100 md:flex">
          <MessageSquare className="h-4 w-4" />
          Stock-scoped chat
        </div>
      </div>

      <ChatAssistant className="h-full" defaultSymbol={state.stock.symbol} />
    </div>
  )
}
