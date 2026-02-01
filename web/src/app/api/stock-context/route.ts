import { NextRequest, NextResponse } from "next/server"
import { createClient } from "@supabase/supabase-js"

// Lazy Supabase client initialization to avoid module load errors
function getSupabase() {
  const supabaseUrl = process.env.NEXT_PUBLIC_SUPABASE_URL
  const supabaseKey = process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY

  if (!supabaseUrl || !supabaseKey) {
    throw new Error("Missing Supabase environment variables")
  }

  return createClient(supabaseUrl, supabaseKey)
}

export interface RAGContextResponse {
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
    }
  }
  metadata: {
    totalResults: number
    sourcesSearched: string[]
    retrievalTimeMs: number
  }
  error?: string
}

export async function GET(request: NextRequest) {
  const startTime = Date.now()
  const searchParams = request.nextUrl.searchParams
  const symbol = searchParams.get("symbol")
  const includeAnalyses = searchParams.get("analyses") !== "false"
  const includeSignals = searchParams.get("signals") !== "false"
  const includeNews = searchParams.get("news") !== "false"
  const includeKnowledge = searchParams.get("knowledge") !== "false"

  if (!symbol) {
    return NextResponse.json(
      { error: "Symbol parameter is required" },
      { status: 400 }
    )
  }

  try {
    const supabase = getSupabase()
    const sanitizedSymbol = symbol.replace(/[^A-Za-z0-9.\-^]/g, "").toUpperCase()
    const sourcesSearched: string[] = []
    let totalResults = 0

    // Get stock ID first
    const { data: stockData } = await supabase
      .from("stocks")
      .select("id, sector")
      .eq("symbol", sanitizedSymbol)
      .single()

    const stockId = stockData?.id
    const sector = stockData?.sector

    // Fetch recent analyses for this stock (no vector search needed)
    let similarAnalyses: Array<{
      id: number
      symbol: string
      signal: string
      confidence: number
      reasoning: string
      createdAt: string
      similarity: number
    }> = []

    if (includeAnalyses && stockId) {
      sourcesSearched.push("analyses")
      const { data: analyses } = await supabase
        .from("analysis")
        .select("id, signal, confidence, reasoning, technical_summary, created_at")
        .eq("stock_id", stockId)
        .order("created_at", { ascending: false })
        .limit(5)

      if (analyses) {
        similarAnalyses = analyses.map((a, idx) => ({
          id: a.id,
          symbol: sanitizedSymbol,
          signal: a.signal || "",
          confidence: a.confidence || 0,
          reasoning: a.reasoning || a.technical_summary || "",
          createdAt: a.created_at?.substring(0, 10) || "",
          similarity: 1 - idx * 0.1, // Fake similarity based on recency
        }))
        totalResults += similarAnalyses.length
      }
    }

    // Fetch recent signals for this stock
    let relatedSignals: Array<{
      id: number
      symbol: string
      signal: string
      priceAtSignal: number
      reason: string
      importance: string
      similarity: number
    }> = []

    if (includeSignals && stockId) {
      sourcesSearched.push("signals")
      const { data: signals } = await supabase
        .from("signals")
        .select("id, signal, price_at_signal, reason, importance, created_at")
        .eq("stock_id", stockId)
        .order("created_at", { ascending: false })
        .limit(5)

      if (signals) {
        relatedSignals = signals.map((s, idx) => ({
          id: s.id,
          symbol: sanitizedSymbol,
          signal: s.signal || "",
          priceAtSignal: s.price_at_signal || 0,
          reason: s.reason || "",
          importance: s.importance || "medium",
          similarity: 1 - idx * 0.1,
        }))
        totalResults += relatedSignals.length
      }
    }

    // Fetch recent news for this stock
    let relevantNews: Array<{
      id: number
      headline: string
      summary: string
      source: string
      sentimentLabel: string
      publishedAt: string
      similarity: number
    }> = []

    if (includeNews && stockId) {
      sourcesSearched.push("news")
      const { data: news } = await supabase
        .from("news")
        .select("id, headline, summary, source, sentiment_label, published_at")
        .eq("stock_id", stockId)
        .order("published_at", { ascending: false })
        .limit(5)

      if (news) {
        relevantNews = news.map((n, idx) => ({
          id: n.id,
          headline: n.headline || "",
          summary: n.summary || "",
          source: n.source || "",
          sentimentLabel: n.sentiment_label || "neutral",
          publishedAt: n.published_at?.substring(0, 10) || "",
          similarity: 1 - idx * 0.1,
        }))
        totalResults += relevantNews.length
      }
    }

    // Fetch knowledge base entries (if table exists)
    let knowledgeBase: Array<{
      id: number
      title: string
      content: string
      category: string
      similarity: number
    }> = []

    if (includeKnowledge) {
      sourcesSearched.push("knowledge")
      try {
        const { data: knowledge } = await supabase
          .from("knowledge_base")
          .select("id, title, content, category")
          .or(`stock_symbols.cs.{${sanitizedSymbol}},is_public.eq.true`)
          .order("created_at", { ascending: false })
          .limit(5)

        if (knowledge) {
          knowledgeBase = knowledge.map((k, idx) => ({
            id: k.id,
            title: k.title || "",
            content: k.content || "",
            category: k.category || "general",
            similarity: 1 - idx * 0.1,
          }))
          totalResults += knowledgeBase.length
        }
      } catch {
        // Table might not exist yet
      }
    }

    // Get sector sentiment summary
    let sectorSentiment = null
    if (sector) {
      try {
        // Count positive/negative news in sector
        const { data: sectorStocks } = await supabase
          .from("stocks")
          .select("id")
          .eq("sector", sector)

        if (sectorStocks && sectorStocks.length > 0) {
          const stockIds = sectorStocks.map(s => s.id)

          const { data: sectorNews } = await supabase
            .from("news")
            .select("sentiment_label")
            .in("stock_id", stockIds)
            .gte("published_at", new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString())

          if (sectorNews) {
            const positiveCount = sectorNews.filter(n => n.sentiment_label === "positive").length
            const negativeCount = sectorNews.filter(n => n.sentiment_label === "negative").length
            const sentiment = positiveCount > negativeCount ? "Bullish" : negativeCount > positiveCount ? "Bearish" : "Neutral"

            sectorSentiment = {
              sector,
              sentimentSummary: `${sentiment} (${positiveCount} positive, ${negativeCount} negative news in last 7 days)`,
              positiveCount,
              negativeCount,
            }
          }
        }
      } catch {
        // Ignore sector sentiment errors
      }
    }

    const retrievalTimeMs = Date.now() - startTime

    return NextResponse.json({
      success: true,
      symbol: sanitizedSymbol,
      context: {
        similarAnalyses,
        relatedSignals,
        relevantNews,
        knowledgeBase,
        sectorSentiment,
      },
      metadata: {
        totalResults,
        sourcesSearched,
        retrievalTimeMs,
      },
    })
  } catch (error) {
    console.error("RAG context retrieval error:", error)
    const message = error instanceof Error ? error.message : "Unknown error"
    return NextResponse.json({
      success: false,
      error: message,
      symbol: "",
      context: {
        similarAnalyses: [],
        relatedSignals: [],
        relevantNews: [],
        knowledgeBase: [],
        sectorSentiment: null,
      },
      metadata: {
        totalResults: 0,
        sourcesSearched: [],
        retrievalTimeMs: 0,
      },
    }, { status: 500 })
  }
}
