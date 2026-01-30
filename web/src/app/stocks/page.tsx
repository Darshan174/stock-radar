"use client"

import { useEffect, useState } from "react"
import { useRouter } from "next/navigation"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Sparkline } from "@/components/charts"
import { Plus, TrendingUp, TrendingDown, Loader2, Play, CheckCircle, XCircle } from "lucide-react"
import { supabase, Stock } from "@/lib/supabase"

interface StockWithPrice extends Stock {
  latest_price?: number
  price_change?: number
  sparkline_data?: { time: string; value: number }[]
}

export default function StocksPage() {
  const router = useRouter()
  const [stocks, setStocks] = useState<StockWithPrice[]>([])
  const [loading, setLoading] = useState(true)
  const [dialogOpen, setDialogOpen] = useState(false)
  const [newSymbol, setNewSymbol] = useState("")
  const [adding, setAdding] = useState(false)
  const [analyzing, setAnalyzing] = useState<string | null>(null)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState("")

  async function fetchStocks() {
    try {
      const { data: stocksData, error } = await supabase
        .from("stocks")
        .select("*")
        .order("symbol")

      if (error) {
        console.error("Error fetching stocks:", error)
        return
      }

      if (stocksData) {
        // First, quickly load stocks with historical data
        const stocksWithHistoricalPrices = await Promise.all(
          stocksData.map(async (stock) => {
            const { data: priceData } = await supabase
              .from("price_history")
              .select("close, timestamp")
              .eq("stock_id", stock.id)
              .order("timestamp", { ascending: false })
              .limit(30)

            let latest_price = 0
            let price_change = 0
            let sparkline_data: { time: string; value: number }[] = []

            if (priceData && priceData.length > 0) {
              latest_price = priceData[0].close
              if (priceData.length > 1) {
                const prev = priceData[1].close
                price_change = prev > 0 ? ((latest_price - prev) / prev) * 100 : 0
              }
              sparkline_data = [...priceData].reverse().map((p) => ({
                time: p.timestamp.split("T")[0],
                value: p.close,
              }))
            }

            return { ...stock, latest_price, price_change, sparkline_data }
          })
        )

        // Show stocks immediately with historical prices
        setStocks(stocksWithHistoricalPrices)
        setLoading(false)

        // Then fetch live prices in background and update
        const updatedStocks = await Promise.all(
          stocksWithHistoricalPrices.map(async (stock) => {
            try {
              const liveResponse = await fetch(`/api/live-price?symbol=${encodeURIComponent(stock.symbol)}`, {
                cache: "no-store"
              })
              if (liveResponse.ok) {
                const liveData = await liveResponse.json()
                if (liveData.price) {
                  return {
                    ...stock,
                    latest_price: liveData.price,
                    price_change: liveData.changePercent || stock.price_change,
                  }
                }
              }
            } catch (e) {
              console.warn(`Failed to fetch live price for ${stock.symbol}:`, e)
            }
            return stock
          })
        )

        // Update with live prices
        setStocks(updatedStocks)
      }
    } catch (error) {
      console.error("Error:", error)
    } finally {
      setLoading(false)
    }
  }

  // Refresh live prices periodically
  useEffect(() => {
    if (loading || stocks.length === 0) return

    const priceRefreshInterval = setInterval(async () => {
      const updatedStocks = await Promise.all(
        stocks.map(async (stock) => {
          try {
            const liveResponse = await fetch(`/api/live-price?symbol=${encodeURIComponent(stock.symbol)}`, {
              cache: "no-store"
            })
            if (liveResponse.ok) {
              const liveData = await liveResponse.json()
              if (liveData.price) {
                return {
                  ...stock,
                  latest_price: liveData.price,
                  price_change: liveData.changePercent || stock.price_change,
                }
              }
            }
          } catch {
            // Ignore errors on refresh
          }
          return stock
        })
      )
      setStocks(updatedStocks)
    }, 30000) // Refresh every 30 seconds

    return () => clearInterval(priceRefreshInterval)
  }, [loading, stocks.length]) // Only depend on length to avoid infinite loop

  useEffect(() => {
    fetchStocks()

    const channel = supabase
      .channel("stocks-page")
      .on("postgres_changes", { event: "*", schema: "public", table: "stocks" }, () => {
        fetchStocks()
      })
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "analysis" }, () => {
        fetchStocks()
      })
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [])

  async function handleAddStock() {
    if (!newSymbol.trim()) {
      setError("Please enter a stock symbol")
      return
    }

    setAdding(true)
    setError("")
    setSuccess("")

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol: newSymbol.toUpperCase() }),
      })

      const data = await response.json()

      if (!response.ok) {
        setError(data.error || "Failed to add stock")
        return
      }

      setSuccess(`Successfully added ${newSymbol.toUpperCase()}!`)
      setNewSymbol("")

      // Wait a bit then close dialog
      setTimeout(() => {
        setDialogOpen(false)
        setSuccess("")
        fetchStocks()
      }, 1500)
    } catch (err) {
      setError("Failed to connect to server. Make sure the backend is running.")
    } finally {
      setAdding(false)
    }
  }

  async function handleAnalyze(e: React.MouseEvent, symbol: string) {
    e.stopPropagation()
    setAnalyzing(symbol)

    try {
      const response = await fetch("/api/analyze", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ symbol }),
      })

      if (response.ok) {
        fetchStocks()
      }
    } catch (err) {
      console.error("Failed to analyze:", err)
    } finally {
      setAnalyzing(null)
    }
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="flex items-center justify-between mb-8">
          <div>
            <Skeleton className="h-9 w-32 mb-2" />
            <Skeleton className="h-5 w-48" />
          </div>
          <Skeleton className="h-10 w-28" />
        </div>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {[1, 2, 3].map((i) => (
            <Skeleton key={i} className="h-48" />
          ))}
        </div>
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="flex items-center justify-between mb-8">
        <div>
          <h1 className="text-3xl font-bold">Watchlist</h1>
          <p className="text-muted-foreground">Manage your tracked stocks</p>
        </div>
        <Dialog open={dialogOpen} onOpenChange={(open) => {
          setDialogOpen(open)
          if (!open) {
            setError("")
            setSuccess("")
            setNewSymbol("")
          }
        }}>
          <DialogTrigger asChild>
            <Button>
              <Plus className="h-4 w-4 mr-2" />
              Add Stock
            </Button>
          </DialogTrigger>
          <DialogContent>
            <DialogHeader>
              <DialogTitle>Add Stock to Watchlist</DialogTitle>
              <DialogDescription>
                Enter a stock symbol to analyze and add to your watchlist.
                This may take up to 2 minutes.
              </DialogDescription>
            </DialogHeader>
            <div className="space-y-4 py-4">
              <div className="space-y-2">
                <Input
                  placeholder="Enter symbol (e.g., AAPL, RELIANCE.NS)"
                  value={newSymbol}
                  onChange={(e) => setNewSymbol(e.target.value.toUpperCase())}
                  onKeyDown={(e) => e.key === "Enter" && !adding && handleAddStock()}
                  disabled={adding}
                />
                {error && (
                  <div className="flex items-center gap-2 text-sm text-red-500">
                    <XCircle className="h-4 w-4" />
                    {error}
                  </div>
                )}
                {success && (
                  <div className="flex items-center gap-2 text-sm text-green-500">
                    <CheckCircle className="h-4 w-4" />
                    {success}
                  </div>
                )}
              </div>

              {/* Categorized Stock Lists */}
              <div className="space-y-3 max-h-[300px] overflow-y-auto">
                {/* Indian Stocks */}
                {(() => {
                  const stocks = [
                    // Nifty 50 Blue Chips
                    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
                    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "WIPRO.NS", "AXISBANK.NS",
                    "TATAMOTORS.NS", "MARUTI.NS", "ASIANPAINT.NS", "SUNPHARMA.NS", "BAJFINANCE.NS",
                    "HINDUNILVR.NS", "KOTAKBANK.NS", "LT.NS", "HCLTECH.NS", "TITAN.NS",
                    "ULTRACEMCO.NS", "NESTLEIND.NS", "POWERGRID.NS", "NTPC.NS", "ONGC.NS",
                    "TATASTEEL.NS", "JSWSTEEL.NS", "COALINDIA.NS", "BPCL.NS", "IOC.NS",
                    // Metals & Mining
                    "HINDCOPPER.NS", "HINDALCO.NS", "VEDL.NS", "NMDC.NS", "NATIONALUM.NS",
                    // PSU Banks
                    "BANKBARODA.NS", "PNB.NS", "CANBK.NS", "UNIONBANK.NS", "INDIANB.NS",
                    // Auto
                    "M&M.NS", "HEROMOTOCO.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "TVSMOTOR.NS",
                    // Pharma
                    "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "BIOCON.NS", "AUROPHARMA.NS",
                    // IT
                    "TECHM.NS", "LTIM.NS", "MPHASIS.NS", "COFORGE.NS", "PERSISTENT.NS",
                    // Others
                    "ADANIENT.NS", "ADANIPORTS.NS", "GRASIM.NS", "BRITANNIA.NS", "DABUR.NS",
                    "GODREJCP.NS", "PIDILITIND.NS", "HAVELLS.NS", "INDIGO.NS", "ZOMATO.NS"
                  ]
                  const filtered = newSymbol ? stocks.filter(s => s.toLowerCase().includes(newSymbol.toLowerCase()) || s.replace(".NS", "").toLowerCase().includes(newSymbol.toLowerCase())) : stocks
                  if (filtered.length === 0) return null
                  return (
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-primary">🇮🇳 Indian Stocks (NSE)</h4>
                      <div className="flex flex-wrap gap-1.5">
                        {filtered.map((sym) => (
                          <Button
                            key={sym}
                            variant={newSymbol === sym ? "default" : "outline"}
                            size="sm"
                            className="text-xs h-7"
                            onClick={() => setNewSymbol(sym)}
                            disabled={adding}
                          >
                            {sym.replace(".NS", "")}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )
                })()}

                {/* US Stocks */}
                {(() => {
                  const stocks = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V", "JNJ", "WMT", "PG", "DIS", "NFLX", "AMD", "INTC", "CRM", "PYPL", "UBER"]
                  const filtered = newSymbol ? stocks.filter(s => s.toLowerCase().includes(newSymbol.toLowerCase())) : stocks
                  if (filtered.length === 0) return null
                  return (
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-primary">🇺🇸 US Stocks</h4>
                      <div className="flex flex-wrap gap-1.5">
                        {filtered.map((sym) => (
                          <Button
                            key={sym}
                            variant={newSymbol === sym ? "default" : "outline"}
                            size="sm"
                            className="text-xs h-7"
                            onClick={() => setNewSymbol(sym)}
                            disabled={adding}
                          >
                            {sym}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )
                })()}

                {/* ETFs */}
                {(() => {
                  const stocks = ["SPY", "QQQ", "VTI", "VOO", "IWM", "EEM", "VNQ", "GLD", "SLV", "TLT", "HYG", "XLF", "XLK", "XLE", "ARKK"]
                  const filtered = newSymbol ? stocks.filter(s => s.toLowerCase().includes(newSymbol.toLowerCase())) : stocks
                  if (filtered.length === 0) return null
                  return (
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-primary">📊 ETFs</h4>
                      <div className="flex flex-wrap gap-1.5">
                        {filtered.map((sym) => (
                          <Button
                            key={sym}
                            variant={newSymbol === sym ? "default" : "outline"}
                            size="sm"
                            className="text-xs h-7"
                            onClick={() => setNewSymbol(sym)}
                            disabled={adding}
                          >
                            {sym}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )
                })()}

                {/* Crypto */}
                {(() => {
                  const stocks = ["BTC-USD", "ETH-USD", "BNB-USD", "XRP-USD", "SOL-USD", "ADA-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "SHIB-USD", "AVAX-USD", "LINK-USD", "LTC-USD", "UNI-USD", "ATOM-USD"]
                  const filtered = newSymbol ? stocks.filter(s => s.toLowerCase().includes(newSymbol.toLowerCase()) || s.replace("-USD", "").toLowerCase().includes(newSymbol.toLowerCase())) : stocks
                  if (filtered.length === 0) return null
                  return (
                    <div>
                      <h4 className="text-sm font-semibold mb-2 text-primary">🪙 Crypto</h4>
                      <div className="flex flex-wrap gap-1.5">
                        {filtered.map((sym) => (
                          <Button
                            key={sym}
                            variant={newSymbol === sym ? "default" : "outline"}
                            size="sm"
                            className="text-xs h-7"
                            onClick={() => setNewSymbol(sym)}
                            disabled={adding}
                          >
                            {sym.replace("-USD", "")}
                          </Button>
                        ))}
                      </div>
                    </div>
                  )
                })()}
              </div>

              {adding && (
                <div className="bg-muted/50 rounded-lg p-3 text-sm">
                  <div className="flex items-center gap-2 text-muted-foreground">
                    <Loader2 className="h-4 w-4 animate-spin" />
                    <span>Fetching data and running AI analysis...</span>
                  </div>
                </div>
              )}
            </div>
            <DialogFooter>
              <Button variant="outline" onClick={() => setDialogOpen(false)} disabled={adding}>
                Cancel
              </Button>
              <Button onClick={handleAddStock} disabled={adding || !newSymbol.trim()}>
                {adding ? (
                  <>
                    <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                    Analyzing...
                  </>
                ) : (
                  "Add & Analyze"
                )}
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
      </div>

      {stocks.length === 0 ? (
        <Card className="p-8 text-center">
          <div className="space-y-3">
            <p className="text-muted-foreground">No stocks in your watchlist yet.</p>
            <p className="text-sm text-muted-foreground">
              Click &quot;Add Stock&quot; to get started with AI-powered analysis.
            </p>
          </div>
        </Card>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {stocks.map((stock) => (
            <Card
              key={stock.symbol}
              className="cursor-pointer hover:scale-[1.02] transition-all relative overflow-hidden group"
              onClick={() => router.push(`/stocks/${stock.symbol}`)}
            >
              {/* Full-card sparkline background */}
              {stock.sparkline_data && stock.sparkline_data.length > 1 && (
                <div className="absolute inset-0 opacity-40 group-hover:opacity-60 transition-opacity">
                  <Sparkline
                    data={stock.sparkline_data}
                    width={320}
                    height={180}
                    color={stock.price_change && stock.price_change >= 0 ? "#00E676" : "#FF1744"}
                  />
                </div>
              )}

              {/* Gradient overlay for text readability */}
              <div className="absolute inset-0 bg-gradient-to-t from-background via-background/60 to-transparent" />

              <CardHeader className="pb-2 relative z-10">
                <div className="flex items-center justify-between">
                  <CardTitle className="flex items-center gap-2">
                    {stock.price_change && stock.price_change >= 0 ? (
                      <TrendingUp className="h-5 w-5 text-[#00E676]" />
                    ) : (
                      <TrendingDown className="h-5 w-5 text-[#FF1744]" />
                    )}
                    {stock.symbol}
                  </CardTitle>
                  {stock.price_change !== undefined && stock.price_change !== 0 && (
                    <span className={`text-lg font-bold ${stock.price_change >= 0 ? "text-[#00E676]" : "text-[#FF1744]"}`}>
                      {stock.price_change >= 0 ? "+" : ""}{stock.price_change.toFixed(2)}%
                    </span>
                  )}
                </div>
                <CardDescription className="text-foreground/70">{stock.name}</CardDescription>
              </CardHeader>
              <CardContent className="relative z-10">
                <div className="flex items-end justify-between gap-4">
                  <div>
                    <div className="text-3xl font-bold">
                      {stock.latest_price ? (
                        <>
                          {stock.currency === "INR" ? "₹" : "$"}
                          {stock.latest_price.toFixed(2)}
                        </>
                      ) : (
                        <span className="text-muted-foreground text-lg">No data</span>
                      )}
                    </div>
                    <p className="text-sm text-muted-foreground">{stock.sector || stock.exchange}</p>
                  </div>
                  <Button
                    size="sm"
                    variant="secondary"
                    onClick={(e) => handleAnalyze(e, stock.symbol)}
                    disabled={analyzing === stock.symbol}
                    className="shrink-0"
                  >
                    {analyzing === stock.symbol ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <>
                        <Play className="h-3 w-3 mr-1" />
                        Analyze
                      </>
                    )}
                  </Button>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}
    </div>
  )
}
