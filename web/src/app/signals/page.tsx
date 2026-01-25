"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { supabase, Analysis, Stock } from "@/lib/supabase"

interface AnalysisWithStock extends Analysis {
  stocks: Stock
}

function formatDate(dateStr: string) {
  const date = new Date(dateStr)
  return date.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  })
}

function getSignalColor(signal: string) {
  switch (signal) {
    case "strong_buy":
    case "buy":
      return "bg-green-500/10 text-green-500"
    case "strong_sell":
    case "sell":
      return "bg-red-500/10 text-red-500"
    default:
      return "bg-yellow-500/10 text-yellow-500"
  }
}

export default function SignalsPage() {
  const [signals, setSignals] = useState<AnalysisWithStock[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    async function fetchSignals() {
      try {
        const { data, error } = await supabase
          .from("analysis")
          .select("*, stocks(*)")
          .order("created_at", { ascending: false })
          .limit(50)

        if (error) {
          console.error("Error fetching signals:", error)
          return
        }

        if (data) {
          setSignals(data as AnalysisWithStock[])
        }
      } catch (error) {
        console.error("Error:", error)
      } finally {
        setLoading(false)
      }
    }

    fetchSignals()

    // Real-time subscription
    const channel = supabase
      .channel("signals")
      .on("postgres_changes", { event: "INSERT", schema: "public", table: "analysis" }, () => {
        fetchSignals()
      })
      .subscribe()

    return () => {
      supabase.removeChannel(channel)
    }
  }, [])

  if (loading) {
    return (
      <div className="p-8">
        <div className="mb-8">
          <Skeleton className="h-9 w-32 mb-2" />
          <Skeleton className="h-5 w-64" />
        </div>
        <Skeleton className="h-96" />
      </div>
    )
  }

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-3xl font-bold">Signals</h1>
        <p className="text-muted-foreground">Trading signals from AI analysis</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>All Signals</CardTitle>
          <CardDescription>Historical signals across all stocks</CardDescription>
        </CardHeader>
        <CardContent>
          {signals.length === 0 ? (
            <p className="text-muted-foreground text-center py-8">
              No signals yet. Run an analysis to generate signals.
            </p>
          ) : (
            <Table>
              <TableHeader>
                <TableRow>
                  <TableHead>Symbol</TableHead>
                  <TableHead>Signal</TableHead>
                  <TableHead>Confidence</TableHead>
                  <TableHead>Target</TableHead>
                  <TableHead>Stop Loss</TableHead>
                  <TableHead>Model</TableHead>
                  <TableHead>Time</TableHead>
                </TableRow>
              </TableHeader>
              <TableBody>
                {signals.map((signal) => (
                  <TableRow key={signal.id}>
                    <TableCell className="font-medium">
                      {signal.stocks?.symbol || "Unknown"}
                    </TableCell>
                    <TableCell>
                      <Badge variant="outline" className={getSignalColor(signal.signal)}>
                        {signal.signal.toUpperCase()}
                      </Badge>
                    </TableCell>
                    <TableCell>{Math.round(signal.confidence * 100)}%</TableCell>
                    <TableCell>
                      {signal.target_price ? `$${signal.target_price.toFixed(2)}` : "-"}
                    </TableCell>
                    <TableCell>
                      {signal.stop_loss ? `$${signal.stop_loss.toFixed(2)}` : "-"}
                    </TableCell>
                    <TableCell className="text-xs text-muted-foreground">
                      {signal.llm_model?.split("/").pop() || "-"}
                    </TableCell>
                    <TableCell className="text-muted-foreground">
                      {formatDate(signal.created_at)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
