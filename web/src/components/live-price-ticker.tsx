"use client"

import { useEffect, useState } from "react"
import { LivePrice, useLiveStockData } from "@/hooks/use-live-stock-data"
import { TrendingUp, TrendingDown, Wifi, WifiOff } from "lucide-react"
import { cn } from "@/lib/utils"

interface LivePriceTickerProps {
    symbol: string
    currency?: string
    className?: string
    livePriceOverride?: LivePrice | null
}

export function LivePriceTicker({
    symbol,
    currency = "\u20B9",
    className,
    livePriceOverride = null,
}: LivePriceTickerProps) {
    const { livePrice: hookLivePrice } = useLiveStockData({
        symbol,
        enabled: !livePriceOverride,
        updateThrottleMs: 350,
    })

    const livePrice = livePriceOverride || hookLivePrice
    const [nowMs, setNowMs] = useState(() => Date.now())

    useEffect(() => {
        const timer = setInterval(() => setNowMs(Date.now()), 1000)
        return () => clearInterval(timer)
    }, [])

    if (!livePrice) {
        return (
            <div className={cn("flex items-center gap-2 text-muted-foreground", className)}>
                <div className="animate-pulse h-6 w-24 bg-muted rounded" />
            </div>
        )
    }

    const isUp = livePrice.change >= 0
    const ageSec = Math.max(0, Math.floor((nowMs - livePrice.timestamp.getTime()) / 1000))
    const isFresh = ageSec <= 18
    const isDelayed = ageSec > 18 && ageSec <= 120

    return (
        <div className={cn("flex items-center gap-3", className)}>
            {/* Connection/data freshness status */}
            <div className="flex items-center gap-1.5">
                {isFresh ? (
                    <>
                        <Wifi className="h-3 w-3 text-green-500" />
                        <span className="text-xs text-muted-foreground">LIVE</span>
                    </>
                ) : isDelayed ? (
                    <>
                        <Wifi className="h-3 w-3 text-amber-500" />
                        <span className="text-xs text-muted-foreground">DELAYED</span>
                    </>
                ) : (
                    <>
                        <WifiOff className="h-3 w-3 text-red-500" />
                        <span className="text-xs text-muted-foreground">OFFLINE</span>
                    </>
                )}
            </div>

            {/* Price display */}
            <div className="flex items-center gap-2">
                <span className="text-2xl font-bold">
                    {currency}{livePrice.price.toFixed(2)}
                </span>
                <div className={cn(
                    "flex items-center gap-1 px-2 py-0.5 rounded text-sm font-medium",
                    isUp ? "bg-green-500/10 text-green-500" : "bg-red-500/10 text-red-500"
                )}>
                    {isUp ? (
                        <TrendingUp className="h-3.5 w-3.5" />
                    ) : (
                        <TrendingDown className="h-3.5 w-3.5" />
                    )}
                    <span>{isUp ? "+" : ""}{livePrice.change.toFixed(2)}</span>
                    <span>({isUp ? "+" : ""}{livePrice.changePercent.toFixed(2)}%)</span>
                </div>
            </div>

        </div>
    )
}
