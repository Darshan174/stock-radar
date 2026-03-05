"use client"

import { useEffect, useState, useRef } from "react"
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
    const { livePrice: hookLivePrice, error: hookError } = useLiveStockData({
        symbol,
        enabled: !livePriceOverride,
        updateThrottleMs: 350,
    })

    const livePrice = livePriceOverride || hookLivePrice
    const [nowMs, setNowMs] = useState(() => Date.now())
    const prevPriceRef = useRef<number | null>(null)
    const [flashClass, setFlashClass] = useState("")

    useEffect(() => {
        const timer = setInterval(() => setNowMs(Date.now()), 1000)
        return () => clearInterval(timer)
    }, [])

    useEffect(() => {
        if (!livePrice) return

        let timeout: NodeJS.Timeout;
        if (prevPriceRef.current !== null && livePrice.price !== prevPriceRef.current) {
            setFlashClass(livePrice.price > prevPriceRef.current ? "animate-price-up" : "animate-price-down")
            timeout = setTimeout(() => setFlashClass(""), 1000)
        }
        prevPriceRef.current = livePrice.price
        return () => clearTimeout(timeout)
    }, [livePrice?.price])

    if (!livePrice) {
        return (
            <div className={cn("flex items-center gap-2 text-muted-foreground", className)}>
                <div className="h-6 min-w-24 rounded bg-muted/70 px-2 text-xs flex items-center justify-center">
                    {hookError ? "Price unavailable" : "Loading price..."}
                </div>
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
                        <span className="text-xs text-muted-foreground font-semibold tracking-wider">LIVE</span>
                    </>
                ) : isDelayed ? (
                    <>
                        <Wifi className="h-3 w-3 text-amber-500" />
                        <span className="text-xs text-muted-foreground font-semibold tracking-wider">DELAYED</span>
                    </>
                ) : (
                    <>
                        <WifiOff className="h-3 w-3 text-red-500" />
                        <span className="text-xs text-muted-foreground font-semibold tracking-wider">OFFLINE</span>
                    </>
                )}
            </div>

            {/* Price display */}
            <div className="flex items-center gap-2">
                <span
                    key={`${livePrice.price}-${flashClass}`}
                    className={cn("text-2xl font-bold font-mono tabular-nums px-1.5 rounded", flashClass)}
                >
                    {currency}{livePrice.price.toFixed(2)}
                </span>
                <div className={cn(
                    "flex items-center gap-1 px-2 py-0.5 rounded text-sm font-medium font-mono tabular-nums",
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
