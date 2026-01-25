"use client"

import { useLiveStockData } from "@/hooks/use-live-stock-data"
import { TrendingUp, TrendingDown, Wifi, WifiOff } from "lucide-react"
import { cn } from "@/lib/utils"

interface LivePriceTickerProps {
    symbol: string
    currency?: string
    className?: string
}

export function LivePriceTicker({ symbol, currency = "₹", className }: LivePriceTickerProps) {
    const { livePrice, isConnected, error } = useLiveStockData({
        symbol,
        enabled: true,
        refreshInterval: 5000, // Update every 5 seconds
    })

    if (!livePrice) {
        return (
            <div className={cn("flex items-center gap-2 text-muted-foreground", className)}>
                <div className="animate-pulse h-6 w-24 bg-muted rounded" />
            </div>
        )
    }

    const isUp = livePrice.change >= 0

    return (
        <div className={cn("flex items-center gap-3", className)}>
            {/* Connection status */}
            <div className="flex items-center gap-1.5">
                {isConnected ? (
                    <Wifi className="h-3 w-3 text-green-500" />
                ) : (
                    <WifiOff className="h-3 w-3 text-red-500" />
                )}
                <span className="text-xs text-muted-foreground">
                    {isConnected ? "LIVE" : "OFFLINE"}
                </span>
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

            {/* Last update time */}
            <span className="text-xs text-muted-foreground">
                Updated: {livePrice.timestamp.toLocaleTimeString()}
            </span>
        </div>
    )
}
