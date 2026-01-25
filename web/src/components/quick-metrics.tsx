"use client"

import { cn } from "@/lib/utils"
import { TrendingUp, TrendingDown, Activity, BarChart3, Gauge } from "lucide-react"

interface QuickMetricsProps {
    rsi?: number
    vwap?: number
    currentPrice?: number
    volumeRatio?: number
    atr?: number
    currency?: string
}

function getRSIColor(rsi: number): string {
    if (rsi >= 70) return "text-red-400"      // Overbought
    if (rsi <= 30) return "text-green-400"    // Oversold
    return "text-yellow-400"                   // Neutral
}

function getRSILabel(rsi: number): string {
    if (rsi >= 70) return "Overbought"
    if (rsi <= 30) return "Oversold"
    return "Neutral"
}

function getVolumeRatioColor(ratio: number): string {
    if (ratio >= 2) return "text-purple-400"  // High volume
    if (ratio >= 1.5) return "text-blue-400"  // Above average
    if (ratio <= 0.5) return "text-gray-400"  // Low volume
    return "text-foreground"                   // Normal
}

function getVolumeRatioLabel(ratio: number): string {
    if (ratio >= 2) return "High"
    if (ratio >= 1.5) return "Above Avg"
    if (ratio <= 0.5) return "Low"
    return "Normal"
}

export function QuickMetrics({
    rsi,
    vwap,
    currentPrice,
    volumeRatio,
    atr,
    currency = "$",
}: QuickMetricsProps) {
    // Calculate VWAP position relative to current price
    const vwapPosition = vwap && currentPrice
        ? currentPrice > vwap ? "above" : currentPrice < vwap ? "below" : "at"
        : null

    const metrics = [
        rsi !== undefined && {
            label: "RSI(14)",
            value: rsi.toFixed(0),
            sublabel: getRSILabel(rsi),
            color: getRSIColor(rsi),
            icon: <Gauge className="h-3.5 w-3.5" />,
        },
        vwap !== undefined && {
            label: "VWAP",
            value: `${currency}${vwap.toFixed(2)}`,
            sublabel: vwapPosition ? `Price ${vwapPosition}` : undefined,
            color: vwapPosition === "above" ? "text-green-400" : vwapPosition === "below" ? "text-red-400" : "text-foreground",
            icon: <Activity className="h-3.5 w-3.5" />,
        },
        volumeRatio !== undefined && {
            label: "Vol Ratio",
            value: `${volumeRatio.toFixed(1)}x`,
            sublabel: getVolumeRatioLabel(volumeRatio),
            color: getVolumeRatioColor(volumeRatio),
            icon: <BarChart3 className="h-3.5 w-3.5" />,
        },
        atr !== undefined && {
            label: "ATR",
            value: `${currency}${atr.toFixed(2)}`,
            sublabel: "Volatility",
            color: "text-cyan-400",
            icon: <TrendingUp className="h-3.5 w-3.5" />,
        },
    ].filter(Boolean) as Array<{
        label: string
        value: string
        sublabel?: string
        color: string
        icon: React.ReactNode
    }>

    if (metrics.length === 0) return null

    return (
        <div className="flex items-center gap-1 flex-wrap">
            {metrics.map((metric, index) => (
                <div key={metric.label} className="flex items-center">
                    {index > 0 && (
                        <span className="mx-2 text-muted-foreground/30">│</span>
                    )}
                    <div className="flex items-center gap-1.5">
                        <span className={cn("opacity-60", metric.color)}>
                            {metric.icon}
                        </span>
                        <span className="text-xs text-muted-foreground">{metric.label}:</span>
                        <span className={cn("text-sm font-medium", metric.color)}>
                            {metric.value}
                        </span>
                        {metric.sublabel && (
                            <span className="text-xs text-muted-foreground">
                                ({metric.sublabel})
                            </span>
                        )}
                    </div>
                </div>
            ))}
        </div>
    )
}

// Compact version for chart overlay
export function QuickMetricsCompact({
    rsi,
    vwap,
    currentPrice,
    volumeRatio,
    currency = "$",
}: QuickMetricsProps) {
    return (
        <div className="flex items-center gap-4 text-xs font-mono">
            {rsi !== undefined && (
                <span className="flex items-center gap-1">
                    <span className="text-muted-foreground">RSI:</span>
                    <span className={getRSIColor(rsi)}>{rsi.toFixed(0)}</span>
                </span>
            )}
            {vwap !== undefined && (
                <span className="flex items-center gap-1">
                    <span className="text-muted-foreground">VWAP:</span>
                    <span className={
                        currentPrice && currentPrice > vwap
                            ? "text-green-400"
                            : currentPrice && currentPrice < vwap
                                ? "text-red-400"
                                : "text-foreground"
                    }>
                        {currency}{vwap.toFixed(2)}
                    </span>
                </span>
            )}
            {volumeRatio !== undefined && (
                <span className="flex items-center gap-1">
                    <span className="text-muted-foreground">Vol:</span>
                    <span className={getVolumeRatioColor(volumeRatio)}>
                        {volumeRatio.toFixed(1)}x
                    </span>
                </span>
            )}
        </div>
    )
}
