"use client"

import { useEffect, useRef, useState, useCallback } from "react"

interface LivePrice {
    symbol: string
    price: number
    change: number
    changePercent: number
    volume: number
    timestamp: Date
}

interface UseLiveStockDataOptions {
    symbol: string
    enabled?: boolean
    refreshInterval?: number // in milliseconds, default 5000 (5 seconds)
}

/**
 * Custom hook for live stock data using Yahoo Finance polling
 * (Free alternative to WebSocket - updates every 5 seconds)
 */
export function useLiveStockData({
    symbol,
    enabled = true,
    refreshInterval = 5000
}: UseLiveStockDataOptions) {
    const [livePrice, setLivePrice] = useState<LivePrice | null>(null)
    const [isConnected, setIsConnected] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const intervalRef = useRef<NodeJS.Timeout | null>(null)

    const fetchLivePrice = useCallback(async () => {
        if (!symbol) return

        try {
            // Use Yahoo Finance API through our backend
            const response = await fetch(`/api/live-price?symbol=${encodeURIComponent(symbol)}`)

            if (!response.ok) {
                throw new Error("Failed to fetch live price")
            }

            const data = await response.json()

            if (data.price) {
                setLivePrice({
                    symbol: data.symbol || symbol,
                    price: data.price,
                    change: data.change || 0,
                    changePercent: data.changePercent || 0,
                    volume: data.volume || 0,
                    timestamp: new Date(),
                })
                setIsConnected(true)
                setError(null)
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : "Connection error")
            setIsConnected(false)
        }
    }, [symbol])

    useEffect(() => {
        if (!enabled || !symbol) {
            setIsConnected(false)
            return
        }

        // Initial fetch
        fetchLivePrice()

        // Set up interval for polling
        intervalRef.current = setInterval(fetchLivePrice, refreshInterval)

        return () => {
            if (intervalRef.current) {
                clearInterval(intervalRef.current)
                intervalRef.current = null
            }
            setIsConnected(false)
        }
    }, [symbol, enabled, refreshInterval, fetchLivePrice])

    return {
        livePrice,
        isConnected,
        error,
        refresh: fetchLivePrice,
    }
}
