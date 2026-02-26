"use client"

import { useEffect, useRef, useState, useCallback } from "react"

export interface LivePrice {
    symbol: string
    price: number
    change: number
    changePercent: number
    volume: number
    timestamp: Date
    previousClose?: number
}

interface UseLiveStockDataOptions {
    symbol: string
    enabled?: boolean
    refreshInterval?: number
    preferWebSocket?: boolean
    updateThrottleMs?: number
}

interface FinnhubTrade {
    s: string
    p: number
    t: number
    v?: number
}

interface FinnhubTradeMessage {
    type?: string
    data?: FinnhubTrade[]
}

export function useLiveStockData({
    symbol,
    enabled = true,
    refreshInterval = 3000,
    preferWebSocket = true,
    updateThrottleMs = 450,
}: UseLiveStockDataOptions) {
    const [livePrice, setLivePrice] = useState<LivePrice | null>(null)
    const [isConnected, setIsConnected] = useState(false)
    const [error, setError] = useState<string | null>(null)
    const [transport, setTransport] = useState<"idle" | "websocket" | "polling">("idle")

    const wsRef = useRef<WebSocket | null>(null)
    const pollRef = useRef<ReturnType<typeof setInterval> | null>(null)
    const reconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null)
    const throttleRef = useRef<ReturnType<typeof setTimeout> | null>(null)
    const disconnectRef = useRef<ReturnType<typeof setTimeout> | null>(null)
    const shouldReconnectRef = useRef(false)
    const previousCloseRef = useRef<number | null>(null)
    const lastPublishedAtRef = useRef(0)
    const pendingTradeRef = useRef<FinnhubTrade | null>(null)
    const consecutivePollFailuresRef = useRef(0)

    const toFinnhubSymbol = useCallback((raw: string): string => {
        const s = raw.toUpperCase().trim()
        if (s.endsWith(".NS")) return `NSE:${s.replace(".NS", "")}`
        if (s.endsWith(".BO")) return `BSE:${s.replace(".BO", "")}`
        return s
    }, [])

    const clearAllTimers = useCallback(() => {
        if (pollRef.current) {
            clearInterval(pollRef.current)
            pollRef.current = null
        }
        if (reconnectRef.current) {
            clearTimeout(reconnectRef.current)
            reconnectRef.current = null
        }
        if (throttleRef.current) {
            clearTimeout(throttleRef.current)
            throttleRef.current = null
        }
        if (disconnectRef.current) {
            clearTimeout(disconnectRef.current)
            disconnectRef.current = null
        }
    }, [])

    const markConnected = useCallback(() => {
        consecutivePollFailuresRef.current = 0
        if (disconnectRef.current) {
            clearTimeout(disconnectRef.current)
            disconnectRef.current = null
        }
        setIsConnected(true)
    }, [])

    const scheduleDisconnectGrace = useCallback((delayMs: number = 15000) => {
        if (disconnectRef.current) return
        disconnectRef.current = setTimeout(() => {
            setIsConnected(false)
            disconnectRef.current = null
        }, delayMs)
    }, [])

    const publishTrade = useCallback(
        (trade: FinnhubTrade) => {
            const lastPrice = Number(trade.p)
            const tradeTs = Number(trade.t)
            if (!Number.isFinite(lastPrice)) return

            setLivePrice((prev) => {
                if (prev && lastPrice === prev.price) {
                    return {
                        ...prev,
                        timestamp:
                            Number.isFinite(tradeTs) && tradeTs > 0
                                ? new Date(tradeTs)
                                : prev.timestamp,
                    }
                }

                const previousClose =
                    previousCloseRef.current ??
                    prev?.previousClose ??
                    (prev ? prev.price - prev.change : lastPrice)
                if (typeof previousClose === "number" && previousClose > 0) {
                    previousCloseRef.current = previousClose
                }

                const change = lastPrice - previousClose
                const changePercent = previousClose > 0 ? (change / previousClose) * 100 : 0
                const nextTimestamp =
                    Number.isFinite(tradeTs) && tradeTs > 0
                        ? new Date(tradeTs)
                        : new Date()

                return {
                    symbol: trade.s || symbol.toUpperCase(),
                    price: lastPrice,
                    change,
                    changePercent,
                    volume: prev?.volume || 0,
                    timestamp: nextTimestamp,
                    previousClose,
                }
            })

            lastPublishedAtRef.current = Date.now()
            markConnected()
        },
        [symbol, markConnected]
    )

    const flushPendingTrade = useCallback(() => {
        throttleRef.current = null
        if (!pendingTradeRef.current) return
        const pending = pendingTradeRef.current
        pendingTradeRef.current = null
        publishTrade(pending)
    }, [publishTrade])

    const queueTrade = useCallback(
        (trade: FinnhubTrade) => {
            const now = Date.now()
            const elapsed = now - lastPublishedAtRef.current

            if (elapsed >= updateThrottleMs && !throttleRef.current) {
                publishTrade(trade)
                return
            }

            pendingTradeRef.current = trade
            if (!throttleRef.current) {
                const remaining = Math.max(50, updateThrottleMs - elapsed)
                throttleRef.current = setTimeout(flushPendingTrade, remaining)
            }
        },
        [flushPendingTrade, publishTrade, updateThrottleMs]
    )

    const fetchLivePrice = useCallback(async () => {
        if (!symbol) return

        try {
            const response = await fetch(`/api/live-price?symbol=${encodeURIComponent(symbol)}`, {
                cache: "no-store",
            })

            if (!response.ok) {
                throw new Error("Failed to fetch live price")
            }

            const data = await response.json()

            if (data.price) {
                const previousClose =
                    typeof data.previousClose === "number" ? data.previousClose : undefined
                if (typeof previousClose === "number" && previousClose > 0) {
                    previousCloseRef.current = previousClose
                }

                setLivePrice({
                    symbol: data.symbol || symbol,
                    price: data.price,
                    change: data.change || 0,
                    changePercent: data.changePercent || 0,
                    volume: data.volume || 0,
                    timestamp: data.timestamp ? new Date(data.timestamp) : new Date(),
                    previousClose,
                })

                markConnected()
                setError(null)
            }
        } catch (err) {
            const wsOpen = wsRef.current?.readyState === WebSocket.OPEN
            if (!wsOpen) {
                consecutivePollFailuresRef.current += 1
                setError(err instanceof Error ? err.message : "Connection error")
                if (consecutivePollFailuresRef.current >= 3) {
                    setIsConnected(false)
                }
            }
        }
    }, [symbol, markConnected])

    const startPolling = useCallback(() => {
        if (!enabled || !symbol) return
        clearAllTimers()
        setTransport("polling")
        fetchLivePrice()
        pollRef.current = setInterval(fetchLivePrice, refreshInterval)
    }, [enabled, symbol, clearAllTimers, fetchLivePrice, refreshInterval])

    const connectWebSocket = useCallback(async () => {
        if (!enabled || !symbol || !preferWebSocket) return

        try {
            const streamSymbol = toFinnhubSymbol(symbol)
            const configRes = await fetch("/api/live-stream-token", { cache: "no-store" })
            if (!configRes.ok) {
                throw new Error("WebSocket config unavailable")
            }

            const config = await configRes.json()
            if (!config?.enabled || !config?.wsUrl) {
                startPolling()
                return
            }

            if (pollRef.current) {
                clearInterval(pollRef.current)
                pollRef.current = null
            }

            const ws = new WebSocket(config.wsUrl as string)
            wsRef.current = ws

            ws.onopen = () => {
                setTransport("websocket")
                markConnected()
                setError(null)
                ws.send(
                    JSON.stringify({
                        type: "subscribe",
                        symbol: streamSymbol,
                    })
                )

                // Keep a periodic snapshot for volume/prev-close consistency.
                if (!pollRef.current) {
                    pollRef.current = setInterval(fetchLivePrice, 60000)
                }
            }

            ws.onmessage = (event) => {
                try {
                    const payload = JSON.parse(event.data as string) as FinnhubTradeMessage
                    if (payload?.type !== "trade" || !Array.isArray(payload?.data)) return

                    const trades = payload.data.filter((t) => t?.s === streamSymbol)
                    if (!trades.length) return

                    const lastTrade = trades[trades.length - 1]
                    queueTrade(lastTrade)
                } catch {
                    // Ignore malformed messages
                }
            }

            ws.onerror = () => {
                setError("WebSocket error")
            }

            ws.onclose = () => {
                scheduleDisconnectGrace(15000)
                startPolling()
                if (shouldReconnectRef.current) {
                    reconnectRef.current = setTimeout(() => {
                        connectWebSocket()
                    }, 3000)
                }
            }
        } catch {
            startPolling()
        }
    }, [
        enabled,
        symbol,
        preferWebSocket,
        startPolling,
        fetchLivePrice,
        queueTrade,
        toFinnhubSymbol,
        markConnected,
        scheduleDisconnectGrace,
    ])

    useEffect(() => {
        if (!enabled || !symbol) {
            clearAllTimers()
            if (wsRef.current) {
                wsRef.current.close()
                wsRef.current = null
            }
            setIsConnected(false)
            setTransport("idle")
            return
        }

        shouldReconnectRef.current = true
        fetchLivePrice()
        if (preferWebSocket) {
            connectWebSocket()
        } else {
            startPolling()
        }

        return () => {
            shouldReconnectRef.current = false
            clearAllTimers()
            if (wsRef.current) {
                try {
                    wsRef.current.send(
                        JSON.stringify({
                            type: "unsubscribe",
                            symbol: toFinnhubSymbol(symbol),
                        })
                    )
                } catch {
                    // no-op
                }
                wsRef.current.close()
                wsRef.current = null
            }
            setIsConnected(false)
            setTransport("idle")
        }
    }, [
        symbol,
        enabled,
        preferWebSocket,
        refreshInterval,
        fetchLivePrice,
        connectWebSocket,
        startPolling,
        clearAllTimers,
        toFinnhubSymbol,
        updateThrottleMs,
    ])

    return {
        livePrice,
        isConnected,
        error,
        transport,
        refresh: fetchLivePrice,
    }
}
