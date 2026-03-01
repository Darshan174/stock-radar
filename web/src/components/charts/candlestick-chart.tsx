"use client"

import { useEffect, useRef, useState, useMemo, useCallback } from "react"
import {
  createChart,
  ColorType,
  IChartApi,
  CandlestickData,
  Time,
  CrosshairMode,
  LineStyle,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  ISeriesApi,
} from "lightweight-charts"
import { ChartToolbar, ChartIndicators, defaultIndicators, ChartType } from "./chart-toolbar"
import {
  calculateSMA,
  calculateEMA,
  calculateBollingerBands,
} from "@/lib/indicators"

function useTheme() {
  const [isDark, setIsDark] = useState(true)

  useEffect(() => {
    const checkTheme = () => {
      setIsDark(document.documentElement.classList.contains("dark"))
    }
    checkTheme()
    const observer = new MutationObserver(checkTheme)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })
    return () => observer.disconnect()
  }, [])

  return isDark
}

interface CandlestickChartProps {
  data: {
    time: number
    open: number
    high: number
    low: number
    close: number
    volume?: number
  }[]
  liveTicks?: {
    time: number
    value: number
  }[]
  signals?: {
    time: string
    type: "buy" | "sell"
    price: number
  }[]
  height?: number
  showVolume?: boolean
  currency?: string
  isIntraday?: boolean
  timezone?: string
  interval?: string
}

function toEpochSeconds(time: Time | undefined): number | null {
  if (time === undefined) return null
  if (typeof time === "number") return Math.floor(time)
  if (typeof time === "string") {
    const dt = new Date(`${time}T00:00:00Z`)
    if (!Number.isNaN(dt.getTime())) return Math.floor(dt.getTime() / 1000)
    return null
  }
  if (typeof time === "object" && "year" in time) {
    return Math.floor(Date.UTC(time.year, time.month - 1, time.day) / 1000)
  }
  return null
}

function formatTime(
  value: Time | undefined,
  isIntraday: boolean,
  timezone: string,
  forCrosshair: boolean,
  showSeconds: boolean = false
): string {
  const epoch = toEpochSeconds(value)
  if (epoch == null) return ""
  const date = new Date(epoch * 1000)

  try {
    if (isIntraday) {
      return new Intl.DateTimeFormat("en-US", {
        timeZone: timezone,
        month: forCrosshair ? "short" : undefined,
        day: forCrosshair ? "2-digit" : undefined,
        hour: "2-digit",
        minute: "2-digit",
        second: showSeconds ? "2-digit" : undefined,
        hour12: false,
      }).format(date)
    }

    return new Intl.DateTimeFormat("en-US", {
      timeZone: timezone,
      month: "short",
      day: "2-digit",
      year: forCrosshair ? "numeric" : undefined,
    }).format(date)
  } catch {
    return date.toISOString()
  }
}

function ChartLegend({
  ohlc,
  change,
  volume,
  currency,
}: {
  ohlc: { o: number; h: number; l: number; c: number } | null
  change: number
  volume: number
  currency: string
}) {
  if (!ohlc) return null

  const isUp = ohlc.c >= ohlc.o
  const changeColor = isUp ? "#16a34a" : "#dc2626"

  return (
    <div className="absolute top-3 left-14 z-10 flex flex-wrap items-center gap-4 text-sm font-mono max-w-[calc(100%-120px)]">
      <span className="text-muted-foreground">
        O <span className="text-foreground font-medium">{currency}{ohlc.o.toFixed(2)}</span>
      </span>
      <span className="text-muted-foreground">
        H <span className="text-green-600 dark:text-green-400 font-medium">{currency}{ohlc.h.toFixed(2)}</span>
      </span>
      <span className="text-muted-foreground">
        L <span className="text-red-600 dark:text-red-400 font-medium">{currency}{ohlc.l.toFixed(2)}</span>
      </span>
      <span className="text-muted-foreground">
        C <span style={{ color: changeColor }} className="font-medium">{currency}{ohlc.c.toFixed(2)}</span>
      </span>
      <span style={{ color: changeColor }} className="font-medium">
        {change >= 0 ? "+" : ""}{change.toFixed(2)}%
      </span>
      {volume > 0 && (
        <span className="text-muted-foreground">
          Vol <span className="text-foreground font-medium">
            {volume >= 1000000 ? `${(volume / 1000000).toFixed(2)}M` : `${(volume / 1000).toFixed(0)}K`}
          </span>
        </span>
      )}
    </div>
  )
}

function IndicatorLegend({ indicators }: { indicators: ChartIndicators }) {
  const activeIndicators = []
  if (indicators.ema12) activeIndicators.push({ color: "#f59e0b", label: "EMA 12" })
  if (indicators.ema26) activeIndicators.push({ color: "#6366f1", label: "EMA 26" })
  if (indicators.sma20) activeIndicators.push({ color: "#fb923c", label: "SMA 20" })
  if (indicators.sma50) activeIndicators.push({ color: "#ec4899", label: "SMA 50" })
  if (indicators.sma200) activeIndicators.push({ color: "#8b5cf6", label: "SMA 200" })
  if (indicators.bollingerBands) activeIndicators.push({ color: "#38bdf8", label: "BB(20,2)" })

  if (activeIndicators.length === 0) return null

  return (
    <div className="absolute top-10 right-3 z-10 flex items-center gap-3 text-xs font-mono bg-background/80 backdrop-blur-sm px-2 py-1 rounded">
      {activeIndicators.map(({ color, label }) => (
        <span key={label} className="flex items-center gap-1.5">
          <span className="w-3 h-0.5 rounded" style={{ backgroundColor: color }} />
          <span className="text-muted-foreground">{label}</span>
        </span>
      ))}
    </div>
  )
}

export function CandlestickChart({
  data,
  liveTicks = [],
  signals = [],
  height = 600,
  showVolume = true,
  currency = "$",
  isIntraday = false,
  timezone = "UTC",
}: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const liveDotRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const mainSeriesRef = useRef<ISeriesApi<"Candlestick"> | ISeriesApi<"Line"> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<"Histogram"> | null>(null)
  const lineLengthRef = useRef(0)
  const visibleRangeRef = useRef<{ from: number; to: number } | null>(null)
  const chartTypeRef = useRef<ChartType>("line")
  const lineDataRef = useRef<{ time: Time; value: number }[]>([])
  const [isFullscreen, setIsFullscreen] = useState(false)
  const isDark = useTheme()
  const [indicators, setIndicators] = useState<ChartIndicators>(defaultIndicators)
  const [chartType, setChartType] = useState<ChartType>("line")
  const [legendData, setLegendData] = useState<{
    ohlc: { o: number; h: number; l: number; c: number } | null
    change: number
    volume: number
  }>({ ohlc: null, change: 0, volume: 0 })

  const fallbackLegendData = useMemo(() => {
    if (!data.length) return { ohlc: null, change: 0, volume: 0 }
    const lastCandle = data[data.length - 1]
    const prevCandle = data.length > 1 ? data[data.length - 2] : lastCandle
    const change = prevCandle.close > 0 ? ((lastCandle.close - prevCandle.close) / prevCandle.close) * 100 : 0
    return {
      ohlc: { o: lastCandle.open, h: lastCandle.high, l: lastCandle.low, c: lastCandle.close },
      change,
      volume: lastCandle.volume || 0,
    }
  }, [data])

  const indicatorData = useMemo(() => {
    if (!data.length) return null
    const closes = data.map((d) => d.close)
    const volumes = data.map((d) => d.volume || 0)
    return {
      ema12: calculateEMA(closes, 12),
      ema26: calculateEMA(closes, 26),
      sma20: calculateSMA(closes, 20),
      sma50: calculateSMA(closes, 50),
      sma200: calculateSMA(closes, 200),
      bollingerBands: calculateBollingerBands(closes, 20, 2),
      volumeSma: calculateSMA(volumes, 20),
    }
  }, [data])

  const chartData = useMemo(() => {
    if (!data.length) return { candles: [], lineData: [], volumes: [] }

    const candles: CandlestickData<Time>[] = data.map((d) => ({
      time: d.time as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }))

    const baseLineData = data.map((d) => ({
      time: d.time as Time,
      value: d.close,
    }))

    const lineData = [...baseLineData]
    if (liveTicks.length > 0) {
      let lastTime = Number(lineData[lineData.length - 1]?.time ?? 0)
      for (const tick of liveTicks) {
        if (!Number.isFinite(tick.time) || !Number.isFinite(tick.value)) continue
        const nextTime = tick.time <= lastTime ? lastTime + 0.0001 : tick.time
        lineData.push({
          time: nextTime as Time,
          value: tick.value,
        })
        lastTime = nextTime
      }
    }

    const volumes = data.map((d) => ({
      time: d.time as Time,
      value: d.volume || 0,
      color: d.close >= d.open ? "rgba(22, 163, 74, 0.35)" : "rgba(220, 38, 38, 0.35)",
    }))

    return { candles, lineData, volumes }
  }, [data, liveTicks])

  useEffect(() => {
    chartTypeRef.current = chartType
  }, [chartType])

  useEffect(() => {
    lineDataRef.current = chartData.lineData
  }, [chartData.lineData])

  const handleZoomIn = useCallback(() => {
    if (!chartRef.current) return
    const timeScale = chartRef.current.timeScale()
    const visibleRange = timeScale.getVisibleLogicalRange()
    if (!visibleRange) return
    const newRange = {
      from: visibleRange.from + (visibleRange.to - visibleRange.from) * 0.1,
      to: visibleRange.to - (visibleRange.to - visibleRange.from) * 0.1,
    }
    timeScale.setVisibleLogicalRange(newRange)
  }, [])

  const handleZoomOut = useCallback(() => {
    if (!chartRef.current) return
    const timeScale = chartRef.current.timeScale()
    const visibleRange = timeScale.getVisibleLogicalRange()
    if (!visibleRange) return
    const newRange = {
      from: visibleRange.from - (visibleRange.to - visibleRange.from) * 0.2,
      to: visibleRange.to + (visibleRange.to - visibleRange.from) * 0.2,
    }
    timeScale.setVisibleLogicalRange(newRange)
  }, [])

  const handleResetZoom = useCallback(() => {
    chartRef.current?.timeScale().fitContent()
  }, [])

  const handleFullscreen = useCallback(() => {
    if (!chartContainerRef.current) return
    if (!isFullscreen) {
      chartContainerRef.current.requestFullscreen?.()
    } else {
      document.exitFullscreen?.()
    }
  }, [isFullscreen])

  const handleScreenshot = useCallback(() => {
    if (!chartContainerRef.current) return
    const canvas = chartContainerRef.current.querySelector("canvas")
    if (!canvas) return
    const link = document.createElement("a")
    link.download = `chart-${new Date().toISOString().split("T")[0]}.png`
    link.href = canvas.toDataURL("image/png")
    link.click()
  }, [])

  const handleIndicatorChange = useCallback((key: keyof ChartIndicators, value: boolean) => {
    setIndicators((prev) => ({ ...prev, [key]: value }))
  }, [])

  const updateLiveDotPosition = useCallback(() => {
    const dot = liveDotRef.current
    const chart = chartRef.current
    const mainSeries = mainSeriesRef.current
    if (!dot || !chart || !mainSeries) return

    if (chartTypeRef.current !== "line" || lineDataRef.current.length === 0) {
      dot.style.opacity = "0"
      return
    }

    const lastPoint = lineDataRef.current[lineDataRef.current.length - 1]
    const x = chart.timeScale().timeToCoordinate(lastPoint.time as Time)
    const y = (mainSeries as ISeriesApi<"Line">).priceToCoordinate(lastPoint.value)

    if (x == null || y == null) {
      dot.style.opacity = "0"
      return
    }

    dot.style.opacity = "1"
    dot.style.transform = `translate(${x - 5}px, ${y - 5}px)`
  }, [])

  useEffect(() => {
    const handleFullscreenChange = () => setIsFullscreen(!!document.fullscreenElement)
    document.addEventListener("fullscreenchange", handleFullscreenChange)
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange)
  }, [])

  useEffect(() => {
    if (!chartRef.current || !mainSeriesRef.current) return

    if (chartType === "candlestick") {
      ;(mainSeriesRef.current as ISeriesApi<"Candlestick">).setData(chartData.candles)
      lineLengthRef.current = chartData.lineData.length
    } else {
      const lineSeries = mainSeriesRef.current as ISeriesApi<"Line">
      if (
        lineLengthRef.current > 0 &&
        chartData.lineData.length === lineLengthRef.current + 1
      ) {
        lineSeries.update(chartData.lineData[chartData.lineData.length - 1])
      } else {
        lineSeries.setData(chartData.lineData)
      }
      lineLengthRef.current = chartData.lineData.length
    }

    if (showVolume && volumeSeriesRef.current) {
      volumeSeriesRef.current.setData(chartData.volumes)
    }
    updateLiveDotPosition()
  }, [chartData, chartType, showVolume, updateLiveDotPosition])

  // Rebuild chart only on structural/view changes, not on every data tick.
  // eslint-disable react-hooks/exhaustive-deps
  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return

    if (chartRef.current) {
      const existingRange = chartRef.current.timeScale().getVisibleLogicalRange()
      if (existingRange) {
        visibleRangeRef.current = { from: existingRange.from, to: existingRange.to }
      }
      chartRef.current.remove()
      chartRef.current = null
      mainSeriesRef.current = null
      volumeSeriesRef.current = null
    }

    const chartHeight = isFullscreen ? window.innerHeight : height
    const showSecondLabels = isIntraday && chartType === "line"
    const bgColor = isDark ? "#060b15" : "#ffffff"
    const textColor = isDark ? "#c9d1e1" : "#0f172a"
    const gridColor = isDark ? "rgba(148, 163, 184, 0.12)" : "rgba(15, 23, 42, 0.08)"
    const crosshairLabelBg = isDark ? "#0f172a" : "#f8fafc"
    const borderColor = isDark ? "rgba(148, 163, 184, 0.18)" : "rgba(15, 23, 42, 0.16)"

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: bgColor },
        textColor,
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
      },
      grid: {
        vertLines: { color: gridColor, style: LineStyle.Solid },
        horzLines: { color: gridColor, style: LineStyle.Solid },
      },
      width: chartContainerRef.current.clientWidth,
      height: chartHeight,
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#64748b", width: 1, style: LineStyle.Dashed, labelBackgroundColor: crosshairLabelBg },
        horzLine: { color: "#64748b", width: 1, style: LineStyle.Dashed, labelBackgroundColor: crosshairLabelBg },
      },
      timeScale: {
        borderColor,
        timeVisible: isIntraday,
        secondsVisible: showSecondLabels,
        fixLeftEdge: true,
        rightOffset: 2,
        barSpacing: isIntraday ? 6 : 8,
        tickMarkFormatter: (time: Time) =>
          formatTime(time, isIntraday, timezone, false, showSecondLabels),
      },
      rightPriceScale: {
        borderColor,
        scaleMargins: { top: 0.08, bottom: showVolume ? 0.24 : 0.08 },
      },
      localization: {
        locale: "en-US",
        priceFormatter: (price: number) => `${currency}${price.toFixed(2)}`,
        timeFormatter: (time: Time) =>
          formatTime(time, isIntraday, timezone, true, showSecondLabels),
      },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    })

    chartRef.current = chart

    let mainSeries: ISeriesApi<"Candlestick"> | ISeriesApi<"Line">
    let lineSeries: ISeriesApi<"Line"> | null = null

    if (chartType === "candlestick") {
      mainSeries = chart.addSeries(CandlestickSeries, {
        upColor: "#16a34a",
        downColor: "#dc2626",
        borderUpColor: "#16a34a",
        borderDownColor: "#dc2626",
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
        borderVisible: true,
      })
      mainSeries.setData(chartData.candles)
    } else {
      lineSeries = chart.addSeries(LineSeries, {
        color: "#2563eb",
        lineWidth: 2,
        crosshairMarkerVisible: false,
      })
      mainSeries = lineSeries
      mainSeries.setData(chartData.lineData)
      lineLengthRef.current = chartData.lineData.length
    }
    mainSeriesRef.current = mainSeries

    if (indicatorData && chartType === "candlestick") {
      if (indicators.ema12) {
        const ema12Series = chart.addSeries(LineSeries, {
          color: "#f59e0b",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const ema12Data = data
          .map((d, i) => ({ time: d.time as Time, value: indicatorData.ema12[i] }))
          .filter((d) => d.value !== null) as { time: Time; value: number }[]
        ema12Series.setData(ema12Data)
      }

      if (indicators.ema26) {
        const ema26Series = chart.addSeries(LineSeries, {
          color: "#6366f1",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const ema26Data = data
          .map((d, i) => ({ time: d.time as Time, value: indicatorData.ema26[i] }))
          .filter((d) => d.value !== null) as { time: Time; value: number }[]
        ema26Series.setData(ema26Data)
      }

      if (indicators.sma20) {
        const sma20Series = chart.addSeries(LineSeries, {
          color: "#fb923c",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const sma20Data = data
          .map((d, i) => ({ time: d.time as Time, value: indicatorData.sma20[i] }))
          .filter((d) => d.value !== null) as { time: Time; value: number }[]
        sma20Series.setData(sma20Data)
      }

      if (indicators.sma50) {
        const sma50Series = chart.addSeries(LineSeries, {
          color: "#ec4899",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const sma50Data = data
          .map((d, i) => ({ time: d.time as Time, value: indicatorData.sma50[i] }))
          .filter((d) => d.value !== null) as { time: Time; value: number }[]
        sma50Series.setData(sma50Data)
      }

      if (indicators.sma200) {
        const sma200Series = chart.addSeries(LineSeries, {
          color: "#8b5cf6",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const sma200Data = data
          .map((d, i) => ({ time: d.time as Time, value: indicatorData.sma200[i] }))
          .filter((d) => d.value !== null) as { time: Time; value: number }[]
        sma200Series.setData(sma200Data)
      }

      if (indicators.bollingerBands) {
        const bb = indicatorData.bollingerBands

        const upperSeries = chart.addSeries(LineSeries, {
          color: "rgba(56, 189, 248, 0.7)",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        upperSeries.setData(
          data
            .map((d, i) => ({ time: d.time as Time, value: bb.upper[i] }))
            .filter((d) => d.value !== null) as { time: Time; value: number }[]
        )

        const middleSeries = chart.addSeries(LineSeries, {
          color: "#38bdf8",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        middleSeries.setData(
          data
            .map((d, i) => ({ time: d.time as Time, value: bb.middle[i] }))
            .filter((d) => d.value !== null) as { time: Time; value: number }[]
        )

        const lowerSeries = chart.addSeries(LineSeries, {
          color: "rgba(56, 189, 248, 0.7)",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        lowerSeries.setData(
          data
            .map((d, i) => ({ time: d.time as Time, value: bb.lower[i] }))
            .filter((d) => d.value !== null) as { time: Time; value: number }[]
        )
      }
    }

    if (showVolume && chartData.volumes.length > 0) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      })
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.84, bottom: 0 },
      })
      volumeSeries.setData(chartData.volumes)
      volumeSeriesRef.current = volumeSeries
    }

    if (data.length > 0) {
      const lastPrice = data[data.length - 1].close
      mainSeries.createPriceLine({
        price: lastPrice,
        color: data[data.length - 1].close >= data[data.length - 1].open ? "#16a34a" : "#dc2626",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "",
      })
    }

    const indexByTime = new Map<number, number>()
    const pointByTime = new Map<number, (typeof data)[number]>()
    data.forEach((row, idx) => {
      indexByTime.set(row.time, idx)
      pointByTime.set(row.time, row)
    })

    chart.subscribeCrosshairMove((param) => {
      const epoch = toEpochSeconds(param.time)
      if (epoch == null || !param.seriesData) {
        setLegendData((prev) => (prev.ohlc ? { ohlc: null, change: 0, volume: 0 } : prev))
        return
      }

      const dataPoint = pointByTime.get(epoch)
      const idx = indexByTime.get(epoch)
      if (!dataPoint || idx == null) return

      const prevPoint = idx > 0 ? data[idx - 1] : dataPoint
      const change = prevPoint.close > 0 ? ((dataPoint.close - prevPoint.close) / prevPoint.close) * 100 : 0

      const seriesData = param.seriesData.get(mainSeries)
      const ohlcData =
        seriesData && "open" in seriesData
          ? {
              o: (seriesData as CandlestickData<Time>).open,
              h: (seriesData as CandlestickData<Time>).high,
              l: (seriesData as CandlestickData<Time>).low,
              c: (seriesData as CandlestickData<Time>).close,
            }
          : { o: dataPoint.open, h: dataPoint.high, l: dataPoint.low, c: dataPoint.close }

      setLegendData({
        ohlc: ohlcData,
        change,
        volume: dataPoint.volume || 0,
      })
    })

    if (visibleRangeRef.current) {
      chart.timeScale().setVisibleLogicalRange(visibleRangeRef.current)
    } else {
      chart.timeScale().fitContent()
    }
    updateLiveDotPosition()

    const handleVisibleRangeChange = () => {
      const range = chart.timeScale().getVisibleLogicalRange()
      if (range) {
        visibleRangeRef.current = { from: range.from, to: range.to }
      }
      updateLiveDotPosition()
    }
    chart.timeScale().subscribeVisibleLogicalRangeChange(handleVisibleRangeChange)

    const handleResize = () => {
      if (!chartContainerRef.current || !chartRef.current) return
      chartRef.current.applyOptions({
        width: chartContainerRef.current.clientWidth,
        height: isFullscreen ? window.innerHeight : height,
      })
      updateLiveDotPosition()
    }

    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleVisibleRangeChange)
      chart.remove()
      if (chartRef.current === chart) {
      chartRef.current = null
      }
    }
  }, [
    data.length,
    signals,
    height,
    showVolume,
    currency,
    indicators,
    isFullscreen,
    isDark,
    isIntraday,
    timezone,
    chartType,
    updateLiveDotPosition,
  ])
  // eslint-enable react-hooks/exhaustive-deps

  if (!data.length) {
    return (
      <div
        style={{ height }}
        className="bg-background rounded-lg flex items-center justify-center border"
      >
        <span className="text-muted-foreground">No chart data available</span>
      </div>
    )
  }

  return (
    <div
      ref={chartContainerRef}
      className="relative w-full rounded-lg overflow-hidden"
      style={{ height: isFullscreen ? "100vh" : height, backgroundColor: isDark ? "#060b15" : "#ffffff" }}
    >
      <ChartToolbar
        onZoomIn={handleZoomIn}
        onZoomOut={handleZoomOut}
        onResetZoom={handleResetZoom}
        onFullscreen={handleFullscreen}
        onScreenshot={handleScreenshot}
        isFullscreen={isFullscreen}
        indicators={indicators}
        onIndicatorChange={handleIndicatorChange}
        chartType={chartType}
        onToggleChartType={() => setChartType((prev) => (prev === "line" ? "candlestick" : "line"))}
      />
      <div
        ref={liveDotRef}
        className="pointer-events-none absolute z-10 h-2.5 w-2.5 rounded-full border-2 border-white/80 bg-blue-500 shadow-[0_0_12px_rgba(37,99,235,0.95)] stockradar-live-dot"
        style={{ opacity: 0, transform: "translate(-9999px,-9999px)" }}
      />
      <ChartLegend
        ohlc={legendData.ohlc || fallbackLegendData.ohlc}
        change={legendData.ohlc ? legendData.change : fallbackLegendData.change}
        volume={legendData.ohlc ? legendData.volume : fallbackLegendData.volume}
        currency={currency}
      />
      {chartType === "candlestick" && <IndicatorLegend indicators={indicators} />}
    </div>
  )
}
