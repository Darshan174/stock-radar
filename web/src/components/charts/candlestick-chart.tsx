"use client"

import { useEffect, useRef, useState, useMemo, useCallback } from "react"

// Theme detection hook
function useTheme() {
  const [isDark, setIsDark] = useState(true)

  useEffect(() => {
    const checkTheme = () => {
      setIsDark(document.documentElement.classList.contains("dark"))
    }
    checkTheme()

    // Watch for theme changes
    const observer = new MutationObserver(checkTheme)
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ["class"] })

    return () => observer.disconnect()
  }, [])

  return isDark
}
import {
  createChart,
  ColorType,
  IChartApi,
  CandlestickData,
  Time,
  CrosshairMode,
  LineStyle,
  LineType,
  CandlestickSeries,
  LineSeries,
  HistogramSeries,
  AreaSeries,
  BarSeries,
  BaselineSeries,
  ISeriesApi,
} from "lightweight-charts"
import { ChartToolbar, ChartIndicators, defaultIndicators, ChartType } from "./chart-toolbar"
import {
  calculateSMA,
  calculateEMA,
  calculateBollingerBands,
  calculateRSI,
} from "@/lib/indicators"

interface CandlestickChartProps {
  data: {
    time: string
    open: number
    high: number
    low: number
    close: number
    volume?: number
  }[]
  signals?: {
    time: string
    type: "buy" | "sell"
    price: number
  }[]
  height?: number
  showVolume?: boolean
  currency?: string
}

// Heikin Ashi calculation helper
function calculateHeikinAshi(data: CandlestickChartProps['data']): CandlestickData<Time>[] {
  if (data.length === 0) return []

  const result: CandlestickData<Time>[] = []

  for (let i = 0; i < data.length; i++) {
    const current = data[i]
    const prev = i > 0 ? result[i - 1] : null

    const haClose = (current.open + current.high + current.low + current.close) / 4
    const haOpen = prev
      ? (prev.open + prev.close) / 2
      : (current.open + current.close) / 2
    const haHigh = Math.max(current.high, haOpen, haClose)
    const haLow = Math.min(current.low, haOpen, haClose)

    result.push({
      time: current.time as Time,
      open: haOpen,
      high: haHigh,
      low: haLow,
      close: haClose,
    })
  }

  return result
}

// Custom Legend Component
function ChartLegend({
  ohlc,
  change,
  volume,
  currency,
  indicators,
}: {
  ohlc: { o: number; h: number; l: number; c: number } | null
  change: number
  volume: number
  currency: string
  indicators: ChartIndicators
}) {
  if (!ohlc) return null

  const isUp = ohlc.c >= ohlc.o
  const changeColor = isUp ? "#26a69a" : "#ef5350"

  return (
    <div className="absolute top-3 left-14 z-10 flex flex-wrap items-center gap-4 text-sm font-mono max-w-[calc(100%-120px)]">
      <span className="text-muted-foreground">
        O <span className="text-foreground font-medium">{currency}{ohlc.o.toFixed(2)}</span>
      </span>
      <span className="text-muted-foreground">
        H <span className="text-[#26a69a] font-medium">{currency}{ohlc.h.toFixed(2)}</span>
      </span>
      <span className="text-muted-foreground">
        L <span className="text-[#ef5350] font-medium">{currency}{ohlc.l.toFixed(2)}</span>
      </span>
      <span className="text-muted-foreground">
        C <span style={{ color: changeColor }} className="font-medium">{currency}{ohlc.c.toFixed(2)}</span>
      </span>
      <span style={{ color: changeColor }} className="font-medium">
        {isUp ? "+" : ""}{change.toFixed(2)}%
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

// Indicator legend
function IndicatorLegend({ indicators }: { indicators: ChartIndicators }) {
  const activeIndicators = []
  if (indicators.ema12) activeIndicators.push({ color: "#f7931a", label: "EMA 12" })
  if (indicators.ema26) activeIndicators.push({ color: "#627eea", label: "EMA 26" })
  if (indicators.sma20) activeIndicators.push({ color: "#ff9800", label: "SMA 20" })
  if (indicators.sma50) activeIndicators.push({ color: "#e91e63", label: "SMA 50" })
  if (indicators.sma200) activeIndicators.push({ color: "#9c27b0", label: "SMA 200" })
  if (indicators.bollingerBands) activeIndicators.push({ color: "#2196f3", label: "BB(20,2)" })

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
  signals = [],
  height = 600,
  showVolume = true,
  currency = "$",
}: CandlestickChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const [isFullscreen, setIsFullscreen] = useState(false)
  const isDark = useTheme()
  const [indicators, setIndicators] = useState<ChartIndicators>(defaultIndicators)
  const [chartType, setChartType] = useState<ChartType>("candlestick")
  const [legendData, setLegendData] = useState<{
    ohlc: { o: number; h: number; l: number; c: number } | null
    change: number
    volume: number
  }>({ ohlc: null, change: 0, volume: 0 })

  // Calculate all indicator data
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
      rsi: calculateRSI(closes, 14),
      volumeSma: calculateSMA(volumes, 20),
    }
  }, [data])

  // Format data for the chart
  const chartData = useMemo(() => {
    if (!data.length) return { candles: [], heikinAshi: [], lineData: [], volumes: [], baselineValue: 0 }

    const candles: CandlestickData<Time>[] = data.map((d) => ({
      time: d.time as Time,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    }))

    const heikinAshi = calculateHeikinAshi(data)

    const lineData = data.map((d) => ({
      time: d.time as Time,
      value: d.close,
    }))

    const volumes = data.map((d) => ({
      time: d.time as Time,
      value: d.volume || 0,
      color: d.close >= d.open ? "rgba(38, 166, 154, 0.4)" : "rgba(239, 83, 80, 0.4)",
    }))

    // Baseline value - use first close price
    const baselineValue = data[0].close

    return { candles, heikinAshi, lineData, volumes, baselineValue }
  }, [data, chartType])

  // Toolbar handlers
  const handleZoomIn = useCallback(() => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale()
      const visibleRange = timeScale.getVisibleLogicalRange()
      if (visibleRange) {
        const newRange = {
          from: visibleRange.from + (visibleRange.to - visibleRange.from) * 0.1,
          to: visibleRange.to - (visibleRange.to - visibleRange.from) * 0.1,
        }
        timeScale.setVisibleLogicalRange(newRange)
      }
    }
  }, [])

  const handleZoomOut = useCallback(() => {
    if (chartRef.current) {
      const timeScale = chartRef.current.timeScale()
      const visibleRange = timeScale.getVisibleLogicalRange()
      if (visibleRange) {
        const newRange = {
          from: visibleRange.from - (visibleRange.to - visibleRange.from) * 0.2,
          to: visibleRange.to + (visibleRange.to - visibleRange.from) * 0.2,
        }
        timeScale.setVisibleLogicalRange(newRange)
      }
    }
  }, [])

  const handleResetZoom = useCallback(() => {
    if (chartRef.current) {
      chartRef.current.timeScale().fitContent()
    }
  }, [])

  const handleFullscreen = useCallback(() => {
    if (!chartContainerRef.current) return

    if (!isFullscreen) {
      if (chartContainerRef.current.requestFullscreen) {
        chartContainerRef.current.requestFullscreen()
      }
    } else {
      if (document.exitFullscreen) {
        document.exitFullscreen()
      }
    }
  }, [isFullscreen])

  const handleScreenshot = useCallback(() => {
    if (!chartContainerRef.current) return

    // Use html2canvas or simple canvas capture
    const canvas = chartContainerRef.current.querySelector("canvas")
    if (canvas) {
      const link = document.createElement("a")
      link.download = `chart-${new Date().toISOString().split("T")[0]}.png`
      link.href = canvas.toDataURL("image/png")
      link.click()
    }
  }, [])

  const handleIndicatorChange = useCallback((key: keyof ChartIndicators, value: boolean) => {
    setIndicators((prev) => ({ ...prev, [key]: value }))
  }, [])

  // Fullscreen event listener
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(!!document.fullscreenElement)
    }
    document.addEventListener("fullscreenchange", handleFullscreenChange)
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange)
  }, [])

  // Main chart effect
  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return

    // Clear previous chart
    if (chartRef.current) {
      chartRef.current.remove()
      chartRef.current = null
    }

    const chartHeight = isFullscreen ? window.innerHeight : height

    // Theme-aware colors
    const bgColor = isDark ? "#000000" : "#ffffff"
    const textColor = isDark ? "#d1d4dc" : "#131722"
    const gridColor = isDark ? "rgba(42, 46, 57, 0.6)" : "rgba(42, 46, 57, 0.2)"
    const crosshairLabelBg = isDark ? "#2a2e39" : "#f0f3fa"
    const borderColor = isDark ? "rgba(42, 46, 57, 0.8)" : "rgba(42, 46, 57, 0.3)"

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: bgColor },
        textColor: textColor,
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
        vertLine: { color: "#758696", width: 1, style: LineStyle.Dashed, labelBackgroundColor: crosshairLabelBg },
        horzLine: { color: "#758696", width: 1, style: LineStyle.Dashed, labelBackgroundColor: crosshairLabelBg },
      },
      timeScale: {
        borderColor: borderColor,
        timeVisible: true,
        secondsVisible: false,
        fixLeftEdge: true,
        // Allow scrolling past right edge for room
      },
      rightPriceScale: {
        borderColor: borderColor,
        scaleMargins: { top: 0.1, bottom: showVolume ? 0.25 : 0.1 },
      },
      localization: {
        locale: "en-IN",
        priceFormatter: (price: number) => currency + price.toFixed(2),
        timeFormatter: (time: number | string) => {
          let date: Date

          // Handle string dates (daily data uses "YYYY-MM-DD" format)
          if (typeof time === "string") {
            date = new Date(time + "T00:00:00+05:30") // Parse as IST
            return date.toLocaleDateString("en-IN", {
              day: "numeric",
              month: "short"
            })
          }

          // Handle Unix timestamps (intraday data)
          // If timestamp is in seconds (< year 2100 in ms), multiply by 1000
          const ts = time < 10000000000 ? time * 1000 : time
          date = new Date(ts)

          // Check if valid date
          if (isNaN(date.getTime())) {
            return String(time)
          }

          // For intraday, show time in IST
          return date.toLocaleTimeString("en-IN", {
            timeZone: "Asia/Kolkata",
            hour: "2-digit",
            minute: "2-digit",
            hour12: false,
          })
        },
      },
      handleScroll: { mouseWheel: true, pressedMouseMove: true },
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
    })

    chartRef.current = chart

    // Create main series based on chart type
    let mainSeries: ISeriesApi<"Candlestick"> | ISeriesApi<"Line"> | ISeriesApi<"Area"> | ISeriesApi<"Bar"> | ISeriesApi<"Baseline">

    switch (chartType) {
      case "candlestick":
        mainSeries = chart.addSeries(CandlestickSeries, {
          upColor: "#26a69a",
          downColor: "#ef5350",
          borderUpColor: "#26a69a",
          borderDownColor: "#ef5350",
          wickUpColor: "#26a69a",
          wickDownColor: "#ef5350",
          borderVisible: false,
        })
        mainSeries.setData(chartData.candles)
        break

      case "hollowCandles":
        mainSeries = chart.addSeries(CandlestickSeries, {
          upColor: "transparent",
          downColor: "#ef5350",
          borderUpColor: "#26a69a",
          borderDownColor: "#ef5350",
          wickUpColor: "#26a69a",
          wickDownColor: "#ef5350",
          borderVisible: true,
        })
        mainSeries.setData(chartData.candles)
        break

      case "bars":
        mainSeries = chart.addSeries(BarSeries, {
          upColor: "#26a69a",
          downColor: "#ef5350",
          openVisible: true,
          thinBars: false,
        })
        mainSeries.setData(chartData.candles)
        break

      case "heikinAshi":
        mainSeries = chart.addSeries(CandlestickSeries, {
          upColor: "#26a69a",
          downColor: "#ef5350",
          borderUpColor: "#26a69a",
          borderDownColor: "#ef5350",
          wickUpColor: "#26a69a",
          wickDownColor: "#ef5350",
          borderVisible: false,
        })
        mainSeries.setData(chartData.heikinAshi)
        break

      case "line":
        mainSeries = chart.addSeries(LineSeries, {
          color: "#2962FF",
          lineWidth: 2,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 4,
        })
        mainSeries.setData(chartData.lineData)
        break

      case "stepLine":
        mainSeries = chart.addSeries(LineSeries, {
          color: "#2962FF",
          lineWidth: 2,
          lineType: LineType.WithSteps,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 4,
        })
        mainSeries.setData(chartData.lineData)
        break

      case "area":
        mainSeries = chart.addSeries(AreaSeries, {
          topColor: "rgba(41, 98, 255, 0.4)",
          bottomColor: "rgba(41, 98, 255, 0.0)",
          lineColor: "#2962FF",
          lineWidth: 2,
          crosshairMarkerVisible: true,
        })
        mainSeries.setData(chartData.lineData)
        break

      case "baseline":
        mainSeries = chart.addSeries(BaselineSeries, {
          baseValue: { type: "price", price: chartData.baselineValue },
          topLineColor: "#26a69a",
          topFillColor1: "rgba(38, 166, 154, 0.2)",
          topFillColor2: "rgba(38, 166, 154, 0.0)",
          bottomLineColor: "#ef5350",
          bottomFillColor1: "rgba(239, 83, 80, 0.0)",
          bottomFillColor2: "rgba(239, 83, 80, 0.2)",
          lineWidth: 2,
        })
        mainSeries.setData(chartData.lineData)
        break

      default:
        mainSeries = chart.addSeries(CandlestickSeries, {
          upColor: "#26a69a",
          downColor: "#ef5350",
          borderUpColor: "#26a69a",
          borderDownColor: "#ef5350",
          wickUpColor: "#26a69a",
          wickDownColor: "#ef5350",
          borderVisible: false,
        })
        mainSeries.setData(chartData.candles)
    }

    // Add indicator overlays
    if (indicatorData) {
      // EMA 12
      if (indicators.ema12) {
        const ema12Series = chart.addSeries(LineSeries, {
          color: "#f7931a",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const ema12Data = data.map((d, i) => ({
          time: d.time as Time,
          value: indicatorData.ema12[i],
        })).filter((d) => d.value !== null) as { time: Time; value: number }[]
        ema12Series.setData(ema12Data)
      }

      // EMA 26
      if (indicators.ema26) {
        const ema26Series = chart.addSeries(LineSeries, {
          color: "#627eea",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const ema26Data = data.map((d, i) => ({
          time: d.time as Time,
          value: indicatorData.ema26[i],
        })).filter((d) => d.value !== null) as { time: Time; value: number }[]
        ema26Series.setData(ema26Data)
      }

      // SMA 20
      if (indicators.sma20) {
        const sma20Series = chart.addSeries(LineSeries, {
          color: "#ff9800",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const sma20Data = data.map((d, i) => ({
          time: d.time as Time,
          value: indicatorData.sma20[i],
        })).filter((d) => d.value !== null) as { time: Time; value: number }[]
        sma20Series.setData(sma20Data)
      }

      // SMA 50
      if (indicators.sma50) {
        const sma50Series = chart.addSeries(LineSeries, {
          color: "#e91e63",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const sma50Data = data.map((d, i) => ({
          time: d.time as Time,
          value: indicatorData.sma50[i],
        })).filter((d) => d.value !== null) as { time: Time; value: number }[]
        sma50Series.setData(sma50Data)
      }

      // SMA 200
      if (indicators.sma200) {
        const sma200Series = chart.addSeries(LineSeries, {
          color: "#9c27b0",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const sma200Data = data.map((d, i) => ({
          time: d.time as Time,
          value: indicatorData.sma200[i],
        })).filter((d) => d.value !== null) as { time: Time; value: number }[]
        sma200Series.setData(sma200Data)
      }

      // Bollinger Bands
      if (indicators.bollingerBands) {
        const bb = indicatorData.bollingerBands

        // Upper band
        const upperSeries = chart.addSeries(LineSeries, {
          color: "rgba(33, 150, 243, 0.6)",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const upperData = data.map((d, i) => ({
          time: d.time as Time,
          value: bb.upper[i],
        })).filter((d) => d.value !== null) as { time: Time; value: number }[]
        upperSeries.setData(upperData)

        // Middle band (SMA)
        const middleSeries = chart.addSeries(LineSeries, {
          color: "#2196f3",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const middleData = data.map((d, i) => ({
          time: d.time as Time,
          value: bb.middle[i],
        })).filter((d) => d.value !== null) as { time: Time; value: number }[]
        middleSeries.setData(middleData)

        // Lower band
        const lowerSeries = chart.addSeries(LineSeries, {
          color: "rgba(33, 150, 243, 0.6)",
          lineWidth: 1,
          priceLineVisible: false,
          lastValueVisible: false,
        })
        const lowerData = data.map((d, i) => ({
          time: d.time as Time,
          value: bb.lower[i],
        })).filter((d) => d.value !== null) as { time: Time; value: number }[]
        lowerSeries.setData(lowerData)
      }
    }

    // Volume series
    if (showVolume && chartData.volumes.length > 0) {
      const volumeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: "volume" },
        priceScaleId: "volume",
      })
      volumeSeries.priceScale().applyOptions({
        scaleMargins: { top: 0.85, bottom: 0 },
      })
      volumeSeries.setData(chartData.volumes)
    }

    // Price line
    if (data.length > 0) {
      const lastPrice = data[data.length - 1].close
      mainSeries.createPriceLine({
        price: lastPrice,
        color: data[data.length - 1].close >= data[data.length - 1].open ? "#26a69a" : "#ef5350",
        lineWidth: 1,
        lineStyle: LineStyle.Dashed,
        axisLabelVisible: true,
        title: "",
      })
    }

    // Set initial legend
    if (data.length > 0) {
      const lastCandle = data[data.length - 1]
      const prevCandle = data.length > 1 ? data[data.length - 2] : lastCandle
      const change = prevCandle.close > 0 ? ((lastCandle.close - prevCandle.close) / prevCandle.close) * 100 : 0
      setLegendData({
        ohlc: { o: lastCandle.open, h: lastCandle.high, l: lastCandle.low, c: lastCandle.close },
        change,
        volume: lastCandle.volume || 0,
      })
    }

    // Crosshair move handler
    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.seriesData) {
        if (data.length > 0) {
          const lastCandle = data[data.length - 1]
          const prevCandle = data.length > 1 ? data[data.length - 2] : lastCandle
          const change = prevCandle.close > 0 ? ((lastCandle.close - prevCandle.close) / prevCandle.close) * 100 : 0
          setLegendData({
            ohlc: { o: lastCandle.open, h: lastCandle.high, l: lastCandle.low, c: lastCandle.close },
            change,
            volume: lastCandle.volume || 0,
          })
        }
        return
      }

      const seriesData = param.seriesData.get(mainSeries)
      if (seriesData) {
        const timeStr = param.time as string
        const dataPoint = data.find((d) => d.time === timeStr)
        const idx = data.findIndex((d) => d.time === timeStr)
        const prevPoint = idx > 0 ? data[idx - 1] : dataPoint

        if (dataPoint && prevPoint) {
          const change = prevPoint.close > 0 ? ((dataPoint.close - prevPoint.close) / prevPoint.close) * 100 : 0

          // Handle both OHLC and line data formats
          const ohlcData = 'open' in seriesData
            ? { o: (seriesData as CandlestickData<Time>).open, h: (seriesData as CandlestickData<Time>).high, l: (seriesData as CandlestickData<Time>).low, c: (seriesData as CandlestickData<Time>).close }
            : { o: dataPoint.open, h: dataPoint.high, l: dataPoint.low, c: dataPoint.close }

          setLegendData({
            ohlc: ohlcData,
            change,
            volume: dataPoint.volume || 0,
          })
        }
      }
    })

    chart.timeScale().fitContent()

    // Resize handler
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
          height: isFullscreen ? window.innerHeight : height,
        })
      }
    }

    window.addEventListener("resize", handleResize)

    return () => {
      window.removeEventListener("resize", handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [data, signals, height, showVolume, chartData, currency, indicators, indicatorData, isFullscreen, isDark])

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
      style={{ height: isFullscreen ? "100vh" : height, backgroundColor: isDark ? "#000000" : "#ffffff" }}
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
        onChartTypeChange={setChartType}
      />
      <ChartLegend
        ohlc={legendData.ohlc}
        change={legendData.change}
        volume={legendData.volume}
        currency={currency}
        indicators={indicators}
      />
      <IndicatorLegend indicators={indicators} />
    </div>
  )
}
