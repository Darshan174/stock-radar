"use client"

import { useEffect, useRef } from "react"
import { createChart, ColorType, IChartApi, Time, AreaSeries } from "lightweight-charts"

interface SparklineProps {
  data: { time: string; value: number }[]
  width?: number
  height?: number
  color?: string
}

export function Sparkline({
  data,
  width = 100,
  height = 40,
  color,
}: SparklineProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)

  // Determine color based on price change
  const priceChange = data.length >= 2
    ? data[data.length - 1].value - data[0].value
    : 0
  const lineColor = color || (priceChange >= 0 ? "#22c55e" : "#ef4444")

  useEffect(() => {
    if (!chartContainerRef.current || data.length === 0) return

    if (chartRef.current) {
      chartRef.current.remove()
    }

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: "transparent" },
        textColor: "transparent",
      },
      grid: {
        vertLines: { visible: false },
        horzLines: { visible: false },
      },
      width,
      height,
      rightPriceScale: {
        visible: false,
      },
      timeScale: {
        visible: false,
      },
      crosshair: {
        mode: 0,
      },
      handleScroll: false,
      handleScale: false,
    })

    chartRef.current = chart

    // Use AreaSeries for sparkline (v5 API)
    const areaSeries = chart.addSeries(AreaSeries, {
      lineColor: lineColor,
      topColor: `${lineColor}40`,
      bottomColor: `${lineColor}05`,
      lineWidth: 2,
      priceLineVisible: false,
      lastValueVisible: false,
      crosshairMarkerVisible: false,
    })

    const formattedData = data.map((d) => ({
      time: d.time as Time,
      value: d.value,
    }))

    areaSeries.setData(formattedData)
    chart.timeScale().fitContent()

    return () => {
      if (chartRef.current) {
        chartRef.current.remove()
        chartRef.current = null
      }
    }
  }, [data, width, height, lineColor])

  if (data.length === 0) {
    return <div style={{ width, height }} className="bg-muted/20 rounded" />
  }

  return <div ref={chartContainerRef} />
}
