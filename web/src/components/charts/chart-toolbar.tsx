"use client"

import { Button } from "@/components/ui/button"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"
import {
    DropdownMenu,
    DropdownMenuContent,
    DropdownMenuCheckboxItem,
    DropdownMenuLabel,
    DropdownMenuSeparator,
    DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import {
    ZoomIn,
    ZoomOut,
    Maximize2,
    Minimize2,
    Download,
    RotateCcw,
    TrendingUp,
    Minus,
    BarChart3,
    CandlestickChart as CandlestickIcon,
    ChartLine,
} from "lucide-react"

// Chart types
export type ChartType = "line" | "candlestick"

export interface ChartIndicators {
    ema12: boolean
    ema26: boolean
    sma20: boolean
    sma50: boolean
    sma200: boolean
    bollingerBands: boolean
    rsi: boolean
    macd: boolean
    volumeSma: boolean
}

export interface ChartToolbarProps {
    onZoomIn: () => void
    onZoomOut: () => void
    onResetZoom: () => void
    onFullscreen: () => void
    onScreenshot: () => void
    isFullscreen: boolean
    indicators: ChartIndicators
    onIndicatorChange: (key: keyof ChartIndicators, value: boolean) => void
    onAddHorizontalLine?: () => void
    chartType?: ChartType
    onToggleChartType?: () => void
}

export function ChartToolbar({
    onZoomIn,
    onZoomOut,
    onResetZoom,
    onFullscreen,
    onScreenshot,
    isFullscreen,
    indicators,
    onIndicatorChange,
    onAddHorizontalLine,
    chartType,
    onToggleChartType,
}: ChartToolbarProps) {
    return (
        <TooltipProvider delayDuration={300}>
            <div className="absolute top-12 left-3 z-20 flex flex-col gap-1 bg-[#1e222d]/90 backdrop-blur-sm rounded-lg p-1 border border-[#2a2e39]">
                {/* Zoom Controls */}
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-[#787b86] hover:text-white hover:bg-[#2a2e39]"
                            onClick={onZoomIn}
                        >
                            <ZoomIn className="h-4 w-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="right">Zoom In</TooltipContent>
                </Tooltip>

                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-[#787b86] hover:text-white hover:bg-[#2a2e39]"
                            onClick={onZoomOut}
                        >
                            <ZoomOut className="h-4 w-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="right">Zoom Out</TooltipContent>
                </Tooltip>

                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-[#787b86] hover:text-white hover:bg-[#2a2e39]"
                            onClick={onResetZoom}
                        >
                            <RotateCcw className="h-4 w-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="right">Fit to View</TooltipContent>
                </Tooltip>

                <div className="w-full h-px bg-[#2a2e39] my-1" />

                {/* Indicators Dropdown */}
                <DropdownMenu>
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <DropdownMenuTrigger asChild>
                                <Button
                                    variant="ghost"
                                    size="icon"
                                    className="h-8 w-8 text-[#787b86] hover:text-white hover:bg-[#2a2e39]"
                                >
                                    <TrendingUp className="h-4 w-4" />
                                </Button>
                            </DropdownMenuTrigger>
                        </TooltipTrigger>
                        <TooltipContent side="right">Indicators</TooltipContent>
                    </Tooltip>
                    <DropdownMenuContent side="right" align="start" className="w-48">
                        <DropdownMenuLabel>Moving Averages</DropdownMenuLabel>
                        <DropdownMenuCheckboxItem
                            checked={indicators.ema12}
                            onCheckedChange={(checked) => onIndicatorChange("ema12", checked)}
                        >
                            <span className="flex items-center gap-2">
                                <span className="w-3 h-0.5 bg-[#f7931a] rounded" />
                                EMA 12
                            </span>
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                            checked={indicators.ema26}
                            onCheckedChange={(checked) => onIndicatorChange("ema26", checked)}
                        >
                            <span className="flex items-center gap-2">
                                <span className="w-3 h-0.5 bg-[#627eea] rounded" />
                                EMA 26
                            </span>
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                            checked={indicators.sma20}
                            onCheckedChange={(checked) => onIndicatorChange("sma20", checked)}
                        >
                            <span className="flex items-center gap-2">
                                <span className="w-3 h-0.5 bg-[#ff9800] rounded" />
                                SMA 20
                            </span>
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                            checked={indicators.sma50}
                            onCheckedChange={(checked) => onIndicatorChange("sma50", checked)}
                        >
                            <span className="flex items-center gap-2">
                                <span className="w-3 h-0.5 bg-[#e91e63] rounded" />
                                SMA 50
                            </span>
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                            checked={indicators.sma200}
                            onCheckedChange={(checked) => onIndicatorChange("sma200", checked)}
                        >
                            <span className="flex items-center gap-2">
                                <span className="w-3 h-0.5 bg-[#9c27b0] rounded" />
                                SMA 200
                            </span>
                        </DropdownMenuCheckboxItem>

                        <DropdownMenuSeparator />
                        <DropdownMenuLabel>Overlays</DropdownMenuLabel>
                        <DropdownMenuCheckboxItem
                            checked={indicators.bollingerBands}
                            onCheckedChange={(checked) => onIndicatorChange("bollingerBands", checked)}
                        >
                            <span className="flex items-center gap-2">
                                <span className="w-3 h-0.5 bg-[#2196f3] rounded" />
                                Bollinger Bands
                            </span>
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                            checked={indicators.volumeSma}
                            onCheckedChange={(checked) => onIndicatorChange("volumeSma", checked)}
                        >
                            <span className="flex items-center gap-2">
                                <BarChart3 className="h-3 w-3 text-[#607d8b]" />
                                Volume SMA
                            </span>
                        </DropdownMenuCheckboxItem>

                        <DropdownMenuSeparator />
                        <DropdownMenuLabel>Panels</DropdownMenuLabel>
                        <DropdownMenuCheckboxItem
                            checked={indicators.rsi}
                            onCheckedChange={(checked) => onIndicatorChange("rsi", checked)}
                        >
                            RSI (14)
                        </DropdownMenuCheckboxItem>
                        <DropdownMenuCheckboxItem
                            checked={indicators.macd}
                            onCheckedChange={(checked) => onIndicatorChange("macd", checked)}
                        >
                            MACD (12, 26, 9)
                        </DropdownMenuCheckboxItem>
                    </DropdownMenuContent>
                </DropdownMenu>

                {/* Drawing Tools */}
                {onAddHorizontalLine && (
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="ghost"
                                size="icon"
                                className="h-8 w-8 text-[#787b86] hover:text-white hover:bg-[#2a2e39]"
                                onClick={onAddHorizontalLine}
                            >
                                <Minus className="h-4 w-4" />
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent side="right">Add Horizontal Line</TooltipContent>
                    </Tooltip>
                )}

                <div className="w-full h-px bg-[#2a2e39] my-1" />

                {/* Fullscreen & Screenshot */}
                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-[#787b86] hover:text-white hover:bg-[#2a2e39]"
                            onClick={onFullscreen}
                        >
                            {isFullscreen ? (
                                <Minimize2 className="h-4 w-4" />
                            ) : (
                                <Maximize2 className="h-4 w-4" />
                            )}
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="right">
                        {isFullscreen ? "Exit Fullscreen" : "Fullscreen"}
                    </TooltipContent>
                </Tooltip>

                <Tooltip>
                    <TooltipTrigger asChild>
                        <Button
                            variant="ghost"
                            size="icon"
                            className="h-8 w-8 text-[#787b86] hover:text-white hover:bg-[#2a2e39]"
                            onClick={onScreenshot}
                        >
                            <Download className="h-4 w-4" />
                        </Button>
                    </TooltipTrigger>
                    <TooltipContent side="right">Download Chart</TooltipContent>
                </Tooltip>

                {chartType && onToggleChartType && (
                    <Tooltip>
                        <TooltipTrigger asChild>
                            <Button
                                variant="ghost"
                                size="icon"
                                aria-label={chartType === "line" ? "Switch to candlestick chart" : "Switch to line chart"}
                                title={chartType === "line" ? "Switch to candlestick" : "Switch to line"}
                                className="h-8 w-8 text-[#787b86] hover:text-white hover:bg-[#2a2e39]"
                                onClick={onToggleChartType}
                            >
                                {chartType === "line" ? (
                                    <ChartLine className="h-4 w-4" />
                                ) : (
                                    <CandlestickIcon className="h-4 w-4" />
                                )}
                            </Button>
                        </TooltipTrigger>
                        <TooltipContent side="right">Toggle Chart Type</TooltipContent>
                    </Tooltip>
                )}
            </div>
        </TooltipProvider>
    )
}

// Default indicator settings
export const defaultIndicators: ChartIndicators = {
    ema12: false,
    ema26: false,
    sma20: false,
    sma50: false,
    sma200: false,
    bollingerBands: false,
    rsi: false,
    macd: false,
    volumeSma: false,
}
