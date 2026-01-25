"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import {
    PieChart,
    Pie,
    Cell,
    BarChart,
    Bar,
    LineChart,
    Line,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    Legend,
    ResponsiveContainer,
    ComposedChart,
    Area,
} from "recharts"
import {
    TrendingUp,
    TrendingDown,
    DollarSign,
    Shield,
    PieChart as PieChartIcon,
    BarChart3,
    Brain,
    Zap,
    Target,
    AlertTriangle,
    CheckCircle,
    Info,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface StockFundamentals {
    symbol: string
    name?: string | null
    market_cap?: number | null
    enterprise_value?: number | null
    revenue_ttm?: number | null
    net_income?: number | null
    gross_profit?: number | null
    ebitda?: number | null
    total_cash?: number | null
    total_debt?: number | null
    free_cash_flow?: number | null
    dividend_yield?: number | null
    dividend_rate?: number | null
    payout_ratio?: number | null
    profit_margin?: number | null
    operating_margin?: number | null
    gross_margin?: number | null
    current_ratio?: number | null
    quick_ratio?: number | null
    debt_to_equity?: number | null
    roe?: number | null
    roa?: number | null
    pe_ratio?: number | null
    ps_ratio?: number | null
    pb_ratio?: number | null
    insider_ownership?: number | null
    institutional_ownership?: number | null
    beta?: number | null
    revenue_growth?: number | null
    earnings_growth?: number | null
}

interface AIAlgoPrediction {
    signal: "strong_buy" | "buy" | "hold" | "sell" | "strong_sell"
    confidence: number
    predicted_return_30d: number
    predicted_return_90d: number
    risk_score: number // 1-10
    momentum_score: number // 1-100
    value_score: number // 1-100
    quality_score: number // 1-100
    reasoning: string[]
    key_factors: Array<{ name: string; impact: "positive" | "negative" | "neutral"; weight: number }>
}

interface AdvancedChartsProps {
    fundamentals: StockFundamentals
    aiPrediction?: AIAlgoPrediction | null
    currency?: string
}

// Colors for charts
const COLORS = {
    primary: "#3b82f6",
    secondary: "#8b5cf6",
    success: "#22c55e",
    warning: "#f59e0b",
    danger: "#ef4444",
    muted: "#6b7280",
    cyan: "#06b6d4",
    pink: "#ec4899",
}

// Format functions
function formatLargeNumber(value: number | null | undefined): string {
    if (value === null || value === undefined) return "—"
    const absValue = Math.abs(value)
    if (absValue >= 1e12) return `${(value / 1e12).toFixed(2)}T`
    if (absValue >= 1e9) return `${(value / 1e9).toFixed(2)}B`
    if (absValue >= 1e6) return `${(value / 1e6).toFixed(2)}M`
    if (absValue >= 1e3) return `${(value / 1e3).toFixed(2)}K`
    return value.toFixed(2)
}

function formatPercent(value: number | null | undefined): string {
    if (value === null || value === undefined) return "—"
    return `${(value * 100).toFixed(1)}%`
}

// Info tooltip
function InfoTooltip({ text }: { text: string }) {
    return (
        <Tooltip>
            <TooltipTrigger asChild>
                <Info className="h-3 w-3 text-muted-foreground/50 hover:text-muted-foreground cursor-help ml-1" />
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-xs text-xs">
                {text}
            </TooltipContent>
        </Tooltip>
    )
}

// Payout Ratio Donut Chart
function PayoutRatioChart({ payoutRatio }: { payoutRatio: number | null | undefined }) {
    const ratio = payoutRatio ? Math.min(payoutRatio * 100, 100) : 0
    const data = [
        { name: "Payout", value: ratio },
        { name: "Retained", value: 100 - ratio },
    ]

    return (
        <div className="flex items-center gap-4">
            <div className="relative w-24 h-24">
                <ResponsiveContainer width="100%" height="100%">
                    <PieChart>
                        <Pie
                            data={data}
                            cx="50%"
                            cy="50%"
                            innerRadius={30}
                            outerRadius={40}
                            paddingAngle={2}
                            dataKey="value"
                        >
                            <Cell fill={COLORS.danger} />
                            <Cell fill={COLORS.muted} opacity={0.3} />
                        </Pie>
                    </PieChart>
                </ResponsiveContainer>
                <div className="absolute inset-0 flex items-center justify-center">
                    <span className="text-sm font-bold text-red-400">{ratio.toFixed(0)}%</span>
                </div>
            </div>
            <div className="text-xs text-muted-foreground">
                <div className="flex items-center gap-1 mb-1">
                    <span className="w-2 h-2 rounded-full bg-red-400" />
                    Payout ratio (TTM)
                </div>
            </div>
        </div>
    )
}

// Revenue Breakdown Pie Chart
function RevenueBreakdownChart({ fundamentals, currency }: { fundamentals: StockFundamentals; currency: string }) {
    const data = [
        { name: "Gross Profit", value: fundamentals.gross_profit || 0, color: COLORS.success },
        { name: "COGS", value: (fundamentals.revenue_ttm || 0) - (fundamentals.gross_profit || 0), color: COLORS.warning },
    ].filter(d => d.value > 0)

    if (data.length === 0) return null

    return (
        <div className="space-y-3">
            <h4 className="text-sm font-semibold flex items-center gap-2">
                <PieChartIcon className="h-4 w-4" />
                Revenue to Profit
                <InfoTooltip text="Breakdown of revenue into gross profit and cost of goods sold" />
            </h4>
            <div className="flex items-center gap-6">
                <div className="w-32 h-32">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="50%"
                                innerRadius={35}
                                outerRadius={50}
                                paddingAngle={2}
                                dataKey="value"
                            >
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Pie>
                            <RechartsTooltip
                                formatter={(value) => value !== undefined ? `${currency}${formatLargeNumber(value as number)}` : ''}
                                contentStyle={{ background: "#1f2937", border: "none", borderRadius: "8px" }}
                            />
                        </PieChart>
                    </ResponsiveContainer>
                </div>
                <div className="space-y-2 text-xs">
                    {data.map((item, i) => (
                        <div key={i} className="flex items-center gap-2">
                            <span className="w-3 h-3 rounded" style={{ backgroundColor: item.color }} />
                            <span className="text-muted-foreground">{item.name}</span>
                            <span className="font-medium">{currency}{formatLargeNumber(item.value)}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

// Financial Health Bar Chart
function FinancialHealthChart({ fundamentals, currency }: { fundamentals: StockFundamentals; currency: string }) {
    const data = [
        { name: "Cash", value: fundamentals.total_cash || 0, fill: COLORS.success },
        { name: "Debt", value: fundamentals.total_debt || 0, fill: COLORS.danger },
        { name: "FCF", value: Math.max(0, fundamentals.free_cash_flow || 0), fill: COLORS.cyan },
    ]

    return (
        <div className="space-y-3">
            <h4 className="text-sm font-semibold flex items-center gap-2">
                <Shield className="h-4 w-4" />
                Financial Position
                <InfoTooltip text="Cash, debt, and free cash flow comparison" />
            </h4>
            <div className="h-40">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data} layout="vertical">
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis type="number" tick={{ fill: "#9ca3af", fontSize: 10 }} tickFormatter={(v) => formatLargeNumber(v)} />
                        <YAxis type="category" dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} width={50} />
                        <RechartsTooltip
                            formatter={(value) => value !== undefined ? `${currency}${formatLargeNumber(value as number)}` : ''}
                            contentStyle={{ background: "#1f2937", border: "none", borderRadius: "8px" }}
                        />
                        <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

// Ownership Donut Chart
function OwnershipChart({ fundamentals }: { fundamentals: StockFundamentals }) {
    const insider = (fundamentals.insider_ownership || 0) * 100
    const institutional = (fundamentals.institutional_ownership || 0) * 100
    const publicFloat = Math.max(0, 100 - insider - institutional)

    const data = [
        { name: "Insiders", value: insider, color: COLORS.primary },
        { name: "Institutions", value: institutional, color: COLORS.cyan },
        { name: "Public", value: publicFloat, color: COLORS.muted },
    ].filter(d => d.value > 0)

    return (
        <div className="space-y-3">
            <h4 className="text-sm font-semibold flex items-center gap-2">
                <PieChartIcon className="h-4 w-4" />
                Ownership Structure
                <InfoTooltip text="Distribution of shares among insiders, institutions, and public" />
            </h4>
            <div className="flex items-center gap-6">
                <div className="relative w-32 h-32">
                    <ResponsiveContainer width="100%" height="100%">
                        <PieChart>
                            <Pie
                                data={data}
                                cx="50%"
                                cy="50%"
                                innerRadius={35}
                                outerRadius={50}
                                paddingAngle={2}
                                dataKey="value"
                            >
                                {data.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill={entry.color} />
                                ))}
                            </Pie>
                        </PieChart>
                    </ResponsiveContainer>
                    <div className="absolute inset-0 flex items-center justify-center flex-col">
                        <span className="text-xs text-muted-foreground">Total</span>
                        <span className="text-sm font-bold">100%</span>
                    </div>
                </div>
                <div className="space-y-2 text-xs">
                    {data.map((item, i) => (
                        <div key={i} className="flex items-center gap-2">
                            <span className="w-3 h-3 rounded" style={{ backgroundColor: item.color }} />
                            <span className="text-muted-foreground">{item.name}</span>
                            <span className="font-medium">{item.value.toFixed(1)}%</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

// Profitability Margins Chart
function MarginsChart({ fundamentals }: { fundamentals: StockFundamentals }) {
    const data = [
        { name: "Gross", value: (fundamentals.gross_margin || 0) * 100, fill: COLORS.success },
        { name: "Operating", value: (fundamentals.operating_margin || 0) * 100, fill: COLORS.cyan },
        { name: "Net", value: (fundamentals.profit_margin || 0) * 100, fill: COLORS.primary },
    ]

    return (
        <div className="space-y-3">
            <h4 className="text-sm font-semibold flex items-center gap-2">
                <BarChart3 className="h-4 w-4" />
                Profit Margins
                <InfoTooltip text="Gross, operating, and net profit margins showing profitability at different levels" />
            </h4>
            <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                        <YAxis tick={{ fill: "#9ca3af", fontSize: 10 }} tickFormatter={(v) => `${v}%`} />
                        <RechartsTooltip
                            formatter={(value) => value !== undefined ? `${(value as number).toFixed(1)}%` : ''}
                            contentStyle={{ background: "#1f2937", border: "none", borderRadius: "8px" }}
                        />
                        <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                            {data.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={entry.fill} />
                            ))}
                        </Bar>
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

// Valuation Comparison Chart
function ValuationChart({ fundamentals }: { fundamentals: StockFundamentals }) {
    const data = [
        { name: "P/E", value: fundamentals.pe_ratio || 0 },
        { name: "P/S", value: fundamentals.ps_ratio || 0 },
        { name: "P/B", value: fundamentals.pb_ratio || 0 },
    ]

    return (
        <div className="space-y-3">
            <h4 className="text-sm font-semibold flex items-center gap-2">
                <DollarSign className="h-4 w-4" />
                Valuation Ratios
                <InfoTooltip text="Key valuation multiples - P/E, P/S, and P/B ratios" />
            </h4>
            <div className="h-32">
                <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={data}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis dataKey="name" tick={{ fill: "#9ca3af", fontSize: 11 }} />
                        <YAxis tick={{ fill: "#9ca3af", fontSize: 10 }} />
                        <RechartsTooltip
                            formatter={(value) => value !== undefined ? `${(value as number).toFixed(2)}x` : ''}
                            contentStyle={{ background: "#1f2937", border: "none", borderRadius: "8px" }}
                        />
                        <Bar dataKey="value" fill={COLORS.secondary} radius={[4, 4, 0, 0]} />
                    </BarChart>
                </ResponsiveContainer>
            </div>
        </div>
    )
}

// AI Algo Trading Score Card
function AIScoreCard({
    title,
    score,
    maxScore = 100,
    color,
    description
}: {
    title: string
    score: number
    maxScore?: number
    color: string
    description: string
}) {
    const percentage = (score / maxScore) * 100

    return (
        <div className="p-3 rounded-lg bg-muted/50 space-y-2">
            <div className="flex items-center justify-between">
                <span className="text-sm font-medium">{title}</span>
                <span className="text-lg font-bold" style={{ color }}>{score}</span>
            </div>
            <div className="h-2 bg-muted rounded-full overflow-hidden">
                <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${percentage}%`, backgroundColor: color }}
                />
            </div>
            <p className="text-xs text-muted-foreground">{description}</p>
        </div>
    )
}

// AI Algo Prediction Panel
function AIAlgoPredictionPanel({ prediction, symbol }: { prediction: AIAlgoPrediction; symbol: string }) {
    const signalColors: Record<string, { bg: string; text: string; border: string }> = {
        strong_buy: { bg: "bg-emerald-500/20", text: "text-emerald-400", border: "border-emerald-500/30" },
        buy: { bg: "bg-green-500/20", text: "text-green-400", border: "border-green-500/30" },
        hold: { bg: "bg-yellow-500/20", text: "text-yellow-400", border: "border-yellow-500/30" },
        sell: { bg: "bg-orange-500/20", text: "text-orange-400", border: "border-orange-500/30" },
        strong_sell: { bg: "bg-red-500/20", text: "text-red-400", border: "border-red-500/30" },
    }

    const signalConfig = signalColors[prediction.signal] || signalColors.hold
    const signalLabel = prediction.signal.replace("_", " ").toUpperCase()

    return (
        <div className="space-y-6">
            {/* AI Signal */}
            <div className={cn("p-4 rounded-lg border-2", signalConfig.border, signalConfig.bg)}>
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <div className={cn("p-2 rounded-lg", signalConfig.bg)}>
                            <Brain className={cn("h-6 w-6", signalConfig.text)} />
                        </div>
                        <div>
                            <h3 className="font-bold text-lg">AI Algo Trading Signal</h3>
                            <p className="text-xs text-muted-foreground">Multi-factor quantitative analysis</p>
                        </div>
                    </div>
                    <div className="text-right">
                        <div className={cn("text-2xl font-bold", signalConfig.text)}>{signalLabel}</div>
                        <div className="text-sm text-muted-foreground">{(prediction.confidence * 100).toFixed(0)}% confidence</div>
                    </div>
                </div>

                {/* Predicted Returns */}
                <div className="grid grid-cols-2 gap-4 mb-4">
                    <div className="p-3 rounded-lg bg-background/50">
                        <div className="text-xs text-muted-foreground mb-1">Predicted Return (30D)</div>
                        <div className={cn("text-xl font-bold", prediction.predicted_return_30d >= 0 ? "text-green-400" : "text-red-400")}>
                            {prediction.predicted_return_30d >= 0 ? "+" : ""}{prediction.predicted_return_30d.toFixed(1)}%
                        </div>
                    </div>
                    <div className="p-3 rounded-lg bg-background/50">
                        <div className="text-xs text-muted-foreground mb-1">Predicted Return (90D)</div>
                        <div className={cn("text-xl font-bold", prediction.predicted_return_90d >= 0 ? "text-green-400" : "text-red-400")}>
                            {prediction.predicted_return_90d >= 0 ? "+" : ""}{prediction.predicted_return_90d.toFixed(1)}%
                        </div>
                    </div>
                </div>

                {/* Risk Score */}
                <div className="flex items-center gap-2 text-sm">
                    <Shield className="h-4 w-4" />
                    <span className="text-muted-foreground">Risk Level:</span>
                    <div className="flex gap-0.5">
                        {Array.from({ length: 10 }).map((_, i) => (
                            <div
                                key={i}
                                className={cn(
                                    "w-2 h-4 rounded-sm",
                                    i < prediction.risk_score
                                        ? prediction.risk_score <= 3 ? "bg-green-500" : prediction.risk_score <= 6 ? "bg-yellow-500" : "bg-red-500"
                                        : "bg-muted"
                                )}
                            />
                        ))}
                    </div>
                    <span className="font-medium">{prediction.risk_score}/10</span>
                </div>
            </div>

            {/* Factor Scores */}
            <div className="grid grid-cols-3 gap-4">
                <AIScoreCard
                    title="Momentum"
                    score={prediction.momentum_score}
                    color={prediction.momentum_score >= 70 ? COLORS.success : prediction.momentum_score >= 40 ? COLORS.warning : COLORS.danger}
                    description="Price trend strength"
                />
                <AIScoreCard
                    title="Value"
                    score={prediction.value_score}
                    color={prediction.value_score >= 70 ? COLORS.success : prediction.value_score >= 40 ? COLORS.warning : COLORS.danger}
                    description="Valuation attractiveness"
                />
                <AIScoreCard
                    title="Quality"
                    score={prediction.quality_score}
                    color={prediction.quality_score >= 70 ? COLORS.success : prediction.quality_score >= 40 ? COLORS.warning : COLORS.danger}
                    description="Financial strength"
                />
            </div>

            {/* Key Factors */}
            <div className="space-y-3">
                <h4 className="text-sm font-semibold flex items-center gap-2">
                    <Zap className="h-4 w-4" />
                    Key Factors
                </h4>
                <div className="space-y-2">
                    {prediction.key_factors.map((factor, i) => (
                        <div key={i} className="flex items-center justify-between p-2 rounded-lg bg-muted/30">
                            <div className="flex items-center gap-2">
                                {factor.impact === "positive" ? (
                                    <CheckCircle className="h-4 w-4 text-green-400" />
                                ) : factor.impact === "negative" ? (
                                    <AlertTriangle className="h-4 w-4 text-red-400" />
                                ) : (
                                    <Info className="h-4 w-4 text-yellow-400" />
                                )}
                                <span className="text-sm">{factor.name}</span>
                            </div>
                            <Badge variant="outline" className={cn(
                                factor.impact === "positive" ? "text-green-400 border-green-400/30" :
                                    factor.impact === "negative" ? "text-red-400 border-red-400/30" :
                                        "text-yellow-400 border-yellow-400/30"
                            )}>
                                {(factor.weight * 100).toFixed(0)}% weight
                            </Badge>
                        </div>
                    ))}
                </div>
            </div>

            {/* AI Reasoning */}
            <div className="space-y-3">
                <h4 className="text-sm font-semibold flex items-center gap-2">
                    <Brain className="h-4 w-4" />
                    AI Reasoning
                </h4>
                <div className="space-y-2">
                    {prediction.reasoning.map((reason, i) => (
                        <div key={i} className="flex items-start gap-2 text-sm text-muted-foreground">
                            <span className="text-primary mt-1">•</span>
                            <span>{reason}</span>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    )
}

export function AdvancedChartsPanel({ fundamentals, aiPrediction, currency = "$" }: AdvancedChartsProps) {
    return (
        <TooltipProvider delayDuration={200}>
            <Card>
                <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                        <BarChart3 className="h-5 w-5" />
                        Advanced Analytics
                    </CardTitle>
                    <CardDescription>Visual charts, metrics, and AI-powered predictions</CardDescription>
                </CardHeader>

                <CardContent>
                    <Tabs defaultValue="charts" className="w-full">
                        <TabsList className="grid w-full grid-cols-3 mb-4">
                            <TabsTrigger value="charts">Charts</TabsTrigger>
                            <TabsTrigger value="ratios">Ratios & Margins</TabsTrigger>
                            <TabsTrigger value="ai" className="flex items-center gap-1">
                                <Brain className="h-3 w-3" />
                                AI Algo
                            </TabsTrigger>
                        </TabsList>

                        {/* Charts Tab */}
                        <TabsContent value="charts" className="space-y-6">
                            <div className="grid md:grid-cols-2 gap-6">
                                <RevenueBreakdownChart fundamentals={fundamentals} currency={currency} />
                                <OwnershipChart fundamentals={fundamentals} />
                            </div>
                            <div className="grid md:grid-cols-2 gap-6">
                                <FinancialHealthChart fundamentals={fundamentals} currency={currency} />
                                {fundamentals.payout_ratio !== undefined && fundamentals.payout_ratio !== null && (
                                    <div className="space-y-3">
                                        <h4 className="text-sm font-semibold flex items-center gap-2">
                                            <DollarSign className="h-4 w-4" />
                                            Dividend Summary
                                            <InfoTooltip text="Payout ratio shows what portion of earnings is paid as dividends" />
                                        </h4>
                                        <PayoutRatioChart payoutRatio={fundamentals.payout_ratio} />
                                        <div className="grid grid-cols-2 gap-2 text-sm">
                                            <div>
                                                <span className="text-muted-foreground">Dividend Yield:</span>
                                                <span className="ml-2 font-medium">{formatPercent(fundamentals.dividend_yield)}</span>
                                            </div>
                                            <div>
                                                <span className="text-muted-foreground">Annual Rate:</span>
                                                <span className="ml-2 font-medium">{currency}{fundamentals.dividend_rate?.toFixed(2) || "—"}</span>
                                            </div>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </TabsContent>

                        {/* Ratios Tab */}
                        <TabsContent value="ratios" className="space-y-6">
                            <div className="grid md:grid-cols-2 gap-6">
                                <ValuationChart fundamentals={fundamentals} />
                                <MarginsChart fundamentals={fundamentals} />
                            </div>
                        </TabsContent>

                        {/* AI Algo Tab */}
                        <TabsContent value="ai">
                            {aiPrediction ? (
                                <AIAlgoPredictionPanel prediction={aiPrediction} symbol={fundamentals.symbol} />
                            ) : (
                                <div className="text-center py-12">
                                    <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                                    <h3 className="text-lg font-semibold mb-2">AI Algo Analysis</h3>
                                    <p className="text-muted-foreground text-sm mb-4">
                                        Run an analysis to get AI-powered algo trading predictions with<br />
                                        momentum, value, and quality scores.
                                    </p>
                                </div>
                            )}
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>
        </TooltipProvider>
    )
}
