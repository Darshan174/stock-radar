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
    Building2,
    Globe,
    Users,
    Calendar,
    TrendingUp,
    TrendingDown,
    DollarSign,
    PieChart,
    BarChart3,
    Info,
    ExternalLink,
    Briefcase,
    MapPin,
    Target,
    Shield,
    Activity,
    Percent,
    Layers,
} from "lucide-react"
import { cn } from "@/lib/utils"

interface StockFundamentals {
    symbol: string
    name?: string | null
    short_name?: string | null
    sector?: string | null
    industry?: string | null
    market_cap?: number | null
    enterprise_value?: number | null

    // Company Info
    website?: string | null
    headquarters_city?: string | null
    headquarters_country?: string | null
    employees?: number | null
    description?: string | null

    // Valuation
    pe_ratio?: number | null
    forward_pe?: number | null
    peg_ratio?: number | null
    pb_ratio?: number | null
    ps_ratio?: number | null

    // EPS & Revenue
    eps_ttm?: number | null
    eps_forward?: number | null
    revenue_ttm?: number | null
    gross_profit?: number | null
    ebitda?: number | null
    net_income?: number | null

    // Shares
    shares_outstanding?: number | null
    float_shares?: number | null
    insider_ownership?: number | null
    institutional_ownership?: number | null

    // Risk
    beta?: number | null
    "52_week_high"?: number | null
    "52_week_low"?: number | null

    // Profitability
    profit_margin?: number | null
    operating_margin?: number | null
    gross_margin?: number | null
    roe?: number | null
    roa?: number | null

    // Growth
    revenue_growth?: number | null
    earnings_growth?: number | null

    // Dividends
    dividend_yield?: number | null
    dividend_rate?: number | null
    payout_ratio?: number | null

    // Financial
    current_ratio?: number | null
    quick_ratio?: number | null
    debt_to_equity?: number | null
    total_cash?: number | null
    total_debt?: number | null
    free_cash_flow?: number | null

    // Analyst
    target_high?: number | null
    target_low?: number | null
    target_mean?: number | null
    recommendation?: string | null
    analyst_count?: number | null

    // Earnings
    next_earnings_date?: string | null
}

interface StockInfoPanelProps {
    fundamentals: StockFundamentals
    currentPrice?: number
    currency?: string
}

// Format functions
function formatNumber(value: number | null | undefined, decimals = 2): string {
    if (value === null || value === undefined) return "—"
    return value.toLocaleString(undefined, { minimumFractionDigits: decimals, maximumFractionDigits: decimals })
}

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
    return `${(value * 100).toFixed(2)}%`
}

function formatDate(dateStr: string | null | undefined): string {
    if (!dateStr) return "—"
    const date = new Date(dateStr)
    return date.toLocaleDateString("en-US", { month: "short", day: "numeric", year: "numeric" })
}

// Info tooltip component
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

// Stat card component
function StatItem({
    label,
    value,
    tooltip,
    valueColor,
    icon,
}: {
    label: string
    value: string | number
    tooltip?: string
    valueColor?: string
    icon?: React.ReactNode
}) {
    return (
        <div className="space-y-1">
            <div className="flex items-center text-xs text-muted-foreground">
                {icon && <span className="mr-1.5">{icon}</span>}
                {label}
                {tooltip && <InfoTooltip text={tooltip} />}
            </div>
            <div className={cn("text-sm font-semibold", valueColor)}>
                {value || "—"}
            </div>
        </div>
    )
}

// Recommendation badge
function RecommendationBadge({ recommendation }: { recommendation: string | null | undefined }) {
    if (!recommendation) return <span className="text-muted-foreground">—</span>

    const config: Record<string, { color: string; label: string }> = {
        strongBuy: { color: "bg-emerald-500/20 text-emerald-400 border-emerald-500/30", label: "Strong Buy" },
        buy: { color: "bg-green-500/20 text-green-400 border-green-500/30", label: "Buy" },
        hold: { color: "bg-yellow-500/20 text-yellow-400 border-yellow-500/30", label: "Hold" },
        sell: { color: "bg-orange-500/20 text-orange-400 border-orange-500/30", label: "Sell" },
        strongSell: { color: "bg-red-500/20 text-red-400 border-red-500/30", label: "Strong Sell" },
    }

    const style = config[recommendation] || config.hold

    return (
        <Badge variant="outline" className={style.color}>
            {style.label}
        </Badge>
    )
}

// Progress bar for ownership
function OwnershipBar({ insider, institutional }: { insider?: number | null; institutional?: number | null }) {
    const insiderPct = (insider || 0) * 100
    const instPct = (institutional || 0) * 100
    const publicPct = Math.max(0, 100 - insiderPct - instPct)

    return (
        <div className="space-y-2">
            <div className="flex h-3 w-full overflow-hidden rounded-full bg-muted">
                <div
                    className="bg-blue-500"
                    style={{ width: `${insiderPct}%` }}
                    title={`Insiders: ${insiderPct.toFixed(1)}%`}
                />
                <div
                    className="bg-cyan-500"
                    style={{ width: `${instPct}%` }}
                    title={`Institutions: ${instPct.toFixed(1)}%`}
                />
                <div
                    className="bg-muted-foreground/20"
                    style={{ width: `${publicPct}%` }}
                    title={`Public: ${publicPct.toFixed(1)}%`}
                />
            </div>
            <div className="flex justify-between text-xs text-muted-foreground">
                <span className="flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-blue-500" />
                    Insiders {insiderPct.toFixed(1)}%
                </span>
                <span className="flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-cyan-500" />
                    Institutions {instPct.toFixed(1)}%
                </span>
                <span className="flex items-center gap-1">
                    <span className="h-2 w-2 rounded-full bg-muted-foreground/20" />
                    Public {publicPct.toFixed(1)}%
                </span>
            </div>
        </div>
    )
}

// 52 Week Range indicator
function WeekRangeIndicator({
    current,
    low,
    high,
    currency
}: {
    current?: number
    low?: number | null
    high?: number | null
    currency: string
}) {
    if (!low || !high || !current) return null

    const range = high - low
    const position = range > 0 ? ((current - low) / range) * 100 : 50

    return (
        <div className="space-y-2">
            <div className="flex justify-between text-xs">
                <span className="text-red-400">{currency}{formatNumber(low)}</span>
                <span className="text-muted-foreground">52 Week Range</span>
                <span className="text-green-400">{currency}{formatNumber(high)}</span>
            </div>
            <div className="relative h-2 w-full rounded-full bg-gradient-to-r from-red-500/30 via-yellow-500/30 to-green-500/30">
                <div
                    className="absolute top-1/2 h-4 w-4 -translate-y-1/2 rounded-full bg-white border-2 border-primary shadow-lg"
                    style={{ left: `calc(${Math.min(Math.max(position, 0), 100)}% - 8px)` }}
                />
            </div>
        </div>
    )
}

export function StockInfoPanel({ fundamentals, currentPrice, currency = "$" }: StockInfoPanelProps) {
    return (
        <TooltipProvider delayDuration={200}>
            <Card>
                <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                        <div>
                            <CardTitle className="flex items-center gap-2">
                                <Building2 className="h-5 w-5" />
                                Fundamentals & Stats
                            </CardTitle>
                            <CardDescription>Comprehensive company data</CardDescription>
                        </div>
                        {fundamentals.website && (
                            <a
                                href={fundamentals.website}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="flex items-center gap-1 text-sm text-primary hover:underline"
                            >
                                <Globe className="h-4 w-4" />
                                Website
                                <ExternalLink className="h-3 w-3" />
                            </a>
                        )}
                    </div>
                </CardHeader>

                <CardContent>
                    <Tabs defaultValue="overview" className="w-full">
                        <TabsList className="grid w-full grid-cols-5 mb-4">
                            <TabsTrigger value="overview">Overview</TabsTrigger>
                            <TabsTrigger value="valuation">Valuation</TabsTrigger>
                            <TabsTrigger value="financials">Financials</TabsTrigger>
                            <TabsTrigger value="ownership">Ownership</TabsTrigger>
                            <TabsTrigger value="analyst">Analyst</TabsTrigger>
                        </TabsList>

                        {/* Overview Tab */}
                        <TabsContent value="overview" className="space-y-6">
                            {/* About Section */}
                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <Briefcase className="h-4 w-4" />
                                    About {fundamentals.name || fundamentals.symbol}
                                </h4>
                                <div className="grid grid-cols-3 gap-4">
                                    <StatItem
                                        label="Sector"
                                        value={fundamentals.sector || "—"}
                                        tooltip="The broad category of the company's primary business"
                                    />
                                    <StatItem
                                        label="Industry"
                                        value={fundamentals.industry || "—"}
                                        tooltip="The specific industry within the sector"
                                    />
                                    <StatItem
                                        label="Employees"
                                        value={fundamentals.employees ? formatLargeNumber(fundamentals.employees) : "—"}
                                        tooltip="Number of full-time employees"
                                        icon={<Users className="h-3 w-3" />}
                                    />
                                </div>
                                {fundamentals.headquarters_city && (
                                    <div className="flex items-center gap-1 text-xs text-muted-foreground">
                                        <MapPin className="h-3 w-3" />
                                        {fundamentals.headquarters_city}{fundamentals.headquarters_country ? `, ${fundamentals.headquarters_country}` : ""}
                                    </div>
                                )}
                                {fundamentals.description && (
                                    <p className="text-sm text-muted-foreground line-clamp-3">
                                        {fundamentals.description}
                                    </p>
                                )}
                            </div>

                            {/* Key Stats */}
                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <BarChart3 className="h-4 w-4" />
                                    Key Stats
                                </h4>
                                <div className="grid grid-cols-3 gap-4">
                                    <StatItem
                                        label="Market Cap"
                                        value={`${currency}${formatLargeNumber(fundamentals.market_cap)}`}
                                        tooltip="Total market value of outstanding shares"
                                    />
                                    <StatItem
                                        label="P/E Ratio (TTM)"
                                        value={formatNumber(fundamentals.pe_ratio)}
                                        tooltip="Price to Earnings ratio - measures valuation relative to earnings"
                                    />
                                    <StatItem
                                        label="EPS (TTM)"
                                        value={`${currency}${formatNumber(fundamentals.eps_ttm)}`}
                                        tooltip="Earnings per share over trailing twelve months"
                                    />
                                    <StatItem
                                        label="Revenue (TTM)"
                                        value={`${currency}${formatLargeNumber(fundamentals.revenue_ttm)}`}
                                        tooltip="Total revenue over trailing twelve months"
                                    />
                                    <StatItem
                                        label="Net Income"
                                        value={`${currency}${formatLargeNumber(fundamentals.net_income)}`}
                                        tooltip="Net profit after all expenses"
                                    />
                                    <StatItem
                                        label="Beta (1Y)"
                                        value={formatNumber(fundamentals.beta)}
                                        tooltip="Measures stock volatility relative to market. >1 = more volatile, <1 = less volatile"
                                        valueColor={fundamentals.beta && fundamentals.beta > 1 ? "text-amber-400" : "text-green-400"}
                                    />
                                </div>
                            </div>

                            {/* 52 Week Range */}
                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <Activity className="h-4 w-4" />
                                    Price Range
                                </h4>
                                <WeekRangeIndicator
                                    current={currentPrice}
                                    low={fundamentals["52_week_low"]}
                                    high={fundamentals["52_week_high"]}
                                    currency={currency}
                                />
                            </div>

                            {/* Upcoming Earnings */}
                            {fundamentals.next_earnings_date && (
                                <div className="space-y-3">
                                    <h4 className="text-sm font-semibold flex items-center gap-2">
                                        <Calendar className="h-4 w-4" />
                                        Upcoming Earnings
                                    </h4>
                                    <div className="grid grid-cols-3 gap-4">
                                        <StatItem
                                            label="Next Report Date"
                                            value={formatDate(fundamentals.next_earnings_date)}
                                            tooltip="Expected date of next earnings report"
                                        />
                                        <StatItem
                                            label="EPS Estimate"
                                            value={fundamentals.eps_forward ? `${currency}${formatNumber(fundamentals.eps_forward)}` : "—"}
                                            tooltip="Analyst estimate for next quarter earnings per share"
                                        />
                                    </div>
                                </div>
                            )}
                        </TabsContent>

                        {/* Valuation Tab */}
                        <TabsContent value="valuation" className="space-y-6">
                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <DollarSign className="h-4 w-4" />
                                    Valuation Ratios
                                </h4>
                                <div className="grid grid-cols-3 gap-4">
                                    <StatItem
                                        label="P/E Ratio (TTM)"
                                        value={formatNumber(fundamentals.pe_ratio)}
                                        tooltip="Price to Earnings - compares stock price to earnings per share"
                                    />
                                    <StatItem
                                        label="Forward P/E"
                                        value={formatNumber(fundamentals.forward_pe)}
                                        tooltip="P/E based on projected future earnings"
                                    />
                                    <StatItem
                                        label="PEG Ratio"
                                        value={formatNumber(fundamentals.peg_ratio)}
                                        tooltip="P/E adjusted for growth rate. <1 may indicate undervaluation"
                                        valueColor={fundamentals.peg_ratio && fundamentals.peg_ratio < 1 ? "text-green-400" : undefined}
                                    />
                                    <StatItem
                                        label="P/B Ratio"
                                        value={formatNumber(fundamentals.pb_ratio)}
                                        tooltip="Price to Book - compares market value to book value"
                                    />
                                    <StatItem
                                        label="P/S Ratio"
                                        value={formatNumber(fundamentals.ps_ratio)}
                                        tooltip="Price to Sales - compares stock price to revenue per share"
                                    />
                                    <StatItem
                                        label="Enterprise Value"
                                        value={`${currency}${formatLargeNumber(fundamentals.enterprise_value)}`}
                                        tooltip="Market cap + total debt - cash"
                                    />
                                </div>
                            </div>

                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <Percent className="h-4 w-4" />
                                    Dividends
                                </h4>
                                <div className="grid grid-cols-3 gap-4">
                                    <StatItem
                                        label="Dividend Yield"
                                        value={formatPercent(fundamentals.dividend_yield)}
                                        tooltip="Annual dividend as percentage of stock price"
                                        valueColor={fundamentals.dividend_yield ? "text-green-400" : undefined}
                                    />
                                    <StatItem
                                        label="Dividend Rate"
                                        value={fundamentals.dividend_rate ? `${currency}${formatNumber(fundamentals.dividend_rate)}` : "—"}
                                        tooltip="Annual dividend per share"
                                    />
                                    <StatItem
                                        label="Payout Ratio"
                                        value={formatPercent(fundamentals.payout_ratio)}
                                        tooltip="Percentage of earnings paid as dividends"
                                    />
                                </div>
                            </div>
                        </TabsContent>

                        {/* Financials Tab */}
                        <TabsContent value="financials" className="space-y-6">
                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <TrendingUp className="h-4 w-4" />
                                    Growth & Profitability
                                </h4>
                                <div className="grid grid-cols-3 gap-4">
                                    <StatItem
                                        label="Revenue Growth"
                                        value={formatPercent(fundamentals.revenue_growth)}
                                        tooltip="Year-over-year revenue growth rate"
                                        valueColor={fundamentals.revenue_growth && fundamentals.revenue_growth > 0 ? "text-green-400" : "text-red-400"}
                                    />
                                    <StatItem
                                        label="Earnings Growth"
                                        value={formatPercent(fundamentals.earnings_growth)}
                                        tooltip="Year-over-year earnings growth rate"
                                        valueColor={fundamentals.earnings_growth && fundamentals.earnings_growth > 0 ? "text-green-400" : "text-red-400"}
                                    />
                                    <StatItem
                                        label="Profit Margin"
                                        value={formatPercent(fundamentals.profit_margin)}
                                        tooltip="Net income as percentage of revenue"
                                    />
                                    <StatItem
                                        label="Gross Margin"
                                        value={formatPercent(fundamentals.gross_margin)}
                                        tooltip="Gross profit as percentage of revenue"
                                    />
                                    <StatItem
                                        label="Operating Margin"
                                        value={formatPercent(fundamentals.operating_margin)}
                                        tooltip="Operating income as percentage of revenue"
                                    />
                                    <StatItem
                                        label="EBITDA"
                                        value={`${currency}${formatLargeNumber(fundamentals.ebitda)}`}
                                        tooltip="Earnings before interest, taxes, depreciation, and amortization"
                                    />
                                </div>
                            </div>

                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <Shield className="h-4 w-4" />
                                    Financial Health
                                </h4>
                                <div className="grid grid-cols-3 gap-4">
                                    <StatItem
                                        label="Current Ratio"
                                        value={formatNumber(fundamentals.current_ratio)}
                                        tooltip="Current assets / current liabilities. >1 is healthy"
                                        valueColor={fundamentals.current_ratio && fundamentals.current_ratio > 1 ? "text-green-400" : "text-amber-400"}
                                    />
                                    <StatItem
                                        label="Quick Ratio"
                                        value={formatNumber(fundamentals.quick_ratio)}
                                        tooltip="Liquid assets / current liabilities"
                                    />
                                    <StatItem
                                        label="Debt/Equity"
                                        value={formatNumber(fundamentals.debt_to_equity)}
                                        tooltip="Total debt / shareholders' equity"
                                        valueColor={fundamentals.debt_to_equity && fundamentals.debt_to_equity > 1 ? "text-amber-400" : "text-green-400"}
                                    />
                                    <StatItem
                                        label="Total Cash"
                                        value={`${currency}${formatLargeNumber(fundamentals.total_cash)}`}
                                        tooltip="Cash and cash equivalents on hand"
                                    />
                                    <StatItem
                                        label="Total Debt"
                                        value={`${currency}${formatLargeNumber(fundamentals.total_debt)}`}
                                        tooltip="Total short and long-term debt"
                                    />
                                    <StatItem
                                        label="Free Cash Flow"
                                        value={`${currency}${formatLargeNumber(fundamentals.free_cash_flow)}`}
                                        tooltip="Operating cash flow minus capital expenditures"
                                        valueColor={fundamentals.free_cash_flow && fundamentals.free_cash_flow > 0 ? "text-green-400" : "text-red-400"}
                                    />
                                </div>
                            </div>

                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <Layers className="h-4 w-4" />
                                    Returns
                                </h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <StatItem
                                        label="Return on Equity (ROE)"
                                        value={formatPercent(fundamentals.roe)}
                                        tooltip="Net income as percentage of shareholders' equity"
                                        valueColor={fundamentals.roe && fundamentals.roe > 0.15 ? "text-green-400" : undefined}
                                    />
                                    <StatItem
                                        label="Return on Assets (ROA)"
                                        value={formatPercent(fundamentals.roa)}
                                        tooltip="Net income as percentage of total assets"
                                    />
                                </div>
                            </div>
                        </TabsContent>

                        {/* Ownership Tab */}
                        <TabsContent value="ownership" className="space-y-6">
                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <PieChart className="h-4 w-4" />
                                    Ownership Structure
                                </h4>
                                <OwnershipBar
                                    insider={fundamentals.insider_ownership}
                                    institutional={fundamentals.institutional_ownership}
                                />
                            </div>

                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <Layers className="h-4 w-4" />
                                    Shares
                                </h4>
                                <div className="grid grid-cols-2 gap-4">
                                    <StatItem
                                        label="Shares Outstanding"
                                        value={formatLargeNumber(fundamentals.shares_outstanding)}
                                        tooltip="Total number of shares issued"
                                    />
                                    <StatItem
                                        label="Float Shares"
                                        value={formatLargeNumber(fundamentals.float_shares)}
                                        tooltip="Shares available for public trading"
                                    />
                                </div>
                            </div>
                        </TabsContent>

                        {/* Analyst Tab */}
                        <TabsContent value="analyst" className="space-y-6">
                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <Target className="h-4 w-4" />
                                    Analyst Recommendation
                                </h4>
                                <div className="flex items-center gap-4">
                                    <RecommendationBadge recommendation={fundamentals.recommendation} />
                                    {fundamentals.analyst_count && (
                                        <span className="text-sm text-muted-foreground">
                                            Based on {fundamentals.analyst_count} analyst{fundamentals.analyst_count > 1 ? "s" : ""}
                                        </span>
                                    )}
                                </div>
                            </div>

                            <div className="space-y-3">
                                <h4 className="text-sm font-semibold flex items-center gap-2">
                                    <TrendingUp className="h-4 w-4" />
                                    Price Targets
                                </h4>
                                <div className="grid grid-cols-3 gap-4">
                                    <StatItem
                                        label="Target Low"
                                        value={`${currency}${formatNumber(fundamentals.target_low)}`}
                                        tooltip="Lowest analyst price target"
                                        valueColor="text-red-400"
                                    />
                                    <StatItem
                                        label="Target Mean"
                                        value={`${currency}${formatNumber(fundamentals.target_mean)}`}
                                        tooltip="Average analyst price target"
                                        valueColor="text-amber-400"
                                    />
                                    <StatItem
                                        label="Target High"
                                        value={`${currency}${formatNumber(fundamentals.target_high)}`}
                                        tooltip="Highest analyst price target"
                                        valueColor="text-green-400"
                                    />
                                </div>
                                {currentPrice && fundamentals.target_mean && (
                                    <div className="p-3 rounded-lg bg-muted/50">
                                        <p className="text-sm text-muted-foreground">
                                            {fundamentals.target_mean > currentPrice ? (
                                                <span className="text-green-400">
                                                    ↑ {((fundamentals.target_mean - currentPrice) / currentPrice * 100).toFixed(1)}% upside
                                                </span>
                                            ) : (
                                                <span className="text-red-400">
                                                    ↓ {((currentPrice - fundamentals.target_mean) / currentPrice * 100).toFixed(1)}% downside
                                                </span>
                                            )} to mean target
                                        </p>
                                    </div>
                                )}
                            </div>
                        </TabsContent>
                    </Tabs>
                </CardContent>
            </Card>
        </TooltipProvider>
    )
}
