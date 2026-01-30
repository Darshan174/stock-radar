"use client"

import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"
import {
    TrendingUp,
    TrendingDown,
    Minus,
    Brain,
    Target,
    Shield,
    Newspaper,
    Clock,
    Cpu,
    ChevronDown,
    ChevronUp,
    Info,
    Twitter,
    MessageCircle,
} from "lucide-react"
import { RAGBadge } from "./rag-badge"
import { RAGQualityBadge } from "./rag-quality-badge"
import { useState } from "react"
import { cn } from "@/lib/utils"

interface Analysis {
    id: number
    signal: string
    confidence: number
    mode: string
    reasoning?: string | null
    technical_summary?: string | null
    sentiment_summary?: string | null
    support_level?: number | null
    resistance_level?: number | null
    target_price?: number | null
    stop_loss?: number | null
    risk_reward_ratio?: number | null
    llm_model?: string | null
    llm_tokens_used?: number | null
    tokens_used?: number | null
    analysis_duration_ms?: number | null
    created_at: string
    // RAG fields
    embedding?: number[] | null
    embedding_text?: string | null
    // Social sentiment fields
    social_sentiment?: {
        twitter_mentions?: number
        reddit_mentions?: number
        overall_sentiment?: "bullish" | "bearish" | "neutral"
        trending_topics?: string[]
    } | null
    // RAG validation fields
    rag_validation?: {
        faithfulness_score: number
        context_relevancy_score: number
        groundedness_score: number
        temporal_validity_score: number
        overall_score: number
        quality_grade: string
        claims_verified?: number
        claims_total?: number
        sources_used?: number
        oldest_source_age_hours?: number
        validation_time_ms?: number
    } | null
}

interface AIAnalysisPanelProps {
    analysis: Analysis | null
    currentPrice?: number
    currency?: string
    onRunAnalysis?: () => void
    isAnalyzing?: boolean
}

// Info tooltip descriptions
const INFO_TOOLTIPS = {
    signal: "AI-generated trading recommendation based on technical analysis, market sentiment, and price patterns. Confidence indicates how certain the model is.",
    technical: "Summary of key technical indicators including moving averages, RSI, MACD, and price patterns that influenced the analysis.",
    sentiment: "Analysis of recent news articles, social media mentions, and market sentiment around this stock.",
    keyLevels: "Important price levels identified by the AI. Support is where price may find buyers, resistance is where sellers may appear.",
    riskManagement: "Suggested stop loss to limit potential losses and risk/reward ratio for the trade. A ratio above 2:1 is generally favorable.",
    support: "Price level where buying interest is expected to be strong enough to prevent further decline.",
    resistance: "Price level where selling pressure is expected to be strong enough to prevent further rise.",
    target: "AI-predicted price target based on technical and fundamental analysis.",
    stopLoss: "Suggested exit price to limit losses if the trade goes against expectations.",
    riskReward: "Ratio of potential profit to potential loss. Higher is better - 3:1 means potential gain is 3x the risk.",
    socialSentiment: "Real-time sentiment from social media platforms like Twitter/X and Reddit communities.",
}

// Info button component
function InfoButton({ tooltip }: { tooltip: string }) {
    return (
        <Tooltip>
            <TooltipTrigger asChild>
                <button className="ml-1.5 text-muted-foreground/50 hover:text-muted-foreground transition-colors">
                    <Info className="h-3.5 w-3.5" />
                </button>
            </TooltipTrigger>
            <TooltipContent side="top" className="max-w-xs text-xs">
                {tooltip}
            </TooltipContent>
        </Tooltip>
    )
}

function getSignalConfig(signal: string) {
    const configs: Record<string, {
        color: string
        bgColor: string
        borderColor: string
        icon: React.ReactNode
        label: string
    }> = {
        strong_buy: {
            color: "text-emerald-400",
            bgColor: "bg-emerald-500/10",
            borderColor: "border-emerald-500/30",
            icon: <TrendingUp className="h-5 w-5" />,
            label: "STRONG BUY",
        },
        buy: {
            color: "text-green-400",
            bgColor: "bg-green-500/10",
            borderColor: "border-green-500/30",
            icon: <TrendingUp className="h-4 w-4" />,
            label: "BUY",
        },
        hold: {
            color: "text-yellow-400",
            bgColor: "bg-yellow-500/10",
            borderColor: "border-yellow-500/30",
            icon: <Minus className="h-4 w-4" />,
            label: "HOLD",
        },
        sell: {
            color: "text-orange-400",
            bgColor: "bg-orange-500/10",
            borderColor: "border-orange-500/30",
            icon: <TrendingDown className="h-4 w-4" />,
            label: "SELL",
        },
        strong_sell: {
            color: "text-red-400",
            bgColor: "bg-red-500/10",
            borderColor: "border-red-500/30",
            icon: <TrendingDown className="h-5 w-5" />,
            label: "STRONG SELL",
        },
    }
    return configs[signal.toLowerCase()] || configs.hold
}

function formatTimeAgo(dateStr: string): string {
    const date = new Date(dateStr)
    const now = new Date()
    const diffMs = now.getTime() - date.getTime()
    const diffMins = Math.floor(diffMs / (1000 * 60))
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)

    if (diffMins < 1) return "just now"
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    return `${diffDays}d ago`
}

export function AIAnalysisPanel({
    analysis,
    currentPrice,
    currency = "$",
    onRunAnalysis,
    isAnalyzing = false,
}: AIAnalysisPanelProps) {
    const [isExpanded, setIsExpanded] = useState(true)

    if (!analysis) {
        return (
            <Card className="border-dashed">
                <CardContent className="py-8 text-center">
                    <Brain className="h-12 w-12 mx-auto text-muted-foreground mb-4" />
                    <h3 className="text-lg font-semibold mb-2">No AI Analysis Yet</h3>
                    <p className="text-muted-foreground text-sm mb-4">
                        Run an analysis to get AI-powered trading insights
                    </p>
                    {onRunAnalysis && (
                        <button
                            onClick={onRunAnalysis}
                            disabled={isAnalyzing}
                            className="px-4 py-2 bg-primary text-primary-foreground rounded-lg hover:bg-primary/90 disabled:opacity-50"
                        >
                            {isAnalyzing ? "Analyzing..." : "Run Analysis"}
                        </button>
                    )}
                </CardContent>
            </Card>
        )
    }

    const signalConfig = getSignalConfig(analysis.signal)
    const confidencePercent = Math.round(analysis.confidence * 100)

    return (
        <TooltipProvider delayDuration={200}>
            <Card className={cn("transition-all", signalConfig.borderColor, "border-2")}>
                <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                        <div className="flex items-center gap-3">
                            <div className={cn("p-2 rounded-lg", signalConfig.bgColor)}>
                                <Brain className={cn("h-5 w-5", signalConfig.color)} />
                            </div>
                            <div>
                                <div className="flex items-center">
                                    <CardTitle className="text-lg">AI Analysis</CardTitle>
                                    <InfoButton tooltip={INFO_TOOLTIPS.signal} />
                                </div>
                                <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                                    <Clock className="h-3 w-3" />
                                    <span>{formatTimeAgo(analysis.created_at)}</span>
                                    <span className="text-muted-foreground/50">•</span>
                                    <Cpu className="h-3 w-3" />
                                    <span className="font-medium text-foreground/80">{analysis.llm_model?.split("/").pop() || "AI"}</span>
                                    {(analysis.tokens_used || analysis.llm_tokens_used) && (
                                        <>
                                            <span className="text-muted-foreground/50">•</span>
                                            <span>{analysis.tokens_used || analysis.llm_tokens_used} tokens</span>
                                        </>
                                    )}
                                    {analysis.analysis_duration_ms && (
                                        <>
                                            <span className="text-muted-foreground/50">•</span>
                                            <span>{(analysis.analysis_duration_ms / 1000).toFixed(1)}s</span>
                                        </>
                                    )}
                                    {/* RAG Indicator */}
                                    <span className="text-muted-foreground/50">•</span>
                                    <RAGBadge
                                        isActive={!!analysis.embedding_text}
                                        size="sm"
                                        variant="inline"
                                    />
                                </div>
                            </div>
                        </div>

                        {/* Signal Badge */}
                        <div className="flex items-center gap-3">
                            <div className={cn(
                                "px-4 py-2 rounded-lg font-bold text-lg flex items-center gap-2",
                                signalConfig.bgColor,
                                signalConfig.color
                            )}>
                                {signalConfig.icon}
                                {signalConfig.label}
                            </div>
                            <div className="text-right">
                                <div className="text-2xl font-bold">{confidencePercent}%</div>
                                <div className="text-xs text-muted-foreground">confidence</div>
                            </div>
                        </div>
                    </div>
                </CardHeader>

                <CardContent className="space-y-4">
                    {/* Technical Summary */}
                    {analysis.technical_summary && (
                        <div className="p-3 rounded-lg bg-muted/50">
                            <div className="flex items-center gap-2 text-sm font-medium mb-2">
                                <TrendingUp className="h-4 w-4 text-blue-400" />
                                Technical Summary
                                <InfoButton tooltip={INFO_TOOLTIPS.technical} />
                            </div>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                {analysis.technical_summary}
                            </p>
                        </div>
                    )}

                    {/* Sentiment */}
                    {analysis.sentiment_summary && (
                        <div className="p-3 rounded-lg bg-muted/50">
                            <div className="flex items-center gap-2 text-sm font-medium mb-2">
                                <Newspaper className="h-4 w-4 text-purple-400" />
                                Market Sentiment
                                <InfoButton tooltip={INFO_TOOLTIPS.sentiment} />
                            </div>
                            <p className="text-sm text-muted-foreground leading-relaxed">
                                {analysis.sentiment_summary}
                            </p>
                        </div>
                    )}

                    {/* Social Sentiment (if available) */}
                    {analysis.social_sentiment && (
                        <div className="p-3 rounded-lg bg-muted/50">
                            <div className="flex items-center gap-2 text-sm font-medium mb-2">
                                <MessageCircle className="h-4 w-4 text-cyan-400" />
                                Social Sentiment
                                <InfoButton tooltip={INFO_TOOLTIPS.socialSentiment} />
                            </div>
                            <div className="flex items-center gap-4 text-sm">
                                {analysis.social_sentiment.twitter_mentions !== undefined && (
                                    <div className="flex items-center gap-1.5">
                                        <Twitter className="h-3.5 w-3.5 text-sky-400" />
                                        <span className="text-muted-foreground">
                                            {analysis.social_sentiment.twitter_mentions} mentions
                                        </span>
                                    </div>
                                )}
                                {analysis.social_sentiment.reddit_mentions !== undefined && (
                                    <div className="flex items-center gap-1.5">
                                        <span className="text-orange-500 font-bold text-xs">r/</span>
                                        <span className="text-muted-foreground">
                                            {analysis.social_sentiment.reddit_mentions} mentions
                                        </span>
                                    </div>
                                )}
                                {analysis.social_sentiment.overall_sentiment && (
                                    <Badge variant="outline" className={cn(
                                        analysis.social_sentiment.overall_sentiment === "bullish" ? "text-green-400 border-green-400/30" :
                                            analysis.social_sentiment.overall_sentiment === "bearish" ? "text-red-400 border-red-400/30" :
                                                "text-yellow-400 border-yellow-400/30"
                                    )}>
                                        {analysis.social_sentiment.overall_sentiment}
                                    </Badge>
                                )}
                            </div>
                            {analysis.social_sentiment.trending_topics && analysis.social_sentiment.trending_topics.length > 0 && (
                                <div className="mt-2 flex flex-wrap gap-1">
                                    {analysis.social_sentiment.trending_topics.map((topic, i) => (
                                        <span key={i} className="text-xs px-2 py-0.5 bg-muted rounded-full text-muted-foreground">
                                            #{topic}
                                        </span>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}

                    {/* Key Levels */}
                    {(analysis.support_level || analysis.resistance_level || analysis.target_price) && (
                        <div className="p-3 rounded-lg bg-muted/50">
                            <div className="flex items-center gap-2 text-sm font-medium mb-3">
                                <Target className="h-4 w-4 text-amber-400" />
                                Key Levels
                                <InfoButton tooltip={INFO_TOOLTIPS.keyLevels} />
                            </div>
                            <div className="grid grid-cols-3 gap-4">
                                {analysis.support_level && (
                                    <div>
                                        <div className="text-xs text-muted-foreground mb-1 flex items-center">
                                            Support
                                            <InfoButton tooltip={INFO_TOOLTIPS.support} />
                                        </div>
                                        <div className="text-lg font-semibold text-red-400">
                                            {currency}{analysis.support_level.toFixed(2)}
                                        </div>
                                        {currentPrice && (
                                            <div className="text-xs text-muted-foreground">
                                                {((currentPrice - analysis.support_level) / currentPrice * 100).toFixed(1)}% below
                                            </div>
                                        )}
                                    </div>
                                )}
                                {analysis.resistance_level && (
                                    <div>
                                        <div className="text-xs text-muted-foreground mb-1 flex items-center">
                                            Resistance
                                            <InfoButton tooltip={INFO_TOOLTIPS.resistance} />
                                        </div>
                                        <div className="text-lg font-semibold text-amber-400">
                                            {currency}{analysis.resistance_level.toFixed(2)}
                                        </div>
                                        {currentPrice && (
                                            <div className="text-xs text-muted-foreground">
                                                {((analysis.resistance_level - currentPrice) / currentPrice * 100).toFixed(1)}% above
                                            </div>
                                        )}
                                    </div>
                                )}
                                {analysis.target_price && (
                                    <div>
                                        <div className="text-xs text-muted-foreground mb-1 flex items-center">
                                            Target
                                            <InfoButton tooltip={INFO_TOOLTIPS.target} />
                                        </div>
                                        <div className="text-lg font-semibold text-green-400">
                                            {currency}{analysis.target_price.toFixed(2)}
                                        </div>
                                        {currentPrice && (
                                            <div className="text-xs text-muted-foreground">
                                                {((analysis.target_price - currentPrice) / currentPrice * 100).toFixed(1)}% upside
                                            </div>
                                        )}
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Risk Management */}
                    {(analysis.stop_loss || analysis.risk_reward_ratio) && (
                        <div className="p-3 rounded-lg bg-muted/50">
                            <div className="flex items-center gap-2 text-sm font-medium mb-3">
                                <Shield className="h-4 w-4 text-cyan-400" />
                                Risk Management
                                <InfoButton tooltip={INFO_TOOLTIPS.riskManagement} />
                            </div>
                            <div className="flex items-center gap-6">
                                {analysis.stop_loss && (
                                    <div>
                                        <div className="text-xs text-muted-foreground mb-1 flex items-center">
                                            Stop Loss
                                            <InfoButton tooltip={INFO_TOOLTIPS.stopLoss} />
                                        </div>
                                        <div className="text-lg font-semibold text-red-400">
                                            {currency}{analysis.stop_loss.toFixed(2)}
                                        </div>
                                        {currentPrice && (
                                            <div className="text-xs text-muted-foreground">
                                                {((currentPrice - analysis.stop_loss) / currentPrice * 100).toFixed(1)}% risk
                                            </div>
                                        )}
                                    </div>
                                )}
                                {analysis.risk_reward_ratio && (
                                    <div>
                                        <div className="text-xs text-muted-foreground mb-1 flex items-center">
                                            Risk/Reward
                                            <InfoButton tooltip={INFO_TOOLTIPS.riskReward} />
                                        </div>
                                        <div className="text-lg font-semibold text-cyan-400">
                                            {analysis.risk_reward_ratio.toFixed(1)}:1
                                        </div>
                                        <div className="text-xs text-muted-foreground">
                                            {analysis.risk_reward_ratio >= 2 ? "Favorable" : "Moderate"}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </div>
                    )}

                    {/* Reasoning (Expandable) */}
                    {analysis.reasoning && (
                        <div className="border-t pt-3">
                            <button
                                onClick={() => setIsExpanded(!isExpanded)}
                                className="flex items-center gap-2 text-sm text-muted-foreground hover:text-foreground transition-colors w-full"
                            >
                                {isExpanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
                                Full Reasoning
                            </button>
                            {isExpanded && (
                                <p className="mt-3 text-sm text-muted-foreground leading-relaxed">
                                    {analysis.reasoning}
                                </p>
                            )}
                        </div>
                    )}

                    {/* Model & RAG Info Footer */}
                    <div className="border-t pt-3 mt-3">
                        <div className="flex items-center justify-between text-xs">
                            <div className="flex items-center gap-3">
                                {/* Model Badge */}
                                <div className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-blue-500/10 text-blue-400">
                                    <Cpu className="h-3 w-3" />
                                    <span className="font-medium">{analysis.llm_model?.split("/").pop() || "AI Model"}</span>
                                </div>

                                {/* RAG Badge */}
                                <RAGBadge
                                    isActive={!!analysis.embedding_text}
                                    size="sm"
                                />

                                {/* RAG Quality Badge */}
                                {analysis.rag_validation && (
                                    <RAGQualityBadge
                                        validation={analysis.rag_validation}
                                        size="sm"
                                    />
                                )}
                            </div>

                            {/* Stats */}
                            <div className="flex items-center gap-3 text-muted-foreground">
                                {(analysis.tokens_used || analysis.llm_tokens_used) && (
                                    <span>{analysis.tokens_used || analysis.llm_tokens_used} tokens</span>
                                )}
                                {analysis.analysis_duration_ms && (
                                    <span>{(analysis.analysis_duration_ms / 1000).toFixed(1)}s</span>
                                )}
                            </div>
                        </div>
                    </div>
                </CardContent>
            </Card>
        </TooltipProvider>
    )
}
