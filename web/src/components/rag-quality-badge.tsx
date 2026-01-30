"use client"

import { useState } from "react"
import { cn } from "@/lib/utils"
import {
    Tooltip,
    TooltipContent,
    TooltipProvider,
    TooltipTrigger,
} from "@/components/ui/tooltip"
import { ShieldCheck, ChevronDown, ChevronUp, AlertTriangle, CheckCircle2 } from "lucide-react"

interface RAGValidation {
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
    validation_details?: {
        faithfulness?: { score: number; description: string }
        context_relevancy?: { score: number; description: string }
        groundedness?: { score: number; claims_verified: number; claims_total: number; description: string }
        temporal_validity?: { score: number; oldest_source_hours: number; threshold_hours: number; description: string }
    }
}

interface RAGQualityBadgeProps {
    validation?: RAGValidation | null
    className?: string
    size?: "sm" | "md" | "lg"
    showDetails?: boolean
}

const gradeColors: Record<string, { bg: string; text: string; border: string }> = {
    A: { bg: "bg-emerald-500/10", text: "text-emerald-400", border: "border-emerald-500/30" },
    B: { bg: "bg-green-500/10", text: "text-green-400", border: "border-green-500/30" },
    C: { bg: "bg-yellow-500/10", text: "text-yellow-400", border: "border-yellow-500/30" },
    D: { bg: "bg-orange-500/10", text: "text-orange-400", border: "border-orange-500/30" },
    F: { bg: "bg-red-500/10", text: "text-red-400", border: "border-red-500/30" },
    Unknown: { bg: "bg-gray-500/10", text: "text-gray-400", border: "border-gray-500/30" },
}

const sizeStyles = {
    sm: { container: "px-1.5 py-0.5 text-[10px]", icon: "h-2.5 w-2.5", gap: "gap-1" },
    md: { container: "px-2 py-1 text-xs", icon: "h-3 w-3", gap: "gap-1.5" },
    lg: { container: "px-3 py-1.5 text-sm", icon: "h-4 w-4", gap: "gap-2" },
}

function ScoreBar({ score, label }: { score: number; label: string }) {
    const getColor = (s: number) => {
        if (s >= 80) return "bg-emerald-500"
        if (s >= 60) return "bg-yellow-500"
        if (s >= 40) return "bg-orange-500"
        return "bg-red-500"
    }

    return (
        <div className="space-y-1">
            <div className="flex justify-between text-[10px]">
                <span className="text-muted-foreground">{label}</span>
                <span className="font-medium">{score.toFixed(0)}</span>
            </div>
            <div className="h-1.5 bg-muted rounded-full overflow-hidden">
                <div
                    className={cn("h-full rounded-full transition-all", getColor(score))}
                    style={{ width: `${Math.min(100, Math.max(0, score))}%` }}
                />
            </div>
        </div>
    )
}

export function RAGQualityBadge({
    validation,
    className,
    size = "md",
    showDetails = true,
}: RAGQualityBadgeProps) {
    const [expanded, setExpanded] = useState(false)
    const styles = sizeStyles[size]

    if (!validation) {
        return null
    }

    const grade = validation.quality_grade || "Unknown"
    const colors = gradeColors[grade] || gradeColors.Unknown
    const isGood = grade === "A" || grade === "B"

    const tooltipContent = (
        <div className="space-y-2 min-w-[200px]">
            <div className="flex items-center justify-between">
                <span className="font-medium">RAG Quality Score</span>
                <span className={cn("font-bold text-lg", colors.text)}>
                    {validation.overall_score.toFixed(0)}
                </span>
            </div>

            <div className="space-y-2 pt-1 border-t border-border/50">
                <ScoreBar score={validation.faithfulness_score} label="Faithfulness" />
                <ScoreBar score={validation.context_relevancy_score} label="Context Relevancy" />
                <ScoreBar score={validation.groundedness_score} label="Groundedness" />
                <ScoreBar score={validation.temporal_validity_score} label="Temporal Validity" />
            </div>

            {(validation.claims_verified !== undefined || validation.sources_used !== undefined) && (
                <div className="pt-1 border-t border-border/50 text-[10px] text-muted-foreground space-y-0.5">
                    {validation.claims_verified !== undefined && validation.claims_total !== undefined && (
                        <p>Claims verified: {validation.claims_verified}/{validation.claims_total}</p>
                    )}
                    {validation.sources_used !== undefined && (
                        <p>Sources used: {validation.sources_used}</p>
                    )}
                    {validation.oldest_source_age_hours !== undefined && (
                        <p>Oldest source: {validation.oldest_source_age_hours.toFixed(1)}h ago</p>
                    )}
                </div>
            )}
        </div>
    )

    return (
        <TooltipProvider delayDuration={200}>
            <Tooltip>
                <TooltipTrigger asChild>
                    <div
                        className={cn(
                            "inline-flex items-center rounded-md font-medium cursor-help transition-colors border",
                            styles.container,
                            styles.gap,
                            colors.bg,
                            colors.text,
                            colors.border,
                            className
                        )}
                    >
                        {isGood ? (
                            <ShieldCheck className={styles.icon} />
                        ) : (
                            <AlertTriangle className={styles.icon} />
                        )}
                        <span>Quality: {grade}</span>
                        <span className="opacity-60">({validation.overall_score.toFixed(0)})</span>
                    </div>
                </TooltipTrigger>
                <TooltipContent side="bottom" className="text-xs p-3">
                    {tooltipContent}
                </TooltipContent>
            </Tooltip>
        </TooltipProvider>
    )
}

// Expandable panel version for detailed display
export function RAGQualityPanel({
    validation,
    className,
    defaultExpanded = false,
}: {
    validation?: RAGValidation | null
    className?: string
    defaultExpanded?: boolean
}) {
    const [expanded, setExpanded] = useState(defaultExpanded)

    if (!validation) {
        return null
    }

    const grade = validation.quality_grade || "Unknown"
    const colors = gradeColors[grade] || gradeColors.Unknown
    const isGood = grade === "A" || grade === "B"

    return (
        <div className={cn("rounded-lg border", colors.border, colors.bg, className)}>
            <button
                onClick={() => setExpanded(!expanded)}
                className={cn(
                    "w-full flex items-center justify-between p-3 text-left",
                    colors.text
                )}
            >
                <div className="flex items-center gap-2">
                    {isGood ? (
                        <CheckCircle2 className="h-4 w-4" />
                    ) : (
                        <AlertTriangle className="h-4 w-4" />
                    )}
                    <span className="font-medium">RAG Quality: Grade {grade}</span>
                    <span className="text-sm opacity-70">({validation.overall_score.toFixed(0)}/100)</span>
                </div>
                {expanded ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            </button>

            {expanded && (
                <div className="px-3 pb-3 space-y-3 border-t border-border/30">
                    <div className="pt-3 grid grid-cols-2 gap-3">
                        <ScoreBar score={validation.faithfulness_score} label="Faithfulness" />
                        <ScoreBar score={validation.context_relevancy_score} label="Context Relevancy" />
                        <ScoreBar score={validation.groundedness_score} label="Groundedness" />
                        <ScoreBar score={validation.temporal_validity_score} label="Temporal Validity" />
                    </div>

                    <div className="text-xs text-muted-foreground space-y-1 pt-2 border-t border-border/30">
                        <p><strong className="text-foreground">Faithfulness:</strong> Is the answer grounded in retrieved context?</p>
                        <p><strong className="text-foreground">Context Relevancy:</strong> Are retrieved documents relevant?</p>
                        <p><strong className="text-foreground">Groundedness:</strong> Are claims supported by source data?</p>
                        <p><strong className="text-foreground">Temporal Validity:</strong> Is context recent enough?</p>
                    </div>

                    {(validation.claims_verified !== undefined || validation.sources_used !== undefined) && (
                        <div className="pt-2 border-t border-border/30 text-xs text-muted-foreground flex flex-wrap gap-4">
                            {validation.claims_verified !== undefined && validation.claims_total !== undefined && (
                                <span>Claims verified: <strong className="text-foreground">{validation.claims_verified}/{validation.claims_total}</strong></span>
                            )}
                            {validation.sources_used !== undefined && (
                                <span>Sources: <strong className="text-foreground">{validation.sources_used}</strong></span>
                            )}
                            {validation.validation_time_ms !== undefined && (
                                <span>Validation time: <strong className="text-foreground">{validation.validation_time_ms}ms</strong></span>
                            )}
                        </div>
                    )}
                </div>
            )}
        </div>
    )
}
