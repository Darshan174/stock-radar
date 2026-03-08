"use client"

import { Badge } from "@/components/ui/badge"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

type ChatContextLevel = "locked" | "limited" | "good" | "rich"

interface ChatContextMeta {
  level: ChatContextLevel
  label: string
  description: string
  className: string
}

interface ChatContextBadgeProps {
  analysisCount: number
  className?: string
  showLocked?: boolean
  showCount?: boolean
}

export function getChatContextMeta(analysisCount: number): ChatContextMeta {
  if (analysisCount >= 5) {
    return {
      level: "rich",
      label: "Rich context",
      description: "5+ saved analyses. Best for stronger comparisons, trend consistency checks, and historical Q&A.",
      className: "border-cyan-200 bg-cyan-50 text-cyan-700 dark:border-cyan-500/30 dark:bg-cyan-500/10 dark:text-cyan-300",
    }
  }

  if (analysisCount >= 3) {
    return {
      level: "good",
      label: "Good context",
      description: "3-4 saved analyses. Enough history for solid follow-up questions and pattern comparisons.",
      className: "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-500/30 dark:bg-emerald-500/10 dark:text-emerald-300",
    }
  }

  if (analysisCount >= 1) {
    return {
      level: "limited",
      label: "Limited context",
      description: "1-2 saved analyses. Good for explaining the latest report, but historical context is still thin.",
      className: "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-500/30 dark:bg-amber-500/10 dark:text-amber-200",
    }
  }

  return {
    level: "locked",
    label: "Chat locked",
    description: "No saved analyses yet. Run at least one analysis to unlock stock chat.",
    className: "border-slate-300 bg-slate-100 text-slate-600 grayscale saturate-0 dark:border-white/15 dark:bg-black dark:text-white/70",
  }
}

function formatAnalysisCount(analysisCount: number): string {
  if (analysisCount >= 5) return "5+ analyses"
  if (analysisCount === 1) return "1 analysis"
  return `${analysisCount} analyses`
}

export function ChatContextBadge({
  analysisCount,
  className,
  showLocked = false,
  showCount = false,
}: ChatContextBadgeProps) {
  const meta = getChatContextMeta(analysisCount)

  if (analysisCount < 1 && !showLocked) {
    return null
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span className="inline-flex">
          <Badge variant="outline" className={cn("text-xs", meta.className, className)}>
            {meta.label}
            {showCount ? ` · ${formatAnalysisCount(analysisCount)}` : ""}
          </Badge>
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8}>
        {meta.description}
      </TooltipContent>
    </Tooltip>
  )
}
