"use client"

import { cn } from "@/lib/utils"
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip"
import { Database, Sparkles, Loader2 } from "lucide-react"

interface RAGBadgeProps {
  isActive?: boolean
  isLoading?: boolean
  size?: "sm" | "md" | "lg"
  variant?: "default" | "compact" | "inline"
  showLabel?: boolean
  className?: string
  totalResults?: number
  retrievalTimeMs?: number
  sourcesSearched?: string[]
}

const sizeStyles = {
  sm: {
    container: "px-1.5 py-0.5 text-[10px]",
    icon: "h-2.5 w-2.5",
    gap: "gap-1",
  },
  md: {
    container: "px-2 py-1 text-xs",
    icon: "h-3 w-3",
    gap: "gap-1.5",
  },
  lg: {
    container: "px-3 py-1.5 text-sm",
    icon: "h-4 w-4",
    gap: "gap-2",
  },
}

export function RAGBadge({
  isActive = true,
  isLoading = false,
  size = "md",
  variant = "default",
  showLabel = true,
  className,
  totalResults,
  retrievalTimeMs,
  sourcesSearched,
}: RAGBadgeProps) {
  const styles = sizeStyles[size]

  const tooltipContent = isActive ? (
    <div className="space-y-1.5">
      <p className="font-medium">RAG-Enhanced Analysis</p>
      <p className="text-muted-foreground">
        This uses Retrieval-Augmented Generation (RAG) to include relevant
        historical context from past analyses, signals, news, and knowledge base.
      </p>
      {(totalResults !== undefined || retrievalTimeMs !== undefined || sourcesSearched) && (
        <div className="pt-1 border-t border-border/50 space-y-0.5">
          {totalResults !== undefined && (
            <p className="text-muted-foreground">
              Retrieved: <span className="text-foreground">{totalResults} results</span>
            </p>
          )}
          {sourcesSearched && sourcesSearched.length > 0 && (
            <p className="text-muted-foreground">
              Sources: <span className="text-foreground">{sourcesSearched.join(", ")}</span>
            </p>
          )}
          {retrievalTimeMs !== undefined && (
            <p className="text-muted-foreground">
              Time: <span className="text-foreground">{retrievalTimeMs}ms</span>
            </p>
          )}
        </div>
      )}
    </div>
  ) : (
    <div className="space-y-1">
      <p className="font-medium">Standard Analysis</p>
      <p className="text-muted-foreground">
        RAG context was not used for this analysis. Run a new analysis to
        include historical context.
      </p>
    </div>
  )

  if (variant === "inline") {
    return (
      <TooltipProvider delayDuration={200}>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className={cn(
                "inline-flex items-center cursor-help",
                styles.gap,
                isActive ? "text-purple-400" : "text-muted-foreground",
                className
              )}
            >
              {isLoading ? (
                <Loader2 className={cn(styles.icon, "animate-spin")} />
              ) : isActive ? (
                <Database className={styles.icon} />
              ) : (
                <Sparkles className={styles.icon} />
              )}
              {showLabel && (
                <span className="font-medium">
                  {isLoading ? "Loading RAG..." : isActive ? "RAG" : "Standard"}
                </span>
              )}
            </span>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-xs text-xs">
            {tooltipContent}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  if (variant === "compact") {
    return (
      <TooltipProvider delayDuration={200}>
        <Tooltip>
          <TooltipTrigger asChild>
            <span
              className={cn(
                "inline-flex items-center cursor-help",
                isActive
                  ? "text-purple-400"
                  : "text-muted-foreground",
                className
              )}
            >
              {isLoading ? (
                <Loader2 className={cn(styles.icon, "animate-spin")} />
              ) : isActive ? (
                <Database className={styles.icon} />
              ) : (
                <Sparkles className={styles.icon} />
              )}
            </span>
          </TooltipTrigger>
          <TooltipContent side="bottom" className="max-w-xs text-xs">
            {tooltipContent}
          </TooltipContent>
        </Tooltip>
      </TooltipProvider>
    )
  }

  // Default badge variant
  return (
    <TooltipProvider delayDuration={200}>
      <Tooltip>
        <TooltipTrigger asChild>
          <span
            className={cn(
              "inline-flex items-center rounded-md font-medium cursor-help transition-colors",
              styles.container,
              styles.gap,
              isActive
                ? "bg-purple-500/10 text-purple-400 border border-purple-500/30"
                : "bg-muted text-muted-foreground border border-border",
              className
            )}
          >
            {isLoading ? (
              <Loader2 className={cn(styles.icon, "animate-spin")} />
            ) : isActive ? (
              <Database className={styles.icon} />
            ) : (
              <Sparkles className={styles.icon} />
            )}
            {showLabel && (
              <span>
                {isLoading
                  ? "Loading..."
                  : isActive
                    ? "RAG Enhanced"
                    : "Standard"}
              </span>
            )}
          </span>
        </TooltipTrigger>
        <TooltipContent side="bottom" className="max-w-xs text-xs">
          {tooltipContent}
        </TooltipContent>
      </Tooltip>
    </TooltipProvider>
  )
}

// Loading indicator specifically for RAG operations
export function RAGLoadingIndicator({
  className,
  message = "Retrieving context with RAG...",
}: {
  className?: string
  message?: string
}) {
  return (
    <div className={cn("flex items-center gap-2 text-purple-400", className)}>
      <Database className="h-4 w-4 animate-pulse" />
      <Loader2 className="h-4 w-4 animate-spin" />
      <span className="text-sm">{message}</span>
    </div>
  )
}

// Sources indicator showing what RAG searched
export function RAGSourcesIndicator({
  sources,
  className,
}: {
  sources: string[]
  className?: string
}) {
  if (!sources || sources.length === 0) return null

  return (
    <div className={cn("flex items-center gap-2 text-xs text-muted-foreground", className)}>
      <Database className="h-3 w-3 text-purple-400" />
      <span>Searched: {sources.join(", ")}</span>
    </div>
  )
}
