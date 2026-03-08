"use client"

import { Button } from "@/components/ui/button"
import {
  ANALYSIS_MODE_OPTIONS,
  type AnalysisMode,
  getAnalysisModeDescription,
} from "@/lib/analysis-mode"
import { cn } from "@/lib/utils"

interface AnalysisModeToggleProps {
  mode: AnalysisMode
  onModeChange: (mode: AnalysisMode) => void
  className?: string
  showDescription?: boolean
  compact?: boolean
}

export function AnalysisModeToggle({
  mode,
  onModeChange,
  className,
  showDescription = false,
  compact = false,
}: AnalysisModeToggleProps) {
  return (
    <div className={cn("space-y-2", className)}>
      <div className="inline-flex rounded-full border border-border/70 bg-background/80 p-1 shadow-sm">
        {ANALYSIS_MODE_OPTIONS.map((option) => {
          const isActive = option.value === mode

          return (
            <Button
              key={option.value}
              type="button"
              size={compact ? "sm" : "default"}
              variant={isActive ? "default" : "ghost"}
              className={cn(
                "rounded-full",
                !compact && "min-w-[118px]",
                !isActive && "text-muted-foreground"
              )}
              onClick={() => onModeChange(option.value)}
            >
              {option.label}
            </Button>
          )
        })}
      </div>
      {showDescription && (
        <p className="max-w-xl text-sm text-muted-foreground">
          {getAnalysisModeDescription(mode)}
        </p>
      )}
    </div>
  )
}
