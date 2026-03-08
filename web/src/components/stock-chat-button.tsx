"use client"

import Link from "next/link"
import type * as React from "react"
import { MessageSquare, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip"
import { cn } from "@/lib/utils"

interface StockChatButtonProps {
  symbol: string
  hasAnalysis: boolean
  className?: string
  size?: "default" | "sm" | "lg" | "icon" | "icon-sm" | "icon-lg"
  label?: string
  stopPropagation?: boolean
}

export function StockChatButton({
  symbol,
  hasAnalysis,
  className,
  size = "sm",
  label = "Chat",
  stopPropagation = false,
}: StockChatButtonProps) {
  const handleClick = (event: React.MouseEvent<HTMLElement>) => {
    if (stopPropagation) {
      event.stopPropagation()
    }
  }

  if (hasAnalysis) {
    return (
      <Button
        asChild
        size={size}
        variant="secondary"
        className={cn(
          "border border-cyan-200 bg-cyan-50 text-cyan-700 hover:bg-cyan-100 dark:border-cyan-500/20 dark:bg-cyan-500/10 dark:text-cyan-100 dark:hover:bg-cyan-500/20",
          className
        )}
      >
        <Link href={`/stocks/${encodeURIComponent(symbol)}/chat`} onClick={handleClick}>
          <MessageSquare className="h-3.5 w-3.5" />
          {label}
        </Link>
      </Button>
    )
  }

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <span
          className="inline-flex"
          onClick={handleClick}
          onMouseDown={handleClick}
        >
          <Button
            type="button"
            size={size}
            variant="outline"
            aria-disabled="true"
            className={cn(
              "cursor-not-allowed border-slate-300 bg-slate-100 text-slate-500 grayscale saturate-0 hover:bg-slate-100 hover:text-slate-500 dark:border-white/15 dark:bg-black dark:text-white/60 dark:hover:bg-black dark:hover:text-white/60",
              className
            )}
          >
            <span className="relative inline-flex">
              <MessageSquare className="h-3.5 w-3.5" />
              <X className="absolute -right-1.5 -top-1.5 h-3 w-3 rounded-full bg-slate-100 text-slate-700 dark:bg-black dark:text-white" />
            </span>
            {label}
          </Button>
        </span>
      </TooltipTrigger>
      <TooltipContent side="top" sideOffset={8}>
        AI chat is available only after this stock has a saved analysis.
      </TooltipContent>
    </Tooltip>
  )
}
