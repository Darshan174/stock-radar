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
          "border border-cyan-500/20 bg-cyan-500/10 text-cyan-100 hover:bg-cyan-500/20",
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
              "cursor-not-allowed border-white/15 bg-black text-white/60 grayscale saturate-0 hover:bg-black hover:text-white/60 dark:border-white/15 dark:bg-black dark:hover:bg-black",
              className
            )}
          >
            <span className="relative inline-flex">
              <MessageSquare className="h-3.5 w-3.5" />
              <X className="absolute -right-1.5 -top-1.5 h-3 w-3 rounded-full bg-black text-white" />
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
