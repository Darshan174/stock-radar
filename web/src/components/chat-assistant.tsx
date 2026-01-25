"use client"

import * as React from "react"
import { useState, useRef, useEffect } from "react"
import { cn } from "@/lib/utils"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import {
  MessageCircle,
  Send,
  Loader2,
  Bot,
  User,
  Sparkles,
  FileText,
  TrendingUp,
  Newspaper,
  X,
  Maximize2,
  Minimize2,
} from "lucide-react"

interface ChatMessage {
  id: string
  role: "user" | "assistant"
  content: string
  stockSymbols?: string[]
  sourcesUsed?: Array<{
    type: string
    symbol?: string
    headline?: string
    similarity?: number
  }>
  processingTimeMs?: number
  tokensUsed?: number
  modelUsed?: string
  timestamp: Date
}

interface ChatResponse {
  success: boolean
  answer: string
  stockSymbols: string[]
  sourcesUsed: Array<{
    type: string
    symbol?: string
    headline?: string
    similarity?: number
  }>
  modelUsed: string
  tokensUsed: number
  processingTimeMs: number
  contextRetrieved?: {
    totalResults: number
    sourcesSearched: string[]
    retrievalTimeMs: number
  }
  error?: string
}

interface ChatAssistantProps {
  className?: string
  defaultSymbol?: string
  isFloating?: boolean
}

export function ChatAssistant({
  className,
  defaultSymbol,
  isFloating = false,
}: ChatAssistantProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isOpen, setIsOpen] = useState(!isFloating)
  const [isExpanded, setIsExpanded] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Focus input when chat opens
  useEffect(() => {
    if (isOpen) {
      inputRef.current?.focus()
    }
  }, [isOpen])

  const sendMessage = async () => {
    if (!input.trim() || isLoading) return

    const userMessage: ChatMessage = {
      id: Date.now().toString(),
      role: "user",
      content: input.trim(),
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: userMessage.content,
          symbol: defaultSymbol,
        }),
      })

      const data: ChatResponse = await response.json()

      if (data.error) {
        throw new Error(data.error)
      }

      const assistantMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: data.answer,
        stockSymbols: data.stockSymbols,
        sourcesUsed: data.sourcesUsed,
        processingTimeMs: data.processingTimeMs,
        tokensUsed: data.tokensUsed,
        modelUsed: data.modelUsed,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: ChatMessage = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : "Unknown error"}`,
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      sendMessage()
    }
  }

  const getSourceIcon = (type: string) => {
    switch (type) {
      case "analysis":
        return <TrendingUp className="h-3 w-3" />
      case "news":
        return <Newspaper className="h-3 w-3" />
      case "knowledge":
        return <FileText className="h-3 w-3" />
      default:
        return <Sparkles className="h-3 w-3" />
    }
  }

  const ChatContent = () => (
    <>
      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.length === 0 ? (
          <div className="text-center text-muted-foreground py-8">
            <Bot className="h-12 w-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-medium">Stock Radar AI Assistant</p>
            <p className="text-sm mt-2">
              Ask me anything about stocks, analyses, or market conditions.
            </p>
            <div className="mt-4 space-y-2">
              <p className="text-xs">Try asking:</p>
              <div className="flex flex-wrap gap-2 justify-center">
                {[
                  "What's the latest analysis for RELIANCE?",
                  "Explain the current market sentiment",
                  "Compare TCS and INFY",
                ].map((suggestion) => (
                  <Button
                    key={suggestion}
                    variant="outline"
                    size="sm"
                    className="text-xs"
                    onClick={() => {
                      setInput(suggestion)
                      inputRef.current?.focus()
                    }}
                  >
                    {suggestion}
                  </Button>
                ))}
              </div>
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div
              key={message.id}
              className={cn(
                "flex gap-3",
                message.role === "user" ? "justify-end" : "justify-start"
              )}
            >
              {message.role === "assistant" && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Bot className="h-4 w-4 text-primary" />
                </div>
              )}

              <div
                className={cn(
                  "max-w-[80%] rounded-lg p-3",
                  message.role === "user"
                    ? "bg-primary text-primary-foreground"
                    : "bg-muted"
                )}
              >
                <p className="text-sm whitespace-pre-wrap">{message.content}</p>

                {/* Sources Used */}
                {message.sourcesUsed && message.sourcesUsed.length > 0 && (
                  <div className="mt-3 pt-2 border-t border-border/50">
                    <p className="text-xs text-muted-foreground mb-2">
                      Sources used:
                    </p>
                    <div className="flex flex-wrap gap-1">
                      {message.sourcesUsed.slice(0, 5).map((source, idx) => (
                        <Badge
                          key={idx}
                          variant="secondary"
                          className="text-xs flex items-center gap-1"
                        >
                          {getSourceIcon(source.type)}
                          {source.type === "analysis" && source.symbol
                            ? `${source.symbol} analysis`
                            : source.type === "news" && source.headline
                              ? source.headline.substring(0, 20) + "..."
                              : source.type}
                        </Badge>
                      ))}
                    </div>
                  </div>
                )}

                {/* Stock Symbols */}
                {message.stockSymbols && message.stockSymbols.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {message.stockSymbols.map((symbol) => (
                      <Badge key={symbol} variant="outline" className="text-xs">
                        {symbol}
                      </Badge>
                    ))}
                  </div>
                )}

                {/* Metadata */}
                {message.role === "assistant" && message.processingTimeMs && (
                  <p className="text-xs text-muted-foreground mt-2">
                    {message.modelUsed} | {message.tokensUsed} tokens |{" "}
                    {message.processingTimeMs}ms
                  </p>
                )}
              </div>

              {message.role === "user" && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary flex items-center justify-center">
                  <User className="h-4 w-4 text-primary-foreground" />
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="flex gap-3">
            <div className="flex-shrink-0 w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
              <Bot className="h-4 w-4 text-primary" />
            </div>
            <div className="bg-muted rounded-lg p-3">
              <div className="flex items-center gap-2">
                <Loader2 className="h-4 w-4 animate-spin" />
                <span className="text-sm text-muted-foreground">
                  Analyzing with RAG...
                </span>
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t">
        <div className="flex gap-2">
          <Input
            ref={inputRef}
            placeholder="Ask about stocks, analyses, market conditions..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            disabled={isLoading}
            className="flex-1"
          />
          <Button onClick={sendMessage} disabled={isLoading || !input.trim()}>
            {isLoading ? (
              <Loader2 className="h-4 w-4 animate-spin" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </div>
        {defaultSymbol && (
          <p className="text-xs text-muted-foreground mt-2">
            Focused on: {defaultSymbol}
          </p>
        )}
      </div>
    </>
  )

  // Floating chat bubble version
  if (isFloating) {
    return (
      <>
        {/* Chat Button - Bottom Left */}
        {!isOpen && (
          <Button
            onClick={() => setIsOpen(true)}
            className="fixed bottom-6 left-6 ml-64 h-14 w-14 rounded-full shadow-lg z-50"
            size="icon"
          >
            <MessageCircle className="h-6 w-6" />
          </Button>
        )}

        {/* Chat Window - Bottom Left */}
        {isOpen && (
          <Card
            className={cn(
              "fixed z-50 flex flex-col shadow-2xl transition-all duration-200",
              isExpanded
                ? "bottom-0 left-64 right-0 w-auto h-full rounded-none"
                : "bottom-6 left-6 ml-64 w-[400px] h-[600px] rounded-xl",
              className
            )}
          >
            <CardHeader className="flex-shrink-0 flex flex-row items-center justify-between py-3 px-4 border-b">
              <div className="flex items-center gap-2">
                <div className="w-8 h-8 rounded-full bg-primary/10 flex items-center justify-center">
                  <Sparkles className="h-4 w-4 text-primary" />
                </div>
                <div>
                  <CardTitle className="text-sm">AI Assistant</CardTitle>
                  <CardDescription className="text-xs">
                    RAG-powered stock chat
                  </CardDescription>
                </div>
              </div>
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => setIsExpanded(!isExpanded)}
                >
                  {isExpanded ? (
                    <Minimize2 className="h-4 w-4" />
                  ) : (
                    <Maximize2 className="h-4 w-4" />
                  )}
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  className="h-8 w-8"
                  onClick={() => setIsOpen(false)}
                >
                  <X className="h-4 w-4" />
                </Button>
              </div>
            </CardHeader>
            <ChatContent />
          </Card>
        )}
      </>
    )
  }

  // Full page version
  return (
    <Card className={cn("flex flex-col h-full", className)}>
      <CardHeader className="flex-shrink-0 border-b">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
            <Sparkles className="h-5 w-5 text-primary" />
          </div>
          <div>
            <CardTitle>AI Chat Assistant</CardTitle>
            <CardDescription>
              RAG-powered conversations about stocks and market analysis
            </CardDescription>
          </div>
        </div>
      </CardHeader>
      <CardContent className="flex-1 flex flex-col p-0 overflow-hidden">
        <ChatContent />
      </CardContent>
    </Card>
  )
}

// Floating chat wrapper component for easy integration
export function FloatingChat({ defaultSymbol }: { defaultSymbol?: string }) {
  return <ChatAssistant isFloating defaultSymbol={defaultSymbol} />
}
