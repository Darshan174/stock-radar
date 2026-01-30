"use client"

import { ChatAssistant } from "@/components/chat-assistant"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Sparkles, Database, Brain, MessageSquare, TrendingUp, Newspaper, FileText } from "lucide-react"
import { RAGBadge } from "@/components/rag-badge"

export default function ChatPage() {
  return (
    <div className="p-6 h-screen flex flex-col">
      <div className="mb-6">
        <div className="flex items-center gap-3">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-primary" />
            AI Chat Assistant
          </h1>
          <RAGBadge isActive={true} size="lg" />
        </div>
        <p className="text-muted-foreground mt-1">
          RAG-powered conversations about your stocks and market analysis
        </p>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0">
        {/* Chat Interface */}
        <div className="lg:col-span-3 min-h-0">
          <ChatAssistant className="h-full" />
        </div>

        {/* Info Sidebar */}
        <div className="space-y-4 overflow-y-auto">
          {/* How It Works */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Brain className="h-4 w-4" />
                How RAG Works
              </CardTitle>
            </CardHeader>
            <CardContent className="text-sm text-muted-foreground space-y-3">
              <div className="flex gap-2">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium">
                  1
                </div>
                <p>Your question is converted to a semantic embedding</p>
              </div>
              <div className="flex gap-2">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium">
                  2
                </div>
                <p>We search our database for similar content</p>
              </div>
              <div className="flex gap-2">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium">
                  3
                </div>
                <p>Relevant context is retrieved and ranked</p>
              </div>
              <div className="flex gap-2">
                <div className="flex-shrink-0 w-6 h-6 rounded-full bg-primary/10 flex items-center justify-center text-xs font-medium">
                  4
                </div>
                <p>AI generates an informed response using the context</p>
              </div>
            </CardContent>
          </Card>

          {/* Data Sources */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <Database className="h-4 w-4" />
                Data Sources
              </CardTitle>
              <CardDescription className="text-xs">
                Context retrieved from:
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              <div className="flex items-center gap-2 text-sm">
                <TrendingUp className="h-4 w-4 text-green-500" />
                <span>Past Analyses & Signals</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <Newspaper className="h-4 w-4 text-blue-500" />
                <span>News Articles</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <FileText className="h-4 w-4 text-purple-500" />
                <span>Knowledge Base</span>
              </div>
              <div className="flex items-center gap-2 text-sm">
                <MessageSquare className="h-4 w-4 text-orange-500" />
                <span>Chat History</span>
              </div>
            </CardContent>
          </Card>

          {/* Example Questions */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm flex items-center gap-2">
                <MessageSquare className="h-4 w-4" />
                Example Questions
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2">
              {[
                "What was the last signal for RELIANCE?",
                "Why did TCS get a sell signal?",
                "Compare the momentum of INFY and WIPRO",
                "What news is affecting IT stocks?",
                "Explain the RSI indicator",
                "Which stocks have buy signals today?",
              ].map((question, idx) => (
                <div
                  key={idx}
                  className="text-xs text-muted-foreground p-2 rounded bg-muted/50 hover:bg-muted cursor-pointer transition-colors"
                >
                  "{question}"
                </div>
              ))}
            </CardContent>
          </Card>

          {/* Features */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-sm">Features</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-1">
                <Badge variant="secondary" className="text-xs">Semantic Search</Badge>
                <Badge variant="secondary" className="text-xs">Vector DB</Badge>
                <Badge variant="secondary" className="text-xs">Multi-source</Badge>
                <Badge variant="secondary" className="text-xs">Context-aware</Badge>
                <Badge variant="secondary" className="text-xs">Real-time</Badge>
                <Badge variant="secondary" className="text-xs">Historical</Badge>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
