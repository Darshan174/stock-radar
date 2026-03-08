import Link from "next/link"
import { ArrowRight, MessageSquare, Search } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"

export default function ChatPage() {
  return (
    <div className="app-page">
      <div className="mx-auto max-w-3xl">
        <Card className="border-dashed">
          <CardHeader className="text-center">
            <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-full border border-cyan-200 bg-cyan-50 text-cyan-700 dark:border-cyan-500/20 dark:bg-cyan-500/10 dark:text-cyan-300">
              <MessageSquare className="h-6 w-6" />
            </div>
            <CardTitle>AI chat moved into Stocks</CardTitle>
            <CardDescription>
              Open chat from a stock that already has saved analysis history.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4 pb-8 text-center">
            <div className="rounded-lg border border-white/10 bg-black/40 p-4 text-sm text-muted-foreground">
              Stocks with analysis show an active chat button.
              Stocks without analysis show a disabled black-and-white chat button with an info tooltip.
            </div>
            <div className="flex flex-col items-center justify-center gap-3 sm:flex-row">
              <Button asChild>
                <Link href="/stocks">
                  <Search className="h-4 w-4" />
                  Open Watchlist
                </Link>
              </Button>
              <Button asChild variant="outline">
                <Link href="/signals">
                  View Signals
                  <ArrowRight className="h-4 w-4" />
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
