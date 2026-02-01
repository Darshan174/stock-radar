import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol } from "@/lib/input-validation"

const execAsync = promisify(exec)

export async function handleSocialSentiment(request: NextRequest): Promise<NextResponse> {
  const searchParams = request.nextUrl.searchParams

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const symbol = symbolCheck.value

  const projectPath = path.join(process.cwd(), "..")
  const { stdout, stderr } = await execAsync(
    `cd "${projectPath}" && python3 -c "
import sys, os, json
sys.path.append('src')
from agents.fetcher import StockFetcher
from datetime import datetime

fetcher = StockFetcher()

symbol = os.environ['SR_SYMBOL']

reddit_sentiment = fetcher.get_reddit_sentiment(symbol)
social_sentiment = fetcher.get_social_sentiment(symbol)

result = {
    'symbol': symbol,
    'reddit': {
        'mentions': reddit_sentiment.get('mentions', 0),
        'rank': reddit_sentiment.get('rank', 0),
        'sentiment': reddit_sentiment.get('sentiment', 'neutral'),
        'subreddits': reddit_sentiment.get('subreddits', []),
        'fetched_at': reddit_sentiment.get('fetched_at', datetime.utcnow().isoformat())
    },
    'social_sentiment': {
        'overall': social_sentiment.get('overall', 'neutral'),
        'sources': social_sentiment.get('sources', {})
    },
    'timestamp': datetime.utcnow().isoformat()
}
print(json.dumps(result))
"`,
    {
      timeout: 60000,
      env: { ...process.env, SR_SYMBOL: symbol },
    }
  )

  if (stderr && stderr.includes("Error")) {
    console.error("Social sentiment endpoint error:", stderr)
    return NextResponse.json({ error: "Failed to fetch social sentiment" }, { status: 500 })
  }

  const result = JSON.parse(stdout.trim())
  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  return withX402(request, "/api/agent/social-sentiment", handleSocialSentiment)
}
