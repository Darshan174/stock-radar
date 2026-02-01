import { NextRequest, NextResponse } from "next/server"
import { exec } from "child_process"
import { promisify } from "util"
import path from "path"
import { withX402 } from "@/lib/x402-enforcer"
import { validateSymbol, validateNumericParam } from "@/lib/input-validation"

const execAsync = promisify(exec)

async function handleNewsImpact(request: NextRequest): Promise<NextResponse> {
  const searchParams = request.nextUrl.searchParams

  const symbolCheck = validateSymbol(searchParams.get("symbol"))
  if (!symbolCheck.valid) {
    return NextResponse.json({ error: symbolCheck.error }, { status: 400 })
  }

  const daysCheck = validateNumericParam(searchParams.get("days"), "7", "days", 1, 90)
  if (!daysCheck.valid) {
    return NextResponse.json({ error: daysCheck.error }, { status: 400 })
  }

  const symbol = symbolCheck.value
  const days = daysCheck.value

  const projectPath = path.join(process.cwd(), "..")
  const { stdout, stderr } = await execAsync(
    `cd "${projectPath}" && python3 -c "
import sys, os, json
sys.path.append('src')
from agents.fetcher import StockFetcher
from agents.storage import StockStorage
from agents.rag_retriever import RAGRetriever
from agents.analyzer import StockAnalyzer
from datetime import datetime, timedelta

fetcher = StockFetcher()
storage = StockStorage()
rag_retriever = RAGRetriever()
analyzer = StockAnalyzer()

symbol = os.environ['SR_SYMBOL']
days = int(os.environ['SR_DAYS'])

quote = fetcher.get_quote(symbol)
if not quote:
    print(json.dumps({'error': 'Quote not found'}))
    sys.exit(1)

news_data = storage.get_news(symbol, days=days)
if not news_data or len(news_data) == 0:
    print(json.dumps({'error': 'No news data available'}))
    sys.exit(1)

news_sentiment = storage.get_news_sentiment(symbol, days=days)
if news_sentiment:
    avg_sentiment = news_sentiment.get('avg_sentiment', 0)
    sentiment_label = 'bullish' if avg_sentiment > 0.1 else 'bearish' if avg_sentiment < -0.1 else 'neutral'
else:
    avg_sentiment = 0
    sentiment_label = 'neutral'

recent_news = news_data[:5]
news_items = []
for item in recent_news:
    news_items.append({
        'title': item.get('title', ''),
        'source': item.get('source', ''),
        'published_at': item.get('published_at', ''),
        'sentiment': 'positive' if item.get('sentiment', 0) > 0 else 'negative' if item.get('sentiment', 0) < 0 else 'neutral'
    })

sector = fetcher.get_stock_info(symbol).get('sector', 'Unknown')
sector_context = rag_retriever.get_sector_sentiment_context(sector) if sector != 'Unknown' else None

key_topics = []
if news_items:
    all_titles = ' '.join([item['title'] for item in news_items])
    keywords = ['earnings', 'guidance', 'product', 'merger', 'acquisition', 'regulation', 'lawsuit', 'partnership', 'launch', 'recall', 'outlook', 'revenue', 'profit', 'dividend', 'stock split']
    for keyword in keywords:
        if keyword.lower() in all_titles.lower():
            key_topics.append(keyword)

price_impact = 'neutral'
if avg_sentiment > 0.15:
    price_impact = 'positive'
elif avg_sentiment < -0.15:
    price_impact = 'negative'

result = {
    'symbol': symbol,
    'news_impact': {
        'overall_sentiment': sentiment_label,
        'sentiment_score': round(avg_sentiment, 3),
        'expected_price_impact': price_impact,
        'key_topics': key_topics,
        'news_count': len(news_data),
        'recent_news': news_items
    },
    'sector_context': {
        'sector': sector,
        'sector_sentiment': sector_context.get('sentiment', 'neutral') if sector_context else 'neutral'
    } if sector_context else None,
    'timestamp': datetime.utcnow().isoformat()
}
print(json.dumps(result))
"`,
    {
      timeout: 60000,
      env: { ...process.env, SR_SYMBOL: symbol, SR_DAYS: String(days) },
    }
  )

  if (stderr && stderr.includes("Error")) {
    console.error("News impact endpoint error:", stderr)
    return NextResponse.json({ error: "Failed to generate news impact summary" }, { status: 500 })
  }

  const result = JSON.parse(stdout.trim())
  return NextResponse.json(result)
}

export async function GET(request: NextRequest) {
  return withX402(request, "/api/agent/news-impact", handleNewsImpact)
}
