# Research Radar 🎯

**Competitor Intelligence System** - Automated weekly web scraping, change detection, and Slack alerts.

Crawl competitors → Detect changes → Get notified. Built with Firecrawl, Supabase, Claude, and Trigger.dev.

## 🎯 Features

- **Automated Crawling**: Weekly scrapes of competitor websites via Firecrawl
- **Smart Change Detection**: Claude AI analyzes differences and categorizes them
- **Vector Embeddings**: Ollama embeddings for semantic search of past crawls
- **Slack Alerts**: Rich notifications with importance levels (high/mid/low)
- **Persistent Storage**: All crawls, changes, and alerts stored in Supabase
- **Scheduled Execution**: Trigger.dev webhook integration for weekly jobs

## 📊 How It Works

```
1. CRAWL (Firecrawl)
   └─> Scrape competitor website → extract markdown/HTML

2. EMBED (Ollama)
   └─> Generate vector embeddings → store in pgvector

3. STORE (Supabase)
   └─> Persist crawl data with embeddings

4. ANALYZE (Claude)
   └─> Compare old vs new crawl → detect meaningful changes
   └─> Categorize: pricing, feature, hiring, partnership, other

5. ALERT (Slack)
   └─> Send formatted notification with importance level
   └─> Track which changes were alerted
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone or create project
mkdir research-radar && cd research-radar

# Copy environment template
cp .env.example .env

# Edit .env with your credentials
nano .env
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Start Ollama

```bash
# In a separate terminal
ollama serve

# In another terminal, pull the embedding model
ollama pull nomic-embed-text
```

### 4. Run Setup Verification

```bash
python setup.py
```

### 5. Create Database Schema

Run the migration SQL in your Supabase project:
```bash
# Copy and paste migrations/001_initial_schema.sql
# into your Supabase SQL editor
```

### 6. Add Competitors

```python
from src.agents.storage import SupabaseStorage

storage = SupabaseStorage()
storage.add_competitor("Notion", "https://www.notion.so")
storage.add_competitor("Linear", "https://www.linear.app")
storage.add_competitor("Confluence", "https://www.atlassian.com/software/confluence")
```

### 7. Run Manual Crawl

```bash
python triggers/weekly.py --run-id manual-test-1
```

### 8. Schedule with Trigger.dev

Configure a webhook to hit your deployment with this cron schedule:
```
0 0 * * 0  # Every Sunday at midnight
```

## 📦 Project Structure

```
research-radar/
├── src/
│   └── agents/              # Core agent modules
│       ├── crawler.py       # Firecrawl integration
│       ├── storage.py       # Supabase + Ollama
│       ├── analyzer.py      # Claude change detection
│       └── alerts.py        # Slack notifications
├── triggers/
│   └── weekly.py            # Trigger.dev orchestration
├── migrations/
│   └── 001_initial_schema.sql
├── requirements.txt
├── setup.py
├── .env.example
└── README.md
```

## 🔧 Configuration

### Environment Variables

```bash
# Firecrawl
FIRECRAWL_API_KEY=your_key

# Supabase
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your_anon_key

# Claude
ANTHROPIC_API_KEY=your_key

# Slack
SLACK_BOT_TOKEN=xoxb-...
SLACK_CHANNEL_ID=C...

# Ollama (default: localhost:11434)
OLLAMA_API_URL=http://localhost:11434

# Trigger.dev
TRIGGER_API_KEY=your_key
```

## 🤖 Agents

### CompetitorCrawler
Scrapes competitor websites using Firecrawl API.
```python
from src.agents.crawler import CompetitorCrawler

crawler = CompetitorCrawler()
result = crawler.scrape_url("https://example.com")
results = crawler.scrape_multiple(["url1", "url2"])
```

### SupabaseStorage
Persists data and manages vector embeddings.
```python
from src.agents.storage import SupabaseStorage

storage = SupabaseStorage()
storage.add_competitor("Notion", "notion.so")
storage.store_crawl(competitor_id, markdown, html, url)
storage.semantic_search("pricing changes")
storage.store_change(crawl_id, "pricing", "New Pro plan", "high")
```

### ChangeAnalyzer
Detects and categorizes changes using Claude.
```python
from src.agents.analyzer import ChangeAnalyzer

analyzer = ChangeAnalyzer()
result = analyzer.analyze_changes("Notion", old_markdown, new_markdown)
# Returns: {has_changes, changes[], similarity}
```

### SlackNotifier
Sends formatted notifications to Slack.
```python
from src.agents.alerts import SlackNotifier

notifier = SlackNotifier()
response = notifier.send_change_alert(competitor, change)
# Returns: {success, slack_ts, error}
```

## 📊 Database Schema

### competitors
Track companies to monitor.
```sql
id, name, url, created_at, updated_at
```

### crawls
Store web scrape results with embeddings.
```sql
id, competitor_id, markdown, html, url, embedding, crawl_date
```

### changes
Detected differences between crawls.
```sql
id, crawl_id, type, summary, importance, detected_at
```

### alerts
Slack notifications sent.
```sql
id, change_id, slack_ts, alerted_at
```

## 🔍 Example Workflow

```python
# 1. Get latest crawl for competitor
old_crawl = storage.get_latest_crawl(competitor_id)

# 2. Scrape competitor's website
new_crawl = crawler.scrape_url(competitor.url)

# 3. Store new crawl with embeddings
crawl_record = storage.store_crawl(
    competitor_id,
    new_crawl["markdown"],
    new_crawl["html"],
    new_crawl["url"]
)

# 4. Analyze what changed
analysis = analyzer.analyze_changes(
    competitor.name,
    old_crawl["markdown"],
    new_crawl["markdown"]
)

# 5. Store detected changes
if analysis["has_changes"]:
    for change in analysis["changes"]:
        storage.store_change(
            crawl_record["id"],
            change["type"],
            change["summary"],
            change["importance"]
        )

# 6. Send Slack alerts
for change in changes:
    response = notifier.send_change_alert(competitor, change)
    storage.record_alert(change["id"], response["slack_ts"])
```

## 🧪 Testing

### Test Individual Components

```bash
# Test crawler
python src/agents/crawler.py

# Test storage
python src/agents/storage.py

# Test analyzer
python src/agents/analyzer.py

# Test alerts
python src/agents/alerts.py
```

### Run Full Workflow

```bash
python triggers/weekly.py --run-id test-1
```

## 📝 API Reference

See docstrings in each module:
- `src/agents/crawler.py` - Firecrawl integration
- `src/agents/storage.py` - Supabase & embeddings
- `src/agents/analyzer.py` - Change detection
- `src/agents/alerts.py` - Slack notifications
- `triggers/weekly.py` - Orchestration

## 🚨 Error Handling

All components include:
- Retry logic with exponential backoff
- Comprehensive error logging
- Graceful failure handling
- Connection testing

## 📈 Performance

- Crawler: ~10-30s per URL (depends on page size)
- Embeddings: ~500ms per crawl (768-dim vectors)
- Analysis: ~5-10s per competitor (Claude API)
- Alerts: ~1-2s per Slack message

## 🔐 Security

- All API keys stored in `.env` (never committed)
- JWT tokens for Slack
- Supabase RLS policies (recommended)
- Rate limiting on all API calls

## 📞 Support

- Check logs for detailed error messages
- Run `python setup.py` to diagnose issues
- Verify all environment variables are set
- Ensure Ollama service is running

## 📄 License

MIT
