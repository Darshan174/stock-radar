# Quick Start Guide

Get Research Radar running in 5 minutes.

## Prerequisites

- Python 3.9+
- Ollama running locally (`ollama serve`)
- Supabase project
- API keys for: Firecrawl, Claude, Slack, Trigger.dev

## Step 1: Setup Files

Your project structure:
```
/tmp/claude/research-radar/
├── src/agents/           # Core modules
├── triggers/weekly.py    # Orchestration
├── migrations/           # Database schema
├── setup.py             # Verification script
├── requirements.txt     # Dependencies
└── .env.example        # Configuration template
```

## Step 2: Install Dependencies

```bash
cd /tmp/claude/research-radar
pip install -r requirements.txt
```

## Step 3: Configure Environment

```bash
cp .env.example .env
# Edit .env with your API keys
nano .env
```

Required:
- `FIRECRAWL_API_KEY` - Get from firecrawl.dev
- `SUPABASE_URL` + `SUPABASE_KEY` - From Supabase dashboard
- `ANTHROPIC_API_KEY` - Claude API key
- `SLACK_BOT_TOKEN` + `SLACK_CHANNEL_ID` - From Slack workspace
- `OLLAMA_API_URL` - Usually `http://localhost:11434`

## Step 4: Start Ollama

```bash
# In a new terminal
ollama serve

# In another terminal, pull the embedding model
ollama pull nomic-embed-text
```

## Step 5: Setup Database

1. Go to your Supabase project → SQL editor
2. Copy and paste the contents of `migrations/001_initial_schema.sql`
3. Run the SQL

## Step 6: Verify Setup

```bash
python setup.py
```

This checks:
- ✓ Environment variables
- ✓ Python dependencies
- ✓ Ollama connectivity
- ✓ Database schema
- ✓ API connections

## Step 7: Add Competitors

```python
from src.agents.storage import SupabaseStorage

storage = SupabaseStorage()

# Add competitors to track
storage.add_competitor("Notion", "https://www.notion.so")
storage.add_competitor("Linear", "https://www.linear.app")
storage.add_competitor("Confluence", "https://www.atlassian.com/software/confluence")

# Verify
competitors = storage.list_competitors()
print(f"Tracking {len(competitors)} competitors")
```

## Step 8: Run Manual Test

```bash
# Test all components
python examples/test_manual.py

# Run full workflow
python triggers/weekly.py --run-id manual-test-1
```

## Step 9: Schedule with Trigger.dev

1. Deploy your project (or set up a webhook server)
2. Create a scheduled task in Trigger.dev
3. Configure cron: `0 0 * * 0` (weekly, Sunday midnight)
4. Set webhook URL to your deployment

## Next Steps

- Monitor the Slack channel for alerts
- Check Supabase dashboard for crawls and changes
- Adjust importance thresholds in `analyzer.py`
- Add more competitors as needed

## Troubleshooting

### Ollama not connecting
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Pull the model
ollama pull nomic-embed-text
```

### Supabase connection error
- Verify credentials in `.env`
- Check Supabase project status
- Ensure RLS policies allow your key

### Slack message failed
- Verify bot token and channel ID
- Ensure bot has message permissions
- Check channel is public or bot is invited

### Claude API error
- Verify API key is correct
- Check quota in Anthropic dashboard
- Ensure model name is correct (claude-opus-4-5-20251101)

## Files Reference

| File | Purpose |
|------|---------|
| `src/agents/crawler.py` | Firecrawl web scraping |
| `src/agents/storage.py` | Supabase + Ollama embeddings |
| `src/agents/analyzer.py` | Claude change detection |
| `src/agents/alerts.py` | Slack notifications |
| `triggers/weekly.py` | Workflow orchestration |
| `migrations/001_initial_schema.sql` | Database setup |
| `setup.py` | Environment verification |
| `examples/test_manual.py` | Component testing |

## Example Output

After running `triggers/weekly.py`:

```
2026-01-14 21:45:00 - research_radar - INFO - Starting weekly crawl run_id=manual-test-1
2026-01-14 21:45:05 - research_radar.crawler - INFO - Successfully scraped https://www.notion.so
2026-01-14 21:45:10 - research_radar.storage - INFO - Stored crawl with embedding
2026-01-14 21:45:15 - research_radar.analyzer - INFO - Found 2 changes in Notion
2026-01-14 21:45:20 - research_radar.alerts - INFO - Sent alert to Slack (ts=1234567890.001)
```

## Key Concepts

**Crawl**: Snapshot of a competitor's website at a point in time
**Change**: Meaningful difference detected between two crawls
**Embedding**: 768-dimensional vector representation of crawl content
**Alert**: Slack message notifying about a change
**Importance**: high/mid/low prioritization of changes

## Performance Expectations

- **Crawl**: 10-30s per URL
- **Embedding**: 500ms per crawl
- **Analysis**: 5-10s per competitor
- **Slack**: 1-2s per message
- **Total**: ~30-60s for 3 competitors

Enjoy tracking your competitors! 🎯
