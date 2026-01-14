"""Research Radar Agents

Core agents for competitor intelligence:
- crawler: Firecrawl-based web scraping
- storage: Supabase persistence and vector embeddings
- analyzer: Claude-powered change detection
- alerts: Slack notifications
"""

from .crawler import CompetitorCrawler
from .storage import SupabaseStorage, OllamaEmbeddings
from .analyzer import ChangeAnalyzer
from .alerts import SlackNotifier

__all__ = [
    "CompetitorCrawler",
    "SupabaseStorage",
    "OllamaEmbeddings",
    "ChangeAnalyzer",
    "SlackNotifier"
]
