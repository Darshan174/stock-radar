"""Research Radar Agents

Core agents for competitor intelligence:
- crawler: Firecrawl-based web scraping
- storage: Supabase persistence and vector embeddings
- analyzer: Ollama-powered change detection (FREE, runs locally!)
- alerts: Slack notifications
"""

from .crawler import CompetitorCrawler
from .storage import SupabaseStorage, OllamaEmbeddings
from .analyzer import OllamaAnalyzer
from .alerts import SlackNotifier

__all__ = [
    "CompetitorCrawler",
    "SupabaseStorage",
    "OllamaEmbeddings",
    "OllamaAnalyzer",
    "SlackNotifier"
]
