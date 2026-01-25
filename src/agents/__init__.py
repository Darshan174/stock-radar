"""Stock Radar Agents

Core agents for stock market analysis:
- fetcher: Yahoo Finance stock data fetching
- storage: Supabase persistence with Cohere embeddings
- analyzer: LiteLLM-powered analysis (Groq -> Gemini -> Ollama)
- alerts: Slack & Telegram notifications
"""

from .fetcher import StockFetcher
from .storage import StockStorage, CohereEmbeddings
from .analyzer import StockAnalyzer, TradingMode, Signal
from .alerts import SlackNotifier, TelegramNotifier, NotificationManager

__all__ = [
    "StockFetcher",
    "StockStorage",
    "CohereEmbeddings",
    "StockAnalyzer",
    "TradingMode",
    "Signal",
    "SlackNotifier",
    "TelegramNotifier",
    "NotificationManager",
]
