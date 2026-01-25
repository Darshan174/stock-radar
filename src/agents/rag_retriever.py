"""
Stock Radar - RAG (Retrieval-Augmented Generation) Retriever

This module provides intelligent context retrieval for AI-powered stock analysis.
It searches across multiple data sources to find relevant information for:
- Enhancing analysis prompts with historical context
- Finding similar technical setups and their outcomes
- Retrieving sector-wide sentiment and news
- Supporting conversational Q&A about stocks
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class RAGContext:
    """Container for retrieved RAG context."""
    query: str
    stock_symbol: Optional[str] = None

    # Retrieved content by source
    similar_analyses: List[Dict[str, Any]] = field(default_factory=list)
    similar_signals: List[Dict[str, Any]] = field(default_factory=list)
    relevant_news: List[Dict[str, Any]] = field(default_factory=list)
    knowledge_entries: List[Dict[str, Any]] = field(default_factory=list)
    similar_conversations: List[Dict[str, Any]] = field(default_factory=list)

    # Stock-specific context
    recent_analyses: List[Dict[str, Any]] = field(default_factory=list)
    recent_signals: List[Dict[str, Any]] = field(default_factory=list)
    technical_indicators: Optional[Dict[str, Any]] = None

    # Metadata
    total_results: int = 0
    retrieval_time_ms: int = 0
    sources_searched: List[str] = field(default_factory=list)


class RAGRetriever:
    """
    Intelligent retrieval system for stock analysis context.

    Provides methods to:
    - Find similar historical analyses
    - Discover patterns in past signals
    - Retrieve relevant news and knowledge
    - Build comprehensive context for LLM prompts
    """

    def __init__(self, storage=None):
        """
        Initialize RAG retriever.

        Args:
            storage: StockStorage instance (will create if not provided)
        """
        if storage is None:
            from agents.storage import StockStorage
            self.storage = StockStorage()
        else:
            self.storage = storage

        logger.info("RAGRetriever initialized")

    # -------------------------------------------------------------------------
    # Core Retrieval Methods
    # -------------------------------------------------------------------------

    def retrieve_context(
        self,
        query: str,
        stock_symbol: Optional[str] = None,
        user_id: Optional[str] = None,
        include_analyses: bool = True,
        include_signals: bool = True,
        include_news: bool = True,
        include_knowledge: bool = True,
        include_conversations: bool = False,
        max_results_per_source: int = 3,
        match_threshold: float = 0.5
    ) -> RAGContext:
        """
        Retrieve comprehensive RAG context for a query.

        Args:
            query: The search query or context description
            stock_symbol: Filter by specific stock (optional)
            user_id: User ID for personalized results (optional)
            include_analyses: Search past analyses
            include_signals: Search past signals
            include_news: Search news articles
            include_knowledge: Search knowledge base
            include_conversations: Search chat history
            max_results_per_source: Maximum results per source type
            match_threshold: Minimum similarity score

        Returns:
            RAGContext with all retrieved information
        """
        import time
        start_time = time.time()

        context = RAGContext(query=query, stock_symbol=stock_symbol)
        sources_searched = []

        try:
            # Search similar analyses
            if include_analyses:
                context.similar_analyses = self.storage.search_similar_analyses(
                    query=query,
                    stock_symbol=stock_symbol,
                    limit=max_results_per_source,
                    match_threshold=match_threshold
                )
                sources_searched.append("analyses")

            # Search similar signals
            if include_signals:
                context.similar_signals = self.storage.search_similar_signals(
                    query=query,
                    stock_symbol=stock_symbol,
                    limit=max_results_per_source,
                    match_threshold=match_threshold
                )
                sources_searched.append("signals")

            # Search relevant news
            if include_news:
                context.relevant_news = self.storage.search_news(
                    query=query,
                    limit=max_results_per_source,
                    match_threshold=match_threshold
                )
                sources_searched.append("news")

            # Search knowledge base
            if include_knowledge:
                context.knowledge_entries = self.storage.search_knowledge_base(
                    query=query,
                    user_id=user_id,
                    stock_symbols=[stock_symbol] if stock_symbol else None,
                    limit=max_results_per_source,
                    match_threshold=match_threshold
                )
                sources_searched.append("knowledge")

            # Search chat history
            if include_conversations:
                context.similar_conversations = self.storage.search_chat_history(
                    query=query,
                    user_id=user_id,
                    limit=max_results_per_source,
                    match_threshold=match_threshold
                )
                sources_searched.append("conversations")

            # Calculate totals
            context.total_results = (
                len(context.similar_analyses) +
                len(context.similar_signals) +
                len(context.relevant_news) +
                len(context.knowledge_entries) +
                len(context.similar_conversations)
            )
            context.retrieval_time_ms = int((time.time() - start_time) * 1000)
            context.sources_searched = sources_searched

            logger.info(
                f"RAG retrieval complete: {context.total_results} results "
                f"from {len(sources_searched)} sources in {context.retrieval_time_ms}ms"
            )

            return context

        except Exception as e:
            logger.error(f"Error in RAG retrieval: {str(e)}")
            return context

    def get_analysis_context(
        self,
        symbol: str,
        mode: str,
        current_indicators: Dict[str, Any],
        current_quote: Dict[str, Any],
        news: List[Dict[str, Any]] = None
    ) -> str:
        """
        Get RAG context specifically for analysis enhancement.

        This method builds a context string that can be injected into
        analysis prompts to provide historical perspective.

        Args:
            symbol: Stock symbol
            mode: Trading mode ('intraday' or 'longterm')
            current_indicators: Current technical indicators
            current_quote: Current price quote
            news: Recent news articles (optional)

        Returns:
            Formatted context string for prompt injection
        """
        # Build query from current state
        rsi = current_indicators.get('rsi_14', 'N/A')
        macd = current_indicators.get('macd', 'N/A')
        price = current_quote.get('price', 'N/A')
        change_pct = current_quote.get('change_percent', 0)

        query = (
            f"{symbol} {mode} analysis with RSI {rsi} MACD {macd} "
            f"price {price} change {change_pct}%"
        )

        # Retrieve context
        context = self.retrieve_context(
            query=query,
            stock_symbol=symbol,
            include_analyses=True,
            include_signals=True,
            include_news=True,
            include_knowledge=False,
            include_conversations=False,
            max_results_per_source=3,
            match_threshold=0.4
        )

        # Format context for prompt
        context_parts = []

        # Add similar past analyses
        if context.similar_analyses:
            context_parts.append("SIMILAR PAST ANALYSES:")
            for i, analysis in enumerate(context.similar_analyses[:3], 1):
                signal = analysis.get('signal', 'N/A')
                confidence = analysis.get('confidence', 0)
                reasoning = analysis.get('reasoning', '')[:200]
                created = analysis.get('created_at', '')[:10]
                similarity = analysis.get('similarity', 0)

                context_parts.append(
                    f"{i}. [{created}] Signal: {signal} (conf: {confidence:.0%}, "
                    f"sim: {similarity:.0%})\n   {reasoning}"
                )

        # Add similar signals and their outcomes
        if context.similar_signals:
            context_parts.append("\nSIMILAR PAST SIGNALS:")
            for i, signal in enumerate(context.similar_signals[:3], 1):
                sig_type = signal.get('signal', 'N/A')
                price_at = signal.get('price_at_signal', 'N/A')
                reason = signal.get('reason', '')[:150]
                importance = signal.get('importance', 'N/A')
                similarity = signal.get('similarity', 0)

                context_parts.append(
                    f"{i}. {sig_type.upper()} at {price_at} ({importance}) "
                    f"[sim: {similarity:.0%}]\n   {reason}"
                )

        # Add relevant news context
        if context.relevant_news:
            context_parts.append("\nRELEVANT NEWS CONTEXT:")
            for i, news_item in enumerate(context.relevant_news[:3], 1):
                headline = news_item.get('headline', '')
                sentiment = news_item.get('sentiment_label', 'neutral')
                similarity = news_item.get('similarity', 0)

                context_parts.append(
                    f"{i}. [{sentiment}] {headline} (sim: {similarity:.0%})"
                )

        if not context_parts:
            return ""

        return "\n\nHISTORICAL CONTEXT (from RAG retrieval):\n" + "\n".join(context_parts)

    def find_similar_technical_setups(
        self,
        indicators: Dict[str, Any],
        stock_symbol: Optional[str] = None,
        mode: str = "intraday",
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find past instances with similar technical setups.

        Args:
            indicators: Current technical indicators
            stock_symbol: Filter by specific stock (optional)
            mode: Trading mode
            limit: Maximum results

        Returns:
            List of similar setups with their outcomes
        """
        # Build query from indicators
        rsi = indicators.get('rsi_14', 50)
        macd = indicators.get('macd', 0)
        macd_signal = indicators.get('macd_signal', 0)
        price_vs_sma = indicators.get('price_vs_sma20_pct', 0)

        # Describe the setup
        rsi_desc = "oversold" if rsi < 30 else "overbought" if rsi > 70 else "neutral"
        macd_desc = "bullish crossover" if macd > macd_signal else "bearish crossover"
        trend_desc = "above" if price_vs_sma > 0 else "below"

        query = (
            f"{mode} trading setup with RSI {rsi_desc} ({rsi}), "
            f"{macd_desc}, price {trend_desc} SMA20 by {abs(price_vs_sma):.1f}%"
        )

        return self.storage.search_similar_analyses(
            query=query,
            stock_symbol=stock_symbol,
            mode=mode,
            limit=limit,
            match_threshold=0.4
        )

    def get_sector_sentiment_context(
        self,
        symbol: str,
        sector: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Retrieve recent news/sentiment for the sector.

        Args:
            symbol: Stock symbol to find sector
            sector: Sector name (optional, will be looked up)

        Returns:
            Dictionary with sector sentiment data
        """
        # Get stock info to find sector if not provided
        if not sector:
            stock = self.storage.get_stock_by_symbol(symbol)
            sector = stock.get('sector') if stock else None

        if not sector:
            return {"sector": "Unknown", "news": [], "sentiment_summary": "No sector data"}

        # Search for sector-wide news
        query = f"{sector} sector market news sentiment outlook"
        news = self.storage.search_news(
            query=query,
            limit=5,
            match_threshold=0.3
        )

        # Calculate sentiment summary
        if news:
            sentiments = [n.get('sentiment_label', 'neutral') for n in news]
            positive = sentiments.count('positive')
            negative = sentiments.count('negative')

            if positive > negative:
                sentiment_summary = f"Sector sentiment: Positive ({positive}/{len(sentiments)} positive)"
            elif negative > positive:
                sentiment_summary = f"Sector sentiment: Negative ({negative}/{len(sentiments)} negative)"
            else:
                sentiment_summary = f"Sector sentiment: Mixed ({positive} positive, {negative} negative)"
        else:
            sentiment_summary = "No recent sector news found"

        return {
            "sector": sector,
            "news": news,
            "sentiment_summary": sentiment_summary,
            "positive_count": news.count('positive') if news else 0,
            "negative_count": news.count('negative') if news else 0
        }

    def find_correlated_stock_signals(
        self,
        symbol: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find recent signals from potentially correlated stocks.

        This helps identify if similar stocks are showing the same patterns.

        Args:
            symbol: Stock symbol to find correlations for
            limit: Maximum results

        Returns:
            List of signals from related stocks
        """
        # Get stock info
        stock = self.storage.get_stock_by_symbol(symbol)
        if not stock:
            return []

        sector = stock.get('sector')
        if not sector:
            return []

        # Search for signals in the same sector
        query = f"trading signal {sector} sector stock market"

        signals = self.storage.search_similar_signals(
            query=query,
            limit=limit * 2,  # Get more to filter out the target stock
            match_threshold=0.3
        )

        # Filter out the target stock's own signals
        correlated_signals = [
            s for s in signals
            if s.get('symbol', '').upper() != symbol.upper()
        ][:limit]

        return correlated_signals

    # -------------------------------------------------------------------------
    # Context Formatting Methods
    # -------------------------------------------------------------------------

    def format_context_for_prompt(
        self,
        context: RAGContext,
        max_length: int = 2000
    ) -> str:
        """
        Format RAG context into a string suitable for LLM prompts.

        Args:
            context: RAGContext object
            max_length: Maximum character length

        Returns:
            Formatted context string
        """
        parts = []

        # Add similar analyses
        if context.similar_analyses:
            parts.append("## Historical Analyses (Similar Patterns)")
            for i, analysis in enumerate(context.similar_analyses[:3], 1):
                signal = analysis.get('signal', 'N/A')
                confidence = analysis.get('confidence', 0)
                reasoning = analysis.get('reasoning', '')[:150]
                similarity = analysis.get('similarity', 0)

                parts.append(
                    f"{i}. Signal: {signal} (conf: {confidence:.0%}) [similarity: {similarity:.0%}]\n"
                    f"   Reasoning: {reasoning}..."
                )

        # Add relevant news
        if context.relevant_news:
            parts.append("\n## Relevant News Context")
            for i, news in enumerate(context.relevant_news[:3], 1):
                headline = news.get('headline', '')
                sentiment = news.get('sentiment_label', 'neutral')
                parts.append(f"{i}. [{sentiment.upper()}] {headline}")

        # Add knowledge base entries
        if context.knowledge_entries:
            parts.append("\n## Knowledge Base")
            for i, kb in enumerate(context.knowledge_entries[:2], 1):
                title = kb.get('title', '')
                content = kb.get('content', '')[:200]
                parts.append(f"{i}. {title}\n   {content}...")

        # Add similar signals
        if context.similar_signals:
            parts.append("\n## Past Signals (Similar Context)")
            for i, signal in enumerate(context.similar_signals[:3], 1):
                sig = signal.get('signal', 'N/A')
                price = signal.get('price_at_signal', 'N/A')
                importance = signal.get('importance', 'medium')
                parts.append(f"{i}. {sig.upper()} at {price} ({importance})")

        result = "\n".join(parts)

        # Truncate if too long
        if len(result) > max_length:
            result = result[:max_length-3] + "..."

        return result

    def get_context_summary(self, context: RAGContext) -> Dict[str, Any]:
        """
        Get a summary of retrieved context for logging/display.

        Args:
            context: RAGContext object

        Returns:
            Summary dictionary
        """
        return {
            "query": context.query[:50] + "..." if len(context.query) > 50 else context.query,
            "stock_symbol": context.stock_symbol,
            "total_results": context.total_results,
            "retrieval_time_ms": context.retrieval_time_ms,
            "sources_searched": context.sources_searched,
            "results_by_source": {
                "analyses": len(context.similar_analyses),
                "signals": len(context.similar_signals),
                "news": len(context.relevant_news),
                "knowledge": len(context.knowledge_entries),
                "conversations": len(context.similar_conversations)
            }
        }


# =============================================================================
# Convenience Functions
# =============================================================================

_retriever_instance: Optional[RAGRetriever] = None


def get_retriever() -> RAGRetriever:
    """Get singleton RAGRetriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = RAGRetriever()
    return _retriever_instance


def retrieve_analysis_context(
    symbol: str,
    mode: str,
    indicators: Dict[str, Any],
    quote: Dict[str, Any],
    news: List[Dict[str, Any]] = None
) -> str:
    """
    Convenience function to get RAG context for analysis.

    Args:
        symbol: Stock symbol
        mode: Trading mode
        indicators: Technical indicators
        quote: Price quote
        news: Recent news (optional)

    Returns:
        Formatted context string for prompt injection
    """
    return get_retriever().get_analysis_context(
        symbol=symbol,
        mode=mode,
        current_indicators=indicators,
        current_quote=quote,
        news=news
    )


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    print("Testing RAG Retriever")
    print("=" * 50)

    retriever = RAGRetriever()

    # Test context retrieval
    print("\n1. Testing general context retrieval...")
    context = retriever.retrieve_context(
        query="bullish momentum RSI oversold MACD crossover",
        include_analyses=True,
        include_signals=True,
        include_news=True,
        max_results_per_source=2
    )

    summary = retriever.get_context_summary(context)
    print(f"   Query: {summary['query']}")
    print(f"   Total results: {summary['total_results']}")
    print(f"   Results by source: {summary['results_by_source']}")
    print(f"   Retrieval time: {summary['retrieval_time_ms']}ms")

    # Test analysis context
    print("\n2. Testing analysis context generation...")
    test_indicators = {
        "rsi_14": 35,
        "macd": 2.5,
        "macd_signal": 1.8,
        "price_vs_sma20_pct": -1.5
    }
    test_quote = {
        "price": 2850.0,
        "change_percent": -0.8
    }

    analysis_context = retriever.get_analysis_context(
        symbol="RELIANCE.NS",
        mode="intraday",
        current_indicators=test_indicators,
        current_quote=test_quote
    )

    if analysis_context:
        print(f"   Generated context ({len(analysis_context)} chars):")
        print(analysis_context[:500] + "..." if len(analysis_context) > 500 else analysis_context)
    else:
        print("   No context generated (may need to run some analyses first)")

    print("\n" + "=" * 50)
    print("RAG Retriever tests completed")
