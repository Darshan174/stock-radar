"""
Research Agent — multi-step financial research using ReAct.

Answers complex questions like:
  - "Compare AAPL and MSFT momentum, which is stronger?"
  - "Is TSLA near earnings? What's the risk?"
  - "Find stocks with RSI below 30 among FAANG"
  - "What happened last time AAPL had this RSI level?"

Uses existing service modules as tools.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.react_engine import AgentResult, ReActEngine, ToolDefinition

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Stock Radar's Research Agent — an expert financial analyst with access \
to real-time market data, technical indicators, fundamentals, news, and \
historical analysis records.

CAPABILITIES (tools you can call):
  - get_quote: current price, volume, change for any stock
  - get_indicators: RSI, MACD, SMA, Bollinger, ATR, ADX, etc.
  - get_fundamentals: P/E, P/B, ROE, margins, debt, dividends, etc.
  - get_news: recent news headlines and sentiment
  - search_past_analyses: semantic search over past stock analyses (RAG)
  - compare_stocks: side-by-side comparison of multiple stocks
  - run_scorer: algorithmic scoring (momentum, value, quality, risk)

RULES:
1. Always fetch real data before making claims. Never hallucinate numbers.
2. Cite specific values from tool results in your reasoning.
3. If a tool returns an error, tell the user — don't make up data.
4. Keep your final answer concise but thorough. Use markdown formatting.
5. When comparing stocks, use compare_stocks for efficiency instead of \
   calling get_quote for each one separately.
6. You may call multiple tools in sequence to build a complete picture.
"""


class ResearchAgent:
    """Multi-step research agent for complex financial questions."""

    def __init__(self, model: str | None = None) -> None:
        from services.fetcher import StockFetcher
        from services.scorer import StockScorer

        self._fetcher = StockFetcher()
        self._scorer = StockScorer()

        # RAG retriever (optional)
        self._rag = None
        try:
            from services.rag_retriever import RAGRetriever
            self._rag = RAGRetriever()
        except Exception:
            logger.info("RAG retriever not available — search_past_analyses disabled")

        # ML predictor (optional)
        self._predictor = None
        try:
            from config import settings
            if settings.ml_model_enabled and settings.ml_model_path:
                from training.predictor import SignalPredictor
                self._predictor = SignalPredictor(settings.ml_model_path)
        except Exception:
            pass

        self.engine = ReActEngine(
            model=model,
            tools=self._build_tools(),
            system_prompt=SYSTEM_PROMPT,
            max_steps=10,
        )

    def ask(self, question: str) -> AgentResult:
        return self.engine.run(question)

    def ask_stream(self, question: str):
        return self.engine.run_stream(question)

    # ------------------------------------------------------------------
    #  Tool definitions
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[ToolDefinition]:
        tools = [
            ToolDefinition(
                name="get_quote",
                description=(
                    "Get the current stock quote: price, change, volume, "
                    "market cap, P/E ratio, 52-week range."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {
                            "type": "string",
                            "description": "Stock ticker symbol, e.g. AAPL, RELIANCE.NS",
                        },
                    },
                    "required": ["symbol"],
                },
                function=self._tool_get_quote,
            ),
            ToolDefinition(
                name="get_indicators",
                description=(
                    "Get technical indicators for a stock: RSI(14), MACD, "
                    "SMA(20/50), Bollinger Bands, ATR, ADX, volume ratio."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                        "period": {
                            "type": "string",
                            "description": "History period: 1mo, 3mo, 6mo, 1y, 2y",
                            "enum": ["1mo", "3mo", "6mo", "1y", "2y"],
                        },
                    },
                    "required": ["symbol"],
                },
                function=self._tool_get_indicators,
            ),
            ToolDefinition(
                name="get_fundamentals",
                description=(
                    "Get company fundamentals: P/E, P/B, PEG, ROE, ROA, "
                    "profit margin, debt/equity, dividend yield, revenue growth, "
                    "earnings growth, analyst targets."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                    },
                    "required": ["symbol"],
                },
                function=self._tool_get_fundamentals,
            ),
            ToolDefinition(
                name="get_news",
                description="Get recent news headlines and sentiment for a stock.",
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                    },
                    "required": ["symbol"],
                },
                function=self._tool_get_news,
            ),
            ToolDefinition(
                name="compare_stocks",
                description=(
                    "Compare 2-4 stocks side by side on price, momentum score, "
                    "value score, quality score, risk score, and overall signal."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbols": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of 2-4 stock tickers to compare",
                        },
                    },
                    "required": ["symbols"],
                },
                function=self._tool_compare_stocks,
            ),
            ToolDefinition(
                name="run_scorer",
                description=(
                    "Run the algorithmic scorer on a stock to get momentum, value, "
                    "quality, and risk scores (0-100) plus an overall signal."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                    },
                    "required": ["symbol"],
                },
                function=self._tool_run_scorer,
            ),
        ]

        if self._rag:
            tools.append(
                ToolDefinition(
                    name="search_past_analyses",
                    description=(
                        "Search past stock analyses using semantic search (RAG). "
                        "Useful for finding historical patterns, similar setups, "
                        "and what happened last time a stock was in this situation."
                    ),
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query",
                            },
                            "symbol": {
                                "type": "string",
                                "description": "Optional: filter to a specific stock",
                            },
                        },
                        "required": ["query"],
                    },
                    function=self._tool_search_past_analyses,
                )
            )

        return tools

    # ------------------------------------------------------------------
    #  Tool implementations
    # ------------------------------------------------------------------

    def _tool_get_quote(self, symbol: str) -> dict[str, Any]:
        quote = self._fetcher.get_quote(symbol)
        if not quote:
            return {"error": f"No quote data available for {symbol}"}
        return {
            "symbol": quote.symbol,
            "price": quote.price,
            "change": quote.change,
            "change_percent": quote.change_percent,
            "volume": quote.volume,
            "avg_volume": quote.avg_volume,
            "high": quote.high,
            "low": quote.low,
            "market_cap": quote.market_cap,
            "pe_ratio": quote.pe_ratio,
            "52w_high": quote.fifty_two_week_high,
            "52w_low": quote.fifty_two_week_low,
        }

    def _tool_get_indicators(self, symbol: str, period: str = "6mo") -> dict[str, Any]:
        history = self._fetcher.get_price_history(symbol, period=period)
        if not history:
            return {"error": f"No price history for {symbol}"}
        indicators = self._fetcher.calculate_indicators(history)
        # Return only the most useful indicators to keep context small
        keys = [
            "rsi_14", "macd", "macd_signal", "macd_histogram",
            "sma_20", "sma_50", "price_vs_sma20_pct", "price_vs_sma50_pct",
            "bollinger_upper", "bollinger_lower", "bollinger_width_pct",
            "atr_14", "atr_pct", "adx", "plus_di", "minus_di",
            "volume_ratio",
        ]
        return {k: indicators.get(k) for k in keys if indicators.get(k) is not None}

    def _tool_get_fundamentals(self, symbol: str) -> dict[str, Any]:
        data = self._fetcher.get_fundamentals(symbol)
        if not data:
            return {"error": f"No fundamental data for {symbol}"}
        # Return the most important fields
        keys = [
            "company_name", "sector", "industry",
            "pe_ratio", "forward_pe", "pb_ratio", "peg_ratio",
            "profit_margin", "roe", "roa", "current_ratio", "debt_to_equity",
            "revenue_growth", "earnings_growth", "dividend_yield",
            "target_mean", "recommendation",
            "fifty_two_week_high", "fifty_two_week_low",
        ]
        return {k: data.get(k) for k in keys if data.get(k) is not None}

    def _tool_get_news(self, symbol: str) -> dict[str, Any]:
        articles = self._fetcher.get_news_yahoo(symbol)
        if not articles:
            return {"articles": [], "note": f"No recent news found for {symbol}"}
        return {
            "articles": [
                {
                    "headline": a.headline,
                    "source": a.source,
                    "summary": (a.summary or "")[:200],
                }
                for a in articles[:5]
            ],
        }

    def _tool_compare_stocks(self, symbols: list[str]) -> dict[str, Any]:
        if len(symbols) > 4:
            symbols = symbols[:4]
        results = {}
        for sym in symbols:
            quote = self._fetcher.get_quote(sym)
            if not quote:
                results[sym] = {"error": "No data"}
                continue
            history = self._fetcher.get_price_history(sym, period="6mo")
            indicators = self._fetcher.calculate_indicators(history) if history else {}
            fundamentals = self._fetcher.get_fundamentals(sym) or {}

            q_dict = {
                "price": quote.price,
                "volume": quote.volume,
                "avg_volume": quote.avg_volume,
            }
            scores = self._scorer.calculate_all_scores(
                quote=q_dict,
                indicators=indicators,
                fundamentals=fundamentals,
                price_history_days=len(history) if history else 0,
                has_news=False,
            )
            results[sym] = {
                "price": quote.price,
                "change_pct": quote.change_percent,
                "signal": scores.overall_signal,
                "composite_score": scores.composite_score,
                "momentum": scores.momentum_score,
                "value": scores.value_score,
                "quality": scores.quality_score,
                "risk": scores.risk_score,
                "confidence": scores.confidence_score,
            }
        return results

    def _tool_run_scorer(self, symbol: str) -> dict[str, Any]:
        quote = self._fetcher.get_quote(symbol)
        if not quote:
            return {"error": f"No data for {symbol}"}
        history = self._fetcher.get_price_history(symbol, period="6mo")
        indicators = self._fetcher.calculate_indicators(history) if history else {}
        fundamentals = self._fetcher.get_fundamentals(symbol) or {}

        scores = self._scorer.calculate_all_scores(
            quote={"price": quote.price, "volume": quote.volume, "avg_volume": quote.avg_volume},
            indicators=indicators,
            fundamentals=fundamentals,
            price_history_days=len(history) if history else 0,
            has_news=False,
        )
        return {
            "symbol": symbol,
            "signal": scores.overall_signal,
            "composite_score": scores.composite_score,
            "momentum_score": scores.momentum_score,
            "value_score": scores.value_score,
            "quality_score": scores.quality_score,
            "risk_score": scores.risk_score,
            "confidence_score": scores.confidence_score,
        }

    def _tool_search_past_analyses(self, query: str, symbol: str | None = None) -> dict[str, Any]:
        if not self._rag:
            return {"error": "RAG retriever not available"}
        ctx = self._rag.retrieve_context(
            query=query,
            stock_symbol=symbol,
            max_results_per_source=3,
        )
        return {
            "similar_analyses": ctx.similar_analyses[:3],
            "similar_signals": ctx.similar_signals[:3],
            "relevant_news": ctx.relevant_news[:3],
            "total_results": ctx.total_results,
            "retrieval_time_ms": ctx.retrieval_time_ms,
        }
