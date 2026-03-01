"""
Stock Radar - RAG-Powered Chat Assistant

This module provides a conversational interface for stock-related questions.
It uses RAG (Retrieval-Augmented Generation) to:
- Answer questions about specific stocks
- Explain past analyses and signals
- Provide market context and insights
- Help users understand trading decisions
"""

import os
import re
import uuid
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

from litellm import completion

from agents.storage import StockStorage
from agents.rag_retriever import RAGRetriever, RAGContext
from agents.usage_tracker import get_tracker

try:
    from config import settings
except ImportError:
    settings = None

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A single chat message."""
    role: str  # 'user', 'assistant', 'system'
    content: str
    stock_symbols: List[str] = field(default_factory=list)
    context_used: Optional[Dict[str, Any]] = None
    tokens_used: int = 0
    model_used: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class ChatResponse:
    """Response from the chat assistant."""
    answer: str
    context_retrieved: RAGContext
    stock_symbols: List[str]
    sources_used: List[Dict[str, Any]]
    model_used: str
    tokens_used: int
    processing_time_ms: int


class StockChatAssistant:
    """
    RAG-powered conversational assistant for stock analysis.

    Features:
    - Answers questions about stocks using retrieved context
    - Explains past analyses and trading signals
    - Provides market insights based on historical data
    - Maintains conversation history for context
    """

    DEFAULT_MODELS = ["openai/glm-4.7", "gemini/gemini-2.5-flash"]
    DEFAULT_TASK_ROUTES = {
        "chat": [
            "groq/llama-3.1-70b-versatile",
            "openai/glm-4.7",
            "gemini/gemini-2.5-flash",
        ],
        "analysis": [
            "groq/llama-3.1-70b-versatile",
            "openai/glm-4.7",
            "gemini/gemini-2.5-flash",
        ],
        "sentiment": [
            "groq/llama-3.1-8b-instant",
            "gemini/gemini-2.5-flash",
            "openai/glm-4.7",
        ],
        "news": [
            "groq/llama-3.1-8b-instant",
            "gemini/gemini-2.5-flash",
            "openai/glm-4.7",
        ],
    }

    SYSTEM_PROMPT = """You are Stock Radar's AI analyst — a sharp, data-driven trading assistant.

Your responses must be:
- STRUCTURED with clear sections using markdown headers and bullet points
- SPECIFIC — always cite exact numbers (RSI=31.6, not "RSI is low")
- EDUCATIONAL — briefly explain what each indicator means for non-experts
- ACTIONABLE — give clear levels (entry, target, stop-loss) when data is available

Response format for stock analysis questions:

## Signal & Conviction
State the signal (BUY/SELL/HOLD), confidence %, and a one-line thesis.

## Technical Breakdown
For each indicator, state the value AND what it means:
- RSI: value → what it signals (oversold <30, overbought >70, neutral 30-70)
- MACD: value vs signal line → bullish/bearish crossover explanation
- Price vs SMA: where price sits relative to moving averages → trend direction
- Bollinger Bands: position within bands → volatility read
- Volume: current vs average → conviction behind the move

## Key Levels
Table format: Support | Resistance | Target | Stop Loss

## Algo Scores (if available)
Momentum, Value, Quality, Risk scores with brief interpretation of each.

## Risk Factors
What could go wrong. Divergences between signals. Low confidence areas.

## What This Means (Plain English)
2-3 sentence summary a beginner would understand. No jargon.

Rules:
- Use the provided stock data and context as your primary source
- If algo signal diverges from technical signal, explain WHY
- Always end with a brief risk disclaimer
- Use ₹ for Indian stocks, $ for US stocks
- When no analysis exists, fetch what you can from context and be transparent about gaps"""

    def __init__(
        self,
        storage: Optional[StockStorage] = None,
        retriever: Optional[RAGRetriever] = None,
        zai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        groq_key: Optional[str] = None,
    ):
        """
        Initialize the chat assistant.

        Args:
            storage: StockStorage instance
            retriever: RAGRetriever instance
            zai_key: Zhipu AI (Z.AI) API key for GLM-5
            gemini_key: Gemini API key
            groq_key: Groq API key
        """
        self.storage = storage or StockStorage()
        self.retriever = retriever or RAGRetriever(storage=self.storage)

        # Configure API keys
        self.zai_key = zai_key or (settings.zai_api_key if settings else None) or os.getenv("ZAI_API_KEY")
        self.zai_api_base = (
            (settings.zai_api_base if settings else None)
            or os.getenv("ZAI_API_BASE", "https://open.bigmodel.cn/api/coding/paas/v4")
        )
        self.gemini_key = (
            gemini_key or (settings.gemini_api_key if settings else None) or os.getenv("GEMINI_API_KEY")
        )
        self.groq_key = groq_key or (settings.groq_api_key if settings else None) or os.getenv("GROQ_API_KEY")

        if self.gemini_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_key
        if self.groq_key:
            os.environ["GROQ_API_KEY"] = self.groq_key

        configured_defaults = (
            list(settings.fallback_models) if settings and settings.fallback_models else list(self.DEFAULT_MODELS)
        )
        configured_routes = (
            dict(settings.task_model_routes) if settings and settings.task_model_routes else {}
        )
        self.default_models = configured_defaults
        self.task_routes = dict(self.DEFAULT_TASK_ROUTES)
        self.task_routes.update(configured_routes)

        # Build provider-available model set from defaults + all task routes.
        candidate_models = list(self.default_models)
        for models in self.task_routes.values():
            candidate_models.extend(models)

        unique_candidates = []
        seen = set()
        for model in candidate_models:
            if model in seen:
                continue
            seen.add(model)
            unique_candidates.append(model)

        self.available_models = [m for m in unique_candidates if self._model_is_available(m)]

        # Conversation state
        self.session_id: Optional[str] = None
        self.conversation_history: List[ChatMessage] = []

        logger.info(
            "StockChatAssistant initialized with models=%s task_routes=%s",
            self.available_models,
            sorted(self.task_routes.keys()),
        )

    def _model_is_available(self, model: str) -> bool:
        """Check provider credentials for a model."""
        if model.startswith("openai/"):
            return bool(self.zai_key)
        if model.startswith("gemini/"):
            return bool(self.gemini_key)
        if model.startswith("groq/"):
            return bool(self.groq_key)
        return True

    def _models_for_task(self, task: str = "default") -> List[str]:
        """Resolve model order for task, falling back to global defaults."""
        task_key = (task or "default").lower()
        route = self.task_routes.get(task_key, self.default_models)
        routed = [m for m in route if m in self.available_models]
        if routed:
            return routed

        default_available = [m for m in self.default_models if m in self.available_models]
        if default_available:
            return default_available
        return list(self.available_models)

    def start_session(self, user_id: Optional[str] = None) -> str:
        """
        Start a new chat session.

        Args:
            user_id: Optional user ID for personalization

        Returns:
            Session ID
        """
        self.session_id = str(uuid.uuid4())
        self.conversation_history = []
        self.user_id = user_id
        logger.info(f"Started new chat session: {self.session_id}")
        return self.session_id

    def _extract_stock_symbols(self, text: str) -> List[str]:
        """
        Extract stock symbols from text.

        Args:
            text: Text to search

        Returns:
            List of potential stock symbols
        """
        # Common patterns for stock symbols
        patterns = [
            r'\b([A-Z]{2,20})\.NS\b',  # Indian NSE stocks: RELIANCE.NS, TATAMOTORS.NS
            r'\b([A-Z]{2,20})\.BO\b',  # Indian BSE stocks: RELIANCE.BO
            r'\b([A-Z]{2,20})\b',       # All stocks: RELIANCE, AAPL, BAJFINANCE
        ]

        symbols = set()
        for pattern in patterns:
            matches = re.findall(pattern, text.upper())
            symbols.update(matches)

        # Filter out common words that look like symbols
        common_words = {
            'THE', 'AND', 'FOR', 'ARE', 'BUT', 'NOT', 'YOU', 'ALL',
            'CAN', 'HAD', 'HER', 'WAS', 'ONE', 'OUR', 'OUT', 'HAS',
            'HIS', 'HOW', 'ITS', 'MAY', 'NEW', 'NOW', 'OLD', 'SEE',
            'WAY', 'WHO', 'BOY', 'DID', 'GET', 'LET', 'PUT', 'SAY',
            'SHE', 'TOO', 'USE', 'RSI', 'MACD', 'SMA', 'EMA', 'ATR',
            'WHAT', 'WHEN', 'WHERE', 'WHY', 'WHICH', 'SHOW', 'GIVE',
            'TELL', 'ABOUT', 'LATEST', 'RECENT', 'CURRENT', 'BEST',
            'GOOD', 'BAD', 'HIGH', 'LOW', 'BUY', 'SELL', 'HOLD',
            'STOCK', 'PRICE', 'NEWS', 'FROM', 'WITH', 'THAT', 'THIS',
            'WILL', 'BEEN', 'HAVE', 'EACH', 'MAKE', 'LIKE', 'LONG',
            'LOOK', 'MANY', 'SOME', 'THAN', 'THEM', 'THEN', 'VERY',
            'MUCH', 'ALSO', 'BACK', 'JUST', 'ONLY', 'COME', 'MADE',
            'FIND', 'HERE', 'KNOW', 'TAKE', 'WANT', 'DOES', 'HELP',
            'OVER', 'SUCH', 'WHAT', 'YEAR', 'INTO', 'MOST', 'WELL',
            'ANALYSIS', 'COMPARE', 'BETWEEN', 'SHOULD', 'COULD',
            'WOULD', 'MARKET', 'TRADING', 'SIGNAL', 'EXPLAIN',
            'IS', 'IT', 'IF', 'IN', 'ON', 'OR', 'AT', 'TO', 'UP',
            'SO', 'DO', 'NO', 'AN', 'AS', 'BY', 'GO', 'MY', 'OF',
            'WE', 'ME', 'HE', 'BE', 'NS', 'BO',
        }

        return [s for s in symbols if s not in common_words and len(s) >= 2]

    def _ensure_stock_data(self, symbol: str) -> bool:
        """Check if symbol has data in DB. If not, fetch fresh and store."""
        # Try symbol + common variants
        for variant in [symbol, f"{symbol}.NS", f"{symbol}.BO"]:
            stock = self.storage.get_stock_by_symbol(variant)
            if stock:
                return True

        # No data found — fetch fresh
        try:
            from agents.fetcher import StockFetcher
            import threading

            fetcher = StockFetcher()

            # Try symbol variants: raw, .NS, .BO
            variants = [symbol]
            if not symbol.endswith((".NS", ".BO")):
                variants.extend([f"{symbol}.NS", f"{symbol}.BO"])

            data = {}
            for variant in variants:
                # Fresh container per variant so a timed-out daemon thread
                # from a previous iteration cannot contaminate it.
                result_box: list[dict] = [{}]

                def _do_fetch(sym=variant, box=result_box):
                    try:
                        box[0] = fetcher.get_full_analysis_data(sym, "1y")
                    except Exception:
                        pass

                t = threading.Thread(target=_do_fetch, daemon=True)
                t.start()
                t.join(timeout=15)
                if t.is_alive():
                    logger.warning(f"Fresh fetch timed out for {variant}")
                    continue
                fetched = result_box[0]
                if fetched.get("quote"):
                    data = fetched
                    symbol = variant
                    break
            else:
                return False  # No variant returned a quote

            if not data.get("quote"):
                return False

            # Infer exchange from resolved symbol suffix
            if symbol.endswith(".NS"):
                exchange = "NSE"
            elif symbol.endswith(".BO"):
                exchange = "BSE"
            else:
                exchange = "NASDAQ"

            fundamentals = data.get("fundamentals") or {}
            stock = self.storage.get_or_create_stock(
                symbol=symbol,
                name=fundamentals.get("name") or fundamentals.get("short_name") or symbol,
                exchange=exchange,
                sector=fundamentals.get("sector"),
            )
            stock_id = stock["id"]

            # Store indicators
            if data.get("indicators"):
                self.storage.store_indicators(
                    stock_id,
                    datetime.now(timezone.utc),
                    data["indicators"],
                    timeframe="1d",
                )

            # Store news (up to 10 articles)
            for news_item in (data.get("news") or [])[:10]:
                try:
                    self.storage.store_news(
                        headline=news_item.headline,
                        summary=news_item.summary,
                        source=news_item.source,
                        url=news_item.url,
                        published_at=news_item.published_at,
                        stock_id=stock_id,
                    )
                except Exception:
                    pass  # Skip individual news storage failures

            logger.info(f"Fresh data fetched and stored for {symbol}")
            return True

        except Exception as e:
            logger.warning(f"Fresh fetch failed for {symbol}: {e}")
            return False

    def _fetch_direct_stock_data(self, symbols: List[str]) -> str:
        """
        Directly fetch latest analysis and live data for detected symbols.

        This bypasses RAG similarity search and does exact symbol lookups,
        ensuring we always have data when a stored analysis exists.
        """
        if not symbols:
            return ""

        parts = []
        for symbol in symbols[:3]:  # Limit to 3 symbols
            try:
                # Try symbol variants: RELIANCE, RELIANCE.NS, RELIANCE.BO
                stock = self.storage.get_stock_by_symbol(symbol)
                if not stock:
                    stock = self.storage.get_stock_by_symbol(f"{symbol}.NS")
                if not stock:
                    stock = self.storage.get_stock_by_symbol(f"{symbol}.BO")
                if not stock:
                    continue

                stock_id = stock.get("id")
                stock_name = stock.get("name", symbol)
                exchange = stock.get("exchange", "")

                section = f"=== {symbol} ({stock_name}) - {exchange} ==="

                # Fetch latest stored analysis
                for mode in ("intraday", "longterm"):
                    analysis = self.storage.get_latest_analysis(stock_id, mode=mode)
                    if analysis:
                        signal = analysis.get("signal", "N/A")
                        confidence = analysis.get("confidence", 0)
                        reasoning = analysis.get("reasoning", "")
                        created = str(analysis.get("created_at", ""))[:19]
                        support = analysis.get("support_level")
                        resistance = analysis.get("resistance_level")
                        target = analysis.get("target_price")
                        stop_loss = analysis.get("stop_loss")
                        algo = analysis.get("algo_prediction")

                        section += f"\n\nLatest {mode.upper()} Analysis ({created}):"
                        section += f"\n  Signal: {signal}"
                        section += f"\n  Confidence: {confidence}"
                        section += f"\n  Reasoning: {reasoning}"
                        if support:
                            section += f"\n  Support: {support}"
                        if resistance:
                            section += f"\n  Resistance: {resistance}"
                        if target:
                            section += f"\n  Target Price: {target}"
                        if stop_loss:
                            section += f"\n  Stop Loss: {stop_loss}"
                        if algo and isinstance(algo, dict):
                            section += f"\n  Algo Signal: {algo.get('signal', 'N/A')}"
                            section += f"\n  Algo Confidence: {algo.get('confidence', 'N/A')}"
                            section += f"\n  Momentum Score: {algo.get('momentum_score', 'N/A')}"
                            section += f"\n  Value Score: {algo.get('value_score', 'N/A')}"
                            section += f"\n  Quality Score: {algo.get('quality_score', 'N/A')}"
                            section += f"\n  Risk Score: {algo.get('risk_score', 'N/A')}"
                            if algo.get("scoring_method"):
                                section += f"\n  Scoring Method: {algo['scoring_method']}"
                            if algo.get("market_regime"):
                                section += f"\n  Market Regime: {algo['market_regime']}"
                        break  # Use the first mode that has data

                # Fetch latest technical indicators
                if stock_id:
                    try:
                        ind = self.storage.get_latest_indicators(stock_id)
                        if ind and isinstance(ind, dict):
                            indicators = ind.get("indicators") or ind
                            section += "\n\nTechnical Indicators:"
                            indicator_keys = [
                                ("rsi_14", "RSI(14)"),
                                ("macd", "MACD"),
                                ("macd_signal", "MACD Signal"),
                                ("macd_histogram", "MACD Histogram"),
                                ("sma_20", "SMA(20)"),
                                ("sma_50", "SMA(50)"),
                                ("sma_200", "SMA(200)"),
                                ("price_vs_sma20_pct", "Price vs SMA20 %"),
                                ("price_vs_sma50_pct", "Price vs SMA50 %"),
                                ("bollinger_upper", "Bollinger Upper"),
                                ("bollinger_lower", "Bollinger Lower"),
                                ("atr_pct", "ATR %"),
                                ("adx", "ADX"),
                                ("plus_di", "+DI"),
                                ("minus_di", "-DI"),
                            ]
                            for key, label in indicator_keys:
                                val = indicators.get(key)
                                if val is not None:
                                    section += f"\n  {label}: {val}"
                    except Exception:
                        pass

                # Fetch recent news for this stock
                if stock_id:
                    try:
                        news = self.storage.get_recent_news(stock_id=stock_id, limit=3)
                        if news:
                            section += "\n\nRecent News:"
                            for n in news:
                                headline = n.get("headline", "")
                                sentiment = n.get("sentiment_label", "neutral")
                                date = str(n.get("published_at", ""))[:10]
                                section += f"\n  [{date}] [{sentiment}] {headline}"
                    except Exception:
                        pass

                parts.append(section)

            except Exception as e:
                logger.warning(f"Failed to fetch direct data for {symbol}: {e}")

        if not parts:
            return ""

        return "=== DIRECT STOCK DATA (from database) ===\n" + "\n\n".join(parts)

    def _build_context_prompt(self, context: RAGContext) -> str:
        """
        Build context section for the LLM prompt.

        Args:
            context: Retrieved RAG context

        Returns:
            Formatted context string
        """
        parts = []

        # Add similar analyses
        if context.similar_analyses:
            parts.append("=== RELEVANT PAST ANALYSES ===")
            for i, analysis in enumerate(context.similar_analyses[:3], 1):
                symbol = analysis.get('symbol', 'N/A')
                signal = analysis.get('signal', 'N/A')
                confidence = analysis.get('confidence', 0)
                reasoning = analysis.get('reasoning', '')
                created = str(analysis.get('created_at', ''))[:10]
                similarity = analysis.get('similarity', 0)

                parts.append(
                    f"\n[Analysis {i}] {symbol} - {created}\n"
                    f"Signal: {signal.upper()} (confidence: {confidence:.0%})\n"
                    f"Reasoning: {reasoning}\n"
                    f"Relevance: {similarity:.0%}"
                )

        # Add relevant news
        if context.relevant_news:
            parts.append("\n\n=== RELEVANT NEWS ===")
            for i, news in enumerate(context.relevant_news[:3], 1):
                headline = news.get('headline', '')
                summary = news.get('summary', '')
                sentiment = news.get('sentiment_label', 'neutral')
                source = news.get('source', 'Unknown')
                similarity = news.get('similarity', 0)

                parts.append(
                    f"\n[News {i}] {headline}\n"
                    f"Source: {source} | Sentiment: {sentiment}\n"
                    f"Summary: {summary[:200]}...\n"
                    f"Relevance: {similarity:.0%}"
                )

        # Add signals
        if context.similar_signals:
            parts.append("\n\n=== RECENT TRADING SIGNALS ===")
            for i, signal in enumerate(context.similar_signals[:3], 1):
                symbol = signal.get('symbol', 'N/A')
                sig_type = signal.get('signal', 'N/A')
                price = signal.get('price_at_signal', 'N/A')
                reason = signal.get('reason', '')
                importance = signal.get('importance', 'medium')
                created = str(signal.get('created_at', ''))[:10]

                parts.append(
                    f"\n[Signal {i}] {symbol} - {sig_type.upper()} at {price}\n"
                    f"Date: {created} | Importance: {importance}\n"
                    f"Reason: {reason[:150]}"
                )

        # Add knowledge base entries
        if context.knowledge_entries:
            parts.append("\n\n=== KNOWLEDGE BASE ===")
            for i, kb in enumerate(context.knowledge_entries[:2], 1):
                title = kb.get('title', '')
                content = kb.get('content', '')[:300]
                category = kb.get('category', 'general')

                parts.append(
                    f"\n[Knowledge {i}] {title}\n"
                    f"Category: {category}\n"
                    f"Content: {content}..."
                )

        if not parts:
            return "No relevant context found in our database for this query."

        return "\n".join(parts)

    def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        task: str = "chat",
    ) -> tuple[str, str, int]:
        """
        Call LLM with fallback chain.

        Args:
            messages: Chat messages
            temperature: Response temperature
            task: Logical LLM task route key

        Returns:
            Tuple of (response_text, model_used, tokens_used)
        """
        models = self._models_for_task(task=task)
        if not models:
            raise Exception("No available LLM providers configured for requested task.")

        last_error = None

        for model in models:
            try:
                logger.info(f"Trying model for task={task}: {model}")

                # Pass api_base and api_key for ZAI (OpenAI-compatible endpoint)
                kwargs = dict(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=2500,
                )
                if model.startswith("openai/") and self.zai_key:
                    kwargs["api_base"] = self.zai_api_base
                    kwargs["api_key"] = self.zai_key
                elif model.startswith("groq/") and self.groq_key:
                    kwargs["api_key"] = self.groq_key

                response = completion(**kwargs)

                content = response.choices[0].message.content
                tokens = 0

                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    if hasattr(usage, 'total_tokens') and usage.total_tokens:
                        tokens = usage.total_tokens
                    elif hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                        tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)

                # Track usage
                service = "zai" if model.startswith("openai/glm") else model.split("/")[0]
                get_tracker().track(service, count=1, tokens=tokens)

                logger.info(f"Success with {model} ({tokens} tokens)")
                return content, model, tokens

            except Exception as e:
                last_error = e
                logger.warning(f"{model} failed: {e}")
                continue

        raise Exception(f"All models failed. Last error: {last_error}")

    def ask(
        self,
        question: str,
        stock_symbol: Optional[str] = None,
        include_history: bool = True,
        max_history_messages: int = 5
    ) -> ChatResponse:
        """
        Ask a question and get a RAG-powered response.

        Args:
            question: User's question
            stock_symbol: Specific stock to focus on (optional)
            include_history: Include conversation history
            max_history_messages: Maximum history messages to include

        Returns:
            ChatResponse with answer and metadata
        """
        start_time = time.time()

        # Start session if needed
        if not self.session_id:
            self.start_session()

        # Extract stock symbols from question
        detected_symbols = self._extract_stock_symbols(question)
        if stock_symbol and stock_symbol not in detected_symbols:
            detected_symbols.insert(0, stock_symbol)

        # Ensure we have data for detected symbols (fresh-fetch fallback)
        for sym in detected_symbols[:3]:
            try:
                self._ensure_stock_data(sym)
            except Exception as e:
                logger.warning(f"Data coverage check failed for {sym}: {e}")

        primary_symbol = detected_symbols[0] if detected_symbols else None

        # Retrieve RAG context (semantic search)
        context = self.retriever.retrieve_context(
            query=question,
            stock_symbol=primary_symbol,
            user_id=getattr(self, 'user_id', None),
            include_analyses=True,
            include_signals=True,
            include_news=True,
            include_knowledge=True,
            include_conversations=False,
            max_results_per_source=3,
            match_threshold=0.4
        )

        # Build context prompt from RAG
        context_prompt = self._build_context_prompt(context)

        # Direct data fetch for detected symbols (bypasses RAG similarity)
        direct_data = self._fetch_direct_stock_data(detected_symbols)

        # Build messages
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT}
        ]

        # Add conversation history
        if include_history and self.conversation_history:
            history_to_include = self.conversation_history[-max_history_messages:]
            for msg in history_to_include:
                messages.append({
                    "role": msg.role,
                    "content": msg.content
                })

        # Add user question with context
        user_message = f"""Question: {question}
{f"Primary stock: {primary_symbol}" if primary_symbol else ""}

--- STOCK DATA (direct from database) ---
{direct_data if direct_data else "No stored data found for this symbol."}

--- RAG CONTEXT (semantic search) ---
{context_prompt}
---

Use the structured response format from your instructions. Include every data point
available — don't summarize away the numbers. Explain technical terms inline."""

        messages.append({"role": "user", "content": user_message})

        # Call LLM
        try:
            answer, model_used, tokens_used = self._call_llm(messages, task="chat")
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = f"I apologize, but I encountered an error processing your question: {str(e)}"
            model_used = "error"
            tokens_used = 0

        # Store in conversation history
        user_msg = ChatMessage(
            role="user",
            content=question,
            stock_symbols=detected_symbols
        )
        self.conversation_history.append(user_msg)

        # Build sources list
        sources_used = []
        for analysis in context.similar_analyses[:3]:
            sources_used.append({
                "type": "analysis",
                "symbol": analysis.get('symbol'),
                "signal": analysis.get('signal'),
                "date": str(analysis.get('created_at', ''))[:10],
                "similarity": analysis.get('similarity', 0)
            })
        for news in context.relevant_news[:3]:
            sources_used.append({
                "type": "news",
                "headline": news.get('headline', '')[:50],
                "source": news.get('source'),
                "sentiment": news.get('sentiment_label'),
                "similarity": news.get('similarity', 0)
            })

        # Store context used summary
        context_summary = {
            "total_results": context.total_results,
            "sources_searched": context.sources_searched,
            "retrieval_time_ms": context.retrieval_time_ms
        }

        assistant_msg = ChatMessage(
            role="assistant",
            content=answer,
            stock_symbols=detected_symbols,
            context_used=context_summary,
            tokens_used=tokens_used,
            model_used=model_used
        )
        self.conversation_history.append(assistant_msg)

        # Store in database
        try:
            self.storage.store_chat_message(
                session_id=self.session_id,
                role="user",
                content=question,
                stock_symbols=detected_symbols,
                user_id=getattr(self, 'user_id', None)
            )
            self.storage.store_chat_message(
                session_id=self.session_id,
                role="assistant",
                content=answer,
                stock_symbols=detected_symbols,
                context_used=context_summary,
                tokens_used=tokens_used,
                model_used=model_used,
                user_id=getattr(self, 'user_id', None)
            )
        except Exception as e:
            logger.warning(f"Failed to store chat in database: {e}")

        processing_time_ms = int((time.time() - start_time) * 1000)

        return ChatResponse(
            answer=answer,
            context_retrieved=context,
            stock_symbols=detected_symbols,
            sources_used=sources_used,
            model_used=model_used,
            tokens_used=tokens_used,
            processing_time_ms=processing_time_ms
        )

    def answer(
        self,
        question: str,
        symbol: Optional[str] = None
    ) -> str:
        """
        Simple interface to ask a question and get just the answer.

        Args:
            question: User's question
            symbol: Stock symbol (optional)

        Returns:
            Answer string
        """
        response = self.ask(question, stock_symbol=symbol)
        return response.answer

    def explain_analysis(self, symbol: str, analysis_id: Optional[str] = None) -> str:
        """
        Explain a specific analysis or the latest analysis for a stock.

        Args:
            symbol: Stock symbol
            analysis_id: Specific analysis ID (optional)

        Returns:
            Explanation string
        """
        # Get stock context
        stock_context = self.storage.get_stock_context_for_rag(
            symbol=symbol,
            include_analyses=True,
            include_news=True,
            include_signals=True,
            max_items_per_type=5
        )

        if not stock_context.get('recent_analyses'):
            return f"No analyses found for {symbol}. Try running an analysis first."

        # Get the specific or latest analysis
        if analysis_id:
            analysis = next(
                (a for a in stock_context['recent_analyses'] if str(a.get('id')) == analysis_id),
                stock_context['recent_analyses'][0]
            )
        else:
            analysis = stock_context['recent_analyses'][0]

        # Build explanation prompt
        question = (
            f"Please explain this trading analysis for {symbol} in detail:\n\n"
            f"Signal: {analysis.get('signal')}\n"
            f"Confidence: {analysis.get('confidence')}\n"
            f"Reasoning: {analysis.get('reasoning')}\n"
            f"Technical Summary: {analysis.get('technical_summary')}\n"
            f"Sentiment Summary: {analysis.get('sentiment_summary')}\n"
            f"Target Price: {analysis.get('target_price')}\n"
            f"Stop Loss: {analysis.get('stop_loss')}\n\n"
            f"Help the user understand why this signal was given and what it means for them."
        )

        response = self.ask(question, stock_symbol=symbol)
        return response.answer

    def get_stock_summary(self, symbol: str) -> str:
        """
        Get a comprehensive summary of a stock based on available data.

        Args:
            symbol: Stock symbol

        Returns:
            Summary string
        """
        question = (
            f"Give me a comprehensive summary of {symbol} based on our available data. "
            f"Include recent analyses, signals, news sentiment, and any relevant insights. "
            f"What should a trader know about this stock right now?"
        )

        response = self.ask(question, stock_symbol=symbol)
        return response.answer

    def compare_stocks(self, symbols: List[str]) -> str:
        """
        Compare multiple stocks based on available data.

        Args:
            symbols: List of stock symbols to compare

        Returns:
            Comparison string
        """
        symbols_str = ", ".join(symbols)
        question = (
            f"Compare these stocks based on our available data: {symbols_str}. "
            f"Look at their recent signals, analyses, and news sentiment. "
            f"Which one looks more promising and why?"
        )

        response = self.ask(question)
        return response.answer

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        logger.info(f"Cleared conversation history for session {self.session_id}")


# =============================================================================
# Convenience Functions
# =============================================================================

_assistant_instance: Optional[StockChatAssistant] = None


def get_assistant() -> StockChatAssistant:
    """Get singleton StockChatAssistant instance."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = StockChatAssistant()
    return _assistant_instance


def quick_ask(question: str, symbol: Optional[str] = None) -> str:
    """
    Quick way to ask a question.

    Args:
        question: User's question
        symbol: Stock symbol (optional)

    Returns:
        Answer string
    """
    return get_assistant().answer(question, symbol)


# =============================================================================
# CLI Interface
# =============================================================================

def run_chat_cli():
    """Run interactive chat CLI."""
    from dotenv import load_dotenv
    load_dotenv()

    print("\n" + "=" * 60)
    print("  STOCK RADAR - AI Chat Assistant")
    print("  RAG-powered stock analysis conversations")
    print("=" * 60)
    print("\nCommands:")
    print("  /quit or /exit - Exit the chat")
    print("  /clear - Clear conversation history")
    print("  /explain <symbol> - Explain latest analysis for a stock")
    print("  /summary <symbol> - Get stock summary")
    print("  /compare <sym1> <sym2> - Compare stocks")
    print("\nAsk any question about stocks, analyses, or market conditions!")
    print("-" * 60 + "\n")

    assistant = StockChatAssistant()
    assistant.start_session()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ('/quit', '/exit'):
                print("\nGoodbye! Happy trading!")
                break

            if user_input.lower() == '/clear':
                assistant.clear_history()
                print("Conversation history cleared.")
                continue

            if user_input.lower().startswith('/explain '):
                symbol = user_input[9:].strip().upper()
                print(f"\nAssistant: {assistant.explain_analysis(symbol)}")
                continue

            if user_input.lower().startswith('/summary '):
                symbol = user_input[9:].strip().upper()
                print(f"\nAssistant: {assistant.get_stock_summary(symbol)}")
                continue

            if user_input.lower().startswith('/compare '):
                symbols = user_input[9:].strip().upper().split()
                if len(symbols) < 2:
                    print("Please provide at least 2 symbols to compare.")
                    continue
                print(f"\nAssistant: {assistant.compare_stocks(symbols)}")
                continue

            # Regular question
            response = assistant.ask(user_input)

            print(f"\nAssistant: {response.answer}")
            print(f"\n[{response.model_used} | {response.tokens_used} tokens | "
                  f"{response.processing_time_ms}ms | {len(response.sources_used)} sources]")

        except KeyboardInterrupt:
            print("\n\nGoodbye! Happy trading!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            continue


if __name__ == "__main__":
    run_chat_cli()
