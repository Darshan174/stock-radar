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
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field

import litellm
from litellm import completion

from agents.storage import StockStorage
from agents.rag_retriever import RAGRetriever, RAGContext
from agents.usage_tracker import get_tracker

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

    # LLM model fallback chain
    MODELS = [
        "zai/glm-4.7",
        "gemini/gemini-2.0-flash",
        "ollama/mistral",
    ]

    SYSTEM_PROMPT = """You are Stock Radar's AI assistant, an expert in stock market analysis.
You help users understand stocks, trading signals, and market conditions.

Guidelines:
1. Answer questions based PRIMARILY on the provided context from our database
2. If the context doesn't contain relevant information, say so clearly
3. When discussing specific stocks, reference the data we have (analyses, signals, news)
4. Be concise but thorough - traders value clear, actionable insights
5. If asked about predictions, remind users that past performance doesn't guarantee future results
6. Always cite the sources of your information when available

Context will be provided in the format:
- Historical analyses from our database
- Recent news articles
- Past trading signals
- Knowledge base entries

Use this context to provide informed, data-backed responses."""

    def __init__(
        self,
        storage: Optional[StockStorage] = None,
        retriever: Optional[RAGRetriever] = None,
        zai_key: Optional[str] = None,
        gemini_key: Optional[str] = None
    ):
        """
        Initialize the chat assistant.

        Args:
            storage: StockStorage instance
            retriever: RAGRetriever instance
            zai_key: Zhipu AI (Z.AI) API key for GLM-4.7
            gemini_key: Gemini API key
        """
        self.storage = storage or StockStorage()
        self.retriever = retriever or RAGRetriever(storage=self.storage)

        # Configure API keys
        self.zai_key = zai_key or os.getenv("ZAI_API_KEY")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")

        if self.zai_key:
            os.environ["ZAI_API_KEY"] = self.zai_key
            if not os.getenv("ZAI_API_BASE"):
                os.environ["ZAI_API_BASE"] = "https://open.bigmodel.cn/api/coding/paas/v4"
        if self.gemini_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_key

        # Build available models list
        self.available_models = []
        if self.zai_key:
            self.available_models.append(self.MODELS[0])
        if self.gemini_key:
            self.available_models.append(self.MODELS[1])
        self.available_models.append(self.MODELS[2])  # Ollama backup

        # Conversation state
        self.session_id: Optional[str] = None
        self.conversation_history: List[ChatMessage] = []

        logger.info(f"StockChatAssistant initialized with models: {self.available_models}")

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
            r'\b([A-Z]{2,5})\.NS\b',  # Indian stocks: RELIANCE.NS
            r'\b([A-Z]{2,5})\.BO\b',  # BSE stocks: RELIANCE.BO
            r'\b([A-Z]{1,5})\b',       # US stocks: AAPL, MSFT
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
            'SHE', 'TOO', 'USE', 'RSI', 'MACD', 'SMA', 'EMA', 'ATR'
        }

        return [s for s in symbols if s not in common_words and len(s) >= 2]

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
        temperature: float = 0.3
    ) -> tuple[str, str, int]:
        """
        Call LLM with fallback chain.

        Args:
            messages: Chat messages
            temperature: Response temperature

        Returns:
            Tuple of (response_text, model_used, tokens_used)
        """
        last_error = None

        for model in self.available_models:
            try:
                logger.info(f"Trying model: {model}")

                response = completion(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=1500,
                )

                content = response.choices[0].message.content
                tokens = 0

                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    if hasattr(usage, 'total_tokens') and usage.total_tokens:
                        tokens = usage.total_tokens
                    elif hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                        tokens = (usage.prompt_tokens or 0) + (usage.completion_tokens or 0)

                # Track usage
                service = model.split("/")[0]
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

        primary_symbol = detected_symbols[0] if detected_symbols else None

        # Retrieve RAG context
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

        # Build context prompt
        context_prompt = self._build_context_prompt(context)

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

{f"Focus on stock: {primary_symbol}" if primary_symbol else ""}

Retrieved Context from Database:
{context_prompt}

Please answer the question based on the above context. If the context doesn't contain
relevant information, acknowledge this and provide general guidance."""

        messages.append({"role": "user", "content": user_message})

        # Call LLM
        try:
            answer, model_used, tokens_used = self._call_llm(messages)
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
