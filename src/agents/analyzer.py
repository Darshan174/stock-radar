"""
Stock Analyzer using LiteLLM.
Provides AI-powered stock analysis with fallback chain: Groq -> Gemini -> Ollama
"""

import os
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum

import litellm
from litellm import completion

from agents.usage_tracker import get_tracker
from agents.scorer import StockScorer

# RAG imports (optional, for enhanced analysis)
try:
    from agents.rag_retriever import RAGRetriever, retrieve_analysis_context
    from agents.rag_validator import RAGValidator, validate_rag_analysis
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Configure LiteLLM
litellm.set_verbose = False


class TradingMode(str, Enum):
    """Trading analysis mode."""
    INTRADAY = "intraday"
    LONGTERM = "longterm"


class Signal(str, Enum):
    """Trading signals."""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    HOLD = "hold"
    SELL = "sell"
    STRONG_SELL = "strong_sell"


@dataclass
class AnalysisResult:
    """Result of stock analysis."""
    symbol: str
    mode: TradingMode
    signal: Signal
    confidence: float
    reasoning: str
    technical_summary: str
    sentiment_summary: Optional[str]
    support_level: Optional[float]
    resistance_level: Optional[float]
    target_price: Optional[float]
    stop_loss: Optional[float]
    risk_reward_ratio: Optional[float]
    llm_model: str
    tokens_used: int
    analysis_duration_ms: int
    created_at: str
    # Algo trading fields
    algo_prediction: Optional[Dict] = None
    # RAG validation scores
    rag_validation: Optional[Dict] = None


@dataclass
class AlgoPrediction:
    """AI Algo Trading Prediction Result."""
    signal: Signal
    confidence: float
    predicted_return_30d: float
    predicted_return_90d: float
    risk_score: int  # 1-10
    momentum_score: int  # 0-100
    value_score: int  # 0-100
    quality_score: int  # 0-100
    reasoning: List[str]
    key_factors: List[Dict[str, Any]]


class StockAnalyzer:
    """
    AI-powered stock analyzer using LiteLLM.
    Fallback chain: ZAI GLM-4.7 (primary) -> Gemini (quality) -> Ollama Mistral (backup)
    """

    # Model fallback chain
    MODELS = [
        "zai/glm-4.7",                       # Primary - Zhipu AI GLM-4.7 (200K context)
        "gemini/gemini-2.0-flash",            # Secondary - Google's fast model
        "ollama/mistral",                     # Backup - local (only if cloud fails)
    ]

    def __init__(
        self,
        zai_key: Optional[str] = None,
        gemini_key: Optional[str] = None,
        ollama_url: Optional[str] = None,
        enable_rag: bool = True
    ):
        """
        Initialize stock analyzer with API keys.

        Args:
            zai_key: Zhipu AI (Z.AI) API key for GLM-4.7
            gemini_key: Google Gemini API key
            ollama_url: Ollama API URL for local models
            enable_rag: Whether to enable RAG context retrieval for enhanced analysis
        """
        # Set API keys
        self.zai_key = zai_key or os.getenv("ZAI_API_KEY")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")
        self.ollama_url = ollama_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")

        # Configure LiteLLM
        if self.zai_key:
            os.environ["ZAI_API_KEY"] = self.zai_key
            # Coding Plan uses a different endpoint than the regular API
            if not os.getenv("ZAI_API_BASE"):
                os.environ["ZAI_API_BASE"] = "https://open.bigmodel.cn/api/coding/paas/v4"
        if self.gemini_key:
            os.environ["GEMINI_API_KEY"] = self.gemini_key

        # Build available models list - ZAI GLM-4.7 is primary
        self.available_models = []

        # Primary: ZAI GLM-4.7 (cloud, 200K context)
        if self.zai_key:
            self.available_models.append(self.MODELS[0])  # zai/glm-4.7

        # Secondary: Gemini (if key available)
        if self.gemini_key:
            self.available_models.append(self.MODELS[1])

        # Backup: Ollama Mistral (local, only if cloud fails)
        self.available_models.append(self.MODELS[2])

        # Initialize RAG retriever for enhanced analysis
        self.enable_rag = enable_rag and RAG_AVAILABLE
        self.rag_retriever = None
        self.rag_validator = None
        if self.enable_rag:
            try:
                self.rag_retriever = RAGRetriever()
                self.rag_validator = RAGValidator()
                logger.info("RAG retriever and validator initialized for enhanced analysis")
            except Exception as e:
                logger.warning(f"Failed to initialize RAG components: {e}")
                self.enable_rag = False

        logger.info(f"StockAnalyzer initialized with models: {self.available_models}, RAG: {self.enable_rag}")

    def _call_llm(self, prompt: str, system_prompt: str = "") -> tuple[str, str, int]:
        """
        Call LLM with fallback chain.

        Args:
            prompt: User prompt
            system_prompt: System prompt

        Returns:
            Tuple of (response_text, model_used, tokens_used)
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        last_error = None
        for model in self.available_models:
            try:
                logger.info(f"Trying model: {model}")

                response = completion(
                    model=model,
                    messages=messages,
                    temperature=0.3,
                    max_tokens=2000,
                )

                content = response.choices[0].message.content

                # Extract tokens with multiple fallback methods
                tokens = 0
                if hasattr(response, 'usage') and response.usage:
                    usage = response.usage
                    # Try total_tokens first
                    if hasattr(usage, 'total_tokens') and usage.total_tokens:
                        tokens = usage.total_tokens
                    # Fallback: sum prompt + completion tokens
                    elif hasattr(usage, 'prompt_tokens') and hasattr(usage, 'completion_tokens'):
                        prompt_tokens = usage.prompt_tokens or 0
                        completion_tokens = usage.completion_tokens or 0
                        tokens = prompt_tokens + completion_tokens
                    # Try dict access as fallback
                    elif isinstance(usage, dict):
                        tokens = usage.get('total_tokens', 0) or (
                            usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0)
                        )

                    logger.debug(f"Token usage details: {usage}")

                # Track usage
                service = model.split("/")[0]  # Extract provider: zai, gemini, ollama
                get_tracker().track(service, count=1, tokens=tokens)

                logger.info(f"Success with {model} ({tokens} tokens)")
                return content, model, tokens

            except Exception as e:
                last_error = e
                logger.warning(f"{model} failed: {e}")
                continue

        raise Exception(f"All models failed. Last error: {last_error}")

    def _extract_json(self, text: str) -> Optional[Dict]:
        """Extract JSON from LLM response."""
        try:
            # Try direct parse first
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try to find JSON in response
        try:
            start = text.find('{')
            end = text.rfind('}') + 1
            if start != -1 and end > start:
                return json.loads(text[start:end])
        except json.JSONDecodeError:
            pass

        return None

    def _get_rag_context(
        self,
        symbol: str,
        mode: str,
        quote: Dict[str, Any],
        indicators: Dict[str, Any],
        news: List[Dict[str, Any]] = None
    ) -> tuple[str, List[Dict[str, Any]]]:
        """
        Get RAG context for enhanced analysis.

        Args:
            symbol: Stock symbol
            mode: Trading mode ('intraday' or 'longterm')
            quote: Current price quote
            indicators: Technical indicators
            news: Recent news articles

        Returns:
            Tuple of (formatted context string, raw retrieved documents)
        """
        if not self.enable_rag or not self.rag_retriever:
            return "", []

        try:
            # Get full context object to access raw documents
            rag_context = self.rag_retriever.retrieve_context(
                query=f"{symbol} {mode} analysis",
                stock_symbol=symbol,
                include_analyses=True,
                include_signals=True,
                include_news=True,
                include_knowledge=False,
                max_results_per_source=3,
                match_threshold=0.4
            )
            
            # Collect raw documents for validation
            raw_docs = []
            raw_docs.extend(rag_context.similar_analyses)
            raw_docs.extend(rag_context.similar_signals)
            raw_docs.extend(rag_context.relevant_news)
            
            # Format context string for prompt
            context_str = self.rag_retriever.format_context_for_prompt(rag_context)
            
            if context_str:
                logger.info(f"Retrieved RAG context for {symbol} ({len(context_str)} chars, {len(raw_docs)} docs)")
            
            return context_str, raw_docs
            
        except Exception as e:
            logger.warning(f"Failed to get RAG context: {e}")
            return "", []

    # -------------------------------------------------------------------------
    # Intraday Analysis
    # -------------------------------------------------------------------------

    def analyze_intraday(
        self,
        symbol: str,
        quote: Dict[str, Any],
        indicators: Dict[str, Any],
        news: List[Dict[str, Any]] = None,
        social_sentiment: Dict[str, Any] = None
    ) -> Optional[AnalysisResult]:
        """
        Analyze stock for intraday trading.

        Args:
            symbol: Stock symbol
            quote: Current price quote
            indicators: Technical indicators
            news: Recent news articles
            social_sentiment: Social media sentiment data

        Returns:
            AnalysisResult with trading signal
        """
        start_time = time.time()

        system_prompt = """You are an expert intraday stock trader and technical analyst.
Analyze the provided stock data and give a clear trading signal.
Focus on short-term price movements (minutes to hours).
Consider social media sentiment when available - high Reddit/Twitter buzz can indicate momentum.
Always respond with valid JSON only."""

        # Build news summary
        news_text = "No recent news."
        if news:
            news_items = [f"- {n.get('headline', '')}" for n in news[:5]]
            news_text = "\n".join(news_items)

        # Build social sentiment summary
        social_text = "No social data available."
        if social_sentiment:
            reddit_mentions = social_sentiment.get('reddit_mentions', 0)
            reddit_rank = social_sentiment.get('reddit_rank')
            overall = social_sentiment.get('overall_sentiment', 'neutral')
            
            if reddit_mentions > 0:
                social_text = f"""Reddit Mentions: {reddit_mentions}
Reddit Rank: #{reddit_rank if reddit_rank else 'Not ranked'}
Overall Social Sentiment: {overall}"""

        # Get RAG context for enhanced analysis
        rag_context, rag_docs = self._get_rag_context(
            symbol=symbol,
            mode="intraday",
            quote=quote,
            indicators=indicators,
            news=news
        )

        prompt = f"""Analyze this stock for INTRADAY trading:

STOCK: {symbol}

CURRENT PRICE DATA:
- Price: {quote.get('price')}
- Change: {quote.get('change_percent', 0):.2f}%
- Volume: {quote.get('volume', 0):,}
- Avg Volume: {quote.get('avg_volume', 0):,}
- Day High: {quote.get('high')}
- Day Low: {quote.get('low')}

TECHNICAL INDICATORS:
- RSI(14): {indicators.get('rsi_14', 'N/A')}
- MACD: {indicators.get('macd', 'N/A')}
- MACD Signal: {indicators.get('macd_signal', 'N/A')}
- SMA(20): {indicators.get('sma_20', 'N/A')}
- SMA(50): {indicators.get('sma_50', 'N/A')}
- Bollinger Upper: {indicators.get('bollinger_upper', 'N/A')}
- Bollinger Lower: {indicators.get('bollinger_lower', 'N/A')}
- Price vs SMA20: {indicators.get('price_vs_sma20_pct', 'N/A')}%

RECENT NEWS:
{news_text}

SOCIAL MEDIA SENTIMENT:
{social_text}
{rag_context}

Provide your analysis in this exact JSON format:
{{
    "signal": "strong_buy|buy|hold|sell|strong_sell",
    "confidence": 0.0-1.0,
    "reasoning": "2-3 sentence explanation of your signal",
    "technical_summary": "Brief technical analysis summary",
    "sentiment_summary": "Brief news/sentiment summary including social media buzz if relevant, or null",
    "support_level": price_number_or_null,
    "resistance_level": price_number_or_null,
    "target_price": price_number_or_null,
    "stop_loss": price_number_or_null,
    "risk_reward_ratio": number_or_null
}}

Respond with JSON only:"""

        try:
            response_text, model_used, tokens = self._call_llm(prompt, system_prompt)
            analysis = self._extract_json(response_text)

            if not analysis:
                logger.error("Failed to parse analysis JSON")
                return None

            duration_ms = int((time.time() - start_time) * 1000)
            
            # Run RAG validation if we have context
            rag_validation_result = None
            if rag_docs and self.rag_validator:
                try:
                    validation = self.rag_validator.validate_analysis(
                        query=f"{symbol} intraday analysis",
                        answer=analysis.get("reasoning", "") + " " + analysis.get("technical_summary", ""),
                        retrieved_context=rag_docs,
                        analysis_mode="intraday",
                        source_data={"quote": quote, "indicators": indicators}
                    )
                    rag_validation_result = validation.to_dict()
                    logger.info(f"RAG validation for {symbol}: grade={validation.quality_grade}, score={validation.overall_score:.1f}")
                except Exception as ve:
                    logger.warning(f"RAG validation failed: {ve}")

            return AnalysisResult(
                symbol=symbol,
                mode=TradingMode.INTRADAY,
                signal=Signal(analysis.get("signal", "hold")),
                confidence=float(analysis.get("confidence", 0.5)),
                reasoning=analysis.get("reasoning", ""),
                technical_summary=analysis.get("technical_summary", ""),
                sentiment_summary=analysis.get("sentiment_summary"),
                support_level=analysis.get("support_level"),
                resistance_level=analysis.get("resistance_level"),
                target_price=analysis.get("target_price"),
                stop_loss=analysis.get("stop_loss"),
                risk_reward_ratio=analysis.get("risk_reward_ratio"),
                llm_model=model_used,
                tokens_used=tokens,
                analysis_duration_ms=duration_ms,
                created_at=datetime.now(timezone.utc).isoformat(),
                rag_validation=rag_validation_result
            )

        except Exception as e:
            logger.error(f"Intraday analysis failed for {symbol}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Long-term Analysis
    # -------------------------------------------------------------------------

    def analyze_longterm(
        self,
        symbol: str,
        quote: Dict[str, Any],
        fundamentals: Dict[str, Any],
        indicators: Dict[str, Any],
        news: List[Dict[str, Any]] = None
    ) -> Optional[AnalysisResult]:
        """
        Analyze stock for long-term investing.

        Args:
            symbol: Stock symbol
            quote: Current price quote
            fundamentals: Fundamental data (P/E, revenue, etc.)
            indicators: Technical indicators
            news: Recent news articles

        Returns:
            AnalysisResult with investment signal
        """
        start_time = time.time()

        system_prompt = """You are an expert long-term investor and fundamental analyst.
Analyze the provided stock data and give a clear investment recommendation.
Focus on company fundamentals, valuation, and growth prospects.
Consider both value and growth perspectives.
Always respond with valid JSON only."""

        # Build news summary
        news_text = "No recent news."
        if news:
            news_items = [f"- {n.get('headline', '')}" for n in news[:5]]
            news_text = "\n".join(news_items)

        # Get RAG context for enhanced analysis
        rag_context, rag_docs = self._get_rag_context(
            symbol=symbol,
            mode="longterm",
            quote=quote,
            indicators=indicators,
            news=news
        )

        prompt = f"""Analyze this stock for LONG-TERM investing:

STOCK: {symbol}
COMPANY: {fundamentals.get('name', symbol)}
SECTOR: {fundamentals.get('sector', 'Unknown')}

CURRENT PRICE:
- Price: {quote.get('price')}
- 52-Week High: {quote.get('fifty_two_week_high')}
- 52-Week Low: {quote.get('fifty_two_week_low')}
- Market Cap: {fundamentals.get('market_cap', 'N/A')}

VALUATION METRICS:
- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A')}
- Forward P/E: {fundamentals.get('forward_pe', 'N/A')}
- PEG Ratio: {fundamentals.get('peg_ratio', 'N/A')}
- P/B Ratio: {fundamentals.get('pb_ratio', 'N/A')}
- P/S Ratio: {fundamentals.get('ps_ratio', 'N/A')}

PROFITABILITY:
- Profit Margin: {fundamentals.get('profit_margin', 'N/A')}
- Operating Margin: {fundamentals.get('operating_margin', 'N/A')}
- ROE: {fundamentals.get('roe', 'N/A')}
- ROA: {fundamentals.get('roa', 'N/A')}

GROWTH:
- Revenue Growth: {fundamentals.get('revenue_growth', 'N/A')}
- Earnings Growth: {fundamentals.get('earnings_growth', 'N/A')}

FINANCIAL HEALTH:
- Current Ratio: {fundamentals.get('current_ratio', 'N/A')}
- Debt/Equity: {fundamentals.get('debt_to_equity', 'N/A')}
- Free Cash Flow: {fundamentals.get('free_cash_flow', 'N/A')}

DIVIDENDS:
- Dividend Yield: {fundamentals.get('dividend_yield', 'N/A')}
- Payout Ratio: {fundamentals.get('payout_ratio', 'N/A')}

ANALYST CONSENSUS:
- Target Price (Mean): {fundamentals.get('target_mean', 'N/A')}
- Recommendation: {fundamentals.get('recommendation', 'N/A')}

TECHNICAL (Weekly):
- RSI(14): {indicators.get('rsi_14', 'N/A')}
- Price vs SMA(50): {indicators.get('price_vs_sma50_pct', 'N/A')}%
- SMA(200): {indicators.get('sma_200', 'N/A')}

RECENT NEWS:
{news_text}
{rag_context}

Provide your investment analysis in this exact JSON format:
{{
    "signal": "strong_buy|buy|hold|sell|strong_sell",
    "confidence": 0.0-1.0,
    "reasoning": "3-4 sentence explanation of your recommendation",
    "technical_summary": "Brief technical outlook",
    "sentiment_summary": "Brief fundamental/news assessment",
    "support_level": price_number_or_null,
    "resistance_level": price_number_or_null,
    "target_price": 12_month_target_or_null,
    "stop_loss": suggested_exit_price_or_null,
    "risk_reward_ratio": number_or_null
}}

Respond with JSON only:"""

        try:
            response_text, model_used, tokens = self._call_llm(prompt, system_prompt)
            analysis = self._extract_json(response_text)

            if not analysis:
                logger.error("Failed to parse analysis JSON")
                return None

            duration_ms = int((time.time() - start_time) * 1000)
            
            # Run RAG validation if we have context
            rag_validation_result = None
            if rag_docs and self.rag_validator:
                try:
                    validation = self.rag_validator.validate_analysis(
                        query=f"{symbol} longterm analysis",
                        answer=analysis.get("reasoning", "") + " " + analysis.get("technical_summary", ""),
                        retrieved_context=rag_docs,
                        analysis_mode="longterm",
                        source_data={"quote": quote, "indicators": indicators, "fundamentals": fundamentals}
                    )
                    rag_validation_result = validation.to_dict()
                    logger.info(f"RAG validation for {symbol}: grade={validation.quality_grade}, score={validation.overall_score:.1f}")
                except Exception as ve:
                    logger.warning(f"RAG validation failed: {ve}")

            return AnalysisResult(
                symbol=symbol,
                mode=TradingMode.LONGTERM,
                signal=Signal(analysis.get("signal", "hold")),
                confidence=float(analysis.get("confidence", 0.5)),
                reasoning=analysis.get("reasoning", ""),
                technical_summary=analysis.get("technical_summary", ""),
                sentiment_summary=analysis.get("sentiment_summary"),
                support_level=analysis.get("support_level"),
                resistance_level=analysis.get("resistance_level"),
                target_price=analysis.get("target_price"),
                stop_loss=analysis.get("stop_loss"),
                risk_reward_ratio=analysis.get("risk_reward_ratio"),
                llm_model=model_used,
                tokens_used=tokens,
                analysis_duration_ms=duration_ms,
                created_at=datetime.now(timezone.utc).isoformat(),
                rag_validation=rag_validation_result
            )

        except Exception as e:
            logger.error(f"Long-term analysis failed for {symbol}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Why Is It Moving?
    # -------------------------------------------------------------------------

    def explain_movement(
        self,
        symbol: str,
        price_change_pct: float,
        news: List[Dict[str, Any]],
        volume_ratio: float = 1.0
    ) -> Optional[str]:
        """
        Explain why a stock is moving significantly.

        Args:
            symbol: Stock symbol
            price_change_pct: Price change percentage
            news: Recent news articles
            volume_ratio: Current volume / average volume

        Returns:
            Explanation string
        """
        direction = "UP" if price_change_pct > 0 else "DOWN"

        news_text = "No recent news found."
        if news:
            news_items = [f"- {n.get('headline', '')} ({n.get('source', 'Unknown')})"
                         for n in news[:7]]
            news_text = "\n".join(news_items)

        prompt = f"""Explain why this stock is moving:

STOCK: {symbol}
MOVEMENT: {direction} {abs(price_change_pct):.2f}%
VOLUME: {volume_ratio:.1f}x average

RECENT NEWS:
{news_text}

Provide a brief, clear explanation (2-3 sentences) of the likely catalyst for this move.
If the news doesn't explain it, mention possible reasons (sector movement, market sentiment, technical breakout, etc.)."""

        try:
            response_text, _, _ = self._call_llm(prompt)
            return response_text.strip()
        except Exception as e:
            logger.error(f"Movement explanation failed: {e}")
            return None

    # -------------------------------------------------------------------------
    # Verification (Ollama Inspector)
    # -------------------------------------------------------------------------

    def verify_analysis(
        self,
        analysis: AnalysisResult,
        quote: Dict[str, Any],
        indicators: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Use Ollama to verify/validate an analysis (inspector role).

        Args:
            analysis: The analysis to verify
            quote: Current price data
            indicators: Technical indicators

        Returns:
            Verification result with confidence adjustment
        """
        prompt = f"""You are verifying a stock analysis. Check if the reasoning is sound.

ANALYSIS TO VERIFY:
- Stock: {analysis.symbol}
- Signal: {analysis.signal.value}
- Confidence: {analysis.confidence}
- Reasoning: {analysis.reasoning}
- Support: {analysis.support_level}
- Resistance: {analysis.resistance_level}
- Target: {analysis.target_price}
- Stop Loss: {analysis.stop_loss}

CURRENT DATA:
- Price: {quote.get('price')}
- RSI: {indicators.get('rsi_14')}
- MACD: {indicators.get('macd')}

Verify if:
1. The signal matches the technical data
2. Support/resistance levels are reasonable
3. Risk/reward makes sense

Respond with JSON:
{{
    "is_valid": true/false,
    "confidence_adjustment": -0.2 to +0.2,
    "concerns": ["list of concerns if any"],
    "verification_note": "brief note"
}}"""

        try:
            # Force use Ollama for verification
            response = completion(
                model="ollama/mistral",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=500,
            )

            result = self._extract_json(response.choices[0].message.content)
            return result or {"is_valid": True, "confidence_adjustment": 0}

        except Exception as e:
            logger.warning(f"Verification failed (using original): {e}")
            return {"is_valid": True, "confidence_adjustment": 0}

    # -------------------------------------------------------------------------
    # Algo Trading Prediction
    # -------------------------------------------------------------------------

    def generate_algo_prediction(
        self,
        symbol: str,
        quote: Dict[str, Any],
        indicators: Dict[str, Any],
        fundamentals: Dict[str, Any] = None,
        news: List[Dict[str, Any]] = None,
        price_history_days: int = 0
    ) -> Optional[Dict[str, Any]]:
        """
        Generate algo trading prediction with ALGORITHMIC scores + LLM reasoning.
        
        Scores are calculated using mathemat formulas (not AI-generated):
        - Momentum: RSI, MACD, price trends
        - Value: P/E, P/B, dividend yield
        - Quality: ROE, profit margins, debt/equity
        - Risk: Volatility, debt, profitability
        - Confidence: Based on data availability
        
        LLM is only used for generating the reasoning explanation.
        
        Args:
            symbol: Stock symbol
            quote: Current price data
            indicators: Technical indicators
            fundamentals: Company fundamentals (optional)
            news: Recent news (optional)
            price_history_days: Number of days of price history available
            
        Returns:
            Algo prediction with scores, predicted returns, and reasoning
        """
        start_time = time.time()
        
        # Step 1: Calculate algorithmic scores (no LLM involved)
        scorer = StockScorer()
        algo_scores = scorer.calculate_all_scores(
            quote=quote,
            indicators=indicators,
            fundamentals=fundamentals,
            price_history_days=price_history_days,
            has_news=bool(news)
        )
        
        logger.info(f"Algorithmic scores for {symbol}: M={algo_scores.momentum_score}, V={algo_scores.value_score}, Q={algo_scores.quality_score}, R={algo_scores.risk_score}")
        
        # Step 2: Use LLM only for generating reasoning explanation
        system_prompt = """You are a stock analyst. Given the calculated scores, provide a brief explanation of what they mean for this stock.
Do NOT make up new scores or predictions - use the provided scores.
Always respond with valid JSON only."""

        # Build context for LLM
        fund_summary = "No fundamentals data."
        if fundamentals:
            pe = fundamentals.get('pe_ratio', 'N/A')
            pb = fundamentals.get('pb_ratio', 'N/A')
            roe = fundamentals.get('roe')
            roe_str = f"{roe * 100:.1f}%" if roe else "N/A"
            fund_summary = f"P/E: {pe}, P/B: {pb}, ROE: {roe_str}"

        prompt = f"""Explain these algorithmically-calculated scores for {symbol}:

CALCULATED SCORES (DO NOT CHANGE THESE):
- Signal: {algo_scores.overall_signal.upper()}
- Momentum Score: {algo_scores.momentum_score}/100
- Value Score: {algo_scores.value_score}/100
- Quality Score: {algo_scores.quality_score}/100
- Risk Score: {algo_scores.risk_score}/10
- Confidence: {algo_scores.confidence_score}%

KEY DATA:
- Current Price: {quote.get('price')}
- RSI: {indicators.get('rsi_14', 'N/A')}
- MACD vs Signal: {indicators.get('macd', 'N/A')} vs {indicators.get('macd_signal', 'N/A')}
- Fundamentals: {fund_summary}

Provide explanation in this JSON format:
{{
    "reasoning": [
        "Brief explanation of the signal based on scores",
        "Key momentum/technical insight",
        "Key fundamental insight if available"
    ],
    "key_factors": [
        {{"name": "Most important factor", "impact": "positive|negative|neutral"}},
        {{"name": "Second factor", "impact": "positive|negative|neutral"}}
    ],
    "predicted_return_30d": estimate_percentage_based_on_scores,
    "predicted_return_90d": estimate_percentage_based_on_scores
}}

Respond with JSON only:"""

        try:
            response_text, model_used, tokens = self._call_llm(prompt, system_prompt)
            llm_response = self._extract_json(response_text)
            
            if not llm_response:
                llm_response = {
                    "reasoning": [f"Signal: {algo_scores.overall_signal.upper()} based on algorithmic analysis"],
                    "key_factors": [],
                    "predicted_return_30d": 0,
                    "predicted_return_90d": 0
                }
            
            # Build final prediction with algorithmic scores + LLM reasoning
            prediction = {
                # Algorithmic scores (calculated, not LLM-generated)
                "signal": algo_scores.overall_signal,
                "confidence": algo_scores.confidence_score / 100,  # Convert to 0-1
                "momentum_score": algo_scores.momentum_score,
                "value_score": algo_scores.value_score,
                "quality_score": algo_scores.quality_score,
                "risk_score": algo_scores.risk_score,
                
                # Score breakdowns for transparency
                "score_breakdowns": {
                    "momentum": algo_scores.momentum_breakdown,
                    "value": algo_scores.value_breakdown,
                    "quality": algo_scores.quality_breakdown,
                    "risk": algo_scores.risk_breakdown,
                },
                
                # LLM-generated explanation
                "reasoning": llm_response.get("reasoning", []),
                "key_factors": llm_response.get("key_factors", []),
                "predicted_return_30d": llm_response.get("predicted_return_30d", 0),
                "predicted_return_90d": llm_response.get("predicted_return_90d", 0),
                
                # Metadata
                "scoring_method": "algorithmic",  # Flag to indicate formula-based
                "llm_model": model_used,
                "tokens_used": tokens,
                "analysis_duration_ms": int((time.time() - start_time) * 1000),
                "created_at": datetime.now(timezone.utc).isoformat()
            }
            
            logger.info(f"Algo prediction for {symbol}: {prediction['signal']} (algorithmic)")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Algo prediction failed for {symbol}: {e}")
            # Return algorithmic scores even if LLM fails
            return {
                "signal": algo_scores.overall_signal,
                "confidence": algo_scores.confidence_score / 100,
                "momentum_score": algo_scores.momentum_score,
                "value_score": algo_scores.value_score,
                "quality_score": algo_scores.quality_score,
                "risk_score": algo_scores.risk_score,
                "scoring_method": "algorithmic",
                "reasoning": ["Scores calculated algorithmically from financial data"],
                "error": str(e),
                "created_at": datetime.now(timezone.utc).isoformat()
            }


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    analyzer = StockAnalyzer()

    # Test data
    test_quote = {
        "price": 2847.50,
        "change_percent": 1.2,
        "volume": 2300000,
        "avg_volume": 1500000,
        "high": 2860,
        "low": 2820,
        "fifty_two_week_high": 3000,
        "fifty_two_week_low": 2200
    }

    test_indicators = {
        "rsi_14": 62,
        "macd": 15.5,
        "macd_signal": 12.3,
        "sma_20": 2800,
        "sma_50": 2750,
        "bollinger_upper": 2900,
        "bollinger_lower": 2700,
        "price_vs_sma20_pct": 1.7
    }

    test_news = [
        {"headline": "Reliance Jio announces 5G expansion plans"},
        {"headline": "Oil prices rise on supply concerns"},
    ]

    print("\n" + "=" * 60)
    print("Testing StockAnalyzer - Intraday")
    print("=" * 60)

    result = analyzer.analyze_intraday(
        symbol="RELIANCE.NS",
        quote=test_quote,
        indicators=test_indicators,
        news=test_news
    )

    if result:
        print(f"\nSignal: {result.signal.value.upper()}")
        print(f"Confidence: {result.confidence:.0%}")
        print(f"Reasoning: {result.reasoning}")
        print(f"Model: {result.llm_model}")
        print(f"Duration: {result.analysis_duration_ms}ms")
    else:
        print("Analysis failed")
