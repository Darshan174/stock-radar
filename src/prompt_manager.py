"""
Stock Radar - Prompt Versioning & A/B Testing.

Manages versioned prompt templates so you can iterate on prompts
systematically instead of editing strings in code.

WHY THIS MATTERS (AI Engineering):
- Prompts are the "source code" of AI applications.
- Changing a single word can dramatically change output quality.
- Without versioning, you can't compare results or roll back.
- Interviewers ask: "How do you iterate on prompts?"

HOW IT WORKS:
    1. Prompts are stored as versioned templates (v1, v2, v3...)
    2. The active version is set via PROMPT_VERSION env var
    3. Each version is a complete prompt with {placeholders}
    4. A/B testing: run the same analysis with two versions, compare results

USAGE:
    from prompt_manager import prompt_manager

    # Get the active prompt for intraday analysis
    prompt = prompt_manager.get_prompt(
        "intraday_analysis",
        symbol="AAPL",
        price=150.0,
        rsi=65,
    )

    # Get a specific version for A/B testing
    prompt_v2 = prompt_manager.get_prompt(
        "intraday_analysis",
        version="v2",
        symbol="AAPL",
        price=150.0,
    )
"""

from __future__ import annotations

from typing import Any


# =============================================================================
# Prompt Templates Registry
# =============================================================================

PROMPTS: dict[str, dict[str, dict[str, str]]] = {
    # -----------------------------------------------------------------
    # INTRADAY ANALYSIS
    # -----------------------------------------------------------------
    "intraday_analysis": {
        "v1": {
            "system": (
                "You are an expert intraday stock trader and technical analyst.\n"
                "Analyze the provided stock data and give a clear trading signal.\n"
                "Focus on short-term price movements (minutes to hours).\n"
                "Consider social media sentiment when available.\n"
                "Always respond with valid JSON only."
            ),
            "user": (
                "Analyze this stock for INTRADAY trading:\n\n"
                "STOCK: {symbol}\n\n"
                "CURRENT PRICE DATA:\n"
                "- Price: {price}\n"
                "- Change: {change_percent:.2f}%\n"
                "- Volume: {volume:,}\n"
                "- Avg Volume: {avg_volume:,}\n"
                "- Day High: {high}\n"
                "- Day Low: {low}\n\n"
                "TECHNICAL INDICATORS:\n"
                "- RSI(14): {rsi_14}\n"
                "- MACD: {macd}\n"
                "- MACD Signal: {macd_signal}\n"
                "- SMA(20): {sma_20}\n"
                "- SMA(50): {sma_50}\n"
                "- Bollinger Upper: {bollinger_upper}\n"
                "- Bollinger Lower: {bollinger_lower}\n"
                "- Price vs SMA20: {price_vs_sma20_pct}%\n\n"
                "RECENT NEWS:\n{news_text}\n\n"
                "SOCIAL MEDIA SENTIMENT:\n{social_text}\n"
                "{rag_context}\n\n"
                "Provide your analysis in this exact JSON format:\n"
                '{{"signal": "strong_buy|buy|hold|sell|strong_sell",'
                ' "confidence": 0.0-1.0,'
                ' "reasoning": "2-3 sentence explanation",'
                ' "technical_summary": "Brief technical summary",'
                ' "sentiment_summary": "Brief sentiment summary or null",'
                ' "support_level": price_or_null,'
                ' "resistance_level": price_or_null,'
                ' "target_price": price_or_null,'
                ' "stop_loss": price_or_null,'
                ' "risk_reward_ratio": number_or_null}}\n\n'
                "Respond with JSON only:"
            ),
        },
        "v2": {
            "system": (
                "You are a senior quantitative trader. You combine technical analysis "
                "with sentiment data to generate precise intraday signals.\n\n"
                "RULES:\n"
                "1. Only use data provided - do not hallucinate prices or indicators.\n"
                "2. If data is insufficient, signal 'hold' with low confidence.\n"
                "3. Always explain your reasoning step-by-step.\n"
                "4. Cite specific indicator values in your reasoning.\n"
                "5. Respond with valid JSON only."
            ),
            "user": (
                "## Intraday Analysis Request\n\n"
                "**Stock:** {symbol}\n"
                "**Price:** {price} | **Change:** {change_percent:.2f}%\n"
                "**Volume:** {volume:,} (avg: {avg_volume:,})\n"
                "**Range:** {low} - {high}\n\n"
                "### Technical Indicators\n"
                "| Indicator | Value |\n"
                "|-----------|-------|\n"
                "| RSI(14) | {rsi_14} |\n"
                "| MACD | {macd} |\n"
                "| MACD Signal | {macd_signal} |\n"
                "| SMA(20) | {sma_20} |\n"
                "| SMA(50) | {sma_50} |\n"
                "| Bollinger Upper | {bollinger_upper} |\n"
                "| Bollinger Lower | {bollinger_lower} |\n"
                "| Price vs SMA20 | {price_vs_sma20_pct}% |\n\n"
                "### News\n{news_text}\n\n"
                "### Social Sentiment\n{social_text}\n\n"
                "### Historical Context (RAG)\n{rag_context}\n\n"
                "Think step-by-step, then respond with JSON:\n"
                '{{"signal": "...", "confidence": 0.0-1.0, '
                '"reasoning": "step-by-step analysis", '
                '"technical_summary": "...", "sentiment_summary": "...", '
                '"support_level": ..., "resistance_level": ..., '
                '"target_price": ..., "stop_loss": ..., '
                '"risk_reward_ratio": ...}}'
            ),
        },
    },
    # -----------------------------------------------------------------
    # LONGTERM ANALYSIS
    # -----------------------------------------------------------------
    "longterm_analysis": {
        "v1": {
            "system": (
                "You are an expert long-term investor and fundamental analyst.\n"
                "Analyze the provided stock data and give a clear investment recommendation.\n"
                "Focus on company fundamentals, valuation, and growth prospects.\n"
                "Always respond with valid JSON only."
            ),
            "user": (
                "Analyze this stock for LONG-TERM investing:\n\n"
                "STOCK: {symbol}\n"
                "COMPANY: {company_name}\n"
                "SECTOR: {sector}\n\n"
                "CURRENT PRICE:\n"
                "- Price: {price}\n"
                "- 52-Week High: {fifty_two_week_high}\n"
                "- 52-Week Low: {fifty_two_week_low}\n\n"
                "VALUATION: P/E={pe_ratio}, Fwd P/E={forward_pe}, "
                "PEG={peg_ratio}, P/B={pb_ratio}\n\n"
                "PROFITABILITY: Margin={profit_margin}, ROE={roe}, ROA={roa}\n\n"
                "GROWTH: Revenue={revenue_growth}, Earnings={earnings_growth}\n\n"
                "HEALTH: Current Ratio={current_ratio}, D/E={debt_to_equity}\n\n"
                "DIVIDENDS: Yield={dividend_yield}\n\n"
                "ANALYST: Target={target_mean}, Rec={recommendation}\n\n"
                "TECHNICAL: RSI={rsi_14}, vs SMA50={price_vs_sma50_pct}%\n\n"
                "NEWS:\n{news_text}\n"
                "{rag_context}\n\n"
                "Respond with JSON:\n"
                '{{"signal": "...", "confidence": 0.0-1.0, '
                '"reasoning": "3-4 sentences", "technical_summary": "...", '
                '"sentiment_summary": "...", "support_level": ..., '
                '"resistance_level": ..., "target_price": ..., '
                '"stop_loss": ..., "risk_reward_ratio": ...}}'
            ),
        },
        "v2": {
            "system": (
                "You are a senior fundamental analyst at a top investment firm.\n\n"
                "RULES:\n"
                "1. Base your analysis only on the data provided.\n"
                "2. Compare valuation metrics to sector averages mentally.\n"
                "3. Weight fundamentals 60%, technicals 20%, sentiment 20%.\n"
                "4. Cite specific numbers from the data in your reasoning.\n"
                "5. Respond with valid JSON only."
            ),
            "user": (
                "## Long-Term Investment Analysis\n\n"
                "**{company_name}** ({symbol}) | Sector: {sector}\n"
                "Price: {price} | 52W: {fifty_two_week_low} - {fifty_two_week_high}\n\n"
                "### Valuation\n"
                "P/E: {pe_ratio} | Fwd P/E: {forward_pe} | PEG: {peg_ratio} | P/B: {pb_ratio}\n\n"
                "### Profitability\n"
                "Margin: {profit_margin} | ROE: {roe} | ROA: {roa}\n\n"
                "### Growth\n"
                "Revenue: {revenue_growth} | Earnings: {earnings_growth}\n\n"
                "### Financial Health\n"
                "Current Ratio: {current_ratio} | D/E: {debt_to_equity}\n\n"
                "### Dividends\n"
                "Yield: {dividend_yield}\n\n"
                "### Analyst Consensus\n"
                "Target: {target_mean} | Recommendation: {recommendation}\n\n"
                "### Technical Snapshot\n"
                "RSI: {rsi_14} | vs SMA50: {price_vs_sma50_pct}%\n\n"
                "### News\n{news_text}\n\n"
                "### Historical Context\n{rag_context}\n\n"
                "Analyze step-by-step, then respond with JSON."
            ),
        },
    },
    # -----------------------------------------------------------------
    # ALGO PREDICTION EXPLANATION
    # -----------------------------------------------------------------
    "algo_explanation": {
        "v1": {
            "system": (
                "You are a stock analyst. Given calculated scores, provide a brief "
                "explanation. Do NOT make up new scores - use the provided ones.\n"
                "Always respond with valid JSON only."
            ),
            "user": (
                "Explain these scores for {symbol}:\n\n"
                "Signal: {signal} | Momentum: {momentum_score}/100 | "
                "Value: {value_score}/100 | Quality: {quality_score}/100 | "
                "Risk: {risk_score}/10 | Confidence: {confidence_score}%\n\n"
                "Price: {price} | RSI: {rsi_14} | MACD: {macd} vs {macd_signal}\n"
                "Fundamentals: {fund_summary}\n\n"
                "Respond with JSON:\n"
                '{{"reasoning": ["point 1", "point 2", "point 3"],'
                ' "key_factors": [{{"name": "...", "impact": "positive|negative|neutral"}}],'
                ' "predicted_return_30d": number, "predicted_return_90d": number}}'
            ),
        },
    },
}


class PromptManager:
    """
    Manages versioned prompt templates.

    Loads the active version from config, allows overrides per-request,
    and supports A/B testing by running two versions side by side.
    """

    def __init__(self, default_version: str = "v1") -> None:
        self.default_version = default_version

    def get_prompt(
        self,
        prompt_name: str,
        version: str | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        """
        Get a formatted prompt by name and version.

        Args:
            prompt_name: Name of the prompt template (e.g., "intraday_analysis")
            version: Version to use (e.g., "v1", "v2"). Defaults to active version.
            **kwargs: Values to fill into the template placeholders.

        Returns:
            Dict with "system" and "user" keys containing formatted prompts.
        """
        version = version or self.default_version

        templates = PROMPTS.get(prompt_name)
        if templates is None:
            raise ValueError(f"Unknown prompt: {prompt_name}")

        version_templates = templates.get(version)
        if version_templates is None:
            # Fall back to v1
            version_templates = templates.get("v1")
            if version_templates is None:
                raise ValueError(f"No templates found for {prompt_name}")

        # Format with kwargs, using 'N/A' for missing values
        safe_kwargs = _safe_format_args(kwargs)

        return {
            "system": version_templates["system"].format_map(safe_kwargs),
            "user": version_templates["user"].format_map(safe_kwargs),
        }

    def list_prompts(self) -> dict[str, list[str]]:
        """List all available prompts and their versions."""
        return {name: list(versions.keys()) for name, versions in PROMPTS.items()}

    def get_version(self) -> str:
        """Get the currently active prompt version."""
        return self.default_version


class _SafeDict(dict):
    """Dict that returns 'N/A' for missing keys instead of raising KeyError."""

    def __missing__(self, key: str) -> str:
        return "N/A"


def _safe_format_args(kwargs: dict[str, Any]) -> _SafeDict:
    """Wrap kwargs so missing template vars get 'N/A' instead of errors."""
    return _SafeDict(kwargs)


# Singleton
try:
    from config import settings
    prompt_manager = PromptManager(default_version=settings.prompt_version)
except Exception:
    prompt_manager = PromptManager(default_version="v1")
