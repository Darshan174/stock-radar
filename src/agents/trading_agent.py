"""
Trading Agent — autonomous signal generation with risk management.

Workflow:
  1. Check kill switches (is the system safe to trade?)
  2. Analyze stock (get signal + scores)
  3. Check market regime
  4. Calculate position size
  5. Run pre-trade risk checks
  6. Execute paper trade if all checks pass

Paper trading only.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.react_engine import AgentResult, ReActEngine, ToolDefinition

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
You are Stock Radar's Trading Agent — an autonomous trader that generates \
signals and executes paper trades with full risk management.

AVAILABLE TOOLS:
  - analyze_stock: run full analysis to get signal, scores, and indicators
  - check_regime: detect market regime (trending / mean-reverting / high-vol / neutral)
  - calculate_position_size: risk-aware position sizing based on signal + confidence + volatility
  - check_pre_trade_risk: validate the proposed position against portfolio limits
  - check_kill_switches: check if any safety switches are triggered
  - place_paper_trade: execute a paper trade (simulation only)
  - get_portfolio: see current open positions and P&L

YOUR WORKFLOW (follow this order):
1. First call check_kill_switches. If halted=true, STOP and explain why.
2. Call analyze_stock to get the signal and confidence.
3. If signal is "hold" or confidence < 0.35, do NOT trade. Explain why.
4. Call check_regime to understand market conditions.
5. Call calculate_position_size with the signal, confidence, and volatility.
6. Call check_pre_trade_risk with the proposed position size.
7. If blocked=true, do NOT trade. Explain which risk check failed.
8. If all checks pass, call place_paper_trade.
9. Summarize: what you traded, why, position size, stop loss, take profit.

CRITICAL RULES:
- NEVER skip risk checks. Safety first.
- NEVER trade if kill switches are triggered.
- This is PAPER TRADING only — simulated money.
- Always explain your reasoning at each step.
- If you're unsure, err on the side of not trading.
"""


class TradingAgent:
    """Autonomous trading agent with risk management. Paper trading only."""

    def __init__(self, model: str | None = None) -> None:
        from services.fetcher import StockFetcher
        from services.scorer import StockScorer

        self._fetcher = StockFetcher()
        self._scorer = StockScorer()

        # Paper portfolio
        try:
            from config import settings
            paper_dir = settings.paper_trading_dir
        except Exception:
            paper_dir = "data/paper_trading"

        from training.paper_trading import PaperPortfolio
        self._portfolio = PaperPortfolio(paper_dir=paper_dir)

        self.engine = ReActEngine(
            model=model,
            tools=self._build_tools(),
            system_prompt=SYSTEM_PROMPT,
            max_steps=12,
        )

    def trade(self, instruction: str) -> AgentResult:
        return self.engine.run(instruction)

    def trade_stream(self, instruction: str):
        return self.engine.run_stream(instruction)

    # ------------------------------------------------------------------
    #  Tool definitions
    # ------------------------------------------------------------------

    def _build_tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="analyze_stock",
                description=(
                    "Run full analysis on a stock: technical indicators, "
                    "algorithmic scores (momentum/value/quality/risk 0-100), "
                    "and an overall signal (strong_buy/buy/hold/sell/strong_sell)."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                        "mode": {
                            "type": "string",
                            "enum": ["intraday", "longterm"],
                            "description": "Analysis mode (default: intraday)",
                        },
                    },
                    "required": ["symbol"],
                },
                function=self._tool_analyze_stock,
            ),
            ToolDefinition(
                name="check_regime",
                description=(
                    "Detect the current market regime for a stock: "
                    "trending, mean_reverting, high_volatility, or neutral. "
                    "Also returns regime confidence."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                    },
                    "required": ["symbol"],
                },
                function=self._tool_check_regime,
            ),
            ToolDefinition(
                name="calculate_position_size",
                description=(
                    "Calculate risk-aware position size based on signal, "
                    "confidence, volatility, and market regime. Returns "
                    "position_size (0-1 scale), direction, and reasoning."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "signal": {
                            "type": "string",
                            "enum": ["strong_buy", "buy", "hold", "sell", "strong_sell"],
                        },
                        "confidence": {"type": "number", "description": "0.0 to 1.0"},
                        "volatility_pct": {
                            "type": "number",
                            "description": "ATR as percentage of price (optional)",
                        },
                        "regime": {
                            "type": "string",
                            "description": "Market regime from check_regime (optional)",
                        },
                    },
                    "required": ["signal", "confidence"],
                },
                function=self._tool_calculate_position_size,
            ),
            ToolDefinition(
                name="check_pre_trade_risk",
                description=(
                    "Run pre-trade risk checks: position limit, sector exposure, "
                    "daily loss limit, total exposure limit. Returns blocked=true "
                    "if any check fails."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                        "proposed_size": {
                            "type": "number",
                            "description": "Proposed position size (0-1 scale)",
                        },
                    },
                    "required": ["symbol", "proposed_size"],
                },
                function=self._tool_check_pre_trade_risk,
            ),
            ToolDefinition(
                name="check_kill_switches",
                description=(
                    "Check if any safety kill switches are triggered: "
                    "daily loss limit, stale data, excessive slippage. "
                    "If halted=true, do NOT trade."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                },
                function=self._tool_check_kill_switches,
            ),
            ToolDefinition(
                name="place_paper_trade",
                description=(
                    "Execute a paper trade (simulation). Records the trade "
                    "in the paper portfolio with entry price, size, stop loss, "
                    "and take profit."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string", "description": "Stock ticker"},
                        "signal": {
                            "type": "string",
                            "enum": ["strong_buy", "buy", "sell", "strong_sell"],
                            "description": "Trading signal (do not pass 'hold')",
                        },
                        "position_size": {
                            "type": "number",
                            "description": "Position size (0-1 scale)",
                        },
                        "confidence": {
                            "type": "number",
                            "description": "Signal confidence (0-1)",
                        },
                        "stop_loss": {
                            "type": "number",
                            "description": "Stop loss price (optional)",
                        },
                        "take_profit": {
                            "type": "number",
                            "description": "Take profit price (optional)",
                        },
                        "reason": {
                            "type": "string",
                            "description": "Brief reason for the trade",
                        },
                    },
                    "required": ["symbol", "signal", "position_size", "confidence", "reason"],
                },
                function=self._tool_place_paper_trade,
            ),
            ToolDefinition(
                name="get_portfolio",
                description=(
                    "Get current paper portfolio: open positions, "
                    "closed trades summary, and overall P&L."
                ),
                parameters={
                    "type": "object",
                    "properties": {},
                },
                function=self._tool_get_portfolio,
            ),
        ]

    # ------------------------------------------------------------------
    #  Tool implementations
    # ------------------------------------------------------------------

    def _tool_analyze_stock(self, symbol: str, mode: str = "intraday") -> dict[str, Any]:
        quote = self._fetcher.get_quote(symbol)
        if not quote:
            return {"error": f"No data available for {symbol}"}

        period = "6mo" if mode == "intraday" else "2y"
        history = self._fetcher.get_price_history(symbol, period=period)
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
            "price": quote.price,
            "change_pct": quote.change_percent,
            "signal": scores.overall_signal,
            "composite_score": scores.composite_score,
            "momentum_score": scores.momentum_score,
            "value_score": scores.value_score,
            "quality_score": scores.quality_score,
            "risk_score": scores.risk_score,
            "confidence": round(scores.confidence_score / 100, 2),
            "rsi_14": indicators.get("rsi_14"),
            "macd": indicators.get("macd"),
            "atr_pct": indicators.get("atr_pct"),
            "adx": indicators.get("adx"),
            "volume_ratio": indicators.get("volume_ratio"),
        }

    def _tool_check_regime(self, symbol: str) -> dict[str, Any]:
        from training.regime import classify_market_regime

        history = self._fetcher.get_price_history(symbol, period="6mo")
        indicators = self._fetcher.calculate_indicators(history) if history else {}
        return classify_market_regime(indicators)

    def _tool_calculate_position_size(
        self,
        signal: str,
        confidence: float,
        volatility_pct: float | None = None,
        regime: str | None = None,
    ) -> dict[str, Any]:
        from training.risk import calculate_position_size
        return calculate_position_size(
            signal=signal,
            confidence=confidence,
            volatility_pct=volatility_pct,
            regime=regime,
        )

    def _tool_check_pre_trade_risk(self, symbol: str, proposed_size: float) -> dict[str, Any]:
        from training.pre_trade_risk import check_all_pre_trade_risk

        positions = self._portfolio.get_open_positions()
        return check_all_pre_trade_risk(
            symbol=symbol,
            proposed_size=proposed_size,
            positions=positions,
        )

    def _tool_check_kill_switches(self) -> dict[str, Any]:
        from training.kill_switches import check_all_kill_switches

        closed = self._portfolio.get_closed_trades()
        positions = self._portfolio.get_open_positions()
        return check_all_kill_switches(
            closed_trades=closed,
            positions=positions,
        )

    def _tool_place_paper_trade(
        self,
        symbol: str,
        signal: str,
        position_size: float,
        confidence: float,
        reason: str,
        stop_loss: float | None = None,
        take_profit: float | None = None,
    ) -> dict[str, Any]:
        quote = self._fetcher.get_quote(symbol)
        if not quote:
            return {"error": f"Cannot get current price for {symbol}"}

        self._portfolio.record_signal(
            symbol=symbol,
            signal=signal,
            confidence=confidence,
            price=quote.price,
            position_size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )

        return {
            "status": "filled",
            "symbol": symbol,
            "signal": signal,
            "entry_price": quote.price,
            "position_size": position_size,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "reason": reason,
            "note": "PAPER TRADE — simulated execution",
        }

    def _tool_get_portfolio(self) -> dict[str, Any]:
        positions = self._portfolio.get_open_positions()
        summary = self._portfolio.get_performance_summary()
        return {
            "open_positions": len(positions),
            "positions": {
                sym: {
                    "signal": p.get("signal"),
                    "entry_price": p.get("price"),
                    "size": p.get("position_size"),
                }
                for sym, p in list(positions.items())[:10]
            },
            "performance": summary,
        }
