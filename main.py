#!/usr/bin/env python3
"""
Stock Radar - Main Orchestrator

Analyzes stocks for intraday and long-term traders with AI-powered insights.
Coordinates fetching, analysis, storage, and notifications.
"""

import os
import sys
import time
import logging
import argparse
from datetime import datetime, timezone
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from agents.fetcher import StockFetcher  # noqa: E402
from agents.analyzer import StockAnalyzer  # noqa: E402
from agents.storage import StockStorage  # noqa: E402
from agents.alerts import NotificationManager  # noqa: E402
from agents.usage_tracker import get_tracker  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class StockRadar:
    """
    Main orchestrator for Stock Radar.

    Coordinates the full analysis pipeline:
    1. Fetch stock data (prices, indicators, news)
    2. Run AI analysis (intraday or longterm)
    3. Verify analysis with Ollama inspector
    4. Store results in Supabase
    5. Send alerts to Slack/Telegram
    """

    def __init__(self):
        """Initialize all components."""
        logger.info("Initializing Stock Radar...")

        # Start metrics server in background
        try:
            from monitoring.server import start_server as start_metrics_server
            from config import settings
            self._metrics_server = start_metrics_server(
                port=settings.metrics_port, background=True
            )
            logger.info(f"Metrics server started on port {settings.metrics_port}")

            # Mark system as up
            from metrics import SYSTEM_UP
            SYSTEM_UP.set(1)
        except Exception as e:
            logger.warning(f"Could not start metrics server: {e}")

        self.fetcher = StockFetcher()
        self.analyzer = StockAnalyzer()
        self.storage = StockStorage()
        self.notifications = NotificationManager()

        # Start Finnhub WebSocket for real-time prices (free)
        try:
            from agents.realtime import get_realtime_manager
            self._realtime = get_realtime_manager()
            if self._realtime.start():
                logger.info("Finnhub real-time WebSocket started")
        except Exception as e:
            logger.debug(f"Real-time feed not available: {e}")

        # Verify schema on startup
        if not self.storage.ensure_schema():
            logger.warning("Database schema verification failed - some features may not work")

        logger.info("Stock Radar initialized successfully")
        logger.info(f"Active notification channels: {self.notifications.active_channels}")

    def analyze_stock(
        self,
        symbol: str,
        mode: str = "intraday",
        period: str = "max",
        send_alert: bool = True,
        verify: bool = True
    ) -> dict:
        """
        Run full analysis pipeline for a single stock.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS', 'AAPL')
            mode: Trading mode ('intraday' or 'longterm')
            period: Historical data period for analysis
            send_alert: Whether to send notifications
            verify: Whether to verify with Ollama inspector

        Returns:
            Analysis results dictionary
        """
        start_time = time.time()
        logger.info(f"Starting {mode} analysis for {symbol}")

        try:
            # Step 1: Fetch stock data
            logger.info(f"[1/5] Fetching data for {symbol}...")
            data = self.fetcher.get_full_analysis_data(symbol, period=period)

            if "error" in data:
                logger.error(f"Failed to fetch data: {data['error']}")
                return {"error": data["error"], "symbol": symbol}

            quote = data.get("quote")
            if not quote:
                logger.error(f"No quote data for {symbol}")
                return {"error": "No quote data available", "symbol": symbol}
            
            current_price = quote.price
            # Get company name from fundamentals or use symbol as fallback
            fundamentals = data.get("fundamentals", {}) or {}
            company_name = fundamentals.get("name") or symbol

            logger.info(f"   Current price: {current_price:.2f}")

            # Step 2: Run AI analysis
            logger.info(f"[2/5] Running {mode} AI analysis...")
            # TradingMode enum resolved inline in analyze_intraday/analyze_longterm calls

            # Prepare data for analyzer - convert StockQuote to dict if needed
            quote_dict = {
                "symbol": quote.symbol,
                "price": quote.price,
                "change": quote.change,
                "change_percent": quote.change_percent,
                "volume": quote.volume,
                "avg_volume": quote.avg_volume,
                "high": quote.high,
                "low": quote.low,
                "open": quote.open,
                "prev_close": quote.prev_close,
            }
            indicators = data.get("indicators", {})
            price_history = data.get("price_history", [])
            news = data.get("news", [])
            finnhub_sentiment = data.get("finnhub_sentiment")  # Phase-6
            
            # Fetch social sentiment (Reddit, Twitter)
            logger.info("[2.5/5] Fetching social media sentiment...")
            social_sentiment = self.fetcher.get_social_sentiment(symbol)
            if social_sentiment.get("total_mentions", 0) > 0:
                logger.info(f"   Reddit mentions: {social_sentiment.get('reddit_mentions', 0)}")
            
            # Convert NewsItem objects to dicts if needed
            news_list = []
            for item in news:
                if hasattr(item, "headline"):
                    news_list.append({
                        "headline": item.headline,
                        "summary": item.summary,
                        "source": item.source,
                        "url": item.url,
                        "published_at": item.published_at,  # Phase-6 sentiment momentum
                    })
                else:
                    news_list.append(item)

            if mode == "intraday":
                analysis = self.analyzer.analyze_intraday(
                    symbol=symbol,
                    quote=quote_dict,
                    indicators=indicators,
                    news=news_list,
                    social_sentiment=social_sentiment
                )
            else:
                analysis = self.analyzer.analyze_longterm(
                    symbol=symbol,
                    quote=quote_dict,
                    fundamentals=fundamentals,
                    indicators=indicators,
                    news=news_list
                )

            if not analysis.signal:
                logger.error("Analysis failed - no signal generated")
                return {"error": "Analysis failed", "symbol": symbol}

            logger.info(f"   Signal: {analysis.signal.value} (confidence: {analysis.confidence:.0%})")
            logger.info(f"   Model used: {analysis.llm_model}")

            # Step 3: Verify with Ollama (optional)
            ollama_available = any('ollama' in m for m in self.analyzer.available_models)
            if verify and ollama_available:
                logger.info("[3/5] Verifying analysis with Ollama inspector...")
                verification = self.analyzer.verify_analysis(
                    analysis=analysis,
                    quote=quote_dict,
                    indicators=indicators
                )

                if verification.get("verified"):
                    logger.info("   Verification: PASSED")
                else:
                    logger.warning(f"   Verification: NEEDS REVIEW - {verification.get('concerns', 'N/A')}")
                    analysis.verification = verification
            else:
                logger.info("[3/5] Skipping verification (Ollama not available)")

            # Step 3.5: Generate AI Algo Trading Prediction
            logger.info("[3.5/5] Generating AI algo trading prediction...")
            algo_prediction = None
            try:
                algo_prediction = self.analyzer.generate_algo_prediction(
                    symbol=symbol,
                    quote=quote_dict,
                    indicators=indicators,
                    fundamentals=fundamentals,
                    news=news_list,
                    price_history=price_history,
                    finnhub_sentiment=finnhub_sentiment,
                )
                if algo_prediction:
                    analysis.algo_prediction = algo_prediction
                    logger.info(f"   Algo Signal: {algo_prediction.get('signal')} "
                               f"(momentum={algo_prediction.get('momentum_score')}, "
                               f"value={algo_prediction.get('value_score')}, "
                               f"quality={algo_prediction.get('quality_score')})")
            except Exception as e:
                logger.warning(f"   Algo prediction skipped: {e}")

            # Step 3.6: Paper trading shadow recording
            try:
                from config import settings as _settings
                if _settings.paper_trading_enabled and algo_prediction:
                    from training.paper_trading import PaperPortfolio
                    from training.broker import Order, PaperBroker, submit_with_retry
                    from training.pre_trade_risk import check_all_pre_trade_risk
                    from training.canary import (
                        check_canary_breach,
                        enable_canary,
                        is_canary_eligible,
                        load_canary_state,
                        record_canary_trade,
                        save_canary_state,
                    )
                    from training.audit import append_audit_event
                    pp = PaperPortfolio(
                        paper_dir=_settings.paper_trading_dir,
                        initial_capital=_settings.paper_trading_capital,
                    )
                    audit_enabled = bool(getattr(_settings, "audit_enabled", True))
                    actionable_signals = {"buy", "strong_buy", "sell", "strong_sell"}
                    algo_signal = str(algo_prediction.get("signal", "hold")).lower()
                    sector = (fundamentals or {}).get("sector") or "unknown"
                    proposed_size = abs(float(algo_prediction.get("position_size", 0.0) or 0.0))
                    allow_open = True
                    risk_result = None

                    def _audit(event_type: str, data: dict) -> None:
                        if not audit_enabled:
                            return
                        try:
                            append_audit_event(
                                event_type=event_type,
                                data=data,
                                audit_dir=_settings.audit_dir,
                            )
                        except Exception as audit_err:
                            logger.debug(f"Audit event failed ({event_type}): {audit_err}")

                    _audit(
                        "signal",
                        {
                            "symbol": symbol,
                            "signal": algo_signal,
                            "confidence": float(algo_prediction.get("confidence", 0.0) or 0.0),
                            "price": float(current_price or 0.0),
                            "position_size": proposed_size,
                            "model_version": algo_prediction.get("model_version"),
                            "market_regime": algo_prediction.get("market_regime"),
                        },
                    )

                    # Kill-switch check before recording
                    kill_switch_halted = False
                    if _settings.kill_switch_enabled:
                        from training.kill_switches import check_all_kill_switches
                        rt_data = None
                        rt_connected = False
                        try:
                            from agents.realtime import get_realtime_manager
                            rt = get_realtime_manager()
                            rt_connected = bool(getattr(rt, "is_connected", False))
                            if rt_connected and hasattr(rt, "get_latest"):
                                rt_data = rt.get_latest(symbol)
                        except Exception:
                            pass

                        # Run stale-data check only when realtime feed is
                        # connected and we have at least one snapshot.
                        stale_check_enabled = rt_connected and (rt_data is not None)
                        stale_input = rt_data if stale_check_enabled else {"age_ms": 0}
                        stale_ms = (
                            _settings.kill_switch_max_stale_ms
                            if stale_check_enabled
                            else 2**63
                        )

                        ks_result = check_all_kill_switches(
                            closed_trades=pp.get_closed_trades(),
                            positions=pp.get_open_positions(),
                            latest_realtime_data=stale_input,
                            max_daily_loss_pct=_settings.kill_switch_max_daily_loss_pct,
                            max_stale_ms=stale_ms,
                            slippage_threshold_pct=_settings.kill_switch_slippage_threshold_pct,
                        )
                        if ks_result["halted"]:
                            kill_switch_halted = True
                            for reason in ks_result["reasons"]:
                                logger.warning(f"   KILL SWITCH: {reason}")
                            _audit(
                                "kill_switch",
                                {
                                    "symbol": symbol,
                                    "reasons": ks_result.get("reasons", []),
                                    "checks": ks_result.get("checks", {}),
                                },
                            )

                    if not kill_switch_halted:
                        # Baseline closed-trade count for post-step canary accounting.
                        closed_before = pp.get_closed_trades()
                        closed_before_count = len(closed_before)

                        # Check stop-loss/take-profit on existing positions first
                        pp.update_prices({symbol: current_price})

                        # Canary gate (symbol allow-list + breach checks)
                        canary_state = None
                        canary_reasons: list[str] = []
                        if _settings.canary_enabled:
                            canary_state = load_canary_state(_settings.canary_dir)
                            if not canary_state.get("enabled", False):
                                if canary_state.get("disabled_reason"):
                                    allow_open = False
                                    canary_reasons.append(canary_state["disabled_reason"])
                                else:
                                    canary_state = enable_canary(_settings.canary_dir)

                            if allow_open and algo_signal in actionable_signals:
                                if not is_canary_eligible(symbol, _settings.canary_symbol_list):
                                    allow_open = False
                                    canary_reasons.append(
                                        f"Symbol {symbol} not in canary allow-list"
                                    )

                            if allow_open and algo_signal in actionable_signals:
                                canary_check = check_canary_breach(
                                    canary_state,
                                    max_trades=_settings.canary_max_trades,
                                    max_loss_pct=_settings.canary_max_loss_pct,
                                )
                                canary_state = canary_check["state"]
                                if canary_check["breached"]:
                                    allow_open = False
                                    canary_reasons.extend(canary_check["reasons"])
                                    _audit(
                                        "canary_breach",
                                        {
                                            "symbol": symbol,
                                            "reasons": canary_check["reasons"],
                                            "state": canary_state,
                                        },
                                    )

                        # Pre-trade risk gate (only for actionable new entries)
                        if (
                            _settings.pre_trade_risk_enabled
                            and algo_signal in actionable_signals
                            and proposed_size > 0
                        ):
                            open_positions = pp.get_open_positions()
                            today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                            closed_today = [
                                t for t in closed_before
                                if str(t.get("exit_time", ""))[:10] == today
                            ]
                            sector_map = {
                                sym: (pos.get("sector") or "unknown")
                                for sym, pos in open_positions.items()
                            }
                            sector_map[symbol] = sector

                            risk_result = check_all_pre_trade_risk(
                                symbol=symbol,
                                proposed_size=proposed_size,
                                positions=open_positions,
                                closed_trades_today=closed_today,
                                sector_map=sector_map,
                                max_single_position=_settings.pre_trade_max_position,
                                max_sector_weight=_settings.pre_trade_max_sector,
                                max_daily_loss_pct=_settings.pre_trade_max_daily_loss_pct,
                                max_total_exposure=_settings.pre_trade_max_exposure,
                            )
                            _audit(
                                "risk_check",
                                {
                                    "symbol": symbol,
                                    "blocked": risk_result["blocked"],
                                    "reasons": risk_result["reasons"],
                                    "checks": risk_result["checks"],
                                },
                            )
                            if risk_result["blocked"]:
                                allow_open = False
                                for reason in risk_result["reasons"]:
                                    logger.warning(f"   PRE-TRADE RISK: {reason}")

                        if canary_reasons:
                            for reason in canary_reasons:
                                logger.warning(f"   CANARY BLOCK: {reason}")
                            _audit(
                                "canary_breach",
                                {
                                    "symbol": symbol,
                                    "reasons": canary_reasons,
                                },
                            )

                        # Broker execution adapter path (paper/live abstraction).
                        fill_data = None
                        if algo_signal in actionable_signals and proposed_size > 0 and allow_open:
                            broker_mode = str(_settings.broker_mode or "paper").lower()
                            if broker_mode != "paper":
                                logger.warning(
                                    "Broker mode '%s' not implemented yet; using paper broker.",
                                    broker_mode,
                                )
                            broker = PaperBroker(paper_dir=_settings.paper_trading_dir)
                            side = "buy" if algo_signal in ("buy", "strong_buy") else "sell"
                            order = Order(
                                order_id=f"{symbol}:{int(time.time() * 1000)}:{side}",
                                symbol=symbol,
                                side=side,
                                quantity=max(proposed_size, 0.0001),
                                order_type="market",
                                stop_loss=algo_prediction.get("stop_loss"),
                                take_profit=algo_prediction.get("take_profit"),
                                metadata={
                                    "current_price": float(current_price),
                                    "signal": algo_signal,
                                    "confidence": float(algo_prediction.get("confidence", 0.0) or 0.0),
                                    "sector": sector,
                                    "model_version": str(algo_prediction.get("model_version", "")),
                                },
                            )
                            _audit(
                                "order",
                                {
                                    "symbol": symbol,
                                    "order_id": order.order_id,
                                    "side": order.side,
                                    "quantity": order.quantity,
                                    "order_type": order.order_type,
                                    "stop_loss": order.stop_loss,
                                    "take_profit": order.take_profit,
                                },
                            )
                            fill = submit_with_retry(
                                broker,
                                order,
                                max_retries=_settings.broker_retry_max,
                                backoff_base=_settings.broker_retry_backoff,
                            )
                            fill_data = {
                                "symbol": fill.symbol,
                                "order_id": fill.order_id,
                                "fill_id": fill.fill_id,
                                "status": fill.status,
                                "fill_price": fill.fill_price,
                                "filled_quantity": fill.filled_quantity,
                                "error": fill.error,
                            }
                            _audit("fill", fill_data)
                        elif algo_signal in actionable_signals:
                            _audit(
                                "order",
                                {
                                    "symbol": symbol,
                                    "blocked": True,
                                    "blocked_by": (
                                        "pre_trade_risk"
                                        if risk_result and risk_result.get("blocked")
                                        else "canary"
                                    ),
                                },
                            )

                        pp.record_signal(
                            symbol=symbol,
                            signal=algo_prediction["signal"],
                            confidence=algo_prediction["confidence"],
                            price=current_price,
                            regime=algo_prediction.get("market_regime", "neutral"),
                            position_size=algo_prediction.get("position_size", 0.0),
                            sector=sector,
                            stop_loss=algo_prediction.get("stop_loss"),
                            take_profit=algo_prediction.get("take_profit"),
                            model_version=str(algo_prediction.get("model_version", "")),
                            allow_open=allow_open,
                            metadata={
                                "risk_check": risk_result,
                                "fill": fill_data,
                            },
                        )

                        # Update canary stats from newly closed trades this step.
                        if _settings.canary_enabled:
                            if canary_state is None:
                                canary_state = load_canary_state(_settings.canary_dir)
                            closed_after = pp.get_closed_trades()
                            for trade in closed_after[closed_before_count:]:
                                try:
                                    pnl_pct = float(trade.get("pnl_pct", 0.0))
                                except (TypeError, ValueError):
                                    pnl_pct = 0.0
                                canary_state = record_canary_trade(canary_state, pnl_pct)
                            canary_check = check_canary_breach(
                                canary_state,
                                max_trades=_settings.canary_max_trades,
                                max_loss_pct=_settings.canary_max_loss_pct,
                            )
                            canary_state = canary_check["state"]
                            if canary_check["breached"]:
                                _audit(
                                    "canary_breach",
                                    {
                                        "symbol": symbol,
                                        "reasons": canary_check["reasons"],
                                        "state": canary_state,
                                    },
                                )
                            save_canary_state(canary_state, _settings.canary_dir)

                        logger.info(f"   Paper trade recorded for {symbol}")
                    else:
                        logger.warning(f"   Paper trade SKIPPED for {symbol} (kill switch)")
            except Exception as e:
                logger.debug(f"Paper trading recording skipped: {e}")

            # Step 4: Store in database
            logger.info("[4/5] Storing analysis in database...")

            # Get or create stock record
            stock = self.storage.get_stock_by_symbol(symbol)
            if not stock:
                # Extract exchange from symbol
                exchange = "NSE" if ".NS" in symbol else "BSE" if ".BO" in symbol else "NASDAQ"
                currency = "INR" if exchange in ("NSE", "BSE") else "USD"
                stock = self.storage.get_or_create_stock(
                    symbol=symbol,
                    name=company_name,
                    exchange=exchange,
                    sector=fundamentals.get("sector") if fundamentals else None,
                    currency=currency
                )

            stock_id = stock["id"]

            # Store price data - convert PriceData objects to dicts
            if price_history:
                price_dicts = []
                for p in price_history:
                    if hasattr(p, "timestamp"):
                        # Convert PriceData dataclass to dict
                        price_dicts.append({
                            "timestamp": p.timestamp,
                            "open": p.open,
                            "high": p.high,
                            "low": p.low,
                            "close": p.close,
                            "volume": p.volume,
                        })
                    else:
                        price_dicts.append(p)
                self.storage.store_price_data(
                    stock_id=stock_id,
                    prices=price_dicts,
                    timeframe="1d"
                )

            # Store indicators
            indicators = data.get("indicators", {})
            if indicators:
                self.storage.store_indicators(
                    stock_id=stock_id,
                    timestamp=datetime.now(timezone.utc),
                    indicators=indicators
                )

            # Store analysis with embedding for RAG
            analysis_record = self.storage.store_analysis_with_embedding(
                stock_id=stock_id,
                mode=mode,
                signal=analysis.signal.value,
                confidence=analysis.confidence,
                reasoning=analysis.reasoning,
                technical_summary=analysis.technical_summary,
                sentiment_summary=analysis.sentiment_summary,
                support_level=analysis.support_level,
                resistance_level=analysis.resistance_level,
                target_price=analysis.target_price,
                stop_loss=analysis.stop_loss,
                llm_model=analysis.llm_model,
                llm_tokens_used=analysis.tokens_used,
                analysis_duration_ms=int((time.time() - start_time) * 1000),
                algo_prediction=getattr(analysis, 'algo_prediction', None),
                generate_embedding=True  # Enable RAG embeddings
            )

            # Store signal if actionable
            if analysis.signal.value in ("strong_buy", "buy", "strong_sell", "sell"):
                signal_type = "entry"
                importance = "high" if analysis.confidence >= 0.8 else "medium"

                self.storage.store_signal(
                    stock_id=stock_id,
                    signal_type=signal_type,
                    signal="buy" if "buy" in analysis.signal.value else "sell",
                    price_at_signal=current_price,
                    reason=analysis.reasoning[:500],
                    analysis_id=analysis_record.get("id"),
                    importance=importance
                )

            # Step 5: Send notifications
            if send_alert and analysis.signal.value in ("strong_buy", "buy", "strong_sell", "sell"):
                logger.info("[5/5] Sending notifications...")

                alert_result = self.notifications.send_analysis_alert(
                    symbol=symbol,
                    name=company_name,
                    signal=analysis.signal.value,
                    confidence=analysis.confidence,
                    reasoning=analysis.reasoning,
                    mode=mode,
                    current_price=current_price,
                    target_price=analysis.target_price,
                    stop_loss=analysis.stop_loss,
                    support=analysis.support_level,
                    resistance=analysis.resistance_level,
                )

                # Record alerts in database
                for channel, result in alert_result.get("channels", {}).items():
                    if result.get("success"):
                        self.storage.store_alert(
                            user_id="system",  # System-generated alert
                            stock_id=stock_id,
                            channel=channel,
                            message=f"{analysis.signal.value}: {analysis.reasoning[:200]}",
                            external_id=result.get("timestamp") or result.get("message_id"),
                            status="sent"
                        )

                logger.info(f"   Alerts sent to: {list(alert_result.get('channels', {}).keys())}")
            else:
                logger.info("[5/5] Skipping notifications (hold signal or disabled)")

            # Send API usage summary to Slack
            tracker = get_tracker()
            usage_summary = tracker.get_session_summary(symbol)
            if usage_summary:
                logger.info(f"   Usage: {usage_summary.replace(chr(10), ' | ')}")
                # Send usage summary via Slack
                try:
                    self.notifications.slack.send_text(usage_summary)
                except Exception as e:
                    logger.debug(f"Could not send usage summary: {e}")
                tracker.clear_session()

            elapsed = time.time() - start_time
            logger.info(f"Analysis complete in {elapsed:.1f}s")

            result = {
                "symbol": symbol,
                "name": company_name,
                "mode": mode,
                "signal": analysis.signal.value,
                "confidence": analysis.confidence,
                "reasoning": analysis.reasoning,
                "current_price": current_price,
                "target_price": analysis.target_price,
                "stop_loss": analysis.stop_loss,
                "support": analysis.support_level,
                "resistance": analysis.resistance_level,
                "model_used": analysis.llm_model,
                "tokens_used": analysis.tokens_used,
                "duration_seconds": elapsed,
            }

            if algo_prediction:
                result.update({
                    "market_regime": algo_prediction.get("market_regime"),
                    "momentum_score": algo_prediction.get("momentum_score"),
                    "value_score": algo_prediction.get("value_score"),
                    "quality_score": algo_prediction.get("quality_score"),
                    "risk_score": algo_prediction.get("risk_score"),
                    "position_size_pct": algo_prediction.get("position_size_pct"),
                    "algo_stop_loss": algo_prediction.get("stop_loss"),
                    "algo_take_profit": algo_prediction.get("take_profit"),
                    "scoring_method": algo_prediction.get("scoring_method"),
                })

            return result

        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}", exc_info=True)
            return {"error": str(e), "symbol": symbol}

    def analyze_watchlist(
        self,
        user_email: str,
        mode: Optional[str] = None,
        send_alerts: bool = True
    ) -> list:
        """
        Analyze all stocks in a user's watchlist.

        Args:
            user_email: User's email to get watchlist
            mode: Override trading mode (or use per-stock mode)
            send_alerts: Whether to send notifications

        Returns:
            List of analysis results
        """
        logger.info(f"Analyzing watchlist for {user_email}")

        # Get or create user
        user = self.storage.get_or_create_user(user_email)
        user_id = user["id"]
        user_mode = mode or user.get("trading_mode", "intraday")

        # Get watchlist
        watchlist = self.storage.get_user_watchlist(user_id)

        if not watchlist:
            logger.warning(f"No stocks in watchlist for {user_email}")
            return []

        logger.info(f"Found {len(watchlist)} stocks in watchlist")

        results = []
        for entry in watchlist:
            stock = entry.get("stocks", {})
            symbol = stock.get("symbol")
            stock_mode = entry.get("mode", user_mode)

            if not symbol:
                continue

            logger.info(f"\nAnalyzing {symbol}...")
            result = self.analyze_stock(
                symbol=symbol,
                mode=mode or stock_mode,
                send_alert=send_alerts and entry.get("alerts_enabled", True)
            )
            results.append(result)

            # Small delay between analyses
            time.sleep(1)

        # Summary
        successful = sum(1 for r in results if "error" not in r)
        logger.info(f"\nWatchlist analysis complete: {successful}/{len(results)} successful")

        return results

    def explain_movement(self, symbol: str) -> dict:
        """
        Explain why a stock is moving (up or down).

        Args:
            symbol: Stock symbol

        Returns:
            Explanation dictionary
        """
        logger.info(f"Explaining movement for {symbol}")

        # Fetch data
        data = self.fetcher.get_full_analysis_data(symbol, period="5d")

        if "error" in data:
            return {"error": data["error"]}

        # Get AI explanation
        explanation = self.analyzer.explain_movement(data)

        return {
            "symbol": symbol,
            "name": data.get("quote", {}).get("name", symbol),
            "current_price": data.get("quote", {}).get("current_price"),
            "change_percent": data.get("quote", {}).get("change_percent"),
            "explanation": explanation,
        }

    def run_continuous(
        self,
        symbols: list,
        mode: str = "intraday",
        interval_minutes: int = 15,
        max_iterations: Optional[int] = None
    ):
        """
        Run continuous analysis at regular intervals.

        Args:
            symbols: List of stock symbols to analyze
            mode: Trading mode
            interval_minutes: Minutes between analysis runs
            max_iterations: Maximum number of iterations (None for infinite)
        """
        logger.info(f"Starting continuous analysis for {len(symbols)} stocks")
        logger.info(f"Mode: {mode}, Interval: {interval_minutes} minutes")

        iteration = 0
        while max_iterations is None or iteration < max_iterations:
            iteration += 1
            logger.info(f"\n{'='*50}")
            logger.info(f"ITERATION {iteration} - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"{'='*50}")

            for symbol in symbols:
                try:
                    self.analyze_stock(symbol, mode=mode)
                except Exception as e:
                    logger.error(f"Error analyzing {symbol}: {e}")

                time.sleep(2)  # Small delay between stocks

            if max_iterations is None or iteration < max_iterations:
                logger.info(f"\nSleeping for {interval_minutes} minutes...")
                time.sleep(interval_minutes * 60)


def _print_paper_dashboard(pp, settings) -> None:
    """Print formatted terminal dashboard of paper-trading state."""
    from training.paper_trading import check_paper_trading_gates
    from training.kill_switches import check_all_kill_switches

    summary = pp.get_performance_summary()
    positions = pp.get_open_positions()
    closed_trades = pp.get_closed_trades()

    # === PnL Summary ===
    print(f"\n{'='*60}")
    print("  PAPER TRADING DASHBOARD")
    print(f"{'='*60}")
    print("\n--- PnL Summary ---")
    print(f"  Total Trades:   {summary['total_trades']}")
    if summary["total_trades"] > 0:
        print(f"  Win Rate:       {summary['win_rate']:.1%}")
        print(f"  Avg P&L:        {summary['avg_pnl_pct']:+.2f}%")
        print(f"  Total P&L:      {summary['total_pnl_pct']:+.2f}%")
        print(f"  Best Trade:     {summary['best_trade_pct']:+.2f}%")
        print(f"  Worst Trade:    {summary['worst_trade_pct']:+.2f}%")

    # === Rolling Window ===
    rolling = pp.get_rolling_performance(last_n_trades=50)
    print("\n--- Rolling Window (last 50 trades) ---")
    if rolling["total_trades"] > 0:
        print(f"  Sharpe:         {rolling['sharpe']:.4f}")
        print(f"  Max Drawdown:   {rolling['max_drawdown']:.2f}%")
        print(f"  Win Rate:       {rolling['win_rate']:.1%}")
        print(f"  Turnover:       {rolling['turnover']:.2f}")
    else:
        print("  No trades in window")

    # === Open Exposure ===
    print(f"\n--- Open Exposure ({len(positions)} positions) ---")
    long_exp = 0.0
    short_exp = 0.0
    for sym, pos in positions.items():
        direction = pos.get("direction", "long")
        size = abs(pos.get("position_size", 0.0))
        entry = pos.get("entry_price", 0.0)
        sl = pos.get("stop_loss")
        tp = pos.get("take_profit")
        print(f"  {sym}: {direction} @ {entry:.2f} size={size:.4f} SL={sl} TP={tp}")
        if direction == "long":
            long_exp += size
        else:
            short_exp += size
    net_exp = long_exp - short_exp
    print(f"  Long: {long_exp:.4f}  Short: {short_exp:.4f}  Net: {net_exp:+.4f}")

    # === Regime Distribution ===
    signals = pp._read_jsonl(pp.signals_path)
    regimes: dict[str, int] = {}
    for sig in signals:
        r = sig.get("regime", "unknown")
        regimes[r] = regimes.get(r, 0) + 1
    print(f"\n--- Regime Distribution ({len(signals)} signals) ---")
    for regime, count in sorted(regimes.items(), key=lambda x: -x[1]):
        bar = "#" * min(count, 40)
        print(f"  {regime:20s} {count:4d} {bar}")

    # === Recent Signals ===
    recent = signals[-10:] if signals else []
    print(f"\n--- Recent Signals (last {len(recent)}) ---")
    for sig in recent:
        ts = sig.get("timestamp", "")[:19]
        sym = sig.get("symbol", "")
        signal = sig.get("signal", "")
        conf = sig.get("confidence", 0.0)
        print(f"  {ts}  {sym:8s}  {signal:12s}  conf={conf:.2f}")

    # === Kill-Switch Status ===
    print("\n--- Kill-Switch Status ---")
    try:
        ks = check_all_kill_switches(
            closed_trades=closed_trades,
            positions=positions,
            latest_realtime_data={"age_ms": 0},
            max_daily_loss_pct=settings.kill_switch_max_daily_loss_pct,
            max_stale_ms=settings.kill_switch_max_stale_ms,
            slippage_threshold_pct=settings.kill_switch_slippage_threshold_pct,
        )
        for name, check in ks.get("checks", {}).items():
            status = "TRIGGERED" if check.get("triggered") else "OK"
            print(f"  {name:20s} [{status}]")
        if ks["halted"]:
            print(f"  >> TRADING HALTED: {'; '.join(ks['reasons'])}")
        else:
            print("  >> All clear")
    except Exception as e:
        print(f"  Error running kill switches: {e}")

    # === Promotion Gates ===
    print("\n--- Promotion Gates ---")
    try:
        gates = check_paper_trading_gates(
            paper_dir=settings.paper_trading_dir,
            initial_capital=settings.paper_trading_capital,
        )
        for gate_name, gate_detail in gates.get("checks", {}).items():
            status = "PASS" if gate_detail.get("passed") else "FAIL"
            val = gate_detail.get("value", "N/A")
            thresh = gate_detail.get("threshold", "N/A")
            print(f"  {gate_name:20s} [{status}]  value={val}  threshold={thresh}")
        if not gates.get("checks"):
            print(f"  {gates.get('reason', 'No checks run')}")
        overall = "PASS" if gates["passed"] else "FAIL"
        print(f"  >> Overall: [{overall}] {gates['reason']}")
    except Exception as e:
        print(f"  Error running gates: {e}")

    print()


def main():
    """Main entry point with CLI argument parsing."""
    parser = argparse.ArgumentParser(
        description="Stock Radar - AI-powered stock analysis for traders"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze a single stock")
    analyze_parser.add_argument("symbol", help="Stock symbol (e.g., RELIANCE.NS, AAPL)")
    analyze_parser.add_argument(
        "-m", "--mode",
        choices=["intraday", "longterm"],
        default="intraday",
        help="Trading mode (default: intraday)"
    )
    analyze_parser.add_argument(
        "-p", "--period",
        default="max",
        help="Data period for historical data (default: max for full history). Options: 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, max"
    )
    analyze_parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Don't send notifications"
    )
    analyze_parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip Ollama verification"
    )

    # watchlist command
    watchlist_parser = subparsers.add_parser("watchlist", help="Analyze user's watchlist")
    watchlist_parser.add_argument("email", help="User email")
    watchlist_parser.add_argument(
        "-m", "--mode",
        choices=["intraday", "longterm"],
        help="Override trading mode"
    )
    watchlist_parser.add_argument(
        "--no-alert",
        action="store_true",
        help="Don't send notifications"
    )

    # explain command
    explain_parser = subparsers.add_parser("explain", help="Explain stock movement")
    explain_parser.add_argument("symbol", help="Stock symbol")

    # continuous command
    continuous_parser = subparsers.add_parser("continuous", help="Run continuous analysis")
    continuous_parser.add_argument(
        "symbols",
        nargs="+",
        help="Stock symbols to analyze"
    )
    continuous_parser.add_argument(
        "-m", "--mode",
        choices=["intraday", "longterm"],
        default="intraday",
        help="Trading mode"
    )
    continuous_parser.add_argument(
        "-i", "--interval",
        type=int,
        default=15,
        help="Analysis interval in minutes (default: 15)"
    )
    continuous_parser.add_argument(
        "-n", "--iterations",
        type=int,
        help="Maximum iterations (default: infinite)"
    )

    # test command
    subparsers.add_parser("test", help="Test all connections")

    # usage command
    usage_parser = subparsers.add_parser("usage", help="Show API usage status")
    usage_parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset all usage counters"
    )

    # chat command
    chat_parser = subparsers.add_parser("chat", help="Start interactive AI chat assistant")
    chat_parser.add_argument(
        "-s", "--symbol",
        help="Focus chat on a specific stock symbol"
    )

    # ask command (single question)
    ask_parser = subparsers.add_parser("ask", help="Ask a single question")
    ask_parser.add_argument("question", help="Question to ask the AI assistant")
    ask_parser.add_argument(
        "-s", "--symbol",
        help="Focus on a specific stock symbol"
    )

    # paper command - paper trading management
    paper_parser = subparsers.add_parser("paper", help="Paper trading management")
    paper_parser.add_argument(
        "action",
        choices=["status", "trades", "reset", "dashboard"],
        help="Paper trading action: status, trades, reset, or dashboard",
    )

    # canary command - canary rollout controls
    canary_parser = subparsers.add_parser("canary", help="Canary rollout controls")
    canary_parser.add_argument(
        "action",
        choices=["status", "enable", "disable"],
        help="Canary action: status, enable, disable",
    )

    # audit command - immutable audit trail reports
    audit_parser = subparsers.add_parser("audit", help="Audit trail reporting")
    audit_parser.add_argument(
        "action",
        choices=["report"],
        help="Audit action: report",
    )
    audit_parser.add_argument(
        "--date",
        dest="report_date",
        help="Report date in YYYY-MM-DD (default: today UTC)",
    )

    # backfill command - fetch full historical data for stocks
    backfill_parser = subparsers.add_parser("backfill", help="Backfill full historical price data for stocks")
    backfill_parser.add_argument(
        "symbols",
        nargs="*",
        help="Stock symbols to backfill (leave empty for all stocks)"
    )
    backfill_parser.add_argument(
        "-p", "--period",
        default="max",
        help="Data period to fetch (default: max for full history)"
    )
    backfill_parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear existing price history before backfilling"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Initialize Stock Radar
    radar = StockRadar()

    if args.command == "analyze":
        result = radar.analyze_stock(
            symbol=args.symbol,
            mode=args.mode,
            period=args.period,
            send_alert=not args.no_alert,
            verify=not args.no_verify
        )

        if "error" in result:
            print(f"\nError: {result['error']}")
            sys.exit(1)

        print(f"\n{'='*50}")
        print(f"ANALYSIS RESULT: {result['symbol']}")
        print(f"{'='*50}")
        print(f"Company: {result['name']}")
        print(f"Mode: {result['mode']}")
        print(f"Signal: {result['signal'].upper()}")
        print(f"Confidence: {result['confidence']:.0%}")
        print(f"Current Price: {result['current_price']:.2f}")
        if result.get('target_price'):
            print(f"Target Price: {result['target_price']:.2f}")
        if result.get('stop_loss'):
            print(f"Stop Loss: {result['stop_loss']:.2f}")
        print(f"\nReasoning:\n{result['reasoning']}")
        print(f"\nModel: {result['model_used']}")
        print(f"Duration: {result['duration_seconds']:.1f}s")

        # --- Algo Intelligence section ---
        if result.get('scoring_method'):
            print(f"\n{'─'*50}")
            print("ALGO INTELLIGENCE")
            print(f"{'─'*50}")
            if result.get('market_regime'):
                print(f"Market Regime:   {result['market_regime']} ({result['scoring_method']})")
            if result.get('momentum_score') is not None:
                print(f"Momentum Score:  {result['momentum_score']}/100")
            if result.get('value_score') is not None:
                print(f"Value Score:     {result['value_score']}/100")
            if result.get('quality_score') is not None:
                print(f"Quality Score:   {result['quality_score']}/100")
            if result.get('risk_score') is not None:
                print(f"Risk Score:      {result['risk_score']}/10")
            if result.get('position_size_pct') is not None:
                print(f"Position Size:   {result['position_size_pct']}%")
            if result.get('algo_stop_loss') and result.get('algo_take_profit'):
                print(f"Algo SL / TP:    ${result['algo_stop_loss']:.2f} / ${result['algo_take_profit']:.2f}")

        # --- Paper Trading Performance section ---
        try:
            from config import settings as _settings
            if _settings.paper_trading_enabled:
                from training.paper_trading import PaperPortfolio
                pp = PaperPortfolio(
                    paper_dir=_settings.paper_trading_dir,
                    initial_capital=_settings.paper_trading_capital,
                )
                perf = pp.get_rolling_performance()
                if perf["total_trades"] > 0:
                    print(f"\n{'─'*50}")
                    print("PAPER TRADING PERFORMANCE")
                    print(f"{'─'*50}")
                    print(f"Trades: {perf['total_trades']}  |  "
                          f"Win Rate: {perf['win_rate']:.1%}  |  "
                          f"Avg P&L: {perf['avg_pnl_pct']:+.2f}%  |  "
                          f"Sharpe: {perf['sharpe']:.2f}")
        except Exception:
            pass

    elif args.command == "watchlist":
        results = radar.analyze_watchlist(
            user_email=args.email,
            mode=args.mode,
            send_alerts=not args.no_alert
        )

        print(f"\n{'='*50}")
        print("WATCHLIST ANALYSIS SUMMARY")
        print(f"{'='*50}")

        for result in results:
            if "error" in result:
                print(f"{result['symbol']}: ERROR - {result['error']}")
            else:
                print(f"{result['symbol']}: {result['signal'].upper()} ({result['confidence']:.0%})")

    elif args.command == "explain":
        result = radar.explain_movement(args.symbol)

        if "error" in result:
            print(f"\nError: {result['error']}")
            sys.exit(1)

        print(f"\n{'='*50}")
        print(f"WHY IS {result['symbol']} MOVING?")
        print(f"{'='*50}")
        print(f"Current Price: {result['current_price']:.2f}")
        print(f"Change: {result['change_percent']:.2f}%")
        print(f"\n{result['explanation']}")

    elif args.command == "continuous":
        radar.run_continuous(
            symbols=args.symbols,
            mode=args.mode,
            interval_minutes=args.interval,
            max_iterations=args.iterations
        )

    elif args.command == "test":
        print("Testing Stock Radar connections...")
        print("-" * 50)

        # Test notifications
        print("\nNotification Channels:")
        connections = radar.notifications.test_all_connections()
        for channel, status in connections.items():
            status_text = "CONNECTED" if status else "NOT CONFIGURED"
            print(f"  {channel}: {status_text}")

        # Test database
        print("\nDatabase:")
        if radar.storage.ensure_schema():
            print("  Supabase: CONNECTED")
        else:
            print("  Supabase: SCHEMA ERROR")

        # Test fetcher
        print("\nData Fetcher:")
        test_quote = radar.fetcher.get_quote("AAPL")
        if test_quote and hasattr(test_quote, 'price'):
            print(f"  Yahoo Finance: CONNECTED (AAPL = ${test_quote.price:.2f})")
        else:
            print("  Yahoo Finance: ERROR")

        # Test analyzer
        print("\nAI Analyzer:")
        print(f"  Available Models: {radar.analyzer.available_models}")
        ollama_in_models = any('ollama' in m for m in radar.analyzer.available_models)
        print(f"  Ollama Backup: {'AVAILABLE' if ollama_in_models else 'NOT AVAILABLE'}")

        print("\n" + "-" * 50)
        print("Test complete!")

    elif args.command == "usage":
        from agents.usage_tracker import get_tracker

        tracker = get_tracker()

        if args.reset:
            tracker.reset()
            print("✅ All usage counters have been reset.")
        else:
            print(tracker.get_status_report())

    elif args.command == "chat":
        from agents.chat_assistant import run_chat_cli
        run_chat_cli()

    elif args.command == "ask":
        from agents.chat_assistant import StockChatAssistant

        print("Processing your question...")
        assistant = StockChatAssistant()
        assistant.start_session()

        response = assistant.ask(
            question=args.question,
            stock_symbol=args.symbol.upper() if args.symbol else None
        )

        print(f"\n{'='*60}")
        print("ANSWER")
        print(f"{'='*60}")
        print(f"\n{response.answer}")

        if response.stock_symbols:
            print(f"\nStocks mentioned: {', '.join(response.stock_symbols)}")

        if response.sources_used:
            print(f"\nSources used: {len(response.sources_used)}")
            for source in response.sources_used[:5]:
                print(f"  - {source.get('type')}: {source.get('symbol') or source.get('headline', '')[:40]}")

        print(f"\n[{response.model_used} | {response.tokens_used} tokens | {response.processing_time_ms}ms]")

    elif args.command == "paper":
        from training.paper_trading import PaperPortfolio
        from config import settings as _settings

        pp = PaperPortfolio(
            paper_dir=_settings.paper_trading_dir,
            initial_capital=_settings.paper_trading_capital,
        )

        if args.action == "status":
            positions = pp.get_open_positions()
            summary = pp.get_performance_summary()

            print(f"\n{'='*50}")
            print("PAPER TRADING STATUS")
            print(f"{'='*50}")
            print(f"Open Positions: {len(positions)}")
            for sym, pos in positions.items():
                print(f"  {sym}: {pos['direction']} @ {pos['entry_price']:.2f} "
                      f"(SL={pos.get('stop_loss')}, TP={pos.get('take_profit')})")
            print(f"\nClosed Trades: {summary['total_trades']}")
            if summary['total_trades'] > 0:
                print(f"  Win Rate: {summary['win_rate']:.1%}")
                print(f"  Avg P&L: {summary['avg_pnl_pct']:.2f}%")
                print(f"  Total P&L: {summary['total_pnl_pct']:.2f}%")
                print(f"  Best Trade: {summary['best_trade_pct']:.2f}%")
                print(f"  Worst Trade: {summary['worst_trade_pct']:.2f}%")

        elif args.action == "trades":
            trades = pp.get_closed_trades()
            if not trades:
                print("No closed trades yet.")
            else:
                print(f"\n{'='*50}")
                print("CLOSED TRADES")
                print(f"{'='*50}")
                for t in trades:
                    pnl = t.get('pnl_pct', 0)
                    marker = "+" if pnl > 0 else ""
                    print(f"  {t['symbol']}: {t['direction']} "
                          f"entry={t['entry_price']:.2f} -> exit={t['exit_price']:.2f} "
                          f"({marker}{pnl:.2f}%)")

        elif args.action == "reset":
            confirm = input("Reset all paper trading data? (yes/no): ")
            if confirm.strip().lower() == "yes":
                pp.reset()
                print("Paper trading data cleared.")
            else:
                print("Reset cancelled.")

        elif args.action == "dashboard":
            _print_paper_dashboard(pp, _settings)

    elif args.command == "canary":
        from config import settings as _settings
        from training.canary import disable_canary, enable_canary, load_canary_state

        if args.action == "status":
            state = load_canary_state(_settings.canary_dir)
            print(f"\n{'='*50}")
            print("CANARY STATUS")
            print(f"{'='*50}")
            print(f"Enabled: {state.get('enabled', False)}")
            print(f"Total trades: {state.get('total_trades', 0)}")
            print(f"Total P&L: {state.get('total_pnl_pct', 0.0):+.2f}%")
            print(f"Breach count: {state.get('breach_count', 0)}")
            if state.get("disabled_reason"):
                print(f"Disabled reason: {state.get('disabled_reason')}")
            print(f"Allow-list symbols: {', '.join(_settings.canary_symbol_list) or 'ALL'}")

        elif args.action == "enable":
            state = enable_canary(_settings.canary_dir)
            print("Canary mode enabled.")
            print(f"State: trades={state['total_trades']}, pnl={state['total_pnl_pct']:+.2f}%")

        elif args.action == "disable":
            state = disable_canary(_settings.canary_dir, reason="manual_cli_disable")
            print("Canary mode disabled.")
            print(f"Reason: {state.get('disabled_reason')}")

    elif args.command == "audit":
        from config import settings as _settings
        from training.audit import generate_daily_report, print_daily_report

        if args.action == "report":
            report = generate_daily_report(
                dt=args.report_date,
                audit_dir=_settings.audit_dir,
                paper_dir=_settings.paper_trading_dir,
            )
            print_daily_report(report)

    elif args.command == "backfill":
        print("Backfilling historical price data...")
        print("-" * 50)

        # Get stocks to backfill
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        else:
            # Get all active stocks
            stocks = radar.storage.list_stocks()
            symbols = [s["symbol"] for s in stocks]

        if not symbols:
            print("No stocks found to backfill.")
            sys.exit(1)

        print(f"Backfilling {len(symbols)} stock(s) with period={args.period}")
        if args.clear:
            print("WARNING: Clearing existing price history before backfilling")

        success_count = 0
        total_records = 0

        for symbol in symbols:
            try:
                print(f"\n[{symbol}] Fetching historical data...")

                # Get stock record
                stock = radar.storage.get_stock_by_symbol(symbol)
                if not stock:
                    print("  Stock not found in database, skipping")
                    continue

                stock_id = stock["id"]

                # Clear existing data if requested
                if args.clear:
                    radar.storage.client.table("price_history").delete().eq("stock_id", stock_id).execute()
                    print("  Cleared existing price history")

                # Fetch price history
                prices = radar.fetcher.get_price_history(symbol, period=args.period)

                if not prices:
                    print("  No price data returned")
                    continue

                # Convert to dict format
                price_dicts = []
                for p in prices:
                    price_dicts.append({
                        "timestamp": p.timestamp,
                        "open": p.open,
                        "high": p.high,
                        "low": p.low,
                        "close": p.close,
                        "volume": p.volume
                    })

                # Store price data
                count = radar.storage.store_price_data(
                    stock_id=stock_id,
                    prices=price_dicts,
                    timeframe="1d"
                )

                print(f"  Stored {count} records (fetched {len(prices)} total)")
                success_count += 1
                total_records += count

            except Exception as e:
                print(f"  Error: {str(e)}")

        print("\n" + "-" * 50)
        print(f"Backfill complete: {success_count}/{len(symbols)} stocks, {total_records} total records")


if __name__ == "__main__":
    main()
