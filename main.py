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

from agents.fetcher import StockFetcher
from agents.analyzer import StockAnalyzer, TradingMode
from agents.storage import StockStorage
from agents.alerts import NotificationManager
from agents.usage_tracker import get_tracker

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

        self.fetcher = StockFetcher()
        self.analyzer = StockAnalyzer()
        self.storage = StockStorage()
        self.notifications = NotificationManager()

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
            trading_mode = TradingMode.INTRADAY if mode == "intraday" else TradingMode.LONGTERM

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
            news = data.get("news", [])
            
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
            try:
                algo_prediction = self.analyzer.generate_algo_prediction(
                    symbol=symbol,
                    quote=quote_dict,
                    indicators=indicators,
                    fundamentals=fundamentals,
                    news=news_list
                )
                if algo_prediction:
                    analysis.algo_prediction = algo_prediction
                    logger.info(f"   Algo Signal: {algo_prediction.get('signal')} "
                               f"(momentum={algo_prediction.get('momentum_score')}, "
                               f"value={algo_prediction.get('value_score')}, "
                               f"quality={algo_prediction.get('quality_score')})")
            except Exception as e:
                logger.warning(f"   Algo prediction skipped: {e}")

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
            price_history = data.get("price_history", [])
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

            return {
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
    test_parser = subparsers.add_parser("test", help="Test all connections")

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

    elif args.command == "watchlist":
        results = radar.analyze_watchlist(
            user_email=args.email,
            mode=args.mode,
            send_alerts=not args.no_alert
        )

        print(f"\n{'='*50}")
        print(f"WATCHLIST ANALYSIS SUMMARY")
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
                    print(f"  Stock not found in database, skipping")
                    continue

                stock_id = stock["id"]

                # Clear existing data if requested
                if args.clear:
                    radar.storage.client.table("price_history").delete().eq("stock_id", stock_id).execute()
                    print(f"  Cleared existing price history")

                # Fetch price history
                prices = radar.fetcher.get_price_history(symbol, period=args.period)

                if not prices:
                    print(f"  No price data returned")
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
