"""
Real-time stock data via Finnhub WebSocket (free tier).

Maintains a persistent WebSocket connection that receives trade updates
in real-time (<100ms latency). Caches latest prices in-memory so
get_quote() can return instantly for subscribed symbols.

Finnhub free tier: unlimited WebSocket connections, real-time US trades.
No credit card required.

Usage:
    from agents.realtime import RealtimeManager

    rt = RealtimeManager(finnhub_key="your_key")
    rt.subscribe(["AAPL", "MSFT", "GOOGL"])

    # Later - instant price lookup (no HTTP call)
    price = rt.get_latest("AAPL")
    # -> {"price": 185.42, "volume": 350, "timestamp": 1707..., "age_ms": 120}
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


@dataclass
class RealtimeTrade:
    """A single trade from the WebSocket feed."""
    symbol: str
    price: float
    volume: float
    timestamp_ms: int  # Unix milliseconds from Finnhub
    received_at: float = field(default_factory=time.time)


class RealtimeManager:
    """
    Finnhub WebSocket client for real-time stock prices.

    Free tier: real-time US stock trades, no credit card required.
    URL: wss://ws.finnhub.io?token=<API_KEY>

    Message format:
        Subscribe:   {"type":"subscribe","symbol":"AAPL"}
        Unsubscribe: {"type":"unsubscribe","symbol":"AAPL"}
        Trade data:  {"data":[{"p":185.42,"s":"AAPL","t":1707...,"v":100}],"type":"trade"}
    """

    def __init__(
        self,
        finnhub_key: str | None = None,
        on_trade: Callable[[RealtimeTrade], None] | None = None,
    ):
        self._key = finnhub_key or os.getenv("FINNHUB_API_KEY")
        self._on_trade = on_trade
        self._ws = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._subscribed: Set[str] = set()

        # Latest trade per symbol (in-memory cache)
        self._latest: Dict[str, RealtimeTrade] = {}
        self._lock = threading.Lock()

        if not self._key:
            logger.warning("No FINNHUB_API_KEY set - realtime feed disabled")

    @property
    def is_connected(self) -> bool:
        return self._running and self._ws is not None

    def start(self) -> bool:
        """Start the WebSocket connection in a background thread."""
        if not self._key:
            logger.warning("Cannot start realtime feed: no API key")
            return False

        if self._running:
            return True

        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="finnhub-ws")
        self._thread.start()
        logger.info("Finnhub WebSocket started")
        return True

    def stop(self):
        """Stop the WebSocket connection."""
        self._running = False
        if self._ws:
            try:
                self._ws.close()
            except Exception:
                pass
        self._ws = None
        logger.info("Finnhub WebSocket stopped")

    def subscribe(self, symbols: List[str]):
        """Subscribe to real-time trades for given symbols."""
        if not self._running:
            self.start()

        # Wait briefly for connection
        for _ in range(20):
            if self._ws:
                break
            time.sleep(0.1)

        for symbol in symbols:
            clean = symbol.split(".")[0].upper()
            if clean not in self._subscribed:
                self._send({"type": "subscribe", "symbol": clean})
                self._subscribed.add(clean)
                logger.info(f"Subscribed to {clean}")

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols."""
        for symbol in symbols:
            clean = symbol.split(".")[0].upper()
            if clean in self._subscribed:
                self._send({"type": "unsubscribe", "symbol": clean})
                self._subscribed.discard(clean)

    def get_latest(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get the latest trade for a symbol. Returns None if no data yet.

        Returns:
            Dict with price, volume, timestamp, age_ms
        """
        clean = symbol.split(".")[0].upper()
        with self._lock:
            trade = self._latest.get(clean)

        if not trade:
            return None

        age_ms = int((time.time() - trade.received_at) * 1000)

        return {
            "price": trade.price,
            "volume": trade.volume,
            "timestamp": trade.timestamp_ms,
            "age_ms": age_ms,
            "symbol": clean,
        }

    def get_subscribed_symbols(self) -> List[str]:
        """Get list of currently subscribed symbols."""
        return list(self._subscribed)

    # --- internal ---

    def _send(self, msg: dict):
        """Send a message on the WebSocket."""
        if self._ws:
            try:
                import websocket
                self._ws.send(json.dumps(msg))
            except Exception as e:
                logger.warning(f"WebSocket send failed: {e}")

    def _run(self):
        """WebSocket event loop (runs in background thread)."""
        try:
            import websocket
        except ImportError:
            logger.error("websocket-client not installed. Run: pip install websocket-client")
            self._running = False
            return

        url = f"wss://ws.finnhub.io?token={self._key}"

        while self._running:
            try:
                logger.info("Connecting to Finnhub WebSocket...")
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close,
                    on_open=self._on_open,
                )
                self._ws.run_forever(ping_interval=30, ping_timeout=10)
            except Exception as e:
                logger.error(f"WebSocket error: {e}")

            if self._running:
                logger.info("Reconnecting in 5s...")
                time.sleep(5)

    def _on_open(self, ws):
        """Re-subscribe to all symbols on reconnect."""
        logger.info("Finnhub WebSocket connected")
        for symbol in self._subscribed:
            self._send({"type": "subscribe", "symbol": symbol})

    def _on_message(self, ws, message):
        """Process incoming trade data."""
        try:
            data = json.loads(message)
            if data.get("type") != "trade":
                return

            for trade_data in data.get("data", []):
                symbol = trade_data.get("s", "")
                price = trade_data.get("p", 0)
                volume = trade_data.get("v", 0)
                timestamp_ms = trade_data.get("t", 0)

                trade = RealtimeTrade(
                    symbol=symbol,
                    price=price,
                    volume=volume,
                    timestamp_ms=timestamp_ms,
                )

                with self._lock:
                    self._latest[symbol] = trade

                if self._on_trade:
                    self._on_trade(trade)

        except Exception as e:
            logger.warning(f"Error processing WebSocket message: {e}")

    def _on_error(self, ws, error):
        logger.warning(f"Finnhub WebSocket error: {error}")

    def _on_close(self, ws, close_status_code, close_msg):
        logger.info(f"Finnhub WebSocket closed: {close_status_code} {close_msg}")


# Singleton instance
_manager: RealtimeManager | None = None


def get_realtime_manager() -> RealtimeManager:
    """Get or create the singleton RealtimeManager."""
    global _manager
    if _manager is None:
        _manager = RealtimeManager()
    return _manager
