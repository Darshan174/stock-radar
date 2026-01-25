"""
Stock Radar - Notification engine for trading signals.
Sends alerts to Slack and Telegram with rich formatting.
"""

import os
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import requests
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Manages Slack notifications for stock trading signals.
    Formats messages with trading-specific emojis and rich blocks.
    """

    # Signal type emoji mappings
    SIGNAL_EMOJIS = {
        "strong_buy": "🟢🟢",
        "buy": "🟢",
        "hold": "🟡",
        "sell": "🔴",
        "strong_sell": "🔴🔴",
    }

    # Trading mode emojis
    MODE_EMOJIS = {
        "intraday": "⚡",
        "longterm": "📈",
    }

    # Signal type emoji mappings for alerts
    ALERT_TYPE_EMOJIS = {
        "entry": "🎯",
        "exit": "🚪",
        "stop_loss": "🛑",
        "target_hit": "🎉",
    }

    # Importance level colors (Slack attachment colors)
    IMPORTANCE_COLORS = {
        "high": "#FF0000",
        "medium": "#FFA500",
        "low": "#00FF00",
    }

    MAX_RETRIES = 3
    RETRY_DELAY = 2

    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel_id: Optional[str] = None
    ):
        """
        Initialize Slack notifier.

        Args:
            bot_token: Slack bot token (defaults to SLACK_BOT_TOKEN env var)
            channel_id: Channel ID (defaults to SLACK_CHANNEL_ID env var)
        """
        self.bot_token = bot_token or os.getenv("SLACK_BOT_TOKEN")
        self.channel_id = channel_id or os.getenv("SLACK_CHANNEL_ID")

        if not self.bot_token or not self.channel_id:
            logger.warning("Slack credentials not configured - notifications disabled")
            self.client = None
        else:
            self.client = WebClient(token=self.bot_token)
            logger.info(f"SlackNotifier initialized for channel: {self.channel_id}")

    def is_configured(self) -> bool:
        """Check if Slack is properly configured."""
        return self.client is not None

    def format_analysis_alert(
        self,
        symbol: str,
        name: str,
        signal: str,
        confidence: float,
        reasoning: str,
        mode: str = "intraday",
        current_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        support: Optional[float] = None,
        resistance: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Format an analysis result into a Slack block message.

        Args:
            symbol: Stock symbol
            name: Company name
            signal: Trading signal (strong_buy, buy, hold, sell, strong_sell)
            confidence: Confidence score (0-1)
            reasoning: AI reasoning text
            mode: Trading mode (intraday, longterm)
            current_price: Current stock price
            target_price: Target price
            stop_loss: Stop loss level
            support: Support level
            resistance: Resistance level

        Returns:
            Formatted Slack message dictionary
        """
        signal_emoji = self.SIGNAL_EMOJIS.get(signal, "🟡")
        mode_emoji = self.MODE_EMOJIS.get(mode, "📊")
        confidence_pct = int(confidence * 100)

        # Confidence bar visualization
        filled = int(confidence * 10)
        confidence_bar = "█" * filled + "░" * (10 - filled)

        # Build header
        header_text = f"{signal_emoji} {symbol} - {signal.upper().replace('_', ' ')}"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": header_text,
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Company:*\n{name}",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Mode:*\n{mode_emoji} {mode.title()}",
                    },
                ],
            },
        ]

        # Price information section
        price_fields = []
        if current_price:
            price_fields.append({
                "type": "mrkdwn",
                "text": f"*Current Price:*\n₹{current_price:,.2f}",
            })
        if target_price:
            if current_price:
                pct_change = ((target_price - current_price) / current_price) * 100
                arrow = "↑" if pct_change > 0 else "↓"
                price_fields.append({
                    "type": "mrkdwn",
                    "text": f"*Target:*\n₹{target_price:,.2f} ({arrow}{abs(pct_change):.1f}%)",
                })
            else:
                price_fields.append({
                    "type": "mrkdwn",
                    "text": f"*Target:*\n₹{target_price:,.2f}",
                })
        if stop_loss:
            price_fields.append({
                "type": "mrkdwn",
                "text": f"*Stop Loss:*\n₹{stop_loss:,.2f}",
            })

        if price_fields:
            blocks.append({
                "type": "section",
                "fields": price_fields[:4],  # Max 4 fields per section
            })

        # Support/Resistance levels
        if support or resistance:
            level_fields = []
            if support:
                level_fields.append({
                    "type": "mrkdwn",
                    "text": f"*Support:*\n₹{support:,.2f}",
                })
            if resistance:
                level_fields.append({
                    "type": "mrkdwn",
                    "text": f"*Resistance:*\n₹{resistance:,.2f}",
                })
            blocks.append({
                "type": "section",
                "fields": level_fields,
            })

        # Confidence bar
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Confidence:* {confidence_bar} {confidence_pct}%",
            },
        })

        # Reasoning section
        # Truncate if too long
        truncated_reasoning = reasoning[:2000] + "..." if len(reasoning) > 2000 else reasoning
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Analysis:*\n{truncated_reasoning}",
            },
        })

        # Footer with timestamp
        blocks.append({
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": f"_Generated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_",
                }
            ],
        })

        blocks.append({"type": "divider"})

        # Determine color based on signal
        if signal in ("strong_buy", "buy"):
            color = "#00FF00"
        elif signal in ("strong_sell", "sell"):
            color = "#FF0000"
        else:
            color = "#FFA500"

        return {
            "blocks": blocks,
            "text": f"{signal_emoji} {symbol}: {signal.upper().replace('_', ' ')} - {confidence_pct}% confidence",
            "attachments": [{"color": color, "blocks": []}],
        }

    def format_signal_alert(
        self,
        symbol: str,
        name: str,
        signal_type: str,
        signal: str,
        price: float,
        reason: str,
        importance: str = "medium",
    ) -> Dict[str, Any]:
        """
        Format a trading signal into a Slack message.

        Args:
            symbol: Stock symbol
            name: Company name
            signal_type: 'entry', 'exit', 'stop_loss', 'target_hit'
            signal: 'buy', 'sell', 'hold'
            price: Price at signal
            reason: Signal reason
            importance: 'high', 'medium', 'low'

        Returns:
            Formatted Slack message dictionary
        """
        type_emoji = self.ALERT_TYPE_EMOJIS.get(signal_type, "📊")
        signal_emoji = self.SIGNAL_EMOJIS.get(signal, "🟡")
        color = self.IMPORTANCE_COLORS.get(importance, "#FFA500")

        header_text = f"{type_emoji} {signal_emoji} {symbol} - {signal_type.upper().replace('_', ' ')}"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": header_text,
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {
                        "type": "mrkdwn",
                        "text": f"*Stock:*\n{name} ({symbol})",
                    },
                    {
                        "type": "mrkdwn",
                        "text": f"*Price:*\n₹{price:,.2f}",
                    },
                ],
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Reason:*\n{reason}",
                },
            },
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": f"_Importance: {importance.upper()} | {datetime.now().strftime('%H:%M:%S')}_",
                    }
                ],
            },
            {"type": "divider"},
        ]

        return {
            "blocks": blocks,
            "text": f"{type_emoji} {symbol}: {signal_type} at ₹{price:,.2f}",
            "attachments": [{"color": color, "blocks": []}],
        }

    def _send_with_retry(self, message: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """Send message with retry logic."""
        if not self.client:
            return False, None, "Slack not configured"

        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat_postMessage(
                    channel=self.channel_id,
                    blocks=message.get("blocks"),
                    text=message.get("text"),
                    attachments=message.get("attachments"),
                )

                timestamp = response.get("ts")
                logger.info(f"Slack message sent, ts: {timestamp}")
                return True, timestamp, None

            except SlackApiError as e:
                last_error = str(e)
                error_code = e.response.get("error")
                logger.warning(f"Slack API error (attempt {attempt + 1}): {error_code}")

                if error_code in ["invalid_auth", "token_expired", "channel_not_found"]:
                    return False, None, f"Slack error: {error_code}"

                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)

            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error: {e}")
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)

        return False, None, f"Failed after {self.MAX_RETRIES} attempts: {last_error}"

    def send_analysis_alert(
        self,
        symbol: str,
        name: str,
        signal: str,
        confidence: float,
        reasoning: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Send an analysis alert to Slack."""
        try:
            message = self.format_analysis_alert(
                symbol=symbol,
                name=name,
                signal=signal,
                confidence=confidence,
                reasoning=reasoning,
                **kwargs
            )

            success, timestamp, error = self._send_with_retry(message)

            return {
                "success": success,
                "timestamp": timestamp,
                "error": error,
                "symbol": symbol,
                "signal": signal,
                "channel": "slack",
                "sent_at": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "timestamp": None,
                "error": str(e),
                "symbol": symbol,
                "signal": signal,
                "channel": "slack",
                "sent_at": datetime.now().isoformat(),
            }

    def send_signal_alert(
        self,
        symbol: str,
        name: str,
        signal_type: str,
        signal: str,
        price: float,
        reason: str,
        importance: str = "medium",
    ) -> Dict[str, Any]:
        """Send a trading signal alert to Slack."""
        try:
            message = self.format_signal_alert(
                symbol=symbol,
                name=name,
                signal_type=signal_type,
                signal=signal,
                price=price,
                reason=reason,
                importance=importance,
            )

            success, timestamp, error = self._send_with_retry(message)

            return {
                "success": success,
                "timestamp": timestamp,
                "error": error,
                "symbol": symbol,
                "signal_type": signal_type,
                "channel": "slack",
                "sent_at": datetime.now().isoformat(),
            }

        except Exception as e:
            return {
                "success": False,
                "timestamp": None,
                "error": str(e),
                "symbol": symbol,
                "signal_type": signal_type,
                "channel": "slack",
                "sent_at": datetime.now().isoformat(),
            }

    def send_text(self, text: str) -> Dict[str, Any]:
        """
        Send a simple text message to Slack.

        Args:
            text: Plain text message

        Returns:
            Result dictionary with success, timestamp, error
        """
        if not self.client:
            return {"success": False, "error": "Slack not configured"}

        try:
            response = self.client.chat_postMessage(
                channel=self.channel_id,
                text=text,
            )
            timestamp = response.get("ts")
            logger.info(f"Slack text message sent, ts: {timestamp}")
            return {"success": True, "timestamp": timestamp, "error": None}
        except SlackApiError as e:
            error = str(e.response.get("error", e))
            logger.error(f"Slack text message failed: {error}")
            return {"success": False, "timestamp": None, "error": error}
        except Exception as e:
            logger.error(f"Slack text message error: {e}")
            return {"success": False, "timestamp": None, "error": str(e)}

    def test_connection(self) -> bool:
        """Test Slack connection."""
        if not self.client:
            return False
        try:
            response = self.client.auth_test()
            logger.info(f"Slack connected: {response.get('user_id')}")
            return True
        except SlackApiError as e:
            logger.error(f"Slack connection failed: {e}")
            return False


class TelegramNotifier:
    """
    Manages Telegram notifications for stock trading signals.
    Good for mobile alerts and quick notifications.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token (defaults to TELEGRAM_BOT_TOKEN env var)
            chat_id: Chat ID (defaults to TELEGRAM_CHAT_ID env var)
        """
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.api_url = f"https://api.telegram.org/bot{self.bot_token}" if self.bot_token else None

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram credentials not configured - notifications disabled")
        else:
            logger.info("TelegramNotifier initialized")

    def is_configured(self) -> bool:
        """Check if Telegram is properly configured."""
        return self.bot_token is not None and self.chat_id is not None

    def format_analysis_message(
        self,
        symbol: str,
        name: str,
        signal: str,
        confidence: float,
        reasoning: str,
        mode: str = "intraday",
        current_price: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
    ) -> str:
        """Format analysis into a Telegram message with HTML formatting."""
        signal_emoji = {
            "strong_buy": "🟢🟢",
            "buy": "🟢",
            "hold": "🟡",
            "sell": "🔴",
            "strong_sell": "🔴🔴",
        }.get(signal, "🟡")

        mode_emoji = "⚡" if mode == "intraday" else "📈"
        confidence_pct = int(confidence * 100)

        lines = [
            f"{signal_emoji} <b>{symbol} - {signal.upper().replace('_', ' ')}</b>",
            f"",
            f"📊 <b>Company:</b> {name}",
            f"{mode_emoji} <b>Mode:</b> {mode.title()}",
        ]

        if current_price:
            lines.append(f"💵 <b>Price:</b> ₹{current_price:,.2f}")
        if target_price:
            lines.append(f"🎯 <b>Target:</b> ₹{target_price:,.2f}")
        if stop_loss:
            lines.append(f"🛑 <b>Stop Loss:</b> ₹{stop_loss:,.2f}")

        lines.extend([
            f"",
            f"📈 <b>Confidence:</b> {confidence_pct}%",
            f"",
            f"<b>Analysis:</b>",
            reasoning[:500] + "..." if len(reasoning) > 500 else reasoning,
        ])

        return "\n".join(lines)

    def format_signal_message(
        self,
        symbol: str,
        name: str,
        signal_type: str,
        signal: str,
        price: float,
        reason: str,
        importance: str = "medium",
    ) -> str:
        """Format a trading signal into a Telegram message."""
        type_emoji = {
            "entry": "🎯",
            "exit": "🚪",
            "stop_loss": "🛑",
            "target_hit": "🎉",
        }.get(signal_type, "📊")

        signal_emoji = {
            "buy": "🟢",
            "sell": "🔴",
            "hold": "🟡",
        }.get(signal, "🟡")

        importance_indicator = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢",
        }.get(importance, "🟡")

        lines = [
            f"{type_emoji} {signal_emoji} <b>{symbol}</b>",
            f"",
            f"<b>{signal_type.upper().replace('_', ' ')}</b> at ₹{price:,.2f}",
            f"",
            f"{name}",
            f"",
            f"<b>Reason:</b> {reason}",
            f"",
            f"{importance_indicator} Importance: {importance.upper()}",
        ]

        return "\n".join(lines)

    def send_message(self, text: str, parse_mode: str = "HTML") -> Dict[str, Any]:
        """Send a message via Telegram."""
        if not self.is_configured():
            return {
                "success": False,
                "error": "Telegram not configured",
                "channel": "telegram",
            }

        try:
            response = requests.post(
                f"{self.api_url}/sendMessage",
                json={
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": parse_mode,
                },
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()

            if data.get("ok"):
                message_id = data.get("result", {}).get("message_id")
                logger.info(f"Telegram message sent: {message_id}")
                return {
                    "success": True,
                    "message_id": message_id,
                    "error": None,
                    "channel": "telegram",
                    "sent_at": datetime.now().isoformat(),
                }
            else:
                error = data.get("description", "Unknown error")
                logger.error(f"Telegram error: {error}")
                return {
                    "success": False,
                    "message_id": None,
                    "error": error,
                    "channel": "telegram",
                    "sent_at": datetime.now().isoformat(),
                }

        except requests.exceptions.RequestException as e:
            logger.error(f"Telegram request error: {e}")
            return {
                "success": False,
                "message_id": None,
                "error": str(e),
                "channel": "telegram",
                "sent_at": datetime.now().isoformat(),
            }

    def send_analysis_alert(
        self,
        symbol: str,
        name: str,
        signal: str,
        confidence: float,
        reasoning: str,
        **kwargs
    ) -> Dict[str, Any]:
        """Send an analysis alert via Telegram."""
        message = self.format_analysis_message(
            symbol=symbol,
            name=name,
            signal=signal,
            confidence=confidence,
            reasoning=reasoning,
            **kwargs
        )
        result = self.send_message(message)
        result["symbol"] = symbol
        result["signal"] = signal
        return result

    def send_signal_alert(
        self,
        symbol: str,
        name: str,
        signal_type: str,
        signal: str,
        price: float,
        reason: str,
        importance: str = "medium",
    ) -> Dict[str, Any]:
        """Send a trading signal alert via Telegram."""
        message = self.format_signal_message(
            symbol=symbol,
            name=name,
            signal_type=signal_type,
            signal=signal,
            price=price,
            reason=reason,
            importance=importance,
        )
        result = self.send_message(message)
        result["symbol"] = symbol
        result["signal_type"] = signal_type
        return result

    def test_connection(self) -> bool:
        """Test Telegram connection by getting bot info."""
        if not self.is_configured():
            return False
        try:
            response = requests.get(f"{self.api_url}/getMe", timeout=5)
            response.raise_for_status()
            data = response.json()
            if data.get("ok"):
                logger.info(f"Telegram connected: @{data['result']['username']}")
                return True
            return False
        except Exception as e:
            logger.error(f"Telegram connection failed: {e}")
            return False


class NotificationManager:
    """
    Unified notification manager for Stock Radar.
    Handles sending alerts through all configured channels.
    """

    def __init__(self):
        """Initialize all notification channels."""
        self.slack = SlackNotifier()
        self.telegram = TelegramNotifier()

        # Track which channels are active
        self.active_channels = []
        if self.slack.is_configured():
            self.active_channels.append("slack")
        if self.telegram.is_configured():
            self.active_channels.append("telegram")

        logger.info(f"NotificationManager initialized with channels: {self.active_channels}")

    def send_analysis_alert(
        self,
        symbol: str,
        name: str,
        signal: str,
        confidence: float,
        reasoning: str,
        channels: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Send analysis alert to specified channels.

        Args:
            symbol: Stock symbol
            name: Company name
            signal: Trading signal
            confidence: Confidence score
            reasoning: AI reasoning
            channels: List of channels to use (defaults to all active)
            **kwargs: Additional arguments (mode, prices, etc.)

        Returns:
            Results from each channel
        """
        channels = channels or self.active_channels
        results = {}

        for channel in channels:
            if channel == "slack" and self.slack.is_configured():
                results["slack"] = self.slack.send_analysis_alert(
                    symbol=symbol,
                    name=name,
                    signal=signal,
                    confidence=confidence,
                    reasoning=reasoning,
                    **kwargs
                )
            elif channel == "telegram" and self.telegram.is_configured():
                results["telegram"] = self.telegram.send_analysis_alert(
                    symbol=symbol,
                    name=name,
                    signal=signal,
                    confidence=confidence,
                    reasoning=reasoning,
                    **kwargs
                )

        return {
            "symbol": symbol,
            "signal": signal,
            "channels": results,
            "sent_at": datetime.now().isoformat(),
        }

    def send_signal_alert(
        self,
        symbol: str,
        name: str,
        signal_type: str,
        signal: str,
        price: float,
        reason: str,
        importance: str = "medium",
        channels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Send trading signal alert to specified channels.

        Args:
            symbol: Stock symbol
            name: Company name
            signal_type: Type of signal
            signal: Buy/sell/hold
            price: Current price
            reason: Signal reason
            importance: Alert importance
            channels: List of channels to use

        Returns:
            Results from each channel
        """
        channels = channels or self.active_channels
        results = {}

        for channel in channels:
            if channel == "slack" and self.slack.is_configured():
                results["slack"] = self.slack.send_signal_alert(
                    symbol=symbol,
                    name=name,
                    signal_type=signal_type,
                    signal=signal,
                    price=price,
                    reason=reason,
                    importance=importance,
                )
            elif channel == "telegram" and self.telegram.is_configured():
                results["telegram"] = self.telegram.send_signal_alert(
                    symbol=symbol,
                    name=name,
                    signal_type=signal_type,
                    signal=signal,
                    price=price,
                    reason=reason,
                    importance=importance,
                )

        return {
            "symbol": symbol,
            "signal_type": signal_type,
            "channels": results,
            "sent_at": datetime.now().isoformat(),
        }

    def send_batch_alerts(
        self,
        alerts: List[Dict[str, Any]],
        channels: Optional[List[str]] = None,
        delay: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Send multiple alerts in batch.

        Args:
            alerts: List of alert dictionaries with 'type' and data
            channels: Channels to use
            delay: Delay between messages (rate limiting)

        Returns:
            Batch results summary
        """
        results = []
        successful = 0
        failed = 0

        for alert in alerts:
            alert_type = alert.get("type", "analysis")

            if alert_type == "analysis":
                result = self.send_analysis_alert(
                    channels=channels,
                    **{k: v for k, v in alert.items() if k != "type"}
                )
            else:
                result = self.send_signal_alert(
                    channels=channels,
                    **{k: v for k, v in alert.items() if k != "type"}
                )

            results.append(result)

            # Count successes
            for channel_result in result.get("channels", {}).values():
                if channel_result.get("success"):
                    successful += 1
                else:
                    failed += 1

            time.sleep(delay)

        return {
            "total": len(alerts),
            "successful": successful,
            "failed": failed,
            "results": results,
            "sent_at": datetime.now().isoformat(),
        }

    def test_all_connections(self) -> Dict[str, bool]:
        """Test all notification channel connections."""
        return {
            "slack": self.slack.test_connection() if self.slack.is_configured() else False,
            "telegram": self.telegram.test_connection() if self.telegram.is_configured() else False,
        }


if __name__ == "__main__":
    # Test notification functionality
    print("Testing Stock Radar Notification Module")
    print("=" * 50)

    manager = NotificationManager()
    print(f"\nActive channels: {manager.active_channels}")

    # Test connections
    print("\nTesting connections...")
    connections = manager.test_all_connections()
    for channel, status in connections.items():
        status_text = "Connected" if status else "Not configured/Failed"
        print(f"  {channel}: {status_text}")

    # Example alert format
    print("\n" + "=" * 50)
    print("Example Slack message format:")
    print("-" * 50)

    if manager.slack.is_configured():
        message = manager.slack.format_analysis_alert(
            symbol="RELIANCE.NS",
            name="Reliance Industries",
            signal="buy",
            confidence=0.85,
            reasoning="Strong momentum with RSI at 65. MACD showing bullish crossover. Price above all major moving averages.",
            mode="intraday",
            current_price=2450.50,
            target_price=2520.00,
            stop_loss=2400.00,
            support=2380.00,
            resistance=2550.00,
        )
        print(f"Text: {message['text']}")
    else:
        print("Slack not configured - skipping format test")

    print("\n" + "=" * 50)
    print("Notification module test completed")
