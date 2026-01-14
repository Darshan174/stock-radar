"""
Notification engine for Research Radar competitor intelligence system.
Formats alerts and sends them to Slack with rich formatting and tracking.
"""

import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Manages Slack notifications for competitor intelligence changes.
    Formats messages with emojis and importance indicators.
    """

    # Change type emoji mappings
    CHANGE_TYPE_EMOJIS = {
        "pricing": "💰",
        "feature": "✨",
        "hiring": "👥",
        "partnership": "🤝",
        "other": "ℹ️",
    }

    # Importance level emoji mappings
    IMPORTANCE_EMOJIS = {
        "high": "🔴",
        "mid": "🟡",
        "low": "🟢",
    }

    # Retry configuration
    MAX_RETRIES = 3
    RETRY_DELAY = 2  # seconds

    def __init__(self, bot_token: str, channel_id: str):
        """
        Initialize Slack notifier with authentication and channel.

        Args:
            bot_token: Slack bot token for authentication
            channel_id: Channel ID where alerts will be sent

        Raises:
            ValueError: If bot_token or channel_id is empty
        """
        if not bot_token or not channel_id:
            raise ValueError("bot_token and channel_id are required")

        self.client = WebClient(token=bot_token)
        self.channel_id = channel_id
        self.bot_token = bot_token

        logger.info(f"SlackNotifier initialized for channel: {channel_id}")

    def _get_change_type_emoji(self, change_type: str) -> str:
        """Get emoji for change type, default to 'other'."""
        return self.CHANGE_TYPE_EMOJIS.get(change_type.lower(), self.CHANGE_TYPE_EMOJIS["other"])

    def _get_importance_emoji(self, importance: str) -> str:
        """Get emoji for importance level, default to 'low'."""
        return self.IMPORTANCE_EMOJIS.get(importance.lower(), self.IMPORTANCE_EMOJIS["low"])

    def format_change_message(
        self,
        competitor: str,
        change: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Format a change into a Slack block message.

        Args:
            competitor: Name of the competitor
            change: Dictionary containing change details with keys:
                - type: Change type (pricing, feature, hiring, partnership, other)
                - summary: Brief summary of the change
                - importance: Level (high, mid, low)
                - details: Detailed description (optional)
                - url: Link to more information (optional)
                - date: Change date/discovery date (optional)

        Returns:
            Dictionary containing formatted Slack blocks and metadata
        """
        change_type = change.get("type", "other")
        summary = change.get("summary", "No summary provided")
        importance = change.get("importance", "low")
        details = change.get("details", "")
        url = change.get("url", "")
        change_date = change.get("date", datetime.now().isoformat())

        type_emoji = self._get_change_type_emoji(change_type)
        importance_emoji = self._get_importance_emoji(importance)

        # Format timestamp
        try:
            date_obj = datetime.fromisoformat(change_date.replace("Z", "+00:00"))
            formatted_date = date_obj.strftime("%b %d, %Y")
        except (ValueError, AttributeError):
            formatted_date = str(change_date)

        # Build header section with competitor, type, and importance
        header_text = f"{type_emoji} {importance_emoji} *{competitor}* - {change_type.title()}"

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
                "text": {
                    "type": "mrkdwn",
                    "text": f"*Summary:*\n{summary}",
                },
            },
        ]

        # Add details section if provided
        if details:
            blocks.append(
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Details:*\n{details}",
                    },
                }
            )

        # Add metadata footer with date and optional URL
        footer_parts = [f"_Detected: {formatted_date}_"]
        if url:
            footer_parts.append(f"<{url}|View Source>")

        blocks.append(
            {
                "type": "context",
                "elements": [
                    {
                        "type": "mrkdwn",
                        "text": " | ".join(footer_parts),
                    }
                ],
            }
        )

        # Add divider
        blocks.append({"type": "divider"})

        return {
            "blocks": blocks,
            "text": f"{type_emoji} {importance_emoji} {competitor}: {summary}",
            "metadata": {
                "competitor": competitor,
                "change_type": change_type,
                "importance": importance,
                "timestamp": datetime.now().isoformat(),
            },
        }

    def _send_with_retry(self, message: Dict[str, Any]) -> Tuple[bool, Optional[str], Optional[str]]:
        """
        Send message to Slack with retry logic.

        Args:
            message: Formatted message to send

        Returns:
            Tuple of (success: bool, timestamp: Optional[str], error: Optional[str])
        """
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.chat_postMessage(
                    channel=self.channel_id,
                    blocks=message.get("blocks"),
                    text=message.get("text"),
                )

                timestamp = response.get("ts")
                logger.info(f"Message sent successfully to {self.channel_id}, ts: {timestamp}")
                return True, timestamp, None

            except SlackApiError as e:
                last_error = str(e)
                error_code = e.response.get("error")
                logger.warning(
                    f"Slack API error (attempt {attempt + 1}/{self.MAX_RETRIES}): {error_code} - {e}"
                )

                # Don't retry on authentication errors
                if error_code in ["invalid_auth", "token_expired", "token_revoked"]:
                    logger.error(f"Authentication error: {error_code}")
                    return False, None, f"Authentication failed: {error_code}"

                # Don't retry on channel errors
                if error_code in ["channel_not_found", "not_in_channel"]:
                    logger.error(f"Channel error: {error_code}")
                    return False, None, f"Channel error: {error_code}"

                # Retry on transient errors
                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)

            except Exception as e:
                last_error = str(e)
                logger.error(f"Unexpected error sending message (attempt {attempt + 1}/{self.MAX_RETRIES}): {e}")

                if attempt < self.MAX_RETRIES - 1:
                    time.sleep(self.RETRY_DELAY)

        error_message = f"Failed to send message after {self.MAX_RETRIES} attempts: {last_error}"
        logger.error(error_message)
        return False, None, error_message

    def send_change_alert(
        self,
        competitor: str,
        change: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Send a single change alert to Slack.

        Args:
            competitor: Name of the competitor
            change: Dictionary containing change details

        Returns:
            Dictionary with keys:
                - success: Boolean indicating if send was successful
                - timestamp: Slack message timestamp if successful
                - error: Error message if failed
                - competitor: Name of competitor
                - change_type: Type of change
        """
        try:
            # Format the message
            formatted_message = self.format_change_message(competitor, change)

            # Send with retry logic
            success, timestamp, error = self._send_with_retry(formatted_message)

            return {
                "success": success,
                "timestamp": timestamp,
                "error": error,
                "competitor": competitor,
                "change_type": change.get("type", "other"),
                "importance": change.get("importance", "low"),
                "sent_at": datetime.now().isoformat(),
            }

        except Exception as e:
            error_message = f"Error processing change alert: {str(e)}"
            logger.error(error_message, exc_info=True)
            return {
                "success": False,
                "timestamp": None,
                "error": error_message,
                "competitor": competitor,
                "change_type": change.get("type", "other"),
                "importance": change.get("importance", "low"),
                "sent_at": datetime.now().isoformat(),
            }

    def send_batch_alerts(
        self,
        changes: List[Dict[str, Any]],
        batch_delay: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Send multiple change alerts in batch.

        Args:
            changes: List of dictionaries with keys:
                - competitor: Competitor name
                - change: Change details (same format as send_change_alert)
            batch_delay: Delay between messages in seconds (default 0.5)

        Returns:
            Dictionary with:
                - total: Total number of changes
                - successful: Number of successful sends
                - failed: Number of failed sends
                - results: List of results from each send
                - timestamps: List of successful message timestamps
                - errors: List of error messages
        """
        results = []
        timestamps = []
        errors = []

        logger.info(f"Sending batch alerts for {len(changes)} changes")

        for idx, item in enumerate(changes):
            try:
                competitor = item.get("competitor", "Unknown")
                change = item.get("change", {})

                result = self.send_change_alert(competitor, change)
                results.append(result)

                if result["success"]:
                    timestamps.append(result["timestamp"])
                else:
                    errors.append(
                        {
                            "competitor": competitor,
                            "error": result["error"],
                        }
                    )

                # Add delay between messages to avoid rate limiting
                if idx < len(changes) - 1:
                    time.sleep(batch_delay)

            except Exception as e:
                error_msg = f"Error processing batch item {idx}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(
                    {
                        "competitor": item.get("competitor", "Unknown"),
                        "error": error_msg,
                    }
                )

        successful = sum(1 for r in results if r["success"])
        failed = len(results) - successful

        batch_result = {
            "total": len(changes),
            "successful": successful,
            "failed": failed,
            "results": results,
            "timestamps": timestamps,
            "errors": errors,
            "sent_at": datetime.now().isoformat(),
        }

        logger.info(
            f"Batch send complete: {successful} successful, {failed} failed out of {len(changes)} total"
        )

        return batch_result

    def test_connection(self) -> bool:
        """
        Test Slack connection by calling auth.test.

        Returns:
            True if connection is successful, False otherwise
        """
        try:
            response = self.client.auth_test()
            logger.info(f"Slack connection test successful: {response.get('user_id')}")
            return True
        except SlackApiError as e:
            logger.error(f"Slack connection test failed: {e}")
            return False
