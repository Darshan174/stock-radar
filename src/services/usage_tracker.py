"""
API Usage Tracker for Stock Radar.
Tracks API usage with per-request summaries and threshold alerts.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Any, List, Callable

logger = logging.getLogger(__name__)

# API service limits
API_LIMITS = {
    "zai": {"limit": 10000, "period": "daily", "unit": "requests"},
    "gemini": {"limit": 1500, "period": "daily", "unit": "requests"},
    "groq": {"limit": 10000, "period": "daily", "unit": "requests"},
    "cohere": {"limit": 1000, "period": "monthly", "unit": "embeds"},
    "finnhub": {"limit": 30000, "period": "daily", "unit": "calls"},
    "ollama": {"limit": None, "period": None, "unit": "calls"},  # Local, no limit
}

ALERT_THRESHOLDS = [50, 75, 90, 95, 100]


class UsageTracker:
    """
    Tracks API usage across services with:
    - Per-request usage summaries
    - Threshold alerts at 50%, 75%, 90%, 95%, 100%
    - Auto-reset based on daily/monthly periods
    """

    def __init__(self, storage_path: Optional[str] = None, slack_notifier: Optional[Callable] = None):
        """
        Initialize usage tracker.

        Args:
            storage_path: Path to JSON storage file (default: ~/.stock-radar/usage.json)
            slack_notifier: Optional callback function to send Slack notifications
        """
        if storage_path:
            self.storage_path = Path(storage_path)
        else:
            self.storage_path = Path.home() / ".stock-radar" / "usage.json"

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.slack_notifier = slack_notifier
        self._session_usage: Dict[str, Dict] = {}  # Track usage within current session
        self._load_usage()

    def _load_usage(self):
        """Load usage data from storage file."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, "r") as f:
                    self._data = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._data = self._init_data()
        else:
            self._data = self._init_data()

        # Check for reset
        self._check_reset()

    def _init_data(self) -> Dict:
        """Initialize empty usage data structure."""
        now = datetime.now(timezone.utc).isoformat()
        return {
            "services": {
                service: {
                    "count": 0,
                    "tokens": 0,
                    "last_reset": now,
                    "alerted_thresholds": [],
                }
                for service in API_LIMITS
            },
            "created_at": now,
        }

    def _save_usage(self):
        """Save usage data to storage file."""
        try:
            with open(self.storage_path, "w") as f:
                json.dump(self._data, f, indent=2, default=str)
        except IOError as e:
            logger.error(f"Failed to save usage data: {e}")

    def _check_reset(self):
        """Check if any service needs to be reset based on its period."""
        now = datetime.now(timezone.utc)

        for service, config in API_LIMITS.items():
            if config["period"] is None:
                continue

            service_data = self._data["services"].get(service, {})
            last_reset_str = service_data.get("last_reset")

            if not last_reset_str:
                continue

            try:
                last_reset = datetime.fromisoformat(last_reset_str.replace("Z", "+00:00"))
            except ValueError:
                continue

            should_reset = False

            if config["period"] == "daily":
                # Reset if it's a new day (UTC)
                should_reset = last_reset.date() < now.date()
            elif config["period"] == "monthly":
                # Reset if it's a new month
                should_reset = (
                    last_reset.year < now.year or
                    (last_reset.year == now.year and last_reset.month < now.month)
                )

            if should_reset:
                logger.info(f"Resetting usage for {service} (period: {config['period']})")
                self._data["services"][service] = {
                    "count": 0,
                    "tokens": 0,
                    "last_reset": now.isoformat(),
                    "alerted_thresholds": [],
                }

        self._save_usage()

    def track(self, service: str, count: int = 1, tokens: int = 0) -> Dict[str, Any]:
        """
        Track API usage for a service.

        Args:
            service: Service name (zai, gemini, cohere, finnhub, ollama)
            count: Number of API calls made
            tokens: Number of tokens used (for LLMs)

        Returns:
            Current usage stats for the service
        """
        service = service.lower()

        if service not in API_LIMITS:
            logger.warning(f"Unknown service: {service}")
            return {}

        # Update persistent storage
        if service not in self._data["services"]:
            self._data["services"][service] = {
                "count": 0,
                "tokens": 0,
                "last_reset": datetime.now(timezone.utc).isoformat(),
                "alerted_thresholds": [],
            }

        self._data["services"][service]["count"] += count
        self._data["services"][service]["tokens"] += tokens
        self._save_usage()

        # Update session tracking (for per-request summary)
        if service not in self._session_usage:
            self._session_usage[service] = {"count": 0, "tokens": 0}
        self._session_usage[service]["count"] += count
        self._session_usage[service]["tokens"] += tokens

        # Check thresholds
        self._check_thresholds(service)

        return self.get_usage(service)

    def get_usage(self, service: str) -> Dict[str, Any]:
        """Get current usage for a service."""
        service = service.lower()
        config = API_LIMITS.get(service, {})
        service_data = self._data["services"].get(service, {})

        count = service_data.get("count", 0)
        limit = config.get("limit")
        period = config.get("period", "N/A")

        percentage = (count / limit * 100) if limit else 0

        return {
            "service": service,
            "count": count,
            "tokens": service_data.get("tokens", 0),
            "limit": limit,
            "period": period,
            "percentage": round(percentage, 2),
            "remaining": (limit - count) if limit else None,
            "unit": config.get("unit", "calls"),
        }

    def get_all_usage(self) -> List[Dict[str, Any]]:
        """Get usage for all tracked services."""
        return [self.get_usage(service) for service in API_LIMITS]

    def get_session_summary(self, symbol: str = "") -> str:
        """
        Get a formatted summary of usage in the current session.

        Args:
            symbol: Stock symbol being analyzed (for context)

        Returns:
            Formatted usage summary message
        """
        if not self._session_usage:
            return ""

        lines = [f"📊 Usage Summary{f' for {symbol} analysis' if symbol else ''}:"]

        for service, session_data in self._session_usage.items():
            if session_data["count"] == 0:
                continue

            usage = self.get_usage(service)
            config = API_LIMITS.get(service, {})

            token_info = f" ({session_data['tokens']} tokens)" if session_data["tokens"] else ""
            limit_info = f"{usage['count']}/{usage['limit']}" if usage["limit"] else f"{usage['count']}"
            pct_info = f" ({usage['percentage']:.1f}%)" if usage["limit"] else ""

            lines.append(
                f"• {service.capitalize()}: {session_data['count']} {config.get('unit', 'calls')}"
                f"{token_info} → {limit_info} {usage['period']}{pct_info}"
            )

        return "\n".join(lines)

    def clear_session(self):
        """Clear session usage tracking (call after sending summary)."""
        self._session_usage = {}

    def _check_thresholds(self, service: str):
        """Check if usage crossed any alert thresholds."""
        usage = self.get_usage(service)

        if not usage["limit"]:
            return  # No limit to check

        percentage = usage["percentage"]
        service_data = self._data["services"][service]
        alerted = service_data.get("alerted_thresholds", [])

        for threshold in ALERT_THRESHOLDS:
            if percentage >= threshold and threshold not in alerted:
                self._send_threshold_alert(service, usage, threshold)
                alerted.append(threshold)
                service_data["alerted_thresholds"] = alerted
                self._save_usage()

    def _send_threshold_alert(self, service: str, usage: Dict, threshold: int):
        """Send a threshold alert notification."""
        emoji = "⚠️" if threshold < 100 else "🚨"
        message = (
            f"{emoji} API USAGE ALERT: {service.capitalize()} at {threshold}%\n"
            f"Used: {usage['count']:,} / {usage['limit']:,} {usage['period']} {usage['unit']}\n"
            f"Remaining: {usage['remaining']:,}"
        )

        logger.warning(message.replace("\n", " | "))

        if self.slack_notifier:
            try:
                self.slack_notifier(message)
            except Exception as e:
                logger.error(f"Failed to send Slack alert: {e}")

    def send_session_summary(self, symbol: str = "") -> bool:
        """
        Send session usage summary to Slack.

        Args:
            symbol: Stock symbol for context

        Returns:
            True if sent successfully
        """
        summary = self.get_session_summary(symbol)

        if not summary or not self.slack_notifier:
            return False

        try:
            self.slack_notifier(summary)
            self.clear_session()
            return True
        except Exception as e:
            logger.error(f"Failed to send usage summary: {e}")
            return False

    def reset(self, service: Optional[str] = None):
        """
        Manually reset usage counters.

        Args:
            service: Service to reset (or None for all)
        """
        now = datetime.now(timezone.utc).isoformat()

        if service:
            if service in self._data["services"]:
                self._data["services"][service] = {
                    "count": 0,
                    "tokens": 0,
                    "last_reset": now,
                    "alerted_thresholds": [],
                }
                logger.info(f"Reset usage for {service}")
        else:
            self._data = self._init_data()
            logger.info("Reset usage for all services")

        self._save_usage()

    def get_status_report(self) -> str:
        """Get a formatted status report of all API usage."""
        lines = ["📈 API Usage Status", "=" * 30]

        for usage in self.get_all_usage():
            if usage["limit"] is None:
                status = f"{usage['count']} {usage['unit']} (unlimited)"
            else:
                bar_length = 20
                filled = int(usage["percentage"] / 100 * bar_length)
                bar = "█" * filled + "░" * (bar_length - filled)
                status = f"[{bar}] {usage['percentage']:.1f}% ({usage['count']}/{usage['limit']})"

            lines.append(f"{usage['service'].capitalize():10} {status}")

        return "\n".join(lines)


# Global tracker instance (lazy initialization)
_tracker: Optional[UsageTracker] = None


def get_tracker(slack_notifier: Optional[Callable] = None) -> UsageTracker:
    """Get or create the global usage tracker instance."""
    global _tracker
    if _tracker is None:
        _tracker = UsageTracker(slack_notifier=slack_notifier)
    return _tracker
