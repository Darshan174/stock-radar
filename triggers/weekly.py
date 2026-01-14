"""
Weekly competitor intelligence crawl orchestration for Trigger.dev.

This module handles the complete flow:
crawl -> embed -> analyze -> alert

Designed to be triggered via Trigger.dev webhook or scheduled task.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from enum import Enum

import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)
from supabase import create_client, Client
from anthropic import Anthropic

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("research-radar.weekly")


# -----------------------------------------------------------------------------
# Data Classes
# -----------------------------------------------------------------------------

class StageStatus(str, Enum):
    """Status of each pipeline stage."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class StageResult:
    """Result of a single pipeline stage."""
    stage: str
    status: StageStatus
    duration_ms: int = 0
    error: Optional[str] = None
    data: Optional[dict] = None


@dataclass
class CompetitorResult:
    """Result of processing a single competitor."""
    competitor_id: int
    competitor_name: str
    url: str
    stages: list[StageResult] = field(default_factory=list)
    changes_detected: bool = False
    change_summary: Optional[str] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        """Check if all stages succeeded or were skipped."""
        return all(
            s.status in (StageStatus.SUCCESS, StageStatus.SKIPPED)
            for s in self.stages
        )


@dataclass
class ExecutionSummary:
    """Summary of the entire weekly crawl execution."""
    run_id: str
    started_at: str
    completed_at: Optional[str] = None
    total_competitors: int = 0
    successful: int = 0
    failed: int = 0
    changes_detected: int = 0
    alerts_sent: int = 0
    competitor_results: list[CompetitorResult] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "total_competitors": self.total_competitors,
            "successful": self.successful,
            "failed": self.failed,
            "changes_detected": self.changes_detected,
            "alerts_sent": self.alerts_sent,
            "competitor_results": [asdict(r) for r in self.competitor_results],
            "error": self.error,
        }


# -----------------------------------------------------------------------------
# Service Clients
# -----------------------------------------------------------------------------

class ServiceClients:
    """Manages connections to external services."""

    def __init__(self):
        self._supabase: Optional[Client] = None
        self._anthropic: Optional[Anthropic] = None
        self._firecrawl_key: Optional[str] = None
        self._slack_token: Optional[str] = None
        self._slack_channel: Optional[str] = None
        self._ollama_url: Optional[str] = None

    @property
    def supabase(self) -> Client:
        """Get or create Supabase client."""
        if self._supabase is None:
            url = os.getenv("SUPABASE_URL")
            key = os.getenv("SUPABASE_KEY")
            if not url or not key:
                raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")
            self._supabase = create_client(url, key)
        return self._supabase

    @property
    def anthropic(self) -> Anthropic:
        """Get or create Anthropic client."""
        if self._anthropic is None:
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY must be set")
            self._anthropic = Anthropic(api_key=api_key)
        return self._anthropic

    @property
    def firecrawl_key(self) -> str:
        """Get Firecrawl API key."""
        if self._firecrawl_key is None:
            self._firecrawl_key = os.getenv("FIRECRAWL_API_KEY")
            if not self._firecrawl_key:
                raise ValueError("FIRECRAWL_API_KEY must be set")
        return self._firecrawl_key

    @property
    def slack_token(self) -> str:
        """Get Slack bot token."""
        if self._slack_token is None:
            self._slack_token = os.getenv("SLACK_BOT_TOKEN")
            if not self._slack_token:
                raise ValueError("SLACK_BOT_TOKEN must be set")
        return self._slack_token

    @property
    def slack_channel(self) -> str:
        """Get Slack channel ID."""
        if self._slack_channel is None:
            self._slack_channel = os.getenv("SLACK_CHANNEL_ID")
            if not self._slack_channel:
                raise ValueError("SLACK_CHANNEL_ID must be set")
        return self._slack_channel

    @property
    def ollama_url(self) -> str:
        """Get Ollama API URL."""
        if self._ollama_url is None:
            self._ollama_url = os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        return self._ollama_url


# Global service clients instance
clients = ServiceClients()


# -----------------------------------------------------------------------------
# Stage 1: Get Competitors
# -----------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
)
def get_competitors() -> list[dict]:
    """
    Fetch all active competitors from Supabase.

    Returns:
        List of competitor records with id, name, url fields.
    """
    logger.info("Fetching competitors from Supabase")

    try:
        result = clients.supabase.table("competitors").select(
            "id, name, url"
        ).eq("active", True).execute()

        competitors = result.data if result.data else []
        logger.info(f"Found {len(competitors)} active competitors")
        return competitors

    except Exception as e:
        logger.error(f"Failed to fetch competitors: {str(e)}")
        raise


# -----------------------------------------------------------------------------
# Stage 2: Crawl with Firecrawl
# -----------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
)
def crawl_competitor(url: str) -> dict:
    """
    Scrape a competitor URL using Firecrawl API.

    Args:
        url: The competitor website URL to scrape.

    Returns:
        Dictionary with markdown content and metadata.
    """
    logger.info(f"Crawling {url}")

    response = requests.post(
        "https://api.firecrawl.dev/v1/scrape",
        headers={
            "Authorization": f"Bearer {clients.firecrawl_key}",
            "Content-Type": "application/json",
        },
        json={
            "url": url,
            "formats": ["markdown", "html"],
            "onlyMainContent": True,
            "removeTags": ["nav", "footer", "script", "style"],
        },
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if not data.get("success"):
        raise ValueError(f"Firecrawl failed: {data.get('error', 'Unknown error')}")

    crawl_data = {
        "url": url,
        "markdown": data.get("data", {}).get("markdown", ""),
        "html": data.get("data", {}).get("html", ""),
        "crawl_date": datetime.utcnow().isoformat(),
    }

    logger.info(f"Successfully crawled {url} ({len(crawl_data['markdown'])} chars)")
    return crawl_data


# -----------------------------------------------------------------------------
# Stage 3: Generate Embeddings with Ollama
# -----------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.RequestException, ConnectionError)),
)
def generate_embedding(text: str) -> list[float]:
    """
    Generate embedding vector using Ollama.

    Args:
        text: Text to embed (truncated to 2000 chars).

    Returns:
        Embedding vector as list of floats.
    """
    # Truncate to avoid token limits
    truncated_text = text[:2000]

    response = requests.post(
        f"{clients.ollama_url}/api/embed",
        json={
            "model": "nomic-embed-text",
            "input": truncated_text,
        },
        timeout=30,
    )
    response.raise_for_status()

    embeddings = response.json().get("embeddings", [])
    if embeddings:
        return embeddings[0]
    return []


# -----------------------------------------------------------------------------
# Stage 4: Store Crawl in Supabase
# -----------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def store_crawl(
    competitor_id: int,
    crawl_data: dict,
    embedding: list[float],
) -> dict:
    """
    Store crawl data with embedding in Supabase.

    Args:
        competitor_id: ID of the competitor.
        crawl_data: Crawl data from Firecrawl.
        embedding: Embedding vector from Ollama.

    Returns:
        Stored crawl record.
    """
    logger.info(f"Storing crawl for competitor {competitor_id}")

    data = {
        "competitor_id": competitor_id,
        "markdown": crawl_data["markdown"],
        "html": crawl_data.get("html", ""),
        "url": crawl_data["url"],
        "crawl_date": crawl_data["crawl_date"],
        "embedding": embedding if embedding else None,
    }

    result = clients.supabase.table("crawls").insert(data).execute()
    stored = result.data[0] if result.data else data
    logger.info(f"Stored crawl ID: {stored.get('id')}")
    return stored


# -----------------------------------------------------------------------------
# Stage 5: Get Previous Crawl
# -----------------------------------------------------------------------------

def get_previous_crawl(competitor_id: int, current_crawl_id: int) -> Optional[dict]:
    """
    Get the previous crawl for comparison.

    Args:
        competitor_id: ID of the competitor.
        current_crawl_id: ID of current crawl to exclude.

    Returns:
        Previous crawl record or None if first crawl.
    """
    logger.info(f"Fetching previous crawl for competitor {competitor_id}")

    try:
        result = clients.supabase.table("crawls").select("*").eq(
            "competitor_id", competitor_id
        ).neq(
            "id", current_crawl_id
        ).order(
            "crawl_date", desc=True
        ).limit(1).execute()

        if result.data:
            logger.info(f"Found previous crawl ID: {result.data[0].get('id')}")
            return result.data[0]

        logger.info("No previous crawl found (first crawl)")
        return None

    except Exception as e:
        logger.error(f"Error fetching previous crawl: {str(e)}")
        return None


# -----------------------------------------------------------------------------
# Stage 6: Analyze Changes with Claude
# -----------------------------------------------------------------------------

ANALYSIS_PROMPT = """You are analyzing competitor website changes for a business intelligence system.

Compare the PREVIOUS content with the CURRENT content and identify significant changes.

Focus on:
1. Pricing changes (new prices, discounts, plan changes)
2. Feature announcements (new features, deprecations)
3. Hiring signals (job postings, team growth)
4. Strategic shifts (partnerships, integrations, positioning)

PREVIOUS CONTENT:
{previous_content}

CURRENT CONTENT:
{current_content}

Respond in JSON format:
{{
    "has_changes": true/false,
    "changes": [
        {{
            "type": "pricing|feature|hiring|strategic",
            "importance": "high|mid|low",
            "summary": "Brief description of the change",
            "details": "More detailed explanation"
        }}
    ],
    "overall_summary": "One paragraph summary of all changes"
}}

If there are no significant changes, return:
{{
    "has_changes": false,
    "changes": [],
    "overall_summary": "No significant changes detected."
}}
"""


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=4, max=30),
)
def analyze_changes(
    previous_content: str,
    current_content: str,
    competitor_name: str,
) -> dict:
    """
    Analyze changes between crawls using Claude.

    Args:
        previous_content: Markdown from previous crawl.
        current_content: Markdown from current crawl.
        competitor_name: Name of competitor for context.

    Returns:
        Analysis result with detected changes.
    """
    logger.info(f"Analyzing changes for {competitor_name}")

    # Truncate content to fit context window
    max_content_length = 15000
    prev_truncated = previous_content[:max_content_length]
    curr_truncated = current_content[:max_content_length]

    message = clients.anthropic.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        messages=[
            {
                "role": "user",
                "content": ANALYSIS_PROMPT.format(
                    previous_content=prev_truncated,
                    current_content=curr_truncated,
                ),
            }
        ],
    )

    # Parse response
    response_text = message.content[0].text

    # Extract JSON from response
    try:
        # Handle potential markdown code blocks
        if "```json" in response_text:
            json_start = response_text.find("```json") + 7
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()
        elif "```" in response_text:
            json_start = response_text.find("```") + 3
            json_end = response_text.find("```", json_start)
            response_text = response_text[json_start:json_end].strip()

        analysis = json.loads(response_text)
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse Claude response as JSON: {e}")
        analysis = {
            "has_changes": False,
            "changes": [],
            "overall_summary": "Analysis failed to parse.",
            "raw_response": response_text,
        }

    logger.info(f"Analysis complete: has_changes={analysis.get('has_changes')}")
    return analysis


# -----------------------------------------------------------------------------
# Stage 7: Store Changes
# -----------------------------------------------------------------------------

def store_changes(crawl_id: int, analysis: dict) -> list[dict]:
    """
    Store detected changes in Supabase.

    Args:
        crawl_id: ID of the crawl where changes were detected.
        analysis: Analysis result from Claude.

    Returns:
        List of stored change records.
    """
    if not analysis.get("has_changes") or not analysis.get("changes"):
        return []

    stored_changes = []
    for change in analysis["changes"]:
        try:
            data = {
                "crawl_id": crawl_id,
                "type": change.get("type", "unknown"),
                "summary": change.get("summary", ""),
                "importance": change.get("importance", "low"),
                "details": change.get("details", ""),
                "detected_at": datetime.utcnow().isoformat(),
            }

            result = clients.supabase.table("changes").insert(data).execute()
            if result.data:
                stored_changes.append(result.data[0])
                logger.info(f"Stored change: {change.get('summary')}")

        except Exception as e:
            logger.error(f"Failed to store change: {str(e)}")

    return stored_changes


# -----------------------------------------------------------------------------
# Stage 8: Send Slack Alert
# -----------------------------------------------------------------------------

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
)
def send_slack_alert(
    competitor_name: str,
    analysis: dict,
    change_ids: list[int],
) -> Optional[str]:
    """
    Send Slack alert for detected changes.

    Args:
        competitor_name: Name of the competitor.
        analysis: Analysis result with changes.
        change_ids: IDs of stored changes.

    Returns:
        Slack message timestamp if successful.
    """
    if not analysis.get("has_changes"):
        return None

    logger.info(f"Sending Slack alert for {competitor_name}")

    # Build message blocks
    changes = analysis.get("changes", [])
    importance_emoji = {"high": ":", "mid": ":", "low": ":"}

    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"Competitor Update: {competitor_name}",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": f"*Summary:* {analysis.get('overall_summary', 'Changes detected.')}",
            },
        },
        {"type": "divider"},
    ]

    for change in changes[:5]:  # Limit to 5 changes in alert
        emoji = importance_emoji.get(change.get("importance", "low"), ":")
        blocks.append({
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"{emoji} *{change.get('type', 'Change').upper()}* "
                    f"({change.get('importance', 'low')})\n"
                    f"{change.get('summary', '')}"
                ),
            },
        })

    blocks.append({
        "type": "context",
        "elements": [
            {
                "type": "mrkdwn",
                "text": f"Detected at {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}",
            }
        ],
    })

    # Send to Slack
    response = requests.post(
        "https://slack.com/api/chat.postMessage",
        headers={
            "Authorization": f"Bearer {clients.slack_token}",
            "Content-Type": "application/json",
        },
        json={
            "channel": clients.slack_channel,
            "blocks": blocks,
            "text": f"Competitor Update: {competitor_name}",
        },
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()

    if not result.get("ok"):
        logger.error(f"Slack API error: {result.get('error')}")
        return None

    slack_ts = result.get("ts")
    logger.info(f"Slack alert sent: {slack_ts}")

    # Record alerts in database
    for change_id in change_ids:
        try:
            clients.supabase.table("alerts").insert({
                "change_id": change_id,
                "slack_ts": slack_ts,
                "alerted_at": datetime.utcnow().isoformat(),
            }).execute()
        except Exception as e:
            logger.warning(f"Failed to record alert for change {change_id}: {e}")

    return slack_ts


# -----------------------------------------------------------------------------
# Competitor Processing Pipeline
# -----------------------------------------------------------------------------

def process_competitor(competitor: dict) -> CompetitorResult:
    """
    Process a single competitor through the full pipeline.

    Stages:
    1. Crawl website with Firecrawl
    2. Generate embeddings with Ollama
    3. Store crawl in Supabase
    4. Get previous crawl
    5. Analyze changes with Claude
    6. Store changes
    7. Send Slack alert

    Args:
        competitor: Competitor record with id, name, url.

    Returns:
        CompetitorResult with all stage results.
    """
    result = CompetitorResult(
        competitor_id=competitor["id"],
        competitor_name=competitor["name"],
        url=competitor["url"],
    )

    crawl_data = None
    crawl_record = None
    embedding = []
    previous_crawl = None
    analysis = None
    stored_changes = []

    # Stage 1: Crawl
    stage_start = datetime.utcnow()
    try:
        crawl_data = crawl_competitor(competitor["url"])
        result.stages.append(StageResult(
            stage="crawl",
            status=StageStatus.SUCCESS,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            data={"chars": len(crawl_data.get("markdown", ""))},
        ))
    except Exception as e:
        logger.error(f"Crawl failed for {competitor['name']}: {e}")
        result.stages.append(StageResult(
            stage="crawl",
            status=StageStatus.FAILED,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            error=str(e),
        ))
        result.error = f"Crawl failed: {e}"
        return result

    # Stage 2: Generate Embedding
    stage_start = datetime.utcnow()
    try:
        embedding = generate_embedding(crawl_data["markdown"])
        result.stages.append(StageResult(
            stage="embed",
            status=StageStatus.SUCCESS,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            data={"dimensions": len(embedding)},
        ))
    except Exception as e:
        logger.warning(f"Embedding failed for {competitor['name']}: {e}")
        result.stages.append(StageResult(
            stage="embed",
            status=StageStatus.FAILED,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            error=str(e),
        ))
        # Continue without embedding - not critical

    # Stage 3: Store Crawl
    stage_start = datetime.utcnow()
    try:
        crawl_record = store_crawl(competitor["id"], crawl_data, embedding)
        result.stages.append(StageResult(
            stage="store",
            status=StageStatus.SUCCESS,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            data={"crawl_id": crawl_record.get("id")},
        ))
    except Exception as e:
        logger.error(f"Store failed for {competitor['name']}: {e}")
        result.stages.append(StageResult(
            stage="store",
            status=StageStatus.FAILED,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            error=str(e),
        ))
        result.error = f"Store failed: {e}"
        return result

    # Stage 4: Get Previous Crawl
    stage_start = datetime.utcnow()
    try:
        previous_crawl = get_previous_crawl(
            competitor["id"],
            crawl_record.get("id"),
        )
        if previous_crawl:
            result.stages.append(StageResult(
                stage="get_previous",
                status=StageStatus.SUCCESS,
                duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
                data={"previous_crawl_id": previous_crawl.get("id")},
            ))
        else:
            result.stages.append(StageResult(
                stage="get_previous",
                status=StageStatus.SKIPPED,
                duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
                data={"reason": "first_crawl"},
            ))
            return result  # No previous crawl, nothing to compare
    except Exception as e:
        logger.warning(f"Get previous failed for {competitor['name']}: {e}")
        result.stages.append(StageResult(
            stage="get_previous",
            status=StageStatus.FAILED,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            error=str(e),
        ))
        return result

    # Stage 5: Analyze Changes
    stage_start = datetime.utcnow()
    try:
        analysis = analyze_changes(
            previous_crawl.get("markdown", ""),
            crawl_data["markdown"],
            competitor["name"],
        )
        result.stages.append(StageResult(
            stage="analyze",
            status=StageStatus.SUCCESS,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            data={
                "has_changes": analysis.get("has_changes"),
                "change_count": len(analysis.get("changes", [])),
            },
        ))
        result.changes_detected = analysis.get("has_changes", False)
        result.change_summary = analysis.get("overall_summary")
    except Exception as e:
        logger.error(f"Analysis failed for {competitor['name']}: {e}")
        result.stages.append(StageResult(
            stage="analyze",
            status=StageStatus.FAILED,
            duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
            error=str(e),
        ))
        result.error = f"Analysis failed: {e}"
        return result

    # Stage 6 & 7: Store Changes and Alert (only if changes detected)
    if analysis.get("has_changes"):
        # Stage 6: Store Changes
        stage_start = datetime.utcnow()
        try:
            stored_changes = store_changes(crawl_record.get("id"), analysis)
            result.stages.append(StageResult(
                stage="store_changes",
                status=StageStatus.SUCCESS,
                duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
                data={"changes_stored": len(stored_changes)},
            ))
        except Exception as e:
            logger.error(f"Store changes failed for {competitor['name']}: {e}")
            result.stages.append(StageResult(
                stage="store_changes",
                status=StageStatus.FAILED,
                duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
                error=str(e),
            ))

        # Stage 7: Send Alert
        stage_start = datetime.utcnow()
        try:
            change_ids = [c.get("id") for c in stored_changes if c.get("id")]
            slack_ts = send_slack_alert(competitor["name"], analysis, change_ids)
            result.stages.append(StageResult(
                stage="alert",
                status=StageStatus.SUCCESS if slack_ts else StageStatus.FAILED,
                duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
                data={"slack_ts": slack_ts},
            ))
        except Exception as e:
            logger.error(f"Alert failed for {competitor['name']}: {e}")
            result.stages.append(StageResult(
                stage="alert",
                status=StageStatus.FAILED,
                duration_ms=int((datetime.utcnow() - stage_start).total_seconds() * 1000),
                error=str(e),
            ))
    else:
        result.stages.append(StageResult(
            stage="store_changes",
            status=StageStatus.SKIPPED,
            duration_ms=0,
            data={"reason": "no_changes"},
        ))
        result.stages.append(StageResult(
            stage="alert",
            status=StageStatus.SKIPPED,
            duration_ms=0,
            data={"reason": "no_changes"},
        ))

    return result


# -----------------------------------------------------------------------------
# Main Orchestration
# -----------------------------------------------------------------------------

def run_weekly_crawl(run_id: Optional[str] = None) -> ExecutionSummary:
    """
    Main orchestration function for weekly competitor crawl.

    Flow:
    1. Fetch all active competitors
    2. For each competitor:
       - Crawl website
       - Generate embeddings
       - Store crawl
       - Get previous crawl
       - Analyze changes
       - Store changes
       - Send alert
    3. Return execution summary

    Args:
        run_id: Optional run identifier for tracking.

    Returns:
        ExecutionSummary with results for all competitors.
    """
    if run_id is None:
        run_id = f"weekly_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"

    logger.info(f"Starting weekly crawl run: {run_id}")
    summary = ExecutionSummary(
        run_id=run_id,
        started_at=datetime.utcnow().isoformat(),
    )

    # Fetch competitors
    try:
        competitors = get_competitors()
        summary.total_competitors = len(competitors)
    except Exception as e:
        logger.error(f"Failed to fetch competitors: {e}")
        summary.error = f"Failed to fetch competitors: {e}"
        summary.completed_at = datetime.utcnow().isoformat()
        return summary

    if not competitors:
        logger.warning("No active competitors found")
        summary.completed_at = datetime.utcnow().isoformat()
        return summary

    # Process each competitor
    for competitor in competitors:
        logger.info(f"Processing competitor: {competitor['name']}")
        try:
            result = process_competitor(competitor)
            summary.competitor_results.append(result)

            if result.success:
                summary.successful += 1
            else:
                summary.failed += 1

            if result.changes_detected:
                summary.changes_detected += 1
                # Count alerts sent
                alert_stage = next(
                    (s for s in result.stages if s.stage == "alert"),
                    None
                )
                if alert_stage and alert_stage.status == StageStatus.SUCCESS:
                    summary.alerts_sent += 1

        except Exception as e:
            logger.error(f"Unexpected error processing {competitor['name']}: {e}")
            summary.failed += 1
            summary.competitor_results.append(CompetitorResult(
                competitor_id=competitor["id"],
                competitor_name=competitor["name"],
                url=competitor["url"],
                error=str(e),
            ))

    summary.completed_at = datetime.utcnow().isoformat()
    logger.info(
        f"Weekly crawl complete: {summary.successful}/{summary.total_competitors} "
        f"successful, {summary.changes_detected} changes detected, "
        f"{summary.alerts_sent} alerts sent"
    )

    return summary


# -----------------------------------------------------------------------------
# Trigger.dev Webhook Handler
# -----------------------------------------------------------------------------

def handle_trigger_webhook(payload: dict) -> dict:
    """
    Handle incoming Trigger.dev webhook request.

    This is the main entry point for Trigger.dev scheduled jobs.

    Args:
        payload: Webhook payload from Trigger.dev.

    Returns:
        Response dictionary with execution results.
    """
    run_id = payload.get("runId") or payload.get("run_id")

    logger.info(f"Received Trigger.dev webhook: run_id={run_id}")

    try:
        summary = run_weekly_crawl(run_id)
        return {
            "success": True,
            "summary": summary.to_dict(),
        }
    except Exception as e:
        logger.error(f"Weekly crawl failed: {e}")
        return {
            "success": False,
            "error": str(e),
        }


# -----------------------------------------------------------------------------
# Flask/FastAPI Endpoint (for local testing or direct webhook)
# -----------------------------------------------------------------------------

def create_webhook_endpoint():
    """
    Create a simple webhook endpoint for Trigger.dev.

    Returns a Flask or FastAPI-compatible handler function.
    """
    from flask import Flask, request, jsonify

    app = Flask(__name__)

    @app.route("/webhook/weekly-crawl", methods=["POST"])
    def webhook_handler():
        payload = request.get_json() or {}
        result = handle_trigger_webhook(payload)
        status_code = 200 if result.get("success") else 500
        return jsonify(result), status_code

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"}), 200

    return app


# -----------------------------------------------------------------------------
# CLI Entry Point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Research Radar Weekly Crawl Orchestration"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        help="Optional run identifier",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start webhook server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Server port (default: 8080)",
    )

    args = parser.parse_args()

    if args.serve:
        # Start webhook server
        app = create_webhook_endpoint()
        logger.info(f"Starting webhook server on port {args.port}")
        app.run(host="0.0.0.0", port=args.port)
    else:
        # Run crawl directly
        from dotenv import load_dotenv
        load_dotenv()

        summary = run_weekly_crawl(args.run_id)
        print(json.dumps(summary.to_dict(), indent=2))
