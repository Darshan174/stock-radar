"""
Change analyzer for competitor intelligence.
Uses Claude API to detect and categorize meaningful changes between crawls.
"""

import os
import json
import logging
import difflib
from typing import Optional
from dataclasses import dataclass

import anthropic

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class Change:
    """Represents a detected change."""
    type: str  # pricing, feature, hiring, partnership, other
    summary: str
    importance: str  # high, mid, low
    details: str


class ChangeAnalyzer:
    """Analyzes competitor content changes using Claude API."""

    # Patterns to filter out trivial changes
    TRIVIAL_PATTERNS = [
        "copyright",
        "all rights reserved",
        "last updated",
        "last modified",
        "privacy policy",
        "terms of service",
        "cookie",
        "utm_",
        "tracking",
        "analytics",
        "footer",
        "navigation",
        "__next",
        "webpack",
        "bundle",
    ]

    # Similarity threshold - below this we consider content too different (site redesign)
    SIMILARITY_THRESHOLD = 0.15
    # Above this threshold, changes are too minor to report
    TRIVIAL_SIMILARITY_THRESHOLD = 0.98

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-opus-4-5-20250514"):
        """
        Initialize the analyzer with Claude API.

        Args:
            api_key: Anthropic API key (defaults to env var)
            model: Claude model to use
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def _compute_similarity(self, old_text: str, new_text: str) -> float:
        """
        Compute similarity ratio between two texts.

        Args:
            old_text: Previous version
            new_text: Current version

        Returns:
            Similarity ratio between 0 and 1
        """
        if not old_text and not new_text:
            return 1.0
        if not old_text or not new_text:
            return 0.0

        return difflib.SequenceMatcher(None, old_text, new_text).ratio()

    def _generate_diff(self, old_text: str, new_text: str, context_lines: int = 3) -> str:
        """
        Generate a unified diff between old and new text.

        Args:
            old_text: Previous version
            new_text: Current version
            context_lines: Number of context lines around changes

        Returns:
            Unified diff string
        """
        old_lines = old_text.splitlines(keepends=True)
        new_lines = new_text.splitlines(keepends=True)

        diff = difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="previous",
            tofile="current",
            n=context_lines
        )

        return "".join(diff)

    def _filter_trivial_lines(self, text: str) -> str:
        """
        Remove lines that typically contain trivial content.

        Args:
            text: Raw text content

        Returns:
            Filtered text
        """
        lines = text.split("\n")
        filtered = []

        for line in lines:
            line_lower = line.lower().strip()
            if not line_lower:
                continue

            is_trivial = any(pattern in line_lower for pattern in self.TRIVIAL_PATTERNS)
            if not is_trivial:
                filtered.append(line)

        return "\n".join(filtered)

    def _truncate_for_api(self, text: str, max_chars: int = 15000) -> str:
        """
        Truncate text to fit within API limits.

        Args:
            text: Text to truncate
            max_chars: Maximum characters

        Returns:
            Truncated text
        """
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n\n[... truncated ...]"

    def analyze_changes(
        self,
        competitor_name: str,
        old_markdown: str,
        new_markdown: str
    ) -> dict:
        """
        Analyze changes between old and new crawls.

        Args:
            competitor_name: Name of the competitor
            old_markdown: Previous crawl markdown content
            new_markdown: Current crawl markdown content

        Returns:
            Dictionary with:
                - has_changes: bool
                - changes: list of change dicts
                - similarity: float
                - diff: str (unified diff)
        """
        result = {
            "competitor_name": competitor_name,
            "has_changes": False,
            "changes": [],
            "similarity": 1.0,
            "diff": "",
            "analysis_status": "success"
        }

        # Handle edge cases
        if not old_markdown and not new_markdown:
            result["analysis_status"] = "no_content"
            return result

        if not old_markdown:
            result["has_changes"] = True
            result["changes"] = [{
                "type": "other",
                "summary": "Initial crawl - no previous data to compare",
                "importance": "low",
                "details": "First crawl of this competitor, baseline established."
            }]
            result["similarity"] = 0.0
            return result

        if not new_markdown:
            result["has_changes"] = True
            result["changes"] = [{
                "type": "other",
                "summary": "Website content unavailable or removed",
                "importance": "high",
                "details": "The competitor's website content could not be retrieved. This may indicate a site issue or major redesign."
            }]
            result["similarity"] = 0.0
            return result

        # Filter trivial content
        filtered_old = self._filter_trivial_lines(old_markdown)
        filtered_new = self._filter_trivial_lines(new_markdown)

        # Compute similarity
        similarity = self._compute_similarity(filtered_old, filtered_new)
        result["similarity"] = similarity

        # Generate diff
        diff = self._generate_diff(filtered_old, filtered_new)
        result["diff"] = diff

        # Check for trivial changes (very high similarity)
        if similarity >= self.TRIVIAL_SIMILARITY_THRESHOLD:
            result["analysis_status"] = "no_meaningful_changes"
            return result

        # Check for site redesign (very low similarity)
        if similarity < self.SIMILARITY_THRESHOLD:
            result["has_changes"] = True
            result["changes"] = [{
                "type": "other",
                "summary": "Major site redesign or restructuring detected",
                "importance": "high",
                "details": f"Content similarity is very low ({similarity:.1%}). The website may have undergone a complete redesign. Manual review recommended."
            }]
            return result

        # Use Claude to analyze the diff semantically
        changes = self._analyze_with_claude(competitor_name, filtered_old, filtered_new, diff)

        if changes:
            result["has_changes"] = True
            result["changes"] = changes

        return result

    def _analyze_with_claude(
        self,
        competitor_name: str,
        old_content: str,
        new_content: str,
        diff: str
    ) -> list[dict]:
        """
        Use Claude to semantically analyze changes.

        Args:
            competitor_name: Name of the competitor
            old_content: Filtered previous content
            new_content: Filtered current content
            diff: Unified diff

        Returns:
            List of change dictionaries
        """
        # Truncate content for API
        truncated_old = self._truncate_for_api(old_content, 10000)
        truncated_new = self._truncate_for_api(new_content, 10000)
        truncated_diff = self._truncate_for_api(diff, 5000)

        prompt = f"""You are a competitive intelligence analyst. Analyze the changes between two versions of {competitor_name}'s website content.

## Previous Content
{truncated_old}

## Current Content
{truncated_new}

## Diff (changes highlighted)
{truncated_diff}

## Task
Identify meaningful business changes. Focus on:
1. **Pricing changes**: New pricing, price increases/decreases, plan changes, discounts
2. **Feature changes**: New features, removed features, feature updates, product announcements
3. **Hiring changes**: New job postings, team growth signals, leadership changes
4. **Partnership changes**: New integrations, partnerships, acquisitions, collaborations
5. **Other significant changes**: Messaging shifts, positioning changes, market expansion

IGNORE trivial changes like:
- Date/time updates
- Minor wording tweaks without meaning change
- Footer/navigation changes
- Tracking code or technical artifacts
- Formatting-only changes

## Response Format
Return a JSON array of changes. Each change should have:
- "type": one of "pricing", "feature", "hiring", "partnership", "other"
- "summary": brief one-line summary (max 100 chars)
- "importance": "high" (immediate business impact), "mid" (notable but not urgent), "low" (minor interest)
- "details": 2-3 sentences explaining the change and its significance

If there are no meaningful changes, return an empty array: []

Return ONLY valid JSON, no other text."""

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2000,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )

            # Extract JSON from response
            response_text = response.content[0].text.strip()

            # Handle potential markdown code blocks
            if response_text.startswith("```"):
                lines = response_text.split("\n")
                response_text = "\n".join(lines[1:-1])

            changes = json.loads(response_text)

            # Validate and clean changes
            validated_changes = []
            for change in changes:
                if self._validate_change(change):
                    validated_changes.append({
                        "type": change.get("type", "other"),
                        "summary": change.get("summary", "")[:150],
                        "importance": change.get("importance", "low"),
                        "details": change.get("details", "")
                    })

            return validated_changes

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse Claude response as JSON: {e}")
            return []
        except anthropic.APIError as e:
            logger.error(f"Claude API error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error analyzing with Claude: {e}")
            return []

    def _validate_change(self, change: dict) -> bool:
        """
        Validate a change dictionary has required fields.

        Args:
            change: Change dictionary to validate

        Returns:
            True if valid
        """
        required_fields = ["type", "summary", "importance"]

        for field in required_fields:
            if field not in change or not change[field]:
                return False

        valid_types = ["pricing", "feature", "hiring", "partnership", "other"]
        valid_importance = ["high", "mid", "low"]

        if change.get("type") not in valid_types:
            return False
        if change.get("importance") not in valid_importance:
            return False

        return True

    def batch_analyze(
        self,
        competitors: list[dict]
    ) -> list[dict]:
        """
        Analyze changes for multiple competitors.

        Args:
            competitors: List of dicts with:
                - name: competitor name
                - old_markdown: previous content
                - new_markdown: current content

        Returns:
            List of analysis results for each competitor
        """
        results = []

        for competitor in competitors:
            name = competitor.get("name", "Unknown")
            old_md = competitor.get("old_markdown", "")
            new_md = competitor.get("new_markdown", "")

            logger.info(f"Analyzing changes for {name}...")

            try:
                analysis = self.analyze_changes(name, old_md, new_md)
                results.append(analysis)

                # Log summary
                if analysis["has_changes"]:
                    change_count = len(analysis["changes"])
                    high_priority = sum(1 for c in analysis["changes"] if c.get("importance") == "high")
                    logger.info(f"{name}: {change_count} changes detected ({high_priority} high priority)")
                else:
                    logger.info(f"{name}: No meaningful changes")

            except Exception as e:
                logger.error(f"Error analyzing {name}: {e}")
                results.append({
                    "competitor_name": name,
                    "has_changes": False,
                    "changes": [],
                    "similarity": 0.0,
                    "diff": "",
                    "analysis_status": "error",
                    "error": str(e)
                })

        return results

    def get_high_priority_changes(self, analysis_results: list[dict]) -> list[dict]:
        """
        Filter for high priority changes across all competitors.

        Args:
            analysis_results: Results from batch_analyze

        Returns:
            List of high priority changes with competitor context
        """
        high_priority = []

        for result in analysis_results:
            if not result.get("has_changes"):
                continue

            for change in result.get("changes", []):
                if change.get("importance") == "high":
                    high_priority.append({
                        "competitor": result.get("competitor_name"),
                        **change
                    })

        return high_priority


if __name__ == "__main__":
    # Test the analyzer
    analyzer = ChangeAnalyzer()

    # Example test data
    old_content = """
    # Acme Corp

    ## Pricing
    - Starter: $10/month
    - Pro: $29/month
    - Enterprise: Contact us

    ## Features
    - Real-time sync
    - Team collaboration
    - API access
    """

    new_content = """
    # Acme Corp

    ## Pricing
    - Starter: $12/month
    - Pro: $29/month
    - Business: $79/month
    - Enterprise: Contact us

    ## Features
    - Real-time sync
    - Team collaboration
    - API access
    - AI-powered insights (NEW)

    ## Careers
    We're hiring! Join our growing team.
    """

    result = analyzer.analyze_changes("Acme Corp", old_content, new_content)
    print(json.dumps(result, indent=2))
