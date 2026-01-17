"""
Supabase storage and vector embeddings using Ollama.
Persists crawls, competitors, changes, and alerts with semantic search capabilities.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional, Any

import requests
from supabase import create_client, Client

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class OllamaEmbeddings:
    """Generate embeddings using local Ollama instance with nomic-embed-text model."""

    # nomic-embed-text produces 768-dimensional embeddings
    EMBEDDING_DIMENSION = 768

    def __init__(
        self,
        api_url: Optional[str] = None,
        model: str = "nomic-embed-text",
        timeout: int = 60
    ):
        """
        Initialize Ollama embeddings client.

        Args:
            api_url: Ollama API URL (defaults to OLLAMA_API_URL env var or localhost)
            model: Embedding model name (defaults to nomic-embed-text)
            timeout: Request timeout in seconds
        """
        self.api_url = api_url or os.getenv("OLLAMA_API_URL", "http://localhost:11434")
        self.model = model
        self.timeout = timeout
        logger.info(f"Initialized OllamaEmbeddings with model={model}, url={self.api_url}")

    def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding vector for text using Ollama.

        Args:
            text: Text to embed (will be truncated if too long)

        Returns:
            Embedding vector as list of floats, or empty list on failure
        """
        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return []

        # Truncate very long text to avoid timeouts (nomic-embed-text handles ~8k tokens)
        max_chars = 8000
        if len(text) > max_chars:
            logger.debug(f"Truncating text from {len(text)} to {max_chars} chars")
            text = text[:max_chars]

        try:
            response = requests.post(
                f"{self.api_url}/api/embed",
                json={
                    "model": self.model,
                    "input": text
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data.get("embeddings", [])

            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                logger.debug(f"Generated embedding with {len(embedding)} dimensions")
                return embedding

            logger.warning("No embeddings returned from Ollama")
            return []

        except requests.exceptions.Timeout:
            logger.error(f"Timeout generating embedding (>{self.timeout}s)")
            return []
        except requests.exceptions.ConnectionError:
            logger.error(f"Failed to connect to Ollama at {self.api_url}")
            return []
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from Ollama: {e.response.status_code} - {e.response.text}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {type(e).__name__}: {str(e)}")
            return []

    def is_available(self) -> bool:
        """Check if Ollama service is available and model is loaded."""
        try:
            response = requests.get(
                f"{self.api_url}/api/tags",
                timeout=5
            )
            response.raise_for_status()
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]
            return self.model in model_names
        except Exception as e:
            logger.warning(f"Ollama availability check failed: {str(e)}")
            return False


class SupabaseStorage:
    """
    Manages competitor intelligence data in Supabase with vector search.

    Tables:
        - competitors: Tracked competitor companies
        - crawls: Web crawl results with embeddings
        - changes: Detected changes from crawls
        - alerts: Sent Slack alerts for changes
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        ollama_url: Optional[str] = None
    ):
        """
        Initialize Supabase client and embeddings.

        Args:
            url: Supabase project URL (defaults to SUPABASE_URL env var)
            key: Supabase API key (defaults to SUPABASE_KEY env var)
            ollama_url: Ollama API URL for embeddings

        Raises:
            ValueError: If Supabase credentials are missing
        """
        url = url or os.getenv("SUPABASE_URL")
        key = key or os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                "Supabase credentials required. Set SUPABASE_URL and SUPABASE_KEY "
                "environment variables or pass them as arguments."
            )

        # Create Supabase client
        self.client: Client = create_client(url, key)
        self.embeddings = OllamaEmbeddings(api_url=ollama_url)

        logger.info(f"Initialized SupabaseStorage connected to {url}")

    def ensure_schema(self) -> bool:
        """
        Verify that required tables and extensions exist in the database.

        This method checks for the existence of required tables but does not
        create them - schema should be managed via migrations.

        Returns:
            True if schema is valid, False otherwise
        """
        required_tables = ["competitors", "crawls", "changes", "alerts"]
        missing_tables = []

        try:
            for table in required_tables:
                try:
                    # Attempt to query each table to verify it exists
                    result = self.client.table(table).select("*").limit(0).execute()
                    logger.debug(f"Table '{table}' exists")
                except Exception as e:
                    error_msg = str(e).lower()
                    if "does not exist" in error_msg or "relation" in error_msg:
                        missing_tables.append(table)
                        logger.warning(f"Table '{table}' not found")
                    else:
                        # Other errors might be permission issues, still log them
                        logger.warning(f"Error checking table '{table}': {str(e)}")

            if missing_tables:
                logger.error(
                    f"Missing required tables: {missing_tables}. "
                    "Please run database migrations to create the schema."
                )
                return False

            # Check if search_crawls function exists by calling it with empty params
            try:
                self.client.rpc(
                    "search_crawls",
                    {"query_embedding": [0.0] * OllamaEmbeddings.EMBEDDING_DIMENSION, "match_count": 0}
                ).execute()
                logger.debug("RPC function 'search_crawls' exists")
            except Exception as e:
                if "function" in str(e).lower() and "does not exist" in str(e).lower():
                    logger.warning(
                        "RPC function 'search_crawls' not found. "
                        "Semantic search may not work correctly."
                    )

            logger.info("Schema verification completed successfully")
            return True

        except Exception as e:
            logger.error(f"Error verifying schema: {type(e).__name__}: {str(e)}")
            return False

    def add_competitor(self, name: str, url: str) -> dict[str, Any]:
        """
        Add a new competitor to track.

        Args:
            name: Competitor company name
            url: Competitor website URL (should be the main domain)

        Returns:
            Created competitor record with id, name, url, created_at

        Raises:
            ValueError: If name or url is empty
            Exception: On database errors
        """
        if not name or not name.strip():
            raise ValueError("Competitor name cannot be empty")
        if not url or not url.strip():
            raise ValueError("Competitor URL cannot be empty")

        # Normalize URL
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"

        try:
            data = {
                "name": name.strip(),
                "url": url,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("competitors").insert(data).execute()

            if result.data:
                competitor = result.data[0]
                logger.info(f"Added competitor: {name} (id={competitor.get('id')})")
                return competitor

            logger.warning(f"No data returned when adding competitor: {name}")
            return data

        except Exception as e:
            error_msg = str(e)
            if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
                logger.error(f"Competitor already exists: {name} ({url})")
                raise ValueError(f"Competitor '{name}' or URL '{url}' already exists")
            logger.error(f"Error adding competitor '{name}': {str(e)}")
            raise

    def get_competitor(self, competitor_id: int) -> Optional[dict[str, Any]]:
        """
        Get a competitor by ID.

        Args:
            competitor_id: The competitor's database ID

        Returns:
            Competitor record or None if not found
        """
        try:
            result = self.client.table("competitors").select("*").eq(
                "id", competitor_id
            ).execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error getting competitor {competitor_id}: {str(e)}")
            return None

    def list_competitors(self) -> list[dict[str, Any]]:
        """
        List all tracked competitors.

        Returns:
            List of competitor records
        """
        try:
            result = self.client.table("competitors").select("*").order(
                "created_at", desc=False
            ).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error listing competitors: {str(e)}")
            return []

    def store_crawl(
        self,
        competitor_id: int,
        markdown: str,
        html: str,
        url: str
    ) -> dict[str, Any]:
        """
        Store a web crawl with vector embedding for semantic search.

        Args:
            competitor_id: ID of the competitor this crawl belongs to
            markdown: Extracted markdown content from the page
            html: Raw HTML content from the page
            url: The specific URL that was crawled

        Returns:
            Created crawl record with id, embedding status, etc.

        Raises:
            ValueError: If required fields are missing
            Exception: On database errors
        """
        if not competitor_id:
            raise ValueError("competitor_id is required")
        if not url or not url.strip():
            raise ValueError("url is required")

        try:
            # Generate embedding for the markdown content
            # Use first portion for embedding to balance context and performance
            embedding_text = markdown[:4000] if markdown else ""
            embedding = self.embeddings.embed_text(embedding_text) if embedding_text else []

            crawl_date = datetime.now(timezone.utc).isoformat()

            data = {
                "competitor_id": competitor_id,
                "markdown": markdown or "",
                "html": html or "",
                "url": url.strip(),
                "crawl_date": crawl_date,
                "embedding": embedding if embedding else None
            }

            result = self.client.table("crawls").insert(data).execute()

            if result.data:
                crawl = result.data[0]
                embedding_status = "with embedding" if embedding else "without embedding"
                logger.info(
                    f"Stored crawl for competitor {competitor_id} "
                    f"(id={crawl.get('id')}, {len(markdown or '')} chars, {embedding_status})"
                )
                return crawl

            logger.warning(f"No data returned when storing crawl for competitor {competitor_id}")
            return data

        except Exception as e:
            logger.error(f"Error storing crawl for competitor {competitor_id}: {str(e)}")
            raise

    def get_latest_crawl(self, competitor_id: int) -> Optional[dict[str, Any]]:
        """
        Get the most recent crawl for a competitor.

        Args:
            competitor_id: ID of the competitor

        Returns:
            Latest crawl record or None if no crawls exist
        """
        try:
            result = self.client.table("crawls").select(
                "id, competitor_id, markdown, html, url, crawl_date"
            ).eq(
                "competitor_id", competitor_id
            ).order(
                "crawl_date", desc=True
            ).limit(1).execute()

            if result.data:
                crawl = result.data[0]
                logger.debug(f"Found latest crawl for competitor {competitor_id}: {crawl.get('id')}")
                return crawl

            logger.debug(f"No crawls found for competitor {competitor_id}")
            return None

        except Exception as e:
            logger.error(f"Error getting latest crawl for competitor {competitor_id}: {str(e)}")
            return None

    def get_crawl_history(
        self,
        competitor_id: int,
        limit: int = 10
    ) -> list[dict[str, Any]]:
        """
        Get crawl history for a competitor.

        Args:
            competitor_id: ID of the competitor
            limit: Maximum number of crawls to return

        Returns:
            List of crawl records, most recent first
        """
        try:
            result = self.client.table("crawls").select(
                "id, competitor_id, url, crawl_date"
            ).eq(
                "competitor_id", competitor_id
            ).order(
                "crawl_date", desc=True
            ).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting crawl history for competitor {competitor_id}: {str(e)}")
            return []

    def semantic_search(
        self,
        query: str,
        competitor_id: Optional[int] = None,
        limit: int = 5,
        match_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Search crawls using semantic similarity via pgvector.

        This method requires the 'search_crawls' RPC function to be defined
        in Supabase. See migrations for the function definition.

        Args:
            query: Natural language search query
            competitor_id: Filter results to a specific competitor (optional)
            limit: Maximum number of results to return
            match_threshold: Minimum similarity score (0-1, higher is more similar)

        Returns:
            List of matching crawl records with similarity scores
        """
        if not query or not query.strip():
            logger.warning("Empty query provided for semantic search")
            return []

        try:
            # Generate embedding for the search query
            query_embedding = self.embeddings.embed_text(query.strip())

            if not query_embedding:
                logger.warning("Failed to generate query embedding, falling back to empty results")
                return []

            # Call the Supabase RPC function for vector similarity search
            params = {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": limit
            }

            if competitor_id is not None:
                params["filter_competitor_id"] = competitor_id

            response = self.client.rpc("search_crawls", params).execute()

            results = response.data if response.data else []
            logger.info(f"Semantic search for '{query[:50]}...' returned {len(results)} results")
            return results

        except Exception as e:
            error_msg = str(e)
            if "function" in error_msg.lower() and "does not exist" in error_msg.lower():
                logger.error(
                    "RPC function 'search_crawls' not found. "
                    "Please create the function via database migration."
                )
            else:
                logger.error(f"Error in semantic search: {str(e)}")
            return []

    def store_change(
        self,
        crawl_id: int,
        change_type: str,
        summary: str,
        importance: str
    ) -> dict[str, Any]:
        """
        Store a detected change from comparing crawls.

        Args:
            crawl_id: ID of the crawl where the change was detected
            change_type: Category of change (e.g., 'pricing', 'feature', 'hiring', 'content')
            summary: Human-readable description of the change
            importance: Priority level ('high', 'medium', 'low')

        Returns:
            Created change record

        Raises:
            ValueError: If required fields are invalid
            Exception: On database errors
        """
        if not crawl_id:
            raise ValueError("crawl_id is required")

        valid_types = {"pricing", "feature", "hiring", "content", "product", "announcement", "other"}
        change_type = change_type.lower().strip() if change_type else "other"
        if change_type not in valid_types:
            logger.warning(f"Unknown change type '{change_type}', defaulting to 'other'")
            change_type = "other"

        valid_importance = {"high", "medium", "low"}
        importance = importance.lower().strip() if importance else "low"
        if importance not in valid_importance:
            logger.warning(f"Unknown importance '{importance}', defaulting to 'low'")
            importance = "low"

        if not summary or not summary.strip():
            raise ValueError("Change summary cannot be empty")

        try:
            data = {
                "crawl_id": crawl_id,
                "type": change_type,
                "summary": summary.strip(),
                "importance": importance,
                "detected_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("changes").insert(data).execute()

            if result.data:
                change = result.data[0]
                logger.info(
                    f"Stored {importance} {change_type} change (id={change.get('id')}): "
                    f"{summary[:100]}..."
                )
                return change

            logger.warning(f"No data returned when storing change for crawl {crawl_id}")
            return data

        except Exception as e:
            logger.error(f"Error storing change for crawl {crawl_id}: {str(e)}")
            raise

    def get_unalerted_changes(self) -> list[dict[str, Any]]:
        """
        Get all changes that haven't been sent as alerts yet.

        This performs a left join to find changes without corresponding alert records.

        Returns:
            List of unalerted change records with crawl and competitor info
        """
        try:
            # Query changes that don't have an alert record
            # Using a subquery approach since Supabase doesn't support LEFT JOIN WHERE NULL directly
            result = self.client.table("changes").select(
                "*, crawls(id, url, competitor_id, crawl_date, competitors(id, name, url))"
            ).is_("id", "not.in.(select change_id from alerts)").order(
                "detected_at", desc=False
            ).execute()

            # Fallback: if the above doesn't work, get all changes and filter
            if result.data is None:
                # Get all change IDs that have alerts
                alerts_result = self.client.table("alerts").select("change_id").execute()
                alerted_ids = {a["change_id"] for a in (alerts_result.data or [])}

                # Get all changes and filter out alerted ones
                changes_result = self.client.table("changes").select(
                    "*, crawls(id, url, competitor_id, crawl_date, competitors(id, name, url))"
                ).order("detected_at", desc=False).execute()

                unalerted = [
                    c for c in (changes_result.data or [])
                    if c["id"] not in alerted_ids
                ]
                logger.info(f"Found {len(unalerted)} unalerted changes (fallback method)")
                return unalerted

            unalerted = result.data if result.data else []
            logger.info(f"Found {len(unalerted)} unalerted changes")
            return unalerted

        except Exception as e:
            logger.error(f"Error getting unalerted changes: {str(e)}")
            # Attempt simpler fallback
            try:
                alerts_result = self.client.table("alerts").select("change_id").execute()
                alerted_ids = {a["change_id"] for a in (alerts_result.data or [])}

                changes_result = self.client.table("changes").select("*").order(
                    "detected_at", desc=False
                ).execute()

                unalerted = [
                    c for c in (changes_result.data or [])
                    if c["id"] not in alerted_ids
                ]
                logger.info(f"Found {len(unalerted)} unalerted changes (simple fallback)")
                return unalerted
            except Exception as fallback_error:
                logger.error(f"Fallback also failed: {str(fallback_error)}")
                return []

    def record_alert(self, change_id: int, slack_ts: str) -> dict[str, Any]:
        """
        Record that an alert was sent for a change.

        Args:
            change_id: ID of the change that was alerted
            slack_ts: Slack message timestamp (used for threading/updating)

        Returns:
            Created alert record

        Raises:
            ValueError: If required fields are missing
            Exception: On database errors
        """
        if not change_id:
            raise ValueError("change_id is required")
        if not slack_ts or not slack_ts.strip():
            raise ValueError("slack_ts is required")

        try:
            data = {
                "change_id": change_id,
                "slack_ts": slack_ts.strip(),
                "alerted_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("alerts").insert(data).execute()

            if result.data:
                alert = result.data[0]
                logger.info(f"Recorded alert for change {change_id} (slack_ts={slack_ts})")
                return alert

            logger.warning(f"No data returned when recording alert for change {change_id}")
            return data

        except Exception as e:
            error_msg = str(e)
            if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
                logger.warning(f"Alert already recorded for change {change_id}")
                raise ValueError(f"Alert already exists for change {change_id}")
            logger.error(f"Error recording alert for change {change_id}: {str(e)}")
            raise

    def get_change_with_context(self, change_id: int) -> Optional[dict[str, Any]]:
        """
        Get a change with full context (crawl and competitor info).

        Args:
            change_id: ID of the change

        Returns:
            Change record with nested crawl and competitor data, or None
        """
        try:
            result = self.client.table("changes").select(
                "*, crawls(id, url, competitor_id, crawl_date, markdown, competitors(id, name, url))"
            ).eq("id", change_id).execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error getting change {change_id} with context: {str(e)}")
            return None

    def get_recent_changes(
        self,
        limit: int = 20,
        importance: Optional[str] = None,
        change_type: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """
        Get recent changes with optional filtering.

        Args:
            limit: Maximum number of changes to return
            importance: Filter by importance level (optional)
            change_type: Filter by change type (optional)

        Returns:
            List of recent changes
        """
        try:
            query = self.client.table("changes").select(
                "*, crawls(url, competitors(name))"
            )

            if importance:
                query = query.eq("importance", importance.lower())
            if change_type:
                query = query.eq("type", change_type.lower())

            result = query.order("detected_at", desc=True).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting recent changes: {str(e)}")
            return []


# SQL for creating the search_crawls RPC function (for reference/migrations):
SEARCH_CRAWLS_SQL = """
-- Enable pgvector extension (run once)
CREATE EXTENSION IF NOT EXISTS vector WITH SCHEMA extensions;

-- Create the semantic search function
CREATE OR REPLACE FUNCTION search_crawls(
    query_embedding vector(768),
    match_threshold float DEFAULT 0.5,
    match_count int DEFAULT 5,
    filter_competitor_id int DEFAULT NULL
)
RETURNS TABLE (
    id bigint,
    competitor_id bigint,
    url text,
    markdown text,
    crawl_date timestamptz,
    similarity float
)
LANGUAGE plpgsql
AS $$
BEGIN
    RETURN QUERY
    SELECT
        c.id,
        c.competitor_id,
        c.url,
        c.markdown,
        c.crawl_date,
        1 - (c.embedding <=> query_embedding) AS similarity
    FROM crawls c
    WHERE
        c.embedding IS NOT NULL
        AND (filter_competitor_id IS NULL OR c.competitor_id = filter_competitor_id)
        AND 1 - (c.embedding <=> query_embedding) > match_threshold
    ORDER BY c.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;
"""


if __name__ == "__main__":
    # Test storage functionality
    import sys

    print("Testing Research Radar Storage Module")
    print("=" * 50)

    # Test Ollama embeddings
    print("\n1. Testing OllamaEmbeddings...")
    embeddings = OllamaEmbeddings()

    if embeddings.is_available():
        print(f"   Ollama is available with model: {embeddings.model}")
        test_embedding = embeddings.embed_text("Test embedding generation")
        if test_embedding:
            print(f"   Generated embedding with {len(test_embedding)} dimensions")
        else:
            print("   Warning: Failed to generate test embedding")
    else:
        print(f"   Warning: Ollama not available or model '{embeddings.model}' not loaded")
        print("   Run: ollama pull nomic-embed-text")

    # Test Supabase storage
    print("\n2. Testing SupabaseStorage...")
    try:
        storage = SupabaseStorage()
        print("   Supabase client initialized")

        if storage.ensure_schema():
            print("   Schema verification passed")
        else:
            print("   Warning: Schema verification failed - check migrations")

        # List existing competitors
        competitors = storage.list_competitors()
        print(f"   Found {len(competitors)} existing competitors")

    except ValueError as e:
        print(f"   Error: {str(e)}")
        print("   Set SUPABASE_URL and SUPABASE_KEY environment variables")
        sys.exit(1)
    except Exception as e:
        print(f"   Unexpected error: {str(e)}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Storage module tests completed")
