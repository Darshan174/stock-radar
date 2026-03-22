"""
Stock Radar - Supabase storage and vector embeddings.
Persists stocks, price data, analysis, signals, and alerts with semantic search.
"""

import hashlib
import os
import logging
import math
import threading
import time as _time
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional, Any
from decimal import Decimal

import requests
from supabase import create_client, Client

try:
    from config import settings as _cfg
except ImportError:
    _cfg = None

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


@dataclass
class EmbeddingResult:
    """Embedding vector plus metadata about how it was generated."""

    vector: list[float]
    provider: str
    model: str
    dimension: int


# ── In-process TTL cache for embedding vectors ─────────────────────────────
# Identical queries generate identical embeddings; cache to avoid redundant
# API calls (Cohere/Gemini).  Default TTL = cache_embedding_ttl_sec (86400s).

_embed_cache: dict[str, tuple[float, list[float]]] = {}
_embed_cache_lock = threading.Lock()

_EMBED_CACHE_TTL: int = (
    getattr(_cfg, "cache_embedding_ttl_sec", 86400) if _cfg else 86400
)


def _embed_cache_key(text: str, input_type: str) -> str:
    h = hashlib.sha256(f"{input_type}:{text}".encode()).hexdigest()[:24]
    return f"emb:{h}"


def _embed_cache_get(key: str) -> list[float] | None:
    with _embed_cache_lock:
        entry = _embed_cache.get(key)
        if entry and entry[0] > _time.monotonic():
            return entry[1]
    return None


def _embed_cache_set(key: str, vec: list[float]) -> None:
    with _embed_cache_lock:
        _embed_cache[key] = (_time.monotonic() + _EMBED_CACHE_TTL, vec)


class BaseEmbeddings:
    """Common interface for embedding providers."""

    provider_name = "unknown"

    def __init__(
        self,
        model: str,
        configured_dimension: Optional[int] = None,
        timeout: int = 30,
    ):
        self.model = model
        self.configured_dimension = configured_dimension
        self.timeout = timeout

    @staticmethod
    def _truncate_text(text: str, max_chars: int = 4000) -> str:
        if len(text) > max_chars:
            logger.debug(f"Truncating text from {len(text)} to {max_chars} chars")
            return text[:max_chars]
        return text

    @staticmethod
    def _normalize_vector(values: list[float]) -> list[float]:
        magnitude = math.sqrt(sum(v * v for v in values))
        if magnitude <= 0:
            return values
        return [v / magnitude for v in values]

    def embed(
        self,
        text: str,
        input_type: str = "search_document",
        title: Optional[str] = None,
    ) -> Optional[EmbeddingResult]:
        raise NotImplementedError

    def embed_text(
        self,
        text: str,
        input_type: str = "search_document",
        title: Optional[str] = None,
    ) -> list[float]:
        key = _embed_cache_key(text, input_type)
        cached = _embed_cache_get(key)
        if cached is not None:
            return cached
        result = self.embed(text=text, input_type=input_type, title=title)
        vec = result.vector if result else []
        if vec:
            _embed_cache_set(key, vec)
        return vec

    def is_available(self) -> bool:
        try:
            result = self.embed("test")
            return bool(result and result.vector)
        except Exception:
            return False


class NullEmbeddings(BaseEmbeddings):
    """Disabled embeddings provider used when configuration is incomplete."""

    provider_name = "disabled"

    def __init__(self):
        super().__init__(model="disabled", configured_dimension=None, timeout=0)

    def embed(
        self,
        text: str,
        input_type: str = "search_document",
        title: Optional[str] = None,
    ) -> Optional[EmbeddingResult]:
        _ = (text, input_type, title)
        return None


class CohereEmbeddings(BaseEmbeddings):
    """Generate embeddings using Cohere API for financial text."""

    provider_name = "cohere"
    DEFAULT_MODEL = "embed-english-v3.0"
    DEFAULT_DIMENSION = 1024

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        configured_dimension: Optional[int] = DEFAULT_DIMENSION,
        timeout: int = 30,
    ):
        """
        Initialize Cohere embeddings client.

        Args:
            api_key: Cohere API key (defaults to COHERE_API_KEY env var)
            model: Embedding model name
            timeout: Request timeout in seconds
        """
        super().__init__(
            model=model,
            configured_dimension=configured_dimension or self.DEFAULT_DIMENSION,
            timeout=timeout,
        )
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        self.api_url = "https://api.cohere.ai/v1/embed"

        if not self.api_key:
            logger.warning("Cohere API key not set - embeddings will be disabled")
        else:
            logger.info(f"Initialized CohereEmbeddings with model={model}")

    def embed(
        self,
        text: str,
        input_type: str = "search_document",
        title: Optional[str] = None,
    ) -> Optional[EmbeddingResult]:
        """
        Generate embedding vector for text using Cohere.

        Args:
            text: Text to embed
            input_type: 'search_document' for storage, 'search_query' for queries

        Returns:
            Embedding metadata, or None on failure
        """
        _ = title
        if not self.api_key:
            logger.warning("Cohere API key not set, skipping embedding")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        text = self._truncate_text(text)

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.model,
                    "texts": [text],
                    "input_type": input_type,
                    "truncate": "END"
                },
                timeout=self.timeout
            )
            response.raise_for_status()

            data = response.json()
            embeddings = data.get("embeddings", [])

            if embeddings and len(embeddings) > 0:
                embedding = embeddings[0]
                dimension = len(embedding)
                logger.debug(f"Generated embedding with {dimension} dimensions")
                if self.configured_dimension and dimension != self.configured_dimension:
                    logger.warning(
                        "Configured embedding_dim=%s but Cohere returned %s dimensions",
                        self.configured_dimension,
                        dimension,
                    )
                # Track Cohere usage
                from services.usage_tracker import get_tracker
                get_tracker().track("cohere", count=1)
                return EmbeddingResult(
                    vector=embedding,
                    provider=self.provider_name,
                    model=self.model,
                    dimension=dimension,
                )

            logger.warning("No embeddings returned from Cohere")
            return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout generating embedding (>{self.timeout}s)")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error from Cohere: {e.response.status_code} - {e.response.text}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error generating embedding: {type(e).__name__}: {str(e)}")
            return None


class GoogleEmbeddings(BaseEmbeddings):
    """Generate embeddings using the Gemini embeddings REST API."""

    provider_name = "google"
    DEFAULT_MODEL = "gemini-embedding-001"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        configured_dimension: Optional[int] = None,
        timeout: int = 30,
    ):
        super().__init__(
            model=model,
            configured_dimension=configured_dimension,
            timeout=timeout,
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.api_url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:embedContent"
        )

        if not self.api_key:
            logger.warning("Gemini API key not set - Google embeddings will be disabled")
        else:
            logger.info(
                "Initialized GoogleEmbeddings with model=%s output_dim=%s",
                self.model,
                self.configured_dimension or "default",
            )

    def embed(
        self,
        text: str,
        input_type: str = "search_document",
        title: Optional[str] = None,
    ) -> Optional[EmbeddingResult]:
        if not self.api_key:
            logger.warning("Gemini API key not set, skipping embedding")
            return None

        if not text or not text.strip():
            logger.warning("Empty text provided for embedding")
            return None

        text = self._truncate_text(text)
        task_type = (
            "RETRIEVAL_QUERY" if input_type == "search_query" else "RETRIEVAL_DOCUMENT"
        )
        payload: dict[str, Any] = {
            "model": f"models/{self.model}",
            "content": {
                "parts": [{"text": text}],
            },
            "taskType": task_type,
        }
        if self.configured_dimension:
            payload["outputDimensionality"] = self.configured_dimension
        if title and task_type == "RETRIEVAL_DOCUMENT":
            payload["title"] = title

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "x-goog-api-key": self.api_key,
                },
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()

            data = response.json()
            embedding = data.get("embedding", {}).get("values", [])
            if not embedding:
                logger.warning("No embeddings returned from Google")
                return None

            if self.configured_dimension and self.configured_dimension != 3072:
                embedding = self._normalize_vector(embedding)

            from services.usage_tracker import get_tracker

            get_tracker().track("gemini", count=1)
            return EmbeddingResult(
                vector=embedding,
                provider=self.provider_name,
                model=self.model,
                dimension=len(embedding),
            )

        except requests.exceptions.Timeout:
            logger.error(f"Timeout generating Google embedding (>{self.timeout}s)")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(
                f"HTTP error from Gemini embeddings: {e.response.status_code} - {e.response.text}"
            )
            return None
        except Exception as e:
            logger.error(
                f"Unexpected error generating Google embedding: {type(e).__name__}: {str(e)}"
            )
            return None


def create_embeddings_client(
    provider: str,
    model: Optional[str],
    dimension: Optional[int],
    cohere_key: Optional[str] = None,
    gemini_key: Optional[str] = None,
    timeout: int = 30,
) -> BaseEmbeddings:
    """Create the configured embeddings client."""

    normalized_provider = (provider or "cohere").strip().lower()

    if normalized_provider in ("disabled", "none", "off"):
        return NullEmbeddings()

    if normalized_provider == "cohere":
        resolved_model = model or CohereEmbeddings.DEFAULT_MODEL
        resolved_dimension = dimension or CohereEmbeddings.DEFAULT_DIMENSION
        return CohereEmbeddings(
            api_key=cohere_key,
            model=resolved_model,
            configured_dimension=resolved_dimension,
            timeout=timeout,
        )

    if normalized_provider in ("google", "gemini"):
        resolved_model = model
        if not resolved_model or resolved_model == CohereEmbeddings.DEFAULT_MODEL:
            resolved_model = GoogleEmbeddings.DEFAULT_MODEL
        return GoogleEmbeddings(
            api_key=gemini_key,
            model=resolved_model,
            configured_dimension=dimension,
            timeout=timeout,
        )

    logger.warning("Unknown embedding provider '%s'; embeddings disabled", provider)
    return NullEmbeddings()


class StockStorage:
    """
    Manages stock market data in Supabase with vector search.

    Tables:
        - users: User accounts with subscription plans
        - stocks: Master stock list
        - watchlist: User's tracked stocks
        - price_history: OHLCV data
        - technical_indicators: RSI, MACD, etc.
        - news: News articles with embeddings
        - analysis: AI analysis results
        - signals: Trading signals
        - alerts: Sent notifications
    """

    def __init__(
        self,
        url: Optional[str] = None,
        key: Optional[str] = None,
        cohere_key: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        gemini_key: Optional[str] = None,
    ):
        """
        Initialize Supabase client and embeddings.

        Args:
            url: Supabase project URL (defaults to SUPABASE_URL env var)
            key: Supabase API key (defaults to SUPABASE_KEY env var)
            cohere_key: Cohere API key for embeddings
            embedding_provider: Embedding backend ('cohere', 'google', 'disabled')
            embedding_model: Embedding model name
            embedding_dim: Expected / requested embedding dimension
            gemini_key: Gemini API key for Google embeddings

        Raises:
            ValueError: If Supabase credentials are missing
        """
        settings_url = getattr(_cfg, "supabase_url", None) if _cfg else None
        settings_key = getattr(_cfg, "supabase_key", None) if _cfg else None
        url = url or settings_url or os.getenv("SUPABASE_URL")
        key = key or settings_key or os.getenv("SUPABASE_KEY")

        if not url or not key:
            raise ValueError(
                "Supabase credentials required. Set SUPABASE_URL and SUPABASE_KEY "
                "environment variables or pass them as arguments."
            )

        self.client: Client = create_client(url, key)
        self.embedding_provider = (
            embedding_provider
            or (getattr(_cfg, "embedding_provider", None) if _cfg else None)
            or os.getenv("EMBEDDING_PROVIDER", "cohere")
        )
        self.embedding_model = (
            embedding_model
            or (getattr(_cfg, "embedding_model", None) if _cfg else None)
            or os.getenv("EMBEDDING_MODEL")
            or CohereEmbeddings.DEFAULT_MODEL
        )
        self.embedding_dim = (
            embedding_dim
            if embedding_dim is not None
            else (getattr(_cfg, "embedding_dim", None) if _cfg else None)
        )
        if self.embedding_dim is None:
            raw_dim = os.getenv("EMBEDDING_DIM")
            self.embedding_dim = int(raw_dim) if raw_dim else CohereEmbeddings.DEFAULT_DIMENSION

        cohere_key = (
            cohere_key
            or (getattr(_cfg, "cohere_api_key", None) if _cfg else None)
            or os.getenv("COHERE_API_KEY")
        )
        gemini_key = (
            gemini_key
            or (getattr(_cfg, "gemini_api_key", None) if _cfg else None)
            or os.getenv("GEMINI_API_KEY")
        )

        self.embeddings = create_embeddings_client(
            provider=self.embedding_provider,
            model=self.embedding_model,
            dimension=self.embedding_dim,
            cohere_key=cohere_key,
            gemini_key=gemini_key,
        )
        self.embedding_provider = self.embeddings.provider_name
        self.embedding_model = self.embeddings.model
        self.embedding_dim = self.embeddings.configured_dimension or self.embedding_dim

        self._embed_pool = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embed-bg")

        logger.info(
            "Initialized StockStorage connected to %s using embeddings provider=%s model=%s dim=%s",
            url,
            self.embedding_provider,
            self.embedding_model,
            self.embedding_dim,
        )

    def _backfill_embedding(
        self,
        table: str,
        row_id: Any,
        text: str,
        input_type: str = "search_document",
        title: Optional[str] = None,
        id_column: str = "id",
        prefix: str = "embedding",
    ) -> None:
        """Generate an embedding and UPDATE the row in the background."""
        try:
            result = self.embeddings.embed(text, input_type=input_type, title=title)
            if not result:
                return
            payload = self._embedding_metadata(result, prefix=prefix)
            self.client.table(table).update(payload).eq(id_column, row_id).execute()
        except Exception as exc:
            logger.warning("Background embedding backfill failed for %s[%s]: %s", table, row_id, exc)

    @staticmethod
    def _embedding_columns(prefix: str = "embedding") -> tuple[str, ...]:
        return (
            prefix,
            f"{prefix}_provider",
            f"{prefix}_model_name",
            f"{prefix}_dimension",
        )

    @staticmethod
    def _embedding_metadata(
        result: Optional[EmbeddingResult],
        prefix: str = "embedding",
    ) -> dict[str, Any]:
        if not result:
            return {}
        return {
            prefix: result.vector,
            f"{prefix}_provider": result.provider,
            f"{prefix}_model_name": result.model,
            f"{prefix}_dimension": result.dimension,
        }

    @staticmethod
    def _build_signal_embedding_text(
        signal_type: str,
        signal: str,
        reason: str,
        price_at_signal: float,
        importance: str,
    ) -> str:
        return (
            f"{signal_type} {signal} trading signal at {price_at_signal}. "
            f"Importance: {importance}. Reason: {reason}"
        )

    def _insert_with_legacy_retry(
        self,
        table_name: str,
        data: dict[str, Any],
        optional_fields: tuple[str, ...] = (),
    ):
        try:
            return self.client.table(table_name).insert(data).execute()
        except Exception as e:
            error_msg = str(e).lower()
            if optional_fields and "column" in error_msg:
                logger.warning(
                    "Insert into %s failed due to missing optional columns. Retrying without %s. "
                    "Run the latest migrations to enable embedding metadata.",
                    table_name,
                    ", ".join(optional_fields),
                )
                legacy_data = {
                    key: value for key, value in data.items() if key not in optional_fields
                }
                return self.client.table(table_name).insert(legacy_data).execute()
            raise

    # -------------------------------------------------------------------------
    # SCHEMA VERIFICATION
    # -------------------------------------------------------------------------

    def ensure_schema(self) -> bool:
        """
        Verify that required tables exist in the database.

        Probes all tables in parallel (one round-trip each but concurrent)
        instead of serial one-at-a-time checks.

        Returns:
            True if schema is valid, False otherwise
        """
        required_tables = [
            "users", "stocks", "watchlist", "price_history",
            "technical_indicators", "news", "analysis", "signals", "alerts",
            "chat_history", "knowledge_base", "analyze_jobs",
        ]

        def _probe(table: str) -> str | None:
            try:
                self.client.table(table).select("*").limit(0).execute()
                return None
            except Exception as e:
                error_msg = str(e).lower()
                if "does not exist" in error_msg or "relation" in error_msg:
                    return table
                return None

        try:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=len(required_tables)) as pool:
                results = pool.map(_probe, required_tables)
            missing_tables = [t for t in results if t is not None]
        except Exception as e:
            logger.error(f"Error verifying schema: {type(e).__name__}: {str(e)}")
            return False

        if missing_tables:
            logger.error(
                f"Missing required tables: {missing_tables}. "
                "Please run database migrations."
            )
            return False

        logger.info("Schema verification completed successfully")
        return True

    # -------------------------------------------------------------------------
    # USERS
    # -------------------------------------------------------------------------

    def get_or_create_user(self, email: str, name: Optional[str] = None) -> dict[str, Any]:
        """
        Get existing user or create a new one.

        Args:
            email: User's email address
            name: User's name (optional)

        Returns:
            User record
        """
        try:
            # Try to get existing user
            result = self.client.table("users").select("*").eq("email", email).execute()

            if result.data:
                logger.debug(f"Found existing user: {email}")
                return result.data[0]

            # Create new user
            data = {
                "email": email,
                "name": name,
                "plan": "free",
                "stocks_limit": 3,
                "trading_mode": "intraday",
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("users").insert(data).execute()

            if result.data:
                logger.info(f"Created new user: {email}")
                return result.data[0]

            return data

        except Exception as e:
            logger.error(f"Error getting/creating user {email}: {str(e)}")
            raise

    def update_user_plan(self, user_id: str, plan: str, stocks_limit: int) -> dict[str, Any]:
        """Update user's subscription plan."""
        try:
            result = self.client.table("users").update({
                "plan": plan,
                "stocks_limit": stocks_limit,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", user_id).execute()

            if result.data:
                logger.info(f"Updated user {user_id} to plan: {plan}")
                return result.data[0]

            raise ValueError(f"User {user_id} not found")

        except Exception as e:
            logger.error(f"Error updating user plan: {str(e)}")
            raise

    def update_user_trading_mode(self, user_id: str, mode: str) -> dict[str, Any]:
        """Update user's default trading mode."""
        if mode not in ("intraday", "longterm"):
            raise ValueError(f"Invalid trading mode: {mode}")

        try:
            result = self.client.table("users").update({
                "trading_mode": mode,
                "updated_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", user_id).execute()

            if result.data:
                logger.info(f"Updated user {user_id} trading mode to: {mode}")
                return result.data[0]

            raise ValueError(f"User {user_id} not found")

        except Exception as e:
            logger.error(f"Error updating trading mode: {str(e)}")
            raise

    # -------------------------------------------------------------------------
    # STOCKS
    # -------------------------------------------------------------------------

    def get_or_create_stock(
        self,
        symbol: str,
        name: str,
        exchange: str,
        sector: Optional[str] = None,
        currency: str = "INR"
    ) -> dict[str, Any]:
        """
        Get existing stock or create a new one.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS')
            name: Company name
            exchange: Exchange (e.g., 'NSE', 'NASDAQ')
            sector: Industry sector
            currency: Trading currency

        Returns:
            Stock record
        """
        try:
            # Try to get existing stock
            result = self.client.table("stocks").select("*").eq(
                "symbol", symbol
            ).eq("exchange", exchange).execute()

            if result.data:
                logger.debug(f"Found existing stock: {symbol}")
                return result.data[0]

            # Create new stock
            data = {
                "symbol": symbol,
                "name": name,
                "exchange": exchange,
                "sector": sector,
                "currency": currency,
                "is_active": True,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("stocks").insert(data).execute()

            if result.data:
                logger.info(f"Created new stock: {symbol} on {exchange}")
                return result.data[0]

            return data

        except Exception as e:
            logger.error(f"Error getting/creating stock {symbol}: {str(e)}")
            raise

    def get_stock_by_symbol(self, symbol: str) -> Optional[dict[str, Any]]:
        """Get stock by symbol."""
        try:
            result = self.client.table("stocks").select("*").eq("symbol", symbol).execute()
            return result.data[0] if result.data else None
        except Exception as e:
            logger.error(f"Error getting stock {symbol}: {str(e)}")
            return None

    def list_stocks(self, exchange: Optional[str] = None) -> list[dict[str, Any]]:
        """List all active stocks, optionally filtered by exchange."""
        try:
            query = self.client.table("stocks").select("*").eq("is_active", True)

            if exchange:
                query = query.eq("exchange", exchange)

            result = query.order("symbol").execute()
            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error listing stocks: {str(e)}")
            return []

    # -------------------------------------------------------------------------
    # WATCHLIST
    # -------------------------------------------------------------------------

    def add_to_watchlist(
        self,
        user_id: str,
        stock_id: int,
        mode: str = "intraday",
        support: Optional[float] = None,
        resistance: Optional[float] = None,
        target: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> dict[str, Any]:
        """
        Add a stock to user's watchlist.

        Args:
            user_id: User's ID
            stock_id: Stock's ID
            mode: Trading mode ('intraday' or 'longterm')
            support: User-defined support level
            resistance: User-defined resistance level
            target: Target price
            stop_loss: Stop loss price

        Returns:
            Watchlist entry
        """
        try:
            data = {
                "user_id": user_id,
                "stock_id": stock_id,
                "mode": mode,
                "alerts_enabled": True,
                "support_level": support,
                "resistance_level": resistance,
                "target_price": target,
                "stop_loss": stop_loss,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("watchlist").insert(data).execute()

            if result.data:
                logger.info(f"Added stock {stock_id} to user {user_id} watchlist")
                return result.data[0]

            return data

        except Exception as e:
            error_msg = str(e)
            if "duplicate" in error_msg.lower() or "unique" in error_msg.lower():
                logger.warning(f"Stock already in watchlist")
                raise ValueError("Stock already in watchlist")
            logger.error(f"Error adding to watchlist: {str(e)}")
            raise

    def get_user_watchlist(self, user_id: str) -> list[dict[str, Any]]:
        """Get user's watchlist with stock details."""
        try:
            result = self.client.table("watchlist").select(
                "*, stocks(id, symbol, name, exchange, sector, currency)"
            ).eq("user_id", user_id).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting watchlist for user {user_id}: {str(e)}")
            return []

    def remove_from_watchlist(self, user_id: str, stock_id: int) -> bool:
        """Remove a stock from user's watchlist."""
        try:
            self.client.table("watchlist").delete().eq(
                "user_id", user_id
            ).eq("stock_id", stock_id).execute()

            logger.info(f"Removed stock {stock_id} from user {user_id} watchlist")
            return True

        except Exception as e:
            logger.error(f"Error removing from watchlist: {str(e)}")
            return False

    def update_watchlist_levels(
        self,
        user_id: str,
        stock_id: int,
        support: Optional[float] = None,
        resistance: Optional[float] = None,
        target: Optional[float] = None,
        stop_loss: Optional[float] = None
    ) -> dict[str, Any]:
        """Update price levels for a watchlist entry."""
        try:
            updates = {}
            if support is not None:
                updates["support_level"] = support
            if resistance is not None:
                updates["resistance_level"] = resistance
            if target is not None:
                updates["target_price"] = target
            if stop_loss is not None:
                updates["stop_loss"] = stop_loss

            result = self.client.table("watchlist").update(updates).eq(
                "user_id", user_id
            ).eq("stock_id", stock_id).execute()

            if result.data:
                return result.data[0]

            raise ValueError("Watchlist entry not found")

        except Exception as e:
            logger.error(f"Error updating watchlist levels: {str(e)}")
            raise

    # -------------------------------------------------------------------------
    # PRICE HISTORY
    # -------------------------------------------------------------------------

    def store_price_data(
        self,
        stock_id: int,
        prices: list[dict[str, Any]],
        timeframe: str = "1d"
    ) -> int:
        """
        Store OHLCV price data for a stock.

        Args:
            stock_id: Stock's ID
            prices: List of price dictionaries with timestamp, open, high, low, close, volume
            timeframe: Data timeframe ('1m', '5m', '15m', '1h', '1d', '1w')

        Returns:
            Number of records stored
        """
        if not prices:
            return 0

        try:
            records = []
            for p in prices:
                records.append({
                    "stock_id": stock_id,
                    "timestamp": p["timestamp"].isoformat() if hasattr(p["timestamp"], "isoformat") else p["timestamp"],
                    "timeframe": timeframe,
                    "open": float(p["open"]),
                    "high": float(p["high"]),
                    "low": float(p["low"]),
                    "close": float(p["close"]),
                    "volume": int(p.get("volume", 0)) if p.get("volume") else None,
                    "created_at": datetime.now(timezone.utc).isoformat()
                })

            # Upsert to handle duplicates
            result = self.client.table("price_history").upsert(
                records,
                on_conflict="stock_id,timestamp,timeframe"
            ).execute()

            count = len(result.data) if result.data else 0
            logger.info(f"Stored {count} price records for stock {stock_id}")
            return count

        except Exception as e:
            logger.error(f"Error storing price data: {str(e)}")
            raise

    def get_price_history(
        self,
        stock_id: int,
        timeframe: str = "1d",
        limit: int = 100
    ) -> list[dict[str, Any]]:
        """Get price history for a stock."""
        try:
            result = self.client.table("price_history").select("*").eq(
                "stock_id", stock_id
            ).eq("timeframe", timeframe).order(
                "timestamp", desc=True
            ).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting price history: {str(e)}")
            return []

    # -------------------------------------------------------------------------
    # TECHNICAL INDICATORS
    # -------------------------------------------------------------------------

    def store_indicators(
        self,
        stock_id: int,
        timestamp: datetime,
        indicators: dict[str, Any],
        timeframe: str = "1d"
    ) -> dict[str, Any]:
        """
        Store technical indicators for a stock.

        Args:
            stock_id: Stock's ID
            timestamp: Data timestamp
            indicators: Dictionary of indicator values
            timeframe: Data timeframe

        Returns:
            Stored record
        """
        try:
            data = {
                "stock_id": stock_id,
                "timestamp": timestamp.isoformat() if hasattr(timestamp, "isoformat") else timestamp,
                "timeframe": timeframe,
                "sma_20": indicators.get("sma_20"),
                "sma_50": indicators.get("sma_50"),
                "sma_200": indicators.get("sma_200"),
                "ema_12": indicators.get("ema_12"),
                "ema_26": indicators.get("ema_26"),
                "rsi_14": indicators.get("rsi_14"),
                "macd": indicators.get("macd"),
                "macd_signal": indicators.get("macd_signal"),
                "macd_histogram": indicators.get("macd_histogram"),
                "bollinger_upper": indicators.get("bollinger_upper"),
                "bollinger_middle": indicators.get("bollinger_middle"),
                "bollinger_lower": indicators.get("bollinger_lower"),
                "atr_14": indicators.get("atr_14"),
                "volume_sma_20": indicators.get("volume_sma_20"),
                "obv": indicators.get("obv"),
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("technical_indicators").upsert(
                data,
                on_conflict="stock_id,timestamp,timeframe"
            ).execute()

            if result.data:
                logger.debug(f"Stored indicators for stock {stock_id}")
                return result.data[0]

            return data

        except Exception as e:
            logger.error(f"Error storing indicators: {str(e)}")
            raise

    def get_latest_indicators(
        self,
        stock_id: int,
        timeframe: str = "1d"
    ) -> Optional[dict[str, Any]]:
        """Get latest technical indicators for a stock."""
        try:
            result = self.client.table("technical_indicators").select("*").eq(
                "stock_id", stock_id
            ).eq("timeframe", timeframe).order(
                "timestamp", desc=True
            ).limit(1).execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error getting indicators: {str(e)}")
            return None

    # -------------------------------------------------------------------------
    # NEWS
    # -------------------------------------------------------------------------

    def store_news(
        self,
        headline: str,
        summary: Optional[str] = None,
        source: Optional[str] = None,
        url: Optional[str] = None,
        published_at: Optional[datetime] = None,
        stock_id: Optional[int] = None,
        sentiment_score: Optional[float] = None,
        sentiment_label: Optional[str] = None
    ) -> dict[str, Any]:
        """
        Store a news article with embedding.

        Args:
            headline: News headline
            summary: Article summary
            source: News source
            url: Article URL
            published_at: Publication timestamp
            stock_id: Related stock ID (optional)
            sentiment_score: Sentiment score (-1.0 to 1.0)
            sentiment_label: 'positive', 'negative', or 'neutral'

        Returns:
            Stored news record
        """
        try:
            embed_text = f"{headline}. {summary}" if summary else headline

            data = {
                "stock_id": stock_id,
                "headline": headline,
                "summary": summary,
                "source": source,
                "url": url,
                "published_at": published_at.isoformat() if published_at else None,
                "sentiment_score": sentiment_score,
                "sentiment_label": sentiment_label,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self._insert_with_legacy_retry(
                "news",
                data,
                optional_fields=self._embedding_columns(),
            )

            row = result.data[0] if result.data else data
            logger.info(f"Stored news: {headline[:50]}...")

            if result.data:
                self._embed_pool.submit(
                    self._backfill_embedding,
                    table="news",
                    row_id=row["id"],
                    text=embed_text,
                    title=headline,
                )

            return row

        except Exception as e:
            logger.error(f"Error storing news: {str(e)}")
            raise

    def search_news(
        self,
        query: str,
        stock_id: Optional[int] = None,
        limit: int = 10,
        match_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Search news using semantic similarity.

        Args:
            query: Search query
            stock_id: Filter by stock (optional)
            limit: Maximum results
            match_threshold: Minimum similarity score

        Returns:
            List of matching news articles with similarity scores
        """
        if not query or not query.strip():
            return []

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_text(query.strip(), input_type="search_query")

            if not query_embedding:
                logger.warning("Failed to generate query embedding")
                return []

            # Call RPC function for vector search
            params = {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": limit
            }

            if stock_id is not None:
                params["filter_stock_id"] = stock_id

            response = self.client.rpc("search_news", params).execute()

            results = response.data if response.data else []
            logger.info(f"News search for '{query[:30]}...' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in news search: {str(e)}")
            return []

    def get_recent_news(
        self,
        stock_id: Optional[int] = None,
        limit: int = 20,
        sentiment: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get recent news articles."""
        try:
            query = self.client.table("news").select(
                "id, stock_id, headline, summary, source, url, published_at, "
                "sentiment_score, sentiment_label"
            )

            if stock_id:
                query = query.eq("stock_id", stock_id)
            if sentiment:
                query = query.eq("sentiment_label", sentiment)

            result = query.order("published_at", desc=True).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting recent news: {str(e)}")
            return []

    # -------------------------------------------------------------------------
    # ANALYSIS
    # -------------------------------------------------------------------------

    def store_analysis(
        self,
        stock_id: int,
        mode: str,
        signal: str,
        confidence: float,
        reasoning: str,
        technical_score: Optional[float] = None,
        technical_summary: Optional[str] = None,
        fundamental_score: Optional[float] = None,
        fundamental_summary: Optional[str] = None,
        sentiment_score: Optional[float] = None,
        sentiment_summary: Optional[str] = None,
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        llm_model: Optional[str] = None,
        llm_tokens_used: Optional[int] = None,
        analysis_duration_ms: Optional[int] = None,
        algo_prediction: Optional[dict] = None
    ) -> dict[str, Any]:
        """
        Store AI analysis results.

        Args:
            stock_id: Stock's ID
            mode: 'intraday' or 'longterm'
            signal: 'strong_buy', 'buy', 'hold', 'sell', 'strong_sell'
            confidence: Confidence score (0-1)
            reasoning: AI explanation
            algo_prediction: AI algo trading prediction (optional)
            ... other analysis metrics

        Returns:
            Stored analysis record
        """
        try:
            data = {
                "stock_id": stock_id,
                "mode": mode,
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "technical_score": technical_score,
                "technical_summary": technical_summary,
                "fundamental_score": fundamental_score,
                "fundamental_summary": fundamental_summary,
                "sentiment_score": sentiment_score,
                "sentiment_summary": sentiment_summary,
                "support_level": support_level,
                "resistance_level": resistance_level,
                "target_price": target_price,
                "stop_loss": stop_loss,
                "llm_model": llm_model,
                "llm_tokens_used": llm_tokens_used,
                "analysis_duration_ms": analysis_duration_ms,
                "algo_prediction": algo_prediction,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("analysis").insert(data).execute()

            if result.data:
                logger.info(f"Stored {mode} analysis for stock {stock_id}: {signal}")
                return result.data[0]

            return data

        except Exception as e:
            logger.error(f"Error storing analysis: {str(e)}")
            raise

    def get_latest_analysis(
        self,
        stock_id: int,
        mode: str = "intraday"
    ) -> Optional[dict[str, Any]]:
        """Get the most recent analysis for a stock."""
        try:
            result = self.client.table("analysis").select("*").eq(
                "stock_id", stock_id
            ).eq("mode", mode).order(
                "created_at", desc=True
            ).limit(1).execute()

            return result.data[0] if result.data else None

        except Exception as e:
            logger.error(f"Error getting latest analysis: {str(e)}")
            return None

    def get_analyses_for_watchlist(
        self,
        stock_ids: list[int],
        mode: str = "intraday"
    ) -> list[dict[str, Any]]:
        """Get latest analysis for multiple stocks using RPC function."""
        if not stock_ids:
            return []

        try:
            response = self.client.rpc("get_latest_analysis", {
                "p_stock_ids": stock_ids,
                "p_mode": mode
            }).execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Error getting analyses: {str(e)}")
            return []

    # -------------------------------------------------------------------------
    # SIGNALS
    # -------------------------------------------------------------------------

    def store_signal(
        self,
        stock_id: int,
        signal_type: str,
        signal: str,
        price_at_signal: float,
        reason: str,
        analysis_id: Optional[int] = None,
        importance: str = "medium"
    ) -> dict[str, Any]:
        """
        Store a trading signal.

        Args:
            stock_id: Stock's ID
            signal_type: 'entry', 'exit', 'stop_loss', 'target_hit'
            signal: 'buy', 'sell', 'hold'
            price_at_signal: Current price
            reason: Signal reason
            analysis_id: Related analysis ID
            importance: 'high', 'medium', 'low'

        Returns:
            Stored signal record
        """
        try:
            embed_text = self._build_signal_embedding_text(
                signal_type=signal_type,
                signal=signal,
                reason=reason,
                price_at_signal=price_at_signal,
                importance=importance,
            )
            data = {
                "stock_id": stock_id,
                "analysis_id": analysis_id,
                "signal_type": signal_type,
                "signal": signal,
                "price_at_signal": price_at_signal,
                "reason": reason,
                "importance": importance,
                "is_triggered": False,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self._insert_with_legacy_retry(
                "signals",
                data,
                optional_fields=self._embedding_columns(prefix="context_embedding"),
            )

            row = result.data[0] if result.data else data
            logger.info(f"Stored {importance} {signal} signal for stock {stock_id}")

            if result.data:
                self._embed_pool.submit(
                    self._backfill_embedding,
                    table="signals",
                    row_id=row["id"],
                    text=embed_text,
                    title=f"{signal_type} {signal}",
                    prefix="context_embedding",
                )

            return row

        except Exception as e:
            logger.error(f"Error storing signal: {str(e)}")
            raise

    def get_active_signals(
        self,
        stock_id: Optional[int] = None,
        importance: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get active (non-triggered) signals."""
        try:
            query = self.client.table("signals").select(
                "*, stocks(symbol, name)"
            ).eq("is_triggered", False)

            if stock_id:
                query = query.eq("stock_id", stock_id)
            if importance:
                query = query.eq("importance", importance)

            result = query.order("created_at", desc=True).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting active signals: {str(e)}")
            return []

    def mark_signal_triggered(self, signal_id: int) -> dict[str, Any]:
        """Mark a signal as triggered."""
        try:
            result = self.client.table("signals").update({
                "is_triggered": True,
                "triggered_at": datetime.now(timezone.utc).isoformat()
            }).eq("id", signal_id).execute()

            if result.data:
                return result.data[0]

            raise ValueError(f"Signal {signal_id} not found")

        except Exception as e:
            logger.error(f"Error marking signal triggered: {str(e)}")
            raise

    # -------------------------------------------------------------------------
    # ALERTS
    # -------------------------------------------------------------------------

    def store_alert(
        self,
        user_id: str,
        stock_id: int,
        channel: str,
        message: str,
        signal_id: Optional[int] = None,
        external_id: Optional[str] = None,
        status: str = "sent"
    ) -> dict[str, Any]:
        """
        Store a sent alert/notification.

        Args:
            user_id: User's ID
            stock_id: Stock's ID
            channel: 'slack', 'telegram', 'email'
            message: Alert message content
            signal_id: Related signal ID
            external_id: External message ID (Slack ts, Telegram ID)
            status: 'pending', 'sent', 'failed'

        Returns:
            Stored alert record
        """
        try:
            data = {
                "user_id": user_id,
                "stock_id": stock_id,
                "signal_id": signal_id,
                "channel": channel,
                "message": message,
                "status": status,
                "external_id": external_id,
                "sent_at": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self.client.table("alerts").insert(data).execute()

            if result.data:
                logger.info(f"Stored alert for user {user_id} via {channel}")
                return result.data[0]

            return data

        except Exception as e:
            logger.error(f"Error storing alert: {str(e)}")
            raise

    def get_user_alerts(
        self,
        user_id: str,
        limit: int = 20,
        channel: Optional[str] = None
    ) -> list[dict[str, Any]]:
        """Get alerts for a user."""
        try:
            query = self.client.table("alerts").select(
                "*, stocks(symbol, name)"
            ).eq("user_id", user_id)

            if channel:
                query = query.eq("channel", channel)

            result = query.order("sent_at", desc=True).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting user alerts: {str(e)}")
            return []

    def get_unsent_signals(self, user_id: str) -> list[dict[str, Any]]:
        """Get signals that haven't been alerted to the user yet."""
        try:
            # Get user's watchlist stock IDs
            watchlist = self.get_user_watchlist(user_id)
            stock_ids = [w["stock_id"] for w in watchlist if w.get("alerts_enabled", True)]

            if not stock_ids:
                return []

            # Get signals for those stocks
            signals_result = self.client.table("signals").select(
                "*, stocks(symbol, name)"
            ).in_("stock_id", stock_ids).eq("is_triggered", False).execute()

            signals = signals_result.data if signals_result.data else []

            # Get already alerted signal IDs
            alerts_result = self.client.table("alerts").select("signal_id").eq(
                "user_id", user_id
            ).execute()
            alerted_ids = {a["signal_id"] for a in (alerts_result.data or []) if a["signal_id"]}

            # Filter out already alerted
            unsent = [s for s in signals if s["id"] not in alerted_ids]

            logger.info(f"Found {len(unsent)} unsent signals for user {user_id}")
            return unsent

        except Exception as e:
            logger.error(f"Error getting unsent signals: {str(e)}")
            return []


    # -------------------------------------------------------------------------
    # RAG (Retrieval-Augmented Generation) Methods
    # -------------------------------------------------------------------------

    def store_analysis_with_embedding(
        self,
        stock_id: int,
        mode: str,
        signal: str,
        confidence: float,
        reasoning: str,
        technical_summary: Optional[str] = None,
        sentiment_summary: Optional[str] = None,
        support_level: Optional[float] = None,
        resistance_level: Optional[float] = None,
        target_price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        llm_model: Optional[str] = None,
        llm_tokens_used: Optional[int] = None,
        analysis_duration_ms: Optional[int] = None,
        algo_prediction: Optional[dict] = None,
        generate_embedding: bool = True
    ) -> dict[str, Any]:
        """
        Store AI analysis results with embedding for RAG retrieval.

        Args:
            ... same as store_analysis ...
            generate_embedding: Whether to generate embedding for semantic search

        Returns:
            Stored analysis record with embedding
        """
        try:
            # Build embedding text from analysis content
            embedding_text = f"{signal} signal for stock analysis. {reasoning}"
            if technical_summary:
                embedding_text += f" Technical: {technical_summary}"
            if sentiment_summary:
                embedding_text += f" Sentiment: {sentiment_summary}"

            data = {
                "stock_id": stock_id,
                "mode": mode,
                "signal": signal,
                "confidence": confidence,
                "reasoning": reasoning,
                "technical_summary": technical_summary,
                "sentiment_summary": sentiment_summary,
                "support_level": support_level,
                "resistance_level": resistance_level,
                "target_price": target_price,
                "stop_loss": stop_loss,
                "llm_model": llm_model,
                "llm_tokens_used": llm_tokens_used,
                "analysis_duration_ms": analysis_duration_ms,
                "algo_prediction": algo_prediction,
                "embedding_text": embedding_text,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self._insert_with_legacy_retry(
                "analysis",
                data,
                optional_fields=self._embedding_columns() + ("embedding_text",),
            )

            row = result.data[0] if result.data else data
            logger.info(f"Stored {mode} analysis for stock {stock_id}: {signal}")

            # Backfill embedding in background so the write path isn't blocked
            if generate_embedding and result.data:
                self._embed_pool.submit(
                    self._backfill_embedding,
                    table="analysis",
                    row_id=row["id"],
                    text=embedding_text,
                    title=f"{mode} {signal} analysis",
                )

            return row

        except Exception as e:
            logger.error(f"Error storing analysis with embedding: {str(e)}")
            raise

    def search_similar_analyses(
        self,
        query: str,
        stock_symbol: Optional[str] = None,
        mode: Optional[str] = None,
        limit: int = 5,
        match_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Search for semantically similar past analyses.

        Args:
            query: Search query text
            stock_symbol: Filter by stock symbol (optional)
            mode: Filter by trading mode (optional)
            limit: Maximum results
            match_threshold: Minimum similarity score

        Returns:
            List of similar analyses with similarity scores
        """
        if not query or not query.strip():
            return []

        try:
            # Generate query embedding
            query_embedding = self.embeddings.embed_text(query.strip(), input_type="search_query")

            if not query_embedding:
                logger.warning("Failed to generate query embedding for analysis search")
                return []

            # Call RPC function for vector search
            params = {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": limit
            }

            if stock_symbol:
                params["filter_stock_symbol"] = stock_symbol
            if mode:
                params["filter_mode"] = mode

            response = self.client.rpc("search_similar_analyses", params).execute()

            results = response.data if response.data else []
            logger.info(f"Analysis search for '{query[:30]}...' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in analysis search: {str(e)}")
            return []

    def search_similar_signals(
        self,
        query: str,
        stock_symbol: Optional[str] = None,
        signal_type: Optional[str] = None,
        limit: int = 5,
        match_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Search for signals with similar context.

        Args:
            query: Search query text
            stock_symbol: Filter by stock symbol (optional)
            signal_type: Filter by signal type (optional)
            limit: Maximum results
            match_threshold: Minimum similarity score

        Returns:
            List of similar signals with similarity scores
        """
        if not query or not query.strip():
            return []

        try:
            query_embedding = self.embeddings.embed_text(query.strip(), input_type="search_query")

            if not query_embedding:
                logger.warning("Failed to generate query embedding for signal search")
                return []

            params = {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": limit
            }

            if stock_symbol:
                params["filter_stock_symbol"] = stock_symbol
            if signal_type:
                params["filter_signal_type"] = signal_type

            response = self.client.rpc("search_similar_signals", params).execute()

            results = response.data if response.data else []
            logger.info(f"Signal search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in signal search: {str(e)}")
            return []

    def rag_search(
        self,
        query: str,
        stock_symbol: Optional[str] = None,
        user_id: Optional[str] = None,
        match_threshold: float = 0.4,
        max_per_source: int = 3
    ) -> list[dict[str, Any]]:
        """
        Comprehensive RAG search across all data sources.

        Args:
            query: Search query text
            stock_symbol: Filter by stock symbol (optional)
            user_id: Filter by user ID for knowledge base (optional)
            match_threshold: Minimum similarity score
            max_per_source: Maximum results per source type

        Returns:
            List of results from news, analyses, and knowledge base
        """
        if not query or not query.strip():
            return []

        try:
            query_embedding = self.embeddings.embed_text(query.strip(), input_type="search_query")

            if not query_embedding:
                logger.warning("Failed to generate query embedding for RAG search")
                return []

            params = {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "max_results_per_source": max_per_source
            }

            if stock_symbol:
                params["filter_stock_symbol"] = stock_symbol
            if user_id:
                params["filter_user_id"] = user_id

            response = self.client.rpc("rag_search", params).execute()

            results = response.data if response.data else []
            logger.info(f"RAG search for '{query[:30]}...' returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error in RAG search: {str(e)}")
            return []

    # -------------------------------------------------------------------------
    # Chat History for RAG
    # -------------------------------------------------------------------------

    def store_chat_message(
        self,
        session_id: str,
        role: str,
        content: str,
        user_id: Optional[str] = None,
        stock_symbols: Optional[list[str]] = None,
        context_used: Optional[dict] = None,
        tokens_used: Optional[int] = None,
        model_used: Optional[str] = None,
        generate_embedding: bool = True
    ) -> dict[str, Any]:
        """
        Store a chat message with embedding for conversation retrieval.

        Args:
            session_id: Chat session ID
            role: 'user', 'assistant', or 'system'
            content: Message content
            user_id: User's ID (optional)
            stock_symbols: Stocks mentioned in the message
            context_used: RAG context that was retrieved
            tokens_used: Tokens used for this message
            model_used: LLM model used
            generate_embedding: Whether to generate embedding

        Returns:
            Stored chat message record
        """
        try:
            data = {
                "session_id": session_id,
                "user_id": user_id,
                "role": role,
                "content": content,
                "stock_symbols": stock_symbols,
                "context_used": context_used,
                "tokens_used": tokens_used,
                "model_used": model_used,
                "created_at": datetime.now(timezone.utc).isoformat()
            }

            result = self._insert_with_legacy_retry(
                "chat_history",
                data,
                optional_fields=self._embedding_columns(),
            )

            row = result.data[0] if result.data else data
            logger.debug(f"Stored chat message for session {session_id}")

            if generate_embedding and content and result.data:
                self._embed_pool.submit(
                    self._backfill_embedding,
                    table="chat_history",
                    row_id=row["id"],
                    text=content,
                    title=role,
                )

            return row

        except Exception as e:
            logger.error(f"Error storing chat message: {str(e)}")
            raise

    def get_chat_history(
        self,
        session_id: str,
        limit: int = 50
    ) -> list[dict[str, Any]]:
        """Get chat history for a session."""
        try:
            result = self.client.table("chat_history").select(
                "id, role, content, stock_symbols, context_used, created_at"
            ).eq("session_id", session_id).order(
                "created_at", desc=False
            ).limit(limit).execute()

            return result.data if result.data else []

        except Exception as e:
            logger.error(f"Error getting chat history: {str(e)}")
            return []

    def search_chat_history(
        self,
        query: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        limit: int = 10,
        match_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Search past chat conversations semantically.

        Args:
            query: Search query
            user_id: Filter by user
            session_id: Filter by session
            limit: Maximum results
            match_threshold: Minimum similarity

        Returns:
            List of similar chat messages
        """
        if not query or not query.strip():
            return []

        try:
            query_embedding = self.embeddings.embed_text(query.strip(), input_type="search_query")

            if not query_embedding:
                return []

            params = {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": limit
            }

            if user_id:
                params["filter_user_id"] = user_id
            if session_id:
                params["filter_session_id"] = session_id

            response = self.client.rpc("search_chat_history", params).execute()

            return response.data if response.data else []

        except Exception as e:
            logger.error(f"Error searching chat history: {str(e)}")
            return []

    # -------------------------------------------------------------------------
    # Knowledge Base for RAG
    # -------------------------------------------------------------------------

    def store_knowledge(
        self,
        title: str,
        content: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        stock_symbols: Optional[list[str]] = None,
        tags: Optional[list[str]] = None,
        source_url: Optional[str] = None,
        is_public: bool = False
    ) -> dict[str, Any]:
        """
        Store a knowledge base entry with embedding.

        Args:
            title: Entry title
            content: Entry content
            user_id: Owner user ID
            category: Category (strategy, research, notes, etc.)
            stock_symbols: Related stock symbols
            tags: Tags for categorization
            source_url: Source URL if applicable
            is_public: Whether publicly accessible

        Returns:
            Stored knowledge entry
        """
        try:
            embed_text = f"{title}. {content}"

            data = {
                "user_id": user_id,
                "title": title,
                "content": content,
                "category": category,
                "stock_symbols": stock_symbols,
                "tags": tags,
                "source_url": source_url,
                "is_public": is_public,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat()
            }

            result = self._insert_with_legacy_retry(
                "knowledge_base",
                data,
                optional_fields=self._embedding_columns(),
            )

            row = result.data[0] if result.data else data
            logger.info(f"Stored knowledge entry: {title[:50]}...")

            if result.data:
                self._embed_pool.submit(
                    self._backfill_embedding,
                    table="knowledge_base",
                    row_id=row["id"],
                    text=embed_text,
                    title=title,
                )

            return row

        except Exception as e:
            logger.error(f"Error storing knowledge: {str(e)}")
            raise

    def search_knowledge_base(
        self,
        query: str,
        user_id: Optional[str] = None,
        category: Optional[str] = None,
        stock_symbols: Optional[list[str]] = None,
        include_public: bool = True,
        limit: int = 10,
        match_threshold: float = 0.5
    ) -> list[dict[str, Any]]:
        """
        Search knowledge base semantically.

        Args:
            query: Search query
            user_id: Filter by user
            category: Filter by category
            stock_symbols: Filter by related stocks
            include_public: Include public entries
            limit: Maximum results
            match_threshold: Minimum similarity

        Returns:
            List of similar knowledge entries
        """
        if not query or not query.strip():
            return []

        try:
            query_embedding = self.embeddings.embed_text(query.strip(), input_type="search_query")

            if not query_embedding:
                return []

            params = {
                "query_embedding": query_embedding,
                "include_public": include_public,
                "match_threshold": match_threshold,
                "match_count": limit
            }

            if user_id:
                params["filter_user_id"] = user_id
            if category:
                params["filter_category"] = category
            if stock_symbols:
                params["filter_symbols"] = stock_symbols

            response = self.client.rpc("search_knowledge_base", params).execute()

            results = response.data if response.data else []
            logger.info(f"Knowledge search returned {len(results)} results")
            return results

        except Exception as e:
            logger.error(f"Error searching knowledge base: {str(e)}")
            return []

    def get_stock_context_for_rag(
        self,
        symbol: str,
        include_analyses: bool = True,
        include_news: bool = True,
        include_signals: bool = True,
        max_items_per_type: int = 5
    ) -> dict[str, Any]:
        """
        Get comprehensive context for a stock for RAG.

        Args:
            symbol: Stock symbol
            include_analyses: Include recent analyses
            include_news: Include recent news
            include_signals: Include recent signals
            max_items_per_type: Maximum items per category

        Returns:
            Dictionary with all available context
        """
        context = {"symbol": symbol}

        try:
            # Get stock ID
            stock = self.get_stock_by_symbol(symbol)
            if not stock:
                return context

            stock_id = stock["id"]
            context["stock"] = stock

            # Get recent analyses
            if include_analyses:
                result = self.client.table("analysis").select(
                    "id, mode, signal, confidence, reasoning, technical_summary, "
                    "sentiment_summary, target_price, stop_loss, created_at"
                ).eq("stock_id", stock_id).order(
                    "created_at", desc=True
                ).limit(max_items_per_type).execute()
                context["recent_analyses"] = result.data if result.data else []

            # Get recent news
            if include_news:
                result = self.client.table("news").select(
                    "id, headline, summary, source, sentiment_label, sentiment_score, published_at"
                ).eq("stock_id", stock_id).order(
                    "published_at", desc=True
                ).limit(max_items_per_type).execute()
                context["recent_news"] = result.data if result.data else []

            # Get recent signals
            if include_signals:
                result = self.client.table("signals").select(
                    "id, signal_type, signal, price_at_signal, reason, importance, created_at"
                ).eq("stock_id", stock_id).order(
                    "created_at", desc=True
                ).limit(max_items_per_type).execute()
                context["recent_signals"] = result.data if result.data else []

            # Get latest indicators
            indicators = self.get_latest_indicators(stock_id)
            if indicators:
                context["latest_indicators"] = indicators

            return context

        except Exception as e:
            logger.error(f"Error getting stock context: {str(e)}")
            return context


if __name__ == "__main__":
    # Test storage functionality
    import sys

    print("Testing Stock Radar Storage Module")
    print("=" * 50)

    # Test configured embeddings provider
    print("\n1. Testing embeddings client...")
    embeddings = create_embeddings_client(
        provider=getattr(_cfg, "embedding_provider", None) if _cfg else os.getenv("EMBEDDING_PROVIDER", "cohere"),
        model=getattr(_cfg, "embedding_model", None) if _cfg else os.getenv("EMBEDDING_MODEL"),
        dimension=getattr(_cfg, "embedding_dim", None) if _cfg else None,
        cohere_key=getattr(_cfg, "cohere_api_key", None) if _cfg else os.getenv("COHERE_API_KEY"),
        gemini_key=getattr(_cfg, "gemini_api_key", None) if _cfg else os.getenv("GEMINI_API_KEY"),
    )

    if embeddings.is_available():
        print(
            f"   Embeddings available via provider={embeddings.provider_name} model={embeddings.model}"
        )
        test_embedding = embeddings.embed_text("RELIANCE stock is bullish today")
        if test_embedding:
            print(f"   Generated embedding with {len(test_embedding)} dimensions")
        else:
            print("   Warning: Failed to generate test embedding")
    else:
        print("   Warning: Embeddings not available - check provider API keys and configuration")

    # Test Supabase storage
    print("\n2. Testing StockStorage...")
    try:
        storage = StockStorage()
        print("   Supabase client initialized")

        if storage.ensure_schema():
            print("   Schema verification passed")
        else:
            print("   Warning: Schema verification failed - run migrations")

        # List existing stocks
        stocks = storage.list_stocks()
        print(f"   Found {len(stocks)} existing stocks")

    except ValueError as e:
        print(f"   Error: {str(e)}")
        print("   Set SUPABASE_URL and SUPABASE_KEY environment variables")
        sys.exit(1)
    except Exception as e:
        print(f"   Unexpected error: {str(e)}")
        sys.exit(1)

    print("\n" + "=" * 50)
    print("Storage module tests completed")
