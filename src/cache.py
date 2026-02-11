"""
Stock Radar - Caching Layer.

Provides a unified caching interface with Redis (production) or
in-memory dict (development) backends. Caches expensive operations
like LLM calls, data fetches, and embeddings.

WHY THIS MATTERS (AI Engineering):
- LLM calls cost money and take time. Caching identical requests saves both.
- Stock quotes don't change every millisecond - a 60s TTL is fine.
- Embeddings for the same text never change - cache them permanently.
- Interviewers ask: "How do you optimize performance and reduce costs?"

CACHE STRATEGY:
    | Data Type     | TTL      | Key Pattern                        |
    |---------------|----------|------------------------------------|
    | Quotes        | 60s      | quote:{symbol}                     |
    | Fundamentals  | 1 hour   | fundamentals:{symbol}              |
    | Analysis      | 15 min   | analysis:{symbol}:{mode}:{prompt}  |
    | Embeddings    | 24 hours | embedding:{hash(text)}             |
    | Indicators    | 5 min    | indicators:{symbol}                |

USAGE:
    from cache import cache

    # Try cache first, compute on miss
    result = cache.get("quote:AAPL")
    if result is None:
        result = fetch_quote("AAPL")
        cache.set("quote:AAPL", result, ttl=60)
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from metrics import CACHE_HITS, CACHE_MISSES


class InMemoryCache:
    """
    Simple in-memory cache with TTL support.

    Used when Redis is not available (development, testing).
    Data is lost on restart - that's fine for dev.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Any, float]] = {}  # key -> (value, expires_at)

    def get(self, key: str) -> Any | None:
        """Get a value from cache. Returns None on miss or expiry."""
        entry = self._store.get(key)
        if entry is None:
            return None

        value, expires_at = entry
        if expires_at > 0 and time.time() > expires_at:
            del self._store[key]
            return None

        return value

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache (must be JSON-serializable)
            ttl: Time-to-live in seconds (0 = no expiry)
        """
        expires_at = time.time() + ttl if ttl > 0 else 0
        self._store[key] = (value, expires_at)

    def delete(self, key: str) -> None:
        """Delete a key from cache."""
        self._store.pop(key, None)

    def clear(self) -> None:
        """Clear all cached data."""
        self._store.clear()

    def stats(self) -> dict[str, int]:
        """Get cache statistics."""
        now = time.time()
        alive = sum(1 for _, (_, exp) in self._store.items() if exp == 0 or exp > now)
        return {"total_keys": len(self._store), "alive_keys": alive}


class RedisCache:
    """
    Redis-backed cache for production use.

    Falls back to InMemoryCache if Redis is unavailable.
    """

    def __init__(self, redis_url: str) -> None:
        try:
            import redis
            self._client = redis.from_url(redis_url, decode_responses=True)
            self._client.ping()
            self._available = True
        except Exception:
            self._available = False
            self._fallback = InMemoryCache()

    def get(self, key: str) -> Any | None:
        if not self._available:
            return self._fallback.get(key)
        try:
            raw = self._client.get(key)
            if raw is None:
                return None
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        if not self._available:
            self._fallback.set(key, value, ttl)
            return
        try:
            raw = json.dumps(value, default=str)
            if ttl > 0:
                self._client.setex(key, ttl, raw)
            else:
                self._client.set(key, raw)
        except Exception:
            pass

    def delete(self, key: str) -> None:
        if not self._available:
            self._fallback.delete(key)
            return
        try:
            self._client.delete(key)
        except Exception:
            pass

    def clear(self) -> None:
        if not self._available:
            self._fallback.clear()
            return
        try:
            self._client.flushdb()
        except Exception:
            pass


class StockRadarCache:
    """
    High-level caching API with domain-specific methods.

    Automatically tracks cache hits/misses in Prometheus.
    """

    def __init__(self, backend: InMemoryCache | RedisCache) -> None:
        self._backend = backend

    def get(self, key: str, cache_type: str = "general") -> Any | None:
        """Get from cache, tracking metrics."""
        result = self._backend.get(key)
        if result is not None:
            CACHE_HITS.labels(cache_type=cache_type).inc()
        else:
            CACHE_MISSES.labels(cache_type=cache_type).inc()
        return result

    def set(self, key: str, value: Any, ttl: int = 0) -> None:
        """Set in cache."""
        self._backend.set(key, value, ttl)

    def delete(self, key: str) -> None:
        """Delete from cache."""
        self._backend.delete(key)

    # -------------------------------------------------------------------------
    # Domain-specific helpers
    # -------------------------------------------------------------------------

    def get_quote(self, symbol: str) -> dict | None:
        return self.get(f"quote:{symbol}", cache_type="quote")

    def set_quote(self, symbol: str, data: dict, ttl: int = 60) -> None:
        self.set(f"quote:{symbol}", data, ttl)

    def get_fundamentals(self, symbol: str) -> dict | None:
        return self.get(f"fundamentals:{symbol}", cache_type="fundamentals")

    def set_fundamentals(self, symbol: str, data: dict, ttl: int = 3600) -> None:
        self.set(f"fundamentals:{symbol}", data, ttl)

    def get_analysis(self, symbol: str, mode: str, prompt_version: str) -> dict | None:
        key = f"analysis:{symbol}:{mode}:{prompt_version}"
        return self.get(key, cache_type="analysis")

    def set_analysis(
        self, symbol: str, mode: str, prompt_version: str, data: dict, ttl: int = 900
    ) -> None:
        key = f"analysis:{symbol}:{mode}:{prompt_version}"
        self.set(key, data, ttl)

    def get_embedding(self, text: str) -> list[float] | None:
        key = f"embedding:{_hash_text(text)}"
        return self.get(key, cache_type="embedding")

    def set_embedding(self, text: str, vector: list[float], ttl: int = 86400) -> None:
        key = f"embedding:{_hash_text(text)}"
        self.set(key, vector, ttl)

    def get_indicators(self, symbol: str) -> dict | None:
        return self.get(f"indicators:{symbol}", cache_type="indicators")

    def set_indicators(self, symbol: str, data: dict, ttl: int = 300) -> None:
        self.set(f"indicators:{symbol}", data, ttl)


def _hash_text(text: str) -> str:
    """Create a short hash of text for cache keys."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def create_cache(redis_url: str | None = None) -> StockRadarCache:
    """
    Create the appropriate cache backend.

    Uses Redis if REDIS_URL is set, otherwise falls back to in-memory.
    """
    if redis_url:
        backend = RedisCache(redis_url)
    else:
        backend = InMemoryCache()

    return StockRadarCache(backend)


# Singleton - import and use directly
try:
    from config import settings
    cache = create_cache(settings.redis_url)
except Exception:
    cache = create_cache(None)
