"""
Stock Radar - Centralized Configuration Management.

Uses Pydantic BaseSettings to load configuration from environment variables
and .env files with type safety, validation, and sensible defaults.

WHY THIS MATTERS (AI Engineering):
- Every production AI system needs centralized, type-safe configuration.
- API keys, model names, thresholds should never be hardcoded.
- Pydantic validates types automatically - catches config errors at startup.
- The `.env` file keeps secrets out of git.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    All configuration for Stock Radar in one place.

    Values are loaded in this priority order:
    1. Environment variables (highest priority)
    2. .env file
    3. Default values defined here (lowest priority)
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Don't fail on unknown env vars
    )

    # -------------------------------------------------------------------------
    # App
    # -------------------------------------------------------------------------
    app_env: str = Field(default="dev", alias="APP_ENV")
    app_name: str = Field(default="stock-radar", alias="APP_NAME")
    log_json: bool = Field(default=False, alias="LOG_JSON")

    # -------------------------------------------------------------------------
    # LLM Providers
    # -------------------------------------------------------------------------
    zai_api_key: str | None = Field(default=None, alias="ZAI_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    ollama_api_url: str = Field(
        default="http://localhost:11434", alias="OLLAMA_API_URL"
    )

    # Model fallback order (comma-separated)
    llm_fallback_order: str = Field(
        default="zai/glm-4.7,gemini/gemini-2.0-flash,ollama/mistral",
        alias="LLM_FALLBACK_ORDER",
    )
    llm_temperature: float = Field(default=0.3, alias="LLM_TEMPERATURE")
    llm_max_tokens: int = Field(default=2000, alias="LLM_MAX_TOKENS")
    llm_timeout_sec: int = Field(default=60, alias="LLM_TIMEOUT_SEC")

    # -------------------------------------------------------------------------
    # Cost tracking (per 1K tokens)
    # -------------------------------------------------------------------------
    cost_zai_input: float = Field(default=0.0, alias="COST_ZAI_INPUT_PER_1K")
    cost_zai_output: float = Field(default=0.0, alias="COST_ZAI_OUTPUT_PER_1K")
    cost_gemini_input: float = Field(default=0.0, alias="COST_GEMINI_INPUT_PER_1K")
    cost_gemini_output: float = Field(default=0.0, alias="COST_GEMINI_OUTPUT_PER_1K")
    cost_cohere_per_call: float = Field(default=0.0001, alias="COST_COHERE_PER_CALL")

    # -------------------------------------------------------------------------
    # Data Providers
    # -------------------------------------------------------------------------
    twelve_data_api_key: str | None = Field(default=None, alias="TWELVE_DATA_API_KEY")
    finnhub_api_key: str | None = Field(default=None, alias="FINNHUB_API_KEY")
    alpha_vantage_api_key: str | None = Field(
        default=None, alias="ALPHA_VANTAGE_API_KEY"
    )

    # -------------------------------------------------------------------------
    # Database
    # -------------------------------------------------------------------------
    supabase_url: str | None = Field(default=None, alias="SUPABASE_URL")
    supabase_key: str | None = Field(default=None, alias="SUPABASE_KEY")

    # -------------------------------------------------------------------------
    # Embeddings
    # -------------------------------------------------------------------------
    cohere_api_key: str | None = Field(default=None, alias="COHERE_API_KEY")
    embedding_model: str = Field(
        default="embed-english-v3.0", alias="EMBEDDING_MODEL"
    )
    embedding_dim: int = Field(default=1024, alias="EMBEDDING_DIM")

    # -------------------------------------------------------------------------
    # Notifications
    # -------------------------------------------------------------------------
    slack_bot_token: str | None = Field(default=None, alias="SLACK_BOT_TOKEN")
    slack_channel: str | None = Field(default=None, alias="SLACK_CHANNEL")
    telegram_bot_token: str | None = Field(default=None, alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str | None = Field(default=None, alias="TELEGRAM_CHAT_ID")

    # -------------------------------------------------------------------------
    # Scoring & Analysis
    # -------------------------------------------------------------------------
    scoring_preset: str = Field(default="balanced", alias="SCORING_PRESET")
    top_k_rag: int = Field(default=5, alias="TOP_K_RAG")
    rag_match_threshold: float = Field(default=0.4, alias="RAG_MATCH_THRESHOLD")

    # -------------------------------------------------------------------------
    # Retry
    # -------------------------------------------------------------------------
    retry_max_attempts: int = Field(default=3, alias="RETRY_MAX_ATTEMPTS")
    retry_min_wait_sec: int = Field(default=1, alias="RETRY_MIN_WAIT_SEC")
    retry_max_wait_sec: int = Field(default=10, alias="RETRY_MAX_WAIT_SEC")

    # -------------------------------------------------------------------------
    # Cache (Redis)
    # -------------------------------------------------------------------------
    redis_url: str | None = Field(default=None, alias="REDIS_URL")
    cache_quote_ttl_sec: int = Field(default=60, alias="CACHE_QUOTE_TTL_SEC")
    cache_fundamentals_ttl_sec: int = Field(
        default=3600, alias="CACHE_FUNDAMENTALS_TTL_SEC"
    )
    cache_analysis_ttl_sec: int = Field(default=900, alias="CACHE_ANALYSIS_TTL_SEC")
    cache_embedding_ttl_sec: int = Field(
        default=86400, alias="CACHE_EMBEDDING_TTL_SEC"
    )

    # -------------------------------------------------------------------------
    # Metrics
    # -------------------------------------------------------------------------
    metrics_enabled: bool = Field(default=True, alias="METRICS_ENABLED")

    # -------------------------------------------------------------------------
    # Guardrails
    # -------------------------------------------------------------------------
    guardrails_enabled: bool = Field(default=True, alias="GUARDRAILS_ENABLED")
    guardrails_max_confidence: float = Field(
        default=0.95, alias="GUARDRAILS_MAX_CONFIDENCE"
    )
    guardrails_require_reasoning: bool = Field(
        default=True, alias="GUARDRAILS_REQUIRE_REASONING"
    )

    # -------------------------------------------------------------------------
    # Prompt Versioning
    # -------------------------------------------------------------------------
    prompt_version: str = Field(default="v1", alias="PROMPT_VERSION")

    @property
    def fallback_models(self) -> list[str]:
        """Parse the LLM fallback order into a list."""
        return [m.strip() for m in self.llm_fallback_order.split(",") if m.strip()]


# Singleton - created once, imported everywhere
settings = Settings()
