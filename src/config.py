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
    zai_api_base: str = Field(
        default="https://open.bigmodel.cn/api/coding/paas/v4", alias="ZAI_API_BASE"
    )
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")

    # Model fallback order (comma-separated)
    llm_fallback_order: str = Field(
        default="openai/glm-4.7,gemini/gemini-2.5-flash",
        alias="LLM_FALLBACK_ORDER",
    )
    # Task-based routes, format:
    # "analysis=model_a,model_b;chat=model_x,model_y;sentiment=model_p,model_q"
    llm_task_routes: str = Field(default="", alias="LLM_TASK_ROUTES")
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
    # ML Model
    # -------------------------------------------------------------------------
    ml_model_path: str | None = Field(default=None, alias="ML_MODEL_PATH")
    ml_model_enabled: bool = Field(default=True, alias="ML_MODEL_ENABLED")

    # -------------------------------------------------------------------------
    # Paper Trading
    # -------------------------------------------------------------------------
    paper_trading_enabled: bool = Field(default=False, alias="PAPER_TRADING_ENABLED")
    paper_trading_dir: str = Field(default="data/paper_trading", alias="PAPER_TRADING_DIR")
    paper_trading_capital: float = Field(default=100_000.0, alias="PAPER_TRADING_CAPITAL")

    # -------------------------------------------------------------------------
    # Kill Switches
    # -------------------------------------------------------------------------
    kill_switch_enabled: bool = Field(default=True, alias="KILL_SWITCH_ENABLED")
    kill_switch_max_daily_loss_pct: float = Field(default=5.0, alias="KILL_SWITCH_MAX_DAILY_LOSS_PCT")
    kill_switch_max_stale_ms: int = Field(default=60_000, alias="KILL_SWITCH_MAX_STALE_MS")
    kill_switch_slippage_threshold_pct: float = Field(default=2.0, alias="KILL_SWITCH_SLIPPAGE_THRESHOLD_PCT")

    # -------------------------------------------------------------------------
    # Metrics Server
    # -------------------------------------------------------------------------
    metrics_port: int = Field(default=9090, alias="METRICS_PORT")
    health_check_enabled: bool = Field(default=True, alias="HEALTH_CHECK_ENABLED")

    # -------------------------------------------------------------------------
    # Prompt Versioning
    # -------------------------------------------------------------------------
    prompt_version: str = Field(default="v1", alias="PROMPT_VERSION")

    # -------------------------------------------------------------------------
    # Broker / Execution
    # -------------------------------------------------------------------------
    broker_mode: str = Field(default="paper", alias="BROKER_MODE")
    broker_retry_max: int = Field(default=3, alias="BROKER_RETRY_MAX")
    broker_retry_backoff: float = Field(default=1.0, alias="BROKER_RETRY_BACKOFF")

    # -------------------------------------------------------------------------
    # Pre-Trade Risk
    # -------------------------------------------------------------------------
    pre_trade_risk_enabled: bool = Field(default=True, alias="PRE_TRADE_RISK_ENABLED")
    pre_trade_max_position: float = Field(default=0.20, alias="PRE_TRADE_MAX_POSITION")
    pre_trade_max_sector: float = Field(default=0.35, alias="PRE_TRADE_MAX_SECTOR")
    pre_trade_max_daily_loss_pct: float = Field(default=5.0, alias="PRE_TRADE_MAX_DAILY_LOSS_PCT")
    pre_trade_max_exposure: float = Field(default=1.0, alias="PRE_TRADE_MAX_EXPOSURE")

    # -------------------------------------------------------------------------
    # Canary Mode
    # -------------------------------------------------------------------------
    canary_enabled: bool = Field(default=False, alias="CANARY_ENABLED")
    canary_dir: str = Field(default="data/canary", alias="CANARY_DIR")
    canary_symbols: str = Field(default="", alias="CANARY_SYMBOLS")
    canary_max_trades: int = Field(default=50, alias="CANARY_MAX_TRADES")
    canary_max_loss_pct: float = Field(default=3.0, alias="CANARY_MAX_LOSS_PCT")
    canary_capital: float = Field(default=10_000.0, alias="CANARY_CAPITAL")

    # -------------------------------------------------------------------------
    # Audit
    # -------------------------------------------------------------------------
    audit_enabled: bool = Field(default=True, alias="AUDIT_ENABLED")
    audit_dir: str = Field(default="data/audit", alias="AUDIT_DIR")

    @property
    def fallback_models(self) -> list[str]:
        """Parse the LLM fallback order into a list."""
        return [m.strip() for m in self.llm_fallback_order.split(",") if m.strip()]

    @property
    def task_model_routes(self) -> dict[str, list[str]]:
        """Parse task-specific model routes from LLM_TASK_ROUTES."""
        routes: dict[str, list[str]] = {}
        if not self.llm_task_routes:
            return routes

        for raw_clause in self.llm_task_routes.split(";"):
            clause = raw_clause.strip()
            if not clause or "=" not in clause:
                continue
            task, model_csv = clause.split("=", 1)
            task = task.strip().lower()
            models = [m.strip() for m in model_csv.split(",") if m.strip()]
            if task and models:
                routes[task] = models
        return routes

    @property
    def canary_symbol_list(self) -> list[str]:
        """Parse canary symbols into a list."""
        return [s.strip().upper() for s in self.canary_symbols.split(",") if s.strip()]


# Singleton - created once, imported everywhere
settings = Settings()
