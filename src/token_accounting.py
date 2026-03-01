"""
Stock Radar - Token Accounting & Cost Estimation.

Tracks every LLM call's token usage and calculates cost estimates.
Every API response includes a `meta` block with full cost transparency.

WHY THIS MATTERS (AI Engineering):
- LLM APIs charge per token. Without tracking, costs spiral out of control.
- Interviewers ask: "How do you manage AI costs at scale?"
- This module answers: by tracking every token in and out, per model, per request.

USAGE:
    from token_accounting import TokenAccountant

    accountant = TokenAccountant()

    # Record an LLM call
    accountant.record_llm_call(
        model="zai/glm-5",
        tokens_in=1200,
        tokens_out=350,
        latency_sec=1.2,
    )

    # Get cost summary for a request
    meta = accountant.get_request_meta()
    # Returns: {
    #   "tokens_in": 1200, "tokens_out": 350, "total_tokens": 1550,
    #   "cost_usd": 0.004, "model_used": "zai/glm-5",
    #   "latency_sec": 1.2, "data_sources_used": ["twelvedata", "finnhub"]
    # }
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from metrics import (
    LLM_LATENCY,
    LLM_TOKENS,
    LLM_REQUESTS,
    API_COST,
    ANALYSIS_DURATION,
)


# Cost rates per 1K tokens (update as pricing changes)
MODEL_COSTS: dict[str, dict[str, float]] = {
    "openai/glm-4.7": {"input": 0.0007, "output": 0.0007},
    "gemini/gemini-2.5-flash": {"input": 0.0, "output": 0.0},  # Free tier
    "groq/llama-3.1-70b-versatile": {"input": 0.0, "output": 0.0},  # Free tier / route default
    "groq/llama-3.1-8b-instant": {"input": 0.0, "output": 0.0},  # Free tier / route default
}


@dataclass
class LLMCallRecord:
    """Record of a single LLM API call."""
    model: str
    tokens_in: int
    tokens_out: int
    latency_sec: float
    cost_usd: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class DataSourceRecord:
    """Record of a data source fetch."""
    provider: str
    operation: str
    latency_sec: float
    success: bool


class TokenAccountant:
    """
    Tracks all LLM calls and data fetches for a single request/analysis.

    Create a new instance per request. After the request completes,
    call get_request_meta() to get the full cost breakdown.
    """

    def __init__(self) -> None:
        self.llm_calls: list[LLMCallRecord] = []
        self.data_sources: list[DataSourceRecord] = []
        self.start_time: float = time.time()
        self._analysis_mode: str = "unknown"

    def set_mode(self, mode: str) -> None:
        """Set the analysis mode (intraday/longterm)."""
        self._analysis_mode = mode

    def record_llm_call(
        self,
        model: str,
        tokens_in: int,
        tokens_out: int,
        latency_sec: float,
    ) -> LLMCallRecord:
        """
        Record an LLM API call with token usage and cost.

        Also updates Prometheus metrics automatically.
        """
        # Calculate cost
        rates = MODEL_COSTS.get(model, {"input": 0.0, "output": 0.0})
        cost = (tokens_in / 1000) * rates["input"] + (tokens_out / 1000) * rates["output"]

        record = LLMCallRecord(
            model=model,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_sec=latency_sec,
            cost_usd=cost,
        )
        self.llm_calls.append(record)

        # Update Prometheus metrics
        provider = model.split("/")[0] if "/" in model else model
        LLM_LATENCY.labels(model=provider).observe(latency_sec)
        LLM_REQUESTS.labels(model=provider, status="success").inc()
        LLM_TOKENS.labels(direction="input", model=provider).inc(tokens_in)
        LLM_TOKENS.labels(direction="output", model=provider).inc(tokens_out)
        if cost > 0:
            API_COST.labels(provider=provider).inc(cost)

        return record

    def record_llm_error(self, model: str) -> None:
        """Record a failed LLM call."""
        provider = model.split("/")[0] if "/" in model else model
        LLM_REQUESTS.labels(model=provider, status="error").inc()

    def record_data_fetch(
        self,
        provider: str,
        operation: str,
        latency_sec: float,
        success: bool = True,
    ) -> None:
        """Record a data provider fetch (yfinance, twelvedata, etc.)."""
        self.data_sources.append(
            DataSourceRecord(
                provider=provider,
                operation=operation,
                latency_sec=latency_sec,
                success=success,
            )
        )

    def get_request_meta(self) -> dict[str, Any]:
        """
        Build the `meta` block that gets included in every API response.

        Returns:
            Dictionary with full transparency on tokens, cost, latency, sources.
        """
        total_tokens_in = sum(c.tokens_in for c in self.llm_calls)
        total_tokens_out = sum(c.tokens_out for c in self.llm_calls)
        total_cost = sum(c.cost_usd for c in self.llm_calls)
        total_latency = time.time() - self.start_time

        # Which model was actually used (last successful call)
        model_used = self.llm_calls[-1].model if self.llm_calls else "none"

        # Unique data sources
        sources_used = list({ds.provider for ds in self.data_sources if ds.success})

        # Record total analysis duration
        ANALYSIS_DURATION.labels(mode=self._analysis_mode).observe(total_latency)

        return {
            "tokens_in": total_tokens_in,
            "tokens_out": total_tokens_out,
            "total_tokens": total_tokens_in + total_tokens_out,
            "cost_usd": round(total_cost, 6),
            "model_used": model_used,
            "models_tried": len(self.llm_calls),
            "latency_sec": round(total_latency, 2),
            "data_sources_used": sources_used,
            "llm_calls": [
                {
                    "model": c.model,
                    "tokens_in": c.tokens_in,
                    "tokens_out": c.tokens_out,
                    "cost_usd": round(c.cost_usd, 6),
                    "latency_sec": round(c.latency_sec, 2),
                }
                for c in self.llm_calls
            ],
        }
