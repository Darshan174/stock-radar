"""
Stock Radar - Streaming LLM Responses (Server-Sent Events).

Streams LLM tokens to the frontend in real-time instead of waiting
for the complete response. Uses Server-Sent Events (SSE) protocol.

WHY THIS MATTERS (AI Engineering):
- Every production AI app (ChatGPT, Claude, etc.) streams responses.
- Users see tokens appear immediately instead of waiting 5-10 seconds.
- If you can't implement streaming, that's a red flag in interviews.

HOW SSE WORKS:
    1. Client makes HTTP request with Accept: text/event-stream
    2. Server keeps the connection open
    3. Server sends chunks: "data: {json}\n\n"
    4. Client receives chunks in real-time
    5. Final chunk: "data: [DONE]\n\n"

USAGE (Python):
    from streaming import stream_llm_response

    async def handler():
        async for chunk in stream_llm_response(prompt, system_prompt):
            yield f"data: {json.dumps(chunk)}\n\n"
        yield "data: [DONE]\n\n"

USAGE (Next.js API route):
    // In your route.ts
    const response = await fetch('http://python-backend/stream', {
        method: 'POST',
        body: JSON.stringify({ symbol: 'AAPL', mode: 'intraday' }),
    });

    const reader = response.body.getReader();
    while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const text = new TextDecoder().decode(value);
        // Parse SSE events from text
    }
"""

from __future__ import annotations

import json
import time
from typing import Any, AsyncGenerator, Generator

import litellm


def stream_llm_response(
    prompt: str,
    system_prompt: str = "",
    models: list[str] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 2000,
) -> Generator[dict[str, Any], None, None]:
    """
    Stream LLM response token by token with model fallback.

    Yields SSE-compatible event dicts.

    Args:
        prompt: User prompt
        system_prompt: System prompt
        models: List of models to try (fallback chain)
        temperature: LLM temperature
        max_tokens: Max tokens to generate

    Yields:
        Dict events: {"type": "token", "content": "..."} for each token
        Dict events: {"type": "done", "model": "...", "tokens": ..., "latency": ...}
        Dict events: {"type": "error", "message": "..."}
    """
    if models is None:
        try:
            from config import settings
            models = settings.fallback_models
        except Exception:
            models = ["zai/glm-4.7", "ollama/mistral"]

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    last_error = None

    for model in models:
        try:
            start = time.time()
            total_tokens = 0
            content_buffer = ""

            # Yield start event
            yield {
                "type": "start",
                "model": model,
            }

            # Use litellm streaming
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            for chunk in response:
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    token_text = delta.content
                    content_buffer += token_text
                    total_tokens += 1

                    yield {
                        "type": "token",
                        "content": token_text,
                    }

            elapsed = time.time() - start

            # Yield completion event
            yield {
                "type": "done",
                "model": model,
                "content": content_buffer,
                "tokens": total_tokens,
                "latency_sec": round(elapsed, 2),
            }

            return  # Success, stop trying models

        except Exception as e:
            last_error = str(e)

            yield {
                "type": "fallback",
                "failed_model": model,
                "error": str(e),
            }

            continue

    # All models failed
    yield {
        "type": "error",
        "message": f"All models failed. Last error: {last_error}",
    }


def format_sse_event(data: dict[str, Any]) -> str:
    """Format a dict as an SSE event string."""
    return f"data: {json.dumps(data)}\n\n"


def stream_analysis_sse(
    symbol: str,
    mode: str = "intraday",
    prompt_version: str = "v1",
) -> Generator[str, None, None]:
    """
    High-level SSE stream for stock analysis.

    Combines data fetching progress + LLM streaming into one stream.

    Yields:
        SSE-formatted strings ready to send over HTTP.
    """
    # Phase 1: Data fetching
    yield format_sse_event({
        "type": "status",
        "phase": "fetching",
        "message": f"Fetching data for {symbol}...",
    })

    try:
        from agents.fetcher import StockFetcher
        from prompt_manager import prompt_manager

        fetcher = StockFetcher()
        quote = fetcher.get_quote(symbol)

        if not quote:
            yield format_sse_event({
                "type": "error",
                "message": f"No data available for {symbol}",
            })
            return

        yield format_sse_event({
            "type": "status",
            "phase": "indicators",
            "message": f"Calculating indicators... Price: {quote.price}",
        })

        history = fetcher.get_price_history(symbol, period="3mo")
        indicators = fetcher.calculate_indicators(history)

        # Phase 2: Build prompt
        yield format_sse_event({
            "type": "status",
            "phase": "analyzing",
            "message": "Starting AI analysis...",
        })

        prompt_name = "intraday_analysis" if mode == "intraday" else "longterm_analysis"

        # Build kwargs for the prompt
        kwargs: dict[str, Any] = {
            "symbol": symbol,
            "price": quote.price,
            "change_percent": quote.change_percent,
            "volume": quote.volume,
            "avg_volume": quote.avg_volume,
            "high": quote.high,
            "low": quote.low,
            "news_text": "No recent news.",
            "social_text": "No social data.",
            "rag_context": "",
        }
        kwargs.update(indicators)

        prompts = prompt_manager.get_prompt(prompt_name, version=prompt_version, **kwargs)

        # Phase 3: Stream LLM
        for event in stream_llm_response(
            prompt=prompts["user"],
            system_prompt=prompts["system"],
        ):
            yield format_sse_event(event)

    except Exception as e:
        yield format_sse_event({
            "type": "error",
            "message": str(e),
        })

    yield "data: [DONE]\n\n"
