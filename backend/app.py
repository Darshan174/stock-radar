from __future__ import annotations

import os
import re
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Query, Request, status

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

load_dotenv(ROOT_DIR / ".env")

from main import StockRadar  # noqa: E402
from agents.chat_assistant import ChatMessage, StockChatAssistant  # noqa: E402
from agents.fetcher import NewsItem, StockFetcher  # noqa: E402
from agents.rag_retriever import RAGRetriever  # noqa: E402
from agents.scorer import StockScorer  # noqa: E402
from agents.storage import StockStorage  # noqa: E402

from backend.auth import verify_backend_auth  # noqa: E402
from backend.jobs import AnalyzeJobManager  # noqa: E402
from backend.schemas import (  # noqa: E402
    AnalyzeJobCreateRequest,
    AnalyzeJobCreated,
    AnalyzeJobStatus,
    AskRequest,
    AskResponse,
)

SYMBOL_RE = re.compile(r"^[A-Za-z0-9.\-^]{1,20}$")
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
VALID_PERIODS = {"1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "max"}

POSITIVE_NEWS_TERMS = {
    "beat",
    "beats",
    "surge",
    "rally",
    "growth",
    "record",
    "upgrade",
    "partnership",
    "profit",
    "bullish",
    "strong",
}
NEGATIVE_NEWS_TERMS = {
    "miss",
    "misses",
    "drop",
    "fall",
    "downgrade",
    "lawsuit",
    "loss",
    "bearish",
    "weak",
    "recall",
    "probe",
}

app = FastAPI(
    title="Stock Radar Backend API",
    version="1.0.0",
    description="Private backend API for Vercel + Railway split deployment",
)

api = APIRouter(prefix="/v1", dependencies=[Depends(verify_backend_auth)])
jobs = AnalyzeJobManager(max_workers=2, ttl_seconds=1800)

_radar_lock = threading.Lock()
_radar: StockRadar | None = None


@app.on_event("startup")
def _startup() -> None:
    app.state.started_at = datetime.now(timezone.utc)


def _validate_symbol(symbol: str) -> str:
    normalized = symbol.strip().upper()
    if not SYMBOL_RE.match(normalized):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid symbol format. Use letters, numbers, dots, hyphens, carets (max 20 chars).",
        )
    return normalized


def _quote_to_dict(quote: Any) -> dict[str, Any]:
    return {
        "symbol": getattr(quote, "symbol", ""),
        "price": float(getattr(quote, "price", 0) or 0),
        "change": float(getattr(quote, "change", 0) or 0),
        "change_percent": float(getattr(quote, "change_percent", 0) or 0),
        "volume": int(getattr(quote, "volume", 0) or 0),
        "avg_volume": int(getattr(quote, "avg_volume", 0) or 0),
        "high": float(getattr(quote, "high", 0) or 0),
        "low": float(getattr(quote, "low", 0) or 0),
        "open": float(getattr(quote, "open", 0) or 0),
        "prev_close": float(getattr(quote, "prev_close", 0) or 0),
    }


def _news_item_to_dict(item: NewsItem) -> dict[str, Any]:
    return {
        "title": item.headline,
        "source": item.source,
        "published_at": item.published_at.isoformat() if item.published_at else None,
        "url": item.url,
    }


def _get_radar() -> StockRadar:
    global _radar
    if _radar is None:
        with _radar_lock:
            if _radar is None:
                _radar = StockRadar()
    return _radar


def _build_fetcher() -> StockFetcher:
    return StockFetcher()


def _build_storage() -> StockStorage:
    try:
        return StockStorage()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Storage initialization failed: {exc}",
        ) from exc


def _run_analysis(symbol: str, mode: str, period: str) -> dict[str, Any]:
    radar = _get_radar()
    return radar.analyze_stock(symbol=symbol, mode=mode, period=period)


def _calculate_rsi_series(closes: list[float], period: int) -> list[float]:
    if len(closes) < period + 1:
        return []

    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]
    gains = [d if d > 0 else 0.0 for d in deltas]
    losses = [-d if d < 0 else 0.0 for d in deltas]

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    rsis: list[float] = []
    rs = (avg_gain / avg_loss) if avg_loss else 0
    rsis.append(100 - (100 / (1 + rs)) if avg_loss else 100.0)

    for idx in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[idx]) / period
        avg_loss = (avg_loss * (period - 1) + losses[idx]) / period
        if avg_loss == 0:
            rsis.append(100.0)
        else:
            rs = avg_gain / avg_loss
            rsis.append(100 - (100 / (1 + rs)))

    return rsis


def _headline_sentiment_score(headline: str) -> float:
    words = set(re.findall(r"[A-Za-z]+", headline.lower()))
    positive = len(words & POSITIVE_NEWS_TERMS)
    negative = len(words & NEGATIVE_NEWS_TERMS)
    if positive == negative == 0:
        return 0.0
    return (positive - negative) / max(positive + negative, 1)


@api.post("/analyze/jobs", response_model=AnalyzeJobCreated, status_code=status.HTTP_202_ACCEPTED)
def create_analyze_job(payload: AnalyzeJobCreateRequest, request: Request) -> AnalyzeJobCreated:
    symbol = _validate_symbol(payload.symbol)
    period = payload.period if payload.period in VALID_PERIODS else "max"

    job_id = jobs.submit(_run_analysis, symbol, payload.mode, period)
    status_url = f"{request.base_url}v1/analyze/jobs/{job_id}"

    return AnalyzeJobCreated(jobId=job_id, statusUrl=status_url, status="queued")


@api.get("/analyze/jobs/{job_id}", response_model=AnalyzeJobStatus)
def get_analyze_job(job_id: str) -> AnalyzeJobStatus:
    record = jobs.get(job_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")

    return AnalyzeJobStatus(
        jobId=record.job_id,
        status=record.status,  # type: ignore[arg-type]
        result=record.result,
        error=record.error,
    )


@api.post("/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Question is required")

    stock_symbol = _validate_symbol(payload.symbol) if payload.symbol else None

    assistant = StockChatAssistant()

    if payload.sessionId and SESSION_RE.match(payload.sessionId):
        assistant.session_id = payload.sessionId
        try:
            history = assistant.storage.get_chat_history(session_id=payload.sessionId, limit=40)
            for msg in history:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role not in ("user", "assistant", "system") or not content:
                    continue
                assistant.conversation_history.append(
                    ChatMessage(
                        role=role,
                        content=content,
                        stock_symbols=msg.get("stock_symbols") or [],
                        context_used=msg.get("context_used"),
                        created_at=msg.get("created_at") or datetime.now(timezone.utc).isoformat(),
                    )
                )
        except Exception:
            # If history retrieval fails we still allow a clean session continuation.
            pass
    else:
        assistant.start_session()

    response = assistant.ask(question=question, stock_symbol=stock_symbol)
    if response.model_used == "error":
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=response.answer or "All chat models failed",
        )

    return AskResponse(
        answer=response.answer,
        stockSymbols=response.stock_symbols,
        sourcesUsed=response.sources_used,
        modelUsed=response.model_used,
        tokensUsed=response.tokens_used,
        processingTimeMs=response.processing_time_ms,
        sessionId=assistant.session_id or "",
        contextRetrieved={
            "totalResults": response.context_retrieved.total_results,
            "sourcesSearched": response.context_retrieved.sources_searched,
            "retrievalTimeMs": response.context_retrieved.retrieval_time_ms,
        },
    )


@api.get("/fundamentals")
def fundamentals(symbol: str = Query(..., min_length=1, max_length=20)) -> dict[str, Any]:
    normalized = _validate_symbol(symbol)
    fetcher = _build_fetcher()
    data = fetcher.get_fundamentals(normalized)
    if not data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No fundamentals found")
    return data


@api.get("/agent/momentum")
def agent_momentum(symbol: str = Query(..., min_length=1, max_length=20)) -> dict[str, Any]:
    normalized = _validate_symbol(symbol)
    fetcher = _build_fetcher()
    scorer = StockScorer()

    quote = fetcher.get_quote(normalized)
    history = fetcher.get_price_history(normalized, period="6mo", interval="1d")
    indicators = fetcher.calculate_indicators(history)

    if quote is None or not indicators:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Insufficient data for momentum analysis")

    momentum_score, breakdown = scorer.calculate_momentum_score(indicators)

    return {
        "symbol": normalized,
        "momentum_score": momentum_score,
        "signal": "bullish" if momentum_score > 60 else "bearish" if momentum_score < 40 else "neutral",
        "breakdown": breakdown,
        "timestamp": quote.timestamp.isoformat() if quote.timestamp else datetime.now(timezone.utc).isoformat(),
    }


@api.get("/agent/rsi-divergence")
def agent_rsi_divergence(
    symbol: str = Query(..., min_length=1, max_length=20),
    period: int = Query(default=14, ge=1, le=200),
    lookback: int = Query(default=5, ge=1, le=100),
) -> dict[str, Any]:
    normalized = _validate_symbol(symbol)
    fetcher = _build_fetcher()

    history = fetcher.get_price_history(normalized, period="1y", interval="1d")
    if len(history) < max(period + lookback + 5, 30):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Insufficient historical data")

    closes = [float(p.close) for p in history]
    rsi_series = _calculate_rsi_series(closes, period)
    if len(rsi_series) < lookback + 1:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Insufficient RSI history")

    prices = closes[-lookback:]
    rsi_values = rsi_series[-lookback:]

    price_high_idx = max(range(len(prices)), key=lambda i: prices[i])
    price_low_idx = min(range(len(prices)), key=lambda i: prices[i])

    bearish_divergence = False
    bullish_divergence = False

    if price_high_idx > 0:
        prev_high_idx = max(range(price_high_idx), key=lambda i: prices[i])
        if prices[price_high_idx] > prices[prev_high_idx] and rsi_values[price_high_idx] < rsi_values[prev_high_idx]:
            bearish_divergence = True

    if price_low_idx < len(prices) - 1:
        trailing = range(price_low_idx + 1, len(prices))
        prev_low_idx = min(trailing, key=lambda i: prices[i], default=price_low_idx)
        if prices[price_low_idx] < prices[prev_low_idx] and rsi_values[price_low_idx] > rsi_values[prev_low_idx]:
            bullish_divergence = True

    signal = "neutral"
    confidence = 0
    if bearish_divergence:
        signal = "bearish"
        confidence = 70
    elif bullish_divergence:
        signal = "bullish"
        confidence = 70

    return {
        "symbol": normalized,
        "rsi_divergence": {
            "signal": signal,
            "confidence": confidence,
            "bearish_divergence": bearish_divergence,
            "bullish_divergence": bullish_divergence,
            "current_rsi": round(rsi_values[-1], 2),
            "rsi_trend": "up" if rsi_values[-1] > rsi_values[-2] else "down",
            "price_trend": "up" if prices[-1] > prices[-2] else "down",
            "lookback_period": lookback,
            "data_points": len(prices),
        },
        "timestamp": history[-1].timestamp.isoformat(),
    }


@api.get("/agent/social-sentiment")
def agent_social_sentiment(symbol: str = Query(..., min_length=1, max_length=20)) -> dict[str, Any]:
    normalized = _validate_symbol(symbol)
    fetcher = _build_fetcher()

    reddit = fetcher.get_reddit_sentiment(normalized)
    social = fetcher.get_social_sentiment(normalized)
    overall = social.get("overall") or social.get("overall_sentiment") or "neutral"

    return {
        "symbol": normalized,
        "reddit": {
            "mentions": reddit.get("mentions", 0),
            "rank": reddit.get("rank", 0),
            "sentiment": reddit.get("sentiment", "neutral"),
            "subreddits": reddit.get("subreddits", []),
            "fetched_at": reddit.get("fetched_at", datetime.now(timezone.utc).isoformat()),
        },
        "social_sentiment": {
            "overall": overall,
            "sources": {
                "reddit": {
                    "mentions": social.get("reddit_mentions", 0),
                    "rank": social.get("reddit_rank"),
                }
            },
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@api.get("/agent/support-resistance")
def agent_support_resistance(
    symbol: str = Query(..., min_length=1, max_length=20),
    period: int = Query(default=14, ge=1, le=365),
) -> dict[str, Any]:
    normalized = _validate_symbol(symbol)
    fetcher = _build_fetcher()

    history = fetcher.get_price_history(normalized, period="1y", interval="1d")
    if len(history) < max(period, 20):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Insufficient historical data")

    recent = history[-period:]
    close = float(recent[-1].close)
    high = max(float(p.high) for p in recent)
    low = min(float(p.low) for p in recent)

    pivot = (high + low + close) / 3
    r1 = 2 * pivot - low
    r2 = pivot + (high - low)
    r3 = high + 2 * (pivot - low)
    s1 = 2 * pivot - high
    s2 = pivot - (high - low)
    s3 = low - 2 * (high - pivot)

    indicators = fetcher.calculate_indicators(history)

    return {
        "symbol": normalized,
        "support_resistance": {
            "pivot_points": {
                "pivot": round(pivot, 2),
                "resistance_1": round(r1, 2),
                "resistance_2": round(r2, 2),
                "resistance_3": round(r3, 2),
                "support_1": round(s1, 2),
                "support_2": round(s2, 2),
                "support_3": round(s3, 2),
            },
            "bollinger_bands": {
                "upper": indicators.get("bollinger_upper"),
                "middle": indicators.get("bollinger_middle"),
                "lower": indicators.get("bollinger_lower"),
            },
            "atr": {
                "value": indicators.get("atr_14"),
                "period": 14,
            },
        },
        "current_price": round(close, 2),
        "price_position": "above_pivot" if close > pivot else "below_pivot",
        "timestamp": recent[-1].timestamp.isoformat(),
    }


@api.get("/agent/stock-score")
def agent_stock_score(symbol: str = Query(..., min_length=1, max_length=20)) -> dict[str, Any]:
    normalized = _validate_symbol(symbol)
    fetcher = _build_fetcher()
    scorer = StockScorer()

    quote = fetcher.get_quote(normalized)
    fundamentals = fetcher.get_fundamentals(normalized) or {}
    history = fetcher.get_price_history(normalized, period="1y", interval="1d")
    indicators = fetcher.calculate_indicators(history)

    if quote is None or not indicators:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Insufficient data for stock scoring")

    scores = scorer.calculate_all_scores(
        quote=_quote_to_dict(quote),
        indicators=indicators,
        fundamentals=fundamentals,
        price_history_days=len(history),
        has_news=False,
    )

    return {
        "symbol": normalized,
        "overall_score": scores.composite_score,
        "recommendation": scores.overall_signal,
        "confidence": round(scores.confidence_score / 100, 2),
        "scores": {
            "momentum": {
                "score": scores.momentum_score,
                "breakdown": scores.momentum_breakdown,
            },
            "value": {
                "score": scores.value_score,
                "breakdown": scores.value_breakdown,
            },
            "quality": {
                "score": scores.quality_score,
                "breakdown": scores.quality_breakdown,
            },
            "risk": {
                "score": scores.risk_score,
                "breakdown": scores.risk_breakdown,
            },
        },
        "current_price": quote.price,
        "timestamp": quote.timestamp.isoformat(),
    }


@api.get("/agent/news-impact")
def agent_news_impact(
    symbol: str = Query(..., min_length=1, max_length=20),
    days: int = Query(default=7, ge=1, le=90),
) -> dict[str, Any]:
    normalized = _validate_symbol(symbol)
    fetcher = _build_fetcher()

    to_date = datetime.now(timezone.utc)
    from_date = to_date - timedelta(days=days)

    news = fetcher.get_news_yahoo(normalized)
    news.extend(fetcher.get_news_finnhub(normalized, from_date=from_date, to_date=to_date))

    if not news:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No news data available")

    deduped: list[NewsItem] = []
    seen = set()
    for item in sorted(news, key=lambda n: n.published_at, reverse=True):
        key = (item.headline or "", item.source or "")
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    recent_news = deduped[:5]

    sentiment_scores = [_headline_sentiment_score(item.headline or "") for item in deduped]
    avg_sentiment = sum(sentiment_scores) / max(len(sentiment_scores), 1)
    if avg_sentiment > 0.15:
        sentiment_label = "bullish"
        expected_impact = "positive"
    elif avg_sentiment < -0.15:
        sentiment_label = "bearish"
        expected_impact = "negative"
    else:
        sentiment_label = "neutral"
        expected_impact = "neutral"

    key_topics = []
    topic_candidates = sorted(POSITIVE_NEWS_TERMS | NEGATIVE_NEWS_TERMS)
    all_headlines = " ".join(item.headline.lower() for item in recent_news if item.headline)
    for term in topic_candidates:
        if term in all_headlines:
            key_topics.append(term)

    fundamentals = fetcher.get_fundamentals(normalized) or {}
    sector_context = None
    sector = fundamentals.get("sector")
    if sector:
        try:
            retriever = RAGRetriever()
            context = retriever.get_sector_sentiment_context(sector)
            sector_context = {
                "sector": sector,
                "sector_sentiment": context.get("sentiment_summary", "neutral"),
            }
        except Exception:
            sector_context = {"sector": sector, "sector_sentiment": "neutral"}

    return {
        "symbol": normalized,
        "news_impact": {
            "overall_sentiment": sentiment_label,
            "sentiment_score": round(avg_sentiment, 3),
            "expected_price_impact": expected_impact,
            "key_topics": key_topics,
            "news_count": len(deduped),
            "recent_news": [
                {
                    "title": item.headline,
                    "source": item.source,
                    "published_at": item.published_at.isoformat() if item.published_at else None,
                    "sentiment": (
                        "positive"
                        if _headline_sentiment_score(item.headline or "") > 0
                        else "negative"
                        if _headline_sentiment_score(item.headline or "") < 0
                        else "neutral"
                    ),
                }
                for item in recent_news
            ],
        },
        "sector_context": sector_context,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/health")
def health() -> dict[str, Any]:
    started_at = getattr(app.state, "started_at", datetime.now(timezone.utc))
    now = datetime.now(timezone.utc)

    dependencies: dict[str, Any] = {
        "supabase": {"status": "unknown"},
        "llm": {"status": "unknown"},
    }

    try:
        storage = _build_storage()
        dependencies["supabase"] = {
            "status": "healthy" if storage.ensure_schema() else "degraded",
        }
    except HTTPException as exc:
        dependencies["supabase"] = {"status": "unhealthy", "error": exc.detail}
    except Exception as exc:
        dependencies["supabase"] = {"status": "unhealthy", "error": str(exc)}

    llm_available = any(
        bool(os.getenv(k))
        for k in ("ZAI_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY")
    )
    dependencies["llm"] = {
        "status": "healthy" if llm_available else "degraded",
        "configuredProviders": [
            provider
            for provider, key in (
                ("zai", "ZAI_API_KEY"),
                ("groq", "GROQ_API_KEY"),
                ("gemini", "GEMINI_API_KEY"),
            )
            if os.getenv(key)
        ],
    }

    return {
        "status": "ok",
        "service": "stock-radar-backend",
        "timestamp": now.isoformat(),
        "uptimeSeconds": round((now - started_at).total_seconds(), 3),
        "dependencies": dependencies,
    }


app.include_router(api)
