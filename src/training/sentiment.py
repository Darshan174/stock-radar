"""
Sentiment feature engine — Phase 6 of the Quant Upgrade Roadmap.

Computes 8 numerical sentiment features from three sources:
  1.  News headlines → FinBERT scores (with VADER fallback)
  2.  Finnhub news-sentiment endpoint (buzz, bullish %)
  3.  Earnings calendar proximity

All functions are pure-functional; no global state.
FinBERT is loaded lazily on first call so the module can be imported
without heavyweight dependencies (``transformers``, ``torch``).

Feature names (canonical order — appended after Phase-5 slot [37:45]):
  news_sentiment_mean      – avg FinBERT/VADER score (−1 … +1)
  news_sentiment_std       – std of per-headline scores
  news_volume_7d           – log₂(1 + number of articles in last 7 d)
  news_sentiment_momentum  – Δ sentiment: last 3 d avg − last 7 d avg
  finnhub_buzz_score       – Finnhub buzz metric (normalised 0-1)
  finnhub_bullish_pct      – Finnhub bullish percent (0-1)
  earnings_proximity       – −log₂(1 + days to next earnings) (higher = closer)
  sentiment_vs_sector      – stock sentiment − sector avg sentiment
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
#  Canonical feature names (8)
# ---------------------------------------------------------------------------

SENTIMENT_FEATURE_NAMES: List[str] = [
    "news_sentiment_mean",
    "news_sentiment_std",
    "news_volume_7d",
    "news_sentiment_momentum",
    "finnhub_buzz_score",
    "finnhub_bullish_pct",
    "earnings_proximity",
    "sentiment_vs_sector",
]

# ---------------------------------------------------------------------------
#  Headline scoring — FinBERT (primary) / VADER (fallback)
# ---------------------------------------------------------------------------

_FINBERT_PIPELINE = None  # lazy singleton
_SCORER = None  # VADER fallback singleton


def _load_finbert():
    """
    Lazily load the ProsusAI/finbert sentiment pipeline.

    Returns the transformers pipeline, or None if unavailable.
    """
    global _FINBERT_PIPELINE  # noqa: PLW0603
    if _FINBERT_PIPELINE is not None:
        return _FINBERT_PIPELINE

    try:
        from transformers import pipeline as hf_pipeline

        _FINBERT_PIPELINE = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            tokenizer="ProsusAI/finbert",
            truncation=True,
            max_length=512,
        )
        logger.info("FinBERT loaded successfully")
        return _FINBERT_PIPELINE
    except Exception as exc:
        logger.warning("FinBERT unavailable (%s); falling back to VADER", exc)
        return None


def _load_vader():
    """Lazily load the VADER sentiment analyser (lightweight fallback)."""
    global _SCORER  # noqa: PLW0603
    if _SCORER is not None:
        return _SCORER

    try:
        import nltk
        from nltk.sentiment.vader import SentimentIntensityAnalyzer

        # Ensure lexicon is available
        try:
            nltk.data.find("sentiment/vader_lexicon.zip")
        except LookupError:
            nltk.download("vader_lexicon", quiet=True)

        _SCORER = SentimentIntensityAnalyzer()
        logger.info("VADER fallback loaded")
        return _SCORER
    except Exception as exc:
        logger.warning("VADER unavailable (%s); headline scoring disabled", exc)
        return None


def score_headline(headline: str) -> float:
    """
    Score a single headline on [−1, +1].

    Tries FinBERT first, then falls back to VADER, then returns NaN.

    FinBERT returns one of {positive, negative, neutral} with a score.
    We map: positive → +score, negative → −score, neutral → ~0.
    """
    # Try FinBERT
    pipe = _load_finbert()
    if pipe is not None:
        try:
            result = pipe(headline[:512])[0]
            label = result["label"].lower()
            score = float(result["score"])
            if label == "negative":
                return -score
            if label == "positive":
                return score
            return score * 0.1  # neutral → near-zero
        except Exception:
            pass

    # VADER fallback
    vader = _load_vader()
    if vader is not None:
        try:
            return float(vader.polarity_scores(headline)["compound"])
        except Exception:
            pass

    return np.nan


def score_headlines(headlines: Sequence[str]) -> np.ndarray:
    """Score a batch of headlines. Returns array of shape (n,) on [−1, +1]."""
    if not headlines:
        return np.array([], dtype=np.float64)

    # Batch FinBERT for efficiency
    pipe = _load_finbert()
    if pipe is not None:
        try:
            results = pipe([h[:512] for h in headlines])
            scores = []
            for r in results:
                label = r["label"].lower()
                s = float(r["score"])
                if label == "negative":
                    scores.append(-s)
                elif label == "positive":
                    scores.append(s)
                else:
                    scores.append(s * 0.1)
            return np.array(scores, dtype=np.float64)
        except Exception:
            pass

    # VADER fallback (headline-by-headline)
    vader = _load_vader()
    if vader is not None:
        try:
            return np.array(
                [vader.polarity_scores(h)["compound"] for h in headlines],
                dtype=np.float64,
            )
        except Exception:
            pass

    return np.full(len(headlines), np.nan, dtype=np.float64)


# ---------------------------------------------------------------------------
#  Headline timestamping helpers
# ---------------------------------------------------------------------------

def _parse_ts(ts: Any) -> Optional[datetime]:
    """Best-effort parse of a timestamp field."""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        dt = ts
    elif isinstance(ts, (int, float)):
        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
    else:
        try:
            from dateutil.parser import parse as dateparse
            dt = dateparse(str(ts))
        except Exception:
            return None
    # Ensure timezone-aware
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


# ---------------------------------------------------------------------------
#  Core: compute_sentiment_features
# ---------------------------------------------------------------------------

def compute_sentiment_features(
    headlines: Sequence[str] | None = None,
    headline_timestamps: Sequence[Any] | None = None,
    *,
    finnhub_sentiment: Dict[str, Any] | None = None,
    fundamentals: Dict[str, Any] | None = None,
    reference_date: datetime | None = None,
) -> Dict[str, float]:
    """
    Compute 8 sentiment features.

    Args:
        headlines: list of news headline strings (most-recent-first is fine)
        headline_timestamps: parallel list of timestamps (datetime | epoch | ISO str)
        finnhub_sentiment: dict from ``fetcher.get_sentiment_finnhub()``
        fundamentals: dict from ``fetcher.get_fundamentals()``
            (used for ``next_earnings_date``, ``sector``)
        reference_date: "now" for temporal calculations (default: utcnow)

    Returns:
        Dict keyed by SENTIMENT_FEATURE_NAMES with float values (NaN for missing).
    """
    now = reference_date or datetime.now(timezone.utc)
    out: Dict[str, float] = {name: np.nan for name in SENTIMENT_FEATURE_NAMES}

    # ── News headline features ──
    if headlines and len(headlines) > 0:
        scores = score_headlines(list(headlines))
        valid = scores[~np.isnan(scores)]

        if len(valid) > 0:
            out["news_sentiment_mean"] = float(np.mean(valid))
            out["news_sentiment_std"] = float(np.std(valid)) if len(valid) > 1 else 0.0

        # Volume: log₂(1 + n_articles)
        out["news_volume_7d"] = math.log2(1 + len(headlines))

        # Momentum: sentiment of last 3 days vs last 7 days
        if headline_timestamps and len(headline_timestamps) == len(headlines):
            parsed = [_parse_ts(ts) for ts in headline_timestamps]
            recent_scores = []
            all_scores_list = []
            for ts, sc, headline in zip(parsed, scores, headlines):
                if np.isnan(sc):
                    continue
                all_scores_list.append(sc)
                if ts is not None:
                    delta = (now - ts).total_seconds() / 86400
                    if delta <= 3:
                        recent_scores.append(sc)

            if recent_scores and all_scores_list:
                recent_avg = float(np.mean(recent_scores))
                all_avg = float(np.mean(all_scores_list))
                out["news_sentiment_momentum"] = recent_avg - all_avg
        elif len(valid) >= 4:
            # No timestamps → approximate by position (first = most recent)
            n_recent = max(1, len(valid) // 3)
            recent_avg = float(np.mean(valid[:n_recent]))
            all_avg = float(np.mean(valid))
            out["news_sentiment_momentum"] = recent_avg - all_avg

    # ── Finnhub sentiment features ──
    fh = finnhub_sentiment or {}
    buzz = fh.get("buzz_score")
    if buzz is not None:
        try:
            out["finnhub_buzz_score"] = float(buzz)
        except (TypeError, ValueError):
            pass

    bullish = fh.get("sentiment_score")
    if bullish is not None:
        try:
            out["finnhub_bullish_pct"] = float(bullish)
        except (TypeError, ValueError):
            pass

    sector_avg = fh.get("sector_avg_sentiment")
    if sector_avg is not None and out["finnhub_bullish_pct"] is not np.nan:
        try:
            out["sentiment_vs_sector"] = float(out["finnhub_bullish_pct"]) - float(sector_avg)
        except (TypeError, ValueError):
            pass
    elif out.get("news_sentiment_mean") is not np.nan and sector_avg is not None:
        # Fallback: use news mean vs sector
        try:
            out["sentiment_vs_sector"] = float(out["news_sentiment_mean"]) - float(sector_avg)
        except (TypeError, ValueError):
            pass

    # ── Earnings proximity ──
    fund = fundamentals or {}
    next_earnings = fund.get("next_earnings_date")
    if next_earnings:
        try:
            if isinstance(next_earnings, str):
                from dateutil.parser import parse as dateparse
                earnings_dt = dateparse(next_earnings)
            elif isinstance(next_earnings, datetime):
                earnings_dt = next_earnings
            else:
                earnings_dt = None

            if earnings_dt is not None:
                if earnings_dt.tzinfo is None:
                    earnings_dt = earnings_dt.replace(tzinfo=timezone.utc)
                days_to_earnings = (earnings_dt - now).total_seconds() / 86400
                if days_to_earnings < 0:
                    days_to_earnings = 0  # earnings already passed → imminent risk
                # Higher value = closer to earnings
                out["earnings_proximity"] = -math.log2(1 + days_to_earnings)
        except Exception:
            pass

    return out
