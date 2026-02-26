# Phase 6 — Sentiment Feature Layer

**Status:** ✅ Complete  
**Date:** 2026-02-21  
**Scope:** FinBERT / VADER news scoring + Finnhub sentiment + earnings proximity → 8 new ML features

---

## Overview

Phase 6 extends the ML feature vector from **37 to 45 columns** by adding 8 sentiment features derived from three data sources:

1. **News headlines** — Scored via ProsusAI/FinBERT (or VADER as a lightweight fallback)
2. **Finnhub sentiment API** — Buzz score and bullish percentage
3. **Earnings calendar** — Proximity to next earnings event

All features are numerical and follow the same integration pattern established in Phase 5: NaN for missing data, imputer-safe, backward-compatible with older models.

---

## New Features (8)

| # | Name | Range | Source | Description |
|---|------|-------|--------|-------------|
| 37 | `news_sentiment_mean` | [−1, +1] | Headlines | Average FinBERT/VADER sentiment score |
| 38 | `news_sentiment_std` | [0, ∞) | Headlines | Dispersion of headline scores (consensus vs. disagreement) |
| 39 | `news_volume_7d` | [0, ∞) | Headlines | log₂(1 + article count) — information flow intensity |
| 40 | `news_sentiment_momentum` | [−2, +2] | Headlines | Last-3-day avg − last-7-day avg (sentiment acceleration) |
| 41 | `finnhub_buzz_score` | [0, 1] | Finnhub | Article activity / buzz metric |
| 42 | `finnhub_bullish_pct` | [0, 1] | Finnhub | Bullish sentiment percentage |
| 43 | `earnings_proximity` | (−∞, 0] | Fundamentals | −log₂(1 + days to earnings); higher = closer |
| 44 | `sentiment_vs_sector` | [−1, +1] | Finnhub / Headlines | Stock sentiment minus sector average |

---

## Architecture

### Headline Scoring Pipeline

```
                     ┌─────────────┐
Headlines ──────────►│  FinBERT    │──► scores (−1 to +1)
  (list[str])        │ (lazy load) │
                     └──────┬──────┘
                            │ fallback
                     ┌──────▼──────┐
                     │   VADER     │──► compound score (−1 to +1)
                     │ (nltk)      │
                     └──────┬──────┘
                            │ fallback
                         NaN (graceful)
```

- **FinBERT** (`ProsusAI/finbert`) is loaded lazily on first use
- **VADER** (`nltk.sentiment.vader`) is the lightweight fallback
- If neither is available, headline features are NaN — the model still works via imputer

### Feature Computation Flow

```
                    ┌──────────────────────────────┐
                    │   compute_sentiment_features │
                    │                              │
  headlines ───────►│  ① score_headlines()         │──► news_sentiment_mean
  timestamps ──────►│  ② temporal binning          │──► news_sentiment_std
                    │  ③ volume counting           │──► news_volume_7d
                    │  ④ momentum calculation      │──► news_sentiment_momentum
  finnhub_sent ────►│  ⑤ buzz/bullish extraction   │──► finnhub_buzz_score
                    │  ⑥ sector comparison         │──► finnhub_bullish_pct
  fundamentals ────►│  ⑦ earnings date parsing     │──► earnings_proximity
                    │  ⑧ relative sentiment        │──► sentiment_vs_sector
                    └──────────────────────────────┘
```

---

## Integration Points

### Feature Engineering (`feature_engineering.py`)

- `FEATURE_NAMES` expanded: 20 (base) + 17 (Phase-5) + **8 (Phase-6)** = **45**
- `extract_features()` gains a `sentiment` keyword argument
- `extract_features_batch()` handles new columns from CSV (backward-compatible)

### Predictor (`predictor.py`)

`SignalPredictor.predict()` gains three new optional parameters:

```python
def predict(
    self,
    indicators, fundamentals, quote,
    *,
    closes=None, highs=None, lows=None, volumes=None,
    # ──── Phase-6 ────
    headlines=None,              # list[str]
    headline_timestamps=None,    # list[datetime | epoch | ISO]
    finnhub_sentiment=None,      # dict from fetcher.get_sentiment_finnhub()
) -> dict:
```

### Cross-Sectional Module (`cross_sectional.py`)

`ALL_NEW_FEATURE_NAMES` now includes `SENTIMENT_FEATURE_NAMES` (25 total vs. 17).

### Backward Compatibility

- Models trained on 20 features (v1) or 37 features (v2) are **fully supported**
- Feature vector is sliced to `model_n_features` at prediction time
- Sentiment slots are NaN if no data is provided → imputer fills them

---

## Dependencies

| Package | Status | Purpose |
|---------|--------|---------|
| `transformers` + `torch` | **Optional** | FinBERT inference (gold standard) |
| `nltk` | **Optional** | VADER fallback (lightweight) |
| `python-dateutil` | **Optional** | Timestamp parsing for headline timestamps |

No new **required** dependencies. The system gracefully degrades:
- With both → FinBERT sentiment + full features
- With VADER only → VADER sentiment + full features
- With neither → NaN sentiment slots (model uses imputer defaults)

---

## Files Modified / Created

### New
- `src/training/sentiment.py` — Sentiment feature engine (8 features, FinBERT/VADER scoring)
- `tests/test_sentiment.py` — 21 focused tests for sentiment features
- `docs/PHASE6_COMPLETION.md` — This document

### Modified
- `src/training/cross_sectional.py` — Added `SENTIMENT_FEATURE_NAMES` import, extended `ALL_NEW_FEATURE_NAMES`
- `src/training/feature_engineering.py` — Updated docstring (37→45), added `sentiment` kwarg to `extract_features()`
- `src/training/predictor.py` — Added `headlines`, `headline_timestamps`, `finnhub_sentiment` params to `predict()`
- `src/agents/fetcher.py` — `get_full_analysis_data()` now fetches `finnhub_sentiment` in parallel
- `src/agents/analyzer.py` — `generate_algo_prediction()` extracts headlines + passes Phase-6 data to `predictor.predict()`
- `main.py` — Passes `finnhub_sentiment` through from fetcher → analyzer; preserves `published_at` in news dicts
- `src/training/dataset_builder.py` — Updated docstrings and comments for 45-feature vector
- `tests/test_phase5_integration.py` — Updated vector shape assertion (37→45)
- `tests/test_cross_sectional.py` — Updated feature count assertions (37→45, 17→25)

---

## Test Results

```
113 passed, 0 failed, 9 warnings (pydantic deprecation only)
```

### Test Coverage (21 new tests)

| Test Class | Tests | Focus |
|------------|-------|-------|
| `TestSentimentFeatureNames` | 3 | Feature count, vector size, name presence |
| `TestHeadlineScoring` | 4 | Batch scoring, empty list, missing scorers |
| `TestComputeSentimentFeatures` | 10 | All-NaN default, headline features, momentum with/without timestamps, Finnhub, earnings proximity, full integration |
| `TestSentimentIntegration` | 4 | extract_features with/without sentiment, partial dicts, backward compat |

---

## Known Limitations

1. **FinBERT/VADER not pre-installed** — Users need `pip install transformers torch` or `pip install nltk` to activate headline scoring. Without them, sentiment features are NaN.
2. **Historical training data** — The dataset builder doesn't fetch news for historical days; sentiment slots are NaN in training CSVs. Future work can integrate a news archive API.
3. **Finnhub rate limits** — The Finnhub sentiment API has rate limits; the fetcher already handles retries.
4. **Sector sentiment** — `sentiment_vs_sector` requires `sector_avg_sentiment` from Finnhub; falls back to NaN if unavailable.

---

## Next Steps

- ~~Wire news fetching into live inference pipeline~~ ✅ Done
- ~~Fetch Finnhub sentiment in parallel~~ ✅ Done
- **Integrate historical news API** into dataset builder for richer training data
- **Add VADER to requirements.txt** — Lightweight enough to be a default dependency
- **Phase 7** — Continue with the next upgrade roadmap item
