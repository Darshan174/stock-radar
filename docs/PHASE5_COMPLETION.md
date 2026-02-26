# Phase 5 — Quant Feature Upgrade: Completion Summary

_Last updated: 2026-02-21_

---

## 1. Data Assumptions

| Assumption | Detail |
|---|---|
| **Price source** | Yahoo Finance via `yfinance`, daily OHLCV bars |
| **Minimum history** | 60 trading days per symbol (configurable `min_history`) |
| **Benchmark** | SPY used for excess-return and cross-sectional features |
| **Fundamentals** | Static-per-symbol; sourced once per dataset build via `get_fundamentals()` |
| **Sector classification** | Provider-based (Yahoo Finance `sector` field), cached per build run, with a minimal static fallback map for API failures |
| **Universe** | Defined in `data/training_symbols.txt` (~30 symbols); cross-sectional z-scores computed within this universe on each calendar date |
| **Label horizon** | 5-day forward return (`horizon_days=5`), mapped to 5-class signal via configurable thresholds |
| **Feature count** | 37 total: 20 base (technical + fundamental + sentiment) + 6 cross-sectional + 6 factor-style + 5 microstructure-lite |

### Feature Families

| Family | Count | Features |
|---|:---:|---|
| **Base Technical** | 8 | rsi_14, macd, macd_signal, adx, atr_pct, bb_position, price_vs_sma20_pct, price_vs_sma50_pct |
| **Base Volatility** | 4 | stoch_k, stoch_d, obv_slope, macd_histogram |
| **Base Valuation** | 4 | pe_ratio, pb_ratio, dividend_yield, market_cap_log |
| **Base Quality** | 4 | roe, profit_margin, volume_ratio, sentiment_score |
| **Cross-Sectional** | 6 | xs_return_zscore, xs_momentum_zscore, xs_volatility_zscore, xs_volume_zscore, relative_strength_sector, excess_return_vs_benchmark |
| **Factor-Style** | 6 | momentum_12_1, momentum_1m, quality_score, low_vol_factor, earnings_yield, book_to_price |
| **Microstructure** | 5 | vwap_deviation_pct, volume_imbalance, volatility_compression, high_low_range_pct, close_location_value |

---

## 2. Leakage Guards

### 2.1 Train/Test Split — Purged Walk-Forward

- **Method:** `_purged_walk_forward_split()` in `train.py`
- **Mechanism:** Data is sorted by calendar date. The first ~80 % of dates form the training set; remaining dates form the test set.
- **Embargo:** A configurable gap (`embargo_days`, default 5 calendar days) is enforced between the last training date and the first test date. All rows falling within the embargo window are excluded from both sets.
- **Why it matters:** The 5-day forward-return label means that a training row from date _T_ uses information up to _T+5_. Without an embargo, a test row at _T+1_ would share label information with the training set.

### 2.2 Optuna Cross-Validation — TimeSeriesSplit

- **Before Phase 5:** `StratifiedKFold(shuffle=True)` — violates temporal order within the tuning objective.
- **After Phase 5:** `TimeSeriesSplit(n_splits=min(5, N//50))` — folds respect chronological order.
- **Impact:** Prevents the hyperparameter search from selecting parameters that exploit future information.

### 2.3 Cross-Sectional Feature Computation

- **Two-pass approach:**
  1. Pass 1: Compute per-symbol base, factor, and microstructure features.
  2. Pass 2: Compute universe-relative z-scores using only data available on each calendar date.
- **No look-ahead:** Cross-sectional statistics (mean, std) are computed per-date using only the symbols present on that date. No aggregate from future dates leaks into the features.

### 2.4 Feature Engineering — No Future Data

- All technical indicators (`RSI`, `MACD`, `ADX`, etc.) are computed on `history[:i+1]` — the window up to and including the current bar.
- Factor features (momentum, quality, vol) use only the close-price series up to date _i_.
- Microstructure features use only OHLCV data up to date _i_.

---

## 3. Backtest Realism

### 3.1 Transaction Costs

- **Flat cost:** Configurable `transaction_cost_bps` (default 10 bps per turnover unit).
- **Slippage:** Volume-dependent market-impact model:
  `slippage = slippage_bps × √(trade_notional / ADV_notional)`
  - Models the empirical square-root market impact observed in limit-order-book literature.
  - Disabled by default (`slippage_bps=0`); enable for realistic cost estimation.

### 3.2 Liquidity Constraints

- **ADV participation cap:** Positions are capped at `max_adv_participation × avg_daily_volume` (default 1 % ADV).
- **Effect:** Prevents the backtest from allocating unrealistically large positions in illiquid names.
- **Reporting:** The `liquidity_limited_pct` metric shows what fraction of trades were constrained.

### 3.3 Position Sizing

- **Regime-aware sizing:** When `use_position_sizing=True`, the backtest uses `calculate_position_size()` which adjusts position magnitude based on signal confidence, volatility (ATR), and market regime classification.

---

## 4. Model Promotion Criteria (Rollout Gates)

Implemented in `model_registry.promote_if_better()`.

### Gate Checks

A candidate model is promoted to active **only if ALL three criteria pass:**

| Metric | Condition | Default Threshold |
|---|---|:---:|
| **Sharpe** | `candidate.sharpe ≥ incumbent.sharpe + sharpe_min_improvement` | 0.0 |
| **CAGR** | `candidate.cagr ≥ incumbent.cagr + cagr_min_improvement` | 0.0 |
| **Max Drawdown** | `candidate.max_dd ≥ incumbent.max_dd − max_dd_tolerance` | 0.0 |

- If no incumbent exists, the candidate is **auto-promoted**.
- Thresholds are configurable via the `thresholds` argument.
- The function returns a structured verdict with per-metric pass/fail, enabling automated CI/CD pipelines to gate deployments.

### Recommended Production Thresholds

For a risk-sensitive deployment:

```python
promote_if_better(
    candidate_version=3,
    thresholds={
        "sharpe_min_improvement": 0.05,    # must beat by ≥0.05
        "cagr_min_improvement": 0.005,      # must beat by ≥0.5%
        "max_dd_tolerance": 0.02,           # may be ≤2% worse on drawdown
    },
)
```

---

## 5. Feature Health Monitoring

Implemented in `feature_health.py`.

### Checks

| Check | Thresholds | Action |
|---|---|---|
| **NaN Rate** | WARN >25 %, CRITICAL >50 % | Indicates broken data pipeline or missing provider data |
| **Distribution Drift** | WARN >2σ shift, CRITICAL >4σ shift | Indicates regime change or data-source alteration |
| **Out-of-Range** | Fraction outside training [P5, P95] | Detects extrapolation risk |

### Usage

```python
from training.feature_health import (
    compute_reference_stats,
    check_feature_health,
    log_feature_health,
)

# At training time: save reference stats with the model
ref_stats = compute_reference_stats(X_train, FEATURE_NAMES)

# At inference time: monitor incoming feature vectors
report = check_feature_health(
    feature_vectors=X_live,
    reference_stats=ref_stats,
    phase5_only=True,  # focus on new 17 features
)
log_feature_health(report)
```

---

## 6. Files Modified / Created in Phase 5

| File | Change |
|---|---|
| `src/training/cross_sectional.py` | **New** — 17 cross-sectional, factor, microstructure features |
| `src/training/feature_health.py` | **New** — NaN rate + drift monitoring |
| `src/training/model_registry.py` | **Extended** — `promote_if_better()` rollout gates |
| `src/training/backtesting.py` | **Extended** — slippage model, liquidity constraints |
| `src/training/dataset_builder.py` | **Extended** — two-pass feature pipeline, provider-based sector resolution |
| `src/training/train.py` | **Extended** — purged walk-forward split, TimeSeriesSplit CV, IC diagnostics |
| `src/training/predictor.py` | **Extended** — backward-compatible feature slicing, Phase-5 feature extraction at inference |
| `src/training/feature_engineering.py` | **Extended** — 37-feature vector builder |
| `src/agents/analyzer.py` | **Extended** — passes OHLCV history to predictor |
| `main.py` | **Extended** — passes `price_history` through the analysis pipeline |
| `tests/test_phase5_integration.py` | **New** — 11 integration tests covering backward compat, E2E wiring, embargo logic, analyzer wiring |
| `tests/test_rollout_health_slippage.py` | **New** — 25 focused tests for rollout gates, feature health, and slippage/liquidity |

---

## 7. Known Limitations & Future Work

1. **Sector classification coverage:** The fallback sector map covers ~30 symbols. Symbols not in Yahoo Finance's sector data and not in the fallback map default to "Unknown", which degrades cross-sectional relative-strength calculation.

2. **Single-benchmark assumption:** Only SPY is used as the benchmark. Multi-benchmark support (e.g., sector ETFs) would improve excess-return accuracy.

3. **Static fundamentals:** Fundamentals are fetched once per dataset build. Quarterly re-fetching would better reflect changing company financials during multi-year training periods.

4. **Slippage calibration:** The square-root impact coefficient (`slippage_bps`) is currently user-supplied. Empirical calibration from live fill data would improve realism.

5. **Feature health in CI/CD:** `check_feature_health()` is now wired into `SignalPredictor.predict()` and will log warnings/critical alerts when NaN rates or drift exceed thresholds. Reference stats are loaded from model metadata (`feature_reference_stats` key). For full CI/CD integration, save `compute_reference_stats()` output into the model's `_meta.json` during training.
