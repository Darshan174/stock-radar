"""Unit tests for Phase-5 cross-sectional, factor, and microstructure features."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from training.cross_sectional import (
    ALL_NEW_FEATURE_NAMES,
    CROSS_SECTIONAL_FEATURE_NAMES,
    FACTOR_FEATURE_NAMES,
    MICROSTRUCTURE_FEATURE_NAMES,
    compute_cross_sectional_features,
    compute_factor_features,
    compute_microstructure_features,
    _returns_from_closes,
    _zscore,
)
from training.feature_engineering import (
    BASE_FEATURE_NAMES,
    FEATURE_NAMES,
    extract_features,
)


class TestFeatureNamesIntegrity:
    """Ensure the combined feature vector is well-formed."""

    def test_total_feature_count(self):
        assert len(FEATURE_NAMES) == 45

    def test_base_count(self):
        assert len(BASE_FEATURE_NAMES) == 20

    def test_new_feature_count(self):
        assert len(ALL_NEW_FEATURE_NAMES) == 25

    def test_group_sizes(self):
        assert len(CROSS_SECTIONAL_FEATURE_NAMES) == 6
        assert len(FACTOR_FEATURE_NAMES) == 6
        assert len(MICROSTRUCTURE_FEATURE_NAMES) == 5

    def test_no_duplicate_names(self):
        assert len(set(FEATURE_NAMES)) == len(FEATURE_NAMES)

    def test_ordering(self):
        """Base features come first, then XS, factor, micro."""
        assert FEATURE_NAMES[:20] == BASE_FEATURE_NAMES
        assert FEATURE_NAMES[20:26] == CROSS_SECTIONAL_FEATURE_NAMES
        assert FEATURE_NAMES[26:32] == FACTOR_FEATURE_NAMES
        assert FEATURE_NAMES[32:37] == MICROSTRUCTURE_FEATURE_NAMES


class TestFactorFeatures:
    """Test factor-style feature computation."""

    @pytest.fixture
    def price_series(self):
        """300-day synthetic close prices."""
        np.random.seed(42)
        return list(np.cumsum(np.random.randn(300) * 0.02) + 100)

    @pytest.fixture
    def fundamentals(self):
        return {
            "pe_ratio": 25,
            "pb_ratio": 3.5,
            "roe": 0.15,
            "profit_margin": 0.12,
            "debt_to_equity": 45.0,
            "current_ratio": 1.8,
        }

    def test_returns_all_factor_keys(self, price_series, fundamentals):
        result = compute_factor_features(price_series, fundamentals)
        for name in FACTOR_FEATURE_NAMES:
            assert name in result, f"Missing key: {name}"

    def test_momentum_12_1_direction(self, price_series):
        """If price trended up, momentum should be positive- or close thereto."""
        up_prices = list(np.linspace(100, 120, 300))
        result = compute_factor_features(up_prices)
        assert result["momentum_12_1"] > 0
        assert result["momentum_1m"] > 0

    def test_quality_score_range(self, price_series, fundamentals):
        result = compute_factor_features(price_series, fundamentals)
        qs = result["quality_score"]
        assert 0.0 <= qs <= 1.0

    def test_low_vol_factor_range(self, price_series):
        result = compute_factor_features(price_series)
        lvf = result["low_vol_factor"]
        assert 0.0 < lvf <= 1.0

    def test_earnings_yield_inverse_pe(self, price_series, fundamentals):
        result = compute_factor_features(price_series, fundamentals)
        expected = 1.0 / 25.0
        assert abs(result["earnings_yield"] - expected) < 1e-6

    def test_book_to_price_inverse_pb(self, price_series, fundamentals):
        result = compute_factor_features(price_series, fundamentals)
        expected = 1.0 / 3.5
        assert abs(result["book_to_price"] - expected) < 1e-4

    def test_short_series_returns_nan(self):
        """With fewer than 60 bars, most factors should be NaN."""
        result = compute_factor_features(list(range(10, 30)))
        assert np.isnan(result["momentum_12_1"])
        assert np.isnan(result["low_vol_factor"])

    def test_no_fundamentals_still_computes(self, price_series):
        result = compute_factor_features(price_series, None)
        assert not np.isnan(result["momentum_12_1"])
        assert np.isnan(result["earnings_yield"])  # no PE


class TestMicrostructureFeatures:
    """Test microstructure-lite feature computation."""

    @pytest.fixture
    def price_data(self):
        np.random.seed(42)
        n = 100
        c = np.cumsum(np.random.randn(n) * 0.5) + 100
        h = c + np.random.rand(n) * 2
        lo = c - np.random.rand(n) * 2
        v = np.random.randint(100_000, 500_000, size=n).astype(float)
        return list(c), list(h), list(lo), list(v)

    def test_returns_all_micro_keys(self, price_data):
        c, h, lo, v = price_data
        result = compute_microstructure_features(c, h, lo, v)
        for name in MICROSTRUCTURE_FEATURE_NAMES:
            assert name in result, f"Missing key: {name}"

    def test_close_location_value_range(self, price_data):
        c, h, lo, v = price_data
        result = compute_microstructure_features(c, h, lo, v)
        clv = result["close_location_value"]
        if not np.isnan(clv):
            assert 0.0 <= clv <= 1.0

    def test_volume_imbalance_range(self, price_data):
        c, h, lo, v = price_data
        result = compute_microstructure_features(c, h, lo, v)
        vi = result["volume_imbalance"]
        if not np.isnan(vi):
            assert -1.0 <= vi <= 1.0

    def test_vwap_deviation_sign(self):
        """If price is above VWAP, deviation should be positive."""
        # Create a series where close > typical price
        n = 30
        c = [100.0] * n
        c[-1] = 110.0  # last close far above
        h = [101.0] * n
        lo = [99.0] * n
        v = [1_000_000.0] * n
        result = compute_microstructure_features(c, h, lo, v)
        assert result["vwap_deviation_pct"] > 0

    def test_short_series(self):
        """Short series should return mostly NaN."""
        result = compute_microstructure_features([100, 101], [102, 103], [98, 99], [1000, 2000])
        assert np.isnan(result["vwap_deviation_pct"])
        assert np.isnan(result["volume_imbalance"])


class TestCrossSectionalFeatures:
    """Test universe-relative cross-sectional features."""

    @pytest.fixture
    def universe_data(self):
        return {
            "trailing_returns": np.array([5.5, -2.1, 3.3, 0.1, -1.5, 7.2, 4.0]),
            "momentums": np.array([12.0, -5.0, 8.0, 1.0, -3.0, 20.0, 10.0]),
            "vols": np.array([0.2, 0.35, 0.18, 0.25, 0.3, 0.15, 0.22]),
            "vol_ratios": np.array([1.1, 0.8, 1.3, 0.9, 1.0, 1.5, 1.2]),
        }

    def test_returns_all_xs_keys(self, universe_data):
        result = compute_cross_sectional_features(
            symbol="AAPL",
            trailing_return_pct=5.5,
            momentum_12_1=12.0,
            realised_vol=0.2,
            avg_volume_ratio=1.1,
            sector="Technology",
            universe_trailing_returns=universe_data["trailing_returns"],
            universe_momentums=universe_data["momentums"],
            universe_vols=universe_data["vols"],
            universe_volume_ratios=universe_data["vol_ratios"],
        )
        for name in CROSS_SECTIONAL_FEATURE_NAMES:
            assert name in result, f"Missing key: {name}"

    def test_excess_return_calculation(self, universe_data):
        result = compute_cross_sectional_features(
            symbol="AAPL",
            trailing_return_pct=5.5,
            momentum_12_1=12.0,
            realised_vol=0.2,
            avg_volume_ratio=1.1,
            sector="Technology",
            universe_trailing_returns=universe_data["trailing_returns"],
            universe_momentums=universe_data["momentums"],
            universe_vols=universe_data["vols"],
            universe_volume_ratios=universe_data["vol_ratios"],
            benchmark_return_pct=2.0,
        )
        assert abs(result["excess_return_vs_benchmark"] - 3.5) < 1e-6

    def test_relative_strength_with_sector_peers(self, universe_data):
        sector_rets = np.array([5.5, 3.3, 4.0, -1.0])
        result = compute_cross_sectional_features(
            symbol="AAPL",
            trailing_return_pct=5.5,
            momentum_12_1=12.0,
            realised_vol=0.2,
            avg_volume_ratio=1.1,
            sector="Technology",
            universe_trailing_returns=universe_data["trailing_returns"],
            universe_momentums=universe_data["momentums"],
            universe_vols=universe_data["vols"],
            universe_volume_ratios=universe_data["vol_ratios"],
            sector_returns=sector_rets,
        )
        # 5.5 is the max in sector → should be rank ~1.0
        assert result["relative_strength_sector"] == 1.0

    def test_no_sector_peers_neutral(self, universe_data):
        result = compute_cross_sectional_features(
            symbol="AAPL",
            trailing_return_pct=5.5,
            momentum_12_1=12.0,
            realised_vol=0.2,
            avg_volume_ratio=1.1,
            sector="Technology",
            universe_trailing_returns=universe_data["trailing_returns"],
            universe_momentums=universe_data["momentums"],
            universe_vols=universe_data["vols"],
            universe_volume_ratios=universe_data["vol_ratios"],
            sector_returns=None,
        )
        assert result["relative_strength_sector"] == 0.5

    def test_zscore_symmetric(self):
        vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        z3 = _zscore(3.0, vals)
        assert abs(z3) < 1e-6  # 3 is the mean


class TestExtractFeaturesWithPhase5:
    """Test that extract_features integrates Phase-5 features correctly."""

    def test_full_vector_with_all_dicts(self):
        indicators = {"rsi_14": 55, "macd": 1.0, "adx": 25}
        fund = {"pe_ratio": 20, "roe": 0.15}
        quote = {"price": 100, "volume": 1000, "avg_volume": 800}

        xs = {name: 0.5 for name in CROSS_SECTIONAL_FEATURE_NAMES}
        fac = {name: 0.3 for name in FACTOR_FEATURE_NAMES}
        mic = {name: 0.7 for name in MICROSTRUCTURE_FEATURE_NAMES}

        features = extract_features(indicators, fund, quote,
                                    cross_sectional=xs, factor=fac, microstructure=mic)
        assert features.shape == (45,)
        # Check Phase-5 values were placed correctly
        assert features[20] == 0.5  # first XS feature
        assert features[26] == 0.3  # first factor feature
        assert features[32] == 0.7  # first micro feature

    def test_missing_phase5_fills_nan(self):
        """If no Phase-5 dicts are provided, those slots should be NaN."""
        indicators = {"rsi_14": 55}
        features = extract_features(indicators, None, None)
        assert features.shape == (45,)
        # All Phase-5 slots should be NaN
        assert np.isnan(features[20:]).all()

    def test_backward_compatible_base_features(self):
        """First 20 features should work identically to old module."""
        indicators = {"rsi_14": 62.5, "macd": 3.2, "macd_signal": 2.1}
        features = extract_features(indicators, None, None)
        assert features[0] == 62.5
        assert features[1] == 3.2


class TestHelpers:
    """Test utility functions."""

    def test_returns_from_closes(self):
        closes = [100, 102, 101]
        rets = _returns_from_closes(closes)
        assert len(rets) == 2
        assert abs(rets[0] - 0.02) < 1e-10
        assert abs(rets[1] - (-1 / 102)) < 1e-10

    def test_returns_from_closes_empty(self):
        rets = _returns_from_closes([100])
        assert len(rets) == 0
