import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from agents.fetcher import StockFetcher


def test_intraday_defaults_to_recent_higher_frequency_bars():
    fetcher = StockFetcher()

    period, interval = fetcher._resolve_analysis_history_config("intraday", "max")

    assert period == "5d"
    assert interval == "15m"


def test_intraday_allows_single_day_high_frequency_bars():
    fetcher = StockFetcher()

    period, interval = fetcher._resolve_analysis_history_config("intraday", "1d")

    assert period == "1d"
    assert interval == "5m"


def test_longterm_defaults_to_weekly_multi_year_context():
    fetcher = StockFetcher()

    period, interval = fetcher._resolve_analysis_history_config("longterm", "1d")

    assert period == "5y"
    assert interval == "1wk"


def test_longterm_preserves_supported_multi_year_periods():
    fetcher = StockFetcher()

    period, interval = fetcher._resolve_analysis_history_config("longterm", "2y")

    assert period == "2y"
    assert interval == "1wk"
