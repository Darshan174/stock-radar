"""
Stock Data Fetcher for Stock Radar.
Fetches price data, technical indicators, and news from multiple sources.
"""

import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from enum import Enum

import yfinance as yf
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Twelve Data - Primary source for price data (30+ years history)
try:
    from twelvedata import TDClient
    TWELVEDATA_AVAILABLE = True
except ImportError:
    TWELVEDATA_AVAILABLE = False

from agents.usage_tracker import get_tracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class Timeframe(str, Enum):
    """Supported timeframes for price data."""
    MINUTE_1 = "1m"
    MINUTE_5 = "5m"
    MINUTE_15 = "15m"
    HOUR_1 = "1h"
    DAY_1 = "1d"
    WEEK_1 = "1w"


@dataclass
class PriceData:
    """OHLCV price data."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str


@dataclass
class StockQuote:
    """Current stock quote."""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    avg_volume: int
    high: float
    low: float
    open: float
    prev_close: float
    market_cap: Optional[int]
    pe_ratio: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    timestamp: datetime


@dataclass
class NewsItem:
    """News article."""
    headline: str
    summary: Optional[str]
    source: str
    url: str
    published_at: datetime
    related_symbols: List[str]


class StockFetcher:
    """
    Fetches stock data from multiple sources.
    Primary: Twelve Data (30+ years history, real-time, 800 req/day free)
    Fallback: Yahoo Finance (free, no API key)
    Secondary: Finnhub, Alpha Vantage (for additional data)
    """

    def __init__(
        self,
        finnhub_key: Optional[str] = None,
        alpha_vantage_key: Optional[str] = None,
        twelve_data_key: Optional[str] = None
    ):
        """
        Initialize stock fetcher.

        Args:
            finnhub_key: Finnhub API key for news/sentiment
            alpha_vantage_key: Alpha Vantage API key for technical indicators
            twelve_data_key: Twelve Data API key for price data (primary)
        """
        self.finnhub_key = finnhub_key or os.getenv("FINNHUB_API_KEY")
        self.alpha_vantage_key = alpha_vantage_key or os.getenv("ALPHA_VANTAGE_API_KEY")
        self.twelve_data_key = twelve_data_key or os.getenv("TWELVE_DATA_API_KEY")
        
        # Initialize Twelve Data client if available
        self.td_client = None
        if TWELVEDATA_AVAILABLE and self.twelve_data_key:
            self.td_client = TDClient(apikey=self.twelve_data_key)
            logger.info("Twelve Data client initialized (primary source)")
        else:
            logger.info("Twelve Data not available, using Yahoo Finance as primary")
        
        logger.info("StockFetcher initialized")

    # -------------------------------------------------------------------------
    # Twelve Data (Primary - 30+ years history, 800 req/day free)
    # -------------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Optional[StockQuote]:
        """
        Get current stock quote. Uses Twelve Data if available, falls back to Yahoo Finance.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS', 'AAPL')

        Returns:
            StockQuote with current price and stats
        """
        # Try Twelve Data first
        if self.td_client:
            quote = self._get_quote_twelvedata(symbol)
            if quote:
                return quote
            logger.warning(f"Twelve Data failed for {symbol}, falling back to yfinance")
        
        # Fallback to Yahoo Finance
        return self._get_quote_yfinance(symbol)

    def _get_quote_twelvedata(self, symbol: str) -> Optional[StockQuote]:
        """Get quote from Twelve Data."""
        try:
            # Clean symbol for Twelve Data (remove .NS, .BO suffixes for quote, add exchange)
            clean_symbol = symbol.replace(".NS", "").replace(".BO", "")
            
            # Get real-time quote
            quote_data = self.td_client.quote(symbol=clean_symbol).as_json()
            
            if not quote_data or "close" not in quote_data:
                return None
            
            price = float(quote_data.get("close", 0))
            prev_close = float(quote_data.get("previous_close", price))
            change = price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            get_tracker().track("twelvedata", count=1)
            
            return StockQuote(
                symbol=symbol,
                price=price,
                change=round(change, 2),
                change_percent=round(change_pct, 2),
                volume=int(quote_data.get("volume", 0)),
                avg_volume=int(quote_data.get("average_volume", 0)) if quote_data.get("average_volume") else 0,
                high=float(quote_data.get("high", 0)),
                low=float(quote_data.get("low", 0)),
                open=float(quote_data.get("open", 0)),
                prev_close=prev_close,
                market_cap=None,  # Not available in basic quote
                pe_ratio=None,
                fifty_two_week_high=float(quote_data.get("fifty_two_week", {}).get("high", 0)) if quote_data.get("fifty_two_week") else None,
                fifty_two_week_low=float(quote_data.get("fifty_two_week", {}).get("low", 0)) if quote_data.get("fifty_two_week") else None,
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Twelve Data quote error for {symbol}: {e}")
            return None

    def _get_quote_yfinance(self, symbol: str) -> Optional[StockQuote]:
        """Get quote from Yahoo Finance (fallback)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info or "regularMarketPrice" not in info:
                logger.warning(f"No data found for {symbol}")
                return None

            return StockQuote(
                symbol=symbol,
                price=info.get("regularMarketPrice", 0),
                change=info.get("regularMarketChange", 0),
                change_percent=info.get("regularMarketChangePercent", 0),
                volume=info.get("regularMarketVolume", 0),
                avg_volume=info.get("averageVolume", 0),
                high=info.get("regularMarketDayHigh", 0),
                low=info.get("regularMarketDayLow", 0),
                open=info.get("regularMarketOpen", 0),
                prev_close=info.get("regularMarketPreviousClose", 0),
                market_cap=info.get("marketCap"),
                pe_ratio=info.get("trailingPE"),
                fifty_two_week_high=info.get("fiftyTwoWeekHigh"),
                fifty_two_week_low=info.get("fiftyTwoWeekLow"),
                timestamp=datetime.now(timezone.utc)
            )
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return None

    def get_price_history(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> List[PriceData]:
        """
        Get historical price data. Uses Twelve Data (30+ years) with yfinance fallback.

        Args:
            symbol: Stock symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', '20y', 'max')
            interval: Data interval ('1min', '5min', '15min', '1h', '1day', '1week')

        Returns:
            List of PriceData objects
        """
        # Try Twelve Data first
        if self.td_client:
            prices = self._get_price_history_twelvedata(symbol, period, interval)
            if prices:
                return prices
            logger.warning(f"Twelve Data history failed for {symbol}, falling back to yfinance")
        
        # Fallback to Yahoo Finance
        return self._get_price_history_yfinance(symbol, period, interval)

    def _get_price_history_twelvedata(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> List[PriceData]:
        """Get historical price data from Twelve Data (supports 30+ years)."""
        try:
            # Clean symbol - for Indian stocks, add exchange suffix
            if symbol.endswith(".NS"):
                clean_symbol = symbol.replace(".NS", "") + ":NSE"
            elif symbol.endswith(".BO"):
                clean_symbol = symbol.replace(".BO", "") + ":BSE"
            else:
                clean_symbol = symbol
            
            # Map period to timedelta for date-based fetching
            period_days = {
                "1d": 1,
                "5d": 5,
                "1mo": 30,
                "3mo": 90,
                "6mo": 180,
                "1y": 365,
                "2y": 730,
                "5y": 1825,
                "10y": 3650,
                "20y": 7300,
                "max": 10000
            }
            days = period_days.get(period, 365)
            
            # Calculate start and end dates
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Map interval format
            interval_map = {
                "1m": "1min",
                "5m": "5min",
                "15m": "15min",
                "1h": "1h",
                "1d": "1day",
                "1wk": "1week"
            }
            td_interval = interval_map.get(interval, "1day")
            
            # Fetch time series data using date range instead of outputsize
            ts = self.td_client.time_series(
                symbol=clean_symbol,
                interval=td_interval,
                start_date=start_date.strftime("%Y-%m-%d"),
                end_date=end_date.strftime("%Y-%m-%d")
            )
            df = ts.as_pandas()
            
            if df is None or df.empty:
                return []
            
            # Reverse to get oldest first
            df = df.iloc[::-1]
            
            prices = []
            for idx, row in df.iterrows():
                # Handle both datetime index and string index
                if isinstance(idx, str):
                    ts_dt = datetime.fromisoformat(idx.replace("Z", "+00:00"))
                else:
                    ts_dt = idx.to_pydatetime()
                
                # Ensure timezone aware
                if ts_dt.tzinfo is None:
                    ts_dt = ts_dt.replace(tzinfo=timezone.utc)
                
                prices.append(PriceData(
                    timestamp=ts_dt,
                    open=round(float(row.get("open", 0)), 2),
                    high=round(float(row.get("high", 0)), 2),
                    low=round(float(row.get("low", 0)), 2),
                    close=round(float(row.get("close", 0)), 2),
                    volume=int(row.get("volume", 0)) if row.get("volume") else 0,
                    timeframe=interval
                ))
            
            get_tracker().track("twelvedata", count=1)
            logger.info(f"Twelve Data: Fetched {len(prices)} price records for {symbol} ({period})")
            return prices
            
        except Exception as e:
            logger.error(f"Twelve Data history error for {symbol}: {e}")
            return []

    def _get_price_history_yfinance(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> List[PriceData]:
        """Get historical price data from Yahoo Finance (fallback)."""
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period, interval=interval)

            if hist.empty:
                logger.warning(f"No history found for {symbol}")
                return []

            prices = []
            for idx, row in hist.iterrows():
                prices.append(PriceData(
                    timestamp=idx.to_pydatetime(),
                    open=round(row["Open"], 2),
                    high=round(row["High"], 2),
                    low=round(row["Low"], 2),
                    close=round(row["Close"], 2),
                    volume=int(row["Volume"]),
                    timeframe=interval
                ))

            logger.info(f"yfinance: Fetched {len(prices)} price records for {symbol}")
            return prices

        except Exception as e:
            logger.error(f"Error fetching history for {symbol}: {e}")
            return []

    def get_fundamentals(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get fundamental data for long-term analysis.

        Args:
            symbol: Stock symbol

        Returns:
            Dictionary with fundamental metrics
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            if not info:
                return None

            # Get earnings dates
            try:
                calendar = ticker.calendar
                earnings_date = None
                if calendar is not None and not calendar.empty:
                    if 'Earnings Date' in calendar.index:
                        earnings_date = str(calendar.loc['Earnings Date'].iloc[0])
            except:
                earnings_date = None

            return {
                "symbol": symbol,
                "name": info.get("longName"),
                "short_name": info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                
                # Company Info
                "website": info.get("website"),
                "headquarters_city": info.get("city"),
                "headquarters_country": info.get("country"),
                "employees": info.get("fullTimeEmployees"),
                "description": info.get("longBusinessSummary"),
                "logo_url": info.get("logo_url"),
                
                # Valuation
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "pb_ratio": info.get("priceToBook"),
                "ps_ratio": info.get("priceToSalesTrailing12Months"),
                
                # Basic EPS & Revenue
                "eps_ttm": info.get("trailingEps"),
                "eps_forward": info.get("forwardEps"),
                "revenue_ttm": info.get("totalRevenue"),
                "gross_profit": info.get("grossProfits"),
                "ebitda": info.get("ebitda"),
                "net_income": info.get("netIncomeToCommon"),
                
                # Shares & Float
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
                "shares_short": info.get("sharesShort"),
                "short_ratio": info.get("shortRatio"),
                "short_percent_of_float": info.get("shortPercentOfFloat"),
                "insider_ownership": info.get("heldPercentInsiders"),
                "institutional_ownership": info.get("heldPercentInstitutions"),
                
                # Risk & Volatility
                "beta": info.get("beta"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "50_day_avg": info.get("fiftyDayAverage"),
                "200_day_avg": info.get("twoHundredDayAverage"),
                
                # Profitability
                "profit_margin": info.get("profitMargins"),
                "operating_margin": info.get("operatingMargins"),
                "gross_margin": info.get("grossMargins"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                
                # Growth
                "revenue_growth": info.get("revenueGrowth"),
                "earnings_growth": info.get("earningsGrowth"),
                "quarterly_revenue_growth": info.get("revenueQuarterlyGrowth"),
                "quarterly_earnings_growth": info.get("earningsQuarterlyGrowth"),
                
                # Dividends
                "dividend_yield": info.get("dividendYield"),
                "dividend_rate": info.get("dividendRate"),
                "payout_ratio": info.get("payoutRatio"),
                "ex_dividend_date": info.get("exDividendDate"),
                
                # Financial health
                "current_ratio": info.get("currentRatio"),
                "quick_ratio": info.get("quickRatio"),
                "debt_to_equity": info.get("debtToEquity"),
                "total_cash": info.get("totalCash"),
                "total_debt": info.get("totalDebt"),
                "free_cash_flow": info.get("freeCashflow"),
                "operating_cash_flow": info.get("operatingCashflow"),
                
                # Analyst data
                "target_high": info.get("targetHighPrice"),
                "target_low": info.get("targetLowPrice"),
                "target_mean": info.get("targetMeanPrice"),
                "target_median": info.get("targetMedianPrice"),
                "recommendation": info.get("recommendationKey"),
                "recommendation_mean": info.get("recommendationMean"),
                "analyst_count": info.get("numberOfAnalystOpinions"),
                
                # Earnings
                "next_earnings_date": earnings_date,
                
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching fundamentals for {symbol}: {e}")
            return None

    def get_news_yahoo(self, symbol: str) -> List[NewsItem]:
        """Get news from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            news = ticker.news or []

            items = []
            for article in news[:10]:  # Limit to 10 articles
                items.append(NewsItem(
                    headline=article.get("title", ""),
                    summary=None,  # Yahoo doesn't provide summary
                    source=article.get("publisher", "Yahoo Finance"),
                    url=article.get("link", ""),
                    published_at=datetime.fromtimestamp(
                        article.get("providerPublishTime", 0),
                        tz=timezone.utc
                    ),
                    related_symbols=[symbol]
                ))

            logger.info(f"Fetched {len(items)} news articles for {symbol}")
            return items

        except Exception as e:
            logger.error(f"Error fetching Yahoo news for {symbol}: {e}")
            return []

    # -------------------------------------------------------------------------
    # Finnhub (Secondary - News & Sentiment)
    # -------------------------------------------------------------------------

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
    def get_news_finnhub(
        self,
        symbol: str,
        from_date: Optional[datetime] = None,
        to_date: Optional[datetime] = None
    ) -> List[NewsItem]:
        """
        Get news from Finnhub API.

        Args:
            symbol: Stock symbol (without exchange suffix for US stocks)
            from_date: Start date for news
            to_date: End date for news

        Returns:
            List of NewsItem objects
        """
        if not self.finnhub_key:
            logger.warning("Finnhub API key not set, skipping")
            return []

        try:
            # Remove exchange suffix for Finnhub
            clean_symbol = symbol.split(".")[0]

            from_date = from_date or datetime.now(timezone.utc) - timedelta(days=7)
            to_date = to_date or datetime.now(timezone.utc)

            url = "https://finnhub.io/api/v1/company-news"
            params = {
                "symbol": clean_symbol,
                "from": from_date.strftime("%Y-%m-%d"),
                "to": to_date.strftime("%Y-%m-%d"),
                "token": self.finnhub_key
            }

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            articles = response.json()

            items = []
            for article in articles[:15]:  # Limit to 15
                items.append(NewsItem(
                    headline=article.get("headline", ""),
                    summary=article.get("summary"),
                    source=article.get("source", "Finnhub"),
                    url=article.get("url", ""),
                    published_at=datetime.fromtimestamp(
                        article.get("datetime", 0),
                        tz=timezone.utc
                    ),
                    related_symbols=article.get("related", [symbol])
                ))

            logger.info(f"Fetched {len(items)} Finnhub news for {symbol}")
            get_tracker().track("finnhub", count=1)  # Track API call
            return items

        except Exception as e:
            logger.error(f"Error fetching Finnhub news for {symbol}: {e}")
            return []

    def get_sentiment_finnhub(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get social sentiment from Finnhub."""
        if not self.finnhub_key:
            return None

        try:
            clean_symbol = symbol.split(".")[0]
            url = "https://finnhub.io/api/v1/news-sentiment"
            params = {"symbol": clean_symbol, "token": self.finnhub_key}

            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            return {
                "symbol": symbol,
                "buzz_score": data.get("buzz", {}).get("buzz"),
                "articles_this_week": data.get("buzz", {}).get("articlesInLastWeek"),
                "sentiment_score": data.get("sentiment", {}).get("bullishPercent"),
                "sector_avg_sentiment": data.get("sectorAverageBullishPercent"),
                "fetched_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error fetching Finnhub sentiment for {symbol}: {e}")
            return None

    # -------------------------------------------------------------------------
    # Technical Indicators (Calculated)
    # -------------------------------------------------------------------------

    def calculate_indicators(self, prices: List[PriceData]) -> Dict[str, Any]:
        """
        Calculate technical indicators from price data.

        Args:
            prices: List of PriceData (oldest first)

        Returns:
            Dictionary with calculated indicators
        """
        if len(prices) < 20:
            logger.warning("Not enough data for indicator calculation")
            return {}

        closes = [p.close for p in prices]
        highs = [p.high for p in prices]
        lows = [p.low for p in prices]
        volumes = [p.volume for p in prices]

        try:
            # Simple Moving Averages
            sma_20 = sum(closes[-20:]) / 20 if len(closes) >= 20 else None
            sma_50 = sum(closes[-50:]) / 50 if len(closes) >= 50 else None
            sma_200 = sum(closes[-200:]) / 200 if len(closes) >= 200 else None

            # RSI (14-period)
            rsi = self._calculate_rsi(closes, 14)

            # MACD
            macd, signal, histogram = self._calculate_macd(closes)

            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = self._calculate_bollinger(closes, 20)

            # ATR (14-period) - for volatility measurement
            atr = self._calculate_atr(highs, lows, closes, 14)

            # VWAP - for intraday institutional activity
            vwap = self._calculate_vwap(highs, lows, closes, volumes)

            # ADX (14-period) - for trend strength
            adx_data = self._calculate_adx(highs, lows, closes, 14)

            # Volume SMA
            vol_sma = sum(volumes[-20:]) / 20 if len(volumes) >= 20 else None

            # Current price relative to indicators
            current_price = closes[-1]
            price_vs_sma20 = ((current_price - sma_20) / sma_20 * 100) if sma_20 else None
            price_vs_sma50 = ((current_price - sma_50) / sma_50 * 100) if sma_50 else None

            # ATR as percentage of price (volatility %)
            atr_pct = (atr / current_price * 100) if atr and current_price else None

            # Price position relative to VWAP
            price_vs_vwap = ((current_price - vwap) / vwap * 100) if vwap else None

            return {
                "sma_20": round(sma_20, 2) if sma_20 else None,
                "sma_50": round(sma_50, 2) if sma_50 else None,
                "sma_200": round(sma_200, 2) if sma_200 else None,
                "rsi_14": round(rsi, 2) if rsi else None,
                "macd": round(macd, 4) if macd else None,
                "macd_signal": round(signal, 4) if signal else None,
                "macd_histogram": round(histogram, 4) if histogram else None,
                "bollinger_upper": round(bb_upper, 2) if bb_upper else None,
                "bollinger_middle": round(bb_middle, 2) if bb_middle else None,
                "bollinger_lower": round(bb_lower, 2) if bb_lower else None,
                "atr_14": round(atr, 2) if atr else None,
                "atr_pct": round(atr_pct, 2) if atr_pct else None,
                "vwap": round(vwap, 2) if vwap else None,
                "price_vs_vwap_pct": round(price_vs_vwap, 2) if price_vs_vwap else None,
                "adx": round(adx_data["adx"], 2) if adx_data else None,
                "plus_di": round(adx_data["plus_di"], 2) if adx_data else None,
                "minus_di": round(adx_data["minus_di"], 2) if adx_data else None,
                "volume_sma_20": int(vol_sma) if vol_sma else None,
                "price_vs_sma20_pct": round(price_vs_sma20, 2) if price_vs_sma20 else None,
                "price_vs_sma50_pct": round(price_vs_sma50, 2) if price_vs_sma50 else None,
                "calculated_at": datetime.now(timezone.utc).isoformat()
            }

        except Exception as e:
            logger.error(f"Error calculating indicators: {e}")
            return {}

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> Optional[float]:
        """Calculate Relative Strength Index using Wilder's smoothing method."""
        if len(closes) < period + 1:
            return None

        # Calculate price changes
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        # First average: simple average of first 'period' values
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period

        # Apply Wilder's smoothing for remaining values
        # Formula: new_avg = (prev_avg * (period - 1) + current_value) / period
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _calculate_macd(
        self,
        closes: List[float],
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> tuple:
        """Calculate MACD, Signal line, and Histogram using proper EMA calculations."""
        if len(closes) < slow + signal:
            return None, None, None

        # Calculate EMA series for fast and slow periods
        ema_fast_series = self._calculate_ema_series(closes, fast)
        ema_slow_series = self._calculate_ema_series(closes, slow)

        if ema_fast_series is None or ema_slow_series is None:
            return None, None, None

        # MACD line = Fast EMA - Slow EMA (for each point where both exist)
        # Align series - slow EMA starts later, so we need to align them
        offset = slow - fast
        macd_series = []
        for i in range(len(ema_slow_series)):
            macd_value = ema_fast_series[i + offset] - ema_slow_series[i]
            macd_series.append(macd_value)

        if len(macd_series) < signal:
            return None, None, None

        # Signal line = 9-period EMA of MACD line
        signal_series = self._calculate_ema_series(macd_series, signal)

        if signal_series is None or len(signal_series) == 0:
            return None, None, None

        # Get the most recent values
        macd_line = macd_series[-1]
        signal_line = signal_series[-1]
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def _calculate_ema_series(self, data: List[float], period: int) -> Optional[List[float]]:
        """Calculate EMA series for all points after the initial period."""
        if len(data) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period  # Start with SMA
        ema_series = [ema]

        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
            ema_series.append(ema)

        return ema_series

    def _calculate_ema(self, data: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return None

        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period  # Start with SMA

        for price in data[period:]:
            ema = (price - ema) * multiplier + ema

        return ema

    def _calculate_bollinger(
        self,
        closes: List[float],
        period: int = 20,
        std_dev: int = 2
    ) -> tuple:
        """Calculate Bollinger Bands."""
        if len(closes) < period:
            return None, None, None

        recent = closes[-period:]
        middle = sum(recent) / period

        # Calculate standard deviation
        variance = sum((x - middle) ** 2 for x in recent) / period
        std = variance ** 0.5

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return upper, middle, lower

    def _calculate_atr(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> Optional[float]:
        """Calculate Average True Range (ATR) using Wilder's smoothing."""
        if len(closes) < period + 1:
            return None

        # Calculate True Range for each period
        true_ranges = []
        for i in range(1, len(closes)):
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i - 1])
            low_close_prev = abs(lows[i] - closes[i - 1])
            true_range = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(true_range)

        if len(true_ranges) < period:
            return None

        # First ATR: simple average of first 'period' true ranges
        atr = sum(true_ranges[:period]) / period

        # Apply Wilder's smoothing for remaining values
        for i in range(period, len(true_ranges)):
            atr = (atr * (period - 1) + true_ranges[i]) / period

        return atr

    def _calculate_vwap(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float]
    ) -> Optional[float]:
        """Calculate Volume Weighted Average Price (VWAP)."""
        if len(closes) < 1 or len(volumes) < 1:
            return None

        # Typical price = (High + Low + Close) / 3
        typical_prices = [(h + l + c) / 3 for h, l, c in zip(highs, lows, closes)]

        # VWAP = Sum(Typical Price * Volume) / Sum(Volume)
        cumulative_tp_vol = sum(tp * v for tp, v in zip(typical_prices, volumes))
        cumulative_vol = sum(volumes)

        if cumulative_vol == 0:
            return None

        return cumulative_tp_vol / cumulative_vol

    def _calculate_adx(
        self,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        period: int = 14
    ) -> Optional[dict]:
        """Calculate Average Directional Index (ADX) with +DI and -DI."""
        if len(closes) < period * 2:
            return None

        # Calculate True Range, +DM, -DM
        true_ranges = []
        plus_dm = []
        minus_dm = []

        for i in range(1, len(closes)):
            # True Range
            high_low = highs[i] - lows[i]
            high_close_prev = abs(highs[i] - closes[i - 1])
            low_close_prev = abs(lows[i] - closes[i - 1])
            tr = max(high_low, high_close_prev, low_close_prev)
            true_ranges.append(tr)

            # Directional Movement
            up_move = highs[i] - highs[i - 1]
            down_move = lows[i - 1] - lows[i]

            if up_move > down_move and up_move > 0:
                plus_dm.append(up_move)
            else:
                plus_dm.append(0)

            if down_move > up_move and down_move > 0:
                minus_dm.append(down_move)
            else:
                minus_dm.append(0)

        if len(true_ranges) < period:
            return None

        # Smooth TR, +DM, -DM using Wilder's smoothing
        smoothed_tr = sum(true_ranges[:period])
        smoothed_plus_dm = sum(plus_dm[:period])
        smoothed_minus_dm = sum(minus_dm[:period])

        dx_values = []

        for i in range(period, len(true_ranges)):
            # Wilder's smoothing: smoothed = prev - (prev/period) + current
            smoothed_tr = smoothed_tr - (smoothed_tr / period) + true_ranges[i]
            smoothed_plus_dm = smoothed_plus_dm - (smoothed_plus_dm / period) + plus_dm[i]
            smoothed_minus_dm = smoothed_minus_dm - (smoothed_minus_dm / period) + minus_dm[i]

            # Calculate +DI and -DI
            if smoothed_tr > 0:
                plus_di = (smoothed_plus_dm / smoothed_tr) * 100
                minus_di = (smoothed_minus_dm / smoothed_tr) * 100
            else:
                plus_di = 0
                minus_di = 0

            # Calculate DX
            di_sum = plus_di + minus_di
            if di_sum > 0:
                dx = abs(plus_di - minus_di) / di_sum * 100
                dx_values.append(dx)

        if len(dx_values) < period:
            return None

        # ADX = Smoothed average of DX values
        adx = sum(dx_values[:period]) / period
        for i in range(period, len(dx_values)):
            adx = (adx * (period - 1) + dx_values[i]) / period

        # Calculate final +DI and -DI
        if smoothed_tr > 0:
            final_plus_di = (smoothed_plus_dm / smoothed_tr) * 100
            final_minus_di = (smoothed_minus_dm / smoothed_tr) * 100
        else:
            final_plus_di = 0
            final_minus_di = 0

        return {
            "adx": adx,
            "plus_di": final_plus_di,
            "minus_di": final_minus_di
        }

    # -------------------------------------------------------------------------
    # Batch Operations
    # -------------------------------------------------------------------------

    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, StockQuote]:
        """Get quotes for multiple symbols."""
        quotes = {}
        for symbol in symbols:
            quote = self.get_quote(symbol)
            if quote:
                quotes[symbol] = quote
        return quotes

    def get_full_analysis_data(self, symbol: str, period: str = "2y") -> Dict[str, Any]:
        """
        Get all data needed for analysis using parallel fetching.

        Runs independent API calls concurrently via ThreadPoolExecutor,
        reducing total latency from sum(all calls) to max(single call).
        Also checks the Finnhub WebSocket cache for instant quote lookups.

        Args:
            symbol: Stock symbol (e.g., 'RELIANCE.NS', 'AAPL')
            period: Historical data period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max')

        Returns:
            Dictionary with quote, history, indicators, fundamentals, and news
        """
        import time as _time
        fetch_start = _time.time()
        logger.info(f"Fetching full analysis data for {symbol} (period={period}, parallel=True)")

        # Try realtime WebSocket cache for instant quote
        rt_quote = None
        try:
            from agents.realtime import get_realtime_manager
            rt = get_realtime_manager()
            rt_data = rt.get_latest(symbol)
            if rt_data and rt_data["age_ms"] < 60_000:  # fresh within 60s
                logger.info(f"  Realtime cache hit for {symbol}: ${rt_data['price']:.2f} (age={rt_data['age_ms']}ms)")
                rt_quote = rt_data
        except Exception:
            pass

        # Run independent fetches in parallel
        results: Dict[str, Any] = {}

        with ThreadPoolExecutor(max_workers=5, thread_name_prefix="fetch") as pool:
            futures = {}

            # Only fetch quote via API if no realtime cache hit
            if not rt_quote:
                futures["quote"] = pool.submit(self.get_quote, symbol)

            futures["history"] = pool.submit(self.get_price_history, symbol, period, "1d")
            futures["fundamentals"] = pool.submit(self.get_fundamentals, symbol)
            futures["news_yahoo"] = pool.submit(self.get_news_yahoo, symbol)

            if self.finnhub_key:
                futures["news_finnhub"] = pool.submit(self.get_news_finnhub, symbol)
                futures["finnhub_sentiment"] = pool.submit(
                    self.get_sentiment_finnhub, symbol
                )

            for key, future in futures.items():
                try:
                    results[key] = future.result(timeout=30)
                except Exception as e:
                    logger.warning(f"  Parallel fetch failed for {key}: {e}")
                    results[key] = None if key != "history" else []

        # Use realtime quote or API quote
        quote = results.get("quote")
        if rt_quote and quote:
            # Update API quote with latest realtime price
            quote.price = rt_quote["price"]
        elif rt_quote and not quote:
            # Fallback: build minimal quote from realtime data
            quote = None  # Let caller handle

        history = results.get("history", [])
        indicators = self.calculate_indicators(history) if history else {}

        news_yahoo = results.get("news_yahoo") or []
        news_finnhub = results.get("news_finnhub") or []

        elapsed = _time.time() - fetch_start
        logger.info(f"  Parallel fetch complete for {symbol} in {elapsed:.1f}s")

        return {
            "symbol": symbol,
            "quote": quote,
            "price_history": history,
            "indicators": indicators,
            "fundamentals": results.get("fundamentals"),
            "news": news_yahoo + news_finnhub,
            "finnhub_sentiment": results.get("finnhub_sentiment"),
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }

    # -------------------------------------------------------------------------
    # Social Sentiment (Reddit, Twitter)
    # -------------------------------------------------------------------------

    def get_reddit_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Get Reddit sentiment from ApeWisdom API (free tier).
        Tracks mentions from r/wallstreetbets, r/stocks, r/investing.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict with mentions, rank, sentiment
        """
        try:
            # ApeWisdom public API - no key required
            url = f"https://apewisdom.io/api/v1.0/filter/all-stocks/page/1"
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                
                # Find our symbol in the results
                for stock in results:
                    if stock.get("ticker", "").upper() == symbol.upper().replace(".NS", "").replace(".BO", ""):
                        mentions = stock.get("mentions", 0)
                        upvotes = stock.get("upvotes", 0)
                        rank = stock.get("rank", 0)
                        name = stock.get("name", "")
                        
                        # Calculate sentiment based on rank and mentions
                        sentiment = "neutral"
                        if rank <= 10 and mentions > 50:
                            sentiment = "bullish"
                        elif mentions < 5:
                            sentiment = "neutral"
                            
                        logger.info(f"Reddit sentiment for {symbol}: {mentions} mentions, rank #{rank}")
                        
                        return {
                            "source": "reddit",
                            "mentions": mentions,
                            "upvotes": upvotes,
                            "rank": rank,
                            "sentiment": sentiment,
                            "subreddits": ["wallstreetbets", "stocks", "investing"],
                            "fetched_at": datetime.now(timezone.utc).isoformat()
                        }
                
                # Symbol not found in top results
                logger.info(f"Reddit: {symbol} not in trending stocks")
                return {
                    "source": "reddit",
                    "mentions": 0,
                    "sentiment": "neutral",
                    "fetched_at": datetime.now(timezone.utc).isoformat()
                }
                
        except Exception as e:
            logger.error(f"Error fetching Reddit sentiment for {symbol}: {e}")
            return {"source": "reddit", "error": str(e)}

    def get_social_sentiment(self, symbol: str) -> Dict[str, Any]:
        """
        Aggregate social sentiment from multiple sources.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dict with aggregated social sentiment
        """
        reddit_data = self.get_reddit_sentiment(symbol)
        
        # Aggregate results
        total_mentions = reddit_data.get("mentions", 0)
        
        # Determine overall sentiment
        if reddit_data.get("sentiment") == "bullish":
            overall = "bullish"
        elif reddit_data.get("sentiment") == "bearish":
            overall = "bearish"
        else:
            overall = "neutral"
            
        return {
            "reddit_mentions": reddit_data.get("mentions", 0),
            "reddit_rank": reddit_data.get("rank"),
            "twitter_mentions": None,  # Future: add Twitter API
            "overall_sentiment": overall,
            "total_mentions": total_mentions,
            "trending_topics": [],  # Future: extract trending topics
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }


# =============================================================================
# CLI Testing
# =============================================================================

if __name__ == "__main__":
    import json
    from dotenv import load_dotenv
    load_dotenv()

    fetcher = StockFetcher()

    # Test with Reliance (Indian stock)
    print("\n" + "=" * 60)
    print("Testing StockFetcher with RELIANCE.NS")
    print("=" * 60)

    quote = fetcher.get_quote("RELIANCE.NS")
    if quote:
        print(f"\nQuote: {quote.symbol}")
        print(f"  Price: ₹{quote.price:,.2f}")
        print(f"  Change: {quote.change_percent:+.2f}%")
        print(f"  Volume: {quote.volume:,}")

    history = fetcher.get_price_history("RELIANCE.NS", period="1mo", interval="1d")
    print(f"\nPrice History: {len(history)} records")

    indicators = fetcher.calculate_indicators(history)
    print(f"\nIndicators:")
    print(f"  RSI(14): {indicators.get('rsi_14')}")
    print(f"  SMA(20): ₹{indicators.get('sma_20'):,.2f}" if indicators.get('sma_20') else "  SMA(20): N/A")

    news = fetcher.get_news_yahoo("RELIANCE.NS")
    print(f"\nNews: {len(news)} articles")
    for article in news[:3]:
        print(f"  - {article.headline[:60]}...")
