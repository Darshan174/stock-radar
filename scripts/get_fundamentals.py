#!/usr/bin/env python3
"""
Simple script to fetch stock fundamentals and output as JSON.
Usage: python get_fundamentals.py SYMBOL
"""

import sys
import json
import yfinance as yf
from datetime import datetime, timezone


def get_fundamentals(symbol: str) -> dict:
    """Fetch comprehensive fundamentals for a stock."""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        if not info:
            return {"error": "No data found", "symbol": symbol}

        # Get earnings dates
        earnings_date = None
        try:
            calendar = ticker.calendar
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar.index:
                    earnings_date = str(calendar.loc['Earnings Date'].iloc[0])
        except:
            pass

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
            
            # Valuation
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "peg_ratio": info.get("pegRatio"),
            "pb_ratio": info.get("priceToBook"),
            "ps_ratio": info.get("priceToSalesTrailing12Months"),
            
            # EPS & Revenue
            "eps_ttm": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            "revenue_ttm": info.get("totalRevenue"),
            "gross_profit": info.get("grossProfits"),
            "ebitda": info.get("ebitda"),
            "net_income": info.get("netIncomeToCommon"),
            
            # Shares
            "shares_outstanding": info.get("sharesOutstanding"),
            "float_shares": info.get("floatShares"),
            "insider_ownership": info.get("heldPercentInsiders"),
            "institutional_ownership": info.get("heldPercentInstitutions"),
            
            # Risk
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
            
            # Dividends
            "dividend_yield": info.get("dividendYield"),
            "dividend_rate": info.get("dividendRate"),
            "payout_ratio": info.get("payoutRatio"),
            
            # Financial
            "current_ratio": info.get("currentRatio"),
            "quick_ratio": info.get("quickRatio"),
            "debt_to_equity": info.get("debtToEquity"),
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "free_cash_flow": info.get("freeCashflow"),
            
            # Analyst
            "target_high": info.get("targetHighPrice"),
            "target_low": info.get("targetLowPrice"),
            "target_mean": info.get("targetMeanPrice"),
            "recommendation": info.get("recommendationKey"),
            "analyst_count": info.get("numberOfAnalystOpinions"),
            
            # Earnings
            "next_earnings_date": earnings_date,
            
            "fetched_at": datetime.now(timezone.utc).isoformat()
        }

    except Exception as e:
        return {"error": str(e), "symbol": symbol}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Symbol required"}))
        sys.exit(1)
    
    symbol = sys.argv[1]
    result = get_fundamentals(symbol)
    print(json.dumps(result))
