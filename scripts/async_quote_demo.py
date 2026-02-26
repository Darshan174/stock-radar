"""
Real async stock quote demo.

This version calls a real quote endpoint and runs requests concurrently.
No fake sleep and no hardcoded response values.
"""

import asyncio
from time import perf_counter
from typing import Any

import requests

YAHOO_QUOTE_URL = "https://query1.finance.yahoo.com/v7/finance/quote"


def fetch_stock_data_sync(ticker: str) -> dict[str, Any]:
    """Blocking HTTP request for one ticker."""
    response = requests.get(
        YAHOO_QUOTE_URL,
        params={"symbols": ticker},
        timeout=10,
    )
    response.raise_for_status()

    payload = response.json()
    results = payload.get("quoteResponse", {}).get("result", [])
    if not results:
        raise ValueError(f"No data returned for ticker: {ticker}")

    quote = results[0]
    price = quote.get("regularMarketPrice")
    if price is None:
        raise ValueError(f"Missing regularMarketPrice for ticker: {ticker}")

    return {
        "ticker": ticker,
        "price": float(price),
        "change": float(quote.get("regularMarketChange", 0.0)),
        "change_percent": float(quote.get("regularMarketChangePercent", 0.0)),
        "currency": quote.get("currency"),
        "market_time": quote.get("regularMarketTime"),
    }


async def fetch_stock_data(ticker: str) -> dict[str, Any]:
    """
    Async wrapper around the blocking request.

    asyncio.to_thread keeps the event loop free while HTTP runs in a worker thread.
    """
    print(f"Fetching data for {ticker}...")
    result = await asyncio.to_thread(fetch_stock_data_sync, ticker)
    print(f"Data received for {ticker}!")
    return result


async def main() -> None:
    tickers = ["AAPL", "TSLA", "MSFT", "NVDA"]

    start = perf_counter()
    tasks = [fetch_stock_data(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = perf_counter() - start

    print("\nResults:")
    for ticker, result in zip(tickers, results):
        if isinstance(result, Exception):
            print(f"- {ticker}: ERROR -> {result}")
            continue
        print(
            f"- {ticker}: ${result['price']:.2f} "
            f"({result['change']:+.2f}, {result['change_percent']:+.2f}%)"
        )

    print(f"\nDone in {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(main())
