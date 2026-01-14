"""
Firecrawl-based web crawler for competitor intelligence.
Scrapes competitor websites and extracts markdown content with retry logic.
"""

import os
from datetime import datetime
from typing import Optional, Dict, List
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

from firecrawl import FirecrawlApp

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)


class CompetitorCrawler:
    """Crawls competitor websites using Firecrawl API with retry logic."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Firecrawl crawler.

        Args:
            api_key: Firecrawl API key (defaults to FIRECRAWL_API_KEY env var)

        Raises:
            ValueError: If API key is not provided
        """
        api_key = api_key or os.getenv("FIRECRAWL_API_KEY")
        if not api_key:
            raise ValueError("FIRECRAWL_API_KEY environment variable not set")

        self.app = FirecrawlApp(api_key=api_key)
        logger.info("CompetitorCrawler initialized")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def scrape_url(self, url: str) -> Dict[str, any]:
        """
        Scrape a single competitor URL and extract structured content.

        Includes exponential backoff retry logic for network failures.

        Args:
            url: The competitor website URL to scrape

        Returns:
            Dictionary with structure:
            {
                'url': str,
                'markdown': str,
                'html': str,
                'crawl_date': str (ISO format),
                'status': str ('success' or 'failed'),
                'error': str (optional, only if status is 'failed'),
                'metadata': dict (optional scraped metadata)
            }
        """
        crawl_timestamp = datetime.utcnow().isoformat()

        try:
            logger.info(f"[{crawl_timestamp}] Scraping URL: {url}")

            # Call Firecrawl API with markdown and HTML formats
            result = self.app.scrape_url(
                url,
                params={
                    "formats": ["markdown", "html"],
                    "onlyMainContent": True,
                    "removeBase64Images": True,
                    "includeTags": ["article", "main", "div", "section"],
                    "excludeTags": ["nav", "footer", "script", "style", "iframe", "noscript"]
                }
            )

            # Check API response
            if not result or not result.get("success", False):
                error_msg = result.get("error", "Unknown error") if result else "Empty response"
                logger.error(f"Failed to scrape {url}: {error_msg}")
                return {
                    "url": url,
                    "markdown": "",
                    "html": "",
                    "crawl_date": crawl_timestamp,
                    "status": "failed",
                    "error": error_msg
                }

            # Extract content
            markdown_content = result.get("markdown", "")
            html_content = result.get("html", "")
            metadata = result.get("metadata", {})

            crawl_data = {
                "url": url,
                "markdown": markdown_content,
                "html": html_content,
                "crawl_date": crawl_timestamp,
                "status": "success",
                "metadata": {
                    "title": metadata.get("title", ""),
                    "description": metadata.get("description", ""),
                    "language": metadata.get("language", ""),
                    "sourceURL": metadata.get("sourceURL", url)
                }
            }

            logger.info(
                f"Successfully scraped {url} - "
                f"Markdown: {len(markdown_content)} chars, "
                f"HTML: {len(html_content)} chars"
            )

            return crawl_data

        except ConnectionError as e:
            logger.error(f"Connection error scraping {url}: {str(e)}")
            raise  # Retry via tenacity

        except TimeoutError as e:
            logger.error(f"Timeout error scraping {url}: {str(e)}")
            raise  # Retry via tenacity

        except Exception as e:
            logger.error(f"Unexpected error scraping {url}: {str(e)}", exc_info=True)
            return {
                "url": url,
                "markdown": "",
                "html": "",
                "crawl_date": crawl_timestamp,
                "status": "failed",
                "error": str(e)
            }

    def scrape_multiple(self, urls: List[str]) -> List[Dict[str, any]]:
        """
        Scrape multiple competitor URLs sequentially.

        Continues scraping even if individual URLs fail.

        Args:
            urls: List of competitor website URLs to scrape

        Returns:
            List of crawl data dictionaries (both successful and failed scrapes)
        """
        logger.info(f"Starting batch scrape of {len(urls)} URLs")
        results = []

        for idx, url in enumerate(urls, 1):
            logger.info(f"Processing URL {idx}/{len(urls)}: {url}")

            try:
                crawl_data = self.scrape_url(url)
                results.append(crawl_data)

                if crawl_data["status"] == "success":
                    logger.info(f"Successfully scraped {url}")
                else:
                    logger.warning(f"Failed to scrape {url}: {crawl_data.get('error', 'Unknown')}")

            except Exception as e:
                logger.error(f"Critical error scraping {url}: {str(e)}", exc_info=True)
                results.append({
                    "url": url,
                    "markdown": "",
                    "html": "",
                    "crawl_date": datetime.utcnow().isoformat(),
                    "status": "failed",
                    "error": f"Critical error: {str(e)}"
                })

        successful = sum(1 for r in results if r["status"] == "success")
        logger.info(f"Batch scrape complete: {successful}/{len(urls)} successful")

        return results


if __name__ == "__main__":
    # Test crawler with sample competitor URLs
    crawler = CompetitorCrawler()

    test_urls = [
        "https://www.notion.so",
        "https://www.linear.app",
        "https://www.atlassian.com/software/confluence"
    ]

    print("\nTesting CompetitorCrawler...")
    print(f"Scraping {len(test_urls)} URLs\n")

    results = crawler.scrape_multiple(test_urls)

    print("\n" + "="*70)
    print("SCRAPE RESULTS")
    print("="*70)

    for result in results:
        print(f"\nURL: {result['url']}")
        print(f"Status: {result['status']}")
        print(f"Crawl Date: {result['crawl_date']}")

        if result['status'] == 'success':
            print(f"Markdown Length: {len(result['markdown'])} chars")
            print(f"HTML Length: {len(result['html'])} chars")
            if result.get('metadata'):
                print(f"Title: {result['metadata'].get('title', 'N/A')}")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
        print("-" * 70)
