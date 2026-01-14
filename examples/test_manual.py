#!/usr/bin/env python3
"""
Manual testing script for Research Radar components.
Tests each agent individually before running the full workflow.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.crawler import CompetitorCrawler
from src.agents.storage import SupabaseStorage, OllamaEmbeddings
from src.agents.analyzer import ChangeAnalyzer
from src.agents.alerts import SlackNotifier


def test_crawler():
    """Test Firecrawl crawler."""
    print("\n" + "="*60)
    print("Testing CompetitorCrawler")
    print("="*60)

    try:
        crawler = CompetitorCrawler()
        print("✓ Crawler initialized")

        # Test single URL
        url = "https://www.notion.so"
        print(f"\nScraping {url}...")
        result = crawler.scrape_url(url)

        if result["status"] == "success":
            print(f"✓ Successfully scraped {url}")
            print(f"  - Markdown: {len(result['markdown'])} chars")
            print(f"  - HTML: {len(result['html'])} chars")
            return result
        else:
            print(f"✗ Failed to scrape: {result.get('error')}")
            return None

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def test_storage():
    """Test Supabase storage."""
    print("\n" + "="*60)
    print("Testing SupabaseStorage")
    print("="*60)

    try:
        storage = SupabaseStorage()
        print("✓ Storage initialized")

        # Test schema
        if storage.ensure_schema():
            print("✓ Schema verified")
        else:
            print("⚠️  Schema incomplete - run migrations")

        # Test listing competitors
        competitors = storage.list_competitors()
        print(f"✓ Found {len(competitors)} competitors")

        return storage

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def test_embeddings():
    """Test Ollama embeddings."""
    print("\n" + "="*60)
    print("Testing OllamaEmbeddings")
    print("="*60)

    try:
        embeddings = OllamaEmbeddings()

        if embeddings.is_available():
            print("✓ Ollama is available")

            test_text = "Test embedding generation"
            embedding = embeddings.embed_text(test_text)

            if embedding:
                print(f"✓ Generated embedding with {len(embedding)} dimensions")
                return embeddings
            else:
                print("✗ Failed to generate embedding")
                return None
        else:
            print("✗ Ollama not available")
            print("  Run: ollama pull nomic-embed-text")
            return None

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def test_analyzer():
    """Test Claude analyzer."""
    print("\n" + "="*60)
    print("Testing ChangeAnalyzer")
    print("="*60)

    try:
        analyzer = ChangeAnalyzer()
        print("✓ Analyzer initialized")

        # Test with sample data
        old_text = """
        # Product Pricing
        Pro Plan: $10/month
        Enterprise: Custom pricing
        """

        new_text = """
        # Product Pricing
        Pro Plan: $12/month
        Premium Plan: $20/month (NEW)
        Enterprise: Custom pricing
        """

        print("\nAnalyzing sample changes...")
        result = analyzer.analyze_changes("Test", old_text, new_text)

        if result and result.get("has_changes"):
            print(f"✓ Detected {len(result['changes'])} changes")
            for change in result["changes"]:
                print(f"  - {change['type']}: {change['summary']}")
        else:
            print("⚠️  No changes detected in test data")

        return result

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def test_slack():
    """Test Slack notifier."""
    print("\n" + "="*60)
    print("Testing SlackNotifier")
    print("="*60)

    try:
        notifier = SlackNotifier()
        print("✓ Slack notifier initialized")

        # Test connection
        if notifier.test_connection():
            print("✓ Connected to Slack")
            return notifier
        else:
            print("⚠️  Could not verify Slack connection")
            return notifier

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        return None


def test_full_workflow():
    """Test the complete workflow."""
    print("\n" + "="*60)
    print("Testing Full Workflow")
    print("="*60)

    # Get test data
    print("\n1. Testing crawler...")
    crawl = test_crawler()
    if not crawl:
        print("✗ Crawler test failed, skipping workflow")
        return

    print("\n2. Testing storage...")
    storage = test_storage()
    if not storage:
        print("✗ Storage test failed, skipping workflow")
        return

    print("\n3. Testing embeddings...")
    embeddings = test_embeddings()
    if not embeddings:
        print("⚠️  Embeddings test failed, continuing without embeddings")

    print("\n4. Testing analyzer...")
    analysis = test_analyzer()
    if not analysis:
        print("⚠️  Analyzer test failed")

    print("\n5. Testing Slack...")
    notifier = test_slack()
    if not notifier:
        print("⚠️  Slack test failed")

    print("\n" + "="*60)
    print("✅ Workflow testing complete!")
    print("="*60)


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 Research Radar Component Tests")
    print("="*60)

    test_crawler()
    test_storage()
    test_embeddings()
    test_analyzer()
    test_slack()

    # Full workflow test
    test_full_workflow()

    print("\n" + "="*60)
    print("All tests completed!")
    print("="*60)


if __name__ == "__main__":
    main()
