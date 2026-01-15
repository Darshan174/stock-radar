#!/usr/bin/env python3
"""
Simple test script for the Ollama-based analyzer.
Run this to verify Ollama is working before running the full system.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.analyzer import OllamaAnalyzer

def test_ollama_connection():
    """Test if Ollama is running and accessible."""
    print("\n" + "="*60)
    print("TEST 1: Ollama Connection")
    print("="*60)

    analyzer = OllamaAnalyzer(model="mistral")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            if "mistral" in model_names:
                print("✓ Ollama is running with mistral model")
                return True
            else:
                print("✗ Mistral model not found. Available models:", model_names)
                print("\nTo fix, run: ollama pull mistral")
                return False
        else:
            print("✗ Ollama returned status:", response.status_code)
            return False
    except Exception as e:
        print(f"✗ Cannot connect to Ollama: {str(e)}")
        print("\nMake sure Ollama is running: ollama serve")
        return False


def test_simple_analysis():
    """Test analyzer with simple example."""
    print("\n" + "="*60)
    print("TEST 2: Simple Change Detection")
    print("="*60)

    analyzer = OllamaAnalyzer(model="mistral")

    # Simple test case
    old_content = """
    Notion - The All-in-One Workspace

    PRICING
    Free Plan - $0/month
    Pro Plan - $10/month
    Team Plan - $15/person/month

    FEATURES
    - Database blocks
    - Calendar views
    - Time tracking
    """

    new_content = """
    Notion - The All-in-One Workspace

    PRICING
    Free Plan - $0/month
    Pro Plan - $12/month (INCREASED)
    Team Plan - $15/person/month
    Plus Plan - $25/month (NEW!)

    FEATURES
    - Database blocks
    - Calendar views
    - Time tracking
    - AI Summaries (NEW)
    """

    print("Testing analysis of Notion pricing + feature changes...")
    result = analyzer.analyze_changes(
        competitor_name="Notion",
        old_markdown=old_content,
        new_markdown=new_content
    )

    if result is None:
        print("✗ Analysis returned None")
        return False

    print(f"✓ Analysis completed successfully")
    print(f"\nResult:")
    print(f"  Has changes: {result.get('has_changes')}")
    print(f"  Number of changes: {len(result.get('changes', []))}")

    if result.get("changes"):
        print(f"\n  Detected changes:")
        for i, change in enumerate(result["changes"], 1):
            print(f"    {i}. {change.get('type', 'unknown').upper()}")
            print(f"       Summary: {change.get('summary', 'N/A')}")
            print(f"       Importance: {change.get('importance', 'unknown')}")

    return result.get("has_changes", False)


def test_no_changes():
    """Test analyzer with no real changes."""
    print("\n" + "="*60)
    print("TEST 3: No Changes Detection")
    print("="*60)

    analyzer = OllamaAnalyzer(model="mistral")

    # Same content = no changes
    content = """
    Linear - Issue Tracking

    PRICING
    Free: $0
    Pro: $8/month
    Scale: Custom

    FEATURES
    - Issue management
    - Collaboration
    - Automation
    """

    print("Analyzing identical content (should detect no changes)...")
    result = analyzer.analyze_changes(
        competitor_name="Linear",
        old_markdown=content,
        new_markdown=content
    )

    if result is None:
        print("✓ Correctly returned None for <1% change")
        return True

    has_changes = result.get("has_changes", False)
    if has_changes:
        print("✗ False positive: Detected changes in identical content")
        return False
    else:
        print("✓ Correctly identified no significant changes")
        return True


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("🧪 OLLAMA ANALYZER TEST SUITE")
    print("="*60)
    print("\nTesting Ollama-based change analyzer (costs $0, not $0.05!)\n")

    results = []

    # Test 1: Connection
    test1 = test_ollama_connection()
    results.append(("Ollama Connection", test1))

    if not test1:
        print("\n" + "="*60)
        print("❌ OLLAMA NOT RUNNING")
        print("="*60)
        print("\nCannot proceed without Ollama. Please:")
        print("1. Start Ollama: ollama serve")
        print("2. Pull mistral: ollama pull mistral")
        print("3. Run this test again")
        return 1

    # Test 2: Simple analysis
    try:
        test2 = test_simple_analysis()
        results.append(("Simple Analysis", test2))
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        results.append(("Simple Analysis", False))

    # Test 3: No changes
    try:
        test3 = test_no_changes()
        results.append(("No Changes Detection", test3))
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        results.append(("No Changes Detection", False))

    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {test_name}")

    all_passed = all(passed for _, passed in results)

    if all_passed:
        print("\n" + "="*60)
        print("✅ ALL TESTS PASSED!")
        print("="*60)
        print("\nYour Ollama analyzer is ready to use.")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Set up .env with your credentials")
        print("3. Create database schema in Supabase")
        print("4. Run: python triggers/weekly.py --run-id test-1")
        return 0
    else:
        print("\n" + "="*60)
        print("❌ SOME TESTS FAILED")
        print("="*60)
        print("\nFix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
