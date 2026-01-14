#!/usr/bin/env python3
"""
Setup script for Research Radar - Competitor Intelligence System

Steps:
1. Create virtual environment
2. Install dependencies
3. Configure environment variables
4. Verify Ollama is running
5. Test connections to all services
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv


def run_command(cmd, description):
    """Run a shell command and report status."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, shell=True)
    return result.returncode == 0


def check_env_vars():
    """Check if all required environment variables are set."""
    load_dotenv()

    required_vars = {
        "FIRECRAWL_API_KEY": "Firecrawl API key",
        "SUPABASE_URL": "Supabase project URL",
        "SUPABASE_KEY": "Supabase API key",
        "ANTHROPIC_API_KEY": "Claude API key",
        "SLACK_BOT_TOKEN": "Slack bot token",
        "SLACK_CHANNEL_ID": "Slack channel ID",
        "TRIGGER_API_KEY": "Trigger.dev API key"
    }

    print(f"\n{'='*60}")
    print("▶ Checking Environment Variables")
    print(f"{'='*60}")

    missing = []
    for var, description in required_vars.items():
        value = os.getenv(var)
        if value:
            masked = value[:4] + "*" * (len(value) - 8) + value[-4:]
            print(f"✓ {var}: {masked}")
        else:
            print(f"✗ {var}: NOT SET ({description})")
            missing.append(var)

    if missing:
        print(f"\n⚠️  Missing {len(missing)} environment variables")
        print("\nCopy .env.example to .env and fill in your credentials:")
        print("  cp .env.example .env")
        return False

    print("\n✓ All environment variables are set")
    return True


def check_ollama():
    """Check if Ollama is running and nomic-embed-text is loaded."""
    print(f"\n{'='*60}")
    print("▶ Checking Ollama Service")
    print(f"{'='*60}")

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            models = response.json().get("models", [])
            model_names = [m.get("name", "").split(":")[0] for m in models]

            if "nomic-embed-text" in model_names:
                print("✓ Ollama is running with nomic-embed-text model")
                return True
            else:
                print("⚠️  Ollama is running but nomic-embed-text not loaded")
                print("\nPull the model with:")
                print("  ollama pull nomic-embed-text")
                return False
        else:
            print("✗ Ollama returned error:", response.status_code)
            return False

    except requests.exceptions.ConnectionError:
        print("✗ Ollama is not running")
        print("\nStart Ollama with:")
        print("  ollama serve")
        return False
    except Exception as e:
        print(f"✗ Error checking Ollama: {str(e)}")
        return False


def test_imports():
    """Test that all Python dependencies can be imported."""
    print(f"\n{'='*60}")
    print("▶ Testing Python Dependencies")
    print(f"{'='*60}")

    modules = {
        "firecrawl": "Firecrawl",
        "supabase": "Supabase",
        "anthropic": "Anthropic",
        "slack_sdk": "Slack SDK",
        "tenacity": "Tenacity",
        "requests": "Requests"
    }

    missing = []
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name}")
            missing.append(module)

    if missing:
        print(f"\n⚠️  Missing {len(missing)} dependencies")
        print("\nInstall with:")
        print("  pip install -r requirements.txt")
        return False

    print("\n✓ All dependencies are installed")
    return True


def test_connections():
    """Test connections to all services."""
    print(f"\n{'='*60}")
    print("▶ Testing Service Connections")
    print(f"{'='*60}")

    all_ok = True

    # Test Firecrawl
    try:
        from src.agents.crawler import CompetitorCrawler
        crawler = CompetitorCrawler()
        print("✓ Firecrawl API configured")
    except Exception as e:
        print(f"✗ Firecrawl: {str(e)}")
        all_ok = False

    # Test Supabase
    try:
        from src.agents.storage import SupabaseStorage
        storage = SupabaseStorage()
        if storage.ensure_schema():
            print("✓ Supabase connected and schema verified")
        else:
            print("⚠️  Supabase schema incomplete - run migrations")
            all_ok = False
    except Exception as e:
        print(f"✗ Supabase: {str(e)}")
        all_ok = False

    # Test Claude
    try:
        from src.agents.analyzer import ChangeAnalyzer
        analyzer = ChangeAnalyzer()
        print("✓ Claude API configured")
    except Exception as e:
        print(f"✗ Claude: {str(e)}")
        all_ok = False

    # Test Slack
    try:
        from src.agents.alerts import SlackNotifier
        notifier = SlackNotifier()
        print("✓ Slack configured")
    except Exception as e:
        print(f"✗ Slack: {str(e)}")
        all_ok = False

    # Test Ollama
    try:
        from src.agents.storage import OllamaEmbeddings
        embeddings = OllamaEmbeddings()
        if embeddings.is_available():
            print("✓ Ollama embeddings available")
        else:
            print("⚠️  Ollama not available")
            all_ok = False
    except Exception as e:
        print(f"✗ Ollama: {str(e)}")
        all_ok = False

    return all_ok


def main():
    """Run setup steps."""
    print("\n" + "="*60)
    print("🚀 Research Radar Setup")
    print("="*60)

    # Change to project directory
    project_dir = Path(__file__).parent
    os.chdir(project_dir)

    # Check steps
    checks = [
        ("Installing dependencies", lambda: run_command("pip install -r requirements.txt", "Installing Python dependencies")),
        ("Checking environment variables", check_env_vars),
        ("Testing Python imports", test_imports),
        ("Checking Ollama", check_ollama),
        ("Testing service connections", test_connections)
    ]

    results = []
    for check_name, check_func in checks:
        result = check_func()
        results.append((check_name, result))

    # Summary
    print(f"\n{'='*60}")
    print("📋 Setup Summary")
    print(f"{'='*60}")

    for check_name, result in results:
        status = "✓" if result else "✗"
        print(f"{status} {check_name}")

    all_passed = all(r for _, r in results)

    if all_passed:
        print(f"\n✅ Setup complete! Your system is ready.")
        print(f"\nNext steps:")
        print(f"1. Add competitors to track:")
        print(f"   python -c \"from src.agents.storage import SupabaseStorage; s = SupabaseStorage(); s.add_competitor('Notion', 'notion.so')\"")
        print(f"\n2. Run a manual crawl:")
        print(f"   python triggers/weekly.py --run-id manual-test-1")
        print(f"\n3. Set up weekly scheduling with Trigger.dev")
        return 0
    else:
        print(f"\n❌ Setup incomplete. Fix the issues above and try again.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
