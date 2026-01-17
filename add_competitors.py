#!/usr/bin/env python3
"""Add competitors to the database."""

from dotenv import load_dotenv
load_dotenv()

from src.agents.storage import SupabaseStorage

s = SupabaseStorage()

# Add competitors
competitors = [
    ('Notion', 'https://www.notion.so'),
    ('Linear', 'https://www.linear.app'),
    ('Confluence', 'https://www.atlassian.com/software/confluence')
]

for name, url in competitors:
    try:
        result = s.add_competitor(name, url)
        if result:
            print(f"✓ Added {name}")
        else:
            print(f"⚠️  {name} may already exist or failed")
    except Exception as e:
        print(f"✗ Failed to add {name}: {e}")

# List all competitors
print("\nCurrent competitors:")
for c in s.list_competitors():
    print(f"  - {c['name']}: {c['url']}")
