  """
  Ollama-based change detection (no API costs!)
  Uses local Ollama instance instead of Claude API
  """

  import os
  import json
  import logging
  from datetime import datetime
  from typing import Optional
  from difflib import unified_diff
  import requests

  logger = logging.getLogger(__name__)
  logging.basicConfig(level=logging.INFO)


  class OllamaAnalyzer:
      """Analyzes changes using local Ollama (free!)"""

      def __init__(self, 
                   api_url: Optional[str] = None,
                   model: str = "mistral"):
          """
          Initialize Ollama analyzer.
          
          Args:
              api_url: Ollama API URL (default: http://localhost:11434)
              model: Model name (default: mistral - best balance)
          """
          self.api_url = api_url or os.getenv("OLLAMA_API_URL",
                                               "http://localhost:11434")
          self.model = model
          logger.info(f"Initialized OllamaAnalyzer with model={model}")

      def _extract_diff(self, old_text: str, new_text: str) -> str:
          """Create a diff showing changes"""
          old_lines = old_text.split("\n")
          new_lines = new_text.split("\n")

          diff = unified_diff(
              old_lines,
              new_lines,
              fromfile="previous",
              tofile="current",
              lineterm="",
              n=3
          )
          return "\n".join(diff)

      def analyze_changes(self, 
                         competitor_name: str,
                         old_markdown: str,
                         new_markdown: str) -> Optional[dict]:
          """
          Analyze changes using local Ollama model (FREE!)
          
          Returns:
              {
                  "competitor_name": str,
                  "has_changes": bool,
                  "changes": [
                      {
                          "type": "pricing|feature|hiring|partnership|other",
                          "summary": "Description of change",
                          "importance": "high|medium|low",
                          "details": "More details"
                      }
                  ]
              }
          """
          try:
              # Check if changes are significant (>1% different)
              old_len = len(old_markdown)
              new_len = len(new_markdown)
              change_ratio = abs(new_len - old_len) / max(old_len, 1)

              if change_ratio < 0.01:
                  logger.info(f"Skipping {competitor_name}: <1% change")
                  return None

              # Create diff
              diff = self._extract_diff(old_markdown, new_markdown)

              # Limit diff size for Ollama (it's slower than Claude)
              if len(diff) > 2000:
                  diff = diff[:2000] + "\n... (truncated)"

              # Create the prompt for Ollama
              prompt = f"""You are analyzing website changes for competitor intelligence.

  Competitor: {competitor_name}
  Website diff showing old vs new content:

  {diff}

  Analyze the changes and respond with ONLY a JSON object (no other text).

  Return JSON with:
  - "has_changes": true/false (are there meaningful changes?)
  - "changes": array of change objects with:
    - "type": one of [pricing, feature, hiring, partnership, content, other]
    - "summary": 1-2 sentence summary of the change
    - "importance": one of [high, medium, low]
    - "details": 2-3 sentence explanation

  Ignore trivial changes like date updates, timestamps, analytics code.
  Focus on business-relevant changes.

  Example response:
  {{
    "has_changes": true,
    "changes": [
      {{
        "type": "pricing",
        "summary": "Pro plan price increased from $10 to $12 per month",
        "importance": "high",
        "details": "The pricing page now shows the new pricing tier..."
      }}
    ]
  }}

  Respond with only valid JSON:"""

              # Call Ollama
              logger.info(f"Analyzing {competitor_name} with Ollama...")

              response = requests.post(
                  f"{self.api_url}/api/generate",
                  json={
                      "model": self.model,
                      "prompt": prompt,
                      "stream": False,  # Wait for full response
                      "temperature": 0.3  # Lower = more consistent
                  },
                  timeout=60
              )

              response.raise_for_status()

              result = response.json()
              response_text = result.get("response", "")

              # Extract JSON from response (Ollama sometimes adds extra text)
              try:
                  # Try to find JSON in the response
                  start = response_text.find('{')
                  end = response_text.rfind('}') + 1

                  if start != -1 and end > start:
                      json_str = response_text[start:end]
                      analysis = json.loads(json_str)
                  else:
                      logger.error(f"No JSON found in response")
                      return None
              except json.JSONDecodeError as e:
                  logger.error(f"Failed to parse JSON: {e}")
                  logger.debug(f"Raw response: {response_text}")
                  return None

              analysis["competitor_name"] = competitor_name
              analysis["change_ratio"] = change_ratio

              if analysis.get("has_changes"):
                  logger.info(f"Found {len(analysis.get('changes', []))} changes")
              else:
                  logger.info(f"No meaningful changes in {competitor_name}")

              return analysis

          except requests.exceptions.ConnectionError:
              logger.error(f"Cannot connect to Ollama at {self.api_url}")
              logger.error("Make sure Ollama is running: ollama serve")
              return None
          except requests.exceptions.Timeout:
              logger.error(f"Ollama timeout (model too slow or overloaded)")
              return None
          except Exception as e:
              logger.error(f"Error analyzing changes: {str(e)}")
              return None

      def batch_analyze(self, competitors: list) -> list:
          """Analyze multiple competitors"""
          results = []
          for competitor in competitors:
              try:
                  analysis = self.analyze_changes(
                      competitor["name"],
                      competitor["old_markdown"],
                      competitor["new_markdown"]
                  )
                  if analysis:
                      results.append(analysis)
              except Exception as e:
                  logger.error(f"Failed to analyze {competitor['name']}: {e}")
                  continue

          return results


  if __name__ == "__main__":
      # Test the analyzer
      analyzer = OllamaAnalyzer(model="mistral")

      test_data = {
          "name": "Notion",
          "old_markdown": "# Notion\nPricing: Pro Plan $10/month",
          "new_markdown": "# Notion\nPricing: Pro Plan $12/month\nNew: AI summaries feature"
      }

      result = analyzer.analyze_changes(
          test_data["name"],
          test_data["old_markdown"],
          test_data["new_markdown"]
      )

      if result:
          print(json.dumps(result, indent=2))
      else:
          print("Analysis failed")