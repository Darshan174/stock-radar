# Stock-Radar DevRel Learning Track

This folder implements the 2-week DevRel plan for learning Stock-Radar's
technology stack and Python concepts with practical, demo-ready outputs.

## What is included

- `2-week-curriculum.md`: day-by-day execution guide (10 working days).
- `workshop-pack.md`: reusable templates for slides, live demo script,
  API handout, FAQ, and troubleshooting appendix.
- `../../scripts/devrel_validate.sh`: validation command runner.
- `../../scripts/devrel_api_flow.sh`: end-to-end API flow smoke runner.

## How to run this plan

1. Start each day in `2-week-curriculum.md`.
2. Run the listed commands for the day.
3. Fill in the corresponding artifact sections in `workshop-pack.md`.
4. Use `scripts/devrel_validate.sh quick` daily.
5. Use `scripts/devrel_validate.sh full` on Day 10.

## Required local tools

- Python 3.11+
- Node.js 20+
- `pytest`
- `ruff`
- `curl`
- `jq` (recommended for JSON output)

## Suggested daily cadence

- 25 min: focused reading/code walkthrough
- 50 min: execute commands and verify outputs
- 30 min: produce DevRel artifact
- 15 min: rehearsal/talk-through

## Exit criteria after Day 10

- You can explain architecture and tradeoffs clearly.
- You can run and narrate async analysis + polling live.
- You can explain LLM routing, guardrails, risk, and observability.
- You have a workshop-ready pack from `workshop-pack.md`.
