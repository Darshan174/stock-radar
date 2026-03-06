# Workshop Pack Template (DevRel)

Use this file as the final output artifact for Day 10.

## 1) Slide Outline (45-60 minutes)

### Slide 1
Title: Stock-Radar in Production: Python AI + Market Intelligence

### Slide 2
Problem statement and target users.

### Slide 3
System architecture (Web, Backend, Supabase, LLM/Data providers).

### Slide 4
Python service layer patterns.
- FastAPI API design
- Pydantic settings and schemas
- dataclasses and typed contracts

### Slide 5
Async analysis jobs and API contracts.
- create job
- poll status
- terminal states

### Slide 6
LLM reliability.
- task routing
- fallback chain
- guardrails
- token/cost accounting

### Slide 7
Observability and operations.
- Prometheus metrics
- structured logging
- cache behavior

### Slide 8
ML pipeline and risk controls.
- feature engineering and training
- inference and regime logic
- backtesting realism
- pre-trade risk and canary controls

### Slide 9
Cross-stack integration.
- Next.js backend proxy
- optional Aptos/Move agent registry

### Slide 10
Live demo summary and Q&A.

## 2) Live Demo Script (10-15 minutes)

### Setup
```bash
cd /Users/darshann/Desktop/stock-radar
./scripts/devrel_validate.sh quick
```

### Demo sequence
1. Show API discovery and health path.
2. Submit async analyze request.
3. Poll job until terminal state.
4. Show where routing/guardrails are implemented in code.
5. Show one risk/backtest test case and why it matters.

### Demo talk track notes
- "This design isolates expensive analysis in backend while web handles public UX and rate-limiting."
- "The async job contract avoids long request blocking and gives clear operational states."
- "Routing + guardrails are explicit, testable controls, not hidden prompt magic."

## 3) API Handout (Quick Reference)

### Backend endpoints
- `POST /v1/analyze/jobs`
- `GET /v1/analyze/jobs/{job_id}`
- `POST /v1/ask`
- `GET /v1/fundamentals`
- `GET /v1/agent/*`

### Shared types
- `AnalyzeJobCreated`
- `AnalyzeJobStatus`

### Typical async flow
1. Submit job.
2. Receive `jobId` and initial `queued` status.
3. Poll status route.
4. Handle `succeeded` or `failed`.

## 4) FAQ Template

1. Why not run analysis directly in Next.js routes?
Answer: ________________________

2. How do you prevent LLM output drift and unsafe responses?
Answer: ________________________

3. How is latency/cost monitored over time?
Answer: ________________________

4. What makes backtesting realistic in this project?
Answer: ________________________

5. What parts are optional for non-web3 users?
Answer: ________________________

## 5) Troubleshooting Appendix

### Symptom: backend unavailable
Checks:
- verify `PY_BACKEND_URL`
- verify `PY_BACKEND_API_KEY`
- verify backend `/health`

### Symptom: async job fails repeatedly
Checks:
- LLM key availability (`ZAI_API_KEY`, `GROQ_API_KEY`, `GEMINI_API_KEY`)
- provider limits or upstream timeout
- request payload validation

### Symptom: empty context in chat
Checks:
- Supabase credentials
- schema/migrations
- storage + retrieval code path

### Symptom: metrics missing
Checks:
- metrics endpoint
- Prometheus target config
- service/network routing

## 6) Final Readiness Checklist

- [ ] Can explain architecture in 2 minutes.
- [ ] Can run async analyze flow live without errors.
- [ ] Can explain routing + guardrails from source code.
- [ ] Can explain one risk control and one backtest realism control.
- [ ] Can answer top 5 audience questions confidently.
