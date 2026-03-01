# Deploy Runbook: Vercel Web + Railway Backend

This runbook deploys Stock-Radar in split mode:
- `web` app on Vercel (public entrypoint, x402, UI, public API routes)
- Python backend on Railway (private API for analysis/agents)
- Supabase as the database
- Aptos testnet for payment/reputation flow
- Upstash Redis REST for IP rate limits

## 1) Preflight checks (local)

Run from repository root:

```bash
pytest tests/ -v
ruff check src/ tests/ backend/
```

Run from `web/`:

```bash
npm run lint
npm run build
```

## 2) Deploy backend to Railway first

1. Create a new Railway service from this repo.
2. Set `Dockerfile Path` to `Dockerfile.backend`.
3. Railway start command:

```bash
uvicorn backend.app:app --host 0.0.0.0 --port $PORT
```

4. Configure backend environment variables:
   - `BACKEND_API_KEY` (required)
   - `SUPABASE_URL` (required)
   - `SUPABASE_KEY` (required)
   - At least one LLM key (required): `ZAI_API_KEY` or `GROQ_API_KEY` or `GEMINI_API_KEY`
   - `APTOS_NETWORK` (testnet recommended for demo)
   - `APTOS_RECIPIENT_ADDRESS`
   - Optional: `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `COHERE_API_KEY`
5. Deploy and copy the Railway public URL.

Health check:

```bash
curl -sS <RAILWAY_URL>/health | jq
```

Expected:
- top-level `status: "ok"`
- `dependencies.supabase.status` is healthy/degraded (not crash)
- `dependencies.llm.configuredProviders` contains at least one provider

## 3) Deploy web to Vercel

1. Import project in Vercel.
2. Set root directory to `web`.
3. Configure web environment variables:
   - `PY_BACKEND_URL=<RAILWAY_URL>`
   - `PY_BACKEND_API_KEY=<same as BACKEND_API_KEY>`
   - `NEXT_PUBLIC_SUPABASE_URL`
   - `NEXT_PUBLIC_SUPABASE_ANON_KEY`
   - `APTOS_NETWORK` (testnet)
   - `APTOS_RECIPIENT_ADDRESS`
   - `APTOS_PRICE_PER_REQUEST`
   - `X402_USE_FACILITATOR`
   - `X402_VERIFICATION_MODE`
   - `INTERNAL_API_KEY`
   - `UPSTASH_REDIS_REST_URL`
   - `UPSTASH_REDIS_REST_TOKEN`
4. Deploy.

## 4) Smoke test checklist (post-deploy)

Replace `<VERCEL_URL>` with your deployment URL.

1. Health endpoint:

```bash
curl -i https://<VERCEL_URL>/api/health
```

2. Capability discovery:

```bash
curl -i https://<VERCEL_URL>/api/agent/discover
```

3. Payment challenge (no payment header):

```bash
curl -i "https://<VERCEL_URL>/api/agent/momentum?symbol=AAPL"
```

Expected: HTTP `402`.

4. Async analyze submit:

```bash
curl -i -X POST https://<VERCEL_URL>/api/analyze \
  -H "Content-Type: application/json" \
  -H "X-Internal-Key: <INTERNAL_API_KEY>" \
  -d '{"symbol":"AAPL","mode":"intraday"}'
```

Expected: HTTP `202` with `jobId`.

5. Analyze polling:

```bash
curl -i "https://<VERCEL_URL>/api/analyze/status?jobId=<JOB_ID>"
```

Expected status progression: `queued` -> `running` -> `succeeded` or `failed`.

6. Chat ask:

```bash
curl -i -X POST https://<VERCEL_URL>/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize AAPL momentum","symbol":"AAPL"}'
```

7. Rate limit:
- Call a free route (`/api/agent/discover`) repeatedly.
- Expect HTTP `429` after threshold.

## 5) Failure behavior to verify

- Stop Railway service temporarily; Vercel proxy routes should return stable `503` style errors.
- Invalid symbol should return HTTP `400`.
- Invalid backend key should not leak secret details through Vercel responses.
- Analyze jobs that exceed timeout should show structured failed status.

## 6) Manual rollout order

1. Railway backend deploy + validate `/health`
2. Vercel web deploy with backend URL/key
3. Run smoke tests
4. Keep previous Vercel deployment ready for rollback

## 7) Rollback

1. Promote previous Vercel deployment.
2. If backend instability persists, temporarily disable analyze/ask UX via maintenance flag.
3. Keep Supabase schema/data unchanged (no destructive migration in this rollout).
