# Stock-Radar

Pay-per-use financial intelligence APIs with blockchain-native payments, async AI analysis jobs, and an operator-ready dashboard.

## Why Stock-Radar
Stock-Radar is built for developers who want production-style market intelligence without subscription lock-in:

- Query only what you need, pay only for what you call.
- Expose discoverable machine-to-machine endpoints (`/api/agent/discover`, `/api/agent/message`).
- Support payment-gated routes through x402 + Aptos.
- Keep analysis state in Supabase and run compute in a dedicated Python backend.

## What You Get
- Next.js app (`web`) with dashboard + public API routes.
- FastAPI backend (`backend`) for heavy analysis workloads.
- Async job flow for long-running analysis (`/api/analyze` + `/api/analyze/status`).
- Agent endpoints for momentum, scoring, sentiment, fundamentals, and orchestration.
- On-chain agent identity/reputation integration on Aptos testnet.

## Architecture
```mermaid
flowchart LR
    U[User / Client] --> V[Vercel: Next.js Web]
    V -->|X-Backend-Api-Key| R[Railway: FastAPI Backend]
    V --> S[(Supabase)]
    R --> S
    R --> L[LLM Providers]
    R --> M[Market Data Providers]
    V --> A[Aptos Testnet / x402]
```

## Repository Map
- `web/`: Next.js 16 app, UI, public API routes, x402 enforcement/proxy
- `backend/`: FastAPI service used by `web` for analysis/agent compute
- `src/`: Python analysis engine, agents, storage, orchestration logic
- `move-agent-registry/`: Move contracts for agent registry/reputation
- `demo-client/`: CLI demo for payment flow
- `docs/deploy-vercel-railway.md`: Production deployment runbook

## Quick Start (Local)
### 1. Prerequisites
- Node.js 20+
- Python 3.11+
- npm
- Supabase project (URL + keys)

### 2. Configure Environment
From repo root:
```bash
cp .env.example .env
```

From `web/`:
```bash
cp .env.example .env.local
```

Minimum required values:

Backend (`.env`):
- `BACKEND_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_KEY`
- one of `ZAI_API_KEY` or `GROQ_API_KEY` or `GEMINI_API_KEY`

Web (`web/.env.local`):
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `PY_BACKEND_URL=http://localhost:8000`
- `PY_BACKEND_API_KEY=<same as BACKEND_API_KEY>`

### 3. Install Dependencies
```bash
# Python deps
pip install -r requirements.txt

# Web deps
cd web
npm install
```

### 4. Run Backend
From repo root:
```bash
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

Health check:
```bash
curl http://localhost:8000/health
```

### 5. Run Web App
In a second terminal:
```bash
cd web
npm run dev
```

Open:
- `http://localhost:3000`
- `http://localhost:3000/x402-demo`

## API Quickstart
### Service Health
```bash
curl -s http://localhost:3000/api/health | jq .
```

### Agent Discovery
```bash
curl -s http://localhost:3000/api/agent/discover | jq .
```

### Submit Async Analysis
```bash
curl -s -X POST http://localhost:3000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"symbol":"AAPL","mode":"intraday"}' | jq .
```

### Poll Analysis Status
```bash
curl -s "http://localhost:3000/api/analyze/status?jobId=<JOB_ID>" | jq .
```

### Ask Endpoint
```bash
curl -s -X POST http://localhost:3000/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"Summarize AAPL momentum","symbol":"AAPL"}' | jq .
```

## x402-Paid Endpoint Examples
Without payment header, paid routes return HTTP `402`:
```bash
curl -i "http://localhost:3000/api/agent/momentum?symbol=AAPL"
```

For internal/testing bypass flows, configure `INTERNAL_API_KEY` and pass:
```bash
-H "X-Internal-Key: <INTERNAL_API_KEY>"
```

## Deploy (Recommended: Vercel + Railway)
Use split deployment:
- `web` on Vercel
- `backend` on Railway (Dockerfile: `Dockerfile.backend`)

Authoritative guide:
- [docs/deploy-vercel-railway.md](docs/deploy-vercel-railway.md)

High-level order:
1. Deploy Railway backend first.
2. Set Railway envs (`BACKEND_API_KEY`, Supabase, LLM keys).
3. Deploy Vercel web with `PY_BACKEND_URL` + matching `PY_BACKEND_API_KEY`.
4. Run smoke tests (`/api/health`, `/api/analyze`, `/api/analyze/status`).

## Common Troubleshooting
### `Set PY_BACKEND_URL and PY_BACKEND_API_KEY`
Vercel web env is missing backend proxy settings.

### `Backend service unavailable`
Usually key mismatch:
- Vercel `PY_BACKEND_API_KEY` must equal Railway `BACKEND_API_KEY`.

### `supabaseUrl is required`
Set both:
- `NEXT_PUBLIC_SUPABASE_URL`
- `NEXT_PUBLIC_SUPABASE_ANON_KEY`

### Build fails on Docker `COPY data/ data/`
`data/` is gitignored. Use current Dockerfiles that create `data/` during build.

## Quality Checks
From repo root:
```bash
pytest tests/ -v
ruff check src/ tests/ backend/
```

From `web/`:
```bash
npm run lint
npm run build
```

## Contributing
1. Create a branch from `main`.
2. Keep API contracts stable (`web/src/lib/analyze-contracts.ts`).
3. Update docs if env vars or endpoints change.
4. Open a PR with:
   - what changed
   - why
   - verification steps/screenshots

## Security Notes
- Never commit `.env` files or private keys.
- Rotate any key that was ever pasted publicly.
- Keep backend auth enabled (`BACKEND_API_KEY`).

## License
No license file is currently included. Add one before open-source distribution.
