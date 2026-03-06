# 2-Week DevRel Curriculum (Intermediate Python)

This curriculum is implementation-ready: each day includes what to read,
what to run, what to produce, and the pass criteria.

## Day 1: Architecture and Runtime Map

### Focus
Understand the full request/data flow and service boundaries.

### Read
- `README.md`
- `docker-compose.yml`
- `docs/deploy-vercel-railway.md`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "Architecture|Repository Map|Quick Start|Deploy" README.md docs/deploy-vercel-railway.md
cat docker-compose.yml
```

### Produce
Create a one-page architecture explainer with:
- request path (Web -> Backend -> LLM/Data providers)
- deployment split (Vercel + Railway + Supabase)
- where auth, payments, and monitoring happen

### Exit criteria
You can explain why backend compute is separated from the public web tier.

---

## Day 2: Python Foundations in This Codebase

### Focus
Learn the Python patterns used here, not generic syntax.

### Read
- `main.py`
- `src/config.py`
- `src/agents/fetcher.py`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "class |@dataclass|Enum|def main\(|argparse|Settings\(" main.py src/config.py src/agents/fetcher.py
```

### Produce
Build a "Python patterns used here" cheat sheet covering:
- dataclasses and enums
- type hints and return contracts
- settings/env aliasing with Pydantic
- CLI command/subcommand style

### Exit criteria
You can point to at least three concrete examples of type-safe Python design.

---

## Day 3: Backend API Design and Contracts

### Focus
Master async job API behavior and validation boundaries.

### Read
- `backend/app.py`
- `backend/schemas.py`
- `backend/jobs.py`
- `web/src/lib/analyze-contracts.ts`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "@api\.(get|post)\(" backend/app.py
rg -n "AnalyzeJob" backend/schemas.py web/src/lib/analyze-contracts.ts
```

### Produce
Write an API flow card with:
- submit job endpoint
- status polling endpoint
- terminal states and error handling

### Exit criteria
You can explain why this job manager uses thread pool + TTL cleanup.

---

## Day 4: LLM Orchestration and Reliability

### Focus
Explain how LLM calls stay safe, deterministic, and cost-visible.

### Read
- `src/agents/analyzer.py`
- `src/agents/chat_assistant.py`
- `src/guardrails.py`
- `src/token_accounting.py`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "DEFAULT_TASK_ROUTES|_models_for_task|fallback|guardrail|record_llm_call" src/agents/analyzer.py src/agents/chat_assistant.py src/guardrails.py src/token_accounting.py
pytest tests/test_llm_routing.py -v
```

### Produce
Create a 10-minute demo segment:
- model routing by task
- fallback behavior
- guardrail checks
- token/cost meta transparency

### Exit criteria
You can explain how routing choices are tested and why they are predictable.

---

## Day 5: Observability and Operations

### Focus
Translate telemetry into operational decisions.

### Read
- `src/metrics.py`
- `src/logging_config.py`
- `src/cache.py`
- `monitoring/prometheus.yml`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "Counter\(|Histogram\(|Gauge\(|CACHE_|configure_logging" src/metrics.py src/cache.py src/logging_config.py
```

### Produce
Draft an operator runbook page with:
- key metrics
- likely incident signals
- first triage steps

### Exit criteria
You can explain one cache win scenario and one stale-data risk scenario.

---

## Day 6: Data Layer and RAG

### Focus
Understand how storage + embeddings support retrieval-driven chat.

### Read
- `src/agents/storage.py`
- `src/agents/rag_retriever.py`
- `migrations/001_stock_schema.sql`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "class StockStorage|embed|vector|store_analysis_with_embedding|semantic" src/agents/storage.py src/agents/rag_retriever.py
sed -n '1,220p' migrations/001_stock_schema.sql
```

### Produce
Write a "How RAG works in Stock-Radar" walkthrough:
- what gets stored
- what gets embedded
- what gets retrieved at ask time

### Exit criteria
You can explain retrieval context sources and fallback behavior when data is sparse.

---

## Day 7: ML Pipeline Fundamentals

### Focus
Explain training from dataset creation to model artifacts.

### Read
- `src/training/dataset_builder.py`
- `src/training/train.py`
- `src/training/feature_engineering.py`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "extract_features|walk-forward|optuna|GradientBoosting|metadata|feature" src/training/train.py src/training/dataset_builder.py src/training/feature_engineering.py
```

### Produce
Prepare workshop section "raw data -> features -> trained model" with one simple diagram.

### Exit criteria
You can explain why purged walk-forward split reduces leakage.

---

## Day 8: Inference, Risk, and Backtesting

### Focus
Connect prediction outputs to trading/risk realism.

### Read
- `src/training/predictor.py`
- `src/training/regime.py`
- `src/training/backtesting.py`
- `src/training/pre_trade_risk.py`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "predict\(|position_size|regime|slippage|liquidity|risk" src/training/predictor.py src/training/regime.py src/training/backtesting.py src/training/pre_trade_risk.py
pytest tests/test_backtesting.py tests/test_regime_risk.py tests/test_phase10.py -v
```

### Produce
Create advanced Q&A notes for:
- position sizing rationale
- slippage and liquidity limits
- pre-trade gate examples

### Exit criteria
You can defend why backtest realism controls are necessary for go-live trust.

---

## Day 9: Cross-Stack DevRel Narrative

### Focus
Tell a coherent story across web, backend, and on-chain components.

### Read
- `web/src/app/api/analyze/route.ts`
- `web/src/lib/backend-client.ts`
- `move-agent-registry/sources/agent_registry.move`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
rg -n "backendRequest|X-Backend-Api-Key|statusUrl|AnalyzeJob" web/src/lib/backend-client.ts web/src/app/api/analyze/route.ts
rg -n "register_agent|record_request|submit_rating" move-agent-registry/sources/agent_registry.move
```

### Produce
Create your 45-60 minute workshop outline:
- problem and architecture
- Python backend internals
- AI/ML production controls
- web + chain integration

### Exit criteria
You can explain why chain components are optional for core analysis flow but valuable for agent economy features.

---

## Day 10: Final Rehearsal and Packaging

### Focus
Convert knowledge into a stable workshop package.

### Read
- `tests/test_llm_routing.py`
- `tests/test_phase10.py`
- `tests/test_training.py`
- `docs/devrel/workshop-pack.md`

### Execute
```bash
cd /Users/darshann/Desktop/stock-radar
./scripts/devrel_validate.sh quick
# Optional final gate:
# ./scripts/devrel_validate.sh full
```

### Produce
Complete all sections in `workshop-pack.md`:
- slides outline
- demo run script
- API handout
- FAQ
- troubleshooting appendix

### Exit criteria
You can run one uninterrupted end-to-end dry run with clear narration and fallback talking points.

---

## Daily Validation Rule

Run at the end of each day:

```bash
cd /Users/darshann/Desktop/stock-radar
./scripts/devrel_validate.sh quick
```

If quick validation fails, fix understanding gaps before starting the next day.
