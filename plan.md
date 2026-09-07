# Plan — replenish-copilot (4 phases)

## Phase 1 — Data Engineering & Ingestion Setup

- [ ] Run `uv run python scripts/generate_data.py` (Kaggle-seeded; 75 SKUs / 15 suppliers / 150 POs / 600 demand / 6 policies with `doc_id` headers).
- [ ] Write `src/ingest.py`: create SQLite schemas per `architecture.md`, bulk-load 4 CSVs, chunk policies (500/100), embed with MiniLM, build minsearch text + vector indexes (persist via sqlitesearch).
- [ ] Acceptance: `data/replenish.db` row counts match CSVs; hybrid index returns hits for spot-check queries; `ingest.py` re-runnable (idempotent).

## Phase 2 — Dual-RAG Core Engine (`src/rag.py`)

- [ ] Implement deterministic router: regex SKU/SUP extract + keyword intent (stockout/reorder/SLA/safety/valuation).
- [ ] Implement `query_sqlite()` (facts: stock vs rop/safety, SLA aggregates, cover-days) + `query_qdrant()` (top_k=5) run in parallel via `ThreadPoolExecutor`.
- [ ] Wire single-pass OpenRouter synthesis (`minimax/minimax-m3:free`, temp 0.0, fixed template with citations + "I don't know" fallback).
- [ ] Validate on 8 baseline queries: stockout list, SKU-014 why-reorder, SLA violators, planner action, policy-grounded reorder, SKU-032 safety math, worst on-time suppliers, demand-spike SOP.
- [ ] Acceptance: all 8 return grounded answers with `doc_id`/row citations, no agent loop, latency logged.

## Phase 3 — Offline & LLM-as-a-Judge Evaluation

- [ ] Freeze `data/ground_truth.csv` (~20 rows: `question, doc_id, expected_facts`) — doc_id is the join label.
- [ ] Build `src/eval_retrieval.py`: `compute_relevance()` -> Hit Rate@k + MRR via `evaluate(ground_truth, search_fn)`; compare text (minsearch) vs vector (MiniLM) vs hybrid (RRF merge); grid-tune boosts + `top_k`; save `data/eval_results/retrieval.json`.
- [ ] Build `src/eval_llm_judge.py`: OpenRouter judge scores each RAG answer (`good/bad` + reasoning; relevance + faithfulness); save `data/eval_results/llm_judge.csv`; read bad rows to fix prompt/retrieval.
- [ ] Acceptance: best retrieval config selected by numbers (not vibes); judge pass rate reported in README.

## Phase 4 — UI, Telemetry Dashboard & Docker Deployment

- [ ] `src/app.py` (Streamlit): chat input, answer + source citations (doc_id, SKU numbers), +1/-1 + text feedback, writes `conversations` + `feedback` rows; per-call judge (sampled) stored as `source=judge`.
- [ ] `src/db.py` + `src/metrics.py`: `LLMCallRecord` (model, tokens, latency, cost), `save_conversation() RETURNING id`, `save_feedback()`.
- [ ] Dashboard tab with 5 charts: Request Volume, Latency p50/p95, Cost Tracking, Feedback Score (+1/-1 split), Online Relevance (judge label distribution) + recent conversations table.
- [ ] `Dockerfile` + `docker-compose.yml` (single app container; all retrieval is local SQLite/minsearch): `docker-compose up --build`, verify cold-start `ingest` + 8 queries pass.
- [ ] Finalize README (setup, arch diagram, eval numbers, repro steps). Acceptance: clone -> compose up -> working demo.
