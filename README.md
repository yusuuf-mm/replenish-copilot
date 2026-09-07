# replenish-copilot

RAG-based inventory replenishment & supplier-delay decision-support copilot
(LLM Zoomcamp capstone). Deterministic dual-retrieval: SQLite facts + Qdrant policy search -> single-pass synthesis via OpenRouter (`minimax/minimax-m3:free`).

## Quickstart

```bash
uv sync
cp .env.example .env   # set OPENROUTER_API_KEY, QDRANT_HOST/PORT

uv run python scripts/generate_data.py   # seed data (uses supply_chain_dataset1.csv when present)
uv run python src/ingest.py              # load SQLite + index Qdrant
uv run streamlit run src/app.py
```

Docker (Qdrant runs externally — start it first):

```bash
docker-compose up --build
```

## How it works

1. Router extracts `SKU-XXX` / `SUP-XX` via regex (code, not LLM).
2. Parallel retrieval: SQLite (stock, reorder, SLA, demand facts) + Qdrant `policy_chunks` (MiniLM 384-dim, cosine, top 5).
3. Fixed prompt merges facts + cited chunks -> one OpenRouter call (temp 0.0).
4. Every answer logs to telemetry (`conversations` + `feedback`); dashboard shows Volume, Latency p50/p95, Cost, Feedback, Relevance.

No agent loops — full flow is a single deterministic pass.

## Data

- `supply_chain_dataset1.csv` (repo root, Kaggle, 91k rows) seeds realistic values: 50 real SKUs + 10 real supplier lead-times sampled into `data/csv/`; topped up synthetically to 75 SKUs / 15 suppliers / 150 POs / 600 demand rows.
- `data/policies/` holds 6 versioned SOPs (`POL-001`..`POL-006`), each with a `doc_id` header used as the eval join key.

## Evaluation

- Retrieval: Hit Rate@k + MRR, text vs vector vs hybrid (`src/eval_retrieval.py`).
- Answers: LLM-as-a-Judge good/bad + reasoning (`src/eval_llm_judge.py`).
- See `architecture.md` (design) and `plan.md` (4-phase build plan).
