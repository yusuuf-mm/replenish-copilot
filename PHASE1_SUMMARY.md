# Phase 1 Summary — Data & Retrieval Foundation (Done)

**Exit criteria:** one working end-to-end query returning a grounded answer
combining retrieved policy text and a computed fact. **Met.**

## What was built

| Piece | Status | Evidence |
|---|---|---|
| Problem statement + README skeleton | Done | `README.md` has problem, scope, quickstart |
| 6 policy docs with `doc_id` | Done (+1 fix) | `data/policies/POL-001..007` — POL-007 (demand-spike exceptions) added after live test showed Q8 had no covering doc |
| 4 synthetic CSVs (Kaggle-seeded) | Done | 75 SKUs / 15 suppliers / 150 POs / 600 demand rows, counts verified |
| `src/ingest.py` (SQLite + hybrid index) | Done | `data/replenish.db` rebuilt; 29 chunks in text + vector indexes |
| Analytics (stockout/SLA/cover-days) | Done | Merged into `src/rag.py` (see Decisions log in tracker) |
| `src/rag.py` (deterministic hybrid RAG) | Done | Entity regex + parallel SQLite/hybrid-policy retrieval + single OpenRouter call |
| Manual test, 5 sample questions | Done | Full grounded answer verified (see below) |

## Verified end-to-end answer

Q: *Why should SKU-014 be reordered now?*
A (abridged): stock 20 far below reorder point 259 and safety stock 69 —
at-risk and critical **[SKU-014]**; critical SKUs get immediate POs **[POL-003]**,
expedited when lead time > 7 days **[POL-004]**.

## Stack decisions (approved)

- Local hybrid retrieval (SQLite + minsearch text + MiniLM vector, RRF merge).
  No external vector DB.
- LLM: `google/gemma-4-26b-a4b-it:free` via OpenRouter (minimax-m3:free was
  delisted; model is overridable via `OPENROUTER_MODEL` in `.env`).
- Fixed bug: `OPENROUTER_MODEL` is now read after dotenv load so `.env`
  changes take effect.

## Known issues / watch items

- Free-tier models rate-limit (upstream 429) and reasoning models occasionally
  return empty completions — `synthesize()` retries once, then abstains.
- If huggingface.co is unreachable, embedding load hangs on retries: set
  `HF_HUB_OFFLINE=1` (model is cached locally after first download).
- `*.db` artefacts are gitignored; `ingest.py` rebuilds them (reproducibility
  holds via committed CSVs + policies).
- HF Hub warnings on first embedding load are harmless (cached model).

## Next: Phase 2 (Evaluation)

Freeze `data/ground_truth.csv` (~20 Q&A with `doc_id` labels), build
`src/eval_retrieval.py` (hit rate + MRR: text vs vector vs hybrid) and
`src/eval_llm_judge.py` (good/bad + reasoning).
