# PROJECT_PLAN.md — replenish-copilot, 3-Day Execution Tracker

Passing threshold: 11/26. Target: all 18 core points, then bonus points only if ahead
of schedule. Update the status column as work completes — do not skip ahead while a
row above is still "Not started" or "Blocked."

## Day 1 — Data & Retrieval Foundation

| # | Task | Rubric criterion | Status |
|---|---|---|---|
| 1 | Problem statement + README skeleton | Problem Description (2) | Not started |
| 2 | Write 5–8 policy markdown docs | Retrieval Flow (2) | Not started |
| 3 | Generate synthetic CSVs (inventory, suppliers, POs, demand) | Retrieval Flow (2) | Not started |
| 4 | Build `ingest.py`: load, chunk, embed, index policy docs | Ingestion Pipeline (2) | Not started |
| 5 | Load CSVs into SQLite | Retrieval Flow (2) | Not started |
| 6 | Build `analytics.py`: reorder point, stockout risk, SLA breach calcs | Retrieval Flow (2) | Not started |
| 7 | Implement base RAG flow: query → retrieve → facts → prompt → LLM → answer | Retrieval Flow (2) | Not started |
| 8 | Manual test with 5 sample questions | — | Not started |

**Day 1 exit criteria:** one working end-to-end query returning a grounded answer
combining retrieved policy text and a computed fact.

## Day 2 — Evaluation

| # | Task | Rubric criterion | Status |
|---|---|---|---|
| 1 | Build 15–20 gold-standard Q&A pairs | Retrieval/LLM Evaluation (4) | Not started |
| 2 | Implement hit rate + MRR scoring | Retrieval Evaluation (2) | Not started |
| 3 | Compare at least two retrieval configs, pick winner | Retrieval Evaluation (2) | Not started |
| 4 | Implement LLM-as-judge scoring | LLM Evaluation (2) | Not started |
| 5 | Compare at least two prompts/models, pick winner | LLM Evaluation (2) | Not started |
| 6 | (If ahead) query rewriting | Bonus +1 | Not started |
| 7 | (If ahead) hybrid search + RRF | Bonus +1 | Not started |

**Day 2 exit criteria:** a results table (hit rate/MRR per config, judge scores per
prompt/model) ready to paste into the README.

## Day 3 — Interface, Monitoring, Containerization

| # | Task | Rubric criterion | Status |
|---|---|---|---|
| 1 | Streamlit chat UI with source citations | Interface (2) | Not started |
| 2 | Thumbs up/down feedback capture | Monitoring (2) | Not started |
| 3 | Dashboard with 5+ charts | Monitoring (2) | Not started |
| 4 | `docker-compose.yml` covering app + all services | Containerization (2) | Not started |
| 5 | README finalization: setup, architecture, eval results | Reproducibility (2) | Not started |
| 6 | Full reproducibility test: clone → compose up → working app | Reproducibility (2) | Not started |
| 7 | (If ahead) cloud deploy | Bonus +2 | Not started |
| 8 | (If ahead) re-ranking | Bonus +1 | Not started |

**Day 3 exit criteria:** complete, dockerized, reproducible submission.

## Risk log
*(Claude Code: append here immediately if any core rubric item looks at risk — do not
wait until end of day to surface it.)*

## Decisions log
*(Claude Code: append here any deviation from ARCHITECTURE.md, with a one-line reason.)*