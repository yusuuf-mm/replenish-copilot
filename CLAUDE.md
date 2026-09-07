# CLAUDE.md — replenish-copilot

## Stack

- Python 3.11+, Streamlit, SQLite (+ sqlitesearch), minsearch, Pytest
- LLM via OpenRouter API: chat model `minimax/minimax-m3:free`
- Embeddings (local, free): `sentence-transformers all-MiniLM-L6-v2` (384-dim, cosine)
- Docker + docker-compose for app shipping (no external vector DB — all retrieval is local)

## Execution Commands (uv — no pip, no manual venv)

```bash
uv sync                          # create .venv + install from pyproject.toml / uv.lock

uv run python scripts/generate_data.py   # build data/csv/*.csv + data/policies/*.md (seeded from supply_chain_dataset1.csv when present)
uv run python src/ingest.py              # CSV -> SQLite + policy chunks -> minsearch/SQLite index
uv run streamlit run src/app.py          # chat UI + telemetry dashboard

uv run pytest -q                # unit + retrieval tests
docker-compose up --build         # app container (needs OPENROUTER_API_KEY set; no external services)
```

Env (`.env`, never commit):

```text
OPENROUTER_API_KEY=sk-or-...
OPENROUTER_MODEL=minimax/minimax-m3:free
```

## Architectural Rules

1. **Modular `src/` layout**: `ingest.py`, `rag.py`, `eval_retrieval.py`, `eval_llm_judge.py`, `app.py`, `db.py`, `metrics.py`. No logic in notebooks.
2. **Type hints mandatory** on all public functions (`def f(x: str) -> list[dict]:`).
3. **Deterministic hybrid retrieval only**:
   - SQLite route = tabular/entity facts (SKU-XXX, SUP-XX extracted via regex, never guessed by LLM).
   - Policy route = hybrid: minsearch text search + MiniLM vector search (384-dim, cosine, top_k=5), merged with RRF; sqlitesearch for persistence.
   - Merge in code, then single LLM synthesis call. No multi-turn tool use.
4. **Zero non-deterministic agent loops**: no `while` LLM-decides-tools loops, no function-calling retries. One pass: extract -> query both stores in parallel -> synthesize -> log.
5. **Stable IDs**: `doc_id` (`POL-00X`) in every policy header; `sku_id` (`SKU-XXX`), `supplier_id` (`SUP-XX`) verbatim from CSV. Eval joins on these — never rename mid-pipeline.
6. **Grounding**: every answer cites `doc_id`/table row used; empty retrieval -> respond "I don't know" (no hallucination).
