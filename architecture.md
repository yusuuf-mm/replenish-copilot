# Architecture — replenish-copilot

## System Flow

```text
User query (Streamlit)
        |
        v
Deterministic router (code, not LLM)
  - regex entity extract: SKU-\d{3}, SUP-\d{2}
  - keyword detect: stockout / reorder / SLA / delay / safety stock
        |
   +----+----+
   |         |
   v         v
SQLite route        Policy route (hybrid, local)
(relational)        (minsearch text + MiniLM vector 384-dim cosine top_k=5, RRF merge)
   |         |
   +----+----+
        |
        v
Context synthesis (facts table + policy chunks with doc_id)
        |
        v
OpenRouter Gemini-compatible call (minimax/minimax-m3:free, single pass)
        |
        v
Answer + citations -> telemetry log (conversations + feedback)
```

Parallel execution: both stores queried per request (threads); either may return empty.
If SQLite returns no entity match, answer from policies only (and vice versa).
If both empty -> "I don't know."

## Schemas

### SQLite — app data (`data/replenish.db`, built by `src/ingest.py`)

**inventory** (`sku_id` PK)

| column | type | notes |
|---|---|---|
| sku_id | TEXT PK | `SKU-001`..`SKU-075` |
| product_name | TEXT | |
| category | TEXT | Electronics/Hardware/Consumables/... |
| stock_on_hand | INTEGER | forced low for SKU-007/014/032 (eval) |
| reorder_point | INTEGER | at-risk when stock <= rop |
| safety_stock | INTEGER | critical when stock <= safety |
| unit_cost_usd | REAL | valuation + expedite math |
| supplier_id | TEXT FK | -> suppliers |

**suppliers** (`supplier_id` PK)

| column | type | notes |
|---|---|---|
| supplier_id | TEXT PK | `SUP-01`..`SUP-15` |
| supplier_name | TEXT | |
| lead_time_days | INTEGER | seeded from Kaggle avg; synthetic top-up random |
| sla_compliance_rate | REAL | SUP-04/SUP-11 seeded low (0.68-0.82) for SLA queries |
| contact_email | TEXT | mandatory before PO |

**purchase_orders**

| column | type | notes |
|---|---|---|
| po_id | TEXT PK | `PO-0001`.. |
| sku_id | TEXT FK | |
| supplier_id | TEXT FK | |
| order_qty | INTEGER | |
| order_date | TEXT (ISO) | |
| expected_delivery | TEXT (ISO) | breach = status late OR past expected + 2d grace |
| status | TEXT | delivered / late / pending |

**demand**

| column | type | notes |
|---|---|---|
| date | TEXT (ISO) | |
| sku_id | TEXT FK | |
| daily_demand | INTEGER | 70% sampled from Kaggle pool, 30% synthetic |
| forecasted_demand | REAL | |

### Policy index — SQLite-backed hybrid (`data/policy_index/`, built by `src/ingest.py`)

- Text search: minsearch index over `content` (text fields: `content`, `section_title`; keyword field: `doc_id`), persisted via sqlitesearch.
- Vector search: MiniLM `all-MiniLM-L6-v2` embeddings (384-dim, cosine) over the same chunks.
- Retrieval per query: text top-5 + vector top-5, merged with RRF (k=60) keyed on chunk id; exact `doc_id` mention boosted to top.
- Chunking: 500 chars, 100 overlap, per markdown section; payload/fields: `{doc_id, section_title, content}`.
- 6 docs: POL-001 safety stock, POL-002 SLA penalties, POL-003 stockout prioritization, POL-004 expedited shipping, POL-005 vendor onboarding, POL-006 valuation.
- No external vector DB — everything runs local (course-lesson pattern; keeps Docker + eval reproducible).

### Telemetry — SQLite (`data/telemetry.db`)

**conversations**: `id PK, question, answer, model, latency_s, prompt_tokens, completion_tokens, cost_usd, created_at`
**feedback**: `id PK, conversation_id FK, source (user/judge), score (+1/-1 or RELEVANT/PARTLY/NON_RELEVANT), note, created_at`

## Retrieval Mechanics

1. **Entity extraction (code)**: `re.findall(r"SKU-\d{3}", q)` + `re.findall(r"SUP-\d{2}", q)`.
   SQL uses `WHERE sku_id IN (...)` — never `LIKE`, never LLM-written SQL.
2. **Fact queries**: stock/rop/safety join inventory+suppliers; SLA query aggregates POs by supplier (`late` count, on-time rate); demand query averages last-N daily/forecast for cover-days = stock / avg_daily.
3. **Policy search (hybrid)**: minsearch text top-5 + MiniLM vector top-5 (cosine), RRF-merged (k=60); filter none (policies are global). Boost exact `doc_id` mention to top.
4. **Synthesis prompt**: fixed template — instructions (grounded, cite doc_id + sku numbers, "I don't know" fallback) + facts block + chunks block + user question. One OpenRouter call, temperature 0.0.
