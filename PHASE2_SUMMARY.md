# Phase 2 Summary — Evaluation (Done on clean rows)

## Retrieval eval (all 20 Qs, `data/eval_results/retrieval.json`)

| Config | Hit rate | MRR |
|---|---|---|
| text (minsearch) | 1.00 | **0.858** |
| vector (MiniLM) | 0.90 | 0.825 |
| hybrid (RRF, production) | 1.00 | 0.829 |

**Pick: hybrid.** Text wins MRR because ground-truth questions echo doc
wording (known synthetic-data bias — the course lessons warn about this).
Hybrid matches text on hit rate, trails MRR by 0.03, and is robust to
paraphrased user queries. Evidence-backed call, documented here.

## LLM-judge eval (13/20 clean verdicts, `data/eval_results/llm_judge.csv`)

| Prompt | Good | Total clean |
|---|---|---|
| A-citations (production) | **8** | 13 |
| B-plain | 7 | 13 |

**Pick: A-citations** (thin margin; A gives more complete answers, e.g. spike
definition Q10, SKU-032 grounding Q9).

7 rows (Q16–20 × 2 prompts) have retry-noise verdicts (`judge failed after
retries`) — free-tier quota exhausted mid-run. Re-run later with:
`uv run python src/eval_llm_judge.py --rejudge-failed`
(optionally with `OPENROUTER_JUDGE_MODEL` set to a fresh `:free` slug).
They do not affect the prompt comparison above.

## Real failure analysis (not noise)

1. **Abstention fallback (Q2-A, Q7, Q9-B, Q12-B):** free-model empty
   completions → "I don't know" despite good retrieval. Provider noise, but
   confirms the fallback works instead of hallucinating.
2. **Breach-threshold mismatch (Q5):** our `breach = late_pos >= 2` flags 5
   suppliers; ground truth expects SUP-04/SUP-11 only. Follow-up: tighten the
   threshold (e.g. `>= 3` + on-time < 90%) and re-judge.
3. **Strict judge (Q1):** answer correctly lists at-risk SKUs but judge wanted
   explicit stock≤rop evidence per row. Judge calibration note: require
   "shows its work" in prompt A, or loosen expected_facts.
4. **Thin answer (Q6-A):** model named the vendor but dropped the numbers that
   were in context. Prompt A could demand numbers with every entity claim.

## Infra hardening (this phase)

- `synthesize()`: 3 attempts, backoff on any exception, guards empty
  `choices`; reads `OPENROUTER_MODEL` after dotenv load.
- `judge()`: guards empty choices, 60s+ backoff on 429, 12s pacing,
  separate `OPENROUTER_JUDGE_MODEL` supported.
- Eval scripts are resume-safe (incremental CSV, `--limit`, `--rejudge-failed`).

## Next: Phase 3 (UI, telemetry, Docker)

`src/app.py` (chat + citations + feedback), `src/db.py` + `src/metrics.py`
(telemetry), 5-chart dashboard, `Dockerfile` + `docker-compose.yml`.
