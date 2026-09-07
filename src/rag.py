"""Deterministic hybrid RAG engine (Day 1, tasks 6+7).

One pass, no agent loops:
  extract entities (regex) -> SQLite facts + hybrid policy search (parallel)
  -> single OpenRouter synthesis call -> grounded answer with citations.

Run retrieval-only check without an API key:
  uv run python src/rag.py "Which SKUs are at stockout risk?"
Full answers need OPENROUTER_API_KEY in .env.
"""

from __future__ import annotations

import os
import re
import sqlite3
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "replenish.db"
INDEX_DIR = ROOT / "data" / "policy_index"
TEXT_DB = INDEX_DIR / "text.db"
VECTOR_DB = INDEX_DIR / "vectors.db"

MODEL = os.environ.get("OPENROUTER_MODEL", "minimax/minimax-m3:free")
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
RRF_K = 60

INSTRUCTIONS = """You are a supply-chain replenishment assistant. Answer ONLY from the
facts and policy chunks below. Cite every claim as [SKU-XXX] or [POL-00X].
If the context does not contain the answer, respond exactly: I don't know."""

PROMPT_TEMPLATE = """Facts:
{facts}

Policy chunks:
{chunks}

Question: {question}
Answer with citations:"""

SKU_RE = re.compile(r"SKU-\d{3}")
SUP_RE = re.compile(r"SUP-\d{2}")
DOC_RE = re.compile(r"POL-\d{3}")
INTENTS = ("stockout", "reorder", "sla", "delay", "late", "safety",
           "expedit", "valuation", "onboard", "forecast", "demand")


def extract_entities(question: str) -> dict[str, list[str]]:
    """Extract SKU/SUP/doc IDs + keyword intents via regex (code, not LLM)."""
    return {
        "skus": sorted(set(SKU_RE.findall(question))),
        "suppliers": sorted(set(SUP_RE.findall(question))),
        "docs": sorted(set(DOC_RE.findall(question))),
        "intents": [w for w in INTENTS if w in question.lower()],
    }


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def stock_facts(skus: list[str]) -> list[dict]:
    """Per-SKU stock vs reorder/safety + supplier + cover-days."""
    conn = _connect()
    try:
        if skus:
            rows = conn.execute(
                "SELECT i.*, s.supplier_name, s.lead_time_days FROM inventory i "
                "JOIN suppliers s ON i.supplier_id = s.supplier_id "
                f"WHERE i.sku_id IN ({','.join('?' for _ in skus)})", skus,
            ).fetchall()
        else:  # no entity -> at-risk list
            rows = conn.execute(
                "SELECT i.*, s.supplier_name, s.lead_time_days FROM inventory i "
                "JOIN suppliers s ON i.supplier_id = s.supplier_id "
                "WHERE i.stock_on_hand <= i.reorder_point "
                "ORDER BY i.stock_on_hand - i.safety_stock LIMIT 10",
            ).fetchall()
        facts: list[dict] = []
        for r in rows:
            d = dict(r)
            avg = conn.execute(
                "SELECT AVG(daily_demand) FROM demand WHERE sku_id = ?",
                (d["sku_id"],)).fetchone()[0] or 1.0
            d["cover_days"] = round(d["stock_on_hand"] / max(avg, 0.1), 1)
            d["at_risk"] = d["stock_on_hand"] <= d["reorder_point"]
            d["critical"] = d["stock_on_hand"] <= d["safety_stock"]
            facts.append(d)
        return facts
    finally:
        conn.close()


def sla_facts(suppliers: list[str]) -> list[dict]:
    """Per-supplier PO aggregates + breach flags."""
    conn = _connect()
    try:
        if suppliers:
            where = f"WHERE p.supplier_id IN ({','.join('?' for _ in suppliers)})"
            params: tuple = tuple(suppliers)
        else:
            where, params = "", ()
        rows = conn.execute(
            "SELECT p.supplier_id, s.supplier_name, COUNT(*) AS pos, "
            "SUM(CASE WHEN p.status='late' THEN 1 ELSE 0 END) AS late_pos "
            "FROM purchase_orders p JOIN suppliers s "
            "ON p.supplier_id = s.supplier_id "
            f"{where} GROUP BY p.supplier_id ORDER BY late_pos DESC", params,
        ).fetchall()
        return [dict(r) | {"breach": (r["late_pos"] or 0) >= 2} for r in rows]
    finally:
        conn.close()


def policy_search(question: str, top_k: int = TOP_K) -> list[dict]:
    """Hybrid: minsearch text top-k + MiniLM vector top-k, RRF-merged."""
    from sqlitesearch import TextSearchIndex, VectorSearchIndex

    text_index = TextSearchIndex(
        text_fields=["content", "section_title"],
        keyword_fields=["doc_id"],
        db_path=str(TEXT_DB),
    )
    text_hits = text_index.search(question, num_results=top_k)
    text_index.close()

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(EMBED_MODEL)
    qvec = model.encode(question)
    vec_index = VectorSearchIndex(
        keyword_fields=["doc_id"], mode="lsh", db_path=str(VECTOR_DB))
    vec_hits = vec_index.search(qvec, num_results=top_k)
    vec_index.close()

    fused: dict[str, tuple[float, dict]] = {}
    for rank, h in enumerate(text_hits):
        key = h.get("chunk_id", h["content"][:60])
        fused[key] = (fused.get(key, (0.0, h))[0] + 1 / (RRF_K + rank + 1), h)
    for rank, h in enumerate(vec_hits):
        key = h.get("chunk_id", h["content"][:60])
        fused[key] = (fused.get(key, (0.0, h))[0] + 1 / (RRF_K + rank + 1), h)
    ranked = sorted(fused.values(), key=lambda t: -t[0])
    docs = [d for d in DOC_RE.findall(question)]
    if docs:  # exact doc_id mention jumps to top
        ranked.sort(key=lambda t: t[1].get("doc_id") not in docs)
    return [h for _, h in ranked[:top_k]]


def retrieve(question: str) -> dict:
    """Parallel SQLite facts + policy search. No LLM call."""
    ent = extract_entities(question)
    with ThreadPoolExecutor(max_workers=2) as pool:
        f_stock = pool.submit(stock_facts, ent["skus"])
        f_sla = pool.submit(sla_facts, ent["suppliers"])
        f_pol = pool.submit(policy_search, question)
        return {"entities": ent, "stock": f_stock.result(),
                "sla": f_sla.result(), "policies": f_pol.result()}


def build_prompt(question: str, ctx: dict) -> str:
    """Assemble the fixed synthesis prompt from retrieved context."""
    fact_lines = [f"{k}={r}" for k in ("stock", "sla") for r in ctx[k]]
    chunk_lines = [f"[{c.get('doc_id')}] {c.get('section_title')}: "
                   f"{c.get('content', '')[:400]}" for c in ctx["policies"]]
    return PROMPT_TEMPLATE.format(
        facts="\n".join(fact_lines) or "(no tabular facts)",
        chunks="\n".join(chunk_lines) or "(no policy chunks)",
        question=question,
    )


def synthesize(question: str, ctx: dict) -> str:
    """Single OpenRouter call (temperature 0.0). Needs OPENROUTER_API_KEY."""
    from dotenv import load_dotenv
    from openai import OpenAI

    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing — add it to .env")
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
    resp = client.chat.completions.create(
        model=MODEL, temperature=0.0,
        messages=[{"role": "system", "content": INSTRUCTIONS},
                  {"role": "user", "content": build_prompt(question, ctx)}],
    )
    return resp.choices[0].message.content or "I don't know."


def rag(question: str) -> dict:
    """Full pipeline: retrieve -> synthesize. Returns answer + sources."""
    ctx = retrieve(question)
    return {"answer": synthesize(question, ctx), "sources": ctx}


if __name__ == "__main__":
    q = sys.argv[1] if len(sys.argv) > 1 else "Which SKUs are at stockout risk?"
    ctx = retrieve(q)
    print(f"entities: {ctx['entities']}")
    print(f"stock rows: {len(ctx['stock'])}, sla rows: {len(ctx['sla'])}, "
          f"policy chunks: {[c.get('doc_id') for c in ctx['policies']]}")
    for r in ctx["stock"][:5]:
        print(f"  {r['sku_id']} stock={r['stock_on_hand']} rop={r['reorder_point']} "
              f"cover={r['cover_days']}d at_risk={r['at_risk']}")
