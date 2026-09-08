"""Retrieval evaluation (Day 2, tasks 2+3).

Compares text-only vs vector-only vs hybrid (RRF) policy search on the frozen
ground_truth.csv using hit rate and MRR on doc_id labels.

Run: uv run python src/eval_retrieval.py
Writes: data/eval_results/retrieval.json
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.rag import TOP_K, policy_search_text, policy_search_vector, rrf_merge  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUND_TRUTH = ROOT / "data" / "ground_truth.csv"
RESULTS_DIR = ROOT / "data" / "eval_results"


def load_ground_truth() -> list[dict]:
    """Load frozen (question, doc_id, expected_facts) rows."""
    with open(GROUND_TRUTH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def search_hybrid(question: str, top_k: int = TOP_K) -> list[dict]:
    """RRF merge of text + vector top-k (same as production policy_search)."""
    return rrf_merge(policy_search_text(question, top_k),
                     policy_search_vector(question, top_k), top_k)


def relevance(doc_id: str, hits: list[dict]) -> list[int]:
    """1 where the hit's doc_id matches ground truth, else 0, in rank order."""
    return [int(h.get("doc_id") == doc_id) for h in hits]


def hit_rate(all_rel: list[list[int]]) -> float:
    """Share of queries with the right doc anywhere in the list."""
    return sum(1 for r in all_rel if 1 in r) / len(all_rel)


def mrr(all_rel: list[list[int]]) -> float:
    """Mean reciprocal rank of the first correct doc (0 on miss)."""
    total = 0.0
    for r in all_rel:
        for rank, v in enumerate(r):
            if v == 1:
                total += 1 / (rank + 1)
                break
    return total / len(all_rel)


def evaluate(rows: list[dict], search_fn: Callable[[str], list[dict]],
             top_k: int = TOP_K) -> dict:
    """Score one search config on the frozen set."""
    all_rel = [relevance(r["doc_id"], search_fn(r["question"], top_k))
               for r in rows]
    return {"hit_rate": round(hit_rate(all_rel), 3),
            "mrr": round(mrr(all_rel), 3), "n": len(rows), "top_k": top_k}


def main() -> None:
    rows = load_ground_truth()
    print(f"Ground truth: {len(rows)} questions")
    results: dict[str, dict] = {}
    for name, fn in [("text", policy_search_text),
                     ("vector", policy_search_vector),
                     ("hybrid", search_hybrid)]:
        results[name] = evaluate(rows, fn)
        print(f"{name:>6}: {results[name]}")
    best = max(results, key=lambda k: (results[k]["mrr"], results[k]["hit_rate"]))
    results["winner"] = best
    print(f"Winner: {best}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "retrieval.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved -> {RESULTS_DIR / 'retrieval.json'}")


if __name__ == "__main__":
    main()
