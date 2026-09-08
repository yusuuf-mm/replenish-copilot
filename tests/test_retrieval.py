"""Offline unit tests — no network, no API keys (Phase 3)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.eval_retrieval import hit_rate, mrr  # noqa: E402
from src.metrics import estimate_cost  # noqa: E402
from src.rag import extract_entities, rrf_merge  # noqa: E402


def test_extract_entities() -> None:
    """Regex extracts SKU/SUP/doc IDs and intents."""
    ent = extract_entities("Why should SKU-014 from SUP-04 be reordered? See POL-003.")
    assert ent["skus"] == ["SKU-014"]
    assert ent["suppliers"] == ["SUP-04"]
    assert ent["docs"] == ["POL-003"]
    assert "reorder" in ent["intents"]


def test_extract_entities_empty() -> None:
    """No entities -> empty lists (at-risk fallback path)."""
    ent = extract_entities("Which SKUs are at stockout risk?")
    assert ent["skus"] == [] and ent["suppliers"] == []
    assert "stockout" in ent["intents"]


def test_hit_rate() -> None:
    """Hit when a 1 appears anywhere."""
    assert hit_rate([[1, 0], [0, 0], [0, 1]]) == 2 / 3


def test_mrr() -> None:
    """Rank-1 scores 1.0, rank-2 scores 0.5, miss scores 0."""
    assert mrr([[1, 0], [0, 1], [0, 0]]) == (1 + 0.5 + 0) / 3


def test_rrf_merge_prefers_overlap() -> None:
    """A chunk in both lists outranks single-list chunks."""
    a = {"chunk_id": "x", "content": "x" * 61}
    b = {"chunk_id": "y", "content": "y" * 61}
    merged = rrf_merge([a, b], [a])
    assert merged[0]["chunk_id"] == "x"


def test_estimate_cost_free_is_zero() -> None:
    """Free-tier models bill nothing."""
    assert estimate_cost("google/gemma-4-26b-a4b-it:free", 2000, 500) == 0.0
