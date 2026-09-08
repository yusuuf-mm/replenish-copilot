"""Per-call telemetry receipt (Phase 3).

LLMCallRecord is the unit everything downstream consumes: what went in/out,
what it cost, how long the user waited. Cost is 0.0 for :free models.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class LLMCallRecord:
    model: str
    question: str
    answer: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_s: float = 0.0
    cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def estimate_cost(model: str, prompt_tokens: int,
                  completion_tokens: int) -> float:
    """USD cost for one call. Free-tier models bill 0."""
    if ":free" in model or not model:
        return 0.0
    return (prompt_tokens * 1.0 + completion_tokens * 3.0) / 1_000_000


def timed_call(model: str, question: str,
               fn: Callable[[], tuple[str, dict[str, int]]]) -> tuple[str, LLMCallRecord]:
    """Run fn() -> (answer, token_counts), return answer + filled record."""
    start = time.time()
    answer, tokens = fn()
    latency = time.time() - start
    pt = tokens.get("prompt_tokens", 0)
    ct = tokens.get("completion_tokens", 0)
    return answer, LLMCallRecord(
        model=model, question=question, answer=answer,
        prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct,
        latency_s=round(latency, 2),
        cost_usd=estimate_cost(model, pt, ct),
    )


def to_dict(record: LLMCallRecord) -> dict[str, Any]:
    """Plain dict for storage/display."""
    return asdict(record)
