"""LLM-as-a-Judge evaluation (Day 2, tasks 4+5).

Compares two synthesis prompts (A: citations-required, B: plain) by generating
answers for ground_truth.csv and judging each vs expected_facts
(good/bad + reasoning). Writes incrementally — safe to resume.

Run: uv run python src/eval_llm_judge.py [--limit N]
Writes: data/eval_results/llm_judge.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from dotenv import load_dotenv  # noqa: E402
from openai import OpenAI  # noqa: E402

from src.rag import INSTRUCTIONS, retrieve, synthesize  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval_results" / "llm_judge.csv"

PROMPT_B = """Answer the question using the facts and policy chunks below.
Be concise."""

JUDGE_INSTRUCTIONS = """You grade a RAG answer against expected facts.
Reply with exactly two lines:
score: good|bad
reasoning: one sentence why"""

JUDGE_TEMPLATE = """Question: {question}
Expected facts: {expected}
Answer: {answer}"""


def get_client() -> tuple[OpenAI, str]:
    """OpenRouter client + model from .env."""
    load_dotenv(ROOT / ".env")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY missing — add it to .env")
    model = os.environ.get("OPENROUTER_MODEL", "")
    return OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key), model


def judge(client: OpenAI, model: str, question: str, expected: str,
          answer: str) -> tuple[str, str]:
    """Return (score, reasoning) with one retry on empty verdicts."""
    for _ in range(3):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.0,
                messages=[
                    {"role": "system", "content": JUDGE_INSTRUCTIONS},
                    {"role": "user", "content": JUDGE_TEMPLATE.format(
                        question=question, expected=expected, answer=answer)},
                ],
            )
            text = ""
            choices = resp.choices or []
            if choices and choices[0].message.content:
                text = choices[0].message.content.strip()
            score = "good" if "score: good" in text.lower() else (
                "bad" if "score: bad" in text.lower() else "")
            reason = ""
            for line in text.splitlines():
                if line.lower().startswith("reasoning:"):
                    reason = line.split(":", 1)[1].strip()
            if score:
                return score, reason
        except Exception as e:  # noqa: BLE001 — free-tier flakiness; back off
            print(f"judge retry ({e.__class__.__name__})", flush=True)
            time.sleep(10)
        time.sleep(2)
    return "bad", "judge failed after retries"


def load_done() -> set[tuple[str, str]]:
    """(question, prompt) pairs already recorded — resume support."""
    if not OUT.exists():
        return set()
    with open(OUT, newline="", encoding="utf-8") as f:
        return {(r["question"], r["prompt"]) for r in csv.DictReader(f)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    with open(ROOT / "data" / "ground_truth.csv", newline="",
              encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]

    client, model = get_client()
    done = load_done()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    new_file = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["question", "doc_id", "prompt",
                                          "answer", "score", "reasoning"])
        if new_file:
            w.writeheader()
        for i, row in enumerate(rows, 1):
            ctx = retrieve(row["question"])
            for name, instr in [("A-citations", INSTRUCTIONS),
                                ("B-plain", PROMPT_B)]:
                if (row["question"], name) in done:
                    continue
                answer = synthesize(row["question"], ctx, instructions=instr)
                score, reason = judge(client, model, row["question"],
                                      row["expected_facts"], answer)
                w.writerow({"question": row["question"], "doc_id": row["doc_id"],
                            "prompt": name, "answer": answer, "score": score,
                            "reasoning": reason})
                f.flush()
                safe_reason = reason[:80].encode("ascii", "replace").decode()
                print(f"[{i}/{len(rows)}] {name}: {score} - {safe_reason}",
                      flush=True)
    print(f"Saved -> {OUT}")


if __name__ == "__main__":
    main()
