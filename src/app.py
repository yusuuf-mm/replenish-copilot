"""Streamlit UI: chat + telemetry dashboard (Phase 3).

Run: uv run streamlit run src/app.py
Tab 1 (Chat): conversational Q&A with citations, metrics, +1/-1 feedback.
Tab 2 (Dashboard): 5 charts (volume, latency, cost, feedback, relevance).
"""

from __future__ import annotations

import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st  # noqa: E402

from src import db  # noqa: E402
from src.metrics import LLMCallRecord, estimate_cost  # noqa: E402
from src.rag import MODEL, retrieve, synthesize  # noqa: E402

JUDGE_PROMPT = """Rate this RAG answer's relevance to the question.
Reply with one word: RELEVANT, PARTLY_RELEVANT, or NON_RELEVANT.
Question: {question}
Answer: {answer}"""

CSS = """
<style>
/* Theme-adaptive: transparent surfaces, translucent tints, inherited text.
   Hardcoded light backgrounds go unreadable in dark mode. */
.answer-panel { background: rgba(127,127,127,0.08);
  border: 1px solid rgba(127,127,127,0.35);
  border-radius: 10px; padding: 1rem 1.2rem; margin: 0.5rem 0; }
.src-chip { display: inline-block; background: rgba(47,93,58,0.18);
  border: 1px solid rgba(47,93,58,0.45); color: inherit;
  border-radius: 999px; padding: 0.1rem 0.7rem; margin: 0.1rem 0.2rem 0.1rem 0;
  font-size: 0.8rem; font-weight: 600; }
.pill { display: inline-block; border-radius: 999px; padding: 0.15rem 0.8rem;
  font-size: 0.8rem; font-weight: 700; color: inherit; }
.pill-ok { background: rgba(47,93,58,0.22); }
.pill-warn { background: rgba(138,90,0,0.22); }
.pill-bad { background: rgba(143,45,34,0.22); }
.stat-line { opacity: 0.7; font-size: 0.85rem; }
h2 { letter-spacing: -0.01em; }
</style>
"""

PILL = {"RELEVANT": ("pill-ok", "relevant"),
        "PARTLY_RELEVANT": ("pill-warn", "partly relevant"),
        "NON_RELEVANT": ("pill-bad", "not relevant")}


def pill(verdict: str) -> str:
    """Full-tint status pill for a judge verdict."""
    cls, label = PILL.get(verdict, ("pill-warn", verdict.lower()))
    return f'<span class="pill {cls}">{label}</span>'


def chips(ids: list[str]) -> str:
    """Inline source chips for doc/sku ids."""
    return " ".join(f'<span class="src-chip">{i}</span>' for i in ids) or "—"


def online_judge(question: str, answer: str) -> str:
    """No-ground-truth relevance verdict. Returns '' on any failure."""
    try:
        from dotenv import load_dotenv
        from openai import OpenAI

        load_dotenv(Path(__file__).resolve().parents[1] / ".env")
        key = os.environ.get("OPENROUTER_API_KEY", "")
        if not key:
            return ""
        client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=key)
        model = os.environ.get("OPENROUTER_JUDGE_MODEL",
                               os.environ.get("OPENROUTER_MODEL", MODEL))
        resp = client.chat.completions.create(
            model=model, temperature=0.0,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=question, answer=answer[:2000])}])
        text = ((resp.choices or [None])[0].message.content or "").strip()
        for label in ("PARTLY_RELEVANT", "NON_RELEVANT", "RELEVANT"):
            if label in text:
                return label
        return ""
    except Exception:
        return ""


def answer_turn(question: str) -> None:
    """Run one Q&A turn: retrieve, synthesize, log, judge."""
    t0 = time.time()
    ctx = retrieve(question)
    try:
        answer = synthesize(question, ctx)
    except Exception as e:  # noqa: BLE001 — show, don't crash
        st.error(f"LLM call failed: {e}")
        return
    latency = round(time.time() - t0, 2)
    pt = len(question) // 4 + 1500  # rough estimate, labeled as such
    ct = len(answer) // 4
    record = LLMCallRecord(
        model=os.environ.get("OPENROUTER_MODEL", MODEL), question=question,
        answer=answer, prompt_tokens=pt, completion_tokens=ct,
        total_tokens=pt + ct, latency_s=latency,
        cost_usd=estimate_cost(MODEL, pt, ct))
    cid = db.save_conversation(record)
    verdict = online_judge(question, answer)
    if verdict:
        db.save_feedback(cid, "judge", verdict)
    st.session_state["history"].append({
        "answer": answer, "policies": [c.get("doc_id") for c in ctx["policies"]],
        "skus": [r["sku_id"] for r in ctx["stock"][:10]],
        "latency": latency, "tokens": pt + ct, "verdict": verdict, "cid": cid})
    st.session_state["cid"] = cid


def chat_tab() -> None:
    """Conversational Q&A with history, citations, and feedback."""
    st.header("Replenishment copilot")
    st.caption("Grounded in live stock facts and cited policy. "
               "Citations after every claim.")
    if "history" not in st.session_state:
        st.session_state["history"] = []
    for turn in st.session_state["history"]:
        with st.chat_message("assistant"):
            st.markdown(f'<div class="answer-panel">{turn["answer"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(chips(turn["policies"] + turn["skus"][:5]),
                        unsafe_allow_html=True)
            meta = (f'<span class="stat-line">{turn["latency"]}s, '
                    f'about {turn["tokens"]} tokens (est.)</span>')
            if turn["verdict"]:
                meta += " " + pill(turn["verdict"])
            st.markdown(meta, unsafe_allow_html=True)
    q = st.chat_input("Which SKUs are at stockout risk?")
    if q and q.strip():
        with st.chat_message("user"):
            st.write(q.strip())
        with st.chat_message("assistant"):
            with st.spinner("Checking stock and policy..."):
                answer_turn(q.strip())
            turn = st.session_state["history"][-1]
            st.markdown(f'<div class="answer-panel">{turn["answer"]}</div>',
                        unsafe_allow_html=True)
            st.markdown(chips(turn["policies"] + turn["skus"][:5]),
                        unsafe_allow_html=True)
    if "cid" in st.session_state:
        st.divider()
        c1, c2 = st.columns([1, 1])
        if c1.button("This helped", use_container_width=True):
            db.save_feedback(st.session_state["cid"], "user", "+1")
            st.success("Noted, thanks.")
        if c2.button("Not quite right", use_container_width=True):
            db.save_feedback(st.session_state["cid"], "user", "-1")
            st.success("Flagged for review.")


def dashboard_tab() -> None:
    """5-chart telemetry dashboard + recent conversations."""
    st.header("Operations telemetry")
    stats = db.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Questions asked", stats["total"])
    c2.metric("Mean latency", f'{stats["avg_latency"]:.1f}s')
    c3.metric("Spend to date", f'${stats["total_cost"]:.4f}')
    c4.metric("Helpful votes", f'{stats["thumbs_up"]}/{stats["thumbs_up"] + stats["thumbs_down"]}')
    rows = db.get_recent(100)
    if not rows:
        st.info("Quiet here. Answers from the Chat tab will populate "
                "these charts.")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    df["day"] = df["created_at"].str[:10]
    st.divider()
    st.subheader("Request volume")
    st.caption("Questions per day. Gaps mean nobody asked, not missing data.")
    st.bar_chart(df.groupby("day").size(), color="#2f5d3a")
    st.divider()
    st.subheader("Latency")
    lat = sorted(df["latency_s"].tolist())
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    st.caption(f"Median {p50:.1f}s, p95 {p95:.1f}s across "
               f"{len(lat)} answers. Embeddings load once, then stay warm.")
    st.line_chart(df.iloc[::-1].reset_index(drop=True)["latency_s"],
                  color="#2f5d3a")
    st.divider()
    st.subheader("Spend")
    st.caption("Free-tier models record $0. Paid usage accumulates below.")
    st.line_chart(df.iloc[::-1]["cost_usd"].cumsum().reset_index(drop=True),
                  color="#8a5a00")
    st.divider()
    st.subheader("Human feedback")
    st.bar_chart({"helpful": stats["thumbs_up"],
                  "not quite": stats["thumbs_down"]}, color="#2f5d3a")
    st.divider()
    st.subheader("Judge relevance")
    rel = db.get_relevance_stats()
    if rel:
        st.bar_chart(rel, color="#2f5d3a")
    else:
        st.caption("No judge verdicts yet. They appear after answered "
                   "questions.")
    st.divider()
    st.subheader("Recent answers")
    st.dataframe(df[["created_at", "question", "answer", "latency_s",
                      "total_tokens"]].head(20), use_container_width=True)


def main() -> None:
    """App entry: init telemetry tables, render tabs."""
    st.set_page_config(page_title="Replenishment copilot",
                       page_icon="boxes", layout="wide")
    st.markdown(CSS, unsafe_allow_html=True)
    db.init_db()
    st.title("Replenishment copilot")
    tab1, tab2 = st.tabs(["Ask", "Telemetry"])
    with tab1:
        chat_tab()
    with tab2:
        dashboard_tab()


if __name__ == "__main__":
    main()
