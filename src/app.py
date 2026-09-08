"""Streamlit UI: chat + telemetry dashboard (Phase 3).

Run: uv run streamlit run src/app.py
Tab 1 (Chat): ask -> grounded answer + citations + metrics, +1/-1 feedback.
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


def chat_tab() -> None:
    """Q&A with citations, metrics, and feedback buttons."""
    st.header("Replenishment Copilot")
    q = st.text_input("Ask about stock, reorders, or supplier delays:",
                      placeholder="Which SKUs are at stockout risk?")
    if st.button("Ask") and q.strip():
        with st.spinner("Retrieving + synthesizing..."):
            t0 = time.time()
            ctx = retrieve(q)
            try:
                answer = synthesize(q, ctx)
            except Exception as e:  # noqa: BLE001 — show, don't crash
                st.error(f"LLM call failed: {e}")
                return
            latency = round(time.time() - t0, 2)
        st.subheader("Answer")
        st.write(answer)
        with st.expander("Sources"):
            st.write("Policies:",
                     [c.get("doc_id") for c in ctx["policies"]])
            st.write("SKUs:",
                     [r["sku_id"] for r in ctx["stock"][:10]])
        pt = len(q) // 4 + 1500  # rough estimate, labeled as such
        ct = len(answer) // 4
        record = LLMCallRecord(
            model=os.environ.get("OPENROUTER_MODEL", MODEL), question=q,
            answer=answer, prompt_tokens=pt, completion_tokens=ct,
            total_tokens=pt + ct, latency_s=latency,
            cost_usd=estimate_cost(MODEL, pt, ct))
        cid = db.save_conversation(record)
        st.session_state["cid"] = cid
        st.caption(f"~{latency}s | ~{pt + ct} tokens (est.) | "
                   f"cost ${record.cost_usd:.4f}")
        verdict = online_judge(q, answer)
        if verdict:
            db.save_feedback(cid, "judge", verdict)
            st.caption(f"Online judge: {verdict}")
    if "cid" in st.session_state:
        c1, c2 = st.columns(2)
        if c1.button("+1 helpful"):
            db.save_feedback(st.session_state["cid"], "user", "+1")
            st.success("Thanks!")
        if c2.button("-1 not helpful"):
            db.save_feedback(st.session_state["cid"], "user", "-1")
            st.success("Thanks — we'll review.")


def dashboard_tab() -> None:
    """5-chart telemetry dashboard + recent conversations."""
    st.header("Telemetry Dashboard")
    stats = db.get_stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Conversations", stats["total"])
    c2.metric("Avg latency (s)", round(stats["avg_latency"], 2))
    c3.metric("Total cost ($)", round(stats["total_cost"], 4))
    c4.metric("Avg tokens", int(stats["avg_tokens"]))
    rows = db.get_recent(100)
    if not rows:
        st.info("No traffic yet — ask a question in the Chat tab.")
        return
    import pandas as pd

    df = pd.DataFrame(rows)
    df["day"] = df["created_at"].str[:10]
    st.subheader("1. Request volume (per day)")
    st.bar_chart(df.groupby("day").size())
    st.subheader("2. Latency over time (s)")
    st.line_chart(df.iloc[::-1].reset_index(drop=True)["latency_s"])
    lat = sorted(df["latency_s"].tolist())
    p50 = statistics.median(lat)
    p95 = lat[min(len(lat) - 1, int(len(lat) * 0.95))]
    st.caption(f"p50 = {p50:.2f}s | p95 = {p95:.2f}s")
    st.subheader("3. Cumulative cost ($)")
    st.line_chart(df.iloc[::-1]["cost_usd"].cumsum().reset_index(drop=True))
    st.subheader("4. User feedback")
    st.bar_chart({"thumbs up": stats["thumbs_up"],
                  "thumbs down": stats["thumbs_down"]})
    st.subheader("5. Online judge relevance")
    rel = db.get_relevance_stats()
    if rel:
        st.bar_chart(rel)
    else:
        st.caption("No judge verdicts yet.")
    st.subheader("Recent conversations")
    st.dataframe(df[["created_at", "question", "answer", "latency_s",
                      "total_tokens"]].head(20))


def main() -> None:
    """App entry: init telemetry tables, render tabs."""
    db.init_db()
    st.title("replenish-copilot")
    tab1, tab2 = st.tabs(["Chat", "Dashboard"])
    with tab1:
        chat_tab()
    with tab2:
        dashboard_tab()


if __name__ == "__main__":
    main()
