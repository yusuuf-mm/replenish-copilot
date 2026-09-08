"""Telemetry storage — SQLite (Phase 3).

Two tables: conversations (one row per answered question) and feedback
(user votes + online judge verdicts, keyed by conversation id).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from src.metrics import LLMCallRecord

ROOT = Path(__file__).resolve().parents[1]
TELEMETRY_DB = ROOT / "data" / "telemetry.db"


def _connect() -> sqlite3.Connection:
    TELEMETRY_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TELEMETRY_DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if missing. Safe to call on every startup."""
    conn = _connect()
    try:
        conn.execute("""CREATE TABLE IF NOT EXISTS conversations (
            id INTEGER PRIMARY KEY AUTOINCREMENT, question TEXT NOT NULL,
            answer TEXT NOT NULL, model TEXT NOT NULL,
            prompt_tokens INTEGER DEFAULT 0, completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0, latency_s REAL DEFAULT 0,
            cost_usd REAL DEFAULT 0, created_at TEXT NOT NULL)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id INTEGER REFERENCES conversations(id),
            source TEXT NOT NULL, score TEXT NOT NULL,
            note TEXT DEFAULT '', created_at TEXT NOT NULL)""")
        conn.commit()
    finally:
        conn.close()


def save_conversation(record: LLMCallRecord) -> int:
    """Insert one conversation row. Returns its id for feedback linkage."""
    from datetime import datetime

    conn = _connect()
    try:
        cur = conn.execute(
            """INSERT INTO conversations (question, answer, model, prompt_tokens,
               completion_tokens, total_tokens, latency_s, cost_usd, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (record.question, record.answer, record.model, record.prompt_tokens,
             record.completion_tokens, record.total_tokens, record.latency_s,
             record.cost_usd, datetime.now().isoformat()),
        )
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def save_feedback(conversation_id: int, source: str, score: str,
                  note: str = "") -> None:
    """Store a user vote (+1/-1) or judge verdict (RELEVANT/...)."""
    from datetime import datetime

    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO feedback (conversation_id, source, score, note,"
            " created_at) VALUES (?,?,?,?,?)",
            (conversation_id, source, score, note, datetime.now().isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_stats() -> dict:
    """Headline aggregates for the dashboard."""
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT COUNT(*), AVG(latency_s), SUM(cost_usd),"
            " AVG(total_tokens) FROM conversations").fetchone()
        fb = conn.execute(
            "SELECT SUM(CASE WHEN score='+1' THEN 1 ELSE 0 END),"
            " SUM(CASE WHEN score='-1' THEN 1 ELSE 0 END)"
            " FROM feedback WHERE source='user'").fetchone()
        return {"total": row[0] or 0, "avg_latency": row[1] or 0.0,
                "total_cost": row[2] or 0.0, "avg_tokens": row[3] or 0.0,
                "thumbs_up": fb[0] or 0, "thumbs_down": fb[1] or 0}
    finally:
        conn.close()


def get_recent(limit: int = 100) -> list[dict]:
    """Newest conversations first (for charts + table)."""
    conn = _connect()
    try:
        return [dict(r) for r in conn.execute(
            "SELECT * FROM conversations ORDER BY id DESC LIMIT ?",
            (limit,)).fetchall()]
    finally:
        conn.close()


def get_relevance_stats() -> dict[str, int]:
    """Online-judge verdict distribution."""
    conn = _connect()
    try:
        return {r[0]: r[1] for r in conn.execute(
            "SELECT score, COUNT(*) FROM feedback WHERE source='judge'"
            " GROUP BY score").fetchall()}
    finally:
        conn.close()
