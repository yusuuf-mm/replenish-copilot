"""Ingestion pipeline for replenish-copilot (Day 1, tasks 4+5).

- Loads data/csv/*.csv into SQLite (data/replenish.db) — idempotent rebuild.
- Chunks data/policies/*.md (500 chars / 100 overlap per ## section),
  indexes chunks in a local hybrid policy index:
    text   -> sqlitesearch TextSearchIndex (BM25-ish keyword search)
    vector -> sqlitesearch VectorSearchIndex (MiniLM 384-dim, cosine)
- Run: uv run python src/ingest.py
"""

from __future__ import annotations

import csv
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSV_DIR = ROOT / "data" / "csv"
POLICY_DIR = ROOT / "data" / "policies"
DB_PATH = ROOT / "data" / "replenish.db"
INDEX_DIR = ROOT / "data" / "policy_index"
TEXT_DB = INDEX_DIR / "text.db"
VECTOR_DB = INDEX_DIR / "vectors.db"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBED_MODEL = "all-MiniLM-L6-v2"

TABLES: dict[str, str] = {
    "inventory": """CREATE TABLE inventory (
        sku_id TEXT PRIMARY KEY, product_name TEXT, category TEXT,
        stock_on_hand INTEGER, reorder_point INTEGER, safety_stock INTEGER,
        unit_cost_usd REAL, supplier_id TEXT)""",
    "suppliers": """CREATE TABLE suppliers (
        supplier_id TEXT PRIMARY KEY, supplier_name TEXT, lead_time_days INTEGER,
        sla_compliance_rate REAL, contact_email TEXT)""",
    "purchase_orders": """CREATE TABLE purchase_orders (
        po_id TEXT PRIMARY KEY, sku_id TEXT, supplier_id TEXT, order_qty INTEGER,
        order_date TEXT, expected_delivery TEXT, status TEXT)""",
    "demand": """CREATE TABLE demand (
        date TEXT, sku_id TEXT, daily_demand INTEGER, forecasted_demand REAL)""",
}

INT_COLS = {"stock_on_hand", "reorder_point", "safety_stock", "lead_time_days",
            "order_qty", "daily_demand"}
REAL_COLS = {"unit_cost_usd", "sla_compliance_rate", "forecasted_demand"}


def coerce(row: dict[str, str]) -> dict[str, object]:
    """Coerce numeric CSV strings to int/float, keep the rest as text."""
    out: dict[str, object] = {}
    for k, v in row.items():
        if k in INT_COLS:
            out[k] = int(float(v))
        elif k in REAL_COLS:
            out[k] = float(v)
        else:
            out[k] = v
    return out


def load_csvs(db_path: Path = DB_PATH) -> dict[str, int]:
    """Rebuild SQLite tables from CSVs. Returns row counts per table."""
    counts: dict[str, int] = {}
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        for table, ddl in TABLES.items():
            conn.execute(f"DROP TABLE IF EXISTS {table}")
            conn.execute(ddl)
            csv_path = CSV_DIR / f"{table}.csv"
            if table == "purchase_orders":
                csv_path = CSV_DIR / "purchase_orders.csv"
            with open(csv_path, newline="", encoding="utf-8") as f:
                rows = [coerce(r) for r in csv.DictReader(f)]
            cols = list(rows[0].keys())
            conn.executemany(
                f"INSERT INTO {table} ({','.join(cols)}) "
                f"VALUES ({','.join('?' for _ in cols)})",
                [tuple(r[c] for c in cols) for r in rows],
            )
            counts[table] = len(rows)
        conn.commit()
    finally:
        conn.close()
    return counts


def parse_policy(path: Path) -> tuple[str, str, str]:
    """Return (doc_id, title, body) from a policy markdown file."""
    text = path.read_text(encoding="utf-8")
    m = re.match(r"---\s*\ndoc_id:\s*(\S+)\s*\ntitle:\s*(.+?)\s*\n---\s*\n(.*)",
                 text, re.DOTALL)
    if not m:
        raise ValueError(f"{path.name}: missing doc_id/title frontmatter")
    return m.group(1), m.group(2), m.group(3)


def chunk_text(body: str, size: int = CHUNK_SIZE,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Sliding-window char chunks over whitespace-normalized text."""
    text = re.sub(r"\s+", " ", body).strip()
    chunks: list[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):
            break
        start += size - overlap
    return chunks


def build_chunks() -> list[dict]:
    """Chunk all policies into {chunk_id, doc_id, section_title, content}."""
    chunks: list[dict] = []
    for path in sorted(POLICY_DIR.glob("*.md")):
        doc_id, title, body = parse_policy(path)
        sections = re.split(r"(?m)^##\s+", body)
        for s in sections:
            s = s.strip()
            if not s:
                continue
            lines = s.split("\n", 1)
            section_title = lines[0].strip()[:80]
            section_body = lines[1] if len(lines) > 1 else lines[0]
            for c in chunk_text(section_body):
                chunks.append({
                    "chunk_id": f"{doc_id}-c{len(chunks):02d}",
                    "doc_id": doc_id,
                    "section_title": f"{title} / {section_title}",
                    "content": c,
                })
    return chunks


def build_text_index(chunks: list[dict]) -> int:
    """Index chunks with sqlitesearch (persistent). Returns chunk count."""
    from sqlitesearch import TextSearchIndex

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    if TEXT_DB.exists():
        TEXT_DB.unlink()
    index = TextSearchIndex(
        text_fields=["content", "section_title"],
        keyword_fields=["doc_id"],
        db_path=str(TEXT_DB),
    )
    for ch in chunks:
        index.add(ch)
    index.close()
    return len(chunks)


def build_vector_index(chunks: list[dict]) -> int:
    """Embed chunks with MiniLM and index vectors. Returns chunk count."""
    from sentence_transformers import SentenceTransformer
    from sqlitesearch import VectorSearchIndex

    model = SentenceTransformer(EMBED_MODEL)
    texts = [f"{c['section_title']} {c['content']}" for c in chunks]
    vectors = model.encode(texts, batch_size=32, show_progress_bar=False)
    if VECTOR_DB.exists():
        VECTOR_DB.unlink()
    index = VectorSearchIndex(
        keyword_fields=["doc_id"],
        mode="lsh",
        db_path=str(VECTOR_DB),
    )
    index.fit(list(vectors), chunks)
    index.close()
    return len(chunks)


def main() -> None:
    counts = load_csvs()
    print("SQLite:", {k: f"{v} rows" for k, v in counts.items()}, f"-> {DB_PATH}")
    chunks = build_chunks()
    n_text = build_text_index(chunks)
    print(f"Text index: {n_text} chunks -> {TEXT_DB}")
    n_vec = build_vector_index(chunks)
    print(f"Vector index: {n_vec} chunks ({EMBED_MODEL}) -> {VECTOR_DB}")
    print("Ingest OK.")


if __name__ == "__main__":
    sys.exit(main())
