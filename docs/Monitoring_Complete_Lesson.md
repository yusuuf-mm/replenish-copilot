# Monitoring: complete practical lesson

Module 4 measured your system before anyone touched it. This module watches it after real people show up. Offline numbers say whether the system works. Monitoring says whether it still works, how much it costs, and whether anyone likes the answers.

You build three pieces on top of the RAG pipeline you already have: a chat interface people type into, a database that stores every interaction, and dashboards that turn the stored rows into charts. Streamlit first, Grafana when you outgrow it.

Read top to bottom. Each section adds one file or one table, shows the exact code, and says how to check it works.

---
## 0. What we are building

```text
assistant.py       RAG pipeline (search → prompt → answer), unchanged logic
        ↓
app.py             Streamlit chat: ask, answer, metrics, judge score, +1/-1
        ↓
metrics.py         LLMCallRecord + RAGWithMetrics: time, tokens, cost per call
        ↓
PostgreSQL         conversations table + feedback table, every interaction saved
        ↓
db_query.py        read it back: recent rows, aggregate stats, feedback counts
        ↓
dashboard.py       Streamlit dashboard: totals, charts, recent answers, feedback
        ↓
generate_data.py   fake traffic so the charts have something to show
        ↓
Grafana            real dashboards: time series, pies, gauges, tables, refresh
        ↓
compose            one command for postgres + grafana + streamlit
```

The order is deliberate. You cannot dashboard data you never stored, and you cannot store what you never captured. Interface first, capture second, store third, visualize last. Each layer only talks to the one below it.

---
## 1. Learning objectives

When you finish, you should be able to:

- Explain why offline metrics stop being enough once real traffic arrives
- Wrap the module 1 RAG pipeline in a Streamlit chat app without changing its logic
- Instrument an LLM call with a dataclass: model, prompt, tokens, time, cost
- Price a call from the `usage` object at 0.15/M input and 0.60/M output
- Run Postgres in Docker, create the `conversations` table, and save every Q&A with its metrics
- Read rows back as records and compute aggregate stats in SQL
- Build a Streamlit dashboard with totals, time charts, and recent conversations
- Collect thumbs up/down into a `feedback` table keyed by conversation id
- Run an online LLM judge (no ground truth, three relevance labels) and store its verdicts
- Fill an empty dashboard with synthetic traffic
- Build Grafana panels in SQL: time series, bar, pie, gauge, table
- Say when the Streamlit dashboard is enough and when Grafana earns its keep
- Describe what changes for production: async writes, sampling the judge, better log storage

---
## 2. The big mental model

Offline evaluation asked "does it work" on a frozen test set. Monitoring asks "is it still working" on live traffic nobody controls. Different questions, different machinery:

```text
Offline (module 4)              Online (this module)
─────────────────────           ─────────────────────
Fixed questions                 Whatever users type
Known-correct docs              No ground truth
Hit rate, MRR, judge vs A       Latency, cost, thumbs, judge vs nothing
Run before shipping             Runs while serving
```

Per question, there is a surprising amount worth keeping: the instructions, the prompt, the model name, token counts, cost, how long the user waited, whether they liked the answer, and whether a judge found it relevant. That list is the schema design for the whole module. Every table and panel below exists to capture or display one of those items.

Three new pieces, each with one job:

```text
Interface  →  where people ask (Streamlit chat app)
Database   →  where interactions land (PostgreSQL)
Dashboard  →  where rows become charts (Streamlit, then Grafana)
```

The RAG pipeline itself does not change. Search, prompt, answer, same as module 1. Monitoring wraps it; it never rewrites it. That is also why the agent homework works: the same wrap fits any pipeline that produces answers.

---
## 3. Environment setup

The `code/` folder is a uv project (`llm-zoomcamp-2026-monitoring`, Python 3.12+). Dependencies grow as the module goes:

```text
jupyter, minsearch, openai, psycopg[binary], python-dotenv, requests, streamlit
```

Start with `python-dotenv` and add the rest when each lesson needs them. You need an OpenAI key:

```text
OPENAI_API_KEY=YOUR_API_KEY
```

Two helpers carry over from module 1. Fetch them if they are missing:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main

wget ${PREFIX}/01-agentic-rag/code/ingest.py
wget ${PREFIX}/01-agentic-rag/code/rag_helper.py
```

This module's copies match module 1: `ingest.py` keeps `id` as-is and indexes question/section/answer with `course` as the keyword field; `RAGBase` defaults to `gpt-5.4-mini` with question boost 3.0. Nothing new to learn here, which is the point. Reuse, don't rebuild.

One repo quirk to know about before it bites you. The shipped `Makefile` line 3 reads `make run:` where the lesson text says the target should be `run:`. As written, `make run` will not do what the lesson promises. Either fix that line to `run:` or just run the full `uv run python ...` commands. The lesson text is correct; the file has the typo.

---
## 4. The assistant

Something has to answer before anything can be monitored. `assistant.py` wires the two helpers into one factory:

```python
import sys

from dotenv import load_dotenv
from openai import OpenAI

from ingest import load_faq_data, build_index
from rag_helper import RAGBase

def create_assistant():
    load_dotenv()

    documents = load_faq_data()
    index = build_index(documents)

    return RAGBase(
        index=index,
        llm_client=OpenAI(),
    )
```

No custom instructions passed. `RAGBase` ships with a system prompt for course questions, and a second one would just fight it.

The `__main__` block takes an optional question argument so CLI tests stay quick:

```python
if __name__ == "__main__":
    assistant = create_assistant()

    query = "How do I join the course?"
    if len(sys.argv) > 1:
        query = sys.argv[1]

    answer = assistant.rag(query)
    print(answer)
```

```bash
uv run python assistant.py
uv run python assistant.py "How do I join the course?"
```

Repetitive commands go in the `Makefile` (`run`, and later `chat`, `postgres`, `query`). With the typo from section 3 fixed, `make run` works. An answer on the console proves the pipeline is alive. A console is not where users live, so next comes the interface.

---
## 5. Chat app

Streamlit turns the CLI into a page with a text box. The first version is tiny on purpose:

```python
import streamlit as st
from assistant import create_assistant

assistant = create_assistant()

st.title("Course Assistant")

user_input = st.text_input("Enter your question:")

if st.button("Ask"):
    with st.spinner("Processing..."):
        answer = assistant.rag(user_input)
        st.success("Completed!")
        st.write(answer)
```

```bash
uv add streamlit
uv run streamlit run app.py
```

Makefile target `chat`, then `make chat`. In Codespaces the port forwards automatically; open the link, ask how to join the course, get an answer.

The lesson is upfront that this UI is plain and will stay plain. If you want it prettier, hand it to a coding assistant and describe the layout. The teaching value is in what gets wired behind the button, not the button.

Right now each answer evaporates. No timing, no tokens, no cost. That gap is the entire module.

---
## 6. Capturing metrics

Every LLM call should leave a receipt. A dataclass beats a dict here because the fields are spelled out where anyone can read them:

```python
import time
from dataclasses import dataclass, field
from datetime import datetime

from rag_helper import RAGBase

@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
```

Model, prompt, instructions, answer: what went in and what came out. Tokens in and out plus total: what it consumed. Response time: what the user felt. Cost: what you paid. Timestamp defaults to now so nobody forgets to set it.

### 6.1 Pricing one call

The provider charges per million tokens, different rates each direction. The `usage` object on the response carries the counts:

```python
def calculate_cost(model, usage):
    cost = 0
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost
```

Two honest notes. The rates here (0.15/0.60) differ from module 4's `calc_price` (0.75/4.50); pricing moves and each module froze its own snapshot, so check current numbers before budgeting anything real. And the field names follow the API (`prompt_tokens`, `completion_tokens`); `input_tokens`/`output_tokens` read better, rename freely in your own code.

### 6.2 The instrumented subclass

Same subclass trick as module 4's `RAGWithUsage`, but aimed at live capture instead of batch accounting. Override only `llm`, time it, stash the record:

```python
class RAGWithMetrics(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        return response.output_text
```

`_call_llm` sends the request exactly the way `RAGBase.llm` did. `_log_response` builds the record, prints it, and parks it on `self.last_call`. Shared mutable state is not thread-safe, and the lesson says so plainly. For one person clicking through Streamlit it is fine; for concurrent traffic you would return the record instead of stashing it.

`assistant.py` switches its import from `RAGBase` to `RAGWithMetrics`. Nothing else changes there. `app.py` needs no structural changes either, but now it can show the receipt under each answer:

```python
record = assistant.last_call
st.write(f"Response time: {record.response_time:.2f}s")
st.write(f"Prompt tokens: {record.prompt_tokens}")
st.write(f"Completion tokens: {record.completion_tokens}")
st.write(f"Cost: ${record.cost:.4f}")
```

`make chat` again. Each answer now carries its numbers. They still vanish on close, which is what the database fixes.

---
## 7. Storing everything in PostgreSQL

Memory ends at process exit. Postgres keeps the rows, speaks SQL for the dashboards, and plugs straight into Grafana later. This database serves monitoring only; nothing else in the system touches it.

### 7.1 Containers and network

Grafana will reach Postgres by container name, so both need a shared network:

```bash
docker network create monitoring
```

```bash
docker run -it \
    --name course-assistant-pg \
    --network monitoring \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=password \
    -e POSTGRES_DB=course_assistant \
    -p 5432:5432 \
    -v pgdata:/var/lib/postgresql/data \
    postgres:17
```

Named volume `pgdata` means data survives restarts. In the Makefile these become `network` and `postgres` targets (the latter depending on the former), so `make postgres` does both. Driver for Python:

```bash
uv add "psycopg[binary]"
```

### 7.2 The conversations table

`db_init.py` holds the connection helper (env vars with defaults matching the container) and the schema. Thirteen columns, mirroring the record plus the user's raw question and course:

```sql
CREATE TABLE conversations (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    course TEXT NOT NULL,
    model TEXT NOT NULL,
    instructions TEXT NOT NULL,
    prompt TEXT NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    total_tokens INTEGER NOT NULL,
    response_time FLOAT NOT NULL,
    cost FLOAT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
)
```

Two columns deserve their explanations. `course` exists because one assistant can serve many courses; everything is `llm-zoomcamp` today, but the column saves a migration later. `timestamp` is timezone-aware on purpose: without the zone, Grafana misplaces points on its time axis. (The table name `conversations` is inherited from older materials; `llm_call_records` would describe it better. Rename in your own projects.)

`init_db(drop=False)` creates the table, with an opt-in drop for schema iterations. The warning stands: never point a drop at a real database. Run once; the volume keeps it around. Re-run only when the schema changes.

```bash
uv run python db_init.py
```

### 7.3 Saving with the id back

`db_save.py` inserts one record. `id` is `SERIAL`, so Postgres assigns it, and `RETURNING id` hands it back. That id matters later: feedback rows point at it, so the insert must return it.

```python
from datetime import datetime
from db_init import get_db_connection, DB_TIMEZONE

def save_conversation(record, question, course):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO conversations (
                    question, answer, course, model, instructions, prompt,
                    prompt_tokens, completion_tokens, total_tokens,
                    response_time, cost, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                )
                RETURNING id
                """,
                (
                    question,
                    record.answer,
                    course,
                    record.model,
                    record.instructions,
                    record.prompt,
                    record.prompt_tokens,
                    record.completion_tokens,
                    record.total_tokens,
                    record.response_time,
                    record.cost,
                    timestamp,
                ),
            )
            conversation_id = cur.fetchone()[0]
        conn.commit()
    finally:
        conn.close()
    return conversation_id
```

Why pass `question` separately when the record already has a prompt? The prompt is the full model input, shaped by how we call the LLM. The question is what the human typed. Different things, and dashboards want the human one verbatim. Keep them apart.

Wire it into the CLI test in `assistant.py` (import, then `save_conversation(assistant.last_call, query, "llm-zoomcamp")` after the answer) and check with psql:

```bash
uv run python assistant.py "How do I join the course?"
docker exec -it course-assistant-pg psql -U user -d course_assistant \
    -c "SELECT id, question, response_time, cost FROM conversations;"
```

Then the same one-liner in `app.py` after the metrics display, plus stashing the id in session state for the feedback buttons coming in section 10:

```python
from db_save import save_conversation

conversation_id = save_conversation(record, user_input, "llm-zoomcamp")
st.session_state.conversation_id = conversation_id
```

Every answer now persists. Next: reading them back.

---
## 8. Querying it back

Dashboards read; this is the read path. `db_query.py` converts raw tuples back into the dataclass (positional indexes are nobody's friend) and offers the two reads everything else needs.

```python
from dataclasses import dataclass

from db_init import get_db_connection
from metrics import LLMCallRecord

def row_to_record(row):
    return LLMCallRecord(
        model=row[4],
        prompt=row[6],
        instructions=row[5],
        answer=row[2],
        prompt_tokens=row[7],
        completion_tokens=row[8],
        total_tokens=row[9],
        response_time=row[10],
        cost=row[11],
        timestamp=row[12],
    )

def get_conversations(limit=10):
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, question, answer, course, model,
                       instructions, prompt,
                       prompt_tokens, completion_tokens, total_tokens,
                       response_time, cost, timestamp
                FROM conversations
                ORDER BY timestamp DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    return [row_to_record(row) for row in rows]
```

Newest first, capped by `limit`. One performance note the lesson doesn't hide: no index on `timestamp`, though `id` is indexed and grows monotonically, so ordering by `id` would be faster, as would adding the index. At dashboard volumes it doesn't matter; know it before the table gets big.

Makefile gets a `query` target; `uv run python db_query.py` prints records (a wall of text, but proof the round trip works). The dashboard makes it readable.

---
## 9. Streamlit dashboard

Before reaching for Grafana, check whether Streamlit already answers your questions. Latency, cost, and recent conversations on one page covers a lot of projects, and stopping here means SQLite could replace Postgres and Docker could go away entirely. We stay on Postgres only because Grafana connects to it easily later.

`db_query.py` gains aggregates first:

```python
@dataclass
class Stats:
    total: int
    avg_response_time: float
    total_cost: float
    avg_tokens: float

def get_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    COUNT(*),
                    AVG(response_time),
                    SUM(cost),
                    AVG(total_tokens)
                FROM conversations
            """)
            row = cur.fetchone()
    finally:
        conn.close()

    return Stats(
        total=row[0],
        avg_response_time=row[1],
        total_cost=row[2],
        avg_tokens=row[3],
    )
```

Then `dashboard.py`: four headline numbers, two time charts off the last 100 rows, recent conversations as plain text.

```python
import streamlit as st
from dataclasses import asdict
import pandas as pd
from db_query import get_conversations, get_stats

st.title("Course Assistant Dashboard")

stats = get_stats()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", stats.total)
col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
col3.metric("Total cost", f"${stats.total_cost:.4f}")
col4.metric("Avg tokens", f"{stats.avg_tokens:.0f}")

records = get_conversations(limit=100)
df = pd.DataFrame([asdict(r) for r in records])

st.subheader("Cost over time")
st.line_chart(df, x="timestamp", y="cost")

st.subheader("Response time over time")
st.line_chart(df, x="timestamp", y="response_time")

st.subheader("Recent conversations")
records = get_conversations(limit=20)

for record in records:
    st.write(f"**{record.prompt[:80]}...**")
    st.write(f"{record.answer[:200]}...")
    st.write(f"Time: {record.response_time:.2f}s | Cost: ${record.cost:.4f}")
    st.divider()
```

Fetching whole records to chart two columns is wasteful; a lean version would select only timestamp plus the plotted value. At our volume nobody cares, so the code stays short. Port 8501 is taken by the chat app, so the dashboard runs on 8502:

```bash
uv run streamlit run dashboard.py --server.port 8502
```

Plain text instead of tables, four numbers, two lines. Already real visibility. Grafana later adds alerting and richer panels, not the concept.

---
## 10. User feedback

Timing and tokens never say whether the answer was any good. The user knows, so give them buttons. Thumbs up/down per answer, stored per conversation.

One table serves both feedback sources, human and machine, distinguished by `source`. The judge-only columns (`relevance`, `explanation`) sit null for user rows; user rows use `score` (+1/-1) and leave the judge columns null:

```python
def init_feedback():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE IF EXISTS feedback")

            cur.execute("""
                CREATE TABLE feedback (
                    id SERIAL PRIMARY KEY,
                    conversation_id INTEGER REFERENCES conversations(id),
                    source TEXT NOT NULL,
                    relevance TEXT,
                    explanation TEXT,
                    score INTEGER,
                    timestamp TIMESTAMP WITH TIME ZONE NOT NULL
                )
            """)
        conn.commit()
    finally:
        conn.close()
```

`db_init.py`'s `__main__` calls both `init_db()` and `init_feedback()`; re-run it. `db_feedback.py` is one insert function:

```python
from datetime import datetime
from db_init import get_db_connection, DB_TIMEZONE

def save_feedback(conversation_id, source, relevance=None,
                  explanation=None, score=None):
    timestamp = datetime.now(DB_TIMEZONE)

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO feedback (
                    conversation_id, source, relevance,
                    explanation, score, timestamp
                ) VALUES (
                    %s, %s, %s, %s, %s, %s
                )
                """,
                (conversation_id, source, relevance,
                 explanation, score, timestamp),
            )
        conn.commit()
    finally:
        conn.close()
```

The app side has one subtlety. Streamlit reruns the whole script on every click, so the conversation id must survive from the Ask press to the button press. That is what `st.session_state.conversation_id` (set in section 7) is for:

```python
from db_feedback import save_feedback

col1, col2 = st.columns(2)
with col1:
    if st.button("+1"):
        cid = st.session_state.conversation_id
        save_feedback(cid, "user", score=1)
        st.write("Thanks!")

with col2:
    if st.button("-1"):
        cid = st.session_state.conversation_id
        save_feedback(cid, "user", score=-1)
        st.write("Thanks for the feedback!")
```

Buttons render always, not just after an answer; gating them would be nicer and the lesson leaves it as an exercise. Two honest caveats from the lesson: clicks are noisy (misclicks, generous ratings, the instructor admits to both), yet a sudden pile of thumbs-down still means go look at what broke. And good user labels double as alignment data for the judge: where users and judge disagree, the judge prompt needs work.

---
## 11. Built-in judge

Module 4 judged answers offline against ground truth. Online there is no reference answer, so the judge works harder with less: question and answer only, no A to compare against. The instructions compensate by describing good answers more carefully.

Same structured-output machinery from module 4 (`llm_structured_retry` from `evaluation_utils.py`; fetch that file if it is missing). Three labels instead of two, since live answers are rarely cleanly right or wrong:

```python
import json

from pydantic import BaseModel
from typing import Literal
from openai import OpenAI
from dotenv import load_dotenv

from evaluation_utils import llm_structured_retry

class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str

judge_instructions = """
You are an expert evaluator for a RAG system.
Analyze the relevance of the generated answer to the given question.

Classify the answer as:
- RELEVANT: the answer addresses the question
- PARTLY_RELEVANT: the answer partially addresses the question
- NON_RELEVANT: the answer does not address the question
""".strip()

judge_prompt = """
Question: {question}
Generated Answer: {answer}
""".strip()
```

The `explanation` field earns its keep: forcing the judge to reason before labeling tends to improve the label. The retry wrapper covers the occasional malformed-JSON response (rare on small OpenAI models, more common elsewhere).

```python
def evaluate_relevance(question, answer, client=None):
    if client is None:
        client = OpenAI()

    prompt = judge_prompt.format(
        question=question,
        answer=answer
    )

    result, usage = llm_structured_retry(
        client,
        judge_instructions,
        prompt,
        RelevanceVerdict,
    )

    return result.relevance, result.explanation
```

Test standalone (`uv run python judge.py` with the join-the-course example), then wire into `app.py` right after the save: judge, store with `source="judge"`, display both fields. Judge rows and user rows now share the `feedback` table, which is what lets a dashboard compare machine opinion against human opinion.

Three production caveats, all from the lesson. The judge is an extra LLM call per question: latency plus money, so real systems run it asynchronously and answer first. Its own cost deserves separate tracking (a `judge_feedback` table with cost would do). And at real traffic volumes, judge a sample (one in ten keeps the signal at a tenth of the price). Here it runs inline on every call to keep the code readable.

Treat this judge as a starting point, not a verdict machine. It will mislabel both ways. The fix is alignment: collect user labels, tune the prompt until the judge agrees with them. The lesson points at an Evidently talk on automated prompt optimization for the longer version of that loop.

---
## 12. Feedback on the dashboard

Two queries, two panels, added to the dashboard from section 9. Judge relevance as a distribution, user feedback as counts:

```python
def get_relevance_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT relevance, COUNT(*)
                FROM feedback
                WHERE source = 'judge'
                GROUP BY relevance
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    return dict(rows)
```

```python
def get_user_feedback_stats():
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT
                    SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END),
                    SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END)
                FROM feedback
                WHERE source = 'user'
            """)
            row = cur.fetchone()
    finally:
        conn.close()
    return row
```

```python
from db_query import get_conversations, get_stats, get_relevance_stats, get_user_feedback_stats

st.subheader("Judge relevance")
relevance = get_relevance_stats()
st.bar_chart(relevance)

st.subheader("User feedback")
thumbs_up, thumbs_down = get_user_feedback_stats()
col1, col2 = st.columns(2)
col1.metric("Thumbs up", int(thumbs_up or 0))
col2.metric("Thumbs down", int(thumbs_down or 0))
```

The `or 0` guards against NULL when nobody has voted yet. Quality now sits next to cost and speed. Problem: with three real conversations the charts are empty. Synthetic traffic fixes that.

---
## 13. Synthetic data

Clicking through the app to fill a dashboard is slow and dull. `generate_data.py` pumps fake rows straight into the same tables the app writes, one conversation per second until you stop it.

```python
import time
import random

from metrics import LLMCallRecord
from db_save import save_conversation
from db_feedback import save_feedback

SAMPLE_QUESTIONS = [
    "How do I install Docker?",
    "Can I still join the course?",
    "What are the prerequisites?",
    "How do I submit homework?",
    "When are the office hours?",
]

SAMPLE_ANSWERS = [
    "You can install Docker by downloading Docker Desktop from the official website.",
    "Yes, you can join at any time. The materials remain available.",
    "You need basic Python knowledge and familiarity with the command line.",
    "Submit your homework through the course portal before the deadline.",
    "Office hours are held weekly. Check the calendar for details.",
]

RELEVANCE = ["RELEVANT", "PARTLY_RELEVANT", "NON_RELEVANT"]
```

Records get plausible random metrics (tokens, 0.5–5s response times, tiny costs). Feedback is probabilistic: judge verdicts on ~70% of rows, user votes on ~50%, with thumbs skewed 4:1 positive to look like a mostly-working system:

```python
def random_score():
    return random.choice([1, 1, 1, 1, -1])

def generate_one():
    question = random.choice(SAMPLE_QUESTIONS)
    answer = random.choice(SAMPLE_ANSWERS)
    record = fake_record(question, answer)

    conversation_id = save_conversation(
        record, question, "llm-zoomcamp"
    )

    if random.random() < 0.7:
        relevance = random.choice(RELEVANCE)
        save_feedback(
            conversation_id, "judge",
            relevance=relevance,
            explanation=f"Answer is {relevance.lower()}.",
        )

    if random.random() < 0.5:
        score = random_score()
        save_feedback(conversation_id, "user", score=score)

def generate_live():
    print("Starting live data generation (Ctrl+C to stop)...", flush=True)
    while True:
        generate_one()
        time.sleep(1)
```

```bash
uv run python generate_data.py
```

Leave it running while building the Grafana panels next. Charts that move as you work beat static screenshots for learning what each query does.

---
## 14. Grafana

Streamlit got you this far. Grafana goes further: more panel types, more data sources, and alerting when a metric crosses a line. The price is another service to run. Simple needs, Streamlit is enough; when you want more, Grafana reads the Postgres you already have.

```bash
docker run -d \
    --name grafana \
    --network monitoring \
    -p 3000:3000 \
    -v grafana_data:/var/lib/grafana \
    grafana/grafana
```

Same `monitoring` network so the hostname `course-assistant-pg` resolves. Detached (`-d`); drop it if you want to watch logs during setup. The volume keeps data sources and dashboards across restarts. UI at `http://localhost:3000`, first login admin/admin (it prompts for a new password; admin/admin again is fine locally).

Data source: Configuration → Data Sources → PostgreSQL. Host `course-assistant-pg:5432`, database `course_assistant`, user `user`, password `password`, SSL disabled. Save & Test should answer "Database Connection OK".

### 14.1 Two query habits

Alias the time column as `time` or Grafana cannot place points on the x-axis. And filter every panel on the selected range with `$__timeFrom()` / `$__timeTo()` so panels follow the picker instead of scanning the whole table. `$__timeGroup(column, interval)` buckets rows into intervals; `$__interval` auto-sizes the buckets.

### 14.2 The seven panels

Response time, raw points (one row is already one call):

```sql
SELECT
  timestamp AS time,
  response_time
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
ORDER BY timestamp
```

Time series. Token usage, bucketed averages (raw points would drown long ranges):

```sql
SELECT
  $__timeGroup(timestamp, $__interval) AS time,
  AVG(total_tokens) AS avg_tokens
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY 1
ORDER BY 1
```

Time series. `GROUP BY 1` means the first column, the bucket.

Cost, cumulative per bucket:

```sql
SELECT
  $__timeGroup(timestamp, $__interval) AS time,
  SUM(cost) AS total_cost
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
  AND cost > 0
GROUP BY 1
ORDER BY 1
```

Time series. Model usage:

```sql
SELECT
  model,
  COUNT(*) as count
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY model
```

Bar chart. Judge relevance:

```sql
SELECT
  relevance,
  COUNT(*) as count
FROM feedback
WHERE source = 'judge'
  AND timestamp BETWEEN $__timeFrom() AND $__timeTo()
GROUP BY relevance
```

Pie chart. User feedback:

```sql
SELECT
  SUM(CASE WHEN score > 0 THEN 1 ELSE 0 END) as thumbs_up,
  SUM(CASE WHEN score < 0 THEN 1 ELSE 0 END) as thumbs_down
FROM feedback
WHERE source = 'user'
  AND timestamp BETWEEN $__timeFrom() AND $__timeTo()
```

Gauge or pie. Recent conversations:

```sql
SELECT
  timestamp AS time,
  question,
  answer,
  response_time,
  cost
FROM conversations
WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()
ORDER BY timestamp DESC
LIMIT 5
```

Table.

Set auto-refresh to 30 seconds and a default range like last 6 hours. Suggested layout: wide conversations table on top, model bar and relevance pie in the middle, response time / tokens / cost along the bottom. Panel types are suggestions, not law; try the same query as bar, pie, and series and keep whatever reads best. With the generator still running, charts move while you arrange them.

Speed, cost, relevance, and ratings on one page. That is the monitoring story in a single screen.

---
## 15. One command for everything

Three services started by hand means remembering the network, retyping long commands, and Docker complaining about existing container names on retry. Compose collapses it into one file. Note: neither the compose file nor the Dockerfile ships in `code/`; the lesson describes them and you create them.

Layout:

```text
code/
├── docker-compose.yaml
├── Dockerfile
├── .env
├── pyproject.toml
├── uv.lock
├── .python-version
├── app.py           # Streamlit app
├── assistant.py     # RAG pipeline + LLM
├── db_init.py       # Database init
├── db_save.py       # Save conversations
└── dashboard.py     # Streamlit dashboard
```

The app container builds from python:3.12-slim with uv copied in, deps synced locked, Streamlit on 8501 bound to all interfaces:

```dockerfile
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app
ENV PATH="/app/.venv/bin:$PATH"

COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked

COPY . .

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

`.env` carries Postgres credentials plus `POSTGRES_HOST=postgres` (the compose service name, not localhost: inside the compose network, hosts are service names) and the OpenAI key.

Compose: postgres:17 with env from the file and a `postgres_data` volume; Grafana with a `grafana_data` volume and admin password preset; Streamlit built from the Dockerfile with DB vars plus the API key, both app services depending on postgres.

```bash
docker-compose up
uv run python db_init.py
```

App at 8501, Grafana at 3000, `docker-compose down` to stop. Volumes keep Postgres rows and Grafana dashboards across restarts.

---
## 16. Production reality

What you built is a teaching rig, and the lesson names exactly where reality diverges.

Overhead first. Every call now writes to the database, and the judge adds a whole LLM call. Inline is fine for learning; production pushes both behind the request. Answer the user, then score and store asynchronously, ideally through a queue. Same for storage: Postgres handles our volume, but high-rate event streams belong in something like Kafka with downstream systems doing the keeping.

Second, the build-vs-buy call. Hand-rolled instrumentation captures exactly what you want where you want it, at the cost of writing everything. The packaged alternatives: Langfuse and Arize Phoenix for tracing LLM apps, Pydantic Logfire for near-zero-effort instrumentation plus dashboard (the instructor's pick), Evidently straddling monitoring and evaluation. The tradeoff cuts both ways. Free dashboard, but opaque capture and framework spelunking when you want something different. Try a couple before committing.

Underneath most of the packaged tools sits OpenTelemetry, the instrumentation standard worth learning regardless of which dashboard you end up on. Conceptually your system keeps looking like what you built here. The technology behind it changes; the shape does not.

The module's suggested extensions, if you want more: instrument an agent the same way (each tool call captured like an LLM call), generate synthetic traffic and watch Grafana fill, or move the stack to Compose.

---
## 17. Things to try

1. Ask three questions, then read the raw rows with psql. Match each column to a field on `LLMCallRecord` by hand once; the schema will feel obvious after that.
2. Break the judge on purpose. Ask something the FAQ cannot answer and watch what relevance comes back. Then decide whether the verdict or your expectations were wrong.
3. Vote against yourself. Give a good answer a thumbs-down and find both rows in `feedback`. That pair (same conversation, two sources disagreeing) is what judge alignment runs on.
4. Run the generator for five minutes with Grafana on 30s refresh. Watch each panel fill and learn which queries feel live vs static.
5. Change a panel type on the same query (bar, pie, series) and keep the one you read fastest. There is no correct answer, only faster reading.
6. Sample the judge. Edit `app.py` to judge roughly one call in ten and compare dashboard behavior and cost. This is the cheapest production lesson in the module.
7. Time the `timestamp` vs `id` ordering on a full table and decide whether the index conversation from section 8 was premature or overdue.
8. Swap SQLite in for Postgres plus the Streamlit dashboard only, no Docker. See how far the simple stack goes before you miss Grafana.

---
## 18. Architecture review

```text
Serve (per question)
  Streamlit ask → RAG (search, prompt, answer) → metrics captured → judge scores
        ↓
Store (per question)
  conversations row (RETURNING id) → feedback rows (user score, judge verdict)
        ↓
Read (per dashboard load)
  get_conversations / get_stats / relevance + feedback counts
        ↓
See (two dashboards)
  Streamlit: totals, lines, recent answers, feedback bars
  Grafana: time series, pies, gauges, tables, 30s refresh
        ↓
Fill (while building)
  generate_data.py: one fake conversation per second, both feedback sources
        ↓
Ship (one command)
  compose: postgres + grafana + streamlit, volumes persist
```

Three sentences for anyone asking what the module was about: offline metrics say the system works on frozen questions; monitoring says it keeps working on live ones. You capture per-call receipts, store them with user and judge feedback, and chart the lot. Everything past that, sampling, async writes, Kafka, OpenTelemetry, is the same shape at bigger scale.

---
## 19. What you should now understand

- Offline evaluation and monitoring answer different questions: works vs keeps working. You need both, in that order.
- The RAG pipeline never changes. Monitoring wraps it with interface, storage, and display.
- A dataclass receipt (model, prompt, tokens, time, cost, timestamp) is the unit everything downstream consumes.
- Cost comes from `usage` times per-million rates. Rates move; the module froze two different snapshots, so verify before budgeting.
- The raw question and the model prompt are different columns. Dashboards want the human wording.
- `RETURNING id` exists so feedback rows can point at their conversation. Save-then-rate depends on it.
- Timezone-aware timestamps are not pedantry; Grafana needs the zone to place points.
- Session state bridges Streamlit reruns: the Ask press and the button press are different executions.
- One feedback table with a `source` column lets human and machine verdicts sit side by side for comparison.
- The online judge is harder than the offline one (no reference answer) and gets three labels plus an explanation to compensate.
- Judge verdicts need alignment against user labels before you trust them at scale.
- Synthetic traffic exists to make dashboards readable during development, with skewed-positive feedback to look like a working system.
- Grafana panels are SQL plus two habits: alias time as `time`, filter on the selected range.
- Streamlit suffices until you need alerting, richer panels, or many sources. That is the honest Grafana threshold.
- Production moves writes and judging off the request path and moves high-volume logs off Postgres.

---
## 20. Self-test

Try these from memory. Answers follow.

1. What three pieces does monitoring add, and what job does each do?
2. Why doesn't the RAG pipeline change in this module?
3. What nine fields (plus default) make up `LLMCallRecord`, and why is each kept?
4. How is per-call cost computed, and where do the token counts come from?
5. Why subclass `RAGBase` instead of editing it?
6. What breaks if two requests share one `RAGWithMetrics` instance?
7. Why does Postgres get its own database used by nothing else?
8. Why is `timestamp` timezone-aware?
9. What does `RETURNING id` buy you?
10. Why store `question` separately from `prompt`?
11. What does `row_to_record` fix?
12. When is the Streamlit dashboard enough, and what does Grafana add?
13. What do the `source` values in `feedback` mean, and why one table?
14. How does `st.session_state.conversation_id` survive Streamlit reruns?
15. Why does the online judge get three labels while the offline judge got two?
16. Why does the judge return an explanation nobody displays?
17. Name three production adjustments for the judge and the writes.
18. What do the synthetic feedback probabilities (0.7 judge, 0.5 user, 4:1 positive) simulate?
19. What two SQL habits do Grafana panels need, and why?
20. Your dashboard shows a wave of thumbs-down in the last hour. What do you do?

---

Answers:

1. Interface (where people ask), database (where interactions land), dashboard (where rows become charts).
2. Monitoring observes the pipeline; changing it would mix the thing measured with the measurement. The wrap works for any answer-producing pipeline, agents included.
3. Model, prompt, instructions, answer (what went in/out); prompt/completion/total tokens (consumption); response time (user-felt latency); cost (money); timestamp defaulting to now (ordering and charts).
4. Input tokens × input rate plus output tokens × output rate, per million. Counts come from `response.usage` on the LLM response.
5. One method overridden (`llm`), everything else inherited. Same pattern as module 4's usage tracker; minimal diff, no fork.
6. Both calls share `last_call`; the second overwrites the first's record. Fine for one user, wrong for concurrent traffic.
7. Monitoring storage stays isolated: no contention with app data, free choice of schema and retention, Grafana reads without touching anything operational.
8. Grafana places points using the time column; naive timestamps misalign on its axis. The lesson calls this out explicitly.
9. The new row's id, needed to attach feedback (user votes, judge verdicts) to the right conversation.
10. The prompt is model input shaped by code; the question is what the human typed. Dashboards and debugging want the human version verbatim.
11. Positional tuples (remembering column 4 is the model) become named dataclass fields the rest of the code already uses.
12. Streamlit covers totals, lines, and recent answers. Grafana adds panel variety, multiple sources, and alerting. No alerting need, no Grafana need.
13. `user` for thumbs votes, `judge` for relevance verdicts. One table lets dashboards compare machine opinion against human opinion directly.
14. It doesn't survive automatically; the code writes it there after Ask (`st.session_state.conversation_id = conversation_id`) and the button handlers read it back on the next rerun.
15. Live answers are rarely cleanly right or wrong, and there is no reference answer to compare against. Three labels plus careful instructions compensate.
16. Reasoning before labeling improves the label. The field also gives humans something to read when auditing verdicts.
17. Run judging asynchronously off the request path, track judge cost separately, sample (e.g. one in ten) instead of judging everything. Same async treatment for DB writes; Kafka-style pipeline at scale.
18. A mostly-healthy system: most answers judged, about half voted on, users happy ~80% of the time. Charts with shape instead of flat zeros.
19. Alias the time column as `time` (Grafana needs it for the x-axis) and filter `WHERE timestamp BETWEEN $__timeFrom() AND $__timeTo()` (panels follow the selected range).
20. Treat it as an incident signal, not a metric to admire. Read the recent conversations, check relevance verdicts for the same window, look at what changed upstream (retrieval, prompt, model), fix, watch the panels recover.

---
## 21. Where to go from here

Instrumentation skills transfer: point the same receipt-capture at an agent's tool calls and you have agent monitoring for free. Try Logfire first if a framework appeals (least code for a working dashboard), Phoenix or Langfuse if tracing is the itch, Evidently if you want eval and monitoring in one place. Learn OpenTelemetry regardless; it sits under most of them.

Older cohort material took a different road (2024 used Elasticsearch and local models via Ollama) if you want contrast. The 2026 homework file is currently a TODO placeholder, so the lesson-14 extensions are your assignment in the meantime: agent instrumentation, synthetic traffic plus Grafana, full Compose stack.
