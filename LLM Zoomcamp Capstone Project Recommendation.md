# LLM Zoomcamp Capstone Project Recommendation

## Improved Prompt

Here is a clearer, more complete version of your request:

> "I'm working on the DataTalksClub LLM Zoomcamp capstone project and want to finish it in 3 days. I have a transcript from an agentic AI crash course (LangChain, multi-source RAG, shopping agent, telecom RAG chatbot) and I'm interested in supply chain / operations research applications. Given my background in operations research, data engineering, and LLM applications, recommend the best project idea that: (1) covers the full LLM Zoomcamp evaluation rubric, (2) leverages my OR/supply chain domain knowledge, (3) borrows architecture patterns from the agentic AI transcript, (4) is achievable in 3 days, and (5) is genuinely impactful for my portfolio and learning. Provide a comprehensive review including architecture, data sources, rubric coverage, a 3-day execution plan, and what to avoid."

---

## LLM Zoomcamp Capstone — Official Requirements

The LLM Zoomcamp capstone requires you to build a complete, end-to-end RAG application demonstrating mastery of all course concepts. You must build ([DataTalksClub Blog](https://datatalks.club/blog/llm-zoomcamp.html)):

1. **A searchable knowledge base** — Choose a dataset, ingest, clean, and store it for retrieval
2. **A retrieval pipeline** — Implement the full RAG flow: retrieve context, assemble prompts, call an LLM, return grounded answers
3. **An evaluation process** — Measure retrieval and answer quality using search metrics or LLM-as-a-Judge
4. **A user-facing interface** — A simple UI or API (Streamlit, FastAPI, or similar)
5. **Monitoring & feedback loops** — Track queries, feedback, and performance over time

To earn a certificate, you must: (1) complete the capstone project, (2) peer review 3 other students' projects, and (3) meet submission deadlines. The passing threshold is **11 points** (out of 26 maximum) ([DataTalksClub FAQ](https://datatalks.club/faq/llm-zoomcamp.html), [2026 Dashboard](https://courses.datatalks.club/llm-zoomcamp-2026/dashboard)).

### Evaluation Rubric (from [GitHub project.md](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md))

Each criterion is scored 0-2 points. Total: **18 core points + 8 bonus points = 26 maximum**.

| Criterion | 0 Points | 1 Point | 2 Points |
|---|---|---|---|
| **Problem Description** | Not described | Briefly/unclearly described | Well-described, clear what problem it solves |
| **Retrieval Flow** | No knowledge base or LLM | No knowledge base, direct LLM query | Both knowledge base AND LLM used |
| **Retrieval Evaluation** | No evaluation | One retrieval approach evaluated | Multiple approaches evaluated, best selected |
| **LLM Evaluation** | No evaluation | One approach evaluated | Multiple approaches evaluated, best selected |
| **Interface** | No interaction | CLI, script, or Jupyter notebook | UI, web app, or API |
| **Ingestion Pipeline** | No ingestion | Semi-automated (Jupyter) | Automated Python script or tool |
| **Monitoring** | No monitoring | Feedback OR dashboard | Feedback AND dashboard with 5+ charts |
| **Containerization** | None | Dockerfile OR docker-compose for deps only | Everything in docker-compose |
| **Reproducibility** | No instructions/missing data | Incomplete instructions OR missing data | Clear instructions, accessible data, working code, versions specified |

### Bonus Points

- **Hybrid search** (combining text + vector search, at least evaluating it) — 1 point
- **Document re-ranking** — 1 point
- **User query rewriting** — 1 point
- **Cloud deployment** — 2 points
- **Extra bonus** (innovative features, up to reviewer discretion) — up to 3 points

### Key FAQ Insights ([DataTalksClub FAQ](https://datatalks.club/faq/llm-zoomcamp.html))

- **No minimum dataset size** — choose enough to demonstrate and meaningfully evaluate your flow
- **Structured data is allowed** — you can use database schemas, table descriptions, or records; you still need to retrieve/query data as part of the RAG or agent flow
- **Self-created datasets are fine** — a static file or API-backed source works; provide a script that lets reviewers rebuild the knowledge base
- **Python script for ingestion = full 2 points** — a Jupyter notebook only earns 1 point
- **Orchestrators are optional** — only use one if it genuinely fits (e.g., recurring daily ingestion)

---

## Recommended Project: Inventory Replenishment & Supplier Delay Copilot

### The Idea

Build a RAG-based decision support assistant that answers supply chain operations questions by:

1. **Retrieving from a knowledge base** of supply chain policies, SOPs, and supplier documentation (unstructured documents)
2. **Querying a structured dataset** of inventory, suppliers, and purchase orders (CSV/SQLite)
3. **Generating grounded recommendations** that combine policy context with calculated operational facts (stockout risk, reorder points, supplier delays)

This is not a broad "supply chain control tower" — it is one focused vertical slice: **inventory replenishment decisions and supplier delay analysis**.

### Why This Is the Best Fit for You

| Factor | Why It Matches |
|---|---|
| **Your OR background** | Inventory management, reorder points, safety stock, lead times, and decision support are core OR concepts you already understand |
| **Your data engineering trajectory** | You'll build an ingestion pipeline, work with mixed data sources (docs + structured data), and design a retrieval architecture |
| **LLM Zoomcamp alignment** | Covers every rubric criterion: knowledge base, RAG flow, evaluation, interface, monitoring, Docker, reproducibility |
| **Transcript patterns** | The agentic AI crash course transcript you provided covers multi-source RAG (CSV + SQLite + PDF), evaluation with gold-standard Q&A, and Streamlit UIs — all directly transferable |
| **3-day feasibility** | One narrow domain (inventory/supplier ops) with a small self-contained dataset, not a full multi-agent system |
| **Portfolio impact** | Demonstrates domain expertise + technical RAG skills + evaluation rigor — exactly what differentiates an AI Systems Engineer from a generic LLM developer |

### What the Transcript Teaches You (and How We Adapt It)

The transcript is an agentic AI / LangChain crash course that covers ([from the transcript you added](file://transcript)):

- **Multi-source RAG** — using CSV, SQLite, and PDF as knowledge sources in a single RAG system (the telecom chatbot project)
- **Evaluation** — creating gold-standard Q&A pairs and evaluating agent responses
- **Multi-step reasoning** — agents that decide which tools to call (database, review API, checkout)
- **Streamlit UI** — both projects use Streamlit as the interface
- **Tool use / function calling** — LLMs deciding autonomously which data source to query

**How we adapt it**: Instead of telecom data, we use supply chain data. Instead of a telecom FAQ chatbot, we build a decision support copilot that retrieves policy context AND queries inventory data to answer operational questions. The architecture patterns (multi-source ingestion, chunking, evaluation, Streamlit UI) are identical.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    USER (Streamlit UI)                    │
│   "Which SKUs are at stockout risk this week?"           │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              RAG ORCHESTRATION LAYER                      │
│                                                          │
│  1. Query understanding (optional: query rewriting)      │
│  2. Retrieve from knowledge base (policies/SOPs)          │
│  3. Query structured data (inventory/supplier tables)    │
│  4. Combine context + calculated facts                   │
│  5. Generate grounded answer with citations              │
└──────────┬───────────────────────────┬──────────────────┘
           │                           │
           ▼                           ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  KNOWLEDGE BASE      │  │  STRUCTURED DATA              │
│  (Vector Store)      │  │  (SQLite / Pandas)            │
│                      │  │                               │
│  • Inventory policy  │  │  • inventory.csv              │
│  • Supplier SLAs    │  │  • suppliers.csv              │
│  • Reorder policy    │  │  • purchase_orders.csv        │
│  • Exception SOPs    │  │  • demand_history.csv         │
│  • Data dictionary   │  │                               │
└──────────────────────┘  └───────────────────────────────┘
           │                           │
           ▼                           ▼
┌──────────────────────┐  ┌───────────────────────────────┐
│  INGESTION PIPELINE  │  │  ANALYTICS LAYER              │
│  (Python script)      │  │  (Pandas / SQL functions)     │
│                      │  │                               │
│  • Load & chunk docs  │  │  • Stockout days calculation  │
│  • Embed chunks       │  │  • Reorder point flag         │
│  • Index in vector DB│  │  • Supplier delay risk        │
│  • Load CSV → SQLite  │  │  • Safety stock check         │
└──────────────────────┘  └───────────────────────────────┘
```

### Component Details

| Component | Technology | Purpose |
|---|---|---|
| Knowledge base (unstructured) | Text documents (markdown/PDF) | Supply chain policies, SOPs, supplier SLAs, reorder policies |
| Knowledge base (structured) | SQLite + CSV files | Inventory, suppliers, purchase orders, demand history |
| Vector store | Elasticsearch, Qdrant, or pgvector | Index and search document chunks |
| Embedding model | OpenAI text-embedding-3-small or sentence-transformers | Convert text to vectors |
| LLM | OpenAI GPT-4o-mini or Gemini (via OpenRouter) | Generate grounded answers |
| RAG framework | LangChain or custom Python | Orchestrate retrieval + generation |
| Interface | Streamlit | User-facing chat/Q&A interface |
| Monitoring | SQLite query log + Streamlit dashboard | Track queries, feedback, performance |
| Containerization | Docker + docker-compose | Reproducible deployment |

---

## Data Sources

### Knowledge Base (Unstructured Documents)

Create 5-8 markdown documents covering supply chain operations policies:

1. **Inventory Management Policy** — reorder point formula, safety stock calculation, ABC classification
2. **Supplier SLA Guidelines** — lead time expectations, delay thresholds, escalation procedures
3. **Reorder Policy SOP** — when to reorder, approval workflow, exception handling
4. **Stockout Prevention Procedures** — risk indicators, mitigation steps, escalation
5. **Supplier Performance Review** — evaluation criteria, scorecard methodology
6. **Exception Handling SOP** — what to do when demand spikes, supplier fails, etc.
7. **Data Dictionary** — definitions of SKU, lead time, safety stock, reorder point, etc.
8. **Demand Forecasting Guidelines** — methods, data requirements, review cycle

### Structured Data (CSV Files → SQLite)

Create a small self-contained dataset (committed to the repo, no login required):

| File | Columns | Rows (approx) | Description |
|---|---|---|---|
| `inventory.csv` | sku, product_name, current_stock, reorder_point, safety_stock, unit_cost, supplier_id, lead_time_days | 50-100 | Current inventory levels |
| `suppliers.csv` | supplier_id, supplier_name, country, avg_lead_time_days, sla_days, on_time_rate, quality_score | 10-20 | Supplier master data |
| `purchase_orders.csv` | po_id, sku, supplier_id, order_date, expected_date, actual_date, quantity, status | 100-200 | Purchase order history |
| `demand_history.csv` | date, sku, units_sold | 500-1000 | Historical demand by SKU |

**Why self-contained**: The FAQ explicitly states self-created datasets are allowed and a script to rebuild the knowledge base earns full reproducibility points. No Kaggle login or API setup needed for reviewers.

---

## Sample User Questions

These make the project feel real and will form your evaluation dataset:

1. "Which SKUs are at stockout risk this week?"
2. "Why should SKU-014 be reordered now?"
3. "Which supplier delays are violating SLA?"
4. "What action should the planner take for product X?"
5. "Explain the reorder recommendation using company policy."
6. "What is the safety stock level for SKU-032 and how was it calculated?"
7. "Which suppliers have the worst on-time performance?"
8. "What does the exception handling SOP say about demand spikes?"

---

## Rubric Coverage

### Core Criteria (18 points)

| Criterion | How to Score 2 Points | Difficulty |
|---|---|---|
| **Problem Description** (2) | Clear README explaining the inventory replenishment + supplier delay problem and how the copilot solves it | Easy |
| **Retrieval Flow** (2) | Knowledge base (policy docs) + structured data (inventory CSV) + LLM = full RAG flow | Easy |
| **Retrieval Evaluation** (2) | Compare vector search vs text search vs hybrid search; use hit rate and MRR metrics | Medium |
| **LLM Evaluation** (2) | Compare 2-3 prompt templates and/or 2 LLM models using LLM-as-a-Judge | Medium |
| **Interface** (2) | Streamlit chat interface with query input, answer display, and source citations | Easy |
| **Ingestion Pipeline** (2) | Python script (`ingest.py`) that loads docs, chunks them, embeds, indexes in vector DB, and loads CSV to SQLite | Medium |
| **Monitoring** (2) | User feedback (thumbs up/down) + Streamlit dashboard with 5+ charts (query volume, response times, feedback distribution, popular topics, retrieval scores) | Medium |
| **Containerization** (2) | docker-compose with app, vector DB, and SQLite | Medium |
| **Reproducibility** (2) | Clear README, data committed to repo, versions in requirements.txt, one-command setup | Easy |

### Bonus Points (up to 8)

| Bonus | Points | Priority | How |
|---|---|---|---|
| **Hybrid search** | 1 | Should-have | Combine BM25 text search + vector search with boosting; evaluate vs each alone |
| **Query rewriting** | 1 | Should-have | Rewrite user queries before retrieval (e.g., expand "SKU-014 stockout" to full query) |
| **Re-ranking** | 1 | Stretch | Re-rank retrieved chunks using a cross-encoder or LLM re-ranker |
| **Cloud deployment** | 2 | Stretch | Deploy to AWS EC2 (you're already learning this) or Google Cloud Run |
| **Extra** | up to 3 | Stretch | OR calculation tool (reorder point, safety stock), supplier scorecard visualization |

### Priority Tiers

- **Must-have (Day 1-2)**: All 18 core rubric points — this alone exceeds the 11-point passing threshold
- **Should-have (Day 2-3)**: Hybrid search + query rewriting (+2 bonus points)
- **Stretch (Day 3 or later)**: Re-ranking, cloud deployment, OR calculation tools

---

## 3-Day Execution Plan

### Day 1: Data, Ingestion, and Retrieval (6-8 hours)

| Step | Task | Time |
|---|---|---|
| 1 | Define the problem statement and write initial README | 30 min |
| 2 | Create knowledge base documents (5-8 markdown files with supply chain policies) | 1 hr |
| 3 | Create structured dataset (4 CSV files with sample inventory/supplier/PO/demand data) | 1 hr |
| 4 | Build `ingest.py` — load docs, chunk, embed, index in vector DB (Elasticsearch or Qdrant) | 2 hrs |
| 5 | Load CSV files into SQLite | 30 min |
| 6 | Implement basic RAG flow: query → retrieve → prompt → LLM → answer | 1.5 hrs |
| 7 | Test with sample questions | 30 min |

**End of Day 1**: Working RAG pipeline that retrieves from policy docs and answers questions.

### Day 2: Evaluation, Hybrid Search, and LLM Comparison (6-8 hours)

| Step | Task | Time |
|---|---|---|
| 1 | Create gold-standard evaluation dataset (15-20 Q&A pairs with expected answers) | 1 hr |
| 2 | Implement retrieval evaluation: hit rate, MRR for vector vs text vs hybrid | 1.5 hrs |
| 3 | Implement hybrid search (BM25 + vector with boosting) and compare | 1 hr |
| 4 | Implement LLM-as-a-Judge evaluation (compare 2-3 prompt templates) | 1.5 hrs |
| 5 | Add query rewriting (optional — rewrite queries before retrieval) | 1 hr |
| 6 | Add re-ranking (optional — cross-encoder or LLM re-ranker) | 1 hr |

**End of Day 2**: Full RAG pipeline with evaluation results showing which approach works best.

### Day 3: Interface, Monitoring, Docker, and Deploy (6-8 hours)

| Step | Task | Time |
|---|---|---|
| 1 | Build Streamlit UI — chat interface with source citations | 1.5 hrs |
| 2 | Add user feedback collection (thumbs up/down + text feedback) | 30 min |
| 3 | Build monitoring dashboard with 5+ charts inside Streamlit | 1.5 hrs |
| 4 | Write `docker-compose.yml` (app + vector DB + SQLite) | 1 hr |
| 5 | Finalize README with setup instructions, architecture diagram, evaluation results | 1 hr |
| 6 | Test full reproducibility: clone → docker-compose up → working app | 30 min |
| 7 | (Stretch) Deploy to AWS EC2 or Google Cloud Run | 1-2 hrs |

**End of Day 3**: Complete, reproducible, Dockerized RAG application ready for submission.

---

## Technical Stack

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | Your primary language |
| RAG framework | LangChain or custom Python classes | Transcript uses LangChain; you can use either |
| Vector store | Elasticsearch (local) or Qdrant | Elasticsearch is covered in the course |
| Embeddings | OpenAI text-embedding-3-small or sentence-transformers | OpenAI is simpler; sentence-transformers is free |
| LLM | OpenAI GPT-4o-mini or Gemini via OpenRouter | You already use OpenRouter and Gemini APIs |
| Structured data | SQLite + Pandas | Simple, no external database needed |
| UI | Streamlit | Covered in transcript and course |
| Monitoring | SQLite query log + Streamlit dashboard | No Grafana needed — simpler for 3 days |
| Containerization | Docker + docker-compose | You're learning Docker fundamentals |

---

## What to Avoid

1. **Don't build a multi-agent system** — A single RAG pipeline with one optional tool function (inventory calculation) is enough. Multi-agent systems are harder to evaluate and debug in 3 days.

2. **Don't use Kaggle datasets requiring login** — Reviewers need to reproduce your project. Use self-contained data committed to the repo.

3. **Don't skip evaluation** — Retrieval evaluation (multiple approaches) and LLM evaluation (multiple prompts) are each worth 2 points. This is 4 easy points you cannot afford to lose.

4. **Don't use a Jupyter notebook for ingestion** — A Python script earns 2 points; a notebook only earns 1. The FAQ is explicit about this.

5. **Don't over-scope the monitoring** — 5 simple charts inside Streamlit is sufficient. Don't try to set up Grafana + Prometheus in 3 days.

6. **Don't reuse course homework datasets** — The FAQ explicitly prohibits this. Your supply chain data must be original.

7. **Don't skip the README** — Reproducibility (2 points) is about clear instructions, accessible data, and working code. A good README is the easiest 2 points.

8. **Don't try to do everything in the bonus section** — Focus on core 18 points first (passing = 11). Hybrid search and query rewriting are achievable. Cloud deployment is a stretch goal.

---

## Why This Project Stands Out for Your Portfolio

This project is strategically positioned at the intersection of your three career goals:

1. **Operations Research + AI**: The domain (inventory, reorder points, supplier management) is pure OR. The AI layer (RAG, retrieval, evaluation) demonstrates you can apply LLMs to real operational decisions — not just chatbots.

2. **Data Engineering Pipeline**: The ingestion pipeline (load → chunk → embed → index → store) is a data engineering pattern that translates directly to production AI systems. Working with mixed data sources (documents + structured data) is a skill that enterprise AI teams value.

3. **AI Systems Engineering**: Evaluation (multiple retrieval approaches, LLM-as-a-Judge), monitoring (feedback loops, dashboards), and containerization (Docker) demonstrate production-readiness — the difference between someone who can prompt an LLM and someone who can build a reliable AI system.

This project tells a coherent story: "I understand operations, I can build data pipelines, and I can deploy and evaluate AI systems." That is exactly the profile of a Decision Intelligence Architect.

---

## Summary

| Aspect | Detail |
|---|---|
| **Project** | Inventory Replenishment & Supplier Delay Copilot |
| **Type** | RAG-based decision support assistant |
| **Domain** | Supply chain / operations research |
| **Data** | Self-created policy docs (markdown) + sample inventory CSVs |
| **Rubric coverage** | All 18 core points achievable + 2-4 bonus points |
| **Time to complete** | 3 days (6-8 hours/day) |
| **Passing threshold** | 11 points (easily exceeded) |
| **Key technologies** | Python, LangChain, Elasticsearch/Qdrant, Streamlit, Docker, OpenAI/Gemini |
| **What you learn** | Multi-source RAG, retrieval evaluation, LLM-as-a-Judge, monitoring, containerization |
| **What it demonstrates** | OR domain expertise + data engineering + AI systems engineering |

---

*Sources: [LLM Zoomcamp Blog](https://datatalks.club/blog/llm-zoomcamp.html), [LLM Zoomcamp FAQ](https://datatalks.club/faq/llm-zoomcamp.html), [GitHub project.md](https://github.com/DataTalksClub/llm-zoomcamp/blob/main/project.md), [DeepWiki Evaluation Criteria](https://deepwiki.com/DataTalksClub/llm-zoomcamp/4-projects-and-assessment), [2026 Course Dashboard](https://courses.datatalks.club/llm-zoomcamp-2026/dashboard), [DataTalksClub Documentation](https://datatalks.club/docs/courses/llm-zoomcamp/project/). Transcript content from the file you uploaded to this project.*
