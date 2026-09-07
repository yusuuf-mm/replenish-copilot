# Vector Search — Complete Practical Lesson

This lesson reconstructs the entire LLM Zoomcamp Vector Search module into one coherent learning journey.
Follow from section 0 to the end to understand and reproduce the full Vector Search system.

---
## 0. What We Are Building

Vector search is the bridge between raw text and the information an LLM needs to answer questions.
This lesson takes you from raw FAQ documents to a working RAG pipeline that uses semantic similarity instead of exact keyword matching.

### The Complete Architecture

```text
                    ┌─────────────────┐
                    │     Documents   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Chunk / Prepare │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Embedding Model │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Vector Storage  │
                    └────────┬────────┘
                             │
                    User Query
                         │
                         ▼
                    ┌─────────────────┐
                    │ Query Embedding │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Similarity Search│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relevant Context│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │       LLM       │
                    └────────┬────────┘
                             │
                             ▼
                           Answer
```

### What You Will Learn

By the end of this lesson, you will understand:

1. **What embeddings are** — how text becomes a 384-dimensional vector
2. **Why we need vector search** — the limitation of keyword search
3. **How similarity search works** — dot product and cosine similarity
4. **How to store vectors** — in-memory (numpy), SQLite (persistent), PostgreSQL (production)
5. **How vector search connects to RAG** — retrieval ends where generation begins
6. **Three deployment options** — minsearch, sqlitesearch, PGVector
7. **Lightweight embeddings with ONNX** — 33x smaller deployment

---
## 1. Learning Objectives

After completing this lesson, you should be able to:

- Explain why vector search is needed (keyword search cannot match paraphrases)
- Convert text into embeddings using `sentence-transformers` and `all-MiniLM-L6-v2`
- Generate embeddings for a dataset of 1208 FAQ documents
- Perform vector search using NumPy (brute-force dot product)
- Perform vector search using `minsearch.VectorSearch` (in-memory with keyword filtering)
- Perform persistent vector search using `sqlitesearch.VectorSearchIndex` (SQLite with ANN modes)
- Perform production vector search using PostgreSQL + `pgvector` (HNSW indexes)
- Swap vector search into the RAG pipeline by overriding the `search` method
- Use ONNX Runtime for 33x lighter embedding deployment
- Explain the complete data flow from documents to LLM answer
- Debug retrieval by inspecting embeddings, scores, and top-K results

---
## 2. The Big Mental Model

### From Text to Answer: The Complete Flow

```text
User Question
      ↓
Query Embedding (sentence-transformers → all-MiniLM-L6-v2)
      ↓     384-dimensional vector
      ↓
Similarity Search (dot product / cosine distance)
      ↓
Top-K Relevant Documents
      ↓
Retrieved Context (Q + A pairs)
      ↓
Prompt (question + context)
      ↓
LLM (generates answer)
      ↓
Answer
```

**Key insight**: Vector search is **not** the LLM. The vector database does not generate answers — it only finds relevant information. The LLM uses that information to generate the answer.

### Two Stages of Vector Search

**Stage 1: Offline Indexing (once, after data is available)**
- Load the FAQ dataset
- For each document (question + answer), generate an embedding vector
- Store the vectors + metadata in a vector store (numpy array, SQLite, or PostgreSQL)

**Stage 2: Online Querying (for every user question)**
- Encode the user's query into a vector using the same embedding model
- Compare the query vector against all stored document vectors using cosine similarity
- Return the top-K most similar documents
- Pass those documents as context to the LLM

### Why This Two-Stage Design?

The indexing stage is **expensive** — it runs the neural network over every document.
We pay this cost once. The querying stage is **cheap** — we only encode one query vector and compute dot products.
This split enables interactive search without making users wait minutes for each question.

---
## 3. Environment Setup

### 3.1 Install Required Packages

We need the core libraries for this module. If you're starting fresh:

```bash
mkdir llm-zoomcamp-code
cd llm-zoomcamp-code
uv init
uv add sentence-transformers minsearch sqlitesearch openai python-dotenv
```

**Why these packages?**

- `sentence-transformers` — wraps PyTorch and provides the `all-MiniLM-L6-v2` embedding model
- `minsearch` — in-memory vector search library (same API as text search from Module 1)
- `sqlitesearch` — persistent vector search backed by SQLite with ANN modes (lsh/ivf/hnsw)
- `openai` — OpenAI client for the LLM step in RAG
- `python-dotenv` — load environment variables from `.env` file

### 3.2 Environment Variables

Create a `.env` file in your project root:

```text
OPENAI_API_KEY=YOUR_API_KEY
```

**Why?** The `RAGBase` class from Module 1 needs an OpenAI client to generate answers.
Replace `YOUR_API_KEY` with your actual key, or skip the LLM step and just test the search logic.

### 3.3 Verify Your Environment

Run this to verify everything is installed:

```python
import sentence_transformers
import minsearch
import sqlitesearch
import openai
print("All packages imported successfully!")
print(f"sentence-transformers version: {sentence_transformers.__version__}")
```

> We need `sentence-transformers` because it provides the embedding model that turns text into vectors.
> We need `minsearch` and `sqlitesearch` as our vector search backends.
> We need `openai` and `python-dotenv` for the RAG pipeline.

---
## 4. Understanding the Dataset/Documents

### 4.1 Load the FAQ Data

We'll use the same FAQ dataset from Module 1. It contains course-related questions and answers across multiple courses.

```python
from ingest import load_faq_data

documents = load_faq_data()
len(documents)
```

> We end up with 1208 documents (question + answer pairs) across multiple courses:
> data-engineering-zoomcamp, llm-zoomcamp, machine-learning-zoomcamp, mlops-zoomcamp.

### 4.2 Inspect a Sample Document

```python
documents[10]
```

> Each document is a Python dictionary with keys: `id`, `course`, `section`, `question`, `answer`.

### 4.3 Text Preparation: Concatenate Question + Answer

For vector search, we embed the entire question-answer pair together. This way, a query can match against both the question text and the answer text.

```python
texts = []
for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)
```

> **Why concatenate?** If we only embedded the question, we'd lose the answer information.
> By concatenating question + answer, the embedding captures the full meaning,
> and a query can retrieve the relevant answer along with the question.

---
## 5. Embeddings — Introducing the Embedding Model

### 5.1 What Is an Embedding?

An **embedding** is a fixed-length array of numbers that represents text in a high-dimensional vector space.
The key property: **texts with similar meanings land near each other** in this space.

### 5.2 Why Do We Need Embeddings?

Keyword search matches exact words. If you search for "Docker", the document must contain "Docker".
But two questions can mean the same thing while sharing almost no words:

- "Can I still join the course after the start date?"
- "Is it possible to enroll late?"

These mean the same thing but share almost no words. **Vector search matches meaning, not words.**

### 5.3 The Embedding Model: `all-MiniLM-L6-v2`

We'll use `sentence-transformers` with the `all-MiniLM-L6-v2` model:

- **384-dimensional vectors** (compact)
- **Fast on CPU** (no GPU required for experiments)
- **Good quality for general English text**
- **Uses cosine similarity** (dot product of normalized vectors = cosine similarity)

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
```

> **Why this model?** It's a small transformer model (90 MB) trained on sentence pairs so that
> similar sentences land near each other in the 384-dimensional vector space.
> It's the right size for our FAQ dataset of short English texts.

### 5.4 First Experiment: Encode Two Similar Questions

```python
q1 = "Can I still join the course after the start date?"
q2 = "I just found out about the program, can I still enroll?"

v1 = model.encode(q1)
v2 = model.encode(q2)
```

> What just happened? The model turned each question into a 384-dimensional vector.
> These vectors capture the meaning of the questions, not just the words.

### 5.5 Experiment: Compare Similarity

```python
# Dot product gives cosine similarity (vectors are normalized)
similarity = v1.dot(v2)
similarity
```

> We get a score near 1.0 because the two questions mean the same thing.
> A score near 0 would mean the questions are unrelated.

### 5.6 Experiment: Compare Unrelated Questions

```python
q3 = "How do I install Docker on Windows?"
v3 = model.encode(q3)

unrelated_similarity = v1.dot(v3)
unrelated_similarity
```

> We get a score near 0.0 because installing Docker has nothing to do with course enrollment.
> This demonstrates that the embedding model maps semantically similar texts to similar vectors.

---
## 6. Inspecting Embeddings — Dimensions, Shapes, and Examples

### 6.1 Shape of a Single Vector

```python
v1.shape
```

> `(384,)` — a 1-dimensional array of 384 numbers. Each number represents some concept
> the model learned. We cannot read off what any single number means, but combinations of numbers
> capture semantic meaning.

### 6.2 Shape of the Full Embedding Matrix

We'll embed all 1208 documents and organize them as a matrix where:
- **Rows** = documents (vectors)
- **Columns** = dimensions of the vectors (384)

```python
import numpy as np

X = np.array(vectors)
X.shape
```

> `(1208, 384)` — 1208 documents × 384 dimensions.
> Row `i` is the embedding for document `i`. Column `j` is the j-th dimension across all documents.

### 6.3 What Does a Vector Component Represent?

We cannot point to any single number and say "this means 'course enrollment'".
Instead, each dimension is a learned feature that contributes to the overall vector representation.
The model has learned that certain patterns of activations across all 384 dimensions correspond to certain topics, intents, and concepts.

> **Key point**: Vector dimensions are NOT "words". They are learned concepts.
> The model might use dimension 42 to encode "start date" ideas across many different contexts,
> but we cannot read off "dimension 42 = start date" — it's a distributed representation.

### 6.4 Experiment: Inspect Embedding Values

```python
# First 5 components of q1's vector
v1[:5]
```

> You'll see 5 floating-point numbers between -1 and 1 (or whatever range the model outputs).
> The important thing is that these numbers, taken together, capture the meaning of the question.

---
## 7. Preparing Documents — Chunking and Text Preparation

### 7.1 Build One Text Per Document

```python
texts = []
for doc in documents:
    text = doc["question"] + " " + doc["answer"]
    texts.append(text)
```

> **Result**: 1208 texts, one per document, each being "question answer".

### 7.2 Why This Structure?

When a user asks a question, we'll embed their query and compare against these combined question+answer vectors.
The retrieved documents will contain both the original question and the answer,
which we then pass as context to the LLM.

---
## 8. Generating Embeddings for the Full Dataset

### 8.1 Batch Embedding (Crucial for Memory and Observation)

We cannot hand all 1208 texts to the model at once — it would be slow and we couldn't see what's happening inside. Instead, we split into batches.

```python
from tqdm.auto import tqdm

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)
```

> **What `tqdm` does**: Shows a progress bar so we can watch the embedding generation.
> **Why batch_size=50**: A good compromise — large enough to be efficient, small enough to observe.
>
> **Runtime on a typical PC**: the first run downloads the ~90 MB model (~30–60 s),
> then embedding all 1208 texts takes ~2–4 minutes on CPU. No GPU or Jupyter required —
> this runs fine as a plain `.py` script.

### 8.2 Verify the Result

```python
len(vectors)
```

> `1208` — one vector per document.

### 8.3 Convert to NumPy Matrix

```python
X = np.array(vectors)
X.shape
```

> `(1208, 384)` — confirmed: 1208 documents, 384 dimensions each.

---
## 9. Vector Search from First Principles (NumPy)

### 9.1 The Core Idea

We have a matrix `X` with 1208 document vectors. When a query comes in, we:
1. Encode the query into a vector `v_query`
2. Compute the dot product of `v_query` against every document row in `X`
3. Return the top-K documents with highest scores

### 9.2 Encode a Query

```python
query = "Can I still join the course after the start date?"
v_query = model.encode(query)
v_query.shape
```

> `(384,)` — the query vector has the same shape as all document vectors.

### 9.3 Compute Scores Against All Documents

```python
scores = X.dot(v_query)
```

> **What this does**: Matrix-vector multiplication. Each element `scores[i]` is the cosine similarity
> between document `i` (row `i` of `X`) and `v_query`. This is much faster than a Python for loop
> because NumPy runs optimized C code.

### 9.4 Find the Best Match

```python
idx = np.argmax(scores)
idx, scores[idx]
```

> Returns the index and score of the most similar document.
> For this query, document 553 has score 0.76 (high cosine similarity).

### 9.5 Show the Best Match

```python
documents[idx]
```

> We see the document about joining the course late — the right answer for this question!

### 9.6 Top-5 Results (The Idiomatic Way)

Usually we want more than the single best match. Let's get the top 5:

```python
top5 = np.argsort(-scores)[:5]
```

> **Why `-scores`?** `np.argsort` sorts from lowest to highest by default.
> Negating inverts the order, so the largest scores (most similar) come first.
> Then `[:5]` takes the top 5.

### 9.7 Inspect Top-5 Scores and Documents

```python
scores[top5]
```

> Gives us the 5 similarity scores.

```python
for idx in top5:
    print(scores[idx])
    print(documents[idx])
    print()
```

> **Expected output**: 5 documents with decreasing similarity scores,
> starting with the most relevant document for the query.

### 9.8 Shorter Trick (One-Liner)

```python
top5 = np.argsort(-scores)[:5]
```

> This is the idiomatic one-liner for "top-K max sort".
> It looks cryptic the first time, but it's a common pattern.

---
## 10. Vector Search with minsearch

### 10.1 Why Use minsearch?

Writing the numpy argsort code every time gets old, and we can't filter by course.
minsearch wraps all of it with a simple API.

### 10.2 Create the Vector Search Index

We already have our documents and vectors from the previous section.

```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)
```

> **What happens**: The index is built in memory. `keyword_fields=["course"]`
> allows us to filter search results by course later.

### 10.3 Search for a Question

```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vindex.search(query_vector, num_results=5)
```

> **What happens under the hood**: Same thing we just did with numpy —
> compute dot product between each vector (after filtering) and our query vector.

### 10.4 Inspect the Top Result

```python
results[0]
```

> Should return the document about joining the course late:
>
> ```python
> {"id": "74eb249bbf",
>  "course": "llm-zoomcamp",
>  "section": "General Course-Related Questions",
>  "question": "I just discovered the course. Can I still join?",
>  "answer": "Yes, but if you want to receive a certificate, you need to submit your project while we're still accepting submissions."}
> ```

### 10.5 Filtering by Course

Like the text index, we can filter by keyword fields. This matters for user experience —
a student in LLM Zoom Camp doesn't care about answers from the data engineering course.

```python
results = vindex.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

> Now we only get results from the llm-zoomcamp course.

---
## 11. Understanding Search Results — Scores, Ranking, top-K

### 11.1 What Do the Scores Mean?

Each result comes with a similarity score (cosine similarity, range ~[0, 1] for normalized vectors).
Higher = more similar. The scores tell us how close the query vector is to each document vector.

### 11.2 Why Top-5 and Not Top-1?

The answer to a question can be spread across several documents. One holds part of it,
another fills in the rest. Sometimes the top result isn't the right one but the second is.
We send all 5 to the LLM and let it combine them.

### 11.3 Experiment: Change top-K

Compare retrieval with different values of `top_k`:

```python
# top_k = 1
results_one = vindex.search(query_vector, num_results=1)

# top_k = 3
results_three = vindex.search(query_vector, num_results=3)

# top_k = 5
results_five = vindex.search(query_vector, num_results=5)

# top_k = 10
results_ten = vindex.search(query_vector, num_results=10)
```

**What changes?**
- `top_k = 1`: Only the single most similar document is retrieved.
  If the answer is split across documents, we lose information.
- `top_k = 3`: More context, but still limited.
- `top_k = 5`: The default — enough context for the LLM to work with.
- `top_k = 10`: More context, but also more noise.
  The LLM has to read through more irrelevant information.

> **Rule of thumb**: Start with `top_k = 5`. Evaluate whether 3 or 10 works better for your data.

### 11.4 Experiment: Inspect Retrieved Context

```python
for result in results_five:
    print(f"Question: {result['question']}")
    print(f"Answer: {result['answer'][:100]}...")  # first 100 chars
    print(f"Similarity: {result.get('score', 'N/A')}")
    print()
```

> This shows exactly what information is returned before it is passed to the LLM.
> The LLM will use this as context to generate its answer.

---
## 12. Connecting Retrieval to RAG

### 12.1 The RAG Pipeline from Module 1

In Module 1, we built a RAG pipeline with three steps:

```python
def rag(question):
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)
```

The `search` step used keyword search. Now we swap in vector search.

### 12.2 The RAGBase Class

From Module 1, we have `RAGBase` with methods: `search`, `build_context`, `build_prompt`, `llm`, `rag`.

```python
from rag_helper import RAGBase
```

### 12.3 Why We Can't Pass minsearch VectorSearch Directly

Text search takes the query string directly, but vector search needs the query as a vector first.
So we subclass `RAGBase` and override `search` to encode the query before searching.

### 12.4 The RAGVector Subclass

```python
class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )
```

> **What this does**:
> 1. `__init__` adds an `embedder` argument for the sentence transformer
> 2. `search` encodes the query into a vector, then queries `vindex` with that vector
> 3. Everything else (prompt building, LLM call) is inherited from `RAGBase`

### 12.5 Initialize the Vector-Enabled RAG Assistant

```python
vector_assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=openai_client,
)
```

### 12.6 Test It

```python
query = "I just found out about the program, can I still sign up?"
vector_assistant.rag(query)
```

> The answers should be close to what we got with keyword search,
> but vector search handles rephrased questions better.

> **Critical understanding**:
> - Vector search is responsible for **finding relevant information**
> - The LLM is responsible for **using that information to generate an answer**
> - Retrieval ends after step 3 (returning relevant documents)
> - Generation begins after step 3 (LLM produces the answer)

---
## 13. Persistent Vector Search with sqlitesearch

### 13.1 Why sqlitesearch?

minsearch keeps everything in memory and rebuilds the index on every startup.
For a real application, we need persistence — the index should survive process restarts.

### 13.2 Install sqlitesearch

```bash
uv add sqlitesearch
```

### 13.3 Initialize the VectorSearchIndex

Three ANN modes are available:
- `lsh` (default): up to 100K vectors, random hyperplane projections
- `ivf`: 10K-500K vectors, K-means clustering
- `hnsw`: 10K-1M+ vectors, proximity graph (highest recall)

For our small dataset (1208 vectors), `lsh` is fine.

```python
from sqlitesearch import VectorSearchIndex

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```

> **Why `ivf`**: Good for 10K-500K vectors. Our dataset is smaller, but any mode works.

### 13.4 Index the Data

```python
vs_index.fit(vectors, documents)
```

> **Important**: The index is saved to `faq_vectors2.db`. Unlike minsearch,
> this file persists on disk. We can search immediately, or reopen the index later
> without re-indexing.

### 13.5 Search with Course Filtering

```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)

results = vs_index.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

> Same API as minsearch, but the data lives on disk.

### 13.6 Close the Connection

```python
vs_index.close()
```

> Properly closes the SQLite connection.

### 13.7 Reopening the Index (Without Re-Embedding)

In a new Python session, we can reopen the index without re-computing embeddings:

```python
from sentence_transformers import SentenceTransformer
from sqlitesearch import VectorSearchIndex

model = SentenceTransformer("all-MiniLM-L6-v2")

vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)
```

> **Key insight**: We still load the embedding model to encode the query,
> but we don't re-embed all the documents. The index is already built and waiting on disk.

### 13.8 Using sqlitesearch Vector Search in RAG

```python
from rag_helper import RAGBase
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

# Open the index (same as the "Reopening the index" section above)
vs_index = VectorSearchIndex(
    keyword_fields=["course"],
    mode="ivf",
    db_path="faq_vectors2.db"
)

class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )

vector_assistant = RAGVector(
    embedder=model,
    index=vs_index,
    llm_client=openai_client,
)

vector_assistant.rag("the program has already begun, can I still sign up?")
```

> **Same pattern** as with minsearch, but the index lives persistently in SQLite.

### 13.9 Comparing minsearch and sqlitesearch

| | minsearch VectorSearch | sqlitesearch VectorSearchIndex |
|---|---|---|
| Storage | In-memory (numpy) | SQLite file on disk |
| Search type | Exact cosine (brute force) | ANN (LSH/IVF/HNSW) with exact rerank |
| Persists across restarts | No | Yes |
| Setup required | None | None |
| Concurrent access | No | Limited |
| Best for | Experiments, notebooks | Projects, persistence |

> sqlitesearch is probably the last you'll hear of it for teaching — it was built to show
> the ingestion-then-deployment split. For most work you'll reach for PGVector.

---
## 14. Production Vector Search with PGVector

### 14.1 Why PGVector?

For production systems, we need a real database with concurrent access, transactions,
and the ability to handle millions of records. PGVector turns PostgreSQL into a vector database.

### 14.2 Start Postgres with pgvector

```bash
docker run -it \
    --name pgvector \
    -e POSTGRES_USER=user \
    -e POSTGRES_PASSWORD=pswd \
    -e POSTGRES_DB=faq \
    -v pgvector_data:/var/lib/postgresql/data \
    -p 5432:5432 \
    pgvector/pgvector:pg17
```

> This image has pgvector pre-installed. The `-v` flag creates a named volume
> so data persists across container restarts.

### 14.3 Install the Python Client

```bash
uv add psycopg[binary]
```

> We use `psycopg` (v3) which supports `conn.execute()` directly without creating a cursor.
> Note: this is different from `psycopg2`.

### 14.4 Connect to Postgres and Enable pgvector

```python
import psycopg

conn = psycopg.connect(
    "postgresql://user:pswd@localhost:5432/faq"
)
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
```

> The second line activates pgvector. It adds the `vector` column type
> and the `<=>` similarity search operator.

### 14.5 Prepare the Data

We need the FAQ documents and their embeddings. Here's the script from the lessons:

```python
from ingest import load_faq_data
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

documents = load_faq_data()
texts = [doc["question"] + " " + doc["answer"] for doc in documents]

batch_size = 50
vectors = []

from tqdm.auto import tqdm

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)
```

### 14.6 Create the Table

```python
conn.execute("""
    DROP TABLE IF EXISTS documents
""")

conn.execute("""
    CREATE TABLE documents (
        id SERIAL PRIMARY KEY,
        course TEXT,
        section TEXT,
        question TEXT,
        answer TEXT,
        embedding vector(384)
    )
""")
```

> The `vector(384)` column stores our 384-dimensional embeddings from `all-MiniLM-L6-v2`.

### 14.7 Insert Documents with Embeddings

We need a helper function to convert a numpy vector to a Postgres-compatible string:

```python
def vec_to_str(vector):
    return "[" + ",".join(str(x) for x in vector) + "]"
```

Then loop over documents and insert each one:

```python
for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (doc["course"], doc["section"], doc["question"], doc["answer"],
         vec_to_str(vec))
    )

conn.commit()
```

> We hand Postgres the vector as text, and the `::vector` cast tells it to
> parse that string back into a vector.

### 14.8 Search with Cosine Similarity

```python
query = "I just discovered the course. Can I still join it?"
query_vector = model.encode(query)
query_str = vec_to_str(query_vector)

results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, query_str)
).fetchall()

for row in results:
    print(f"[{row[0]}] {row[1]} (similarity: {row[3]:.4f})")
```

> **Key**: The `<=>` operator computes cosine distance (1 - cosine similarity).
> We order by ascending distance, so the closest vectors come first.
> Subtracting from 1 converts it to similarity.

### 14.9 Filtering by Course

```python
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE course = %s
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, "llm-zoomcamp", query_str)
).fetchall()
```

> Because this is plain SQL, filtering by course is just an extra `WHERE` clause.

### 14.10 Creating an Index for Faster Search

For a larger dataset, create an HNSW index:

```python
conn.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
""")
```

> This builds an HNSW (Hierarchical Navigable Small World) index, the same state-of-the-art algorithm
> dedicated vector databases use. It makes search faster at the cost of a small accuracy trade-off.

### 14.11 Wrapping Search in a Function

```python
def pgvector_search(query, course="llm-zoomcamp", num_results=5):
    query_vector = model.encode(query)
    query_str = vec_to_str(query_vector)
    rows = conn.execute(
        """
        SELECT course, section, question, answer
        FROM documents
        WHERE course = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (course, query_str, num_results)
    ).fetchall()

    return [
        {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
        for r in rows
    ]
```

### 14.12 Using PGVector in RAG

We take the same `search` function and move it into a class. We pass the Postgres connection
instead of an index. We set `index=None` because `RAGBase` expects an index and would complain otherwise.

```python
from rag_helper import RAGBase

class RAGPgVector(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {"course": r[0], "section": r[1], "question": r[2], "answer": r[3]}
            for r in rows
        ]
```

Initialize:

```python
vector_assistant = RAGPgVector(
    embedder=model,
    conn=conn,
    llm_client=openai_client,
)
```

Test:

```python
vector_assistant.rag("the program has already begun, can I still sign up?")
```

### 14.13 Comparing the Three Backends

| | minsearch | sqlitesearch | PGVector |
|---|---|---|---|
| Storage | In-memory (numpy) | SQLite file on disk | PostgreSQL in Docker |
| Search type | Exact cosine (brute force) | ANN (LSH/IVF/HNSW) | ANN (HNSW) |
| Persists across restarts | No | Yes | Yes |
| Setup required | None | None | Docker + Postgres |
| Concurrent access | No | Limited | Full |
| Best for | Experiments, learning | Pet projects, demos | Production |

> **Reach for PGVector when you need production features**:
> concurrent reads and writes, transactions, integration with an existing Postgres-based application.

---
## 15. Lightweight Embeddings with ONNX Runtime

### 15.1 Why ONNX Runtime?

When you move to production, you want to cut overhead — both the dependencies
and the size of your deployment. `sentence-transformers` drags in PyTorch plus a pile of Nvidia libraries,
which is a lot (4.8 GB, 58 packages). ONNX Runtime serves the same model without that weight.

### 15.2 Size Comparison

- `sentence-transformers`: 4.8 GB, 58 packages
- `ONNX Runtime`: 147 MB, 27 packages

That's 33x smaller for the same embeddings and the same results.

### 15.3 Setup a Separate Project

```bash
mkdir llm-zoomcamp-onnx && cd llm-zoomcamp-onnx
uv init --no-workspace
uv add onnxruntime tokenizers numpy tqdm minsearch
uv add --dev huggingface-hub jupyter
uv run python -m ipykernel install --user --name llm-zoomcamp-onnx --display-name "llm-zoomcamp-onnx"
```

### 15.4 Download the ONNX Model

```bash
uv run python download.py
```

> This creates: `models/Xenova/all-MiniLM-L6-v2/tokenizer.json` and `model.onnx`

### 15.5 The Embedder Class

Copy `embedder.py` from the `embed/` directory. It does four things under the hood:

1. **Tokenize** - convert text into integer IDs and attention masks
2. **Run ONNX model** - execute the model graph on CPU
3. **Mean pooling** - average the token embeddings, weighted by the attention mask
4. **Normalize** - divide by L2 norm so vectors can be compared with dot product

```python
from embedder import Embedder

embed = Embedder()
```

> Same `encode` interface as sentence-transformers, but none of the PyTorch weight.

### 15.6 Same Pipeline, No PyTorch

Confirm the numbers match:

```python
embed = Embedder()

q1 = "Can I still join the course after the start date?"
q2 = "How to install Docker on Windows?"
d  = "You don't need to register. You're accepted. You can also just start learning and submitting homework without registering."

v1 = embed.encode(q1)
v2 = embed.encode(q2)
dv = embed.encode(d)

v1.dot(dv)  # Same result as before (~0.32)
v2.dot(dv)  # Same result as before (~0.01)
```

> Same results, same pipeline, but ~33x lighter.

### 15.7 Available ONNX Models

All work with the same code - just change the model name in `download.py` and the path in `Embedder()`:

- `Xenova/all-MiniLM-L6-v2` (384d) - best small general-purpose
- `Xenova/all-MiniLM-L12-v2` (384d) - better quality, slower
- `Xenova/paraphrase-MiniLM-L6-v2` (384d) - paraphrase detection
- `Xenova/multilingual-e5-small` (384d) - multilingual retrieval
- `Xenova/bge-small-en-v1.5` (384d) - strong retrieval
- `Xenova/gte-small` (384d) - lightweight modern model

> To use a different model, add it to `download.py`, run the download,
> then update the path: `Embedder("models/Xenova/bge-small-en-v1.5")`.

---
## 16. Experiments

### Experiment 1 — Compare Similar Sentences

```python
sentences = [
    "Can I still join the course after the start date?",
    "Is it possible to enroll late?",
    "I just discovered the course, can I still join?",
    "How do I install Docker on Windows?",
    "What is the course enrollment process?",
]

for s in sentences:
    v = model.encode(s)
    print(f"'{s}' → shape {v.shape}")
```

> Observe how semantically similar sentences ("Can I still join..." and "Is it possible to enroll late...")
> produce similar vectors, while unrelated sentences produce dissimilar vectors.

### Experiment 2 — Change the Query

Use different queries against the same vector database and observe:

```python
queries = [
    "Can I still join the course after the start date?",
    "Is it possible to enroll late?",
    "I just found out about the program, can I still sign up?",
]

for q in queries:
    v = model.encode(q)
    results = vindex.search(v, num_results=3)
    print(f"Query: {q}")
    for r in results:
        print(f"  - {r['question'][:50]}... (similarity: {r.get('score', 'N/A')})")
    print()
```

> **What to observe**: Different queries may return different documents,
> different rankings, and different similarity scores.
> But semantically similar queries should return similar top documents.

### Experiment 3 — Change top-K

Compare `top_k = 1`, `top_k = 3`, `top_k = 5`, `top_k = 10`:

```python
for k in [1, 3, 5, 10]:
    results = vindex.search(query_vector, num_results=k)
    print(f"top_k={k}: {[r['question'][:40] for r in results]}")
```

> **What changes**:
> - `top_k = 1`: Only the single most similar document
> - `top_k = 3`: More context, but answer may be split
> - `top_k = 5`: Default — enough for LLM to combine
> - `top_k = 10`: More context, but also more noise

### Experiment 4 — Inspect the Retrieved Context

```python
results = vindex.search(query_vector, num_results=5)
context = "\n".join([
    f"Q: {r['question']}\nA: {r['answer'][:200]}..."
    for r in results
])
print(context)
```

> This shows exactly what information is returned before it is passed to the LLM.
> The LLM will use this as context to generate its answer.

---
## 17. Architecture Review

Let me reconstruct the entire system from memory:

### The Complete Data Flow

```text
                    ┌─────────────────┐
                    │     Documents   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Chunk / Prepare │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Embedding Model │
                    └────────┬────────┘
                             │   384-dim vectors
                             ▼
                    ┌─────────────────┐
                    │ Vector Storage  │
                    │ (numpy / SQLite / PGVector) │
                    └────────┬────────┘
                             │
                    User Query
                         │
                         ▼
                    ┌─────────────────┐
                    │ Query Encoding  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Similarity Search│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Relevant Context│
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │       LLM       │
                    └────────┬────────┘
                             │
                             ▼
                           Answer
```

### Key Architectural Principles

1. **Two-stage design**: Indexing is expensive (neural network over all documents), querying is cheap (one query vector)
2. **Same model for query and documents**: The embedding model must be the same for both stages, otherwise vectors wouldn't be comparable
3. **RAG modularity**: The RAG pipeline has three clear steps: search → prompt → LLM. We only swap the search step
4. **Three deployment options**: minsearch (learning), sqlitesearch (persistent), PGVector (production)

### Where Retrieval Ends and Generation Begins

```text
User Question
      ↓
Query Embedding      ← Vector search responsibility
      ↓
Similarity Search
      ↓
Top-K Documents      ← Vector search responsibility
      ↓
Retrieved Context
      ↓
Prompt               ← LLM responsibility (starts here)
      ↓
LLM Generation       ← LLM responsibility
      ↓
Answer
```

**Vector search is NOT the LLM.** The vector database does not generate answers — it only finds relevant information.
The LLM uses that information to generate the answer.

---
## 18. What You Should Now Understand

After completing this lesson, you should be able to explain:

1. **Why vector search exists**: Keyword search cannot match paraphrases. "Can I still join after the start date?" and "Is it possible to enroll late?" mean the same thing but share almost no words.

2. **What an embedding is**: A 384-dimensional vector that represents text in a high-dimensional space where semantically similar texts land near each other.

3. **Why we embed both query and documents**: The embedding model must produce comparable vectors. If we used different models or the query was not embedded, we couldn't compute similarity.

4. **How cosine similarity works**: Dot product of normalized vectors = cosine similarity. Ranges from 0 (unrelated) to 1 (same direction/similar).

5. **The two-stage design**: Offline indexing (expensive, runs once) vs. online querying (cheap, per question).

6. **The three vector search backends**:
   - minsearch: in-memory, exact, for experiments
   - sqlitesearch: persistent SQLite, ANN modes, for projects
   - PGVector: production PostgreSQL, HNSW indexes, concurrent access

7. **How vector search connects to RAG**: Vector search finds relevant documents → those documents become context for the LLM → LLM generates the answer.
   Retrieval ends where generation begins.

8. **Why we store metadata alongside vectors**: The `course`, `section`, `question`, `answer` fields allow filtering and provide context to the LLM.

9. **What top-K means**: Retrieving the top K most similar documents. Start with K=5, evaluate for your data.

10. **Where retrieval ends and generation begins**: After the relevant documents are retrieved and passed as context to the LLM, the LLM takes over to generate the answer.

---
## 19. Self-Test

Test your understanding by answering these questions without looking at the code:

1. Why can't we simply search the database using the raw user question?

2. What does an embedding model actually produce?

3. What is the difference between an embedding and a document?

4. Why do we store metadata alongside vectors?

5. What happens when a user submits a query?

6. Why do we embed the query?

7. What does similarity search actually calculate?

8. What does top-K mean?

9. What is the role of the vector database?

10. What is the role of the LLM?

11. Where does retrieval end?

12. Where does generation begin?

13. Why is Vector Search useful for RAG?

14. What would happen if we retrieved irrelevant documents?

15. What would happen if we retrieved too many documents?

---

**Answers (attempt the questions first, then check):**

1. Raw text cannot be compared against stored vector embeddings. The vector store only understands vector representations, not natural language.

2. A 384-dimensional vector (for `all-MiniLM-L6-v2`) that represents the text's meaning in a vector space where similar texts have similar vectors.

3. An embedding is the vector representation; a document is the original text (question + answer pair) with metadata.

4. Metadata (course, section, question, answer) allows filtering (e.g., by course) and provides context to the LLM when generating answers.

5. The query is encoded into a vector using the same embedding model, then compared against all stored document vectors using cosine similarity, and the top-K most similar documents are returned.

6. To convert the user's natural language question into a vector that can be compared against the stored document vectors. Without embedding, the search engine wouldn't know how to match the query.

7. Cosine similarity (or cosine distance `1 - cosine similarity`). It measures the angle between two vectors, ignoring their length. Higher similarity = smaller angle = more similar meaning.

8. top-K means retrieving the K most similar documents. K=5 is the default — enough context for the LLM to combine, but not so many that we include too much noise.

9. The vector database stores document vectors and performs similarity search. Its role is to find the most relevant documents for a given query vector.

10. The LLM generates the final answer based on the retrieved context. Its role is to use the provided information to produce a helpful response.

11. Retrieval ends after the top-K most relevant documents are returned, along with their metadata (questions and answers), forming the context that will be passed to the LLM.

12. Generation begins when the LLM receives the prompt (question + context) and starts producing the answer text.

13. Vector search is useful for RAG because it finds semantically similar documents even when the user's query uses different words than what's in the documents. This handles paraphrased questions and natural language better than keyword search.

14. If we retrieve irrelevant documents, the LLM will have wrong or misleading context, potentially leading to incorrect or hallucinated answers.

15. If we retrieve too many documents, the LLM has to process more context, which can increase token usage, slow down generation, and include noise that confuses the answer.

---
## 20. Final System Diagram

```text
                    ┌─────────────────┐
                    │     Documents   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Chunk / Prepare │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Embedding Model │
                    │ (all-MiniLM-L6-v2)│
                    └────────┬────────┘
                             │   384-dim vectors
                             ▼
                    ┌─────────────────────┐
                    │ Vector Storage      │
                    │ (numpy / SQLite /   │
                    │  PGVector / Docker) │
                    └────────┬────────────┘
                             │
                    User Query              │
                         │                │
                         ▼                │
                    ┌─────────────────┐    │
                    │ Query Embedding │    │
                    └────────┬────────┘    │
                             │                │
                             ▼                │
                    ┌─────────────────┐    │
                    │ Similarity Search│    │
                    └────────┬────────┘    │
                             │                │
                             ▼                │
                    ┌─────────────────┐    │
                    │ Relevant Context│    │
                    └────────┬────────┘    │
                             │                │
                             ▼                │
                    ┌─────────────────┐    │
                    │       LLM       │    │
                    └────────┬────────┘    │
                             │                │
                             ▼                │
                           Answer              │
```

**Key**: The vector database (any of the three backends) is responsible for finding relevant information.
The LLM is responsible for generating the answer using that information.
Vector search ≠ LLM. The system splits responsibilities: retrieval → LLM generation.
