# Best practices for RAG: complete practical lesson

A note before starting: this module is from 2024 and the repo marks it optional. The stack shows its age (Elasticsearch, older embedding model, older LangChain imports). The ideas have not aged. Hybrid search and reranking still do more for retrieval quality per hour of work than almost anything else, which is why this file exists.

You take the same FAQ data from earlier modules, index it three ways in Elasticsearch, and measure each step with the hit rate / MRR machinery from module 4. Then you redo the retrieval through LangChain and confirm the numbers don't move.

Read top to bottom. Each section explains the idea, shows the exact code, and gives the recorded metric so you know what to expect.

---
## 0. What we are building

```text
documents-with-ids.json (948 docs: text, section, question, course, id)
        ↓
embed 3 fields per doc (question, text, question+text) with multi-qa-MiniLM-L6-cos-v1
        ↓
Elasticsearch index "course-questions" (dense_vector x3, cosine)
        ↓
knn on questions → 0.773 / 0.667
knn on texts → 0.829 / 0.706
knn on question+text → 0.917 / 0.824
        ↓
hybrid (knn + keyword in one request) + RRF rerank → 0.925 / 0.851
        ↓
same thing through LangChain's ElasticsearchRetriever → 0.925 / 0.851
```

Two notebooks, in order:

```text
hybrid-search-and-reranking-es.ipynb (42 cells)   ES direct: index, hybrid, RRF, eval
hybrid-search-langchain.ipynb (28 cells)          same retrieval via LangChain retriever
```

Ground truth is `ground-truth-data.csv`: 4627 rows of (question, course, document). I verified the count with the csv module. Different file from module 4's 395-row set: older, bigger, all courses, and each row carries its course because every query filters on it.

---
## 1. Learning objectives

When you finish, you should be able to:

- Name the five retrieval improvements and say which two this module covers and why
- Explain what hybrid search mixes and what the alpha knob does at 0, 1, and between
- Work an RRF example by hand on two ranked lists
- Index FAQ docs in Elasticsearch with three dense_vector fields
- Run knn and keyword search in a single ES request and read the staged metrics
- Implement RRF yourself with k=60 keyed on doc id, and say how it differs from the lesson-02 sketch
- Rebuild the same hybrid retrieval with LangChain's `ElasticsearchRetriever` and confirm identical scores
- Decide when reranking is worth it (hint: noisy data, bigger gains) and when to skip it

---
## 2. The big mental model

Retrieval is where RAG answers are won or lost, and the module opens with five ways to win more often:

1. Small-to-big chunks: index small pieces, hand the LLM the surrounding context
2. Metadata: titles, topics, dates as filters and boosts
3. Hybrid search: semantic plus lexical in one query
4. Query rewriting: let an LLM restate the question before searching
5. Reranking: re-order results with a better score after first retrieval

Only 3 and 5 get the full treatment here. The lesson's reasoning is effort-per-gain, and the numbers back it: hybrid plus RRF lifts this dataset from 0.773 to 0.925 hit rate with no model changes and no new data.

Why the two mix well is straightforward. Vector search knows what words mean and misses exact terms ("pandas", an error code, a name). Keyword search nails exact terms and misses paraphrases. Each covers the other's blind spot, so running both and merging beats either alone. Reranking then fixes the ordering problem: cosine similarity puts the right doc sixth when you only keep five, and a second scoring pass pulls it up.

```text
Index once (3 vector fields + text fields)
        ↓
Retrieve twice per question (knn list + keyword list)
        ↓
Merge once (RRF over ranks, not scores)
        ↓
Evaluate same as module 4 (hit rate, MRR, frozen ground truth)
```

Scores from different methods don't mix (a cosine similarity and a BM25 score share no scale), which is why fusion uses ranks. Keep that sentence; it explains every formula below.

---
## 3. Environment setup

You need Elasticsearch running locally on port 9200 (the notebooks point `Elasticsearch('http://localhost:9200')` at it), plus:

```bash
uv add sentence-transformers elasticsearch pandas tqdm
```

LangChain arrives in section 8:

```bash
uv add langchain langchain-elasticsearch langchain-huggingface
```

Data files sit next to the notebooks: `documents-with-ids.json` (948 docs) and `ground-truth-data.csv` (4627 rows). No downloads, no API keys, no secrets. The embedding model `multi-qa-MiniLM-L6-cos-v1` downloads from HuggingFace on first use (384 dims, same width as the MiniLM you already know, trained on question-answer pairs which suits FAQ retrieval).

One honest caveat: RRF as a native ES feature needs a recent server version, and the built-in `rank: {"rrf": {}}` needs a paid subscription. Free tier errors out. That is exactly why the module implements RRF by hand too. Know which path your cluster supports before planning around the native one.

---
## 4. Indexing: three vectors per doc

Each doc carries `text` (the answer body), `question`, `section`, `course`, `id`. We embed three views of it because the staged metrics will show each view matters:

```python
model_name = 'multi-qa-MiniLM-L6-cos-v1'
model = SentenceTransformer(model_name)
```

```python
for doc in tqdm(documents):
    question = doc['question']
    text = doc['text']
    qt = question + ' ' + text

    doc['question_vector'] = model.encode(question)
    doc['text_vector'] = model.encode(text)
    doc['question_text_vector'] = model.encode(qt)
```

Question alone, answer alone, both concatenated. The concatenated field wins later, which tells you queries match best against the full Q&A sense of a doc, not either half.

The index mapping stores text fields for keyword search and three cosine `dense_vector` fields for knn, with `course` and `id` as keywords (filtering and labels, same roles as ever):

```python
es_client = Elasticsearch('http://localhost:9200')

index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"},
            "id": {"type": "keyword"},
            "question_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
            "question_text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            },
        }
    }
}

index_name = "course-questions"

es_client.indices.delete(index=index_name, ignore_unavailable=True)
es_client.indices.create(index=index_name, body=index_settings)
```

Then index every doc:

```python
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)
```

Single shard, no replicas: a local dev index, not a production topology. Delete-then-create makes re-runs idempotent, which you will appreciate while iterating.

---
## 5. Hybrid search, the idea

Run both searches, merge the lists. Two merge styles exist and they are not the same thing.

Weighted sum mixes scores:

```text
score = alpha * vector_score + (1 - alpha) * keyword_score
```

Alpha 1 is pure vector, 0 is pure keyword, between is a blend. Simple, but it mixes numbers that were never on one scale, so weights need tuning per dataset and the tuning rots as data drifts.

RRF mixes ranks instead, which sidesteps the scale problem entirely. For each doc, add `1 / (k + rank)` over every list it appears in (ranks here start at 0 in the lesson's worked example, with the formula written `1 / (k + rank + 1)` and k=1). Work it once by hand because the mechanics stick better that way:

```text
text search:   [A, B, C, D, E]
vector search: [C, B, F, G, A]

A = 1/(1+0+1) + 1/(1+4+1) = 0.500 + 0.167 = 0.667
B = 1/(1+1+1) + 1/(1+1+1) = 0.333 + 0.333 = 0.667
C = 1/(1+2+1) + 1/(1+0+1) = 0.250 + 0.500 = 0.750
D = 1/(1+3+1)             = 0.200
F =             1/(1+2+1) = 0.250

Final: C, then A/B tied, then F, then D/G tied, then E
```

C wins by placing high in both lists. Single-list docs score lower no matter how high they placed, which is the behavior you want: agreement across methods outranks dominance in one. Higher k flattens rank differences; the constant just tunes how much position matters. And it generalizes to any number of lists, not only two.

The generic version, keyed on question text, capped at N results:

```python
def rrf(search_results, k=1, num_results=10):
    scores = {}
    doc_map = {}

    for results in search_results:
        for rank, doc in enumerate(results):
            key = doc["question"]
            if key not in scores:
                scores[key] = 0
                doc_map[key] = doc
            scores[key] += 1 / (k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:num_results]]
```

Then hybrid is three lines: run both, fuse:

```python
def hybrid_search(query, course="data-engineering-zoomcamp", num_results=10):
    keyword_results = keyword_search(query, course=course, num_results=num_results)
    vector_results = vector_search(query, course=course, num_results=num_results)
    return rrf([keyword_results, vector_results], num_results=num_results)
```

Note the keying choice: question text as the dedup key works when questions are unique, doc id is safer in general. The ES implementation in section 7 uses ids. Remember the difference; it matters the moment two docs share a question string.

---
## 6. Hybrid search in Elasticsearch

One request carries both halves. The knn half searches a vector field with the encoded query; the keyword half runs `multi_match` over question (boosted 3x), text, and section. Both filter on course. The 0.5 boosts are per-clause weights inside ES scoring:

```python
knn_query = {
    "field": "text_vector",
    "query_vector": v_q,
    "k": 5,
    "num_candidates": 10000,
    "boost": 0.5,
    "filter": {
        "term": {
            "course": course
        }
    }
}
```

```python
keyword_query = {
    "bool": {
        "must": {
            "multi_match": {
                "query": query,
                "fields": ["question^3", "text", "section"],
                "type": "best_fields",
                "boost": 0.5,
            }
        },
        "filter": {
            "term": {
                "course": course
            }
        }
    }
}
```

`k: 5` with `num_candidates: 10000` means approximate search over a wide candidate pool, then the top 5. Fine at this scale; the candidates number is what you raise when recall slips on bigger indexes.

The pipeline wrapper used for evaluation takes the vector field as a parameter (so the staged comparison just swaps field names), returns `_source` docs:

```python
def elastic_search_hybrid(field, query, vector, course):
    # ... knn_query and keyword_query as above ...

    search_query = {
        "knn": knn_query,
        "query": keyword_query,
        "size": 5,
        "_source": ["text", "section", "question", "course", "id"]
    }

    es_results = es_client.search(
        index=index_name,
        body=search_query
    )

    result_docs = []

    for hit in es_results['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs
```

Evaluation reuses module 4's machinery verbatim (same `hit_rate`, `mrr`, `evaluate` shape, ground truth rows carry `course` so each query filters correctly). Recorded progression, and this is the number ladder to memorize:

```text
knn on question vectors:       hit 0.773, MRR 0.667
knn on text vectors:           hit 0.829, MRR 0.706
knn on question+text vectors:  hit 0.917, MRR 0.824
hybrid + RRF rerank:           hit 0.925, MRR 0.851
```

Two readings. Combined-field vectors beat either half alone, same lesson as module 2's question-plus-answer concatenation. And hybrid plus rerank adds the last couple of points on top. Small in absolute terms, real in production terms, where a few percent over millions of queries is a different support queue.

One labeling caution: lesson 03 calls 0.917/0.824 "hybrid without reranking," but the notebook's staged cells show those numbers coming from knn on the combined field. The 0.925/0.851 figure is the hybrid-with-RRF result in both sources. I trust the notebook's staging since you can see each step run.

---
## 7. Reranking with RRF, for real

Cosine similarity orders by angle, not by usefulness. The right doc lands sixth, you keep five, the LLM never sees it. Reranking is a second scoring pass over retrieved docs, and RRF is the cheapest version that works: no model, no training, just ranks.

Native ES (8.9+) does it server-side:

```python
    response = es_client.search(
        index=index_name,
        query=keyword_query,
        knn=knn_query,
        size=5,
        rank={"rrf": {}}
    )

    return [hit["_source"] for hit in response["hits"]["hits"]]
```

Paid subscription required on most tiers; free tier errors. So the module also does it client-side: run knn and keyword as separate searches, score both hit lists, merge on doc id:

```python
def compute_rrf(rank, k=60):
    return 1 / (k + rank)
```

```python
    rrf_scores = {}

    for rank, hit in enumerate(knn_response["hits"]["hits"]):
        doc_id = hit["_source"]["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + compute_rrf(rank, k)

    for rank, hit in enumerate(keyword_response["hits"]["hits"]):
        doc_id = hit["_source"]["id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + compute_rrf(rank, k)

    all_docs = {
        hit["_source"]["id"]: hit["_source"]
        for hit in knn_response["hits"]["hits"] + keyword_response["hits"]["hits"]
    }

    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [all_docs[doc_id] for doc_id, _ in sorted_docs[:5]]
```

Docs in both lists get their scores summed, which pushes agreement to the top, same as the hand example. Two differences from the lesson-02 sketch, both deliberate: k=60 (the literature default; rank position matters less, so messy lists don't swing the outcome) and doc id as the merge key instead of question text (unique by construction).

Evaluated through the same `evaluate()` with a thin wrapper:

```python
def question_text_hybrid_rrf(q):
    question = q["question"]
    course = q["course"]
    v_q = model.encode(question)
    return elastic_search_hybrid_rrf("question_text_vector", question, v_q, course)

evaluate(ground_truth, question_text_hybrid_rrf)
```

Result: 0.925 / 0.851. Modest here, bigger where data is noisier and queries messier. That conditional is the honest version of "should I rerank": measure on your data, keep it if the delta pays for the complexity.

References the lesson leaves: the ES RRF guide, the original Cormack paper, and the subscriptions page (to check whether your tier has native RRF before writing the manual version).

---
## 8. The same thing through LangChain

Nothing about retrieval changes in this section. Same index, same fields, same query shape. LangChain just wraps the ES client so retrieval plugs into larger chains later.

```bash
uv add langchain langchain-elasticsearch langchain-huggingface
```

Embeddings come from the HuggingFace wrapper around the same model family:

```python
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Dict
from langchain_elasticsearch import ElasticsearchRetriever

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
)

es_url = "http://localhost:9200"
```

A body function builds the hybrid request per query (note the hardcoded course here; the evaluated version parameterizes it):

```python
def hybrid_query(search_query: str) -> Dict:
    vector = embedding.embed_query(search_query)
    return {
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": search_query,
                        "fields": ["question^3", "text", "section"],
                        "type": "best_fields",
                    }
                },
                "filter": {
                    "term": {
                        "course": "data-engineering-zoomcamp"
                    }
                }
            }
        },
        "knn": {
            "field": "question_text_vector",
            "query_vector": vector,
            "k": 5,
            "num_candidates": 10000,
        },
        "size": 5
    }
```

```python
hybrid_retriever = ElasticsearchRetriever.from_es_params(
    url=es_url,
    index_name="course-questions",
    body_func=hybrid_query,
    content_field="text"
)
```

```python
query = "I just discovered the course. Can I still join it?"
results = hybrid_retriever.invoke(query)

for result in results:
    print(result.metadata["_source"]["question"])
    print(result.metadata["_score"])
```

Results arrive as Documents; the payload lives under `metadata["_source"]`, the score under `metadata["_score"]`. For evaluation the retriever gets wrapped to return plain dicts with id, question, text:

```python
    results = retriever.invoke(query)
    return [
        {
            "id": r.metadata["_source"]["id"],
            "question": r.metadata["_source"]["question"],
            "text": r.metadata["_source"]["text"],
        }
        for r in results
    ]

def question_text_hybrid(q):
    return elastic_search_hybrid("question_text_vector", q["question"], q["course"])

evaluate(ground_truth, question_text_hybrid)
```

Recorded: hit rate 0.925, MRR 0.851. Identical to the hand-rolled path, which is the point. The wrapper changes ergonomics, not outcomes. Worth it when retrieval feeds a bigger chain; overhead when it doesn't.

---
## 9. The three techniques left on the table

The intro promised five; two got built. Pointers for the rest:

Small-to-big retrieval: embed small chunks (precise matching), hand the LLM the chunk plus its neighbors (full context). LangChain's ParentDocumentRetriever is the ready-made version.

Metadata: titles, topics, dates as keyword fields, used to filter before ranking rather than hoping the ranker figures it out. The `course` term filter in every query above is already the baby version of this.

Query rewriting: an LLM restates the user's question into a cleaner search query first. Costs a call per question, pays when queries are vague or jargon-heavy. Measure it like everything else.

Deeper reading: the RAG survey (arXiv 2312.10997) for the full technique zoo, LangChain's retriever docs for the wrapping layer, plus the RRF paper and ES guide from section 7.

---
## 10. Things to try

1. Work the lesson-02 RRF example by hand before running anything. If C doesn't come out on top, reread the formula, don't rerun the code.
2. Change k in the manual RRF (1, 20, 60, 200) and re-evaluate. Watch how little the top moves; that insensitivity is why 60 is a safe default.
3. Merge on question text vs doc id with a duplicate question planted in the index. See which key survives.
4. Drop the course filter from one query function and compare. That delta is what metadata filtering buys you.
5. Evaluate knn on each vector field separately (the notebook stages this). Confirm the combined field wins on your run too, not just in the recorded numbers.
6. Try alpha-style weighted fusion against RRF on the same ground truth. RRF should win or tie with zero tuning; note where it doesn't.
7. Run the LangChain notebook end to end and diff its scores against the ES notebook. They should match to the decimal; if not, the body function drifted from the hand query.
8. Pick one of the three unbuilt techniques and prototype it against the same 4627 rows. Same `evaluate()`, same frozen set, honest delta.

---
## 11. Architecture review

```text
Index once
  948 docs × 3 embeddings (question, text, both) → "course-questions"
        ↓
Retrieve twice per question
  knn (vector field, k=5, 10k candidates) + keyword (multi_match, course filter)
        ↓
Merge once
  RRF over ranks (k=60, doc-id keys) → top 5
        ↓
Measure always
  4627 frozen questions → hit rate + MRR → 0.925 / 0.851
        ↓
Wrap optionally
  LangChain retriever around the same request → same numbers, chainable
```

Three sentences for anyone asking what the module was about: vector search understands meaning and misses exact terms, keyword search is the reverse, so you run both and fuse ranks instead of scores. RRF is the fusion: a doc's score is the sum of reciprocal ranks across lists, which rewards agreement without any tuning. LangChain changes how you call it, not what it computes.

---
## 12. What you should now understand

- Retrieval quality caps answer quality, which is why this module exists at all.
- Hybrid means semantic plus lexical per question; alpha blends scores, RRF blends ranks.
- Ranks fuse cleanly, scores don't. Methods with different scales can't share a weighted sum without tuning that rots.
- k in RRF sets how much position matters; 60 is the default because results barely move around it.
- Merge keys matter: doc ids are unique by construction, question text is unique by luck.
- Combined question-plus-text vectors beat either half, same pattern as module 2's concatenation.
- Native ES RRF needs a recent version and usually a paid tier; the manual version needs neither.
- The course filter is metadata retrieval in miniature; generalize it before reaching for fancier tricks.
- LangChain's retriever is ergonomics. Same request, same scores, easier chains.
- Every claim here has a number behind it because every variant ran against the same 4627 frozen questions. That frozen set is what makes the deltas mean something.

---
## 13. Self-test

Try these from memory. Answers follow.

1. Name the five techniques and say which two this module builds.
2. What breaks if you only run vector search? Only keyword search?
3. What does alpha do at 0, at 1, and at 0.5?
4. Why fuse ranks instead of scores?
5. Work the A/B/C example: which doc wins and why?
6. What does k control in RRF, and why is 60 the default?
7. Which three vector fields get indexed per doc, and which one wins?
8. What do `k: 5` and `num_candidates: 10000` mean in the knn clause?
9. Why does every query filter on course?
10. Native RRF vs manual RRF: when does each apply?
11. The manual RRF keys on doc id; the lesson-02 sketch keys on question text. Which is safer and why?
12. Recite the four recorded metric stages in order.
13. Lesson 03 calls 0.917/0.824 "hybrid without reranking" while the notebook stages it as knn on the combined field. Does the disagreement matter?
14. What does LangChain change about the retrieval, precisely?
15. When is reranking worth the complexity?

---

Answers:

1. Small-to-big chunks, metadata, hybrid search, query rewriting, reranking. Built: hybrid search and reranking, picked for gain per effort.
2. Vector-only misses exact terms (names, codes, "pandas"). Keyword-only misses paraphrases. Each fails exactly where the other works.
3. Alpha 1 is pure vector, 0 is pure keyword, 0.5 an even blend. In between trades semantic against lexical weight.
4. Scores from different methods share no scale (cosine vs BM25), so weighted sums need per-dataset tuning that drifts. Ranks are unitless and comparable directly.
5. C wins: high in both lists, so its reciprocal-rank sum beats docs that dominate only one list or place low in both.
6. How much rank position matters; higher k flattens differences. 60 because results barely move around it, making it a safe untuned default.
7. Question, text, and question-plus-text vectors. The combined field wins (0.917/0.824 vs 0.773/0.667 and 0.829/0.706).
8. Approximate search considers 10,000 candidates, returns the top 5. Candidates bound recall effort; k bounds output size.
9. Ground truth rows carry a course and questions are course-specific; unfiltered search returns right-answer-wrong-course docs. Metadata filtering before ranking.
10. Native (`rank: {"rrf": {}}`) when the server is recent enough and the tier includes it; manual (two searches plus client-side scoring) everywhere else, including free tier.
11. Doc id. Ids are unique by construction; question text collides the moment two docs share a question string, merging their scores wrongly.
12. knn questions 0.773/0.667, knn texts 0.829/0.706, knn combined 0.917/0.824, hybrid plus RRF 0.925/0.851 (hit rate / MRR).
13. No, for decisions: both sources agree the final hybrid-plus-RRF number is 0.925/0.851, and the notebook's staged cells show where each figure comes from. Trust staged runs over prose labels.
14. Nothing about retrieval itself: same index, same fields, same query shape, identical scores. It changes the interface (Documents, chainable retriever) for use in larger pipelines.
15. When measurement says so on your data. Modest here (+0.008/+0.027), bigger on noisier data with messier queries. Prototype against frozen ground truth and keep it if the delta pays.

---
## 14. Where to go from here

The module has no homework, so the section-10 tries are the assignment. If one direction calls: prototype a third technique (rewriting is the most fun, metadata the most practical) against the same frozen rows. For the survey-level view, the RAG techniques paper covers the zoo this module only sampled. And a reminder on age: imports and server versions have moved since 2024, so treat exact API spellings as check-against-current-docs, while the ideas (fuse ranks, rerank late, measure everything) carry over untouched.
