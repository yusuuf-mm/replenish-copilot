# Evaluation: complete practical lesson

Modules 1 to 3 built things: search, RAG, agents. Nobody asked which version was actually better. This module fixes that. You generate test questions with an LLM, run your systems against them, and get numbers: hit rate and MRR for search, an LLM judge for answers, and answer-plus-trajectory scores for agents.

Read top to bottom. Each section explains the idea, shows the exact code, and tells you what number to expect.

---
## 0. What we are building

A test harness in three layers, each one resting on the one below:

```text
Ground truth (LLM writes 5 questions per FAQ doc, doc id = label)
        ↓
Search eval (does the right doc show up? hit rate, MRR)
        ↓
RAG eval (is the generated answer right? LLM judge, good/bad)
        ↓
Agent eval (is the answer right AND were the tool calls sane?)
```

Four notebooks carry this out, in order:

```text
01-data-gen.ipynb      make the questions → data/ground_truth-new.csv (395 rows)
02-search-eval.ipynb   relevance lists → hit_rate / mrr / evaluate() → tune boosts
03-rag-evals.ipynb     run RAG on all 395 questions → data/rag-answers-new.csv
04-llm-judge.ipynb     judge all 395 answers → data/rag-evaluations-new.csv
```

Plus the agent part (lesson 14, no dedicated notebook): run the ToyAIKit agent on 50 questions, save answers plus tool calls, judge both.

The data folder already holds every artifact, so you can skip any expensive step and download instead. I verified the row counts: 395 ground truth rows, 395 RAG answers, 395 RAG judgments, 50 agent answers, 50 agent judgments. (If you count lines with `wc -l` you will see bigger numbers. Answers contain newlines inside quoted CSV fields, so count records with pandas or the csv module, not lines.)

---
## 1. Learning objectives

When you finish, you should be able to:

- Explain the A → Q* → A′ pattern and why the doc id is the linchpin of the whole thing
- Generate ground truth questions with structured output (`responses.parse` + a Pydantic model)
- Run batch generation in parallel with retries and track the dollar cost
- Turn search results into relevance lists and compute hit rate and MRR by hand
- Use `evaluate()` to compare search functions fairly on a fixed dataset
- Tune field boosts with a grid search and read the result without fooling yourself
- Run RAG over all ground truth questions with `RAGWithUsage` and save A′ records
- Judge answers with an LLM (instructions, template, good/bad + reasoning)
- Evaluate an agent's answer and its tool-call trajectory separately
- Say when synthetic data stops being trustworthy and what replaces it

---
## 2. The big mental model

The whole module is one trick: create questions from answers, so you already know the right answer before you test.

```text
A  = original FAQ answer (you have this)
Q* = question the LLM invents from A (you generate this)
A′ = what your system answers when asked Q* (you measure this)
```

Search eval checks: is the source doc of A in the results for Q*? RAG eval checks: does A′ say the same thing as A? Agent eval adds: did the tool calls that led to A′ make sense?

Three levels, and the order matters. Search gets most of the time because everything downstream depends on it. Wrong documents in, no prompt or model gets you out. So: test retrieval alone first, then the full pipeline on top.

Two kinds of evaluation, two different moments:

```text
Offline: test dataset, metrics, before you ship (this module)
Online:  real traffic, feedback, dashboards, after you ship (module 5)
```

Offline lets you compare settings, prompts, and models on identical questions. The dataset never changes between runs, so any metric movement is the change you made, not noise.

One warning the lesson repeats and I will too: synthetic questions resemble the FAQ text they came from, which pushes scores up. Numbers above 95% deserve suspicion, not celebration. Generated data gets you started; real user queries replace it as soon as you have them.

---
## 3. Environment setup

The `code/` folder is a uv project (`llm-zoomcamp-2026-evals`, Python 3.12+). Dependencies:

```text
jupyter, minsearch, openai, pandas, python-dotenv, requests, tqdm
```

Note what is missing: no sentence-transformers, no vector DB. Search eval here runs on minsearch keyword search. The evaluation machinery does not care which search sits behind it, which is exactly the point of section 6.

You need an OpenAI key in `.env`:

```text
OPENAI_API_KEY=YOUR_API_KEY
```

And three helpers in the notebook directory. If they are not there, fetch them:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main

wget ${PREFIX}/01-agentic-rag/code/ingest.py
wget ${PREFIX}/01-agentic-rag/code/rag_helper.py
wget ${PREFIX}/04-evaluation/code/evaluation_utils.py
```

Two things differ from the module 1 copies, worth knowing. This `ingest.py` keeps the `id` field as-is (module 1 renamed it to `doc_id`), and `build_index` uses text fields question/section/answer plus `course` as the keyword field. The ids matter here in a way they did not before: the id is the label your metrics check against.

Notebook map, so you always know where you are:

| Notebook | Lesson | Job |
|---|---|---|
| `01-data-gen.ipynb` (36 cells) | 02, 03 | One doc, then all docs → ground truth CSV |
| `02-search-eval.ipynb` (44 cells) | 04, 05, 06 | Relevance lists → metrics → boost tuning |
| `03-rag-evals.ipynb` (27 cells) | 12 | RAG on all questions → answers CSV |
| `04-llm-judge.ipynb` (24 cells) | 13 | Judge all answers → evaluations CSV |

---
## 4. Ground truth, one document

Ground truth means questions with known-correct documents. Three ways to get it: pay annotators (best, expensive), label real user queries (needs a live system), or have an LLM invent questions from your docs (cheap, what we do). For each FAQ doc we ask for 5 questions it would answer, and the doc id becomes the label.

### 4.1 Load and narrow the docs

```python
from ingest import load_faq_data
documents = load_faq_data()
```

Generating for every course would take longer and cost more, so we keep LLM Zoomcamp only (79 docs) and rename the list to `documents` for the rest of the module:

```python
documents_llm = []

for doc in documents:
    if doc["course"] == "llm-zoomcamp":
        documents_llm.append(doc)

documents = documents_llm
```

Pick one doc and look at it:

```python
doc = documents[0]
print(doc["id"])
print(doc["question"])
print(doc["answer"])
```

That id is the whole game. Every question we generate from this doc carries this id, and later we check whether search returns it.

### 4.2 Structured output

Free text is a pain for pipelines: you would have to parse questions out of a paragraph and hope the format holds. Structured output makes the model return a Python object with a fixed shape instead.

```python
from pydantic import BaseModel

class Questions(BaseModel):
    questions: list[str]
```

The instructions ask the model to roleplay a student, write 5 complete questions, and reuse as few words from the record as possible. That last bit matters: wording that copies the FAQ makes retrieval trivially easy and inflates your metrics later.

```python
data_gen_instructions = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()
```

Send the doc as JSON, then call `responses.parse` instead of `responses.create`. The `text_format=Questions` argument is what forces the shape; the parsed object lands in `response.output_parsed`:

```python
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
openai_client = OpenAI()

user_prompt = json.dumps(doc)

messages = [
    {"role": "developer", "content": data_gen_instructions},
    {"role": "user", "content": user_prompt}
]

response = openai_client.responses.parse(
    model="gpt-5.4-mini",
    input=messages,
    text_format=Questions
)

result = response.output_parsed
print(result.questions)
```

Five questions about the first FAQ doc. From here on, `evaluation_utils.py` carries the repeated patterns so the notebooks stay readable:

- `llm_structured`: the parse call above in one function, returns `(parsed, usage)`
- `llm_structured_retry`: same call with retries and exponential backoff (`2 ** attempt` seconds), gives up after 3 tries
- `calc_price` / `calc_total_price`: dollars from token usage at 0.75/M input, 4.50/M output
- `map_progress`: parallel fan-out with a progress bar (next section)
- `RAGWithUsage`: `RAGBase` that records usage and bakes in the tuned boosts (section 9)

Same call through the helper:

```python
from evaluation_utils import llm_structured

result, usage = llm_structured(
    openai_client,
    data_gen_instructions,
    user_prompt,
    Questions
)
```

### 4.3 Cost of one call, and the record shape

Usage rides along on every response. Price it:

```python
from evaluation_utils import calc_price

cost = calc_price(usage)
```

Then convert questions into the records the rest of the module consumes:

```python
records = []

for q in result.questions:
    records.append({
        "question": q,
        "document": doc["id"]
    })
```

Two fields, nothing else: the question, and the id of the doc that answers it. That pairing is the entire contract downstream code relies on.

---
## 5. Ground truth, all documents

One doc works. Now do it 79 times without babysitting failures or waiting all day.

### 5.1 The per-doc function with retries

Batch jobs die on transient errors: one rate limit, one network blip, and a naive loop loses everything. So the per-doc function uses the retrying variant:

```python
from evaluation_utils import llm_structured_retry

def generate_ground_truth(doc):
    user_prompt = json.dumps(doc)

    out, usage = llm_structured_retry(
        openai_client,
        data_gen_instructions,
        user_prompt,
        Questions
    )

    results = []

    for q in out.questions:
        results.append({
            "question": q,
            "document": doc["id"]
        })

    return results, usage
```

Try it on 5 docs sequentially first, to confirm the shape before spending money at scale:

```python
from tqdm.auto import tqdm

ground_truth = []
usages = []

for doc in tqdm(documents[:5]):
    records, usage = generate_ground_truth(doc)
    ground_truth.extend(records)
    usages.append(usage)
```

Works, but each call waits on the network in turn. For 79 docs that is a lot of idle time.

### 5.2 Parallel fan-out

LLM calls are network-bound, so threads help. `map_progress` submits one job per doc, ticks the bar as each finishes, and gathers results in order. Six workers is the default here: enough to stop waiting, few enough to stay under rate limits.

```python
from concurrent.futures import ThreadPoolExecutor
from evaluation_utils import map_progress

with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, documents, generate_ground_truth)
```

Split the pairs back apart:

```python
ground_truth = []
usages = []

for records, usage in results:
    ground_truth.extend(records)
    usages.append(usage)

len(ground_truth)
```

Five questions per doc, so roughly 5x the doc count.

### 5.3 Total cost, then save

```python
from evaluation_utils import calc_total_price

calc_total_price(usages)
```

```python
import pandas as pd

df_ground_truth = pd.DataFrame(ground_truth)
df_ground_truth.to_csv("data/ground_truth-new.csv", index=False)
```

The recorded run (May 29, 2026): 79 docs, 395 questions, $0.057 total. About six cents. If you would rather skip the spend:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main

wget -O data/ground_truth-new.csv ${PREFIX}/04-evaluation/data/ground_truth-new.csv
```

FAQ data drifts over time, so a fresh run may produce different docs, questions, costs, and downstream numbers. That is expected, not a bug.

---
## 6. Search evaluation

Now we ask, for each of the 395 questions: did search bring back the right doc?

New notebook (`02-search-eval.ipynb`). No LLM calls in this part at all, just the index and the CSV:

```python
import pandas as pd

df_ground_truth = pd.read_csv("data/ground_truth-new.csv")
ground_truth = df_ground_truth.to_dict(orient="records")
```

```python
from ingest import load_faq_data, build_index

documents = load_faq_data()

documents_llm = []

for doc in documents:
    if doc["course"] == "llm-zoomcamp":
        documents_llm.append(doc)

documents = documents_llm
index = build_index(documents)
```

### 6.1 The swappable search function

Wrap search in a plain function on purpose. Downstream code only needs "query in, results out", so `text_search` today can become `vector_search` or hybrid tomorrow with zero changes elsewhere:

```python
def text_search(query):
    boost_dict = {"question": 3.0, "section": 0.5}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict
    )
```

### 6.2 Relevance lists

Take one record, search, and compare ids:

```python
q = ground_truth[0]

doc_id = q["document"]
results = text_search(query=q["question"])

for d in results:
    print(f'{d["id"]} == {doc_id}: {d["id"] == doc_id}')
```

Then compress that comparison into a list of 0s and 1s, one per retrieved doc:

```python
relevance = []

for d in results:
    relevance.append(int(d["id"] == doc_id))

relevance
# [1, 0, 0, 0, 0]
```

Correct doc first. A 1 anywhere else would mean it showed up lower; all zeros means it missed entirely.

As a function, first tied to text search, then generalized to take any search function:

```python
def compute_relevance(q, search_function):
    doc_id = q["document"]
    results = search_function(query=q["question"])

    relevance = []

    for d in results:
        relevance.append(int(d["id"] == doc_id))

    return relevance
```

```python
def compute_relevance_total(ground_truth, search_function):
    relevance_total = []

    for q in tqdm(ground_truth):
        relevance = compute_relevance(q, search_function)
        relevance_total.append(relevance)

    return relevance_total
```

Smoke-test on 15 questions before the full run (spot the one all-zero row, a real miss, in the recorded output):

```python
ground_truth_sample = ground_truth[:15]
relevance_total = compute_relevance_total(ground_truth_sample, text_search)
```

Then the full set:

```python
relevance_total = compute_relevance_total(ground_truth, text_search)
```

Relevance lists are the raw material. Metrics turn them into numbers.

---
## 7. Search metrics

### 7.1 Hit rate

Hit rate (same as Recall@k here, since each query has exactly one right doc): the share of queries where a 1 appears anywhere in the list.

```python
def hit_rate(relevance):
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance)
```

On the 15-question sample: 14 hits out of 15, so 0.933. It answers "did we find it" and nothing else.

### 7.2 MRR

Mean reciprocal rank cares where the hit landed. First correct doc at rank 1 scores 1.0, rank 2 scores 1/2, rank 3 scores 1/3, a miss scores 0. Average over queries.

```python
def mrr(relevance):
    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance)
```

The `rank + 1` is there because Python counts from zero and 1/0 is not a score anyone wants. On the sample: 0.822. MRR always sits at or below hit rate, since anything below rank 1 drags the average down.

### 7.3 One function for any search

```python
def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)

    return {
        "hit_rate": hit_rate(relevance_total),
        "mrr": mrr(relevance_total),
    }
```

Recorded result for text search on the full set: `{"hit_rate": 0.899, "mrr": 0.769}`. Roughly 90% found, usually near the top.

### 7.4 Reading the numbers without lying to yourself

One relevant doc per query is an assumption, not a fact. Other retrieved docs can be perfectly good answers; a 50% hit rate means the source doc missed half the time, not that half your results are garbage.

Synthetic closeness inflates both metrics. Above 95%, check whether your questions just echo the FAQ before celebrating.

Thresholds depend on the job. Some apps live happily at 50%; others need 90%+. Two things set the bar: how well your LLM copes with imperfect context, and how much wrongness your users tolerate.

And read the pair together. High MRR means the right doc sits near the top where the prompt actually uses it. High hit rate with low MRR means it is in there somewhere, buried under noise the model has to wade through.

---
## 8. Tuning search with evidence

Until now the boosts were vibes: question at 3.0 because matching the FAQ question "should" count more. Reasonable guess. Now we check it.

The loop never changes: fix the dataset, change one thing, re-run `evaluate()`, read the delta. Same questions every time, so the comparison is fair.

### 8.1 Sweep one boost

```python
def search_boost(query, question_boost):
    boost_dict = {"question": question_boost, "section": 0.5}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
    )
```

```python
for boost in [0.5, 1.0, 3.0, 5.0, 10.0]:
    result = evaluate(
        ground_truth,
        lambda query, boost=boost: search_boost(query, boost)
    )
    print(f"boost={boost}: {result}")
```

Recorded:

```text
boost=0.5:  hit_rate 0.911, mrr 0.801
boost=1.0:  hit_rate 0.924, mrr 0.814
boost=3.0:  hit_rate 0.899, mrr 0.769
boost=5.0:  hit_rate 0.871, mrr 0.740
boost=10.0: hit_rate 0.858, mrr 0.712
```

More question weight makes things worse. The best value is 1.0, no boost at all. The intuition was backwards, and we would never have known without measuring. (The `lambda query, boost=boost` wrapper exists because `evaluate` passes only the query; the default-arg trick freezes each boost value into its own function.)

### 8.2 Grid search over all three fields

```python
def search_boosts(query, question_boost, answer_boost, section_boost):
    boost_dict = {
        "question": question_boost,
        "section": section_boost,
        "answer": answer_boost,
    }

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
    )
```

3 × 4 × 3 combinations, each evaluated, results collected with their scores, sorted by MRR. Top rows, recorded:

```text
question  answer  section  hit_rate  mrr
1.0       2.0     0.1      0.975     0.885
2.0       4.0     0.2      0.975     0.885
5.0       10.0    0.5      0.975     0.885
```

The top rows share one ratio: question : answer : section = 1 : 2 : 0.1. Only relative weights matter, so take the small readable ones. And the answer text carries twice the question weight, the reverse of where we started.

Lock it in as the new `text_search`:

```python
def text_search(query):
    boost_dict = {
        "question": 1.0,
        "answer": 2.0,
        "section": 0.1,
    }

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
    )
```

Hit rate 0.975, MRR 0.885, up from 0.899/0.769. That gap is the value of the module so far.

### 8.3 Tuning habits

Grid search is fine for a handful of knobs at ~1 second per combo. When each evaluation costs minutes, switch to random sampling, Bayesian optimization (hyperopt gets a namecheck in the lesson), or a held-out validation split so you stop overfitting the test set.

On top-K: 10 results would lift hit rate mechanically, more tickets in the lottery. But every extra doc is more prompt tokens, more money, and more noise for the model to sift. Five is the sane default for short FAQ docs.

---
## 9. From retrieval to answers

Search eval asks "did the right doc show up". RAG eval asks "did the final answer come out right". A bad final answer can come from three places: wrong docs retrieved, good docs dropped from the prompt, or the model ignoring context it was given. The judge in section 11 scores the outcome; reading the failures tells you which stage to blame.

New notebook (`03-rag-evals.ipynb`). Load questions, docs, index exactly like before, plus one new structure: a lookup from doc id to the full doc, so each question can find its original answer:

```python
doc_idx = {}

for doc in documents:
    doc_idx[doc["id"]] = doc
```

For the RAG runs we use `RAGWithUsage` from `evaluation_utils.py`. It subclasses module 1's `RAGBase`, so `rag()` works identically, with two additions: it records token usage on every LLM call (for `total_cost()`), and its `search` already carries the tuned boosts from section 8:

```python
from evaluation_utils import RAGWithUsage

assistant = RAGWithUsage(
    index=index,
    llm_client=openai_client,
)
```

Note the layering: `RAGBase.search` still has the old 3.0/0.5 boosts, but `RAGWithUsage` overrides `search` with 1.0/2.0/0.1. The tuned numbers flow into every RAG answer from here on without touching anything else.

---
## 10. Generating RAG answers

One question first, to see the record shape:

```python
rec = ground_truth[0]
question = rec["question"]

answer_llm = assistant.rag(question)
assistant.total_cost()

doc_id = rec["document"]
answer_orig = doc_idx[doc_id]["answer"]

rag_result = {
    "question": question,
    "answer_llm": answer_llm,
    "answer_orig": answer_orig,
    "document": doc_id,
}
```

Four fields: what was asked, what the system said, what the FAQ said, which doc it came from. As a function:

```python
def generate_rag_answer(rec):
    question = rec["question"]
    doc_id = rec["document"]
    original_doc = doc_idx[doc_id]

    answer_llm = assistant.rag(question)
    answer_orig = original_doc["answer"]

    result = {
        "question": question,
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": doc_id,
    }

    return result
```

Reset the usage counters (the test calls above already logged some), then fan out over all 395 questions, six workers, same `map_progress` helper:

```python
assistant.reset_usage()

from concurrent.futures import ThreadPoolExecutor
from evaluation_utils import map_progress

with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, ground_truth, generate_rag_answer)

answers = []

for answer_record in results:
    answers.append(answer_record)

assistant.total_cost()

df_answers = pd.DataFrame(answers)
df_answers.to_csv("data/rag-answers-new.csv", index=False)
```

Recorded run: 395 questions, $0.343 total, about 34 cents. Skip it if you like:

```bash
PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main

wget -O data/rag-answers-new.csv ${PREFIX}/04-evaluation/data/rag-answers-new.csv
```

I confirmed the shipped file holds exactly 395 records with the four expected columns.

---
## 11. LLM as judge

String matching cannot grade generative answers. The RAG answer will phrase things differently even when it is right, so exact comparison fails on good output. Instead, another LLM call reads the question plus both answers and decides: same key information, or not. That is the whole technique, and "LLM-as-a-judge" is just its name.

New notebook (`04-llm-judge.ipynb`). Load the answers CSV from section 10. In production you would not have the original answer for real user questions, and the judge prompt would have to work from question plus generated answer alone. Here we have the stronger offline setup, so use it.

### 11.1 Output shape: verdict plus reasoning

```python
from pydantic import BaseModel, Field
from typing import Literal

class AnswerEvaluation(BaseModel):
    reasoning: str = Field(
        description="Reasoning about the quality of the answer."
    )
    score: Literal["good", "bad"] = Field(
        description="'good' if the answer is correct and complete, 'bad' otherwise."
    )
```

Two fields for two jobs. The score aggregates into a metric. The reasoning tells you why, which is what you actually read when something fails. Asking for the explanation first generally gets better verdicts than asking for the verdict alone.

### 11.2 Instructions and template

```python
aqa_judge_instructions = """
You are an expert evaluator. You will be given:
1. A question from a student
2. The original answer from the FAQ (ground truth)
3. An answer generated by an AI assistant

Your task is to decide if the AI answer is semantically equivalent to
the original answer.

Rules:
- The AI answer does NOT need to be word-for-word identical
- It should convey the same key information
- Extra detail is fine as long as the core answer is correct
- Mark 'bad' only if the AI answer is wrong or misses the key point

Be fair and focus on correctness, not style.
""".strip()
```

```python
aqa_judge_prompt = """
Question:
{question}

Original Answer (ground truth):
{answer_orig}

AI Answer:
{answer_llm}
""".strip()
```

One record through the retrying helper, price it:

```python
from evaluation_utils import calc_price, calc_total_price, llm_structured_retry, map_progress

eval_result, usage = llm_structured_retry(
    openai_client,
    aqa_judge_instructions,
    prompt,
    AnswerEvaluation,
)

calc_price(usage)
```

Wrap it (the `model` parameter lets you try a stronger judge later without rewriting):

```python
def evaluate_aqa(question, answer_orig, answer_llm, model="gpt-5.4-mini"):
    prompt = aqa_judge_prompt.format(
        question=question,
        answer_orig=answer_orig,
        answer_llm=answer_llm
    )

    result, usage = llm_structured_retry(
        openai_client,
        aqa_judge_instructions,
        prompt,
        AnswerEvaluation,
        model=model,
    )

    return result, usage
```

### 11.3 Judge everything, read the failures

```python
def judge_record(rec):
    eval_result, usage = evaluate_aqa(
        question=rec["question"],
        answer_orig=rec["answer_orig"],
        answer_llm=rec["answer_llm"]
    )

    result = {
        "question": rec["question"],
        "document": rec["document"],
        "score": eval_result.score,
        "reasoning": eval_result.reasoning,
    }

    return result, usage
```

Parallel as usual, split evaluations from usages, price the batch, save:

```python
with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, answers, judge_record)

evaluations = []
usages = []

for evaluation, usage in results:
    evaluations.append(evaluation)
    usages.append(usage)

df_eval = pd.DataFrame(evaluations)
calc_total_price(usages)

good_count = (df_eval["score"] == "good").sum()
total_count = len(df_eval)
print(f"Good: {good_count}/{total_count} = {good_count/total_count:.2%}")

df_eval.to_csv("data/rag-evaluations-new.csv", index=False)
```

Recorded: 379 good, 16 bad out of 395, judge cost $0.251. Or download the file.

The bad rows are the most valuable output of the whole exercise. Typical causes: wrong doc retrieved, answer too generic, or the pipeline saying "I don't know" when the FAQ had the answer. Each one points at a stage to fix.

### 11.4 Who judges the judge

The judge can be wrong too, usually too lenient: good verdict on an answer built from the wrong doc. Catching that needs your own eyes on a sample of both verdicts. No second judge fixes this; a judge checking a judge just moves the trust problem one step over.

The practical version from the lesson: a small Streamlit app showing question, both answers, and verdict side by side, where you mark verdicts right or wrong and tighten the instructions based on what you see. Tedious, manual, and the thing that makes the framework trustworthy.

---
## 12. Agent evaluation

Same A → Q → A′ setup, except A′ comes from an agent instead of the fixed pipeline. And we save one more thing: the trajectory, meaning the tool calls the agent made before answering. (Just the calls here, name plus arguments, not the full message history.)

### 12.1 The agent and its search tool

Reuse the ToyAIKit agent from module 1. It runs the loop and keeps the history. Define search with the tuned boosts from section 8 baked in:

```python
from dotenv import load_dotenv
from openai import OpenAI
from toyaikit.llm import OpenAIClient

load_dotenv()
openai_client = OpenAI()
```

```python
def search(query: str) -> list[dict]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 1.0, "answer": 2.0, "section": 0.1},
        filter_dict={"course": "llm-zoomcamp"}
    )
```

```python
from toyaikit.tools import Tools
from toyaikit.chat.runners import OpenAIResponsesRunner

agent_tools = Tools()
agent_tools.add_tool(search)

instructions = """
You're a course teaching assistant. Answer student questions based on
the FAQ search results. Use the search tool before answering.
""".strip()

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)
```

Each run returns `last_message` (the answer), `all_messages` (full history), and `cost`. One question to see the shape:

```python
rec = ground_truth[0]

result = runner.loop(prompt=rec["question"])
result.all_messages
```

### 12.2 Extracting the trajectory

Walk the history, keep function calls with name and arguments, skip the rest:

```python
def extract_tool_calls(messages):
    tool_calls = []

    for message in messages:
        if isinstance(message, dict):
            continue

        if message.type == "function_call":
            tool_calls.append({
                "name": message.name,
                "arguments": message.arguments,
            })

    return tool_calls
```

Typical output is a single search call with a query string in the arguments. The record bundles everything the judge will need:

```python
agent_result = {
    "question": rec["question"],
    "answer_agent": result.last_message,
    "answer_orig": answer_orig,
    "tool_calls": tool_calls,
    "cost": result.cost.total_cost,
    "document": doc_id,
}
```

### 12.3 Batch over 50 questions

Same shape as a function, same parallel pattern:

```python
def generate_agent_answer(rec):
    doc_id = rec["document"]
    original_doc = doc_idx[doc_id]

    result = runner.loop(prompt=rec["question"])

    tool_calls = extract_tool_calls(result.all_messages)

    answer_record = {
        "question": rec["question"],
        "answer_agent": result.last_message,
        "answer_orig": original_doc["answer"],
        "tool_calls": tool_calls,
        "cost": result.cost.total_cost,
        "document": doc_id,
    }

    return answer_record
```

```python
with ThreadPoolExecutor(max_workers=6) as pool:
    agent_answers = map_progress(pool, ground_truth[:50], generate_agent_answer)

df_agent = pd.DataFrame(agent_answers)
df_agent["cost"].sum()
df_agent.to_csv("data/agent-answers.csv", index=False)
```

ToyAIKit tracks cost per run, so the `cost` column sums directly. Recorded: 50 questions, $0.0699 total, about 7 cents. I verified the shipped CSV: exactly 50 records, costs summing to 0.069933.

### 12.4 Judging answers and trajectories

A good trajectory is not a busy one. For this search agent the rubric is concrete: query relevant to the question, important keywords included, no duplicate calls, refinements that actually refine, usually 1 call, 2-3 acceptable for hard questions, more than 3 needs a reason, calls supporting the final answer, no quitting early and no searching forever.

The judge returns two verdicts now:

```python
from pydantic import BaseModel, Field
from typing import Literal

class AgentEvaluation(BaseModel):
    answer_reasoning: str = Field(
        description="Reasoning about whether the final answer is correct."
    )
    answer_score: Literal["good", "bad"] = Field(
        description="'good' if the final answer matches the original answer."
    )
    trajectory_reasoning: str = Field(
        description="Reasoning about whether the tool calls were useful."
    )
    trajectory_score: Literal["good", "bad"] = Field(
        description="'good' if the tool calls were reasonable for the question."
    )
```

Instructions cover both halves (answer equivalence plus the trajectory rubric), and the prompt template adds the tool calls as a fourth block after question, original, and agent answer. One wrinkle the code handles: records loaded back from CSV carry `tool_calls` as a string, so the judge function parses it with `json.loads` when needed.

```python
import json
from evaluation_utils import calc_total_price, llm_structured_retry

def evaluate_agent_answer(rec, model="gpt-5.4-mini"):
    tool_calls = rec["tool_calls"]

    if isinstance(tool_calls, str):
        tool_calls = json.loads(tool_calls)

    prompt = agent_judge_prompt.format(
        question=rec["question"],
        answer_orig=rec["answer_orig"],
        answer_agent=rec["answer_agent"],
        tool_calls=json.dumps(tool_calls, indent=2),
    )

    result, usage = llm_structured_retry(
        openai_client,
        agent_judge_instructions,
        prompt,
        AgentEvaluation,
        model=model,
    )

    return result, usage
```

The split verdict is what makes this interesting: bad answer with good trajectory means retrieval worked and the model fumbled the context. Bad on both means it searched for the wrong thing or gave up early. Different failures, different fixes.

Batch, split, price, count, save:

```python
def judge_agent_record(rec):
    agent_eval, usage = evaluate_agent_answer(rec)

    result = {
        "question": rec["question"],
        "document": rec["document"],
        "answer_score": agent_eval.answer_score,
        "answer_reasoning": agent_eval.answer_reasoning,
        "trajectory_score": agent_eval.trajectory_score,
        "trajectory_reasoning": agent_eval.trajectory_reasoning,
    }

    return result, usage
```

```python
with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, agent_answers, judge_agent_record)

agent_evaluations = []
usages = []

for evaluation, usage in results:
    agent_evaluations.append(evaluation)
    usages.append(usage)

df_agent_eval = pd.DataFrame(agent_evaluations)
calc_total_price(usages)

df_agent_eval["answer_score"].value_counts()
df_agent_eval["trajectory_score"].value_counts()

df_agent_eval.to_csv("data/agent-evaluations.csv", index=False)
```

Recorded: answers 45 good / 5 bad, trajectories 49 good / 1 bad, judge cost $0.053 (29,228 input tokens, 6,984 output). I verified the shipped file: 50 records with all six columns.

---
## 13. Things to try

1. Hand-compute one metric. Take the 15-row sample from section 6, count the hits, then score the ranks. If your MRR does not come out near 0.82, recheck the `rank + 1`.
2. Break a boost on purpose. Set question to 10.0, re-run `evaluate()`, watch both metrics fall. Then you will believe section 8 instead of just reading it.
3. Read five bad judge rows. For each, decide: retrieval failure, prompt failure, or model failure. The judge tells you where to look; the diagnosis is yours.
4. Tighten the judge. Find a lenient verdict (good score on a wrong-doc answer), add one sentence to the instructions ruling that case out, re-run on the bad subset.
5. Compare RAG vs agent on the same 50 questions. Same ground truth, different A′ producers. Where does the agent win, and is it worth the extra calls?
6. Count trajectory lengths in `agent-answers.csv`. How many runs used more than 1 call? Do those answers score better or just cost more?
7. Swap the judge model via the `model` parameter and see if the good/bad split moves. If it does, your instructions are doing less work than you think.
8. Price your own run end to end. Ground truth (~6c) + RAG answers (~34c) + RAG judge (~25c) + agent (~7c) + agent judge (~5c) lands under a dollar for the full module. Knowing that number makes re-running painless.

---
## 14. Architecture review

```text
Test data (once, ~6 cents)
  79 FAQ docs → LLM invents 5 questions each → 395 (question, doc-id) rows
        ↓
Retrieval eval (no LLM calls)
  each question → search → relevance list → hit_rate + mrr
  tune boosts on the same frozen set → 1.0 / 2.0 / 0.1
        ↓
Answer eval (~34c + ~25c)
  each question → RAGWithUsage → A′ → judge vs A → good/bad + reasoning
        ↓
Agent eval (~7c + ~5c)
  each question → ToyAIKit loop → A′ + tool calls → dual verdicts
        ↓
Decisions
  ship the boosts, fix the bad rows, re-run on every prompt/model change
```

Three sentences for anyone asking what the module was about: you cannot tell which system is better by trying queries by hand, so you manufacture questions with known answers and measure. Retrieval gets numbers (hit rate, MRR); answers get verdicts (an LLM judge); agents get both verdict and trajectory review. Then you re-run the whole thing every time anything changes, because that is the only way the numbers stay honest.

---
## 15. What you should now understand

- Ground truth is questions paired with the id of the doc that answers them. No stable ids, no evaluation. Assign them first on your own data.
- Structured output (`responses.parse` + Pydantic) exists so code can consume model output without parsing prose.
- Batch LLM work needs retries (transient failures are normal) and parallelism (network waits dominate), with worker counts that respect rate limits.
- A relevance list is the bridge between search output and metrics: 1 where the ids match, 0 elsewhere.
- Hit rate says found-or-not; MRR says how far down. Read them as a pair.
- The fixed dataset is what makes comparisons fair. Change one thing at a time or the delta means nothing.
- Intuitions about boosts lose to measurements. Here the answer field mattered twice as much as the question field, against everyone's first guess.
- `evaluate(ground_truth, search_function)` works for any retrieval backend because it only assumes query-in, results-out.
- RAG eval checks the full chain at once, which means failures need triage across search, prompt, and model.
- LLM judges need reasoning attached to verdicts, and the judges themselves need spot-checking by a human. No turtles all the way down.
- Agent trajectories get their own verdict because a right answer via nonsense tool calls is luck, not a system.
- Synthetic data starts the process and real user data finishes it. Above-95% scores on synthetic questions are a smell, not a win.

---
## 16. Self-test

Try these from memory. Answers follow.

1. What three options exist for getting ground truth, and which does this module use?
2. Why does every record need a stable document id?
3. What does `responses.parse` give you that `responses.create` does not?
4. Why ask the generator to reuse as few words from the record as possible?
5. What do `llm_structured_retry` and `map_progress` each protect you from?
6. What is a relevance list, concretely?
7. Why does the MRR code use `rank + 1`?
8. What does `evaluate()` assume about the search function it receives?
9. Your boost sweep says question=1.0 beats question=3.0. What do you conclude?
10. The top grid rows share the ratio 1:2:0.1. Why pick the smallest numbers?
11. Why does `RAGWithUsage` override `search` instead of reusing `RAGBase.search`?
12. What four fields go into a RAG answer record, and why each?
13. Why can't exact string matching grade RAG answers?
14. What two things does the judge return, and what is each one for?
15. Your judge says good but search retrieved the wrong doc. What happened and what do you do?
16. What is a trajectory here, and what is deliberately excluded from it?
17. Bad answer + good trajectory vs bad on both: what does each combination mean?
18. Why does `evaluate_agent_answer` handle `tool_calls` being a string?
19. When is synthetic data no longer enough?
20. What do you re-run when you change a prompt, swap a model, or retune boosts?

---

Answers:

1. Human annotators, labeled real queries, LLM-generated synthetic data. The module uses synthetic: 5 questions per FAQ doc from `gpt-5.4-mini`.
2. The id is the label. Metrics check whether search returned that exact doc; without unique ids the check is impossible.
3. A parsed object in a fixed shape (`output_parsed`) instead of free text, so code can read fields directly.
4. Copycat wording makes retrieval trivially easy and inflates hit rate and MRR. Real users phrase things differently.
5. Retry survives transient API/network failures without killing the batch; parallel fan-out stops you waiting on 79 sequential network round trips.
6. One 0/1 per retrieved doc: 1 where the retrieved id equals the ground truth doc id, in rank order.
7. Python ranks start at 0. Rank 1 must score 1/1, and dividing by raw rank 0 would blow up.
8. Only that it takes a query and returns results with ids. That is why text, vector, or hybrid search all plug in unchanged.
9. The intuition (boost the question field) was wrong for this data. Trust the measurement, ship 1.0, keep going.
10. Only relative weights affect ranking. Small numbers say the same thing and are easier to read.
11. To bake in the tuned boosts (1.0/2.0/0.1) and record token usage per call, while keeping `rag()` identical.
12. Question, LLM answer, original answer, doc id: the question reproduces the run, the two answers are what the judge compares, the id traces failures back to the source doc.
13. Generative answers rephrase correct content. Word overlap punishes right answers for style differences.
14. Score (aggregates into the metric) and reasoning (explains the verdict so you can debug failures).
15. The judge was too lenient. Sample verdicts by hand, tighten the instructions against that case, re-run.
16. The tool calls (names + arguments) the agent made before answering. The full message history is excluded; the judge does not need it.
17. Good trajectory + bad answer: retrieval worked, the model mishandled the context. Both bad: it searched wrong or quit early. Different bugs, different fixes.
18. Records reloaded from CSV carry `tool_calls` as a JSON string, not a list. Parse-then-format keeps both paths working.
19. As soon as you have real traffic. Synthetic questions echo source text and inflate metrics; labeled user queries are the reliable ground truth.
20. Everything. The full offline suite, on the frozen dataset, before anything reaches users.

---
## 17. Where to go from here

Climb the data ladder from lesson 15: synthetic baseline, tune generation until questions look real, deploy, collect user queries, label them by hand, then feed real patterns back into the generator. Manual probing never stops being useful, especially early when automation is thin.

For production setups, the lesson names three frameworks that package these ideas with tracking and dashboards: Ragas (faithfulness, answer relevance, context precision), DeepEval (hallucination detection among others), TruLens (instrumentation plus quality metrics). Monitoring itself, user feedback buttons, query/answer logs, threshold alerts, lives in module 5.

Older cohort versions exist (2024 used Elasticsearch for the monitoring-flavored take, 2025 has its own evaluation module) if you want a second angle. And the homework (`cohorts/2026/04-evaluation/homework.md`) exercises this module end to end; several of the section 13 tries overlap with its questions, deliberately.
