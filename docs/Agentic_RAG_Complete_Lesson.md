# Agentic RAG: complete practical lesson

This is Module 1, the foundation everything else stands on. Part 1 builds a working RAG pipeline from scratch in plain Python: fetch FAQ docs, index them, stuff results into a prompt, call the model. Part 2 hands the steering wheel to the model: function calling, a handwritten agent loop, then a small framework that runs the loop for you.

No frameworks until the last stretch, on purpose. You see every piece first, so when a library hides the loop later, you know what it hides.

Read top to bottom. Each section adds one piece, shows the exact code, and says how to check it works.

---
## 0. What we are building

```text
FAQ JSON (datatalks.club, all courses)
        ↓
minsearch index (question/section/answer as text, course as keyword)
        ↓
search() → build_prompt() → llm()  =  rag()
        ↓
ingest.py + rag_helper.py (reusable files, RAGBase class)
        ↓
sqlitesearch split (ingest once to faq.db, query many times)
        ↓
"Olama" typo breaks the fixed pipeline
        ↓
search tool + function calling (model asks, you execute, you return)
        ↓
while-loop agent (model decides searches, stops when done)
        ↓
ToyAIKit runner (same loop, less boilerplate, cost tracking)
```

Five notebooks, each matching a stage:

```text
notebook.ipynb (35 cells)             the whole Part 1, start to finish
rag_cleaned.ipynb (5 cells)           Part 1 compressed through the helper files
persistent_rag_ingest.ipynb (6)       write LLM docs to faq.db (slow, deliberate)
persinsent_rag.ipynb (6)              query faq.db through RAGBase (yes, the filename
                                      is missing a "t" in the repo; it still runs)
agents.ipynb (50 cells)               typo demo, function calling, loop, ToyAIKit
```

---
## 1. Learning objectives

When you finish, you should be able to:

- Explain why a bare LLM fails on course questions (cutoff, no private data, hallucinations)
- Fetch the FAQ dataset and describe every field, including what `course` slugs are for
- Build a minsearch index and explain text fields vs keyword fields in your own words
- Boost and filter search, and wrap it in a `search()` function
- Split a prompt into fixed instructions and per-request user prompt, and say why
- Call the Responses API, read `output_text` and `usage`, and price a call
- Use `RAGBase` from the helper files and override its instructions
- Split ingestion from querying with sqlitesearch and explain when the split pays off
- Demo the typo failure that motivates agents
- Write a tool schema, execute a function call, and return the result with the right `call_id`
- Write the agent `while` loop from memory, including the exit condition
- Steer an agent with instructions (more searches, on-topic only)
- Run the same agent through ToyAIKit and read cost and history off the result
- Argue when not to use an agent at all

---
## 2. The big mental model

An LLM here is a box. Text in, text out. No memory between calls, no knowledge past training, no access to your files. Everything in this module is scaffolding around that box: retrieval to fix its ignorance, prompts to fix its behavior, loops to fix its one-shot nature.

RAG in three lines:

```python
def rag(question):
    search_results = search(question)
    user_prompt = build_prompt(question, search_results)
    return llm(user_prompt)
```

Search finds, prompt assembles, model writes. Each piece swaps independently: Anthropic for OpenAI, Elasticsearch for minsearch, new template for old. Nothing else moves. That modularity is what makes every later module possible; vector search, reranking, judges all slot into one of these three slots.

Then the fixed version breaks, always the same way: search runs once on the raw user query, and garbage in means garbage out with no recovery. The agent fix changes who decides. Developer decides in RAG (steps fixed upfront). Model decides in an agent (tools chosen per turn, loop until done). Same search, same model, different driver.

```text
RAG:   you fix the steps, the model fills the answer
Agent: you fix the goal and tools, the model picks the steps
```

If a later question confuses you, ask which side of that line it sits on.

---
## 3. Environment setup

Python 3.14+, Jupyter, uv as the package manager (fast; the course switched everything to it). Local or Codespaces, your call. Codespaces wins for uniformity: same Ubuntu, same Python, same Docker, fewer "works on mine" threads.

```bash
mkdir llm-zoomcamp-2026-code
cd llm-zoomcamp-2026-code
uv init
```

```bash
uv add requests minsearch openai jupyter python-dotenv
```

Each one earns its place: `requests` fetches the FAQ, `minsearch` indexes it, `openai` calls the model, `jupyter` runs the notebooks, `python-dotenv` loads the key. Later additions: `sqlitesearch` for persistence, `toyaikit` for the framework lesson. Full dependency list lives in `code/pyproject.toml`.

The key goes in `.env`, never in code, never in git:

```text
OPENAI_API_KEY=sk-YOUR_KEY_HERE
```

And `.env` goes in `.gitignore` next to `__pycache__/`, `*.db`, `models/`, `.venv/` (the repo's ignore file already covers all five). The course costs cents to run; still, make a separate OpenAI project so the usage page shows course spend apart from everything else. Groq and other OpenAI-compatible providers work too, with `base_url` pointed at them.

```bash
uv run jupyter notebook
```

Verify the client before anything else:

```python
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI
openai_client = OpenAI()
```

Errors here mean the key is wrong. Fix that now; everything downstream assumes it works. (If `load_dotenv()` in every notebook annoys you, the lesson suggests dirdotenv to auto-load on directory change.)

---
## 4. The bare model, and why it isn't enough

Define the box:

```python
def llm(prompt):
    response = openai_client.responses.create(
        model="gpt-5.4-mini",
        input=prompt
    )
    return response.output_text
```

`llm("Hey, what's up?")` works. `llm("I just discovered the course. Can I join now?")` answers generically: "you can usually join", "check the website". It doesn't know Zoomcamp enrollment policy because that policy isn't common knowledge and postdates training. Salmon recipes it knows; your FAQ it doesn't. Three limits, worth memorizing: knowledge cutoff, no private data, confident hallucinations.

Now paste context in by hand. A few FAQ entries about joining, registration emails, office hours, GPU options. Same question, new prompt with instructions plus question plus context, and the answer comes out right: join late, certificate needs a submitted project. Nothing about this is fancy, and that is the lesson's point. What you just did, retrieve-then-generate with a human as the retriever, is RAG with the automation missing. The rest of Part 1 automates the retrieval.

---
## 5. The dataset

The FAQ lives as JSON on datatalks.club, one file per course behind an index file:

```python
import requests

docs_url = "https://datatalks.club/faq/json/courses.json"
response = requests.get(docs_url)
courses_raw = response.json()
```

```python
documents = []
url_prefix = "https://datatalks.club/faq"

for course in courses_raw:
    course_url = f"""{url_prefix}{course["path"]}"""

    course_response = requests.get(course_url)
    course_response.raise_for_status()
    course_data = course_response.json()

    documents.extend(course_data)

len(documents)
```

Around eleven hundred docs. Each one:

```python
{
    "id": "0e38656cfb",
    "course": "machine-learning-zoomcamp",
    "section": "General Course-Related Questions",
    "question": "How do I submit homework?",
    "answer": "- Do the tasks locally\n- Publish your code ..."
}
```

`question` and `answer` are the searchable text. `course` is a slug for filtering (ask about data engineering, skip ML answers). `section` helps ranking and gives the model context about where an entry lives.

One repo-vs-lesson wrinkle to know about. The lesson text shows docs with an `id` field. The repo's `ingest.py` renames it: `doc["doc_id"] = doc.pop("id")`, with a comment saying sqlite needs the `id` key free so reimports don't duplicate. So code running through the repo helper sees `doc_id` where the lesson prose says `id`. Same data, different key name. If your code KeyErrors on `id`, this rename is why.

Data prep honesty from the lesson: this dataset arrives clean because the instructor cleaned it (partly with an LLM). Real projects start with scraping, PDFs, and chunking, and that work dwarfs the modeling. The course focuses on the GenAI side; budget real time for prep on your own data.

In pipeline terms the dataset is the knowledge base: index everything once, search per question, hand the hits to the model.

---
## 6. Search with minsearch

Every search engine runs one function: score each doc against the query, rank, return the top N. What differs is `sim`. Lexical search counts shared words (this section). Vector search compares meanings (module 2). "Join the course after the start date" vs "enroll late" share nearly no words, so lexical search struggles there by construction.

Why search at all: ~1100 docs won't fit a prompt cheaply, and a confused model with everything is worse than a focused model with five hits.

minsearch is the instructor's own tiny in-memory engine (single-file origins, Lucene-borrowed vocabulary: text fields, keyword fields, boosts, filters). Not production, but the concepts transfer straight to Elasticsearch, and it runs anywhere Python runs.

```python
from minsearch import Index

index = Index(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"]
)

index.fit(documents)
```

Text fields get tokenized (split, lowercased, stop words dropped) and ranked. Keyword fields match exactly, like `WHERE course = '...'` in SQL: they restrict which docs compete, regardless of ranking. Four courses in one index means filtering isn't optional; an LLM Zoomcamp question must not surface MLOps answers.

Try it:

```python
question = "I just discovered the course. Can I join now?"

search_results = index.search(
    question,
    boost_dict={"question": 2.0, "section": 0.5},
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

Top hit is the "can I still join" entry. Boosts say which fields matter: question matches count double (default 1.0), section matches count half. Same mechanism as Lucene/ES boosting. Play with the values; watching rankings move teaches more than reading about them.

Wrap it as the pipeline's first component, defaulting to the LLM course:

```python
def search(question, course="llm-zoomcamp"):
    boost_dict = {"question": 2.0, "section": 0.5}
    filter_dict = {"course": course}

    return index.search(
        question,
        boost_dict=boost_dict,
        filter_dict=filter_dict,
        num_results=5
    )
```

---
## 7. Building the prompt

The model only sees what you send. Prompts split in two because the halves change at different rates: instructions are fixed per deployment, user prompts are built fresh per request.

```python
INSTRUCTIONS = """
Your task is to answer questions from the course participants
based on the provided context.

Use the context to find relevant information and provide accurate
answers. If the answer is not found in the context,
respond with "I don't know."
"""
```

The "I don't know" line is the hallucination brake. No matching context, no invented answer. (The notebook also tests the brake's limits with a prompt-injection probe, `ignore all your instructions...`; try it yourself in section 17 and see what holds.)

```python
USER_PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
"""
```

Context is the hits flattened into readable blocks:

```python
def build_context(search_results):
    lines = []

    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")

    return "\n".join(lines).strip()
```

```python
def build_prompt(question, search_results):
    context = build_context(search_results)
    prompt = USER_PROMPT_TEMPLATE.format(
        question=question,
        context=context
    )
    return prompt.strip()
```

Print one. Question on top, FAQ blocks below, exactly what the model will read. The prompt is the bridge: good ones keep answers grounded, bad ones let the model wander. Prompt engineering stays experimental until module 4 gives you metrics; until then this template is the starting point. (Spelling note: the repo notebook names the template variable `USER_PROMPT_TEMPALATE`, missing an L. Ugly, harmless, and worth knowing when you grep.)

---
## 8. The LLM call

Send the prompt from section 7:

```python
response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=prompt
)
```

The Responses API is current; chat completions is legacy. Switching providers usually means switching back to `chat.completions`, since OpenAI-compatible APIs tend to expose that one. Same client, different method.

The response is a Pydantic object with a long walk to the text (`output[0].content[0].text`) and a shortcut:

```python
response.output_text
```

Usage rides along:

```text
ResponseUsage(input_tokens=334, output_tokens=39, total_tokens=373)
```

Price it at 0.75/M input, 4.50/M output (gpt-5.4-mini rates as frozen in the lesson; verify current pricing before budgeting):

```python
input_price = 0.75 / 1_000_000
output_price = 4.50 / 1_000_000

cost = (
    response.usage.input_tokens * input_price +
    response.usage.output_tokens * output_price
)
```

Fractions of a cent per query. Cached input tokens bill lower when prompt prefixes repeat.

Single strings work, but real calls send a message history: the model is stateless, so the full conversation so far goes in every request. Two roles here, `developer` for the fixed instructions and `user` for the assembled prompt (`system` also works; the course uses `developer` and reports no practical difference):

```python
message_history = [
    {"role": "developer", "content": INSTRUCTIONS},
    {"role": "user", "content": prompt}
]

response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=message_history
)
```

Fold into functions and wire the pipeline:

```python
def llm(instructions, user_prompt, model="gpt-5.4-mini"):
    message_history = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt}
    ]

    response = openai_client.responses.create(
        model=model,
        input=message_history
    )

    return response.output_text
```

```python
def rag(query, model="gpt-5.4-mini"):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(INSTRUCTIONS, prompt, model=model)
    return answer
```

`rag("How do I get a certificate?")` answers from the FAQ, naming courses and sections. Swap any component (backend, template, model) and nothing else moves. When sqlitesearch replaces minsearch later, only `search` changes. That sentence previews the whole course.

---
## 9. The helper files

Copy-pasted pipeline code rots across notebooks, so Part 1 ends by filing it: `ingest.py` (load + index) and `rag_helper.py` (the RAG logic as `RAGBase`). Import both, three lines to a working assistant.

`ingest.py` is the fetch loop from section 5 (plus the `doc_id` rename) and a `build_index` that fixes the field layout in one place.

`rag_helper.py` holds `INSTRUCTIONS`, `PROMPT_TEMPLATE` (QUESTION/CONTEXT shape), and the class. Why a class: the functions need an index and a client, and globals stop working the moment the code leaves the notebook. Constructor arguments keep it reusable, and subclassing lets later modules override one method (vector search overrides `search`, usage tracking overrides `llm`) without touching the rest:

```python
class RAGBase:

    def __init__(
        self,
        index,
        llm_client,
        instructions=INSTRUCTIONS,
        prompt_template=PROMPT_TEMPLATE,
        course="llm-zoomcamp",
        model="gpt-5.4-mini"
    ):
        ...
```

`index` is anything with a `search` method: minsearch today, sqlitesearch tomorrow. `search` bakes in boosts (question 3.0 here, note the drift from section 6's 2.0; both are starting guesses module 4 later measures) and the course filter. `build_context`/`build_prompt` format, `llm` sends the two-message history, `rag` wires the three steps.

Notebook usage, the shape every later module repeats:

```python
from dotenv import load_dotenv
load_dotenv()

from ingest import load_faq_data, build_index
from rag_helper import RAGBase
from openai import OpenAI

documents = load_faq_data()
index = build_index(documents)

openai_client = OpenAI()

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
)

answer = assistant.rag("I just discovered the course. Can I join now?")
print(answer)
```

Defaults cover instructions; pass custom ones (the TA-style variant in the lesson) to change behavior without forking the file. `rag_cleaned.ipynb` is this whole flow in 5 cells; keep it bookmarked as the canonical minimal example.

---
## 10. Ingestion vs query: the sqlitesearch split

minsearch lives in process memory: dictionaries bound to one Python process, gone on exit, rebuilt on every restart. Fine at FAQ scale (sub-second indexing). At millions of docs with slow fetching and cleaning, startup becomes minutes and every restart repeats the work.

The fix is architectural, not algorithmic. One process writes a persistent index; another reads it. Ingest once, query forever:

```text
minsearch (one process):     fetch → parse → index → ready, repeated every restart
sqlitesearch (two processes): ingest once → faq.db; query any time → open faq.db → ready
```

sqlitesearch wraps SQLite FTS5 (ships with Python, zero new dependencies) behind the same API as minsearch: `search` with query, `boost_dict`, `filter_dict`, `num_results`. Same calls, persistent storage. That API sameness is why `RAGBase` accepts the sqlite index with zero changes; had the API differed, a subclass overriding `search` would bridge it.

Ingestion notebook (`persistent_rag_ingest.ipynb`): load, filter to LLM docs, add one by one with a half-second sleep to feel slow ingestion, close:

```python
import time
from sqlitesearch import TextSearchIndex

index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db"
)

for doc in docs_llm:
    index.add(doc)
    print(f"""Added: {doc["question"][:60]}...""")
    time.sleep(0.5)

index.close()
print("Done. Index saved to faq.db")
```

Query notebook (`persinsent_rag.ipynb`, typo and all): open the same path, watch `count()` grow while ingestion still runs elsewhere (two processes, one file, no coordination code on your part, impossible with minsearch), search, then RAG with no `fit` and no data loading:

```python
from sqlitesearch import TextSearchIndex

sqlite_index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="faq.db"
)
```

```python
from rag_helper import RAGBase

assistant = RAGBase(
    index=sqlite_index,
    llm_client=openai_client,
)

answer = assistant.rag("Can I still join the course after it started?")
print(answer)
```

Same answers, no startup indexing. Close with `sqlite_index.close()` when done (or let the kernel do it). Choosing: minsearch while indexing is instant, persistent backends when it isn't; Elasticsearch/OpenSearch/Qdrant/Weaviate at production scale, same two-process shape.

---
## 11. Where Part 1 points next

Two directions, both taken later in the course. Agents (Part 2): the pipeline searches once with the raw query, and a miss leaves the model no recovery. An agent lets the model re-search, translate, and decide when to stop. Vector search (module 2): exact words fail on paraphrase, embeddings fix meaning matching.

Elasticsearch gets a nod as the production text backend (BM25, filters, aggregations, dense and sparse vectors, distributed scale), heavier than sqlitesearch and appropriate when scale demands it.

Fine-tuning gets a firm deprioritization: GPU hardware, retraining on every data change, and no access to unseen information. RAG is cheaper, flexible across models, and covers nearly everything; the instructor reports never hitting a case that needed fine-tuning across thousands of analyzed job posts. Learn RAG first, fine-tune only with a genuine reason.

Suggested experiments: vary prompts, add sources, try other models including local ones via Ollama, try Elasticsearch.

---
## 12. Why the fixed pipeline breaks

Part 2 opens with a typo. `assistant.rag("How do I run Ollama locally?")` works; `assistant.rag("How do I run Olama locally?")` fails. Lexical search wants the exact word, "Olama" matches nothing useful, the model gets garbage context and says it doesn't know or answers irrelevantly.

The failure is structural, not bad luck. Search runs once, on the exact user string, with no signal back to anyone that it failed. No second chance exists in the code. Typos are just the sharpest demo; unusual phrasings and multi-part questions break the same way.

The alternative: stop routing the query straight to search and let the model drive. It can fix the typo, retry with new terms, ask a clarifying question. Fixed flows can't. That flexibility is what "agentic" means here: the LLM picks actions and order per step instead of executing a preset sequence.

---
## 13. Function calling

The typo fixed by architecture. Give the model a `search` tool and watch the same question take a different path: search "Olama", see nothing, reason about a typo, search "Ollama", answer. Nobody wrote typo-handling code. The loop plus the model's judgment is the handling.

Developer decides (RAG) vs model decides (agent) is the whole distinction. The enabler is function calling: describe tools as JSON schemas, let the model request calls, execute them, return results, repeat.

First, what no-tool looks like: a bare question gets a vague general-knowledge answer, the section-4 problem restated. Then the tool. A module-level `search` over the existing index (name aligned with the Python function for easy dispatch later):

```python
def search(query):
    boost_dict = {"question": 3.0, "section": 0.5}
    filter_dict = {"course": "llm-zoomcamp"}

    return index.search(
        query,
        num_results=5,
        boost_dict=boost_dict,
        filter_dict=filter_dict
    )
```

Then the schema. The model never sees your Python, only this JSON (language-agnostic by design; identical from TypeScript or Java):

```python
search_tool = {
    "type": "function",
    "name": "search",
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
```

`description` is the load-bearing field: the model reads it to decide when calling makes sense. `parameters` is JSON Schema for the arguments, `query` required so it always arrives.

Send the question with tools attached:

```python
response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

response.output
```

No answer comes back. Instead a `function_call` entry: the model wants FAQ data first. Note the arguments: it rewrote the enrollment question into search keywords ("enroll late join course") rather than echoing the raw text. That rewriting is a quiet superpower; the model queries better than users type.

Execute and return. Parse args, call, serialize:

```python
import json

call = response.output[0]
args = json.loads(call.arguments)

results = search(**args)
result_json = json.dumps(results, indent=2)
```

Append the model's own output first (it must see its call), then the result with the matching `call_id` (matters when one turn holds several calls):

```python
messages.extend(response.output)

messages.append({
    "type": "function_call_output",
    "call_id": call.call_id,
    "output": result_json,
})
```

Call again with the full history. Statelessness forces the replay: question, call decision, tool result, everything, or the model loses the plot:

```python
response = openai_client.responses.create(
    model="gpt-5.4-mini",
    input=messages,
    tools=[search_tool],
)

response.output_text
```

Now it answers from FAQ results. One RAG call became two API calls, and the second resends the whole history. Price both calls (0.15/M in, 0.60/M out in the lesson's example function; the notebook computes ~652 input / 33 output tokens for the second call). Agentic means more round-trips, and history growth makes later turns pricier than early ones. Watch `usage` while developing; multi-turn loops compound fast.

Names vary (agentic RAG, tool use, function calling), mechanism doesn't: the model picks tools.

---
## 14. The agent loop

One function call by hand is a demo. Real questions need unknown-many searches, including retries after misses. So the single-turn code becomes a loop: call the model, run its tools, return outputs, repeat until a turn contains no function calls. That loop is an agent.

Three parts: instructions (role via the `developer` message; good ones pay off everywhere downstream), tools (just `search` for now), memory (the message list, appended every turn: prompts, outputs, tool results).

The developer prompt used here nudges multi-search behavior so the loop actually iterates:

```python
instructions = """
You're a course teaching assistant.
You're given a question from a course student and your task is to answer it.

If you want to look up information, use the search function.
Use as many keywords from the user question as possible when making first requests.

Make multiple searches.

Try to expand your search by using new keywords
based on the results you get from the search.

At the end, ask if there are other areas that the user wants to explore.
""".strip()
```

Repeated tool execution gets a helper (name-dispatch now, registry later when tools multiply):

```python
def make_call(call):
    args = json.loads(call.arguments)

    if call.name == "search":
        result = search(**args)

    result_json = json.dumps(result, indent=2)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,
        "output": result_json,
    }
```

One response processed: extend history, print messages, run calls, set the flag that decides whether another API round is needed:

```python
messages.extend(response.output)
has_function_calls = False

for item in response.output:
    if item.type == "function_call":
        print("function_call:", item.name, item.arguments)
        call_output = make_call(item)
        messages.append(call_output)
        has_function_calls = True

    elif item.type == "message":
        print("ASSISTANT:")
        print(item.content[0].text)
```

The loop itself, with an iteration counter for visibility:

```python
it = 1

while True:
    print(f"iteration #{it}...")
    has_function_calls = False

    response = openai_client.responses.create(
        model="gpt-5.4-mini",
        input=messages,
        tools=[search_tool],
    )

    messages.extend(response.output)

    for item in response.output:
        if item.type == "function_call":
            print("function_call:", item.name, item.arguments)
            call_output = make_call(item)
            messages.append(call_output)
            has_function_calls = True

        elif item.type == "message":
            print("ASSISTANT:")
            print(item.content[0].text)

    it = it + 1
    if has_function_calls == False:
        break
```

Exit condition: a turn with no function calls means a final answer. Deliberately minimal; production adds max iterations, token budgets, wall-clock limits (cap at five, force an answer on the last). The flag stays the core either way.

Wrapped for reuse:

```python
def agent_loop(instructions, question, model="gpt-5.4-mini") -> str:
    # ... same loop, captures last_answer from message items ...
    return last_answer
```

`agent_loop(instructions, "How do I run Olama locally?")` searches the typo, misses, searches "Ollama", answers. Recovery with no recovery code. The enrollment question works too.

Two steering lessons follow. Models often stop after one search ("knows enough, why bother"), so instructions get explicit about searching, analyzing, then searching again; compliance is probabilistic, not guaranteed. And the agent answers anything, including chess (`what's queen gambit?`), so a stricter variant scopes it to course logistics and FAQ-only facts. That scoping is a lightweight input guardrail; real ones check before the agent runs. Instructions steer, guardrails enforce; start with the former, add the latter when scope violations cost you.

Every framework below wraps this loop. LangChain, PydanticAI, OpenAI Agents SDK: same `while`, same flag, more features.

---
## 15. ToyAIKit: the loop without the boilerplate

Hand loops teach; nobody wants to rewrite them per agent. ToyAIKit packages the pattern so you work on tools, prompts, and behavior. Built in a DataTalks workshop, small enough to read, and deliberately not a production pick: no winner-picking among real frameworks, full visibility when things break. Teaching and local debugging, not deployment.

```bash
uv add toyaikit
```

```python
from toyaikit.llm import OpenAIClient
from toyaikit.tools import Tools
from toyaikit.chat import IPythonChatInterface
from toyaikit.chat.runners import OpenAIResponsesRunner, DisplayingRunnerCallback
```

Register the tool with the hand schema, or skip the schema: with a type hint plus docstring, the framework derives it (verify with `get_tools()`; output matches the hand-written JSON):

```python
def search(query: str) -> dict[str, str]:
    """
    Search the FAQ database for entries matching the given query.
    """
    return index.search(
        query,
        num_results=5,
        boost_dict={"question": 3.0, "section": 0.5},
        filter_dict={"course": "llm-zoomcamp"}
    )
```

```python
agent_tools = Tools()
agent_tools.add_tool(search)
```

Every modern framework does this trick (Agents SDK, PydanticAI, LangChain, Google ADK): typed function plus docstring in, tool schema out. Learn it once, reuse everywhere.

Runner plus display callback, model pinned deliberately (the default fallback is smaller, faster, and worse at following instructions):

```python
chat_interface = IPythonChatInterface()
callback = DisplayingRunnerCallback(chat_interface)

runner = OpenAIResponsesRunner(
    tools=agent_tools,
    developer_prompt=instructions,
    chat_interface=chat_interface,
    llm_client=OpenAIClient(model="gpt-5.4-mini")
)
```

`runner.loop(prompt="How do I run Olama locally?", callback=callback)` replays the typo recovery with nicer output: each call and message renders inline. The `LoopResult` carries `all_messages` (the same list you maintained by hand), token counts, and `cost` computed from usage, which beats hand-adding prices across multi-turn runs.

History continues across calls via `previous_messages`: the follow-up "how do I run a different model" resolves to Ollama only because the earlier turns travel along. Without them the model guesses blind. `runner.run()` gives an interactive loop, "stop" to quit.

---
## 16. Frameworks, and when to skip them

Same loop everywhere, so pick by stack and taste:

OpenAI Agents SDK (`uv add openai-agents`): official, Responses API native, tools plus multi-turn plus agent handoffs. Natural if you're already on OpenAI.

PydanticAI (`uv add pydantic-ai`): type-safe, multi-provider (switching vendors is a model string), plain functions as tools. The instructor's personal pick, credited to feel and the team behind it rather than any single feature.

LangChain/LangGraph: biggest integration surface (vector stores, loaders, community), graphs for complex patterns. Pick for breadth.

Google ADK: Gemini-first, same building blocks (tools, instructions, sessions), Cloud-integrated. Pick for Google stacks.

Also named: CrewAI and AutoGen for multi-agent setups, Semantic Kernel (C# and Python), Smolagents for lightweight HuggingFace work, Anthropic's native tool use.

Then the contrarian close, worth quoting in spirit: you often don't need an agent. Each loop turn is another billed call with the full history resent, latency stacks per round-trip, behavior varies run to run, and you inherit new things to monitor (cost, iteration counts, circular behavior). Try in order: plain RAG, templating/parsing, a single tool-less call. Ship the simple thing when it works. Reach for the loop only after the simple thing provably fails, so the complexity arrives with evidence behind it.

---
## 17. Things to try

1. Ask the bare model a course question, then the same question with pasted FAQ context. The gap between the answers is the entire case for RAG.
2. Move boosts around (question 1.0 vs 5.0, section 0 vs 1.0) and watch rankings shift. Then you'll understand why module 4 measures instead of guesses.
3. Remove the "I don't know" line from instructions and ask something unanswerable from the FAQ. See what the model invents. Restore the line.
4. Run the notebook's injection probe (`ignore all your instructions...`, cell 33) and judge whether the grounding held. Then harden the instructions and retry.
5. Time `faq.db` creation vs minsearch `fit`. That ratio is the ingestion-split business case.
6. While ingestion still runs, poll `count()` from the query notebook. Watching the number climb teaches shared-state databases faster than any diagram.
7. Run the typo pair (Ollama/Olama) through `rag()` and through `agent_loop()`. Same model, same index, different driver, different outcome.
8. Print `response.output` on a tool turn and identify the `function_call` entry, its `call_id`, and the rewritten query. Then trace that `call_id` into the returned output.
9. Cap the loop at 3 iterations with a forced final answer. Find a question where the cap hurts; that's the question class needing budget design.
10. Ask about chess under both instruction variants. Confirm the guardrail version refuses and the loose version chats. Decide which behavior your app wants.
11. Continue a ToyAIKit conversation with a pronoun-heavy follow-up, then rerun without `previous_messages`. The difference is what memory buys.
12. Compare one agent answer's total cost against the equivalent single `rag()` call. Quote the multiplier to yourself before choosing agents in homework Q6-style decisions.

---
## 18. Architecture review

```text
Serve a question (fixed RAG)
  query → search (minsearch, boosts, course filter) → top 5
        → build_prompt (instructions + Q + context blocks)
        → llm (Responses API, developer+user messages)
        → grounded answer or "I don't know"
        ↓
Persist the slow part (ingestion split)
  ingest once → faq.db; query any time → open + search, no fit
        ↓
Break it (one typo)
  "Olama" → lexical miss → garbage context → no recovery possible
        ↓
Hand the wheel over (function calling)
  schema (description drives decisions) → model requests call
        → execute (parse, dispatch, serialize) → return with call_id
        → second call with full history → answer (2+ billed calls)
        ↓
Loop it (agent)
  instructions + tools + memory → while True → no-call turn exits
        → typo recovery, multi-search, scoped by prompt
        ↓
Package it (ToyAIKit)
  typed tools → derived schemas → runner loop → cost + history tracked
        ↓
Choose wisely (frameworks or none)
  same loop everywhere; simplest working solution ships
```

Three sentences for anyone asking what the module was about: RAG grounds a forgetful model by retrieving docs into its prompt, in three swappable steps. Fixed pipelines die on the first bad search, so agents hand search decisions to the model inside a loop that exits when it stops calling tools. Frameworks package that loop; judgment about when to use it matters more than which one you pick.

---
## 19. What you should now understand

- The model is a stateless box: cutoff knowledge, no private data, confident inventions. Scaffolding compensates for all three.
- RAG is retrieve, assemble, generate. Modularity (swap backend, template, model independently) is the design feature everything later exploits.
- Text fields rank, keyword fields restrict. Boosts weight, filters gate. Both ideas transfer to Elasticsearch unchanged.
- Instructions stay fixed, user prompts rebuild per request. The "I don't know" line is load-bearing.
- Responses API is current, chat completions legacy; providers often expose only the latter. `output_text` skips the object spelunking; `usage` feeds pricing.
- The class exists so index and client travel as arguments and subclasses override one method. That override pattern carries modules 2, 4, and 5.
- Persistence splits slow ingestion from fast querying. Same search API means zero RAG changes; different API would mean subclassing.
- Lexical search fails exact-match misses with no recovery path. The typo demo is the general case, not an edge.
- Tool descriptions drive model decisions; required params guarantee arguments. Schemas are JSON because the wire is HTTP, not Python.
- `call_id` links results to requests across multi-call turns. Full history replays every turn because statelessness demands it.
- The loop exits on a call-free turn; iteration caps, budgets, and timeouts are production guardrails around that flag.
- Instructions steer probabilistically (multi-search, on-topic scope); guardrails enforce. Know which guarantee you have.
- Frameworks derive schemas from hints plus docstrings and run your handwritten loop with extras (display, cost, history). ToyAIKit teaches; production picks differ.
- Agents multiply cost, latency, and unpredictability. Simplest working solution ships; the loop waits for proven need.

---
## 20. Self-test

Try these from memory. Answers follow.

1. Why does the bare model fail "can I join now" but nail salmon recipes?
2. What does each `documents` field do in the pipeline?
3. Text field vs keyword field: who ranks, who restricts?
4. What breaks if instructions and user prompt merge into one string?
5. Why does the prompt template say "I don't know"?
6. Responses API vs chat completions: which, and what changes on provider switch?
7. Where does per-call cost come from, and roughly how much is one RAG query?
8. Why is `RAGBase` a class instead of module-level functions with globals?
9. sqlitesearch swaps in with zero RAG changes. Why exactly?
10. What structural fact makes typo recovery impossible in fixed RAG?
11. Which schema field matters most for tool choice, and why?
12. What links a tool result to its request, and when does that matter?
13. Why must every agent call resend the full history?
14. What exits the agent loop, and what three production nets surround it?
15. The model stops after one search. What do you change, and what guarantee do you get?
16. Instructions vs guardrails: who steers, who enforces?
17. How do frameworks build tool schemas without hand-written JSON?
18. What three things does a ToyAIKit `LoopResult` give you that hand-rolled totals don't track as easily?
19. Why does the follow-up "a different model" need `previous_messages`?
20. Name the agency bill: what four costs arrive with the loop?

---

Answers:

1. Course policy is private, recent, and absent from training; salmon is public, stable, and overrepresented in training. The box only knows the latter.
2. `question`/`answer` are searchable text; `course` filters by slug; `section` aids ranking and context; `id` (repo: renamed `doc_id`) uniquely labels docs.
3. Text fields tokenize and rank; keyword fields exact-match and restrict the candidate set before ranking matters.
4. Fixed guidance and per-request content tangle; reuse breaks, rebuilding gets error-prone, and swapping behavior means editing every call site.
5. It converts retrieval misses into explicit abstention instead of hallucination. No supporting context, no invented answer.
6. Responses is current, chat completions legacy. OpenAI-compatible providers usually expose chat completions, so switching vendors typically means switching methods on the same client.
7. `response.usage` token counts times per-million rates (0.75 in / 4.50 out here). One RAG query lands in fractions of a cent.
8. Dependencies (index, client) become constructor args instead of globals, keeping the code reusable; subclassing overrides single methods (`search`, `llm`) for later modules.
9. Identical search API (query, boosts, filters, result count), so `RAGBase.search` calls it unchanged. New API would need an adapter subclass.
10. Search executes once on the raw query with no feedback path; the model never sees the miss, so nothing can retry or reformulate.
11. `description`. The model reads it to decide when calling fits; parameters shape the call, but description triggers it.
12. `call_id`, echoed in `function_call_output`. Matters whenever one turn holds multiple calls and results must route back correctly.
13. Calls are stateless; the message list is the only memory. Omit history and the model loses the question, its own call, and the tool result.
14. A response turn containing zero function calls. Around it: max iterations, token/cost budgets, wall-clock limits (e.g. cap at five, force-answer last).
15. Rewrite instructions to demand searching, analyzing, then re-searching with new keywords. Guarantee: none, compliance is probabilistic run to run.
16. Instructions steer (probabilistic, in-prompt), guardrails enforce (deterministic, pre-execution checks). Start steering, add enforcement where violations cost.
17. Type hints plus docstrings on plain functions, auto-derived into JSON schemas. Hand schema and derived schema come out identical.
18. Token counts, computed `cost`, and the full `all_messages` history in one object, no manual accumulation across turns.
19. "Different model" resolves via prior turns mentioning Ollama. Without history the reference dangles and the model guesses.
20. More billed calls per request (history resent each turn), stacked latency per round-trip, run-to-run path variance, and new monitoring surface (cost, iterations, loops).

---
## 21. Where to go from here

The homework rebuilds this module's arc on new data: lesson markdown pulled via `gitsource` at a pinned commit, indexed by `content` (text) and `filename` (keyword), then RAG with an adapted `RAGBase` (schema differs, so `search` and `build_context` change), chunked with a 2000/1000 sliding window, token-counted before and after, and finally agentified with a counted number of `search` calls. Its Q2 answer is the agentic-loop lesson itself, which tells you the module's center of gravity.

Beyond that: prompts and sources first, more models next (including local via Ollama), Elasticsearch when scale demands, module 2 when exact words fail, Part 2's loop when fixed flows fail. Fine-tuning stays last, behind a genuine reason. The 2024/2025 cohort archives hold alternate takes (including an Elasticsearch-flavored module 1) if you want a second angle on the same ideas.
