# AI orchestration with Kestra: complete practical lesson

This is the whole Module 3 in one file. Modules 1 and 2 taught you to build RAG by hand: Python code, your own loops, your own index. This module moves that work into an orchestrator. You define flows in YAML, Kestra runs them, and AI fills in three different jobs: writing the flows for you, answering from your data, and doing tasks on its own.

Read top to bottom. Each section explains the idea first, then shows the exact flow or command, then tells you what to look at when you run it.

---
## 0. What we are building

Six flows, in the `zoomcamp` namespace, each one adding something the previous one could not do:

```text
1_chat_without_rag      ask Gemini a question with no context (watch it guess)
        ↓
2_chat_with_rag         ingest docs, embed them, ask again (watch it get specific)
        ↓
3_rag_with_websearch    skip ingestion, retrieve live web results instead
        ↓
4_simple_agent          an agent that summarizes text (goal in, no fixed steps)
        ↓
5_web_research_agent    an agent that searches the web on its own and writes a report
        ↓
6_multi_agent_research  two agents: one gathers, one writes the final JSON
```

The shape of the module:

```text
Context problem (why generic assistants fail)
        ↓
AI Copilot (generate flows from a sentence)
        ↓
RAG (answer from your data, not from training memory)
        ↓
Agents (decide steps at runtime, use tools)
        ↓
Multi-agent (split the job across specialists)
        ↓
Production (cost, secrets, logs, fallbacks)
```

Two things this module is not: it is not a Kestra tutorial (assumes you know what a task and a trigger are), and the KV-store embeddings are not a real vector database. The lesson says this outright, and I will repeat it in section 6 so nobody builds on it by accident.

---
## 1. Learning objectives

When you finish, you should be able to:

- Explain why a generic assistant writes broken Kestra flows, in one sentence: it has no access to current plugin docs
- Start Kestra locally with Docker Compose and get the UI open at localhost:8080
- Store API keys as base64 `SECRET_` variables and reference them with `secret()` in flows
- Import the six example flows through the API or the UI
- Use AI Copilot to generate a flow and apply the 5% rule to finish it
- Tell the ingest phase apart from the query phase in a RAG flow, and say when each one runs
- Explain static RAG vs web search RAG and pick the right one for a given question
- Describe what `AIAgent` does that a normal task list does not: it picks tools and order at runtime
- Read flow 6 and trace the delegation: main agent calls research agent as a tool
- Make the cost call: free-tier Flash for simple work, stronger models only where reasoning pays off
- Say when to use an agent and when to stick with a fixed workflow

---
## 2. The big mental model

Everything in this module is one problem wearing different clothes: the model only knows what you put in front of it.

No context about Kestra's current plugins? It invents property names. No context about Kestra 1.1? It invents features. No tools to act with? It can only talk. Each technique in this module is a different way of fixing the context:

```text
Problem                              Fix
─────────────────────────────────────────────────────────
Assistant guesses plugin syntax      Copilot, grounded in live docs
Model guesses facts                  RAG, grounded in your documents
Fixed steps can't adapt              Agent, picks tools at runtime
One agent does everything badly      Multi-agent, split responsibilities
Nobody watches cost and secrets      Best practices, before prod
```

And the line that separates the two halves of the module:

```text
Copilot and RAG: you define the steps, AI fills in content
Agents:          you define the goal, AI picks the steps
```

That is the whole module in two lines. If a question ever confuses you, ask which side of that line it sits on.

A note on providers, since it comes up in every flow: the examples use Gemini (`gemini-2.5-flash` for chat, `gemini-embedding-001` for embeddings), flow 3 uses OpenAI (`gpt-5-mini`). Kestra's AI plugin supports the other majors too. Swapping providers means changing the `provider` block, nothing else.

---
## 3. Environment setup

You need Docker with Compose, three API keys, and about ten minutes. I will explain each piece as we go because the secrets mechanism trips people up.

### 3.1 Start Kestra

The module ships a `docker-compose.yml`. Two services: Postgres (stores flows, executions, queue state) and Kestra itself (`kestra/kestra:v1.3.21`, `server standalone` mode). The Kestra container mounts the Docker socket, which matters later: the research agent spins up an MCP filesystem server in Docker, and that only works because the socket is mounted.

```bash
cd 03-orchestration
docker compose up -d
```

Open http://localhost:8080. Default login is `admin@kestra.io` / `Admin1234!`. To stop everything later: `docker compose down`.

### 3.2 Get the API keys

You need up to three:

Gemini (required). Go to Google AI Studio, sign in, create a key. The free tier handles light use, but the agent flows burn through quota fast. If you see `429 Resource Exhausted`, wait a minute and retry before assuming something is broken.

OpenAI (required for flow 3 only). Create a key under API keys at platform.openai.com.

Tavily (required for web search in flows 3, 5, 6). Free tier gives 1,000 searches a month, plenty for this module.

### 3.3 Store keys as secrets

This is the part worth reading twice. Kestra reads secrets from environment variables that start with `SECRET_`, and the value must be base64-encoded. In the flow YAML you then call `secret()` with the name minus the prefix.

```bash
export GEMINI_API_KEY="your-gemini-api-key-here" # required
export SECRET_GEMINI_API_KEY=$(echo -n $GEMINI_API_KEY | base64) # required
export SECRET_OPENAI_API_KEY=$(echo -n "your-openai-api-key-here" | base64)   # required for flow 3
export SECRET_TAVILY_API_KEY=$(echo -n "your-tavily-api-key-here" | base64)   # optional
```

Then start or restart Kestra so it picks them up:

```bash
docker compose up -d
```

In flows:

```yaml
apiKey: "{{ secret('GEMINI_API_KEY') }}"
```

Note the missing `SECRET_` prefix. That is not a typo, it is how `secret()` works: you export `SECRET_GEMINI_API_KEY`, you reference `secret('GEMINI_API_KEY')`. Get this wrong and the flow fails with an unhelpful error, so double check it now.

Never commit keys to Git. This gets its own section later, but it bears repeating here because you are about to handle three of them.

### 3.4 Import the six flows

Either paste the YAML into the UI, or import through the API:

```bash
cd 03-orchestration

# Adjust username and password to match your Kestra setup
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/1_chat_without_rag.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/2_chat_with_rag.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/3_rag_with_websearch.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/4_simple_agent.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/5_web_research_agent.yaml
curl -X POST -u 'admin@kestra.io:Admin1234!' http://localhost:8080/api/v1/flows/import -F fileUpload=@flows/6_multi_agent_research.yaml
```

### 3.5 Run something

Go to the UI, open the `zoomcamp` namespace, execute `4_simple_agent`, leave the defaults, watch it run. Then do the same for 5 and 6 and read the logs. If 4 runs, your keys and setup are fine. If it fails, the problem is almost always the secrets step above.

---
## 4. The context problem

Before touching any Kestra feature, the module makes you feel the problem. Do this experiment, it takes two minutes.

Open ChatGPT in a private window (private so no old chat leaks in as accidental context) and enter:

```text
Create a Kestra flow that loads NYC taxi data from a CSV file to BigQuery. The flow should extract data, upload to GCS, and load to BigQuery.
```

You will get YAML back. It will look plausible. It will probably be wrong: renamed task types, properties that do not exist, features that never existed.

Why? A model trained months ago cannot know about last month's plugin rename. It does not know your infra, your versions, or your org's conventions. It fills the gaps with confident guesses. That is all a hallucination is: a gap filled confidently.

The takeaway the module keeps returning to: the model is only as good as what it can see. Copilot fixes this for flow authoring by putting current docs in front of the model. RAG fixes it for Q&A by putting your documents in front of the model. Agents fix it for tasks by giving the model tools to go look things up itself. Same problem, three fixes.

---
## 5. AI Copilot

Writing flows by hand is slow in a specific way. Not hard-slow: lookup-slow. Which plugin, which property name, which order, does this task even still exist. Autocomplete helps with one task at a time. Copilot flips it: describe the inputs and the goal, get a full flow, fix the last bit yourself.

It works because Copilot reads the plugin docs for your running Kestra version. That is the entire difference from the ChatGPT experiment in section 4. Same model class, different context.

### 5.1 Setup

Open source Kestra only supports Gemini for Copilot, so the Gemini key from section 3 is the requirement here. In the UI, create or open a flow and click the sparkle icon in the top-right of the editor.

### 5.2 Try it against ChatGPT

Use the exact same taxi-to-BigQuery prompt from section 4. This time the output should have real task types, real properties, YAML that actually runs. The comparison is the point: same prompt, grounded context, different result.

### 5.3 The 5% rule

Copilot gets the structure right. It does not know your environment. Expect to adjust secrets, variable names, error handling, retry counts, anything site-specific. I read this as: let it do the 95% that is generic, keep the 5% that is yours. Review every generated flow before running it somewhere that matters.

### 5.4 Iterative refinement

The conversation keeps state. Each follow-up edits the existing flow instead of starting over:

1. "Create a flow that downloads a CSV file and loads it to BigQuery" (basic flow)
2. "Add a task that checks data quality in BigQuery" (validation tasks appear)
3. "Schedule the flow to run daily at 9 AM UTC" (a `Schedule` trigger appears)
4. "Send a Slack notification if it fails" (a `SlackIncomingWebhook` task lands in an `errors` branch)

Then you finish by hand: the actual SQL, your channel, your retry count. You work with it instead of handing the whole thing off.

Typical uses: generating a flow from scratch ("sync data from Postgres to GCS"), adding a task ("add an If-task for conditional branching"), configuring a trigger ("add a webhook trigger"), adding error handling ("retry with exponential backoff").

### 5.5 Agent skills as an alternative

If you would rather stay in your editor, Kestra's agent-skills repo gives Claude or Cursor the same grounding Copilot has: current docs, valid properties, best practices. Same idea, different surface.

---
## 6. RAG in Kestra

Copilot fixed flow authoring. RAG fixes answers. Same context problem: ask about Kestra 1.1 with no documents attached and the model guesses. Attach the release notes and it quotes them.

If Modules 1 and 2 are fresh, this will feel familiar. If not: RAG means retrieve relevant text, paste it into the prompt, let the model answer from that instead of memory. The Kestra version just expresses both halves as flow tasks.

### 6.1 The two phases

RAG splits into ingest and query. In the demo flows they run back-to-back so you can see the whole thing. In production you would separate them: ingest on a schedule as docs change, query whenever someone asks.

```text
Ingest (once, or on a schedule)
  Fetch docs → create embeddings → store in KV Store
                          ↓
Query (every question)
  User question → find similar content → add to prompt → LLM answers
```

Ingest: fetch documents, turn text into vectors with an embedding model, store them in Kestra's KV Store. Query: embed the question, find the closest stored vectors, stuff the matching text into the prompt, generate.

One honest caveat from the lesson, worth keeping: the KV Store is fine for learning and small demos. It is not a vector database. Bigger document sets, latency requirements, anything production: use a real vector store (Module 2 covered those). Do not build a serious retrieval system on the KV Store.

### 6.2 Flow 1: without RAG

File: `flows/1_chat_without_rag.yaml`. One `ChatCompletion` task asks Gemini "which features were released in Kestra 1.1, list at least 5", then a `Log` task prints the answer plus a nudge to compare with flow 2.

```yaml
tasks:
  - id: chat_without_rag
    type: io.kestra.plugin.ai.completion.ChatCompletion
    provider:
      type: io.kestra.plugin.ai.provider.GoogleGemini
      modelName: gemini-2.5-flash
      apiKey: "{{ secret('GEMINI_API_KEY') }}"
    messages:
      - type: USER
        content: |
          Which features were released in Kestra 1.1?
          Please list at least 5 major features with brief descriptions.
```

Run it. Expect vague or wrong features, possibly real features from the wrong version. That is the baseline.

### 6.3 Flow 2: with RAG

File: `flows/2_chat_with_rag.yaml`. Three tasks. First ingests the actual 1.1 release post from GitHub into embeddings:

```yaml
  - id: ingest_release_notes
    type: io.kestra.plugin.ai.rag.IngestDocument
    provider:
      type: io.kestra.plugin.ai.provider.GoogleGemini
      modelName: gemini-embedding-001
      apiKey: "{{ secret('GEMINI_API_KEY') }}"
    embeddings:
      type: io.kestra.plugin.ai.embeddings.KestraKVStore
    drop: true
    fromExternalURLs:
      - https://raw.githubusercontent.com/kestra-io/docs/refs/heads/main/src/contents/blogs/release-1-1/index.md
```

Note `drop: true`: it clears the store first, so re-runs do not stack duplicates. Then the chat task, which wires a chat provider and an embedding provider separately plus a system message that says to use the docs and admit when the answer is not there:

```yaml
  - id: chat_with_rag
    type: io.kestra.plugin.ai.rag.ChatCompletion
    chatProvider:
      type: io.kestra.plugin.ai.provider.GoogleGemini
      modelName: gemini-2.5-flash
      apiKey: "{{ secret('GEMINI_API_KEY') }}"
    embeddingProvider:
      type: io.kestra.plugin.ai.provider.GoogleGemini
      modelName: gemini-embedding-001
      apiKey: "{{ secret('GEMINI_API_KEY') }}"
    embeddings:
      type: io.kestra.plugin.ai.embeddings.KestraKVStore
    systemMessage: |
      You are a helpful assistant that answers questions about Kestra.
      Use the provided documentation to give accurate, specific answers.
      If you don't find the information in the context, say so.
    prompt: |
      Which features were released in Kestra 1.1?
      Please list at least 5 major features with brief descriptions.
```

Run it, compare with flow 1. Same question, specific answers with real feature names. The only thing that changed is the context.

### 6.4 Flow 3: web search instead of ingestion

File: `flows/3_rag_with_websearch.yaml`. No ingest task at all. The `TavilyWebSearch` retriever fetches live results at query time and injects them as context:

```yaml
tasks:
  - id: chat_with_rag_and_websearch_content_retriever
    type: io.kestra.plugin.ai.rag.ChatCompletion
    chatProvider:
      type: io.kestra.plugin.ai.provider.OpenAI
      apiKey: "{{ secret('OPENAI_API_KEY') }}"
      modelName: gpt-5-mini
    contentRetrievers:
      - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
        apiKey: "{{ secret('TAVILY_API_KEY') }}"
    systemMessage: You are a helpful assistant that can answer questions about Kestra.
    prompt: What is the latest release of Kestra?
```

Two things to notice. It uses OpenAI, so you need that key configured (to stay on Gemini, swap the provider block for `GoogleGemini`, the rest stays). And there is a tradeoff the lesson states plainly: no ingestion step, but you inherit whatever quality the search engine gives you. Test the retrieved context, do not assume it.

Static RAG vs web search RAG, side by side:

| | Static RAG | Web search RAG |
|---|---|---|
| Data source | Documents you ingested | Live web results |
| Best for | Internal docs, policies, fixed knowledge bases | Time-sensitive or fast-changing info |
| Ingestion step | Required | Not required |
| Example question | "What does our refund policy say?" | "What is the latest release of Kestra?" |

Control the source material, use static RAG. The answer moves faster than you can re-ingest, use web search.

### 6.5 RAG habits

Re-ingest on a cadence so the store matches reality. Chunk big docs into sensible sections before ingesting. Actually check what gets retrieved for your queries instead of trusting it. Pick the retriever per question type, not per habit.

---
## 7. AI agents

In Module 1 you wrote the agent loop yourself: call the model, run whatever tools it asked for, send results back, stop when it answers without tool calls. Kestra's `AIAgent` is that loop as a task type. You give it a goal, tools, and a system message. It drives.

The contrast with everything before it:

```yaml
# Traditional: you fix the steps
tasks:
  - id: step1
    type: Task1
  - id: step2
    type: Task2
  - id: step3
    type: Task3
```

```yaml
# Agent: you fix the goal
tasks:
  - id: agent
    type: io.kestra.plugin.ai.agent.AIAgent
    prompt: "Research data engineering trends and create a report"
    tools:
      - WebSearch
      - TaskExecution
```

Reach for an agent when the steps are not known upfront, decisions depend on live information, or the plan needs to change mid-run. Stick with fixed tasks when the process must be repeatable and auditable, compliance cares about the exact sequence, or you need cost and latency nailed down. Agents are flexible and expensive; fixed workflows are rigid and cheap. Pick per workflow, not per preference.

### 7.1 Anatomy of an agent task

```yaml
id: example_agent
namespace: zoomcamp

tasks:
  - id: agent
    type: io.kestra.plugin.ai.agent.AIAgent

    # who the agent is
    systemMessage: |
      You are a data analyst. Analyze data and provide insights.

    # what it should do
    prompt: "What are the top 3 trends in this data?"

    # which model answers
    provider:
      type: io.kestra.plugin.ai.provider.GoogleGemini
      modelName: gemini-2.5-flash
      apiKey: "{{ secret('GEMINI_API_KEY') }}"

    # what it is allowed to touch
    tools:
      - type: io.kestra.plugin.ai.tool.TavilyWebSearch
        apiKey: "{{ secret('TAVILY_API_KEY') }}"

    # memory across executions
    memory:
      type: io.kestra.plugin.ai.memory.KestraKVStore
      memoryId: analyst_001
```

Five pieces: system message sets the role, prompt sets the job, provider sets the model, tools set what it can do, memory lets it remember across runs. The provider block takes any major vendor, same shape.

### 7.2 Flow 4: the simple agent

File: `flows/4_simple_agent.yaml`. Summarizes a text with length and language picked at runtime through inputs (`short/medium/long`, seven languages). Two agent tasks chained: the first writes the summary, the second compresses it to one English sentence by reading `outputs.multilingual_agent.textOutput`. A `Log` task prints token counts for both.

The boring-looking detail that actually matters: `pluginDefaults` at the bottom sets the Gemini provider once for every `AIAgent` in the flow, so neither task repeats it:

```yaml
pluginDefaults:
  - type: io.kestra.plugin.ai.agent.AIAgent
    values:
      provider:
        type: io.kestra.plugin.ai.provider.GoogleGemini
        modelName: gemini-2.5-flash
        apiKey: "{{ secret('GEMINI_API_KEY') }}"
```

Run it twice with different lengths and watch the token counts move. That is your cheapest cost lesson in the module.

### 7.3 Flow 5: the agent that does research

File: `flows/5_web_research_agent.yaml`. This is where autonomy stops being a slogan. The system message tells it the process (search, evaluate, search again if thin, synthesize into a report with summary, findings, analysis, sources, save to `research_report.md`), and then it decides: how many searches, which queries, when it is done. Nobody wrote those steps in YAML.

Two mechanisms worth naming. `contentRetrievers` gives it Tavily web search (`maxResults: 10`). `tools` gives it a filesystem MCP server via `DockerMcpClient` (image `mcp/filesystem`), which is how the report actually gets written to disk. `outputFiles` declares `research_report.md` so Kestra surfaces it, and the final `Log` task prints where it landed plus total tokens.

Run it with the default topic. Read the logs. Count the searches it chose. That count is the agent making decisions you did not script.

### 7.4 Tools you can hand an agent

| Tool | Purpose | Example use |
|------|---------|-------------|
| `TavilyWebSearch` | Search the web for current information | Market research, news monitoring |
| `GoogleCustomWebSearch` | Search with Google Custom Search API | Google search |
| `CodeExecution` | Run code safely via Judge0 | Math calculations, data validation |
| `KestraTask` | Execute any Kestra task | Run tasks based on 1000+ Kestra plugins |
| `KestraFlow` | Trigger other Kestra flows | Call other flows for modularity |
| `StreamableHttpMcpClient` | Use MCP servers via HTTP/SSE | Connect to remote MCP servers |
| `DockerMcpClient` | Use MCP servers in Docker | MCP servers spun up on-demand via Docker |
| `StdioMcpClient` | Use MCP servers via stdio | Integration with external systems |
| `AIAgent` | Use another agent as a tool | Multi-agent systems, specialized sub-agents |

The last row is the bridge to the next section.

### 7.5 Seeing what the agent did

Kestra records token usage, tool calls, request/response logs, outputs, and timing per execution. When something behaves oddly, turn on the detailed logging:

```yaml
tasks:
  - id: research_agent
    type: io.kestra.plugin.ai.agent.AIAgent
    description: Autonomous research agent with web search capabilities
    provider:
      type: io.kestra.plugin.ai.provider.GoogleGemini
      apiKey: "{{ secret('GEMINI_API_KEY') }}"
      modelName: gemini-2.5-flash
    configuration:
      logRequests: true
      logResponses: true
```

`logRequests` and `logResponses` show the raw conversation. Start there before assuming the tools are broken; most agent failures are prompt failures.

---
## 8. Multi-agent systems

One agent doing everything gets mushy: vague role, vague output, hard to debug. Split the job. Each agent gets one responsibility, and an agent can call another agent as a tool. That last part is the whole pattern.

### 8.1 Flow 6: company research with two agents

File: `flows/6_multi_agent_research.yaml`. Input is a company name (default `kestra.io`). Two roles:

| Agent | Specialization | Tools | Responsibility |
|-------|---------------|-------|----------------|
| Research agent | Web research and data gathering | Tavily web search | Find factual, current information |
| Main analyst agent | Analysis and synthesis | Research agent (used as a tool) | Create structured reports |

The run: main agent gets "research this company", calls the research agent tool, research agent searches Tavily and returns findings, main agent shapes them into strict JSON (`company`, `summary`, `recent_news`, `competitors`, no markdown, no fences, no commentary). A `Log` task then renders the JSON with Pebble templating (`json(...)`, for loops over news and competitors) plus the main agent's token count.

The key lines are the tool declaration inside the main agent:

```yaml
    tools:
      - type: io.kestra.plugin.ai.tool.AIAgent
        description: Web research and data gathering
        ...
        contentRetrievers:
          - type: io.kestra.plugin.ai.retriever.TavilyWebSearch
            apiKey: "{{ secret('TAVILY_API_KEY') }}"
```

The main agent treats the research agent like any other tool call. It asks, it gets text back, it moves on. And `pluginDefaults` again covers the Gemini provider for both levels, so the nested agent config stays short.

Why this shape wins: each agent stays in its lane, and when output looks wrong you know which one to blame. The research agent returning junk is a retrieval problem. The main agent mangling good findings is a prompt problem. Different fixes.

### 8.2 Habits for multi-agent flows

Give each agent one job and write it down in the task description, or the next person will not know why there are two agents. Watch tokens, since every agent is more LLM calls on your bill. And document what each agent does in the flow, because a bare `AIAgent` task with a vague prompt is unreadable a month later.

---
## 9. When to use what

| Scenario | Use this | Why |
|----------|----------|-----|
| Creating/editing flows | AI Copilot | Fastest way to generate YAML flow code |
| Answering questions about your data | RAG | Grounds responses in real data |
| Fixed, repeatable ETL pipelines | Traditional workflows | Deterministic, predictable, compliant |
| Research and analysis tasks | AI Agents | Can adapt to findings and make decisions |
| Complex, multi-step objectives | Multi-agent systems | Specialized agents working together |

If you only remember one row: regulated, repeatable work stays in fixed workflows. Agents are for jobs where the path depends on what you find along the way.

---
## 10. Cost, security, and production readiness

### 10.1 Cost

Every AI feature is metered in tokens. The module's pricing snapshot for Gemini:

| Model | Tier | Input | Output |
|-------|------|-------|--------|
| Gemini 2.5 Flash | Free | $0.00 | $0.00 |
| Gemini 2.5 Flash | Batch / Flex | $0.15 | $1.25 |
| Gemini 3.5 Flash | Free | $0.00 | $0.00 |
| Gemini 3.5 Flash | Standard | $1.50 | $9.00 |
| Gemini 3.5 Flash | Batch / Flex | $0.75 | $4.50 |
| Gemini 3.5 Flash | Priority | $2.70 | $16.20 |

(Prices per 1M tokens; check the pricing page before budgeting, these move.)

Practical version: learn on the free tier, default to 2.5 Flash, reach for the stronger model only when the task needs real reasoning. Cap `maxOutputTokens` so a chatty agent cannot surprise you. Read token usage in the execution outputs; flow 4 even logs it for you. And when determinism is what you need, a plain workflow costs nothing in tokens at all.

### 10.2 Security

Keys stay out of Git and out of YAML literals. Always:

```yaml
# Wrong
apiKey: "sk-abc123def456"

# Correct
apiKey: "{{ secret('GEMINI_API_KEY') }}"
```

Base64-export with the `SECRET_` prefix before starting Kestra (section 3.3), rotate regularly, watch usage for spikes. The Kestra secrets docs cover the rest.

### 10.3 Observability and debugging

Log requests and responses while troubleshooting (section 7.5 shows the two flags). Per execution, watch tokens, which tools fired and what they returned, wall time, and whether the output was actually any good. When debugging: simplify the prompt first, read the LLM's reasoning in the logs, then check tool outputs. In that order. Most "agent is broken" reports I have seen are prompts that never said what good looks like.

### 10.4 Before production

Run each flow repeatedly with varied inputs and check the outputs hold up. Add retries and failure alerts, because API calls fail at the worst time. Cap output tokens. Write down what the agent is supposed to do in the flow and task descriptions, so the on-call person is not reverse-engineering your prompt at 2am.

---
## 11. Things to try

These map to the homework and double as experiments. Do them in order; each one leans on the previous.

1. Same prompt, two surfaces. Run the taxi-to-BigQuery prompt in bare ChatGPT, then in Copilot. The difference you see is the value of grounded docs. (Homework Q1: the answer is the plugin documentation.)
2. Flow 1 vs flow 2. Ask both about Kestra 1.1. Note what changes when the release notes are in the prompt. (Homework Q2: specific, grounded features win.)
3. Token knobs. Run flow 4 with `short`, then `long`. Compare the logged output tokens. Longer summaries cost roughly 2-4x on the output side. (Homework Q3.)
4. Watch autonomy. Run flow 5 and count the searches from the logs. Nobody scripted that number. (Homework Q4: the agent decides.)
5. Trace delegation. Run flow 6 on `kestra.io`, then read the YAML and point at the exact block where the main agent calls the research agent. (Homework Q5: the research agent is a tool for gathering.)
6. The compliance call. Ask yourself which of the six flows you would put in front of an auditor. (Homework Q6: fixed workflows.)
7. Provider swap. Change one flow's provider block to another vendor and compare output. The YAML around it should not need to change.
8. Break RAG on purpose. Re-run flow 2 with `drop: false` twice, or point the ingest URL at something unrelated, and watch retrieval quality fall apart. Cheap way to learn why re-ingest cadence and chunking matter.

---
## 12. Architecture review

The full picture, compressed:

```text
Authoring (design time)
  Your sentence → Copilot (+ live plugin docs) → flow YAML → your 5% edits
        ↓
Grounding (ingest time, scheduled)
  Docs → embeddings (gemini-embedding-001) → KV Store
  (serious scale: real vector store instead)
        ↓
Answering (query time, on demand)
  Question → retrieve (KV Store or Tavily) → prompt + context → answer
        ↓
Acting (runtime, agent decides)
  Goal → AIAgent loop (tools: search, code, tasks, flows, MCP, sub-agents) → output + token bill
        ↓
Collaborating (runtime, split jobs)
  Main agent → research agent as tool → structured JSON → logged report
        ↓
Running it for real
  Secrets via SECRET_ vars, logs on, tokens capped, retries set, docs written
```

Three sentences if anyone asks what the module was about: generic assistants fail because they lack context. Copilot, RAG, and agents are three ways of supplying it: docs for authoring, documents for answering, tools for doing. Fixed workflows stay for anything that must be repeatable; agents earn their keep where the path is unknown.

---
## 13. What you should now understand

- A model without context guesses. Outdated plugins, invented features, vague answers all come from the same gap.
- Copilot beats bare ChatGPT on Kestra flows for exactly one reason: it sees current plugin documentation.
- RAG has two phases with different schedules. Ingest runs once or on a cadence; query runs per question. The demo runs them together for convenience, production should not.
- The KV Store is a teaching stand-in, not a vector database. Know where that boundary is.
- Static RAG fits knowledge you control; web search RAG fits answers that move faster than your ingest cycle.
- An agent is a loop (prompt, act, observe, repeat) that you configure with goal, tools, and system message instead of steps.
- `AIAgent` as a tool is the multi-agent pattern. One agent delegates gathering, the other owns synthesis.
- Tokens are money. Model choice, output caps, and reading the usage logs are cost controls, not trivia.
- Secrets go through `SECRET_`-prefixed base64 vars and `secret()` without the prefix. Keys never touch YAML or Git.
- Regulated, repeatable work stays in fixed workflows, because an auditor needs the exact sequence, not a story about what the agent felt like doing.

---
## 14. Self-test

Try these from memory. Answers follow.

1. Why does bare ChatGPT produce broken Kestra flows for the taxi-to-BigQuery prompt?
2. What does Copilot have that ChatGPT lacks in that comparison?
3. What is the 5% rule?
4. What are the two RAG phases, and when does each run in production?
5. Why is the KV Store not a production vector store?
6. When do you pick web search RAG over static RAG?
7. What does `drop: true` do in the ingest task, and why does it matter on re-runs?
8. What is the difference between a fixed task list and an `AIAgent` task?
9. Name the five parts of an agent task and what each one controls.
10. In flow 5, who decides how many web searches happen?
11. In flow 6, what is the research agent to the main agent?
12. Why does flow 4 use `pluginDefaults`, and what would break without it?
13. How do you reference a secret in YAML if you exported `SECRET_GEMINI_API_KEY`?
14. Which model do you default to for cost, and when do you step up?
15. Which approach fits a financial-reporting pipeline under strict compliance, and why?

---

Answers:

1. Its training data predates current plugin versions, renames, and APIs, so it fills gaps with invented syntax and properties.
2. Grounding in the live plugin docs and valid properties for your running Kestra version.
3. Copilot writes the generic 95%; you finish the environment-specific 5% (secrets, variables, error handling, config tweaks).
4. Ingest (fetch, embed, store) runs once or on a schedule; query (retrieve, augment, generate) runs per question.
5. It is convenient storage for demos and small sets, with none of the scale, latency, or retrieval guarantees of a real vector store.
6. When the answer changes faster than you can re-ingest: latest releases, news, anything live.
7. It clears the store before ingesting, so repeated runs do not pile up duplicate vectors.
8. A task list fixes the steps; an agent fixes the goal and picks tools and order at runtime.
9. System message (role), prompt (job), provider (model), tools (what it can touch), memory (context across runs).
10. The agent itself, driven by its prompt and system message. Nothing in the YAML scripts the count.
11. A tool. The main agent invokes it for web data the way it would call search or a database.
12. It sets the Gemini provider once for all `AIAgent` tasks. Without it each task would need its own provider block (and would fail without one).
13. `{{ secret('GEMINI_API_KEY') }}`. The `SECRET_` prefix is for the environment variable only.
14. Gemini 2.5 Flash (free tier for standard inference). Step up to the stronger Flash when the task needs heavier reasoning.
15. Traditional fixed workflows. Deterministic, repeatable, auditable; agents trade that away for flexibility.

---
## 15. Where to go from here

Swap providers in a flow and compare. Take a pipeline you already run and find the step where the decision depends on live data; that is your first agent candidate. Browse the Blueprints library for patterns worth stealing. If you get stuck, the Kestra Slack community is active.

Docs worth bookmarking: AI tools overview, AI Copilot, AI agents, RAG workflows, the AI plugin reference, the agent task reference, the RAG tasks reference. External: Gemini API docs, AI Studio, Tavily docs, the Kestra GitHub repo. Video playlist for the module is linked in the module README.

And the homework (`cohorts/2026/03-orchestration/homework.md`) covers Q1-Q6 from section 11 with a submission form. Do it while the flows are still running locally; half the questions need execution outputs.
