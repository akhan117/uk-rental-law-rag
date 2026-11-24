# UK Residential Rental Law RAG — Demo

[![CI](https://github.com/akhan117/uk-rental-law-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/akhan117/uk-rental-law-rag/actions/workflows/ci.yml)

A source-grounded legal information assistant for **England residential landlord-tenant law**,
built to a separate design spec (cited in code comments as `Docs.pdf`, not included in this repo): retrieve legislation + case law, generate
a cited answer, and abstain when the evidence doesn't support an answer. Runs fully locally on
Qwen3 via Ollama, or against a free hosted LLM API; ships as one Docker image, and is tested
and eval-gated in CI.

**This is a tech demo, not a production legal tool.** Legal information, not legal advice. See
[Known limitations](#known-limitations) before trusting any specific answer.

**Jump to:** [Architecture](#architecture) · [Deployment and operations](#deployment-and-operations)
· [Agent mode](#agent-mode-tool-calling) · [MCP server](#mcp-server) · [Eval results](#eval-results)
· [Findings](#finding-1-section-21-has-been-repealed-in-the-live-text-and-the-system-initially-hallucinated-around-the-gap)

## Running this if you're not a developer

You don't need to install anything first, and you don't need admin rights.

1. **Download it.** On this page, click the green **Code** button → **Download ZIP**, then
   unzip it (right-click → *Extract All* on Windows; double-click on a Mac).
2. **Double-click `start.bat`** (Windows) or **`start.command`** (Mac).
   - Windows may say *"Windows protected your PC"*: click **More info → Run anyway**.
   - A Mac may say it can't check the developer: **right-click** `start.command` → **Open** →
     **Open**. This is only needed the first time.
3. **The first time only**, it sets itself up (about 5-15 minutes) and asks one question:
   - **Fast**: answers in seconds on any computer. It opens a page where you get a free key
     from Groq (sign in with Google, click *Create API Key*, copy, paste). Questions go to
     Groq's servers.
   - **Private**: everything stays on your computer. It downloads a 2.5GB AI model, and
     without a graphics card each answer takes up to a minute. The setup tells you which suits
     your computer.

After that, double-clicking start opens the app in about 15 seconds. To switch between Fast and
Private later, double-click `setup.bat` / `setup.command`.

The app keeps running quietly in the background after you close its window, until you
restart the computer. Starting it again always starts a fresh copy, so an updated download or a
changed setting takes effect straight away.

> **Downloaded from GitHub?** The repo includes the parsed legislation, and setup builds the
> search index from it. Case law is not included (see the licensing note under
> [Known limitations](#known-limitations)). To add it, run step 1 of the
> [developer path](#running-it-manually-developer-path).

## Architecture

```
legislation.gov.uk + Find Case Law  (behind an AWS WAF JS challenge — fetched via headless
        │                            Chromium, see ingestion/fetch.py)
        ▼
ingestion/  →  structure-aware chunking (Act→Part→Section→Subsection,
                judgment→coherent paragraph groups)  →  data/processed/*.jsonl
        ▼
build_index.py  →  chunk_quality filter (drops heading-only/empty chunks)
                →  BAAI/bge-small-en-v1.5 embeddings (local, CPU)  →  Chroma (data/chroma/)
        ▼
backend/  (FastAPI)
   query_understanding.py → rewrite the question into statutory vocabulary + sub-queries
   scope.py        → keyword + retrieval-confidence gate (abstain before generating)
   retrieval.py    → hybrid dense + BM25, one search per sub-query, interleaved
   agent.py        → optional: the model plans retrieval itself with tools (tools.py)
   generate.py     → citation-enforcing prompt → LLM → schema-constrained JSON
   llm.py          → one interface over Ollama (native) and any OpenAI-compatible API
                     (Groq / Gemini / OpenRouter): structured-output ladder, retries, usage
   verify.py       → Act-qualified citation check against what was actually retrieved
   telemetry.py    → per-request trace → JSON log line + /metrics (Prometheus) + /metrics/summary
   guard.py        → public-demo protection: per-IP rate limit, daily cap, answer cache
   mcp_server.py   → the corpus tools + full pipeline exposed over MCP
        ▼
frontend/  → static chat UI (vanilla JS), renders short answer → applicable law →
             how it applies → uncertainties → sources (with links + expandable raw text),
             agent steps live, and latency/tokens per answer
```

```
git push ──► CI ─┬─ ruff + pytest (100 tests, no model or network)
                 ├─ build index from committed corpus → retrieval eval (no LLM, deterministic)
                 │    gated against eval/baseline_retrieval.json, per question
                 └─ docker build → smoke test the container → push to ghcr.io (main)
weekly / manual: eval.yml → full 62-question eval on the hosted LLM, gated with tolerance
```

## Curated seed corpus

7 Acts, 6 Court of Appeal/Supreme Court cases, narrow topic focus (deposits, notices/possession,
repairs, fees). **Not comprehensive** — see limitations. 417 chunks parsed (362 legislation + 55
case); **408 indexed** — 9 are heading-only with no operative text and are excluded as actively
harmful to retrieval (Finding 5).

Legislation: Housing Act 1988 (core provisions incl. s.7 possession orders + all 39 Schedule 2
grounds), Protection from Eviction Act 1977 ss.1-3 + s.5 (notices to quit), Landlord and Tenant
Act 1985 s.11, Homes (Fitness for Human Habitation) Act 2018, Housing Act 2004 ss.212-215 +
Schedule 10, Tenant Fees Act 2019, **Renters' Rights Act 2025** (the reform that abolished
assured shortholds and s.21 — ss.1-5, 20-21, 58-61), and one statutory instrument — the Housing
(Tenancy Deposits) (Prescribed Information) Order 2007 — because s.213(6) requires information "in
the prescribed form" without saying what that form is (Finding 7).

Cases: Superstrike v Rodrigues [2013], Charalambous v Ng [2014], Johnson v Old [2013],
Taylor v Spencer [2013], Trecarrell House v Rouncefield [2020], Edwards v Kumarasamy [2016].

The pipeline is generic (topic/metadata-driven), so widening the corpus later is "run ingestion
on more Acts/cases," not a re-architecture.

## Running it manually (developer path)

`setup.bat` / `start.bat` above do all of this; this section is the equivalent by hand.

**Prerequisites:** Python 3.10+, [Ollama](https://ollama.com) with `qwen3:4b` pulled
(`ollama pull qwen3:4b`), and `pip install -r requirements.txt`.

The ingest step below additionally needs a headless browser (it isn't in `requirements.txt`
because only ingestion uses it):
`pip install playwright && playwright install chromium`.

```bash
# 1. Ingest - optional after cloning: the parsed legislation is committed, so
#    `python ingestion/build_index.py` alone gives a working legislation-only
#    index. The full fetch below adds case law and refreshes the corpus.
cd ingestion
python fetch_legislation.py   # fetches curated Acts (headless-browser, ~2-3 min)
python fetch_cases.py         # fetches curated cases
python parse_legislation.py   # -> data/processed/legislation_chunks.jsonl
python parse_cases.py         # -> data/processed/case_chunks.jsonl
python build_index.py         # embeds + indexes into Chroma

# 2. Run
ollama serve                  # if not already running
cd ../backend
uvicorn main:app --reload     # serves API + frontend at http://localhost:8000

# 3. Evaluate
cd ../eval
python run_eval.py            # scores against gold_questions.jsonl
python run_eval.py --retrieval-only   # recall + gates only: no LLM, deterministic, ~1 min
python run_eval.py --mode agent       # same benchmark, agent-planned retrieval
python compare_models.py      # side-by-side table of every recorded run
cd ..
pip install -r requirements-dev.txt && pytest   # 100 unit/API tests, no model needed
```

## Deployment and operations

### Choosing the LLM backend

The model is configuration, not code (`backend/llm.py`). Everything else (retrieval, gates,
the citation-constrained answer schema, verification) is identical across backends.

| `LLM_PROVIDER` | Default model | Key variable | Notes |
|---|---|---|---|
| `ollama` (default) | `qwen3:4b` | — | Fully local. Native API, so `think: false`, `num_predict` and `repeat_penalty` apply exactly as measured below. `RAG_MODEL` still works. |
| `groq` | `openai/gpt-oss-20b` | `GROQ_API_KEY` | Free tier, email signup. Strict `json_schema` including the citation `enum`; tool calling. The Docker image's default. |
| `gemini` | `gemini-2.5-flash` | `GEMINI_API_KEY` | Free tier via Google AI Studio. OpenAI-compatible endpoint. |
| `openrouter` | `openai/gpt-oss-20b:free` | `OPENROUTER_API_KEY` | `:free` models; low daily request cap without credits. |
| `openai` | set `LLM_MODEL` | `LLM_API_KEY` | Any OpenAI-compatible URL (`LLM_BASE_URL`), including Ollama's own `/v1`. |

`LLM_MODEL` overrides the model id. All variables are listed in [`.env.example`](.env.example).

**Structured output degrades, it doesn't break.** Providers differ most in how much of JSON
Schema they enforce, and the schema here is load-bearing (Findings 3 and 7). So `llm.py`
walks a ladder: strict `json_schema` → best-effort `json_schema` → `json_object` → schema in
the prompt. Each provider's first rejected rung is remembered for the life of the process. The
one guarantee that used to live only in the schema, that `citations` can hold only retrieved
authorities, is now also enforced in code (`generate.py`), so it holds on every rung.
Free-tier rate limits (429 + `Retry-After`) are waited out, not failed on.

### Docker

```bash
docker build -t uk-rental-law-rag .
docker run -p 7860:7860 -e GROQ_API_KEY=... uk-rental-law-rag     # http://localhost:7860

# or with compose: hosted LLM, or app + Ollama fully local
GROQ_API_KEY=... docker compose up app
LLM_PROVIDER=ollama docker compose --profile local up
```

One image: FastAPI, the bge embedding model and a pre-built Chroma index are baked in at build
time, so a container start downloads nothing and builds nothing. It installs CPU-only torch, not
the default CUDA wheel, which would add about 2.5GB of libraries a CPU host never loads. The
build context is an allowlist (`.dockerignore`), so local-only material can't get into the
image. The image ships the **legislation-only** corpus. Case law stays local because of the
licence question under [Known limitations](#known-limitations). To serve your local full
corpus from the container, mount it: `-v ./data:/app/data`.

### CI/CD

Two workflows, both free on a public repo:

- **`ci.yml`, every push and PR.** `ruff`; `pytest` (100 tests, all offline: the LLM is faked
  at the HTTP layer with `httpx.MockTransport`, so the real provider adapters are what's
  tested); then it builds the index from the committed corpus and runs the **retrieval eval
  with no LLM at all**. That eval is deterministic (README Finding 5: zero in-scope questions
  need the LLM rewrite), so it is gated strictly: thresholds in `eval/thresholds.json`, plus a
  per-question check against `eval/baseline_retrieval.json`. A question that retrieved its
  provision in the baseline and no longer does fails the build by name. After that it builds
  the Docker image, smoke-tests the running container (health, both abstention gates,
  `/metrics`) and pushes it to `ghcr.io` from `main`.
- **`eval.yml`, weekly and on demand.** The full 62-question eval against the hosted model, in
  `rag` or `agent` mode. Generation metrics carry sampling noise (see the variance note
  below), so these are gated with tolerance, as floors that catch a regression rather than
  targets that flap. It skips cleanly if no API key secret is set.

The committed corpus is `data/processed/legislation_chunks.jsonl` (Crown copyright, Open
Government Licence v3.0, which permits republication). Without it, CI would need the
headless-browser fetch described in Architecture.

### Hosting it publicly

The image runs on any container host. If it is exposed to the internet, anyone can spend its
free-tier LLM quota, so the image enables the guards in `backend/guard.py` by default: a
per-visitor rate limit (6 questions a minute), a cap on LLM-backed questions per day (300), and
an answer cache so repeated example questions cost no tokens. Set `RATE_LIMIT_PER_MINUTE=0` and
`DAILY_QUERY_LIMIT=0` to turn them off (`docker-compose.yml` does, for local use).

### Observability

Every request produces one trace (`backend/telemetry.py`) with per-stage latency (scope gate,
retrieval, rewrite, agent planning, generation, verify), LLM calls and tokens, tool calls,
which gate fired, and whether citation verification flagged the answer. Each trace is:

- **logged** as a single JSON line on stdout (visible in `docker logs` or any host's log viewer);
- **returned** to the client as `meta` on every answer (the UI shows it under each answer);
- **aggregated** in a rolling window served at `/metrics` (Prometheus format: request counts
  by mode and outcome, a token counter, a latency histogram, and the citation-flag ratio) and
  `/metrics/summary` (JSON: p50/p95/p99 latency overall and per stage, average tokens and LLM
  calls per answer, estimated cost per answer at a configurable list price, cache hit rate).

The citation-flag ratio is the online version of the eval's hallucinated-authority rate: the
same `verify.py` check, measured on live traffic instead of on the gold set.

`eval/benchmark.py` load-tests any deployment over the same streaming endpoint the browser
uses. It reports p50/p95 latency, time to first sources, tokens and tool calls per answer,
and throughput:

```bash
python eval/benchmark.py --url http://127.0.0.1:8000 --n 30
python eval/benchmark.py --url http://127.0.0.1:7860 --n 20 --mode agent   # e.g. the container
```

**Measured serving numbers, local backend** (Qwen3-4B on Ollama, RTX 3070 Laptop GPU, full
corpus; `benchmark.py --n 20` over HTTP, cache cold, cross-checked against `/metrics/summary`):

| | |
|---|---|
| end-to-end latency p50 / p95 | 9.8s / 13.8s |
| time to first sources p50 / p95 | 0.04s / 0.05s |
| retrieval p95 (hybrid dense + BM25) | 0.046s |
| generation p95 | 13.8s |
| tokens per answer | ~1,680 in / ~340 out, 1.0 LLM calls |
| answers flagged by citation verification | 10% (2 of 20) |

Generation is effectively all of the latency. The sources event arrives in about 40ms, so the
UI shows real evidence immediately while the answer is still being written.

## Agent mode (tool calling)

`mode: "agent"` (the **Agent** toggle in the UI) hands retrieval *planning* to the model. It
gets three tools over the corpus (`backend/tools.py`) and up to four rounds to use them:

| Tool | What it does that single-shot retrieval can't |
|---|---|
| `search_law(query, source_type?, act?)` | The model picks the wording and the filters, and can search again after reading the results. |
| `get_provision(act, provision)` | Exact lookup by Act and section or Ground (`"HA 1988"`, `"s.8(4AA)"`, `"Ground 8"`). Similarity ranking can lose a named provision to a neighbour; a lookup can't. It also reports **repealed** provisions from the manifest, which retrieval can't do because their text is gone (Finding 1). |
| `list_sources()` | What the corpus actually holds, so "not covered" can come from knowledge rather than a miss. |

This targets the two measured weak spots of single-shot RAG: multi-hop questions, where no one
chunk states the combined rule (Known limitations), and questions that name a provision which
similarity search then fails to surface.

**What the agent is not allowed to do is write the answer.** The evidence it gathers goes
through exactly the same path as RAG mode: the citation `enum` built from the evidence, the
schema-constrained answer, and post-generation verification. The evidence cap is the same k=6.
The scope gate still applies, though an exact lookup satisfies it by itself, since the agent
named the provision. If the agent finds nothing, or makes no tool calls, the request falls back
to ordinary retrieval, so agent mode is never worse than RAG mode. Because only the retrieval
strategy differs, `run_eval.py --mode agent` measures that one variable on the same benchmark.

The loop is hand-rolled (`backend/agent.py`, about 150 lines) rather than LangGraph. It is one
bounded loop, and having it written out keeps every step in the trace and debuggable.

**Measured: the local 4B model can't drive it.** With tools attached, Qwen3-4B on Ollama 0.40
writes about 2,000 tokens of reasoning before every tool call, whether `think` is false, true,
or `/no_think` is in the prompt. That is about 60s per planning round on an RTX 3070 Laptop
GPU. With the full research prompt it made no tool call at all within 3,500 tokens. Locally,
agent mode therefore falls back to RAG (the UI says so); `AGENT_PLAN_MAX_TOKENS` raises the
budget for anyone who wants to wait. Agent mode is meant for the hosted backend, where
gpt-oss-20b at `reasoning_effort=low` plans in a few hundred tokens.

## MCP server

`backend/mcp_server.py` exposes the same corpus tools, plus `answer_question` (the full
grounded pipeline using the configured LLM), to any MCP client such as Claude Desktop or Claude
Code. The tools behave exactly as they do for the in-app agent, including exact lookup and
repeal awareness.

```bash
pip install -r requirements-mcp.txt
python backend/mcp_server.py          # stdio transport
```

Client config, e.g. Claude Desktop's `claude_desktop_config.json` (use absolute paths):

```json
{
  "mcpServers": {
    "uk-rental-law": {
      "command": "/path/to/repo/venv/bin/python",
      "args": ["/path/to/repo/backend/mcp_server.py"],
      "env": {"LLM_PROVIDER": "groq", "GROQ_API_KEY": "..."}
    }
  }
}
```

It uses `mcp` 1.x on purpose: 2.x needs newer pydantic and Starlette than the pinned FastAPI
supports.

## Eval results

Run against `eval/gold_questions.jsonl` (30 in-scope, 1 stale-law probe, 14 out-of-scope,
12 must-answer, 5 adversarial) with Qwen3-4B. Full per-question results in `eval/eval_results.json`.

Every generation-dependent figure below is the range over **three repeat runs**, not a single run -
see the variance note for why that matters here.

| Metric | Result |
|---|---|
| In-scope retrieval recall@6 | 25/30 (83%) — identical across runs |
| In-scope citation correctness | 24/30 (80%) — three consecutive runs |
| In-scope answer correctness | 19/22 (86%) — 8 of 30 have no answer key |
| Answers containing a hallucinated authority | 2/30 (7%) |
| Out-of-scope correct abstention | 14/14 (100%) |
| Answerable questions not over-gated | 12/12 (100%) |
| Stale-law (s.21) handled correctly | 1/1 — explains the repeal (Finding 1) |
| Adversarial grounding-override | 5/5 — but see the variance note |

**Answer correctness and citation correctness are not comparable numbers.** The answer key is
keyword presence; citation correctness requires an Act-plus-number match. 86% > 80% says nothing
about the system being better at one than the other. The useful signal is where they *disagree*:
three questions (q09, q15, q23) cite the correct section and state the wrong rule, and under the
earlier metrics all three scored as clean passes. Only 16 of 22 keyed questions get both right.

Broken down, since the question set was deliberately made harder:

| | recall@6 | citation correctness |
|---|---|---|
| Original 23 questions | **87%** | **78%** |
| 7 new post-reform questions | **71%** | **43%** |

Abstention is counted wherever it legitimately happens, not only at the gate: of the 14 out-of-scope
probes, 9 were stopped by the keyword check, 4 by the retrieval-confidence check, and 1 by the
generator declining to answer from evidence that didn't address the question. The `must_answer` set
is the inverse guard and is what keeps the abstention number honest - see Finding 4.

**Retrieval recall became reproducible, and that was a deliberate fix.** Adding query rewriting made
recall@6 depend on a model call, so it stopped being a deterministic property of the index and
started drifting between runs. The cascade in Finding 5 - rewrite only when the question's own
wording retrieves badly - means well-formed questions never touch the model, and recall is now
identical across repeat runs. Citation correctness still inherits generation noise and is reported
as a range.

**The post-reform questions remain the weak point, and the fine-tuning case stands.** They retrieve
at 71% but cite at 43%: the model is handed the right provisions and still won't extract the
operative rule from amendment-instruction drafting ("in section 5, after subsection (1) insert—").
Qwen3-8B scores the same 43%, so this is not simply a capacity ceiling either. That gap - right
evidence, wrong citation, unchanged by doubling the model - is the measured argument for the
fine-tuning pass Docs.pdf §5 proposes.

**Variance warning — measured, not estimated.** Four consecutive runs of the adversarial set on the
same model with no code change scored 3/5, 3/5, 4/5, 3/5. The failures decompose cleanly: `adv04`
fails every time (a real, reproducible weakness), while `adv01` flips between runs at
`temperature=0.1`. So a single adversarial figure carries roughly ±20 points of noise, which is
larger than the gap between the two models compared below — the metric cannot currently rank
anything. One flapping probe out of five is enough to produce that swing, which is the whole
argument for growing this set by an order of magnitude before leaning on it.

One prompt change was reverted on the strength of this eval: an instruction added to stop the model
rambling when evidence was thin gave it an escape hatch it over-used, and it began answering "the
retrieved sources don't cover this" to questions whose governing provision *was* sitting in the
retrieved set (e.g. "can my landlord kick me out without going to court?", with Protection from
Eviction Act 1977 s.3(1) right there). Rebalancing it to "default to answering; only decline when
nothing in the excerpts bears on the question" took citation correctness on the original 23 from
65% back up to 78%.

These are reported as measured, not polished. The findings below are more informative than the
headline numbers.

**Re-run after the backend refactor.** Moving generation behind `llm.py`, the agent and
telemetry plumbing touched every request path, so the full eval was re-run on the same
local Qwen3-4B before shipping it (`eval/results/qwen3-4b_rag_2026-11-16.json`). Every
retrieval, citation, answer, abstention, must-answer, stale-law and adversarial figure is
identical to the table above. Hallucinated authority came in at 3/30 against 2/30, which is
within the run-to-run variance described below. The same file now records latency and tokens
per question: p50 6.3s, p95 13.7s across all 62 items (abstentions included), ~1,540 tokens in
and ~285 out per answered question.

**Legislation-only corpus (CI and Docker).** Without the six cases, the 6
case-law questions are skipped rather than failed, and retrieval on the remaining 24 in-scope
questions is **20/24 (83%)**, with all 14 out-of-scope probes stopped before generation.
That run is `eval/baseline_retrieval.json`, the baseline CI gates every push against.

**Hosted backend.** The *LLM eval* workflow produces the same table for `openai/gpt-oss-20b`
on Groq, in both `rag` and `agent` modes. Those numbers will be added here once that workflow
has run with an API key, rather than estimated in advance.

### Model comparison: Qwen3-4B vs Qwen3-8B

Docs.pdf §6 asks for the base model to be a swappable configuration measured against a fixed
benchmark. It is: `RAG_MODEL=qwen3:8b python run_eval.py`, and `eval/compare_models.py` prints the
table below from the per-model result files.

| | Qwen3-4B | Qwen3-8B |
|---|---|---|
| recall@6 (all in-scope) | 83% | 83% |
| citation correctness (all in-scope) | 67% | **73%** |
| citation correctness (original 23) | 78% | **83%** |
| citation correctness (new 7 post-reform) | 29% | **43%** |
| out-of-scope abstention | 100% | 100% |
| adversarial grounding | 80% | 60% (noise — see below) |
| latency per answer (warm) | **6.0s** | 13.6s |

Retrieval scores are identical across models, as they must be — retrieval doesn't involve the LLM,
so this doubles as a sanity check that the harness isolates the variable it claims to. The
abstention row was model-independent when this comparison was run, because abstention then happened
only at the scope gate, which runs before generation. That is no longer true: abstention is now
two-stage and the generator can decline, so the row became model-dependent after Finding 4. The 8B
column predates both that change and the 14-probe out-of-scope set, and is left as recorded.

**What this answers:** the citation-faithfulness gap is *partly* model capacity. Going 4B → 8B buys
about a third of the gap on the hard post-reform questions (29% → 43%) and 5 points on the original
set — real, but it does not solve the problem. A 43% citation score on questions where retrieval
succeeds 71% of the time means the model is still being handed the right provisions and failing to
cite them correctly. Scaling alone won't close this.

**A caution on reading this table, learned the hard way.** An intermediate version of this README
claimed the comparison had been *overturned* — that query rewriting took 4B to 71% on the
post-reform questions, past what 8B bought. That claim came from a single eval run and did not
reproduce: three repeat runs put 4B at 43%, exactly level with 8B. The lesson is the one the
variance note already makes and which is easy to forget when a number moves the way you hoped:
**at this sample size a single run is an anecdote.** Both columns here are single runs too, which is
why the conclusion above rests on the direction of the gap rather than its precise size.

**On the adversarial drop:** do not read 80% → 60% as 8B being worse at grounding. Two things rule
that out. The models fail *different* probes (4B fails adv04; 8B passes adv04 but fails
adv02/adv03) — and, more decisively, re-running 4B alone four times scored 3/5, 3/5, 4/5, 3/5, so
it reproduces the entire 80% → 60% "difference" against itself with nothing changed. The comparison
is inside the noise floor. That is itself the finding: the adversarial set needs to be much larger
before it can support a model-selection decision.

**Recommendation for the demo:** stay on 4B. 8B costs 2.3× the latency for a modest accuracy gain,
and at 13.6s per answer the demo starts feeling broken to someone clicking through it.

### Retrieval: hybrid dense + BM25

Dense-only search missed provisions written as amendment instructions ("omit Chapter 2 of Part 1")
even when the section heading stated the point plainly ("Abolition of assured shorthold
tenancies") — the wording a user searches with simply isn't in the body text. Fusing BM25 lexical
search with the dense search (round-robin interleave) fixed that class of miss: **recall@6 on the
original gold set went 74% → 87%** on the current corpus.

Four other approaches were measured and **rejected** because they didn't earn their place:

| Tried | Result |
|---|---|
| bge query-instruction prefix | no measurable gain |
| Reciprocal rank fusion (k0=60/10/1) | equal or worse than interleave; RRF buries a provision only the lexical side finds |
| Cross-encoder rerank (ms-marco-MiniLM-L-6-v2) | **zero** gain over hybrid, +1s per query |
| Guaranteeing legislation slots in top-k | actively harmful: 87% → 78%, case-law questions 5/6 → 3/6 |
| Raising k from 6 to 8 | no change on the queries it was meant to fix |

### Retrieval: everyday vocabulary → statutory wording

Users ask about a "lease", "moving out", being "kicked out"; the legislation says "tenancy",
"notice to quit", "possession". Neither embedding nor BM25 bridges that reliably — asking
*"what's the max notice a landlord could require for a leaser to terminate lease?"* retrieved
nothing relevant, while the same question phrased with "tenancy"/"notice" retrieved the right
provision. `expand_query()` in `backend/retrieval.py` appends the statutory wording when it sees
everyday phrasing. Measured: **colloquial phrasings 1/5 → 4/5** retrieved correctly, gold recall
80% → 83% (no regression).

### Finding 1: Section 21 has been repealed in the live text, and the system initially hallucinated around the gap

The corpus was built by fetching legislation.gov.uk's **current revised** text (not a point-in-time
snapshot), which meant it picked up a real, current legislative change: **Housing Act 1988 s.21
("no-fault eviction") is now repealed** — the corpus's `section-21` chunk is just the heading with
no body text, consistent with the Renters' Rights Act 2024/2025 reforms. This is a live instance of
exactly the "legislation currency" risk flagged before building this demo.

Asking the baseline system "what notice period does s.21 require?" is revealing: retrieval correctly
found no chunk actually stating a notice period (none exists), but the *topically* related chunks it
did retrieve (Ground 10, case law mentioning s.21) had similarity scores well inside the
confidence threshold — so the scope gate didn't abstain, and the first-pass prompt let the model
fill the gap from its own training knowledge, confidently citing "s.21(1) and (2)" and "one month,"
neither of which is in any retrieved excerpt. **This is the real failure mode the project's grounding
tests exist to catch.**

Fix applied: the generation prompt now includes an explicit allow-list of only the citations present
in the retrieved excerpts, with an instruction to flag (not guess) if the needed fact isn't in them.
This measurably improved behavior — the model now says *"this is not directly stated in the
retrieved excerpts"* instead of asserting a specific wrong number — but it's a prompt-level mitigation,
not a guarantee. The eval still scores this as abstention-failure under the strict metric (the system
didn't refuse to answer), which is the right call: the honest result is "the baseline doesn't fully
solve this," not a disguised pass. This is precisely the evidence Docs.pdf uses to justify the planned
LoRA fine-tuning pass — citation discipline is a measured gap, not a hypothetical one.

### Finding 2: the adversarial probe was measuring the wrong thing

`adv02` tampers the Housing Act 2004 s.213(3) deposit deadline from 30 to 90 days, and used to
fail. The cause turned out to be the test, not the model: the retrieved set also contained
Charalambous v Ng and s.213(3)'s sibling subsections, which state the real "30 days" themselves.
Tampering one chunk left the true figure sitting in context elsewhere, so the probe wasn't
measuring "does the model follow retrieved text over memory" — it was measuring which of two
*conflicting retrieved sources* the model preferred.

The probe now tampers **every** retrieved chunk containing the target string, making the evidence
internally consistent. `adv02` passes under the corrected test (4 chunks tampered), which confirms
the original diagnosis: it was never parametric hallucination. One probe (`adv04`) still fails
honestly at 4 chunks tampered.

### Finding 3: schema-constrained decoding, after a silent blank-answer bug

With plain `{"format": "json"}` Ollama guarantees *valid* JSON but not the right JSON. On one
question the model returned a well-formed object with keys it invented (`case_name`,
`paragraphs`) and echoed source text back — so every `.get("short_answer", "")` returned empty
and the user saw a **blank answer card** with no error. Generation now passes an explicit JSON
schema to constrain decoding, and falls back to an honest "could not produce a usable answer"
message rather than rendering blanks.

A post-generation check (`backend/verify.py`) also now extracts every section/Ground reference the
answer makes and compares it against what was actually retrieved, appending a warning for any that
aren't supported. This closes the gap Finding 1 identified — the s.21 case cited "s.21(1) and (2)"
when no s.21 text was retrieved at all.

### Finding 4: the abstention metric was scoring half the gate

A user query — *"main leasing law differences between London and Manchester"* — came back as a
confident restatement of Tenant Fees Act 2019 s.1(10), a provision with nothing to do with the
question. Two independent defects, both hidden by the eval.

**The confidence gate measured the wrong quantity.** It abstained only if the single best-matching
chunk exceeded 0.42 cosine distance. In a ~400-chunk corpus the nearest neighbour to *anything* is
closer than that, so the gate was close to inert: of eight deliberately off-corpus probes, five
passed through it, including "what is the best mortgage rate right now?".

Counting *close* hits separates the classes far better than thresholding the best one:

| | best distance | chunks within 0.30 |
|---|---|---|
| in-scope gold (n=30) | 0.117 – 0.255 | 3 – 6 |
| off-corpus probes (n=10) | 0.273 – 0.565 | 0 (8 of 10) |

Best-distance leaves a 0.018 margin to site a threshold in; the close-hit count leaves a gap from 0
to 3. The intuition is that an irrelevant question still has a nearest neighbour — it just doesn't
have *several*. The gate now requires ≥2 chunks within 0.30.

**The second defect no distance threshold can fix.** "London vs Manchester" is squarely on-topic, so
it retrieves six genuinely relevant tenancy provisions that simply don't address the question's
premise. Relevance scoring cannot detect a false premise. This is handled in the keyword layer
instead, which now corrects it directly: residential tenancy law is uniform across England, and what
varies locally is council licensing schemes, which this corpus doesn't hold. That is a statement
about how the statute book is territorially organised rather than a retrieved legal claim, which is
why it sits beside the Scotland/Wales patterns rather than being presented as a cited answer.

**Why the eval never caught either.** `run_out_of_scope` only ever called `keyword_scope_check`. The
100% abstention score was therefore testing regexes that had already been written for the exact
jurisdictions in the gold set, and never once exercised the confidence gate — the component that
handles everything nobody thought to write a regex for. It now runs the full gate as `main.py` does,
and records which check fired.

The out-of-scope set grew 5 → 15 to cover both classes. Five `must_answer` probes were added as the
inverse guard, because a scope gate is the easiest thing here to over-tighten — every abstention
looks responsible in isolation, and nothing in the old metric would have noticed the system
abstaining on everything. They assert the gate does *not* fire on questions that name a city but ask
something ordinary ("my landlord in London hasn't protected my deposit") or that use "difference"
with no geography ("Ground 8 vs Ground 10"). Retrieval and citation scores were unchanged by the
fix, as expected — neither retrieval nor generation was touched.

### Finding 5: the system fell apart on awkward questions, and it was one bug

Stress-tested with 16 deliberately awkward questions - vague, narrative, multi-part, misspelled,
false-premise - the system produced confident answers to questions nobody had asked:

| asked | answered |
|---|---|
| "I've lived here 3 years… is that allowed?" | "The tenant's payment of six months' rent on 29 April 2010 was not a tenancy deposit…" |
| "2 months behind on rent, how long until a court order?" | "The court determined that the February 2018 GSR was not provided to Ms Rouncefield…" |
| "Why does my landlord only need 24 hours notice?" | "The court held that Mr Kumarasamy was not liable for the disrepair…" |

Every one of those is a *summary of a case-law holding*. That is the tell. The corpus holds two very
different kinds of prose - terse statutory drafting and long judicial narrative - and a question
written as a story embeds closer to the narrative. Retrieval returned 4-5 case chunks out of 6, and
the model then did something quite reasonable with what it was given: it summarised the story.

Hand-rewriting the same questions into statutory wording was the decisive test:

| question | as asked | hand-rewritten |
|---|---|---|
| "I've lived here 3 years…" | 0.294 best, **5 of 6** case chunks | 0.181, 0 of 6 |
| "can he just chuck me out??" | 0.488 best, **4 of 6** case chunks | 0.227, 0 of 6 |
| "deposit + black mould" | 0.332 best, 3 of 6 | 0.205, 2 of 6 |

The corpus had the answers the whole time. The query was the defect, so the fix belongs before
retrieval, not in the prompt: `backend/query_understanding.py` has the model rewrite the question
into statutory vocabulary and into one search query per distinct sub-question, and
`Retriever.multi_search` interleaves a search per sub-query so a two-part question can't have one
half crowded out. The user's original wording is kept as a final query, because the rewriter can
drop a term that only exact lexical match would have found and it has no view of what's in the
corpus. Any failure falls back to the raw question.

**Rewriting every question was then found to be a mistake, and it is now a cascade.** With k=6 split
across three sub-queries each gets two slots, so a well-formed question whose own wording would have
found the provision at rank 3 loses it: q09, q16 and q27 all retrieve correctly on the raw question
alone and all missed once their slots were shared with paraphrases. Messy questions were being
improved by quietly degrading good ones, and the aggregate number hid the trade. `main.retrieve` now
searches the question's own wording first and only rewrites when that comes back thin - the
confidence check is already exactly that signal. Measured: **0 rewrites across all 30 in-scope gold
questions**, rewrites firing on precisely the 5 messy `must_answer` probes that need them, recall
restored to 83%, and most questions no longer paying the extra model call at all.

Three further defects surfaced only because this one was fixed:

- **Heading-only chunks were retrieval attractors.** Nine indexed chunks had a heading and no body -
  the provisions the Renters' Rights Act repealed (ss.19A, 20, 21), an emptied cross-heading, and
  four one-line concurrences ("[47] I agree."). A bare heading is a pure topical label with no
  qualifying text, so it sits close to *any* question on its topic: the repealed s.21 stub was the
  single best match (0.182) for a question about deposit deductions, winning a slot and then
  contributing nothing citable. `backend/chunk_quality.py` excludes them from both the dense and
  lexical halves; the chunks remain in `data/processed/` and Finding 1 is unaffected.
- **False premises were inherited.** "Why does my landlord only need to give me 24 hours notice?"
  was answered "a landlord may give a 24-hour notice… under certain conditions" - wrong, and the
  kind of wrong that matters. The rewriter now neutralises a premise rather than carrying it
  through ("whether a landlord can evict on 24 hours notice, and what notice is actually required"),
  and the answer became "the landlord cannot evict you on 24 hours' notice because the law requires
  at least two months'".
- **Citation verification was checking the number but not the Act.** Asked about s.21, the model
  answered "as per section 215 of the Housing Act 1988". Housing Act *2004* s.215 had been
  retrieved, so the bare number looked supported and nothing fired - though Housing Act 1988 s.215
  does not exist. Checks are now Act-qualified (`backend/verify.py`), with `tests/test_verify.py`
  covering both directions, including the two false positives that writing it introduced (a
  typographic apostrophe in "Renters' Rights Act 2025", and answer fields being concatenated so a
  reference in one field picked up an Act named in the next).

**What this cost, and what it actually bought.** With the cascade, the extra ~3s model call is paid
only by questions whose own wording retrieves badly, so a typical warm answer stays at ~6s.

The honest accounting on the headline metrics is that they barely moved: recall@6 is 83% before and
after, citation correctness 67% → 67-70%. An intermediate single run showed 87%/80% and was reported
as a gain; it did not reproduce. What this work actually bought is a *failure class*, not a
percentage - the system no longer answers a question nobody asked. The gold set barely measured that
class, which is precisely why the 12 `must_answer` probes were added: they are the only part of the
eval that would notice if this regressed.

### Finding 6: the chunk id knew something the chunk text didn't

Schedule 2 splits grounds for possession into Part I, "Grounds on which Court **must** order
possession", and Part II, "...**may** order possession" — mandatory versus discretionary, which
decides whether a tenant has a defence at all. The structure-aware chunker preserved that in the
chunk *id* (`schedule-2-part-II-crossheading-ground-10`) and dropped it from the text that gets
embedded and shown to the model. All 39 ground chunks read "Housing Act 1988, Schedule 2, Ground N"
and then the ground's wording, which never states which Part it sits in.

So the system could not answer "is Ground 10 mandatory or discretionary?" — and, worse, filled the
gap by inventing a citation.

**The first fix made it worse, which is the instructive part.** Prefixing each chunk with its Part
title gave all 39 grounds the same long opening line, homogenising their embeddings: a request for
Ground 10 came back with Grounds 5d, 6a, 5e and 5b, and recall fell 25/30 → 24/30. A shared prefix
across many chunks is a good way to destroy what distinguishes them.

The distinction is needed to *answer*, not to *find*. It now rides as chunk metadata and is attached
to the excerpt label at prompt time, leaving the embedded text untouched:

```
[Housing Act 1988, Sch.2 Ground 10 - Grounds on which Court may Order Possession]
[Housing Act 1988, Sch.2 Ground 8  - Grounds on which Court must order possession]
```

Measured: recall back to 25/30, and **answers containing a hallucinated authority fell from 5/30
(17%) to 2/30 (7%)** — q11, q26 and q31 stopped inventing citations once the real distinction was
available to them.

A companion change was tried and **rejected on measurement**: query expansion mapping "mandatory" →
"Court must order possession" and "discretionary" → "Court may order possession". It looks like the
obvious counterpart, and it dropped Ground 10 from rank 1 to rank 23 — an either/or question
contains both words, so both phrasings get appended, and they match every Part heading in the
Schedule.

### Finding 7: three fixes, and the one that would have been wrong

**Constrained citations instead of fine-tuning (for now).** Docs.pdf §5 proposes LoRA partly to
improve citation discipline. A JSON-schema `enum` buys some of that immediately: the answer's
`citations` field may only contain strings that were actually retrieved, so constrained decoding
*cannot* emit an authority that wasn't. Verified against a prompt explicitly instructing the model
to cite a non-retrieved Act. It is partial — only that field is constrained, prose can still mention
an invented Act — which is why `verify.py` still checks the text.

**Statutory instruments, and a parser bug they exposed.** The SIs parsed to nothing but headings.
The cause was in the existing parser and was costing the Acts too: `local_text` dropped every
`<Repeal>` subtree, but legislation.gov.uk marks Wales-only repeals the same way —
`<Repeal RetainText="true" Extent="W">` — and in an England-only corpus that text is still in force.
Across the corpus there are **1,645 retained-text repeals holding ~96KB**, against 26 genuine ones.

The obvious fix would have been wrong. Honouring `RetainText="true"` generally looks right and
isn't: that exact attribute appears in **both** Housing Act 1988 s.21, repealed outright by the
Renters' Rights Act, and s.13, which is in force. The attribute cannot separate them — only
commencement data could, which this pipeline doesn't resolve. So only `Extent="W"` is re-included,
which is unambiguous. Verified afterwards that s.21 still parses empty and the repeal manifest is
unchanged; the alternative was resurrecting repealed law as current, which is the worst failure this
project has.

**A parsing gap nearly became a false repeal.** The Forms Regulations 2015 parsed to five bare
headings because their substance lives in Schedules. Heading-with-no-body is exactly the signal the
repeal manifest uses, so they would have been reported to users as repealed provisions. That SI was
dropped, and `build_index.py` now refuses to treat a provision as repealed when its entire document
parsed empty — a parser gap must not be able to masquerade as a repeal.

**Unbounded generation was a hidden hang.** The eval crashed on the repealed-s.21 question after
600s: with little it was permitted to say, the model restated "the retrieved excerpts do not
cover..." until the request timed out. There was no cap on generation length at all. In the app that
is a ten-minute wait ending in a 500 — worse than a wrong answer, and it was made *more* likely by
raising the client timeout for slow machines. Bounded now by `num_predict` plus a `repeat_penalty`;
the question answers in ~8s. The lesson is that the fix for a slow generation is to bound the
generation, not to extend the wait.

## Known limitations

- **Snapshot, not live.** Corpus reflects legislation.gov.uk/Find Case Law content as fetched on
  this run's date; no re-sync pipeline. See Finding 1 above for why that matters in practice.
- **Curated, not comprehensive.** 7 Acts / 6 cases chosen for topic coverage, not exhaustiveness.
  County court possession case law — which decides most real-world disputes — is largely
  unreported and essentially absent from Find Case Law, hence from this corpus.
- **Amendment-text heuristic, not point-in-time resolution.** legislation.gov.uk marks
  in-progress amendments with `<Repeal>`/`<Addition>` spans; this pipeline drops `<Repeal>` text
  and keeps `<Addition>` text as a heuristic for "current," rather than resolving exact
  commencement dates. Usually right, occasionally produces an empty/thin chunk (see Finding 1).
- **Citation faithfulness is checked, not guaranteed.** `backend/verify.py` flags references the
  retrieved evidence doesn't support, but it matches on section/Ground numbers — it catches an
  invented authority, not a real one applied to the wrong proposition.
- **Conceptual/multi-hop questions still retrieve poorly.** "How does eviction work now that s.21
  is gone?" needs two facts from different chunks (shortholds abolished; possession only on
  Schedule 2 grounds) and no single chunk states the combination. Hybrid retrieval, reranking and
  source balancing were all measured against this and none fixed it — it's a corpus/reasoning gap,
  not a ranking one. Query rewriting helps (these questions now retrieve at 86%) but doesn't
  dissolve it.
- **Questions adjacent to the corpus can still slip the gate.** "What are the rules on dog
  ownership in flats?" has no answer here, but generic tenancy-agreement provisions are its nearest
  neighbours, so it clears the distance gate and relies on the generator to decline. It does so
  currently, but that is a model judgement rather than a guarantee. Tightening the gate to catch it
  is measurably the wrong trade — see Finding 4.
- **Latency is ~6s for a well-formed question, ~9-12s for one that needs rewriting**, since the
  cascade only pays for the extra model call when the first retrieval comes back thin.
- **Case citations are not verified.** `verify.py` checks statutory references Act-qualified, but
  nothing checks neutral citations — the model was observed emitting "[2013] EWCA Civ 1606" where
  the real citation is 1600, and a one-digit error in a neutral citation points at a different case
  entirely.
- **Topic gaps remain**: HMO licensing and HHSRS (Housing Act 2004 Parts 1-3), service charges
  (LTA 1985 ss.18-30), and Rent Repayment Orders are still absent.
- **No National Archives licence has been obtained.** Docs.pdf §2 lists this as a prerequisite for
  using Find Case Law material for machine learning, vector search or other computational analysis,
  and it has not been done. Judgments are Crown copyright and published under the Open Justice
  Licence, which covers ordinary reading and quotation but requires a separate transactional licence
  for computational use at scale. For a six-judgment demo run locally this is a paperwork gap rather
  than a live infringement risk, but it is a real precondition for anything shipped or published,
  and it is unresolved rather than overlooked. **That is why everything published - the git
  repo and the Docker image - carries legislation only** (Open Government
  Licence), and the six cases stay in local builds. Case-law questions are reported as skipped,
  not failed, when the corpus doesn't hold them; adding the licence and the cases is a data
  change, not a code change.
- **Only one statutory instrument is ingested.** The Prescribed Information Order 2007 is in;
  the Assured Tenancies (Forms) Regulations 2015 were tried and dropped because their substance is
  the forms themselves, which sit in Schedules this pipeline doesn't read (Finding 7).
- **Not legal advice**, and no regulated-activity compliance layer — this is a demo of grounding
  mechanics, not a production legal product.
