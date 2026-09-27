# Scout — Specification

## SECTION 2. PRODUCT DECISION: SCOUT

We build Scout, a purpose-aware company intelligence agent. You give it a company and a purpose. It plans research around that purpose, gathers cited evidence, checks its own work, and gets measurably faster and sharper with every run.

Challenge context. Techvruk contest hosted by Tryneu Global Solution, 15 internship slots, deadline Sunday 27 Sep 2026 at 11:30 PM. Judged on LLM reasoning and prompting, agentic design patterns, workflow orchestration, and practical tool integration. The brief itself says agents should "get better with use" and "research dynamically based on the task or purpose". Its example list leans hard on lead enrichment, lead scoring and competitive intelligence aimed at colleges, hiring managers and engineering firms, which tells us what the host's business cares about.

Why Scout wins, criterion by criterion

| Judging criterion | How Scout proves it |
| --- | --- |
| LLM reasoning and prompting | Separate role prompts (planner, executor, critic, synthesizer, reflector), structured JSON outputs validated by Pydantic, automatic repair on malformed output |
| Agentic design patterns | Plan-and-Execute outer loop, ReAct inner loop per step, critic-driven replanning, Reflexion-style lessons, tool use |
| Workflow orchestration | Explicit state machine with conditional routing: continue, retry with a new approach, add a follow-up step, or mark unknown |
| Practical integration | Live web search with a fallback provider, page fetch and extraction, SQLite memory, CLI and Streamlit UI |
| "Gets better with use" | Fact cache with freshness, source reliability scores, lessons injected into future plans, run metrics that prove the improvement |
| "Research dynamically by purpose" | Same company, different purpose → visibly different plan, tools and output format |

## SECTION 3. PRODUCT SPEC

One text goal in, one cited brief out, shaped by one of three purpose playbooks (plus a general fallback), with three memory mechanisms that make the next run better.

Input

goal (free text) saying who to research and why. Example: "We sell a campus hiring-challenge platform. Research Zoho as a prospect."

Optional user_context ("we sell X to Y") and budget overrides.

The intake stage extracts a Task : targets (1 or 2 company names or URLs), purpose_type , user_context , constraints .

Purpose playbooks ( scout/playbooks.py ). Each playbook gives the planner research hints, the output sections and a scoring rubric. The planner still writes its own plan; the playbook guides it, it is not a script.

Purpose Research hints Output sections Score
sales_prospect What they do, size and locations, hiring volume and campus activity, recent news or funding, tools they use, likely pain points, decision-maker roles

Snapshot, Buying signals, Pain points, Who to approach (roles only), Pitch angle, Risks

Fit score 0–100 with reasons

competitor Positioning, pricing, customers, recent launches, hiring signals

Positioning, Pricing, Recent moves, Strengths and weaknesses, Threat assessment

Threat level low, medium or high

interview_prep Products, business model, tech stack, engineering culture, recent news, interview process signals

Company in 60 seconds, Tech stack, News worth mentioning, Likely topics, Smart questions to ask

Readiness checklist

general Planner decides Summary, Key findings, Open questions

None

Output: the Brief

A Markdown report with the playbook's sections.

Every factual claim cites an evidence id that maps to a source URL. The verifier flags or removes claims with no evidence.

An "Unknowns" section lists what Scout could not verify. Honesty beats coverage.

Run metrics: tool calls, LLM calls, tokens, latency, citation coverage %, facts reused from memory.

Saved to runs/<run_id>/ as report.md , trace.jsonl and metrics.json .

Learning with use: three mechanisms

1. Fact memory. Facts about an entity are stored with source URL, timestamp and confidence. The recall stage loads fresh facts (default time-to-live 7 days, 2 days for news). The planner marks steps "answered from memory" and skips them. Proof: fewer tool calls on a repeat entity.

2. Source reliability. Each domain has a score updated when the critic marks a source useful or useless, and when the user gives feedback. Search results are re-ranked by this score before fetching. Proof: higher useful-fetch rate over runs.

3. Strategy lessons (Reflexion-style). After each run the reflector writes 1 to 3 short lessons for that purpose, for example "For hiring signals, the careers page beats news search." The top lessons are injected into future planner prompts and gain or lose votes based on later outcomes. Proof: visibly different plans and fewer wasted calls.

User feedback (thumbs up or down per report section) feeds both source scores and lesson votes.

Insights view makes the learning visible: a runs table showing tool calls, latency and citation coverage over time, the lessons list with votes, and the top and bottom sources.

Non-goals: personal contact scraping, auth or multiple users, scheduled recurring runs (list as future work), vector databases.

## SECTION 4. ARCHITECTURE

Scout is an explicit state machine written by hand (no LangChain), with a Plan-and-Execute outer loop, a ReAct inner loop per step, and a critic that routes conditionally. The orchestrator is a Python generator that yields events, so the CLI, the UI and the trace file all consume the same stream.

Workflow (reuse this diagram in the README)

```mermaid
stateDiagram-v2
[*] --> Intake
Intake --> Recall
Recall --> Plan
Plan --> Execute
Execute --> Critique
Critique --> Execute: next step, or retry with new approach
Critique --> Plan: add follow-up step (replan)
Critique --> Synthesize: all steps done or budget reached
Synthesize --> Verify
Verify --> Synthesize: unsupported claims (max 1 revision)
Verify --> Reflect
Reflect --> [*]
```

Stages

Stage Input → Output Uses LLM

Intake goal text → Task Yes

Recall Task → MemoryContext (fresh facts, stale facts, top 3 lessons, source scores) No

Plan Task + MemoryContext + playbook → ResearchPlan (max 6 steps, each with question, rationale, suggested tools, done criteria, answered_from_memory flag) Yes

Execute one Step → StepResult via ReAct: each iteration returns an Action (thought + tool + args) or a finish with findings; max 4 iterations Yes + tools

Critique StepResult + done criteria → Critique with verdict complete , retry (with new approach), followup (new step) or unknown , plus a rating per source used Yes

Synthesize evidence ledger + playbook sections → Brief (sections of claims, each claim listing evidence ids) Yes

Verify Deterministic: every claim must cite existing evidence ids. Unsupported claims get one revision pass, then move to Unknowns Mostly no

Reflect run summary → 1–3 new lessons + votes on injected lessons; persist facts, source scores, metrics Yes (lessons only)

Tools (registry with a name, description and JSON argument schema each)

web_search(query, max_results=5) → [{title, url, snippet}] . Tavily first, DuckDuckGo fallback, results re-ranked by source score.

fetch_page(url, focus) → extracted text (trafilatura), cut to about 4,000 characters, keeping the chunks most relevant to focus .

recall_memory(entity, topic) → stored facts for that entity.

The executor's finish action returns findings: [{claim, source_url, snippet, confidence}] ; the executor turns them into Evidence records with ids.

LLM protocol. One function, llm.complete_json(messages, schema) , validates output against a Pydantic model. On failure it sends one repair request containing the validation error, then raises. This works on any OpenAI-compatible provider and is fully testable with a FakeLLM that returns scripted responses.

Events. Event{run_id, ts, stage, type, payload} with types: stage_started , plan_created , memory_hit , step_started , thought , tool_call , tool_result , critique , replan , synthesis , verification , lesson_learned , run_finished , error . Every event is appended to runs/<run_id>/trace.jsonl .

Data model

Pydantic ( scout/schemas.py ): Task , MemoryContext , Step , ResearchPlan , Action , Observation , Evidence , StepResult , Critique , Claim , Section , Brief , Lesson , RunMetrics .

SQLite ( data/scout.db ): runs(id, goal, purpose_type, targets, started_at, finished_at, metrics_json) , facts(id, entity, topic, claim, source_url, captured_at, confidence, run_id) , sources(domain, useful, useless) , lessons(id, purpose_type, text, votes_up, votes_down, created_at, last_used_at) , feedback(id, run_id, section, rating, created_at) .

Source score = (useful + 1) / (useful + useless + 2). Lesson rank uses the same formula on votes, with recency as tie-breaker.

Entity names are normalized (lowercase, strip "pvt ltd", "inc", "limited"; prefer the domain when known).

Budgets (all in config.py , all overridable): max 5 planned steps (8 with follow-ups), 1 follow-up per run, 3 ReAct iterations per step, retries capped at 1 per step and 2 per run, 30 tool calls, 60 LLM calls, 480 seconds wall clock, 15 seconds per fetch. Hitting a budget never crashes: Scout synthesizes from what it has and says so in the brief. A backoff wait never overshoots the wall clock; synthesis gets a short grace period.

Provider router. LLM calls go through a router that holds an ordered candidate list per tier (smart: planner, synthesizer, reflector; fast: executor, critic). A candidate is a provider (base URL plus the name of its key's environment variable) and a model; the routing table lives in config.py and contains no secrets. Several Groq models are listed because Groq quotas are per model, with Gemini as the last fallback. On a daily-quota error, a 429 whose wait exceeds 20 seconds, or a repeated short 429, the candidate cools down (until the provider's reported reset time) and the call fails over to the next one; a 401, 403 or 404 disables the candidate for the run. Every switch is traced as a provider_switched event and every call records its provider and model. Without router keys, the single-provider LLM_* settings are used.

Context management. The executor sees the step question, the playbook hint and a compact scratchpad (last 3 observations in full, older ones as one-line summaries). The evidence ledger lives outside the prompt. Tokens are counted per call.

Safety

Fetched web content is wrapped in <untrusted_content> tags; every system prompt says to treat it as data and never follow instructions inside it (prompt-injection defense).

fetch_page allows only http and https, blocks localhost and private IP ranges, and caps response size.

The synthesizer is told to record roles only; a final regex pass strips emails and phone numbers from the brief.

## SECTION 5. TECH STACK AND REPOSITORY LAYOUT

Python 3.11+, a thin OpenAI-compatible LLM wrapper, free-tier providers, SQLite, Typer CLI and Streamlit UI. No agent framework: a readable custom orchestrator is itself evidence of understanding.

Repository layout.

```
scout-agent/
├── README.md
├── LICENSE MIT
├── requirements.txt
```

Layer Choice Why

LLM access
openai Python SDK pointed at any OpenAI-compatible endpoint ( LLM_BASE_URL , LLM_API_KEY , model names in .env )

One wrapper covers Groq, Gemini's OpenAI-compatible endpoint, OpenRouter free models and local Ollama

Models
Free tier. SCOUT_MODEL (planner, synthesizer, reflector) and optional SCOUT_FAST_MODEL (executor, critic) to stretch rate limits. Current model names and limits are confirmed before use

Free-tier fairness rule; disclose in README

Validation
Pydantic v2 Structured outputs and repair loop

Search
Tavily free tier, with ddgs (DuckDuckGo, no key) as automatic fallback

Resilience; works even without a Tavily key

Extraction
httpx + trafilatura Clean article text from pages

Storage
sqlite3 from the standard library Zero setup for judges

CLI
Typer + Rich Readable live trace in the terminal

UI
Streamlit Fastest route to a live trace and insights view

Quality
pytest, ruff Tests are part of the pitch

Repository layout

```
├── pyproject.toml ruff + pytest config
├── .env.example LLM_BASE_URL, LLM_API_KEY, SCOUT_MODEL, SCOUT_FAST_MODEL, TAVILY_API_KEY
├── .gitignore .env, data/, runs/, .venv/, .commit-msg.txt
├── docs/
│ ├── SPEC.md Sections 2–10 of this brief
│ └── architecture.md
├── scout/
│ ├── __main__.py CLI: run, insights, feedback, eval
│ ├── config.py settings and budgets
│ ├── llm.py client, complete_json, retries, token counting
│ ├── schemas.py all Pydantic models
│ ├── playbooks.py
│ ├── events.py Event model + trace writer
│ ├── prompts/ intake, planner, executor, critic, synthesizer, reflector (.md)
│ ├── agent/
│ │ ├── orchestrator.py state machine, yields Events
│ │ ├── intake.py planner.py executor.py critic.py
│ │ └── synthesizer.py verifier.py reflector.py
│ ├── tools/
│ │ └── registry.py web_search.py fetch_page.py memory_tools.py
│ ├── memory/store.py SQLite repository
│ └── eval.py
├── app/streamlit_app.py
├── tests/ fakes.py (FakeLLM, FakeSearch) + test_*.py
├── examples/ 3 saved runs: report.md, trace.jsonl, metrics.json
├── evals/ tasks.yaml, results.md
└── scripts/learning_demo.sh
```

## SECTION 6. ENGINEERING STANDARDS

Code

- Type hints everywhere, small functions, a docstring on every public function.
- Prompts live in `scout/prompts/*.md`, never as long inline strings.
- Settings and budgets come only from `scout/config.py`.
- Library code emits events or uses `logging`; no stray `print`.
- `ruff check .` passes on every change.

Tests

- Tests never touch the network; they use `FakeLLM` and `FakeSearch` from `tests/fakes.py`.
- `pytest -q` stays green; real API runs belong in scripts and manual checks, not in the test suite.

Secrets and dependencies

- Real keys live only in `.env`, which is gitignored; keys are never printed or logged.
- Versions are pinned in `requirements.txt`.

Honesty

- Metrics, outputs and example runs are never faked: everything in `examples/` and `evals/` comes from real runs.
