# Scout architecture

This document explains how Scout works and why it is built this way. The canonical requirements are in [SPEC.md](SPEC.md).

## 1. The state machine

Scout is an explicit state machine written by hand in [`scout/agent/orchestrator.py`](../scout/agent/orchestrator.py). One run moves through these stages:

| Stage | Input → output | LLM? |
|---|---|---|
| Intake | goal text → `Task` (targets, purpose, user context) | yes (fast tier) |
| Recall | task → `MemoryContext` (fresh facts, top lessons, source scores) | no |
| Plan | task + memory + playbook → `ResearchPlan` (≤5 research steps, plus memory steps) | yes (smart tier) |
| Execute | one step → `StepResult` via a ReAct loop (≤3 iterations) | yes (fast tier) + tools |
| Critique | step result → verdict + per-source ratings | yes (fast tier) |
| Synthesize | evidence ledger + playbook sections → `Brief` | yes (smart tier) |
| Verify | brief → checked brief (citations, numbers, one revision) | mostly no |
| Reflect | run summary → lessons + votes; facts, sources and the run are persisted | yes (smart tier) |

The critic makes the graph conditional. After each step it returns one of:

- **complete**: move to the next step;
- **retry** with a `new_approach`: re-run the same step once with that approach injected (at most 1 per step and 2 per run);
- **followup** with a new step: append it to the plan, a *replan* (at most 1 per run);
- **unknown**: stop trying; the question goes to the brief's Unknowns section.

Every stage is a small method that yields events. Failures inside a stage become `error` events and a fallback (a playbook-based plan, a deterministic brief, skipping reflection), so a run always ends with `run_finished`.

**Budgets** (tool calls, LLM calls, wall clock, iterations) live in `Settings` ([`config.py`](../scout/config.py)) and are checked before every call by [`budget.py`](../scout/agent/budget.py). Two LLM calls are reserved for synthesis, so a budget hit always ends in a brief that states it is partial.

## 2. The event stream

The orchestrator is a generator: `for event in Orchestrator(settings).run(goal)`. An `Event` is `{run_id, ts, stage, type, payload}` ([`events.py`](../scout/events.py)). The same stream feeds three consumers:

- the Rich CLI (`python -m scout run`), which prints each event;
- the Streamlit UI, which reduces events into a `RunView` ([`scout/ui/viewmodel.py`](../scout/ui/viewmodel.py));
- `RunRecorder`, which appends each event to `runs/<run_id>/trace.jsonl` and writes `report.md` and `metrics.json` at the end.

Because `run_finished` carries the task, the plan, the brief, the evidence and the metrics, a trace is enough to **replay** a run with no LLM or network: the UI's replay mode feeds the saved events through the same reducer it uses for live runs. The executor is itself a generator that *returns* its `StepResult` (`result = yield from execute_step(...)`), so step-level events stream without callbacks.

The one place a generator cannot help is a blocking wait: when a provider rate-limits a call, the LLM client sleeps inside a single `next()`. For that it calls an optional listener immediately before sleeping (the UI shows "Waiting 12 s: rate limit on qwen3.8-27b") and also records a `rate_limited` event for the trace.

## 3. LLM protocol and the provider router

Every structured call goes through `LLM.complete_json(messages, schema)` ([`llm.py`](../scout/llm.py)):

1. Send the request (JSON mode where the provider supports it).
2. Validate the reply against the Pydantic model; on failure send **one** repair request containing the validation errors; then raise.
3. Retry 429/5xx with exponential backoff that honours `Retry-After` and never waits past the run's deadline.

Stages talk about a *tier* ("smart" for planner, synthesizer and reflector; "fast" for executor and critic), not a model. [`RouterBackend`](../scout/router.py) maps a tier to an ordered candidate list from the routing table in `config.py` (provider base URL + key variable name + model; no secrets):

- short 429: back off and retry the same candidate; a second one in a row counts as long;
- long 429 or daily quota: put the candidate in cooldown until the provider's reported reset time and fail over;
- 401 / 403 / 404: disable the candidate for this run and fail over;
- when the preferred candidate's cooldown ends, it is used again.

Every change of serving candidate becomes a `provider_switched` event, and every call records its provider and model, so tokens per model appear in the metrics. This exists because free tiers are small and per model: Groq gives each model its own daily token quota, so four models on one key is about four times the capacity.

Some models (gpt-oss on Groq) answer with native function calls even when asked for JSON. The provider rejects that output but returns it in `failed_generation`; Scout treats it as the model's answer, and the `Action` schema accepts the `{"name", "arguments"}` shape, so the ReAct loop keeps working.

## 4. Memory design

[`scout/memory/store.py`](../scout/memory/store.py) is a small repository over SQLite (`data/scout.db`) with five tables: `runs`, `facts`, `sources`, `lessons`, `feedback`.

- **Entities** are normalised (lowercase, company suffixes stripped, the domain when a URL is given), so "Zoho Corporation" and "ZOHO Corp." share memory.
- **Facts** are the evidence that survived verification, stored with source URL, snippet, confidence, timestamp and a volatility tag. *Stable* facts stay fresh for 7 days, *news* for 2. Recall ranks fresh facts by relevance to the goal and the purpose's research hints, capped at 15 facts or about 1.5k tokens.
- **Memory steps.** The planner sees facts with ids and may mark a step `answered_from_memory` with the ids it relies on. A deterministic pass (`attach_memory`) converts research steps that match stored fact topics and, if fresh facts exist but none were used, prepends a background memory step. Memory steps cost no tools and no LLM calls; their facts enter the evidence ledger with their original URLs, so citations stay valid.
- **Sources** keep useful/useless counts per domain from critic ratings and user feedback. Score = `(useful + 1) / (useful + useless + 2)`; unseen domains are 0.5. Search results are re-ranked by `rank + 0.5 × (score − 0.5)`, which can move a trusted domain about one place, never from the bottom to the top.
- **Lessons** are 1–3 short, general sentences per run from the reflector, scoped to a purpose. A new lesson that overlaps an existing one by keywords is merged as an up-vote instead of inserted. The top 3 by vote score (recency breaks ties) are injected into the planner prompt; the reflector then votes on whether they helped.

## 5. The verifier

[`scout/agent/verifier.py`](../scout/agent/verifier.py) runs after synthesis and is deterministic:

1. **Citation check:** every claim must cite at least one evidence id that exists in the ledger.
2. **Numeric fidelity:** every number in a claim (after normalising "30,300" to "30300" and ignoring evidence ids) must appear in the snippet of an evidence item it cites.

Flagged claims get **one** LLM revision pass (rewrite using only the cited evidence, or drop). Whatever still fails moves to the Unknowns section as "Could not verify: …". Then a regex pass scrubs emails and phone numbers, and the score is normalised to the playbook's fixed format ("Fit 75/100", "Threat: Medium", "Readiness 4/6"). The synthesizer prompt also tells the model to state the most recent dated figure when sources conflict and to note the conflict.

## 6. Safety

- Web text crosses into prompts only through the tool registry, wrapped in `<untrusted_content>` with embedded closing tags neutralised; every prompt treats it as data.
- `fetch_page` has an SSRF guard: http/https only; localhost, private, loopback, link-local (cloud metadata), CGNAT and other non-global addresses blocked; each redirect hop re-validated; size and time capped.
- Roles only: personal profile pages are never fetched or cited, lessons that recommend profiles or contact details are discarded, and the brief is scrubbed.

## 7. Trade-offs

| Decision | Why | Cost |
|---|---|---|
| Hand-written state machine, no framework | Every edge, budget and failure is visible and unit-tested; replay falls out of the event stream | More code to own than a framework's defaults |
| JSON actions instead of native tool calling | Portable across OpenAI-compatible providers; validated and repairable | Some models need the native-call shape accepted too |
| SQLite | Zero setup, transactional, inspectable | Single-user; no fuzzy (vector) recall |
| Deterministic checks behind the LLM | Citations, numbers, personal data and memory reuse do not depend on model quality | Exact-match numeric check can be strict |
| Free-tier router | Works without paid keys | Rate limits slow runs; quality varies by model |
| Replay-first UI | Demos and judging spend no quota and are reproducible | A replay must always be labelled as such (the UI does) |
