---
marp: true
theme: default
paginate: true
size: 16:9
style: |
  section { font-size: 26px; }
  h1 { font-size: 44px; }
  table { font-size: 20px; }
  section.tight { font-size: 22px; }
  section.tight h1 { font-size: 38px; }
---

# Scout 🔭
### Research that depends on *why* you are asking

- Same company, different purpose → different facts: a **sales rep**, a **competitor analyst** and an **interview candidate** need different briefs about Zoho
- Most agents run one generic prompt. Scout turns the purpose into a **playbook-guided plan**, cites every claim, checks its own work, and **learns with use**

![bg right:48% fit](img/ui.png)

---

# Architecture

Hand-written state machine (no framework). The orchestrator is a generator of events → CLI, UI and trace all consume the same stream.

**Intake → Recall → Plan → [Execute ⇄ Critique]\* → Synthesize → Verify → Reflect**

| Layer | Choice |
|---|---|
| LLM | OpenAI-compatible wrapper, JSON + Pydantic + 1 repair; **router** across 3 Groq models + Gemini (free tiers) |
| Tools | Tavily → DuckDuckGo search, SSRF-guarded fetch (httpx + trafilatura), `recall_memory` |
| Memory | SQLite: facts, source scores, lessons, feedback, runs |
| UI | Typer + Rich CLI, Streamlit with live runs and **replay** |

---

<!-- _class: tight -->

# Agentic patterns in action

- **Plan-and-Execute**: purpose-shaped plan from playbook + lessons
- **ReAct** per step: think → search / fetch / recall → observe → cited findings
- **Critic routing**: complete · **retry** (new approach) · **follow-up** (replan) · unknown
- **Verifier**: citations must exist; **numbers must appear in a cited snippet**; 1 revision, then Unknowns
- **Reflexion**: lessons written after each run, voted on later

![bg right:45% fit](img/trace.png)

---

# Learns with use

1. **Fact memory**: verified facts with source + freshness (7 d stable, 2 d news) answer steps for free
2. **Source reliability**: critic ratings + 👍/👎 → `(useful+1)/(useful+useless+2)` re-ranks search
3. **Strategy lessons**: top 3 per purpose injected into the next plan

Same goal ("Prep me for an SDE intern interview at Zoho"), before → after memory:

| | tool calls | LLM calls | tokens | seconds |
|---|---|---|---|---|
| no memory | 11 | 27 | 38,190 | 122 |
| 1 step from memory, 6 facts | **5** | **16** | **18,861** | **32** |

<!-- DEMO_TABLE: replace with the clean learning_demo.sh numbers -->

---

# Results and next steps

- **Evals** (9 recorded runs with the verifier): 100% citation coverage; 13 claims flagged, 5 moved to Unknowns instead of published
- **Safety**: untrusted-content wrapping, SSRF guard, roles only (no profiles or contact data), budgets that end in honest partial briefs, traced provider failover
- **Quality**: 238 offline tests; replayable runs; zero-key example replays

**Next:** scheduled competitor tracking · vector memory · parallel steps · Scout as an MCP server

**Repo:** `scout-agent` (README → quickstart in 5 commands) · MIT
