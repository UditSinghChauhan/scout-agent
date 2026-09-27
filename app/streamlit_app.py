"""Scout UI: live runs, replays of recorded runs, insights and a how-it-works page.

Launch from the repo root:  python -m scout ui   (or: streamlit run app/streamlit_app.py)
Replay mode is the default and makes no LLM or network calls.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from scout.agent.orchestrator import Orchestrator  # noqa: E402
from scout.agent.synthesizer import escape_md, presentable, render_parts  # noqa: E402
from scout.config import (  # noqa: E402
    PROVIDERS,
    ROUTES,
    Settings,
    llm_configured,
    load_settings,
    provider_key,
    router_enabled,
)
from scout.events import RunRecorder  # noqa: E402
from scout.insights import apply_feedback  # noqa: E402
from scout.memory.store import MemoryStore, reset_memory, score  # noqa: E402
from scout.playbooks import purpose_label  # noqa: E402
from scout.ui.recorded import (  # noqa: E402
    RecordedRun,
    recorded_lessons,
    recorded_runs,
    recorded_sources,
)
from scout.ui.runs import SavedRun, list_runs, load_events  # noqa: E402
from scout.ui.viewmodel import (  # noqa: E402
    RunView,
    StepView,
    TraceEntry,
    build_view,
    reduce,
    wait_line,
)

logger = logging.getLogger("scout.ui")

_PITCH = (
    "We sell a campus hiring-challenge platform. "
    "Research {} as a prospect and tell us how to pitch."
)
HERO_GOALS = (
    _PITCH.format("Zoho"),
    "Prep me for an SDE intern interview at Zoho.",
    _PITCH.format("Freshworks"),
)
STATUS_COLORS = {
    "pending": "gray",
    "running": "blue",
    "done": "green",
    "from memory": "violet",
    "retried": "orange",
    "unknown": "yellow",
    "skipped (budget)": "gray",
    "not reached": "gray",
    "error": "red",
    "n/a": "gray",
}
ENTRY_ICONS = {
    "thought": "💭",
    "tool": "🔧",
    "result": "👁",
    "critique": "⚖️",
    "retry": "🔁",
    "switch": "🔀",
    "error": "⚠️",
    "memory": "🧠",
    "wait": "⏳",
}
NO_KEYS_MESSAGE = "Live mode needs API keys; see README. Replay the example runs below."
QUIET_EVENTS = {"llm_call"}  # not worth a re-render during streaming
WORKFLOW_DOT = """
digraph scout {
  rankdir=LR; bgcolor="transparent"; nodesep=0.35; ranksep=0.45;
  node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica", fontsize=18];
  edge [fontname="Helvetica", fontsize=14, color="#667085", fontcolor="#667085"];
  Intake -> Recall -> Plan -> Execute -> Critique;
  Critique -> Execute [label="next step / retry"];
  Critique -> Plan [label="follow-up (replan)"];
  Critique -> Synthesize [label="done or budget"];
  Synthesize -> Verify;
  Verify -> Synthesize [label="1 revision"];
  Verify -> Reflect;
  Memory [shape=cylinder, fillcolor="#f3eefb", label="Memory\\nfacts · sources · lessons"];
  Memory -> Recall [style=dashed]; Reflect -> Memory [style=dashed];
  Tools [shape=component, fillcolor="#eefbf1", label="Tools\\nsearch · fetch · memory"];
  Execute -> Tools [dir=both, style=dashed];
  Router [shape=hexagon, fillcolor="#fbf6ee", label="Model router\\nGroq ×3 → Gemini"];
  subgraph router_links {
    edge [style=dotted, arrowhead=none, color="#b08a4a", fontcolor="#b08a4a"];
    Router -> Plan [label="LLM calls"]; Router -> Execute; Router -> Synthesize;
  }
}
"""


# --- helpers ---------------------------------------------------------------------------------


def settings() -> Settings:
    """Current settings (read on every rerun so .env changes are picked up)."""
    return load_settings()


def open_store(cfg: Settings) -> MemoryStore:
    """A fresh store connection for this rerun."""
    return MemoryStore(cfg.db_path)


def citation_linker(tail_md: str) -> Callable[[str], str]:
    """A function that turns [n] citations into links, using the Sources list in ``tail_md``."""
    sources = dict(re.findall(r"^(\d+)\. <(.+)>$", tail_md, re.M))

    def link(text: str) -> str:
        def one(match: re.Match[str]) -> str:
            num = match.group(1)
            return f"[[{num}]]({sources[num]})" if num in sources else match.group(0)

        return re.sub(r"(?<!\[)\[(\d+)\](?!\()", one, text)

    return link


def linkify(report_md: str) -> str:
    """Turn numbered citations [n] into links to the matching source URL."""
    body, sep, tail = report_md.partition("\n## Sources")
    return citation_linker(tail)(body) + sep + tail


def demote(markdown: str) -> str:
    """Demote report headings so the brief sits inside the page (H1 -> H3, H2 -> H4)."""
    return re.sub(r"^(#{1,2}) ", lambda m: "#" * (len(m.group(1)) + 2) + " ", markdown, flags=re.M)


def fmt_metric(label: str, value: Any) -> str:
    """Readable metric values: 29,723 tokens, 56.4 s, 100%."""
    if value is None:
        return "–"
    if label == "Seconds":
        return f"{float(value):.1f} s"
    if label == "Citation coverage":
        pct = float(value)
        return f"{pct:.0f}%" if pct == int(pct) else f"{pct:.1f}%"
    if isinstance(value, int | float):
        return f"{int(value):,}"
    return str(value)


def provider_lines(cfg: Settings) -> list[str]:
    """Configured providers and models (never keys)."""
    if router_enabled(cfg):
        lines = []
        for tier, candidates in ROUTES.items():
            usable = [c for c in candidates if provider_key(cfg, PROVIDERS[c.provider])]
            items = "\n".join(f"{i}. {c.model} ({c.provider})" for i, c in enumerate(usable, 1))
            lines.append(f"**{tier} tier** (in failover order)\n\n{items or 'none configured'}")
        return lines
    host = cfg.llm_base_url.split("://", 1)[-1].split("/", 1)[0] or "not configured"
    return [
        f"**single provider** {host}",
        f"models: {cfg.scout_model or '?'} / {cfg.fast_model or '?'}",
    ]


def friendly_error(where: str, exc: Exception) -> None:
    """Log the details, show a short message (never a stack trace)."""
    logger.exception("UI error in %s", where)
    st.error(
        f"Something went wrong while {where}. The details were logged; try another run or reload."
    )


# --- run view --------------------------------------------------------------------------------


def render_badge(badge: str | None) -> None:
    """Always-visible banner saying whether this is a replay or a live run."""
    if badge:
        st.info(badge, icon="🔁" if badge.startswith("Replay") else "🟢")


def render_recall(view: RunView) -> None:
    """The recall banner: what Scout remembered before planning."""
    who = ", ".join(view.targets or view.recalled_entities) or "this company"
    if view.recalled_facts:
        st.success(f"🧠 Recalled {view.recalled_facts} facts about {who} from earlier runs")
    elif view.stages.get("recall") != "pending":
        st.caption(f"Cold start: no stored facts about {who} yet.")
    if view.lessons_injected:
        lessons = "\n".join(f"- {escape_md(lesson['text'])}" for lesson in view.lessons_injected)
        st.success(
            f"📚 Using {len(view.lessons_injected)} lessons learned from earlier runs\n\n{lessons}"
        )


def render_stages(view: RunView) -> None:
    """One chip per workflow stage."""
    cols = st.columns(len(view.stages))
    for col, (name, status) in zip(cols, view.stages.items(), strict=True):
        with col:
            st.badge(f"{name}: {status}", color=STATUS_COLORS.get(status, "gray"))


def step_label(step: StepView) -> str:
    """Plan row text."""
    tag = " · added by replan" if step.replanned else ""
    facts = f" · facts {step.fact_ids}" if step.status == "from memory" and step.fact_ids else ""
    return f"**{step.id}.** {escape_md(step.question)}{tag}{facts}"


def render_plan(view: RunView) -> None:
    """Plan panel with a status chip per step; replanned steps highlighted."""
    st.subheader(f"Plan · {purpose_label(view.purpose)}" if view.purpose else "Plan")
    if not view.steps:
        st.caption("Waiting for the plan…")
    for step in view.steps:
        chip, text = st.columns([1.8, 6])
        with chip:
            st.badge(step.status, color=STATUS_COLORS.get(step.status, "gray"))
        with text:
            if step.replanned:
                st.warning(step_label(step), icon="➕")
            else:
                st.markdown(step_label(step))


def render_entries(entries: list[TraceEntry]) -> None:
    """A step's compact timeline: 💭 thought → 🔧 tool call → 👁 result → ⚖️ critic."""
    for entry in entries:
        icon = ENTRY_ICONS.get(entry.kind, "•")
        text = entry.text if len(entry.text) <= 400 else entry.text[:400] + "…"
        text = escape_md(text)
        if entry.kind == "recovered_error":
            st.caption("↻ Model returned malformed JSON; retried.")
        elif entry.kind == "error":
            st.error(f"{icon} {text}")
        elif entry.kind in ("wait", "switch"):
            st.caption(f"{icon} {text}")
        elif entry.kind == "tool":
            st.markdown(f"{icon} `{entry.text[:300]}`")
        else:
            st.markdown(f"{icon} {text}")


def render_trace(view: RunView) -> None:
    """One expander per step titled with its one-line summary, plus run-level events.

    While a run streams (or a replay animates) the active step is open; afterwards, steps with a
    retry, replan, memory hit or unknown verdict are open.
    """
    st.subheader("Trace")
    for step in view.steps:
        if not step.entries:
            continue
        expanded = step.notable if view.finished else step.id == view._current
        with st.expander(step.summary, expanded=expanded):
            st.caption(escape_md(step.question))
            render_entries(step.entries)
    if view.run_entries:
        with st.expander(f"Run events ({len(view.run_entries)})"):
            render_entries(view.run_entries)


def render_metrics(view: RunView) -> None:
    """Metrics row; the fixed-format score is the first and largest tile (omitted if none)."""
    m = view.metrics
    cells = [
        ("Tool calls", m.get("tool_calls")),
        ("LLM calls", m.get("llm_calls")),
        ("Tokens", m.get("total_tokens")),
        ("Seconds", m.get("latency_s")),
        ("Citation coverage", m.get("citation_coverage", 0)),
        ("Memory steps", m.get("memory_steps", 0)),
        ("Provider switches", m.get("provider_switches", len(view.switches))),
    ]
    widths = ([2.2] if view.score else []) + [1] * len(cells)
    cols = st.columns(widths)
    if view.score:
        cols[0].metric("Score", view.score, border=True)
        cols = cols[1:]
    for col, (label, value) in zip(cols, cells, strict=True):
        col.metric(label, fmt_metric(label, value))


def _feedback_changed(cfg: Settings, run_id: str, key: str, title: str, final: dict) -> None:
    """st.feedback callback: store a thumbs up/down for one section."""
    value = st.session_state.get(key)
    if value is None:
        return
    store = open_store(cfg)
    try:
        result = apply_feedback(store, run_id, final, title, value == 1)
    finally:
        store.close()
    if result:
        _, domains, lessons = result
        st.session_state["feedback_toast"] = (
            f"Saved: {len(domains)} sources and {len(lessons)} lessons updated"
        )


def render_brief(view: RunView, cfg: Settings, interactive: bool) -> None:
    """The brief with clickable citations and a 👍/👎 pair beside each section heading."""
    st.subheader("Brief")
    final = {"brief": view.brief, "evidence": view.evidence}
    ready = presentable(final) if view.brief else None
    if view.status == "failed" or (ready is None and not view.report_md):
        st.warning("No brief was produced for this run.")
        return
    if ready is None:  # very old traces: only the rendered report exists
        st.markdown(demote(linkify(escape_md(view.report_md))))
        return
    parts = render_parts(*ready, view.run_id)
    link = citation_linker(parts.tail)
    st.markdown(demote(link(parts.header)))
    for i, (title, body) in enumerate(parts.sections):
        if interactive:
            thumbs, heading = st.columns([0.55, 9.45], vertical_alignment="center", gap="small")
            heading.markdown(f"#### {escape_md(title)}")
            key = f"fb-{view.run_id}-{i}"
            with thumbs:
                st.feedback(
                    "thumbs",
                    key=key,
                    on_change=_feedback_changed,
                    args=(cfg, view.run_id, key, title, final),
                )
        else:
            st.markdown(f"#### {escape_md(title)}")
        st.markdown(link(body))
    st.markdown(demote(parts.tail))
    message = st.session_state.pop("feedback_toast", None)
    if message:
        st.toast(message)


def render_view(view: RunView, cfg: Settings, badge: str | None, interactive: bool) -> None:
    """Draw a whole run (used while streaming and for the final view)."""
    render_badge(badge)
    st.markdown(f"**Goal:** {escape_md(view.goal) if view.goal else '…'}")
    render_recall(view)
    render_stages(view)
    left, right = st.columns([2, 3])
    with left:
        render_plan(view)
    with right:
        render_trace(view)
    if view.finished:
        render_metrics(view)
        render_brief(view, cfg, interactive)


def replay_badge(run: SavedRun) -> str:
    """The replay label (a replay must never pass as a live run)."""
    recorded = run.recorded.replace("T", " ")
    return f"Replay of recorded run {run.run_id}, recorded {recorded}"


def show_replay(run: SavedRun, delay: float, cfg: Settings) -> None:
    """Replay a saved run: animate once when a delay is set, then show the final view."""
    events = load_events(run)
    badge = replay_badge(run)
    animated = st.session_state.setdefault("animated", set())
    # One placeholder for every frame and for the final view: nothing from a previously shown
    # run lingers (faded) below the animation.
    placeholder = st.empty()
    if delay > 0 and run.run_id not in animated:
        view = RunView()
        for event in events:
            reduce(view, event)
            if event.type in QUIET_EVENTS:
                continue
            with placeholder.container():
                render_view(view, cfg, badge, interactive=False)
            time.sleep(delay)
        animated.add(run.run_id)
    with placeholder.container():
        render_view(build_view(events), cfg, badge, interactive=True)


def run_live(goal: str, cfg: Settings) -> str | None:
    """Stream a live run into the page; returns its run id."""
    status = st.empty()

    def on_wait(kind: str, payload: dict[str, Any]) -> None:
        if kind == "rate_limited":
            status.info(f"⏳ {wait_line(payload)}")

    orchestrator = Orchestrator(cfg, listener=on_wait)
    recorder = RunRecorder(cfg.runs_dir, orchestrator.run_id)
    badge = f"Live run {orchestrator.run_id} (streaming now)"
    placeholder = st.empty()
    view = RunView()
    for event in recorder.record(orchestrator.run(goal)):
        reduce(view, event)
        if event.type in QUIET_EVENTS:
            continue
        status.empty()
        with placeholder.container():
            render_view(view, cfg, badge, interactive=False)
    placeholder.empty()
    status.empty()
    return orchestrator.run_id


# --- tabs ------------------------------------------------------------------------------------


def run_tab(cfg: Settings, mode: str, picked: SavedRun | None, delay: float) -> None:
    """The Run tab for either mode."""
    if mode == "Replay saved run":
        if picked is None:
            st.info('No saved runs yet. Switch to Live run, or run `python -m scout run "<goal>"`.')
            return
        show_replay(picked, delay, cfg)
        return
    goal = st.text_area(
        "Goal", key="goal", height=80, placeholder="Who should Scout research, and why?"
    )
    if st.button("Run Scout", type="primary", disabled=not goal.strip()):
        st.session_state["live_run_id"] = run_live(goal.strip(), cfg)
    run_id = st.session_state.get("live_run_id")
    if run_id:
        run_dir = cfg.runs_dir / run_id
        events = load_events(SavedRun(run_id, goal, "", "", "", run_dir))
        render_view(
            build_view(events), cfg, f"Live run {run_id} (finished; shown from its trace)", True
        )


def runs_frame(runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Runs (memory-store rows or recorded runs) as a table."""
    rows = []
    for r in runs:
        m = r["metrics"]
        rows.append(
            {
                "run": r["id"],
                "purpose": purpose_label(r["purpose_type"]),
                "target": ", ".join(r["targets"]),
                "tool calls": m.get("tool_calls", 0),
                "cache hits": m.get("cache_hits", 0),
                "memory steps": m.get("memory_steps", 0),
                "LLM calls": m.get("llm_calls", 0),
                "tokens": m.get("total_tokens", 0),
                "seconds": m.get("latency_s", 0),
                "coverage %": m.get("citation_coverage", 0),
            }
        )
    return pd.DataFrame(rows)


COMPARE_KEYS = (
    ("tool calls", "tool_calls"),
    ("LLM calls", "llm_calls"),
    ("tokens", "total_tokens"),
    ("seconds", "latency_s"),
    ("memory steps", "memory_steps"),
    ("facts reused", "facts_reused"),
    ("cache hits", "cache_hits"),
    ("citation coverage %", "citation_coverage"),
    ("provider switches", "provider_switches"),
)


def run_option(r: dict[str, Any]) -> str:
    """Picker label for a run row."""
    return f"{r['id']} · {purpose_label(r['purpose_type'])} · {', '.join(r['targets'])}"


def compare_runs(runs: list[dict[str, Any]]) -> None:
    """Pick two runs and show their metrics side by side with the difference."""
    st.subheader("Compare two runs")
    if len(runs) < 2:
        st.caption("Needs at least two runs.")
        return
    labels = {run_option(r): r for r in runs}
    names = list(labels)
    left, right = st.columns(2)
    a = labels[left.selectbox("Run A", names, index=0, key="cmp-a")]
    b = labels[right.selectbox("Run B", names, index=len(names) - 1, key="cmp-b")]
    rows = []
    for label, key in COMPARE_KEYS:
        va, vb = a["metrics"].get(key, 0) or 0, b["metrics"].get(key, 0) or 0
        rows.append(
            {"metric": label, "Run A": va, "Run B": vb, "change (B − A)": round(vb - va, 1)}
        )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")


def plan_markdown(run: RecordedRun) -> str:
    """A run's goal, purpose, plan (memory and replanned steps marked) and injected lessons."""
    lines = [
        f"**{purpose_label(run.purpose_type)}** · {escape_md(', '.join(run.targets))}",
        "",
        f"_{escape_md(run.goal)}_",
        "",
    ]
    for step in run.steps:
        marks = ""
        if step.get("answered_from_memory"):
            marks += " 🧠 _from memory_"
        if step.get("is_followup"):
            marks += " ➕ _added by replan_"
        lines.append(f"{step.get('id', '?')}. {escape_md(str(step.get('question', '')))}{marks}")
    if run.lessons_injected:
        lines += ["", "**Lessons injected:**"]
        lines += [f"- 📚 {escape_md(t)}" for t in run.lessons_injected]
    return "\n".join(lines)


def compare_plans(recorded: list[RecordedRun]) -> None:
    """Two runs' plans side by side: same company, different purpose -> different plan."""
    st.subheader("Compare plans")
    with_plans = [r for r in recorded if r.steps]
    if len(with_plans) < 2:
        st.caption("Needs at least two recorded runs with a plan.")
        return
    labels = {run_option(r.as_row()): r for r in with_plans}
    names = list(labels)
    left, right = st.columns(2)
    a = labels[left.selectbox("Plan A", names, index=0, key="plan-a")]
    b = labels[right.selectbox("Plan B", names, index=len(names) - 1, key="plan-b")]
    left.markdown(plan_markdown(a))
    right.markdown(plan_markdown(b))


def sources_tables(sources: list[dict[str, Any]]) -> None:
    """Top and bottom 5 sources."""
    top, bottom = st.columns(2)
    worst = sorted(sources, key=lambda s: (s["score"], -s["useless"]))[:5]
    for col, title, rows in (
        (top, "Top 5 sources", sources[:5]),
        (bottom, "Bottom 5 sources", worst),
    ):
        col.markdown(f"**{title}**")
        if rows:
            table = pd.DataFrame(rows).set_index("domain")
            col.table(table.assign(score=table["score"].map(lambda v: f"{v:.2f}")))
        else:
            col.caption("None yet.")


def store_lessons(store: MemoryStore) -> list[dict[str, Any]]:
    """Lessons from the memory database."""
    uses = store.lesson_uses()
    return [
        {
            "purpose": purpose_label(lesson.purpose_type),
            "lesson": lesson.text,
            "votes": f"+{lesson.votes_up}/-{lesson.votes_down}",
            "score": f"{score(lesson.votes_up, lesson.votes_down):.2f}",
            "uses": uses.get(lesson.id or 0, 0),
        }
        for lesson in store.lessons()
    ]


INSIGHT_SOURCES = ("Recorded runs (runs/ + examples/)", "Memory database")


def insights_tab(cfg: Settings) -> None:
    """Runs, charts, lessons, sources, run comparison and plan comparison."""
    recorded = recorded_runs(cfg.runs_dir, cfg.examples_dir)
    store = open_store(cfg)
    try:
        db_runs = store.runs()
        default = 1 if len(db_runs) >= 2 else 0
        source = st.radio(
            "Source", INSIGHT_SOURCES, index=default, horizontal=True, key="insights-source"
        )
        if source == "Memory database":
            runs, sources = db_runs, store.sources()
            lessons = store_lessons(store)
        else:
            runs = [r.as_row() for r in recorded]
            sources = recorded_sources(recorded)
            lessons = [
                {k: r[k] for k in ("purpose", "lesson", "votes", "score", "uses", "learned in")}
                | {"purpose": purpose_label(r["purpose"])}
                for r in recorded_lessons(recorded)
            ]
    finally:
        store.close()
    frame = runs_frame(runs)
    st.subheader("Runs")
    if frame.empty:
        st.caption("No runs yet.")
    else:
        st.dataframe(frame, hide_index=True, width="stretch")
        chart = frame.assign(
            run=[
                f"{r[-4:]} · {p.split()[0].lower()}"
                for r, p in zip(frame["run"], frame["purpose"], strict=True)
            ]
        ).set_index("run")
        c1, c2 = st.columns(2)
        c1.markdown("**Tokens per run**")
        c1.bar_chart(chart[["tokens"]])
        c2.markdown("**Tool calls per run**")
        c2.bar_chart(chart[["tool calls"]])
    st.subheader("Lessons")
    if lessons:
        st.table(pd.DataFrame(lessons))
    else:
        st.caption("No lessons yet.")
    sources_tables(sources)
    compare_runs(runs)
    compare_plans(recorded)


def how_it_works_tab() -> None:
    """Workflow diagram and a short explanation."""
    st.graphviz_chart(WORKFLOW_DOT)
    st.markdown(
        "- **Intake → Plan:** the goal becomes a task with a purpose; the planner writes a "
        "purpose-shaped plan guided by a playbook and by lessons from earlier runs.\n"
        "- **Execute (ReAct):** each step thinks, calls a tool (search, fetch, memory), observes, "
        "and finishes with cited findings.\n"
        "- **Critique:** after each step a critic routes the run: next step, retry with a new "
        "approach, add a follow-up step, or mark it unknown.\n"
        "- **Synthesize → Verify:** the brief cites evidence ids; a deterministic verifier checks "
        "citations and numbers, revises once, and moves anything unsupported to Unknowns.\n"
        "- **Learns with use:** (1) fact memory skips steps already answered, (2) source scores "
        "re-rank search results, (3) Reflexion-style lessons shape future plans.\n"
        "- **Budgets and router:** tool, LLM and wall-clock budgets end runs gracefully; a "
        "provider router fails over between free-tier models."
    )


# --- sidebar and page ------------------------------------------------------------------------


def sidebar(cfg: Settings) -> tuple[str, SavedRun | None, float]:
    """Mode, replay picker, live options, providers, reset. Returns (mode, run, delay)."""
    st.sidebar.title("🔭 Scout")
    live_ok = llm_configured(cfg)
    modes = ["Replay saved run", "Live run"] if live_ok else ["Replay saved run"]
    mode = st.sidebar.radio("Mode", modes, key="mode")
    if not live_ok:
        st.sidebar.info(NO_KEYS_MESSAGE)
    picked: SavedRun | None = None
    delay = 0.0
    if mode == "Replay saved run":
        runs = list_runs(cfg.runs_dir, cfg.examples_dir)
        if runs:
            picked = st.sidebar.selectbox(
                "Recorded run", runs, format_func=lambda r: r.label, key="run"
            )
            st.sidebar.caption(picked.caption)
            st.sidebar.badge(f"REPLAY · {picked.run_id}", color="orange", icon="🔁")
        delay = st.sidebar.slider(
            "Replay speed (seconds per event)", 0.0, 0.5, 0.0, 0.05, key="delay"
        )
    else:
        st.sidebar.markdown("**Example goals**")
        for i, goal in enumerate(HERO_GOALS):
            if st.sidebar.button(goal[:60] + "…", key=f"hero-{i}"):
                st.session_state["goal"] = goal
        st.session_state["max_tool_calls"] = st.sidebar.slider(
            "Max tool calls", 3, 30, 12, key="tools"
        )
        st.session_state["max_steps"] = st.sidebar.slider("Max planned steps", 1, 5, 5, key="steps")
    with st.sidebar.expander("Providers (read-only)"):
        for line in provider_lines(cfg):
            st.markdown(line)
    with st.sidebar.expander("Reset memory"):
        confirm = st.checkbox(
            "I understand this deletes all facts, sources and lessons", key="confirm"
        )
        if st.button("Reset memory", disabled=not confirm, key="reset"):
            removed = reset_memory(cfg.db_path)
            st.toast(f"Memory reset ({len(removed)} files removed)")
    return mode, picked, delay


def main() -> None:
    """Build the page."""
    st.set_page_config(page_title="Scout", page_icon="🔭", layout="wide")
    try:
        cfg = settings()
        mode, picked, delay = sidebar(cfg)
        if mode == "Live run":
            cfg = cfg.with_overrides(
                max_tool_calls=st.session_state.get("max_tool_calls", cfg.max_tool_calls),
                max_planned_steps=st.session_state.get("max_steps", cfg.max_planned_steps),
            )
    except Exception as exc:  # noqa: BLE001 - never show a stack trace
        friendly_error("loading settings", exc)
        return
    st.markdown("### 🔭 Scout")
    st.caption("Purpose-aware company research that cites its sources and learns with use.")
    run_t, insights_t, how_t = st.tabs(["Run", "Insights", "How it works"])
    with run_t:
        try:
            run_tab(cfg, mode, picked, delay)
        except Exception as exc:  # noqa: BLE001
            friendly_error("showing this run", exc)
    with insights_t:
        try:
            insights_tab(cfg)
        except Exception as exc:  # noqa: BLE001
            friendly_error("loading insights", exc)
    with how_t:
        how_it_works_tab()


if __name__ == "__main__":
    main()
