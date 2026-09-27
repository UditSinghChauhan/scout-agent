"""Scout UI: live runs, replays of recorded runs, insights and a how-it-works page.

Launch from the repo root:  python -m scout ui   (or: streamlit run app/streamlit_app.py)
Replay mode is the default and makes no LLM or network calls.
"""

from __future__ import annotations

import logging
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from scout.agent.orchestrator import Orchestrator  # noqa: E402
from scout.config import (  # noqa: E402
    PROVIDERS,
    ROUTES,
    Settings,
    load_settings,
    provider_key,
    router_enabled,
)
from scout.events import RunRecorder  # noqa: E402
from scout.insights import apply_feedback  # noqa: E402
from scout.memory.store import MemoryStore, reset_memory, score  # noqa: E402
from scout.ui.runs import SavedRun, list_runs, load_events  # noqa: E402
from scout.ui.viewmodel import RunView, StepView, TraceEntry, build_view, reduce  # noqa: E402

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
    "error": "red",
}
ENTRY_ICONS = {
    "thought": "💭",
    "tool": "🔧",
    "result": "📄",
    "critique": "🧐",
    "retry": "🔁",
    "switch": "🔀",
    "error": "⚠️",
    "memory": "🧠",
}
QUIET_EVENTS = {"llm_call"}  # not worth a re-render during streaming
WORKFLOW_DOT = """
digraph scout {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", fontname="Helvetica"];
  Intake -> Recall -> Plan -> Execute -> Critique;
  Critique -> Execute [label="next step / retry"];
  Critique -> Plan [label="follow-up (replan)"];
  Critique -> Synthesize [label="done or budget"];
  Synthesize -> Verify;
  Verify -> Synthesize [label="1 revision"];
  Verify -> Reflect;
  Memory [shape=cylinder, fillcolor="#f3eefb", label="Memory\\nfacts · sources · lessons"];
  Memory -> Recall [style=dashed]; Reflect -> Memory [style=dashed];
}
"""


# --- helpers ---------------------------------------------------------------------------------


def settings() -> Settings:
    """Current settings (read on every rerun so .env changes are picked up)."""
    return load_settings()


def open_store(cfg: Settings) -> MemoryStore:
    """A fresh store connection for this rerun."""
    return MemoryStore(cfg.db_path)


def linkify(report_md: str) -> str:
    """Turn numbered citations [n] into links to the matching source URL."""
    body, sep, tail = report_md.partition("\n## Sources")
    sources = dict(re.findall(r"^(\d+)\. <(.+)>$", tail, re.M))

    def link(match: re.Match[str]) -> str:
        num = match.group(1)
        return f"[[{num}]]({sources[num]})" if num in sources else match.group(0)

    return re.sub(r"\[(\d+)\]", link, body) + sep + tail


def provider_lines(cfg: Settings) -> list[str]:
    """Configured providers and models (never keys)."""
    if router_enabled(cfg):
        lines = []
        for tier, candidates in ROUTES.items():
            usable = [c.label for c in candidates if provider_key(cfg, PROVIDERS[c.provider])]
            lines.append(f"**{tier}**: " + (" → ".join(usable) or "none configured"))
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
        lessons = "\n".join(f"- {lesson['text']}" for lesson in view.lessons_injected)
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
    tag = " · replanned" if step.replanned else ""
    facts = f" · facts {step.fact_ids}" if step.status == "from memory" and step.fact_ids else ""
    return f"**{step.id}.** {step.question}{tag}{facts}"


def render_plan(view: RunView) -> None:
    """Plan panel with a status chip per step; replanned steps highlighted."""
    st.subheader("Plan")
    if not view.steps:
        st.caption("Waiting for the plan…")
    for step in view.steps:
        chip, text = st.columns([1, 6])
        with chip:
            st.badge(step.status, color=STATUS_COLORS.get(step.status, "gray"))
        with text:
            if step.replanned:
                st.warning(step_label(step), icon="➕")
            else:
                st.markdown(step_label(step))


def render_entries(entries: list[TraceEntry]) -> None:
    """Lines of a trace."""
    for entry in entries:
        icon = ENTRY_ICONS.get(entry.kind, "•")
        text = entry.text if len(entry.text) <= 400 else entry.text[:400] + "…"
        if entry.kind == "error":
            st.error(f"{icon} {text}")
        else:
            st.markdown(f"{icon} {text}")


def render_trace(view: RunView, expanded_step: int | None = None) -> None:
    """One expander per step, plus run-level events."""
    st.subheader("Trace")
    for step in view.steps:
        if not step.entries:
            continue
        with st.expander(
            f"Step {step.id} · {step.status} · {step.question}", expanded=step.id == expanded_step
        ):
            render_entries(step.entries)
    if view.run_entries:
        with st.expander(f"Run events ({len(view.run_entries)})"):
            render_entries(view.run_entries)


def render_metrics(view: RunView) -> None:
    """Headline metrics row."""
    m = view.metrics
    cells = [
        ("Tool calls", m.get("tool_calls")),
        ("LLM calls", m.get("llm_calls")),
        ("Tokens", m.get("total_tokens")),
        ("Seconds", m.get("latency_s")),
        ("Citation coverage", f"{m.get('citation_coverage', 0)}%"),
        ("Memory steps", m.get("memory_steps", 0)),
        ("Provider switches", m.get("provider_switches", len(view.switches))),
    ]
    for col, (label, value) in zip(st.columns(len(cells)), cells, strict=True):
        col.metric(label, "–" if value is None else value)


def render_feedback(view: RunView, cfg: Settings) -> None:
    """Thumbs up/down per brief section, stored via the feedback function."""
    if not view.brief:
        return
    st.markdown("**Was this section useful?**")
    for i, section in enumerate(view.brief.get("sections", [])):
        title, up, down = st.columns([6, 1, 1])
        title.markdown(section.get("title", f"Section {i + 1}"))
        for col, is_up, icon in ((up, True, "👍"), (down, False, "👎")):
            if col.button(icon, key=f"fb-{view.run_id}-{i}-{is_up}"):
                store = open_store(cfg)
                try:
                    final = {"brief": view.brief, "evidence": view.evidence}
                    result = apply_feedback(store, view.run_id, final, section["title"], is_up)
                finally:
                    store.close()
                if result:
                    _, domains, lessons = result
                    st.toast(f"Saved: {len(domains)} sources and {len(lessons)} lessons updated")


def render_brief(view: RunView, cfg: Settings, interactive: bool) -> None:
    """Score, brief with clickable citations, feedback."""
    st.subheader("Brief")
    if view.status == "failed" or not view.report_md:
        st.warning("No brief was produced for this run.")
        return
    if view.score:
        st.metric("Score", view.score)
    st.markdown(linkify(view.report_md))
    if interactive:
        render_feedback(view, cfg)


def render_view(view: RunView, cfg: Settings, badge: str | None, interactive: bool) -> None:
    """Draw a whole run (used while streaming and for the final view)."""
    render_badge(badge)
    st.markdown(f"**Goal:** {view.goal or '…'}")
    render_recall(view)
    render_stages(view)
    left, right = st.columns([2, 3])
    with left:
        render_plan(view)
    with right:
        render_trace(view, expanded_step=None if view.finished else view._current)
    if view.finished:
        render_metrics(view)
        render_brief(view, cfg, interactive)


def replay_badge(run: SavedRun) -> str:
    """The replay label (the video must never pass a replay off as live)."""
    recorded = run.recorded.replace("T", " ")
    return f"Replay of recorded run {run.run_id}, recorded {recorded}"


def show_replay(run: SavedRun, delay: float, cfg: Settings) -> None:
    """Replay a saved run: animate once when a delay is set, then show the final view."""
    events = load_events(run)
    badge = replay_badge(run)
    animated = st.session_state.setdefault("animated", set())
    if delay > 0 and run.run_id not in animated:
        placeholder = st.empty()
        view = RunView()
        for event in events:
            reduce(view, event)
            if event.type in QUIET_EVENTS:
                continue
            with placeholder.container():
                render_view(view, cfg, badge, interactive=False)
            time.sleep(delay)
        placeholder.empty()
        animated.add(run.run_id)
    render_view(build_view(events), cfg, badge, interactive=True)


def run_live(goal: str, cfg: Settings) -> str | None:
    """Stream a live run into the page; returns its run id."""
    orchestrator = Orchestrator(cfg)
    recorder = RunRecorder(cfg.runs_dir, orchestrator.run_id)
    badge = f"Live run {orchestrator.run_id} (streaming now)"
    placeholder = st.empty()
    view = RunView()
    for event in recorder.record(orchestrator.run(goal)):
        reduce(view, event)
        if event.type in QUIET_EVENTS:
            continue
        with placeholder.container():
            render_view(view, cfg, badge, interactive=False)
    placeholder.empty()
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


def runs_frame(store: MemoryStore) -> pd.DataFrame:
    """Runs from the memory store as a table."""
    rows = []
    for r in store.runs():
        m = r["metrics"]
        rows.append(
            {
                "run": r["id"],
                "purpose": r["purpose_type"],
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


def compare_runs(runs: list[dict[str, Any]]) -> None:
    """Pick two runs and show their metrics side by side with the difference."""
    st.subheader("Compare two runs")
    if len(runs) < 2:
        st.caption("Needs at least two runs in memory.")
        return
    labels = {f"{r['id']} · {r['purpose_type']} · {', '.join(r['targets'])}": r for r in runs}
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


def insights_tab(cfg: Settings) -> None:
    """Runs, charts, lessons, sources and the run comparison."""
    store = open_store(cfg)
    try:
        frame = runs_frame(store)
        st.subheader("Runs")
        if frame.empty:
            st.caption("No runs in memory yet.")
        else:
            st.dataframe(frame, hide_index=True, width="stretch")
            chart = frame.set_index("run")
            c1, c2 = st.columns(2)
            c1.markdown("**Tokens per run**")
            c1.bar_chart(chart[["tokens"]])
            c2.markdown("**Tool calls per run**")
            c2.bar_chart(chart[["tool calls"]])
        st.subheader("Lessons")
        uses = store.lesson_uses()
        lessons = [
            {
                "id": lesson.id,
                "purpose": lesson.purpose_type,
                "lesson": lesson.text,
                "votes": f"+{lesson.votes_up}/-{lesson.votes_down}",
                "score": round(score(lesson.votes_up, lesson.votes_down), 2),
                "uses": uses.get(lesson.id or 0, 0),
            }
            for lesson in store.lessons()
        ]
        if lessons:
            st.dataframe(pd.DataFrame(lessons), hide_index=True, width="stretch")
        else:
            st.caption("No lessons yet.")
        sources = store.sources()
        top, bottom = st.columns(2)
        worst = sorted(sources, key=lambda s: (s["score"], -s["useless"]))[:5]
        for col, title, rows in (
            (top, "Top 5 sources", sources[:5]),
            (bottom, "Bottom 5 sources", worst),
        ):
            col.markdown(f"**{title}**")
            if rows:
                col.dataframe(pd.DataFrame(rows), hide_index=True)
            else:
                col.caption("None yet.")
        compare_runs(store.runs())
    finally:
        store.close()


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
    mode = st.sidebar.radio("Mode", ["Replay saved run", "Live run"], key="mode")
    picked: SavedRun | None = None
    delay = 0.0
    if mode == "Replay saved run":
        runs = list_runs(cfg.runs_dir)
        if runs:
            picked = st.sidebar.selectbox(
                "Recorded run", runs, format_func=lambda r: r.label, key="run"
            )
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
    st.title("Scout")
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
