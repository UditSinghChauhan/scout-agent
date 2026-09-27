"""View model for the UI: reduce a run's event stream into a ``RunView``.

The same reducer serves live runs (events arrive one by one) and replays (events read back from
``trace.jsonl``). It is pure Python and tolerant: traces from older phases with missing fields
reduce without errors.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from scout.events import Event
from scout.scoring import normalize_score

STAGES = ("intake", "recall", "plan", "execute", "synthesize", "verify", "reflect")
_STAGE_ALIASES = {"critique": "execute"}  # critique alternates with execution per step
STEP_STATUSES = ("pending", "running", "done", "from memory", "retried", "unknown")


@dataclass
class TraceEntry:
    """One line in a step's trace."""

    kind: str  # thought | tool | result | critique | retry | switch | error | memory | wait
    text: str
    ok: bool | None = None


@dataclass
class StepView:
    """A plan step and everything that happened to it."""

    id: int
    question: str
    status: str = "pending"
    replanned: bool = False
    fact_ids: list[int] = field(default_factory=list)
    retries: int = 0
    verdict: str | None = None
    entries: list[TraceEntry] = field(default_factory=list)


@dataclass
class RunView:
    """Everything the UI needs to draw one run."""

    run_id: str = ""
    goal: str = ""
    started: str = ""
    purpose: str = ""
    targets: list[str] = field(default_factory=list)
    stages: dict[str, str] = field(default_factory=lambda: dict.fromkeys(STAGES, "pending"))
    steps: list[StepView] = field(default_factory=list)
    recalled_facts: int = 0
    recalled_entities: list[str] = field(default_factory=list)
    lessons_injected: list[dict[str, Any]] = field(default_factory=list)
    lessons_learned: list[dict[str, Any]] = field(default_factory=list)
    memory_hits: int = 0
    switches: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    run_entries: list[TraceEntry] = field(default_factory=list)
    verification: dict[str, Any] = field(default_factory=dict)
    finished: bool = False
    status: str = "running"
    brief: dict[str, Any] | None = None
    report_md: str = ""
    evidence: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    events: int = 0
    _current: int | None = None

    def step(self, step_id: Any) -> StepView | None:
        """Step by id (None if unknown)."""
        try:
            wanted = int(step_id)
        except (TypeError, ValueError):
            return None
        return next((s for s in self.steps if s.id == wanted), None)

    @property
    def score(self) -> str | None:
        """The brief's score in the fixed per-playbook format (None if not valid)."""
        brief = self.brief or {}
        purpose = self.purpose or str(brief.get("purpose_type", ""))
        return normalize_score(purpose, brief.get("score"), brief.get("score_reasons") or [])


def _lesson(item: Any) -> dict[str, Any]:
    """Normalise a lesson entry (older traces stored plain strings)."""
    if isinstance(item, dict):
        return {"id": item.get("id"), "text": str(item.get("text", ""))}
    return {"id": None, "text": str(item)}


def _mark_stage(view: RunView, stage: str) -> None:
    """Mark ``stage`` running and every earlier stage done."""
    stage = _STAGE_ALIASES.get(stage, stage)
    if stage not in view.stages:
        return
    for name in STAGES:
        if name == stage:
            if view.stages[name] != "error":
                view.stages[name] = "running"
            break
        if view.stages[name] in ("pending", "running"):
            view.stages[name] = "done"


def _add_steps(view: RunView, steps: Iterable[Any]) -> None:
    """Create step views from a plan payload."""
    for raw in steps:
        if not isinstance(raw, dict) or view.step(raw.get("id")) is not None:
            continue
        view.steps.append(
            StepView(
                id=int(raw.get("id", len(view.steps) + 1)),
                question=str(raw.get("question", "")),
                replanned=bool(raw.get("is_followup", False)),
                fact_ids=list(raw.get("memory_fact_ids") or []),
            )
        )


def _entry_target(view: RunView, payload: dict[str, Any]) -> list[TraceEntry]:
    """Trace list for an event: its step if known, else the current step, else the run."""
    step = view.step(payload.get("step_id")) or view.step(view._current)
    return step.entries if step else view.run_entries


def reduce(view: RunView, event: Event) -> RunView:
    """Apply one event to the view (mutates and returns it)."""
    p = event.payload or {}
    view.events += 1
    view.run_id = view.run_id or event.run_id
    view.started = view.started or event.ts.isoformat(timespec="seconds")
    _mark_stage(view, event.stage)
    handler = _HANDLERS.get(event.type)
    if handler:
        handler(view, p, event)
    return view


def _on_stage_started(view: RunView, p: dict[str, Any], event: Event) -> None:
    if event.stage == "intake":
        view.goal = str(p.get("goal", view.goal))
    elif event.stage == "plan" and isinstance(p.get("task"), dict):
        view.purpose = str(p["task"].get("purpose_type", view.purpose))
        view.targets = list(p["task"].get("targets", view.targets))
    elif event.stage == "recall":
        view.recalled_facts = int(p.get("fresh_facts", 0) or 0)
        view.recalled_entities = list(p.get("entities", []))
        view.lessons_injected = [_lesson(x) for x in p.get("lessons", [])]


def _on_plan_created(view: RunView, p: dict[str, Any], event: Event) -> None:
    view.purpose = str(p.get("purpose_type", view.purpose))
    _add_steps(view, p.get("steps", []))
    if "lessons" in p:
        view.lessons_injected = [_lesson(x) for x in p.get("lessons", [])]


def _on_step_started(view: RunView, p: dict[str, Any], event: Event) -> None:
    _add_steps(view, [p])
    step = view.step(p.get("id"))
    if step is None:
        return
    view._current = step.id
    if p.get("approach"):
        step.retries += 1
        step.entries.append(TraceEntry("retry", f"Retry with new approach: {p['approach']}"))
    step.status = "running"


def _on_memory_hit(view: RunView, p: dict[str, Any], event: Event) -> None:
    _add_steps(view, [{"id": p.get("step_id"), "question": p.get("question", "")}])
    step = view.step(p.get("step_id"))
    view.memory_hits += 1
    if step is None:
        return
    step.status = "from memory"
    step.fact_ids = list(p.get("fact_ids", step.fact_ids))
    sources = ", ".join(p.get("sources", []))
    step.entries.append(
        TraceEntry("memory", f"Answered from memory: facts {step.fact_ids} ({sources})", True)
    )


def _on_thought(view: RunView, p: dict[str, Any], event: Event) -> None:
    if p.get("text"):
        _entry_target(view, p).append(TraceEntry("thought", str(p["text"])))


def _on_tool_call(view: RunView, p: dict[str, Any], event: Event) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in (p.get("args") or {}).items())
    _entry_target(view, p).append(TraceEntry("tool", f"{p.get('tool', '?')}({args})"))


def _on_tool_result(view: RunView, p: dict[str, Any], event: Event) -> None:
    text = str(p.get("summary", ""))
    if p.get("repeated"):
        text = f"(repeat, not re-run) {text}"
    _entry_target(view, p).append(TraceEntry("result", text, bool(p.get("ok", True))))


def _on_critique(view: RunView, p: dict[str, Any], event: Event) -> None:
    step = view.step(p.get("step_id"))
    verdict = str(p.get("verdict", ""))
    text = f"Critic: {verdict} — {p.get('reason', '')}"
    ratings = p.get("source_ratings") or {}
    if ratings:
        text += " | sources: " + ", ".join(f"{u} {r}" for u, r in ratings.items())
    _entry_target(view, p).append(TraceEntry("critique", text, verdict != "retry"))
    if step is None:
        return
    step.verdict = verdict
    if verdict == "unknown":
        step.status = "unknown"
    elif verdict == "retry":
        step.status = "running"  # becomes "retried" only if a retry event follows
    else:
        step.status = "retried" if step.retries else "done"


def _on_retry(view: RunView, p: dict[str, Any], event: Event) -> None:
    step = view.step(p.get("step_id"))
    if step is not None:
        step.status = "retried"


def _on_replan(view: RunView, p: dict[str, Any], event: Event) -> None:
    added = p.get("added_step")
    if isinstance(added, dict):
        _add_steps(view, [{**added, "is_followup": True}])
        view.run_entries.append(
            TraceEntry("critique", f"Replan: added step {added.get('id')}: {added.get('question')}")
        )


def _on_switch(view: RunView, p: dict[str, Any], event: Event) -> None:
    view.switches.append(dict(p))
    route = f"{p.get('from')} → {p.get('to')}"
    text = f"Provider switch ({p.get('tier', '?')}): {route} ({p.get('reason', '')})"
    target = _entry_target(view, p) if event.stage == "execute" else view.run_entries
    target.append(TraceEntry("switch", text))


def wait_line(p: dict[str, Any]) -> str:
    """Status line for a rate-limit wait, e.g. "Waiting 12 s: rate limit on qwen3.8-27b"."""
    model = str(p.get("model") or "the model").split("/")[-1]
    return f"Waiting {float(p.get('wait_s', 0) or 0):.0f} s: rate limit on {model}"


def _on_rate_limited(view: RunView, p: dict[str, Any], event: Event) -> None:
    target = _entry_target(view, p) if event.stage == "execute" else view.run_entries
    target.append(TraceEntry("wait", wait_line(p)))


def _on_error(view: RunView, p: dict[str, Any], event: Event) -> None:
    message = str(p.get("message", "error"))
    view.errors.append(message)
    _entry_target(view, p).append(TraceEntry("error", message, False))


def _on_verification(view: RunView, p: dict[str, Any], event: Event) -> None:
    view.verification = dict(p)


def _on_lesson_learned(view: RunView, p: dict[str, Any], event: Event) -> None:
    view.lessons_learned.append(
        {"id": p.get("lesson_id"), "text": p.get("text", ""), "merged": p.get("merged", False)}
    )


def _on_run_finished(view: RunView, p: dict[str, Any], event: Event) -> None:
    view.finished = True
    view.status = str(p.get("status", "ok"))
    view.report_md = str(p.get("report_md") or "")
    view.brief = p.get("brief") if isinstance(p.get("brief"), dict) else None
    view.evidence = list(p.get("evidence") or [])
    view.metrics = dict(p.get("metrics") or {})
    task = p.get("task")
    if isinstance(task, dict):
        view.goal = str(task.get("goal", view.goal))
        view.purpose = str(task.get("purpose_type", view.purpose))
        view.targets = list(task.get("targets", view.targets))
    plan = p.get("plan")
    if isinstance(plan, dict):
        _add_steps(view, plan.get("steps", []))
    for step in view.steps:
        if step.status == "running":
            # A retry verdict the run could not honour (cap reached) ends unresolved.
            step.status = "unknown" if step.verdict == "retry" else "done"
    for name, status in view.stages.items():
        if status == "running":
            view.stages[name] = "done"
    view._current = None


_HANDLERS = {
    "stage_started": _on_stage_started,
    "plan_created": _on_plan_created,
    "step_started": _on_step_started,
    "memory_hit": _on_memory_hit,
    "thought": _on_thought,
    "tool_call": _on_tool_call,
    "tool_result": _on_tool_result,
    "critique": _on_critique,
    "retry": _on_retry,
    "replan": _on_replan,
    "provider_switched": _on_switch,
    "rate_limited": _on_rate_limited,
    "error": _on_error,
    "verification": _on_verification,
    "lesson_learned": _on_lesson_learned,
    "run_finished": _on_run_finished,
}


def build_view(events: Iterable[Event]) -> RunView:
    """Reduce a whole event sequence."""
    view = RunView()
    for event in events:
        reduce(view, event)
    return view
