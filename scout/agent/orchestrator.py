"""Orchestrator: Scout's explicit state machine, written as an event generator (docs/SPEC.md §4).

Flow: Intake -> (Recall) -> Plan -> [Execute -> Critique]* -> Synthesize -> Verify -> (Reflect).
The critic routes each step: ``complete`` moves on, ``retry`` re-runs the step once with the
critic's new approach, ``followup`` appends one new step (replan, max ``max_followups`` per run),
``unknown`` records the question for the Unknowns section. Budgets from config are enforced
throughout; on a budget hit Scout synthesizes from the evidence it has. Exceptions become
``error`` events; ``run`` always ends with ``run_finished``. Router notices (provider switches,
per-call provider/model) are turned into ``provider_switched`` and ``llm_call`` events.
"""

from __future__ import annotations

import logging
import uuid
from collections import Counter
from collections.abc import Generator, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

from scout.agent.budget import Budget
from scout.agent.critic import critique_step
from scout.agent.executor import EvidenceLedger, execute_step
from scout.agent.intake import run_intake
from scout.agent.planner import fallback_plan, make_plan
from scout.agent.reflector import reflect
from scout.agent.synthesizer import citation_stats, fallback_brief, render_markdown, synthesize
from scout.agent.verifier import (
    apply_revisions,
    find_issues,
    move_to_unknowns,
    revise_claims,
)
from scout.config import Settings
from scout.events import Event, EventType
from scout.llm import LLM, DeadlineExceededError
from scout.memory.store import MemoryStore, domain_of, normalize_entity, now_utc
from scout.playbooks import Playbook, get_playbook
from scout.safety import scrub_brief
from scout.schemas import (
    Brief,
    Critique,
    Evidence,
    Fact,
    MemoryContext,
    ResearchPlan,
    RunMetrics,
    Step,
    StepResult,
    Task,
)
from scout.textutil import dedupe
from scout.tools.registry import ToolRegistry, build_registry

logger = logging.getLogger(__name__)
T = TypeVar("T")
UNKNOWN_OVERLAP = 0.5  # an Unknowns line sharing half its keywords with an earlier one is a repeat


def new_run_id() -> str:
    """Sortable, unique run id such as 20260927-071503-a1b2."""
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:4]}"


@dataclass
class RunState:
    """Everything the state machine carries between stages."""

    run_id: str
    goal: str
    task: Task | None = None
    playbook: Playbook | None = None
    memory: MemoryContext = field(default_factory=MemoryContext)
    plan: ResearchPlan | None = None
    results: list[StepResult] = field(default_factory=list)
    ledger: EvidenceLedger = field(default_factory=EvidenceLedger)
    critiques: list[Critique] = field(default_factory=list)
    critic_unknowns: list[str] = field(default_factory=list)
    retries: int = 0
    followups: int = 0
    brief: Brief | None = None
    report_md: str = ""
    flagged_numeric: list[str] = field(default_factory=list)
    unsupported: int = 0
    revised: int = 0
    memory_steps: int = 0
    facts_reused: int = 0
    step_log: list[dict[str, Any]] = field(default_factory=list)
    stage: str = "intake"


class Orchestrator:
    """Runs one research goal end to end, yielding Events."""

    def __init__(
        self,
        settings: Settings,
        llm: LLM | None = None,
        registry: ToolRegistry | None = None,
        run_id: str | None = None,
        store: MemoryStore | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or LLM(settings=settings)
        self.store = store if store is not None else MemoryStore(settings.db_path)
        self.registry = registry or build_registry(settings, store=self.store)
        self.run_id = run_id or new_run_id()
        self.started_at = now_utc()
        self.clock = self.llm._clock
        self.budget = Budget(settings, self.llm, clock=self.clock)
        self._started = self.clock()
        self._switches = 0
        # A4: backoff waits never overshoot the wall clock.
        self.llm.deadline = self._started + settings.max_wall_clock_s

    # --- events -----------------------------------------------------------------------------

    def _event(self, stage: str, type_: EventType, payload: dict[str, Any] | None = None) -> Event:
        """Create an event for this run."""
        return Event(run_id=self.run_id, stage=stage, type=type_, payload=payload or {})

    def _notices(self, stage: str) -> Iterator[Event]:
        """Turn pending LLM notices (calls, provider switches) into events."""
        for type_, payload in self.llm.drain_notices():
            if type_ == "provider_switched":
                self._switches += 1
            yield self._event(stage, type_, payload)  # type: ignore[arg-type]

    def _pump(self, gen: Generator[Event, None, T], stage: str) -> Generator[Event, None, T]:
        """Relay a stage generator's events, inserting LLM notices; return its value."""
        while True:
            try:
                event = next(gen)
            except StopIteration as stop:
                yield from self._notices(stage)
                return stop.value
            yield from self._notices(stage)
            yield event

    # --- run --------------------------------------------------------------------------------

    def run(self, goal: str, user_context: str | None = None) -> Iterator[Event]:
        """Execute the workflow; always ends with a ``run_finished`` event."""
        state = RunState(run_id=self.run_id, goal=goal)
        try:
            yield from self._intake(state, user_context)
            if state.task is not None:
                yield from self._recall(state)
                yield from self._plan(state)
                yield from self._execute(state)
                yield from self._synthesize(state)
                yield from self._verify(state)
                yield from self._reflect(state)
        except Exception as exc:  # noqa: BLE001 - never crash the run
            logger.exception("Unhandled error in stage %s", state.stage)
            yield self._event(state.stage, "error", {"message": f"{type(exc).__name__}: {exc}"})
        yield from self._notices(state.stage)
        yield self._finish(state)

    # --- stages -----------------------------------------------------------------------------

    def _intake(self, state: RunState, user_context: str | None) -> Iterator[Event]:
        state.stage = "intake"
        yield self._event("intake", "stage_started", {"goal": state.goal})
        try:
            state.task = run_intake(self.llm, state.goal, user_context)
        except Exception as exc:  # noqa: BLE001
            yield from self._notices("intake")
            yield self._event("intake", "error", {"message": f"Intake failed: {exc}"})
            return
        yield from self._notices("intake")
        state.playbook = get_playbook(state.task.purpose_type)

    def _recall(self, state: RunState) -> Iterator[Event]:
        """Load fresh facts for the targets, the top lessons for the purpose and source scores."""
        state.stage = "recall"
        assert state.task is not None
        s = self.settings
        fresh: list[Fact] = []
        stale: list[Fact] = []
        for target in state.task.targets:
            f, st = self.store.recall(
                normalize_entity(target),
                s.fact_ttl_days,
                s.news_ttl_days,
                s.memory_max_facts - len(fresh),
                s.memory_max_chars,
            )
            fresh += f
            stale += st
        lessons = self.store.top_lessons(state.task.purpose_type, s.lessons_top_n)
        self.store.mark_used(lesson.id for lesson in lessons if lesson.id is not None)
        state.memory = MemoryContext(
            fresh_facts=fresh,
            stale_facts=stale,
            lessons=lessons,
            source_scores=self.store.source_scores(),
        )
        yield self._event(
            "recall",
            "stage_started",
            {
                "entities": [normalize_entity(t) for t in state.task.targets],
                "fresh_facts": len(fresh),
                "stale_facts": len(stale),
                "lessons": [{"id": lesson.id, "text": lesson.text} for lesson in lessons],
                "known_sources": len(state.memory.source_scores),
            },
        )

    def _plan(self, state: RunState) -> Iterator[Event]:
        state.stage = "plan"
        assert state.task is not None and state.playbook is not None
        yield self._event("plan", "stage_started", {"task": state.task.model_dump()})
        try:
            state.plan = make_plan(
                self.llm,
                state.task,
                state.memory,
                state.playbook,
                self.settings,
                self.registry.describe(),
            )
        except Exception as exc:  # noqa: BLE001
            yield from self._notices("plan")
            yield self._event(
                "plan", "error", {"message": f"Planner failed, using fallback: {exc}"}
            )
            state.plan = fallback_plan(state.task, state.playbook, self.settings.max_planned_steps)
        yield from self._notices("plan")
        yield self._event(
            "plan",
            "plan_created",
            {
                "purpose_type": state.task.purpose_type,
                "steps": [s.model_dump() for s in state.plan.steps],
                "lessons": [
                    {"id": lesson.id, "text": lesson.text} for lesson in state.memory.lessons
                ],
            },
        )

    def _budget_ok(self) -> bool:
        """True if another step (LLM + tool calls) fits the budget."""
        return self.budget.llm_available() and self.budget.tool_available()

    def _execute(self, state: RunState) -> Iterator[Event]:
        state.stage = "execute"
        assert state.plan is not None
        steps: list[Step] = list(state.plan.steps[: self.settings.max_total_steps])
        index = 0
        while index < len(steps):
            step = steps[index]
            if step.answered_from_memory:
                yield from self._memory_step(state, step)
                index += 1
                continue
            if not self._budget_ok():
                break
            approach: str | None = None
            for attempt in range(1 + self.settings.max_retries_per_step):
                result = yield from self._run_step(state, step, approach)
                if result is None:
                    return  # wall clock hit inside the step
                critique = yield from self._critique(state, step, result)
                if critique is None or critique.verdict == "complete":
                    break
                if critique.verdict == "unknown":
                    state.critic_unknowns.append(step.question)
                    break
                if critique.verdict == "followup":
                    yield from self._add_followup(state, steps, critique)
                    break
                # retry
                if (
                    attempt >= self.settings.max_retries_per_step
                    or state.retries >= self.settings.max_retries_per_run
                    or not self._budget_ok()
                ):
                    break
                approach = critique.new_approach or "use a different query or source"
                state.retries += 1
                yield self._event(
                    "execute", "retry", {"step_id": step.id, "new_approach": approach}
                )
            if self.budget.exhausted_reason:
                break
            index += 1
        state.plan = state.plan.model_copy(update={"steps": steps})

    def _memory_step(self, state: RunState, step: Step) -> Iterator[Event]:
        """Answer a step from remembered facts: no tools, no LLM; citations keep their URLs."""
        facts = {f.id: f for f in state.memory.fresh_facts}
        evidence = [
            state.ledger.add_fact(facts[i], step.id) for i in step.memory_fact_ids if i in facts
        ]
        state.memory_steps += 1
        state.facts_reused += len(evidence)
        state.step_log.append({"step_id": step.id, "question": step.question, "verdict": "memory"})
        yield self._event(
            "execute",
            "memory_hit",
            {
                "step_id": step.id,
                "question": step.question,
                "fact_ids": step.memory_fact_ids,
                "evidence_ids": [e.id for e in evidence],
                "sources": sorted({e.source_url for e in evidence}),
            },
        )

    def _run_step(
        self, state: RunState, step: Step, approach: str | None
    ) -> Generator[Event, None, StepResult | None]:
        """Execute one step; None if the wall clock ran out mid-step."""
        assert state.task is not None and state.playbook is not None
        yield self._event("execute", "step_started", {**step.model_dump(), "approach": approach})

        def emit(type_: EventType, payload: dict[str, Any]) -> Event:
            return self._event("execute", type_, payload)

        try:
            result = yield from self._pump(
                execute_step(
                    step,
                    task=state.task,
                    playbook=state.playbook,
                    llm=self.llm,
                    registry=self.registry,
                    budget=self.budget,
                    settings=self.settings,
                    ledger=state.ledger,
                    emit=emit,
                    approach=approach,
                ),
                "execute",
            )
        except DeadlineExceededError as exc:
            self.budget.exhausted_reason = f"wall clock ({self.settings.max_wall_clock_s:.0f}s)"
            yield from self._notices("execute")
            yield self._event("execute", "error", {"step_id": step.id, "message": str(exc)})
            return None
        except Exception as exc:  # noqa: BLE001 - one failed step must not end the run
            yield from self._notices("execute")
            yield self._event("execute", "error", {"step_id": step.id, "message": str(exc)})
            return StepResult(step_id=step.id, status="error")
        state.results.append(result)
        return result

    def _critique(
        self, state: RunState, step: Step, result: StepResult
    ) -> Generator[Event, None, Critique | None]:
        """Ask the critic for a verdict; None when the budget leaves no room for it."""
        assert state.playbook is not None
        if not self.budget.llm_available():
            return None
        try:
            critique = critique_step(self.llm, step, result, state.playbook)
        except DeadlineExceededError:
            self.budget.exhausted_reason = f"wall clock ({self.settings.max_wall_clock_s:.0f}s)"
            yield from self._notices("critique")
            return None
        except Exception as exc:  # noqa: BLE001
            yield from self._notices("critique")
            yield self._event("critique", "error", {"step_id": step.id, "message": str(exc)})
            return None
        yield from self._notices("critique")
        state.critiques.append(critique)
        state.step_log.append(
            {
                "step_id": step.id,
                "question": step.question,
                "verdict": critique.verdict,
                "reason": critique.reason,
                "new_approach": critique.new_approach,
                "evidence": len(result.evidence),
            }
        )
        yield self._event(
            "critique",
            "critique",
            {
                "step_id": step.id,
                "verdict": critique.verdict,
                "reason": critique.reason,
                "new_approach": critique.new_approach,
                "followup": critique.followup.question if critique.followup else None,
                "evidence": len(result.evidence),
                "source_ratings": critique.source_ratings,
            },
        )
        return critique

    def _add_followup(
        self, state: RunState, steps: list[Step], critique: Critique
    ) -> Iterator[Event]:
        """Append the critic's follow-up step if the follow-up and step caps allow (replan)."""
        if (
            critique.followup is None
            or state.followups >= self.settings.max_followups
            or len(steps) >= self.settings.max_total_steps
        ):
            return
        new_step = critique.followup.model_copy(update={"id": len(steps) + 1, "is_followup": True})
        steps.append(new_step)
        state.followups += 1
        yield self._event(
            "plan", "replan", {"added_step": new_step.model_dump(), "reason": critique.reason}
        )

    def _synthesize(self, state: RunState) -> Iterator[Event]:
        state.stage = "synthesize"
        assert state.task is not None and state.playbook is not None
        # Synthesis always runs; it gets a short grace period past the wall clock.
        self.llm.deadline = (
            self._started + self.settings.max_wall_clock_s + self.settings.synthesis_grace_s
        )
        note = self.budget.exhausted_reason
        if note:
            note = f"budget reached: {note}"
        yield self._event(
            "synthesize",
            "stage_started",
            {"evidence": len(state.ledger.items), "budget_note": note},
        )
        try:
            state.brief = synthesize(
                self.llm, state.task, state.playbook, state.ledger, note, state.critic_unknowns
            )
        except Exception as exc:  # noqa: BLE001
            yield from self._notices("synthesize")
            yield self._event("synthesize", "error", {"message": f"Synthesis failed: {exc}"})
            state.brief = fallback_brief(state.task, state.playbook, state.ledger, note)
        yield from self._notices("synthesize")
        cited, total = citation_stats(state.brief, state.ledger)
        yield self._event("synthesize", "synthesis", {"claims": total, "cited_claims": cited})

    def _verify(self, state: RunState) -> Iterator[Event]:
        """Deterministic claim checks, one revision pass, then move leftovers to Unknowns."""
        state.stage = "verify"
        assert state.brief is not None
        issues = find_issues(state.brief, state.ledger)
        state.flagged_numeric = [i.text for i in issues if i.kind == "numeric"]
        before = len(issues)
        if issues and self.budget.llm.usage.llm_calls < self.settings.max_llm_calls:
            try:
                revisions = revise_claims(self.llm, issues, state.ledger)
                state.brief = apply_revisions(state.brief, revisions)
                state.revised = len(revisions.revisions)
            except Exception as exc:  # noqa: BLE001 - fall through to moving claims
                yield from self._notices("verify")
                yield self._event("verify", "error", {"message": f"Revision failed: {exc}"})
            yield from self._notices("verify")
            issues = find_issues(state.brief, state.ledger)
        if issues:
            state.brief = move_to_unknowns(state.brief, issues)
        missing = [f"Not found: {q}" for q in state.critic_unknowns]
        unknowns = dedupe(list(state.brief.unknowns) + missing, UNKNOWN_OVERLAP)
        state.brief = state.brief.model_copy(update={"unknowns": unknowns})
        state.unsupported = len(issues)
        state.brief = scrub_brief(state.brief)
        state.report_md = render_markdown(state.brief, state.ledger, self.run_id)
        yield self._event(
            "verify",
            "verification",
            {
                "flagged": before,
                "flagged_numeric": state.flagged_numeric,
                "revised": state.revised,
                "moved_to_unknowns": [i.text for i in issues],
            },
        )

    def _reflect(self, state: RunState) -> Iterator[Event]:
        """Persist verified facts and source ratings, then learn lessons (one LLM call)."""
        state.stage = "reflect"
        assert state.task is not None and state.brief is not None
        self._persist_facts(state)
        for critique in state.critiques:
            for url, rating in critique.source_ratings.items():
                self.store.rate_source(url, rating == "useful")
        if self.llm.usage.llm_calls >= self.settings.max_llm_calls:
            return
        try:
            reflection = reflect(
                self.llm,
                state.task.purpose_type,
                self._run_summary(state),
                state.memory.lessons,
                self.settings.reflector_max_lessons,
            )
        except Exception as exc:  # noqa: BLE001 - learning is optional, the brief is done
            yield from self._notices("reflect")
            yield self._event("reflect", "error", {"message": f"Reflection failed: {exc}"})
            return
        yield from self._notices("reflect")
        for vote in reflection.votes:
            self.store.vote(vote.lesson_id, vote.helpful)
        for text in reflection.lessons:
            lesson, merged = self.store.add_or_merge_lesson(
                state.task.purpose_type, text, self.settings.lesson_merge_overlap
            )
            yield self._event(
                "reflect",
                "lesson_learned",
                {"lesson_id": lesson.id, "text": lesson.text, "proposed": text, "merged": merged},
            )

    def _persist_facts(self, state: RunState) -> None:
        """Store evidence cited by the verified brief as facts, keyed by normalized entity."""
        assert state.task is not None and state.brief is not None
        cited = {i for sec in state.brief.sections for c in sec.claims for i in c.evidence_ids}
        by_entity: dict[str, list[Evidence]] = {}
        for e in state.ledger.items:
            if e.id not in cited or e.from_memory:
                continue
            target = next(
                (t for t in state.task.targets if normalize_entity(t) in e.claim.lower()),
                state.task.targets[0],
            )
            by_entity.setdefault(normalize_entity(target), []).append(e)
        for entity, items in by_entity.items():
            self.store.add_facts(entity, items, self.run_id)

    def _run_summary(self, state: RunState) -> str:
        """Compact run summary for the reflector."""
        assert state.task is not None
        lines = [f"Goal: {state.goal}", f"Purpose: {state.task.purpose_type}", "Steps:"]
        for entry in state.step_log:
            line = f"- step {entry['step_id']}: {entry['question']} -> {entry['verdict']}"
            if entry.get("reason"):
                line += f" ({entry['reason']})"
            if entry.get("new_approach"):
                line += f"; retry approach: {entry['new_approach']}"
            lines.append(line)
        useful: set[str] = set()
        useless: set[str] = set()
        for critique in state.critiques:
            for url, rating in critique.source_ratings.items():
                (useful if rating == "useful" else useless).add(domain_of(url))
        lines.append(f"Useful domains: {', '.join(sorted(useful)) or 'none'}")
        lines.append(f"Useless domains: {', '.join(sorted(useless - useful)) or 'none'}")
        lines.append(
            f"Tool calls {self.budget.tool_calls}, LLM calls {self.llm.usage.llm_calls}, "
            f"evidence {len(state.ledger.items)}, steps from memory {state.memory_steps}, "
            f"budget {self.budget.exhausted_reason or 'ok'}"
        )
        if state.brief is not None and state.brief.unknowns:
            lines.append("Unknowns: " + "; ".join(state.brief.unknowns[:5]))
        return "\n".join(lines)

    # --- finish -----------------------------------------------------------------------------

    def metrics(self, state: RunState) -> RunMetrics:
        """Collect run metrics."""
        cited, total = citation_stats(state.brief, state.ledger) if state.brief else (0, 0)
        usage = self.llm.usage
        calls_by_model = Counter(
            f"{c.provider}/{c.model}" if c.provider else c.model for c in usage.calls
        )
        return RunMetrics(
            run_id=self.run_id,
            tool_calls=self.budget.tool_calls,
            llm_calls=usage.llm_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_s=round(self.clock() - self._started, 1),
            citation_coverage=round(100.0 * cited / total, 1) if total else 0.0,
            unsupported_claims=state.unsupported,
            flagged_numeric_claims=len(state.flagged_numeric),
            revised_claims=state.revised,
            budget_exhausted=self.budget.exhausted_reason is not None,
            provider_switches=self._switches,
            tokens_by_model=usage.tokens_by_model(),
            calls_by_model=dict(calls_by_model),
            critique_verdicts=dict(Counter(c.verdict for c in state.critiques)),
            retries=state.retries,
            replans=state.followups,
            memory_steps=state.memory_steps,
            facts_reused=state.facts_reused,
            cache_hits=self.registry.stats.get("cache_hits", 0),
            lessons_injected=len(state.memory.lessons),
        )

    def _finish(self, state: RunState) -> Event:
        metrics = self.metrics(state)
        cited, total = citation_stats(state.brief, state.ledger) if state.brief else (0, 0)
        metrics_payload = {
            **metrics.model_dump(),
            "total_tokens": metrics.total_tokens,
            "claims": total,
            "cited_claims": cited,
            "evidence_items": len(state.ledger.items),
            "budget_reason": self.budget.exhausted_reason,
            "injected_lessons": [lesson.id for lesson in state.memory.lessons],
        }
        if state.task is not None:
            try:
                self.store.save_run(
                    self.run_id,
                    state.goal,
                    state.task.purpose_type,
                    state.task.targets,
                    self.started_at,
                    now_utc(),
                    metrics_payload,
                )
            except Exception:  # noqa: BLE001 - never fail the run on bookkeeping
                logger.exception("Could not save run %s", self.run_id)
        return self._event(
            "finish",
            "run_finished",
            {
                "status": "ok" if state.brief else "failed",
                "task": state.task.model_dump() if state.task else None,
                "plan": state.plan.model_dump() if state.plan else None,
                "report_md": state.report_md,
                "brief": state.brief.model_dump() if state.brief else None,
                "evidence": [e.model_dump() for e in state.ledger.items],
                "metrics": metrics_payload,
            },
        )


def run_orchestrator(
    goal: str,
    settings: Settings,
    llm: LLM | None = None,
    registry: ToolRegistry | None = None,
) -> Iterator[Event]:
    """Convenience wrapper: build an Orchestrator and yield its events."""
    return Orchestrator(settings, llm=llm, registry=registry).run(goal)
