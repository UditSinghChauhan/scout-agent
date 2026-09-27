"""Orchestrator: Scout's explicit state machine, written as an event generator (docs/SPEC.md §4).

Phase 1 flow: Intake -> (Recall) -> Plan -> Execute each step -> Synthesize -> (Verify)
-> (Reflect). Bracketed stages are seams filled in later phases. Budgets from config are enforced
throughout; on a budget hit Scout synthesizes from the evidence it has. Exceptions become
``error`` events; ``run_orchestrator`` always ends with ``run_finished``.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from scout.agent.budget import Budget
from scout.agent.executor import EvidenceLedger, execute_step
from scout.agent.intake import run_intake
from scout.agent.planner import fallback_plan, make_plan
from scout.agent.synthesizer import citation_stats, fallback_brief, render_markdown, synthesize
from scout.config import Settings
from scout.events import Event, EventType
from scout.llm import LLM
from scout.playbooks import Playbook, get_playbook
from scout.schemas import Brief, MemoryContext, ResearchPlan, RunMetrics, StepResult, Task
from scout.tools.registry import ToolRegistry, build_registry

logger = logging.getLogger(__name__)


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
    brief: Brief | None = None
    report_md: str = ""
    stage: str = "intake"


class Orchestrator:
    """Runs one research goal end to end, yielding Events."""

    def __init__(
        self,
        settings: Settings,
        llm: LLM | None = None,
        registry: ToolRegistry | None = None,
        run_id: str | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm or LLM(settings=settings)
        self.registry = registry or build_registry(settings)
        self.run_id = run_id or new_run_id()
        self.budget = Budget(settings, self.llm)
        self._started = time.monotonic()

    def _event(self, stage: str, type_: EventType, payload: dict[str, Any] | None = None) -> Event:
        """Create an event for this run."""
        return Event(run_id=self.run_id, stage=stage, type=type_, payload=payload or {})

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
        yield self._finish(state)

    # --- stages -----------------------------------------------------------------------------

    def _intake(self, state: RunState, user_context: str | None) -> Iterator[Event]:
        state.stage = "intake"
        yield self._event("intake", "stage_started", {"goal": state.goal})
        try:
            state.task = run_intake(self.llm, state.goal, user_context)
        except Exception as exc:  # noqa: BLE001
            yield self._event("intake", "error", {"message": f"Intake failed: {exc}"})
            return
        state.playbook = get_playbook(state.task.purpose_type)

    def _recall(self, state: RunState) -> Iterator[Event]:
        """Seam for Phase 3 (fact memory, lessons, source scores). Empty context for now."""
        state.stage = "recall"
        state.memory = MemoryContext()
        yield from ()

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
            yield self._event(
                "plan", "error", {"message": f"Planner failed, using fallback: {exc}"}
            )
            state.plan = fallback_plan(state.task, state.playbook, self.settings.max_planned_steps)
        yield self._event(
            "plan",
            "plan_created",
            {
                "purpose_type": state.task.purpose_type,
                "steps": [s.model_dump() for s in state.plan.steps],
                "lessons": [lesson.text for lesson in state.memory.lessons],
            },
        )

    def _execute(self, state: RunState) -> Iterator[Event]:
        state.stage = "execute"
        assert state.task is not None and state.playbook is not None and state.plan is not None
        steps = state.plan.steps[: self.settings.max_total_steps]
        for step in steps:
            if not (self.budget.llm_available() and self.budget.tool_available()):
                break
            yield self._event("execute", "step_started", step.model_dump())
            emit = lambda type_, payload: self._event("execute", type_, payload)  # noqa: E731
            try:
                result = yield from execute_step(
                    step,
                    task=state.task,
                    playbook=state.playbook,
                    llm=self.llm,
                    registry=self.registry,
                    budget=self.budget,
                    settings=self.settings,
                    ledger=state.ledger,
                    emit=emit,
                )
            except Exception as exc:  # noqa: BLE001 - one failed step must not end the run
                yield self._event("execute", "error", {"step_id": step.id, "message": str(exc)})
                continue
            state.results.append(result)
            yield from self._critique(state, result)
            if self.budget.exhausted_reason:
                break

    def _critique(self, state: RunState, result: StepResult) -> Iterator[Event]:
        """Seam for Phase 2 critic routing (retry / followup / unknown). Always 'next' for now."""
        yield from ()

    def _synthesize(self, state: RunState) -> Iterator[Event]:
        state.stage = "synthesize"
        assert state.task is not None and state.playbook is not None
        note = self.budget.exhausted_reason
        if note:
            note = f"budget reached: {note}"
        yield self._event(
            "synthesize",
            "stage_started",
            {"evidence": len(state.ledger.items), "budget_note": note},
        )
        try:
            state.brief = synthesize(self.llm, state.task, state.playbook, state.ledger, note)
        except Exception as exc:  # noqa: BLE001
            yield self._event("synthesize", "error", {"message": f"Synthesis failed: {exc}"})
            state.brief = fallback_brief(state.task, state.playbook, state.ledger, note)
        state.report_md = render_markdown(state.brief, state.ledger, self.run_id)
        cited, total = citation_stats(state.brief, state.ledger)
        yield self._event("synthesize", "synthesis", {"claims": total, "cited_claims": cited})

    def _verify(self, state: RunState) -> Iterator[Event]:
        """Seam for Phase 2 deterministic verifier."""
        yield from ()

    def _reflect(self, state: RunState) -> Iterator[Event]:
        """Seam for Phase 3 reflector (lessons, persistence)."""
        yield from ()

    # --- finish -----------------------------------------------------------------------------

    def metrics(self, state: RunState) -> RunMetrics:
        """Collect run metrics."""
        cited, total = citation_stats(state.brief, state.ledger) if state.brief else (0, 0)
        usage = self.llm.usage
        return RunMetrics(
            run_id=self.run_id,
            tool_calls=self.budget.tool_calls,
            llm_calls=usage.llm_calls,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            latency_s=round(time.monotonic() - self._started, 1),
            citation_coverage=round(100.0 * cited / total, 1) if total else 0.0,
            budget_exhausted=self.budget.exhausted_reason is not None,
        )

    def _finish(self, state: RunState) -> Event:
        metrics = self.metrics(state)
        cited, total = citation_stats(state.brief, state.ledger) if state.brief else (0, 0)
        return self._event(
            "finish",
            "run_finished",
            {
                "status": "ok" if state.brief else "failed",
                "report_md": state.report_md,
                "brief": state.brief.model_dump() if state.brief else None,
                "evidence": [e.model_dump() for e in state.ledger.items],
                "metrics": {
                    **metrics.model_dump(),
                    "total_tokens": metrics.total_tokens,
                    "claims": total,
                    "cited_claims": cited,
                    "evidence_items": len(state.ledger.items),
                    "budget_reason": self.budget.exhausted_reason,
                },
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
