"""Executor stage: one Step -> StepResult via a ReAct loop (docs/SPEC.md §4).

``execute_step`` is a generator: it yields ``thought``/``tool_call``/``tool_result`` events and
returns the ``StepResult`` (use ``result = yield from execute_step(...)``). Each iteration asks
the LLM for one :class:`Action`; tool results are observed; a ``finish`` action returns findings
that become ledger :class:`Evidence` with ids. If the loop runs out of iterations or tool budget,
the model is asked once more to finish from what it has.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any

from scout.agent.budget import SYNTHESIS_RESERVE, Budget
from scout.config import Settings
from scout.events import Event, EventType
from scout.llm import LLM, LLMValidationError
from scout.playbooks import Playbook
from scout.prompts import load_prompt
from scout.safety import is_personal_profile
from scout.schemas import Action, Evidence, Fact, Finding, Observation, Step, StepResult, Task
from scout.tools.registry import REPEAT_NOTE, ToolRegistry

logger = logging.getLogger(__name__)

EmitFn = Callable[[EventType, dict[str, Any]], Event]

_TAG_RE = re.compile(r"</?untrusted_content[^>]*>")


def normalize_url(url: str) -> str:
    """Comparable form of a URL (case-insensitive scheme/host, no trailing slash or fragment)."""
    url = url.strip().split("#", 1)[0].rstrip("/")
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    host, slash, path = rest.partition("/")
    return f"{scheme.lower()}://{host.lower()}{slash}{path}"


@dataclass
class EvidenceLedger:
    """All evidence gathered in a run; lives outside the prompts."""

    items: list[Evidence] = field(default_factory=list)

    def _duplicate(self, claim: str, url: str) -> bool:
        """True if the same claim from the same source is already in the ledger."""
        key = (claim.strip().lower(), normalize_url(url))
        return any(
            (i.claim.strip().lower(), normalize_url(i.source_url)) == key for i in self.items
        )

    def add(self, finding: Finding, step_id: int, topic: str = "") -> Evidence | None:
        """Add a finding as Evidence with the next id; skip exact duplicates."""
        if self._duplicate(finding.claim, finding.source_url):
            return None
        evidence = Evidence(
            id=f"E{len(self.items) + 1}",
            claim=finding.claim.strip(),
            source_url=finding.source_url.strip(),
            snippet=finding.snippet.strip()[:500],
            confidence=finding.confidence,
            step_id=step_id,
            topic=topic,
            volatility=finding.volatility,
        )
        self.items.append(evidence)
        return evidence

    def add_fact(self, fact: Fact, step_id: int) -> Evidence:
        """Add a remembered fact as Evidence; it keeps its original source URL for citations."""
        for item in self.items:
            if item.fact_id is not None and item.fact_id == fact.id:
                return item
        evidence = Evidence(
            id=f"E{len(self.items) + 1}",
            claim=fact.claim,
            source_url=fact.source_url,
            snippet=fact.snippet,
            confidence=fact.confidence,
            step_id=step_id,
            from_memory=True,
            fact_id=fact.id,
            topic=fact.topic,
            volatility=fact.volatility,
        )
        self.items.append(evidence)
        return evidence

    def by_id(self) -> dict[str, Evidence]:
        """Map evidence id -> Evidence."""
        return {e.id: e for e in self.items}


@dataclass
class _Entry:
    """One scratchpad entry: the observation plus the URLs it exposed."""

    observation: Observation
    urls: tuple[str, ...]


def summarize_observation(entry: _Entry) -> str:
    """One-line summary for older scratchpad entries."""
    obs = entry.observation
    args = json.dumps(obs.args, ensure_ascii=False)[:120]
    if not obs.ok:
        return f"{obs.tool}({args}) -> {obs.error}"[:300]
    if entry.urls:
        return f"{obs.tool}({args}) -> ok; URLs: " + ", ".join(entry.urls[:6])
    text = " ".join(_TAG_RE.sub("", obs.content).split())
    return f"{obs.tool}({args}) -> {text[:160]}"


def render_scratchpad(entries: list[_Entry], full: int = 2) -> str:
    """Older observations as one-liners, the last ``full`` in full."""
    if not entries:
        return "No observations yet. Start with a tool call."
    lines: list[str] = []
    cutoff = len(entries) - full
    for i, entry in enumerate(entries, 1):
        if i <= cutoff:
            lines.append(f"[{i}] (summary) {summarize_observation(entry)}")
        else:
            obs = entry.observation
            args = json.dumps(obs.args, ensure_ascii=False)
            lines.append(f"[{i}] {obs.tool}({args})\n{obs.content}")
    return "\n\n".join(lines)


def _messages(
    system: str, entries: list[_Entry], iteration: int, force_finish: bool, full: int
) -> list:
    """Build the per-iteration prompt from the compact scratchpad."""
    user = f"## Observations\n{render_scratchpad(entries, full)}\n\n"
    if force_finish:
        user += "No research turns left. Reply with a finish action now."
    else:
        user += f"Turn {iteration}. Reply with your next action."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _result_summary(obs: Observation, urls: tuple[str, ...]) -> str:
    """Short human-readable summary of a tool result for the trace."""
    if not obs.ok:
        return obs.error or "error"
    if obs.content.startswith(REPEAT_NOTE):
        return "repeat of an earlier call (not re-run)"
    if obs.tool == "web_search":
        return f"{len(urls)} results"
    text = " ".join(_TAG_RE.sub("", obs.content).split())
    return f"{len(text)} chars: {text[:140]}"


def execute_step(
    step: Step,
    *,
    task: Task,
    playbook: Playbook,
    llm: LLM,
    registry: ToolRegistry,
    budget: Budget,
    settings: Settings,
    ledger: EvidenceLedger,
    emit: EmitFn,
    approach: str | None = None,
) -> Generator[Event, None, StepResult]:
    """Run the ReAct loop for one step, yielding trace events; return the StepResult.

    ``approach`` is the critic's ``new_approach`` when the step is being retried.
    """
    purpose = f"{playbook.purpose_type}: {playbook.description}"
    if task.user_context:
        purpose += f" User context: {task.user_context}"
    retry_note = f"\nRetry: an earlier attempt failed. New approach: {approach}" if approach else ""
    system = load_prompt(
        "executor",
        question=step.question + retry_note,
        done_criteria=step.done_criteria or "a sourced answer to the question",
        purpose=purpose,
        tools=registry.describe(with_thought=True),
        max_iterations=settings.max_react_iterations,
    )
    full = settings.scratchpad_full_observations
    entries: list[_Entry] = []
    seen_urls: set[str] = set()
    finish: Action | None = None
    iterations = 0

    for iteration in range(1, settings.max_react_iterations + 1):
        if not budget.llm_available():
            break
        iterations = iteration
        try:
            action = llm.complete_json(
                _messages(system, entries, iteration, False, full),
                Action,
                fast=True,
                include_schema=False,
            )
        except LLMValidationError as exc:
            obs = Observation(tool="(invalid action)", ok=False, error=str(exc)[:300])
            entries.append(_Entry(obs, ()))
            yield emit("error", {"step_id": step.id, "message": "invalid action JSON"})
            continue
        yield emit("thought", {"step_id": step.id, "iteration": iteration, "text": action.thought})
        if action.type == "finish":
            finish = action
            break
        if not budget.tool_available():
            break
        yield emit("tool_call", {"step_id": step.id, "tool": action.tool, "args": action.args})
        run = registry.run(action.tool or "", action.args)
        if not run.repeated:
            budget.charge_tool()
        seen_urls.update(normalize_url(u) for u in run.urls)
        entries.append(_Entry(run.observation, run.urls))
        yield emit(
            "tool_result",
            {
                "step_id": step.id,
                "tool": run.observation.tool,
                "ok": run.observation.ok,
                "repeated": run.repeated,
                "summary": _result_summary(run.observation, run.urls),
            },
        )

    # Forced finish ignores the wall clock (one call) but never eats the synthesis reserve.
    llm_room = budget.llm.usage.llm_calls < settings.max_llm_calls - SYNTHESIS_RESERVE
    if finish is None and entries and llm_room:
        try:
            forced = llm.complete_json(
                _messages(system, entries, 0, True, full), Action, fast=True, include_schema=False
            )
            yield emit(
                "thought", {"step_id": step.id, "iteration": "final", "text": forced.thought}
            )
            finish = forced if forced.type == "finish" else None
        except LLMValidationError:
            yield emit("error", {"step_id": step.id, "message": "invalid final action JSON"})

    evidence: list[Evidence] = []
    dropped = 0
    for finding in finish.findings if finish else []:
        if normalize_url(finding.source_url) not in seen_urls or is_personal_profile(
            finding.source_url, settings.personal_profile_patterns
        ):
            dropped += 1
            continue
        item = ledger.add(finding, step.id, topic=step.question[:160])
        if item:
            evidence.append(item)
    if dropped:
        logger.info("Step %d: dropped %d findings citing unseen URLs", step.id, dropped)

    status = "done" if evidence else "no_findings"
    if budget.exhausted_reason:
        status = "budget_exhausted"
    return StepResult(
        step_id=step.id,
        status=status,
        evidence=evidence,
        observations=[e.observation for e in entries],
        iterations=iterations,
        sources_used=sorted(
            {u for e in entries for u in e.urls if e.observation.tool == "fetch_page"}
            | {e.source_url for e in evidence}
        ),
    )
