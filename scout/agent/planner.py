"""Planner stage: Task + MemoryContext + playbook -> ResearchPlan (docs/SPEC.md §4)."""

from __future__ import annotations

from scout.config import Settings
from scout.llm import LLM
from scout.playbooks import Playbook, playbook_brief
from scout.prompts import load_prompt
from scout.schemas import Fact, MemoryContext, ResearchPlan, Step, Task
from scout.textutil import overlap

MAX_MEMORY_STEPS = 3  # memory-answered steps are free, but keep the plan readable


def _memory_text(memory: MemoryContext) -> str:
    """Fresh facts with ids, for the planner to mark steps answered from memory."""
    if not memory.fresh_facts:
        return "Nothing yet."
    return "\n".join(
        f"- [F{f.id}] {f.claim} ({f.volatility}, captured {f.captured_at:%Y-%m-%d})"
        for f in memory.fresh_facts
    )


def _lessons_text(memory: MemoryContext) -> str:
    """Injected lessons with ids."""
    return "\n".join(f"- [L{lesson.id}] {lesson.text}" for lesson in memory.lessons) or "None yet."


def normalize_plan(
    plan: ResearchPlan, max_steps: int, fact_ids: set[int] | None = None
) -> ResearchPlan:
    """Validate memory steps, cap executed steps at ``max_steps``, renumber from 1.

    A step counts as answered from memory only if it cites at least one real fresh fact id;
    such steps cost no tool calls, so they do not count against ``max_steps``.
    """
    fact_ids = fact_ids or set()
    steps: list[Step] = []
    executed = remembered = 0
    for step in plan.steps:
        ids = [i for i in step.memory_fact_ids if i in fact_ids]
        from_memory = step.answered_from_memory and bool(ids)
        if from_memory and remembered < MAX_MEMORY_STEPS:
            remembered += 1
        elif not from_memory and executed < max_steps:
            executed += 1
            ids = []
        else:
            continue
        update = {"id": len(steps) + 1, "answered_from_memory": from_memory, "memory_fact_ids": ids}
        steps.append(step.model_copy(update=update))
    return plan.model_copy(update={"steps": steps})


def attach_memory(
    plan: ResearchPlan,
    facts: list[Fact],
    target: str,
    topic_overlap: float = 0.6,
    background_facts: int = 6,
) -> ResearchPlan:
    """Deterministic memory pass after planning (does not rely on the LLM flagging steps).

    1. A research step whose question closely matches the topics of 2+ fresh facts (a fact's
       topic is the question that produced it) becomes a memory step citing those facts.
    2. If fresh facts exist but no step uses memory, a background step answered from memory is
       prepended, citing up to ``background_facts`` facts (stable first).
    Steps are renumbered from 1.
    """
    if not facts:
        return plan
    used = {i for s in plan.steps for i in s.memory_fact_ids}
    steps: list[Step] = []
    for step in plan.steps:
        if not step.answered_from_memory:
            ids = [
                f.id
                for f in facts
                if f.id is not None
                and f.id not in used
                and overlap(step.question, f.topic) >= topic_overlap
            ]
            if len(ids) >= 2:
                used.update(ids)
                step = step.model_copy(
                    update={"answered_from_memory": True, "memory_fact_ids": ids[:8]}
                )
        steps.append(step)
    if not any(s.answered_from_memory for s in steps):
        ordered = sorted(facts, key=lambda f: f.volatility != "stable")
        ids = [f.id for f in ordered if f.id is not None][:background_facts]
        background = Step(
            id=1,
            question=f"What does Scout already know about {target}? (from memory)",
            rationale="Fresh facts from earlier runs cost no research.",
            answered_from_memory=True,
            memory_fact_ids=ids,
        )
        steps.insert(0, background)
    renumbered = [s.model_copy(update={"id": i}) for i, s in enumerate(steps, 1)]
    return plan.model_copy(update={"steps": renumbered})


def fallback_plan(task: Task, playbook: Playbook, max_steps: int) -> ResearchPlan:
    """Deterministic plan from playbook hints, used if the planner call fails."""
    target = " and ".join(task.targets)
    hints = [h for h in playbook.research_hints if h != "Planner decides"] or ["Company overview"]
    steps = [
        Step(
            id=i,
            question=f"{target}: {hint.lower()}?",
            rationale=f"Playbook hint for {playbook.purpose_type}.",
            suggested_tools=["web_search", "fetch_page"],
            done_criteria=f"Sourced facts about {hint.lower()}.",
        )
        for i, hint in enumerate(hints[:max_steps], 1)
    ]
    return ResearchPlan(steps=steps, notes="Fallback plan from playbook hints.")


def make_plan(
    llm: LLM,
    task: Task,
    memory: MemoryContext,
    playbook: Playbook,
    settings: Settings,
    tools_description: str,
) -> ResearchPlan:
    """Ask the LLM for a purpose-shaped plan, capped at the configured step budget."""
    prompt = load_prompt(
        "planner",
        goal=task.goal,
        targets=", ".join(task.targets),
        user_context=task.user_context or "not stated",
        constraints="; ".join(task.constraints) or "none",
        playbook=playbook_brief(playbook),
        memory=_memory_text(memory),
        lessons=_lessons_text(memory),
        tools=tools_description,
        max_steps=settings.max_planned_steps,
    )
    plan = llm.complete_json(
        [{"role": "system", "content": prompt}, {"role": "user", "content": "Write the plan."}],
        ResearchPlan,
        include_schema=False,
    )
    fact_ids = {f.id for f in memory.fresh_facts if f.id is not None}
    plan = normalize_plan(plan, settings.max_planned_steps, fact_ids)
    return attach_memory(plan, memory.fresh_facts, task.targets[0])
