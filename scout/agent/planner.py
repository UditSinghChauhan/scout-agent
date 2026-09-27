"""Planner stage: Task + MemoryContext + playbook -> ResearchPlan (docs/SPEC.md §4)."""

from __future__ import annotations

from scout.config import Settings
from scout.llm import LLM
from scout.playbooks import Playbook, playbook_brief
from scout.prompts import load_prompt
from scout.schemas import MemoryContext, ResearchPlan, Step, Task


def _memory_text(memory: MemoryContext) -> str:
    """Compact memory summary for the prompt (empty until Phase 3)."""
    if not memory.fresh_facts:
        return "Nothing yet."
    return "\n".join(f"- [{f.topic}] {f.claim} ({f.source_url})" for f in memory.fresh_facts)


def _lessons_text(memory: MemoryContext) -> str:
    """Injected lessons (empty until Phase 3)."""
    return "\n".join(f"- {lesson.text}" for lesson in memory.lessons) or "None yet."


def normalize_plan(plan: ResearchPlan, max_steps: int) -> ResearchPlan:
    """Cap the plan at ``max_steps`` and renumber steps from 1."""
    steps = [s.model_copy(update={"id": i}) for i, s in enumerate(plan.steps[:max_steps], 1)]
    return plan.model_copy(update={"steps": steps})


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
    return normalize_plan(plan, settings.max_planned_steps)
