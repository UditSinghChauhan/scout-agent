"""Critic stage: StepResult + done criteria -> Critique (docs/SPEC.md §4, Phase 2 B1)."""

from __future__ import annotations

from scout.agent.executor import _Entry, summarize_observation
from scout.llm import LLM
from scout.playbooks import Playbook
from scout.prompts import load_prompt
from scout.schemas import Critique, Step, StepResult


def _actions_text(result: StepResult) -> str:
    """One line per tool call the executor made."""
    lines = [f"- {summarize_observation(_Entry(o, ()))}" for o in result.observations]
    return "\n".join(lines) or "- no tool calls"


def _evidence_text(result: StepResult) -> str:
    """Evidence items, one per line."""
    lines = [f"- [{e.id}] {e.claim} ({e.source_url})" for e in result.evidence]
    return "\n".join(lines) or "- none"


def critique_step(llm: LLM, step: Step, result: StepResult, playbook: Playbook) -> Critique:
    """Ask the fast-tier model for a verdict; ratings are limited to sources actually used."""
    prompt = load_prompt(
        "critic",
        purpose=f"{playbook.purpose_type}: {playbook.description}",
        question=step.question,
        done_criteria=step.done_criteria or "a sourced answer to the question",
        actions=_actions_text(result),
        evidence=_evidence_text(result),
        sources="\n".join(f"- {u}" for u in result.sources_used) or "- none",
    )
    critique = llm.complete_json(
        [{"role": "system", "content": prompt}, {"role": "user", "content": "Critique the step."}],
        Critique,
        fast=True,
        include_schema=False,
    )
    ratings = {u: r for u, r in critique.source_ratings.items() if u in result.sources_used}
    return critique.model_copy(update={"source_ratings": ratings})
