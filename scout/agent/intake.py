"""Intake stage: goal text -> Task (docs/SPEC.md §4)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from scout.llm import LLM
from scout.prompts import load_prompt
from scout.schemas import PurposeType, Task


class IntakeReply(BaseModel):
    """What the LLM extracts; the goal itself is copied in by code, not the model."""

    targets: list[str] = Field(min_length=1, max_length=2)
    purpose_type: PurposeType = "general"
    user_context: str | None = None
    constraints: list[str] = Field(default_factory=list)


def run_intake(llm: LLM, goal: str, user_context: str | None = None) -> Task:
    """Extract a Task from the user's goal."""
    user = f"Goal: {goal}"
    if user_context:
        user += f"\nUser context: {user_context}"
    reply = llm.complete_json(
        [{"role": "system", "content": load_prompt("intake")}, {"role": "user", "content": user}],
        IntakeReply,
        fast=True,
    )
    return Task(
        goal=goal,
        targets=reply.targets,
        purpose_type=reply.purpose_type,
        user_context=user_context or reply.user_context,
        constraints=reply.constraints,
    )
