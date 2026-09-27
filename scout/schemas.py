"""Pydantic v2 data models for Scout (docs/SPEC.md §4 "Data model").

Models that the LLM produces (Task, ResearchPlan, Action, Critique, Brief, Lesson) keep their
fields simple and JSON-friendly so they validate through ``llm.complete_json``. Length budgets
(max steps, iterations) are enforced by the agent from ``scout.config``, not hard-coded here.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

PurposeType = Literal["sales_prospect", "competitor", "interview_prep", "general"]


def _now() -> datetime:
    """Timezone-aware UTC timestamp."""
    return datetime.now(UTC)


class Task(BaseModel):
    """Intake output: who to research and why."""

    goal: str = Field(description="The user's original goal text.")
    targets: list[str] = Field(
        min_length=1, max_length=2, description="1 or 2 company names or URLs."
    )
    purpose_type: PurposeType = "general"
    user_context: str | None = Field(default=None, description="e.g. 'we sell X to Y'.")
    constraints: list[str] = Field(default_factory=list)


class Fact(BaseModel):
    """A stored fact about an entity (row shape of the SQLite ``facts`` table)."""

    entity: str
    topic: str
    claim: str
    source_url: str
    captured_at: datetime = Field(default_factory=_now)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    run_id: str | None = None


class Lesson(BaseModel):
    """A Reflexion-style strategy lesson for one purpose type."""

    id: int | None = None
    purpose_type: PurposeType
    text: str
    votes_up: int = Field(default=0, ge=0)
    votes_down: int = Field(default=0, ge=0)
    created_at: datetime = Field(default_factory=_now)
    last_used_at: datetime | None = None


class MemoryContext(BaseModel):
    """Recall output: what Scout already knows before planning."""

    fresh_facts: list[Fact] = Field(default_factory=list)
    stale_facts: list[Fact] = Field(default_factory=list)
    lessons: list[Lesson] = Field(default_factory=list, description="Top lessons (3).")
    source_scores: dict[str, float] = Field(
        default_factory=dict, description="domain -> reliability score in [0, 1]."
    )


class Step(BaseModel):
    """One research step in a plan."""

    id: int = Field(ge=1)
    question: str
    rationale: str = ""
    suggested_tools: list[str] = Field(default_factory=list)
    done_criteria: str = ""
    answered_from_memory: bool = False
    is_followup: bool = False


class ResearchPlan(BaseModel):
    """Planner output: an ordered list of steps."""

    steps: list[Step] = Field(min_length=1)
    notes: str = ""


class Finding(BaseModel):
    """One finding returned by the executor's finish action."""

    claim: str
    source_url: str
    snippet: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)


class Action(BaseModel):
    """One ReAct iteration: call a tool, or finish the step with findings."""

    thought: str = ""
    type: Literal["tool", "finish"]
    tool: str | None = None
    args: dict[str, Any] = Field(default_factory=dict)
    findings: list[Finding] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _accept_function_call_shape(cls, data: Any) -> Any:
        """Accept native function-call style ``{"name", "arguments"}`` as an action.

        ``finish`` maps to a finish action; any other name is a tool call. A ``thought`` inside
        the arguments is lifted out so the trace still shows the model's reasoning.
        """
        if not (isinstance(data, dict) and "type" not in data and "name" in data):
            return data
        args = dict(data.get("arguments") or data.get("args") or {})
        thought = str(args.pop("thought", data.get("thought", "")))
        if data["name"] == "finish":
            return {"thought": thought, "type": "finish", "findings": args.get("findings", [])}
        return {"thought": thought, "type": "tool", "tool": data["name"], "args": args}


class Observation(BaseModel):
    """Result of executing a tool call."""

    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    ok: bool = True
    content: str = ""
    error: str | None = None


class Evidence(BaseModel):
    """A citable piece of evidence in the run's ledger."""

    id: str = Field(description="Stable id such as 'E1', cited by claims.")
    claim: str
    source_url: str
    snippet: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    step_id: int | None = None
    from_memory: bool = False


class StepResult(BaseModel):
    """Executor output for one step."""

    step_id: int
    status: Literal["done", "no_findings", "budget_exhausted", "error"] = "done"
    evidence: list[Evidence] = Field(default_factory=list)
    observations: list[Observation] = Field(default_factory=list)
    iterations: int = Field(default=0, ge=0)
    sources_used: list[str] = Field(default_factory=list)


class Critique(BaseModel):
    """Critic output: routing verdict plus per-source ratings."""

    verdict: Literal["complete", "retry", "followup", "unknown"]
    reason: str = ""
    new_approach: str | None = Field(default=None, description="Required for 'retry'.")
    followup: Step | None = Field(default=None, description="Required for 'followup'.")
    source_ratings: dict[str, Literal["useful", "useless"]] = Field(
        default_factory=dict, description="source URL -> rating."
    )


class Claim(BaseModel):
    """A factual statement in the brief, backed by evidence ids."""

    text: str
    evidence_ids: list[str] = Field(default_factory=list)


class Section(BaseModel):
    """One section of the brief (named by the playbook)."""

    title: str
    claims: list[Claim] = Field(default_factory=list)


class Brief(BaseModel):
    """Synthesizer output: the final cited report."""

    title: str
    purpose_type: PurposeType = "general"
    sections: list[Section] = Field(default_factory=list)
    score: str | None = Field(
        default=None, description="Fit score, threat level or readiness, per playbook."
    )
    score_reasons: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    budget_note: str | None = Field(default=None, description="Set when a budget was hit.")


class RunMetrics(BaseModel):
    """Per-run metrics written to runs/<run_id>/metrics.json."""

    run_id: str
    tool_calls: int = 0
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_s: float = 0.0
    citation_coverage: float = Field(default=0.0, ge=0.0, le=100.0, description="Percent.")
    unsupported_claims: int = 0
    facts_reused: int = 0
    budget_exhausted: bool = False

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens
