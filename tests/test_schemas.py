from __future__ import annotations

import pytest
from pydantic import BaseModel, ValidationError

from scout import schemas
from scout.schemas import Action, Brief, Claim, Critique, RunMetrics, Section, Task

ALL_MODELS = [
    "Task", "MemoryContext", "Step", "ResearchPlan", "Action", "Observation", "Evidence",
    "StepResult", "Critique", "Claim", "Section", "Brief", "Lesson", "RunMetrics",
]


@pytest.mark.parametrize("name", ALL_MODELS)
def test_spec_models_exist_and_have_json_schema(name: str) -> None:
    model = getattr(schemas, name)
    assert issubclass(model, BaseModel)
    assert model.model_json_schema()["type"] == "object"


def test_task_limits_targets() -> None:
    Task(goal="g", targets=["Zoho"], purpose_type="sales_prospect")
    with pytest.raises(ValidationError):
        Task(goal="g", targets=[])
    with pytest.raises(ValidationError):
        Task(goal="g", targets=["a", "b", "c"])
    with pytest.raises(ValidationError):
        Task(goal="g", targets=["a"], purpose_type="gossip")


def test_action_tool_and_finish_parse_from_json() -> None:
    tool = Action.model_validate_json(
        '{"thought": "search", "type": "tool", "tool": "web_search", "args": {"query": "zoho"}}'
    )
    assert tool.args == {"query": "zoho"}
    finish = Action.model_validate(
        {
            "thought": "done",
            "type": "finish",
            "findings": [{"claim": "c", "source_url": "https://x.com", "confidence": 0.9}],
        }
    )
    assert finish.findings[0].confidence == 0.9
    with pytest.raises(ValidationError):
        Action.model_validate({"thought": "t", "type": "finish", "findings": [
            {"claim": "c", "source_url": "u", "confidence": 2}
        ]})


def test_critique_verdicts() -> None:
    c = Critique(verdict="retry", new_approach="try the careers page",
                 source_ratings={"https://a.com": "useful"})
    assert c.verdict == "retry"
    with pytest.raises(ValidationError):
        Critique(verdict="maybe")


def test_brief_round_trip() -> None:
    brief = Brief(
        title="Zoho",
        purpose_type="sales_prospect",
        sections=[Section(title="Snapshot", claims=[Claim(text="x", evidence_ids=["E1"])])],
        unknowns=["revenue"],
    )
    assert Brief.model_validate_json(brief.model_dump_json()) == brief


def test_run_metrics_total_tokens() -> None:
    m = RunMetrics(run_id="r1", prompt_tokens=10, completion_tokens=5)
    assert m.total_tokens == 15
