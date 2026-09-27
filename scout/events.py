"""Event model and run recorder (docs/SPEC.md §4 "Events").

The orchestrator yields :class:`Event` objects; the CLI, the UI and :class:`RunRecorder` all
consume the same stream. The recorder appends every event to ``runs/<run_id>/trace.jsonl`` and,
on ``run_finished``, writes ``report.md`` and ``metrics.json``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

EventType = Literal[
    "stage_started",
    "plan_created",
    "memory_hit",
    "step_started",
    "thought",
    "tool_call",
    "tool_result",
    "critique",
    "replan",
    "synthesis",
    "verification",
    "lesson_learned",
    "run_finished",
    "error",
    "retry",
    "provider_switched",
    "llm_call",
]


class Event(BaseModel):
    """One entry in the run trace."""

    run_id: str
    ts: datetime = Field(default_factory=lambda: datetime.now(UTC))
    stage: str
    type: EventType
    payload: dict[str, Any] = Field(default_factory=dict)


class RunRecorder:
    """Persists a run's event stream and final artifacts under ``runs_dir/<run_id>/``."""

    def __init__(self, runs_dir: Path, run_id: str) -> None:
        self.run_dir = runs_dir / run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace_path = self.run_dir / "trace.jsonl"

    def write(self, event: Event) -> None:
        """Append one event to trace.jsonl; on run_finished also write report and metrics."""
        with self.trace_path.open("a", encoding="utf-8") as fh:
            fh.write(event.model_dump_json() + "\n")
        if event.type == "run_finished":
            report = event.payload.get("report_md")
            if report:
                (self.run_dir / "report.md").write_text(report, encoding="utf-8")
            metrics = event.payload.get("metrics")
            if metrics is not None:
                (self.run_dir / "metrics.json").write_text(
                    json.dumps(metrics, indent=2), encoding="utf-8"
                )

    def record(self, events: Iterable[Event]) -> Iterator[Event]:
        """Pass events through unchanged, writing each one first."""
        for event in events:
            self.write(event)
            yield event
