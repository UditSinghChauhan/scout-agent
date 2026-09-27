"""Saved runs for replay: list the ones whose trace loads cleanly, newest first."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from scout.events import Event, load_trace

logger = logging.getLogger(__name__)
_SHORT_PURPOSE = {
    "sales_prospect": "sales",
    "competitor": "competitor",
    "interview_prep": "interview",
    "general": "general",
}


@dataclass(frozen=True)
class SavedRun:
    """A replayable run on disk."""

    run_id: str
    goal: str
    purpose: str
    recorded: str  # ISO timestamp of the first event
    status: str
    path: Path
    source: str = "run"  # "run" (runs/) or "example" (committed examples/)
    targets: tuple[str, ...] = ()

    @property
    def label(self) -> str:
        """Compact picker label (fits the sidebar): example tag, time, purpose, company."""
        tag = "example · " if self.source == "example" else ""
        purpose = _SHORT_PURPOSE.get(self.purpose, self.purpose or "?")
        who = ", ".join(self.targets) or (
            self.goal[:16] + "…" if len(self.goal) > 16 else self.goal
        )
        return f"{tag}{self.recorded[11:16]} {purpose} · {who}"

    @property
    def caption(self) -> str:
        """Longer description shown under the picker: date and full goal."""
        return f"{self.recorded[:16].replace('T', ' ')} — {self.goal}"


def summarize(run_dir: Path, source: str = "run") -> SavedRun | None:
    """Summary of one run directory, or None if its trace is missing or unreadable."""
    trace = run_dir / "trace.jsonl"
    if not trace.is_file():
        return None
    try:
        events = load_trace(trace)
    except (OSError, ValueError, ValidationError) as exc:
        logger.info("Skipping %s: %s", run_dir.name, exc)
        return None
    if not events:
        return None
    goal = purpose = ""
    targets: tuple[str, ...] = ()
    status = "incomplete"
    for event in events:
        p = event.payload or {}
        if event.type == "stage_started" and event.stage == "intake":
            goal = str(p.get("goal", ""))
        elif event.type == "stage_started" and event.stage == "plan":
            task = p.get("task")  # older traces carry the task only here
            if isinstance(task, dict) and not targets:
                targets = tuple(str(t) for t in task.get("targets", []))
        elif event.type == "plan_created":
            purpose = str(p.get("purpose_type", purpose))
        elif event.type == "run_finished":
            status = str(p.get("status", "ok"))
            task = p.get("task") or {}
            if isinstance(task, dict) and task:
                purpose = str(task.get("purpose_type", purpose))
                targets = tuple(str(t) for t in task.get("targets", [])) or targets
    return SavedRun(
        run_id=run_dir.name,
        goal=goal,
        purpose=purpose,
        recorded=events[0].ts.astimezone().isoformat(timespec="seconds"),
        status=status,
        path=run_dir,
        source=source,
        targets=targets,
    )


def _scan(directory: Path | None, source: str) -> list[SavedRun]:
    """Summaries of every run directory under ``directory``."""
    if directory is None or not directory.is_dir():
        return []
    found = [summarize(d, source) for d in sorted(directory.iterdir()) if d.is_dir()]
    return [r for r in found if r is not None]


def list_runs(
    runs_dir: Path, examples_dir: Path | None = None, finished_only: bool = True
) -> list[SavedRun]:
    """Replayable runs, newest first; committed examples are included and labelled.

    A run present in both places is listed once, as an example. Runs without a clean trace
    are left out.
    """
    examples = _scan(examples_dir, "example")
    seen = {r.run_id for r in examples}
    runs = examples + [r for r in _scan(runs_dir, "run") if r.run_id not in seen]
    ok = [r for r in runs if r.status != "incomplete" or not finished_only]
    return sorted(ok, key=lambda r: r.recorded, reverse=True)


def load_events(run: SavedRun) -> list[Event]:
    """All events of a saved run."""
    return load_trace(run.path / "trace.jsonl")
