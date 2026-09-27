"""Saved runs for replay: list the ones whose trace loads cleanly, newest first."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from scout.events import Event, load_trace

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SavedRun:
    """A replayable run on disk."""

    run_id: str
    goal: str
    purpose: str
    recorded: str  # ISO timestamp of the first event
    status: str
    path: Path

    @property
    def label(self) -> str:
        """Picker label: date, purpose, goal."""
        goal = self.goal if len(self.goal) <= 70 else self.goal[:67] + "..."
        return f"{self.recorded[:16].replace('T', ' ')} · {self.purpose or '?'} · {goal}"


def summarize(run_dir: Path) -> SavedRun | None:
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
    status = "incomplete"
    for event in events:
        p = event.payload or {}
        if event.type == "stage_started" and event.stage == "intake":
            goal = str(p.get("goal", ""))
        elif event.type == "plan_created":
            purpose = str(p.get("purpose_type", purpose))
        elif event.type == "run_finished":
            status = str(p.get("status", "ok"))
            task = p.get("task") or {}
            purpose = str(task.get("purpose_type", purpose)) if isinstance(task, dict) else purpose
    return SavedRun(
        run_id=run_dir.name,
        goal=goal,
        purpose=purpose,
        recorded=events[0].ts.astimezone().isoformat(timespec="seconds"),
        status=status,
        path=run_dir,
    )


def list_runs(runs_dir: Path, finished_only: bool = True) -> list[SavedRun]:
    """Replayable runs, newest first (runs without a clean trace are left out)."""
    if not runs_dir.is_dir():
        return []
    runs = [summarize(d) for d in runs_dir.iterdir() if d.is_dir()]
    ok = [r for r in runs if r is not None and (r.status != "incomplete" or not finished_only)]
    return sorted(ok, key=lambda r: r.recorded, reverse=True)


def load_events(run: SavedRun) -> list[Event]:
    """All events of a saved run."""
    return load_trace(run.path / "trace.jsonl")
