"""Insights built from recorded traces alone (no memory database needed).

A judge who clones the repo has an empty database but the committed ``examples/``; everything
the Insights tab shows can be rebuilt from those traces: the runs table (metrics), lessons
(``lesson_learned`` / ``lesson_voted`` events and injections in ``plan_created``), source scores
(the critic's per-source ratings, same Laplace formula as the store) and each run's plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from scout.events import Event
from scout.memory.store import domain_of, score
from scout.ui.runs import SavedRun, list_runs, load_events


@dataclass
class RecordedRun:
    """One recorded run: summary fields, metrics, plan and lessons, from its trace."""

    id: str
    goal: str
    purpose_type: str
    targets: list[str]
    started_at: str
    metrics: dict[str, Any]
    steps: list[dict[str, Any]] = field(default_factory=list)
    lessons_injected: list[str] = field(default_factory=list)
    events: list[Event] = field(default_factory=list, repr=False)

    def as_row(self) -> dict[str, Any]:
        """Same shape as ``MemoryStore.runs()`` rows (for shared UI code)."""
        return {
            "id": self.id,
            "goal": self.goal,
            "purpose_type": self.purpose_type,
            "targets": self.targets,
            "started_at": self.started_at,
            "metrics": self.metrics,
        }


def _load(run: SavedRun) -> RecordedRun:
    """Rebuild a run's insights data from its trace."""
    events = load_events(run)
    metrics: dict[str, Any] = {}
    steps: list[dict[str, Any]] = []
    lessons: list[str] = []
    for event in events:
        p = event.payload or {}
        if event.type == "plan_created":
            steps = [dict(s) for s in p.get("steps", []) if isinstance(s, dict)]
            lessons = [
                str(x.get("text", "")) if isinstance(x, dict) else str(x)
                for x in p.get("lessons", [])
            ]
        elif event.type == "replan" and isinstance(p.get("added_step"), dict):
            steps.append({**p["added_step"], "is_followup": True})
        elif event.type == "run_finished":
            metrics = dict(p.get("metrics") or {})
    return RecordedRun(
        id=run.run_id,
        goal=run.goal,
        purpose_type=run.purpose,
        targets=list(run.targets),
        started_at=run.recorded,
        metrics=metrics,
        steps=steps,
        lessons_injected=lessons,
        events=events,
    )


def recorded_runs(runs_dir: Path, examples_dir: Path | None) -> list[RecordedRun]:
    """Finished recorded runs (runs/ + examples/), oldest first."""
    return [_load(r) for r in reversed(list_runs(runs_dir, examples_dir))]


def recorded_lessons(runs: list[RecordedRun]) -> list[dict[str, Any]]:
    """Lessons learned across runs: merges and votes count up/down, injections count as uses."""
    lessons: dict[tuple[str, str], dict[str, Any]] = {}

    def entry(purpose: str, text: str, run_id: str) -> dict[str, Any]:
        key = (purpose, text.strip())
        if key not in lessons:
            lessons[key] = {
                "purpose": purpose,
                "lesson": text.strip(),
                "learned in": run_id,
                "up": 0,
                "down": 0,
                "uses": 0,
            }
        return lessons[key]

    for run in runs:
        for event in run.events:
            p = event.payload or {}
            if event.type == "lesson_learned" and p.get("text"):
                item = entry(run.purpose_type, str(p["text"]), run.id)
                if p.get("merged"):
                    item["up"] += 1
            elif event.type == "lesson_voted" and p.get("text"):
                item = entry(run.purpose_type, str(p["text"]), run.id)
                item["up" if p.get("helpful") else "down"] += 1
        for text in run.lessons_injected:
            if text:
                entry(run.purpose_type, text, run.id)["uses"] += 1
    rows = []
    for item in lessons.values():
        rows.append(
            {
                **item,
                "votes": f"+{item['up']}/-{item['down']}",
                "score": round(score(item["up"], item["down"]), 2),
            }
        )
    return sorted(rows, key=lambda r: (-r["score"], -r["uses"]))


def recorded_sources(runs: list[RecordedRun]) -> list[dict[str, Any]]:
    """Domain reliability from the critic's per-source ratings, best first."""
    counts: dict[str, list[int]] = {}
    for run in runs:
        for event in run.events:
            if event.type != "critique":
                continue
            for url, rating in (event.payload.get("source_ratings") or {}).items():
                domain = domain_of(url)
                if domain:
                    pair = counts.setdefault(domain, [0, 0])
                    pair[0 if rating == "useful" else 1] += 1
    rows = [
        {"domain": d, "useful": u, "useless": x, "score": score(u, x)}
        for d, (u, x) in counts.items()
    ]
    return sorted(rows, key=lambda r: (-r["score"], -(r["useful"] + r["useless"])))
