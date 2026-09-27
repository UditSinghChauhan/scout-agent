"""Eval harness: score recorded runs (no LLM calls) or run a small task set live.

``python -m scout eval --from-runs`` reads ``runs/`` and ``examples/``, keeps runs whose trace shows
the verifier (Phase 2 or later), and writes ``evals/results.md``.
``python -m scout eval --live evals/tasks.yaml`` runs each task through the orchestrator first
(real LLM and search calls), then scores those runs the same way.
"""

from __future__ import annotations

import statistics
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from scout.events import load_trace

COLUMNS = (
    "run",
    "purpose",
    "date",
    "tool calls",
    "LLM calls",
    "tokens",
    "seconds",
    "coverage %",
    "flagged",
    "to Unknowns",
    "retries",
    "replans",
    "memory steps",
    "switches",
    "budget stop",
)


def score_run(run_dir: Path) -> dict[str, Any] | None:
    """Metrics row for one recorded run, or None if it has no verifier stage or is unreadable."""
    trace = run_dir / "trace.jsonl"
    try:
        events = load_trace(trace)
    except (OSError, ValueError):
        return None
    verification = next((e.payload for e in events if e.type == "verification"), None)
    finished = next((e.payload for e in events if e.type == "run_finished"), None)
    if verification is None or finished is None or not events:
        return None
    m = finished.get("metrics") or {}
    task = finished.get("task") or {}
    purpose = task.get("purpose_type") or next(
        (e.payload.get("purpose_type") for e in events if e.type == "plan_created"), "?"
    )
    return {
        "run": run_dir.name,
        "purpose": purpose,
        "date": events[0].ts.astimezone().strftime("%Y-%m-%d %H:%M"),
        "tool calls": m.get("tool_calls", 0),
        "LLM calls": m.get("llm_calls", 0),
        "tokens": m.get("total_tokens", 0),
        "seconds": m.get("latency_s", 0.0),
        "coverage %": m.get("citation_coverage", 0.0),
        "flagged": verification.get("flagged", 0),
        "to Unknowns": len(verification.get("moved_to_unknowns") or []),
        "retries": m.get("retries", 0),
        "replans": m.get("replans", 0),
        "memory steps": m.get("memory_steps", 0),
        "switches": m.get("provider_switches", 0),
        "budget stop": "yes" if m.get("budget_exhausted") else "no",
    }


def collect(dirs: Iterable[Path]) -> list[dict[str, Any]]:
    """Scored runs from every directory (first occurrence of a run id wins), oldest first."""
    rows: dict[str, dict[str, Any]] = {}
    for directory in dirs:
        if not directory.is_dir():
            continue
        for run_dir in sorted(directory.iterdir()):
            if run_dir.is_dir() and run_dir.name not in rows:
                row = score_run(run_dir)
                if row:
                    rows[run_dir.name] = row
    return sorted(rows.values(), key=lambda r: r["run"])


def aggregates(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per purpose: runs, median tokens and seconds, mean citation coverage; plus an 'all' row."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(row["purpose"], []).append(row)
    out = []
    for purpose, items in sorted(groups.items()) + ([("all", rows)] if rows else []):
        out.append(
            {
                "purpose": purpose,
                "runs": len(items),
                "median tokens": int(statistics.median(r["tokens"] for r in items)),
                "median seconds": round(statistics.median(r["seconds"] for r in items), 1),
                "mean coverage %": round(statistics.fmean(r["coverage %"] for r in items), 1),
                "budget stops": sum(r["budget stop"] == "yes" for r in items),
            }
        )
    return out


def _table(rows: list[dict[str, Any]], columns: Iterable[str]) -> list[str]:
    """Markdown table lines."""
    cols = list(columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows]
    return lines


def render(rows: list[dict[str, Any]], generated: datetime | None = None) -> str:
    """The evals/results.md document."""
    stamp = (generated or datetime.now().astimezone()).strftime("%Y-%m-%d %H:%M %Z")
    agg = aggregates(rows)
    lines = [
        "# Scout eval results",
        "",
        f"Generated {stamp} by `python -m scout eval --from-runs` from {len(rows)} recorded runs.",
        "These are real development runs (Phase 2 onwards: only runs whose trace contains the "
        "verifier stage are included), not a curated benchmark. Budgets and code changed between "
        "runs; each row's trace is in `runs/` or `examples/`.",
        "",
        "## Summary by purpose",
        "",
        *_table(agg, agg[0].keys() if agg else ["purpose"]),
        "",
        "## Per run",
        "",
        "`flagged` = claims the verifier flagged (missing citation or a number not in the cited "
        "snippet); `to Unknowns` = claims still unsupported after one revision.",
        "",
        *_table(rows, COLUMNS),
        "",
    ]
    return "\n".join(lines)


def write_results(dirs: Iterable[Path], out: Path) -> list[dict[str, Any]]:
    """Score recorded runs and write the results file; returns the rows."""
    rows = collect(dirs)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(rows), encoding="utf-8")
    return rows


def load_tasks(path: Path) -> list[dict[str, str]]:
    """Tasks from YAML: ``tasks: [{id, purpose, goal}, ...]``."""
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list) or not all(isinstance(t, dict) and t.get("goal") for t in tasks):
        raise ValueError(f"{path}: expected 'tasks' as a list of mappings with a 'goal'")
    return tasks
