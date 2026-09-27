"""Comparison table for learning-demo runs, built only from saved runs/<id>/ files.

Usage: python scripts/demo_summary.py OUT.md RUN_ID [RUN_ID ...]
Prints the Markdown summary and writes it to OUT.md. No LLM or network calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout.config import load_settings  # noqa: E402
from scout.events import load_trace  # noqa: E402

COLUMNS = (
    ("tool calls", "tool_calls"),
    ("cache hits", "cache_hits"),
    ("memory steps", "memory_steps"),
    ("facts reused", "facts_reused"),
    ("LLM calls", "llm_calls"),
    ("tokens", "total_tokens"),
    ("seconds", "latency_s"),
    ("coverage %", "citation_coverage"),
)


def run_row(runs_dir: Path, run_id: str) -> tuple[list[str], list[str], list[str]]:
    """(table cells, injected lessons, lessons learned) for one run."""
    metrics = json.loads((runs_dir / run_id / "metrics.json").read_text())
    trace = load_trace(runs_dir / run_id / "trace.jsonl")
    plan = next((e.payload for e in trace if e.type == "plan_created"), {})
    finished = next((e.payload for e in trace if e.type == "run_finished"), {})
    task = finished.get("task") or {}
    target = ", ".join(task.get("targets", []))
    cells = [run_id, task.get("purpose_type", "?"), target]
    cells += [str(metrics.get(key, 0)) for _, key in COLUMNS]
    injected = [f"[L{lesson['id']}] {lesson['text']}" for lesson in plan.get("lessons", [])]
    learned = [
        f"[L{e.payload['lesson_id']}] {e.payload['text']}"
        + (" (merged)" if e.payload.get("merged") else "")
        for e in trace
        if e.type == "lesson_learned"
    ]
    return cells, injected, learned


def main() -> int:
    """Build and write the summary."""
    out_path, run_ids = Path(sys.argv[1]), sys.argv[2:]
    runs_dir = load_settings().runs_dir
    header = ["run", "purpose", "target", *(name for name, _ in COLUMNS)]
    lines = [
        "# Scout learning demo",
        "",
        "| " + " | ".join(header) + " |",
        "|" + "---|" * len(header),
    ]
    details: list[str] = []
    for n, run_id in enumerate(run_ids, 1):
        cells, injected, learned = run_row(runs_dir, run_id)
        lines.append("| " + " | ".join(cells) + " |")
        details += [f"## Run {n}: {run_id}", "", "Lessons injected into the plan:"]
        details += [f"- {x}" for x in injected] or ["- none"]
        details += ["", "Lessons learned:"]
        details += [f"- {x}" for x in learned] or ["- none"]
        details.append("")
    text = "\n".join(lines + [""] + details)
    out_path.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
