"""Scout CLI: ``python -m scout run "<goal>"`` with a Rich live trace."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from scout.agent.orchestrator import Orchestrator
from scout.config import load_settings
from scout.events import Event, RunRecorder, load_trace
from scout.insights import apply_feedback, insights_tables
from scout.memory.store import MemoryStore, reset_memory

app = typer.Typer(add_completion=False, help="Scout: purpose-aware company research agent.")
console = Console()


@app.callback()
def main() -> None:
    """Scout command group."""


def _render_plan(payload: dict[str, Any]) -> None:
    """Show the plan as a table."""
    table = Table(title=f"Plan ({payload.get('purpose_type')})", show_lines=False)
    table.add_column("#", style="bold")
    table.add_column("Question")
    table.add_column("Tools", style="dim")
    for step in payload.get("steps", []):
        tools = ", ".join(step["suggested_tools"])
        if step.get("answered_from_memory"):
            tools = f"memory {['F' + str(i) for i in step.get('memory_fact_ids', [])]}"
        table.add_row(str(step["id"]), step["question"], tools)
    console.print(table)
    for lesson in payload.get("lessons", []):
        console.print(f"  [green]lesson injected[/] L{lesson['id']}: {lesson['text']}")


def render_event(event: Event) -> None:
    """Print one trace event."""
    p = event.payload
    if event.type == "stage_started":
        console.rule(f"[bold cyan]{event.stage}")
        if event.stage == "recall":
            console.print(
                f"  memory: {p.get('fresh_facts')} fresh facts, {p.get('stale_facts')} stale, "
                f"{len(p.get('lessons', []))} lessons, {p.get('known_sources')} rated sources"
            )
    elif event.type == "memory_hit":
        console.print(
            f"\n[bold green]Step {p['step_id']} from memory:[/] {p['question']} "
            f"(facts {p['fact_ids']} -> {p['evidence_ids']})"
        )
    elif event.type == "lesson_learned":
        tag = "merged into" if p.get("merged") else "new"
        console.print(f"  [bold green]lesson {tag}[/] L{p.get('lesson_id')}: {p.get('text')}")
    elif event.type == "plan_created":
        _render_plan(p)
    elif event.type == "step_started":
        console.print(f"\n[bold magenta]Step {p['id']}:[/] {p['question']}")
    elif event.type == "thought":
        console.print(f"  [yellow]thought[/] {p.get('text', '')}")
    elif event.type == "tool_call":
        args = json.dumps(p.get("args", {}), ensure_ascii=False)
        console.print(f"  [blue]tool[/] {p.get('tool')}({args})")
    elif event.type == "tool_result":
        style = "green" if p.get("ok") else "red"
        console.print(f"  [{style}]result[/] {p.get('summary', '')}")
    elif event.type == "critique":
        verdict = f"step {p.get('step_id')}: {p.get('verdict')} — {p.get('reason')}"
        console.print(f"  [bold cyan]critic[/] {verdict}")
    elif event.type == "retry":
        console.print(f"  [bold yellow]retry[/] step {p.get('step_id')}: {p.get('new_approach')}")
    elif event.type == "replan":
        console.print(f"  [bold yellow]replan[/] + {p.get('added_step', {}).get('question')}")
    elif event.type == "provider_switched":
        console.print(
            f"  [bold magenta]switch[/] {p.get('tier')}: {p.get('from')} -> {p.get('to')} "
            f"({p.get('reason')})"
        )
    elif event.type == "verification":
        console.print(
            f"  verifier: flagged={p.get('flagged')} numeric={len(p.get('flagged_numeric', []))} "
            f"revised={p.get('revised')} moved_to_unknowns={len(p.get('moved_to_unknowns', []))}"
        )
    elif event.type == "synthesis":
        console.print(f"  claims: {p.get('claims')}, cited: {p.get('cited_claims')}")
    elif event.type == "error":
        console.print(f"  [bold red]error[/] {p.get('message', '')}")
    elif event.type == "run_finished":
        _render_finish(event)


def _render_finish(event: Event) -> None:
    """Print the brief and a metrics line."""
    p = event.payload
    if p.get("report_md"):
        console.print(Panel(Markdown(p["report_md"]), title="Brief", border_style="green"))
    m = p.get("metrics", {})
    console.print(
        f"[bold]run {event.run_id}[/] status={p.get('status')} tool_calls={m.get('tool_calls')} "
        f"llm_calls={m.get('llm_calls')} tokens={m.get('total_tokens')} "
        f"seconds={m.get('latency_s')} citation_coverage={m.get('citation_coverage')}% "
        f"budget={m.get('budget_reason') or 'ok'}"
    )
    console.print(
        f"models={m.get('tokens_by_model')} switches={m.get('provider_switches')} "
        f"verdicts={m.get('critique_verdicts')} retries={m.get('retries')} "
        f"replans={m.get('replans')} flagged_numeric={m.get('flagged_numeric_claims')} "
        f"unsupported={m.get('unsupported_claims')} memory_steps={m.get('memory_steps')} "
        f"facts_reused={m.get('facts_reused')} cache_hits={m.get('cache_hits')}"
    )


@app.command()
def run(
    goal: Annotated[str, typer.Argument(help="Who to research and why.")],
    max_tool_calls: Annotated[int | None, typer.Option(help="Override tool-call budget.")] = None,
    max_steps: Annotated[int | None, typer.Option(help="Override max planned steps.")] = None,
    max_llm_calls: Annotated[int | None, typer.Option(help="Override LLM-call budget.")] = None,
    user_context: Annotated[str | None, typer.Option(help="e.g. 'we sell X to Y'.")] = None,
) -> None:
    """Research a company for a purpose and write a cited brief to runs/<run_id>/."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    settings = load_settings()
    overrides: dict[str, int] = {}
    if max_tool_calls is not None:
        overrides["max_tool_calls"] = max_tool_calls
    if max_llm_calls is not None:
        overrides["max_llm_calls"] = max_llm_calls
    if max_steps is not None:
        overrides["max_planned_steps"] = max_steps
        overrides["max_total_steps"] = max(max_steps, settings.max_total_steps)
    settings = settings.with_overrides(**overrides)

    orchestrator = Orchestrator(settings)
    recorder = RunRecorder(settings.runs_dir, orchestrator.run_id)
    console.print(f"[dim]run_id {orchestrator.run_id} -> {recorder.run_dir}[/]")
    for event in recorder.record(orchestrator.run(goal, user_context)):
        render_event(event)


def _load_final(runs_dir: Path, run_id: str) -> dict[str, Any] | None:
    """The run_finished payload of a saved run (brief + evidence), or None."""
    trace = runs_dir / run_id / "trace.jsonl"
    if not trace.is_file():
        return None
    for event in reversed(load_trace(trace)):
        if event.type == "run_finished":
            return event.payload
    return None


@app.command()
def feedback(
    run_id: Annotated[str, typer.Argument(help="Run id, e.g. 20260927-091420-ac42.")],
    section: Annotated[str, typer.Argument(help="Section title (prefix, case-insensitive).")],
    rating: Annotated[str, typer.Argument(help="up or down.")],
) -> None:
    """Thumbs up/down on a report section: updates its cited domains and the run's lessons."""
    if rating not in ("up", "down"):
        raise typer.BadParameter("rating must be 'up' or 'down'")
    settings = load_settings()
    final = _load_final(settings.runs_dir, run_id)
    if not final or not final.get("brief"):
        console.print(f"[red]No finished run {run_id} in {settings.runs_dir}[/]")
        raise typer.Exit(1)
    store = MemoryStore(settings.db_path)
    result = apply_feedback(store, run_id, final, section, rating == "up")
    if result is None:
        titles = [s["title"] for s in final["brief"]["sections"]]
        console.print(f"[red]No section matching {section!r}. Sections: {titles}[/]")
        raise typer.Exit(1)
    title, domains, lesson_ids = result
    console.print(
        f"Recorded {rating} for [bold]{title}[/]: {len(domains)} domains "
        f"({', '.join(domains) or 'none'}), {len(lesson_ids)} lesson votes."
    )


@app.command()
def insights() -> None:
    """Show runs, lessons and the most and least reliable sources."""
    settings = load_settings()
    store = MemoryStore(settings.db_path)
    for table in insights_tables(store):
        console.print(table)


@app.command()
def reset(
    yes: Annotated[
        bool, typer.Option("--yes", help="Confirm deleting the memory database.")
    ] = False,
    cache: Annotated[bool, typer.Option("--cache", help="Also delete data/cache.")] = False,
) -> None:
    """Wipe Scout's memory (data/scout.db) and optionally the search/fetch cache."""
    if not yes:
        console.print("Refusing to reset without --yes.")
        raise typer.Exit(1)
    settings = load_settings()
    removed = reset_memory(settings.db_path, settings.cache_dir if cache else None)
    console.print(f"Removed: {', '.join(removed) or 'nothing (already clean)'}")


@app.command()
def ui(
    port: Annotated[int, typer.Option(help="Port for the Streamlit server.")] = 8501,
) -> None:
    """Launch the Streamlit UI (live runs, replays, insights)."""
    app_path = Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    raise typer.Exit(subprocess.call(cmd))


if __name__ == "__main__":
    app()
