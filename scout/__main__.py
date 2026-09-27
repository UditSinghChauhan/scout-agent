"""Scout CLI: ``python -m scout run "<goal>"`` with a Rich live trace."""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from scout.agent.orchestrator import Orchestrator
from scout.config import load_settings
from scout.events import Event, RunRecorder

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
        table.add_row(str(step["id"]), step["question"], ", ".join(step["suggested_tools"]))
    console.print(table)


def render_event(event: Event) -> None:
    """Print one trace event."""
    p = event.payload
    if event.type == "stage_started":
        console.rule(f"[bold cyan]{event.stage}")
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
        f"unsupported={m.get('unsupported_claims')}"
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


if __name__ == "__main__":
    app()
