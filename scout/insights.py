"""Insights and feedback over the memory store (docs/SPEC.md §3: learning made visible)."""

from __future__ import annotations

from typing import Any

from rich.table import Table

from scout.memory.store import MemoryStore, domain_of, score


def find_section(brief: dict[str, Any], name: str) -> dict[str, Any] | None:
    """Section whose title starts with (or else contains) ``name``, case-insensitive."""
    wanted = name.strip().lower()
    sections = brief.get("sections", [])
    for sec in sections:
        if sec["title"].lower().startswith(wanted):
            return sec
    return next((s for s in sections if wanted in s["title"].lower()), None)


def apply_feedback(
    store: MemoryStore, run_id: str, final: dict[str, Any], section: str, up: bool
) -> tuple[str, list[str], list[int]] | None:
    """Rate the domains cited in a section and vote on the lessons injected into the run.

    Returns (section title, domains rated, lesson ids voted), or None if no section matches.
    """
    sec = find_section(final["brief"], section)
    if sec is None:
        return None
    evidence = {e["id"]: e for e in final.get("evidence", [])}
    cited = {i for claim in sec["claims"] for i in claim.get("evidence_ids", [])}
    domains = sorted({domain_of(evidence[i]["source_url"]) for i in cited if i in evidence})
    for domain in domains:
        store.rate_source(domain, up)
    run = store.run(run_id) or {}
    lesson_ids = [int(i) for i in run.get("metrics", {}).get("injected_lessons", []) if i]
    for lesson_id in lesson_ids:
        store.vote(lesson_id, up)
    store.add_feedback(run_id, sec["title"], "up" if up else "down")
    return sec["title"], domains, lesson_ids


def _runs_table(store: MemoryStore) -> Table:
    """One row per run with the learning-relevant metrics."""
    table = Table(title="Runs")
    for col in (
        "run",
        "purpose",
        "target",
        "tools",
        "cache",
        "memory steps",
        "LLM",
        "tokens",
        "seconds",
        "coverage %",
    ):
        table.add_column(
            col, justify="right" if col not in ("run", "purpose", "target") else "left"
        )
    for r in store.runs():
        m = r["metrics"]
        table.add_row(
            r["id"],
            r["purpose_type"],
            ", ".join(r["targets"]),
            str(m.get("tool_calls", 0)),
            str(m.get("cache_hits", 0)),
            str(m.get("memory_steps", 0)),
            str(m.get("llm_calls", 0)),
            str(m.get("total_tokens", 0)),
            str(m.get("latency_s", 0)),
            str(m.get("citation_coverage", 0)),
        )
    if not table.rows:
        table.caption = "No runs yet."
    return table


def _lessons_table(store: MemoryStore) -> Table:
    """Lessons with votes, score and how often they were injected."""
    table = Table(title="Lessons")
    for col in ("id", "purpose", "lesson", "votes", "score", "uses"):
        table.add_column(col)
    uses = store.lesson_uses()
    for lesson in store.lessons():
        table.add_row(
            str(lesson.id),
            lesson.purpose_type,
            lesson.text,
            f"+{lesson.votes_up}/-{lesson.votes_down}",
            f"{score(lesson.votes_up, lesson.votes_down):.2f}",
            str(uses.get(lesson.id or 0, 0)),
        )
    if not table.rows:
        table.caption = "No lessons yet."
    return table


def _sources_table(title: str, rows: list[dict[str, Any]]) -> Table:
    """Sources with useful/useless counts and score."""
    table = Table(title=title)
    for col in ("domain", "useful", "useless", "score"):
        table.add_column(col)
    for r in rows:
        table.add_row(r["domain"], str(r["useful"]), str(r["useless"]), f"{r['score']:.2f}")
    if not table.rows:
        table.caption = "No rated sources yet."
    return table


def insights_tables(store: MemoryStore, n: int = 5) -> list[Table]:
    """Runs, lessons, top-n and bottom-n sources."""
    sources = store.sources()
    bottom = list(reversed(sources[-n:])) if len(sources) > n else []
    return [
        _runs_table(store),
        _lessons_table(store),
        _sources_table(f"Top {n} sources", sources[:n]),
        _sources_table(f"Bottom {n} sources", bottom),
    ]
