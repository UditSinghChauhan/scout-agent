"""Synthesizer stage: evidence ledger + playbook sections -> Brief -> Markdown (docs/SPEC.md §4).

Citations are rendered as numbered references, one number per distinct source URL, followed by a
Sources list and an Unknowns section.
"""

from __future__ import annotations

from scout.agent.executor import EvidenceLedger
from scout.llm import LLM
from scout.playbooks import Playbook, playbook_brief
from scout.prompts import load_prompt
from scout.schemas import Brief, Claim, Section, Task
from scout.scoring import SCORE_FORMATS, normalize_score


def _ledger_text(ledger: EvidenceLedger) -> str:
    """One line per evidence item for the prompt."""
    if not ledger.items:
        return "(no evidence was gathered)"
    return "\n".join(
        f"[{e.id}] {e.claim} ({e.source_url}) — {e.snippet[:200]}" for e in ledger.items
    )


def synthesize(
    llm: LLM,
    task: Task,
    playbook: Playbook,
    ledger: EvidenceLedger,
    budget_note: str | None = None,
    critic_unknowns: list[str] | None = None,
) -> Brief:
    """Ask the LLM to write the Brief from the ledger only."""
    note = ""
    if budget_note:
        note = f"- The run stopped early: {budget_note}. Say so in `unknowns` for uncovered areas."
    prompt = load_prompt(
        "synthesizer",
        goal=task.goal,
        targets=", ".join(task.targets),
        user_context=task.user_context or "not stated",
        playbook=playbook_brief(playbook),
        sections="\n".join(f"{i}. {s}" for i, s in enumerate(playbook.sections, 1)),
        score_rule=f"{playbook.score_rubric or 'No score.'} Format: "
        + SCORE_FORMATS.get(playbook.purpose_type, SCORE_FORMATS["general"]),
        evidence=_ledger_text(ledger),
        critic_unknowns="\n".join(f"- {q}" for q in critic_unknowns or []) or "- none",
        budget_note=note,
    )
    brief = llm.complete_json(
        [{"role": "system", "content": prompt}, {"role": "user", "content": "Write the brief."}],
        Brief,
        include_schema=False,
    )
    score = normalize_score(task.purpose_type, brief.score, brief.score_reasons)
    return brief.model_copy(
        update={"purpose_type": task.purpose_type, "budget_note": budget_note, "score": score}
    )


def fallback_brief(
    task: Task, playbook: Playbook, ledger: EvidenceLedger, budget_note: str | None
) -> Brief:
    """Deterministic brief (no LLM) listing each evidence item as a cited claim."""
    claims = [Claim(text=e.claim, evidence_ids=[e.id]) for e in ledger.items]
    return Brief(
        title=f"{' and '.join(task.targets)}: research notes",
        purpose_type=task.purpose_type,
        sections=[Section(title="Key findings", claims=claims)],
        unknowns=[f"Synthesis failed; sections {', '.join(playbook.sections)} were not written."],
        budget_note=budget_note,
    )


def citation_stats(brief: Brief, ledger: EvidenceLedger) -> tuple[int, int]:
    """Return (claims citing at least one existing evidence id, total claims)."""
    known = ledger.by_id()
    claims = [c for s in brief.sections for c in s.claims]
    cited = sum(1 for c in claims if any(i in known for i in c.evidence_ids))
    return cited, len(claims)


def render_markdown(brief: Brief, ledger: EvidenceLedger, run_id: str) -> str:
    """Render the Brief with numbered citations, a Sources list and Unknowns."""
    known = ledger.by_id()
    numbers: dict[str, int] = {}  # source URL -> citation number

    def cite(claim: Claim) -> str:
        refs = []
        for eid in claim.evidence_ids:
            evidence = known.get(eid)
            if evidence is None:
                continue
            num = numbers.setdefault(evidence.source_url, len(numbers) + 1)
            if num not in refs:
                refs.append(num)
        return "".join(f"[{n}]" for n in sorted(refs)) if refs else "_(unsupported)_"

    lines = [f"# {brief.title}", "", f"_Purpose: {brief.purpose_type} · Run `{run_id}`_", ""]
    if brief.budget_note:
        lines += [f"> **Partial brief:** the run stopped early ({brief.budget_note}).", ""]
    if brief.score:
        lines += [f"**Score:** {brief.score}", ""]
        lines += [f"- {r}" for r in brief.score_reasons]
        lines.append("")
    for section in brief.sections:
        lines += [f"## {section.title}", ""]
        if not section.claims:
            lines += ["_No verified findings._", ""]
            continue
        lines += [f"- {c.text} {cite(c)}".rstrip() for c in section.claims]
        lines.append("")
    lines += ["## Unknowns", ""]
    lines += [f"- {u}" for u in brief.unknowns] or ["- None"]
    lines += ["", "## Sources", ""]
    lines += [f"{n}. <{url}>" for url, n in numbers.items()] or ["_No sources cited._"]
    return "\n".join(lines) + "\n"
