"""Synthesizer stage: evidence ledger + playbook sections -> Brief -> Markdown (docs/SPEC.md §4).

Citations are rendered as numbered references, one number per distinct source URL, followed by a
Sources list and an Unknowns section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scout.agent.executor import EvidenceLedger
from scout.briefclean import clean_brief
from scout.llm import LLM
from scout.playbooks import Playbook, playbook_brief, purpose_label
from scout.prompts import load_prompt
from scout.schemas import Brief, Claim, Evidence, Section, Task
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
        takeaways=TAKEAWAYS.get(playbook.purpose_type, TAKEAWAYS["general"]),
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
        title=f"{' and '.join(task.targets)}: verified facts",
        purpose_type=task.purpose_type,
        sections=[Section(title="Key facts", claims=claims)],
        unknowns=[f"Not written this time: {', '.join(playbook.sections)}."],
        budget_note=budget_note,
    )


def citation_stats(brief: Brief, ledger: EvidenceLedger) -> tuple[int, int]:
    """Return (claims citing at least one existing evidence id, total claims)."""
    known = ledger.by_id()
    claims = [c for s in brief.sections for c in s.claims]
    cited = sum(1 for c in claims if any(i in known for i in c.evidence_ids))
    return cited, len(claims)


def escape_md(text: str) -> str:
    """Escape "$" so Streamlit and GitHub do not render money amounts as LaTeX."""
    return re.sub(r"(?<!\\)\$", r"\\$", text)


TAKEAWAYS = {
    "sales_prospect": "For a sales prospect: the fit verdict, the pitch angle, and the main risk.",
    "competitor": "For a competitor: the threat verdict, their strongest move, and our opening.",
    "interview_prep": "For interview prep: the 3 things the candidate must prepare.",
    "general": "The 3 facts that best answer the goal.",
}


@dataclass
class BriefParts:
    """The rendered brief split around section headings (the UI adds feedback beside them)."""

    header: str
    sections: list[tuple[str, str]]  # (heading, body)
    tail: str

    def markdown(self) -> str:
        """The whole brief as one Markdown document."""
        parts = [self.header]
        parts += [f"## {escape_md(title)}\n\n{body}" for title, body in self.sections]
        parts.append(self.tail)
        return "\n".join(parts) + "\n"


def render_parts(brief: Brief, ledger: EvidenceLedger, run_id: str) -> BriefParts:
    """Render the Brief with numbered citations (one number per source URL), escaped text."""
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

    def bullet(claim: Claim) -> str:
        return f"- {escape_md(claim.text)} {cite(claim)}".rstrip()

    head = [
        f"# {escape_md(brief.title)}",
        "",
        f"_Purpose: {purpose_label(brief.purpose_type)} · Run `{run_id}`_",
        "",
    ]
    if brief.budget_note:
        head += [f"> **Partial brief:** the run stopped early ({brief.budget_note}).", ""]
    if brief.score:
        head += [f"**Score:** {brief.score}", ""]
        head += [f"- {escape_md(r)}" for r in brief.score_reasons]
        head.append("")
    if brief.summary:
        head += ["**Key takeaways**", ""]
        head += [bullet(c) for c in brief.summary]
        head.append("")
    sections = []
    for section in brief.sections:
        body = (
            "\n".join(bullet(c) for c in section.claims)
            if section.claims
            else ("_Nothing verified for this section._")
        )
        sections.append((section.title, body + "\n"))
    tail = ["## Unknowns", ""]
    tail += [f"- {escape_md(u)}" for u in brief.unknowns] or ["- None"]
    tail += ["", "## Sources", ""]
    tail += [f"{n}. <{url}>" for url, n in numbers.items()] or ["_No sources cited._"]
    return BriefParts("\n".join(head), sections, "\n".join(tail))


def render_markdown(brief: Brief, ledger: EvidenceLedger, run_id: str) -> str:
    """Render the Brief with numbered citations, a Sources list and Unknowns."""
    return render_parts(brief, ledger, run_id).markdown()


def presentable(payload: dict) -> tuple[Brief, EvidenceLedger] | None:
    """Brief and ledger from a run_finished payload, with today's display rules applied.

    Recorded briefs from older runs get the fixed score format, the jargon filter and claim
    dedupe, so a replay and a regenerated report.md show what a new run would.
    """
    raw = payload.get("brief")
    if not isinstance(raw, dict):
        return None
    ledger = EvidenceLedger(items=[Evidence.model_validate(e) for e in payload.get("evidence", [])])
    brief = Brief.model_validate(raw)
    score = normalize_score(brief.purpose_type, brief.score, brief.score_reasons)
    return clean_brief(brief.model_copy(update={"score": score})), ledger


def render_from_payload(payload: dict, run_id: str) -> str | None:
    """Re-render report.md from a run_finished payload (brief + evidence), e.g. for examples."""
    result = presentable(payload)
    return render_markdown(*result, run_id) if result else None
