"""Verifier stage (docs/SPEC.md §4, Phase 2 B2).

Deterministic checks on every claim of the Brief:

- citation: it must cite at least one existing evidence id;
- numeric fidelity: every number in the claim must appear in the snippet of an evidence item it
  cites.

Flagged claims get one LLM revision pass; anything still failing moves to Unknowns.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field

from scout.agent.executor import EvidenceLedger
from scout.llm import LLM
from scout.prompts import load_prompt
from scout.schemas import Brief, Claim

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*")
_EVIDENCE_ID_RE = re.compile(r"\bE\d+\b")


@dataclass(frozen=True)
class ClaimIssue:
    """One flagged claim."""

    claim_id: str  # "S<section>C<claim>", 1-based
    text: str
    kind: Literal["unsupported", "numeric"]
    detail: str


class ClaimRevision(BaseModel):
    """LLM rewrite (or drop) of one flagged claim."""

    id: str
    text: str = ""
    evidence_ids: list[str] = Field(default_factory=list)
    drop: bool = False


class ClaimRevisions(BaseModel):
    """Revision pass output."""

    revisions: list[ClaimRevision] = Field(default_factory=list)


def numbers_in(text: str) -> set[str]:
    """Normalised numbers in ``text`` ("30,300" -> "30300", "1.05" stays, "2023" stays)."""
    found = set()
    for raw in _NUMBER_RE.findall(_EVIDENCE_ID_RE.sub(" ", text)):
        value = raw.replace(",", "")
        if "." in value:
            value = value.rstrip("0").rstrip(".")
        found.add(value)
    return found


def check_claim(claim: Claim, ledger: EvidenceLedger) -> tuple[str, str] | None:
    """Return (kind, detail) if the claim fails a check, else None."""
    known = ledger.by_id()
    cited = [known[i] for i in claim.evidence_ids if i in known]
    if not cited:
        return "unsupported", f"cites no existing evidence id ({claim.evidence_ids or 'none'})"
    available: set[str] = set()
    for evidence in cited:
        available |= numbers_in(evidence.snippet)
    missing = sorted(numbers_in(claim.text) - available)
    if missing:
        return "numeric", f"numbers {missing} not in cited snippets"
    return None


def claim_ids(brief: Brief) -> dict[str, tuple[int, int]]:
    """Map claim id -> (section index, claim index)."""
    return {
        f"S{si + 1}C{ci + 1}": (si, ci)
        for si, section in enumerate(brief.sections)
        for ci, _ in enumerate(section.claims)
    }


def find_issues(brief: Brief, ledger: EvidenceLedger) -> list[ClaimIssue]:
    """All flagged claims in the brief."""
    issues: list[ClaimIssue] = []
    for cid, (si, ci) in claim_ids(brief).items():
        claim = brief.sections[si].claims[ci]
        problem = check_claim(claim, ledger)
        if problem:
            issues.append(ClaimIssue(cid, claim.text, problem[0], problem[1]))
    return issues


def apply_revisions(brief: Brief, revisions: ClaimRevisions) -> Brief:
    """Replace or drop revised claims (unknown ids are ignored)."""
    index = claim_ids(brief)
    sections = [s.model_copy(update={"claims": list(s.claims)}) for s in brief.sections]
    dropped: set[tuple[int, int]] = set()
    for rev in revisions.revisions:
        if rev.id not in index:
            continue
        si, ci = index[rev.id]
        if rev.drop or not rev.text.strip():
            dropped.add((si, ci))
        else:
            sections[si].claims[ci] = Claim(text=rev.text.strip(), evidence_ids=rev.evidence_ids)
    for si, section in enumerate(sections):
        section.claims = [c for ci, c in enumerate(section.claims) if (si, ci) not in dropped]
    return brief.model_copy(update={"sections": sections})


def move_to_unknowns(brief: Brief, issues: list[ClaimIssue]) -> Brief:
    """Remove flagged claims from sections and list them under Unknowns."""
    flagged = {i.claim_id for i in issues}
    index = claim_ids(brief)
    keep = {pos for cid, pos in index.items() if cid not in flagged}
    sections = [
        s.model_copy(update={"claims": [c for ci, c in enumerate(s.claims) if (si, ci) in keep]})
        for si, s in enumerate(brief.sections)
    ]
    unknowns = list(brief.unknowns) + [f"Could not verify: {i.text}" for i in issues]
    return brief.model_copy(update={"sections": sections, "unknowns": unknowns})


def revise_claims(llm: LLM, issues: list[ClaimIssue], ledger: EvidenceLedger) -> ClaimRevisions:
    """One LLM revision pass over the flagged claims."""
    flagged = "\n".join(f"- {i.claim_id}: {i.text} [problem: {i.detail}]" for i in issues)
    evidence = "\n".join(
        f"[{e.id}] {e.claim} ({e.source_url}) snippet: {e.snippet}" for e in ledger.items
    )
    prompt = load_prompt("revise", flagged=flagged, evidence=evidence)
    return llm.complete_json(
        [{"role": "system", "content": prompt}, {"role": "user", "content": "Revise."}],
        ClaimRevisions,
        include_schema=False,
    )
