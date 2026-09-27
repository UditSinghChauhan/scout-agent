"""Deterministic post-pass on the Brief: no internal jargon, no duplicate claims.

The brief is for the user, so it must not mention how Scout works internally ("ledger",
"evidence id", "E12", "finding", "step"). Evidence-id tags are removed, a few terms are
rewritten, and a line that still contains jargon is dropped. A claim that repeats an earlier one
(same evidence ids, or keyword overlap >= 0.8) is dropped, keeping the first occurrence.
"""

from __future__ import annotations

import re

from scout.playbooks import PURPOSE_LABELS
from scout.schemas import Brief, Claim, Section
from scout.textutil import overlap

DUPLICATE_OVERLAP = 0.8
_ID_TAG_RE = re.compile(
    r"\s*[\(\[]\s*(?:evidence(?: ids?)?:?\s*)?E\d+(?:\s*[,;/&]\s*(?:and\s+)?E\d+)*\s*[\)\]]",
    re.IGNORECASE,
)
_BARE_ID_RE = re.compile(r"\b(?:evidence\s+)?E\d+\b")
_REWRITES = (
    (re.compile(r"\b(?:the\s+)?evidence\s+ledger\b", re.I), "the sources"),
    (re.compile(r"\b(?:the\s+)?ledger\b", re.I), "the sources"),
    (re.compile(r"\bstep-by-step\b", re.I), "detailed"),
    (re.compile(r"\bfindings\b", re.I), "facts"),
    (re.compile(r"\bfinding\b", re.I), "fact"),
    (re.compile(r"\bsteps\b", re.I), "stages"),
    (re.compile(r"\bstep\b", re.I), "stage"),
)
JARGON_RE = re.compile(r"\b(?:ledger|evidence ids?|E\d+|findings?|steps?)\b", re.IGNORECASE)


def humanize_purposes(text: str) -> str:
    """Replace purpose identifiers ("sales_prospect") with their labels ("Sales prospect")."""
    for key, label in PURPOSE_LABELS.items():
        text = re.sub(rf"\b{key}\b", label, text)
    return text


def clean_text(text: str) -> str | None:
    """Text without internal jargon, or None if the line cannot be salvaged."""
    out = _ID_TAG_RE.sub("", humanize_purposes(text))
    out = _BARE_ID_RE.sub("", out)
    for pattern, replacement in _REWRITES:
        out = pattern.sub(replacement, out)
    out = re.sub(r"\s{2,}", " ", out).replace(" .", ".").replace(" ,", ",").strip(" ,;")
    if not out or JARGON_RE.search(out):
        return None
    return out


def _clean_claims(claims: list[Claim]) -> list[Claim]:
    """Claims with cleaned text; claims that cannot be cleaned are dropped."""
    kept = []
    for claim in claims:
        text = clean_text(claim.text)
        if text:
            kept.append(claim.model_copy(update={"text": text}))
    return kept


def _is_duplicate(claim: Claim, seen: list[Claim]) -> bool:
    """Same evidence ids as an earlier claim, or near-identical wording."""
    ids = sorted(claim.evidence_ids)
    for earlier in seen:
        if ids and ids == sorted(earlier.evidence_ids):
            return True
        if overlap(claim.text, earlier.text) >= DUPLICATE_OVERLAP:
            return True
    return False


def dedupe_claims(brief: Brief) -> Brief:
    """Drop claims that repeat an earlier section's claim (first occurrence wins)."""
    seen: list[Claim] = []
    sections = []
    for section in brief.sections:
        kept = [c for c in section.claims if not _is_duplicate(c, seen)]
        seen += kept
        sections.append(section.model_copy(update={"claims": kept}))
    return brief.model_copy(update={"sections": sections})


def clean_brief(brief: Brief) -> Brief:
    """Jargon filter over every user-facing text field, then cross-section dedupe."""
    sections = [Section(title=s.title, claims=_clean_claims(s.claims)) for s in brief.sections]
    cleaned = brief.model_copy(
        update={
            "title": humanize_purposes(brief.title),
            "summary": _clean_claims(brief.summary),
            "sections": sections,
            "score_reasons": [t for r in brief.score_reasons if (t := clean_text(r))],
            "unknowns": [t for u in brief.unknowns if (t := clean_text(u))],
        }
    )
    return dedupe_claims(cleaned)
