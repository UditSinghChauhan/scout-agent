"""Fixed score formats per playbook, enforced in code (never free text).

- sales_prospect: "Fit 75/100"
- competitor:     "Threat: Low" | "Threat: Medium" | "Threat: High"
- interview_prep: "Readiness 4/6" (checklist items ready out of total)
- general:        no score

``normalize_score`` accepts what a model tends to write ("75", "Fit score: 75/100", "medium",
"4 of 6") and returns the canonical string, or None when nothing valid can be read.
"""

from __future__ import annotations

import re

_INT_RE = re.compile(r"\d+")
_RATIO_RE = re.compile(r"(\d+)\s*(?:/|of|out of)\s*(\d+)", re.IGNORECASE)
_THREAT_RE = re.compile(r"\b(low|medium|moderate|high)\b", re.IGNORECASE)
_READY_PREFIX = re.compile(r"^\s*(?:\[x\]|✅|ready\b)", re.IGNORECASE)
_TODO_PREFIX = re.compile(r"^\s*(?:\[ \]|⬜|todo\b|not ready\b)", re.IGNORECASE)

SCORE_FORMATS = {
    "sales_prospect": 'Exactly "Fit NN/100" (NN = 0-100), e.g. "Fit 75/100".',
    "competitor": 'Exactly one of "Threat: Low", "Threat: Medium", "Threat: High".',
    "interview_prep": (
        'Exactly "Readiness X/Y": Y checklist items in `score_reasons`, each starting with '
        '"ready: " or "todo: "; X = number of ready items.'
    ),
    "general": "null (no score for this purpose).",
}


def _fit(raw: str) -> str | None:
    """Fit score out of 100."""
    ratio = _RATIO_RE.search(raw)
    if ratio and int(ratio.group(2)) == 100:
        value = int(ratio.group(1))
    else:
        numbers = [int(n) for n in _INT_RE.findall(raw)]
        if len(numbers) != 1:
            return None
        value = numbers[0]
    return f"Fit {value}/100" if 0 <= value <= 100 else None


def _threat(raw: str) -> str | None:
    """Threat level."""
    match = _THREAT_RE.search(raw)
    if not match:
        return None
    level = match.group(1).lower()
    level = "medium" if level == "moderate" else level
    return f"Threat: {level.capitalize()}"


def _readiness(raw: str, reasons: list[str]) -> str | None:
    """Checklist readiness: from "X/Y" in the score, else from ready/todo checklist items."""
    ratio = _RATIO_RE.search(raw)
    if ratio:
        ready, total = int(ratio.group(1)), int(ratio.group(2))
        return f"Readiness {ready}/{total}" if 0 <= ready <= total and total > 0 else None
    marked = [r for r in reasons if _READY_PREFIX.match(r) or _TODO_PREFIX.match(r)]
    if not marked:
        return None
    ready = sum(1 for r in marked if _READY_PREFIX.match(r))
    return f"Readiness {ready}/{len(marked)}"


def normalize_score(purpose: str, raw: str | None, reasons: list[str] | None = None) -> str | None:
    """Canonical score for ``purpose``, or None if the raw score cannot be validated."""
    text = (raw or "").strip()
    if purpose == "sales_prospect":
        return _fit(text) if text else None
    if purpose == "competitor":
        return _threat(text) if text else None
    if purpose == "interview_prep":
        return _readiness(text, reasons or [])
    return None
