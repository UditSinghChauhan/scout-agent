"""Output safety (docs/SPEC.md §4 Safety): strip emails and phone numbers from the brief."""

from __future__ import annotations

import re

from scout.schemas import Brief

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_PHONE_CANDIDATE_RE = re.compile(r"\+?\(?\d[\d\s().-]{7,}\d")
EMAIL_MASK = "[email removed]"
PHONE_MASK = "[phone removed]"


def _mask_phone(match: re.Match[str]) -> str:
    """Mask a digit run that looks like a phone number (10–15 digits with separators)."""
    text = match.group(0)
    digits = sum(ch.isdigit() for ch in text)
    has_separator = any(ch in text for ch in " ()-.+")
    return PHONE_MASK if 10 <= digits <= 15 and has_separator else text


def is_personal_profile(url: str, patterns: tuple[str, ...]) -> bool:
    """True if ``url`` is an individual's profile page (never fetched or cited)."""
    low = url.lower()
    return any(p in low for p in patterns)


def scrub_text(text: str) -> str:
    """Replace emails and phone numbers in ``text``."""
    return _PHONE_CANDIDATE_RE.sub(_mask_phone, EMAIL_RE.sub(EMAIL_MASK, text))


def scrub_brief(brief: Brief) -> Brief:
    """Scrub every free-text field of the brief (citations and URLs are untouched)."""
    sections = [
        s.model_copy(
            update={"claims": [c.model_copy(update={"text": scrub_text(c.text)}) for c in s.claims]}
        )
        for s in brief.sections
    ]
    return brief.model_copy(
        update={
            "title": scrub_text(brief.title),
            "sections": sections,
            "unknowns": [scrub_text(u) for u in brief.unknowns],
            "score_reasons": [scrub_text(r) for r in brief.score_reasons],
        }
    )
