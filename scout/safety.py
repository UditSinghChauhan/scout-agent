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


_CONTACT_POLICY_RE = re.compile(
    r"linkedin\s*(?:profile|/in\b|people|person|member|employee)|"
    r"(?:personal|individual|employee)\s+profiles?|\be-?mails?\b|\bphone|contact (?:details|info)",
    re.IGNORECASE,
)
_META_UNKNOWN_RE = re.compile(
    r"^\s*(?:none|n/?a|nothing(?: (?:else|further))?|no (?:critical |major |significant |key |"
    r"other |further |remaining )?(?:unknowns?|gaps?|missing information)\b.*)\s*\.?\s*$",
    re.IGNORECASE,
)


def violates_contact_policy(text: str) -> bool:
    """True if text recommends personal profiles or contact details (roles-only policy)."""
    return bool(_CONTACT_POLICY_RE.search(text))


def is_meta_unknown(text: str) -> bool:
    """True for meta statements like "No critical unknowns..." or "None" in the Unknowns list."""
    return bool(_META_UNKNOWN_RE.match(text))


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
    summary = [c.model_copy(update={"text": scrub_text(c.text)}) for c in brief.summary]
    return brief.model_copy(
        update={
            "title": scrub_text(brief.title),
            "summary": summary,
            "sections": sections,
            "unknowns": [scrub_text(u) for u in brief.unknowns],
            "score_reasons": [scrub_text(r) for r in brief.score_reasons],
        }
    )
