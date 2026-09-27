"""Tiny text helpers: keyword sets and overlap similarity (Unknowns and lesson dedupe)."""

from __future__ import annotations

import re

_WORD_RE = re.compile(r"[a-z0-9]{3,}")
STOPWORDS = frozenset(
    "the and for with from that this what who how are was were their its about into over have "
    "has had not but they them you your our any all can will does did which when where there "
    "than then also such use uses used using found find could would should company".split()
)


def keywords(text: str) -> set[str]:
    """Lowercase content words (3+ chars, stopwords removed, naive plural strip)."""
    words = set()
    for w in _WORD_RE.findall(text.lower()):
        if w in STOPWORDS:
            continue
        words.add(w[:-1] if len(w) > 4 and w.endswith("s") else w)
    return words


def overlap(a: str, b: str) -> float:
    """Overlap coefficient of keyword sets: |A∩B| / min(|A|, |B|); 0 when either is empty."""
    ka, kb = keywords(a), keywords(b)
    if not ka or not kb:
        return 0.0
    return len(ka & kb) / min(len(ka), len(kb))


def dedupe(items: list[str], threshold: float = 0.6) -> list[str]:
    """Keep the first of any items whose keyword overlap reaches ``threshold``."""
    kept: list[str] = []
    for item in items:
        if not any(overlap(item, k) >= threshold for k in kept):
            kept.append(item)
    return kept
