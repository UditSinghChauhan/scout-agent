"""Prompt templates stored as Markdown (docs/SPEC.md §6: prompts live in scout/prompts/*.md)."""

from __future__ import annotations

from functools import cache
from pathlib import Path
from string import Template

_DIR = Path(__file__).resolve().parent


@cache
def _read(name: str) -> str:
    """Read a prompt file once."""
    return (_DIR / f"{name}.md").read_text(encoding="utf-8")


def load_prompt(name: str, **values: object) -> str:
    """Load ``scout/prompts/<name>.md`` and fill ``$placeholders`` with ``values``."""
    return Template(_read(name)).safe_substitute({k: str(v) for k, v in values.items()})
