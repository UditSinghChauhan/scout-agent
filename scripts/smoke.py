"""Phase 0 manual smoke test: real LLM, real search, real fetch.

Run from the repo root:  python scripts/smoke.py
Needs .env (see .env.example). Never prints API keys. Exits non-zero if any stage fails.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pydantic import BaseModel, Field  # noqa: E402

from scout.config import Settings, load_settings  # noqa: E402
from scout.llm import LLM  # noqa: E402
from scout.tools.fetch_page import fetch_page  # noqa: E402
from scout.tools.web_search import default_providers, web_search  # noqa: E402

SEARCH_QUERY = "Zoho Corporation company overview"
FETCH_URL = "https://en.wikipedia.org/wiki/Zoho_Corporation"
FETCH_FOCUS = "products headquarters founded employees"


class CompanySnapshot(BaseModel):
    """Tiny schema used only to prove structured output works end to end."""

    company: str
    headquarters_country: str
    founded_year: int = Field(ge=1800, le=2100)
    known_for: list[str] = Field(min_length=1, max_length=3)


def banner(title: str) -> None:
    """Print a stage header."""
    print(f"\n=== {title} ===")


def check_config(settings: Settings) -> bool:
    """Show which settings are present without revealing secrets."""
    banner("1. Configuration")
    host = urlsplit(settings.llm_base_url).hostname or "(openai default)"
    print(f"LLM endpoint host : {host}")
    print(f"LLM_API_KEY       : {'set' if settings.llm_api_key else 'MISSING'}")
    print(f"SCOUT_MODEL       : {settings.scout_model or 'MISSING'}")
    print(f"SCOUT_FAST_MODEL  : {settings.scout_fast_model or '(falls back to SCOUT_MODEL)'}")
    print(f"TAVILY_API_KEY    : {'set' if settings.has_tavily else 'not set (DuckDuckGo only)'}")
    print(f"Search providers  : {[name for name, _ in default_providers(settings)]}")
    print(f"Budgets           : steps={settings.max_planned_steps}/{settings.max_total_steps}, "
          f"react={settings.max_react_iterations}, tools={settings.max_tool_calls}, "
          f"llm={settings.max_llm_calls}, wall={settings.max_wall_clock_s}s, "
          f"fetch={settings.fetch_timeout_s}s")
    return bool(settings.llm_api_key and settings.scout_model)


def check_llm(settings: Settings) -> bool:
    """Ask the real LLM for a validated CompanySnapshot."""
    banner("2. LLM complete_json (real call)")
    llm = LLM(settings=settings)
    messages = [
        {"role": "system", "content": "You are a concise company research assistant."},
        {"role": "user", "content": "Give a snapshot of Zoho Corporation."},
    ]
    started = time.monotonic()
    snapshot = llm.complete_json(messages, CompanySnapshot)
    print(f"Model: {settings.scout_model}")
    print("Validated JSON:", snapshot.model_dump_json(indent=2))
    print(f"LLM calls: {llm.usage.llm_calls} | tokens: {llm.usage.prompt_tokens} prompt + "
          f"{llm.usage.completion_tokens} completion | {time.monotonic() - started:.1f}s")
    return True


def check_search(settings: Settings) -> bool:
    """Run a real web search and show three results."""
    banner(f"3. web_search({SEARCH_QUERY!r}, max_results=3)")
    results = web_search(SEARCH_QUERY, max_results=3, settings=settings)
    for i, r in enumerate(results, 1):
        print(f"[{i}] {r['title']}\n    {r['url']}\n    {r['snippet'][:160]}")
    print(f"Results: {len(results)}")
    return len(results) == 3


def check_fetch(settings: Settings) -> bool:
    """Fetch a public page and show about 500 characters of extracted text."""
    banner(f"4. fetch_page({FETCH_URL!r})")
    text = fetch_page(FETCH_URL, FETCH_FOCUS, settings=settings)
    print(f"Extracted {len(text)} chars (showing first 500):\n")
    print(text[:500])
    return len(text) >= 500


def main() -> int:
    """Run every stage; report each outcome; exit 1 if any failed."""
    settings = load_settings()
    outcomes: dict[str, bool] = {"config": check_config(settings)}
    for name, stage in (("llm", check_llm), ("search", check_search), ("fetch", check_fetch)):
        try:
            outcomes[name] = stage(settings)
        except Exception as exc:  # noqa: BLE001 - a smoke test reports every failure
            print(f"FAILED: {type(exc).__name__}: {exc}")
            outcomes[name] = False
    banner("Summary")
    for name, ok in outcomes.items():
        print(f"{name:7s}: {'PASS' if ok else 'FAIL'}")
    return 0 if all(outcomes.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
