"""Manual check (real API calls): does each candidate model produce clean executor JSON?

Run from the repo root:  python scripts/executor_check.py [provider/model ...]
Defaults to every candidate in the router's fast tier. For each model it sends two executor
turns (a first move, and a turn whose observation already answers the question) and reports
whether the reply validated, whether it had a thought, which action it chose, tokens and latency.
Never prints keys.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout.config import PROVIDERS, ROUTES, Candidate, load_settings, provider_key  # noqa: E402
from scout.llm import LLM, LLMError, OpenAIBackend  # noqa: E402
from scout.prompts import load_prompt  # noqa: E402
from scout.schemas import Action  # noqa: E402
from scout.tools.registry import build_registry  # noqa: E402

FIRST = (
    "## Observations\nNo observations yet. Start with a tool call.\n\n"
    "Turn 1. Reply with your next action."
)
ANSWERED = (
    '## Observations\n[1] web_search({"query": "Zoho employee count"})\n'
    "1. Zoho Corporation - Wikipedia\n   URL: https://en.wikipedia.org/wiki/Zoho_Corporation\n"
    "   Zoho Corporation ... Number of employees 18,000+ (2025). Headquarters Chennai, India.\n\n"
    "Turn 2. Reply with your next action."
)


MAX_CALLS = int(os.environ.get("SCOUT_CHECK_MAX_CALLS", "3"))


class CappedBackend:
    """Refuse requests beyond MAX_CALLS per model (the probe must stay cheap)."""

    def __init__(self, inner: OpenAIBackend) -> None:
        self.inner = inner
        self.calls = 0

    def chat(self, **kwargs: Any) -> Any:
        """Forward one request unless the cap is reached."""
        if self.calls >= MAX_CALLS:
            raise LLMError(f"call cap of {MAX_CALLS} reached")
        self.calls += 1
        return self.inner.chat(**kwargs)


def check(candidate: Candidate) -> str:
    """Run the two executor turns against one candidate; return a one-line verdict."""
    settings = load_settings()
    provider = PROVIDERS[candidate.provider]
    key = provider_key(settings, provider)
    if not key:
        return f"{candidate.label}: SKIPPED (no {provider.key_env})"
    backend = OpenAIBackend(
        key, provider.base_url, settings.llm_timeout_s, provider.name, dict(candidate.params)
    )
    llm = LLM(settings=settings.with_overrides(llm_max_retries=0), backend=CappedBackend(backend))
    system = load_prompt(
        "executor",
        question="What is Zoho's total employee count?",
        done_criteria="headcount with a source",
        purpose="interview_prep",
        tools=build_registry(settings, cache=None).describe(with_thought=True),
        max_iterations=3,
    )
    parts = []
    for name, user in (("first", FIRST), ("answered", ANSWERED)):
        started = time.monotonic()
        try:
            action = llm.complete_json(
                [{"role": "system", "content": system}, {"role": "user", "content": user}],
                Action,
                model=candidate.model,
                include_schema=False,
            )
            thought = "thought" if action.thought.strip() else "NO-THOUGHT"
            what = action.tool if action.type == "tool" else f"finish({len(action.findings)})"
            parts.append(f"{name}: {what} {thought} {time.monotonic() - started:.1f}s")
        except LLMError as exc:
            parts.append(f"{name}: FAIL {type(exc).__name__}: {str(exc)[:80]}")
    calls = llm.usage.llm_calls
    return f"{candidate.label}: {' | '.join(parts)} | calls={calls} tokens={llm.usage.total_tokens}"


def main() -> int:
    """Check the requested or default candidates."""
    wanted = sys.argv[1:]
    if wanted:
        cands = [Candidate(w.split("/", 1)[0], w.split("/", 1)[1]) for w in wanted]
    else:
        cands = list(ROUTES["fast"])
    for cand in cands:
        print(check(cand), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
