"""Deterministic fakes for tests: no network, no paid APIs.

- :class:`FakeLLM` is a ``ChatBackend`` that replays scripted replies (text, dicts, Pydantic
  models, or exceptions), so ``scout.llm.LLM`` retry and repair logic runs for real.
- :class:`FakeSearch` is a search provider returning deterministic results.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from scout.config import Settings
from scout.llm import LLM, ChatResult, Message, estimate_tokens
from scout.tools.web_search import SearchResult

ScriptItem = str | dict[str, Any] | BaseModel | Exception


@dataclass
class FakeRequest:
    """A request the FakeLLM received."""

    model: str
    messages: list[Message]
    json_mode: bool
    temperature: float
    tier: str = "smart"


class FakeLLM:
    """Scripted ChatBackend. Each call consumes the next script item."""

    def __init__(self, script: Sequence[ScriptItem] = ()) -> None:
        self.script: list[ScriptItem] = list(script)
        self.requests: list[FakeRequest] = []
        self.sleeps: list[float] = []

    def add(self, *items: ScriptItem) -> FakeLLM:
        """Append more scripted replies; returns self for chaining."""
        self.script.extend(items)
        return self

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
        tier: str = "smart",
    ) -> ChatResult:
        """Record the request and return (or raise) the next scripted item."""
        request = FakeRequest(model, [dict(m) for m in messages], json_mode, temperature, tier)
        self.requests.append(request)
        if not self.script:
            raise AssertionError("FakeLLM script exhausted")
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        text = _to_text(item)
        prompt = sum(estimate_tokens(m.get("content", "")) for m in messages)
        return ChatResult(text=text, prompt_tokens=prompt, completion_tokens=estimate_tokens(text))

    def llm(
        self, settings: Settings | None = None, clock: Callable[[], float] | None = None
    ) -> LLM:
        """Build a real ``LLM`` wired to this fake, with instant (recorded) backoff sleeps."""
        base = settings or Settings(scout_model="fake-model", scout_fast_model="fake-fast")
        if clock is None:
            return LLM(settings=base, backend=self, sleep=self.sleeps.append)
        return LLM(settings=base, backend=self, sleep=self.sleeps.append, clock=clock)

    @property
    def calls(self) -> int:
        """How many chat requests were made."""
        return len(self.requests)


def _to_text(item: ScriptItem) -> str:
    """Serialize a scripted reply to the raw text the model would return."""
    if isinstance(item, BaseModel):
        return item.model_dump_json()
    if isinstance(item, dict):
        return json.dumps(item)
    return str(item)


@dataclass
class FakeSearch:
    """Deterministic search provider: ``FakeSearch()(query, max_results)``."""

    results_by_query: dict[str, list[SearchResult]] = field(default_factory=dict)
    error: Exception | None = None
    queries: list[str] = field(default_factory=list)

    def __call__(self, query: str, max_results: int) -> list[SearchResult]:
        """Return scripted results for ``query`` or generated ones; raise ``error`` if set."""
        self.queries.append(query)
        if self.error is not None:
            raise self.error
        if query in self.results_by_query:
            return self.results_by_query[query][:max_results]
        slug = "-".join(query.lower().split()) or "empty"
        return [
            {
                "title": f"Result {i} for {query}",
                "url": f"https://example.com/{slug}/{i}",
                "snippet": f"Deterministic snippet {i} about {query}.",
            }
            for i in range(1, max_results + 1)
        ]

    def provider(self, name: str = "fake") -> tuple[str, FakeSearch]:
        """Return a ``(name, provider)`` pair for ``web_search(providers=[...])``."""
        return (name, self)
