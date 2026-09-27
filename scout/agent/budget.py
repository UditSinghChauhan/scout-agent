"""Run budgets (docs/SPEC.md §4): tool calls, LLM calls and wall clock, all from config."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field

from scout.config import Settings
from scout.llm import LLM

# LLM calls held back so synthesis (plus one repair) can always run after a budget hit.
SYNTHESIS_RESERVE = 2


@dataclass
class Budget:
    """Tracks usage against the configured limits. Never raises; callers check and stop."""

    settings: Settings
    llm: LLM
    clock: Callable[[], float] = time.monotonic
    tool_calls: int = 0
    started: float = field(default=0.0)
    exhausted_reason: str | None = None

    def __post_init__(self) -> None:
        self.started = self.clock()

    @property
    def elapsed_s(self) -> float:
        """Seconds since the run started."""
        return self.clock() - self.started

    def tool_available(self) -> bool:
        """True if another tool call fits the budget (records the reason when not)."""
        if self.tool_calls >= self.settings.max_tool_calls:
            return self._exhaust(f"tool calls ({self.settings.max_tool_calls})")
        return self._time_ok()

    def llm_available(self) -> bool:
        """True if another agent LLM call fits, keeping the synthesis reserve."""
        limit = self.settings.max_llm_calls - SYNTHESIS_RESERVE
        if self.llm.usage.llm_calls >= limit:
            return self._exhaust(f"LLM calls ({self.settings.max_llm_calls})")
        return self._time_ok()

    def charge_tool(self) -> None:
        """Count one tool call."""
        self.tool_calls += 1

    def _time_ok(self) -> bool:
        """Wall-clock check."""
        if self.elapsed_s >= self.settings.max_wall_clock_s:
            return self._exhaust(f"wall clock ({self.settings.max_wall_clock_s:.0f}s)")
        return True

    def _exhaust(self, reason: str) -> bool:
        """Remember the first budget that ran out; always returns False."""
        self.exhausted_reason = self.exhausted_reason or reason
        return False
