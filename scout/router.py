"""Multi-provider LLM router (Phase 2, A1).

``RouterBackend`` holds an ordered candidate list per tier (``scout.config.ROUTES``) and tries
them in order:

- daily quota, or a 429 whose wait exceeds ``router_max_wait_s`` -> cooldown (until the reported
  reset time when the provider gives one) and fail over to the next candidate;
- 401/403/404 -> candidate disabled for the rest of the run, fail over;
- a short 429/5xx -> re-raised as ``TransientLLMError`` so ``LLM.complete`` backs off and retries.

Every change of the candidate serving a tier is recorded as a ``provider_switched`` notice
(from, to, reason). When a cooldown ends, the preferred candidate is used again.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace

from scout.config import PROVIDERS, ROUTES, Candidate, Settings, provider_key
from scout.llm import (
    CandidateUnavailableError,
    ChatBackend,
    ChatResult,
    LLMError,
    Message,
    OpenAIBackend,
    QuotaExhaustedError,
    TransientLLMError,
)

logger = logging.getLogger(__name__)

DAILY_COOLDOWN_S = 24 * 3600.0  # daily quota with no reported reset: sit out the whole run


@dataclass
class _State:
    """Runtime state of one candidate."""

    cooldown_until: float = 0.0
    disabled: bool = False
    reason: str = ""
    recent_429: int = 0  # consecutive short rate-limit errors


@dataclass
class RouterBackend:
    """ChatBackend that fails over between (provider, model) candidates per tier."""

    routes: dict[str, list[Candidate]]
    backends: dict[str, ChatBackend]
    max_wait_s: float = 20.0
    clock: Callable[[], float] = time.monotonic
    is_router: bool = True
    _state: dict[str, _State] = field(default_factory=dict)
    _active: dict[str, str] = field(default_factory=dict)
    _switches: list[dict[str, object]] = field(default_factory=list)

    @classmethod
    def from_settings(
        cls, settings: Settings, clock: Callable[[], float] = time.monotonic
    ) -> RouterBackend:
        """Build candidates from ``ROUTES``, skipping providers whose key is missing."""
        backends: dict[str, ChatBackend] = {}
        routes: dict[str, list[Candidate]] = {}
        for tier, candidates in ROUTES.items():
            routes[tier] = []
            for cand in candidates:
                provider = PROVIDERS[cand.provider]
                key = provider_key(settings, provider)
                if not key:
                    continue
                if cand.label not in backends:
                    backends[cand.label] = OpenAIBackend(
                        key,
                        provider.base_url,
                        settings.llm_timeout_s,
                        provider.name,
                        dict(cand.params),
                    )
                routes[tier].append(cand)
        if not any(routes.values()):
            raise LLMError("No LLM provider key found (set GROQ_API_KEY or GEMINI_API_KEY)")
        return cls(routes, backends, settings.router_max_wait_s, clock)

    def pop_switches(self) -> list[dict[str, object]]:
        """Return and clear recorded provider switches."""
        items, self._switches = self._switches, []
        return items

    def status(self) -> dict[str, str]:
        """Candidate label -> 'ok' | 'cooldown Ns' | 'disabled: reason' (for diagnostics)."""
        now = self.clock()
        out: dict[str, str] = {}
        for label in self.backends:
            st = self._state.get(label, _State())
            if st.disabled:
                out[label] = f"disabled: {st.reason}"
            elif st.cooldown_until > now:
                out[label] = f"cooldown {st.cooldown_until - now:.0f}s: {st.reason}"
            else:
                out[label] = "ok"
        return out

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
        tier: str = "smart",
    ) -> ChatResult:
        """Send the request to the first usable candidate of ``tier``, failing over as needed."""
        candidates = self.routes.get(tier) or self.routes.get("smart") or []
        reason = ""
        for cand in candidates:
            st = self._state.setdefault(cand.label, _State())
            if st.disabled or st.cooldown_until > self.clock():
                reason = reason or f"{cand.label} unavailable: {st.reason}"
                continue
            try:
                result = self.backends[cand.label].chat(
                    model=cand.model,
                    messages=messages,
                    json_mode=json_mode,
                    temperature=temperature,
                    tier=tier,
                )
            except CandidateUnavailableError as exc:
                st.disabled, st.reason = True, str(exc)
                reason = f"unavailable ({exc})"
            except QuotaExhaustedError as exc:
                wait = exc.reset_after if exc.reset_after else DAILY_COOLDOWN_S
                st.cooldown_until, st.reason = self.clock() + wait, "daily quota"
                reason = f"{exc} (reset in {wait:.0f}s)"
            except TransientLLMError as exc:
                short = exc.retry_after is None or exc.retry_after <= self.max_wait_s
                if short and st.recent_429 == 0:
                    st.recent_429 = 1
                    exc.model = cand.label
                    raise  # first short wait: LLM.complete backs off and retries this candidate
                wait = max(exc.retry_after or 0.0, self.max_wait_s if short else 0.0)
                st.cooldown_until, st.reason = self.clock() + wait, "rate limited"
                st.recent_429 = 0
                reason = f"rate limited ({wait:.0f}s cooldown)"
            else:
                st.recent_429 = 0
                self._note_active(tier, cand, self._switch_reason(tier, cand, reason))
                return replace(result, provider=cand.provider, model=cand.model)
            logger.warning("Router: %s -> %s", cand.label, reason)
            self._active.setdefault(tier, cand.label)
        raise self._exhausted(candidates, reason)

    def _switch_reason(self, tier: str, cand: Candidate, reason: str) -> str:
        """Why ``cand`` now serves ``tier`` (failover reason or a recovered preference)."""
        labels = [c.label for c in self.routes.get(tier, [])]
        previous = self._active.get(tier)
        if previous in labels and labels.index(cand.label) < labels.index(previous):
            return "preferred candidate available again"
        return reason or "failover"

    def _note_active(self, tier: str, cand: Candidate, reason: str) -> None:
        """Record a switch when a different candidate now serves ``tier``."""
        previous = self._active.get(tier)
        if previous is not None and previous != cand.label:
            self._switches.append(
                {"tier": tier, "from": previous, "to": cand.label, "reason": reason}
            )
        self._active[tier] = cand.label

    def _exhausted(self, candidates: list[Candidate], reason: str) -> LLMError:
        """Error when no candidate is usable: transient if one cools down soon, else fatal."""
        now = self.clock()
        waits = [
            self._state[c.label].cooldown_until - now
            for c in candidates
            if c.label in self._state and not self._state[c.label].disabled
        ]
        soonest = min(waits, default=math.inf)
        if soonest <= self.max_wait_s:
            return TransientLLMError(
                f"all candidates cooling down ({reason})", max(soonest, 0.0), "all candidates"
            )
        return LLMError(f"No LLM candidate available ({reason or 'none configured'})")
