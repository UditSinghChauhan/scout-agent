"""Thin OpenAI-compatible LLM wrapper: ``complete``, ``complete_json``, retries, token counting.

The wrapper talks to a small :class:`ChatBackend` protocol. :class:`OpenAIBackend` adapts the
``openai`` SDK to any OpenAI-compatible endpoint (Groq, Gemini, OpenRouter, Ollama); tests plug in
``tests.fakes.FakeLLM`` instead, so the retry and repair logic below is exercised without network.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

import openai
from pydantic import BaseModel, ValidationError

from scout.config import Settings, load_settings

logger = logging.getLogger(__name__)

Message = dict[str, str]
ModelT = TypeVar("ModelT", bound=BaseModel)

# Protocol messages for structured output. Kept short; stage prompts live in scout/prompts/*.md.
_JSON_INSTRUCTION = (
    "Respond with a single JSON object only, no prose or code fences. "
    "It must validate against this JSON Schema:\n{schema}"
)
_REPAIR_INSTRUCTION = (
    "Your previous reply was not a valid JSON object of the required shape. Errors:\n{errors}\n"
    "Reply again with only the corrected JSON object."
)
_MAX_ERROR_CHARS = 2_000
_WAIT_RE = re.compile(
    r"(?:retry|try again) in\s+((?:\d+h)?\s*(?:\d+m(?!s))?\s*(?:[\d.]+s)?)", re.IGNORECASE
)
_REJECTED_GENERATION_CODES = {"tool_use_failed", "json_validate_failed"}
_LIMIT_KIND_RE = re.compile(r"on ((?:tokens|requests) per \w+ \(\w+\))", re.IGNORECASE)
_DAILY_QUOTA_RE = re.compile(r"per ?day|RPD\b", re.IGNORECASE)


class LLMError(Exception):
    """A non-recoverable LLM failure (bad request, auth, exhausted retries)."""


class TransientLLMError(LLMError):
    """A retryable failure: HTTP 429, HTTP 5xx, or a connection problem."""

    def __init__(
        self, message: str, retry_after: float | None = None, model: str | None = None
    ) -> None:
        super().__init__(message)
        self.retry_after = retry_after
        self.model = model  # "provider/model" when known (set by the router)


class QuotaExhaustedError(LLMError):
    """A daily (or other long-window) quota is used up; ``reset_after`` seconds if known."""

    def __init__(self, message: str, reset_after: float | None = None) -> None:
        super().__init__(message)
        self.reset_after = reset_after


class CandidateUnavailableError(LLMError):
    """401/403/404: this provider or model cannot be used in this run."""


class DeadlineExceededError(LLMError):
    """Waiting for the provider would overshoot the run's wall-clock budget."""


class LLMValidationError(LLMError):
    """The model output failed schema validation even after the repair request."""

    def __init__(self, message: str, raw: str) -> None:
        super().__init__(message)
        self.raw = raw


@dataclass(frozen=True)
class ChatResult:
    """One chat completion: text plus token usage (estimated when the provider omits it)."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    estimated: bool = False
    provider: str = ""
    model: str = ""


class ChatBackend(Protocol):
    """Minimal transport the wrapper needs; implemented by OpenAIBackend, RouterBackend, FakeLLM.

    ``tier`` is "smart" (planner, synthesizer, reflector) or "fast" (executor, critic); a router
    uses it to pick a model, a single-provider backend ignores it.
    """

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
        tier: str = "smart",
    ) -> ChatResult:
        """Send one chat request and return the reply."""
        ...


@dataclass(frozen=True)
class CallRecord:
    """Token and latency accounting for one LLM request."""

    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    estimated: bool
    provider: str = ""
    tier: str = "smart"


@dataclass
class UsageTracker:
    """Accumulates per-call records; feeds RunMetrics in later phases."""

    calls: list[CallRecord] = field(default_factory=list)

    @property
    def llm_calls(self) -> int:
        """Number of successful LLM requests."""
        return len(self.calls)

    @property
    def prompt_tokens(self) -> int:
        """Total prompt tokens."""
        return sum(c.prompt_tokens for c in self.calls)

    @property
    def completion_tokens(self) -> int:
        """Total completion tokens."""
        return sum(c.completion_tokens for c in self.calls)

    @property
    def total_tokens(self) -> int:
        """Prompt plus completion tokens."""
        return self.prompt_tokens + self.completion_tokens

    def tokens_by_model(self) -> dict[str, int]:
        """Total tokens per ``provider/model``."""
        totals: dict[str, int] = {}
        for c in self.calls:
            key = f"{c.provider}/{c.model}" if c.provider else c.model
            totals[key] = totals.get(key, 0) + c.prompt_tokens + c.completion_tokens
        return totals

    def reset(self) -> None:
        """Forget all recorded calls."""
        self.calls.clear()


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for providers that omit usage."""
    return max(1, len(text) // 4) if text else 0


def is_daily_quota(exc: Exception) -> bool:
    """True for a 429 caused by a per-day quota: retrying within the run cannot help."""
    return (
        isinstance(exc, openai.APIStatusError)
        and exc.status_code == 429
        and bool(_DAILY_QUOTA_RE.search(str(exc)))
    )


def is_unavailable(exc: Exception) -> bool:
    """401/403/404: bad or inactive key, no access, or unknown model."""
    return isinstance(exc, openai.APIStatusError) and exc.status_code in (401, 403, 404)


def is_transient(exc: Exception) -> bool:
    """Return True for OpenAI SDK errors worth retrying: 429, 5xx, connection or timeout."""
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        if is_daily_quota(exc):
            return False
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def rejected_generation(exc: Exception) -> str | None:
    """Model output the provider refused on its own validation, or None for other errors.

    Groq answers 400 ``tool_use_failed`` when a model (e.g. gpt-oss) emits a native tool call,
    and ``json_validate_failed`` when JSON mode output does not parse; either way the output (maybe
    empty) is in ``failed_generation``. Returning it lets ``complete_json`` validate it and send
    its single repair request instead of failing the whole step.
    """
    if not isinstance(exc, openai.BadRequestError) or not isinstance(exc.body, dict):
        return None
    error = exc.body.get("error", exc.body)
    if not isinstance(error, dict) or error.get("code") not in _REJECTED_GENERATION_CODES:
        return None
    generation = error.get("failed_generation")
    return generation if isinstance(generation, str) else ""


def parse_wait(text: str) -> float | None:
    """Seconds from hints like "retry in 31.6s" or "try again in 24m47.8s" / "1h2m3s"."""
    match = _WAIT_RE.search(text)
    if not match or not match.group(1).strip():
        return None
    total = 0.0
    for value, unit in re.findall(r"([\d.]+)\s*([hms])", match.group(1)):
        total += float(value) * {"h": 3600, "m": 60, "s": 1}[unit]
    return total


def _retry_after(exc: Exception) -> float | None:
    """Retry delay from the Retry-After header, or a wait hint in the error message."""
    if not isinstance(exc, openai.APIStatusError):
        return None
    value = exc.response.headers.get("retry-after")
    try:
        if value is not None:
            return float(value)
    except ValueError:
        pass
    return parse_wait(str(exc))


class OpenAIBackend:
    """ChatBackend over the ``openai`` SDK for one OpenAI-compatible endpoint."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "",
        timeout: float = 60.0,
        provider: str = "",
        params: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise LLMError("LLM_API_KEY is not set (see .env.example)")
        self.provider = provider
        self.params = params or {}
        # SDK retries are disabled: LLM.complete owns backoff so behaviour is uniform and testable.
        self._client = openai.OpenAI(
            api_key=api_key, base_url=base_url or None, timeout=timeout, max_retries=0
        )

    @classmethod
    def from_settings(cls, settings: Settings) -> OpenAIBackend:
        """Single-provider backend from the LLM_* settings."""
        return cls(settings.llm_api_key, settings.llm_base_url, settings.llm_timeout_s)

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
        tier: str = "smart",
    ) -> ChatResult:
        """Call ``chat.completions.create`` and map SDK errors onto Scout's LLM errors."""
        extra: dict[str, object] = dict(self.params)
        if json_mode:
            extra["response_format"] = {"type": "json_object"}
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=list(messages),  # type: ignore[arg-type]
                temperature=temperature,
                **extra,  # type: ignore[arg-type]
            )
        except openai.OpenAIError as exc:
            rejected = rejected_generation(exc)
            if rejected is not None:
                logger.info("Provider rejected a native tool call; using its text as the reply")
                prompt = sum(estimate_tokens(m.get("content", "")) for m in messages)
                return ChatResult(
                    rejected, prompt, estimate_tokens(rejected), True, self.provider, model
                )
            if is_transient(exc):
                raise TransientLLMError(type(exc).__name__, _retry_after(exc)) from exc
            if is_daily_quota(exc):
                reset = _retry_after(exc)
                limit = _LIMIT_KIND_RE.search(str(exc))
                kind = limit.group(1) if limit else "per-day limit"
                raise QuotaExhaustedError(f"quota exhausted: {kind}", reset) from exc
            if is_unavailable(exc):
                code = getattr(exc, "status_code", "?")
                raise CandidateUnavailableError(f"HTTP {code} for {model}") from exc
            raise LLMError(f"{type(exc).__name__}: {str(exc)[:300]}") from exc
        text = (response.choices[0].message.content or "") if response.choices else ""
        usage = response.usage
        if usage is None:
            prompt = sum(estimate_tokens(m.get("content", "")) for m in messages)
            return ChatResult(text, prompt, estimate_tokens(text), True, self.provider, model)
        return ChatResult(
            text, usage.prompt_tokens, usage.completion_tokens, False, self.provider, model
        )


def extract_json_text(text: str) -> str:
    """Strip code fences or surrounding prose so only the JSON object remains."""
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.split("\n", 1)[1] if "\n" in stripped else ""
        stripped = stripped.rsplit("```", 1)[0].strip()
    if not stripped.startswith("{"):
        start, end = stripped.find("{"), stripped.rfind("}")
        if start != -1 and end > start:
            stripped = stripped[start : end + 1]
    return stripped


def format_validation_error(exc: ValidationError) -> str:
    """Compact, model-readable summary of a Pydantic validation error."""
    lines = []
    for err in exc.errors(include_url=False):
        loc = ".".join(str(p) for p in err["loc"]) or "<root>"
        lines.append(f"- {loc}: {err['msg']}")
    return "\n".join(lines)[:_MAX_ERROR_CHARS]


def _with_schema_instruction(messages: Sequence[Message], schema: type[BaseModel]) -> list[Message]:
    """Prepend the JSON-schema instruction to the system message (or add one)."""
    instruction = _JSON_INSTRUCTION.format(schema=json.dumps(schema.model_json_schema()))
    msgs = [dict(m) for m in messages]
    if msgs and msgs[0].get("role") == "system":
        msgs[0]["content"] = f"{msgs[0]['content']}\n\n{instruction}"
    else:
        msgs.insert(0, {"role": "system", "content": instruction})
    return msgs


class LLM:
    """Scout's LLM client: plain completions, validated JSON, backoff and token accounting.

    ``deadline`` (a ``clock()`` value) bounds backoff waits: a wait that would overshoot it raises
    :class:`DeadlineExceededError` instead of sleeping (Phase 2, A4). ``notices`` collects one
    entry per call and per provider switch; the orchestrator turns them into trace events.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        backend: ChatBackend | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings or load_settings()
        self._backend = backend
        self._sleep = sleep
        self._clock = clock
        self.deadline: float | None = None
        self.usage = UsageTracker()
        self.notices: list[tuple[str, dict[str, object]]] = []
        # Called synchronously before a backoff wait, so a UI can show it while it waits.
        self.listener: Callable[[str, dict[str, object]], None] | None = None

    @property
    def backend(self) -> ChatBackend:
        """The transport, created lazily so constructing an LLM never needs a key."""
        if self._backend is None:
            self._backend = make_backend(self.settings, self._clock)
        return self._backend

    @property
    def routed(self) -> bool:
        """True when a RouterBackend picks the model per tier."""
        return bool(getattr(self.backend, "is_router", False))

    def _model(self, model: str | None, fast: bool) -> str:
        """Resolve which model name to use for a call (the router picks its own)."""
        chosen = model or (self.settings.fast_model if fast else self.settings.scout_model)
        if not chosen and not self.routed:
            raise LLMError("SCOUT_MODEL is not set (see .env.example)")
        return chosen

    def drain_notices(self) -> list[tuple[str, dict[str, object]]]:
        """Return and clear pending (event_type, payload) notices, including router switches."""
        pop = getattr(self.backend, "pop_switches", None)
        switches = pop() if callable(pop) else []
        items = [("provider_switched", sw) for sw in switches] + self.notices
        self.notices = []
        return items

    def complete(
        self,
        messages: Sequence[Message],
        *,
        fast: bool = False,
        model: str | None = None,
        json_mode: bool = False,
        temperature: float | None = None,
    ) -> str:
        """Return the reply text, retrying 429/5xx/connection errors with exponential backoff."""
        name = self._model(model, fast)
        tier = "fast" if fast else "smart"
        temp = self.settings.llm_temperature if temperature is None else temperature
        attempts = 1 + max(0, self.settings.llm_max_retries)
        for attempt in range(attempts):
            started = self._clock()
            try:
                result = self.backend.chat(
                    model=name, messages=messages, json_mode=json_mode, temperature=temp, tier=tier
                )
            except TransientLLMError as exc:
                if attempt == attempts - 1:
                    raise LLMError(f"LLM call failed after {attempts} attempts: {exc}") from exc
                delay = self._backoff(attempt, exc.retry_after)
                self._check_deadline(delay)
                logger.warning("Transient LLM error (%s); retrying in %.1fs", exc, delay)
                self._announce_wait(exc.model or name or tier, tier, delay)
                self._sleep(delay)
                continue
            self._record(result, name, tier, self._clock() - started)
            return result.text
        raise AssertionError("unreachable")  # pragma: no cover

    def _record(self, result: ChatResult, name: str, tier: str, latency: float) -> None:
        """Account tokens and queue an ``llm_call`` notice for the trace."""
        record = CallRecord(
            model=result.model or name,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_s=round(latency, 2),
            estimated=result.estimated,
            provider=result.provider,
            tier=tier,
        )
        self.usage.calls.append(record)
        self.notices.append(
            (
                "llm_call",
                {
                    "provider": record.provider,
                    "model": record.model,
                    "tier": tier,
                    "prompt_tokens": record.prompt_tokens,
                    "completion_tokens": record.completion_tokens,
                    "latency_s": record.latency_s,
                },
            )
        )

    def _announce_wait(self, model: str, tier: str, delay: float) -> None:
        """Record a ``rate_limited`` notice and tell the listener right away."""
        payload: dict[str, object] = {"model": model, "tier": tier, "wait_s": round(delay, 1)}
        self.notices.append(("rate_limited", payload))
        if self.listener is not None:
            try:
                self.listener("rate_limited", payload)
            except Exception:  # noqa: BLE001 - a UI callback must never break a run
                logger.exception("rate_limited listener failed")

    def _check_deadline(self, delay: float) -> None:
        """Refuse a backoff wait that would overshoot the wall-clock deadline."""
        if self.deadline is not None and self._clock() + delay > self.deadline:
            raise DeadlineExceededError(f"backoff of {delay:.0f}s would overshoot the wall clock")

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        """Exponential delay for ``attempt`` (0-based), honouring Retry-After, capped."""
        delay = self.settings.llm_backoff_base_s * (2**attempt)
        if retry_after is not None:
            delay = max(delay, retry_after)
        return min(delay, self.settings.llm_backoff_max_s)

    def complete_json(
        self,
        messages: Sequence[Message],
        schema: type[ModelT],
        *,
        fast: bool = False,
        model: str | None = None,
        include_schema: bool = True,
    ) -> ModelT:
        """Return a validated ``schema`` instance; one repair request on failure, then raise.

        ``include_schema=False`` skips embedding the JSON Schema when the prompt already spells
        out the format (saves tokens); validation is unchanged.
        """
        msgs = _with_schema_instruction(messages, schema) if include_schema else list(messages)
        json_mode = self.settings.llm_json_mode
        raw = self.complete(msgs, fast=fast, model=model, json_mode=json_mode)
        try:
            return schema.model_validate_json(extract_json_text(raw))
        except ValidationError as exc:
            errors = format_validation_error(exc)
            logger.info("complete_json: %s failed validation, sending repair", schema.__name__)
        previous = [{"role": "assistant", "content": raw}] if raw.strip() else []
        repair = [
            *msgs,
            *previous,
            {"role": "user", "content": _REPAIR_INSTRUCTION.format(errors=errors)},
        ]
        raw = self.complete(repair, fast=fast, model=model, json_mode=json_mode)
        try:
            return schema.model_validate_json(extract_json_text(raw))
        except ValidationError as exc:
            detail = format_validation_error(exc)
            raise LLMValidationError(
                f"{schema.__name__} still invalid after repair:\n{detail}", raw
            ) from exc


def make_backend(settings: Settings, clock: Callable[[], float] = time.monotonic) -> ChatBackend:
    """RouterBackend when provider keys are configured, else the single LLM_* provider."""
    from scout.config import router_enabled
    from scout.router import RouterBackend

    if router_enabled(settings):
        return RouterBackend.from_settings(settings, clock)
    return OpenAIBackend.from_settings(settings)


_default_llm: LLM | None = None


def get_llm() -> LLM:
    """Return the process-wide default LLM built from environment settings."""
    global _default_llm
    if _default_llm is None:
        _default_llm = LLM()
    return _default_llm


def complete(messages: Sequence[Message], **kwargs: object) -> str:
    """Module-level shortcut for ``get_llm().complete``."""
    return get_llm().complete(messages, **kwargs)  # type: ignore[arg-type]


def complete_json(messages: Sequence[Message], schema: type[ModelT], **kwargs: object) -> ModelT:
    """Module-level shortcut for ``get_llm().complete_json`` (docs/SPEC.md §4 LLM protocol)."""
    return get_llm().complete_json(messages, schema, **kwargs)  # type: ignore[arg-type]
