"""Thin OpenAI-compatible LLM wrapper: ``complete``, ``complete_json``, retries, token counting.

The wrapper talks to a small :class:`ChatBackend` protocol. :class:`OpenAIBackend` adapts the
``openai`` SDK to any OpenAI-compatible endpoint (Groq, Gemini, OpenRouter, Ollama); tests plug in
``tests.fakes.FakeLLM`` instead, so the retry and repair logic below is exercised without network.
"""

from __future__ import annotations

import json
import logging
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
    "Your previous reply did not validate against the required JSON Schema. Errors:\n{errors}\n"
    "Reply again with only the corrected JSON object."
)
_MAX_ERROR_CHARS = 2_000


class LLMError(Exception):
    """A non-recoverable LLM failure (bad request, auth, exhausted retries)."""


class TransientLLMError(LLMError):
    """A retryable failure: HTTP 429, HTTP 5xx, or a connection problem."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


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


class ChatBackend(Protocol):
    """Minimal transport the wrapper needs; implemented by OpenAIBackend and FakeLLM."""

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
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

    def reset(self) -> None:
        """Forget all recorded calls."""
        self.calls.clear()


def estimate_tokens(text: str) -> int:
    """Rough token estimate (~4 characters per token) for providers that omit usage."""
    return max(1, len(text) // 4) if text else 0


def is_transient(exc: Exception) -> bool:
    """Return True for OpenAI SDK errors worth retrying: 429, 5xx, connection or timeout."""
    if isinstance(exc, openai.APIConnectionError):
        return True
    if isinstance(exc, openai.APIStatusError):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


def _retry_after(exc: Exception) -> float | None:
    """Read a numeric Retry-After header from an SDK status error, if present."""
    if not isinstance(exc, openai.APIStatusError):
        return None
    value = exc.response.headers.get("retry-after")
    try:
        return float(value) if value is not None else None
    except ValueError:
        return None


class OpenAIBackend:
    """ChatBackend over the ``openai`` SDK, pointed at ``LLM_BASE_URL``."""

    def __init__(self, settings: Settings) -> None:
        if not settings.llm_api_key:
            raise LLMError("LLM_API_KEY is not set (see .env.example)")
        # SDK retries are disabled: LLM.complete owns backoff so behaviour is uniform and testable.
        self._client = openai.OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url or None,
            timeout=settings.llm_timeout_s,
            max_retries=0,
        )

    def chat(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        json_mode: bool,
        temperature: float,
    ) -> ChatResult:
        """Call ``chat.completions.create`` and map SDK errors onto Scout's LLM errors."""
        extra = {"response_format": {"type": "json_object"}} if json_mode else {}
        try:
            response = self._client.chat.completions.create(
                model=model,
                messages=list(messages),  # type: ignore[arg-type]
                temperature=temperature,
                **extra,
            )
        except openai.OpenAIError as exc:
            if is_transient(exc):
                raise TransientLLMError(type(exc).__name__, _retry_after(exc)) from exc
            raise LLMError(f"{type(exc).__name__}: {exc}") from exc
        text = (response.choices[0].message.content or "") if response.choices else ""
        usage = response.usage
        if usage is None:
            prompt = sum(estimate_tokens(m.get("content", "")) for m in messages)
            return ChatResult(text, prompt, estimate_tokens(text), estimated=True)
        return ChatResult(text, usage.prompt_tokens, usage.completion_tokens)


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
    """Scout's LLM client: plain completions, validated JSON, backoff and token accounting."""

    def __init__(
        self,
        settings: Settings | None = None,
        backend: ChatBackend | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.settings = settings or load_settings()
        self._backend = backend
        self._sleep = sleep
        self.usage = UsageTracker()

    @property
    def backend(self) -> ChatBackend:
        """The transport, created lazily so constructing an LLM never needs a key."""
        if self._backend is None:
            self._backend = OpenAIBackend(self.settings)
        return self._backend

    def _model(self, model: str | None, fast: bool) -> str:
        """Resolve which model name to use for a call."""
        chosen = model or (self.settings.fast_model if fast else self.settings.scout_model)
        if not chosen:
            raise LLMError("SCOUT_MODEL is not set (see .env.example)")
        return chosen

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
        temp = self.settings.llm_temperature if temperature is None else temperature
        attempts = 1 + max(0, self.settings.llm_max_retries)
        for attempt in range(attempts):
            started = time.monotonic()
            try:
                result = self.backend.chat(
                    model=name, messages=messages, json_mode=json_mode, temperature=temp
                )
            except TransientLLMError as exc:
                if attempt == attempts - 1:
                    raise LLMError(f"LLM call failed after {attempts} attempts: {exc}") from exc
                delay = self._backoff(attempt, exc.retry_after)
                logger.warning("Transient LLM error (%s); retrying in %.1fs", exc, delay)
                self._sleep(delay)
                continue
            self.usage.calls.append(
                CallRecord(
                    model=name,
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    latency_s=time.monotonic() - started,
                    estimated=result.estimated,
                )
            )
            return result.text
        raise AssertionError("unreachable")  # pragma: no cover

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
    ) -> ModelT:
        """Return a validated ``schema`` instance; one repair request on failure, then raise."""
        msgs = _with_schema_instruction(messages, schema)
        json_mode = self.settings.llm_json_mode
        raw = self.complete(msgs, fast=fast, model=model, json_mode=json_mode)
        try:
            return schema.model_validate_json(extract_json_text(raw))
        except ValidationError as exc:
            errors = format_validation_error(exc)
            logger.info("complete_json: %s failed validation, sending repair", schema.__name__)
        repair = [
            *msgs,
            {"role": "assistant", "content": raw},
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
