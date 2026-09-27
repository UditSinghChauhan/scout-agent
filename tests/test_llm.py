from __future__ import annotations

import openai
import pytest
from pydantic import BaseModel

from scout.config import Settings
from scout.llm import (
    LLM,
    LLMError,
    LLMValidationError,
    TransientLLMError,
    extract_json_text,
    is_transient,
)
from tests.fakes import FakeLLM


class Company(BaseModel):
    name: str
    founded: int


def _sdk_error(cls: type[openai.OpenAIError], **attrs: object) -> Exception:
    """Instantiate an openai SDK exception without its HTTP-client-specific constructor."""
    err = cls.__new__(cls)
    for name, value in attrs.items():
        setattr(err, name, value)
    return err


def _status_error(cls: type[openai.APIStatusError], code: int) -> Exception:
    return _sdk_error(cls, status_code=code)


def test_complete_json_valid_first_try() -> None:
    fake = FakeLLM([{"name": "Zoho", "founded": 1996}])
    result = fake.llm().complete_json([{"role": "user", "content": "hi"}], Company)
    assert result == Company(name="Zoho", founded=1996)
    assert fake.calls == 1
    assert fake.requests[0].json_mode is True


def test_schema_instruction_added_to_system_message() -> None:
    fake = FakeLLM([Company(name="A", founded=1)])
    msgs = [{"role": "system", "content": "You are Scout."}, {"role": "user", "content": "x"}]
    fake.llm().complete_json(msgs, Company)
    system = fake.requests[0].messages[0]["content"]
    assert system.startswith("You are Scout.")
    assert '"founded"' in system
    assert msgs[0]["content"] == "You are Scout."  # caller's list untouched


def test_complete_json_repairs_malformed_output() -> None:
    fake = FakeLLM(["this is not json", {"name": "Zoho", "founded": 1996}])
    result = fake.llm().complete_json([{"role": "user", "content": "hi"}], Company)
    assert result.name == "Zoho"
    assert fake.calls == 2
    repair_msgs = fake.requests[1].messages
    assert repair_msgs[-2] == {"role": "assistant", "content": "this is not json"}
    assert "did not validate" in repair_msgs[-1]["content"]


def test_repair_message_contains_validation_error() -> None:
    fake = FakeLLM([{"name": "Zoho", "founded": "long ago"}, {"name": "Zoho", "founded": 1996}])
    fake.llm().complete_json([{"role": "user", "content": "hi"}], Company)
    assert "founded" in fake.requests[1].messages[-1]["content"]


def test_complete_json_raises_after_one_failed_repair() -> None:
    fake = FakeLLM(["{bad", '{"name": "Zoho"}', {"name": "never", "founded": 1}])
    with pytest.raises(LLMValidationError) as info:
        fake.llm().complete_json([{"role": "user", "content": "hi"}], Company)
    assert fake.calls == 2  # exactly one repair request
    assert info.value.raw == '{"name": "Zoho"}'


def test_extract_json_strips_fences_and_prose() -> None:
    assert extract_json_text('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert extract_json_text('Sure! {"a": 1} hope that helps') == '{"a": 1}'


def test_complete_json_accepts_fenced_json() -> None:
    fake = FakeLLM(['```json\n{"name": "Zoho", "founded": 1996}\n```'])
    assert fake.llm().complete_json([{"role": "user", "content": "x"}], Company).founded == 1996
    assert fake.calls == 1


def test_transient_errors_retry_with_exponential_backoff() -> None:
    fake = FakeLLM([TransientLLMError("429"), TransientLLMError("503"), "ok"])
    llm = fake.llm(Settings(scout_model="m", llm_backoff_base_s=1.0, llm_max_retries=4))
    assert llm.complete([{"role": "user", "content": "x"}]) == "ok"
    assert fake.sleeps == [1.0, 2.0]
    assert llm.usage.llm_calls == 1


def test_retry_after_is_honoured_and_capped() -> None:
    fake = FakeLLM([TransientLLMError("429", retry_after=7.0), TransientLLMError("429", 999), "ok"])
    llm = fake.llm(Settings(scout_model="m", llm_backoff_base_s=1.0, llm_backoff_max_s=30.0))
    llm.complete([{"role": "user", "content": "x"}])
    assert fake.sleeps == [7.0, 30.0]


def test_retries_exhausted_raise() -> None:
    fake = FakeLLM([TransientLLMError("500")] * 3)
    llm = fake.llm(Settings(scout_model="m", llm_max_retries=2))
    with pytest.raises(LLMError, match="after 3 attempts"):
        llm.complete([{"role": "user", "content": "x"}])
    assert fake.calls == 3


def test_non_transient_error_is_not_retried() -> None:
    fake = FakeLLM([LLMError("401 bad key"), "never"])
    with pytest.raises(LLMError):
        fake.llm().complete([{"role": "user", "content": "x"}])
    assert fake.calls == 1 and fake.sleeps == []


@pytest.mark.parametrize(
    ("cls", "code", "expected"),
    [
        (openai.RateLimitError, 429, True),
        (openai.InternalServerError, 500, True),
        (openai.InternalServerError, 503, True),
        (openai.BadRequestError, 400, False),
        (openai.AuthenticationError, 401, False),
        (openai.NotFoundError, 404, False),
    ],
)
def test_is_transient_for_sdk_errors(
    cls: type[openai.APIStatusError], code: int, expected: bool
) -> None:
    assert is_transient(_status_error(cls, code)) is expected


def test_connection_errors_are_transient() -> None:
    assert is_transient(_sdk_error(openai.APITimeoutError))
    assert is_transient(_sdk_error(openai.APIConnectionError))


def test_token_counting_accumulates() -> None:
    fake = FakeLLM(["first reply", "second reply"])
    llm = fake.llm()
    llm.complete([{"role": "user", "content": "a" * 400}])
    llm.complete([{"role": "user", "content": "b" * 40}], fast=True)
    assert llm.usage.llm_calls == 2
    assert llm.usage.prompt_tokens == 100 + 10
    assert llm.usage.total_tokens == llm.usage.prompt_tokens + llm.usage.completion_tokens
    assert [r.model for r in fake.requests] == ["fake-model", "fake-fast"]


def test_missing_model_raises() -> None:
    llm = LLM(settings=Settings(), backend=FakeLLM(["x"]))
    with pytest.raises(LLMError, match="SCOUT_MODEL"):
        llm.complete([{"role": "user", "content": "x"}])


def test_missing_api_key_raises_without_leaking() -> None:
    llm = LLM(settings=Settings(scout_model="m"))
    with pytest.raises(LLMError, match="LLM_API_KEY"):
        llm.complete([{"role": "user", "content": "x"}])


def test_daily_quota_429_is_not_retried() -> None:
    err = _sdk_error(openai.RateLimitError, status_code=429)
    err.args = ("Quota exceeded: GenerateRequestsPerDayPerProjectPerModel-FreeTier",)
    assert is_transient(err) is False
    minute = _sdk_error(openai.RateLimitError, status_code=429)
    minute.args = ("Rate limit reached on tokens per minute (TPM)",)
    assert is_transient(minute) is True


def test_rejected_native_tool_call_is_returned_as_text() -> None:
    from scout.llm import rejected_generation

    err = _sdk_error(openai.BadRequestError, status_code=400)
    err.body = {
        "error": {
            "code": "tool_use_failed",
            "failed_generation": '{"name": "web_search", "arguments": {"query": "zoho"}}',
        }
    }
    assert rejected_generation(err) == '{"name": "web_search", "arguments": {"query": "zoho"}}'
    empty = _sdk_error(openai.BadRequestError, status_code=400)
    empty.body = {"error": {"code": "json_validate_failed", "failed_generation": ""}}
    assert rejected_generation(empty) == ""
    other = _sdk_error(openai.BadRequestError, status_code=400)
    other.body = {"error": {"code": "invalid_request_error"}}
    assert rejected_generation(other) is None
