from __future__ import annotations

import pytest

from scout.schemas import Claim
from tests.fakes import FakeLLM, FakeSearch


def test_fake_llm_replays_script_in_order() -> None:
    fake = FakeLLM(["one", {"a": 1}, Claim(text="t")])
    llm = fake.llm()
    msgs = [{"role": "user", "content": "x"}]
    assert llm.complete(msgs) == "one"
    assert llm.complete(msgs) == '{"a": 1}'
    assert '"text":"t"' in llm.complete(msgs)
    assert fake.calls == 3


def test_fake_llm_raises_scripted_exception_and_when_exhausted() -> None:
    fake = FakeLLM([ValueError("scripted")])
    with pytest.raises(ValueError, match="scripted"):
        fake.chat(model="m", messages=[], json_mode=False, temperature=0)
    with pytest.raises(AssertionError, match="exhausted"):
        fake.chat(model="m", messages=[], json_mode=False, temperature=0)


def test_fake_search_is_deterministic() -> None:
    search = FakeSearch()
    first = search("Zoho careers", 3)
    assert first == search("Zoho careers", 3)
    assert len(first) == 3
    assert first[0]["url"] == "https://example.com/zoho-careers/1"
    assert search.queries == ["Zoho careers", "Zoho careers"]


def test_fake_search_scripted_and_error() -> None:
    scripted = [{"title": "T", "url": "https://z.com", "snippet": "s"}]
    assert FakeSearch(results_by_query={"q": scripted})("q", 5) == scripted
    with pytest.raises(RuntimeError):
        FakeSearch(error=RuntimeError("down"))("q", 5)
