from __future__ import annotations

import pytest

from scout.config import Settings
from scout.tools.web_search import SearchError, default_providers, normalize_results, web_search
from tests.fakes import FakeSearch

SETTINGS = Settings()


def test_first_provider_wins() -> None:
    primary, backup = FakeSearch(), FakeSearch()
    results = web_search("zoho", 3, settings=SETTINGS,
                         providers=[primary.provider("tavily"), backup.provider("ddgs")])
    assert len(results) == 3
    assert set(results[0]) == {"title", "url", "snippet"}
    assert backup.queries == []


def test_falls_back_when_primary_errors() -> None:
    tavily = FakeSearch(error=RuntimeError("429 rate limited"))
    ddgs = FakeSearch()
    results = web_search("zoho", 5, settings=SETTINGS,
                         providers=[tavily.provider("tavily"), ddgs.provider("ddgs")])
    assert len(results) == 5
    assert tavily.queries == ["zoho"] and ddgs.queries == ["zoho"]


def test_falls_back_when_primary_is_empty() -> None:
    empty = FakeSearch(results_by_query={"zoho": []})
    results = web_search("zoho", 2, settings=SETTINGS,
                         providers=[empty.provider("tavily"), FakeSearch().provider("ddgs")])
    assert len(results) == 2


def test_all_providers_failing_raises() -> None:
    with pytest.raises(SearchError, match="tavily.*ddgs"):
        web_search("zoho", settings=SETTINGS, providers=[
            FakeSearch(error=RuntimeError("a")).provider("tavily"),
            FakeSearch(error=RuntimeError("b")).provider("ddgs"),
        ])


def test_no_results_without_errors_returns_empty() -> None:
    empty = FakeSearch(results_by_query={"zzz": []})
    assert web_search("zzz", settings=SETTINGS, providers=[empty.provider()]) == []


def test_empty_query_rejected() -> None:
    with pytest.raises(ValueError):
        web_search("   ", settings=SETTINGS, providers=[FakeSearch().provider()])


def test_default_providers_skip_tavily_without_key() -> None:
    assert [n for n, _ in default_providers(Settings())] == ["ddgs"]
    assert [n for n, _ in default_providers(Settings(tavily_api_key="k"))] == ["tavily", "ddgs"]


def test_normalize_tavily_shape() -> None:
    raw = [
        {"title": "Zoho", "url": "https://zoho.com", "content": "  Zoho   makes\nsoftware "},
        {"title": "dup", "url": "https://zoho.com", "content": "x"},
        {"title": "bad", "url": "ftp://zoho.com", "content": "x"},
        {"title": "Wiki", "url": "https://en.wikipedia.org/wiki/Zoho", "content": "y" * 900},
    ]
    out = normalize_results(raw, url_key="url", snippet_key="content", max_results=5)
    assert [r["url"] for r in out] == ["https://zoho.com", "https://en.wikipedia.org/wiki/Zoho"]
    assert out[0]["snippet"] == "Zoho makes software"
    assert len(out[1]["snippet"]) == 400


def test_normalize_ddgs_shape_and_limit() -> None:
    raw = [{"title": f"t{i}", "href": f"https://e.com/{i}", "body": "b"} for i in range(10)]
    out = normalize_results(raw, url_key="href", snippet_key="body", max_results=3)
    assert out == [
        {"title": "t0", "url": "https://e.com/0", "snippet": "b"},
        {"title": "t1", "url": "https://e.com/1", "snippet": "b"},
        {"title": "t2", "url": "https://e.com/2", "snippet": "b"},
    ]
