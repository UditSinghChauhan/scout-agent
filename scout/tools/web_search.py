"""Web search tool: Tavily first, DuckDuckGo (``ddgs``) fallback.

``web_search(query, max_results=5)`` returns ``[{"title", "url", "snippet"}]``. Providers are
plain callables ``(query, max_results) -> results`` so tests inject ``FakeSearch`` and never touch
the network. Source-score re-ranking is added in Phase 3.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from scout.config import Settings, load_settings

logger = logging.getLogger(__name__)

SearchResult = dict[str, str]
SearchProvider = Callable[[str, int], list[SearchResult]]


class SearchError(Exception):
    """Every configured search provider failed."""


def normalize_results(
    raw: Iterable[Mapping[str, Any]],
    *,
    url_key: str,
    snippet_key: str,
    max_results: int,
    snippet_chars: int = 400,
) -> list[SearchResult]:
    """Map provider-specific result dicts to ``{title, url, snippet}``, deduplicated by URL."""
    results: list[SearchResult] = []
    seen: set[str] = set()
    for item in raw:
        url = str(item.get(url_key) or "").strip()
        if not url.startswith(("http://", "https://")) or url in seen:
            continue
        seen.add(url)
        snippet = " ".join(str(item.get(snippet_key) or "").split())
        results.append(
            {
                "title": str(item.get("title") or "").strip(),
                "url": url,
                "snippet": snippet[:snippet_chars],
            }
        )
        if len(results) >= max_results:
            break
    return results


def tavily_provider(settings: Settings) -> SearchProvider:
    """Build a provider backed by ``tavily.TavilyClient.search``."""

    def search(query: str, max_results: int) -> list[SearchResult]:
        from tavily import TavilyClient

        client = TavilyClient(api_key=settings.tavily_api_key)
        response = client.search(
            query, max_results=max_results, timeout=settings.search_timeout_s
        )
        return normalize_results(
            response.get("results", []),
            url_key="url",
            snippet_key="content",
            max_results=max_results,
            snippet_chars=settings.search_snippet_chars,
        )

    return search


def ddgs_provider(settings: Settings) -> SearchProvider:
    """Build a provider backed by ``ddgs.DDGS().text`` (no API key needed)."""

    def search(query: str, max_results: int) -> list[SearchResult]:
        from ddgs import DDGS

        raw = DDGS(timeout=int(settings.search_timeout_s)).text(query, max_results=max_results)
        return normalize_results(
            raw,
            url_key="href",
            snippet_key="body",
            max_results=max_results,
            snippet_chars=settings.search_snippet_chars,
        )

    return search


def default_providers(settings: Settings) -> list[tuple[str, SearchProvider]]:
    """Providers in priority order: Tavily (only when keyed), then DuckDuckGo."""
    providers: list[tuple[str, SearchProvider]] = []
    if settings.has_tavily:
        providers.append(("tavily", tavily_provider(settings)))
    providers.append(("ddgs", ddgs_provider(settings)))
    return providers


def web_search(
    query: str,
    max_results: int = 5,
    *,
    settings: Settings | None = None,
    providers: list[tuple[str, SearchProvider]] | None = None,
) -> list[SearchResult]:
    """Search the web, falling through providers on error or empty results.

    Raises :class:`SearchError` only when every provider raised; returns ``[]`` when providers
    ran cleanly but found nothing.
    """
    if not query.strip():
        raise ValueError("query must not be empty")
    settings = settings or load_settings()
    chain = providers if providers is not None else default_providers(settings)
    failures: list[str] = []
    for name, provider in chain:
        try:
            results = provider(query, max_results)
        except Exception as exc:  # noqa: BLE001 - any provider failure triggers fallback
            logger.warning("Search provider %s failed: %s", name, type(exc).__name__)
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            continue
        if results:
            return results[:max_results]
        logger.info("Search provider %s returned no results; trying next", name)
    if failures and len(failures) == len(chain):
        raise SearchError("All search providers failed: " + "; ".join(failures))
    return []
