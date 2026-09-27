"""Tool registry: name, description and JSON argument schema per tool (docs/SPEC.md §4).

``ToolRegistry.run`` never raises: unknown tools, bad arguments and tool failures all become an
``Observation`` with ``ok=False`` so the ReAct loop can recover. Web text is wrapped in
``<untrusted_content>`` here, the single choke point between the internet and the prompts.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, NamedTuple

from scout.config import Settings
from scout.memory.store import MemoryStore, normalize_entity
from scout.safety import is_personal_profile
from scout.schemas import Observation
from scout.tools.cache import DiskCache
from scout.tools.fetch_page import fetch_text, focus_text
from scout.tools.web_search import SearchProvider, SearchResult, rerank_by_source, web_search

logger = logging.getLogger(__name__)

ToolFn = Callable[..., "ToolOutput"]
REPEAT_NOTE = "(already done; try a different query or source)"


@dataclass(frozen=True)
class ToolOutput:
    """Tool result: prompt-ready text plus the source URLs it exposed."""

    text: str
    urls: tuple[str, ...] = ()


@dataclass(frozen=True)
class Tool:
    """A callable tool with its JSON argument schema."""

    name: str
    description: str
    parameters: dict[str, Any]
    fn: ToolFn

    def signature(self, with_thought: bool = False) -> str:
        """One-line description for prompts; optionally lists the executor's ``thought`` arg."""
        props = self.parameters.get("properties", {})
        required = set(self.parameters.get("required", []))
        parts = ["thought: string"] if with_thought else []
        parts += [
            f"{k}: {v.get('type', 'any')}{'' if k in required else ' (optional)'}"
            for k, v in props.items()
        ]
        return f"- {self.name}({', '.join(parts)}): {self.description}"


def wrap_untrusted(text: str, source: str) -> str:
    """Wrap web text in <untrusted_content>, neutralising any embedded closing tag."""
    safe = text.replace("<untrusted_content", "&lt;untrusted_content").replace(
        "</untrusted_content", "&lt;/untrusted_content"
    )
    return f'<untrusted_content source="{source}">\n{safe}\n</untrusted_content>'


class ToolRun(NamedTuple):
    """Outcome of ``ToolRegistry.run``."""

    observation: Observation
    urls: tuple[str, ...]
    repeated: bool = False


def _call_key(name: str, args: dict[str, Any]) -> str:
    """Normalised identity of a tool call, for in-run dedupe."""
    norm = {k: " ".join(v.lower().split()) if isinstance(v, str) else v for k, v in args.items()}
    return f"{name}:{json.dumps(norm, sort_keys=True, default=str)}"


class ToolRegistry:
    """Named tools the executor may call.

    Within one run (one registry instance), repeating an identical call returns the earlier
    observation with a note instead of running the tool again (Phase 2, A3).
    """

    def __init__(self, tools: list[Tool], stats: dict[str, int] | None = None) -> None:
        self.tools = {t.name: t for t in tools}
        self._done: dict[str, ToolRun] = {}
        self.stats = stats if stats is not None else {"cache_hits": 0}

    def describe(self, with_thought: bool = False) -> str:
        """All tool signatures, one per line."""
        return "\n".join(t.signature(with_thought) for t in self.tools.values())

    def run(self, name: str, args: dict[str, Any]) -> ToolRun:
        """Execute a tool; return the observation, its source URLs and whether it repeated."""
        tool = self.tools.get(name)
        if tool is None:
            known = ", ".join(self.tools)
            return ToolRun(self._fail(name, args, f"Unknown tool {name!r}. Available: {known}"), ())
        props = tool.parameters.get("properties", {})
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        if missing:
            return ToolRun(self._fail(name, args, f"Missing required argument(s): {missing}"), ())
        clean = {k: v for k, v in args.items() if k in props}
        key = _call_key(name, clean)
        if key in self._done:
            earlier = self._done[key]
            obs = earlier.observation.model_copy(
                update={"args": args, "content": f"{REPEAT_NOTE}\n{earlier.observation.content}"}
            )
            return ToolRun(obs, earlier.urls, repeated=True)
        try:
            output = tool.fn(**clean)
            result = ToolRun(
                Observation(tool=name, args=args, ok=True, content=output.text), output.urls
            )
        except Exception as exc:  # noqa: BLE001 - every tool failure becomes an observation
            logger.info("Tool %s failed: %s", name, exc)
            result = ToolRun(self._fail(name, args, f"{type(exc).__name__}: {exc}"), ())
        self._done[key] = result
        return result

    @staticmethod
    def _fail(name: str, args: dict[str, Any], error: str) -> Observation:
        """Build a failed observation."""
        return Observation(tool=name, args=args, ok=False, error=error, content=f"ERROR: {error}")


def format_search_results(query: str, results: list[SearchResult]) -> str:
    """Numbered search results for the prompt, wrapped as untrusted content."""
    if not results:
        return f"No results for {query!r}."
    lines = [
        f"{i}. {r['title']}\n   URL: {r['url']}\n   {r['snippet']}"
        for i, r in enumerate(results, 1)
    ]
    return wrap_untrusted("\n".join(lines), f"web_search: {query}")


def build_registry(
    settings: Settings,
    search_providers: list[tuple[str, SearchProvider]] | None = None,
    fetcher: Callable[[str], str] | None = None,
    cache: DiskCache | None = None,
    store: MemoryStore | None = None,
) -> ToolRegistry:
    """Default registry; tests inject fake search providers, a fake fetcher, cache and store.

    ``cache`` defaults to a 24 h DiskCache under ``data/cache`` when caching is enabled. With a
    ``store``, search results are re-ranked by source reliability and ``recall_memory`` is added.
    """
    if cache is None and settings.cache_enabled:
        cache = DiskCache(settings.cache_dir, settings.cache_ttl_s)
    stats = {"cache_hits": 0}

    def search_tool(query: str, max_results: int = settings.search_max_results) -> ToolOutput:
        n = max(1, min(int(max_results), settings.search_max_results))
        key = f"{' '.join(query.lower().split())}|{n}"
        results = cache.get("search", key) if cache else None
        if results is None:
            results = web_search(query, n, settings=settings, providers=search_providers)
            if cache and results:
                cache.set("search", key, results)
        else:
            stats["cache_hits"] += 1
        if store is not None:
            results = rerank_by_source(
                results, store.source_scores(), settings.source_rerank_weight
            )
        return ToolOutput(format_search_results(query, results), tuple(r["url"] for r in results))

    def fetch_tool(url: str, focus: str = "") -> ToolOutput:
        if is_personal_profile(url, settings.personal_profile_patterns):
            raise ValueError("personal profile pages are not read (roles only)")
        text = cache.get("fetch", url.strip()) if cache else None
        if text is None:
            text = fetcher(url) if fetcher else fetch_text(url, settings=settings)
            if cache:
                cache.set("fetch", url.strip(), text)
        else:
            stats["cache_hits"] += 1
        cut = focus_text(text, focus, settings.fetch_max_chars)
        return ToolOutput(wrap_untrusted(cut, url), (url,))

    tools = [
        Tool(
            name="web_search",
            description="Search the web; returns titles, URLs and snippets.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer"},
                },
                "required": ["query"],
            },
            fn=search_tool,
        ),
        Tool(
            name="fetch_page",
            description=(
                f"Read a public web page (~{settings.fetch_max_chars} chars of main text, "
                "the parts most relevant to `focus`)."
            ),
            parameters={
                "type": "object",
                "properties": {"url": {"type": "string"}, "focus": {"type": "string"}},
                "required": ["url"],
            },
            fn=fetch_tool,
        ),
    ]
    if store is not None:
        tools.append(_recall_tool(settings, store))
    return ToolRegistry(tools, stats)


def _recall_tool(settings: Settings, store: MemoryStore) -> Tool:
    """``recall_memory(entity, topic)``: fresh stored facts about an entity, with their sources."""

    def recall_memory(entity: str, topic: str = "") -> ToolOutput:
        key = normalize_entity(entity)
        fresh, _ = store.recall(
            key,
            settings.fact_ttl_days,
            settings.news_ttl_days,
            settings.memory_max_facts,
            settings.memory_max_chars,
        )
        facts = store.search_facts(key, topic, fresh)
        if not facts:
            return ToolOutput(f"No fresh facts stored for {entity!r} about {topic!r}.")
        lines = [f"- {f.claim}\n  URL: {f.source_url}\n  {f.snippet[:200]}" for f in facts]
        return ToolOutput(
            wrap_untrusted("\n".join(lines), f"memory: {key}"),
            tuple(f.source_url for f in facts),
        )

    return Tool(
        name="recall_memory",
        description="Facts Scout already verified in earlier runs (fresh only), with sources.",
        parameters={
            "type": "object",
            "properties": {"entity": {"type": "string"}, "topic": {"type": "string"}},
            "required": ["entity"],
        },
        fn=recall_memory,
    )
