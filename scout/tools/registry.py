"""Tool registry: name, description and JSON argument schema per tool (docs/SPEC.md §4).

``ToolRegistry.run`` never raises: unknown tools, bad arguments and tool failures all become an
``Observation`` with ``ok=False`` so the ReAct loop can recover. Web text is wrapped in
``<untrusted_content>`` here, the single choke point between the internet and the prompts.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from scout.config import Settings
from scout.schemas import Observation
from scout.tools.fetch_page import fetch_page
from scout.tools.web_search import SearchProvider, SearchResult, web_search

logger = logging.getLogger(__name__)

ToolFn = Callable[..., "ToolOutput"]


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


class ToolRegistry:
    """Named tools the executor may call."""

    def __init__(self, tools: list[Tool]) -> None:
        self.tools = {t.name: t for t in tools}

    def describe(self, with_thought: bool = False) -> str:
        """All tool signatures, one per line."""
        return "\n".join(t.signature(with_thought) for t in self.tools.values())

    def run(self, name: str, args: dict[str, Any]) -> tuple[Observation, tuple[str, ...]]:
        """Execute a tool; return the observation and the source URLs it exposed."""
        tool = self.tools.get(name)
        if tool is None:
            known = ", ".join(self.tools)
            return self._fail(name, args, f"Unknown tool {name!r}. Available: {known}"), ()
        props = tool.parameters.get("properties", {})
        missing = [k for k in tool.parameters.get("required", []) if k not in args]
        if missing:
            return self._fail(name, args, f"Missing required argument(s): {missing}"), ()
        clean = {k: v for k, v in args.items() if k in props}
        try:
            output = tool.fn(**clean)
        except Exception as exc:  # noqa: BLE001 - every tool failure becomes an observation
            logger.info("Tool %s failed: %s", name, exc)
            return self._fail(name, args, f"{type(exc).__name__}: {exc}"), ()
        return Observation(tool=name, args=args, ok=True, content=output.text), output.urls

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
    fetcher: Callable[[str, str], str] | None = None,
) -> ToolRegistry:
    """Default registry; tests inject fake search providers and a fake fetcher."""

    def search_tool(query: str, max_results: int = settings.search_max_results) -> ToolOutput:
        n = max(1, min(int(max_results), settings.search_max_results))
        results = web_search(query, n, settings=settings, providers=search_providers)
        return ToolOutput(format_search_results(query, results), tuple(r["url"] for r in results))

    def fetch_tool(url: str, focus: str = "") -> ToolOutput:
        text = fetcher(url, focus) if fetcher else fetch_page(url, focus, settings=settings)
        return ToolOutput(wrap_untrusted(text, url), (url,))

    return ToolRegistry(
        [
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
                    "Fetch a public web page and return its main text (~4000 chars), keeping "
                    "the parts most relevant to `focus`."
                ),
                parameters={
                    "type": "object",
                    "properties": {"url": {"type": "string"}, "focus": {"type": "string"}},
                    "required": ["url"],
                },
                fn=fetch_tool,
            ),
        ]
    )
