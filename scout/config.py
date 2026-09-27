"""Central settings and budgets for Scout.

Every tunable value lives here (docs/SPEC.md §4 "Budgets", §6 "Settings and budgets come only
from scout/config.py"). Values come from environment variables, optionally seeded from a local
``.env`` file, and any field can be overridden per run with :meth:`Settings.with_overrides`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = REPO_ROOT / ".env"


@dataclass(frozen=True)
class Settings:
    """Immutable Scout configuration. API keys are excluded from ``repr``."""

    # LLM access (any OpenAI-compatible endpoint).
    llm_base_url: str = ""
    llm_api_key: str = field(default="", repr=False)
    scout_model: str = ""
    scout_fast_model: str = ""
    llm_temperature: float = 0.2
    llm_timeout_s: float = 60.0
    llm_max_retries: int = 4
    llm_backoff_base_s: float = 1.0
    llm_backoff_max_s: float = 30.0
    llm_json_mode: bool = True

    # Search.
    tavily_api_key: str = field(default="", repr=False)
    search_max_results: int = 5
    search_timeout_s: float = 10.0
    search_snippet_chars: int = 400

    # Fetching.
    fetch_timeout_s: float = 15.0
    fetch_max_bytes: int = 2_000_000
    fetch_max_chars: int = 4_000
    fetch_max_redirects: int = 5
    fetch_user_agent: str = "ScoutResearchBot/0.1 (+https://github.com/scout-agent)"

    # Budgets (docs/SPEC.md §4).
    max_planned_steps: int = 6
    max_total_steps: int = 8
    max_followups: int = 2
    max_react_iterations: int = 4
    max_tool_calls: int = 30
    max_llm_calls: int = 60
    max_wall_clock_s: float = 240.0

    # Memory freshness (docs/SPEC.md §3), used from Phase 3.
    fact_ttl_days: int = 7
    news_ttl_days: int = 2

    # Paths.
    data_dir: Path = REPO_ROOT / "data"
    runs_dir: Path = REPO_ROOT / "runs"

    @property
    def fast_model(self) -> str:
        """Model for the executor and critic; falls back to the main model."""
        return self.scout_fast_model or self.scout_model

    @property
    def has_tavily(self) -> bool:
        """Whether a Tavily key is configured."""
        return bool(self.tavily_api_key)

    def with_overrides(self, **overrides: Any) -> Settings:
        """Return a copy with the given fields replaced (e.g. per-run budget overrides)."""
        return replace(self, **overrides)


# Environment variable name for each settings field. Fields not listed use SCOUT_<FIELD>.
_ENV_NAMES: dict[str, str] = {
    "llm_base_url": "LLM_BASE_URL",
    "llm_api_key": "LLM_API_KEY",
    "scout_model": "SCOUT_MODEL",
    "scout_fast_model": "SCOUT_FAST_MODEL",
    "tavily_api_key": "TAVILY_API_KEY",
}


def env_name(field_name: str) -> str:
    """Return the environment variable that sets ``field_name``."""
    return _ENV_NAMES.get(field_name, f"SCOUT_{field_name.upper()}")


def read_env_file(path: Path) -> dict[str, str]:
    """Parse a simple ``KEY=VALUE`` .env file; missing file yields an empty dict."""
    if not path.is_file():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        values[key.strip()] = value.strip().strip("'\"")
    return values


def _coerce(name: str, raw: str, default: Any) -> Any:
    """Convert a raw env string to the type of the field default."""
    try:
        if isinstance(default, bool):
            return raw.strip().lower() in {"1", "true", "yes", "on"}
        if isinstance(default, int):
            return int(raw)
        if isinstance(default, float):
            return float(raw)
        if isinstance(default, Path):
            return Path(raw).expanduser()
    except ValueError as exc:
        expected = type(default).__name__
        raise ValueError(f"Invalid value for {env_name(name)}: expected {expected}") from exc
    return raw.strip()


def load_settings(
    env: Mapping[str, str] | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
) -> Settings:
    """Build settings from ``env`` (default ``os.environ``) layered over ``env_file``.

    Real environment variables win over the .env file. Empty values keep the default.
    Pass ``env_file=None`` to ignore any .env file (tests do this).
    """
    merged: dict[str, str] = read_env_file(env_file) if env_file else {}
    merged.update(os.environ if env is None else env)
    defaults = Settings()
    kwargs: dict[str, Any] = {}
    for f in fields(Settings):
        raw = merged.get(env_name(f.name), "")
        if raw.strip():
            kwargs[f.name] = _coerce(f.name, raw, getattr(defaults, f.name))
    return Settings(**kwargs)
