from __future__ import annotations

from pathlib import Path

import pytest

from scout.config import Settings, env_name, load_settings, read_env_file


def test_budget_defaults() -> None:
    # SPEC §4 budgets, with the Phase 2 token-diet overrides (5 steps, 3 iterations, 480 s).
    s = load_settings(env={}, env_file=None)
    assert s.max_planned_steps == 5
    assert s.max_total_steps == 8
    assert s.max_react_iterations == 3
    assert s.max_tool_calls == 30
    assert s.max_llm_calls == 60
    assert s.max_wall_clock_s == 480.0
    assert s.fetch_timeout_s == 15.0
    assert s.fetch_max_chars == 2_000 and s.search_snippet_chars == 200
    assert s.max_followups == 1
    assert s.fact_ttl_days == 7 and s.news_ttl_days == 2


def test_env_overrides_and_types() -> None:
    env = {
        "LLM_BASE_URL": "https://llm.example/v1",
        "SCOUT_MODEL": "big",
        "SCOUT_MAX_TOOL_CALLS": "5",
        "SCOUT_FETCH_TIMEOUT_S": "3.5",
        "SCOUT_LLM_JSON_MODE": "false",
    }
    s = load_settings(env=env, env_file=None)
    assert s.llm_base_url == "https://llm.example/v1"
    assert s.max_tool_calls == 5
    assert s.fetch_timeout_s == 3.5
    assert s.llm_json_mode is False


def test_fast_model_falls_back_to_main() -> None:
    assert Settings(scout_model="big").fast_model == "big"
    assert Settings(scout_model="big", scout_fast_model="small").fast_model == "small"


def test_env_file_is_read_and_real_env_wins(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("# comment\nSCOUT_MODEL='from-file'\nexport SCOUT_FAST_MODEL=fast\n")
    assert read_env_file(env_file)["SCOUT_MODEL"] == "from-file"
    s = load_settings(env={"SCOUT_MODEL": "from-env"}, env_file=env_file)
    assert s.scout_model == "from-env"
    assert s.scout_fast_model == "fast"


def test_missing_env_file_is_fine(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / "nope.env") == {}


def test_invalid_number_names_the_variable() -> None:
    with pytest.raises(ValueError, match="SCOUT_MAX_TOOL_CALLS"):
        load_settings(env={"SCOUT_MAX_TOOL_CALLS": "many"}, env_file=None)


def test_repr_hides_keys() -> None:
    s = Settings(llm_api_key="secret-value-123", tavily_api_key="another-secret")
    assert "secret-value-123" not in repr(s)
    assert "another-secret" not in repr(s)


def test_with_overrides_returns_copy() -> None:
    s = Settings()
    t = s.with_overrides(max_tool_calls=5)
    assert t.max_tool_calls == 5 and s.max_tool_calls == 30


def test_env_names() -> None:
    assert env_name("llm_api_key") == "LLM_API_KEY"
    assert env_name("max_llm_calls") == "SCOUT_MAX_LLM_CALLS"
