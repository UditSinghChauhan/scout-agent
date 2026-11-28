from __future__ import annotations

from pathlib import Path

import pytest

from scout.agent.orchestrator import Orchestrator, OrchestratorStateError, RunState
from scout.config import Settings
from scout.tools.registry import build_registry
from tests.fakes import FakeLLM, FakeSearch


def test_stage_without_its_state_raises_typed_error(tmp_path: Path) -> None:
    settings = Settings(
        scout_model="fake-model", runs_dir=tmp_path, data_dir=tmp_path, cache_enabled=False
    )
    registry = build_registry(settings, [FakeSearch().provider()], lambda url: "", cache=None)
    orch = Orchestrator(settings, llm=FakeLLM().llm(settings), registry=registry)
    with pytest.raises(OrchestratorStateError, match="recall needs a task"):
        list(orch._recall(RunState(run_id="r", goal="g")))
    with pytest.raises(OrchestratorStateError, match="verify needs a brief"):
        list(orch._verify(RunState(run_id="r", goal="g")))
