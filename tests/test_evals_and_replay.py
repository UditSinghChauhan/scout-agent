from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scout.agent.orchestrator import Orchestrator
from scout.config import Settings
from scout.eval import aggregates, collect, load_tasks, render, write_results
from scout.events import load_trace
from scout.llm import TransientLLMError
from scout.scoring import normalize_score
from scout.tools.registry import build_registry
from scout.ui.runs import list_runs
from scout.ui.viewmodel import build_view, wait_line
from tests.fakes import FakeLLM, FakeSearch

FIXTURES = Path(__file__).parent / "fixtures"
DEMO_ID = "20260101-120000-demo"
REPO = Path(__file__).parent.parent


# --- fixed score formats -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("purpose", "raw", "reasons", "expected"),
    [
        ("sales_prospect", "75", [], "Fit 75/100"),
        ("sales_prospect", "Fit score: 82/100", [], "Fit 82/100"),
        ("sales_prospect", "Fit 75/100", [], "Fit 75/100"),
        ("sales_prospect", "high fit", [], None),
        ("sales_prospect", "140", [], None),
        ("sales_prospect", "between 60 and 70", [], None),
        ("competitor", "medium", [], "Threat: Medium"),
        ("competitor", "Threat: HIGH", [], "Threat: High"),
        ("competitor", "moderate threat", [], "Threat: Medium"),
        ("competitor", "unclear", [], None),
        ("interview_prep", "Readiness 4/6", [], "Readiness 4/6"),
        ("interview_prep", "4 of 6", [], "Readiness 4/6"),
        (
            "interview_prep",
            "Pass",
            ["ready: stack", "todo: stipend", "ready: process"],
            "Readiness 2/3",
        ),
        ("interview_prep", "Readiness checklist", ["free text reason"], None),
        ("interview_prep", "7/6", [], None),
        ("general", "90", [], None),
    ],
)
def test_score_format(purpose: str, raw: str, reasons: list[str], expected: str | None) -> None:
    assert normalize_score(purpose, raw, reasons) == expected


def test_synthesis_score_is_validated(tmp_path: Path) -> None:
    settings = Settings(scout_model="m", runs_dir=tmp_path, data_dir=tmp_path, cache_enabled=False)
    fake = FakeLLM(
        [
            {"targets": ["Zoho"], "purpose_type": "sales_prospect"},
            {"steps": [{"id": 1, "question": "q?"}]},
            {"thought": "s", "type": "tool", "tool": "web_search", "args": {"query": "zoho"}},
            {
                "thought": "f",
                "type": "finish",
                "findings": [
                    {"claim": "c", "source_url": "https://example.com/zoho/1", "snippet": "c"}
                ],
            },
            {"verdict": "complete"},
            {
                "title": "t",
                "score": "seventy-ish, maybe 70",
                "sections": [
                    {"title": "Snapshot", "claims": [{"text": "c", "evidence_ids": ["E1"]}]}
                ],
            },
            {"lessons": ["Search the careers page first."]},
        ]
    )
    reg = build_registry(settings, [FakeSearch().provider()], lambda u: "t", cache=None)
    events = list(Orchestrator(settings, llm=fake.llm(settings), registry=reg).run("goal"))
    final = events[-1].payload
    assert final["brief"]["score"] == "Fit 70/100"
    assert "**Score:** Fit 70/100" in final["report_md"]
    assert final["metrics"]["git_commit"]


# --- rate_limited ------------------------------------------------------------------------------


def test_rate_limited_event_and_listener() -> None:
    heard: list[tuple[str, dict]] = []
    fake = FakeLLM([TransientLLMError("429", retry_after=12, model="groq/qwen/qwen3.8-27b"), "ok"])
    llm = fake.llm()
    llm.listener = lambda kind, payload: heard.append((kind, payload))
    assert llm.complete([{"role": "user", "content": "x"}], fast=True) == "ok"
    assert heard == [
        ("rate_limited", {"model": "groq/qwen/qwen3.8-27b", "tier": "fast", "wait_s": 12.0})
    ]
    assert ("rate_limited", heard[0][1]) in llm.drain_notices()
    assert wait_line(heard[0][1]) == "Waiting 12 s: rate limit on qwen3.8-27b"


def test_listener_errors_never_break_a_run() -> None:
    fake = FakeLLM([TransientLLMError("429", retry_after=1), "ok"])
    llm = fake.llm()

    def boom(kind: str, payload: dict) -> None:
        raise RuntimeError("ui gone")

    llm.listener = boom
    assert llm.complete([{"role": "user", "content": "x"}]) == "ok"


def test_rate_limited_shows_in_trace() -> None:
    view = build_view(load_trace(FIXTURES / "runs" / DEMO_ID / "trace.jsonl"))
    waits = [e for s in view.steps for e in s.entries if e.kind == "wait"]
    waits += [e for e in view.run_entries if e.kind == "wait"]
    assert [e.text for e in waits] == ["Waiting 2 s: rate limit on qwen3.8-27b"]


# --- examples and keyless UI -------------------------------------------------------------------


def test_examples_appear_in_picker(tmp_path: Path) -> None:
    runs, examples = tmp_path / "runs", tmp_path / "examples"
    shutil.copytree(FIXTURES / "runs" / DEMO_ID, examples / DEMO_ID)
    shutil.copytree(FIXTURES / "runs" / DEMO_ID, runs / DEMO_ID)  # same run in both places
    listed = list_runs(runs, examples)
    assert len(listed) == 1 and listed[0].source == "example"
    assert listed[0].label.startswith("example · ")
    assert list_runs(runs, None)[0].source == "run"


def test_repo_examples_are_replayable() -> None:
    listed = list_runs(REPO / "no-such-runs-dir", REPO / "examples")
    assert len(listed) >= 4 and all(r.source == "example" for r in listed)
    for run in listed:
        assert build_view(load_trace(run.path / "trace.jsonl")).finished


def test_live_mode_hidden_without_keys(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    examples = tmp_path / "examples"
    shutil.copytree(FIXTURES / "runs" / DEMO_ID, examples / DEMO_ID)
    monkeypatch.setenv("SCOUT_ENV_FILE", "none")
    for key in ("LLM_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY"):
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("SCOUT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SCOUT_EXAMPLES_DIR", str(examples))
    monkeypatch.setenv("SCOUT_DATA_DIR", str(tmp_path / "data"))
    at = AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert list(at.sidebar.radio(key="mode").options) == ["Replay saved run"]
    assert any("Live mode needs API keys" in i.value for i in at.sidebar.info)
    assert any(i.value.startswith(f"Replay of recorded run {DEMO_ID}") for i in at.info)


# --- eval harness ------------------------------------------------------------------------------


def test_eval_harness_on_fixture_runs(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    shutil.copytree(FIXTURES / "runs" / DEMO_ID, runs / DEMO_ID)
    (runs / "old").mkdir()
    shutil.copy(FIXTURES / "old_trace.jsonl", runs / "old" / "trace.jsonl")  # no verifier
    rows = collect([runs])
    assert [r["run"] for r in rows] == [DEMO_ID]
    row = rows[0]
    assert row["purpose"] == "interview_prep" and row["retries"] == 1 and row["replans"] == 1
    assert row["memory steps"] == 1 and row["coverage %"] == 100.0 and row["budget stop"] == "no"
    agg = aggregates(rows)
    assert agg[-1]["purpose"] == "all" and agg[-1]["mean coverage %"] == 100.0
    out = tmp_path / "evals" / "results.md"
    write_results([runs], out)
    text = out.read_text()
    assert "only runs whose trace contains the verifier" in text and DEMO_ID in text
    assert render([]).startswith("# Scout eval results")


def test_tasks_yaml_loads() -> None:
    tasks = load_tasks(REPO / "evals" / "tasks.yaml")
    assert {t["purpose"] for t in tasks} == {"sales_prospect", "competitor", "interview_prep"}


def test_bad_tasks_file(tmp_path: Path) -> None:
    bad = tmp_path / "t.yaml"
    bad.write_text("tasks:\n  - id: x\n")
    with pytest.raises(ValueError):
        load_tasks(bad)
