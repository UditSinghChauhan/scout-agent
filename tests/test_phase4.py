from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from scout.agent.reflector import reflect
from scout.events import Event, load_trace
from scout.memory.store import MemoryStore
from scout.safety import is_meta_unknown, violates_contact_policy
from scout.schemas import Evidence
from scout.ui.runs import list_runs
from scout.ui.viewmodel import RunView, build_view, reduce
from tests.fakes import FakeLLM

FIXTURES = Path(__file__).parent / "fixtures"
DEMO_ID = "20260101-120000-demo"
NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)


def demo_events() -> list[Event]:
    return load_trace(FIXTURES / "runs" / DEMO_ID / "trace.jsonl")


def ev(type_: str, payload: dict, stage: str = "execute") -> Event:
    return Event(run_id="r", stage=stage, type=type_, payload=payload)  # type: ignore[arg-type]


# --- view model --------------------------------------------------------------------------------


def test_reducer_happy_path_on_fixture() -> None:
    view = build_view(demo_events())
    assert view.finished and view.status == "ok"
    assert view.goal == "Prep me for an SDE intern interview at Zoho."
    assert view.purpose == "interview_prep" and view.targets == ["Zoho"]
    assert all(status == "done" for status in view.stages.values())
    assert view.recalled_facts == 1
    assert [lesson["text"] for lesson in view.lessons_injected] == [
        "Read the careers page before generic news searches."
    ]
    assert view.brief and view.score == "Readiness 2/3"
    assert view.metrics["citation_coverage"] == 100.0
    assert view.lessons_learned and view.lessons_learned[0]["text"].startswith("For interview")


def test_reducer_memory_step() -> None:
    view = build_view(demo_events())
    step = view.step(1)
    assert step is not None and step.status == "from memory" and step.fact_ids == [1]
    assert view.memory_hits == 1
    assert step.entries[0].kind == "memory"


def test_reducer_retry_and_replan() -> None:
    view = build_view(demo_events())
    step2 = view.step(2)
    assert step2 is not None and step2.retries == 1 and step2.status == "retried"
    kinds = [e.kind for e in step2.entries]
    assert "retry" in kinds and kinds.count("critique") == 2
    step3 = view.step(3)
    assert step3 is not None and step3.replanned and step3.status == "done"


def test_reducer_retry_refused_ends_unknown() -> None:
    view = RunView()
    reduce(view, ev("plan_created", {"steps": [{"id": 1, "question": "q"}]}, "plan"))
    reduce(view, ev("step_started", {"id": 1, "question": "q"}))
    reduce(view, ev("critique", {"step_id": 1, "verdict": "retry", "reason": "thin"}, "critique"))
    reduce(view, ev("run_finished", {"status": "ok"}, "finish"))
    assert view.step(1).status == "unknown"


def test_reducer_provider_switch_and_error() -> None:
    view = RunView()
    reduce(view, ev("plan_created", {"steps": [{"id": 1, "question": "q"}]}, "plan"))
    reduce(view, ev("step_started", {"id": 1, "question": "q"}))
    reduce(
        view,
        ev(
            "provider_switched", {"tier": "fast", "from": "groq/a", "to": "groq/b", "reason": "TPD"}
        ),
    )
    reduce(view, ev("error", {"step_id": 1, "message": "invalid action JSON"}))
    reduce(view, ev("provider_switched", {"tier": "smart", "from": "x", "to": "y"}, "synthesize"))
    step = view.step(1)
    assert [e.kind for e in step.entries] == ["switch", "error"]
    assert "groq/a → groq/b" in step.entries[0].text
    assert len(view.switches) == 2 and view.run_entries[0].kind == "switch"
    assert view.errors == ["invalid action JSON"]
    assert view.stages["execute"] == "done" and view.stages["synthesize"] == "running"


def test_reducer_tolerates_old_trace() -> None:
    view = build_view(load_trace(FIXTURES / "old_trace.jsonl"))
    assert view.finished and view.goal == "Research Zoho"
    assert view.lessons_injected == [{"id": None, "text": "an old plain-string lesson"}]
    assert [s.status for s in view.steps] == ["done", "not reached"]
    assert view.errors == ["error"]
    assert view.brief is None and view.report_md.startswith("# Zoho")


def test_list_runs_skips_broken_traces(tmp_path: Path) -> None:
    shutil.copytree(FIXTURES / "runs" / DEMO_ID, tmp_path / DEMO_ID)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "trace.jsonl").write_text("{not json\n")
    (tmp_path / "empty").mkdir()
    runs = list_runs(tmp_path)
    assert [r.run_id for r in runs] == [DEMO_ID]
    assert runs[0].purpose == "interview_prep" and "Zoho" in runs[0].label


# --- Streamlit app smoke test ------------------------------------------------------------------


def test_app_replay_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from streamlit.testing.v1 import AppTest

    runs_dir = tmp_path / "runs"
    shutil.copytree(FIXTURES / "runs" / DEMO_ID, runs_dir / DEMO_ID)
    monkeypatch.setenv("SCOUT_RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("SCOUT_DATA_DIR", str(tmp_path / "data"))
    app = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    at = AppTest.from_file(str(app), default_timeout=60)
    at.run()
    assert not at.exception
    infos = [i.value for i in at.info]
    assert any(i.startswith(f"Replay of recorded run {DEMO_ID}") for i in infos)
    assert any("Recalled 1 facts about Zoho" in s.value for s in at.success)
    assert any("Using 1 lessons learned" in s.value for s in at.success)
    markdown = " ".join(m.value for m in at.markdown)
    assert "Company in 60 seconds?" in markdown  # plan panel
    assert "## Sources" in markdown and "[[1]](https://en.wikipedia.org" in markdown  # brief
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Score"] == "Readiness 2/3" and metrics["Memory steps"] == "1"
    labels = [e.label for e in at.expander]
    assert any(label.startswith("Step 2 · retried") for label in labels)


# --- step 0 fixes ------------------------------------------------------------------------------


def test_recall_ranks_by_relevance(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "db.sqlite")
    items = [
        Evidence(id="E1", claim="Zoho opened an office in Austin.", source_url="https://a.x"),
        Evidence(
            id="E2",
            claim="Zoho interview process has a coding round.",
            source_url="https://b.x",
            topic="interview process",
        ),
        Evidence(id="E3", claim="Zoho revenue grew.", source_url="https://c.x"),
    ]
    for i, e in enumerate(items):
        store.add_facts("zoho", [e], "r", now=NOW + timedelta(minutes=i))
    fresh, _ = store.recall("zoho", 7, 2, 2, 6000, now=NOW, relevance="interview process coding")
    assert fresh[0].claim.startswith("Zoho interview")
    assert fresh[1].claim == "Zoho revenue grew."  # then newest first
    plain, _ = store.recall("zoho", 7, 2, 3, 6000, now=NOW)
    assert [f.claim for f in plain][0] == "Zoho revenue grew."


@pytest.mark.parametrize(
    ("text", "bad"),
    [
        ("Identify decision-makers via LinkedIn profiles before outreach.", True),
        ("Collect recruiter emails from the careers page.", True),
        ("Find the HR phone number early.", True),
        ("Check employee profiles on social sites.", True),
        ("Use the LinkedIn company page for headcount trends.", False),
        ("Search the careers page before news for hiring signals.", False),
    ],
)
def test_contact_policy_filter(text: str, bad: bool) -> None:
    assert violates_contact_policy(text) is bad


def test_reflector_discards_profile_lessons() -> None:
    fake = FakeLLM(
        [
            {
                "lessons": [
                    "Use LinkedIn profiles to find hiring managers.",
                    "Search the careers page first.",
                ]
            }
        ]
    )
    out = reflect(fake.llm(), "sales_prospect", "summary", [], 3)
    assert out.lessons == ["Search the careers page first."]


@pytest.mark.parametrize(
    ("text", "meta"),
    [
        ("No critical unknowns prevent the candidate from preparing effectively.", True),
        ("None", True),
        ("No unknowns.", True),
        ("N/A", True),
        ("Exact intern stipend is not public.", False),
        ("No public data on campus hiring volume.", False),
    ],
)
def test_meta_unknown_filter(text: str, meta: bool) -> None:
    assert is_meta_unknown(text) is meta


def _app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # noqa: ANN202 - AppTest type
    from streamlit.testing.v1 import AppTest

    runs_dir = tmp_path / "runs"
    if not (runs_dir / DEMO_ID).exists():
        shutil.copytree(FIXTURES / "runs" / DEMO_ID, runs_dir / DEMO_ID)
    monkeypatch.setenv("SCOUT_RUNS_DIR", str(runs_dir))
    monkeypatch.setenv("SCOUT_DATA_DIR", str(tmp_path / "data"))
    app = Path(__file__).parent.parent / "app" / "streamlit_app.py"
    return AppTest.from_file(str(app), default_timeout=60)


def test_feedback_click_persists_and_changes_insights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    at = _app(tmp_path, monkeypatch)
    at.run()
    at.feedback(key=f"fb-{DEMO_ID}-0").set_value(1).run()  # 👍 beside "Company in 60 seconds"
    assert not at.exception
    store = MemoryStore(tmp_path / "data" / "scout.db")
    assert store.source_scores()["en.wikipedia.org"] == pytest.approx(2 / 3)
    assert store.feedback()[0]["section"] == "Company in 60 seconds"
    at.radio(key="insights-source").set_value("Memory database").run()
    tables = [t.value for t in at.table]
    assert any("en.wikipedia.org" in t.to_string() for t in tables)  # Insights top sources


def test_compare_two_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = MemoryStore(tmp_path / "data" / "scout.db")
    store.save_run(
        "run-a",
        "g",
        "sales_prospect",
        ["Zoho"],
        NOW,
        NOW,
        {"tool_calls": 13, "llm_calls": 33, "total_tokens": 44729, "memory_steps": 0},
    )
    store.save_run(
        "run-b",
        "g",
        "interview_prep",
        ["Zoho"],
        NOW + timedelta(minutes=5),
        NOW,
        {"tool_calls": 5, "llm_calls": 16, "total_tokens": 18861, "memory_steps": 1},
    )
    at = _app(tmp_path, monkeypatch)
    at.run()
    assert not at.exception
    table = next(df.value for df in at.dataframe if "change (B − A)" in df.value.columns)
    rows = table.set_index("metric")
    assert rows.loc["tool calls", "Run A"] == 13 and rows.loc["tool calls", "Run B"] == 5
    assert rows.loc["tool calls", "change (B − A)"] == -8
    assert rows.loc["tokens", "change (B − A)"] == 18861 - 44729
