from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scout.agent.executor import EvidenceLedger
from scout.agent.synthesizer import escape_md, render_markdown, render_parts
from scout.briefclean import clean_brief, clean_text, dedupe_claims
from scout.events import Event, load_trace
from scout.playbooks import purpose_label
from scout.schemas import Brief, Claim, Finding, Section
from scout.ui.recorded import recorded_lessons, recorded_runs, recorded_sources
from scout.ui.viewmodel import RunView, build_view, reduce

FIXTURES = Path(__file__).parent / "fixtures"
DEMO_ID = "20260101-120000-demo"
REPO = Path(__file__).parent.parent


def ev(type_: str, payload: dict, stage: str = "execute") -> Event:
    return Event(run_id="r", stage=stage, type=type_, payload=payload)  # type: ignore[arg-type]


# --- dollar escaping ---------------------------------------------------------------------------


def test_two_dollar_amounts_on_one_line_are_escaped() -> None:
    line = "FY2024 revenue of $720.4 million, up 21% from $596.4 million in FY2023"
    assert escape_md(line) == (
        "FY2024 revenue of \\$720.4 million, up 21% from \\$596.4 million in FY2023"
    )
    assert escape_md(escape_md(line)) == escape_md(line)  # idempotent


def test_report_md_escapes_dollars() -> None:
    ledger = EvidenceLedger()
    ledger.add(Finding(claim="rev", source_url="https://ir.example.com", snippet="$720.4"), 1)
    brief = Brief(
        title="Freshworks",
        purpose_type="interview_prep",
        sections=[
            Section(
                title="Company in 60 seconds",
                claims=[
                    Claim(
                        text="Revenue was $720.4 million, up from $596.4 million.",
                        evidence_ids=["E1"],
                    )
                ],
            )
        ],
    )
    md = render_markdown(brief, ledger, "run")
    assert "\\$720.4 million, up from \\$596.4 million. [1]" in md
    assert "_Purpose: Interview prep ·" in md


# --- final step statuses -----------------------------------------------------------------------


def _plan(n: int) -> list[Event]:
    steps = [{"id": i, "question": f"q{i}"} for i in range(1, n + 1)]
    return [
        ev("plan_created", {"steps": steps}, "plan"),
        ev("step_started", {"id": 1, "question": "q1"}),
        ev("critique", {"step_id": 1, "verdict": "complete"}, "critique"),
    ]


def test_budget_stop_marks_unreached_steps_skipped() -> None:
    finished = ev("run_finished", {"status": "ok", "metrics": {"budget_exhausted": True}}, "finish")
    view = build_view([*_plan(3), finished])
    assert [s.status for s in view.steps] == ["done", "skipped (budget)", "skipped (budget)"]


def test_unreached_steps_without_budget_stop() -> None:
    view = build_view([*_plan(2), ev("run_finished", {"status": "ok", "metrics": {}}, "finish")])
    assert [s.status for s in view.steps] == ["done", "not reached"]
    assert all(s.status != "pending" for s in view.steps)


def test_step_summary_and_notable() -> None:
    view = build_view(load_trace(FIXTURES / "runs" / DEMO_ID / "trace.jsonl"))
    step2 = view.step(2)
    assert step2.summary.startswith("Step 2 · retried → done · 2 tool calls · critic: followup")
    assert "new approach: search for the written test format" in step2.summary
    assert view.step(1).summary == "Step 1 · from memory · 1 facts from memory"
    assert [s.notable for s in view.steps] == [True, True, True]  # memory, retry, replan


def test_active_step_tracked_while_running() -> None:
    view = RunView()
    for event in _plan(2)[:2]:
        reduce(view, event)
    assert view._current == 1 and not view.finished


# --- jargon filter and dedupe ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "High hiring volume with active fresher drives (E6, E7).",
            "High hiring volume with active fresher drives.",
        ),
        ("Headcount is 18,000 [E1].", "Headcount is 18,000."),
        (
            "Based on the evidence ledger, Zoho hires 800 freshers.",
            "Based on the sources, Zoho hires 800 freshers.",
        ),
        (
            "Key findings show a step-by-step interview process.",
            "Key facts show a detailed interview process.",
        ),
        ("The interview has five steps.", "The interview has five stages."),
        ("E3", None),
    ],
)
def test_jargon_filter(text: str, expected: str | None) -> None:
    assert clean_text(text) == expected


def test_clean_brief_covers_every_field() -> None:
    brief = Brief(
        title="t",
        summary=[Claim(text="Fit is strong (E1).", evidence_ids=["E1"])],
        sections=[Section(title="Snapshot", claims=[Claim(text="E2", evidence_ids=["E2"])])],
        score_reasons=["Large base (E1, E3)."],
        unknowns=["Budget owner per the ledger"],
    )
    out = clean_brief(brief)
    assert out.summary[0].text == "Fit is strong."
    assert out.sections[0].claims == []  # nothing left after removing the id: dropped
    assert out.score_reasons == ["Large base."]
    assert out.unknowns == ["Budget owner per the sources"]


def test_claim_dedupe_keeps_first_occurrence() -> None:
    brief = Brief(
        title="t",
        sections=[
            Section(
                title="A",
                claims=[Claim(text="Zoho has 18,000 employees in 2025.", evidence_ids=["E1"])],
            ),
            Section(
                title="B",
                claims=[
                    Claim(text="Different words, same source.", evidence_ids=["E1"]),
                    Claim(text="In 2025 Zoho has 18,000 employees.", evidence_ids=["E9"]),
                    Claim(text="Zoho uses Java.", evidence_ids=["E2"]),
                ],
            ),
        ],
    )
    out = dedupe_claims(brief)
    assert [c.text for c in out.sections[0].claims] == ["Zoho has 18,000 employees in 2025."]
    assert [c.text for c in out.sections[1].claims] == ["Zoho uses Java."]


def test_render_parts_split_for_section_feedback() -> None:
    ledger = EvidenceLedger()
    ledger.add(Finding(claim="c", source_url="https://a.com"), 1)
    brief = Brief(
        title="T",
        purpose_type="sales_prospect",
        score="Fit 70/100",
        summary=[Claim(text="Pitch the skills-first angle.", evidence_ids=["E1"])],
        sections=[
            Section(title="Snapshot", claims=[Claim(text="x", evidence_ids=["E1"])]),
            Section(title="Risks"),
        ],
    )
    parts = render_parts(brief, ledger, "r")
    assert "**Key takeaways**" in parts.header and "Sales prospect" in parts.header
    assert [t for t, _ in parts.sections] == ["Snapshot", "Risks"]
    assert parts.sections[1][1].startswith("_Nothing verified")
    assert parts.markdown() == render_markdown(brief, ledger, "r")


# --- labels and metrics ------------------------------------------------------------------------


def test_purpose_labels() -> None:
    assert purpose_label("interview_prep") == "Interview prep"
    assert purpose_label("sales_prospect") == "Sales prospect"
    assert purpose_label("competitor") == "Competitor"
    assert "_" not in purpose_label("some_new_purpose")


def test_metric_formatting() -> None:
    import importlib.util

    spec = importlib.util.spec_from_file_location("app_mod", REPO / "app" / "streamlit_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)  # main() only runs under __main__
    assert app.fmt_metric("Tokens", 29723) == "29,723"
    assert app.fmt_metric("Seconds", 56.4) == "56.4 s"
    assert app.fmt_metric("Citation coverage", 100.0) == "100%"
    assert app.fmt_metric("Citation coverage", 95.5) == "95.5%"
    assert app.fmt_metric("Tool calls", None) == "–"


# --- insights from traces ----------------------------------------------------------------------


def test_insights_from_traces(tmp_path: Path) -> None:
    runs_dir = tmp_path / "runs"
    shutil.copytree(FIXTURES / "runs" / DEMO_ID, runs_dir / DEMO_ID)
    recorded = recorded_runs(runs_dir, None)
    [run] = recorded
    row = run.as_row()
    assert row["purpose_type"] == "interview_prep" and row["targets"] == ["Zoho"]
    assert row["metrics"]["memory_steps"] == 1
    lessons = recorded_lessons(recorded)
    texts = {lesson["lesson"]: lesson for lesson in lessons}
    assert texts["Read the careers page before generic news searches."]["uses"] == 1
    assert texts["Read the careers page before generic news searches."]["up"] == 1  # vote event
    assert "For interview prep, search the written test format early." in texts
    sources = recorded_sources(recorded)
    assert sources == [{"domain": "example.com", "useful": 1, "useless": 0, "score": 2 / 3}]


def test_compare_plans_marks_memory_and_replan(tmp_path: Path) -> None:
    import importlib.util

    shutil.copytree(FIXTURES / "runs" / DEMO_ID, tmp_path / DEMO_ID)
    [run] = recorded_runs(tmp_path, None)
    spec = importlib.util.spec_from_file_location("app_mod", REPO / "app" / "streamlit_app.py")
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    md = app.plan_markdown(run)
    assert md.startswith("**Interview prep** · Zoho")
    assert "1. Company in 60 seconds? 🧠 _from memory_" in md
    assert "➕ _added by replan_" in md
    assert "📚 Read the careers page before generic news searches." in md


def test_app_insights_defaults_to_recorded_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamlit.testing.v1 import AppTest

    examples = tmp_path / "examples"
    for run in (REPO / "examples").iterdir():
        shutil.copytree(run, examples / run.name)
    monkeypatch.setenv("SCOUT_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setenv("SCOUT_EXAMPLES_DIR", str(examples))
    monkeypatch.setenv("SCOUT_DATA_DIR", str(tmp_path / "data"))  # empty database
    at = AppTest.from_file(str(REPO / "app" / "streamlit_app.py"), default_timeout=60)
    at.run()
    assert not at.exception
    assert at.radio(key="insights-source").value.startswith("Recorded runs")
    runs_table = next(df.value for df in at.dataframe if "tool calls" in df.value.columns)
    assert len(runs_table) == 4 and "Sales prospect" in set(runs_table["purpose"])
    assert any("unstop.com" in t.value.to_string() for t in at.table)  # sources from traces
    plan_md = " ".join(m.value for m in at.markdown)
    assert "🧠 _from memory_" in plan_md  # compare plans rendered
    assert "Plan A" in [s.label for s in at.selectbox]
