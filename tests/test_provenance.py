"""Evidence provenance: finding snippets must come from the source text the model was shown."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from scout.agent.budget import Budget
from scout.agent.executor import (
    EvidenceLedger,
    _accept_findings,
    execute_step,
    find_snippet_span,
    snippet_in_source,
)
from scout.agent.orchestrator import Orchestrator
from scout.config import Settings
from scout.events import Event, RunRecorder, load_trace
from scout.playbooks import get_playbook
from scout.schemas import Finding, RunMetrics, Step, StepResult, Task
from scout.tools.registry import ToolRegistry, build_registry
from scout.ui.viewmodel import build_view
from tests.fakes import FakeLLM, FakeSearch

REPO = Path(__file__).resolve().parent.parent
PAGE_URL = "https://zoho.example/careers"
PAGE_TEXT = "Zoho hires 800 freshers a year through campus drives."
SOURCE = "Zoho Corporation hires 800 freshers a year through “campus drives” – mostly in Chennai."
PROFILE_URL = "https://in.linkedin.com/in/some-person"


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    base = Settings(
        scout_model="fake-model", runs_dir=tmp_path, data_dir=tmp_path, cache_enabled=False
    )
    return base.with_overrides(**overrides)


def registry(settings: Settings, search: FakeSearch | None = None) -> ToolRegistry:
    return build_registry(
        settings,
        [(search or FakeSearch()).provider("fake")],
        lambda url: f"{PAGE_TEXT} Page {url}.",
        cache=None,
    )


def span_text(snippet: str, source: str = SOURCE) -> str | None:
    span = find_snippet_span(snippet, source)
    return source[span[0] : span[1]] if span else None


# --- matching ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("snippet", "expected"),
    [
        ("hires 800 freshers a year", True),  # exact
        ("  ZOHO   corporation\nhires 800 ", True),  # whitespace and case
        ('through "campus drives" - mostly', True),  # straight quotes and hyphen vs curly / dash
        ('<untrusted_content source="u">hires 800 freshers</untrusted_content>', True),  # wrapper
        ("Zoho Corporation hires 800 freshers a year through!", True),  # near match, above 0.85
        ("Zoho Corporation hires 900 freshers a year", False),  # changed number, below 0.85
        ("Acme raised $5M in 2024", False),  # unrelated
        ("", False),  # empty
        ("   \n ", False),  # whitespace only
        ('<untrusted_content source="u"></untrusted_content>', False),  # wrapper only
    ],
)
def test_snippet_in_source(snippet: str, expected: bool) -> None:
    assert snippet_in_source(snippet, SOURCE) is expected


def test_near_match_respects_threshold() -> None:
    near = "Zoho Corporation hires 800 freshers a year through!"
    assert snippet_in_source(near, SOURCE, min_ratio=0.85)
    assert not snippet_in_source(near, SOURCE, min_ratio=0.99)


@pytest.mark.parametrize(
    ("snippet", "stored"),
    [
        ("ZOHO CORPORATION HIRES 800", "Zoho Corporation hires 800"),  # source's own case
        ('through "campus drives"', "through “campus drives”"),  # source's own quotes
        ("hires 80", "hires 800"),  # a cut never shortens a number
        ("oration hires", "Corporation hires"),  # nor a word
    ],
)
def test_span_is_original_source_text(snippet: str, stored: str) -> None:
    assert span_text(snippet) == stored


def test_span_widens_across_digit_separators_and_maps_expanded_chars() -> None:
    assert span_text("staff of 1,", "A staff of 1,200 people.") == "staff of 1,200"
    assert span_text("STRASSE 5", "Office: Straße 5, Berlin") == "Straße 5"


def test_fuzzy_match_keeps_only_the_matched_block() -> None:
    snippet = "Zoho hires 800 freshers a year through campus drives, up 40%"
    assert span_text(snippet, PAGE_TEXT) == "Zoho hires 800 freshers a year through campus drives"


# --- acceptance ----------------------------------------------------------------------------------


def test_accept_findings_checks_only_the_cited_urls_sources(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    sources = {"https://a.com": ["Alpha has 100 staff."], "https://b.com": ["Beta is new."]}
    findings = [
        Finding(claim="a", source_url="https://a.com", snippet="alpha has 100 staff"),
        Finding(claim="b", source_url="https://b.com", snippet="Alpha has 100 staff."),
        Finding(claim="c", source_url="https://c.com", snippet="anything"),
        Finding(claim="d", source_url="https://b.com", snippet=" "),
        Finding(claim="e", source_url=PROFILE_URL, snippet="CTO"),
    ]
    accepted, rejected = _accept_findings(findings, sources, settings)
    assert [(f.claim, f.snippet) for f in accepted] == [("a", "Alpha has 100 staff")]
    assert [(r.url, r.reason) for r in rejected] == [
        ("https://b.com", "snippet_not_in_source"),  # copied from a.com
        ("https://c.com", "unseen_url"),
        ("https://b.com", "empty_snippet"),
        (PROFILE_URL, "personal_profile"),
    ]
    assert rejected[-1].payload(3) == {
        "step_id": 3,
        "reason": "personal_profile",
        "domain": "in.linkedin.com",
    }


def _execute(settings: Settings, script: list) -> tuple[list[Event], StepResult, EvidenceLedger]:
    llm = FakeLLM(script).llm(settings)
    ledger = EvidenceLedger()
    gen = execute_step(
        Step(id=1, question="Q?"),
        task=Task(goal="g", targets=["Zoho"]),
        playbook=get_playbook("general"),
        llm=llm,
        registry=registry(settings),
        budget=Budget(settings, llm),
        settings=settings,
        ledger=ledger,
        emit=lambda t, p: Event(run_id="r", stage="execute", type=t, payload=p),
    )
    events: list[Event] = []
    while True:
        try:
            events.append(next(gen))
        except StopIteration as stop:
            return events, stop.value, ledger


def test_search_and_fetched_text_of_one_url_are_both_sources(tmp_path: Path) -> None:
    url = "https://example.com/zoho-careers/1"
    script = [
        {"thought": "s", "type": "tool", "tool": "web_search", "args": {"query": "zoho careers"}},
        {"thought": "r", "type": "tool", "tool": "fetch_page", "args": {"url": url}},
        {
            "thought": "f",
            "type": "finish",
            "findings": [
                {"claim": "hires", "source_url": url, "snippet": "hires 800 freshers a year"},
                {"claim": "snip", "source_url": url, "snippet": "Deterministic snippet 1"},
                {"claim": "made up", "source_url": url, "snippet": "Zoho hires 9,000 interns"},
            ],
        },
    ]
    events, result, ledger = _execute(settings_for(tmp_path), script)
    assert [e.snippet for e in ledger.items] == [
        "hires 800 freshers a year",
        "Deterministic snippet 1",
    ]
    assert result.snippets_rejected == 1
    rejected = [e.payload for e in events if e.type == "evidence_rejected"]
    assert rejected == [{"step_id": 1, "reason": "snippet_not_in_source", "url": url}]


# --- end to end ----------------------------------------------------------------------------------


def test_invented_tail_number_is_not_stored_and_its_claim_is_unverified(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    claim = "Zoho hires 800 freshers a year, up 40%."
    snippet = "Zoho hires 800 freshers a year through campus drives, up 40%"
    fake = FakeLLM(
        [
            {"targets": ["Zoho"], "purpose_type": "sales_prospect"},
            {"steps": [{"id": 1, "question": "Hiring?"}]},
            {"thought": "r", "type": "tool", "tool": "fetch_page", "args": {"url": PAGE_URL}},
            {
                "thought": "f",
                "type": "finish",
                "findings": [{"claim": claim, "source_url": PAGE_URL, "snippet": snippet}],
            },
            {"verdict": "complete"},
            {
                "title": "Zoho",
                "sections": [{"title": "S", "claims": [{"text": claim, "evidence_ids": ["E1"]}]}],
            },
            {"revisions": []},
            {"lessons": []},
        ]
    )
    orch = Orchestrator(settings, llm=fake.llm(settings), registry=registry(settings))
    events = list(orch.run("g"))
    final = events[-1].payload
    [evidence] = final["evidence"]
    assert evidence["snippet"] == "Zoho hires 800 freshers a year through campus drives"
    assert "40" not in evidence["snippet"]
    assert not any(e.type == "evidence_rejected" for e in events)
    assert all(c["text"] != claim for s in final["brief"]["sections"] for c in s["claims"])
    assert any(claim in u for u in final["brief"]["unknowns"])
    assert final["metrics"]["snippets_rejected"] == 0


def test_profile_url_never_reaches_trace_or_logs(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = settings_for(tmp_path)
    profile = {"title": "P", "url": PROFILE_URL, "snippet": "CTO at Zoho"}
    search = FakeSearch(results_by_query={"zoho cto": [profile]})
    finding = {"claim": "Zoho has a CTO", "source_url": PROFILE_URL, "snippet": "CTO"}
    fake = FakeLLM(
        [
            {"targets": ["Zoho"], "purpose_type": "sales_prospect"},
            {"steps": [{"id": 1, "question": "Who leads tech?"}]},
            {"thought": "s", "type": "tool", "tool": "web_search", "args": {"query": "zoho cto"}},
            {"thought": "f", "type": "finish", "findings": [finding]},
            {"verdict": "complete"},
            {"title": "Zoho", "sections": []},
            {"lessons": []},
        ]
    )
    orch = Orchestrator(settings, llm=fake.llm(settings), registry=registry(settings, search))
    recorder = RunRecorder(tmp_path, orch.run_id)
    with caplog.at_level(logging.DEBUG, logger="scout"):
        events = list(recorder.record(orch.run("Research Zoho")))
    rejected = [e.payload for e in events if e.type == "evidence_rejected"]
    assert rejected == [{"step_id": 1, "reason": "personal_profile", "domain": "in.linkedin.com"}]
    trace = (tmp_path / orch.run_id / "trace.jsonl").read_text(encoding="utf-8")
    assert "linkedin.com/in/" not in trace
    assert "linkedin.com/in/" not in caplog.text


# --- compatibility -------------------------------------------------------------------------------


def test_replay_tolerates_evidence_rejected(tmp_path: Path) -> None:
    def ev(type_: str, payload: dict, stage: str = "execute") -> str:
        return Event(run_id="r", stage=stage, type=type_, payload=payload).model_dump_json()  # type: ignore[arg-type]

    lines = [
        ev("plan_created", {"steps": [{"id": 1, "question": "Q?"}]}, "plan"),
        ev("step_started", {"id": 1, "question": "Q?"}),
        ev("evidence_rejected", {"step_id": 1, "reason": "snippet_not_in_source", "url": "u"}),
        ev("evidence_rejected", {"step_id": 1, "reason": "personal_profile", "domain": "d.com"}),
        ev("run_finished", {"status": "ok", "metrics": {}}, "finish"),
    ]
    path = tmp_path / "trace.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    view = build_view(load_trace(path))
    entries = [(e.kind, e.text, e.ok) for e in view.steps[0].entries]
    assert ("rejected", "Evidence rejected (snippet not in source): u", False) in entries
    assert ("rejected", "Evidence rejected (personal profile): d.com", False) in entries
    assert view.finished and not view.errors


def test_old_metrics_load_with_zero_rejections() -> None:
    paths = sorted((REPO / "examples").glob("*/metrics.json"))
    assert paths
    for path in paths:
        metrics = RunMetrics.model_validate(json.loads(path.read_text(encoding="utf-8")))
        assert metrics.snippets_rejected == 0
