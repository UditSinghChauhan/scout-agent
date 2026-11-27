from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from scout.agent.orchestrator import Orchestrator
from scout.agent.planner import normalize_plan
from scout.agent.reflector import Reflection, reflect
from scout.config import Settings
from scout.events import RunRecorder, load_trace
from scout.insights import apply_feedback, insights_tables
from scout.memory.store import MemoryStore, normalize_entity, score
from scout.schemas import Evidence, Fact, Lesson, ResearchPlan, Step
from scout.textutil import dedupe
from scout.tools.registry import build_registry
from scout.tools.web_search import rerank_by_source
from tests.fakes import FakeLLM, FakeSearch

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=UTC)
FACT_URL = "https://en.wikipedia.org/wiki/Zoho_Corporation"


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    base = Settings(
        scout_model="fake-model",
        runs_dir=tmp_path / "runs",
        data_dir=tmp_path,
        cache_enabled=False,
    )
    return base.with_overrides(**overrides)


def evidence(eid: str, claim: str, url: str, snippet: str = "", vol: str = "stable") -> Evidence:
    return Evidence(
        id=eid,
        claim=claim,
        source_url=url,
        snippet=snippet or claim,
        topic="overview",
        volatility=vol,
    )  # type: ignore[arg-type]


def seeded_store(tmp_path: Path) -> MemoryStore:
    store = MemoryStore(tmp_path / "scout.db")
    store.add_facts(
        "zoho",
        [
            evidence("E1", "Zoho was founded in 1996 in Chennai.", FACT_URL),
            evidence("E2", "Zoho has 18,000+ employees (2025).", FACT_URL),
        ],
        run_id="old-run",
    )
    return store


# --- entity normalization and freshness -------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Zoho Corporation", "zoho"),
        ("ZOHO Corp.", "zoho"),
        ("Freshworks Inc.", "freshworks"),
        ("Tata Consultancy Services Limited", "tata consultancy services"),
        ("Unstop Pvt Ltd", "unstop"),
        ("Acme Private Limited", "acme"),
        ("https://www.zoho.com/about", "zoho.com"),
        ("www.freshworks.com", "freshworks.com"),
    ],
)
def test_entity_normalization(raw: str, expected: str) -> None:
    assert normalize_entity(raw) == expected


def test_ttl_freshness_stable_vs_news(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "db.sqlite")
    store.add_facts("zoho", [evidence("E1", "stable fact", FACT_URL)], "r", now=NOW)
    store.add_facts("zoho", [evidence("E2", "news fact", FACT_URL, vol="news")], "r", now=NOW)
    for days, expected in ((1, {"stable fact", "news fact"}), (3, {"stable fact"}), (8, set())):
        fresh, stale = store.recall("zoho", 7, 2, 15, 6000, now=NOW + timedelta(days=days))
        assert {f.claim for f in fresh} == expected
        assert len(fresh) + len(stale) == 2


def test_recall_caps_count_and_chars(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "db.sqlite")
    items = [evidence(f"E{i}", f"fact number {i} " + "x" * 200, FACT_URL) for i in range(30)]
    store.add_facts("zoho", items, "r", now=NOW)
    fresh, _ = store.recall("zoho", 7, 2, 15, 6000, now=NOW)
    assert len(fresh) == 15
    fresh, _ = store.recall("zoho", 7, 2, 15, 1000, now=NOW)
    assert sum(len(f.claim) for f in fresh) <= 1000


def test_identical_fact_refreshes_instead_of_duplicating(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "db.sqlite")
    store.add_facts("zoho", [evidence("E1", "same", FACT_URL)], "r1", now=NOW)
    added = store.add_facts("zoho", [evidence("E9", "same", FACT_URL)], "r2", now=NOW)
    assert added == 0 and len(store.facts("zoho")) == 1


# --- memory steps ----------------------------------------------------------------------------


def test_plan_normalization_validates_memory_steps() -> None:
    plan = ResearchPlan(
        steps=[
            Step(id=1, question="Overview?", answered_from_memory=True, memory_fact_ids=[1, 99]),
            Step(id=2, question="Bogus memory?", answered_from_memory=True, memory_fact_ids=[99]),
            *[Step(id=3 + i, question=f"R{i}?") for i in range(6)],
        ]
    )
    out = normalize_plan(plan, max_steps=5, fact_ids={1, 2})
    assert out.steps[0].answered_from_memory and out.steps[0].memory_fact_ids == [1]
    assert not out.steps[1].answered_from_memory and out.steps[1].memory_fact_ids == []
    assert sum(not s.answered_from_memory for s in out.steps) == 5  # memory steps are extra
    assert [s.id for s in out.steps] == list(range(1, 7))


INTAKE = {"targets": ["Zoho Corporation"], "purpose_type": "interview_prep"}
SEARCH = {"thought": "s", "type": "tool", "tool": "web_search", "args": {"query": "zoho stack"}}
FINISH = {
    "thought": "f",
    "type": "finish",
    "findings": [
        {
            "claim": "Zoho uses Java.",
            "source_url": "https://example.com/zoho-stack/1",
            "snippet": "Zoho uses Java",
            "volatility": "stable",
        }
    ],
}
COMPLETE = {
    "verdict": "complete",
    "reason": "ok",
    "source_ratings": {"https://example.com/zoho-stack/1": "useful"},
}
REFLECTION = {"lessons": ["Check the engineering blog before generic searches for tech stack."]}


def memory_plan() -> dict:
    return {
        "steps": [
            {
                "id": 1,
                "question": "Company in 60 seconds?",
                "answered_from_memory": True,
                "memory_fact_ids": [1, 2],
            },
            {"id": 2, "question": "What stack does Zoho use?"},
        ]
    }


def brief() -> dict:
    return {
        "title": "Zoho interview prep",
        "sections": [
            {
                "title": "Company in 60 seconds",
                "claims": [{"text": "Zoho was founded in 1996.", "evidence_ids": ["E1"]}],
            },
            {
                "title": "Tech stack",
                "claims": [{"text": "Zoho uses Java.", "evidence_ids": ["E3"]}],
            },
        ],
        "unknowns": [],
    }


def run_with(tmp_path: Path, store: MemoryStore, script: list, **overrides: object) -> tuple:
    settings = settings_for(tmp_path, **overrides)
    fake = FakeLLM(script)
    reg = build_registry(
        settings, [FakeSearch().provider()], lambda u: "text", cache=None, store=store
    )
    orch = Orchestrator(settings, llm=fake.llm(settings), registry=reg, store=store)
    recorder = RunRecorder(settings.runs_dir, orch.run_id)
    events = list(recorder.record(orch.run("Prep me for an SDE intern interview at Zoho.")))
    return events, fake, orch


def test_memory_steps_skip_execution_and_keep_citations(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    script = [INTAKE, memory_plan(), SEARCH, FINISH, COMPLETE, brief(), REFLECTION]
    events, fake, _ = run_with(tmp_path, store, script)
    types = [e.type for e in events]
    [hit] = [e for e in events if e.type == "memory_hit"]
    assert hit.payload["fact_ids"] == [1, 2] and hit.payload["evidence_ids"] == ["E1", "E2"]
    assert types.count("step_started") == 1  # only the research step ran
    assert "[F1] Zoho was founded in 1996 in Chennai." in fake.requests[1].messages[0]["content"]
    final = events[-1].payload
    assert f"<{FACT_URL}>" in final["report_md"]  # memory evidence keeps its source URL
    assert "Zoho was founded in 1996. [1]" in final["report_md"]
    m = final["metrics"]
    assert m["memory_steps"] == 1 and m["facts_reused"] == 2 and m["tool_calls"] == 1
    assert m["citation_coverage"] == 100.0


def test_run_persists_facts_sources_lessons_and_run(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    events, _, orch = run_with(
        tmp_path, store, [INTAKE, memory_plan(), SEARCH, FINISH, COMPLETE, brief(), REFLECTION]
    )
    claims = {f.claim for f in store.facts("zoho")}
    assert "Zoho uses Java." in claims  # new verified evidence stored under the entity
    assert store.source_scores()["example.com"] == pytest.approx(2 / 3)
    [lesson] = store.lessons("interview_prep")
    assert lesson.text.startswith("Check the engineering blog")
    learned = [e for e in events if e.type == "lesson_learned"]
    assert learned[0].payload["merged"] is False
    assert store.run(orch.run_id)["metrics"]["memory_steps"] == 1


def test_lessons_injected_into_planner_and_listed_in_plan_event(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    lesson, _ = store.add_or_merge_lesson("interview_prep", "Read the careers page first.", 0.6)
    reflection = {
        "lessons": ["Prefer official engineering blogs for stack details."],
        "votes": [{"lesson_id": lesson.id, "helpful": True}, {"lesson_id": 999, "helpful": False}],
    }
    events, fake, _ = run_with(
        tmp_path, store, [INTAKE, memory_plan(), SEARCH, FINISH, COMPLETE, brief(), reflection]
    )
    plan_event = next(e for e in events if e.type == "plan_created")
    assert plan_event.payload["lessons"] == [{"id": lesson.id, "text": lesson.text}]
    assert f"[L{lesson.id}] Read the careers page first." in fake.requests[1].messages[0]["content"]
    updated = store.lesson(lesson.id)
    assert updated.votes_up == 1 and updated.votes_down == 0  # unknown id 999 ignored
    assert store.lesson_uses()[lesson.id] == 1


def test_recall_memory_tool(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    reg = build_registry(
        settings_for(tmp_path), [FakeSearch().provider()], lambda u: "t", cache=None, store=store
    )
    run = reg.run("recall_memory", {"entity": "Zoho Corp", "topic": "founded Chennai"})
    assert run.observation.ok and "1996" in run.observation.content
    assert run.urls == (FACT_URL,)


# --- source reliability ----------------------------------------------------------------------


def test_source_score_math(tmp_path: Path) -> None:
    assert score(0, 0) == 0.5
    assert score(3, 1) == pytest.approx(4 / 6)
    store = MemoryStore(tmp_path / "db.sqlite")
    store.rate_source("https://www.zoho.com/careers", True)
    store.rate_source("zoho.com", True)
    store.rate_source("https://spam.example/x", False)
    scores = store.source_scores()
    assert scores["zoho.com"] == pytest.approx(3 / 4)
    assert scores["spam.example"] == pytest.approx(1 / 3)


def results(*domains: str) -> list[dict[str, str]]:
    return [{"title": d, "url": f"https://{d}/p", "snippet": ""} for d in domains]


def test_rerank_is_gentle() -> None:
    res = results("a.com", "b.com", "c.com", "d.com", "e.com")
    # A perfect bottom result can climb at most one place, never to the top.
    out = rerank_by_source(res, {"e.com": 1.0}, 0.5)
    assert [r["title"] for r in out] == ["a.com", "b.com", "c.com", "e.com", "d.com"]
    # A distrusted top result with a trusted runner-up swaps with it, nothing more.
    out = rerank_by_source(res, {"a.com": 0.0, "b.com": 1.0}, 0.5)
    assert [r["title"] for r in out][:3] == ["b.com", "a.com", "c.com"]
    assert rerank_by_source(res, {}, 0.5) == res  # unseen domains are neutral


# --- lessons ---------------------------------------------------------------------------------


def test_lesson_dedupe_merges_and_votes(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "db.sqlite")
    first, merged = store.add_or_merge_lesson(
        "sales_prospect", "For hiring signals, the careers page beats news search.", 0.6
    )
    assert not merged
    again, merged = store.add_or_merge_lesson(
        "sales_prospect", "Careers pages beat news search for hiring signals.", 0.6
    )
    assert merged and again.id == first.id and again.votes_up == 1
    other, merged = store.add_or_merge_lesson(
        "sales_prospect", "Check pricing pages directly instead of review sites.", 0.6
    )
    assert not merged and other.id != first.id
    # Same text under another purpose is a separate lesson.
    _, merged = store.add_or_merge_lesson(
        "competitor", "For hiring signals, the careers page beats news search.", 0.6
    )
    assert not merged


def test_top3_lessons_by_votes_then_recency(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "db.sqlite")
    ids = []
    for i, text in enumerate(
        ["alpha lesson one", "bravo lesson two", "charlie lesson three", "delta lesson four"]
    ):
        lesson, _ = store.add_or_merge_lesson("general", text, 0.99, now=NOW + timedelta(minutes=i))
        ids.append(lesson.id)
    store.vote(ids[0], True)
    store.vote(ids[0], True)
    store.vote(ids[1], False)
    top = store.top_lessons("general", 3)
    # alpha (score .75) first; then the two unvoted (.5), newest first; bravo (.33) drops out.
    assert [lesson.id for lesson in top] == [ids[0], ids[3], ids[2]]


def test_reflection_validation() -> None:
    Reflection(lessons=["Short useful lesson."])
    with pytest.raises(ValidationError):
        Reflection(lessons=[" ".join(["word"] * 25)])
    with pytest.raises(ValidationError):
        Reflection(lessons=["a b", "c d", "e f", "g h"])
    with pytest.raises(ValidationError):
        Reflection(lessons=[])


def test_reflector_repairs_then_filters_votes() -> None:
    too_long = {"lessons": [" ".join(["word"] * 30)]}
    good = {
        "lessons": ["Search the careers page first."],
        "votes": [{"lesson_id": 7, "helpful": True}, {"lesson_id": 8, "helpful": False}],
    }
    fake = FakeLLM([too_long, good])
    injected = [Lesson(id=7, purpose_type="general", text="x")]
    out = reflect(fake.llm(), "general", "summary", injected, 3)
    assert fake.calls == 2  # one repair request
    assert [v.lesson_id for v in out.votes] == [7]


# --- feedback, insights, replay, unknowns ----------------------------------------------------


def test_feedback_updates_domains_and_lesson_votes(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "db.sqlite")
    lesson, _ = store.add_or_merge_lesson("sales_prospect", "Use careers pages.", 0.6)
    store.save_run(
        "r1", "goal", "sales_prospect", ["Zoho"], NOW, NOW, {"injected_lessons": [lesson.id]}
    )
    final = {
        "brief": {
            "sections": [
                {
                    "title": "Buying signals",
                    "claims": [{"text": "x", "evidence_ids": ["E1", "E2"]}],
                },
                {"title": "Risks", "claims": [{"text": "y", "evidence_ids": ["E3"]}]},
            ]
        },
        "evidence": [
            {"id": "E1", "source_url": "https://www.zoho.com/careers"},
            {"id": "E2", "source_url": "https://unstop.com/blog/x"},
            {"id": "E3", "source_url": "https://spam.example/z"},
        ],
    }
    title, domains, lessons = apply_feedback(store, "r1", final, "buying", up=False)
    assert title == "Buying signals" and domains == ["unstop.com", "zoho.com"]
    assert lessons == [lesson.id]
    assert store.source_scores()["zoho.com"] == pytest.approx(1 / 3)
    assert store.lesson(lesson.id).votes_down == 1
    assert store.feedback()[0]["rating"] == "down"
    assert apply_feedback(store, "r1", final, "nonexistent", up=True) is None


def test_insights_with_empty_database(tmp_path: Path) -> None:
    tables = insights_tables(MemoryStore(tmp_path / "empty.db"))
    assert [t.title for t in tables] == ["Runs", "Lessons", "Top 5 sources", "Bottom 5 sources"]
    assert all(t.row_count == 0 and t.caption for t in tables)


def test_trace_is_replay_ready(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    events, _, orch = run_with(
        tmp_path, store, [INTAKE, memory_plan(), SEARCH, FINISH, COMPLETE, brief(), REFLECTION]
    )
    trace = load_trace(tmp_path / "runs" / orch.run_id / "trace.jsonl")
    assert [e.type for e in trace] == [e.type for e in events]
    final = trace[-1].payload
    for key in ("task", "plan", "brief", "evidence", "report_md", "metrics"):
        assert final[key], key
    assert any(e.type == "plan_created" for e in trace)
    assert any(e.type == "llm_call" for e in trace)


def test_unknowns_dedupe() -> None:
    items = [
        "Zoho's annual campus hiring volume is not disclosed.",
        "Not found: What is Zoho's annual campus hiring volume?",
        "Budget for assessment tools is unknown.",
    ]
    assert dedupe(items, 0.5) == [items[0], items[2]]


def test_fact_model_roundtrip(tmp_path: Path) -> None:
    store = seeded_store(tmp_path)
    fact: Fact = store.facts("zoho")[0]
    assert fact.id and fact.snippet and fact.volatility == "stable"


def _unused(_: Any) -> None:  # keep imports honest for type checkers
    pass


def _fact(fid: int, topic: str, vol: str = "stable") -> Fact:
    return Fact(
        id=fid,
        entity="zoho",
        topic=topic,
        claim=f"claim {fid}",
        source_url=FACT_URL,
        volatility=vol,
    )  # type: ignore[arg-type]


def test_attach_memory_converts_matching_research_step() -> None:
    from scout.agent.planner import attach_memory

    facts = [
        _fact(1, "What programming languages and cloud platforms does Zoho use?"),
        _fact(2, "What programming languages and cloud platforms does Zoho use?"),
        _fact(3, "Zoho headcount and offices?"),
    ]
    plan = ResearchPlan(
        steps=[
            Step(id=1, question="Which programming languages and cloud platforms does Zoho use?"),
            Step(id=2, question="Zoho interview rounds?"),
        ]
    )
    out = attach_memory(plan, facts, "Zoho")
    assert out.steps[0].answered_from_memory and out.steps[0].memory_fact_ids == [1, 2]
    assert not out.steps[1].answered_from_memory
    assert len(out.steps) == 2  # no background step needed


def test_attach_memory_prepends_background_step() -> None:
    from scout.agent.planner import attach_memory

    facts = [_fact(1, "news topic", "news"), _fact(2, "headcount"), _fact(3, "offices")]
    plan = ResearchPlan(steps=[Step(id=1, question="Zoho interview rounds?")])
    out = attach_memory(plan, facts, "Zoho")
    assert out.steps[0].answered_from_memory
    assert out.steps[0].memory_fact_ids == [2, 3, 1]  # stable facts first
    assert [s.id for s in out.steps] == [1, 2]
    assert attach_memory(plan, [], "Zoho") == plan
