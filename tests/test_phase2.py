from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from scout.agent.executor import EvidenceLedger
from scout.agent.orchestrator import Orchestrator
from scout.agent.verifier import check_claim, find_issues, move_to_unknowns, numbers_in
from scout.config import Candidate, Settings, router_enabled
from scout.events import Event
from scout.llm import (
    LLM,
    CandidateUnavailableError,
    LLMError,
    QuotaExhaustedError,
    TransientLLMError,
    parse_wait,
)
from scout.router import RouterBackend
from scout.safety import EMAIL_MASK, PHONE_MASK, scrub_brief, scrub_text
from scout.schemas import Brief, Claim, Finding, Section
from scout.tools.cache import DiskCache
from scout.tools.registry import REPEAT_NOTE, build_registry
from tests.fakes import FakeLLM, FakeSearch

SEARCH_URL = "https://example.com/zoho/1"


class Clock:
    """Manually advanced clock."""

    def __init__(self, t: float = 1000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def settings_for(tmp_path: Path, **overrides: object) -> Settings:
    base = Settings(
        scout_model="fake-model", runs_dir=tmp_path, data_dir=tmp_path, cache_enabled=False
    )
    return base.with_overrides(**overrides)


def registry(settings: Settings, search: FakeSearch | None = None) -> Any:
    return build_registry(
        settings,
        [(search or FakeSearch()).provider("fake")],
        lambda url: "Page text 800.",
        cache=None,
    )


INTAKE = {"targets": ["Zoho"], "purpose_type": "interview_prep"}


def plan(n: int) -> dict:
    return {"steps": [{"id": i, "question": f"Q{i}?"} for i in range(1, n + 1)]}


def search(q: str = "zoho") -> dict:
    return {"thought": "s", "type": "tool", "tool": "web_search", "args": {"query": q}}


def finish(snippet: str = "Zoho hires 800 freshers") -> dict:
    return {
        "thought": "f",
        "type": "finish",
        "findings": [
            {"claim": "Zoho hires 800 freshers", "source_url": SEARCH_URL, "snippet": snippet}
        ],
    }


def critique(verdict: str, **extra: Any) -> dict:
    return {"verdict": verdict, "reason": f"because {verdict}", **extra}


def brief(text: str = "Zoho hires 800 freshers a year.", ids: list[str] | None = None) -> dict:
    return {
        "title": "Zoho",
        "sections": [
            {"title": "Snapshot", "claims": [{"text": text, "evidence_ids": ids or ["E1"]}]}
        ],
        "unknowns": [],
    }


def run(tmp_path: Path, script: list, **overrides: object) -> tuple[list[Event], FakeLLM]:
    settings = settings_for(tmp_path, **overrides)
    fake = FakeLLM(script)
    orch = Orchestrator(settings, llm=fake.llm(settings), registry=registry(settings))
    return list(orch.run("Prep me for Zoho")), fake


def synth_prompt(fake: FakeLLM) -> str:
    return next(
        r.messages[0]["content"]
        for r in reversed(fake.requests)
        if "Scout's synthesizer" in r.messages[0]["content"]
    )


def types(events: list[Event]) -> list[str]:
    return [e.type for e in events]


# --- critic branches ------------------------------------------------------------------------


def test_critic_complete_moves_to_next_step(tmp_path: Path) -> None:
    events, _ = run(
        tmp_path,
        [
            INTAKE,
            plan(2),
            search(),
            finish(),
            critique("complete"),
            search("b"),
            finish(),
            critique("complete"),
            brief(),
        ],
    )
    assert types(events).count("step_started") == 2
    verdicts = [e.payload["verdict"] for e in events if e.type == "critique"]
    assert verdicts == ["complete", "complete"]
    assert "retry" not in types(events) and "replan" not in types(events)


def test_critic_retry_reruns_step_with_new_approach(tmp_path: Path) -> None:
    script = [
        INTAKE,
        plan(1),
        search(),
        finish(),
        critique("retry", new_approach="careers page"),
        search("zoho careers page"),
        finish(),
        critique("complete"),
        brief(),
    ]
    events, fake = run(tmp_path, script)
    starts = [e for e in events if e.type == "step_started"]
    assert len(starts) == 2 and starts[1].payload["approach"] == "careers page"
    assert next(e for e in events if e.type == "retry").payload["new_approach"] == "careers page"
    retry_prompt = fake.requests[5].messages[0]["content"]
    assert "New approach: careers page" in retry_prompt
    assert events[-1].payload["metrics"]["retries"] == 1


def test_critic_retry_capped_per_step(tmp_path: Path) -> None:
    script = [
        INTAKE,
        plan(1),
        search(),
        finish(),
        critique("retry", new_approach="a"),
        search("x"),
        finish(),
        critique("retry", new_approach="b"),
        brief(),
    ]
    events, _ = run(tmp_path, script)
    assert types(events).count("retry") == 1
    assert types(events).count("step_started") == 2


def test_critic_followup_adds_one_step(tmp_path: Path) -> None:
    follow = {"id": 9, "question": "What does Zoho's interview loop look like?"}
    script = [
        INTAKE,
        plan(1),
        search(),
        finish(),
        critique("followup", followup=follow),
        search("loop"),
        finish(),
        critique("followup", followup=follow),
        brief(),
    ]
    events, _ = run(tmp_path, script)
    replans = [e for e in events if e.type == "replan"]
    assert len(replans) == 1  # max 1 follow-up per run
    assert replans[0].payload["added_step"]["id"] == 2
    assert replans[0].payload["added_step"]["is_followup"] is True
    assert types(events).count("step_started") == 2


def test_critic_unknown_goes_to_synthesizer(tmp_path: Path) -> None:
    events, fake = run(
        tmp_path, [INTAKE, plan(1), search(), finish(), critique("unknown"), brief()]
    )
    assert "- Q1?" in synth_prompt(fake)
    assert events[-1].payload["metrics"]["critique_verdicts"] == {"unknown": 1}


def test_source_ratings_recorded_in_trace(tmp_path: Path) -> None:
    ratings = {SEARCH_URL: "useful", "https://not-used.example": "useless"}
    events, _ = run(
        tmp_path,
        [
            INTAKE,
            plan(1),
            search(),
            finish(),
            critique("complete", source_ratings=ratings),
            brief(),
        ],
    )
    payload = next(e for e in events if e.type == "critique").payload
    assert payload["source_ratings"] == {SEARCH_URL: "useful"}  # only sources actually used


# --- verifier -------------------------------------------------------------------------------


def ledger_with(*snippets: str) -> EvidenceLedger:
    ledger = EvidenceLedger()
    for i, snip in enumerate(snippets):
        ledger.add(Finding(claim=f"c{i}", source_url=f"https://s{i}.com", snippet=snip), 1)
    return ledger


def test_numbers_normalised() -> None:
    assert numbers_in("30,300 staff, $1.05 billion in FY2023, cites E12") == {
        "30300",
        "1.05",
        "2023",
    }


def test_numeric_fidelity_flag() -> None:
    ledger = ledger_with("Zoho has 18,000+ employees (2025)")
    assert check_claim(Claim(text="Zoho has 18,000 employees", evidence_ids=["E1"]), ledger) is None
    kind, detail = check_claim(Claim(text="Zoho has 12,000 employees", evidence_ids=["E1"]), ledger)
    assert kind == "numeric" and "12000" in detail


def test_unsupported_claim_flag() -> None:
    ledger = ledger_with("x")
    assert check_claim(Claim(text="t", evidence_ids=["E9"]), ledger)[0] == "unsupported"


def test_conflict_note_passes_when_both_figures_cited() -> None:
    ledger = ledger_with("12,000 employees in 2023", "18,000+ employees in 2025")
    claim = Claim(
        text="Zoho has 18,000+ employees (2025); an older 2023 source says 12,000.",
        evidence_ids=["E1", "E2"],
    )
    assert check_claim(claim, ledger) is None


def test_synthesizer_prompt_has_conflict_rule(tmp_path: Path) -> None:
    _, fake = run(tmp_path, [INTAKE, plan(1), search(), finish(), critique("complete"), brief()])
    assert "most recent dated figure" in synth_prompt(fake)


def test_verifier_revision_fixes_numeric_claim(tmp_path: Path) -> None:
    revision = {
        "revisions": [{"id": "S1C1", "text": "Zoho hires 800 freshers.", "evidence_ids": ["E1"]}]
    }
    events, _ = run(
        tmp_path,
        [
            INTAKE,
            plan(1),
            search(),
            finish(),
            critique("complete"),
            brief("Zoho hires 950 freshers."),
            revision,
        ],
    )
    ver = next(e for e in events if e.type == "verification").payload
    assert ver["flagged"] == 1 and ver["flagged_numeric"] == ["Zoho hires 950 freshers."]
    assert ver["moved_to_unknowns"] == []
    report = events[-1].payload["report_md"]
    assert "Zoho hires 800 freshers. [1]" in report
    assert events[-1].payload["metrics"]["citation_coverage"] == 100.0


def test_verifier_moves_unfixable_claims_to_unknowns(tmp_path: Path) -> None:
    revision = {
        "revisions": [{"id": "S1C1", "text": "Zoho hires 999 freshers.", "evidence_ids": ["E1"]}]
    }
    events, _ = run(
        tmp_path,
        [
            INTAKE,
            plan(1),
            search(),
            finish(),
            critique("complete"),
            brief("Zoho hires 950 freshers."),
            revision,
        ],
    )
    report = events[-1].payload["report_md"]
    assert "Could not verify: Zoho hires 999 freshers." in report
    assert events[-1].payload["metrics"]["unsupported_claims"] == 1


def test_move_to_unknowns_keeps_good_claims() -> None:
    ledger = ledger_with("Zoho has 800 interns")
    b = Brief(
        title="t",
        sections=[
            Section(
                title="s",
                claims=[
                    Claim(text="Zoho has 800 interns", evidence_ids=["E1"]),
                    Claim(text="Zoho has 5 offices", evidence_ids=["E1"]),
                ],
            )
        ],
    )
    out = move_to_unknowns(b, find_issues(b, ledger))
    assert [c.text for c in out.sections[0].claims] == ["Zoho has 800 interns"]
    assert out.unknowns == ["Could not verify: Zoho has 5 offices"]


# --- safety ---------------------------------------------------------------------------------


def test_scrub_emails_and_phones() -> None:
    text = "Mail hr.team@zoho.com or call +91 44 6744 7070 / (512) 555-0147. Revenue 1,050,000,000."
    out = scrub_text(text)
    assert "zoho.com" not in out and EMAIL_MASK in out
    assert out.count(PHONE_MASK) == 2
    assert "1,050,000,000" in out  # numbers without phone separators survive
    assert scrub_text("Founded 1996–2009, FY 2023-2024") == "Founded 1996–2009, FY 2023-2024"


def test_scrub_brief_fields() -> None:
    b = Brief(
        title="t",
        sections=[Section(title="s", claims=[Claim(text="a@b.io")])],
        unknowns=["call 080-4455-6677"],
    )
    out = scrub_brief(b)
    assert out.sections[0].claims[0].text == EMAIL_MASK
    assert out.unknowns == [f"call {PHONE_MASK}"]


# --- tools: dedupe, cache, search fallback --------------------------------------------------


def test_dedupe_returns_earlier_observation_without_charging(tmp_path: Path) -> None:
    fake_search = FakeSearch()
    script = [
        INTAKE,
        plan(1),
        search("zoho"),
        search("  ZOHO "),
        finish(),
        critique("complete"),
        brief(),
    ]
    settings = settings_for(tmp_path)
    fake = FakeLLM(script)
    orch = Orchestrator(settings, llm=fake.llm(settings), registry=registry(settings, fake_search))
    events = list(orch.run("Prep me"))
    results = [e.payload for e in events if e.type == "tool_result"]
    assert [r["repeated"] for r in results] == [False, True]
    assert fake_search.queries == ["zoho"]  # tool ran once
    assert events[-1].payload["metrics"]["tool_calls"] == 1
    assert REPEAT_NOTE in fake.requests[4].messages[1]["content"]


def test_disk_cache_serves_second_registry(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    calls: list[str] = []

    def fetcher(url: str) -> str:
        calls.append(url)
        return "Cached page text."

    cache = DiskCache(tmp_path / "cache", ttl_s=3600)
    for _ in range(2):
        reg = build_registry(settings, [FakeSearch().provider()], fetcher, cache=cache)
        assert reg.run("fetch_page", {"url": "https://a.example"}).observation.ok
    assert calls == ["https://a.example"]


def test_disk_cache_expires(tmp_path: Path) -> None:
    clock = Clock(0.0)
    cache = DiskCache(tmp_path, ttl_s=10, clock=clock)
    cache.set("search", "q", [1])
    assert cache.get("search", "q") == [1]
    clock.t = 11
    assert cache.get("search", "q") is None


def test_search_fallback_inside_registry(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    reg = build_registry(
        settings,
        [
            FakeSearch(error=RuntimeError("tavily 432")).provider("tavily"),
            FakeSearch().provider("ddgs"),
        ],
        lambda url: "x",
        cache=None,
    )
    run_ = reg.run("web_search", {"query": "zoho"})
    assert run_.observation.ok and len(run_.urls) == 5


# --- router ---------------------------------------------------------------------------------


A = Candidate("groq", "model-a")
B = Candidate("groq", "model-b")
C = Candidate("gemini", "model-c")


def router(clock: Clock, **backends: FakeLLM) -> RouterBackend:
    labels = {"a": A, "b": B, "c": C}
    return RouterBackend(
        routes={"smart": [A, B, C], "fast": [B, A]},
        backends={labels[k].label: v for k, v in backends.items()},
        max_wait_s=20,
        clock=clock,
    )


def chat(r: RouterBackend, tier: str = "smart") -> Any:
    return r.chat(
        model="",
        messages=[{"role": "user", "content": "x"}],
        json_mode=False,
        temperature=0,
        tier=tier,
    )


def test_router_fails_over_on_daily_quota_and_records_switch() -> None:
    clock = Clock()
    r = router(clock, a=FakeLLM([QuotaExhaustedError("TPD", reset_after=600)]), b=FakeLLM(["ok"]))
    result = chat(r)
    assert (result.provider, result.model, result.text) == ("groq", "model-b", "ok")
    [switch] = r.pop_switches()
    assert switch["from"] == A.label and switch["to"] == B.label
    assert switch["reason"] == "TPD (reset in 600s)"


def test_router_cooldown_recovery() -> None:
    clock = Clock()
    a = FakeLLM([QuotaExhaustedError("TPD", reset_after=600), "back"])
    r = router(clock, a=a, b=FakeLLM(["b1", "b2"]))
    assert chat(r).model == "model-b"
    clock.t += 100
    assert chat(r).model == "model-b"  # A still cooling down
    clock.t += 600
    result = chat(r)
    assert result.model == "model-a" and result.text == "back"
    switches = r.pop_switches()
    assert [s["to"] for s in switches] == [B.label, A.label]


def test_router_long_429_cools_down_short_429_retries() -> None:
    clock = Clock()
    r = router(clock, a=FakeLLM([TransientLLMError("429", retry_after=5)]), b=FakeLLM(["b"]))
    with pytest.raises(TransientLLMError):
        chat(r)  # short wait: LLM.complete backs off on the same candidate
    r2 = router(clock, a=FakeLLM([TransientLLMError("429", retry_after=45)]), b=FakeLLM(["b"]))
    assert chat(r2).model == "model-b"
    assert "rate limited" in r2.pop_switches()[0]["reason"]


def test_router_disables_candidate_on_401_403() -> None:
    clock = Clock()
    a = FakeLLM([CandidateUnavailableError("HTTP 401")])
    r = router(clock, a=a, b=FakeLLM(["b1", "b2"]))
    assert chat(r).model == "model-b"
    clock.t += 10**6
    assert chat(r).model == "model-b"
    assert a.calls == 1  # never retried in this run
    assert r.status()[A.label].startswith("disabled")


def test_router_all_exhausted_raises_llm_error() -> None:
    clock = Clock()
    r = router(
        clock,
        a=FakeLLM([QuotaExhaustedError("d")]),
        b=FakeLLM([QuotaExhaustedError("d")]),
        c=FakeLLM([CandidateUnavailableError("403")]),
    )
    with pytest.raises(LLMError, match="No LLM candidate"):
        chat(r)


def test_router_uses_tier_order() -> None:
    r = router(Clock(), a=FakeLLM(["a"]), b=FakeLLM(["b"]))
    assert chat(r, "fast").model == "model-b"


def test_llm_records_provider_and_model_per_call() -> None:
    r = router(
        Clock(),
        a=FakeLLM(
            [
                QuotaExhaustedError("d"),
            ]
        ),
        b=FakeLLM(["one"]),
    )
    llm = LLM(settings=Settings(), backend=r)
    llm.complete([{"role": "user", "content": "hi"}])
    notices = llm.drain_notices()
    assert [n[0] for n in notices] == ["provider_switched", "llm_call"]
    assert notices[1][1]["model"] == "model-b" and notices[1][1]["provider"] == "groq"
    assert llm.usage.tokens_by_model().keys() == {"groq/model-b"}


def test_router_mode_selection() -> None:
    assert not router_enabled(Settings(llm_api_key="k", llm_base_url="https://api.groq.com/v1"))
    assert router_enabled(Settings(groq_api_key="k"))
    groq_via_llm = Settings(
        llm_router="on", llm_api_key="k", llm_base_url="https://api.groq.com/openai/v1"
    )
    assert router_enabled(groq_via_llm)
    assert not router_enabled(Settings(groq_api_key="k", llm_router="off"))


def test_parse_wait_formats() -> None:
    assert parse_wait("Please try again in 24m47.8s. Need more") == pytest.approx(1487.8)
    assert parse_wait("Please retry in 31.6s") == pytest.approx(31.6)
    assert parse_wait("try again in 1h2m3s") == 3723
    assert parse_wait("no hint") is None


# --- wall clock (A4) ------------------------------------------------------------------------


def test_backoff_never_overshoots_wall_clock(tmp_path: Path) -> None:
    clock = Clock()
    settings = settings_for(tmp_path, max_wall_clock_s=10)
    fake = FakeLLM([INTAKE, plan(2), TransientLLMError("429", retry_after=15), brief([], [])])
    orch = Orchestrator(settings, llm=fake.llm(settings, clock=clock), registry=registry(settings))
    events = list(orch.run("Prep me"))
    assert fake.sleeps == []  # the 15 s wait was refused, not slept
    final = events[-1].payload
    assert final["status"] == "ok"
    assert "wall clock" in final["metrics"]["budget_reason"]
    assert "Partial brief" in final["report_md"]
    assert types(events).count("step_started") == 1


def test_router_second_short_429_fails_over() -> None:
    clock = Clock()
    a = FakeLLM([TransientLLMError("429", retry_after=3), TransientLLMError("429", retry_after=3)])
    r = router(clock, a=a, b=FakeLLM(["b"]))
    with pytest.raises(TransientLLMError):
        chat(r)  # first short 429: back off on the same candidate
    assert chat(r).model == "model-b"  # second one: cool down and fail over
    assert "rate limited" in r.pop_switches()[0]["reason"]


def test_critic_unknowns_always_listed(tmp_path: Path) -> None:
    events, _ = run(tmp_path, [INTAKE, plan(1), search(), finish(), critique("unknown"), brief()])
    assert "- Not found: Q1?" in events[-1].payload["report_md"]


def test_personal_profiles_never_fetched_or_cited(tmp_path: Path) -> None:
    settings = settings_for(tmp_path)
    reg = registry(settings)
    out = reg.run("fetch_page", {"url": "https://in.linkedin.com/in/some-person"})
    assert not out.observation.ok and "personal profile" in out.observation.error
    profile = {"title": "P", "url": "https://in.linkedin.com/in/some-person", "snippet": "CTO"}
    fake_search = FakeSearch(results_by_query={"zoho cto": [profile]})
    finding = {
        "thought": "f",
        "type": "finish",
        "findings": [{"claim": "Zoho has a CTO", "source_url": profile["url"], "snippet": "CTO"}],
    }
    fake = FakeLLM([INTAKE, plan(1), search("zoho cto"), finding, critique("complete"), brief()])
    orch = Orchestrator(settings, llm=fake.llm(settings), registry=registry(settings, fake_search))
    events = list(orch.run("Prep me"))
    assert events[-1].payload["evidence"] == []
