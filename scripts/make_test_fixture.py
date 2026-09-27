"""Regenerate tests/fixtures/runs/20260101-120000-demo from fakes (no LLM, no network).

The fixture exercises recall, an injected lesson, a memory step, a retry, a replan, a
rate-limit wait, the verifier and a fixed-format score. Run from the repo root:
    python scripts/make_test_fixture.py
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scout.agent.orchestrator import Orchestrator  # noqa: E402
from scout.config import Settings  # noqa: E402
from scout.events import RunRecorder  # noqa: E402
from scout.llm import TransientLLMError  # noqa: E402
from scout.memory.store import MemoryStore  # noqa: E402
from scout.schemas import Evidence  # noqa: E402
from scout.tools.registry import build_registry  # noqa: E402
from tests.fakes import FakeLLM, FakeSearch  # noqa: E402

RUN_ID = "20260101-120000-demo"
WIKI = "https://en.wikipedia.org/wiki/Zoho_Corporation"
EX = "https://example.com"


def script(lesson_id: int) -> list:
    """Scripted LLM replies, in call order."""
    return [
        {"targets": ["Zoho"], "purpose_type": "interview_prep"},
        {
            "steps": [
                {
                    "id": 1,
                    "question": "Company in 60 seconds?",
                    "answered_from_memory": True,
                    "memory_fact_ids": [1],
                },
                {"id": 2, "question": "What does the Zoho interview process look like?"},
            ]
        },
        TransientLLMError("429", retry_after=2, model="groq/qwen/qwen3.8-27b"),
        {
            "thought": "Search for the interview process.",
            "type": "tool",
            "tool": "web_search",
            "args": {"query": "zoho interview process"},
        },
        {
            "thought": "Enough to answer.",
            "type": "finish",
            "findings": [
                {
                    "claim": "Zoho interviews include 5 rounds.",
                    "source_url": f"{EX}/zoho-interview-process/1",
                    "snippet": "Zoho interviews include 5 rounds",
                    "confidence": 0.8,
                }
            ],
        },
        {
            "verdict": "retry",
            "reason": "Need more detail.",
            "new_approach": "search for the written test format",
        },
        {
            "thought": "Try the written test format.",
            "type": "tool",
            "tool": "web_search",
            "args": {"query": "zoho written test format"},
        },
        {
            "thought": "Done.",
            "type": "finish",
            "findings": [
                {
                    "claim": "The first round is a written aptitude test.",
                    "source_url": f"{EX}/zoho-written-test-format/1",
                    "snippet": "first round is a written aptitude test",
                }
            ],
        },
        {
            "verdict": "followup",
            "reason": "Culture matters too.",
            "followup": {"id": 9, "question": "What is Zoho's engineering culture?"},
        },
        {
            "thought": "Search culture.",
            "type": "tool",
            "tool": "web_search",
            "args": {"query": "zoho engineering culture"},
        },
        {
            "thought": "Done.",
            "type": "finish",
            "findings": [
                {
                    "claim": "Zoho favours long-term product ownership.",
                    "source_url": f"{EX}/zoho-engineering-culture/1",
                    "snippet": "long-term product ownership",
                }
            ],
        },
        {
            "verdict": "complete",
            "reason": "Answered.",
            "source_ratings": {f"{EX}/zoho-engineering-culture/1": "useful"},
        },
        {
            "title": "Zoho SDE intern interview prep",
            "score": "Readiness 2/3",
            "score_reasons": [
                "ready: company basics",
                "ready: interview format",
                "todo: intern stipend",
            ],
            "sections": [
                {
                    "title": "Company in 60 seconds",
                    "claims": [
                        {"text": "Zoho was founded in 1996 in Chennai.", "evidence_ids": ["E1"]}
                    ],
                },
                {
                    "title": "Likely topics",
                    "claims": [
                        {
                            "text": "Zoho interviews include 5 rounds, "
                            "starting with a written aptitude test.",
                            "evidence_ids": ["E2", "E3"],
                        }
                    ],
                },
                {
                    "title": "Smart questions to ask",
                    "claims": [
                        {
                            "text": "Ask how teams keep long-term ownership of products.",
                            "evidence_ids": ["E4"],
                        }
                    ],
                },
            ],
            "unknowns": ["Exact intern stipend."],
        },
        {
            "lessons": ["For interview prep, search the written test format early."],
            "votes": [{"lesson_id": lesson_id, "helpful": True}],
        },
    ]


def main() -> int:
    """Build the fixture run in a temp dir, then copy it into tests/fixtures/runs/."""
    tmp = Path(tempfile.mkdtemp())
    store = MemoryStore(tmp / "scout.db")
    store.add_facts(
        "zoho",
        [
            Evidence(
                id="E1",
                claim="Zoho was founded in 1996 in Chennai.",
                source_url=WIKI,
                snippet="founded in 1996 in Chennai",
                topic="overview",
            )
        ],
        "earlier-run",
    )
    lesson, _ = store.add_or_merge_lesson(
        "interview_prep", "Read the careers page before generic news searches.", 0.6
    )
    settings = Settings(
        scout_model="fake", runs_dir=tmp / "runs", data_dir=tmp, cache_enabled=False
    )
    fake = FakeLLM(script(lesson.id or 0))
    registry = build_registry(
        settings, [FakeSearch().provider()], lambda url: "text", cache=None, store=store
    )
    orch = Orchestrator(
        settings, llm=fake.llm(settings), registry=registry, store=store, run_id=RUN_ID
    )
    list(
        RunRecorder(settings.runs_dir, RUN_ID).record(
            orch.run("Prep me for an SDE intern interview at Zoho.")
        )
    )
    dst = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "runs" / RUN_ID
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(tmp / "runs" / RUN_ID, dst)
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
