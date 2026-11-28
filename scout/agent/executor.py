"""Executor stage: one Step -> StepResult via a ReAct loop (docs/SPEC.md §4).

``execute_step`` is a generator: it yields ``thought``/``tool_call``/``tool_result`` events and
returns the ``StepResult`` (use ``result = yield from execute_step(...)``). Each iteration asks
the LLM for one :class:`Action`; tool results are observed; a ``finish`` action returns findings
that become ledger :class:`Evidence` with ids. If the loop runs out of iterations or tool budget,
the model is asked once more to finish from what it has.

Provenance: a finding is accepted only if its snippet is found in the text the model was shown for
the cited URL. The stored snippet is then that span of the source, never the model's own wording.
"""

from __future__ import annotations

import difflib
import json
import logging
import re
import unicodedata
from collections import Counter
from collections.abc import Callable, Generator, Iterable
from dataclasses import dataclass, field
from typing import Any, Literal

from scout.agent.budget import SYNTHESIS_RESERVE, Budget
from scout.config import Settings
from scout.events import Event, EventType
from scout.llm import LLM, LLMValidationError
from scout.memory.store import domain_of
from scout.playbooks import Playbook
from scout.prompts import load_prompt
from scout.safety import is_personal_profile
from scout.schemas import Action, Evidence, Fact, Finding, Observation, Step, StepResult, Task
from scout.tools.registry import REPEAT_NOTE, ToolRegistry

logger = logging.getLogger(__name__)

EmitFn = Callable[[EventType, dict[str, Any]], Event]
RejectReason = Literal[
    "personal_profile", "unseen_url", "empty_snippet", "snippet_too_short", "snippet_not_in_source"
]
SNIPPET_REASONS: frozenset[str] = frozenset(
    {"empty_snippet", "snippet_too_short", "snippet_not_in_source"}
)

_TAG_RE = re.compile(r"</?untrusted_content[^>]*>")
_PUNCT = str.maketrans(
    dict.fromkeys("‘’‚‛′", "'") | dict.fromkeys("“”„‟″«»", '"') | dict.fromkeys("‐‑‒–—―−", "-")
)


def normalize_url(url: str) -> str:
    """Comparable form of a URL (case-insensitive scheme/host, no trailing slash or fragment)."""
    url = url.strip().split("#", 1)[0].rstrip("/")
    scheme, sep, rest = url.partition("://")
    if not sep:
        return url
    host, slash, path = rest.partition("/")
    return f"{scheme.lower()}://{host.lower()}{slash}{path}"


# --- provenance ----------------------------------------------------------------------------------


def _normalize(text: str) -> tuple[str, list[int]]:
    """Matching form of ``text`` plus, per output character, its offset in ``text``.

    Drops ``<untrusted_content>`` tags, applies NFKC, casefolds, unifies quotes and dashes and
    collapses whitespace (no leading or trailing space). A character that expands (``ß`` -> ``ss``)
    maps every output character back to the same original offset.
    """
    masked = _TAG_RE.sub(lambda m: " " * len(m.group()), text)
    out: list[str] = []
    index: list[int] = []
    pending_space: int | None = None
    for pos, ch in enumerate(masked):
        for c in unicodedata.normalize("NFKC", ch).casefold().translate(_PUNCT):
            if c.isspace():
                if pending_space is None:
                    pending_space = pos
                continue
            if pending_space is not None and out:
                out.append(" ")
                index.append(pending_space)
            pending_space = None
            out.append(c)
            index.append(pos)
    return "".join(out), index


def _is_token_char(text: str, i: int) -> bool:
    """True if ``text[i]`` belongs to a word or number (``,``/``.`` count between digits)."""
    if not 0 <= i < len(text):
        return False
    ch = text[i]
    if ch.isalnum():
        return True
    return ch in ",." and 0 < i < len(text) - 1 and text[i - 1].isdigit() and text[i + 1].isdigit()


def _widen(text: str, lo: int, hi: int) -> tuple[int, int]:
    """Grow ``text[lo:hi]`` to whole tokens, so a cut never turns "800" into "80"."""
    while lo > 0 and _is_token_char(text, lo - 1) and _is_token_char(text, lo):
        lo -= 1
    while hi < len(text) and _is_token_char(text, hi) and _is_token_char(text, hi - 1):
        hi += 1
    return lo, hi


def _match(snippet: str, source: str, min_ratio: float) -> tuple[int, int, float] | None:
    """(start, end, ratio) of ``snippet`` in ``source`` as original offsets; None if no match.

    Exact (normalised) substring first, ratio 1.0. Otherwise the longest common block, accepted
    when it covers at least ``min_ratio`` of the snippet; only that block is returned, so text the
    snippet adds beyond the source never becomes part of the span.
    """
    snip, _ = _normalize(snippet)
    src, index = _normalize(source)
    if not snip or not src:
        return None
    start = src.find(snip)
    size = len(snip)
    if start < 0:
        block = difflib.SequenceMatcher(None, snip, src, autojunk=False).find_longest_match(
            0, len(snip), 0, len(src)
        )
        if block.size == 0 or block.size / len(snip) < min_ratio:
            return None
        start, size = block.b, block.size
    lo, hi = _widen(source, index[start], index[start + size - 1] + 1)
    return lo, hi, size / len(snip)


def find_snippet_span(
    snippet: str, source: str, min_ratio: float = 0.85
) -> tuple[int, int] | None:
    """Original-text span of ``source`` that ``snippet`` was taken from; None if not found."""
    found = _match(snippet, source, min_ratio)
    return (found[0], found[1]) if found else None


def snippet_in_source(snippet: str, source: str, min_ratio: float = 0.85) -> bool:
    """True if ``snippet`` is in ``source`` (exactly or as a near match), after normalising."""
    return find_snippet_span(snippet, source, min_ratio) is not None


def _best_span(
    snippet: str, sources: Iterable[str], min_ratio: float, whole_source: bool = False
) -> str | None:
    """The source text that best matches ``snippet`` (an exact match wins), or None.

    Returns the matched span, or with ``whole_source`` the entire matching source text.
    """
    best: tuple[float, str] | None = None
    for source in sources:
        found = _match(snippet, source, min_ratio)
        if found and (best is None or found[2] > best[0]):
            text = source if whole_source else source[found[0] : found[1]]
            best = (found[2], text.strip())
    return best[1] if best else None


@dataclass(frozen=True)
class Rejection:
    """A finding the executor refused to put in the ledger."""

    url: str
    reason: RejectReason

    def payload(self, step_id: int) -> dict[str, Any]:
        """Trace payload; a personal profile is identified by its domain only, never its URL."""
        payload: dict[str, Any] = {"step_id": step_id, "reason": self.reason}
        if self.reason == "personal_profile":
            payload["domain"] = domain_of(self.url)
        else:
            payload["url"] = self.url
        return payload


def _accept_findings(
    findings: Iterable[Finding], sources: dict[str, list[str]], settings: Settings
) -> tuple[list[Finding], list[Rejection]]:
    """Split findings into accepted (snippet replaced by the source span) and rejected.

    ``sources`` maps a normalised URL the model saw to the texts it was shown for it. Profiles
    are checked first so a profile URL is never reported under another reason. A snippet
    shorter than ``snippet_min_chars`` may only match a source text that is itself that short,
    and then the whole source text becomes the stored snippet.
    """
    minimum = settings.snippet_min_chars
    accepted: list[Finding] = []
    rejected: list[Rejection] = []
    for finding in findings:
        url = normalize_url(finding.source_url)
        size = len(_normalize(finding.snippet)[0])
        texts = sources.get(url, [])
        short = size < minimum
        if short:
            texts = [t for t in texts if len(_normalize(t)[0]) < minimum]
        reason: RejectReason
        if is_personal_profile(finding.source_url, settings.personal_profile_patterns):
            reason = "personal_profile"
        elif url not in sources:
            reason = "unseen_url"
        elif not size:
            reason = "empty_snippet"
        elif short and not texts:
            reason = "snippet_too_short"
        else:
            span = _best_span(finding.snippet, texts, settings.snippet_match_ratio, short)
            if span:
                accepted.append(finding.model_copy(update={"snippet": span}))
                continue
            reason = "snippet_not_in_source"
        rejected.append(Rejection(finding.source_url.strip(), reason))
    return accepted, rejected


# --- ledger --------------------------------------------------------------------------------------


@dataclass
class EvidenceLedger:
    """All evidence gathered in a run; lives outside the prompts."""

    items: list[Evidence] = field(default_factory=list)

    def _duplicate(self, claim: str, url: str) -> bool:
        """True if the same claim from the same source is already in the ledger."""
        key = (claim.strip().lower(), normalize_url(url))
        return any(
            (i.claim.strip().lower(), normalize_url(i.source_url)) == key for i in self.items
        )

    def add(self, finding: Finding, step_id: int, topic: str = "") -> Evidence | None:
        """Add a finding as Evidence with the next id; skip exact duplicates."""
        if self._duplicate(finding.claim, finding.source_url):
            return None
        evidence = Evidence(
            id=f"E{len(self.items) + 1}",
            claim=finding.claim.strip(),
            source_url=finding.source_url.strip(),
            snippet=finding.snippet.strip()[:500],
            confidence=finding.confidence,
            step_id=step_id,
            topic=topic,
            volatility=finding.volatility,
        )
        self.items.append(evidence)
        return evidence

    def add_fact(self, fact: Fact, step_id: int) -> Evidence:
        """Add a remembered fact as Evidence; it keeps its original source URL for citations."""
        for item in self.items:
            if item.fact_id is not None and item.fact_id == fact.id:
                return item
        evidence = Evidence(
            id=f"E{len(self.items) + 1}",
            claim=fact.claim,
            source_url=fact.source_url,
            snippet=fact.snippet,
            confidence=fact.confidence,
            step_id=step_id,
            from_memory=True,
            fact_id=fact.id,
            topic=fact.topic,
            volatility=fact.volatility,
        )
        self.items.append(evidence)
        return evidence

    def by_id(self) -> dict[str, Evidence]:
        """Map evidence id -> Evidence."""
        return {e.id: e for e in self.items}


@dataclass
class _Entry:
    """One scratchpad entry: the observation plus the URLs it exposed."""

    observation: Observation
    urls: tuple[str, ...]


def summarize_observation(entry: _Entry) -> str:
    """One-line summary for older scratchpad entries."""
    obs = entry.observation
    args = json.dumps(obs.args, ensure_ascii=False)[:120]
    if not obs.ok:
        return f"{obs.tool}({args}) -> {obs.error}"[:300]
    if entry.urls:
        return f"{obs.tool}({args}) -> ok; URLs: " + ", ".join(entry.urls[:6])
    text = " ".join(_TAG_RE.sub("", obs.content).split())
    return f"{obs.tool}({args}) -> {text[:160]}"


def render_scratchpad(entries: list[_Entry], full: int = 2) -> str:
    """Older observations as one-liners, the last ``full`` in full."""
    if not entries:
        return "No observations yet. Start with a tool call."
    lines: list[str] = []
    cutoff = len(entries) - full
    for i, entry in enumerate(entries, 1):
        if i <= cutoff:
            lines.append(f"[{i}] (summary) {summarize_observation(entry)}")
        else:
            obs = entry.observation
            args = json.dumps(obs.args, ensure_ascii=False)
            lines.append(f"[{i}] {obs.tool}({args})\n{obs.content}")
    return "\n\n".join(lines)


def _messages(
    system: str, entries: list[_Entry], iteration: int, force_finish: bool, full: int
) -> list:
    """Build the per-iteration prompt from the compact scratchpad."""
    user = f"## Observations\n{render_scratchpad(entries, full)}\n\n"
    if force_finish:
        user += "No research turns left. Reply with a finish action now."
    else:
        user += f"Turn {iteration}. Reply with your next action."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _result_summary(obs: Observation, urls: tuple[str, ...]) -> str:
    """Short human-readable summary of a tool result for the trace."""
    if not obs.ok:
        return obs.error or "error"
    if obs.content.startswith(REPEAT_NOTE):
        return "repeat of an earlier call (not re-run)"
    if obs.tool == "web_search":
        return f"{len(urls)} results"
    text = " ".join(_TAG_RE.sub("", obs.content).split())
    return f"{len(text)} chars: {text[:140]}"


def execute_step(
    step: Step,
    *,
    task: Task,
    playbook: Playbook,
    llm: LLM,
    registry: ToolRegistry,
    budget: Budget,
    settings: Settings,
    ledger: EvidenceLedger,
    emit: EmitFn,
    approach: str | None = None,
) -> Generator[Event, None, StepResult]:
    """Run the ReAct loop for one step, yielding trace events; return the StepResult.

    ``approach`` is the critic's ``new_approach`` when the step is being retried.
    """
    purpose = f"{playbook.purpose_type}: {playbook.description}"
    if task.user_context:
        purpose += f" User context: {task.user_context}"
    retry_note = f"\nRetry: an earlier attempt failed. New approach: {approach}" if approach else ""
    system = load_prompt(
        "executor",
        question=step.question + retry_note,
        done_criteria=step.done_criteria or "a sourced answer to the question",
        purpose=purpose,
        tools=registry.describe(with_thought=True),
        max_iterations=settings.max_react_iterations,
        snippet_min_chars=settings.snippet_min_chars,
    )
    full = settings.scratchpad_full_observations
    entries: list[_Entry] = []
    sources: dict[str, list[str]] = {}  # normalised URL seen -> texts the model was shown for it
    finish: Action | None = None
    iterations = 0

    for iteration in range(1, settings.max_react_iterations + 1):
        if not budget.llm_available():
            break
        iterations = iteration
        try:
            action = llm.complete_json(
                _messages(system, entries, iteration, False, full),
                Action,
                fast=True,
                include_schema=False,
            )
        except LLMValidationError as exc:
            obs = Observation(tool="(invalid action)", ok=False, error=str(exc)[:300])
            entries.append(_Entry(obs, ()))
            yield emit("error", {"step_id": step.id, "message": "invalid action JSON"})
            continue
        yield emit("thought", {"step_id": step.id, "iteration": iteration, "text": action.thought})
        if action.type == "finish":
            finish = action
            break
        if not budget.tool_available():
            break
        yield emit("tool_call", {"step_id": step.id, "tool": action.tool, "args": action.args})
        run = registry.run(action.tool or "", action.args)
        if not run.repeated:
            budget.charge_tool()
        for url in run.urls:
            sources.setdefault(normalize_url(url), [])
        for url, text in run.sources:
            texts = sources.setdefault(normalize_url(url), [])
            if text not in texts:
                texts.append(text)
        entries.append(_Entry(run.observation, run.urls))
        yield emit(
            "tool_result",
            {
                "step_id": step.id,
                "tool": run.observation.tool,
                "ok": run.observation.ok,
                "repeated": run.repeated,
                "summary": _result_summary(run.observation, run.urls),
            },
        )

    # Forced finish ignores the wall clock (one call) but never eats the synthesis reserve.
    llm_room = budget.llm.usage.llm_calls < settings.max_llm_calls - SYNTHESIS_RESERVE
    if finish is None and entries and llm_room:
        try:
            forced = llm.complete_json(
                _messages(system, entries, 0, True, full), Action, fast=True, include_schema=False
            )
            yield emit(
                "thought", {"step_id": step.id, "iteration": "final", "text": forced.thought}
            )
            finish = forced if forced.type == "finish" else None
        except LLMValidationError:
            yield emit("error", {"step_id": step.id, "message": "invalid final action JSON"})

    accepted, rejected = _accept_findings(finish.findings if finish else [], sources, settings)
    for rejection in rejected:
        yield emit("evidence_rejected", rejection.payload(step.id))
    if rejected:
        reasons = dict(Counter(r.reason for r in rejected))
        logger.info("Step %d: rejected %d findings %s", step.id, len(rejected), reasons)
    evidence: list[Evidence] = []
    for finding in accepted:
        item = ledger.add(finding, step.id, topic=step.question[:160])
        if item:
            evidence.append(item)

    status = "done" if evidence else "no_findings"
    if budget.exhausted_reason:
        status = "budget_exhausted"
    return StepResult(
        step_id=step.id,
        status=status,
        evidence=evidence,
        observations=[e.observation for e in entries],
        iterations=iterations,
        sources_used=sorted(
            {u for e in entries for u in e.urls if e.observation.tool == "fetch_page"}
            | {e.source_url for e in evidence}
        ),
        snippets_rejected=sum(r.reason in SNIPPET_REASONS for r in rejected),
    )
