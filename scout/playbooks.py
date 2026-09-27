"""Purpose playbooks (docs/SPEC.md §3): research hints, output sections and a scoring rubric.

The planner writes its own plan; a playbook only guides it.
"""

from __future__ import annotations

from dataclasses import dataclass

from scout.schemas import PurposeType


@dataclass(frozen=True)
class Playbook:
    """Guidance for one purpose type."""

    purpose_type: PurposeType
    description: str
    research_hints: tuple[str, ...]
    sections: tuple[str, ...]
    score_name: str | None
    score_rubric: str | None


PLAYBOOKS: dict[str, Playbook] = {
    "sales_prospect": Playbook(
        purpose_type="sales_prospect",
        description="Qualify a company as a sales prospect for the user's product.",
        research_hints=(
            "What they do",
            "Size and locations",
            "Hiring volume and campus activity",
            "Recent news or funding",
            "Tools they use",
            "Likely pain points",
            "Decision-maker roles",
        ),
        sections=(
            "Snapshot",
            "Buying signals",
            "Pain points",
            "Who to approach (roles only)",
            "Pitch angle",
            "Risks",
        ),
        score_name="Fit score",
        score_rubric="Fit score 0–100 with reasons",
    ),
    "competitor": Playbook(
        purpose_type="competitor",
        description="Assess a company as a competitor.",
        research_hints=(
            "Positioning",
            "Pricing",
            "Customers",
            "Recent launches",
            "Hiring signals",
        ),
        sections=(
            "Positioning",
            "Pricing",
            "Recent moves",
            "Strengths and weaknesses",
            "Threat assessment",
        ),
        score_name="Threat level",
        score_rubric="Threat level low, medium or high",
    ),
    "interview_prep": Playbook(
        purpose_type="interview_prep",
        description="Prepare a candidate for an interview at the company.",
        research_hints=(
            "Products",
            "Business model",
            "Tech stack",
            "Engineering culture",
            "Recent news",
            "Interview process signals",
        ),
        sections=(
            "Company in 60 seconds",
            "Tech stack",
            "News worth mentioning",
            "Likely topics",
            "Smart questions to ask",
        ),
        score_name="Readiness checklist",
        score_rubric="Readiness checklist",
    ),
    "general": Playbook(
        purpose_type="general",
        description="General company research; the planner decides what matters.",
        research_hints=("Planner decides",),
        sections=("Summary", "Key findings", "Open questions"),
        score_name=None,
        score_rubric=None,
    ),
}


def get_playbook(purpose_type: str) -> Playbook:
    """Return the playbook for ``purpose_type``, falling back to ``general``."""
    return PLAYBOOKS.get(purpose_type, PLAYBOOKS["general"])


def playbook_brief(playbook: Playbook) -> str:
    """Render a playbook as compact prompt text."""
    lines = [
        f"Purpose: {playbook.purpose_type} — {playbook.description}",
        "Research hints: " + "; ".join(playbook.research_hints),
        "Output sections: " + "; ".join(playbook.sections),
    ]
    if playbook.score_rubric:
        lines.append(f"Score: {playbook.score_rubric}")
    return "\n".join(lines)
