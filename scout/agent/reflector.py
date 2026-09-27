"""Reflector stage: run summary -> 1–3 lessons + votes on injected lessons (docs/SPEC.md §3)."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from scout.llm import LLM
from scout.prompts import load_prompt
from scout.safety import violates_contact_policy
from scout.schemas import Lesson

MAX_LESSON_WORDS = 25


class LessonVote(BaseModel):
    """Helpful / unhelpful verdict on an injected lesson."""

    lesson_id: int
    helpful: bool


class Reflection(BaseModel):
    """Reflector output, validated: 1–3 short lessons."""

    lessons: list[str] = Field(min_length=1, max_length=3)
    votes: list[LessonVote] = Field(default_factory=list)

    @field_validator("lessons")
    @classmethod
    def _short_lessons(cls, lessons: list[str]) -> list[str]:
        """Each lesson must be non-empty and under the word limit."""
        cleaned = [" ".join(t.split()) for t in lessons]
        for text in cleaned:
            words = len(text.split())
            if not text or words >= MAX_LESSON_WORDS:
                raise ValueError(f"lesson must be 1-{MAX_LESSON_WORDS - 1} words, got {words}")
        return cleaned


def reflect(
    llm: LLM, purpose: str, summary: str, injected: list[Lesson], max_lessons: int
) -> Reflection:
    """One smart-tier call; votes on lessons that were not injected are dropped."""
    prompt = load_prompt(
        "reflector",
        summary=summary,
        injected="\n".join(f"- [id {lesson.id}] {lesson.text}" for lesson in injected) or "none",
        purpose=purpose,
        max_lessons=max_lessons,
        max_words=MAX_LESSON_WORDS,
    )
    reflection = llm.complete_json(
        [{"role": "system", "content": prompt}, {"role": "user", "content": "Reflect."}],
        Reflection,
        include_schema=False,
    )
    allowed = {lesson.id for lesson in injected}
    votes = [v for v in reflection.votes if v.lesson_id in allowed]
    lessons = [t for t in reflection.lessons if not violates_contact_policy(t)][:max_lessons]
    return reflection.model_copy(update={"lessons": lessons, "votes": votes})
