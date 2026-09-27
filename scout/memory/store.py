"""SQLite memory store (docs/SPEC.md §3 learning mechanisms, §4 data model).

Tables (created on startup): runs, facts, sources, lessons, feedback. Beyond the SPEC columns,
facts carry ``volatility`` (stable | news, for time-to-live) and ``snippet`` (so remembered
evidence still passes the verifier's numeric check), and lessons carry ``uses``.
"""

from __future__ import annotations

import json
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from scout.schemas import Evidence, Fact, Lesson
from scout.textutil import keywords, overlap

_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id TEXT PRIMARY KEY, goal TEXT, purpose_type TEXT, targets TEXT,
    started_at TEXT, finished_at TEXT, metrics_json TEXT
);
CREATE TABLE IF NOT EXISTS facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT, entity TEXT, topic TEXT, claim TEXT, source_url TEXT,
    captured_at TEXT, confidence REAL, run_id TEXT, volatility TEXT DEFAULT 'stable',
    snippet TEXT DEFAULT ''
);
CREATE INDEX IF NOT EXISTS facts_entity ON facts(entity);
CREATE TABLE IF NOT EXISTS sources (
    domain TEXT PRIMARY KEY, useful INTEGER DEFAULT 0, useless INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT, purpose_type TEXT, text TEXT,
    votes_up INTEGER DEFAULT 0, votes_down INTEGER DEFAULT 0,
    created_at TEXT, last_used_at TEXT, uses INTEGER DEFAULT 0
);
CREATE TABLE IF NOT EXISTS feedback (
    id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, section TEXT, rating TEXT, created_at TEXT
);
"""

_ENTITY_SUFFIXES = re.compile(
    r"\b(private limited|pvt\.? ltd\.?|pvt|ltd\.?|limited|inc\.?|incorporated|"
    r"corporation|corp\.?|llc|plc|co\.)$"
)


def now_utc() -> datetime:
    """Current UTC time."""
    return datetime.now(UTC)


def domain_of(url: str) -> str:
    """Lowercase host without a leading ``www.``."""
    host = (urlsplit(url.strip()).hostname or "").lower()
    return host[4:] if host.startswith("www.") else host


def normalize_entity(name: str) -> str:
    """Canonical entity key: the domain for URLs/domains, else a lowercase name without suffixes.

    "Zoho Corporation" -> "zoho"; "Freshworks Inc." -> "freshworks";
    "https://www.zoho.com/about" -> "zoho.com"; "zoho.com" -> "zoho.com".
    """
    text = name.strip().lower()
    if "://" in text:
        return domain_of(text)
    if re.fullmatch(r"[a-z0-9-]+(\.[a-z0-9-]+)+", text):
        return text[4:] if text.startswith("www.") else text
    text = re.sub(r"[^\w\s.&-]", " ", text)
    text = " ".join(text.split())
    previous = None
    while previous != text:
        previous = text
        text = _ENTITY_SUFFIXES.sub("", text).strip(" ,.")
    return text


def score(good: int, bad: int) -> float:
    """Laplace-smoothed reliability: (good + 1) / (good + bad + 2); 0.5 when unseen."""
    return (good + 1) / (good + bad + 2)


def _iso(dt: datetime) -> str:
    """Timestamp as ISO string."""
    return dt.isoformat()


def _dt(text: str | None) -> datetime | None:
    """Parse an ISO timestamp."""
    return datetime.fromisoformat(text) if text else None


class MemoryStore:
    """Repository over the SQLite database; one connection per instance."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(_SCHEMA)

    def close(self) -> None:
        """Close the connection."""
        self.conn.close()

    # --- runs -----------------------------------------------------------------------------

    def save_run(
        self,
        run_id: str,
        goal: str,
        purpose_type: str,
        targets: list[str],
        started_at: datetime,
        finished_at: datetime,
        metrics: dict[str, Any],
    ) -> None:
        """Insert or replace a run row."""
        self.conn.execute(
            "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                goal,
                purpose_type,
                json.dumps(targets),
                _iso(started_at),
                _iso(finished_at),
                json.dumps(metrics),
            ),
        )
        self.conn.commit()

    def runs(self) -> list[dict[str, Any]]:
        """All runs, oldest first, with metrics decoded."""
        rows = self.conn.execute("SELECT * FROM runs ORDER BY started_at").fetchall()
        out = []
        for r in rows:
            item = dict(r)
            item["targets"] = json.loads(item["targets"] or "[]")
            item["metrics"] = json.loads(item.pop("metrics_json") or "{}")
            out.append(item)
        return out

    def run(self, run_id: str) -> dict[str, Any] | None:
        """One run by id."""
        return next((r for r in self.runs() if r["id"] == run_id), None)

    # --- facts ----------------------------------------------------------------------------

    def add_facts(
        self, entity: str, evidence: Iterable[Evidence], run_id: str, now: datetime | None = None
    ) -> int:
        """Persist evidence as facts; an identical claim+source refreshes its timestamp."""
        now = now or now_utc()
        added = 0
        for e in evidence:
            existing = self.conn.execute(
                "SELECT id FROM facts WHERE entity=? AND claim=? AND source_url=?",
                (entity, e.claim, e.source_url),
            ).fetchone()
            if existing:
                self.conn.execute(
                    "UPDATE facts SET captured_at=?, run_id=? WHERE id=?",
                    (_iso(now), run_id, existing["id"]),
                )
                continue
            self.conn.execute(
                "INSERT INTO facts (entity, topic, claim, source_url, captured_at, confidence,"
                " run_id, volatility, snippet) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    entity,
                    e.topic,
                    e.claim,
                    e.source_url,
                    _iso(now),
                    e.confidence,
                    run_id,
                    e.volatility,
                    e.snippet,
                ),
            )
            added += 1
        self.conn.commit()
        return added

    def facts(self, entity: str) -> list[Fact]:
        """All facts for an entity, newest first."""
        rows = self.conn.execute(
            "SELECT * FROM facts WHERE entity=? ORDER BY captured_at DESC, id DESC", (entity,)
        ).fetchall()
        return [
            Fact(
                id=r["id"],
                entity=r["entity"],
                topic=r["topic"] or "",
                claim=r["claim"],
                source_url=r["source_url"],
                snippet=r["snippet"] or "",
                captured_at=_dt(r["captured_at"]) or now_utc(),
                confidence=r["confidence"],
                run_id=r["run_id"],
                volatility=r["volatility"] or "stable",
            )
            for r in rows
        ]

    @staticmethod
    def is_fresh(fact: Fact, now: datetime, stable_days: int, news_days: int) -> bool:
        """Time-to-live check: stable facts live ``stable_days``, news ``news_days``."""
        ttl = news_days if fact.volatility == "news" else stable_days
        return now - fact.captured_at <= timedelta(days=ttl)

    def recall(
        self,
        entity: str,
        stable_days: int,
        news_days: int,
        max_facts: int,
        max_chars: int,
        now: datetime | None = None,
    ) -> tuple[list[Fact], list[Fact]]:
        """(fresh, stale) facts for an entity; fresh capped by count and total characters."""
        now = now or now_utc()
        fresh: list[Fact] = []
        stale: list[Fact] = []
        used = 0
        for fact in self.facts(entity):
            if not self.is_fresh(fact, now, stable_days, news_days):
                stale.append(fact)
                continue
            cost = len(fact.claim) + len(fact.source_url) + 20
            if len(fresh) < max_facts and used + cost <= max_chars:
                fresh.append(fact)
                used += cost
        return fresh, stale

    def search_facts(
        self, entity: str, topic: str, facts: list[Fact], limit: int = 8
    ) -> list[Fact]:
        """Rank ``facts`` by keyword overlap with ``topic`` (for the recall_memory tool)."""
        if not topic.strip():
            return facts[:limit]
        wanted = keywords(topic)
        scored = [(len(wanted & keywords(f.claim + " " + f.topic)), f) for f in facts]
        return [f for s, f in sorted(scored, key=lambda x: -x[0]) if s > 0][:limit]

    # --- sources --------------------------------------------------------------------------

    def rate_source(self, url_or_domain: str, useful: bool, weight: int = 1) -> None:
        """Record a useful/useless rating for a domain."""
        domain = domain_of(url_or_domain) if "://" in url_or_domain else url_or_domain.lower()
        if not domain:
            return
        column = "useful" if useful else "useless"
        self.conn.execute(
            f"INSERT INTO sources (domain, {column}) VALUES (?, ?) "  # noqa: S608 - fixed names
            f"ON CONFLICT(domain) DO UPDATE SET {column} = {column} + ?",
            (domain, weight, weight),
        )
        self.conn.commit()

    def source_scores(self) -> dict[str, float]:
        """domain -> reliability score."""
        rows = self.conn.execute("SELECT domain, useful, useless FROM sources").fetchall()
        return {r["domain"]: score(r["useful"], r["useless"]) for r in rows}

    def sources(self) -> list[dict[str, Any]]:
        """All sources with counts and score, best first."""
        rows = self.conn.execute("SELECT domain, useful, useless FROM sources").fetchall()
        items = [{**dict(r), "score": score(r["useful"], r["useless"])} for r in rows]
        return sorted(items, key=lambda x: (-x["score"], -(x["useful"] + x["useless"])))

    # --- lessons --------------------------------------------------------------------------

    def _lesson(self, row: sqlite3.Row) -> Lesson:
        """Row -> Lesson."""
        return Lesson(
            id=row["id"],
            purpose_type=row["purpose_type"],
            text=row["text"],
            votes_up=row["votes_up"],
            votes_down=row["votes_down"],
            created_at=_dt(row["created_at"]) or now_utc(),
            last_used_at=_dt(row["last_used_at"]),
        )

    def lessons(self, purpose_type: str | None = None) -> list[Lesson]:
        """Lessons (optionally for one purpose), ranked by vote score then recency."""
        query = "SELECT * FROM lessons"
        args: tuple[str, ...] = ()
        if purpose_type:
            query += " WHERE purpose_type=?"
            args = (purpose_type,)
        rows = self.conn.execute(query, args).fetchall()
        items = [self._lesson(r) for r in rows]
        return sorted(
            items, key=lambda x: (-score(x.votes_up, x.votes_down), -x.created_at.timestamp())
        )

    def lesson_uses(self) -> dict[int, int]:
        """lesson id -> times injected."""
        return {r["id"]: r["uses"] for r in self.conn.execute("SELECT id, uses FROM lessons")}

    def top_lessons(self, purpose_type: str, n: int) -> list[Lesson]:
        """Top ``n`` lessons for a purpose (vote score, recency tie-break)."""
        return self.lessons(purpose_type)[:n]

    def mark_used(self, lesson_ids: Iterable[int], now: datetime | None = None) -> None:
        """Record that lessons were injected into a plan."""
        stamp = _iso(now or now_utc())
        for lid in lesson_ids:
            self.conn.execute(
                "UPDATE lessons SET uses = uses + 1, last_used_at=? WHERE id=?", (stamp, lid)
            )
        self.conn.commit()

    def add_or_merge_lesson(
        self, purpose_type: str, text: str, merge_overlap: float, now: datetime | None = None
    ) -> tuple[Lesson, bool]:
        """Insert a lesson, or merge into a near-duplicate (adding an up-vote). Returns merged?"""
        for existing in self.lessons(purpose_type):
            if overlap(text, existing.text) >= merge_overlap:
                assert existing.id is not None
                self.vote(existing.id, True)
                merged = self.lesson(existing.id)
                assert merged is not None
                return merged, True
        cur = self.conn.execute(
            "INSERT INTO lessons (purpose_type, text, created_at) VALUES (?, ?, ?)",
            (purpose_type, text.strip(), _iso(now or now_utc())),
        )
        self.conn.commit()
        lesson = self.lesson(int(cur.lastrowid or 0))
        assert lesson is not None
        return lesson, False

    def lesson(self, lesson_id: int) -> Lesson | None:
        """One lesson by id."""
        row = self.conn.execute("SELECT * FROM lessons WHERE id=?", (lesson_id,)).fetchone()
        return self._lesson(row) if row else None

    def vote(self, lesson_id: int, helpful: bool) -> None:
        """Up- or down-vote a lesson."""
        column = "votes_up" if helpful else "votes_down"
        self.conn.execute(
            f"UPDATE lessons SET {column} = {column} + 1 WHERE id=?",  # noqa: S608 - fixed names
            (lesson_id,),
        )
        self.conn.commit()

    # --- feedback -------------------------------------------------------------------------

    def add_feedback(self, run_id: str, section: str, rating: str) -> None:
        """Store one thumbs-up/down."""
        self.conn.execute(
            "INSERT INTO feedback (run_id, section, rating, created_at) VALUES (?, ?, ?, ?)",
            (run_id, section, rating, _iso(now_utc())),
        )
        self.conn.commit()

    def feedback(self) -> list[dict[str, Any]]:
        """All feedback rows."""
        return [dict(r) for r in self.conn.execute("SELECT * FROM feedback ORDER BY id")]
