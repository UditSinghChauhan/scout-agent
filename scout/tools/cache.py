"""Small JSON disk cache with a time-to-live (Phase 2, A3).

Search results are keyed by query, page text by URL. Stored under ``data/cache`` (gitignored).
Failures to read or write the cache are logged and ignored: it is an optimisation only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DiskCache:
    """``get``/``set`` JSON values by (namespace, key), expiring after ``ttl_s`` seconds."""

    def __init__(
        self, directory: Path, ttl_s: float, clock: Callable[[], float] = time.time
    ) -> None:
        self.directory = directory
        self.ttl_s = ttl_s
        self._clock = clock

    def _path(self, namespace: str, key: str) -> Path:
        """File path for a cache entry."""
        digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.directory / namespace / f"{digest}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        """Return the cached value, or None if missing, expired or unreadable."""
        path = self._path(namespace, key)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if self._clock() - float(entry.get("saved_at", 0)) > self.ttl_s:
            return None
        return entry.get("value")

    def set(self, namespace: str, key: str, value: Any) -> None:
        """Store ``value``; errors are logged, never raised."""
        path = self._path(namespace, key)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = {"key": key, "saved_at": self._clock(), "value": value}
            path.write_text(json.dumps(payload), encoding="utf-8")
        except OSError as exc:
            logger.info("Cache write failed: %s", exc)
