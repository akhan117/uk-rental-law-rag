"""Protection for a public demo running on a free-tier LLM quota.

A public URL means anyone can spend the quota, and once it's gone the demo is
down for everyone until it resets. Three cheap layers:

- per-client rate limit (sliding window), so one visitor can't drain it;
- a daily cap on LLM-backed queries, so the demo degrades to a clear "try
  tomorrow" instead of a wall of provider 429s;
- an answer cache, because demo traffic is dominated by the same few example
  questions, and a repeat should cost neither tokens nor seconds.

All in-process and in-memory: a single free container has no shared store to
put this in, and losing it on restart is harmless. All off by default (local
use), switched on by environment variables in the Dockerfile.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict, deque


class RateLimiter:
    def __init__(self, per_minute: int) -> None:
        self.per_minute = per_minute
        self._hits: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def allow(self, client: str, now: float | None = None) -> bool:
        if self.per_minute <= 0:
            return True
        now = time.monotonic() if now is None else now
        with self._lock:
            q = self._hits.setdefault(client, deque())
            while q and now - q[0] >= 60.0:
                q.popleft()
            if len(q) >= self.per_minute:
                return False
            q.append(now)
            # Bound memory: drop clients idle for a minute.
            if len(self._hits) > 10_000:
                for k in [k for k, v in self._hits.items() if not v or now - v[-1] >= 60.0]:
                    del self._hits[k]
            return True


class DailyCap:
    def __init__(self, limit: int) -> None:
        self.limit = limit
        self._day = ""
        self._count = 0
        self._lock = threading.Lock()

    def take(self, today: str | None = None) -> bool:
        if self.limit <= 0:
            return True
        today = today or time.strftime("%Y-%m-%d", time.gmtime())
        with self._lock:
            if today != self._day:
                self._day, self._count = today, 0
            if self._count >= self.limit:
                return False
            self._count += 1
            return True

    @property
    def used(self) -> int:
        return self._count


class AnswerCache:
    """LRU with TTL, keyed on everything that can change the answer."""

    def __init__(self, size: int, ttl_s: float) -> None:
        self.size = size
        self.ttl_s = ttl_s
        self._data: OrderedDict[str, tuple[float, object]] = OrderedDict()
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    @staticmethod
    def key(*parts) -> str:
        raw = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha256(raw.encode()).hexdigest()

    @staticmethod
    def normalize_question(q: str) -> str:
        return " ".join(q.lower().split()).rstrip("?!. ")

    def get(self, key: str):
        if self.size <= 0:
            return None
        with self._lock:
            item = self._data.get(key)
            if item is None or time.monotonic() - item[0] > self.ttl_s:
                if item is not None:
                    del self._data[key]
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return item[1]

    def put(self, key: str, value) -> None:
        if self.size <= 0:
            return
        with self._lock:
            self._data[key] = (time.monotonic(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.size:
                self._data.popitem(last=False)


def _env_num(key: str, default: float) -> float:
    raw = os.environ.get(key, "").strip()
    return float(raw) if raw else default


def from_env() -> tuple[RateLimiter, DailyCap, AnswerCache]:
    return (
        RateLimiter(int(_env_num("RATE_LIMIT_PER_MINUTE", 0))),
        DailyCap(int(_env_num("DAILY_QUERY_LIMIT", 0))),
        AnswerCache(int(_env_num("ANSWER_CACHE_SIZE", 256)),
                    _env_num("ANSWER_CACHE_TTL_S", 24 * 3600)),
    )
