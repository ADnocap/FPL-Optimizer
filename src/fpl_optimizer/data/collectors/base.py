"""Base classes for data collectors: rate limiting, retry logic, caching."""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path("data")


class RateLimiter:
    """Token-bucket rate limiter.

    Args:
        calls_per_second: Maximum sustained request rate.
    """

    def __init__(self, calls_per_second: float = 1.0) -> None:
        self._min_interval = 1.0 / calls_per_second
        self._last_call: float = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        """Block until the next request is allowed (thread-safe)."""
        with self._lock:
            now = time.monotonic()
            elapsed = now - self._last_call
            if elapsed < self._min_interval:
                time.sleep(self._min_interval - elapsed)
            self._last_call = time.monotonic()


class BaseCollector(ABC):
    """Abstract base for all data collectors.

    Provides HTTP retry logic, caching checks, and rate limiting.
    Subclasses implement ``collect_season`` and ``collect_all``.
    """

    def __init__(
        self,
        data_dir: Path = DEFAULT_DATA_DIR,
        rate_limiter: RateLimiter | None = None,
    ) -> None:
        self.data_dir = data_dir
        self.rate_limiter = rate_limiter or RateLimiter()

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def collect_season(self, season: str) -> bool:
        """Collect data for a single season. Returns True on success."""

    @abstractmethod
    def collect_all(self) -> dict[str, bool]:
        """Collect data for all applicable seasons. Returns {season: success}."""

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_cached(self, path: Path) -> bool:
        """Return True if *path* exists and is non-empty."""
        return path.exists() and path.stat().st_size > 0

    def _request_with_retry(
        self,
        url: str,
        *,
        max_retries: int = 3,
        timeout: int = 30,
    ) -> requests.Response:
        """GET *url* with exponential back-off.

        Raises ``requests.RequestException`` after exhausting retries.
        """
        last_exc: Exception | None = None
        for attempt in range(max_retries):
            self.rate_limiter.wait()
            try:
                resp = requests.get(url, timeout=timeout)
                resp.raise_for_status()
                return resp
            except requests.RequestException as exc:
                last_exc = exc
                wait = 2 ** attempt
                logger.warning(
                    "Attempt %d/%d failed for %s: %s — retrying in %ds",
                    attempt + 1,
                    max_retries,
                    url,
                    exc,
                    wait,
                )
                time.sleep(wait)
        raise last_exc  # type: ignore[misc]
