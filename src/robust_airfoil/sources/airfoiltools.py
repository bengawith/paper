from __future__ import annotations

import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import httpx

from robust_airfoil.hashing import sha256_bytes


@dataclass(frozen=True)
class ProbeResult:
    status: str
    allowed: bool
    http_status: int | None
    reason: str
    body_sha256: str | None = None


@dataclass(frozen=True)
class RawFetchResult:
    polar_key: str
    status: str
    http_status: int | None
    body: bytes
    body_sha256: str | None
    reason: str | None = None


@dataclass
class FetchSummary:
    requested: int
    succeeded: int
    stopped_early: bool
    results: list[RawFetchResult] = field(default_factory=list)


class AirfoilToolsClient:
    def __init__(
        self,
        csv_url_template: str,
        robots_url: str,
        canary_key: str,
        contact_env: str = "AIRFOILTOOLS_CONTACT",
        minimum_delay_seconds: float = 2.0,
        timeout: float = 30.0,
        stop_on_status: Sequence[int] = (401, 403, 429),
        max_attempts: int = 3,
        output_dir: Path | None = None,
    ) -> None:
        self.csv_url_template = csv_url_template
        self.robots_url = robots_url
        self.canary_key = canary_key
        self.contact_env = contact_env
        self.minimum_delay_seconds = minimum_delay_seconds
        self.timeout = timeout
        self.stop_on_status = set(stop_on_status)
        self.max_attempts = max_attempts
        self.output_dir = output_dir
        self._last_request = 0.0

    def _headers(self) -> dict[str, str]:
        contact = os.getenv(self.contact_env, "").strip()
        return {"User-Agent": f"robust-airfoil-v2/2.0 contact={contact}"}

    def _wait(self) -> None:
        remaining = self.minimum_delay_seconds - (time.monotonic() - self._last_request)
        if remaining > 0:
            time.sleep(remaining)

    @staticmethod
    def _looks_like_polar(body: bytes) -> bool:
        text = body[:8192].decode("utf-8", errors="ignore").lower()
        return "alpha" in text and "cl" in text and "cd" in text and "<html" not in text

    def probe_canary(self) -> ProbeResult:
        if not os.getenv(self.contact_env, "").strip():
            return ProbeResult("skipped", False, None, f"missing {self.contact_env}")
        result = self.fetch_polar(self.canary_key)
        return ProbeResult(
            status=result.status,
            allowed=result.status == "ok",
            http_status=result.http_status,
            reason=result.reason or "validated polar body",
            body_sha256=result.body_sha256,
        )

    def fetch_polar(self, polar_key: str) -> RawFetchResult:
        if not os.getenv(self.contact_env, "").strip():
            return RawFetchResult(polar_key, "forbidden", None, b"", None, f"missing {self.contact_env}")
        url = self.csv_url_template.format(polar_key=polar_key)
        for attempt in range(1, self.max_attempts + 1):
            self._wait()
            try:
                response = httpx.get(url, headers=self._headers(), timeout=self.timeout, follow_redirects=True)
                self._last_request = time.monotonic()
            except httpx.HTTPError as exc:
                if attempt == self.max_attempts:
                    return RawFetchResult(polar_key, "error", None, b"", None, repr(exc))
                time.sleep(2 ** (attempt - 1))
                continue
            if response.status_code in self.stop_on_status:
                return RawFetchResult(polar_key, "forbidden", response.status_code, response.content, sha256_bytes(response.content), "stop status")
            if response.status_code >= 500 and attempt < self.max_attempts:
                time.sleep(2 ** (attempt - 1))
                continue
            if response.status_code != 200:
                return RawFetchResult(polar_key, "error", response.status_code, response.content, sha256_bytes(response.content), "non-200")
            if not self._looks_like_polar(response.content):
                return RawFetchResult(polar_key, "invalid_body", 200, response.content, sha256_bytes(response.content), "body is not a polar table")
            if self.output_dir:
                self.output_dir.mkdir(parents=True, exist_ok=True)
                (self.output_dir / f"{polar_key}.csv").write_bytes(response.content)
            return RawFetchResult(polar_key, "ok", 200, response.content, sha256_bytes(response.content))
        raise AssertionError("unreachable")

    def fetch_many(self, polar_keys: Sequence[str], limit: int) -> FetchSummary:
        results: list[RawFetchResult] = []
        stopped = False
        for key in list(polar_keys)[:limit]:
            result = self.fetch_polar(key)
            results.append(result)
            if result.http_status in self.stop_on_status or result.status == "forbidden":
                stopped = True
                break
        return FetchSummary(len(list(polar_keys)[:limit]), sum(r.status == "ok" for r in results), stopped, results)
