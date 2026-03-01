from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Callable, Optional
from uuid import uuid4

TERMINAL_STATES = {"succeeded", "failed"}


@dataclass
class AnalyzeJobRecord:
    job_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    result: Optional[dict[str, Any]] = None
    error: Optional[str] = None


class AnalyzeJobManager:
    """In-memory async job manager for demo deployments."""

    def __init__(self, max_workers: int = 2, ttl_seconds: int = 1800):
        self._executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analyze-job")
        self._ttl_seconds = ttl_seconds
        self._jobs: dict[str, AnalyzeJobRecord] = {}
        self._lock = Lock()

    def submit(self, fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> str:
        self._cleanup_expired()
        job_id = uuid4().hex
        now = datetime.now(timezone.utc)
        record = AnalyzeJobRecord(
            job_id=job_id,
            status="queued",
            created_at=now,
            updated_at=now,
        )

        with self._lock:
            self._jobs[job_id] = record

        self._executor.submit(self._run_job, job_id, fn, *args, **kwargs)
        return job_id

    def _run_job(self, job_id: str, fn: Callable[..., dict[str, Any]], *args: Any, **kwargs: Any) -> None:
        self._update(job_id, status="running")
        try:
            result = fn(*args, **kwargs)
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result.get("error")))
            self._update(job_id, status="succeeded", result=result, error=None)
        except Exception as exc:
            self._update(job_id, status="failed", result=None, error=str(exc))

    def get(self, job_id: str) -> Optional[AnalyzeJobRecord]:
        self._cleanup_expired()
        with self._lock:
            return self._jobs.get(job_id)

    def _update(
        self,
        job_id: str,
        *,
        status: Optional[str] = None,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        with self._lock:
            record = self._jobs.get(job_id)
            if not record:
                return
            if status is not None:
                record.status = status
            if result is not None:
                record.result = result
            if error is not None or status == "failed":
                record.error = error
            record.updated_at = datetime.now(timezone.utc)

    def _cleanup_expired(self) -> None:
        now = datetime.now(timezone.utc)
        with self._lock:
            expired = [
                job_id
                for job_id, record in self._jobs.items()
                if record.status in TERMINAL_STATES
                and (now - record.updated_at).total_seconds() > self._ttl_seconds
            ]
            for job_id in expired:
                self._jobs.pop(job_id, None)
