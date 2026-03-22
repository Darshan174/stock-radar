from __future__ import annotations

import os
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional
from uuid import uuid4

from supabase import Client, create_client

logger = logging.getLogger(__name__)

TABLE = "analyze_jobs"

# How long a worker lock is valid before the job is considered abandoned.
LOCK_TTL_SECONDS = int(os.getenv("JOB_LOCK_TTL_SEC", "600"))  # 10 min default

# How often the poller checks for claimable work.
POLL_INTERVAL_SECONDS = int(os.getenv("JOB_POLL_INTERVAL_SEC", "5"))

# Maximum attempts before a job is moved to the dead-letter queue.
MAX_RETRIES = int(os.getenv("JOB_MAX_RETRIES", "3"))


def _supabase_client() -> Client:
    url = os.getenv("SUPABASE_URL", "")
    key = os.getenv("SUPABASE_KEY", "")
    if not url or not key:
        raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set for job persistence")
    return create_client(url, key)


class AnalyzeJobManager:
    """Supabase-backed job manager with distributed claim, crash recovery,
    retry budgets, and dead-letter queue.

    * ``submit()`` inserts a ``queued`` row — any replica can pick it up.
    * A background poller claims rows atomically via
      ``UPDATE … SET status='running', worker_id=… WHERE status='queued'``
      and hands them to a local ``ThreadPoolExecutor``.
    * Stale ``running`` rows (``locked_until < now``) are reclaimed
      automatically, so a crashed worker doesn't leave orphans.
    * Jobs that fail more than ``MAX_RETRIES`` times are moved to
      ``dead_letter`` status and stop being retried.
    * Can run standalone as a dedicated worker process via ``run_worker()``.
    """

    def __init__(
        self,
        *,
        max_workers: int = 4,
        client: Optional[Client] = None,
        run_fn: Optional[Callable[..., dict[str, Any]]] = None,
    ):
        self._client = client or _supabase_client()
        self._worker_id = uuid4().hex[:12]
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="analyze-job",
        )
        self._run_fn = run_fn  # set via set_run_fn() from app.py
        self._max_workers = max_workers
        self._active_count = 0
        self._active_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._poller_thread: Optional[threading.Thread] = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    def set_run_fn(self, fn: Callable[..., dict[str, Any]]) -> None:
        """Register the analysis function (avoids circular import at init)."""
        self._run_fn = fn

    def start(self) -> None:
        """Start the background poller thread."""
        if self._poller_thread and self._poller_thread.is_alive():
            return
        self._stop_event.clear()
        self._poller_thread = threading.Thread(
            target=self._poll_loop, daemon=True, name="job-poller",
        )
        self._poller_thread.start()
        logger.info(
            "Job poller started (worker=%s, max_workers=%d, poll=%ds, lock_ttl=%ds, max_retries=%d)",
            self._worker_id, self._max_workers, POLL_INTERVAL_SECONDS, LOCK_TTL_SECONDS, MAX_RETRIES,
        )

    def stop(self) -> None:
        """Signal the poller to stop and wait for in-flight jobs."""
        self._stop_event.set()
        self._executor.shutdown(wait=True)
        if self._poller_thread:
            self._poller_thread.join(timeout=10)
        logger.info("Job manager stopped (worker=%s)", self._worker_id)

    def run_worker(self) -> None:
        """Block the current thread as a dedicated worker process.

        Usage from CLI::

            python -m backend.jobs

        This lets you run workers separately from the API process for
        better isolation and independent scaling.
        """
        self.start()
        logger.info("Standalone worker running (Ctrl+C to stop)")
        try:
            self._stop_event.wait()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()

    # ── public API ───────────────────────────────────────────────────────

    def submit(self, symbol: str, mode: str, period: str) -> str:
        """Enqueue a new analysis job.  Returns the job_id."""
        job_id = uuid4().hex
        now = datetime.now(timezone.utc).isoformat()

        self._client.table(TABLE).insert({
            "job_id": job_id,
            "status": "queued",
            "symbol": symbol,
            "mode": mode,
            "period": period,
            "attempt": 0,
            "created_at": now,
            "updated_at": now,
        }).execute()

        return job_id

    def get(self, job_id: str) -> Optional[dict[str, Any]]:
        resp = (
            self._client.table(TABLE)
            .select("job_id, status, result, error, attempt, created_at, updated_at")
            .eq("job_id", job_id)
            .maybe_single()
            .execute()
        )
        return resp.data

    def list_dlq(self, limit: int = 50) -> list[dict[str, Any]]:
        """List jobs in the dead-letter queue for manual inspection."""
        resp = (
            self._client.table(TABLE)
            .select("job_id, symbol, mode, error, attempt, created_at, updated_at")
            .eq("status", "dead_letter")
            .order("updated_at", desc=True)
            .limit(limit)
            .execute()
        )
        return resp.data or []

    def retry_dlq_job(self, job_id: str) -> bool:
        """Re-queue a dead-letter job (resets attempt counter)."""
        resp = (
            self._client.table(TABLE)
            .update({
                "status": "queued",
                "attempt": 0,
                "error": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            })
            .eq("job_id", job_id)
            .eq("status", "dead_letter")
            .execute()
        )
        return bool(resp.data)

    # ── polling / claim ──────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                if self._has_capacity():
                    job = self._try_claim()
                    if job:
                        self._dispatch(job)
                        continue  # check again immediately
            except Exception:
                logger.exception("Poller iteration failed")

            self._stop_event.wait(timeout=POLL_INTERVAL_SECONDS)

    def _has_capacity(self) -> bool:
        with self._active_lock:
            return self._active_count < self._max_workers

    def _try_claim(self) -> Optional[dict[str, Any]]:
        """Atomically claim one queued or stale-running job."""
        now = datetime.now(timezone.utc)
        lock_until = (now + timedelta(seconds=LOCK_TTL_SECONDS)).isoformat()

        # 1. Try a queued job first
        resp = (
            self._client.table(TABLE)
            .update({
                "status": "running",
                "worker_id": self._worker_id,
                "locked_until": lock_until,
                "updated_at": now.isoformat(),
            })
            .eq("status", "queued")
            .order("created_at")
            .limit(1)
            .execute()
        )

        if resp.data:
            return resp.data[0]

        # 2. Reclaim a stale running job (crash recovery)
        resp = (
            self._client.table(TABLE)
            .update({
                "status": "running",
                "worker_id": self._worker_id,
                "locked_until": lock_until,
                "updated_at": now.isoformat(),
            })
            .eq("status", "running")
            .lt("locked_until", now.isoformat())
            .order("created_at")
            .limit(1)
            .execute()
        )

        if resp.data:
            logger.warning("Reclaimed stale job %s", resp.data[0]["job_id"])
            return resp.data[0]

        return None

    def _dispatch(self, job: dict[str, Any]) -> None:
        with self._active_lock:
            self._active_count += 1
        self._executor.submit(self._execute, job)

    def _execute(self, job: dict[str, Any]) -> None:
        job_id = job["job_id"]
        attempt = (job.get("attempt") or 0) + 1
        mode = job.get("mode", "unknown")
        t0 = time.monotonic()

        # Increment attempt counter immediately
        try:
            self._client.table(TABLE).update({
                "attempt": attempt,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).eq("job_id", job_id).execute()
        except Exception:
            pass  # non-critical

        try:
            if not self._run_fn:
                raise RuntimeError("No run_fn registered")

            result = self._run_fn(job["symbol"], job["mode"], job["period"])
            if isinstance(result, dict) and result.get("error"):
                raise RuntimeError(str(result["error"]))

            elapsed = time.monotonic() - t0
            logger.info("JOB job_id=%s status=succeeded mode=%s elapsed=%.1fs attempt=%d", job_id, mode, elapsed, attempt)
            self._finish(job_id, status="succeeded", result=result)

        except Exception as exc:
            elapsed = time.monotonic() - t0
            logger.exception("Job %s failed (attempt %d/%d, %.1fs)", job_id, attempt, MAX_RETRIES, elapsed)

            if attempt >= MAX_RETRIES:
                # Exhausted retries → dead-letter queue
                logger.error("JOB job_id=%s status=dead_letter mode=%s attempts=%d", job_id, mode, attempt)
                self._finish(
                    job_id,
                    status="dead_letter",
                    error=f"Exhausted {MAX_RETRIES} retries. Last error: {exc}",
                )
            else:
                # Re-queue for retry
                logger.warning("JOB job_id=%s status=retry mode=%s attempt=%d/%d", job_id, mode, attempt, MAX_RETRIES)
                self._finish(
                    job_id,
                    status="queued",
                    error=f"Attempt {attempt} failed: {exc}",
                )
        finally:
            with self._active_lock:
                self._active_count -= 1

    def _finish(
        self,
        job_id: str,
        *,
        status: str,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "status": status,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "locked_until": None,
        }
        if result is not None:
            payload["result"] = result
        if error is not None:
            payload["error"] = error

        self._client.table(TABLE).update(payload).eq("job_id", job_id).execute()


# ---------------------------------------------------------------------------
# Standalone worker entry point: python -m backend.jobs
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "src"))

    from dotenv import load_dotenv
    load_dotenv(root / ".env")

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    max_w = int(os.getenv("JOB_MAX_WORKERS", "4"))
    mgr = AnalyzeJobManager(max_workers=max_w)

    # Wire up the analysis function
    from main import StockRadar
    radar = StockRadar()

    def _run(symbol: str, mode: str, period: str) -> dict[str, Any]:
        return radar.analyze_stock(symbol=symbol, mode=mode, period=period)

    mgr.set_run_fn(_run)
    mgr.run_worker()
