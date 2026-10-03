"""In-memory job store with disk persistence for the batch synthesis service."""

import asyncio
import json
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config

# Job statuses as defined by the Azure batch synthesis API.
STATUS_NOT_STARTED = "NotStarted"
STATUS_RUNNING = "Running"
STATUS_SUCCEEDED = "Succeeded"
STATUS_FAILED = "Failed"
TERMINAL_STATUSES = (STATUS_SUCCEEDED, STATUS_FAILED)


def utc_now_iso() -> str:
    """Return the current UTC time in Azure's 7-digit ISO-8601 format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f0Z")


def _parse_iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


class JobStore:
    """Thread-safe store of batch synthesis jobs.

    Job records are kept in memory and mirrored to ``output/<id>/job.json``
    so that completed jobs (and their result ZIPs) survive restarts.
    """

    def __init__(self, output_dir: Path = None) -> None:
        self.output_dir = output_dir or config.OUTPUT_DIR
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self._tasks: Dict[str, asyncio.Task] = {}
        self._lock = threading.Lock()

    # -- persistence ------------------------------------------------------

    def job_dir(self, job_id: str) -> Path:
        return self.output_dir / job_id

    def job_record_path(self, job_id: str) -> Path:
        return self.job_dir(job_id) / "job.json"

    def load(self) -> None:
        """Load persisted job records from the output directory."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for record_path in self.output_dir.glob("*/job.json"):
            try:
                record = json.loads(record_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            # A job that was mid-flight when the service stopped can never
            # complete; mark it failed so clients see a terminal state.
            if record.get("status") not in TERMINAL_STATUSES:
                record["status"] = STATUS_FAILED
                record["properties"]["error"] = {
                    "code": "InternalServerError",
                    "message": "The service was restarted before the job completed.",
                }
                record["lastActionDateTime"] = utc_now_iso()
                self._persist(record)
            self._jobs[record["id"]] = record

    def _persist(self, record: Dict[str, Any]) -> None:
        job_dir = self.job_dir(record["id"])
        job_dir.mkdir(parents=True, exist_ok=True)
        payload = {k: v for k, v in record.items() if not k.startswith("_")}
        tmp = self.job_record_path(record["id"]).with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self.job_record_path(record["id"]))

    # -- CRUD -------------------------------------------------------------

    def create(self, record: Dict[str, Any]) -> None:
        with self._lock:
            self._jobs[record["id"]] = record
            self._persist(record)

    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._jobs.get(job_id)

    def all_jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._jobs.values())

    def update(self, job_id: str, **changes: Any) -> Optional[Dict[str, Any]]:
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return None
            record.update(changes)
            self._persist(record)
            return record

    def delete(self, job_id: str) -> bool:
        with self._lock:
            if job_id not in self._jobs:
                return False
            del self._jobs[job_id]
        task = self._tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()
        import shutil

        shutil.rmtree(self.job_dir(job_id), ignore_errors=True)
        return True

    # -- background tasks ---------------------------------------------------

    def attach_task(self, job_id: str, task: asyncio.Task) -> None:
        self._tasks[job_id] = task

    def cancel_task(self, job_id: str) -> None:
        task = self._tasks.pop(job_id, None)
        if task is not None and not task.done():
            task.cancel()

    async def wait_for_terminal(self, job_id: str, timeout: float = 120.0) -> Optional[Dict[str, Any]]:
        """Wait until a job reaches a terminal status (test helper)."""
        import time

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            record = self.get(job_id)
            if record is None or record["status"] in TERMINAL_STATUSES:
                return record
            await asyncio.sleep(0.05)
        return self.get(job_id)

    # -- TTL ----------------------------------------------------------------

    def sweep_expired(self) -> int:
        """Delete completed jobs whose timeToLiveInHours has elapsed."""
        now = datetime.now(timezone.utc)
        removed = 0
        for record in self.all_jobs():
            if record["status"] not in TERMINAL_STATUSES:
                continue
            completed = _parse_iso(record["lastActionDateTime"])
            ttl = timedelta(hours=record["properties"].get("timeToLiveInHours", 744))
            if now > completed + ttl:
                self.delete(record["id"])
                removed += 1
        return removed


def new_internal_id() -> str:
    """Internal GUID used in summary.json, mirroring Azure's jobID."""
    return str(uuid.uuid4())
