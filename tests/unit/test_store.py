"""Unit tests for the job store (in-memory + persistence semantics)."""

import json
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from edge_tts_api import store as store_mod
from edge_tts_api.store import JobStore


@pytest.fixture()
def job_store(tmp_path):
    return JobStore(output_dir=tmp_path)


def _record(job_id="job-a", status="Succeeded"):
    return {
        "id": job_id,
        "status": status,
        "createdDateTime": store_mod.utc_now_iso(),
        "lastActionDateTime": store_mod.utc_now_iso(),
        "inputKind": "PlainText",
        "customVoices": {},
        "properties": {"timeToLiveInHours": 744},
    }


class TestJobStore:
    def test_create_and_get(self, job_store):
        record = _record()
        job_store.create(record)
        assert job_store.get("job-a") == record
        assert job_store.get("missing") is None

    def test_update_persists_changes(self, job_store):
        job_store.create(_record())
        updated = job_store.update("job-a", status="Failed")
        assert updated["status"] == "Failed"
        assert job_store.get("job-a")["status"] == "Failed"

    def test_update_missing_is_none(self, job_store):
        assert job_store.update("ghost", status="Failed") is None

    def test_delete_removes_record_and_dir(self, job_store):
        job_store.create(_record())
        assert job_store.delete("job-a") is True
        assert job_store.get("job-a") is None
        assert not job_store.job_dir("job-a").exists()
        assert job_store.delete("job-a") is False

    def test_all_jobs_snapshot(self, job_store):
        job_store.create(_record("job-a"))
        job_store.create(_record("job-b"))
        assert len(job_store.all_jobs()) == 2

    def test_persist_and_reload(self, job_store, tmp_path):
        job_store.create(_record("keep-me"))
        reloaded = JobStore(output_dir=tmp_path)
        reloaded.load()
        assert reloaded.get("keep-me")["id"] == "keep-me"

    def test_reload_marks_inflight_failed(self, job_store, tmp_path):
        job_store.create(_record("stuck", status="Running"))
        reloaded = JobStore(output_dir=tmp_path)
        reloaded.load()
        record = reloaded.get("stuck")
        assert record["status"] == "Failed"
        assert record["properties"]["error"]["code"] == "InternalServerError"

    def test_sweep_expired_removes_old_jobs(self, job_store):
        old = _record("old-job")
        old["lastActionDateTime"] = (
            datetime.now(timezone.utc) - timedelta(hours=745)
        ).strftime("%Y-%m-%dT%H:%M:%S.%f0Z")
        job_store.create(old)
        job_store.create(_record("fresh-job"))
        removed = job_store.sweep_expired()
        assert removed == 1
        assert job_store.get("old-job") is None
        assert job_store.get("fresh-job") is not None

    def test_sweep_keeps_running_jobs(self, job_store):
        running = _record("still-going", status="Running")
        running["lastActionDateTime"] = (
            datetime.now(timezone.utc) - timedelta(hours=800)
        ).strftime("%Y-%m-%dT%H:%M:%S.%f0Z")
        job_store.create(running)
        assert job_store.sweep_expired() == 0

    def test_utc_now_iso_format(self):
        stamp = store_mod.utc_now_iso()
        assert stamp.endswith("Z")
        assert stamp.count(".") == 1
        datetime.fromisoformat(stamp.replace("Z", "+00:00"))
