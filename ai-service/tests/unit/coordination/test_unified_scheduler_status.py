import sqlite3
import time

from app.coordination.slurm_backend import SlurmJobState, SlurmJobStatus
from app.coordination.unified_scheduler import (
    Backend,
    JobState,
    JobType,
    UnifiedJob,
    UnifiedScheduler,
)


def _make_scheduler(tmp_path: str) -> UnifiedScheduler:
    db_path = tmp_path / "unified_scheduler.db"
    return UnifiedScheduler(
        db_path=str(db_path),
        enable_slurm=False,
        enable_vast=False,
        enable_p2p=False,
    )


def _load_job_row(db_path: str, unified_id: str) -> sqlite3.Row:
    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT state, started_at, finished_at FROM jobs WHERE unified_id = ?",
            (unified_id,),
        ).fetchone()


def test_sync_slurm_updates_running_and_started_at(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    db_path = scheduler.db_path

    job = UnifiedJob(name="test-slurm-running", job_type=JobType.SELFPLAY)
    scheduler._record_job(job, Backend.SLURM)
    scheduler._update_job(job.id, backend_job_id="123", state=JobState.QUEUED)

    slurm_jobs = {
        123: SlurmJobStatus(
            job_id=123,
            name="test-slurm-running",
            state=SlurmJobState.RUNNING,
            partition="gpu-selfplay",
            node="lambda-gh200-a",
            start_time=None,
            run_time=None,
        )
    }

    updates = scheduler._sync_slurm_job_states(slurm_jobs)
    row = _load_job_row(db_path, job.id)

    assert updates == 1
    assert row["state"] == JobState.RUNNING.value
    assert row["started_at"] is not None
    assert row["finished_at"] is None


def test_sync_slurm_marks_stale_unknown_with_finished_at(tmp_path):
    scheduler = _make_scheduler(tmp_path)
    db_path = scheduler.db_path

    job = UnifiedJob(name="test-slurm-stale", job_type=JobType.SELFPLAY)
    scheduler._record_job(job, Backend.SLURM)
    scheduler._update_job(job.id, backend_job_id="124", state=JobState.QUEUED)

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET created_at = ? WHERE unified_id = ?",
            (time.time() - 1000, job.id),
        )
        conn.commit()

    slurm_jobs = {
        999: SlurmJobStatus(
            job_id=999,
            name="other-job",
            state=SlurmJobState.RUNNING,
            partition="gpu-selfplay",
            node="lambda-gh200-b",
            start_time=None,
            run_time=None,
        )
    }

    updates = scheduler._sync_slurm_job_states(slurm_jobs, stale_after_seconds=1)
    row = _load_job_row(db_path, job.id)

    assert updates == 1
    assert row["state"] == JobState.UNKNOWN.value
    assert row["finished_at"] is not None
