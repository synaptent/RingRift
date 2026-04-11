"""Operational script smoke tests for supported training infrastructure."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path


AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
RESTART_DEAD_LOOPS = AI_SERVICE_ROOT / "scripts" / "restart_dead_loops.sh"
CLUSTER_HEALTH = AI_SERVICE_ROOT / "scripts" / "cluster_health.py"
TRAINING_STATUS = AI_SERVICE_ROOT / "scripts" / "training_status.py"
TRAINING_DASHBOARD = AI_SERVICE_ROOT / "scripts" / "training_dashboard.py"
SUPERVISOR_SCRIPT = AI_SERVICE_ROOT / "scripts" / "minimal_loop_supervisor.sh"
EXPECTED_CONFIGS = {"hex8_2p", "square8_2p", "square8_3p", "square8_4p"}


def _python_env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    return env


def test_restart_dead_loops_dry_run_lists_supported_configs() -> None:
    result = subprocess.run(
        ["bash", str(RESTART_DEAD_LOOPS), "--dry-run"],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Dead-loop restart check completed." in result.stdout
    assert result.stdout.count("[DRY] Would probe") == 4
    for config in EXPECTED_CONFIGS:
        assert f"=== {config} (" in result.stdout


def test_cluster_health_help_works() -> None:
    result = subprocess.run(
        [sys.executable, str(CLUSTER_HEALTH), "--help"],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_python_env(),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Comprehensive health report" in result.stdout


def test_training_status_no_s3_json_is_valid() -> None:
    result = subprocess.run(
        [sys.executable, str(TRAINING_STATUS), "--json", "--no-s3"],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_python_env(),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    rows = json.loads(result.stdout)
    assert isinstance(rows, list)
    assert len(rows) == 4
    assert {row["config"] for row in rows} == EXPECTED_CONFIGS


def test_training_dashboard_help_works() -> None:
    result = subprocess.run(
        [sys.executable, str(TRAINING_DASHBOARD), "--help"],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
        env=_python_env(),
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Refreshing terminal dashboard" in result.stdout


def test_minimal_loop_supervisor_heartbeat_contains_runtime_fields() -> None:
    config = f"pytest_phase18_{int(time.time() * 1000)}"
    heartbeat_file = Path("/tmp") / f"supervisor_{config}.heartbeat"
    if heartbeat_file.exists():
        heartbeat_file.unlink()

    env = _python_env()
    env["SUPERVISOR_HEARTBEAT_INTERVAL_SECONDS"] = "1"
    proc = subprocess.Popen(
        [
            "bash",
            str(SUPERVISOR_SCRIPT),
            "--config",
            config,
            "--restart-delay-seconds",
            "1",
            "--max-restarts",
            "1",
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        cwd=AI_SERVICE_ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        payload = None
        deadline = time.time() + 8
        while time.time() < deadline:
            if heartbeat_file.exists():
                try:
                    payload = json.loads(heartbeat_file.read_text())
                except json.JSONDecodeError:
                    time.sleep(0.1)
                    continue
                if payload.get("state") == "running" and payload.get("child_pid"):
                    break
            time.sleep(0.2)

        assert payload is not None, "supervisor heartbeat was not written"
        assert payload["config"] == config
        assert payload["state"] == "running"
        assert payload["supervisor_pid"]
        assert payload["child_pid"]
        assert payload["restart_count"] == 0
        assert payload["last_restart_time"]
        assert isinstance(payload["uptime_seconds"], int)
        assert payload["uptime_seconds"] >= 0
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
        if heartbeat_file.exists():
            heartbeat_file.unlink()
