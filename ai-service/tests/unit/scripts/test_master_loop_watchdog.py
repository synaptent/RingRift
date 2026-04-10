"""Tests for master_loop_watchdog restart command wiring."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

import scripts.master_loop_watchdog as master_loop_watchdog

pytestmark = pytest.mark.timeout(30)


def test_build_master_loop_command_defaults_to_local_venv_and_lean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    script_path = tmp_path / "scripts" / "master_loop.py"
    python_path = tmp_path / ".venv" / "bin" / "python3"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env python3\n")
    python_path.write_text("")

    monkeypatch.delenv("RINGRIFT_MASTER_LOOP_PYTHON", raising=False)
    monkeypatch.delenv("RINGRIFT_MASTER_LOOP_PROFILE", raising=False)

    command = master_loop_watchdog.build_master_loop_command(
        master_loop_script=script_path,
        ai_service_root=tmp_path,
    )

    assert command == [
        str(python_path),
        str(script_path),
        "--profile",
        "lean",
    ]


def test_build_master_loop_command_respects_env_overrides(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    script_path = tmp_path / "scripts" / "master_loop.py"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env python3\n")

    monkeypatch.setenv("RINGRIFT_MASTER_LOOP_PYTHON", "/tmp/custom-python")
    monkeypatch.setenv("RINGRIFT_MASTER_LOOP_PROFILE", "minimal")

    command = master_loop_watchdog.build_master_loop_command(
        master_loop_script=script_path,
        ai_service_root=tmp_path,
    )

    assert command == [
        "/tmp/custom-python",
        str(script_path),
        "--profile",
        "minimal",
    ]


def test_start_master_loop_uses_resolved_command(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    script_path = tmp_path / "scripts" / "master_loop.py"
    python_path = tmp_path / ".venv" / "bin" / "python3"
    script_path.parent.mkdir(parents=True, exist_ok=True)
    python_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text("#!/usr/bin/env python3\n")
    python_path.write_text("")

    process = MagicMock()
    process.poll.return_value = None
    process.pid = 12345

    monkeypatch.delenv("RINGRIFT_MASTER_LOOP_PYTHON", raising=False)
    monkeypatch.delenv("RINGRIFT_MASTER_LOOP_PROFILE", raising=False)
    monkeypatch.setattr(master_loop_watchdog, "AI_SERVICE_ROOT", tmp_path)
    monkeypatch.setattr(master_loop_watchdog, "MASTER_LOOP_SCRIPT", script_path)
    monkeypatch.setattr(master_loop_watchdog.time, "sleep", lambda _: None)

    popen = MagicMock(return_value=process)
    monkeypatch.setattr(master_loop_watchdog.subprocess, "Popen", popen)

    watchdog = master_loop_watchdog.MasterLoopWatchdog()
    assert watchdog._start_master_loop() is True

    assert popen.call_args.args[0] == [
        str(python_path),
        str(script_path),
        "--profile",
        "lean",
    ]
    assert popen.call_args.kwargs["cwd"] == str(tmp_path)
    assert popen.call_args.kwargs["env"]["PYTHONPATH"] == str(tmp_path)
