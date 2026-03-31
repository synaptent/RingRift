"""Focused tests for gauntlet runner download integrity."""

from __future__ import annotations

import importlib
import subprocess
from pathlib import Path


def _load_module():
    return importlib.import_module("scripts.gauntlet_runner")


def test_pull_candidate_cleans_partial_download_on_copy_failure(monkeypatch, tmp_path):
    """Failed S3 copies should not leave partial candidate artifacts behind."""
    gauntlet_runner = _load_module()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr(gauntlet_runner, "MODELS_DIR", models_dir)
    monkeypatch.setattr(gauntlet_runner, "evaluated", {})

    local_path = models_dir / "candidate_hex8_2p.pth"
    sidecar_path = local_path.with_suffix(local_path.suffix + ".sha256")
    sidecar_path.write_text("stale-sidecar")

    def fake_run(cmd, capture_output, text, timeout):
        if cmd[:3] == ["aws", "s3", "ls"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="2026-03-31 00:00:00        123 candidate_hex8_2p.pth\n",
                stderr="",
            )
        if cmd[:3] == ["aws", "s3", "cp"] and str(cmd[2]).endswith(".sha256"):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="NoSuchKey",
            )
        if cmd[:3] == ["aws", "s3", "cp"]:
            local_path.write_text("partial-download")
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="network timeout",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(gauntlet_runner.subprocess, "run", fake_run)

    assert gauntlet_runner.pull_candidate("hex8_2p") is None
    assert not local_path.exists()
    assert not sidecar_path.exists()


def test_pull_candidate_cleans_empty_download(monkeypatch, tmp_path):
    """A zero-byte candidate should be rejected and removed immediately."""
    gauntlet_runner = _load_module()
    models_dir = tmp_path / "models"
    models_dir.mkdir()

    monkeypatch.setattr(gauntlet_runner, "MODELS_DIR", models_dir)
    monkeypatch.setattr(gauntlet_runner, "evaluated", {})

    local_path = models_dir / "candidate_square8_2p.pth"

    def fake_run(cmd, capture_output, text, timeout):
        if cmd[:3] == ["aws", "s3", "ls"]:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="2026-03-31 00:00:00         42 candidate_square8_2p.pth\n",
                stderr="",
            )
        if cmd[:3] == ["aws", "s3", "cp"] and str(cmd[2]).endswith(".sha256"):
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="",
                stderr="NoSuchKey",
            )
        if cmd[:3] == ["aws", "s3", "cp"]:
            local_path.write_bytes(b"")
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=0,
                stdout="",
                stderr="",
            )
        raise AssertionError(f"Unexpected command: {cmd}")

    monkeypatch.setattr(gauntlet_runner.subprocess, "run", fake_run)

    assert gauntlet_runner.pull_candidate("square8_2p") is None
    assert not local_path.exists()
