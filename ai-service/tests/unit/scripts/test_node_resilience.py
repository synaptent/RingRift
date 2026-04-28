"""Focused tests for node_resilience coordinator disk safeguards."""

from __future__ import annotations

import subprocess
from types import SimpleNamespace

from scripts import node_resilience


def _cfg(tmp_path, *, disk_threshold: int = 90) -> node_resilience.NodeConfig:
    ai_service = tmp_path / "ai-service"
    scripts_dir = ai_service / "scripts"
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "disk_monitor.py").write_text("# test stub\n", encoding="utf-8")
    return node_resilience.NodeConfig(
        node_id="mac-studio",
        coordinator_url="http://localhost:8770",
        ai_service_dir=str(ai_service),
        num_gpus=0,
        disk_threshold=disk_threshold,
        min_free_gb=10.0,
    )


def test_node_resilience_default_disk_threshold_is_90() -> None:
    assert node_resilience.NodeConfig(
        node_id="n",
        coordinator_url="http://localhost",
        ai_service_dir=".",
        num_gpus=0,
    ).disk_threshold == 90


def test_subprocess_failure_message_includes_stdout_when_stderr_empty() -> None:
    result = subprocess.CompletedProcess(
        args=["disk_monitor.py"],
        returncode=2,
        stdout="cleanup could not free enough space",
        stderr="",
    )

    message = node_resilience.NodeResilience._subprocess_failure_message(result)

    assert "exit=2" in message
    assert "stdout=cleanup could not free enough space" in message


def test_cleanup_must_relieve_disk_pressure(monkeypatch, tmp_path) -> None:
    cfg = _cfg(tmp_path, disk_threshold=90)
    resilience = node_resilience.NodeResilience(cfg)

    readings = iter([(96.0, 1.0), (95.0, 2.0)])
    monkeypatch.setattr(resilience, "_disk_usage", lambda: next(readings))
    monkeypatch.setattr(
        node_resilience.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    assert resilience.check_and_cleanup_disk() is False


def test_cleanup_success_after_pressure_relieved(monkeypatch, tmp_path) -> None:
    cfg = _cfg(tmp_path, disk_threshold=90)
    resilience = node_resilience.NodeResilience(cfg)

    readings = iter([(96.0, 1.0), (80.0, 100.0)])
    monkeypatch.setattr(resilience, "_disk_usage", lambda: next(readings))
    monkeypatch.setattr(
        node_resilience.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0, stdout="", stderr=""),
    )

    assert resilience.check_and_cleanup_disk() is True


def test_start_p2p_prefers_launchd_on_macos(monkeypatch, tmp_path) -> None:
    cfg = _cfg(tmp_path)
    resilience = node_resilience.NodeResilience(cfg)
    checks = iter([False, True])
    kickstarted: list[str] = []

    monkeypatch.setattr(resilience, "check_p2p_health", lambda: next(checks))
    monkeypatch.setattr(resilience, "_check_port_available", lambda port: True)
    monkeypatch.setattr(resilience, "_systemd_usable", lambda: False)
    monkeypatch.setattr(resilience, "_launchd_job_loaded", lambda label: label == "com.ringrift.p2p")
    monkeypatch.setattr(resilience, "_kickstart_launchd_job", lambda label: kickstarted.append(label) or True)

    assert resilience.start_p2p_orchestrator() is True
    assert kickstarted == ["com.ringrift.p2p"]


def test_direct_p2p_spawn_forces_supervisor(monkeypatch, tmp_path) -> None:
    cfg = _cfg(tmp_path)
    resilience = node_resilience.NodeResilience(cfg)
    commands: list[list[str]] = []

    class FakeProc:
        pid = 12345
        returncode = 0

        def poll(self) -> int:
            return 0

    def fake_popen(cmd, **kwargs):
        commands.append(cmd)
        return FakeProc()

    monkeypatch.setattr(resilience, "check_p2p_health", lambda: False)
    monkeypatch.setattr(resilience, "_check_port_available", lambda port: True)
    monkeypatch.setattr(resilience, "_systemd_usable", lambda: False)
    monkeypatch.setattr(resilience, "_launchd_job_loaded", lambda label: False)
    monkeypatch.setattr(resilience, "_python_for_orchestrator", lambda: "python")
    monkeypatch.setattr(node_resilience.subprocess, "Popen", fake_popen)

    assert resilience.start_p2p_orchestrator() is False
    assert commands
    assert "--force-supervisor" in commands[0]
