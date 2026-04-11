"""Focused tests for training executor fetch/export integrity."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


class _FakeProc:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


def _write_valid_npz(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    features = np.ones((64, 8), dtype=np.float32)
    np.savez(path, features=features)


class _TimeoutProc:
    def __init__(self):
        self.returncode = -9
        self.kill_called = False

    async def communicate(self):
        return b"CUDA stacktrace line 1\nline 2\n", b""

    def kill(self):
        self.kill_called = True

    async def wait(self):
        return self.returncode


class _SuccessProc:
    def __init__(self, candidate_path: Path, stdout: bytes):
        self.returncode = 0
        self._candidate_path = candidate_path
        self._stdout = stdout

    async def communicate(self):
        self._candidate_path.parent.mkdir(parents=True, exist_ok=True)
        self._candidate_path.write_bytes(b"candidate-model" * 256)
        return self._stdout, b""

    async def wait(self):
        return self.returncode


@pytest.mark.asyncio
async def test_try_fetch_model_from_s3_preserves_existing_file_on_failure(monkeypatch, tmp_path):
    """Failed model fetches should not clobber an existing checkpoint."""
    from scripts.p2p.work_executors import training_executor

    final_path = tmp_path / "models" / "candidate_hex8_2p.pth"
    final_path.parent.mkdir(parents=True)
    original_bytes = b"good-model" * 200
    final_path.write_bytes(original_bytes)
    seen: dict[str, Path] = {}

    async def fake_exec(*cmd, **kwargs):
        temp_path = Path(cmd[4])
        seen["temp"] = temp_path
        temp_path.write_bytes(b"partial")
        return _FakeProc(returncode=1, stderr=b"download failed")

    monkeypatch.setattr(training_executor.asyncio, "create_subprocess_exec", fake_exec)

    result = await training_executor._try_fetch_model_from_s3(
        "candidate_hex8_2p.pth",
        str(final_path),
    )

    assert result is False
    assert final_path.read_bytes() == original_bytes
    assert not seen["temp"].exists()


@pytest.mark.asyncio
async def test_try_fetch_npz_from_cluster_cleans_partial_temp_file(monkeypatch, tmp_path):
    """Failed coordinator rsync should remove its temp artifact."""
    from app.config import cluster_config
    from scripts.p2p.work_executors import training_executor

    npz_path = tmp_path / "data" / "training" / "hex8_2p.npz"
    seen: dict[str, Path] = {}

    fake_cfg = SimpleNamespace(
        _raw_config={"preferred_leader": "leader-a"},
        hosts_raw={
            "leader-a": {
                "tailscale_ip": "100.64.0.10",
                "ssh_user": "ubuntu",
                "ringrift_path": "~/ringrift/ai-service",
            }
        },
    )

    async def fake_exec(*cmd, **kwargs):
        temp_path = Path(cmd[4])
        seen["temp"] = temp_path
        temp_path.write_bytes(b"partial")
        return _FakeProc(returncode=1, stderr=b"rsync failed")

    monkeypatch.setattr(cluster_config, "load_cluster_config", lambda: fake_cfg)
    monkeypatch.setattr(training_executor.asyncio, "create_subprocess_exec", fake_exec)

    result = await training_executor._try_fetch_npz_from_cluster(
        tmp_path,
        "hex8_2p",
        npz_path,
    )

    assert result is None
    assert not npz_path.exists()
    assert not seen["temp"].exists()


@pytest.mark.asyncio
async def test_try_local_jsonl_export_promotes_temp_output_on_success(monkeypatch, tmp_path):
    """Successful local export should write via temp path then rename into place."""
    from scripts.p2p.work_executors import training_executor

    ai_service_root = tmp_path
    jsonl_dir = ai_service_root / "data" / "selfplay" / "p2p_gpu" / "hex8_2p"
    jsonl_dir.mkdir(parents=True)
    for idx in range(5):
        (jsonl_dir / f"game_{idx}.jsonl").write_text("{}\n")

    script_path = ai_service_root / "scripts" / "jsonl_to_npz.py"
    script_path.parent.mkdir(parents=True)
    script_path.write_text("# placeholder\n")

    final_path = ai_service_root / "data" / "training" / "hex8_2p.npz"
    seen: dict[str, Path] = {}

    async def fake_exec(*cmd, **kwargs):
        output_idx = cmd.index("--output") + 1
        temp_path = Path(cmd[output_idx])
        seen["temp"] = temp_path
        temp_path.write_bytes(b"x" * 2048)
        return _FakeProc(returncode=0, stdout=b"ok", stderr=b"")

    monkeypatch.setattr(training_executor.asyncio, "create_subprocess_exec", fake_exec)

    result = await training_executor._try_local_jsonl_export(
        ai_service_root,
        "hex8_2p",
        "hex8",
        2,
    )

    assert result == final_path
    assert final_path.exists()
    assert final_path.stat().st_size == 2048
    assert not seen["temp"].exists()


@pytest.mark.asyncio
async def test_execute_training_work_records_structured_failure_when_training_disabled(
    monkeypatch,
    tmp_path,
):
    """Rejected non-training nodes should still return structured context."""
    from scripts.p2p.managers import work_discovery_manager
    from scripts.p2p.work_executors import training_executor

    monkeypatch.setattr(
        work_discovery_manager,
        "_is_training_enabled_for_node",
        lambda: False,
    )

    work_item = {"work_id": "work-1"}
    result = await training_executor.execute_training_work(
        work_item=work_item,
        config={"board_type": "hex8", "num_players": 2, "model_version": "v5-heavy"},
        node_id="lambda-gh200-10",
        ringrift_path=tmp_path,
    )

    assert result is False
    assert work_item["error"] == "training_disabled:lambda-gh200-10"
    assert work_item["result"]["failure_stage"] == "dispatch_guard"
    assert work_item["result"]["config_key"] == "hex8_2p"
    assert work_item["result"]["model_version"] == "v5-heavy"
    assert work_item["result"]["node_id"] == "lambda-gh200-10"


@pytest.mark.asyncio
async def test_execute_training_work_records_timeout_output(monkeypatch, tmp_path):
    """Timeout failures should retain the partial subprocess output."""
    import app.coordination.npz_validation as npz_validation
    from scripts.p2p.managers import work_discovery_manager
    from scripts.p2p.work_executors import training_executor

    monkeypatch.setattr(
        work_discovery_manager,
        "_is_training_enabled_for_node",
        lambda: True,
    )
    monkeypatch.setattr(npz_validation, "quick_npz_check", lambda path: (True, ""))

    npz_path = tmp_path / "data" / "training" / "hex8_2p.npz"
    _write_valid_npz(npz_path)

    async def fake_s3_fetch(*args, **kwargs):
        return False

    async def fake_exec(*args, **kwargs):
        return _TimeoutProc()

    async def fake_wait_for(awaitable, timeout):
        if timeout == 7200:
            awaitable.close()
            raise asyncio.TimeoutError
        return await awaitable

    monkeypatch.setattr(training_executor, "_try_fetch_npz_from_s3", fake_s3_fetch)
    monkeypatch.setattr(training_executor.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(training_executor.asyncio, "wait_for", fake_wait_for)

    work_item = {"work_id": "work-timeout"}
    result = await training_executor.execute_training_work(
        work_item=work_item,
        config={"board_type": "hex8", "num_players": 2},
        node_id="lambda-gh200-10",
        ringrift_path=tmp_path,
    )

    assert result is False
    assert work_item["error"].startswith("training_timeout:7200s:hex8_2p")
    assert work_item["result"]["failure_stage"] == "training_timeout"
    assert work_item["result"]["timed_out"] is True
    assert work_item["result"]["timeout_seconds"] == 7200
    assert "CUDA stacktrace line 1" in work_item["result"]["subprocess_output_tail"]


@pytest.mark.asyncio
async def test_execute_training_work_surfaces_event_and_s3_followup_failures(
    monkeypatch,
    tmp_path,
):
    """Post-training signaling/upload failures should not be silent."""
    import app.coordination.npz_validation as npz_validation
    import app.coordination.event_emission_helpers as event_emission_helpers
    import app.training.game_gauntlet as game_gauntlet
    from scripts.p2p.managers import work_discovery_manager
    from scripts.p2p.work_executors import training_executor

    monkeypatch.setattr(
        work_discovery_manager,
        "_is_training_enabled_for_node",
        lambda: True,
    )
    monkeypatch.setattr(npz_validation, "quick_npz_check", lambda path: (True, ""))

    npz_path = tmp_path / "data" / "training" / "hex8_2p.npz"
    _write_valid_npz(npz_path)

    candidate_path = tmp_path / "models" / "candidate_hex8_2p.pth"
    stdout = (
        b"epoch 1\n"
        b"TRAINING_SUMMARY: loss=0.1234 samples=640 games=16\n"
    )

    async def fake_s3_fetch(*args, **kwargs):
        return False

    async def fake_push(*args, **kwargs):
        return False

    async def fake_exec(*args, **kwargs):
        return _SuccessProc(candidate_path, stdout)

    monkeypatch.setattr(training_executor, "_try_fetch_npz_from_s3", fake_s3_fetch)
    monkeypatch.setattr(training_executor, "_try_push_candidate_to_s3", fake_push)
    monkeypatch.setattr(training_executor.asyncio, "create_subprocess_exec", fake_exec)
    monkeypatch.setattr(game_gauntlet, "create_neural_ai", lambda *args, **kwargs: object())
    monkeypatch.setattr(game_gauntlet, "create_baseline_ai", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        game_gauntlet,
        "play_single_game",
        lambda *args, **kwargs: SimpleNamespace(candidate_won=True),
    )
    monkeypatch.setattr(event_emission_helpers, "safe_emit_event", lambda *args, **kwargs: False)

    work_item = {"work_id": "work-success"}
    result = await training_executor.execute_training_work(
        work_item=work_item,
        config={"board_type": "hex8", "num_players": 2},
        node_id="lambda-gh200-10",
        ringrift_path=tmp_path,
    )

    assert result is True
    assert work_item["result"]["training_event_emitted"] is False
    assert work_item["result"]["candidate_s3_pushed"] is False
    warning_stages = {warning["stage"] for warning in work_item["result"]["warnings"]}
    assert "training_event_emit" in warning_stages
    assert "candidate_s3_push" in warning_stages
