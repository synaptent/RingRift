"""Focused tests for training executor fetch/export integrity."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


class _FakeProc:
    def __init__(self, returncode: int, stdout: bytes = b"", stderr: bytes = b""):
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr


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
