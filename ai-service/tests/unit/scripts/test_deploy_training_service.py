from __future__ import annotations

import subprocess
from pathlib import Path


AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = AI_SERVICE_ROOT / "scripts" / "deploy_training_service.sh"
P2P_SERVICE_PATH = AI_SERVICE_ROOT / "config" / "systemd" / "ringrift-p2p.service"


def test_dry_run_is_role_aware() -> None:
    result = subprocess.run(
        ["bash", str(SCRIPT_PATH), "--dry-run"],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "gh200-8 (trainer / hex8_2p" in result.stdout
    assert "gh200-11 (selfplay-worker / hex8_2p" in result.stdout
    assert "gh200-10 (evaluator" in result.stdout
    assert "ringrift-training.service" in result.stdout
    assert "ringrift-selfplay-worker.service" in result.stdout
    assert "ringrift-evaluator.service" in result.stdout
    assert "install the current ringrift-p2p.service and keep it enabled" in result.stdout


def test_script_references_role_manifest_and_new_services() -> None:
    script_text = SCRIPT_PATH.read_text()
    assert "config/node_roles.yaml" in script_text
    assert "config/systemd/ringrift-p2p.service" in script_text
    assert 'ssh -n "${SSH_OPTS[@]}"' in script_text
    assert "sudo chown -R ubuntu:ubuntu ~/ringrift/ai-service/logs ~/ringrift/ai-service/p2p_state" in script_text
    assert "ringrift-selfplay-worker.service" in script_text
    assert "ringrift-evaluator.service" in script_text
    assert "disable ringrift-p2p" not in script_text
    assert "pkill -f policy_selfplay_worker.py" not in script_text
    assert "pkill -f evaluator_worker.py" not in script_text
    assert "pkill -f minimal_alphazero_loop.py" not in script_text


def test_p2p_service_enables_lock_recovery_flags() -> None:
    service_text = P2P_SERVICE_PATH.read_text()
    assert "--kill-duplicates" in service_text
    assert "--force-takeover" in service_text
    assert "fuser -k 8770/tcp 8772/tcp" in service_text
