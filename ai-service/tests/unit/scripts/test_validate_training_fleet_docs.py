from __future__ import annotations

import json
import subprocess
from pathlib import Path


AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = AI_SERVICE_ROOT / "scripts" / "validate_training_fleet_docs.py"


def test_training_fleet_docs_validator_passes_current_manifest() -> None:
    result = subprocess.run(
        ["python3", str(SCRIPT_PATH), "--json"],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    payload = json.loads(result.stdout)
    assert payload["ok"] is True
    assert payload["checks"] >= 20
    assert payload["errors"] == []


def test_training_fleet_docs_validator_is_local_only() -> None:
    script_text = SCRIPT_PATH.read_text(encoding="utf-8")

    assert "subprocess" not in script_text
    assert "paramiko" not in script_text
    assert "requests" not in script_text
    assert "urllib.request" not in script_text
