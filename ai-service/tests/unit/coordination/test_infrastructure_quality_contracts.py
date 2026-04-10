"""Quality contracts for reusable coordination infrastructure."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


AI_SERVICE_ROOT = Path(__file__).resolve().parents[3]
INFRASTRUCTURE_QUALITY_FILES = (
    Path("tests/unit/coordination/test_event_subscription_completeness.py"),
    Path("tests/unit/coordination/test_exception_specificity.py"),
    Path("tests/unit/coordination/test_handler_base.py"),
    Path("tests/unit/coordination/test_infrastructure_quality_contracts.py"),
    Path("tests/unit/coordination/test_safe_event_emitter.py"),
    Path("tests/unit/scripts/test_master_loop_watchdog.py"),
)
RUFF_NO_TEST_F401_IGNORE_CONFIG = (
    "lint.per-file-ignores = {"
    "'tests/**/*.py' = ['F811', 'B011'], "
    "'scripts/**/*.py' = [], "
    "'**/__init__.py' = ['F401', 'I001']"
    "}"
)


def test_infrastructure_quality_tests_have_no_unused_imports() -> None:
    """Keep active infrastructure tests clean despite the historical test-suite baseline."""
    pytest.importorskip(
        "ruff",
        reason="ruff is optional; skip this lint contract when it is not installed",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "ruff",
            "check",
            *[str(path) for path in INFRASTRUCTURE_QUALITY_FILES],
            "--select",
            "F401",
            "--config",
            RUFF_NO_TEST_F401_IGNORE_CONFIG,
        ],
        cwd=AI_SERVICE_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("profile", ["minimal", "lean", "standard", "full"])
def test_master_loop_supported_profiles_validate(profile: str) -> None:
    from scripts.master_loop import validate_daemon_profile

    assert validate_daemon_profile(profile) == profile


def test_master_loop_rejects_unknown_profile() -> None:
    from scripts.master_loop import SUPPORTED_DAEMON_PROFILES, validate_daemon_profile

    with pytest.raises(ValueError) as exc_info:
        validate_daemon_profile("legacy-everything")

    error = str(exc_info.value)
    assert "legacy-everything" in error
    for profile in SUPPORTED_DAEMON_PROFILES:
        assert profile in error


@pytest.mark.parametrize("profile", ["minimal", "lean", "standard", "full"])
def test_master_loop_profiles_resolve_to_active_registry_daemons(
    profile: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from app.config.env import env
    from app.coordination.daemon_registry import get_deprecated_types
    from app.coordination.daemon_types import DaemonType
    from scripts.master_loop import MasterLoopController

    controller = object.__new__(MasterLoopController)
    controller.daemon_profile = profile
    controller._has_aws_credentials = lambda: False
    controller._has_npx = lambda: False

    monkeypatch.setattr(env, "is_coordinator", False, raising=False)
    monkeypatch.setattr(env, "is_standby_coordinator", False, raising=False)
    monkeypatch.setattr(env, "node_id", "unit-test-node", raising=False)

    daemons = controller._get_daemons_for_profile()

    assert daemons
    assert len(daemons) == len(set(daemons))
    assert all(isinstance(daemon, DaemonType) for daemon in daemons)
    assert not (set(daemons) & set(get_deprecated_types()))
