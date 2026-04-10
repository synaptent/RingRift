"""Exception specificity guardrails for reusable coordination infrastructure."""

from __future__ import annotations

import ast
from pathlib import Path


COORDINATION_DAEMON_FILES = (
    "app/coordination/coordinator_health_monitor_daemon.py",
    "app/coordination/maintenance_daemon.py",
    "app/coordination/memory_monitor_daemon.py",
    "app/coordination/model_registry_daemon.py",
    "app/coordination/progress_watchdog_daemon.py",
    "app/coordination/stale_fallback.py",
)

TRAINING_PIPELINE_DAEMON_FILES = (
    "app/coordination/backlog_evaluation_daemon.py",
    "app/coordination/stale_evaluation_daemon.py",
    "app/coordination/training_data_recovery_daemon.py",
    "app/coordination/training_watchdog_daemon.py",
)

P2P_JOB_MANAGEMENT_FILES = (
    "scripts/p2p/job_state_machine.py",
    "scripts/p2p/managers/job_lifecycle_manager.py",
    "scripts/p2p/managers/worker_pull_controller.py",
    "scripts/p2p/managers/work_discovery_manager.py",
)


def _exception_handler_names(handler_type: ast.expr | None) -> set[str]:
    if handler_type is None:
        return {"BaseException"}
    if isinstance(handler_type, ast.Name):
        return {handler_type.id}
    if isinstance(handler_type, ast.Tuple):
        names: set[str] = set()
        for elt in handler_type.elts:
            names.update(_exception_handler_names(elt))
        return names
    if isinstance(handler_type, ast.Attribute):
        return {handler_type.attr}
    return set()


def test_phase3_exception_catches_are_specific_or_explicitly_justified() -> None:
    service_root = Path(__file__).parents[3]
    target_files = (
        COORDINATION_DAEMON_FILES
        + TRAINING_PIPELINE_DAEMON_FILES
        + P2P_JOB_MANAGEMENT_FILES
    )
    violations: list[str] = []

    for rel_path in target_files:
        path = service_root / rel_path
        lines = path.read_text().splitlines()
        tree = ast.parse("\n".join(lines), filename=str(path))

        for node in ast.walk(tree):
            if not isinstance(node, ast.ExceptHandler):
                continue
            caught = _exception_handler_names(node.type)
            if "Exception" not in caught and "BaseException" not in caught:
                continue

            line = lines[node.lineno - 1]
            if "# noqa: BLE001" not in line:
                violations.append(f"{rel_path}:{node.lineno}: {line.strip()}")

    assert not violations, "\n".join(violations)
