#!/usr/bin/env python3
"""Fail if app/ imports from scripts/ outside the approved legacy baseline."""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path


_IMPORT_RE = re.compile(r"^\s*(from|import)\s+scripts(\.|\\b)")

# Existing app -> scripts imports predate this guard. Keep them explicit so CI
# blocks new layer violations without pretending the historical debt is gone.
_ALLOWED_EXISTING_VIOLATIONS = {
    (
        "app/coordination/auto_promotion_daemon.py",
        "from scripts.model_lineage import register_model, update_performance",
    ),
    (
        "app/coordination/cascade_training.py",
        "from scripts.transfer_2p_to_4p import transfer_2p_to_np",
    ),
    (
        "app/coordination/cluster_resilience_orchestrator.py",
        "from scripts.p2p.health_coordinator import (",
    ),
    (
        "app/coordination/distributed_lock.py",
        "from scripts.p2p.consensus_mixin import (",
    ),
    (
        "app/coordination/distributed_lock.py",
        "from scripts.p2p_orchestrator import P2POrchestrator",
    ),
    (
        "app/coordination/memory_pressure_controller.py",
        "from scripts.p2p.loops.gossip_state_cleanup_loop import GossipStateCleanupLoop",
    ),
    (
        "app/coordination/memory_pressure_controller.py",
        "from scripts.p2p.loops.base import LoopManager",
    ),
    (
        "app/coordination/p2p_recovery_daemon.py",
        "from scripts.p2p.health_coordinator import (",
    ),
    (
        "app/coordination/p2p_recovery_daemon.py",
        "from scripts.p2p.partition_healer import trigger_partition_healing",
    ),
    (
        "app/coordination/socket_leak_recovery_daemon.py",
        "from scripts.p2p.connection_pool import get_connection_pool",
    ),
    (
        "app/coordination/sync_pull_mixin.py",
        "from scripts.lib.transfer import robust_pull, TransferConfig",
    ),
    (
        "app/coordination/sync_push_mixin.py",
        "from scripts.lib.transfer import robust_push, TransferConfig",
    ),
    (
        "app/coordination/voter_config_orchestrator.py",
        "from scripts.p2p.managers.voter_config_manager import get_voter_config_manager",
    ),
}


def _find_layer_violations(app_root: Path) -> list[tuple[Path, int, str]]:
    violations: list[tuple[Path, int, str]] = []

    if shutil.which("rg"):
        result = subprocess.run(
            ["rg", "-n", r"^\s*(from|import)\s+scripts(\.|\\b)", str(app_root)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout:
            for line in result.stdout.strip().splitlines():
                path_str, lineno_str, content = line.split(":", 2)
                violations.append((Path(path_str), int(lineno_str), content.strip()))
        return violations

    for path in app_root.rglob("*.py"):
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if _IMPORT_RE.match(line):
                violations.append((path, lineno, line.strip()))
    return violations


def main() -> int:
    ai_service_root = Path(__file__).resolve().parents[1]
    app_root = ai_service_root / "app"
    violations = _find_layer_violations(app_root)
    unexpected: list[tuple[Path, int, str]] = []

    for path, lineno, module in violations:
        rel_path = path.relative_to(ai_service_root)
        if (rel_path.as_posix(), module) not in _ALLOWED_EXISTING_VIOLATIONS:
            unexpected.append((path, lineno, module))

    if not unexpected:
        return 0

    print("New layer violations detected (app -> scripts):")
    for path, lineno, module in sorted(unexpected):
        rel_path = path.relative_to(ai_service_root)
        print(f"- {rel_path}:{lineno}: {module}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
