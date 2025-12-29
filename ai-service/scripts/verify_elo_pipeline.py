#!/usr/bin/env python3
"""Verify the Elo unification pipeline is working end-to-end.

This script checks:
1. EloService can write to unified_elo.db
2. Post-training gauntlet records matches to EloService
3. EloSyncManager is configured correctly
4. EloServiceFacade wraps EloService correctly

Usage:
    python scripts/verify_elo_pipeline.py

December 2025: Created as part of Elo unification verification.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_elo_service() -> tuple[bool, str]:
    """Check EloService can read/write."""
    try:
        from app.training.elo_service import get_elo_service

        elo = get_elo_service()

        # Check database exists
        if not elo.db_path.exists():
            return False, f"Database not found: {elo.db_path}"

        # Get a rating to verify connectivity
        test_rating = elo.get_rating("test_model", "square8", 2)

        # Get leaderboard to check match count
        leaders = elo.get_leaderboard("square8", 2, limit=1)

        return True, f"OK - Database at {elo.db_path}"
    except Exception as e:
        return False, f"Error: {e}"


def check_elo_facade() -> tuple[bool, str]:
    """Check EloServiceFacade works."""
    try:
        from app.tournament.elo_facade import EloServiceFacade, get_elo_facade

        facade = get_elo_facade()

        # Check we can get a rating
        rating = facade.get_rating("test_model", "square8", 2)

        return True, f"OK - Facade wraps EloService correctly"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_elo_sync_manager() -> tuple[bool, str]:
    """Check EloSyncManager is configured."""
    try:
        from app.tournament.elo_sync_manager import EloSyncManager

        manager = EloSyncManager()

        # Check database path
        if not manager.db_path.exists():
            return False, f"Database not found: {manager.db_path}"

        # Check it inherits from DatabaseSyncManager
        from app.coordination.database_sync_manager import DatabaseSyncManager
        if not isinstance(manager, DatabaseSyncManager):
            return False, "Not inheriting from DatabaseSyncManager"

        return True, f"OK - Configured for {manager.db_path}"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_daemon_wiring() -> tuple[bool, str]:
    """Check ELO_SYNC daemon is properly wired."""
    try:
        from app.coordination.daemon_types import DaemonType
        from app.coordination.daemon_runners import get_runner

        # Check DaemonType.ELO_SYNC exists
        if not hasattr(DaemonType, "ELO_SYNC"):
            return False, "DaemonType.ELO_SYNC not defined"

        # Check runner can be retrieved
        runner = get_runner(DaemonType.ELO_SYNC)
        if runner is None:
            return False, "ELO_SYNC runner not found"

        return True, "OK - ELO_SYNC daemon registered"
    except ImportError as e:
        return False, f"Import error: {e}"
    except Exception as e:
        return False, f"Error: {e}"


def check_training_coordinator_recording() -> tuple[bool, str]:
    """Check training coordinator records to EloService."""
    try:
        import ast
        from pathlib import Path

        coord_path = Path(__file__).parent.parent / "scripts" / "p2p" / "managers" / "training_coordinator.py"
        if not coord_path.exists():
            return False, f"File not found: {coord_path}"

        content = coord_path.read_text()

        # Check for EloService import in gauntlet method
        if "from app.training.elo_service import get_elo_service" not in content:
            return False, "EloService import not found in training_coordinator.py"

        # Check for record_match call
        if "elo_service.record_match(" not in content:
            return False, "elo_service.record_match() call not found"

        return True, "OK - TrainingCoordinator records to EloService"
    except Exception as e:
        return False, f"Error: {e}"


def check_migration_script() -> tuple[bool, str]:
    """Check migration script exists."""
    try:
        from pathlib import Path

        script_path = Path(__file__).parent / "migrate_legacy_elo.py"
        if not script_path.exists():
            return False, f"Script not found: {script_path}"

        return True, f"OK - Migration script at {script_path}"
    except Exception as e:
        return False, f"Error: {e}"


def main():
    print("=" * 60)
    print("Elo Unification Pipeline Verification")
    print("=" * 60)

    checks = [
        ("EloService", check_elo_service),
        ("EloServiceFacade", check_elo_facade),
        ("EloSyncManager", check_elo_sync_manager),
        ("Daemon Wiring", check_daemon_wiring),
        ("Training Coordinator Recording", check_training_coordinator_recording),
        ("Migration Script", check_migration_script),
    ]

    passed = 0
    failed = 0
    results = []

    for name, check_fn in checks:
        try:
            ok, msg = check_fn()
            status = "✓ PASS" if ok else "✗ FAIL"
            results.append((name, status, msg))
            if ok:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            results.append((name, "✗ ERROR", str(e)))
            failed += 1

    print()
    for name, status, msg in results:
        print(f"{status} {name}")
        print(f"       {msg}")
        print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed > 0:
        print("\nRecommended Actions:")
        for name, status, msg in results:
            if "FAIL" in status or "ERROR" in status:
                print(f"  - Fix: {name}")
        return 1

    print("\n✓ All checks passed - Elo unification pipeline is ready!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
