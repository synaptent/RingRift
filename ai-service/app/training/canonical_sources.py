"""Canonical Training Data Sources - Registry and validation utilities.

This module provides utilities for working with TRAINING_DATA_REGISTRY.md,
which is the source of truth for which game databases are approved for
training. It handles:

1. **Registry Parsing**: Extract database status from the markdown table
2. **Validation**: Check if databases meet quality gates (parity, move data)
3. **Discovery**: Find canonical databases by board type and player count
4. **Cluster Support**: Handle pending_gate status for cluster nodes without npx

Key Functions:
    parse_registry(path)
        Parse TRAINING_DATA_REGISTRY.md and return database info.

    get_canonical_db_path(board_type, num_players)
        Get the canonical database path for a configuration.

    validate_db_for_training(db_path, allow_pending_gate=False)
        Check if a database is approved for training.

Environment Variables:
    RINGRIFT_ALLOW_PENDING_GATE
        Set to "1" or "true" to allow databases with pending_gate status.
        Needed on cluster nodes that lack npx for parity validation.

Example:
    from app.training.canonical_sources import (
        get_canonical_db_path,
        validate_db_for_training,
    )

    # Get canonical database for hex8 2-player
    db_path = get_canonical_db_path("hex8", 2)

    # Validate it's approved for training
    is_valid, reason = validate_db_for_training(db_path)
    if not is_valid:
        raise ValueError(f"Database not approved: {reason}")

See Also:
    - TRAINING_DATA_REGISTRY.md for the canonical source registry
    - scripts/export_replay_dataset.py for training data export
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from collections.abc import Iterable
from pathlib import Path

logger = logging.getLogger(__name__)

AI_SERVICE_ROOT = Path(__file__).resolve().parents[2]

# Environment variable to allow pending_gate databases on cluster nodes without npx
ALLOW_PENDING_GATE_ENV = os.environ.get("RINGRIFT_ALLOW_PENDING_GATE", "").lower() in ("1", "true", "yes")


def resolve_registry_path(registry_path: Path | None = None) -> Path:
    """Resolve TRAINING_DATA_REGISTRY.md path with a repo-local default."""
    return registry_path or (AI_SERVICE_ROOT / "TRAINING_DATA_REGISTRY.md")


def parse_registry(registry_path: Path) -> dict[str, dict[str, str]]:
    """Parse TRAINING_DATA_REGISTRY.md and return DB info keyed by filename."""
    if not registry_path.exists():
        return {}

    content = registry_path.read_text(encoding="utf-8")
    result: dict[str, dict[str, str]] = {}

    table_row_pattern = re.compile(
        r"^\s*\|\s*`?([^`|]+\.db)`?\s*\|"
        r"\s*([^|]*)\|"
        r"\s*([^|]*)\|"
        r"\s*\*?\*?([^|*]+)\*?\*?\s*\|"
        r"\s*([^|]*)\|"
        r"\s*([^|]*)\|",
        re.MULTILINE,
    )

    for match in table_row_pattern.finditer(content):
        db_name = match.group(1).strip()
        status = match.group(4).strip().lower()
        gate_summary = match.group(5).strip()

        if db_name.lower() == "database" or status == "status":
            continue

        result[db_name] = {
            "status": status,
            "gate_summary": gate_summary,
        }

    return result


def load_gate_summary(registry_dir: Path, gate_summary_name: str) -> dict | None:
    """Load a gate summary JSON file relative to the registry directory."""
    if not gate_summary_name or gate_summary_name == "-":
        return None

    summary_path = registry_dir / gate_summary_name
    if not summary_path.exists():
        candidate = Path(gate_summary_name)
        if candidate.parent == Path("."):
            summary_path = registry_dir / "data" / "games" / gate_summary_name
        if not summary_path.exists():
            return None

    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def check_database_parity_status(db_path: Path) -> tuple[bool, str]:
    """Check if database has parity_status column and all games passed.

    This is a fallback for when gate summary files aren't available
    (e.g., on cluster nodes without npx).

    Args:
        db_path: Path to database file

    Returns:
        (passed, reason) tuple
    """
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Check if parity_status column exists
        cursor.execute("PRAGMA table_info(games)")
        columns = [row[1] for row in cursor.fetchall()]
        if "parity_status" not in columns:
            conn.close()
            return False, "parity_status column not found"

        # Check game counts by parity status
        cursor.execute("""
            SELECT parity_status, COUNT(*) as count
            FROM games
            WHERE excluded_from_training IS NULL OR excluded_from_training = 0
            GROUP BY parity_status
        """)
        status_counts = dict(cursor.fetchall())
        conn.close()

        total = sum(status_counts.values())
        passed = status_counts.get("passed", 0)
        failed = status_counts.get("failed", 0)
        pending = status_counts.get("pending", 0) + status_counts.get(None, 0)

        if total == 0:
            return False, "no games in database"

        if passed == total:
            return True, f"all {total} games passed parity"

        if failed > 0:
            return False, f"{failed}/{total} games failed parity"

        if pending > 0:
            return False, f"{pending}/{total} games pending parity validation"

        return False, f"unknown parity status distribution: {status_counts}"

    except Exception as e:
        return False, f"error checking parity: {e}"


def validate_canonical_sources(
    registry_path: Path,
    db_paths: Iterable[Path],
    *,
    allowed_statuses: list[str] | None = None,
    allow_pending_gate: bool = False,
    check_database_parity: bool = True,
) -> dict:
    """Validate that all referenced DBs have allowed canonical status.

    Args:
        registry_path: Path to TRAINING_DATA_REGISTRY.md
        db_paths: Database paths to validate
        allowed_statuses: List of allowed status values (default: ["canonical"])
        allow_pending_gate: If True, also allow "pending_gate" status databases.
            Can also be set via RINGRIFT_ALLOW_PENDING_GATE env var.
        check_database_parity: If True (default), check database parity_status
            column as fallback when gate summary files aren't available.

    Returns:
        Dict with "ok", "problems", and "checked" keys
    """
    if allowed_statuses is None:
        allowed_statuses = ["canonical"]

    allowed_statuses = [s.lower() for s in allowed_statuses]

    # Allow pending_gate status if explicitly requested or via env var
    if (allow_pending_gate or ALLOW_PENDING_GATE_ENV) and "pending_gate" not in allowed_statuses:
        allowed_statuses.append("pending_gate")
        logger.info(
            "[canonical-sources] Including 'pending_gate' databases "
            "(RINGRIFT_ALLOW_PENDING_GATE or allow_pending_gate=True)"
        )

    problems: list[str] = []
    checked: list[str] = []

    registry_info = parse_registry(registry_path)
    registry_dir = registry_path.parent

    # Convert db_paths to list and keep original paths for fallback checks
    db_paths_list = list(db_paths)

    for db_path in db_paths_list:
        db_path = Path(db_path)
        db_name = db_path.name

        if db_name not in registry_info:
            problems.append(
                f"Database '{db_name}' not found in registry {registry_path}"
            )
            continue

        info = registry_info[db_name]
        status = info.get("status", "").lower()

        if status not in allowed_statuses:
            problems.append(
                f"Database '{db_name}' has status '{status}' "
                f"(allowed: {allowed_statuses})"
            )
            continue

        gate_summary_name = info.get("gate_summary", "")
        gate_passed = True  # Assume passed unless proven otherwise

        if gate_summary_name and gate_summary_name != "-":
            gate_data = load_gate_summary(registry_dir, gate_summary_name)
            if gate_data:
                if "canonical_ok" in gate_data:
                    gate_passed = bool(gate_data.get("canonical_ok"))
                else:
                    parity_gate = gate_data.get("parity_gate", {})
                    if parity_gate:
                        gate_passed = bool(parity_gate.get("passed_canonical_parity_gate", True))
                    else:
                        gate_passed = bool(gate_data.get("passed_canonical_parity_gate", True))

                if not gate_passed:
                    # Gate summary exists but shows failure - try database fallback
                    if check_database_parity and db_path.exists():
                        db_passed, reason = check_database_parity_status(db_path)
                        if db_passed:
                            logger.info(
                                f"[canonical-sources] Database '{db_name}' gate summary failed "
                                f"but database parity_status passed: {reason}"
                            )
                            gate_passed = True

                    if not gate_passed:
                        problems.append(
                            f"Database '{db_name}' failed canonical gate "
                            f"(gate_summary: {gate_summary_name})"
                        )
                        continue
            else:
                # Gate summary file not found - try database fallback for pending_gate
                if status == "pending_gate" and check_database_parity and db_path.exists():
                    db_passed, reason = check_database_parity_status(db_path)
                    if db_passed:
                        logger.info(
                            f"[canonical-sources] Database '{db_name}' validated via "
                            f"database parity_status: {reason}"
                        )
                    else:
                        logger.debug(
                            f"[canonical-sources] Database '{db_name}' pending_gate, "
                            f"database check: {reason}"
                        )

        checked.append(db_name)

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "checked": checked,
    }


def enforce_canonical_sources(
    db_paths: Iterable[Path],
    *,
    registry_path: Path | None = None,
    allowed_statuses: list[str] | None = None,
    allow_noncanonical: bool = False,
    allow_pending_gate: bool = False,
    check_database_parity: bool = True,
    error_prefix: str = "canonical-source",
) -> None:
    """Raise SystemExit if any DBs are not canonical by registry policy.

    Args:
        db_paths: Database paths to validate
        registry_path: Path to TRAINING_DATA_REGISTRY.md
        allowed_statuses: List of allowed status values
        allow_noncanonical: Skip all validation if True
        allow_pending_gate: Allow "pending_gate" status databases
        check_database_parity: Check database parity_status column as fallback
        error_prefix: Prefix for error messages
    """
    if allow_noncanonical:
        return

    registry = resolve_registry_path(registry_path)
    result = validate_canonical_sources(
        registry_path=registry,
        db_paths=list(db_paths),
        allowed_statuses=allowed_statuses,
        allow_pending_gate=allow_pending_gate,
        check_database_parity=check_database_parity,
    )
    if result.get("ok"):
        return

    issues = result.get("problems", [])
    details = "\n".join(f"- {issue}" for issue in issues) if issues else "Unknown issue"
    raise SystemExit(f"[{error_prefix}] Canonical source validation failed:\n{details}")
