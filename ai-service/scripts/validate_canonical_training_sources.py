#!/usr/bin/env python3
"""
Validate that training replay DBs referenced in pipelines have canonical status.

This script parses TRAINING_DATA_REGISTRY.md to determine whether referenced
replay databases have been validated (canonical status) before training.

Usage:
    cd ai-service
    python scripts/validate_canonical_training_sources.py \
        --registry TRAINING_DATA_REGISTRY.md \
        --db data/games/canonical_square8.db

Programmatic usage:
    from scripts.validate_canonical_training_sources import validate_canonical_sources
    result = validate_canonical_sources(registry_path, [db_path])
    if not result["ok"]:
        for problem in result["problems"]:
            print(problem)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional


def _parse_registry(registry_path: Path) -> Dict[str, Dict[str, str]]:
    """Parse TRAINING_DATA_REGISTRY.md and return DB info keyed by filename.

    Returns a dict like:
        {
            "canonical_square8.db": {"status": "canonical", "gate_summary": "..."},
            ...
        }
    """
    if not registry_path.exists():
        return {}

    content = registry_path.read_text(encoding="utf-8")
    result: Dict[str, Dict[str, str]] = {}

    # Match markdown table rows with format:
    # | `db_name.db` | board | players | **status** | gate_summary | notes |
    # The status may be wrapped in ** for bold
    table_row_pattern = re.compile(
        r"^\s*\|\s*`?([^`|]+\.db)`?\s*\|"  # DB name (may have backticks)
        r"\s*([^|]*)\|"  # Board type
        r"\s*([^|]*)\|"  # Players
        r"\s*\*?\*?([^|*]+)\*?\*?\s*\|"  # Status (may be bold **)
        r"\s*([^|]*)\|"  # Gate summary
        r"\s*([^|]*)\|",  # Notes
        re.MULTILINE,
    )

    for match in table_row_pattern.finditer(content):
        db_name = match.group(1).strip()
        status = match.group(4).strip().lower()
        gate_summary = match.group(5).strip()

        # Skip header row
        if db_name.lower() == "database" or status == "status":
            continue

        result[db_name] = {
            "status": status,
            "gate_summary": gate_summary,
        }

    return result


def _load_gate_summary(
    registry_dir: Path, gate_summary_name: str
) -> Optional[Dict]:
    """Load a gate summary JSON file relative to the registry directory."""
    if not gate_summary_name or gate_summary_name == "-":
        return None

    summary_path = registry_dir / gate_summary_name
    if not summary_path.exists():
        return None

    try:
        return json.loads(summary_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def validate_canonical_sources(
    registry_path: Path,
    db_paths: List[Path],
    *,
    allowed_statuses: Optional[List[str]] = None,
) -> Dict:
    """Validate that all referenced DBs have allowed canonical status.

    Args:
        registry_path: Path to TRAINING_DATA_REGISTRY.md
        db_paths: List of DB paths to validate
        allowed_statuses: List of allowed status values. Defaults to ["canonical"].

    Returns:
        Dict with:
            - ok: bool - True if all DBs pass validation
            - problems: List[str] - List of problem descriptions
            - checked: List[str] - List of successfully validated DB names
    """
    if allowed_statuses is None:
        allowed_statuses = ["canonical"]

    # Normalize to lowercase
    allowed_statuses = [s.lower() for s in allowed_statuses]

    problems: List[str] = []
    checked: List[str] = []

    # Parse the registry
    registry_info = _parse_registry(registry_path)
    registry_dir = registry_path.parent

    for db_path in db_paths:
        db_name = db_path.name

        # Check if DB is in registry
        if db_name not in registry_info:
            problems.append(
                f"Database '{db_name}' not found in registry {registry_path}"
            )
            continue

        info = registry_info[db_name]
        status = info.get("status", "").lower()

        # Check if status is allowed
        if status not in allowed_statuses:
            problems.append(
                f"Database '{db_name}' has status '{status}' "
                f"(allowed: {allowed_statuses})"
            )
            continue

        # If there's a gate summary, optionally validate it
        gate_summary_name = info.get("gate_summary", "")
        if gate_summary_name and gate_summary_name != "-":
            gate_data = _load_gate_summary(registry_dir, gate_summary_name)
            if gate_data:
                # Check parity gate if present
                parity_gate = gate_data.get("parity_gate", {})
                if parity_gate and not parity_gate.get(
                    "passed_canonical_parity_gate", True
                ):
                    problems.append(
                        f"Database '{db_name}' failed parity gate "
                        f"(gate_summary: {gate_summary_name})"
                    )
                    continue

        checked.append(db_name)

    return {
        "ok": len(problems) == 0,
        "problems": problems,
        "checked": checked,
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate canonical training sources against registry"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        required=True,
        help="Path to TRAINING_DATA_REGISTRY.md",
    )
    parser.add_argument(
        "--db",
        type=Path,
        action="append",
        dest="dbs",
        default=[],
        help="DB path to validate (can be specified multiple times)",
    )
    parser.add_argument(
        "--allow-status",
        action="append",
        dest="allowed_statuses",
        default=None,
        help="Allowed status values (default: canonical only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output result as JSON",
    )

    args = parser.parse_args(argv)

    if not args.dbs:
        parser.error("At least one --db path is required")

    result = validate_canonical_sources(
        args.registry,
        args.dbs,
        allowed_statuses=args.allowed_statuses,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["ok"]:
            print("Validation passed:")
            for db in result["checked"]:
                print(f"  ✓ {db}")
        else:
            print("Validation failed:")
            for problem in result["problems"]:
                print(f"  ✗ {problem}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
