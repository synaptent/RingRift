#!/usr/bin/env python3
"""
Validate that training replay DBs referenced in pipelines have canonical status.

Usage:
    cd ai-service
    python scripts/validate_canonical_training_sources.py \
        --registry TRAINING_DATA_REGISTRY.md \
        --db data/games/canonical_square8_2p.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.training.canonical_sources import (
    load_gate_summary,
    parse_registry,
    resolve_registry_path,
    validate_canonical_sources,
)

__all__ = ["resolve_registry_path", "validate_canonical_sources"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate canonical training sources against registry"
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to TRAINING_DATA_REGISTRY.md (default: repo root)",
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
        help="Output JSON result",
    )
    parser.add_argument(
        "--details",
        action="store_true",
        help="Include registry and gate summary details per DB",
    )

    args = parser.parse_args(argv)
    if not args.dbs:
        parser.error("At least one --db path is required")

    registry_path = resolve_registry_path(args.registry)

    result = validate_canonical_sources(
        registry_path=registry_path,
        db_paths=args.dbs,
        allowed_statuses=args.allowed_statuses,
    )

    if args.details:
        registry_info = parse_registry(registry_path)
        registry_dir = registry_path.parent
        details = []
        for db_path in args.dbs:
            db_name = Path(db_path).name
            info = registry_info.get(db_name, {})
            gate_summary = info.get("gate_summary", "")
            gate_data = load_gate_summary(registry_dir, gate_summary) if gate_summary else None
            details.append(
                {
                    "db": str(db_path),
                    "name": db_name,
                    "status": info.get("status"),
                    "gate_summary": gate_summary,
                    "canonical_ok": gate_data.get("canonical_ok") if gate_data else None,
                    "parity_gate": gate_data.get("parity_gate") if gate_data else None,
                }
            )
        result["details"] = details

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["ok"]:
            print(f"OK: All DBs are canonical: {', '.join(result['checked'])}")
            if args.details:
                for detail in result.get("details", []):
                    print(
                        f"  - {detail['name']}: status={detail.get('status')} "
                        f"gate_summary={detail.get('gate_summary')}"
                    )
        else:
            print("ERROR: Non-canonical DBs detected:")
            for problem in result["problems"]:
                print(f"  - {problem}")
            if args.details:
                for detail in result.get("details", []):
                    print(
                        f"  - {detail['name']}: status={detail.get('status')} "
                        f"gate_summary={detail.get('gate_summary')}"
                    )

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
