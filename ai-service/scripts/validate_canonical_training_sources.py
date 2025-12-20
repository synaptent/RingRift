#!/usr/bin/env python3
"""
Validate that training replay DBs referenced in pipelines have canonical status.

Usage:
    cd ai-service
    python scripts/validate_canonical_training_sources.py \
        --registry TRAINING_DATA_REGISTRY.md \
        --db data/games/canonical_square8.db
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from app.training.canonical_sources import (
    resolve_registry_path,
    validate_canonical_sources,
)

__all__ = ["validate_canonical_sources", "resolve_registry_path"]


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

    args = parser.parse_args(argv)
    if not args.dbs:
        parser.error("At least one --db path is required")

    registry_path = resolve_registry_path(args.registry)

    result = validate_canonical_sources(
        registry_path=registry_path,
        db_paths=args.dbs,
        allowed_statuses=args.allowed_statuses,
    )

    if args.json:
        print(json.dumps(result, indent=2))
    else:
        if result["ok"]:
            print(f"OK: All DBs are canonical: {', '.join(result['checked'])}")
        else:
            print("ERROR: Non-canonical DBs detected:")
            for problem in result["problems"]:
                print(f"  - {problem}")

    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
