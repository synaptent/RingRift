#!/usr/bin/env python
from __future__ import annotations

"""
Validate that training scripts and configs only reference canonical_* GameReplayDBs.

This is a lightweight safeguard around the policy documented in
TRAINING_DATA_REGISTRY.md: all new training runs must use canonical,
parity-gated DBs (typically named canonical_<board>.db).

Current behaviour:
  - Scans ai-service/scripts for obvious DB path literals.
  - Flags any *.db paths whose basename does NOT start with "canonical_".
  - Optionally lists legacy/experimental DBs under ai-service/data/games.

Usage (from ai-service/):

  PYTHONPATH=. python scripts/validate_canonical_training_sources.py

Exit codes:
  0 – no non-canonical *.db references found
  1 – at least one non-canonical *.db reference detected
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple


AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


DB_LITERAL_RE = re.compile(r"['\"]([^'\"]+\\.db)['\"]")


def _scan_script_for_dbs(path: Path) -> List[str]:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return []
    return [m.group(1) for m in DB_LITERAL_RE.finditer(text)]


def find_noncanonical_db_references() -> Dict[str, List[Tuple[str, str]]]:
    """
    Scan ai-service/scripts for .db string literals and return a mapping:
      script_path -> [(literal, basename), ...] for non-canonical basenames.
    """
    scripts_dir = AI_SERVICE_ROOT / "scripts"
    results: Dict[str, List[Tuple[str, str]]] = {}

    for path in sorted(scripts_dir.glob("*.py")):
        db_literals = _scan_script_for_dbs(path)
        bad: List[Tuple[str, str]] = []
        for lit in db_literals:
            basename = os.path.basename(lit)
            # Allow canonical_*.db and obvious test/fixture DBs.
            if basename.startswith("canonical_"):
                continue
            if basename in ("minimal_test.db", "training.db"):
                continue
            bad.append((lit, basename))
        if bad:
            results[str(path.relative_to(AI_SERVICE_ROOT))] = bad

    return results


def list_legacy_db_files() -> List[str]:
    """List non-canonical DB files under ai-service/data/games for awareness."""
    games_dir = AI_SERVICE_ROOT / "data" / "games"
    if not games_dir.exists():
        return []
    paths: List[str] = []
    for db_path in sorted(games_dir.glob("*.db")):
        basename = db_path.name
        if basename.startswith("canonical_"):
            continue
        paths.append(str(db_path))
    return paths


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that training-related scripts only reference canonical_*.db "
            "GameReplayDBs and surface any legacy DBs still present under data/games."
        )
    )
    parser.add_argument(
        "--list-legacy-dbs",
        action="store_true",
        help="Also list non-canonical *.db files under ai-service/data/games.",
    )
    args = parser.parse_args(argv)

    noncanonical_refs = find_noncanonical_db_references()
    legacy_dbs = list_legacy_db_files() if args.list_legacy_dbs else []

    if noncanonical_refs:
        print("[canonical-sources] Non-canonical DB references found in scripts:\n")
        for script, refs in noncanonical_refs.items():
            print(f"  {script}:")
            for lit, basename in refs:
                print(f"    literal={lit!r} (basename={basename})")
        print()
        print(
            "Update these scripts to:\n"
            "  - Use canonical_<board>.db for production training paths, or\n"
            "  - Clearly mark non-canonical DBs as test/legacy only."
        )

    if legacy_dbs:
        print("\n[canonical-sources] Non-canonical DB files under ai-service/data/games:")
        for p in legacy_dbs:
            print(f"  {p}")
        print(
            "\nUse scripts/scan_canonical_phase_dbs.py or targeted rm/mv commands "
            "to delete or archive these once you are confident they are no longer needed."
        )

    return 0 if not noncanonical_refs else 1


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    raise SystemExit(main())
