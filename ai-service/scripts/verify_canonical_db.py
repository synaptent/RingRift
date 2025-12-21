#!/usr/bin/env python3
"""Canonical Database Verification CLI.

Lane 2 Completion - Provides CLI to verify canonical status and provenance
for any database path, enabling operational validation of training data.

Usage:
    # Check if a database is canonical
    python scripts/verify_canonical_db.py data/games/selfplay_square8_2p.db

    # Check multiple databases
    python scripts/verify_canonical_db.py data/games/*.db

    # Verbose output with provenance details
    python scripts/verify_canonical_db.py --verbose data/games/merged.db

    # Exit with error if any non-canonical
    python scripts/verify_canonical_db.py --strict data/games/*.db

    # JSON output for automation
    python scripts/verify_canonical_db.py --json data/games/*.db
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.training.canonical_sources import (
    is_canonical_source,
    get_source_metadata,
    CANONICAL_DB_PATTERNS,
)

try:
    from app.utils.canonical_naming import normalize_board_type, parse_config_key
    HAS_NAMING = True
except ImportError:
    HAS_NAMING = False


def get_db_provenance(db_path: Path) -> dict:
    """Get provenance information for a database.

    Returns dict with:
        - is_canonical: bool
        - board_type: extracted board type (if parseable)
        - num_players: extracted player count (if parseable)
        - source: inferred source (selfplay, tournament, etc.)
        - registry_entry: matching registry pattern (if any)
    """
    provenance = {
        "path": str(db_path),
        "exists": db_path.exists(),
        "is_canonical": False,
        "board_type": None,
        "num_players": None,
        "source": None,
        "registry_entry": None,
        "size_mb": None,
    }

    if not db_path.exists():
        return provenance

    # Get file size
    provenance["size_mb"] = round(db_path.stat().st_size / (1024 * 1024), 2)

    # Check canonical status
    provenance["is_canonical"] = is_canonical_source(str(db_path))

    # Get metadata if available
    metadata = get_source_metadata(str(db_path))
    if metadata:
        provenance["registry_entry"] = metadata

    # Parse filename for board type and player count
    filename = db_path.stem
    parts = filename.split("_")

    # Try to extract board type
    for part in parts:
        if part in ("square8", "square19", "hex8", "hexagonal"):
            provenance["board_type"] = part
            break
        if HAS_NAMING:
            try:
                provenance["board_type"] = normalize_board_type(part)
                break
            except ValueError:
                pass

    # Try to extract player count
    for part in parts:
        if part.endswith("p") and part[:-1].isdigit():
            provenance["num_players"] = int(part[:-1])
            break

    # Infer source from filename
    if "selfplay" in filename.lower():
        provenance["source"] = "selfplay"
    elif "tournament" in filename.lower():
        provenance["source"] = "tournament"
    elif "gauntlet" in filename.lower():
        provenance["source"] = "gauntlet"
    elif "canonical" in filename.lower():
        provenance["source"] = "canonical"
    elif "training" in filename.lower():
        provenance["source"] = "training"

    return provenance


def print_provenance(prov: dict, verbose: bool = False) -> None:
    """Print provenance information in human-readable format."""
    status = "✓ CANONICAL" if prov["is_canonical"] else "✗ NON-CANONICAL"
    print(f"{status}: {prov['path']}")

    if verbose:
        print(f"  Exists: {prov['exists']}")
        if prov["size_mb"]:
            print(f"  Size: {prov['size_mb']} MB")
        if prov["board_type"]:
            print(f"  Board Type: {prov['board_type']}")
        if prov["num_players"]:
            print(f"  Players: {prov['num_players']}")
        if prov["source"]:
            print(f"  Source: {prov['source']}")
        if prov["registry_entry"]:
            print(f"  Registry: {prov['registry_entry']}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Verify canonical status and provenance of training databases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s data/games/selfplay_square8_2p.db
  %(prog)s --verbose data/games/*.db
  %(prog)s --strict --json data/canonical_*.db
        """,
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Database paths to verify (supports glob patterns)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Show detailed provenance information",
    )
    parser.add_argument(
        "-s", "--strict",
        action="store_true",
        help="Exit with error code if any database is non-canonical",
    )
    parser.add_argument(
        "-j", "--json",
        action="store_true",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Show summary statistics at the end",
    )

    args = parser.parse_args()

    # Expand paths (handle glob patterns)
    all_paths = []
    for pattern in args.paths:
        p = Path(pattern)
        if p.exists():
            all_paths.append(p)
        else:
            # Try glob
            matches = list(Path(".").glob(pattern))
            if matches:
                all_paths.extend(matches)
            else:
                # File doesn't exist
                all_paths.append(p)

    if not all_paths:
        print("No database files found", file=sys.stderr)
        return 1

    # Check each database
    results = []
    canonical_count = 0
    non_canonical_count = 0

    for db_path in sorted(all_paths):
        prov = get_db_provenance(db_path)
        results.append(prov)

        if prov["is_canonical"]:
            canonical_count += 1
        else:
            non_canonical_count += 1

        if not args.json:
            print_provenance(prov, args.verbose)

    # Output
    if args.json:
        output = {
            "databases": results,
            "summary": {
                "total": len(results),
                "canonical": canonical_count,
                "non_canonical": non_canonical_count,
            }
        }
        print(json.dumps(output, indent=2))
    elif args.summary or len(results) > 3:
        print(f"\nSummary: {canonical_count} canonical, {non_canonical_count} non-canonical")

    # Exit code for strict mode
    if args.strict and non_canonical_count > 0:
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
