#!/usr/bin/env python
"""
Model naming migration script.

Standardizes model file names to follow convention:
- Production: canonical_{board}_{N}p.pth
- Versioned: {board}_{N}p_v{version}.pth

Creates symlinks for backward compatibility.

Usage:
    # Dry run (preview changes)
    python scripts/migrate_model_names.py --dry-run

    # Execute migration
    python scripts/migrate_model_names.py --execute

    # Also archive old names
    python scripts/migrate_model_names.py --execute --archive
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
from datetime import datetime
from pathlib import Path


# Mapping of board type aliases to canonical names
BOARD_ALIASES = {
    "sq8": "square8",
    "sq19": "square19",
    "hex8": "hex8",
    "hexagonal": "hexagonal",
    "hex": "hex8",  # ambiguous but common
    "square8": "square8",
    "square19": "square19",
}

# Regex patterns to extract board type and player count
PATTERNS = [
    # canonical_sq8_2p.pth, canonical_square8_4p.pth
    re.compile(r"canonical_(?P<board>sq8|sq19|hex8|hexagonal|square8|square19)_(?P<players>\d)p"),
    # ringrift_hex8_2p.pth
    re.compile(r"ringrift_(?P<board>sq8|sq19|hex8|hexagonal|square8|square19)_(?P<players>\d)p"),
    # sq8_2p_v3.pth, hex8_4p_v2.pth
    re.compile(r"(?P<board>sq8|sq19|hex8|hexagonal|square8|square19)_(?P<players>\d)p_v\d+"),
    # distilled_sq8_2p_v7.pth
    re.compile(r"distilled_(?P<board>sq8|sq19|hex8|hexagonal)_(?P<players>\d)p"),
    # policy_sq8_4p_20251219.pth
    re.compile(r"policy_(?P<board>sq8|sq19|hex8|hexagonal)_(?P<players>\d)p"),
    # hex8_2p.pth (simple)
    re.compile(r"^(?P<board>sq8|sq19|hex8|hexagonal|square8|square19)_(?P<players>\d)p\.pth$"),
]


def parse_model_name(filename: str) -> tuple[str | None, int | None, str | None]:
    """Extract board type, player count, and variant from model filename.

    Returns:
        (board_type, num_players, variant) or (None, None, None) if unrecognized
    """
    for pattern in PATTERNS:
        match = pattern.search(filename)
        if match:
            board = BOARD_ALIASES.get(match.group("board"), match.group("board"))
            players = int(match.group("players"))

            # Determine variant
            if filename.startswith("canonical_"):
                variant = "canonical"
            elif filename.startswith("distilled_"):
                variant = "distilled"
            elif filename.startswith("policy_"):
                variant = "policy"
            elif "_v" in filename:
                # Extract version
                v_match = re.search(r"_v(\d+[a-z]?)", filename)
                variant = f"v{v_match.group(1)}" if v_match else "versioned"
            else:
                variant = "unknown"

            return board, players, variant

    return None, None, None


def get_canonical_name(board: str, players: int) -> str:
    """Generate canonical model filename."""
    return f"canonical_{board}_{players}p.pth"


def scan_models(models_dir: Path) -> list[dict]:
    """Scan models directory and categorize files."""
    results = []

    for path in models_dir.glob("*.pth"):
        filename = path.name
        board, players, variant = parse_model_name(filename)

        results.append({
            "path": path,
            "filename": filename,
            "board": board,
            "players": players,
            "variant": variant,
            "size_mb": path.stat().st_size / (1024 * 1024),
            "mtime": datetime.fromtimestamp(path.stat().st_mtime),
        })

    return results


def generate_migration_plan(models: list[dict]) -> list[dict]:
    """Generate migration plan with recommended actions."""
    plan = []

    # Group by board+players
    configs = {}
    for model in models:
        if model["board"] and model["players"]:
            key = (model["board"], model["players"])
            if key not in configs:
                configs[key] = []
            configs[key].append(model)

    # For each config, identify canonical and recommend migration
    for (board, players), group in configs.items():
        canonical_name = get_canonical_name(board, players)

        # Find existing canonical or latest
        canonical = None
        latest = None

        for m in group:
            if m["variant"] == "canonical":
                canonical = m
            if latest is None or m["mtime"] > latest["mtime"]:
                latest = m

        # Add to plan
        for m in group:
            action = "keep"
            new_name = None

            if m["variant"] == "canonical":
                # Already canonical, just verify name
                if m["filename"] != canonical_name:
                    action = "rename"
                    new_name = canonical_name
            elif m == latest and canonical is None:
                # This is the latest and no canonical exists
                action = "promote"
                new_name = canonical_name
            else:
                # Archive old versions
                action = "archive"

            plan.append({
                **m,
                "action": action,
                "new_name": new_name,
            })

    # Handle unrecognized models
    for model in models:
        if model["board"] is None:
            plan.append({
                **model,
                "action": "unrecognized",
                "new_name": None,
            })

    return plan


def execute_migration(plan: list[dict], models_dir: Path, archive: bool = False) -> None:
    """Execute the migration plan."""
    archive_dir = models_dir / "_archived" / datetime.now().strftime("%Y%m%d_%H%M%S")

    for item in plan:
        if item["action"] == "keep":
            print(f"  KEEP: {item['filename']}")

        elif item["action"] == "rename":
            old_path = item["path"]
            new_path = models_dir / item["new_name"]
            print(f"  RENAME: {item['filename']} -> {item['new_name']}")
            old_path.rename(new_path)
            # Create symlink for backward compatibility
            old_path.symlink_to(new_path)
            print(f"    + Created symlink: {item['filename']} -> {item['new_name']}")

        elif item["action"] == "promote":
            old_path = item["path"]
            new_path = models_dir / item["new_name"]
            print(f"  PROMOTE: {item['filename']} -> {item['new_name']}")
            shutil.copy2(old_path, new_path)
            print(f"    + Created canonical copy")

        elif item["action"] == "archive" and archive:
            archive_dir.mkdir(parents=True, exist_ok=True)
            old_path = item["path"]
            new_path = archive_dir / item["filename"]
            print(f"  ARCHIVE: {item['filename']} -> _archived/")
            shutil.move(str(old_path), str(new_path))

        elif item["action"] == "unrecognized":
            print(f"  SKIP (unrecognized): {item['filename']}")


def main():
    parser = argparse.ArgumentParser(description="Migrate model names to standard convention")
    parser.add_argument(
        "--models-dir",
        type=Path,
        default=Path(__file__).parent.parent / "models",
        help="Path to models directory",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without executing",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Execute the migration",
    )
    parser.add_argument(
        "--archive",
        action="store_true",
        help="Archive old model versions (with --execute)",
    )
    args = parser.parse_args()

    if not args.dry_run and not args.execute:
        print("Specify --dry-run to preview or --execute to run migration")
        return 1

    models_dir = args.models_dir.resolve()
    print(f"Scanning: {models_dir}\n")

    # Scan and categorize
    models = scan_models(models_dir)
    print(f"Found {len(models)} model files\n")

    # Generate plan
    plan = generate_migration_plan(models)

    # Display plan
    print("=" * 60)
    print("MIGRATION PLAN")
    print("=" * 60)

    # Group by config
    by_config = {}
    for item in plan:
        key = (item["board"], item["players"]) if item["board"] else ("unknown", 0)
        if key not in by_config:
            by_config[key] = []
        by_config[key].append(item)

    for (board, players), items in sorted(by_config.items()):
        if board == "unknown":
            print(f"\n[UNRECOGNIZED]")
        else:
            print(f"\n[{board}_{players}p]")

        for item in sorted(items, key=lambda x: x["filename"]):
            action_str = item["action"].upper()
            if item["new_name"]:
                print(f"  {action_str}: {item['filename']} -> {item['new_name']}")
            else:
                print(f"  {action_str}: {item['filename']}")

    # Summary
    print("\n" + "=" * 60)
    actions = {}
    for item in plan:
        actions[item["action"]] = actions.get(item["action"], 0) + 1
    print("SUMMARY:")
    for action, count in sorted(actions.items()):
        print(f"  {action}: {count}")

    # Execute if requested
    if args.execute:
        print("\n" + "=" * 60)
        print("EXECUTING MIGRATION")
        print("=" * 60)
        execute_migration(plan, models_dir, archive=args.archive)
        print("\nMigration complete!")
    else:
        print("\n[DRY RUN] No changes made. Use --execute to apply.")

    return 0


if __name__ == "__main__":
    exit(main())
