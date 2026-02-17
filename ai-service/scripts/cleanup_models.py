#!/usr/bin/env python3
"""Model archive cleanup with rotation policy.

Keeps the current canonical models and archives/removes intermediate checkpoints.

Rotation policy:
- Keep: active canonical models (canonical_{board}_{n}p.pth) + symlinks
- Keep: latest 2 timestamped checkpoints per config pattern
- Archive: older timestamped checkpoints
- Archive: intermediate models (quality_*, ab_test_*, validated_*, etc.)
- Delete: orphaned .sha256 files (no matching .pth)
- Delete: dot-prefixed trash files
- Delete: broken symlinks

Usage:
    python scripts/cleanup_models.py              # Dry run (default)
    python scripts/cleanup_models.py --execute    # Actually perform cleanup
    python scripts/cleanup_models.py --keep 3     # Keep 3 most recent per pattern
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
ARCHIVE_DIR = MODELS_DIR / "archive"

# Active canonical models - never touch these
ACTIVE_CANONICAL = {
    "canonical_hex8_2p.pth",
    "canonical_hex8_3p.pth",
    "canonical_hex8_4p.pth",
    "canonical_hexagonal_2p.pth",
    "canonical_hexagonal_3p.pth",
    "canonical_hexagonal_4p.pth",
    "canonical_square8_2p.pth",
    "canonical_square8_3p.pth",
    "canonical_square8_4p.pth",
    "canonical_square19_2p.pth",
    "canonical_square19_3p.pth",
    "canonical_square19_4p.pth",
}

# Also keep non-timestamped variant models (v5-heavy, v2, v5, etc.)
KEEP_VARIANT_PATTERN = re.compile(
    r"^canonical_(hex8|hexagonal|square8|square19)_[234]p_(v[0-9][-\w]*)\.pth$"
)

# Timestamped canonical pattern: canonical_{config}_YYYYMMDD_HHMMSS.pth
TIMESTAMPED_PATTERN = re.compile(
    r"^canonical_(.*?)_(\d{8}_\d{6})\.pth$"
)

# Intermediate model patterns (safe to archive)
INTERMEDIATE_PATTERNS = [
    re.compile(r"^quality_.*\.pth$"),
    re.compile(r"^ab_test_.*\.pth$"),
    re.compile(r"^hex\d+_\d+p_validated.*\.pth$"),
    re.compile(r"^hex\d+_\d+p_transfer.*\.pth$"),
    re.compile(r"^hex\d+_\d+p_hybrid.*\.pth$"),
    re.compile(r"^hexagonal_\d+p_(cluster|retrained).*\.pth$"),
    re.compile(r"^square\d+_\d+p_retrained.*\.pth$"),
    re.compile(r"^transfer_.*\.pth$"),
]


def classify_models(models_dir: Path, keep_n: int) -> dict:
    """Classify all files in models directory into action categories."""
    actions = {
        "keep": [],
        "archive": [],
        "delete": [],
        "delete_symlinks": [],
    }

    # Group timestamped models by config pattern
    timestamped_groups: dict[str, list[tuple[str, str]]] = defaultdict(list)

    for entry in sorted(models_dir.iterdir()):
        if entry.name in ("archive", "archived", ".gitkeep"):
            continue

        name = entry.name

        # --- Broken symlinks ---
        if entry.is_symlink() and not entry.exists():
            actions["delete_symlinks"].append(entry)
            continue

        # --- Symlinks pointing to files we'll handle ---
        if entry.is_symlink():
            target = os.readlink(entry)
            # Symlinks to active canonical models: keep
            if target in ACTIVE_CANONICAL or KEEP_VARIANT_PATTERN.match(target or ""):
                actions["keep"].append(entry)
            else:
                # Will be handled after we decide on targets
                continue  # defer symlink decisions
            continue

        # --- Dot-prefixed trash files ---
        if name.startswith("."):
            actions["delete"].append(entry)
            continue

        # --- Orphaned .sha256 files ---
        if name.endswith(".sha256"):
            base = name[:-7]  # Remove .sha256
            if not (models_dir / base).exists():
                actions["delete"].append(entry)
            else:
                actions["keep"].append(entry)
            continue

        # --- Active canonical models ---
        if name in ACTIVE_CANONICAL:
            actions["keep"].append(entry)
            continue

        # --- Non-timestamped variant models ---
        if KEEP_VARIANT_PATTERN.match(name):
            actions["keep"].append(entry)
            continue

        # --- Timestamped canonical checkpoints ---
        ts_match = TIMESTAMPED_PATTERN.match(name)
        if ts_match:
            config_pattern = ts_match.group(1)
            timestamp = ts_match.group(2)
            timestamped_groups[config_pattern].append((timestamp, name))
            continue

        # --- Intermediate models ---
        if any(p.match(name) for p in INTERMEDIATE_PATTERNS):
            actions["archive"].append(entry)
            continue

        # --- Other .pth files not matching any pattern ---
        if name.endswith(".pth"):
            actions["archive"].append(entry)
            continue

        # --- Everything else (keep) ---
        actions["keep"].append(entry)

    # Process timestamped groups: keep latest N, archive the rest
    for config_pattern, entries in sorted(timestamped_groups.items()):
        entries.sort(key=lambda x: x[0], reverse=True)  # newest first
        for i, (timestamp, filename) in enumerate(entries):
            filepath = models_dir / filename
            if i < keep_n:
                actions["keep"].append(filepath)
            else:
                actions["archive"].append(filepath)

    # Now handle symlinks to archived/deleted files
    archived_names = {f.name for f in actions["archive"]}
    deleted_names = {f.name for f in actions["delete"]}
    for entry in sorted(models_dir.iterdir()):
        if not entry.is_symlink() or not entry.exists():
            continue
        if entry in actions["keep"]:
            continue
        target = os.readlink(entry)
        if target in archived_names or target in deleted_names:
            actions["delete_symlinks"].append(entry)
        elif entry not in actions["keep"]:
            # Symlink to something we're keeping
            target_path = models_dir / target
            if target_path.exists() and target_path not in actions["archive"]:
                actions["keep"].append(entry)
            else:
                actions["delete_symlinks"].append(entry)

    return actions


def format_size(size_bytes: int) -> str:
    """Format bytes as human-readable size."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def get_file_size(path: Path) -> int:
    """Get file size, returning 0 for symlinks."""
    if path.is_symlink():
        return 0
    try:
        return path.stat().st_size
    except OSError:
        return 0


def get_protected_models() -> set[str]:
    """Get model stems protected by Elo-based model protection (top 50%).

    Returns a set of model ID stems that should NOT be archived because
    they are in the top 50% by Elo for their config.
    """
    protected = set()
    try:
        from app.tournament.model_culling import get_culling_controller, CONFIG_KEYS
        controller = get_culling_controller()
        for config_key in CONFIG_KEYS:
            config_protected = controller.get_protected_model_set(config_key)
            # Extract just the nn_id/stem part for file matching
            for pid in config_protected:
                if ":" in pid:
                    protected.add(pid.split(":")[0])
                else:
                    protected.add(pid)
    except Exception as e:
        # Non-fatal - if we can't load protection, proceed without it
        print(f"  Note: Could not load Elo protection set: {e}")
    return protected


def main():
    parser = argparse.ArgumentParser(description="Clean up model archives")
    parser.add_argument(
        "--execute", action="store_true",
        help="Actually perform cleanup (default is dry run)",
    )
    parser.add_argument(
        "--keep", type=int, default=2,
        help="Number of most recent timestamped checkpoints to keep per config (default: 2)",
    )
    args = parser.parse_args()

    if not MODELS_DIR.exists():
        print(f"Models directory not found: {MODELS_DIR}")
        sys.exit(1)

    actions = classify_models(MODELS_DIR, args.keep)

    # Feb 2026: Check Elo-protected models before archiving
    protected_stems = get_protected_models()
    if protected_stems:
        elo_protected = []
        remaining_archive = []
        for f in actions["archive"]:
            stem = f.stem
            if stem in protected_stems:
                elo_protected.append(f)
            else:
                remaining_archive.append(f)
        if elo_protected:
            print(f"Elo-protected from archival: {len(elo_protected)} models (top 50% by Elo)")
            actions["keep"].extend(elo_protected)
            actions["archive"] = remaining_archive

    # Calculate sizes
    archive_size = sum(get_file_size(f) for f in actions["archive"])
    delete_size = sum(get_file_size(f) for f in actions["delete"])
    symlink_count = len(actions["delete_symlinks"])

    print("=" * 60)
    print("MODEL ARCHIVE CLEANUP")
    print("=" * 60)
    print(f"Mode: {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"Keep latest: {args.keep} timestamped checkpoints per config")
    print()

    print(f"KEEP:    {len(actions['keep']):>4} files")
    print(f"ARCHIVE: {len(actions['archive']):>4} files ({format_size(archive_size)})")
    print(f"DELETE:  {len(actions['delete']):>4} files ({format_size(delete_size)})")
    print(f"REMOVE SYMLINKS: {symlink_count:>4}")
    print()

    if actions["archive"]:
        print("--- Files to ARCHIVE (move to archive/) ---")
        for f in sorted(actions["archive"], key=lambda x: x.name):
            print(f"  {f.name:60s} {format_size(get_file_size(f)):>10s}")
        print()

    if actions["delete"]:
        print("--- Files to DELETE ---")
        for f in sorted(actions["delete"], key=lambda x: x.name):
            print(f"  {f.name}")
        print()

    if actions["delete_symlinks"]:
        print("--- Symlinks to REMOVE ---")
        for f in sorted(actions["delete_symlinks"], key=lambda x: x.name):
            target = os.readlink(f) if f.is_symlink() else "?"
            print(f"  {f.name} -> {target}")
        print()

    total_freed = archive_size + delete_size
    print(f"Total space freed from models/: {format_size(total_freed)}")
    print(f"  (archived models preserved in models/archive/)")

    if not args.execute:
        print()
        print("This was a DRY RUN. Use --execute to perform cleanup.")
        return

    # Execute cleanup
    ARCHIVE_DIR.mkdir(exist_ok=True)

    archived = 0
    for f in actions["archive"]:
        dest = ARCHIVE_DIR / f.name
        if dest.exists():
            # Add suffix to avoid overwrite
            stem = f.stem
            suffix = f.suffix
            counter = 1
            while dest.exists():
                dest = ARCHIVE_DIR / f"{stem}_{counter}{suffix}"
                counter += 1
        f.rename(dest)
        archived += 1

    deleted = 0
    for f in actions["delete"]:
        f.unlink()
        deleted += 1

    removed_symlinks = 0
    for f in actions["delete_symlinks"]:
        f.unlink()
        removed_symlinks += 1

    # Also archive matching .sha256 files for archived models
    sha256_archived = 0
    for f in actions["archive"]:
        sha_file = MODELS_DIR / f"{f.name}.sha256"
        if sha_file.exists():
            sha_dest = ARCHIVE_DIR / sha_file.name
            sha_file.rename(sha_dest)
            sha256_archived += 1

    print()
    print(f"Done! Archived: {archived}, Deleted: {deleted + removed_symlinks}, SHA256 moved: {sha256_archived}")
    print(f"Space freed: {format_size(total_freed)}")


if __name__ == "__main__":
    main()
