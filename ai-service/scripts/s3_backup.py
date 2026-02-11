#!/usr/bin/env python3
"""Automated S3 backup script for RingRift AI models and training data.

Backs up:
- Best models (all configs)
- NNUE models
- Game databases (training data)
- Promotion history and state
- Elo leaderboards

Usage:
    python scripts/s3_backup.py                    # Full backup
    python scripts/s3_backup.py --models-only      # Models only
    python scripts/s3_backup.py --dry-run          # Show what would be backed up
    python scripts/s3_backup.py --restore latest   # Restore from latest backup
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Configuration
S3_BUCKET = os.getenv("RINGRIFT_S3_BUCKET", "ringrift-models-20251214")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Paths to back up
BACKUP_PATHS = {
    "models/best": AI_SERVICE_ROOT / "models",  # Best models
    "models/nnue": AI_SERVICE_ROOT / "models" / "nnue",  # NNUE models
    "data/games": AI_SERVICE_ROOT / "data" / "games",  # Game databases
    "data/state": AI_SERVICE_ROOT / "data",  # State files (promotion history, etc.)
}

# File patterns to include/exclude
INCLUDE_PATTERNS = {
    "models/best": ["ringrift_best_*.pth"],
    "models/nnue": ["*.pt"],
    # Include WAL and SHM files for complete database backup (Dec 2025 fix)
    "data/games": ["*.db", "*.db-wal", "*.db-shm"],
    "data/state": ["*.json"],
}

EXCLUDE_PATTERNS = ["*.tmp", "*.log", "__pycache__"]


def run_command(cmd: list[str], dry_run: bool = False) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    if dry_run:
        print(f"  DRY RUN: {' '.join(cmd)}")
        return True, ""

    try:
        # Increased timeout: large databases (10GB+) need ~30 min at 50 Mbps
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        return result.returncode == 0, result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        return False, "Command timed out"
    except Exception as e:
        return False, str(e)


def get_file_hash(filepath: Path) -> str:
    """Calculate MD5 hash of a file for change detection."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def list_s3_objects(prefix: str) -> dict[str, dict]:
    """List objects in S3 with metadata."""
    cmd = [
        "aws", "s3api", "list-objects-v2",
        "--bucket", S3_BUCKET,
        "--prefix", prefix,
        "--query", "Contents[].{Key: Key, Size: Size, ETag: ETag}",
        "--output", "json"
    ]
    success, output = run_command(cmd)
    if success and output.strip():
        try:
            objects = json.loads(output)
            return {obj["Key"]: obj for obj in (objects or [])}
        except json.JSONDecodeError:
            pass
    return {}


def sync_to_s3(
    local_path: Path,
    s3_prefix: str,
    include_patterns: list[str],
    dry_run: bool = False
) -> tuple[int, int]:
    """Sync local directory to S3."""
    if not local_path.exists():
        print(f"  Warning: {local_path} does not exist, skipping")
        return 0, 0

    # Build aws s3 sync command with includes
    # NOTE: --delete intentionally omitted to prevent S3 data loss when
    # local files are cleaned up. S3 should be append-only backup.
    cmd = [
        "aws", "s3", "sync",
        str(local_path),
        f"s3://{S3_BUCKET}/{s3_prefix}/",
    ]

    # Add exclude patterns
    for pattern in EXCLUDE_PATTERNS:
        cmd.extend(["--exclude", pattern])

    # Add include patterns (after exclude to take precedence)
    for pattern in include_patterns:
        cmd.extend(["--include", pattern])

    # Exclude everything else
    cmd.extend(["--exclude", "*"])
    for pattern in include_patterns:
        cmd.extend(["--include", pattern])

    if dry_run:
        cmd.append("--dryrun")

    print(f"  Syncing {local_path} -> s3://{S3_BUCKET}/{s3_prefix}/")
    _success, output = run_command(cmd)

    # Parse output to count files
    uploaded = output.count("upload:")
    deleted = output.count("delete:")

    if output.strip():
        for line in output.strip().split("\n")[:10]:  # Show first 10 lines
            print(f"    {line}")
        if output.count("\n") > 10:
            print(f"    ... and {output.count(chr(10)) - 10} more")

    return uploaded, deleted


def backup_models(dry_run: bool = False) -> dict[str, int]:
    """Backup all models to S3."""
    print("\n=== Backing up Models ===")
    stats = {"uploaded": 0, "deleted": 0}

    for s3_prefix, local_path in [
        ("models/best", AI_SERVICE_ROOT / "models"),
        ("models/nnue", AI_SERVICE_ROOT / "models" / "nnue"),
    ]:
        patterns = INCLUDE_PATTERNS.get(s3_prefix, ["*"])
        uploaded, deleted = sync_to_s3(local_path, s3_prefix, patterns, dry_run)
        stats["uploaded"] += uploaded
        stats["deleted"] += deleted

    return stats


def backup_databases(dry_run: bool = False) -> dict[str, int]:
    """Backup game databases to S3."""
    print("\n=== Backing up Game Databases ===")

    games_path = AI_SERVICE_ROOT / "data" / "games"
    if not games_path.exists():
        print(f"  No games directory found at {games_path}")
        return {"uploaded": 0, "deleted": 0}

    uploaded, deleted = sync_to_s3(
        games_path,
        "data/games",
        INCLUDE_PATTERNS.get("data/games", ["*.db"]),
        dry_run
    )

    return {"uploaded": uploaded, "deleted": deleted}


def backup_state(dry_run: bool = False) -> dict[str, int]:
    """Backup state files (promotion history, etc.)."""
    print("\n=== Backing up State Files ===")

    data_path = AI_SERVICE_ROOT / "data"
    if not data_path.exists():
        print(f"  No data directory found at {data_path}")
        return {"uploaded": 0, "deleted": 0}

    uploaded, deleted = sync_to_s3(
        data_path,
        "data/state",
        ["*.json"],
        dry_run
    )

    return {"uploaded": uploaded, "deleted": deleted}


def create_backup_manifest(dry_run: bool = False) -> None:
    """Create a manifest file with backup metadata."""
    manifest = {
        "timestamp": datetime.now().isoformat(),
        "bucket": S3_BUCKET,
        "paths_backed_up": list(BACKUP_PATHS.keys()),
        "machine": os.uname().nodename,
    }

    manifest_path = AI_SERVICE_ROOT / "data" / "backup_manifest.json"

    if not dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Upload manifest
        cmd = [
            "aws", "s3", "cp",
            str(manifest_path),
            f"s3://{S3_BUCKET}/manifests/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        ]
        run_command(cmd)

    print(f"\n  Manifest created: {manifest}")


def restore_from_s3(
    s3_prefix: str,
    local_path: Path,
    dry_run: bool = False
) -> tuple[int, str]:
    """Restore files from S3 to local."""
    cmd = [
        "aws", "s3", "sync",
        f"s3://{S3_BUCKET}/{s3_prefix}/",
        str(local_path),
    ]

    if dry_run:
        cmd.append("--dryrun")

    print(f"  Restoring s3://{S3_BUCKET}/{s3_prefix}/ -> {local_path}")
    _success, output = run_command(cmd)

    downloaded = output.count("download:")
    return downloaded, output


def main():
    parser = argparse.ArgumentParser(description="S3 backup for RingRift AI")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be done")
    parser.add_argument("--models-only", action="store_true", help="Only backup models")
    parser.add_argument("--databases-only", action="store_true", help="Only backup databases")
    parser.add_argument("--restore", type=str, metavar="PREFIX", help="Restore from S3 (e.g., 'models/best')")
    args = parser.parse_args()

    print("RingRift S3 Backup")
    print(f"  Bucket: {S3_BUCKET}")
    print(f"  Region: {AWS_REGION}")
    print(f"  Mode: {'DRY RUN' if args.dry_run else 'LIVE'}")

    if args.restore:
        # Restore mode
        print("\n=== Restoring from S3 ===")
        s3_prefix = args.restore

        if s3_prefix in BACKUP_PATHS:
            local_path = BACKUP_PATHS[s3_prefix]
        else:
            # Assume it's a direct prefix
            local_path = AI_SERVICE_ROOT / s3_prefix.replace("/", os.sep)

        downloaded, _output = restore_from_s3(s3_prefix, local_path, args.dry_run)
        print(f"\n  Downloaded: {downloaded} files")
        return 0

    # Backup mode
    total_stats = {"uploaded": 0, "deleted": 0}

    if args.models_only:
        stats = backup_models(args.dry_run)
        total_stats["uploaded"] += stats["uploaded"]
        total_stats["deleted"] += stats["deleted"]
    elif args.databases_only:
        stats = backup_databases(args.dry_run)
        total_stats["uploaded"] += stats["uploaded"]
        total_stats["deleted"] += stats["deleted"]
    else:
        # Full backup
        for backup_func in [backup_models, backup_databases, backup_state]:
            stats = backup_func(args.dry_run)
            total_stats["uploaded"] += stats["uploaded"]
            total_stats["deleted"] += stats["deleted"]

        create_backup_manifest(args.dry_run)

    print("\n=== Backup Summary ===")
    print(f"  Files uploaded: {total_stats['uploaded']}")
    print(f"  Files deleted from S3: {total_stats['deleted']}")

    if args.dry_run:
        print("\n  This was a DRY RUN. Run without --dry-run to perform backup.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
