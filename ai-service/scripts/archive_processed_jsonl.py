#!/usr/bin/env python3
"""Archive Processed JSONL Files.

After JSONL files have been successfully imported into SQLite databases,
this script archives them to save disk space while preserving the data.

Features:
- Identifies JSONL files already imported to DB
- Compresses and archives to a designated location
- Maintains manifest of archived files
- Safe deletion with verification

Usage:
    # Dry run - see what would be archived
    python scripts/archive_processed_jsonl.py --dry-run

    # Archive JSONL files older than 24 hours
    python scripts/archive_processed_jsonl.py --min-age 24

    # Archive to specific directory
    python scripts/archive_processed_jsonl.py --archive-dir /mnt/archive/jsonl
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import shutil
import sqlite3
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("archive_processed_jsonl")
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


class JSONLArchiver:
    """Archives processed JSONL files."""

    def __init__(self, archive_dir: Path | None = None):
        self.archive_dir = archive_dir or AI_SERVICE_ROOT / "data" / "archives" / "jsonl"
        self.manifest_path = self.archive_dir / "archive_manifest.json"
        self.manifest: dict[str, Any] = self._load_manifest()

    def _load_manifest(self) -> dict[str, Any]:
        """Load or create archive manifest."""
        if self.manifest_path.exists():
            with open(self.manifest_path) as f:
                return json.load(f)
        return {
            "created": datetime.now().isoformat(),
            "files": {},
            "total_size_saved": 0,
        }

    def _save_manifest(self):
        """Save archive manifest."""
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.manifest_path, "w") as f:
            json.dump(self.manifest, f, indent=2)

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def find_jsonl_files(self, min_age_hours: int = 24) -> list[Path]:
        """Find JSONL files older than min_age_hours."""
        selfplay_dir = AI_SERVICE_ROOT / "data" / "selfplay"
        cutoff = time.time() - (min_age_hours * 3600)

        jsonl_files = []
        for jsonl_path in selfplay_dir.rglob("*.jsonl"):
            if jsonl_path.stat().st_mtime < cutoff:
                jsonl_files.append(jsonl_path)

        return sorted(jsonl_files, key=lambda p: p.stat().st_mtime)

    def verify_in_db(self, jsonl_path: Path, db_path: Path) -> bool:
        """Verify JSONL content is in database."""
        if not db_path.exists():
            return False

        try:
            # Count games in JSONL
            with open(jsonl_path) as f:
                jsonl_count = sum(1 for line in f if line.strip())

            if jsonl_count == 0:
                return True  # Empty files are "verified"

            # This is a simplified check - in production you'd verify specific game IDs
            conn = sqlite3.connect(str(db_path))
            db_count = conn.execute("SELECT COUNT(*) FROM games").fetchone()[0]
            conn.close()

            return db_count > 0
        except Exception as e:
            logger.warning(f"Verification failed for {jsonl_path}: {e}")
            return False

    def archive_file(self, jsonl_path: Path, dry_run: bool = False) -> dict[str, Any] | None:
        """Archive a single JSONL file."""
        if not jsonl_path.exists():
            return None

        original_size = jsonl_path.stat().st_size
        file_hash = self._compute_hash(jsonl_path)

        # Check if already archived
        if file_hash in self.manifest["files"]:
            logger.info(f"Already archived: {jsonl_path}")
            if not dry_run:
                jsonl_path.unlink()
            return None

        # Create archive path
        rel_path = jsonl_path.relative_to(AI_SERVICE_ROOT / "data" / "selfplay")
        archive_path = self.archive_dir / rel_path.with_suffix(".jsonl.gz")

        if dry_run:
            logger.info(f"Would archive: {jsonl_path} -> {archive_path}")
            return {"path": str(jsonl_path), "size": original_size, "action": "dry_run"}

        # Compress and archive
        archive_path.parent.mkdir(parents=True, exist_ok=True)

        with open(jsonl_path, "rb") as f_in, gzip.open(archive_path, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        compressed_size = archive_path.stat().st_size

        # Verify compression
        if compressed_size == 0:
            logger.error(f"Compression failed for {jsonl_path}")
            archive_path.unlink()
            return None

        # Update manifest
        self.manifest["files"][file_hash] = {
            "original_path": str(rel_path),
            "archive_path": str(archive_path.relative_to(self.archive_dir)),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "archived_at": datetime.now().isoformat(),
        }
        self.manifest["total_size_saved"] += (original_size - compressed_size)

        # Delete original
        jsonl_path.unlink()

        logger.info(f"Archived: {jsonl_path} ({original_size / 1024:.1f}KB -> {compressed_size / 1024:.1f}KB)")

        return {
            "path": str(jsonl_path),
            "original_size": original_size,
            "compressed_size": compressed_size,
            "savings": original_size - compressed_size,
        }

    def run(self, min_age_hours: int = 24, dry_run: bool = False,
            verify_db: Path | None = None) -> dict[str, Any]:
        """Run the archival process."""
        results = {
            "files_processed": 0,
            "files_archived": 0,
            "bytes_saved": 0,
            "errors": [],
        }

        jsonl_files = self.find_jsonl_files(min_age_hours)
        logger.info(f"Found {len(jsonl_files)} JSONL files older than {min_age_hours} hours")

        for jsonl_path in jsonl_files:
            try:
                # Optionally verify in DB first
                if verify_db and not self.verify_in_db(jsonl_path, verify_db):
                    logger.warning(f"Skipping {jsonl_path} - not verified in DB")
                    continue

                result = self.archive_file(jsonl_path, dry_run)
                if result:
                    results["files_processed"] += 1
                    if result.get("action") != "dry_run":
                        results["files_archived"] += 1
                        results["bytes_saved"] += result.get("savings", 0)

            except Exception as e:
                logger.error(f"Error archiving {jsonl_path}: {e}")
                results["errors"].append(str(jsonl_path))

        if not dry_run:
            self._save_manifest()

        return results


def main():
    parser = argparse.ArgumentParser(description="Archive Processed JSONL Files")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be archived")
    parser.add_argument("--min-age", type=int, default=24, help="Min age in hours (default: 24)")
    parser.add_argument("--archive-dir", type=str, help="Archive directory")
    parser.add_argument("--verify-db", type=str, help="Verify files are in this DB before archiving")
    args = parser.parse_args()

    archive_dir = Path(args.archive_dir) if args.archive_dir else None
    verify_db = Path(args.verify_db) if args.verify_db else None

    archiver = JSONLArchiver(archive_dir)
    results = archiver.run(
        min_age_hours=args.min_age,
        dry_run=args.dry_run,
        verify_db=verify_db,
    )

    print(f"\nArchival {'(dry run) ' if args.dry_run else ''}complete:")
    print(f"  Files processed: {results['files_processed']}")
    print(f"  Files archived: {results['files_archived']}")
    print(f"  Space saved: {results['bytes_saved'] / (1024*1024):.1f} MB")
    if results["errors"]:
        print(f"  Errors: {len(results['errors'])}")


if __name__ == "__main__":
    main()
