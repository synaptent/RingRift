#!/usr/bin/env python3
"""NFS Sync Verification - Ensure code changes propagate reliably across cluster.

This script verifies that critical files match between local and NFS storage,
and syncs them if there are mismatches. This prevents import errors caused by
stale code on NFS.

Usage:
    # Verify sync (report only)
    python scripts/verify_nfs_sync.py --check

    # Verify and sync if needed
    python scripts/verify_nfs_sync.py --sync

    # Force sync all critical files
    python scripts/verify_nfs_sync.py --force-sync

    # Use as library
    from scripts.verify_nfs_sync import NFSSyncVerifier
    verifier = NFSSyncVerifier()
    mismatches = verifier.verify()
    if mismatches:
        verifier.sync_files(mismatches)
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)

# ============================================
# Configuration
# ============================================

# Critical files that must be in sync across the cluster
CRITICAL_FILES = [
    # Core orchestration
    "scripts/p2p_orchestrator.py",
    "scripts/p2p/constants.py",
    "scripts/p2p/types.py",
    "scripts/unified_ai_loop.py",

    # Coordination modules
    "app/coordination/work_queue.py",
    "app/coordination/job_reaper.py",
    "app/coordination/node_policies.py",

    # P2P and models
    "app/p2p/models.py",
    "app/p2p/types.py",

    # Training infrastructure
    "app/training/nnue_trainer.py",
    "app/training/model_registry.py",

    # Engine interface
    "app/engine_interface/selfplay.py",
    "app/engine_interface/engine_wrapper.py",

    # GPU modules
    "app/gpu/gpu_selfplay.py",
    "app/gpu/gumbel_selfplay.py",
]

# Directories to also check for changes
CRITICAL_DIRS = [
    "scripts/p2p",
    "app/coordination",
    "app/p2p",
]

# Default NFS paths to check
DEFAULT_NFS_PATHS = [
    "/mnt/nfs/ringrift/ai-service",
    "/home/shared/ringrift/ai-service",
    "/data/ringrift/ai-service",
]


@dataclass
class FileMismatch:
    """A file that differs between local and NFS."""
    relative_path: str
    local_path: Path
    nfs_path: Path
    local_hash: str
    nfs_hash: str
    local_mtime: float
    nfs_mtime: float
    local_size: int
    nfs_size: int
    nfs_exists: bool = True
    local_exists: bool = True

    @property
    def newer_is_local(self) -> bool:
        """Check if local version is newer."""
        return self.local_mtime > self.nfs_mtime

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "local_hash": self.local_hash[:8] if self.local_hash else None,
            "nfs_hash": self.nfs_hash[:8] if self.nfs_hash else None,
            "local_mtime": datetime.fromtimestamp(self.local_mtime).isoformat() if self.local_mtime else None,
            "nfs_mtime": datetime.fromtimestamp(self.nfs_mtime).isoformat() if self.nfs_mtime else None,
            "local_size": self.local_size,
            "nfs_size": self.nfs_size,
            "newer_is_local": self.newer_is_local,
            "nfs_exists": self.nfs_exists,
            "local_exists": self.local_exists,
        }


@dataclass
class SyncResult:
    """Result of a sync operation."""
    success: bool
    files_synced: int = 0
    files_failed: int = 0
    synced_files: list[str] = field(default_factory=list)
    failed_files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class NFSSyncVerifier:
    """Verify and sync critical files between local and NFS storage."""

    def __init__(
        self,
        local_root: Path | None = None,
        nfs_root: Path | None = None,
        critical_files: list[str] | None = None,
    ):
        self.local_root = local_root or PROJECT_ROOT
        self.nfs_root = nfs_root or self._detect_nfs_root()
        self.critical_files = critical_files or CRITICAL_FILES

        # Stats
        self.last_verify_time: float = 0
        self.last_sync_time: float = 0
        self.total_syncs: int = 0
        self.sync_history: list[dict[str, Any]] = []

    def _detect_nfs_root(self) -> Path | None:
        """Auto-detect NFS mount point."""
        for path in DEFAULT_NFS_PATHS:
            p = Path(path)
            if p.exists() and p.is_dir():
                # Check if it looks like a ringrift ai-service directory
                if (p / "scripts").exists() or (p / "app").exists():
                    logger.info(f"Detected NFS root: {p}")
                    return p

        # Check environment variable
        env_nfs = os.environ.get("RINGRIFT_NFS_PATH", "").strip()
        if env_nfs:
            p = Path(env_nfs)
            if p.exists():
                return p

        logger.warning("Could not auto-detect NFS root. Set RINGRIFT_NFS_PATH environment variable.")
        return None

    def _hash_file(self, path: Path) -> str:
        """Calculate SHA-256 hash of a file."""
        if not path.exists():
            return ""

        sha256 = hashlib.sha256()
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256.update(chunk)
            return sha256.hexdigest()
        except Exception as e:
            logger.error(f"Error hashing {path}: {e}")
            return ""

    def _get_file_info(self, path: Path) -> tuple[str, float, int]:
        """Get hash, mtime, and size of a file."""
        if not path.exists():
            return "", 0.0, 0

        try:
            stat = path.stat()
            return self._hash_file(path), stat.st_mtime, stat.st_size
        except Exception as e:
            logger.error(f"Error getting file info for {path}: {e}")
            return "", 0.0, 0

    def verify(self, files: list[str] | None = None) -> list[FileMismatch]:
        """Verify that files match between local and NFS.

        Args:
            files: List of relative file paths to check. If None, uses CRITICAL_FILES.

        Returns:
            List of FileMismatch objects for files that differ.
        """
        if self.nfs_root is None:
            logger.error("NFS root not configured")
            return []

        files_to_check = files or self.critical_files
        mismatches = []

        self.last_verify_time = time.time()

        for rel_path in files_to_check:
            local_path = self.local_root / rel_path
            nfs_path = self.nfs_root / rel_path

            # Get file info
            local_hash, local_mtime, local_size = self._get_file_info(local_path)
            nfs_hash, nfs_mtime, nfs_size = self._get_file_info(nfs_path)

            # Check for mismatch
            local_exists = local_path.exists()
            nfs_exists = nfs_path.exists()

            if local_hash != nfs_hash:
                mismatches.append(FileMismatch(
                    relative_path=rel_path,
                    local_path=local_path,
                    nfs_path=nfs_path,
                    local_hash=local_hash,
                    nfs_hash=nfs_hash,
                    local_mtime=local_mtime,
                    nfs_mtime=nfs_mtime,
                    local_size=local_size,
                    nfs_size=nfs_size,
                    local_exists=local_exists,
                    nfs_exists=nfs_exists,
                ))

        if mismatches:
            logger.warning(f"Found {len(mismatches)} file mismatches between local and NFS")
        else:
            logger.info(f"All {len(files_to_check)} critical files are in sync")

        return mismatches

    def sync_files(
        self,
        mismatches: list[FileMismatch],
        direction: str = "local_to_nfs",
        dry_run: bool = False,
    ) -> SyncResult:
        """Sync mismatched files.

        Args:
            mismatches: List of FileMismatch objects to sync
            direction: "local_to_nfs" or "nfs_to_local"
            dry_run: If True, only report what would be synced

        Returns:
            SyncResult with details of the sync operation
        """
        result = SyncResult(success=True)

        for mismatch in mismatches:
            try:
                if direction == "local_to_nfs":
                    src = mismatch.local_path
                    dst = mismatch.nfs_path
                else:
                    src = mismatch.nfs_path
                    dst = mismatch.local_path

                if not src.exists():
                    result.errors.append(f"Source does not exist: {src}")
                    result.failed_files.append(mismatch.relative_path)
                    result.files_failed += 1
                    continue

                if dry_run:
                    logger.info(f"[DRY RUN] Would sync: {src} -> {dst}")
                    result.synced_files.append(mismatch.relative_path)
                    result.files_synced += 1
                    continue

                # Ensure parent directory exists
                dst.parent.mkdir(parents=True, exist_ok=True)

                # Copy file
                shutil.copy2(src, dst)

                logger.info(f"Synced: {mismatch.relative_path} ({direction})")
                result.synced_files.append(mismatch.relative_path)
                result.files_synced += 1

            except Exception as e:
                error_msg = f"Error syncing {mismatch.relative_path}: {e}"
                logger.error(error_msg)
                result.errors.append(error_msg)
                result.failed_files.append(mismatch.relative_path)
                result.files_failed += 1
                result.success = False

        if result.files_synced > 0:
            self.last_sync_time = time.time()
            self.total_syncs += result.files_synced
            self.sync_history.append({
                "timestamp": self.last_sync_time,
                "direction": direction,
                "files_synced": result.files_synced,
                "files_failed": result.files_failed,
                "dry_run": dry_run,
            })

        return result

    def sync_all_critical(self, direction: str = "local_to_nfs", dry_run: bool = False) -> SyncResult:
        """Verify and sync all critical files.

        Args:
            direction: "local_to_nfs" or "nfs_to_local"
            dry_run: If True, only report what would be synced

        Returns:
            SyncResult with details of the sync operation
        """
        mismatches = self.verify()
        if not mismatches:
            return SyncResult(success=True)

        return self.sync_files(mismatches, direction=direction, dry_run=dry_run)

    def verify_directory(self, rel_dir: str) -> list[FileMismatch]:
        """Verify all Python files in a directory.

        Args:
            rel_dir: Relative directory path to check

        Returns:
            List of FileMismatch objects for files that differ
        """
        if self.nfs_root is None:
            logger.error("NFS root not configured")
            return []

        local_dir = self.local_root / rel_dir
        if not local_dir.exists():
            logger.warning(f"Local directory does not exist: {local_dir}")
            return []

        # Find all Python files
        python_files = []
        for path in local_dir.rglob("*.py"):
            rel_path = str(path.relative_to(self.local_root))
            python_files.append(rel_path)

        return self.verify(files=python_files)

    def get_stats(self) -> dict[str, Any]:
        """Get verification/sync statistics."""
        return {
            "local_root": str(self.local_root),
            "nfs_root": str(self.nfs_root) if self.nfs_root else None,
            "critical_files_count": len(self.critical_files),
            "last_verify_time": self.last_verify_time,
            "last_sync_time": self.last_sync_time,
            "total_syncs": self.total_syncs,
            "recent_history": self.sync_history[-10:],
        }


def verify_before_startup() -> bool:
    """Run verification before starting orchestrator.

    Returns:
        True if verification passed (files in sync), False otherwise.
    """
    verifier = NFSSyncVerifier()
    if verifier.nfs_root is None:
        logger.info("NFS not configured, skipping sync verification")
        return True

    mismatches = verifier.verify()
    if not mismatches:
        return True

    # Log mismatches
    logger.warning(f"Found {len(mismatches)} file mismatches:")
    for m in mismatches:
        newer = "local" if m.newer_is_local else "nfs"
        logger.warning(f"  {m.relative_path}: {newer} is newer")

    # Auto-sync if configured
    auto_sync = os.environ.get("RINGRIFT_AUTO_SYNC", "").lower() in ("1", "true", "yes")
    if auto_sync:
        result = verifier.sync_files(mismatches, direction="local_to_nfs")
        if result.success:
            logger.info(f"Auto-synced {result.files_synced} files to NFS")
            return True
        else:
            logger.error(f"Auto-sync failed: {result.errors}")
            return False

    return False


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(description="NFS Sync Verification")
    parser.add_argument("--check", action="store_true", help="Check for mismatches (no sync)")
    parser.add_argument("--sync", action="store_true", help="Sync mismatched files (local to NFS)")
    parser.add_argument("--force-sync", action="store_true", help="Force sync all critical files")
    parser.add_argument("--reverse", action="store_true", help="Sync from NFS to local")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be synced")
    parser.add_argument("--dir", help="Check a specific directory")
    parser.add_argument("--nfs-path", help="Override NFS path")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Create verifier
    nfs_root = Path(args.nfs_path) if args.nfs_path else None
    verifier = NFSSyncVerifier(nfs_root=nfs_root)

    if verifier.nfs_root is None:
        print("ERROR: NFS root not found. Set RINGRIFT_NFS_PATH or use --nfs-path")
        sys.exit(1)

    direction = "nfs_to_local" if args.reverse else "local_to_nfs"

    # Check specific directory
    if args.dir:
        mismatches = verifier.verify_directory(args.dir)
    else:
        mismatches = verifier.verify()

    # Report mismatches
    if mismatches:
        print(f"\nFound {len(mismatches)} mismatches:")
        for m in mismatches:
            newer = "LOCAL" if m.newer_is_local else "NFS"
            status = f"[{newer} newer]"
            if not m.local_exists:
                status = "[MISSING locally]"
            elif not m.nfs_exists:
                status = "[MISSING on NFS]"
            print(f"  {status} {m.relative_path}")
    else:
        print("\nAll files in sync!")
        sys.exit(0)

    # Sync if requested
    if args.sync or args.force_sync:
        result = verifier.sync_files(mismatches, direction=direction, dry_run=args.dry_run)

        if args.dry_run:
            print(f"\n[DRY RUN] Would sync {result.files_synced} files")
        else:
            print(f"\nSynced {result.files_synced} files, {result.files_failed} failed")

        if result.errors:
            print("\nErrors:")
            for err in result.errors:
                print(f"  {err}")

        sys.exit(0 if result.success else 1)

    # Just checking - exit with error if mismatches found
    if args.check and mismatches:
        sys.exit(1)


if __name__ == "__main__":
    main()
