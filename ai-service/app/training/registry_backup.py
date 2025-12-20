"""Model Registry Backup and Recovery.

Provides automated backup and recovery capabilities for the model registry
database and associated model files.

Features:
- Automatic backups before risky operations (sync, bulk updates)
- Scheduled periodic backups
- Point-in-time recovery
- Backup rotation with configurable retention
- Verification of backup integrity

Usage:
    from app.training.registry_backup import RegistryBackupManager

    backup_mgr = RegistryBackupManager(registry_path)

    # Create backup
    backup_path = backup_mgr.create_backup(reason="Before sync")

    # List backups
    backups = backup_mgr.list_backups()

    # Restore from backup
    backup_mgr.restore_backup(backup_id="20240101_120000")
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sqlite3
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from app.utils.checksum_utils import compute_file_checksum
from app.utils.paths import AI_SERVICE_ROOT

logger = logging.getLogger(__name__)
DEFAULT_BACKUP_DIR = AI_SERVICE_ROOT / "data" / "registry_backups"


@dataclass
class BackupMetadata:
    """Metadata for a registry backup."""
    backup_id: str
    timestamp: str
    reason: str
    registry_path: str
    db_size_bytes: int
    db_hash: str
    model_count: int
    version_count: int
    source_host: str
    created_by: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BackupMetadata:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class RegistryBackupManager:
    """Manages model registry backups and recovery."""

    def __init__(
        self,
        registry_path: Path,
        backup_dir: Path | None = None,
        max_backups: int = 10,
        auto_backup_interval_hours: float = 24.0,
    ):
        """Initialize the backup manager.

        Args:
            registry_path: Path to the registry database
            backup_dir: Directory to store backups (default: data/registry_backups)
            max_backups: Maximum number of backups to retain
            auto_backup_interval_hours: Minimum hours between auto-backups
        """
        self.registry_path = Path(registry_path)
        self.backup_dir = backup_dir or DEFAULT_BACKUP_DIR
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.max_backups = max_backups
        self.auto_backup_interval_hours = auto_backup_interval_hours

        self._metadata_file = self.backup_dir / "backup_manifest.json"
        self._backups: list[BackupMetadata] = []
        self._load_manifest()

    def _load_manifest(self):
        """Load backup manifest from disk."""
        if self._metadata_file.exists():
            try:
                with open(self._metadata_file) as f:
                    data = json.load(f)
                    self._backups = [
                        BackupMetadata.from_dict(b)
                        for b in data.get("backups", [])
                    ]
            except Exception as e:
                logger.warning(f"Failed to load backup manifest: {e}")

    def _save_manifest(self):
        """Save backup manifest to disk."""
        try:
            with open(self._metadata_file, 'w') as f:
                json.dump({
                    "backups": [b.to_dict() for b in self._backups],
                    "last_updated": datetime.now().isoformat(),
                }, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save backup manifest: {e}")

    def _compute_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        return compute_file_checksum(file_path, return_empty_for_missing=False)

    def _get_registry_stats(self) -> dict[str, int]:
        """Get model and version counts from registry."""
        if not self.registry_path.exists():
            return {"model_count": 0, "version_count": 0}

        try:
            conn = sqlite3.connect(str(self.registry_path))
            cursor = conn.cursor()

            cursor.execute("SELECT COUNT(*) FROM models")
            model_count = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM versions")
            version_count = cursor.fetchone()[0]

            conn.close()
            return {"model_count": model_count, "version_count": version_count}
        except Exception as e:
            logger.warning(f"Failed to get registry stats: {e}")
            return {"model_count": 0, "version_count": 0}

    def create_backup(
        self,
        reason: str = "Manual backup",
        created_by: str = "system",
    ) -> Path | None:
        """Create a backup of the registry database.

        Args:
            reason: Reason for creating the backup
            created_by: User or system that created the backup

        Returns:
            Path to the backup file, or None if backup failed
        """
        if not self.registry_path.exists():
            logger.warning(f"Registry not found at {self.registry_path}")
            return None

        # Generate backup ID
        backup_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_subdir = self.backup_dir / backup_id
        backup_subdir.mkdir(parents=True, exist_ok=True)

        backup_db_path = backup_subdir / "registry.db"
        metadata_path = backup_subdir / "metadata.json"

        try:
            # Copy database using SQLite backup API for consistency
            source_conn = sqlite3.connect(str(self.registry_path))
            dest_conn = sqlite3.connect(str(backup_db_path))

            source_conn.backup(dest_conn)

            source_conn.close()
            dest_conn.close()

            # Get stats and hash
            stats = self._get_registry_stats()
            db_hash = self._compute_hash(backup_db_path)
            db_size = backup_db_path.stat().st_size

            # Create metadata
            metadata = BackupMetadata(
                backup_id=backup_id,
                timestamp=datetime.now().isoformat(),
                reason=reason,
                registry_path=str(self.registry_path),
                db_size_bytes=db_size,
                db_hash=db_hash,
                model_count=stats["model_count"],
                version_count=stats["version_count"],
                source_host=os.uname().nodename,
                created_by=created_by,
            )

            # Save metadata
            with open(metadata_path, 'w') as f:
                json.dump(metadata.to_dict(), f, indent=2)

            # Add to manifest
            self._backups.append(metadata)
            self._save_manifest()

            # Rotate old backups
            self._rotate_backups()

            logger.info(f"Created backup {backup_id}: {stats['model_count']} models, {stats['version_count']} versions")
            return backup_db_path

        except Exception as e:
            logger.error(f"Backup failed: {e}")
            # Clean up failed backup
            if backup_subdir.exists():
                shutil.rmtree(backup_subdir, ignore_errors=True)
            return None

    def _rotate_backups(self):
        """Remove old backups beyond the retention limit."""
        if len(self._backups) <= self.max_backups:
            return

        # Sort by timestamp (oldest first)
        sorted_backups = sorted(self._backups, key=lambda b: b.timestamp)

        # Remove oldest backups
        to_remove = len(self._backups) - self.max_backups
        for backup in sorted_backups[:to_remove]:
            backup_dir = self.backup_dir / backup.backup_id
            if backup_dir.exists():
                shutil.rmtree(backup_dir, ignore_errors=True)
            self._backups.remove(backup)
            logger.info(f"Rotated old backup: {backup.backup_id}")

        self._save_manifest()

    def should_auto_backup(self) -> bool:
        """Check if an auto-backup should be performed."""
        if not self._backups:
            return True

        latest = max(self._backups, key=lambda b: b.timestamp)
        latest_time = datetime.fromisoformat(latest.timestamp)
        hours_since = (datetime.now() - latest_time).total_seconds() / 3600

        return hours_since >= self.auto_backup_interval_hours

    def list_backups(self) -> list[dict[str, Any]]:
        """List all available backups.

        Returns:
            List of backup metadata dicts, sorted by timestamp descending
        """
        return sorted(
            [b.to_dict() for b in self._backups],
            key=lambda b: b["timestamp"],
            reverse=True,
        )

    def get_backup(self, backup_id: str) -> BackupMetadata | None:
        """Get metadata for a specific backup."""
        for backup in self._backups:
            if backup.backup_id == backup_id:
                return backup
        return None

    def verify_backup(self, backup_id: str) -> dict[str, Any]:
        """Verify integrity of a backup.

        Returns:
            Dict with verification results
        """
        result = {
            "backup_id": backup_id,
            "valid": False,
            "errors": [],
        }

        backup = self.get_backup(backup_id)
        if not backup:
            result["errors"].append("Backup not found")
            return result

        backup_dir = self.backup_dir / backup_id
        backup_db = backup_dir / "registry.db"

        if not backup_db.exists():
            result["errors"].append("Backup database file missing")
            return result

        # Verify hash
        current_hash = self._compute_hash(backup_db)
        if current_hash != backup.db_hash:
            result["errors"].append(f"Hash mismatch: expected {backup.db_hash[:16]}..., got {current_hash[:16]}...")
            return result

        # Verify database can be opened
        try:
            conn = sqlite3.connect(str(backup_db))
            cursor = conn.cursor()

            # Check tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = {row[0] for row in cursor.fetchall()}

            required_tables = {"models", "versions", "tags", "stage_transitions"}
            missing = required_tables - tables
            if missing:
                result["errors"].append(f"Missing tables: {missing}")
                conn.close()
                return result

            # Check counts match
            cursor.execute("SELECT COUNT(*) FROM models")
            model_count = cursor.fetchone()[0]
            if model_count != backup.model_count:
                result["errors"].append(f"Model count mismatch: expected {backup.model_count}, got {model_count}")

            cursor.execute("SELECT COUNT(*) FROM versions")
            version_count = cursor.fetchone()[0]
            if version_count != backup.version_count:
                result["errors"].append(f"Version count mismatch: expected {backup.version_count}, got {version_count}")

            conn.close()

        except Exception as e:
            result["errors"].append(f"Database verification failed: {e}")
            return result

        if not result["errors"]:
            result["valid"] = True

        return result

    def restore_backup(
        self,
        backup_id: str,
        create_pre_restore_backup: bool = True,
    ) -> dict[str, Any]:
        """Restore registry from a backup.

        Args:
            backup_id: ID of the backup to restore
            create_pre_restore_backup: Create a backup of current state before restoring

        Returns:
            Dict with restore results
        """
        result = {
            "success": False,
            "backup_id": backup_id,
        }

        # Verify backup first
        verification = self.verify_backup(backup_id)
        if not verification["valid"]:
            result["error"] = f"Backup verification failed: {verification['errors']}"
            return result

        self.get_backup(backup_id)
        backup_db = self.backup_dir / backup_id / "registry.db"

        # Create pre-restore backup
        if create_pre_restore_backup and self.registry_path.exists():
            pre_restore = self.create_backup(
                reason=f"Pre-restore backup (restoring to {backup_id})",
                created_by="restore_operation",
            )
            if pre_restore:
                result["pre_restore_backup"] = str(pre_restore)

        try:
            # Copy backup to registry location
            shutil.copy2(backup_db, self.registry_path)

            # Verify restored database
            stats = self._get_registry_stats()
            result["success"] = True
            result["restored_models"] = stats["model_count"]
            result["restored_versions"] = stats["version_count"]

            logger.info(f"Restored backup {backup_id}: {stats['model_count']} models, {stats['version_count']} versions")

        except Exception as e:
            result["error"] = str(e)
            logger.error(f"Restore failed: {e}")

        return result

    def get_latest_backup(self) -> BackupMetadata | None:
        """Get the most recent backup."""
        if not self._backups:
            return None
        return max(self._backups, key=lambda b: b.timestamp)

    def cleanup_orphaned_backups(self) -> int:
        """Remove backup directories not in the manifest.

        Returns:
            Number of orphaned backups removed
        """
        known_ids = {b.backup_id for b in self._backups}
        removed = 0

        for item in self.backup_dir.iterdir():
            if item.is_dir() and item.name not in known_ids:
                # Check if it looks like a backup directory
                if (item / "registry.db").exists() or (item / "metadata.json").exists():
                    shutil.rmtree(item, ignore_errors=True)
                    removed += 1
                    logger.info(f"Removed orphaned backup: {item.name}")

        return removed

    def get_backup_stats(self) -> dict[str, Any]:
        """Get statistics about backups."""
        if not self._backups:
            return {
                "total_backups": 0,
                "oldest_backup": None,
                "newest_backup": None,
                "total_size_bytes": 0,
            }

        total_size = sum(b.db_size_bytes for b in self._backups)
        oldest = min(self._backups, key=lambda b: b.timestamp)
        newest = max(self._backups, key=lambda b: b.timestamp)

        return {
            "total_backups": len(self._backups),
            "oldest_backup": oldest.backup_id,
            "oldest_timestamp": oldest.timestamp,
            "newest_backup": newest.backup_id,
            "newest_timestamp": newest.timestamp,
            "total_size_bytes": total_size,
            "total_size_mb": total_size / (1024 * 1024),
        }


def backup_before_sync(registry_path: Path) -> Path | None:
    """Create a backup before a sync operation.

    Convenience function for use in sync operations.
    """
    manager = RegistryBackupManager(registry_path)
    return manager.create_backup(reason="Pre-sync backup", created_by="sync_operation")


def auto_backup_if_needed(registry_path: Path) -> Path | None:
    """Create an auto-backup if enough time has passed.

    Convenience function for scheduled backup checks.
    """
    manager = RegistryBackupManager(registry_path)
    if manager.should_auto_backup():
        return manager.create_backup(reason="Scheduled auto-backup", created_by="auto_backup")
    return None
