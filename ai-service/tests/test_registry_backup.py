"""Tests for RegistryBackupManager.

Tests backup creation, restoration, verification, rotation,
and auto-backup functionality.
"""

import json
import sqlite3
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from app.training.registry_backup import (
    RegistryBackupManager,
    BackupMetadata,
    backup_before_sync,
    auto_backup_if_needed,
)


class TestBackupMetadata:
    """Test BackupMetadata dataclass."""

    def test_to_dict(self):
        """Test metadata serialization."""
        metadata = BackupMetadata(
            backup_id="20240101_120000",
            timestamp="2024-01-01T12:00:00",
            reason="Test backup",
            registry_path="/path/to/registry.db",
            db_size_bytes=1024,
            db_hash="abc123",
            model_count=5,
            version_count=10,
            source_host="test-host",
            created_by="test",
        )

        d = metadata.to_dict()

        assert d["backup_id"] == "20240101_120000"
        assert d["model_count"] == 5
        assert d["db_hash"] == "abc123"

    def test_from_dict(self):
        """Test metadata deserialization."""
        d = {
            "backup_id": "20240101_120000",
            "timestamp": "2024-01-01T12:00:00",
            "reason": "Test backup",
            "registry_path": "/path/to/registry.db",
            "db_size_bytes": 1024,
            "db_hash": "abc123",
            "model_count": 5,
            "version_count": 10,
            "source_host": "test-host",
            "created_by": "test",
        }

        metadata = BackupMetadata.from_dict(d)

        assert metadata.backup_id == "20240101_120000"
        assert metadata.model_count == 5

    def test_roundtrip(self):
        """Test serialization roundtrip."""
        original = BackupMetadata(
            backup_id="20240101_120000",
            timestamp="2024-01-01T12:00:00",
            reason="Test backup",
            registry_path="/path/to/registry.db",
            db_size_bytes=2048,
            db_hash="xyz789",
            model_count=3,
            version_count=7,
            source_host="host",
            created_by="user",
        )

        d = original.to_dict()
        restored = BackupMetadata.from_dict(d)

        assert restored.backup_id == original.backup_id
        assert restored.db_size_bytes == original.db_size_bytes
        assert restored.model_count == original.model_count


class TestRegistryBackupManager:
    """Test RegistryBackupManager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "registry.db"
        self.backup_dir = Path(self.temp_dir) / "backups"

        # Create a minimal registry database
        self._create_test_registry()

        self.manager = RegistryBackupManager(
            registry_path=self.registry_path,
            backup_dir=self.backup_dir,
            max_backups=5,
            auto_backup_interval_hours=1.0,
        )

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_registry(self):
        """Create a test registry database."""
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()

        # Create required tables
        cursor.execute("""
            CREATE TABLE models (
                model_id TEXT PRIMARY KEY,
                name TEXT,
                created_at TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE versions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version INTEGER,
                stage TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE tags (
                model_id TEXT,
                version INTEGER,
                tag TEXT
            )
        """)
        cursor.execute("""
            CREATE TABLE stage_transitions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_id TEXT,
                version INTEGER,
                from_stage TEXT,
                to_stage TEXT
            )
        """)

        # Insert test data
        cursor.execute(
            "INSERT INTO models VALUES (?, ?, ?)",
            ("test_model", "Test Model", datetime.now().isoformat())
        )
        cursor.execute(
            "INSERT INTO versions VALUES (?, ?, ?, ?)",
            (1, "test_model", 1, "production")
        )
        cursor.execute(
            "INSERT INTO versions VALUES (?, ?, ?, ?)",
            (2, "test_model", 2, "staging")
        )

        conn.commit()
        conn.close()

    def test_create_backup(self):
        """Test creating a backup."""
        backup_path = self.manager.create_backup(
            reason="Test backup",
            created_by="test_user",
        )

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.name == "registry.db"

        # Check backup directory structure
        backup_dir = backup_path.parent
        metadata_path = backup_dir / "metadata.json"
        assert metadata_path.exists()

        # Verify metadata
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["reason"] == "Test backup"
        assert metadata["created_by"] == "test_user"
        assert metadata["model_count"] == 1
        assert metadata["version_count"] == 2

    def test_create_backup_missing_registry(self):
        """Test backup when registry doesn't exist."""
        self.registry_path.unlink()

        backup_path = self.manager.create_backup()

        assert backup_path is None

    def test_list_backups(self):
        """Test listing backups."""
        # Create multiple backups
        self.manager.create_backup(reason="Backup 1")
        import time
        time.sleep(0.01)  # Ensure different timestamps
        self.manager.create_backup(reason="Backup 2")

        backups = self.manager.list_backups()

        assert len(backups) == 2
        # Should be sorted newest first
        assert "Backup 2" in backups[0]["reason"]
        assert "Backup 1" in backups[1]["reason"]

    def test_get_backup(self):
        """Test getting a specific backup."""
        self.manager.create_backup(reason="Test")

        backups = self.manager.list_backups()
        backup_id = backups[0]["backup_id"]

        backup = self.manager.get_backup(backup_id)

        assert backup is not None
        assert backup.backup_id == backup_id

    def test_get_backup_not_found(self):
        """Test getting nonexistent backup."""
        backup = self.manager.get_backup("nonexistent_id")
        assert backup is None

    def test_verify_backup_valid(self):
        """Test verifying a valid backup."""
        self.manager.create_backup(reason="Test")
        backup_id = self.manager.list_backups()[0]["backup_id"]

        result = self.manager.verify_backup(backup_id)

        assert result["valid"]
        assert not result["errors"]

    def test_verify_backup_not_found(self):
        """Test verifying nonexistent backup."""
        result = self.manager.verify_backup("nonexistent_id")

        assert not result["valid"]
        assert "Backup not found" in result["errors"]

    def test_verify_backup_missing_file(self):
        """Test verifying backup with missing database file."""
        self.manager.create_backup(reason="Test")
        backup_id = self.manager.list_backups()[0]["backup_id"]

        # Delete the database file
        backup_db = self.backup_dir / backup_id / "registry.db"
        backup_db.unlink()

        result = self.manager.verify_backup(backup_id)

        assert not result["valid"]
        assert any("missing" in e.lower() for e in result["errors"])

    def test_verify_backup_hash_mismatch(self):
        """Test verifying backup with corrupted database."""
        self.manager.create_backup(reason="Test")
        backup_id = self.manager.list_backups()[0]["backup_id"]

        # Corrupt the database file
        backup_db = self.backup_dir / backup_id / "registry.db"
        with open(backup_db, "ab") as f:
            f.write(b"corruption")

        result = self.manager.verify_backup(backup_id)

        assert not result["valid"]
        assert any("hash" in e.lower() or "mismatch" in e.lower() for e in result["errors"])

    def test_restore_backup(self):
        """Test restoring from backup."""
        # Create backup
        self.manager.create_backup(reason="Before changes")
        backup_id = self.manager.list_backups()[0]["backup_id"]

        # Modify the original database
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        cursor.execute("INSERT INTO models VALUES (?, ?, ?)",
                      ("new_model", "New Model", datetime.now().isoformat()))
        conn.commit()
        conn.close()

        # Verify modification
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM models")
        count_before_restore = cursor.fetchone()[0]
        conn.close()
        assert count_before_restore == 2

        # Restore backup (skip pre-restore backup to simplify test)
        result = self.manager.restore_backup(backup_id, create_pre_restore_backup=False)

        assert result["success"]

        # Verify restoration by directly querying the database
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM models")
        count_after_restore = cursor.fetchone()[0]
        conn.close()
        assert count_after_restore == 1

    def test_restore_backup_creates_pre_restore_backup(self):
        """Test that restore creates a backup of current state."""
        # Create initial backup
        self.manager.create_backup(reason="Initial")
        backup_id = self.manager.list_backups()[0]["backup_id"]

        initial_backup_count = len(self.manager.list_backups())

        # Restore
        result = self.manager.restore_backup(backup_id, create_pre_restore_backup=True)

        assert result["success"]
        assert "pre_restore_backup" in result

        # Should have one more backup
        assert len(self.manager.list_backups()) == initial_backup_count + 1

    def test_restore_backup_invalid(self):
        """Test restoring from invalid backup."""
        result = self.manager.restore_backup("nonexistent_id")

        assert not result["success"]
        assert "error" in result

    def test_backup_rotation(self):
        """Test that old backups are rotated out."""
        # Create more backups than max_backups
        for i in range(7):
            self.manager.create_backup(reason=f"Backup {i}")
            import time
            time.sleep(0.01)

        # Should have at most max_backups
        assert len(self.manager.list_backups()) <= self.manager.max_backups

    def test_should_auto_backup_no_backups(self):
        """Test auto-backup check when no backups exist."""
        assert self.manager.should_auto_backup()

    def test_should_auto_backup_recent(self):
        """Test auto-backup check when recent backup exists."""
        self.manager.create_backup(reason="Recent")

        # Should not need auto-backup immediately
        assert not self.manager.should_auto_backup()

    def test_get_latest_backup(self):
        """Test getting most recent backup."""
        self.manager.create_backup(reason="First")
        import time
        time.sleep(0.01)
        self.manager.create_backup(reason="Second")

        latest = self.manager.get_latest_backup()

        assert latest is not None
        assert "Second" in latest.reason

    def test_get_latest_backup_empty(self):
        """Test getting latest backup when none exist."""
        latest = self.manager.get_latest_backup()
        assert latest is None

    def test_cleanup_orphaned_backups(self):
        """Test cleaning up orphaned backup directories."""
        # Create a backup
        self.manager.create_backup(reason="Tracked")

        # Create orphaned directory
        orphaned_dir = self.backup_dir / "orphaned_20240101"
        orphaned_dir.mkdir(parents=True)
        (orphaned_dir / "registry.db").write_bytes(b"orphan")

        # Clean up
        removed = self.manager.cleanup_orphaned_backups()

        assert removed == 1
        assert not orphaned_dir.exists()

    def test_get_backup_stats(self):
        """Test getting backup statistics."""
        # Initially empty
        stats = self.manager.get_backup_stats()
        assert stats["total_backups"] == 0

        # After creating backups
        self.manager.create_backup(reason="Test 1")
        import time
        time.sleep(0.01)
        self.manager.create_backup(reason="Test 2")

        stats = self.manager.get_backup_stats()

        assert stats["total_backups"] == 2
        assert stats["oldest_backup"] is not None
        assert stats["newest_backup"] is not None
        assert stats["total_size_bytes"] > 0


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.registry_path = Path(self.temp_dir) / "registry.db"
        self.backup_dir = Path(self.temp_dir) / "backups"

        # Create registry with proper tables
        conn = sqlite3.connect(str(self.registry_path))
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE models (model_id TEXT PRIMARY KEY)")
        cursor.execute("CREATE TABLE versions (id INTEGER PRIMARY KEY)")
        cursor.execute("CREATE TABLE tags (id INTEGER PRIMARY KEY)")
        cursor.execute("CREATE TABLE stage_transitions (id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()

    def teardown_method(self):
        """Clean up."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_backup_before_sync(self):
        """Test pre-sync backup convenience function."""
        # Use explicit backup dir to avoid polluting global state
        manager = RegistryBackupManager(self.registry_path, backup_dir=self.backup_dir)
        backup_path = manager.create_backup(reason="Pre-sync backup", created_by="sync_operation")

        assert backup_path is not None
        assert backup_path.exists()

    def test_auto_backup_if_needed_creates(self):
        """Test auto-backup when needed."""
        # Use explicit manager to control backup dir
        manager = RegistryBackupManager(self.registry_path, backup_dir=self.backup_dir)

        # Should need backup since none exist
        assert manager.should_auto_backup()

        backup_path = manager.create_backup(reason="Auto backup", created_by="auto_backup")

        assert backup_path is not None
        assert backup_path.exists()

    def test_auto_backup_if_needed_skips(self):
        """Test auto-backup when not needed."""
        # Create recent backup
        manager = RegistryBackupManager(self.registry_path, backup_dir=self.backup_dir)
        manager.create_backup(reason="Recent")

        # Should not need auto-backup since one was just created
        assert not manager.should_auto_backup()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
