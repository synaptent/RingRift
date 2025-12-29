"""Unit tests for ConfigUpdater.

Tests the atomic YAML config update functionality for node status changes.

Created: Dec 29, 2025
Phase 4: Test coverage for critical untested modules.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from app.coordination.node_availability.config_updater import (
    ConfigUpdateResult,
    ConfigUpdater,
    BACKUP_DIR_NAME,
    MAX_BACKUPS,
)


class TestConfigUpdateResult:
    """Tests for ConfigUpdateResult dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        result = ConfigUpdateResult(success=True)
        assert result.success is True
        assert result.nodes_updated == []
        assert result.changes == {}
        assert result.backup_path is None
        assert result.error is None
        assert result.dry_run is False
        assert isinstance(result.timestamp, datetime)

    def test_successful_update(self):
        """Test successful update result."""
        result = ConfigUpdateResult(
            success=True,
            nodes_updated=["vast-123", "lambda-456"],
            changes={
                "vast-123": ("ready", "offline"),
                "lambda-456": ("offline", "ready"),
            },
            backup_path=Path("/backups/config_backup.yaml"),
        )
        assert result.success
        assert len(result.nodes_updated) == 2
        assert result.update_count == 2
        assert result.changes["vast-123"] == ("ready", "offline")

    def test_failed_update(self):
        """Test failed update result."""
        result = ConfigUpdateResult(
            success=False,
            error="Config file not found",
        )
        assert not result.success
        assert result.error == "Config file not found"
        assert result.update_count == 0

    def test_dry_run_result(self):
        """Test dry run result."""
        result = ConfigUpdateResult(
            success=True,
            nodes_updated=["node1"],
            dry_run=True,
        )
        assert result.dry_run
        assert str(result) == "DRY RUN: Would update 1 nodes"

    def test_str_success(self):
        """Test string representation for success."""
        result = ConfigUpdateResult(
            success=True,
            nodes_updated=["node1", "node2"],
        )
        assert str(result) == "Updated 2 nodes successfully"

    def test_str_failure(self):
        """Test string representation for failure."""
        result = ConfigUpdateResult(
            success=False,
            error="YAML parse error",
        )
        assert "YAML parse error" in str(result)

    def test_update_count_property(self):
        """Test update_count property."""
        result = ConfigUpdateResult(
            success=True,
            nodes_updated=["a", "b", "c"],
        )
        assert result.update_count == 3


class TestConfigUpdater:
    """Tests for ConfigUpdater class."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "distributed_hosts.yaml"
        self._create_test_config()

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_config(self):
        """Create a test config file."""
        config = {
            "hosts": {
                "vast-12345": {
                    "status": "ready",
                    "tailscale_ip": "100.1.2.3",
                },
                "lambda-gh200-1": {
                    "status": "offline",
                    "tailscale_ip": "100.4.5.6",
                },
                "runpod-h100": {
                    "status": "ready",
                    "tailscale_ip": "100.7.8.9",
                },
            }
        }
        with open(self.config_path, "w") as f:
            yaml.safe_dump(config, f)

    def test_init_with_path(self):
        """Test initialization with explicit path."""
        updater = ConfigUpdater(config_path=self.config_path)
        assert updater.config_path == self.config_path
        assert updater.dry_run is True  # Safe default
        assert updater.max_backups == MAX_BACKUPS

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        updater = ConfigUpdater(
            config_path=self.config_path,
            dry_run=False,
            max_backups=5,
        )
        assert updater.dry_run is False
        assert updater.max_backups == 5

    def test_find_config_path_default(self):
        """Test _find_config_path with default location."""
        updater = ConfigUpdater(config_path=self.config_path)
        # Should return what we passed
        assert updater.config_path == self.config_path

    def test_backup_dir_property(self):
        """Test backup_dir property."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)
        backup_dir = updater.backup_dir
        assert backup_dir == self.config_path.parent / BACKUP_DIR_NAME

    def test_backup_dir_creates_directory(self):
        """Test backup_dir creates directory if needed."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)
        backup_dir = updater.backup_dir
        assert backup_dir.exists()

    def test_backup_dir_dry_run_no_create(self):
        """Test backup_dir doesn't create directory in dry_run mode."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=True)
        backup_dir = updater.backup_dir
        # In dry_run, should return path but not create
        assert backup_dir == self.config_path.parent / BACKUP_DIR_NAME
        # Directory should not exist unless we created it
        # (might exist from other tests in same temp dir)

    def test_load_config(self):
        """Test loading configuration."""
        updater = ConfigUpdater(config_path=self.config_path)
        config = updater.load_config()
        assert "hosts" in config
        assert "vast-12345" in config["hosts"]
        assert config["hosts"]["vast-12345"]["status"] == "ready"

    def test_load_config_file_not_found(self):
        """Test load_config raises on missing file."""
        updater = ConfigUpdater(config_path=Path("/nonexistent/config.yaml"))
        with pytest.raises(FileNotFoundError):
            updater.load_config()

    @pytest.mark.asyncio
    async def test_update_node_statuses_dry_run(self):
        """Test updating node statuses in dry run mode."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=True)

        result = await updater.update_node_statuses({
            "vast-12345": "offline",
        })

        assert result.success
        assert result.dry_run
        assert "vast-12345" in result.nodes_updated
        assert result.changes["vast-12345"] == ("ready", "offline")
        # Config should NOT be changed
        config = updater.load_config()
        assert config["hosts"]["vast-12345"]["status"] == "ready"

    @pytest.mark.asyncio
    async def test_update_node_statuses_actual(self):
        """Test actually updating node statuses."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)

        result = await updater.update_node_statuses({
            "vast-12345": "offline",
        })

        assert result.success
        assert not result.dry_run
        assert "vast-12345" in result.nodes_updated
        # Config SHOULD be changed
        config = updater.load_config()
        assert config["hosts"]["vast-12345"]["status"] == "offline"
        # Backup should exist
        assert result.backup_path is not None

    @pytest.mark.asyncio
    async def test_update_node_statuses_no_change(self):
        """Test update when status already matches."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)

        result = await updater.update_node_statuses({
            "vast-12345": "ready",  # Already ready
        })

        assert result.success
        assert result.nodes_updated == []
        assert result.changes == {}

    @pytest.mark.asyncio
    async def test_update_node_statuses_unknown_node(self):
        """Test update with unknown node."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)

        result = await updater.update_node_statuses({
            "unknown-node": "offline",
        })

        assert result.success
        assert result.nodes_updated == []
        # Unknown nodes are skipped

    @pytest.mark.asyncio
    async def test_update_node_statuses_multiple(self):
        """Test updating multiple nodes."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)

        result = await updater.update_node_statuses({
            "vast-12345": "offline",
            "lambda-gh200-1": "ready",
        })

        assert result.success
        assert len(result.nodes_updated) == 2
        config = updater.load_config()
        assert config["hosts"]["vast-12345"]["status"] == "offline"
        assert config["hosts"]["lambda-gh200-1"]["status"] == "ready"

    @pytest.mark.asyncio
    async def test_update_node_statuses_empty(self):
        """Test update with empty updates dict."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)

        result = await updater.update_node_statuses({})

        assert result.success
        assert result.nodes_updated == []

    @pytest.mark.asyncio
    async def test_update_node_statuses_file_error(self):
        """Test update with file system error."""
        updater = ConfigUpdater(
            config_path=Path("/nonexistent/config.yaml"),
            dry_run=False,
        )

        result = await updater.update_node_statuses({"node": "offline"})

        assert not result.success
        assert "not found" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_current_statuses(self):
        """Test getting current statuses."""
        updater = ConfigUpdater(config_path=self.config_path)

        statuses = await updater.get_current_statuses()

        assert statuses["vast-12345"] == "ready"
        assert statuses["lambda-gh200-1"] == "offline"
        assert statuses["runpod-h100"] == "ready"

    @pytest.mark.asyncio
    async def test_get_current_statuses_missing_status(self):
        """Test get_current_statuses with missing status field."""
        # Create config without status field
        config = {
            "hosts": {
                "node-no-status": {
                    "tailscale_ip": "100.1.2.3",
                }
            }
        }
        with open(self.config_path, "w") as f:
            yaml.safe_dump(config, f)

        updater = ConfigUpdater(config_path=self.config_path)
        statuses = await updater.get_current_statuses()

        assert statuses["node-no-status"] == "unknown"

    @pytest.mark.asyncio
    async def test_get_nodes_by_status(self):
        """Test filtering nodes by status."""
        updater = ConfigUpdater(config_path=self.config_path)

        ready_nodes = await updater.get_nodes_by_status("ready")
        offline_nodes = await updater.get_nodes_by_status("offline")

        assert "vast-12345" in ready_nodes
        assert "runpod-h100" in ready_nodes
        assert "lambda-gh200-1" in offline_nodes
        assert len(ready_nodes) == 2
        assert len(offline_nodes) == 1

    @pytest.mark.asyncio
    async def test_get_nodes_by_provider(self):
        """Test filtering nodes by provider."""
        updater = ConfigUpdater(config_path=self.config_path)

        vast_nodes = await updater.get_nodes_by_provider("vast")
        lambda_nodes = await updater.get_nodes_by_provider("lambda")
        runpod_nodes = await updater.get_nodes_by_provider("runpod")

        assert "vast-12345" in vast_nodes
        assert "lambda-gh200-1" in lambda_nodes
        assert "runpod-h100" in runpod_nodes

    @pytest.mark.asyncio
    async def test_get_nodes_by_provider_underscore(self):
        """Test filtering nodes with underscore separator."""
        # Add node with underscore
        config = updater_load_config_from_path(self.config_path)
        config["hosts"]["vast_67890"] = {"status": "ready"}
        with open(self.config_path, "w") as f:
            yaml.safe_dump(config, f)

        updater = ConfigUpdater(config_path=self.config_path)
        vast_nodes = await updater.get_nodes_by_provider("vast")

        assert "vast-12345" in vast_nodes
        assert "vast_67890" in vast_nodes


class TestConfigUpdaterBackups:
    """Tests for ConfigUpdater backup functionality."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "distributed_hosts.yaml"
        self._create_test_config()

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_config(self):
        """Create a test config file."""
        config = {"hosts": {"node1": {"status": "ready"}}}
        with open(self.config_path, "w") as f:
            yaml.safe_dump(config, f)

    def test_create_backup(self):
        """Test backup creation."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)
        backup_path = updater._create_backup()

        assert backup_path is not None
        assert backup_path.exists()
        assert backup_path.parent.name == BACKUP_DIR_NAME

    def test_create_backup_dry_run(self):
        """Test backup creation in dry run mode."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=True)
        backup_path = updater._create_backup()

        assert backup_path is None  # No backup in dry run

    def test_rotate_backups(self):
        """Test backup rotation."""
        updater = ConfigUpdater(
            config_path=self.config_path,
            dry_run=False,
            max_backups=3,
        )

        # Create 5 backups
        backup_dir = updater.backup_dir
        for i in range(5):
            backup_file = backup_dir / f"distributed_hosts_2025010{i}_120000.yaml"
            backup_file.touch()

        updater._rotate_backups()

        # Should only have 3 backups left
        backups = list(backup_dir.glob("distributed_hosts_*.yaml"))
        assert len(backups) == 3

    def test_rotate_backups_dry_run(self):
        """Test rotation doesn't happen in dry run."""
        updater = ConfigUpdater(
            config_path=self.config_path,
            dry_run=True,
            max_backups=1,
        )

        # Create backup dir and files manually
        backup_dir = self.config_path.parent / BACKUP_DIR_NAME
        backup_dir.mkdir(exist_ok=True)
        for i in range(5):
            backup_file = backup_dir / f"distributed_hosts_2025010{i}_120000.yaml"
            backup_file.touch()

        updater._rotate_backups()

        # All 5 should still exist (no rotation in dry run)
        backups = list(backup_dir.glob("distributed_hosts_*.yaml"))
        assert len(backups) == 5


class TestConfigUpdaterAtomicWrites:
    """Tests for ConfigUpdater atomic write functionality."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "distributed_hosts.yaml"
        self._create_test_config()

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_config(self):
        """Create a test config file."""
        config = {"hosts": {"node1": {"status": "ready"}}}
        with open(self.config_path, "w") as f:
            yaml.safe_dump(config, f)

    def test_write_config_atomic(self):
        """Test atomic config write."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)
        new_config = {"hosts": {"node1": {"status": "offline"}}}

        updater._write_config_atomic(new_config)

        # Verify config was written
        with open(self.config_path) as f:
            written_config = yaml.safe_load(f)
        assert written_config["hosts"]["node1"]["status"] == "offline"

    def test_write_config_atomic_dry_run(self):
        """Test atomic write doesn't happen in dry run."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=True)
        new_config = {"hosts": {"node1": {"status": "offline"}}}

        updater._write_config_atomic(new_config)

        # Verify config was NOT written
        with open(self.config_path) as f:
            written_config = yaml.safe_load(f)
        assert written_config["hosts"]["node1"]["status"] == "ready"

    def test_write_config_atomic_no_temp_file_left(self):
        """Test no temp file is left after atomic write."""
        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)
        new_config = {"hosts": {}}

        updater._write_config_atomic(new_config)

        # Check no temp files in directory
        temp_files = list(self.config_path.parent.glob("distributed_hosts_*.yaml"))
        # Should only have backup file (if created), not temp file
        for f in temp_files:
            assert not f.name.startswith("distributed_hosts_") or \
                   BACKUP_DIR_NAME in str(f.parent)


class TestConfigUpdaterErrorHandling:
    """Tests for ConfigUpdater error handling."""

    def setup_method(self):
        """Create temporary directory for each test."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "distributed_hosts.yaml"

    def teardown_method(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @pytest.mark.asyncio
    async def test_update_yaml_parse_error(self):
        """Test handling of YAML parse errors."""
        # Create invalid YAML
        with open(self.config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        updater = ConfigUpdater(config_path=self.config_path, dry_run=False)
        result = await updater.update_node_statuses({"node": "offline"})

        assert not result.success
        assert "YAML" in result.error or "parse" in result.error.lower()

    @pytest.mark.asyncio
    async def test_get_current_statuses_error(self):
        """Test get_current_statuses with file error."""
        updater = ConfigUpdater(
            config_path=Path("/nonexistent/config.yaml"),
            dry_run=False,
        )

        statuses = await updater.get_current_statuses()

        assert statuses == {}

    @pytest.mark.asyncio
    async def test_get_nodes_by_status_error(self):
        """Test get_nodes_by_status with file error."""
        updater = ConfigUpdater(
            config_path=Path("/nonexistent/config.yaml"),
            dry_run=False,
        )

        nodes = await updater.get_nodes_by_status("ready")

        assert nodes == []

    @pytest.mark.asyncio
    async def test_get_nodes_by_provider_error(self):
        """Test get_nodes_by_provider with file error."""
        updater = ConfigUpdater(
            config_path=Path("/nonexistent/config.yaml"),
            dry_run=False,
        )

        nodes = await updater.get_nodes_by_provider("vast")

        assert nodes == []


# Helper function for tests
def updater_load_config_from_path(path: Path) -> dict:
    """Load config from path for test setup."""
    with open(path) as f:
        return yaml.safe_load(f) or {}
