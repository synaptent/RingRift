"""Tests for app.distributed.manifest_replication module.

This module tests manifest replication for fault tolerance.
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from app.distributed.manifest_replication import (
    ManifestReplicator,
    ReplicaHost,
    ReplicationStatus,
    create_replicator_from_config,
)


# =============================================================================
# ReplicaHost Tests
# =============================================================================


class TestReplicaHost:
    """Tests for ReplicaHost dataclass."""

    def test_default_values(self):
        """Test ReplicaHost default values."""
        host = ReplicaHost(name="test", ssh_host="192.168.1.100")
        assert host.name == "test"
        assert host.ssh_host == "192.168.1.100"
        assert host.ssh_user == "ubuntu"
        assert host.ssh_port == 22
        assert host.remote_path == "~/ringrift/ai-service/data/data_manifest.db"
        assert host.enabled is True
        assert host.last_replicated == 0.0
        assert host.last_checksum == ""

    def test_custom_values(self):
        """Test ReplicaHost with custom values."""
        host = ReplicaHost(
            name="custom-host",
            ssh_host="10.0.0.50",
            ssh_user="root",
            ssh_port=2222,
            remote_path="/data/manifest.db",
            enabled=False,
            last_replicated=1234567890.0,
            last_checksum="abc123",
        )
        assert host.ssh_user == "root"
        assert host.ssh_port == 2222
        assert host.remote_path == "/data/manifest.db"
        assert host.enabled is False
        assert host.last_replicated == 1234567890.0
        assert host.last_checksum == "abc123"


# =============================================================================
# ReplicationStatus Tests
# =============================================================================


class TestReplicationStatus:
    """Tests for ReplicationStatus dataclass."""

    def test_required_fields(self):
        """Test ReplicationStatus with required fields."""
        status = ReplicationStatus(
            local_checksum="abc123",
            local_mtime=1234567890.0,
            local_size=1024,
        )
        assert status.local_checksum == "abc123"
        assert status.local_mtime == 1234567890.0
        assert status.local_size == 1024
        assert status.replicas == {}
        assert status.last_replication_time == 0.0
        assert status.replication_count == 0

    def test_with_replicas(self):
        """Test ReplicationStatus with replicas."""
        status = ReplicationStatus(
            local_checksum="def456",
            local_mtime=1234567890.0,
            local_size=2048,
            replicas={"host1": {"last_replicated": 123.0}},
            last_replication_time=100.0,
            replication_count=5,
        )
        assert "host1" in status.replicas
        assert status.last_replication_time == 100.0
        assert status.replication_count == 5


# =============================================================================
# ManifestReplicator Tests
# =============================================================================


@pytest.fixture
def temp_manifest_db(tmp_path):
    """Create a temporary manifest database."""
    db_path = tmp_path / "data_manifest.db"

    # Create minimal manifest DB
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS synced_games (
            game_id TEXT PRIMARY KEY,
            config_key TEXT,
            synced_at REAL
        )
    """)
    # Add some test data
    cursor.execute(
        "INSERT INTO synced_games VALUES (?, ?, ?)",
        ("game-1", "hex8_2p", time.time()),
    )
    cursor.execute(
        "INSERT INTO synced_games VALUES (?, ?, ?)",
        ("game-2", "hex8_2p", time.time()),
    )
    conn.commit()
    conn.close()

    return db_path


@pytest.fixture
def replica_hosts():
    """Create test replica hosts."""
    return [
        ReplicaHost(name="host-1", ssh_host="192.168.1.100"),
        ReplicaHost(name="host-2", ssh_host="192.168.1.101"),
        ReplicaHost(name="host-3", ssh_host="192.168.1.102"),
    ]


class TestManifestReplicator:
    """Tests for ManifestReplicator class."""

    def test_init(self, temp_manifest_db, replica_hosts):
        """Test initialization."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
        )
        assert replicator.local_path == temp_manifest_db
        assert len(replicator.replica_hosts) == 3
        assert replicator.min_replicas == 3

    def test_init_with_custom_params(self, temp_manifest_db, replica_hosts):
        """Test initialization with custom parameters."""
        external_backup = Path("/tmp/backup/manifest.db")
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
            min_replicas=2,
            replication_interval_seconds=600,
            ssh_timeout=60,
            scp_timeout=180,
            external_backup_path=external_backup,
        )
        assert replicator.min_replicas == 2
        assert replicator.replication_interval == 600
        assert replicator.ssh_timeout == 60
        assert replicator.scp_timeout == 180
        assert replicator.external_backup_path == external_backup

    def test_get_local_manifest_info_exists(self, temp_manifest_db, replica_hosts):
        """Test _get_local_manifest_info when file exists."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
        )
        checksum, mtime, size = replicator._get_local_manifest_info()

        assert len(checksum) == 64  # SHA256 hex
        assert mtime > 0
        assert size > 0

    def test_get_local_manifest_info_not_exists(self, tmp_path, replica_hosts):
        """Test _get_local_manifest_info when file doesn't exist."""
        missing_path = tmp_path / "missing.db"
        replicator = ManifestReplicator(
            local_manifest_path=missing_path,
            replica_hosts=replica_hosts,
        )
        checksum, mtime, size = replicator._get_local_manifest_info()

        assert checksum == ""
        assert mtime == 0.0
        assert size == 0

    def test_build_ssh_args_default_port(self, temp_manifest_db, replica_hosts):
        """Test SSH args with default port."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
            ssh_timeout=30,
        )
        host = ReplicaHost(name="test", ssh_host="192.168.1.1")
        args = replicator._build_ssh_args(host)

        assert "-o ConnectTimeout=30" in args
        assert "-o StrictHostKeyChecking=no" in args
        assert "-o BatchMode=yes" in args
        assert "-p" not in args  # Default port, no -p flag

    def test_build_ssh_args_custom_port(self, temp_manifest_db, replica_hosts):
        """Test SSH args with custom port."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
        )
        host = ReplicaHost(name="test", ssh_host="192.168.1.1", ssh_port=2222)
        args = replicator._build_ssh_args(host)

        assert "-p 2222" in args

    def test_get_status(self, temp_manifest_db, replica_hosts):
        """Test get_status returns correct structure."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
        )
        status = replicator.get_status()

        assert "local_checksum" in status
        assert "local_mtime" in status
        assert "local_size" in status
        assert "last_replication_time" in status
        assert "replication_count" in status
        assert "replicas" in status
        assert "healthy" in status
        assert len(status["replicas"]) == 3


class TestManifestReplicatorAsync:
    """Async tests for ManifestReplicator."""

    @pytest.mark.asyncio
    async def test_replicate_to_host_disabled(self, temp_manifest_db, replica_hosts):
        """Test replication skips disabled hosts."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
        )
        host = ReplicaHost(name="disabled", ssh_host="192.168.1.1", enabled=False)

        result = await replicator._replicate_to_host(host)
        assert result is False

    @pytest.mark.asyncio
    async def test_replicate_to_host_no_local_file(self, tmp_path, replica_hosts):
        """Test replication fails when local file doesn't exist."""
        missing_path = tmp_path / "missing.db"
        replicator = ManifestReplicator(
            local_manifest_path=missing_path,
            replica_hosts=replica_hosts,
        )
        host = ReplicaHost(name="test", ssh_host="192.168.1.1")

        result = await replicator._replicate_to_host(host)
        assert result is False

    @pytest.mark.asyncio
    async def test_replicate_async_no_change(self, temp_manifest_db, replica_hosts):
        """Test replicate_async returns 0 when no change."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
            replication_interval_seconds=300,
        )

        # Set status to match current file
        checksum, mtime, size = replicator._get_local_manifest_info()
        replicator._status.local_checksum = checksum
        replicator._status.last_replication_time = time.time()

        result = await replicator.replicate_async(force=False)
        assert result == 0  # No replication needed

    @pytest.mark.asyncio
    async def test_replicate_async_force(self, temp_manifest_db, replica_hosts):
        """Test replicate_async with force flag."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=[],  # No hosts to actually replicate to
        )

        result = await replicator.replicate_async(force=True)
        assert result == 0  # No hosts configured

    @pytest.mark.asyncio
    async def test_replicate_async_no_hosts(self, temp_manifest_db):
        """Test replicate_async with no replica hosts."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=[],
        )

        result = await replicator.replicate_async(force=True)
        assert result == 0

    @pytest.mark.asyncio
    async def test_recover_if_needed_local_exists(self, temp_manifest_db, replica_hosts):
        """Test recovery is skipped when local is up-to-date."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
        )

        # Mock _get_remote_manifest_info to return older data
        async def mock_get_remote(*args):
            return ("older_checksum", time.time() - 3600, 1024)  # 1 hour old

        with patch.object(replicator, "_get_remote_manifest_info", mock_get_remote):
            result = await replicator.recover_if_needed()
            # Local is newer, no recovery needed
            assert result is False

    @pytest.mark.asyncio
    async def test_recover_if_needed_no_replicas(self, temp_manifest_db):
        """Test recovery returns False when no replicas available."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=[],
        )

        result = await replicator.recover_if_needed()
        assert result is False


class TestExternalBackup:
    """Tests for external drive backup functionality."""

    @pytest.mark.asyncio
    async def test_backup_to_external_no_path(self, temp_manifest_db, replica_hosts):
        """Test backup skips when no external path configured."""
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
            external_backup_path=None,
        )

        result = await replicator._backup_to_external_drive()
        assert result is False

    @pytest.mark.asyncio
    async def test_backup_to_external_rate_limited(self, temp_manifest_db, replica_hosts, tmp_path):
        """Test backup is rate limited."""
        external_path = tmp_path / "backup" / "manifest.db"
        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
            external_backup_path=external_path,
        )
        # Set last backup to recent time
        replicator._last_external_backup = time.time()

        result = await replicator._backup_to_external_drive()
        assert result is True  # Returns True (skipped, considered success)

    @pytest.mark.asyncio
    async def test_backup_to_external_success(self, temp_manifest_db, replica_hosts, tmp_path):
        """Test successful backup to external drive."""
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        external_path = backup_dir / "manifest.db"

        replicator = ManifestReplicator(
            local_manifest_path=temp_manifest_db,
            replica_hosts=replica_hosts,
            external_backup_path=external_path,
        )
        replicator._last_external_backup = 0  # Allow backup

        result = await replicator._backup_to_external_drive()
        assert result is True
        assert external_path.exists()

    @pytest.mark.asyncio
    async def test_recover_from_external_success(self, temp_manifest_db, replica_hosts, tmp_path):
        """Test recovery from external drive."""
        # Create backup
        backup_dir = tmp_path / "backup"
        backup_dir.mkdir()
        external_path = backup_dir / "manifest.db"

        # Create external backup DB
        conn = sqlite3.connect(external_path)
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE synced_games (
                game_id TEXT PRIMARY KEY,
                config_key TEXT,
                synced_at REAL
            )
        """)
        cursor.execute("INSERT INTO synced_games VALUES ('g1', 'hex8_2p', ?)", (time.time(),))
        conn.commit()
        conn.close()

        # Remove local file
        new_local = tmp_path / "new_manifest.db"

        replicator = ManifestReplicator(
            local_manifest_path=new_local,
            replica_hosts=replica_hosts,
            external_backup_path=external_path,
        )

        result = await replicator.recover_from_external_drive()
        assert result is True
        assert new_local.exists()

    @pytest.mark.asyncio
    async def test_recover_from_external_not_exists(self, tmp_path, replica_hosts):
        """Test recovery fails when external backup doesn't exist."""
        replicator = ManifestReplicator(
            local_manifest_path=tmp_path / "manifest.db",
            replica_hosts=replica_hosts,
            external_backup_path=tmp_path / "nonexistent" / "backup.db",
        )

        result = await replicator.recover_from_external_drive()
        assert result is False


# =============================================================================
# Factory Function Tests
# =============================================================================


class TestCreateReplicatorFromConfig:
    """Tests for create_replicator_from_config factory."""

    def test_create_with_empty_config(self, tmp_path):
        """Test creation with empty config file."""
        manifest_path = tmp_path / "manifest.db"
        config_path = tmp_path / "hosts.yaml"

        # Create empty config
        config_path.write_text("hosts: {}")

        replicator = create_replicator_from_config(
            manifest_path=manifest_path,
            hosts_config_path=config_path,
        )

        assert replicator.local_path == manifest_path
        assert len(replicator.replica_hosts) == 0

    def test_create_with_valid_hosts(self, tmp_path):
        """Test creation with valid host config."""
        manifest_path = tmp_path / "manifest.db"
        config_path = tmp_path / "hosts.yaml"

        # Create config with hosts
        config_path.write_text("""
hosts:
  host-1:
    tailscale_ip: "192.168.1.100"
    ssh_user: ubuntu
    p2p_enabled: true
    status: ready
  host-2:
    ssh_host: "10.0.0.50"
    ssh_user: root
    ssh_port: 2222
    p2p_enabled: true
    status: ready
  host-3:
    tailscale_ip: "192.168.1.102"
    status: terminated
    p2p_enabled: true
""")

        replicator = create_replicator_from_config(
            manifest_path=manifest_path,
            hosts_config_path=config_path,
        )

        # host-3 should be excluded (terminated)
        assert len(replicator.replica_hosts) == 2
        assert "host-1" in replicator.replica_hosts
        assert "host-2" in replicator.replica_hosts
        assert "host-3" not in replicator.replica_hosts

    def test_create_excludes_disabled_p2p(self, tmp_path):
        """Test that hosts with p2p_enabled=false are excluded."""
        manifest_path = tmp_path / "manifest.db"
        config_path = tmp_path / "hosts.yaml"

        config_path.write_text("""
hosts:
  host-1:
    tailscale_ip: "192.168.1.100"
    p2p_enabled: true
    status: ready
  host-2:
    tailscale_ip: "192.168.1.101"
    p2p_enabled: false
    status: ready
""")

        replicator = create_replicator_from_config(
            manifest_path=manifest_path,
            hosts_config_path=config_path,
        )

        assert len(replicator.replica_hosts) == 1
        assert "host-1" in replicator.replica_hosts

    def test_create_excludes_training_only(self, tmp_path):
        """Test that training-only hosts are excluded."""
        manifest_path = tmp_path / "manifest.db"
        config_path = tmp_path / "hosts.yaml"

        config_path.write_text("""
hosts:
  training-node:
    tailscale_ip: "192.168.1.100"
    p2p_enabled: true
    role: training
    status: ready
  selfplay-node:
    tailscale_ip: "192.168.1.101"
    p2p_enabled: true
    role: selfplay
    status: ready
""")

        replicator = create_replicator_from_config(
            manifest_path=manifest_path,
            hosts_config_path=config_path,
        )

        assert len(replicator.replica_hosts) == 1
        assert "selfplay-node" in replicator.replica_hosts

    def test_create_limits_hosts(self, tmp_path):
        """Test that hosts are limited to 5."""
        manifest_path = tmp_path / "manifest.db"
        config_path = tmp_path / "hosts.yaml"

        # Create config with many hosts
        hosts = "\n".join([
            f"  host-{i}:\n    tailscale_ip: \"192.168.1.{i}\"\n    p2p_enabled: true\n    status: ready"
            for i in range(10)
        ])
        config_path.write_text(f"hosts:\n{hosts}")

        replicator = create_replicator_from_config(
            manifest_path=manifest_path,
            hosts_config_path=config_path,
        )

        assert len(replicator.replica_hosts) == 5  # Limited to 5

    def test_create_with_custom_min_replicas(self, tmp_path):
        """Test creation with custom min_replicas."""
        manifest_path = tmp_path / "manifest.db"
        config_path = tmp_path / "hosts.yaml"
        config_path.write_text("hosts: {}")

        replicator = create_replicator_from_config(
            manifest_path=manifest_path,
            hosts_config_path=config_path,
            min_replicas=5,
        )

        assert replicator.min_replicas == 5

    def test_create_with_external_backup(self, tmp_path):
        """Test creation with external backup path."""
        manifest_path = tmp_path / "manifest.db"
        config_path = tmp_path / "hosts.yaml"
        external_path = tmp_path / "backup" / "manifest.db"
        config_path.write_text("hosts: {}")

        replicator = create_replicator_from_config(
            manifest_path=manifest_path,
            hosts_config_path=config_path,
            external_backup_path=external_path,
        )

        assert replicator.external_backup_path == external_path

    def test_create_with_legacy_schema(self, tmp_path):
        """Test creation with legacy remote_hosts.yaml schema."""
        manifest_path = tmp_path / "manifest.db"
        config_path = tmp_path / "hosts.yaml"

        # Legacy schema uses standard_hosts
        config_path.write_text("""
standard_hosts:
  legacy-host:
    ssh_host: "192.168.1.100"
    ssh_user: ubuntu
""")

        replicator = create_replicator_from_config(
            manifest_path=manifest_path,
            hosts_config_path=config_path,
        )

        assert len(replicator.replica_hosts) == 1
        assert "legacy-host" in replicator.replica_hosts


# =============================================================================
# Module Exports Tests
# =============================================================================


class TestModuleExports:
    """Tests for module exports."""

    def test_replicahost_exported(self):
        """Test ReplicaHost is exported."""
        from app.distributed.manifest_replication import ReplicaHost
        assert ReplicaHost is not None

    def test_replicationstatus_exported(self):
        """Test ReplicationStatus is exported."""
        from app.distributed.manifest_replication import ReplicationStatus
        assert ReplicationStatus is not None

    def test_manifestreplicator_exported(self):
        """Test ManifestReplicator is exported."""
        from app.distributed.manifest_replication import ManifestReplicator
        assert ManifestReplicator is not None

    def test_factory_exported(self):
        """Test create_replicator_from_config is exported."""
        from app.distributed.manifest_replication import create_replicator_from_config
        assert create_replicator_from_config is not None
