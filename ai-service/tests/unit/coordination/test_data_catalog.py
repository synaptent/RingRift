"""Tests for data_catalog.py - Central data location registry.

This module tests the DataCatalog class and related utilities for
tracking data locations across the cluster.

Test coverage: DataType, DataEntry, DataCatalogConfig, DataCatalog,
singleton functions.
"""

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.coordination.data_catalog import (
    DataCatalog,
    DataCatalogConfig,
    DataEntry,
    DataType,
    get_data_catalog,
    reset_data_catalog,
)


# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_catalog.db"


@pytest.fixture
def config(temp_db_path):
    """Create test configuration with temporary database."""
    return DataCatalogConfig(
        db_path=temp_db_path,
        auto_persist=True,
        manifest_refresh_interval=30.0,
        stale_entry_threshold=3600.0,
        min_replication_factor=2,
    )


@pytest.fixture
def catalog(config):
    """Create a test catalog instance."""
    return DataCatalog(config)


@pytest.fixture
def sample_entry():
    """Create a sample DataEntry for testing."""
    return DataEntry(
        path="games/canonical_hex8_2p.db",
        data_type=DataType.GAMES,
        config_key="hex8_2p",
        size_bytes=1024000,
        checksum="sha256:abc123",
        mtime=time.time(),
        locations={"node-1", "node-2"},
        primary_location="node-1",
        game_count=1000,
    )


@pytest.fixture(autouse=True)
def reset_singleton():
    """Reset singleton before and after each test."""
    reset_data_catalog()
    yield
    reset_data_catalog()


# =============================================================================
# Test DataType Enum
# =============================================================================


class TestDataType:
    """Tests for DataType enum."""

    def test_games_value(self):
        """DataType.GAMES should have correct value."""
        assert DataType.GAMES.value == "games"

    def test_models_value(self):
        """DataType.MODELS should have correct value."""
        assert DataType.MODELS.value == "models"

    def test_npz_value(self):
        """DataType.NPZ should have correct value."""
        assert DataType.NPZ.value == "npz"

    def test_checkpoint_value(self):
        """DataType.CHECKPOINT should have correct value."""
        assert DataType.CHECKPOINT.value == "checkpoint"

    def test_config_value(self):
        """DataType.CONFIG should have correct value."""
        assert DataType.CONFIG.value == "config"

    def test_log_value(self):
        """DataType.LOG should have correct value."""
        assert DataType.LOG.value == "log"

    def test_unknown_value(self):
        """DataType.UNKNOWN should have correct value."""
        assert DataType.UNKNOWN.value == "unknown"

    def test_from_path_db(self):
        """from_path should detect .db as GAMES."""
        assert DataType.from_path("games/canonical_hex8.db") == DataType.GAMES
        assert DataType.from_path("data/GAMES.DB") == DataType.GAMES

    def test_from_path_pth(self):
        """from_path should detect .pth as MODELS."""
        assert DataType.from_path("models/best_model.pth") == DataType.MODELS

    def test_from_path_pt(self):
        """from_path should detect .pt as MODELS."""
        assert DataType.from_path("models/model.pt") == DataType.MODELS

    def test_from_path_npz(self):
        """from_path should detect .npz as NPZ."""
        assert DataType.from_path("training/hex8_2p.npz") == DataType.NPZ

    def test_from_path_checkpoint(self):
        """from_path should detect checkpoint in path as CHECKPOINT."""
        # Note: .pth extension is checked before "checkpoint" in path
        # So use a path without .pth to trigger checkpoint detection
        assert DataType.from_path("checkpoints/epoch_10.bin") == DataType.CHECKPOINT
        assert DataType.from_path("data/checkpoint_latest") == DataType.CHECKPOINT

    def test_from_path_yaml(self):
        """from_path should detect .yaml as CONFIG."""
        assert DataType.from_path("config/hosts.yaml") == DataType.CONFIG
        assert DataType.from_path("settings.yml") == DataType.CONFIG

    def test_from_path_json(self):
        """from_path should detect .json as CONFIG."""
        assert DataType.from_path("config.json") == DataType.CONFIG

    def test_from_path_log(self):
        """from_path should detect .log as LOG."""
        assert DataType.from_path("logs/training.log") == DataType.LOG

    def test_from_path_unknown(self):
        """from_path should return UNKNOWN for unrecognized extensions."""
        assert DataType.from_path("data/file.txt") == DataType.UNKNOWN
        assert DataType.from_path("readme.md") == DataType.UNKNOWN


# =============================================================================
# Test DataEntry Dataclass
# =============================================================================


class TestDataEntry:
    """Tests for DataEntry dataclass."""

    def test_basic_construction(self, sample_entry):
        """DataEntry should construct with required fields."""
        assert sample_entry.path == "games/canonical_hex8_2p.db"
        assert sample_entry.data_type == DataType.GAMES
        assert sample_entry.config_key == "hex8_2p"
        assert sample_entry.size_bytes == 1024000

    def test_locations_as_set(self, sample_entry):
        """Locations should be a set."""
        assert isinstance(sample_entry.locations, set)
        assert "node-1" in sample_entry.locations
        assert "node-2" in sample_entry.locations

    def test_locations_from_list(self):
        """Locations should be converted from list to set."""
        entry = DataEntry(
            path="test.db",
            data_type=DataType.GAMES,
            config_key="test",
            size_bytes=100,
            checksum="abc",
            mtime=time.time(),
            locations=["node-1", "node-2"],  # List input
            primary_location="node-1",
        )
        assert isinstance(entry.locations, set)
        assert entry.locations == {"node-1", "node-2"}

    def test_data_type_from_string(self):
        """data_type should be converted from string to enum."""
        entry = DataEntry(
            path="test.db",
            data_type="games",  # String input
            config_key="test",
            size_bytes=100,
            checksum="abc",
            mtime=time.time(),
            locations=set(),
            primary_location="node-1",
        )
        assert entry.data_type == DataType.GAMES

    def test_filename_property(self, sample_entry):
        """filename property should return just the filename."""
        assert sample_entry.filename == "canonical_hex8_2p.db"

    def test_replication_factor_property(self, sample_entry):
        """replication_factor should return location count."""
        assert sample_entry.replication_factor == 2

    def test_replication_factor_empty(self):
        """replication_factor should be 0 for empty locations."""
        entry = DataEntry(
            path="test.db",
            data_type=DataType.GAMES,
            config_key="test",
            size_bytes=100,
            checksum="abc",
            mtime=time.time(),
            locations=set(),
            primary_location="",
        )
        assert entry.replication_factor == 0

    def test_to_dict(self, sample_entry):
        """to_dict should serialize all fields."""
        data = sample_entry.to_dict()

        assert data["path"] == "games/canonical_hex8_2p.db"
        assert data["data_type"] == "games"
        assert data["config_key"] == "hex8_2p"
        assert data["size_bytes"] == 1024000
        assert data["checksum"] == "sha256:abc123"
        assert set(data["locations"]) == {"node-1", "node-2"}
        assert data["primary_location"] == "node-1"
        assert data["game_count"] == 1000

    def test_from_dict(self, sample_entry):
        """from_dict should deserialize correctly."""
        data = sample_entry.to_dict()
        restored = DataEntry.from_dict(data)

        assert restored.path == sample_entry.path
        assert restored.data_type == sample_entry.data_type
        assert restored.config_key == sample_entry.config_key
        assert restored.size_bytes == sample_entry.size_bytes
        assert restored.locations == sample_entry.locations
        assert restored.game_count == sample_entry.game_count

    def test_from_dict_with_missing_optional(self):
        """from_dict should handle missing optional fields."""
        data = {
            "path": "test.db",
            "data_type": "games",
            "locations": ["node-1"],
        }
        entry = DataEntry.from_dict(data)

        assert entry.path == "test.db"
        assert entry.config_key == ""
        assert entry.size_bytes == 0
        assert entry.game_count == 0


# =============================================================================
# Test DataCatalogConfig
# =============================================================================


class TestDataCatalogConfig:
    """Tests for DataCatalogConfig."""

    def test_default_values(self):
        """Config should have sensible defaults."""
        config = DataCatalogConfig()

        assert config.auto_persist is True
        assert config.manifest_refresh_interval == 60.0
        assert config.stale_entry_threshold == 86400.0
        assert config.min_replication_factor == 3

    def test_custom_values(self, temp_db_path):
        """Config should accept custom values."""
        config = DataCatalogConfig(
            db_path=temp_db_path,
            auto_persist=False,
            min_replication_factor=5,
        )

        assert config.db_path == temp_db_path
        assert config.auto_persist is False
        assert config.min_replication_factor == 5

    def test_db_path_from_env(self, monkeypatch):
        """db_path should use RINGRIFT_CATALOG_DB env var."""
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = f"{tmpdir}/env_catalog.db"
            monkeypatch.setenv("RINGRIFT_CATALOG_DB", env_path)

            config = DataCatalogConfig()
            assert str(config.db_path) == env_path


# =============================================================================
# Test DataCatalog - Initialization
# =============================================================================


class TestDataCatalogInit:
    """Tests for DataCatalog initialization."""

    def test_init_with_defaults(self, temp_db_path):
        """Catalog should initialize with default config."""
        config = DataCatalogConfig(db_path=temp_db_path)
        catalog = DataCatalog(config)

        assert catalog.config == config
        assert len(catalog._entries) == 0

    def test_init_creates_database(self, config):
        """Catalog should create database file."""
        catalog = DataCatalog(config)

        assert config.db_path.exists()

    def test_init_without_auto_persist(self, temp_db_path):
        """Catalog should work without persistence."""
        config = DataCatalogConfig(db_path=temp_db_path, auto_persist=False)
        catalog = DataCatalog(config)

        # Database should not be created
        assert not temp_db_path.exists()

    def test_init_stats(self, catalog):
        """Catalog should initialize stats."""
        stats = catalog.get_stats()

        assert stats["total_entries"] == 0
        assert stats["total_bytes"] == 0
        assert stats["registrations"] == 0
        assert stats["queries"] == 0


# =============================================================================
# Test DataCatalog - Registration
# =============================================================================


class TestDataCatalogRegister:
    """Tests for registration methods."""

    def test_register_new_entry(self, catalog, sample_entry):
        """register should add new entry."""
        catalog.register(sample_entry)

        entry = catalog.get(sample_entry.path)
        assert entry is not None
        assert entry.path == sample_entry.path
        assert entry.config_key == "hex8_2p"

    def test_register_updates_stats(self, catalog, sample_entry):
        """register should update statistics."""
        catalog.register(sample_entry)

        stats = catalog.get_stats()
        assert stats["total_entries"] == 1
        assert stats["registrations"] == 1
        assert stats["total_bytes"] == 1024000

    def test_register_merges_locations(self, catalog, sample_entry):
        """register should merge locations for existing entry."""
        catalog.register(sample_entry)

        # Register again with different location
        updated = DataEntry(
            path=sample_entry.path,
            data_type=DataType.GAMES,
            config_key="hex8_2p",
            size_bytes=1024000,
            checksum="sha256:abc123",
            mtime=time.time(),
            locations={"node-3"},
            primary_location="node-1",
        )
        catalog.register(updated)

        entry = catalog.get(sample_entry.path)
        assert "node-1" in entry.locations
        assert "node-2" in entry.locations
        assert "node-3" in entry.locations

    def test_register_preserves_created_at(self, catalog, sample_entry):
        """register should preserve created_at for existing entry."""
        catalog.register(sample_entry)
        original_created = catalog.get(sample_entry.path).created_at

        time.sleep(0.01)

        updated = DataEntry(
            path=sample_entry.path,
            data_type=DataType.GAMES,
            config_key="hex8_2p",
            size_bytes=2000000,  # Different size
            checksum="sha256:def456",
            mtime=time.time(),
            locations={"node-3"},
            primary_location="node-3",
        )
        catalog.register(updated)

        entry = catalog.get(sample_entry.path)
        assert entry.created_at == original_created

    def test_register_from_manifest(self, catalog):
        """register_from_manifest should register multiple entries."""
        manifest = {
            "games/hex8_2p.db": {
                "size": 1000000,
                "mtime": time.time(),
                "type": "database",
                "sha256": "abc123",
            },
            "models/best_hex8_2p.pth": {
                "size": 50000000,
                "mtime": time.time(),
                "type": "model",
            },
            "training/hex8_2p.npz": {
                "size": 200000000,
                "mtime": time.time(),
                "type": "npz",
            },
        }

        count = catalog.register_from_manifest("node-1", manifest)

        assert count == 3
        assert catalog.get("games/hex8_2p.db") is not None
        assert catalog.get("models/best_hex8_2p.pth") is not None
        assert catalog.get("training/hex8_2p.npz") is not None

    def test_register_from_manifest_extracts_config_key(self, catalog):
        """register_from_manifest should extract config_key from path."""
        manifest = {
            "games/canonical_square8_4p.db": {"size": 100, "mtime": time.time()},
        }

        catalog.register_from_manifest("node-1", manifest)

        entry = catalog.get("games/canonical_square8_4p.db")
        assert entry.config_key == "square8_4p"

    def test_register_from_manifest_infers_type(self, catalog):
        """register_from_manifest should infer type from extension."""
        manifest = {
            "data/file.db": {"size": 100, "mtime": time.time()},
            "data/model.pth": {"size": 100, "mtime": time.time()},
        }

        catalog.register_from_manifest("node-1", manifest)

        assert catalog.get("data/file.db").data_type == DataType.GAMES
        assert catalog.get("data/model.pth").data_type == DataType.MODELS


# =============================================================================
# Test DataCatalog - Mark Synced/Removed
# =============================================================================


class TestDataCatalogMarkSynced:
    """Tests for mark_synced and mark_removed methods."""

    def test_mark_synced_adds_location(self, catalog, sample_entry):
        """mark_synced should add node to locations."""
        catalog.register(sample_entry)

        catalog.mark_synced(sample_entry.path, "node-3")

        entry = catalog.get(sample_entry.path)
        assert "node-3" in entry.locations

    def test_mark_synced_idempotent(self, catalog, sample_entry):
        """mark_synced should be idempotent."""
        catalog.register(sample_entry)
        original_count = len(catalog.get(sample_entry.path).locations)

        catalog.mark_synced(sample_entry.path, "node-1")  # Already exists

        entry = catalog.get(sample_entry.path)
        assert len(entry.locations) == original_count

    def test_mark_synced_unknown_path(self, catalog):
        """mark_synced should be no-op for unknown path."""
        catalog.mark_synced("unknown/path.db", "node-1")
        # Should not raise

    def test_mark_removed_removes_location(self, catalog, sample_entry):
        """mark_removed should remove node from locations."""
        catalog.register(sample_entry)

        catalog.mark_removed(sample_entry.path, "node-1")

        entry = catalog.get(sample_entry.path)
        assert "node-1" not in entry.locations
        assert "node-2" in entry.locations

    def test_mark_removed_idempotent(self, catalog, sample_entry):
        """mark_removed should be idempotent."""
        catalog.register(sample_entry)
        catalog.mark_removed(sample_entry.path, "node-1")

        catalog.mark_removed(sample_entry.path, "node-1")  # Already removed
        # Should not raise


# =============================================================================
# Test DataCatalog - Queries
# =============================================================================


class TestDataCatalogQueries:
    """Tests for query methods."""

    def test_get_existing(self, catalog, sample_entry):
        """get should return existing entry."""
        catalog.register(sample_entry)

        entry = catalog.get(sample_entry.path)
        assert entry is not None
        assert entry.path == sample_entry.path

    def test_get_nonexistent(self, catalog):
        """get should return None for nonexistent path."""
        entry = catalog.get("nonexistent/path.db")
        assert entry is None

    def test_get_increments_queries(self, catalog, sample_entry):
        """get should increment query count."""
        catalog.register(sample_entry)

        catalog.get(sample_entry.path)
        catalog.get(sample_entry.path)

        stats = catalog.get_stats()
        assert stats["queries"] == 2

    def test_get_all_no_filter(self, catalog):
        """get_all without filter should return all entries."""
        for i in range(5):
            catalog.register(
                DataEntry(
                    path=f"data/file{i}.db",
                    data_type=DataType.GAMES,
                    config_key="test",
                    size_bytes=100,
                    checksum="abc",
                    mtime=time.time(),
                    locations={"node-1"},
                    primary_location="node-1",
                )
            )

        entries = catalog.get_all()
        assert len(entries) == 5

    def test_get_all_filter_by_type(self, catalog):
        """get_all should filter by data_type."""
        catalog.register(
            DataEntry(
                path="data.db",
                data_type=DataType.GAMES,
                config_key="",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1"},
                primary_location="node-1",
            )
        )
        catalog.register(
            DataEntry(
                path="model.pth",
                data_type=DataType.MODELS,
                config_key="",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1"},
                primary_location="node-1",
            )
        )

        games = catalog.get_all(data_type=DataType.GAMES)
        models = catalog.get_all(data_type=DataType.MODELS)

        assert len(games) == 1
        assert len(models) == 1
        assert games[0].path == "data.db"
        assert models[0].path == "model.pth"

    def test_get_all_filter_by_config(self, catalog):
        """get_all should filter by config_key."""
        catalog.register(
            DataEntry(
                path="hex8_2p.db",
                data_type=DataType.GAMES,
                config_key="hex8_2p",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1"},
                primary_location="node-1",
            )
        )
        catalog.register(
            DataEntry(
                path="square8_4p.db",
                data_type=DataType.GAMES,
                config_key="square8_4p",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1"},
                primary_location="node-1",
            )
        )

        hex8 = catalog.get_all(config_key="hex8_2p")
        square8 = catalog.get_all(config_key="square8_4p")

        assert len(hex8) == 1
        assert len(square8) == 1

    def test_get_on_node(self, catalog, sample_entry):
        """get_on_node should return entries on specific node."""
        catalog.register(sample_entry)

        entries = catalog.get_on_node("node-1")
        assert len(entries) == 1
        assert entries[0].path == sample_entry.path

    def test_get_on_node_empty(self, catalog, sample_entry):
        """get_on_node should return empty for unknown node."""
        catalog.register(sample_entry)

        entries = catalog.get_on_node("node-99")
        assert len(entries) == 0

    def test_get_missing_on_node(self, catalog):
        """get_missing_on_node should return entries not on node."""
        catalog.register(
            DataEntry(
                path="on_node1.db",
                data_type=DataType.GAMES,
                config_key="test",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1"},
                primary_location="node-1",
            )
        )
        catalog.register(
            DataEntry(
                path="on_both.db",
                data_type=DataType.GAMES,
                config_key="test",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1", "node-2"},
                primary_location="node-1",
            )
        )

        missing = catalog.get_missing_on_node("node-2")
        assert len(missing) == 1
        assert missing[0].path == "on_node1.db"

    def test_get_under_replicated(self, catalog):
        """get_under_replicated should return entries below min factor."""
        catalog.register(
            DataEntry(
                path="single.db",
                data_type=DataType.GAMES,
                config_key="test",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1"},
                primary_location="node-1",
            )
        )
        catalog.register(
            DataEntry(
                path="triple.db",
                data_type=DataType.GAMES,
                config_key="test",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1", "node-2", "node-3"},
                primary_location="node-1",
            )
        )

        # Min factor is 2 in test config
        under_rep = catalog.get_under_replicated()
        assert len(under_rep) == 1
        assert under_rep[0].path == "single.db"

    def test_get_replication_factor(self, catalog, sample_entry):
        """get_replication_factor should return location count."""
        catalog.register(sample_entry)

        factor = catalog.get_replication_factor(sample_entry.path)
        assert factor == 2

    def test_get_replication_factor_unknown(self, catalog):
        """get_replication_factor should return 0 for unknown path."""
        factor = catalog.get_replication_factor("unknown.db")
        assert factor == 0

    def test_get_nodes_with_data(self, catalog, sample_entry):
        """get_nodes_with_data should return set of nodes."""
        catalog.register(sample_entry)

        nodes = catalog.get_nodes_with_data(sample_entry.path)
        assert nodes == {"node-1", "node-2"}

    def test_get_nodes_with_data_unknown(self, catalog):
        """get_nodes_with_data should return empty set for unknown."""
        nodes = catalog.get_nodes_with_data("unknown.db")
        assert nodes == set()

    def test_get_primary_location(self, catalog, sample_entry):
        """get_primary_location should return primary node."""
        catalog.register(sample_entry)

        primary = catalog.get_primary_location(sample_entry.path)
        assert primary == "node-1"

    def test_get_stale_entries(self, catalog):
        """get_stale_entries should return old entries."""
        # Register an entry first
        catalog.register(
            DataEntry(
                path="old.db",
                data_type=DataType.GAMES,
                config_key="test",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1"},
                primary_location="node-1",
            )
        )

        # Manually set updated_at to old time (register sets it to now)
        # This is necessary because register() overwrites updated_at
        old_time = time.time() - 7200  # 2 hours ago
        catalog._entries["old.db"].updated_at = old_time

        # Stale threshold is 3600s (1 hour) in test config
        stale = catalog.get_stale_entries()
        assert len(stale) == 1
        assert stale[0].path == "old.db"


# =============================================================================
# Test DataCatalog - Stats and Health
# =============================================================================


class TestDataCatalogStats:
    """Tests for statistics and health check."""

    def test_get_stats_empty(self, catalog):
        """get_stats should work with empty catalog."""
        stats = catalog.get_stats()

        assert stats["total_entries"] == 0
        assert stats["total_bytes"] == 0
        assert stats["nodes_tracked"] == 0

    def test_get_stats_with_entries(self, catalog, sample_entry):
        """get_stats should reflect registered entries."""
        catalog.register(sample_entry)

        stats = catalog.get_stats()

        assert stats["total_entries"] == 1
        assert stats["total_bytes"] == 1024000
        assert stats["registrations"] == 1
        assert DataType.GAMES.value in str(stats["entries_by_type"])

    def test_health_check_healthy(self, catalog, sample_entry):
        """health_check should return healthy for good state."""
        # Register entry with good replication
        sample_entry.locations = {"node-1", "node-2", "node-3"}
        catalog.register(sample_entry)

        result = catalog.health_check()

        assert result.healthy is True
        assert "healthy" in result.message.lower()

    def test_health_check_empty(self, catalog):
        """health_check should report issue for empty catalog."""
        result = catalog.health_check()

        assert result.healthy is False
        assert "no entries" in result.message.lower()

    def test_health_check_under_replicated(self, catalog):
        """health_check should report under-replicated entries."""
        catalog.register(
            DataEntry(
                path="single.db",
                data_type=DataType.GAMES,
                config_key="test",
                size_bytes=100,
                checksum="abc",
                mtime=time.time(),
                locations={"node-1"},
                primary_location="node-1",
            )
        )

        result = catalog.health_check()

        assert result.healthy is False
        assert "under-replicated" in result.message.lower()


# =============================================================================
# Test DataCatalog - Clear
# =============================================================================


class TestDataCatalogClear:
    """Tests for clear method."""

    def test_clear_removes_entries(self, catalog, sample_entry):
        """clear should remove all entries."""
        catalog.register(sample_entry)
        assert len(catalog.get_all()) == 1

        catalog.clear()

        assert len(catalog.get_all()) == 0

    def test_clear_resets_indices(self, catalog, sample_entry):
        """clear should reset all indices."""
        catalog.register(sample_entry)

        catalog.clear()

        assert len(catalog._by_node) == 0
        assert len(catalog._by_type) == 0
        assert len(catalog._by_config) == 0


# =============================================================================
# Test DataCatalog - Persistence
# =============================================================================


class TestDataCatalogPersistence:
    """Tests for database persistence."""

    def test_entries_persisted(self, config, sample_entry):
        """Entries should survive restart."""
        catalog1 = DataCatalog(config)
        catalog1.register(sample_entry)

        # Create new catalog with same database
        catalog2 = DataCatalog(config)

        entry = catalog2.get(sample_entry.path)
        assert entry is not None
        assert entry.config_key == "hex8_2p"

    def test_manifest_stored(self, config):
        """Manifests should be stored in database."""
        catalog = DataCatalog(config)
        manifest = {"file.db": {"size": 100, "mtime": time.time()}}
        catalog.register_from_manifest("node-1", manifest)

        # Check database directly
        import sqlite3

        conn = sqlite3.connect(str(config.db_path))
        cursor = conn.execute("SELECT * FROM node_manifests WHERE node_id = 'node-1'")
        row = cursor.fetchone()
        conn.close()

        assert row is not None


# =============================================================================
# Test Singleton Functions
# =============================================================================


class TestSingletonFunctions:
    """Tests for module-level singleton functions."""

    def test_get_data_catalog_returns_same_instance(self, temp_db_path):
        """get_data_catalog should return same instance."""
        config = DataCatalogConfig(db_path=temp_db_path)

        catalog1 = get_data_catalog(config)
        catalog2 = get_data_catalog()

        assert catalog1 is catalog2

    def test_reset_data_catalog(self, temp_db_path):
        """reset_data_catalog should clear singleton."""
        config = DataCatalogConfig(db_path=temp_db_path)

        catalog1 = get_data_catalog(config)
        reset_data_catalog()

        # Create new temp path to avoid database conflict
        with tempfile.TemporaryDirectory() as tmpdir:
            config2 = DataCatalogConfig(db_path=Path(tmpdir) / "new.db")
            catalog2 = get_data_catalog(config2)

            assert catalog1 is not catalog2


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_concurrent_registration(self, catalog):
        """Multiple threads registering should be safe."""
        import threading

        errors = []

        def register_entries():
            try:
                for i in range(100):
                    catalog.register(
                        DataEntry(
                            path=f"thread_{threading.current_thread().name}_{i}.db",
                            data_type=DataType.GAMES,
                            config_key="test",
                            size_bytes=100,
                            checksum="abc",
                            mtime=time.time(),
                            locations={"node-1"},
                            primary_location="node-1",
                        )
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_entries) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert catalog.get_stats()["total_entries"] == 500

    def test_extract_config_key_patterns(self, catalog):
        """_extract_config_key should handle various patterns."""
        test_cases = [
            ("canonical_hex8_2p.db", "hex8_2p"),
            ("hex8_4p_selfplay.db", "hex8_4p"),
            ("selfplay_square8_2p.db", "square8_2p"),
            ("square19_3p.npz", "square19_3p"),
            ("hexagonal_4p.pth", "hexagonal_4p"),
            ("random_file.db", ""),  # No match
        ]

        for path, expected in test_cases:
            result = catalog._extract_config_key(path)
            assert result == expected, f"Failed for {path}: got {result}"

    def test_large_manifest(self, catalog):
        """Should handle large manifests efficiently."""
        manifest = {
            f"games/game_{i}.db": {"size": 1000, "mtime": time.time()}
            for i in range(1000)
        }

        count = catalog.register_from_manifest("node-1", manifest)

        assert count == 1000
        assert catalog.get_stats()["total_entries"] == 1000

    def test_special_characters_in_path(self, catalog):
        """Should handle special characters in paths."""
        entry = DataEntry(
            path="data/file with spaces (1).db",
            data_type=DataType.GAMES,
            config_key="test",
            size_bytes=100,
            checksum="abc",
            mtime=time.time(),
            locations={"node-1"},
            primary_location="node-1",
        )
        catalog.register(entry)

        retrieved = catalog.get("data/file with spaces (1).db")
        assert retrieved is not None
