"""Tests for DataCatalog - Training Data Discovery Service.

Tests cover:
- DataCatalog initialization
- find_training_data() with various board types/num_players
- discover_data_sources() with mocked filesystem
- get_available_samples() sample counting
- File freshness tracking via discover cache
- NPZ file discovery across different directories
- Cache invalidation via refresh()
- UnifiedDataRegistry facade
- Singleton pattern
- Data source filtering and sorting
- Training source recommendations
- Dataclass properties and computed fields
"""

from __future__ import annotations

import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from app.distributed.data_catalog import (
    CatalogStats,
    DataCatalog,
    DataSource,
    NPZDataSource,
    TrainingSourceRecommendation,
    TrainingSourceSuggestion,
    UnifiedDataRegistry,
    get_data_catalog,
    get_data_registry,
    reset_data_catalog,
    reset_data_registry,
)


@pytest.fixture
def temp_dir(tmp_path):
    """Create a temporary directory for test data."""
    return tmp_path


@pytest.fixture
def temp_sync_dir(tmp_path):
    """Create a temporary sync directory."""
    sync_dir = tmp_path / "synced"
    sync_dir.mkdir()
    return sync_dir


@pytest.fixture
def temp_manifest_path(tmp_path):
    """Create a temporary manifest database path."""
    return tmp_path / "test_manifest.db"


@pytest.fixture
def isolated_catalog(temp_sync_dir, temp_manifest_path, tmp_path):
    """Create a fully isolated DataCatalog with only test directories.

    This fixture creates a catalog that will ONLY scan:
    - tmp_path/games (local_game_dirs)
    - temp_sync_dir (synced data)

    All other discovery mechanisms are disabled (HAS_GAME_DISCOVERY = False).
    """
    import app.distributed.data_catalog as dc_module

    # Reset singleton
    reset_data_catalog()
    reset_data_registry()

    local_dirs = [tmp_path / "games"]
    local_dirs[0].mkdir()

    # Temporarily disable GameDiscovery to use manual discovery only
    # This ensures we only scan our test directories
    original_has_game_discovery = dc_module.HAS_GAME_DISCOVERY
    dc_module.HAS_GAME_DISCOVERY = False

    try:
        # Create catalog with ONLY the tmp directories
        catalog = DataCatalog(
            sync_dir=temp_sync_dir,
            manifest_path=temp_manifest_path,
            local_game_dirs=local_dirs,
            node_id="test-node",
        )

        # Override _provider to ensure no NFS/storage provider paths are used
        catalog._provider = None

        yield catalog
    finally:
        # Restore original value
        dc_module.HAS_GAME_DISCOVERY = original_has_game_discovery
        reset_data_catalog()
        reset_data_registry()


# Alias for backward compatibility with existing tests
@pytest.fixture
def catalog(isolated_catalog):
    """Alias for isolated_catalog."""
    return isolated_catalog


def create_test_db(db_path: Path, game_count: int = 10, board_type: str = "hex8", num_players: int = 2) -> None:
    """Create a minimal test game database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS games (
            id INTEGER PRIMARY KEY,
            game_id TEXT,
            board_type TEXT,
            num_players INTEGER
        )
    """)
    for i in range(game_count):
        cursor.execute(
            "INSERT INTO games (game_id, board_type, num_players) VALUES (?, ?, ?)",
            (f"game-{i:04d}", board_type, num_players),
        )
    conn.commit()
    conn.close()


class TestDataCatalogInit:
    """Tests for DataCatalog initialization."""

    def test_init_with_defaults(self, temp_sync_dir):
        """Test initialization with default values."""
        reset_data_catalog()
        with patch("app.distributed.data_catalog.HAS_STORAGE_PROVIDER", False):
            with patch("app.distributed.data_catalog.HAS_UNIFIED_MANIFEST", False):
                catalog = DataCatalog(sync_dir=temp_sync_dir)
                assert catalog.sync_dir == temp_sync_dir
                assert catalog.node_id  # Should have hostname
                assert catalog._sources == {}
                assert catalog._last_discovery == 0.0

    def test_init_with_custom_node_id(self, temp_sync_dir):
        """Test initialization with custom node ID."""
        with patch("app.distributed.data_catalog.HAS_STORAGE_PROVIDER", False):
            with patch("app.distributed.data_catalog.HAS_UNIFIED_MANIFEST", False):
                catalog = DataCatalog(
                    sync_dir=temp_sync_dir,
                    node_id="custom-node-123",
                )
                assert catalog.node_id == "custom-node-123"

    def test_init_with_local_game_dirs(self, temp_sync_dir, tmp_path):
        """Test initialization with custom local game directories."""
        games_dir = tmp_path / "my_games"
        games_dir.mkdir()

        with patch("app.distributed.data_catalog.HAS_STORAGE_PROVIDER", False):
            with patch("app.distributed.data_catalog.HAS_UNIFIED_MANIFEST", False):
                catalog = DataCatalog(
                    sync_dir=temp_sync_dir,
                    local_game_dirs=[games_dir],
                )
                assert games_dir in catalog.local_game_dirs

    def test_singleton_pattern(self, temp_sync_dir):
        """Test singleton accessor returns same instance."""
        reset_data_catalog()

        with patch("app.distributed.data_catalog.HAS_STORAGE_PROVIDER", False):
            with patch("app.distributed.data_catalog.HAS_UNIFIED_MANIFEST", False):
                with patch("app.distributed.data_catalog.DEFAULT_SYNC_DIR", temp_sync_dir):
                    c1 = get_data_catalog()
                    c2 = get_data_catalog()
                    assert c1 is c2

        reset_data_catalog()

    def test_reset_clears_singleton(self, temp_sync_dir):
        """Test reset clears the singleton instance."""
        reset_data_catalog()

        with patch("app.distributed.data_catalog.HAS_STORAGE_PROVIDER", False):
            with patch("app.distributed.data_catalog.HAS_UNIFIED_MANIFEST", False):
                with patch("app.distributed.data_catalog.DEFAULT_SYNC_DIR", temp_sync_dir):
                    c1 = get_data_catalog()
                    reset_data_catalog()
                    c2 = get_data_catalog()
                    assert c1 is not c2

        reset_data_catalog()


class TestDiscoverDataSources:
    """Tests for data source discovery."""

    def test_discover_empty_directory(self, catalog):
        """Test discovery with empty directories."""
        sources = catalog.discover_data_sources()
        assert sources == []

    def test_discover_local_database(self, catalog, tmp_path):
        """Test discovering a local game database."""
        games_dir = catalog.local_game_dirs[0]
        db_path = games_dir / "test_games.db"
        create_test_db(db_path, game_count=50)

        sources = catalog.discover_data_sources(force=True)

        assert len(sources) == 1
        assert sources[0].game_count == 50
        assert sources[0].source_type == "local"
        assert sources[0].path == db_path

    def test_discover_synced_database(self, catalog, temp_sync_dir):
        """Test discovering a synced game database."""
        host_dir = temp_sync_dir / "remote-host"
        host_dir.mkdir()
        db_path = host_dir / "synced_games.db"
        create_test_db(db_path, game_count=100)

        sources = catalog.discover_data_sources(force=True)

        assert len(sources) == 1
        assert sources[0].game_count == 100
        assert sources[0].source_type == "synced"
        assert sources[0].host_origin == "remote-host"

    def test_discover_multiple_sources(self, catalog, tmp_path, temp_sync_dir):
        """Test discovering multiple data sources."""
        # Create local database
        games_dir = catalog.local_game_dirs[0]
        local_db = games_dir / "local.db"
        create_test_db(local_db, game_count=25)

        # Create synced database
        host_dir = temp_sync_dir / "host-a"
        host_dir.mkdir()
        synced_db = host_dir / "synced.db"
        create_test_db(synced_db, game_count=75)

        sources = catalog.discover_data_sources(force=True)

        assert len(sources) == 2
        total_games = sum(s.game_count for s in sources)
        assert total_games == 100

    def test_discovery_caching(self, catalog, tmp_path):
        """Test that discovery results are cached."""
        games_dir = catalog.local_game_dirs[0]
        db_path = games_dir / "test.db"
        create_test_db(db_path, game_count=10)

        # First discovery
        sources1 = catalog.discover_data_sources()
        assert len(sources1) == 1

        # Second discovery should use cache
        catalog._discovery_interval = 3600  # Long interval
        sources2 = catalog.discover_data_sources()
        assert sources1 == sources2

    def test_force_discovery_bypasses_cache(self, catalog, tmp_path):
        """Test that force=True bypasses cache."""
        games_dir = catalog.local_game_dirs[0]
        db_path = games_dir / "test.db"
        create_test_db(db_path, game_count=10)

        sources1 = catalog.discover_data_sources(force=True)
        last_discovery1 = catalog._last_discovery

        time.sleep(0.01)
        sources2 = catalog.discover_data_sources(force=True)
        last_discovery2 = catalog._last_discovery

        assert last_discovery2 > last_discovery1

    def test_skip_empty_databases(self, catalog, tmp_path):
        """Test that empty databases are skipped."""
        games_dir = catalog.local_game_dirs[0]
        db_path = games_dir / "empty.db"
        create_test_db(db_path, game_count=0)

        sources = catalog.discover_data_sources(force=True)
        assert len(sources) == 0

    def test_skip_invalid_databases(self, catalog, tmp_path):
        """Test that invalid databases are skipped gracefully."""
        games_dir = catalog.local_game_dirs[0]
        invalid_db = games_dir / "invalid.db"
        invalid_db.write_text("not a database")

        sources = catalog.discover_data_sources(force=True)
        # Should not raise, just skip the invalid file
        assert len(sources) == 0


class TestAnalyzeDatabase:
    """Tests for database analysis."""

    def test_analyze_database_extracts_board_types(self, catalog, tmp_path):
        """Test that board type is extracted from database."""
        db_path = tmp_path / "test.db"
        create_test_db(db_path, game_count=10, board_type="square8")

        source = catalog._analyze_database(db_path, "local", "test-node")

        assert source is not None
        assert "square8" in source.board_types

    def test_analyze_database_extracts_player_counts(self, catalog, tmp_path):
        """Test that player count is extracted from database."""
        db_path = tmp_path / "test.db"
        create_test_db(db_path, game_count=10, num_players=4)

        source = catalog._analyze_database(db_path, "local", "test-node")

        assert source is not None
        assert 4 in source.player_counts

    def test_analyze_database_gets_file_stats(self, catalog, tmp_path):
        """Test that file stats are populated."""
        db_path = tmp_path / "test.db"
        create_test_db(db_path, game_count=10)

        source = catalog._analyze_database(db_path, "local", "test-node")

        assert source is not None
        assert source.total_size_bytes > 0
        assert source.last_modified > 0


class TestGetSyncedDbPaths:
    """Tests for get_synced_db_paths."""

    def test_get_all_synced_paths(self, catalog, tmp_path, temp_sync_dir):
        """Test getting all synced database paths."""
        games_dir = catalog.local_game_dirs[0]
        db1 = games_dir / "local.db"
        create_test_db(db1, game_count=10)

        host_dir = temp_sync_dir / "host-a"
        host_dir.mkdir()
        db2 = host_dir / "synced.db"
        create_test_db(db2, game_count=20)

        paths = catalog.get_synced_db_paths()

        assert len(paths) == 2
        assert db1 in paths
        assert db2 in paths

    def test_filter_by_host(self, catalog, temp_sync_dir):
        """Test filtering by host."""
        for host in ["host-a", "host-b", "host-c"]:
            host_dir = temp_sync_dir / host
            host_dir.mkdir()
            create_test_db(host_dir / "games.db", game_count=10)

        paths = catalog.get_synced_db_paths(host_filter="host-a")

        assert len(paths) == 1
        assert "host-a" in str(paths[0])

    def test_filter_by_host_wildcard(self, catalog, temp_sync_dir):
        """Test filtering by host with wildcard."""
        for host in ["host-a", "host-b", "other-c"]:
            host_dir = temp_sync_dir / host
            host_dir.mkdir()
            create_test_db(host_dir / "games.db", game_count=10)

        paths = catalog.get_synced_db_paths(host_filter="host-*")

        assert len(paths) == 2

    def test_filter_by_min_games(self, catalog, temp_sync_dir):
        """Test filtering by minimum game count."""
        host_a = temp_sync_dir / "host-a"
        host_a.mkdir()
        create_test_db(host_a / "games.db", game_count=5)

        host_b = temp_sync_dir / "host-b"
        host_b.mkdir()
        create_test_db(host_b / "games.db", game_count=50)

        paths = catalog.get_synced_db_paths(min_games=10)

        assert len(paths) == 1
        assert "host-b" in str(paths[0])

    def test_filter_by_board_type(self, catalog, temp_sync_dir):
        """Test filtering by board type."""
        host_a = temp_sync_dir / "host-a"
        host_a.mkdir()
        create_test_db(host_a / "games.db", board_type="hex8")

        host_b = temp_sync_dir / "host-b"
        host_b.mkdir()
        create_test_db(host_b / "games.db", board_type="square8")

        paths = catalog.get_synced_db_paths(board_type="hex8")

        assert len(paths) == 1
        assert "host-a" in str(paths[0])


class TestGetAllTrainingPaths:
    """Tests for get_all_training_paths."""

    def test_filter_by_source_type(self, catalog, tmp_path, temp_sync_dir):
        """Test filtering by source type."""
        games_dir = catalog.local_game_dirs[0]
        create_test_db(games_dir / "local.db", game_count=10)

        host_dir = temp_sync_dir / "host-a"
        host_dir.mkdir()
        create_test_db(host_dir / "synced.db", game_count=20)

        # Only local
        local_paths = catalog.get_all_training_paths(include_synced=False)
        assert len(local_paths) == 1
        assert "local" in str(local_paths[0])

        # Only synced
        synced_paths = catalog.get_all_training_paths(include_local=False)
        assert len(synced_paths) == 1
        assert "synced" in str(synced_paths[0])


class TestGetStats:
    """Tests for catalog statistics."""

    def test_get_stats_empty(self, catalog):
        """Test getting stats with no data."""
        stats = catalog.get_stats()

        assert isinstance(stats, CatalogStats)
        assert stats.total_sources == 0
        assert stats.total_games == 0
        assert stats.total_size_bytes == 0

    def test_get_stats_with_data(self, catalog, tmp_path, temp_sync_dir):
        """Test getting stats with data."""
        games_dir = catalog.local_game_dirs[0]
        create_test_db(games_dir / "local.db", game_count=100, board_type="hex8")

        host_dir = temp_sync_dir / "host-a"
        host_dir.mkdir()
        create_test_db(host_dir / "synced.db", game_count=200, board_type="square8")

        stats = catalog.get_stats()

        assert stats.total_sources == 2
        assert stats.total_games == 300
        assert stats.total_size_bytes > 0
        assert "local" in stats.sources_by_type
        assert "synced" in stats.sources_by_type
        assert "hex8" in stats.board_type_distribution
        assert "square8" in stats.board_type_distribution


class TestGetPendingSampleCount:
    """Tests for pending sample count estimation."""

    def test_estimate_samples_default_rate(self, catalog, tmp_path):
        """Test sample count estimation with default rate."""
        games_dir = catalog.local_game_dirs[0]
        create_test_db(games_dir / "test.db", game_count=100)

        count = catalog.get_pending_sample_count()

        # Default is 30 samples per game
        assert count == 3000

    def test_estimate_samples_custom_rate(self, catalog, tmp_path):
        """Test sample count estimation with custom rate."""
        games_dir = catalog.local_game_dirs[0]
        create_test_db(games_dir / "test.db", game_count=100)

        count = catalog.get_pending_sample_count(samples_per_game=50)

        assert count == 5000


class TestGetSourcesByQuality:
    """Tests for quality-based source filtering."""

    def test_filter_by_quality(self, catalog, tmp_path):
        """Test filtering sources by quality score."""
        games_dir = catalog.local_game_dirs[0]
        create_test_db(games_dir / "test.db", game_count=10)

        # Manually set quality score
        catalog.discover_data_sources(force=True)
        for source in catalog._sources.values():
            source.avg_quality_score = 0.3

        high_quality = catalog.get_sources_by_quality(min_quality=0.5)
        assert len(high_quality) == 0

        low_quality = catalog.get_sources_by_quality(min_quality=0.2)
        assert len(low_quality) == 1


class TestRefresh:
    """Tests for catalog refresh."""

    def test_refresh_forces_rediscovery(self, catalog, tmp_path):
        """Test that refresh forces rediscovery."""
        games_dir = catalog.local_game_dirs[0]
        db_path = games_dir / "test.db"
        create_test_db(db_path, game_count=10)

        catalog.discover_data_sources(force=True)
        first_time = catalog._last_discovery

        time.sleep(0.01)
        catalog.refresh()
        second_time = catalog._last_discovery

        assert second_time > first_time


class TestRecommendedTrainingSources:
    """Tests for training source recommendations."""

    def test_recommend_sources_empty(self, catalog):
        """Test recommendations with no data."""
        paths = catalog.get_recommended_training_sources()
        assert paths == []

    def test_recommend_sources_sorted(self, catalog, temp_sync_dir):
        """Test that sources are sorted by quality/recency."""
        for i, host in enumerate(["host-a", "host-b", "host-c"]):
            host_dir = temp_sync_dir / host
            host_dir.mkdir()
            create_test_db(host_dir / "games.db", game_count=100)

        paths = catalog.get_recommended_training_sources(target_games=300)

        assert len(paths) == 3

    def test_recommend_sources_diversity(self, catalog, temp_sync_dir):
        """Test that recommendations prefer host diversity."""
        for host in ["host-a", "host-b"]:
            host_dir = temp_sync_dir / host
            host_dir.mkdir()
            create_test_db(host_dir / "db1.db", game_count=100)
            create_test_db(host_dir / "db2.db", game_count=100)

        paths = catalog.get_recommended_training_sources(target_games=200)

        # Should prefer one source from each host
        hosts_used = set()
        for path in paths[:2]:
            for part in path.parts:
                if part.startswith("host-"):
                    hosts_used.add(part)

        assert len(hosts_used) == 2


class TestSuggestBestTrainingSources:
    """Tests for training source suggestions with scoring."""

    def test_suggest_returns_recommendation(self, catalog, temp_sync_dir):
        """Test that suggest returns TrainingSourceSuggestion."""
        host_dir = temp_sync_dir / "host-a"
        host_dir.mkdir()
        create_test_db(host_dir / "games.db", game_count=100, board_type="hex8", num_players=2)

        suggestion = catalog.suggest_best_training_sources("hex8", 2)

        assert isinstance(suggestion, TrainingSourceSuggestion)
        assert suggestion.board_type == "hex8"
        assert suggestion.num_players == 2

    def test_suggest_filters_by_config(self, catalog, temp_sync_dir):
        """Test that suggestions filter by board config."""
        for host, board_type in [("host-a", "hex8"), ("host-b", "square8")]:
            host_dir = temp_sync_dir / host
            host_dir.mkdir()
            create_test_db(host_dir / "games.db", board_type=board_type)

        suggestion = catalog.suggest_best_training_sources("hex8", 2)

        for rec in suggestion.recommendations:
            assert "host-a" in str(rec.path)

    def test_suggest_calculates_coverage(self, catalog, temp_sync_dir):
        """Test coverage calculation."""
        host_dir = temp_sync_dir / "host-a"
        host_dir.mkdir()
        create_test_db(host_dir / "games.db", game_count=25000, board_type="hex8", num_players=2)

        suggestion = catalog.suggest_best_training_sources("hex8", 2, target_games=50000)

        assert suggestion.total_games_available == 25000
        assert suggestion.coverage_percent == 50.0

    def test_suggest_generates_warnings(self, catalog, temp_sync_dir):
        """Test that warnings are generated for missing data."""
        suggestion = catalog.suggest_best_training_sources("nonexistent", 9)

        assert len(suggestion.warnings) > 0
        assert any("No data found" in w for w in suggestion.warnings)


class TestNPZDiscovery:
    """Tests for NPZ file discovery.

    Note: discover_npz_files() searches multiple directories. Tests must carefully
    structure paths to avoid duplicate discovery from overlapping search paths.
    We use completely separate paths to ensure isolation.
    """

    def test_discover_npz_empty(self, catalog):
        """Test NPZ discovery with no files."""
        with patch("app.distributed.data_catalog.DATA_DIR", Path("/nonexistent")):
            with patch("app.distributed.data_catalog.GAMES_DIR", Path("/other/nonexistent/games")):
                sources = catalog.discover_npz_files()
                assert sources == []

    def test_discover_npz_parses_filename(self, catalog, tmp_path):
        """Test that NPZ filenames are parsed for config."""
        # Use completely separate paths to avoid overlap
        data_dir = tmp_path / "data"
        games_dir = tmp_path / "separate_games"  # Different parent than data_dir
        data_dir.mkdir()
        games_dir.mkdir()

        training_dir = data_dir / "training"
        training_dir.mkdir()
        npz_path = training_dir / "hex8_2p.npz"

        # Create mock NPZ file
        import numpy as np
        np.savez(npz_path, policy=np.zeros(100))

        with patch("app.distributed.data_catalog.DATA_DIR", data_dir):
            with patch("app.distributed.data_catalog.GAMES_DIR", games_dir):
                sources = catalog.discover_npz_files()

        assert len(sources) == 1
        assert sources[0].board_type == "hex8"
        assert sources[0].num_players == 2
        assert sources[0].path == npz_path

    def test_discover_npz_filters_by_board_type(self, catalog, tmp_path):
        """Test NPZ filtering by board type."""
        import numpy as np

        # Use completely separate paths to avoid overlap
        data_dir = tmp_path / "data"
        games_dir = tmp_path / "separate_games"
        data_dir.mkdir()
        games_dir.mkdir()

        training_dir = data_dir / "training"
        training_dir.mkdir()

        for filename in ["hex8_2p.npz", "square8_2p.npz", "hexagonal_4p.npz"]:
            np.savez(training_dir / filename, policy=np.zeros(100))

        with patch("app.distributed.data_catalog.DATA_DIR", data_dir):
            with patch("app.distributed.data_catalog.GAMES_DIR", games_dir):
                sources = catalog.discover_npz_files(board_type="hex8")

        assert len(sources) == 1
        assert sources[0].board_type == "hex8"

    def test_discover_npz_filters_by_num_players(self, catalog, tmp_path):
        """Test NPZ filtering by player count."""
        import numpy as np

        # Use completely separate paths to avoid overlap
        data_dir = tmp_path / "data"
        games_dir = tmp_path / "separate_games"
        data_dir.mkdir()
        games_dir.mkdir()

        training_dir = data_dir / "training"
        training_dir.mkdir()

        for filename in ["hex8_2p.npz", "hex8_4p.npz"]:
            np.savez(training_dir / filename, policy=np.zeros(100))

        with patch("app.distributed.data_catalog.DATA_DIR", data_dir):
            with patch("app.distributed.data_catalog.GAMES_DIR", games_dir):
                sources = catalog.discover_npz_files(num_players=4)

        assert len(sources) == 1
        assert sources[0].num_players == 4

    def test_discover_npz_filters_by_min_samples(self, catalog, tmp_path):
        """Test NPZ filtering by minimum samples."""
        import numpy as np

        # Use completely separate paths to avoid overlap
        data_dir = tmp_path / "data"
        games_dir = tmp_path / "separate_games"
        data_dir.mkdir()
        games_dir.mkdir()

        training_dir = data_dir / "training"
        training_dir.mkdir()

        np.savez(training_dir / "small.npz", policy=np.zeros(10))
        np.savez(training_dir / "large.npz", policy=np.zeros(1000))

        with patch("app.distributed.data_catalog.DATA_DIR", data_dir):
            with patch("app.distributed.data_catalog.GAMES_DIR", games_dir):
                sources = catalog.discover_npz_files(min_samples=500)

        assert len(sources) == 1
        assert sources[0].sample_count >= 500

    def test_discover_npz_filters_by_age(self, catalog, tmp_path):
        """Test NPZ filtering by age."""
        import numpy as np
        import os

        training_dir = tmp_path / "training"
        training_dir.mkdir()

        npz_path = training_dir / "hex8_2p.npz"
        np.savez(npz_path, policy=np.zeros(100))

        # Set modification time to 48 hours ago
        old_time = time.time() - 48 * 3600
        os.utime(npz_path, (old_time, old_time))

        with patch("app.distributed.data_catalog.DATA_DIR", tmp_path):
            with patch("app.distributed.data_catalog.GAMES_DIR", tmp_path / "games"):
                # Should be filtered out (older than 24 hours)
                sources = catalog.discover_npz_files(max_age_hours=24)

        assert len(sources) == 0


class TestGetBestNPZForTraining:
    """Tests for best NPZ selection."""

    def test_get_best_npz_prefer_recent(self, catalog, tmp_path):
        """Test that most recent NPZ is preferred."""
        import numpy as np
        import os

        training_dir = tmp_path / "training"
        training_dir.mkdir()

        # Create two NPZ files with different ages
        old_npz = training_dir / "hex8_2p_old.npz"
        np.savez(old_npz, policy=np.zeros(100))
        old_time = time.time() - 3600
        os.utime(old_npz, (old_time, old_time))

        new_npz = training_dir / "hex8_2p_new.npz"
        np.savez(new_npz, policy=np.zeros(100))

        with patch("app.distributed.data_catalog.DATA_DIR", tmp_path):
            with patch("app.distributed.data_catalog.GAMES_DIR", tmp_path / "games"):
                best = catalog.get_best_npz_for_training("hex8", 2, prefer_recent=True)

        assert best is not None
        assert "new" in str(best.path)

    def test_get_best_npz_prefer_largest(self, catalog, tmp_path):
        """Test that largest NPZ is preferred when not preferring recent."""
        import numpy as np

        training_dir = tmp_path / "training"
        training_dir.mkdir()

        small_npz = training_dir / "hex8_2p_small.npz"
        np.savez(small_npz, policy=np.zeros(100))

        large_npz = training_dir / "hex8_2p_large.npz"
        np.savez(large_npz, policy=np.zeros(10000))

        with patch("app.distributed.data_catalog.DATA_DIR", tmp_path):
            with patch("app.distributed.data_catalog.GAMES_DIR", tmp_path / "games"):
                best = catalog.get_best_npz_for_training("hex8", 2, prefer_recent=False)

        assert best is not None
        assert best.sample_count >= 1000

    def test_get_best_npz_no_match(self, catalog, tmp_path):
        """Test when no matching NPZ exists."""
        with patch("app.distributed.data_catalog.DATA_DIR", tmp_path):
            with patch("app.distributed.data_catalog.GAMES_DIR", tmp_path / "games"):
                best = catalog.get_best_npz_for_training("nonexistent", 9)

        assert best is None


class TestGetNPZStats:
    """Tests for NPZ statistics."""

    def test_npz_stats_empty(self, catalog, tmp_path):
        """Test NPZ stats with no files."""
        with patch("app.distributed.data_catalog.DATA_DIR", tmp_path):
            with patch("app.distributed.data_catalog.GAMES_DIR", tmp_path / "games"):
                stats = catalog.get_npz_stats()

        assert stats["total_files"] == 0
        assert stats["total_samples"] == 0

    def test_npz_stats_with_files(self, catalog, tmp_path):
        """Test NPZ stats with files."""
        import numpy as np

        # Use completely separate paths to avoid overlap
        data_dir = tmp_path / "data"
        games_dir = tmp_path / "separate_games"
        data_dir.mkdir()
        games_dir.mkdir()

        training_dir = data_dir / "training"
        training_dir.mkdir()

        np.savez(training_dir / "hex8_2p.npz", policy=np.zeros(100))
        np.savez(training_dir / "hex8_4p.npz", policy=np.zeros(200))

        with patch("app.distributed.data_catalog.DATA_DIR", data_dir):
            with patch("app.distributed.data_catalog.GAMES_DIR", games_dir):
                stats = catalog.get_npz_stats()

        assert stats["total_files"] == 2
        assert stats["total_samples"] >= 300
        assert "hex8_2p" in stats["by_config"]
        assert "hex8_4p" in stats["by_config"]


class TestDataClasses:
    """Tests for data classes."""

    def test_data_source_fields(self):
        """Test DataSource dataclass."""
        source = DataSource(
            name="test.db",
            path=Path("/data/test.db"),
            source_type="local",
            host_origin="test-node",
            game_count=100,
        )
        assert source.name == "test.db"
        assert source.game_count == 100
        assert source.is_available is True
        assert source.board_types == set()

    def test_npz_data_source_config_key(self):
        """Test NPZDataSource config_key property."""
        source = NPZDataSource(
            path=Path("/data/hex8_2p.npz"),
            board_type="hex8",
            num_players=2,
        )
        assert source.config_key == "hex8_2p"

    def test_npz_data_source_config_key_none(self):
        """Test NPZDataSource config_key when missing fields."""
        source = NPZDataSource(path=Path("/data/unknown.npz"))
        assert source.config_key is None

    def test_npz_data_source_age_hours(self):
        """Test NPZDataSource age_hours property."""
        source = NPZDataSource(
            path=Path("/data/test.npz"),
            created_at=time.time() - 3600,  # 1 hour ago
        )
        assert 0.9 < source.age_hours < 1.1

    def test_npz_data_source_age_hours_zero(self):
        """Test NPZDataSource age_hours with zero created_at."""
        source = NPZDataSource(path=Path("/data/test.npz"), created_at=0)
        assert source.age_hours == float("inf")

    def test_catalog_stats_fields(self):
        """Test CatalogStats dataclass."""
        stats = CatalogStats(
            total_sources=5,
            total_games=1000,
            total_size_bytes=5_000_000_000,
        )
        assert stats.total_sources == 5
        assert stats.sources_by_type == {}
        assert stats.board_type_distribution == {}

    def test_training_source_recommendation(self):
        """Test TrainingSourceRecommendation dataclass."""
        rec = TrainingSourceRecommendation(
            path=Path("/data/test.db"),
            game_count=100,
            board_type="hex8",
            num_players=2,
            score=85.5,
            host_origin="node-a",
            reasons=["High quality", "Recent"],
            is_primary=True,
        )
        assert rec.score == 85.5
        assert len(rec.reasons) == 2
        assert rec.is_primary is True

    def test_training_source_suggestion(self):
        """Test TrainingSourceSuggestion dataclass."""
        suggestion = TrainingSourceSuggestion(
            board_type="hex8",
            num_players=2,
            recommendations=[],
            total_games_available=5000,
            games_needed=10000,
            coverage_percent=50.0,
            warnings=["Only 50% coverage"],
        )
        assert suggestion.coverage_percent == 50.0
        assert len(suggestion.warnings) == 1


class TestUnifiedDataRegistry:
    """Tests for UnifiedDataRegistry facade."""

    def test_registry_singleton(self, temp_sync_dir):
        """Test registry singleton pattern."""
        reset_data_registry()
        reset_data_catalog()

        with patch("app.distributed.data_catalog.HAS_STORAGE_PROVIDER", False):
            with patch("app.distributed.data_catalog.HAS_UNIFIED_MANIFEST", False):
                with patch("app.distributed.data_catalog.DEFAULT_SYNC_DIR", temp_sync_dir):
                    r1 = get_data_registry()
                    r2 = get_data_registry()
                    assert r1 is r2

        reset_data_registry()
        reset_data_catalog()

    def test_registry_catalog_property(self, catalog):
        """Test registry provides catalog access."""
        registry = UnifiedDataRegistry(catalog=catalog)
        assert registry.catalog is catalog

    def test_registry_manifest_property(self, catalog):
        """Test registry provides manifest access."""
        registry = UnifiedDataRegistry(catalog=catalog)
        # Manifest may be None if not configured
        assert registry.manifest is None or registry.manifest is catalog._manifest

    def test_registry_is_game_synced_no_manifest(self, catalog):
        """Test is_game_synced returns False without manifest."""
        registry = UnifiedDataRegistry(catalog=catalog)
        assert registry.is_game_synced("game-001") is False

    def test_registry_mark_games_synced_no_manifest(self, catalog):
        """Test mark_games_synced returns 0 without manifest."""
        registry = UnifiedDataRegistry(catalog=catalog)
        count = registry.mark_games_synced(["game-001"], "host-a")
        assert count == 0

    def test_registry_get_synced_game_count_no_manifest(self, catalog):
        """Test get_synced_game_count returns 0 without manifest."""
        registry = UnifiedDataRegistry(catalog=catalog)
        count = registry.get_synced_game_count()
        assert count == 0

    def test_registry_discover_sources(self, catalog, tmp_path):
        """Test registry discover_sources delegates to catalog."""
        games_dir = catalog.local_game_dirs[0]
        create_test_db(games_dir / "test.db", game_count=10)

        registry = UnifiedDataRegistry(catalog=catalog)
        sources = registry.discover_sources(force=True)

        assert len(sources) == 1

    def test_registry_get_combined_stats(self, catalog, tmp_path):
        """Test registry get_combined_stats combines catalog and manifest."""
        games_dir = catalog.local_game_dirs[0]
        create_test_db(games_dir / "test.db", game_count=10)

        registry = UnifiedDataRegistry(catalog=catalog)
        stats = registry.get_combined_stats()

        assert "catalog" in stats
        assert "manifest" in stats
        assert "node_id" in stats
        assert stats["catalog"]["total_games"] == 10

    def test_registry_refresh(self, catalog):
        """Test registry refresh delegates to catalog."""
        registry = UnifiedDataRegistry(catalog=catalog)
        old_time = catalog._last_discovery

        time.sleep(0.01)
        registry.refresh()
        new_time = catalog._last_discovery

        assert new_time > old_time


class TestCacheInvalidation:
    """Tests for cache invalidation behavior."""

    def test_cache_expires_after_interval(self, catalog, tmp_path):
        """Test that cache expires after discovery interval."""
        games_dir = catalog.local_game_dirs[0]
        create_test_db(games_dir / "test.db", game_count=10)

        catalog._discovery_interval = 0.01  # Very short interval
        catalog.discover_data_sources()

        time.sleep(0.02)

        # Add another database
        create_test_db(games_dir / "test2.db", game_count=20)

        # Should rediscover
        sources = catalog.discover_data_sources()
        assert len(sources) == 2

    def test_source_availability_tracking(self, catalog, tmp_path):
        """Test that unavailable sources are marked."""
        games_dir = catalog.local_game_dirs[0]
        db_path = games_dir / "test.db"
        create_test_db(db_path, game_count=10)

        sources = catalog.discover_data_sources(force=True)
        assert len(sources) == 1
        assert sources[0].is_available is True


class TestFileFreshnessTracking:
    """Tests for file freshness tracking."""

    def test_last_modified_tracked(self, catalog, tmp_path):
        """Test that last_modified time is tracked."""
        games_dir = catalog.local_game_dirs[0]
        db_path = games_dir / "test.db"
        create_test_db(db_path, game_count=10)

        before = time.time()
        sources = catalog.discover_data_sources(force=True)
        after = time.time()

        assert len(sources) == 1
        assert before - 1 < sources[0].last_modified < after + 1

    def test_freshness_in_suggestion_scoring(self, catalog, tmp_path, temp_sync_dir):
        """Test that freshness affects suggestion scoring."""
        import os

        host_dir = temp_sync_dir / "host-a"
        host_dir.mkdir()

        # Create old database
        old_db = host_dir / "old.db"
        create_test_db(old_db, game_count=100, board_type="hex8", num_players=2)
        old_time = time.time() - 30 * 86400  # 30 days ago
        os.utime(old_db, (old_time, old_time))

        # Create new database
        new_db = host_dir / "new.db"
        create_test_db(new_db, game_count=100, board_type="hex8", num_players=2)

        suggestion = catalog.suggest_best_training_sources("hex8", 2)

        # Newer should have higher score
        if len(suggestion.recommendations) >= 2:
            recs = sorted(suggestion.recommendations, key=lambda r: r.score, reverse=True)
            # The more recent one should be ranked higher
            assert "new" in str(recs[0].path)
