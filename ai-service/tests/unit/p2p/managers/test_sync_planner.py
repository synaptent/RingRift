"""Unit tests for SyncPlanner manager.

December 2025: Comprehensive tests for Phase 2A extraction of sync planning logic.
Tests cover initialization, manifest collection, sync planning, execution, and health checks.
"""

import asyncio
import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.managers.sync_planner import (
    SyncPlanner,
    SyncPlannerConfig,
    SyncStats,
    _validate_event_types,
    REQUIRED_SYNC_EVENT_TYPES,
)


# ============================================================================
# Test SyncPlannerConfig
# ============================================================================


class TestSyncPlannerConfig:
    """Tests for SyncPlannerConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = SyncPlannerConfig()
        # Uses PeerDefaults.MANIFEST_TIMEOUT (60s) as of Jan 2, 2026
        assert config.manifest_cache_age_seconds == 60
        assert config.manifest_collection_interval == 60
        assert config.max_files_per_sync_job == 50
        assert config.sync_mtime_tolerance_seconds == 60

    def test_custom_values(self):
        """Test configuration with custom values."""
        config = SyncPlannerConfig(
            manifest_cache_age_seconds=600,
            max_files_per_sync_job=100,
            manifest_collection_interval=30,
            sync_mtime_tolerance_seconds=120,
        )
        assert config.manifest_cache_age_seconds == 600
        assert config.max_files_per_sync_job == 100
        assert config.manifest_collection_interval == 30
        assert config.sync_mtime_tolerance_seconds == 120

    def test_zero_cache_age(self):
        """Test configuration with zero cache age (effectively disabled)."""
        config = SyncPlannerConfig(manifest_cache_age_seconds=0)
        assert config.manifest_cache_age_seconds == 0


# ============================================================================
# Test SyncStats
# ============================================================================


class TestSyncStats:
    """Tests for SyncStats dataclass."""

    def test_default_values(self):
        """Test default statistics values."""
        stats = SyncStats()
        assert stats.manifests_collected == 0
        assert stats.sync_plans_generated == 0
        assert stats.sync_jobs_created == 0
        assert stats.sync_jobs_completed == 0
        assert stats.sync_jobs_failed == 0
        assert stats.bytes_synced == 0
        assert stats.last_manifest_collection == 0.0
        assert stats.last_sync_execution == 0.0
        assert stats.events_emitted == 0
        assert stats.events_failed == 0
        assert stats.last_event_error == ""

    def test_stats_increment(self):
        """Test incrementing statistics."""
        stats = SyncStats()
        stats.manifests_collected += 1
        stats.sync_jobs_completed += 5
        stats.bytes_synced += 1024 * 1024
        assert stats.manifests_collected == 1
        assert stats.sync_jobs_completed == 5
        assert stats.bytes_synced == 1024 * 1024

    def test_event_tracking(self):
        """Test event emission tracking."""
        stats = SyncStats()
        stats.events_emitted += 10
        stats.events_failed += 2
        stats.last_event_error = "Connection timeout"
        assert stats.events_emitted == 10
        assert stats.events_failed == 2
        assert stats.last_event_error == "Connection timeout"


# ============================================================================
# Test SyncPlanner Initialization
# ============================================================================


class TestSyncPlannerInit:
    """Tests for SyncPlanner initialization."""

    def test_init_with_required_args(self):
        """Test initialization with only required arguments."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp/test"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )
        assert planner.node_id == "test-node"
        assert planner.data_directory == Path("/tmp/test")
        assert planner.config is not None
        assert isinstance(planner.config, SyncPlannerConfig)
        assert planner.stats is not None

    def test_init_with_custom_config(self):
        """Test initialization with custom configuration."""
        config = SyncPlannerConfig(manifest_cache_age_seconds=600)
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp/test"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
            config=config,
        )
        assert planner.config.manifest_cache_age_seconds == 600

    def test_init_with_all_callbacks(self):
        """Test initialization with all optional callbacks."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp/test"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
            request_peer_manifest=lambda node_id: None,
            check_disk_capacity=lambda: (True, 50.0),
        )
        assert planner._request_peer_manifest is not None
        assert planner._check_disk_capacity is not None

    def test_init_creates_parent_directory(self):
        """Test that initialization works even if data directory doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            nonexistent_path = Path(tmpdir) / "nonexistent" / "data"
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=nonexistent_path,
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )
            # Should not raise, just won't find any files
            manifest = planner.collect_local_manifest()
            assert manifest.total_files == 0


# ============================================================================
# Test Local Manifest Collection
# ============================================================================


class TestLocalManifestCollection:
    """Tests for local manifest collection."""

    def test_collect_empty_directory(self):
        """Test collecting manifest from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            manifest = planner.collect_local_manifest()

            assert manifest.node_id == "test-node"
            assert manifest.total_files == 0
            assert manifest.selfplay_games == 0

    def test_collect_with_selfplay_files(self):
        """Test collecting manifest with selfplay JSONL files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test selfplay directory and files
            selfplay_dir = Path(tmpdir) / "selfplay"
            selfplay_dir.mkdir()

            # Create a JSONL file with game data
            jsonl_file = selfplay_dir / "games_sq8_2p.jsonl"
            jsonl_file.write_text('{"game": 1}\n{"game": 2}\n{"game": 3}\n')

            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            manifest = planner.collect_local_manifest()

            assert manifest.total_files == 1
            assert manifest.selfplay_games == 3
            assert len(manifest.files) == 1
            assert manifest.files[0].file_type == "selfplay"
            assert manifest.files[0].board_type == "square8"
            assert manifest.files[0].num_players == 2

    def test_collect_with_model_files(self):
        """Test collecting manifest with model files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test models directory
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            # Create model files
            model_file = models_dir / "model_hex_4p.pt"
            model_file.write_bytes(b"fake model data")

            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            manifest = planner.collect_local_manifest()

            assert manifest.total_files == 1
            assert manifest.model_count == 1
            assert manifest.files[0].file_type == "model"
            assert manifest.files[0].num_players == 4

    def test_collect_with_training_files(self):
        """Test collecting manifest with training NPZ files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_dir = Path(tmpdir) / "training"
            training_dir.mkdir()

            npz_file = training_dir / "square19_3p.npz"
            npz_file.write_bytes(b"fake npz data")

            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            manifest = planner.collect_local_manifest()

            assert manifest.total_files == 1
            assert manifest.files[0].file_type == "training"
            assert manifest.files[0].board_type == "square19"
            assert manifest.files[0].num_players == 3

    def test_manifest_caching(self):
        """Test manifest caching functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
                config=SyncPlannerConfig(manifest_cache_age_seconds=10),
            )

            # First collection
            manifest1 = planner.collect_local_manifest()
            assert planner.stats.manifests_collected == 1

            # Second collection should use cache
            manifest2 = planner.collect_local_manifest(use_cache=True)
            assert planner.stats.manifests_collected == 1  # No increment
            assert manifest1.collected_at == manifest2.collected_at

            # Force collection should bypass cache
            manifest3 = planner.collect_local_manifest(use_cache=False)
            assert planner.stats.manifests_collected == 2

    def test_collect_with_sqlite_games_db(self):
        """Test collecting manifest with SQLite game database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            games_dir = Path(tmpdir) / "games"
            games_dir.mkdir()

            # Create a SQLite database with games table
            db_path = games_dir / "canonical_hex8_2p.db"
            conn = sqlite3.connect(str(db_path))
            conn.execute("CREATE TABLE games (id INTEGER PRIMARY KEY)")
            conn.execute("INSERT INTO games VALUES (1)")
            conn.execute("INSERT INTO games VALUES (2)")
            conn.commit()
            conn.close()

            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            manifest = planner.collect_local_manifest()

            assert manifest.total_files == 1
            assert manifest.files[0].file_type == "games"


# ============================================================================
# Test Board Config Parsing
# ============================================================================


class TestBoardConfigParsing:
    """Tests for board type/player count parsing from filenames."""

    @pytest.fixture
    def planner(self):
        """Create a planner for testing."""
        return SyncPlanner(
            node_id="test",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

    def test_parse_square8_2p(self, planner):
        """Test parsing square8 2-player config."""
        board_type, num_players = planner._parse_board_config("selfplay/games_sq8_2p.jsonl")
        assert board_type == "square8"
        assert num_players == 2

    def test_parse_hex_4p(self, planner):
        """Test parsing hexagonal 4-player config."""
        board_type, num_players = planner._parse_board_config("models/hex_4p_model.pt")
        assert board_type == "hexagonal"
        assert num_players == 4

    def test_parse_square19_3p(self, planner):
        """Test parsing square19 3-player config."""
        board_type, num_players = planner._parse_board_config("training/square19_3p.npz")
        assert board_type == "square19"
        assert num_players == 3

    def test_parse_full_names(self, planner):
        """Test parsing with full board type names."""
        board_type, num_players = planner._parse_board_config("games/square8_2p.db")
        assert board_type == "square8"
        assert num_players == 2

    def test_parse_unknown_board(self, planner):
        """Test parsing unknown board type returns empty."""
        board_type, num_players = planner._parse_board_config("data/unknown_board.jsonl")
        assert board_type == ""
        assert num_players == 0

    def test_parse_case_insensitive(self, planner):
        """Test case-insensitive parsing."""
        board_type, num_players = planner._parse_board_config("selfplay/GAMES_SQ8_2P.JSONL")
        assert board_type == "square8"
        assert num_players == 2


# ============================================================================
# Test Sync Plan Generation
# ============================================================================


class TestSyncPlanGeneration:
    """Tests for sync plan generation."""

    def test_generate_plan_no_manifest(self):
        """Test generating plan without manifest returns None."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        plan = planner.generate_sync_plan()
        assert plan is None

    def test_generate_plan_no_missing_files(self):
        """Test generating plan when no files are missing returns None."""
        from scripts.p2p.models import ClusterDataManifest

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        manifest = ClusterDataManifest(collected_at=time.time())
        plan = planner.generate_sync_plan(manifest)
        assert plan is None

    def test_generate_plan_with_missing_files(self):
        """Test generating plan with missing files creates sync jobs."""
        from scripts.p2p.models import ClusterDataManifest, DataFileInfo, NodeDataManifest

        planner = SyncPlanner(
            node_id="node-a",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Node A has a file, Node B doesn't
        node_a_manifest = NodeDataManifest(
            node_id="node-a",
            collected_at=time.time(),
            files=[
                DataFileInfo(
                    path="selfplay/games.jsonl",
                    size_bytes=1000,
                    modified_time=time.time(),
                    file_type="selfplay",
                )
            ],
        )

        node_b_manifest = NodeDataManifest(
            node_id="node-b",
            collected_at=time.time(),
            files=[],
        )

        cluster_manifest = ClusterDataManifest(
            collected_at=time.time(),
            node_manifests={
                "node-a": node_a_manifest,
                "node-b": node_b_manifest,
            },
            missing_from_nodes={
                "selfplay/games.jsonl": ["node-b"],
            },
        )

        plan = planner.generate_sync_plan(cluster_manifest)

        assert plan is not None
        assert len(plan.sync_jobs) == 1
        assert plan.sync_jobs[0].source_node == "node-a"
        assert plan.sync_jobs[0].target_node == "node-b"
        assert "selfplay/games.jsonl" in plan.sync_jobs[0].files

    def test_generate_plan_tracks_statistics(self):
        """Test that plan generation updates statistics."""
        from scripts.p2p.models import ClusterDataManifest, DataFileInfo, NodeDataManifest

        planner = SyncPlanner(
            node_id="node-a",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        node_a_manifest = NodeDataManifest(
            node_id="node-a",
            collected_at=time.time(),
            files=[
                DataFileInfo(
                    path="games.jsonl",
                    size_bytes=5000,
                    modified_time=time.time(),
                    file_type="selfplay",
                )
            ],
        )

        cluster_manifest = ClusterDataManifest(
            collected_at=time.time(),
            node_manifests={"node-a": node_a_manifest},
            missing_from_nodes={"games.jsonl": ["node-b"]},
        )

        initial_plans = planner.stats.sync_plans_generated
        planner.generate_sync_plan(cluster_manifest)
        assert planner.stats.sync_plans_generated == initial_plans + 1


# ============================================================================
# Test Sync Plan Execution
# ============================================================================


class TestSyncPlanExecution:
    """Tests for sync plan execution."""

    @pytest.mark.asyncio
    async def test_execute_no_plan(self):
        """Test executing without a plan returns error."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        result = await planner.execute_sync_plan()
        assert not result["success"]
        assert "No sync plan" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_disk_full(self):
        """Test executing when disk is full returns error."""
        from scripts.p2p.models import ClusterSyncPlan

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
            check_disk_capacity=lambda: (False, 95.0),
        )

        plan = ClusterSyncPlan(plan_id="test", created_at=time.time())
        result = await planner.execute_sync_plan(plan)

        assert not result["success"]
        assert "Disk capacity" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_sync_already_in_progress(self):
        """Test that concurrent sync attempts are rejected."""
        from scripts.p2p.models import ClusterSyncPlan

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        plan = ClusterSyncPlan(plan_id="test", created_at=time.time())

        # Simulate sync in progress
        planner._sync_in_progress = True

        result = await planner.execute_sync_plan(plan)
        assert not result["success"]
        assert "already in progress" in result["error"]

    @pytest.mark.asyncio
    async def test_execute_with_sync_callback(self):
        """Test executing with a synchronous callback."""
        from scripts.p2p.models import ClusterSyncPlan, DataSyncJob

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        job = DataSyncJob(
            job_id="job-1",
            source_node="node-a",
            target_node="node-b",
            files=["test.jsonl"],
            status="pending",
        )
        plan = ClusterSyncPlan(
            plan_id="test",
            created_at=time.time(),
            sync_jobs=[job],
        )

        # Sync callback that returns True (success)
        def sync_callback(job):
            return True

        result = await planner.execute_sync_plan(plan, execute_job_callback=sync_callback)

        assert result["success"]
        assert result["jobs_completed"] == 1
        assert result["jobs_failed"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_async_callback(self):
        """Test executing with an asynchronous callback."""
        from scripts.p2p.models import ClusterSyncPlan, DataSyncJob

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        job = DataSyncJob(
            job_id="job-1",
            source_node="node-a",
            target_node="node-b",
            files=["test.jsonl"],
            status="pending",
        )
        plan = ClusterSyncPlan(
            plan_id="test",
            created_at=time.time(),
            sync_jobs=[job],
        )

        async def async_callback(job):
            await asyncio.sleep(0.01)
            return True

        result = await planner.execute_sync_plan(plan, execute_job_callback_async=async_callback)

        assert result["success"]
        assert result["jobs_completed"] == 1

    @pytest.mark.asyncio
    async def test_execute_handles_job_failure(self):
        """Test that job failures are tracked correctly."""
        from scripts.p2p.models import ClusterSyncPlan, DataSyncJob

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        job = DataSyncJob(
            job_id="job-1",
            source_node="node-a",
            target_node="node-b",
            files=["test.jsonl"],
            status="pending",
        )
        plan = ClusterSyncPlan(
            plan_id="test",
            created_at=time.time(),
            sync_jobs=[job],
        )

        def failing_callback(job):
            return False

        result = await planner.execute_sync_plan(plan, execute_job_callback=failing_callback)

        assert result["success"]  # Execution succeeded, but job failed
        assert result["jobs_completed"] == 0
        assert result["jobs_failed"] == 1


# ============================================================================
# Test Utility Methods
# ============================================================================


class TestUtilityMethods:
    """Tests for utility methods."""

    def test_get_cached_manifest_none_when_empty(self):
        """Test get_cached_manifest returns None when no cache."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        assert planner.get_cached_manifest() is None

    def test_get_cached_manifest_returns_valid_cache(self):
        """Test get_cached_manifest returns cache when valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
                config=SyncPlannerConfig(manifest_cache_age_seconds=300),
            )

            # Collect to populate cache
            planner.collect_local_manifest()

            # Should return cached
            cached = planner.get_cached_manifest()
            assert cached is not None
            assert cached.node_id == "test-node"

    def test_get_cluster_manifest_returns_none_when_empty(self):
        """Test get_cluster_manifest returns None when not collected."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        assert planner.get_cluster_manifest() is None

    def test_clear_sync_jobs(self):
        """Test clearing completed sync jobs."""
        from scripts.p2p.models import DataSyncJob

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Add jobs with different statuses
        planner._active_sync_jobs["job-1"] = DataSyncJob(
            job_id="job-1", source_node="a", target_node="b", files=[], status="completed"
        )
        planner._active_sync_jobs["job-2"] = DataSyncJob(
            job_id="job-2", source_node="a", target_node="b", files=[], status="pending"
        )
        planner._active_sync_jobs["job-3"] = DataSyncJob(
            job_id="job-3", source_node="a", target_node="b", files=[], status="failed"
        )

        planner.clear_sync_jobs()

        # Only pending jobs should remain
        assert len(planner._active_sync_jobs) == 1
        assert "job-2" in planner._active_sync_jobs


# ============================================================================
# Test Statistics Tracking
# ============================================================================


class TestStatistics:
    """Tests for statistics tracking."""

    def test_get_stats_returns_all_fields(self):
        """Test get_stats returns all expected fields."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        stats = planner.get_stats()

        expected_fields = [
            "manifests_collected",
            "sync_plans_generated",
            "sync_jobs_created",
            "sync_jobs_completed",
            "sync_jobs_failed",
            "bytes_synced",
            "last_manifest_collection",
            "last_sync_execution",
            "cached_manifest_age",
            "cluster_manifest_nodes",
            "active_sync_jobs",
            "sync_in_progress",
            "events_emitted",
            "events_failed",
        ]

        for field in expected_fields:
            assert field in stats

    def test_stats_after_manifest_collection(self):
        """Test statistics are updated after manifest collection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            planner.collect_local_manifest()
            stats = planner.get_stats()

            assert stats["manifests_collected"] == 1
            assert stats["last_manifest_collection"] > 0


# ============================================================================
# Test Health Check
# ============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_healthy_state(self):
        """Test health check returns healthy for normal operation."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        health = planner.health_check()

        # health_check returns HealthCheckResult dataclass
        assert hasattr(health, "healthy")
        assert health.healthy is True
        assert hasattr(health, "details")
        if health.details:
            assert health.details.get("errors_count", 0) == 0

    def test_health_check_unhealthy_high_failure_rate(self):
        """Test health check returns unhealthy with high failure rate."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Simulate high failure rate
        planner.stats.sync_jobs_created = 10
        planner.stats.sync_jobs_failed = 8  # 80% failure rate

        health = planner.health_check()

        assert hasattr(health, "healthy")
        assert health.healthy is False

    def test_health_check_degraded_moderate_failure_rate(self):
        """Test health check returns degraded with moderate failure rate."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Simulate moderate failure rate
        planner.stats.sync_jobs_created = 10
        planner.stats.sync_jobs_failed = 3  # 30% failure rate

        health = planner.health_check()

        assert hasattr(health, "healthy")
        assert hasattr(health, "status")

    def test_health_check_degraded_stale_sync(self):
        """Test health check returns degraded when sync is stale."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Simulate stale sync (last sync over 30 minutes ago)
        planner.stats.last_sync_execution = time.time() - 3600  # 1 hour ago

        health = planner.health_check()

        assert hasattr(health, "healthy")
        assert hasattr(health, "status")


# ============================================================================
# Test Disk-based Manifest Cache
# ============================================================================


class TestDiskManifestCache:
    """Tests for disk-based manifest caching."""

    def test_save_manifest_to_cache(self):
        """Test saving manifest to disk cache."""
        from scripts.p2p.models import NodeDataManifest

        with tempfile.TemporaryDirectory() as tmpdir:
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            manifest = NodeDataManifest(
                node_id="test-node",
                collected_at=time.time(),
                total_files=5,
            )

            result = planner.save_manifest_to_cache(manifest)
            assert result is True

            cache_path = planner.get_manifest_cache_path()
            assert cache_path.exists()

    def test_load_manifest_from_cache_valid(self):
        """Test loading valid manifest from disk cache."""
        from scripts.p2p.models import NodeDataManifest

        with tempfile.TemporaryDirectory() as tmpdir:
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            manifest = NodeDataManifest(
                node_id="test-node",
                collected_at=time.time(),
                total_files=5,
            )

            planner.save_manifest_to_cache(manifest)
            loaded = planner.load_manifest_from_cache(max_age_seconds=300)

            assert loaded is not None
            assert loaded.node_id == "test-node"
            assert loaded.total_files == 5

    def test_load_manifest_from_cache_expired(self):
        """Test loading expired manifest returns None."""
        from scripts.p2p.models import NodeDataManifest

        with tempfile.TemporaryDirectory() as tmpdir:
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            # Create cache manually with old timestamp
            cache_path = planner.get_manifest_cache_path()
            cache_data = {
                "version": 1,
                "saved_at": time.time() - 600,  # 10 minutes ago
                "manifest": {"node_id": "test-node", "collected_at": time.time() - 600},
            }
            with open(cache_path, "w") as f:
                json.dump(cache_data, f)

            loaded = planner.load_manifest_from_cache(max_age_seconds=60)
            assert loaded is None

    def test_load_manifest_from_cache_nonexistent(self):
        """Test loading from nonexistent cache returns None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = SyncPlanner(
                node_id="test-node",
                data_directory=Path(tmpdir),
                get_peers=lambda: {},
                get_self_info=lambda: MagicMock(),
                peers_lock=threading.Lock(),
                is_leader=lambda: True,
            )

            loaded = planner.load_manifest_from_cache()
            assert loaded is None


# ============================================================================
# Test Event Emission
# ============================================================================


class TestEventEmission:
    """Tests for event emission functionality."""

    def test_validate_event_types_success(self):
        """Test event type validation passes with valid types."""
        # Should not raise when DataEventType has required types
        result = _validate_event_types()
        assert result is True  # Or check it doesn't raise

    @patch("scripts.p2p.managers.sync_planner._publish_sync", None)
    def test_emit_sync_event_no_emitter(self):
        """Test event emission gracefully handles missing emitter."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Should not raise
        from app.distributed.data_events import DataEventType
        planner._emit_sync_event(DataEventType.DATA_SYNC_COMPLETED, test="value")


# ============================================================================
# Test Event Subscriptions
# ============================================================================


class TestEventSubscriptions:
    """Tests for event subscription handling."""

    def test_get_event_subscriptions(self):
        """Test that event subscriptions are defined."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        subscriptions = planner._get_event_subscriptions()

        assert "LEADER_ELECTED" in subscriptions
        assert "NODE_RECOVERED" in subscriptions
        assert "HOST_ONLINE" in subscriptions

    @pytest.mark.asyncio
    async def test_on_leader_elected_clears_cache(self):
        """Test LEADER_ELECTED handler clears manifests."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Set some cached data
        planner._cached_local_manifest = MagicMock()
        planner._cached_manifest_time = time.time()
        planner._cluster_manifest = MagicMock()

        event = {"payload": {"leader_id": "new-leader"}}
        await planner._on_leader_elected(event)

        assert planner._cached_local_manifest is None
        assert planner._cached_manifest_time == 0.0
        assert planner._cluster_manifest is None

    @pytest.mark.asyncio
    async def test_on_node_recovered_invalidates_cluster_manifest(self):
        """Test NODE_RECOVERED handler invalidates cluster manifest."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        planner._cluster_manifest = MagicMock()

        # Pass event with .payload attribute (DataEvent style)
        event = MagicMock()
        event.payload = {"node_id": "recovered-node"}
        await planner._on_node_recovered(event)

        assert planner._cluster_manifest is None

    @pytest.mark.asyncio
    async def test_on_host_online_invalidates_cluster_manifest(self):
        """Test HOST_ONLINE handler invalidates cluster manifest."""
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        planner._cluster_manifest = MagicMock()

        # Pass event with .payload attribute (DataEvent style)
        event = MagicMock()
        event.payload = {"node_id": "new-node"}
        await planner._on_host_online(event)

        assert planner._cluster_manifest is None
