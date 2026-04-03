"""Unit tests for SyncPlanner manager.

December 2025: Tests for Phase 2A extraction of sync planning logic.
"""

import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Import after defining mock models to avoid circular import issues
from scripts.p2p.managers.sync_planner import SyncPlanner, SyncPlannerConfig, SyncStats


class TestSyncPlannerConfig:
    """Tests for SyncPlannerConfig."""

    def test_default_values(self):
        config = SyncPlannerConfig()
        assert config.manifest_cache_age_seconds == 60
        assert config.manifest_collection_interval == 60
        assert config.max_files_per_sync_job == 50
        assert config.sync_mtime_tolerance_seconds == 60
        assert config.max_concurrent_syncs == 1

    def test_custom_values(self):
        config = SyncPlannerConfig(
            manifest_cache_age_seconds=600,
            max_files_per_sync_job=100,
        )
        assert config.manifest_cache_age_seconds == 600
        assert config.max_files_per_sync_job == 100


class TestSyncStats:
    """Tests for SyncStats."""

    def test_default_values(self):
        stats = SyncStats()
        assert stats.manifests_collected == 0
        assert stats.sync_plans_generated == 0
        assert stats.sync_jobs_completed == 0

    def test_stats_increment(self):
        stats = SyncStats()
        stats.manifests_collected += 1
        stats.sync_jobs_completed += 5
        assert stats.manifests_collected == 1
        assert stats.sync_jobs_completed == 5


class TestSyncPlannerInit:
    """Tests for SyncPlanner initialization."""

    def test_init_with_required_args(self):
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

    def test_init_with_custom_config(self):
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


class TestLocalManifestCollection:
    """Tests for local manifest collection."""

    def test_collect_empty_directory(self):
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
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test selfplay directory and files
            selfplay_dir = Path(tmpdir) / "selfplay"
            selfplay_dir.mkdir()

            # Create a JSONL file with some game data
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
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test models directory
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            # Create a model file
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

    def test_manifest_caching(self):
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


class TestBoardConfigParsing:
    """Tests for board type/player count parsing."""

    def test_parse_square8_2p(self):
        planner = SyncPlanner(
            node_id="test",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        board_type, num_players = planner._parse_board_config("selfplay/games_sq8_2p.jsonl")
        assert board_type == "square8"
        assert num_players == 2

    def test_parse_hex_4p(self):
        planner = SyncPlanner(
            node_id="test",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        board_type, num_players = planner._parse_board_config("models/hex_4p_model.pt")
        assert board_type == "hexagonal"
        assert num_players == 4

    def test_parse_square19_3p(self):
        planner = SyncPlanner(
            node_id="test",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        board_type, num_players = planner._parse_board_config("training/square19_3p.npz")
        assert board_type == "square19"
        assert num_players == 3


class TestSyncPlanGeneration:
    """Tests for sync plan generation."""

    def test_generate_plan_no_manifest(self):
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
        from scripts.p2p.models import ClusterDataManifest

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Create a manifest with no missing files
        manifest = ClusterDataManifest(collected_at=time.time())

        plan = planner.generate_sync_plan(manifest)
        assert plan is None  # No sync needed

    def test_generate_plan_with_missing_files(self):
        from scripts.p2p.models import ClusterDataManifest, DataFileInfo, NodeDataManifest

        planner = SyncPlanner(
            node_id="node-a",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        # Create manifests for two nodes
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
            files=[],  # Node B has no files
        )

        # Create cluster manifest with missing file
        cluster_manifest = ClusterDataManifest(
            collected_at=time.time(),
            node_manifests={
                "node-a": node_a_manifest,
                "node-b": node_b_manifest,
            },
            missing_from_nodes={
                "selfplay/games.jsonl": ["node-b"],  # Missing from node B
            },
        )

        plan = planner.generate_sync_plan(cluster_manifest)

        assert plan is not None
        assert len(plan.sync_jobs) == 1
        assert plan.sync_jobs[0].source_node == "node-a"
        assert plan.sync_jobs[0].target_node == "node-b"
        assert "selfplay/games.jsonl" in plan.sync_jobs[0].files


class TestSyncPlanExecution:
    """Tests for sync plan execution."""

    @pytest.mark.asyncio
    async def test_execute_no_plan(self):
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
        from scripts.p2p.models import ClusterSyncPlan

        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
            check_disk_capacity=lambda: (False, 95.0),  # Disk full
        )

        plan = ClusterSyncPlan(plan_id="test", created_at=time.time())
        result = await planner.execute_sync_plan(plan)

        assert not result["success"]
        assert "Disk capacity" in result["error"]


class TestStatistics:
    """Tests for statistics tracking."""

    def test_get_stats(self):
        planner = SyncPlanner(
            node_id="test-node",
            data_directory=Path("/tmp"),
            get_peers=lambda: {},
            get_self_info=lambda: MagicMock(),
            peers_lock=threading.Lock(),
            is_leader=lambda: True,
        )

        stats = planner.get_stats()

        assert "manifests_collected" in stats
        assert "sync_plans_generated" in stats
        assert "sync_jobs_completed" in stats
        assert "active_sync_jobs" in stats
        assert "sync_in_progress" in stats

    def test_stats_after_manifest_collection(self):
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
