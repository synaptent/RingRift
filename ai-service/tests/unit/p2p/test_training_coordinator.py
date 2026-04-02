"""Unit tests for TrainingCoordinator.

Tests training readiness checking, job dispatch, and completion workflows.
"""

import asyncio
import hashlib
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.managers.training_coordinator import (
    TrainingCoordinator,
    MIN_MEMORY_GB_FOR_TASKS,
    LEADERLESS_TRAINING_TIMEOUT,
)


# Mock classes for testing
@dataclass
class MockTrainingThresholds:
    """Mock TrainingThresholds for testing."""
    auto_nnue_enabled: bool = True
    auto_cmaes_enabled: bool = True
    _min_games: dict = field(default_factory=lambda: {"nnue": 1000, "cmaes": 5000})
    _incremental: dict = field(default_factory=lambda: {"nnue": 500, "cmaes": 1000})
    _cooldown: int = 300

    def get_effective_min_games(self, job_type: str) -> int:
        return self._min_games.get(job_type, 1000)

    def get_effective_incremental(self, job_type: str) -> int:
        return self._incremental.get(job_type, 500)

    def get_effective_cooldown(self) -> int:
        return self._cooldown

    def update_from_cluster_state(self, gpu_count: int) -> None:
        pass


@dataclass
class MockClusterDataManifest:
    """Mock cluster data manifest."""
    by_board_type: dict = field(default_factory=dict)


@dataclass
class MockTrainingJob:
    """Mock training job."""
    job_id: str
    job_type: str
    board_type: str
    num_players: int
    status: str = "pending"
    worker_node: str = ""
    checkpoint_path: str = ""
    checkpoint_epoch: int = 0
    resume_from_checkpoint: bool = False
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0
    data_games_count: int = 0
    epochs: int = 50
    batch_size: int = 256
    learning_rate: float = 0.001
    output_model_path: str = ""
    error_message: str = ""


@dataclass
class MockNodeInfo:
    """Mock node info."""
    node_id: str
    has_gpu: bool = True
    gpu_name: str = "RTX 4090"
    memory_gb: int = 64

    def is_healthy(self) -> bool:
        return True

    def is_cpu_only_node(self) -> bool:
        return not self.has_gpu

    def gpu_power_score(self) -> float:
        return 100.0 if self.has_gpu else 0.0

    def cpu_power_score(self) -> float:
        return 50.0

    def get_load_score(self) -> float:
        return 0.5


class TestTrainingCoordinatorInit:
    """Test TrainingCoordinator initialization."""

    def test_basic_init(self):
        """Test basic initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            assert coord.ringrift_path == Path(tmpdir)
            assert coord.games_at_last_nnue_train == {}
            assert coord.games_at_last_cmaes_train == {}
            assert coord._training_trigger_cache == {}

    def test_init_with_optional_params(self):
        """Test initialization with optional parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            games_nnue = {"hex8_2p": 1000}
            games_cmaes = {"hex8_2p": 5000}

            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
                games_at_last_nnue_train=games_nnue,
                games_at_last_cmaes_train=games_cmaes,
                auth_headers=lambda: {"Authorization": "test"},
            )

            assert coord.games_at_last_nnue_train == games_nnue
            assert coord.games_at_last_cmaes_train == games_cmaes
            assert coord.auth_headers() == {"Authorization": "test"}


class TestTrainingReadiness:
    """Test training readiness checking."""

    def test_check_training_readiness_empty_manifest(self):
        """Test readiness check with empty manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: None,
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            jobs = coord.check_training_readiness()
            assert jobs == []

    def test_check_training_readiness_below_threshold(self):
        """Test readiness check when games below threshold."""
        manifest = MockClusterDataManifest()
        manifest.by_board_type = {"hex8_2p": {"total_games": 500}}  # Below 1000

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: manifest,
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            jobs = coord.check_training_readiness()
            assert jobs == []

    def test_check_training_readiness_nnue_trigger(self):
        """Test readiness check triggers NNUE training."""
        manifest = MockClusterDataManifest()
        manifest.by_board_type = {"hex8_2p": {"total_games": 1500}}

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: manifest,
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            jobs = coord.check_training_readiness()

            # Should trigger NNUE (>= 1000 games)
            nnue_jobs = [j for j in jobs if j["job_type"] == "nnue"]
            assert len(nnue_jobs) == 1
            assert nnue_jobs[0]["board_type"] == "hex8"
            assert nnue_jobs[0]["num_players"] == 2

    def test_check_training_readiness_cmaes_trigger(self):
        """Test readiness check triggers CMA-ES training."""
        manifest = MockClusterDataManifest()
        manifest.by_board_type = {"hex8_2p": {"total_games": 6000}}

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: manifest,
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            jobs = coord.check_training_readiness()

            # Should trigger both NNUE and CMA-ES
            nnue_jobs = [j for j in jobs if j["job_type"] == "nnue"]
            cmaes_jobs = [j for j in jobs if j["job_type"] == "cmaes"]
            assert len(nnue_jobs) == 1
            assert len(cmaes_jobs) == 1

    def test_check_training_readiness_incremental(self):
        """Test incremental training triggering."""
        manifest = MockClusterDataManifest()
        manifest.by_board_type = {"hex8_2p": {"total_games": 2000}}

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: manifest,
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
                games_at_last_nnue_train={"hex8_2p": 1400},  # Last trained at 1400
            )

            jobs = coord.check_training_readiness()

            # Should trigger (2000 - 1400 = 600 >= 500 incremental)
            nnue_jobs = [j for j in jobs if j["job_type"] == "nnue"]
            assert len(nnue_jobs) == 1

    def test_check_training_readiness_insufficient_incremental(self):
        """Test no trigger when incremental threshold not met."""
        manifest = MockClusterDataManifest()
        manifest.by_board_type = {"hex8_2p": {"total_games": 1800}}

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: manifest,
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
                games_at_last_nnue_train={"hex8_2p": 1500},  # Last trained at 1500
            )

            jobs = coord.check_training_readiness()

            # Should NOT trigger (1800 - 1500 = 300 < 500 incremental)
            nnue_jobs = [j for j in jobs if j["job_type"] == "nnue"]
            assert len(nnue_jobs) == 0


class TestFindJobs:
    """Test job finding methods."""

    def test_find_running_training_job_none(self):
        """Test finding running job when none exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            job = coord.find_running_training_job("nnue", "hex8_2p")
            assert job is None

    def test_find_running_training_job_exists(self):
        """Test finding running job when one exists."""
        running_job = MockTrainingJob(
            job_id="job1",
            job_type="nnue",
            board_type="hex8",
            num_players=2,
            status="running",
        )
        training_jobs = {"job1": running_job}

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: training_jobs,
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            job = coord.find_running_training_job("nnue", "hex8_2p")
            assert job is not None
            assert job.job_id == "job1"

    def test_find_running_training_job_wrong_type(self):
        """Test finding running job with wrong type."""
        running_job = MockTrainingJob(
            job_id="job1",
            job_type="cmaes",  # Wrong type
            board_type="hex8",
            num_players=2,
            status="running",
        )
        training_jobs = {"job1": running_job}

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: training_jobs,
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            job = coord.find_running_training_job("nnue", "hex8_2p")
            assert job is None

    def test_find_resumable_training_job(self):
        """Test finding resumable job with checkpoint."""
        failed_job = MockTrainingJob(
            job_id="job1",
            job_type="nnue",
            board_type="hex8",
            num_players=2,
            status="failed",
            checkpoint_path="/path/to/checkpoint.pth",
            checkpoint_epoch=25,
        )
        training_jobs = {"job1": failed_job}

        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: training_jobs,
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            job = coord.find_resumable_training_job("nnue", "hex8_2p")
            assert job is not None
            assert job.checkpoint_epoch == 25


class TestTrainingIdempotency:
    """Test training trigger idempotency."""

    def test_compute_training_trigger_hash(self):
        """Test training trigger hash computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            hash1 = coord.compute_training_trigger_hash("hex8_2p", 1500)
            hash2 = coord.compute_training_trigger_hash("hex8_2p", 1500)
            hash3 = coord.compute_training_trigger_hash("hex8_2p", 2500)  # Different bucket
            hash4 = coord.compute_training_trigger_hash("square8_2p", 1500)  # Different config

            # Same inputs should produce same hash
            assert hash1 == hash2
            # Different game bucket should produce different hash
            assert hash1 != hash3
            # Different config should produce different hash
            assert hash1 != hash4

    def test_is_training_trigger_duplicate(self):
        """Test duplicate training trigger detection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            # First check - not duplicate
            assert coord.is_training_trigger_duplicate("hash123") is False

            # Record the trigger
            coord.record_training_trigger("hash123")

            # Now it's a duplicate
            assert coord.is_training_trigger_duplicate("hash123") is True

            # Different hash is not duplicate
            assert coord.is_training_trigger_duplicate("hash456") is False

    def test_check_training_idempotency(self):
        """Test combined idempotency check."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            # First check - can proceed
            can_proceed, trigger_hash = coord.check_training_idempotency("hex8_2p", 1500)
            assert can_proceed is True

            # Record the trigger
            coord.record_training_trigger(trigger_hash)

            # Second check - cannot proceed (duplicate)
            can_proceed2, _ = coord.check_training_idempotency("hex8_2p", 1500)
            assert can_proceed2 is False


class TestJobDispatch:
    """Test training job dispatch."""

    @pytest.mark.asyncio
    async def test_dispatch_training_job_no_workers(self):
        """Test dispatch when no workers available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            training_jobs = {}
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: training_jobs,
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self", memory_gb=4),  # Low memory
                training_thresholds=MockTrainingThresholds(),
            )

            job_config = {
                "job_type": "nnue",
                "board_type": "hex8",
                "num_players": 2,
                "config_key": "hex8_2p",
                "total_games": 2000,
            }

            with patch.object(coord, 'urls_for_peer', return_value=[]):
                job = await coord.dispatch_training_job(job_config)
                assert job is None  # No suitable worker

    @pytest.mark.asyncio
    async def test_dispatch_training_job_with_checkpoint(self):
        """Test dispatch with checkpoint resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a failed job with checkpoint
            failed_job = MockTrainingJob(
                job_id="old_job",
                job_type="nnue",
                board_type="hex8",
                num_players=2,
                status="failed",
                checkpoint_path="/path/to/checkpoint.pth",
                checkpoint_epoch=25,
            )
            training_jobs = {"old_job": failed_job}

            self_node = MockNodeInfo("self", memory_gb=64)
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: training_jobs,
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: self_node,
                training_thresholds=MockTrainingThresholds(),
            )

            job_config = {
                "job_type": "nnue",
                "board_type": "hex8",
                "num_players": 2,
                "config_key": "hex8_2p",
                "total_games": 2000,
            }

            # Check that resumable job is found
            resumable = coord.find_resumable_training_job("nnue", "hex8_2p")
            assert resumable is not None
            assert resumable.checkpoint_epoch == 25


class TestModelArchiving:
    """Test model archiving functionality."""

    @pytest.mark.asyncio
    async def test_archive_failed_model_no_path(self):
        """Test archiving when model path is empty."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            # Should not raise
            await coord._archive_failed_model("", "hex8", 2, "test_reason")

    @pytest.mark.asyncio
    async def test_archive_failed_model_nonexistent_path(self):
        """Test archiving when model path doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            # Should not raise
            await coord._archive_failed_model(
                "/nonexistent/model.pth", "hex8", 2, "test_reason"
            )

    @pytest.mark.asyncio
    async def test_archive_failed_model_moves_file(self):
        """Test archiving actually moves the model file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model file
            model_path = Path(tmpdir) / "model.pth"
            model_path.write_text("mock model")

            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            await coord._archive_failed_model(str(model_path), "hex8", 2, "test_reason")

            # Original should be gone
            assert not model_path.exists()

            # Should be in archive
            archive_path = Path(tmpdir) / "ai-service" / "models" / "archived" / "hex8_2p" / "model.pth"
            assert archive_path.exists()


class TestMedianModel:
    """Test median model lookup."""

    def test_get_median_model_no_db(self):
        """Test median model when no ELO database exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            result = coord._get_median_model("hex8_2p")
            assert result is None

    def test_get_median_model_with_db(self):
        """Test median model lookup from ELO database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create ELO database
            elo_dir = Path(tmpdir) / "ai-service" / "data"
            elo_dir.mkdir(parents=True)
            elo_db = elo_dir / "unified_elo.db"

            conn = sqlite3.connect(str(elo_db))
            conn.execute("""
                CREATE TABLE elo_ratings (
                    participant_id TEXT,
                    board_type TEXT,
                    num_players INTEGER,
                    rating REAL,
                    archived_at REAL
                )
            """)
            # Insert models with varying ratings
            conn.execute("INSERT INTO elo_ratings VALUES ('model_low', 'hex8', 2, 1000, NULL)")
            conn.execute("INSERT INTO elo_ratings VALUES ('model_mid', 'hex8', 2, 1200, NULL)")
            conn.execute("INSERT INTO elo_ratings VALUES ('model_high', 'hex8', 2, 1400, NULL)")
            conn.commit()
            conn.close()

            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            result = coord._get_median_model("hex8_2p")
            # Median of 3 models is the middle one (index 1)
            assert result == "model_mid"


class TestPostTrainingGauntlet:
    """Test post-training gauntlet evaluation."""

    @pytest.mark.asyncio
    async def test_run_post_training_gauntlet_skip_env(self):
        """Test gauntlet skipped via environment variable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            job = MockTrainingJob(
                job_id="test",
                job_type="nnue",
                board_type="hex8",
                num_players=2,
                output_model_path="/path/to/model.pth",
            )

            with patch.dict('os.environ', {'RINGRIFT_SKIP_POST_TRAINING_GAUNTLET': '1'}):
                result = await coord._run_post_training_gauntlet(job)
                assert result is True  # Skipped = passed

    @pytest.mark.asyncio
    async def test_run_post_training_gauntlet_no_model(self):
        """Test gauntlet with no model path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            job = MockTrainingJob(
                job_id="test",
                job_type="nnue",
                board_type="hex8",
                num_players=2,
                output_model_path="",
            )

            result = await coord._run_post_training_gauntlet(job)
            assert result is True  # No model = passed (nothing to evaluate)


class TestModelPromotion:
    """Test model promotion."""

    @pytest.mark.asyncio
    async def test_promote_to_baseline(self):
        """Test promoting model to baseline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model file
            model_path = Path(tmpdir) / "model.pth"
            model_path.write_text("mock model")

            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            await coord.promote_to_baseline(str(model_path), "hex8", 2, "nnue")

            # Baseline should exist
            baseline_path = Path(tmpdir) / "ai-service" / "models" / "nnue" / "hex8_2p_best.pt"
            assert baseline_path.exists()

    @pytest.mark.asyncio
    async def test_promote_to_baseline_with_existing(self):
        """Test promoting model when baseline already exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create model directory and existing baseline
            nnue_dir = Path(tmpdir) / "ai-service" / "models" / "nnue"
            nnue_dir.mkdir(parents=True)
            existing_baseline = nnue_dir / "hex8_2p_best.pt"
            existing_baseline.write_text("old model")

            # Create new model
            model_path = Path(tmpdir) / "new_model.pth"
            model_path.write_text("new model")

            coord = TrainingCoordinator(
                ringrift_path=Path(tmpdir),
                get_cluster_data_manifest=lambda: MockClusterDataManifest(),
                get_training_jobs=lambda: {},
                get_training_lock=lambda: threading.Lock(),
                get_peers=lambda: {},
                get_peers_lock=lambda: threading.Lock(),
                get_self_info=lambda: MockNodeInfo("self"),
                training_thresholds=MockTrainingThresholds(),
            )

            await coord.promote_to_baseline(str(model_path), "hex8", 2, "nnue")

            # New baseline should have new content
            assert existing_baseline.read_text() == "new model"

            # Backup should exist
            backups = list(nnue_dir.glob("hex8_2p_prev_*.pt"))
            assert len(backups) == 1
            assert backups[0].read_text() == "old model"


class TestConstants:
    """Test module constants."""

    def test_min_memory_threshold(self):
        """Test MIN_MEMORY_GB_FOR_TASKS constant.

        Dec 2025: Increased to 64GB for GH200 nodes.
        """
        assert MIN_MEMORY_GB_FOR_TASKS == 64

    def test_leaderless_timeout(self):
        """Test LEADERLESS_TRAINING_TIMEOUT constant.

        Dec 2025-Jan 2026: Tightened from 180s and later floor-raised to 45s
        via centralized constants to keep decentralized training responsive
        without flapping too aggressively.
        """
        assert LEADERLESS_TRAINING_TIMEOUT == 45
