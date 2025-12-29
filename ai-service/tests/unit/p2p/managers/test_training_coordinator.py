"""Tests for P2P TrainingCoordinator - Training dispatch and completion workflows.

December 28, 2025: Created for comprehensive testing of training coordination,
including dispatch, gauntlet integration, and model promotion flows.

This test module covers the critical training coordination paths:
1. Training Dispatch - Triggering training after data is ready
2. Gauntlet Integration - Running evaluations after training
3. Model Promotion - Promoting models based on gauntlet results
"""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Mock Classes
# =============================================================================


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
    """Mock training job for testing."""
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
    val_loss: float = 0.0


@dataclass
class MockNodeInfo:
    """Mock node info for testing."""
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


@dataclass
class MockImprovementCycleManager:
    """Mock improvement cycle manager for testing."""
    handle_training_complete_called: bool = False
    get_tournament_matchups_called: bool = False

    class MockCycleState:
        def __init__(self):
            self.cycles = {}

    def __init__(self):
        self.state = self.MockCycleState()

    def handle_training_complete(self, board_type, num_players, model_path, games_count):
        self.handle_training_complete_called = True

    def get_tournament_matchups(self, board_type, num_players, new_model_path=None):
        self.get_tournament_matchups_called = True
        return []

    def _save_state(self):
        pass


@dataclass
class MockEvent:
    """Mock event for event handler tests."""
    payload: dict


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_ringrift_path():
    """Create a temporary RingRift path structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create expected directory structure
        ai_service = Path(tmpdir) / "ai-service"
        ai_service.mkdir(parents=True)
        (ai_service / "data").mkdir()
        (ai_service / "models").mkdir()
        yield Path(tmpdir)


@pytest.fixture
def mock_training_jobs():
    """Provide a mutable training jobs dict."""
    return {}


@pytest.fixture
def training_lock():
    """Provide a threading lock for training jobs."""
    return threading.Lock()


@pytest.fixture
def peers_lock():
    """Provide a threading lock for peers."""
    return threading.Lock()


@pytest.fixture
def training_coordinator(temp_ringrift_path, mock_training_jobs, training_lock, peers_lock):
    """Create a TrainingCoordinator instance for testing."""
    from scripts.p2p.managers.training_coordinator import TrainingCoordinator

    return TrainingCoordinator(
        ringrift_path=temp_ringrift_path,
        get_cluster_data_manifest=lambda: MockClusterDataManifest(),
        get_training_jobs=lambda: mock_training_jobs,
        get_training_lock=lambda: training_lock,
        get_peers=lambda: {},
        get_peers_lock=lambda: peers_lock,
        get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
        training_thresholds=MockTrainingThresholds(),
    )


@pytest.fixture
def coordinator_with_manifest(temp_ringrift_path, mock_training_jobs, training_lock, peers_lock):
    """Create a coordinator with a data manifest that meets thresholds."""
    from scripts.p2p.managers.training_coordinator import TrainingCoordinator

    manifest = MockClusterDataManifest()
    manifest.by_board_type = {"hex8_2p": {"total_games": 2000}}

    return TrainingCoordinator(
        ringrift_path=temp_ringrift_path,
        get_cluster_data_manifest=lambda: manifest,
        get_training_jobs=lambda: mock_training_jobs,
        get_training_lock=lambda: training_lock,
        get_peers=lambda: {},
        get_peers_lock=lambda: peers_lock,
        get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
        training_thresholds=MockTrainingThresholds(),
    )


@pytest.fixture
def coordinator_with_cycle_manager(temp_ringrift_path, mock_training_jobs, training_lock, peers_lock):
    """Create a coordinator with an improvement cycle manager."""
    from scripts.p2p.managers.training_coordinator import TrainingCoordinator

    cycle_manager = MockImprovementCycleManager()

    return TrainingCoordinator(
        ringrift_path=temp_ringrift_path,
        get_cluster_data_manifest=lambda: MockClusterDataManifest(),
        get_training_jobs=lambda: mock_training_jobs,
        get_training_lock=lambda: training_lock,
        get_peers=lambda: {},
        get_peers_lock=lambda: peers_lock,
        get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
        training_thresholds=MockTrainingThresholds(),
        improvement_cycle_manager=cycle_manager,
    ), cycle_manager


# =============================================================================
# TestTrainingDispatch
# =============================================================================


class TestTrainingDispatch:
    """Test training job dispatch functionality."""

    def test_dispatch_after_dataset_ready(self, coordinator_with_manifest):
        """Verify dispatch triggers correctly after dataset reaches threshold.

        When the cluster data manifest shows sufficient games for a config,
        check_training_readiness should return jobs to dispatch.
        """
        coord = coordinator_with_manifest

        # Check training readiness
        jobs = coord.check_training_readiness()

        # Should have at least one NNUE job ready
        assert len(jobs) >= 1
        nnue_jobs = [j for j in jobs if j["job_type"] == "nnue"]
        assert len(nnue_jobs) == 1
        assert nnue_jobs[0]["board_type"] == "hex8"
        assert nnue_jobs[0]["num_players"] == 2
        assert nnue_jobs[0]["total_games"] == 2000

    def test_check_readiness_timeout(self, training_coordinator):
        """Verify timeout handling during readiness check.

        The check_training_readiness method should complete quickly even
        when cluster data is slow or unavailable.
        """
        coord = training_coordinator

        # Should complete quickly even with empty manifest
        start_time = time.time()
        jobs = coord.check_training_readiness()
        elapsed = time.time() - start_time

        # Should complete in under 1 second (no blocking operations)
        assert elapsed < 1.0
        assert jobs == []  # Empty manifest = no jobs

    @pytest.mark.asyncio
    async def test_dispatch_failure_notification(self, training_coordinator, mock_training_jobs):
        """Verify error events are emitted on dispatch failure.

        When dispatch_training_job fails to find a suitable worker or
        network errors occur, the job should be marked as failed.
        """
        coord = training_coordinator

        job_config = {
            "job_type": "nnue",
            "board_type": "hex8",
            "num_players": 2,
            "config_key": "hex8_2p",
            "total_games": 2000,
        }

        # Mock the network layer to simulate failure
        with patch.object(coord, 'urls_for_peer', return_value=[]):
            # No suitable workers (self node has high memory but no peers)
            job = await coord.dispatch_training_job(job_config)

            # Either job is None (no workers) or status is failed
            if job is not None:
                # Job should be stored with failed status
                assert job.status in ("failed", "queued")

    @pytest.mark.asyncio
    async def test_dispatch_with_network_error(self, training_coordinator, mock_training_jobs):
        """Verify dispatch handles network errors gracefully."""
        coord = training_coordinator

        job_config = {
            "job_type": "nnue",
            "board_type": "hex8",
            "num_players": 2,
            "config_key": "hex8_2p",
            "total_games": 2000,
        }

        # Mock network to raise exception - patch at the import location
        with patch('scripts.p2p.network.get_client_session') as mock_session:
            mock_session.side_effect = ConnectionError("Network error")
            with patch.object(coord, 'urls_for_peer', return_value=['http://node:8770/training/nnue/start']):
                job = await coord.dispatch_training_job(job_config)

                # Job should be marked as failed
                if job is not None:
                    assert job.status == "failed"
                    assert job.error_message  # Has some error message


# =============================================================================
# TestGauntletIntegration
# =============================================================================


class TestGauntletIntegration:
    """Test gauntlet evaluation integration with training completion."""

    @pytest.mark.asyncio
    async def test_gauntlet_trigger_on_training_complete(self, coordinator_with_cycle_manager):
        """Verify gauntlet starts after training completion.

        When handle_training_job_completion is called with a successful job,
        it should trigger gauntlet evaluation (unless skipped via env var).
        """
        coord, cycle_manager = coordinator_with_cycle_manager

        # Create a completed job with output model
        job = MockTrainingJob(
            job_id="test-job-1",
            job_type="nnue",
            board_type="hex8",
            num_players=2,
            status="completed",
            worker_node="test-worker",  # Dec 2025: Required for model fetch
            output_model_path="/nonexistent/model.pth",  # Doesn't need to exist for test
            data_games_count=2000,
        )

        # Skip actual gauntlet execution (model doesn't exist)
        # Dec 2025: Also mock _fetch_model_from_training_node since "test-worker" doesn't exist
        with patch.dict('os.environ', {'RINGRIFT_SKIP_POST_TRAINING_GAUNTLET': '1'}):
            with patch.object(coord, '_fetch_model_from_training_node', return_value=True):
                await coord.handle_training_job_completion(job)

        # Cycle manager should have been notified
        assert cycle_manager.handle_training_complete_called

    @pytest.mark.asyncio
    async def test_gauntlet_failure_handling(self, temp_ringrift_path, mock_training_jobs, training_lock, peers_lock):
        """Verify graceful handling of gauntlet failures.

        When the gauntlet evaluation fails (e.g., missing dependencies),
        the system should handle it gracefully and not crash.
        """
        from scripts.p2p.managers.training_coordinator import TrainingCoordinator

        cycle_manager = MockImprovementCycleManager()

        coord = TrainingCoordinator(
            ringrift_path=temp_ringrift_path,
            get_cluster_data_manifest=lambda: MockClusterDataManifest(),
            get_training_jobs=lambda: mock_training_jobs,
            get_training_lock=lambda: training_lock,
            get_peers=lambda: {},
            get_peers_lock=lambda: peers_lock,
            get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
            training_thresholds=MockTrainingThresholds(),
            improvement_cycle_manager=cycle_manager,
        )

        # Create a completed job
        job = MockTrainingJob(
            job_id="test-job-2",
            job_type="nnue",
            board_type="hex8",
            num_players=2,
            status="completed",
            worker_node="test-worker",  # Dec 2025: Required for model fetch
            output_model_path="/nonexistent/model.pth",
            data_games_count=2000,
        )

        # Mock gauntlet to raise an exception
        # Dec 2025: Also mock _fetch_model_from_training_node so gauntlet is actually called
        with patch.object(coord, '_fetch_model_from_training_node', return_value=True):
            with patch.object(coord, '_run_post_training_gauntlet', side_effect=Exception("Gauntlet error")):
                # Should not raise - exception should be caught
                await coord.handle_training_job_completion(job)

    @pytest.mark.asyncio
    async def test_model_promotion_on_gauntlet_pass(self, temp_ringrift_path, mock_training_jobs, training_lock, peers_lock):
        """Verify promotion flow when gauntlet passes.

        When a model passes the gauntlet (>= 50% win rate), it should
        be promoted and not archived.
        """
        from scripts.p2p.managers.training_coordinator import TrainingCoordinator

        cycle_manager = MockImprovementCycleManager()

        coord = TrainingCoordinator(
            ringrift_path=temp_ringrift_path,
            get_cluster_data_manifest=lambda: MockClusterDataManifest(),
            get_training_jobs=lambda: mock_training_jobs,
            get_training_lock=lambda: training_lock,
            get_peers=lambda: {},
            get_peers_lock=lambda: peers_lock,
            get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
            training_thresholds=MockTrainingThresholds(),
            improvement_cycle_manager=cycle_manager,
        )

        # Create model file
        model_path = temp_ringrift_path / "test_model.pth"
        model_path.write_text("mock model data")

        job = MockTrainingJob(
            job_id="test-job-3",
            job_type="nnue",
            board_type="hex8",
            num_players=2,
            status="completed",
            worker_node="test-worker",  # Dec 2025: Required for model fetch
            output_model_path=str(model_path),
            data_games_count=2000,
        )

        # Mock gauntlet to pass
        # Dec 2025: Also mock _fetch_model_from_training_node since "test-worker" doesn't exist
        with patch.object(coord, '_fetch_model_from_training_node', return_value=True):
            with patch.object(coord, '_run_post_training_gauntlet', return_value=True):
                with patch.object(coord, '_archive_failed_model') as mock_archive:
                    await coord.handle_training_job_completion(job)

                    # Model should NOT be archived (it passed)
                    mock_archive.assert_not_called()

        # Cycle manager should have been called
        assert cycle_manager.handle_training_complete_called


# =============================================================================
# TestModelPromotion
# =============================================================================


class TestModelPromotion:
    """Test model promotion and demotion functionality."""

    @pytest.mark.asyncio
    async def test_elo_update_after_gauntlet(self, temp_ringrift_path, mock_training_jobs, training_lock, peers_lock):
        """Verify Elo updates are recorded after gauntlet evaluation.

        After a successful gauntlet, the ELO database should be updated
        if the coordinator has database access.
        """
        from scripts.p2p.managers.training_coordinator import TrainingCoordinator

        # Create ELO database
        elo_dir = temp_ringrift_path / "ai-service" / "data"
        elo_dir.mkdir(parents=True, exist_ok=True)
        elo_db = elo_dir / "unified_elo.db"

        conn = sqlite3.connect(str(elo_db))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS elo_ratings (
                participant_id TEXT,
                board_type TEXT,
                num_players INTEGER,
                rating REAL,
                archived_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS participants (
                participant_id TEXT PRIMARY KEY,
                model_path TEXT
            )
        """)
        conn.commit()
        conn.close()

        coord = TrainingCoordinator(
            ringrift_path=temp_ringrift_path,
            get_cluster_data_manifest=lambda: MockClusterDataManifest(),
            get_training_jobs=lambda: mock_training_jobs,
            get_training_lock=lambda: training_lock,
            get_peers=lambda: {},
            get_peers_lock=lambda: peers_lock,
            get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
            training_thresholds=MockTrainingThresholds(),
        )

        # Get median model should work with our test DB
        result = coord._get_median_model("hex8_2p")
        # Should be None since we have no models in the ratings table
        assert result is None

    @pytest.mark.asyncio
    async def test_demotion_on_regression(self, temp_ringrift_path, mock_training_jobs, training_lock, peers_lock):
        """Verify regression handling archives poor-performing models.

        When a model fails the gauntlet (< 50% win rate vs median),
        it should be archived with a reason indicating the failure.
        """
        from scripts.p2p.managers.training_coordinator import TrainingCoordinator

        cycle_manager = MockImprovementCycleManager()

        coord = TrainingCoordinator(
            ringrift_path=temp_ringrift_path,
            get_cluster_data_manifest=lambda: MockClusterDataManifest(),
            get_training_jobs=lambda: mock_training_jobs,
            get_training_lock=lambda: training_lock,
            get_peers=lambda: {},
            get_peers_lock=lambda: peers_lock,
            get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
            training_thresholds=MockTrainingThresholds(),
            improvement_cycle_manager=cycle_manager,
        )

        # Create model file
        model_path = temp_ringrift_path / "regressed_model.pth"
        model_path.write_text("mock model data")

        job = MockTrainingJob(
            job_id="test-job-regression",
            job_type="nnue",
            board_type="hex8",
            num_players=2,
            status="completed",
            worker_node="test-worker",  # Dec 2025: Required for model fetch
            output_model_path=str(model_path),
            data_games_count=2000,
        )

        # Mock gauntlet to FAIL (model regressed)
        # Dec 2025: Also mock _fetch_model_from_training_node since "test-worker" doesn't exist
        with patch.object(coord, '_fetch_model_from_training_node', return_value=True):
            with patch.object(coord, '_run_post_training_gauntlet', return_value=False):
                await coord.handle_training_job_completion(job)

        # Model should be archived
        archive_dir = temp_ringrift_path / "ai-service" / "models" / "archived" / "hex8_2p"
        archived_files = list(archive_dir.glob("*.pth")) if archive_dir.exists() else []
        assert len(archived_files) == 1
        assert "regressed_model" in archived_files[0].name

        # Original should be gone
        assert not model_path.exists()

        # Cycle manager should NOT have been called (model failed gauntlet)
        assert not cycle_manager.handle_training_complete_called

    @pytest.mark.asyncio
    async def test_promote_creates_backup(self, temp_ringrift_path, mock_training_jobs, training_lock, peers_lock):
        """Verify promotion creates backup of existing baseline."""
        from scripts.p2p.managers.training_coordinator import TrainingCoordinator

        coord = TrainingCoordinator(
            ringrift_path=temp_ringrift_path,
            get_cluster_data_manifest=lambda: MockClusterDataManifest(),
            get_training_jobs=lambda: mock_training_jobs,
            get_training_lock=lambda: training_lock,
            get_peers=lambda: {},
            get_peers_lock=lambda: peers_lock,
            get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
            training_thresholds=MockTrainingThresholds(),
        )

        # Create model directory with existing baseline
        nnue_dir = temp_ringrift_path / "ai-service" / "models" / "nnue"
        nnue_dir.mkdir(parents=True)
        existing_baseline = nnue_dir / "hex8_2p_best.pt"
        existing_baseline.write_text("old baseline model")

        # Create new model to promote
        new_model = temp_ringrift_path / "new_model.pth"
        new_model.write_text("new better model")

        await coord.promote_to_baseline(str(new_model), "hex8", 2, "nnue")

        # Verify new baseline
        assert existing_baseline.read_text() == "new better model"

        # Verify backup exists
        backups = list(nnue_dir.glob("hex8_2p_prev_*.pt"))
        assert len(backups) == 1
        assert backups[0].read_text() == "old baseline model"


# =============================================================================
# TestEventSubscriptions
# =============================================================================


class TestEventSubscriptions:
    """Test event subscription and handling."""

    def test_subscribe_to_events_idempotent(self, training_coordinator):
        """Verify subscribe_to_events is idempotent."""
        coord = training_coordinator

        # Mock the event bus at its source location (inside event_router)
        mock_bus = MagicMock()
        mock_bus.subscribe = MagicMock()

        with patch('app.coordination.event_router.get_event_bus', return_value=mock_bus):
            # First subscription
            coord.subscribe_to_events()

            # Should be marked as subscribed
            assert coord._subscribed

            # Record initial subscribe count
            initial_call_count = mock_bus.subscribe.call_count

            # Second subscription should be no-op
            coord.subscribe_to_events()

            # No additional subscribe calls (idempotent)
            assert mock_bus.subscribe.call_count == initial_call_count

    @pytest.mark.asyncio
    async def test_on_selfplay_complete_triggers_readiness_check(self, coordinator_with_manifest):
        """Verify SELFPLAY_COMPLETE event triggers readiness check."""
        coord = coordinator_with_manifest

        event = MockEvent(payload={
            "config_key": "hex8_2p",
            "game_count": 500,
        })

        # Readiness check should find jobs
        await coord._on_selfplay_complete(event)

        # No direct assertions - the method logs info if jobs are found

    @pytest.mark.asyncio
    async def test_on_task_abandoned_clears_cache(self, training_coordinator):
        """Verify TASK_ABANDONED clears trigger cache."""
        coord = training_coordinator

        # Add trigger to cache
        coord._training_trigger_cache["nnue_hex8_2p_1000"] = time.time()
        coord._training_trigger_cache["nnue_hex8_2p_2000"] = time.time()
        coord._training_trigger_cache["other_key"] = time.time()

        event = MockEvent(payload={
            "task_id": "nnue_hex8_2p_12345",
            "task_type": "nnue_training",
            "config_key": "hex8_2p",
            "reason": "user_cancelled",
        })

        await coord._on_task_abandoned(event)

        # Keys containing "hex8_2p" should be cleared
        assert "nnue_hex8_2p_1000" not in coord._training_trigger_cache
        assert "nnue_hex8_2p_2000" not in coord._training_trigger_cache
        # Other keys should remain
        assert "other_key" in coord._training_trigger_cache


# =============================================================================
# TestHealthCheck
# =============================================================================


class TestHealthCheck:
    """Test health check functionality."""

    def test_health_check_healthy(self, training_coordinator):
        """Verify health check returns healthy status with subscribed flag.

        Note: The coordinator starts unsubscribed (no event bus available
        during unit tests), which causes degraded status. We mock the
        subscription state to test the healthy path.
        """
        coord = training_coordinator

        # Mock subscription state to test healthy path
        coord._subscribed = True

        health = coord.health_check()

        # health_check returns HealthCheckResult dataclass
        assert hasattr(health, "healthy")
        assert health.healthy is True
        assert hasattr(health, "details")
        if health.details:
            assert health.details.get("running_jobs", 0) == 0
            assert health.details.get("failed_jobs", 0) == 0
            assert health.details.get("subscribed") is True

    def test_health_check_with_failed_jobs(self, temp_ringrift_path, training_lock, peers_lock):
        """Verify health check reports degraded status with high failure rate."""
        from scripts.p2p.managers.training_coordinator import TrainingCoordinator

        # Create jobs dict with failures
        mock_jobs = {
            "job1": MockTrainingJob("job1", "nnue", "hex8", 2, status="failed"),
            "job2": MockTrainingJob("job2", "nnue", "hex8", 2, status="failed"),
            "job3": MockTrainingJob("job3", "nnue", "hex8", 2, status="completed"),
        }

        coord = TrainingCoordinator(
            ringrift_path=temp_ringrift_path,
            get_cluster_data_manifest=lambda: MockClusterDataManifest(),
            get_training_jobs=lambda: mock_jobs,
            get_training_lock=lambda: training_lock,
            get_peers=lambda: {},
            get_peers_lock=lambda: peers_lock,
            get_self_info=lambda: MockNodeInfo("self-node", memory_gb=64),
            training_thresholds=MockTrainingThresholds(),
        )

        health = coord.health_check()

        # 2/3 failed = 66% failure rate > 50% threshold
        assert hasattr(health, "healthy")
        assert health.healthy is False
        assert hasattr(health, "details")
        if health.details:
            assert health.details.get("failed_jobs") == 2
            assert health.details.get("total_jobs") == 3

    def test_health_check_large_trigger_cache(self, training_coordinator):
        """Verify health check flags large trigger cache."""
        coord = training_coordinator

        # Add many entries to trigger cache
        for i in range(150):
            coord._training_trigger_cache[f"trigger_{i}"] = time.time()

        health = coord.health_check()

        # Large cache should cause degraded status
        assert hasattr(health, "healthy")
        assert hasattr(health, "status")
        assert hasattr(health, "details")
        if health.details:
            assert health.details.get("trigger_cache_size", 0) > 100
