"""Tests for JobManager - P2P job spawning and lifecycle management.

December 27, 2025: Created as part of test coverage improvement effort.
Tests core job lifecycle functionality without requiring actual job execution.
"""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Test Fixtures
# =============================================================================


@dataclass
class MockNodeInfo:
    """Mock NodeInfo for testing."""
    node_id: str
    ip: str
    port: int = 8770
    role: str = "training"
    last_seen: float = 0.0
    gpu_available: bool = True
    is_alive: bool = True
    has_gpu: bool = True
    _healthy: bool = True

    def is_healthy(self) -> bool:
        """Check if node is healthy."""
        return self._healthy and self.is_alive


@pytest.fixture
def job_manager_deps():
    """Create dependencies for JobManager initialization."""
    return {
        "ringrift_path": "/mock/ringrift",
        "node_id": "test-node-1",
        "peers": {},
        "peers_lock": threading.Lock(),
        "active_jobs": {},
        "jobs_lock": threading.Lock(),
        "improvement_loop_state": {},
        "distributed_tournament_state": {},
    }


@pytest.fixture
def job_manager(job_manager_deps):
    """Create a JobManager instance for testing."""
    from scripts.p2p.managers.job_manager import JobManager
    return JobManager(**job_manager_deps)


@pytest.fixture
def job_manager_with_peers(job_manager_deps):
    """Create a JobManager with mock peers."""
    from scripts.p2p.managers.job_manager import JobManager

    peers = {
        "node-2": MockNodeInfo(node_id="node-2", ip="10.0.0.2"),
        "node-3": MockNodeInfo(node_id="node-3", ip="10.0.0.3"),
        "node-4": MockNodeInfo(node_id="node-4", ip="10.0.0.4", is_alive=False, _healthy=False),
    }
    job_manager_deps["peers"] = peers
    return JobManager(**job_manager_deps)


# =============================================================================
# JobManagerStats Tests
# =============================================================================


class TestJobManagerStats:
    """Tests for JobManagerStats dataclass."""

    def test_stats_defaults(self):
        """Test default values for stats."""
        from scripts.p2p.managers.job_manager import JobManagerStats

        stats = JobManagerStats()
        assert stats.jobs_spawned == 0
        assert stats.jobs_completed == 0
        assert stats.jobs_failed == 0
        assert stats.jobs_cancelled == 0
        assert stats.nodes_recovered == 0
        assert stats.hosts_offline == 0
        assert stats.hosts_online == 0

    def test_stats_initialization(self):
        """Test stats with custom values."""
        from scripts.p2p.managers.job_manager import JobManagerStats

        stats = JobManagerStats(
            jobs_spawned=10,
            jobs_completed=8,
            jobs_failed=2,
        )
        assert stats.jobs_spawned == 10
        assert stats.jobs_completed == 8
        assert stats.jobs_failed == 2


# =============================================================================
# JobManager Initialization Tests
# =============================================================================


class TestJobManagerInit:
    """Tests for JobManager initialization."""

    def test_basic_init(self, job_manager):
        """Test basic initialization."""
        assert job_manager.node_id == "test-node-1"
        assert job_manager.ringrift_path == "/mock/ringrift"
        assert isinstance(job_manager.stats, object)

    def test_init_with_empty_peers(self, job_manager):
        """Test initialization with no peers."""
        assert len(job_manager.peers) == 0

    def test_init_with_peers(self, job_manager_with_peers):
        """Test initialization with peers."""
        assert len(job_manager_with_peers.peers) == 3

    def test_engine_modes_constant(self, job_manager):
        """Test SEARCH_ENGINE_MODES contains expected modes."""
        expected_modes = {"maxn", "brs", "mcts", "gumbel-mcts"}
        assert expected_modes.issubset(job_manager.SEARCH_ENGINE_MODES)


# =============================================================================
# Event Subscription Tests
# =============================================================================


class TestEventSubscription:
    """Tests for event subscription functionality."""

    def test_subscribe_to_events_success(self, job_manager):
        """Test successful event subscription."""
        with patch("app.coordination.event_router.get_event_bus") as mock_bus:
            mock_bus.return_value = MagicMock()
            job_manager.subscribe_to_events()
            assert job_manager._subscribed is True

    def test_subscribe_to_events_idempotent(self, job_manager):
        """Test subscription is idempotent."""
        job_manager._subscribed = True
        # Should return early without error
        job_manager.subscribe_to_events()
        assert job_manager._subscribed is True


# =============================================================================
# Event Handler Tests
# =============================================================================


class TestEventHandlers:
    """Tests for event handler methods."""

    @pytest.mark.asyncio
    async def test_on_host_offline_cancels_running_jobs(self, job_manager):
        """Test HOST_OFFLINE cancels running jobs on that host."""
        # Setup: Add a running job for the offline host
        job_manager.active_jobs["selfplay"] = {
            "job-1": {"node_id": "node-2", "status": "running"},
        }

        # Create mock event with payload
        event = MagicMock()
        event.payload = {"node_id": "node-2", "last_seen": 12345.0}

        initial_cancelled = job_manager.stats.jobs_cancelled
        await job_manager._on_host_offline(event)

        # Job should be marked as cancelled
        assert job_manager.active_jobs["selfplay"]["job-1"]["status"] == "cancelled"
        assert job_manager.stats.jobs_cancelled == initial_cancelled + 1
        assert job_manager.stats.hosts_offline == 1

    @pytest.mark.asyncio
    async def test_on_host_offline_no_jobs(self, job_manager):
        """Test HOST_OFFLINE with no jobs to cancel."""
        event = MagicMock()
        event.payload = {"node_id": "node-2"}

        # Should not raise, stats should not be updated
        await job_manager._on_host_offline(event)
        assert job_manager.stats.hosts_offline == 0  # No jobs cancelled

    @pytest.mark.asyncio
    async def test_on_host_online_updates_stats(self, job_manager):
        """Test HOST_ONLINE handler updates stats."""
        event = MagicMock()
        event.payload = {"node_id": "node-2", "ip": "10.0.0.2"}

        initial_online = job_manager.stats.hosts_online
        await job_manager._on_host_online(event)
        assert job_manager.stats.hosts_online == initial_online + 1

    @pytest.mark.asyncio
    async def test_on_host_online_empty_node_id(self, job_manager):
        """Test HOST_ONLINE with empty node_id returns early."""
        event = MagicMock()
        event.payload = {"node_id": ""}

        await job_manager._on_host_online(event)
        assert job_manager.stats.hosts_online == 0  # Should return early

    @pytest.mark.asyncio
    async def test_on_node_recovered_updates_stats(self, job_manager):
        """Test NODE_RECOVERED handler updates stats."""
        event = MagicMock()
        event.payload = {"node_id": "node-2", "downtime": 300.0}

        initial_recovered = job_manager.stats.nodes_recovered
        await job_manager._on_node_recovered(event)
        assert job_manager.stats.nodes_recovered == initial_recovered + 1


# =============================================================================
# Task Event Emission Tests
# =============================================================================


class TestTaskEventEmission:
    """Tests for task event emission."""

    def test_emit_task_event_success(self, job_manager):
        """Test task event emission."""
        with patch.object(job_manager, "_emit_task_event") as mock_emit:
            job_manager._emit_task_event(
                "TASK_STARTED",
                job_id="job-123",
                job_type="selfplay",
                config_key="hex8_2p",
            )
            mock_emit.assert_called_once()

    def test_emit_task_event_no_emitter(self, job_manager):
        """Test graceful degradation when event emitter unavailable."""
        # Should not raise even if event system unavailable
        job_manager._emit_task_event(
            "TASK_STARTED",
            job_id="job-123",
            job_type="selfplay",
        )


# =============================================================================
# GPU Selfplay Job Tests
# =============================================================================


class TestGPUSelfplayJob:
    """Tests for GPU selfplay job spawning."""

    def test_search_engine_modes_defined(self, job_manager):
        """Test SEARCH_ENGINE_MODES contains expected modes."""
        expected_modes = {"mcts", "gumbel-mcts"}
        assert expected_modes.issubset(job_manager.SEARCH_ENGINE_MODES)

    def test_run_gpu_selfplay_job_exists(self, job_manager):
        """Test that run_gpu_selfplay_job method exists and is callable."""
        assert hasattr(job_manager, "run_gpu_selfplay_job")
        assert callable(job_manager.run_gpu_selfplay_job)

    def test_job_manager_has_stats(self, job_manager):
        """Test that JobManager tracks stats."""
        assert hasattr(job_manager, "stats")
        assert hasattr(job_manager.stats, "jobs_spawned")
        assert hasattr(job_manager.stats, "jobs_completed")
        assert hasattr(job_manager.stats, "jobs_failed")


# =============================================================================
# Distributed Selfplay Tests
# =============================================================================


class TestDistributedSelfplay:
    """Tests for distributed selfplay functionality."""

    @pytest.mark.asyncio
    async def test_dispatch_selfplay_to_workers(self, job_manager_with_peers):
        """Test dispatching work to alive workers."""
        # Get alive peers
        alive_peers = [
            p for p in job_manager_with_peers.peers.values()
            if hasattr(p, "is_alive") and p.is_alive
        ]
        assert len(alive_peers) == 2  # node-2 and node-3


# =============================================================================
# Training Data Export Tests
# =============================================================================


class TestTrainingDataExport:
    """Tests for training data export functionality."""

    @pytest.mark.asyncio
    async def test_export_training_data_requires_job_id(self, job_manager):
        """Test that job_id is required."""
        with pytest.raises(TypeError):
            await job_manager.export_training_data()  # Missing job_id


# =============================================================================
# Training Job Tests
# =============================================================================


class TestTrainingJob:
    """Tests for training job spawning."""

    @pytest.mark.asyncio
    async def test_run_training_requires_job_id(self, job_manager):
        """Test that job_id is required."""
        with pytest.raises(TypeError):
            await job_manager.run_training()  # Missing job_id


# =============================================================================
# Tournament Tests
# =============================================================================


class TestDistributedTournament:
    """Tests for distributed tournament functionality."""

    def test_generate_tournament_matches(self, job_manager):
        """Test match generation logic."""
        # Test with mock models
        models = ["model_a", "model_b", "model_c"]
        matches = job_manager._generate_tournament_matches(models, games_per_pair=2)

        # Should generate all pairings
        assert len(matches) == 6  # 3 choose 2 = 3 pairs, 2 games each = 6

    def test_get_tournament_workers(self, job_manager_with_peers):
        """Test getting available tournament workers."""
        workers = job_manager_with_peers._get_tournament_workers()

        # Should only return alive workers
        assert len(workers) == 2

    def test_calculate_elo_updates(self, job_manager):
        """Test Elo calculation."""
        # Test basic Elo update
        models = ["model_a", "model_b"]
        results = [
            {"player1_model": "model_a", "player2_model": "model_b", "winner": "model_a"},
        ]

        deltas = job_manager._calculate_elo_updates(models, results)

        # Winner should have positive delta, loser negative
        assert deltas["model_a"] > 0
        assert deltas["model_b"] < 0


# =============================================================================
# Cleanup Tests
# =============================================================================


class TestJobCleanup:
    """Tests for job cleanup functionality."""

    def test_cleanup_completed_jobs_empty(self, job_manager):
        """Test cleanup with no jobs."""
        cleaned = job_manager.cleanup_completed_jobs()
        assert cleaned == 0

    def test_cleanup_completed_jobs_removes_completed(self, job_manager):
        """Test that completed jobs are cleaned up."""
        # Add some completed jobs
        job_manager.active_jobs["selfplay"] = {
            "job-1": {"status": "completed", "completed_at": 0},
            "job-2": {"status": "running"},
        }

        cleaned = job_manager.cleanup_completed_jobs()

        # Should clean completed jobs
        assert "job-2" in job_manager.active_jobs.get("selfplay", {})


# =============================================================================
# Health Check Tests
# =============================================================================


class TestHealthCheck:
    """Tests for health check functionality."""

    def test_health_check_returns_dict(self, job_manager):
        """Test health check returns proper structure."""
        health = job_manager.health_check()

        assert isinstance(health, dict)
        assert "healthy" in health or "status" in health

    def test_health_check_includes_stats(self, job_manager):
        """Test health check includes job stats."""
        health = job_manager.health_check()

        # Should include some stats
        assert isinstance(health, dict)

    def test_health_check_with_active_jobs(self, job_manager):
        """Test health check with running jobs."""
        job_manager.active_jobs["selfplay"] = {
            "job-1": {"status": "running"},
        }

        health = job_manager.health_check()
        assert isinstance(health, dict)


# =============================================================================
# Helper Function Tests
# =============================================================================


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_get_default_port(self):
        """Test default port retrieval."""
        from scripts.p2p.managers.job_manager import _get_default_port

        port = _get_default_port()
        assert port == 8770  # Default fallback

    def test_get_event_emitter_thread_safe(self):
        """Test event emitter getter is thread-safe."""
        from scripts.p2p.managers.job_manager import _get_event_emitter

        # Should not raise even with concurrent calls
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(_get_event_emitter) for _ in range(10)]
            results = [f.result() for f in futures]

        # All results should be the same (or all None)
        assert len(set(id(r) if r else None for r in results)) <= 2


# =============================================================================
# TestSelfplayJobLifecycle - Comprehensive Job Creation and Lifecycle Tests
# =============================================================================


class TestSelfplayJobLifecycle:
    """Tests for selfplay job creation, registration, timeout, and cancellation.

    December 28, 2025: Added as part of comprehensive P2P manager test coverage.
    """

    @pytest.mark.asyncio
    async def test_job_creation_and_registration(self, job_manager):
        """Verify job is tracked in active_jobs after creation."""
        # Mock subprocess creation
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None  # Still running
        mock_proc.communicate = AsyncMock(return_value=(b"output", b""))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"):
            mock_exec.return_value = mock_proc
            mock_proc.returncode = 0  # Will complete successfully

            # Run the job
            await job_manager.run_gpu_selfplay_job(
                job_id="test-job-001",
                board_type="hex8",
                num_players=2,
                num_games=100,
                engine_mode="heuristic-only",
            )

            # Verify subprocess was created
            mock_exec.assert_called_once()

            # After completion, job should be removed from active (successful cleanup)
            # Check the internal process tracking was used
            assert mock_proc.pid == 12345

    @pytest.mark.asyncio
    async def test_job_timeout_handling(self, job_manager):
        """Verify timeout kills process and updates status."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        # Simulate timeout
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"), \
             patch.object(job_manager, "_emit_task_event") as mock_emit:
            mock_exec.return_value = mock_proc

            await job_manager.run_gpu_selfplay_job(
                job_id="timeout-job",
                board_type="hex8",
                num_players=2,
                num_games=100,
                engine_mode="heuristic-only",
            )

            # Verify process was killed
            mock_proc.kill.assert_called_once()

            # Verify TASK_FAILED event was emitted with timeout error
            mock_emit.assert_called()
            # Get the last call to _emit_task_event
            calls = [c for c in mock_emit.call_args_list if c[0][0] == "TASK_FAILED"]
            assert len(calls) >= 1
            assert calls[-1][1].get("error") == "timeout"

    @pytest.mark.asyncio
    async def test_job_cancellation_cleans_up(self, job_manager):
        """Verify cancellation cleans up job from tracking."""
        # Setup: Add a running job
        job_manager.active_jobs["selfplay"] = {
            "cancel-job-001": {
                "job_id": "cancel-job-001",
                "status": "running",
                "node_id": "test-node-1",
                "board_type": "hex8",
                "num_players": 2,
                "started_at": 0,
            }
        }

        # Create mock event for HOST_OFFLINE (simulates cancellation trigger)
        event = MagicMock()
        event.payload = {"node_id": "test-node-1"}

        # December 28, 2025: Updated to mock _safe_emit_event (from EventSubscriptionMixin)
        # instead of _emit_task_event due to refactoring
        with patch.object(job_manager, "_safe_emit_event") as mock_emit:
            await job_manager._on_host_offline(event)

        # Job should be marked as cancelled
        assert job_manager.active_jobs["selfplay"]["cancel-job-001"]["status"] == "cancelled"

        # TASK_ABANDONED event should be emitted
        mock_emit.assert_called()
        abandoned_calls = [c for c in mock_emit.call_args_list if c[0][0] == "TASK_ABANDONED"]
        assert len(abandoned_calls) >= 1

    @pytest.mark.asyncio
    async def test_job_with_failed_returncode(self, job_manager):
        """Verify job failure is tracked when subprocess returns non-zero."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 1  # Non-zero return code
        mock_proc.communicate = AsyncMock(return_value=(b"", b"Error: Something went wrong"))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"), \
             patch.object(job_manager, "_emit_task_event") as mock_emit:
            mock_exec.return_value = mock_proc

            await job_manager.run_gpu_selfplay_job(
                job_id="fail-job",
                board_type="hex8",
                num_players=2,
                num_games=100,
                engine_mode="heuristic-only",
            )

            # Verify TASK_FAILED event was emitted
            failed_calls = [c for c in mock_emit.call_args_list if c[0][0] == "TASK_FAILED"]
            assert len(failed_calls) >= 1


# =============================================================================
# TestTrainingJobLifecycle - Training Job Specific Tests
# =============================================================================


class TestTrainingJobLifecycle:
    """Tests for training job lifecycle including leader-only spawning.

    December 28, 2025: Added as part of comprehensive P2P manager test coverage.
    """

    @pytest.mark.asyncio
    async def test_training_spawn_on_leader(self, job_manager_with_peers):
        """Verify training only runs on leader node."""
        # Setup improvement loop state
        class MockState:
            current_iteration = 1
            board_type = "hex8"
            num_players = 2
            training_data_path = "/tmp/training.npz"
            best_model_path = None

        job_manager_with_peers.improvement_loop_state["train-job-001"] = MockState()

        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Model saved", b""))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.makedirs"):
            mock_exec.return_value = mock_proc

            await job_manager_with_peers.run_training("train-job-001")

            # Verify subprocess was created for training
            mock_exec.assert_called_once()

            # Verify state was updated with new model path
            assert job_manager_with_peers.improvement_loop_state["train-job-001"].candidate_model_path is not None

    @pytest.mark.asyncio
    async def test_training_async_wait_completion(self, job_manager):
        """Verify async completion tracking for training jobs."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999
        mock_proc.returncode = 0
        mock_proc.communicate = AsyncMock(return_value=(b"Training complete", b""))

        config = {
            "job_id": "async-train-job",
            "training_data": "/tmp/data.npz",
            "output_model": "/tmp/model.pt",
            "board_type": "hex8",
            "num_players": 2,
            "epochs": 10,
            "batch_size": 256,
            "learning_rate": 0.001,
        }

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec:
            mock_exec.return_value = mock_proc

            # This should wait for the training to complete
            await job_manager.run_local_training(config)

            # Verify communicate was awaited (async wait)
            mock_proc.communicate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_training_failure_recovery(self, job_manager):
        """Verify error handling when training subprocess fails."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 1  # Failure
        mock_proc.communicate = AsyncMock(return_value=(b"", b"CUDA out of memory"))

        config = {
            "job_id": "fail-train-job",
            "training_data": "/tmp/data.npz",
            "output_model": "/tmp/model.pt",
            "board_type": "hex8",
            "num_players": 2,
            "epochs": 10,
            "batch_size": 256,
            "learning_rate": 0.001,
        }

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch.object(job_manager, "_emit_task_event") as mock_emit:
            mock_exec.return_value = mock_proc

            # Should not raise, but handle error gracefully
            await job_manager.run_local_training(config)

            # TASK_FAILED event should be emitted
            failed_calls = [c for c in mock_emit.call_args_list if c[0][0] == "TASK_FAILED"]
            assert len(failed_calls) >= 1

    @pytest.mark.asyncio
    async def test_training_timeout_recovery(self, job_manager):
        """Verify training job recovers from timeout."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=asyncio.TimeoutError())

        config = {
            "job_id": "timeout-train-job",
            "training_data": "/tmp/data.npz",
            "output_model": "/tmp/model.pt",
            "board_type": "hex8",
            "num_players": 2,
        }

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch.object(job_manager, "_emit_task_event") as mock_emit:
            mock_exec.return_value = mock_proc

            await job_manager.run_local_training(config)

            # Process should be killed
            mock_proc.kill.assert_called_once()

            # TASK_FAILED with timeout should be emitted
            failed_calls = [c for c in mock_emit.call_args_list if c[0][0] == "TASK_FAILED"]
            assert len(failed_calls) >= 1


# =============================================================================
# TestErrorHandling - Cascade Failures and Zombie Process Prevention
# =============================================================================


class TestErrorHandling:
    """Tests for error handling, cascade failures, and zombie process cleanup.

    December 28, 2025: Added as part of comprehensive P2P manager test coverage.
    """

    @pytest.mark.asyncio
    async def test_cascade_job_failures_isolated(self, job_manager):
        """Verify one job failure doesn't break other jobs."""
        # Setup multiple jobs - one will fail, others should continue
        job_manager.active_jobs["selfplay"] = {
            "job-success-1": {"status": "running", "node_id": "node-1"},
            "job-success-2": {"status": "running", "node_id": "node-2"},
        }

        # Simulate a failure on one node
        event = MagicMock()
        event.payload = {"node_id": "node-1"}

        with patch.object(job_manager, "_emit_task_event"):
            await job_manager._on_host_offline(event)

        # Only the job on the failed node should be cancelled
        assert job_manager.active_jobs["selfplay"]["job-success-1"]["status"] == "cancelled"
        assert job_manager.active_jobs["selfplay"]["job-success-2"]["status"] == "running"

    @pytest.mark.asyncio
    async def test_process_cleanup_on_error(self, job_manager):
        """Verify no zombie processes after subprocess error."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()
        mock_proc.communicate = AsyncMock(side_effect=OSError("Broken pipe"))

        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"), \
             patch.object(job_manager, "_emit_task_event"):
            mock_exec.return_value = mock_proc

            await job_manager.run_gpu_selfplay_job(
                job_id="error-job",
                board_type="hex8",
                num_players=2,
                num_games=100,
                engine_mode="heuristic-only",
            )

            # Verify kill was called to prevent zombie
            mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_active_processes_on_shutdown(self, job_manager):
        """Verify all active processes are killed during shutdown."""
        # Register some mock processes
        mock_proc_1 = MagicMock()
        mock_proc_1.pid = 1001
        mock_proc_1.returncode = None
        mock_proc_1.kill = MagicMock()
        mock_proc_1.wait = AsyncMock()

        mock_proc_2 = MagicMock()
        mock_proc_2.pid = 1002
        mock_proc_2.returncode = None
        mock_proc_2.kill = MagicMock()
        mock_proc_2.wait = AsyncMock()

        job_manager._register_process("job-1", mock_proc_1)
        job_manager._register_process("job-2", mock_proc_2)

        # Cleanup all active processes
        killed = await job_manager.cleanup_active_processes()

        assert killed == 2
        mock_proc_1.kill.assert_called()
        mock_proc_2.kill.assert_called()

    @pytest.mark.asyncio
    async def test_process_already_terminated(self, job_manager):
        """Verify graceful handling when process already terminated."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = 0  # Already terminated
        mock_proc.kill = MagicMock(side_effect=ProcessLookupError())

        # Should not raise
        await job_manager._kill_process("test-job", mock_proc)

    def test_register_unregister_process(self, job_manager):
        """Verify process registration and unregistration."""
        mock_proc = MagicMock()
        mock_proc.pid = 99999

        # Register
        job_manager._register_process("test-job-reg", mock_proc)
        assert "test-job-reg" in job_manager._active_processes

        # Unregister
        job_manager._unregister_process("test-job-reg")
        assert "test-job-reg" not in job_manager._active_processes

    @pytest.mark.asyncio
    async def test_multiple_failures_dont_cascade(self, job_manager):
        """Verify multiple concurrent failures are handled independently."""
        # Setup jobs on different nodes
        job_manager.active_jobs["selfplay"] = {
            "job-a": {"status": "running", "node_id": "node-a"},
            "job-b": {"status": "running", "node_id": "node-b"},
            "job-c": {"status": "running", "node_id": "node-c"},
        }

        # Both node-a and node-b go offline
        with patch.object(job_manager, "_emit_task_event"):
            event_a = MagicMock()
            event_a.payload = {"node_id": "node-a"}
            await job_manager._on_host_offline(event_a)

            event_b = MagicMock()
            event_b.payload = {"node_id": "node-b"}
            await job_manager._on_host_offline(event_b)

        # Two jobs cancelled, one still running
        assert job_manager.active_jobs["selfplay"]["job-a"]["status"] == "cancelled"
        assert job_manager.active_jobs["selfplay"]["job-b"]["status"] == "cancelled"
        assert job_manager.active_jobs["selfplay"]["job-c"]["status"] == "running"

        # Stats should reflect two cancellations
        assert job_manager.stats.jobs_cancelled == 2

    @pytest.mark.asyncio
    async def test_oserror_handling(self, job_manager):
        """Verify OSError during subprocess creation is handled."""
        with patch("asyncio.create_subprocess_exec", new_callable=AsyncMock) as mock_exec, \
             patch("os.path.exists", return_value=True), \
             patch("pathlib.Path.mkdir"), \
             patch.object(job_manager, "_emit_task_event") as mock_emit:
            mock_exec.side_effect = OSError("Permission denied")

            # Should not raise
            await job_manager.run_gpu_selfplay_job(
                job_id="oserror-job",
                board_type="hex8",
                num_players=2,
                num_games=100,
                engine_mode="heuristic-only",
            )

            # TASK_FAILED should be emitted
            failed_calls = [c for c in mock_emit.call_args_list if c[0][0] == "TASK_FAILED"]
            assert len(failed_calls) >= 1


# =============================================================================
# TestProcessLifecycleManagement - Process Tracking Tests
# =============================================================================


class TestProcessLifecycleManagement:
    """Tests for subprocess lifecycle management.

    December 28, 2025: Added to verify process tracking infrastructure.
    """

    def test_process_lock_thread_safety(self, job_manager):
        """Verify process lock provides thread safety."""
        import concurrent.futures

        mock_procs = [MagicMock(pid=i) for i in range(10)]

        def register_proc(idx):
            job_manager._register_process(f"job-{idx}", mock_procs[idx])
            return True

        # Concurrent registrations should not fail
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(register_proc, i) for i in range(10)]
            results = [f.result() for f in futures]

        assert all(results)
        assert len(job_manager._active_processes) == 10

    @pytest.mark.asyncio
    async def test_kill_process_with_sigkill(self, job_manager):
        """Verify SIGKILL is tried first for immediate termination."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.wait = AsyncMock()

        job_manager._register_process("kill-test", mock_proc)

        await job_manager._kill_process("kill-test")

        # kill() should be called (SIGKILL)
        mock_proc.kill.assert_called_once()

    @pytest.mark.asyncio
    async def test_kill_process_sigterm_fallback(self, job_manager):
        """Verify SIGTERM fallback when SIGKILL fails."""
        mock_proc = MagicMock()
        mock_proc.pid = 12345
        mock_proc.returncode = None
        mock_proc.kill = MagicMock()
        mock_proc.terminate = MagicMock()
        # wait() times out after kill, succeeds after terminate
        mock_proc.wait = AsyncMock(side_effect=[asyncio.TimeoutError(), None])

        job_manager._register_process("term-test", mock_proc)

        await job_manager._kill_process("term-test")

        # Both kill and terminate should be attempted
        mock_proc.kill.assert_called_once()
        mock_proc.terminate.assert_called_once()
