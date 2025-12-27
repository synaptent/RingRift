"""Unit tests for JobManager.

Tests job spawning, lifecycle management, and metrics functionality.
"""

import asyncio
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from scripts.p2p.managers.job_manager import JobManager


class TestJobManagerInit:
    """Test JobManager initialization."""

    def test_basic_init(self):
        """Test basic initialization with required parameters."""
        mgr = JobManager(
            ringrift_path="/path/to/ringrift",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        assert mgr.ringrift_path == "/path/to/ringrift"
        assert mgr.node_id == "test-node"
        assert mgr.peers == {}
        assert mgr.active_jobs == {}
        assert mgr.improvement_loop_state == {}
        assert mgr.distributed_tournament_state == {}

    def test_init_with_optional_state(self):
        """Test initialization with optional state dicts."""
        improvement_state = {"job1": {"status": "running"}}
        tournament_state = {"tourney1": {"status": "pending"}}

        mgr = JobManager(
            ringrift_path="/path/to/ringrift",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
            improvement_loop_state=improvement_state,
            distributed_tournament_state=tournament_state,
        )

        assert mgr.improvement_loop_state == improvement_state
        assert mgr.distributed_tournament_state == tournament_state


class TestSearchEngineModes:
    """Test engine mode classification."""

    def test_search_engine_modes_constant(self):
        """Test SEARCH_ENGINE_MODES contains expected modes."""
        expected = {"maxn", "brs", "mcts", "gumbel-mcts",
                   "policy-only", "nn-descent", "nn-minimax"}
        assert expected == JobManager.SEARCH_ENGINE_MODES

    def test_heuristic_not_in_search_modes(self):
        """Test heuristic modes are NOT search modes."""
        non_search = {"heuristic-only", "random-only", "nnue-guided", "mixed"}
        for mode in non_search:
            assert mode not in JobManager.SEARCH_ENGINE_MODES


class TestJobManagement:
    """Test job management methods."""
    # NOTE: get_all_jobs tests removed Dec 2025 - method was dead code

    def test_cleanup_completed_jobs(self):
        """Test cleanup of completed jobs."""
        active_jobs = {
            "selfplay": {
                "job1": {"status": "completed"},
                "job2": {"status": "running"},
                "job3": {"status": "failed"},
                "job4": {"status": "timeout"},
                "job5": {"status": "error"},
            }
        }

        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs=active_jobs,
            jobs_lock=threading.Lock(),
        )

        cleaned = mgr.cleanup_completed_jobs()

        # Should have cleaned up 4 jobs (completed, failed, timeout, error)
        assert cleaned == 4
        # Only running job should remain
        assert "job2" in active_jobs["selfplay"]
        assert len(active_jobs["selfplay"]) == 1

    def test_cleanup_empty_jobs(self):
        """Test cleanup when no jobs exist."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        cleaned = mgr.cleanup_completed_jobs()
        assert cleaned == 0


class TestSelfplayJobExecution:
    """Test selfplay job execution."""

    @pytest.mark.asyncio
    async def test_run_gpu_selfplay_script_not_found(self):
        """Test GPU selfplay when script doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(
                ringrift_path=tmpdir,
                node_id="node1",
                peers={},
                peers_lock=threading.Lock(),
                active_jobs={},
                jobs_lock=threading.Lock(),
            )

            # Should return early without error
            await mgr.run_gpu_selfplay_job(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
                num_games=10,
                engine_mode="heuristic-only",
            )

            # No jobs should be tracked
            assert mgr.active_jobs == {}

    @pytest.mark.asyncio
    async def test_run_gpu_selfplay_hexagonal_normalization(self):
        """Test that hexagonal is normalized to hex."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(
                ringrift_path=tmpdir,
                node_id="node1",
                peers={},
                peers_lock=threading.Lock(),
                active_jobs={},
                jobs_lock=threading.Lock(),
            )

            # Create mock script
            ai_service = Path(tmpdir) / "ai-service" / "scripts"
            ai_service.mkdir(parents=True)
            (ai_service / "run_gpu_selfplay.py").write_text("# mock")

            # Run and check output dir uses normalized name
            with patch('asyncio.create_subprocess_exec') as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.pid = 12345
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                await mgr.run_gpu_selfplay_job(
                    job_id="test-job",
                    board_type="hexagonal",
                    num_players=2,
                    num_games=10,
                    engine_mode="heuristic-only",
                )

                # Verify the call was made
                assert mock_exec.called

    @pytest.mark.asyncio
    async def test_run_gpu_selfplay_chooses_hybrid_for_search_modes(self):
        """Test that search engine modes use hybrid selfplay script."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(
                ringrift_path=tmpdir,
                node_id="node1",
                peers={},
                peers_lock=threading.Lock(),
                active_jobs={},
                jobs_lock=threading.Lock(),
            )

            # Create both mock scripts
            ai_service = Path(tmpdir) / "ai-service" / "scripts"
            ai_service.mkdir(parents=True)
            (ai_service / "run_gpu_selfplay.py").write_text("# gpu mock")
            (ai_service / "run_hybrid_selfplay.py").write_text("# hybrid mock")

            # Mock GPU availability to prevent fallback to heuristic-only
            mock_torch = MagicMock()
            mock_torch.cuda.is_available.return_value = True
            with patch('asyncio.create_subprocess_exec') as mock_exec, \
                 patch.dict("sys.modules", {"torch": mock_torch}):

                mock_proc = AsyncMock()
                mock_proc.pid = 12345
                mock_proc.communicate = AsyncMock(return_value=(b"", b""))
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                await mgr.run_gpu_selfplay_job(
                    job_id="test-job",
                    board_type="hex8",
                    num_players=2,
                    num_games=10,
                    engine_mode="gumbel-mcts",  # This is a search mode
                )

                # Check that hybrid script was used
                call_args = mock_exec.call_args[0]
                script_path = str(call_args[1])
                assert "run_hybrid_selfplay.py" in script_path


class TestDistributedSelfplay:
    """Test distributed selfplay coordination."""

    @pytest.mark.asyncio
    async def test_run_distributed_selfplay_no_state(self):
        """Test distributed selfplay with no state returns early."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        # Should return without error
        await mgr.run_distributed_selfplay("nonexistent-job")

    @pytest.mark.asyncio
    async def test_run_local_selfplay_updates_progress(self):
        """Test local selfplay updates progress on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            class MockState:
                selfplay_progress = {}

            improvement_state = {"job1": MockState()}

            mgr = JobManager(
                ringrift_path=tmpdir,
                node_id="node1",
                peers={},
                peers_lock=threading.Lock(),
                active_jobs={},
                jobs_lock=threading.Lock(),
                improvement_loop_state=improvement_state,
            )

            with patch('asyncio.create_subprocess_exec') as mock_exec:
                mock_proc = AsyncMock()
                mock_proc.pid = 12345
                mock_proc.communicate = AsyncMock(return_value=(b"output", b""))
                mock_proc.returncode = 0
                mock_exec.return_value = mock_proc

                await mgr.run_local_selfplay(
                    job_id="job1",
                    num_games=10,
                    board_type="hex8",
                    num_players=2,
                    model_path=None,
                    output_dir=tmpdir,
                )

                # Should have updated progress
                assert improvement_state["job1"].selfplay_progress["node1"] == 10


class TestTournamentMethods:
    """Test tournament-related methods."""

    @pytest.mark.asyncio
    async def test_run_distributed_tournament_no_state(self):
        """Test tournament with no state returns early."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        await mgr.run_distributed_tournament("nonexistent-job")

    @pytest.mark.asyncio
    async def test_run_distributed_tournament_completes(self):
        """Test tournament completion updates state."""
        class MockState:
            status = "pending"
            completed_matches = 0
            worker_nodes = []
            models = ["model1.pth", "model2.pth"]  # At least 2 models required
            total_matches = 0
            games_per_pair = 10
            board_type = "hex8"
            num_players = 2

        tournament_state = {"tourney1": MockState()}

        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
            distributed_tournament_state=tournament_state,
        )

        await mgr.run_distributed_tournament("tourney1")

        # Status should be updated
        assert tournament_state["tourney1"].status == "completed"


class TestValidationMethods:
    """Test validation methods."""
    # NOTE: run_parity_validation and run_npz_export tests removed Dec 2025 - methods were dead code

    @pytest.mark.asyncio
    async def test_run_post_training_gauntlet(self):
        """Test post-training gauntlet returns True."""
        class MockJob:
            job_id = "test-job"

        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        result = await mgr.run_post_training_gauntlet(MockJob())
        assert result is True


class TestHealthCheck:
    """Test JobManager.health_check() method."""

    def test_health_check_healthy_no_jobs(self):
        """Test health_check returns healthy when subscribed and no jobs running."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )
        # Mark as subscribed to get healthy status
        mgr._subscribed = True

        health = mgr.health_check()

        assert health["status"] == "healthy"
        assert health["running_jobs"] == 0
        assert health["failed_jobs"] == 0
        assert health["errors_count"] == 0
        assert health["subscribed"] is True

    def test_health_check_with_running_jobs(self):
        """Test health_check reports running jobs correctly."""
        active_jobs = {
            "selfplay": {
                "job1": {"status": "running", "node_id": "node1"},
                "job2": {"status": "running", "node_id": "node2"},
            }
        }
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs=active_jobs,
            jobs_lock=threading.Lock(),
        )
        # Mark as subscribed to get healthy status
        mgr._subscribed = True

        health = mgr.health_check()

        assert health["status"] == "healthy"
        assert health["running_jobs"] == 2
        assert health["operations_count"] == 2
        assert "selfplay" in health["job_types"]

    def test_health_check_degraded_with_failures(self):
        """Test health_check returns degraded with >20% failure rate."""
        # 3 running, 1 failed = 25% failure rate -> degraded
        active_jobs = {
            "selfplay": {
                "job1": {"status": "running", "node_id": "node1"},
                "job2": {"status": "running", "node_id": "node2"},
                "job3": {"status": "running", "node_id": "node3"},
                "job4": {"status": "failed", "node_id": "node4"},
            }
        }
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs=active_jobs,
            jobs_lock=threading.Lock(),
        )

        health = mgr.health_check()

        assert health["status"] == "degraded"
        assert health["failed_jobs"] == 1
        assert "failure rate" in health["last_error"].lower()

    def test_health_check_unhealthy_with_high_failures(self):
        """Test health_check returns unhealthy with >50% failure rate."""
        # 1 running, 2 failed = 66% failure rate -> unhealthy
        active_jobs = {
            "selfplay": {
                "job1": {"status": "running", "node_id": "node1"},
                "job2": {"status": "failed", "node_id": "node2"},
                "job3": {"status": "error", "node_id": "node3"},
            }
        }
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs=active_jobs,
            jobs_lock=threading.Lock(),
        )

        health = mgr.health_check()

        assert health["status"] == "unhealthy"
        assert health["failed_jobs"] == 2
        assert health["errors_count"] == 2

    def test_health_check_degraded_not_subscribed(self):
        """Test health_check degrades when not subscribed to events."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )
        # Not subscribed (default)

        health = mgr.health_check()

        # Should be degraded because not subscribed
        assert health["status"] == "degraded"
        assert health["subscribed"] is False
        assert health["last_error"] == "Not subscribed to events"

    def test_health_check_counts_timeout_and_error_as_failed(self):
        """Test health_check counts timeout and error statuses as failed."""
        active_jobs = {
            "selfplay": {
                "job1": {"status": "failed", "node_id": "node1"},
                "job2": {"status": "error", "node_id": "node2"},
                "job3": {"status": "timeout", "node_id": "node3"},
            }
        }
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs=active_jobs,
            jobs_lock=threading.Lock(),
        )

        health = mgr.health_check()

        # All 3 should be counted as failed
        assert health["failed_jobs"] == 3
        assert health["running_jobs"] == 0


class TestEventSubscriptions:
    """Test event subscription methods."""

    def test_subscribe_to_events_sets_flag(self):
        """Test subscribe_to_events sets _subscribed flag."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )
        assert mgr._subscribed is False

        # Mock the event router to avoid actual subscription
        with patch("scripts.p2p.managers.job_manager._get_event_emitter"):
            with patch("app.coordination.event_router.get_event_bus") as mock_bus:
                mock_bus.return_value.subscribe = MagicMock()
                mgr.subscribe_to_events()

        assert mgr._subscribed is True

    def test_subscribe_to_events_idempotent(self):
        """Test subscribe_to_events only subscribes once."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )
        mgr._subscribed = True  # Already subscribed

        # Mock the event bus
        with patch("app.coordination.event_router.get_event_bus") as mock_bus:
            mgr.subscribe_to_events()
            # Should not call get_event_bus because already subscribed
            mock_bus.assert_not_called()


class TestHostOfflineHandler:
    """Test _on_host_offline handler."""

    @pytest.mark.asyncio
    async def test_on_host_offline_cancels_jobs(self):
        """Test _on_host_offline cancels jobs on offline node."""
        active_jobs = {
            "selfplay": {
                "job1": {"status": "running", "node_id": "offline-node"},
                "job2": {"status": "running", "node_id": "other-node"},
            }
        }
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs=active_jobs,
            jobs_lock=threading.Lock(),
        )

        # Create mock event
        class MockEvent:
            payload = {"node_id": "offline-node"}

        with patch.object(mgr, "_emit_task_event"):
            await mgr._on_host_offline(MockEvent())

        # Job on offline node should be cancelled
        assert active_jobs["selfplay"]["job1"]["status"] == "cancelled"
        # Job on other node should still be running
        assert active_jobs["selfplay"]["job2"]["status"] == "running"
        # Stats should be updated
        assert mgr.stats.jobs_cancelled >= 1
        assert mgr.stats.hosts_offline >= 1

    @pytest.mark.asyncio
    async def test_on_host_offline_empty_node_id(self):
        """Test _on_host_offline handles empty node_id gracefully."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="test-node",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={"selfplay": {"job1": {"status": "running", "node_id": "node1"}}},
            jobs_lock=threading.Lock(),
        )

        class MockEvent:
            payload = {"node_id": ""}

        # Should not raise, should return early
        await mgr._on_host_offline(MockEvent())

        # Job should not be cancelled
        assert mgr.active_jobs["selfplay"]["job1"]["status"] == "running"
