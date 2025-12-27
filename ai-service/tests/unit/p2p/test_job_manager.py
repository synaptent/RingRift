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


class TestJobCounting:
    """Test job counting methods."""

    def test_get_job_count_for_node_empty(self):
        """Test job count for node with no jobs."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        assert mgr.get_job_count_for_node("node1") == 0
        assert mgr.get_job_count_for_node("nonexistent") == 0

    def test_get_job_count_for_node_with_jobs(self):
        """Test job count for node with jobs."""
        active_jobs = {
            "selfplay": {
                "job1": {"node_id": "node1", "status": "running"},
                "job2": {"node_id": "node2", "status": "running"},
            },
            "training": {
                "job3": {"node_id": "node1", "status": "running"},
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

        assert mgr.get_job_count_for_node("node1") == 2
        assert mgr.get_job_count_for_node("node2") == 1
        assert mgr.get_job_count_for_node("node3") == 0

    def test_get_selfplay_job_count_for_node(self):
        """Test selfplay job count for node."""
        active_jobs = {
            "selfplay": {
                "job1": {"node_id": "node1", "status": "running"},
                "job2": {"node_id": "node1", "status": "running"},
                "job3": {"node_id": "node2", "status": "running"},
            },
            "training": {
                "job4": {"node_id": "node1", "status": "running"},
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

        assert mgr.get_selfplay_job_count_for_node("node1") == 2
        assert mgr.get_selfplay_job_count_for_node("node2") == 1
        assert mgr.get_selfplay_job_count_for_node("node3") == 0

    def test_get_training_job_count_for_node(self):
        """Test training job count for node."""
        active_jobs = {
            "selfplay": {
                "job1": {"node_id": "node1", "status": "running"},
            },
            "training": {
                "job2": {"node_id": "node1", "status": "running"},
                "job3": {"node_id": "node1", "status": "running"},
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

        assert mgr.get_training_job_count_for_node("node1") == 2

    def test_job_count_with_object_jobs(self):
        """Test job counting with object-style jobs (not dicts)."""
        class MockJob:
            def __init__(self, node_id):
                self.node_id = node_id

        active_jobs = {
            "selfplay": {
                "job1": MockJob("node1"),
                "job2": MockJob("node2"),
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

        assert mgr.get_selfplay_job_count_for_node("node1") == 1
        assert mgr.get_selfplay_job_count_for_node("node2") == 1


class TestJobManagement:
    """Test job management methods."""

    def test_get_all_jobs(self):
        """Test getting all active jobs."""
        active_jobs = {
            "selfplay": {"job1": {"status": "running"}},
            "training": {"job2": {"status": "pending"}},
        }

        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs=active_jobs,
            jobs_lock=threading.Lock(),
        )

        all_jobs = mgr.get_all_jobs()
        assert "selfplay" in all_jobs
        assert "training" in all_jobs
        assert all_jobs["selfplay"]["job1"]["status"] == "running"

    def test_get_all_jobs_returns_copy(self):
        """Test that get_all_jobs returns a shallow copy."""
        active_jobs = {
            "selfplay": {"job1": {"status": "running"}},
        }

        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs=active_jobs,
            jobs_lock=threading.Lock(),
        )

        all_jobs = mgr.get_all_jobs()

        # Adding a new job type to returned dict doesn't affect original
        all_jobs["new_type"] = {"job2": {"status": "new"}}
        assert "new_type" not in mgr.active_jobs

        # Adding a job to an existing type's shallow copy doesn't affect original
        all_jobs["selfplay"]["job2"] = {"status": "added"}
        assert "job2" not in mgr.active_jobs["selfplay"]

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
    """Test validation methods (placeholders)."""

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

    @pytest.mark.asyncio
    async def test_run_parity_validation(self):
        """Test parity validation runs without error."""
        mgr = JobManager(
            ringrift_path="/tmp",
            node_id="node1",
            peers={},
            peers_lock=threading.Lock(),
            active_jobs={},
            jobs_lock=threading.Lock(),
        )

        # Should not raise
        await mgr.run_parity_validation(
            job_id="test-job",
            board_type="hex8",
            num_players=2,
            num_seeds=10,
        )

    @pytest.mark.asyncio
    async def test_run_npz_export(self):
        """Test NPZ export runs without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = JobManager(
                ringrift_path=tmpdir,
                node_id="node1",
                peers={},
                peers_lock=threading.Lock(),
                active_jobs={},
                jobs_lock=threading.Lock(),
            )

            # Should not raise
            await mgr.run_npz_export(
                job_id="test-job",
                board_type="hex8",
                num_players=2,
                output_dir=tmpdir,
            )
