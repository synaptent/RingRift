"""Unit tests for app.coordination.slurm_backend (December 27, 2025).

Tests for Slurm backend integration:
- SlurmPartition, SlurmJobState enums
- SlurmNode, SlurmJob, SlurmJobStatus dataclasses
- SlurmBackend class
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.coordination.slurm_backend import (
    SlurmBackend,
    SlurmJob,
    SlurmJobState,
    SlurmJobStatus,
    SlurmNode,
    SlurmPartition,
    SLURM_MANAGED_PATTERNS,
)


# =============================================================================
# SlurmPartition Tests
# =============================================================================


class TestSlurmPartition:
    """Tests for SlurmPartition enum."""

    def test_partition_values(self):
        """Test partition enum values."""
        assert SlurmPartition.GPU_TRAIN.value == "gpu-train"
        assert SlurmPartition.GPU_SELFPLAY.value == "gpu-selfplay"
        assert SlurmPartition.CPU_EVAL.value == "cpu-eval"
        assert SlurmPartition.GPU_GH200.value == "gpu-gh200"

    def test_partition_from_string(self):
        """Test creating partition from string."""
        partition = SlurmPartition("gpu-train")
        assert partition == SlurmPartition.GPU_TRAIN


# =============================================================================
# SlurmJobState Tests
# =============================================================================


class TestSlurmJobState:
    """Tests for SlurmJobState enum."""

    def test_job_states(self):
        """Test job state enum values."""
        assert SlurmJobState.PENDING.value == "PENDING"
        assert SlurmJobState.RUNNING.value == "RUNNING"
        assert SlurmJobState.COMPLETED.value == "COMPLETED"
        assert SlurmJobState.FAILED.value == "FAILED"
        assert SlurmJobState.CANCELLED.value == "CANCELLED"
        assert SlurmJobState.TIMEOUT.value == "TIMEOUT"
        assert SlurmJobState.NODE_FAIL.value == "NODE_FAIL"
        assert SlurmJobState.UNKNOWN.value == "UNKNOWN"


# =============================================================================
# SlurmNode Tests
# =============================================================================


class TestSlurmNode:
    """Tests for SlurmNode dataclass."""

    def test_creation(self):
        """Test creating a SlurmNode."""
        node = SlurmNode(
            name="gpu-node-01",
            partition="gpu-train",
            state="idle",
            cpus=64,
            memory_mb=256000,
            features=["gh200"],
            gres="gpu:8",
        )

        assert node.name == "gpu-node-01"
        assert node.partition == "gpu-train"
        assert node.state == "idle"
        assert node.cpus == 64

    def test_is_idle_true(self):
        """Test is_idle property when node is idle."""
        node = SlurmNode(
            name="test", partition="test", state="idle",
            cpus=8, memory_mb=16000
        )
        assert node.is_idle is True

        node_mix = SlurmNode(
            name="test", partition="test", state="mix",
            cpus=8, memory_mb=16000
        )
        assert node_mix.is_idle is True

    def test_is_idle_false(self):
        """Test is_idle property when node is not idle."""
        node = SlurmNode(
            name="test", partition="test", state="allocated",
            cpus=8, memory_mb=16000
        )
        assert node.is_idle is False

    def test_is_allocated(self):
        """Test is_allocated property."""
        node = SlurmNode(
            name="test", partition="test", state="alloc",
            cpus=8, memory_mb=16000
        )
        assert node.is_allocated is True

        node2 = SlurmNode(
            name="test", partition="test", state="idle",
            cpus=8, memory_mb=16000
        )
        assert node2.is_allocated is False

    def test_gpu_type_from_features(self):
        """Test GPU type extraction from features."""
        node = SlurmNode(
            name="test", partition="test", state="idle",
            cpus=8, memory_mb=16000,
            features=["gpu", "gh200"]
        )
        assert node.gpu_type == "gh200"

    def test_gpu_type_h100(self):
        """Test GPU type for H100."""
        node = SlurmNode(
            name="test", partition="test", state="idle",
            cpus=8, memory_mb=16000,
            features=["h100"]
        )
        assert node.gpu_type == "h100"

    def test_gpu_type_unknown(self):
        """Test GPU type when not detectable."""
        node = SlurmNode(
            name="test", partition="test", state="idle",
            cpus=8, memory_mb=16000,
            features=["gpu", "nvidia"]
        )
        assert node.gpu_type == "unknown"


# =============================================================================
# SlurmJob Tests
# =============================================================================


class TestSlurmJob:
    """Tests for SlurmJob dataclass."""

    def test_default_values(self):
        """Test default job values."""
        job = SlurmJob(name="test-job")

        assert job.name == "test-job"
        assert job.partition == SlurmPartition.GPU_SELFPLAY
        assert job.nodes == 1
        assert job.cpus_per_task == 16
        assert job.memory_gb == 64
        assert job.gpus == 1
        assert job.time_limit == "8:00:00"

    def test_custom_values(self):
        """Test custom job values."""
        job = SlurmJob(
            name="training-job",
            partition=SlurmPartition.GPU_TRAIN,
            nodes=4,
            cpus_per_task=32,
            memory_gb=256,
            gpus=8,
            time_limit="24:00:00",
        )

        assert job.partition == SlurmPartition.GPU_TRAIN
        assert job.nodes == 4
        assert job.gpus == 8

    def test_to_sbatch_args_basic(self):
        """Test sbatch argument generation."""
        job = SlurmJob(name="test")
        args = job.to_sbatch_args()

        assert "--job-name=test" in args
        assert "--partition=gpu-selfplay" in args
        assert "--nodes=1" in args
        assert "--cpus-per-task=16" in args
        assert "--mem=64G" in args
        assert "--time=8:00:00" in args
        assert "--gres=gpu:1" in args

    def test_to_sbatch_args_no_gpu(self):
        """Test sbatch args without GPU."""
        job = SlurmJob(name="cpu-job", gpus=0)
        args = job.to_sbatch_args()

        # Should not have --gres
        assert not any("--gres" in arg for arg in args)

    def test_to_sbatch_args_with_nodelist(self):
        """Test sbatch args with specific nodelist."""
        job = SlurmJob(name="test", nodelist="gpu-node-01,gpu-node-02")
        args = job.to_sbatch_args()

        assert "--nodelist=gpu-node-01,gpu-node-02" in args

    def test_to_sbatch_args_with_exclude(self):
        """Test sbatch args with node exclusion."""
        job = SlurmJob(name="test", exclude="gpu-node-bad")
        args = job.to_sbatch_args()

        assert "--exclude=gpu-node-bad" in args

    def test_to_sbatch_args_with_features(self):
        """Test sbatch args with required features."""
        job = SlurmJob(name="test", features="gh200")
        args = job.to_sbatch_args()

        assert "--constraint=gh200" in args

    def test_to_sbatch_args_with_dependency(self):
        """Test sbatch args with job dependency."""
        job = SlurmJob(name="test", dependency="afterok:12345")
        args = job.to_sbatch_args()

        assert "--dependency=afterok:12345" in args

    def test_to_sbatch_args_with_array(self):
        """Test sbatch args for array job."""
        job = SlurmJob(name="test", array="0-9%5")
        args = job.to_sbatch_args()

        assert "--array=0-9%5" in args


# =============================================================================
# SlurmJobStatus Tests
# =============================================================================


class TestSlurmJobStatus:
    """Tests for SlurmJobStatus dataclass."""

    def test_creation(self):
        """Test creating job status."""
        status = SlurmJobStatus(
            job_id=12345,
            name="test-job",
            state=SlurmJobState.RUNNING,
            partition="gpu-train",
            node="gpu-node-01",
            start_time="2025-12-27T10:00:00",
            run_time="01:30:00",
        )

        assert status.job_id == 12345
        assert status.name == "test-job"
        assert status.state == SlurmJobState.RUNNING

    def test_is_running_true(self):
        """Test is_running when job is running."""
        status = SlurmJobStatus(
            job_id=1, name="test", state=SlurmJobState.RUNNING,
            partition="test", node=None, start_time=None, run_time=None
        )
        assert status.is_running is True

    def test_is_running_false(self):
        """Test is_running when job is not running."""
        status = SlurmJobStatus(
            job_id=1, name="test", state=SlurmJobState.PENDING,
            partition="test", node=None, start_time=None, run_time=None
        )
        assert status.is_running is False

    def test_is_finished_completed(self):
        """Test is_finished for completed job."""
        status = SlurmJobStatus(
            job_id=1, name="test", state=SlurmJobState.COMPLETED,
            partition="test", node=None, start_time=None, run_time=None
        )
        assert status.is_finished is True

    def test_is_finished_failed(self):
        """Test is_finished for failed job."""
        status = SlurmJobStatus(
            job_id=1, name="test", state=SlurmJobState.FAILED,
            partition="test", node=None, start_time=None, run_time=None
        )
        assert status.is_finished is True

    def test_is_finished_cancelled(self):
        """Test is_finished for cancelled job."""
        status = SlurmJobStatus(
            job_id=1, name="test", state=SlurmJobState.CANCELLED,
            partition="test", node=None, start_time=None, run_time=None
        )
        assert status.is_finished is True

    def test_is_finished_timeout(self):
        """Test is_finished for timed out job."""
        status = SlurmJobStatus(
            job_id=1, name="test", state=SlurmJobState.TIMEOUT,
            partition="test", node=None, start_time=None, run_time=None
        )
        assert status.is_finished is True

    def test_is_finished_running(self):
        """Test is_finished for running job."""
        status = SlurmJobStatus(
            job_id=1, name="test", state=SlurmJobState.RUNNING,
            partition="test", node=None, start_time=None, run_time=None
        )
        assert status.is_finished is False


# =============================================================================
# SlurmBackend Tests
# =============================================================================


class TestSlurmBackend:
    """Tests for SlurmBackend class."""

    def test_initialization_default(self):
        """Test default initialization."""
        backend = SlurmBackend()

        assert backend.controller_user == "ubuntu"
        assert backend.ssh_timeout == 10
        assert backend._nodes == {}
        assert backend._jobs == {}

    def test_initialization_custom(self):
        """Test custom initialization."""
        backend = SlurmBackend(
            controller_host="slurm.example.com",
            controller_user="admin",
            nfs_base="/data/ringrift",
            ssh_timeout=30,
        )

        assert backend.controller_host == "slurm.example.com"
        assert backend.controller_user == "admin"
        assert str(backend.nfs_base) == "/data/ringrift"
        assert backend.ssh_timeout == 30

    def test_is_slurm_node_empty_patterns(self):
        """Test is_slurm_node with empty patterns."""
        backend = SlurmBackend()

        # With empty SLURM_MANAGED_PATTERNS, no nodes should match
        assert backend.is_slurm_node("any-node") is False
        assert backend.is_slurm_node("lambda-gpu-1") is False

    @patch.object(SlurmBackend, '_ssh_command', new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_refresh_state_throttled(self, mock_ssh):
        """Test that refresh is throttled."""
        backend = SlurmBackend()
        backend._last_refresh = 999999999999  # Far in the future

        await backend.refresh_state()

        # SSH should not be called due to throttling
        mock_ssh.assert_not_called()

    @patch.object(SlurmBackend, '_ssh_command', new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_ssh_command_timeout(self, mock_ssh):
        """Test SSH command timeout handling."""
        mock_ssh.return_value = (-1, "", "SSH timeout")

        backend = SlurmBackend(controller_host="test")
        rc, stdout, stderr = await backend._ssh_command("test")

        assert rc == -1
        assert stderr == "SSH timeout"

    @patch.object(SlurmBackend, '_ssh_command', new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_refresh_nodes_success(self, mock_ssh):
        """Test successful node refresh."""
        mock_ssh.return_value = (
            0,
            "node1|gpu-train|idle|64|256000|gh200|gpu:8\n"
            "node2|gpu-train|allocated|64|256000|h100|gpu:8",
            ""
        )

        backend = SlurmBackend(controller_host="test")
        backend._last_refresh = 0
        await backend._refresh_nodes()

        assert "node1" in backend._nodes
        assert backend._nodes["node1"].state == "idle"
        assert backend._nodes["node1"].cpus == 64

    @patch.object(SlurmBackend, '_ssh_command', new_callable=AsyncMock)
    @pytest.mark.asyncio
    async def test_refresh_nodes_failure(self, mock_ssh):
        """Test node refresh on SSH failure."""
        mock_ssh.return_value = (1, "", "Connection refused")

        backend = SlurmBackend(controller_host="test")
        backend._last_refresh = 0
        await backend._refresh_nodes()

        # Should not crash, nodes should remain empty
        assert backend._nodes == {}


# =============================================================================
# Integration Tests
# =============================================================================


class TestSlurmBackendIntegration:
    """Integration tests for Slurm backend."""

    def test_job_workflow(self):
        """Test creating and configuring a job for submission."""
        # Create a selfplay job
        job = SlurmJob(
            name="selfplay-hex8-2p",
            partition=SlurmPartition.GPU_SELFPLAY,
            nodes=1,
            cpus_per_task=8,
            memory_gb=32,
            gpus=1,
            time_limit="2:00:00",
            command="python -m app.training.selfplay --board hex8",
            env_vars={"PYTHONPATH": "."},
        )

        args = job.to_sbatch_args()

        # Verify critical arguments
        assert "--job-name=selfplay-hex8-2p" in args
        assert "--partition=gpu-selfplay" in args
        assert "--gres=gpu:1" in args

    def test_training_job_config(self):
        """Test training job configuration."""
        job = SlurmJob(
            name="train-hex8-2p",
            partition=SlurmPartition.GPU_TRAIN,
            nodes=1,
            cpus_per_task=32,
            memory_gb=128,
            gpus=8,
            time_limit="24:00:00",
            features="gh200",
        )

        args = job.to_sbatch_args()

        assert "--partition=gpu-train" in args
        assert "--gres=gpu:8" in args
        assert "--constraint=gh200" in args

    def test_array_job_config(self):
        """Test array job configuration."""
        job = SlurmJob(
            name="eval-batch",
            partition=SlurmPartition.CPU_EVAL,
            nodes=1,
            cpus_per_task=16,
            memory_gb=64,
            gpus=0,
            array="0-99%10",  # 100 tasks, max 10 concurrent
        )

        args = job.to_sbatch_args()

        assert "--array=0-99%10" in args
        assert not any("--gres" in arg for arg in args)


# =============================================================================
# Edge Cases
# =============================================================================


class TestSlurmEdgeCases:
    """Tests for edge cases and error handling."""

    def test_node_with_empty_features(self):
        """Test node with no features."""
        node = SlurmNode(
            name="test", partition="test", state="idle",
            cpus=8, memory_mb=16000,
            features=[]
        )
        assert node.gpu_type == "unknown"

    def test_node_state_case_insensitive(self):
        """Test that state checks are case-insensitive."""
        node1 = SlurmNode(
            name="test", partition="test", state="IDLE",
            cpus=8, memory_mb=16000
        )
        assert node1.is_idle is True

        node2 = SlurmNode(
            name="test", partition="test", state="Allocated",
            cpus=8, memory_mb=16000
        )
        assert node2.is_allocated is True

    def test_job_with_env_vars(self):
        """Test job with environment variables."""
        job = SlurmJob(
            name="test",
            env_vars={
                "PYTHONPATH": ".",
                "CUDA_VISIBLE_DEVICES": "0,1",
            }
        )

        assert job.env_vars["PYTHONPATH"] == "."
        assert job.env_vars["CUDA_VISIBLE_DEVICES"] == "0,1"

    def test_job_status_exit_code(self):
        """Test job status with exit code."""
        status = SlurmJobStatus(
            job_id=1, name="test", state=SlurmJobState.FAILED,
            partition="test", node="node1",
            start_time="2025-12-27T10:00:00",
            run_time="00:05:00",
            exit_code=1,
        )

        assert status.exit_code == 1
        assert status.is_finished is True
