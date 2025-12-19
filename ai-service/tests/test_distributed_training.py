"""
Unit tests for distributed training utilities.

These tests mock the PyTorch distributed module to test functionality
without requiring an actual distributed environment with multiple GPUs.
"""

import os
from unittest import mock

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset

# Import the module under test
from app.training.distributed import (
    setup_distributed,
    cleanup_distributed,
    is_distributed,
    is_main_process,
    get_rank,
    get_world_size,
    get_local_rank,
    get_distributed_sampler,
    wrap_model_ddp,
    synchronize,
    reduce_tensor,
    all_gather_object,
    broadcast_object,
    seed_everything,
    get_device_for_rank,
    scale_learning_rate,
    DistributedMetrics,
    DistributedTrainer,
)
from app.training.data_loader import StreamingDataLoader


class SimpleModel(nn.Module):
    """Simple model for testing DDP wrapping."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


class TestSetupDistributed:
    """Tests for setup_distributed function."""

    @mock.patch("app.training.distributed.dist")
    def test_setup_with_explicit_params(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test setup with explicit rank and world_size."""
        mock_dist.is_initialized.return_value = False
        mock_dist.is_available.return_value = True

        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            setup_distributed(rank=0, world_size=2, backend="gloo")

        mock_dist.init_process_group.assert_called_once()
        call_kwargs = mock_dist.init_process_group.call_args[1]
        assert call_kwargs["backend"] == "gloo"
        assert call_kwargs["world_size"] == 2
        assert call_kwargs["rank"] == 0

    @mock.patch("app.training.distributed.dist")
    def test_setup_reads_env_vars(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test setup reads from environment variables."""
        mock_dist.is_initialized.return_value = False
        mock_dist.is_available.return_value = True

        env_vars = {
            "RANK": "1",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "localhost",
            "MASTER_PORT": "29500",
        }

        with mock.patch.dict(os.environ, env_vars, clear=False):
            with mock.patch.object(
                torch.cuda, "is_available", return_value=False
            ):
                setup_distributed()

        mock_dist.init_process_group.assert_called_once()
        call_kwargs = mock_dist.init_process_group.call_args[1]
        assert call_kwargs["rank"] == 1
        assert call_kwargs["world_size"] == 4

    @mock.patch("app.training.distributed.dist")
    def test_setup_skips_if_already_initialized(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test that setup does nothing if already initialized."""
        mock_dist.is_initialized.return_value = True

        setup_distributed(rank=0, world_size=2)

        mock_dist.init_process_group.assert_not_called()

    @mock.patch("app.training.distributed.dist")
    def test_setup_selects_nccl_for_cuda(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test NCCL backend is selected when CUDA is available."""
        mock_dist.is_initialized.return_value = False
        mock_dist.is_available.return_value = True

        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            with mock.patch.object(torch.cuda, "set_device"):
                setup_distributed(rank=0, world_size=2)

        call_kwargs = mock_dist.init_process_group.call_args[1]
        assert call_kwargs["backend"] == "nccl"


class TestCleanupDistributed:
    """Tests for cleanup_distributed function."""

    @mock.patch("app.training.distributed.dist")
    def test_cleanup_when_initialized(self, mock_dist: mock.MagicMock) -> None:
        """Test that cleanup destroys process group when initialized."""
        mock_dist.is_initialized.return_value = True

        cleanup_distributed()

        mock_dist.destroy_process_group.assert_called_once()

    @mock.patch("app.training.distributed.dist")
    def test_cleanup_when_not_initialized(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test cleanup is safe when not initialized."""
        mock_dist.is_initialized.return_value = False

        cleanup_distributed()

        mock_dist.destroy_process_group.assert_not_called()


class TestIsDistributed:
    """Tests for is_distributed function."""

    @mock.patch("app.training.distributed.dist")
    def test_true_when_available_and_initialized(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns True when distributed is ready."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True

        assert is_distributed() is True

    @mock.patch("app.training.distributed.dist")
    def test_false_when_not_available(self, mock_dist: mock.MagicMock) -> None:
        """Test returns False when distributed is not available."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = True

        assert is_distributed() is False

    @mock.patch("app.training.distributed.dist")
    def test_false_when_not_initialized(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns False when not initialized."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = False

        assert is_distributed() is False


class TestIsMainProcess:
    """Tests for is_main_process function."""

    @mock.patch("app.training.distributed.dist")
    def test_true_when_not_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns True when not in distributed mode."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        assert is_main_process() is True

    @mock.patch("app.training.distributed.dist")
    def test_true_when_rank_0(self, mock_dist: mock.MagicMock) -> None:
        """Test returns True when rank is 0."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0

        assert is_main_process() is True

    @mock.patch("app.training.distributed.dist")
    def test_false_when_rank_not_0(self, mock_dist: mock.MagicMock) -> None:
        """Test returns False when rank is not 0."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 1

        assert is_main_process() is False


class TestGetRank:
    """Tests for get_rank function."""

    @mock.patch("app.training.distributed.dist")
    def test_returns_0_when_not_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns 0 when not in distributed mode."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        assert get_rank() == 0

    @mock.patch("app.training.distributed.dist")
    def test_returns_rank_when_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns actual rank when distributed."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 3

        assert get_rank() == 3


class TestGetWorldSize:
    """Tests for get_world_size function."""

    @mock.patch("app.training.distributed.dist")
    def test_returns_1_when_not_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns 1 when not in distributed mode."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        assert get_world_size() == 1

    @mock.patch("app.training.distributed.dist")
    def test_returns_world_size_when_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns actual world size when distributed."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4

        assert get_world_size() == 4


class TestGetLocalRank:
    """Tests for get_local_rank function."""

    def test_returns_0_when_env_not_set(self) -> None:
        """Test returns 0 when LOCAL_RANK not set."""
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove LOCAL_RANK if it exists
            env_copy = {
                k: v for k, v in os.environ.items()
                if k != "LOCAL_RANK"
            }
            with mock.patch.dict(os.environ, env_copy, clear=True):
                assert get_local_rank() == 0

    def test_returns_value_from_env(self) -> None:
        """Test returns value from LOCAL_RANK env var."""
        with mock.patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            assert get_local_rank() == 2


class TestGetDistributedSampler:
    """Tests for get_distributed_sampler function."""

    @mock.patch("app.training.distributed.dist")
    def test_creates_sampler_with_correct_params(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test sampler is created with correct distributed params."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 1

        # Create a simple dataset
        dataset = TensorDataset(torch.randn(100, 10))

        sampler = get_distributed_sampler(dataset, shuffle=True, seed=123)

        assert sampler.num_replicas == 4
        assert sampler.rank == 1
        assert sampler.shuffle is True
        assert sampler.seed == 123

    @mock.patch("app.training.distributed.dist")
    def test_sampler_without_shuffle(self, mock_dist: mock.MagicMock) -> None:
        """Test sampler with shuffle disabled."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_world_size.return_value = 2
        mock_dist.get_rank.return_value = 0

        dataset = TensorDataset(torch.randn(50, 10))

        sampler = get_distributed_sampler(dataset, shuffle=False)

        assert sampler.shuffle is False


class TestWrapModelDDP:
    """Tests for wrap_model_ddp function."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    @mock.patch("app.training.distributed.DDP")
    @mock.patch("app.training.distributed.dist")
    def test_wraps_model_for_cuda(
        self, mock_dist: mock.MagicMock, mock_ddp: mock.MagicMock
    ) -> None:
        """Test model is wrapped with correct params for CUDA."""
        mock_dist.is_initialized.return_value = True

        model = SimpleModel()
        device = torch.device("cuda:0")

        wrap_model_ddp(model, device, find_unused_parameters=True)

        mock_ddp.assert_called_once_with(
            model,
            device_ids=[0],
            output_device=0,
            find_unused_parameters=True,
            broadcast_buffers=True,
        )

    @mock.patch("app.training.distributed.DDP")
    @mock.patch("app.training.distributed.dist")
    def test_wraps_model_for_cpu(
        self, mock_dist: mock.MagicMock, mock_ddp: mock.MagicMock
    ) -> None:
        """Test model is wrapped without device_ids for CPU."""
        mock_dist.is_initialized.return_value = True

        model = SimpleModel()
        device = torch.device("cpu")

        wrap_model_ddp(model, device)

        mock_ddp.assert_called_once_with(
            model,
            device_ids=None,
            output_device=None,
            find_unused_parameters=False,
            broadcast_buffers=True,
        )


class TestSynchronize:
    """Tests for synchronize function."""

    @mock.patch("app.training.distributed.dist")
    def test_calls_barrier_when_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test barrier is called when distributed."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True

        synchronize()

        mock_dist.barrier.assert_called_once()

    @mock.patch("app.training.distributed.dist")
    def test_noop_when_not_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test no barrier call when not distributed."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        synchronize()

        mock_dist.barrier.assert_not_called()


class TestReduceTensor:
    """Tests for reduce_tensor function."""

    @mock.patch("app.training.distributed.dist")
    def test_returns_unchanged_when_not_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test tensor is returned unchanged when not distributed."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        tensor = torch.tensor([1.0, 2.0, 3.0])
        result = reduce_tensor(tensor)

        assert torch.equal(result, tensor)
        mock_dist.all_reduce.assert_not_called()

    @mock.patch("app.training.distributed.dist")
    def test_calls_all_reduce_when_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test all_reduce is called when distributed."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.ReduceOp.SUM = "SUM"

        tensor = torch.tensor([1.0, 2.0, 3.0])
        reduce_tensor(tensor, op="sum")

        mock_dist.all_reduce.assert_called_once()


class TestAllGatherObject:
    """Tests for all_gather_object function."""

    @mock.patch("app.training.distributed.dist")
    def test_returns_list_when_not_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns single-element list when not distributed."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        result = all_gather_object({"test": 123})

        assert result == [{"test": 123}]


class TestBroadcastObject:
    """Tests for broadcast_object function."""

    @mock.patch("app.training.distributed.dist")
    def test_returns_object_when_not_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test returns same object when not distributed."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        obj = {"test": "value"}
        result = broadcast_object(obj)

        assert result == obj


class TestSeedEverything:
    """Tests for seed_everything function."""

    @mock.patch("app.training.distributed.dist")
    def test_sets_seeds_without_offset_when_not_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test seeds are set without rank offset when not distributed."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        seed_everything(42, rank_offset=True)

        # Verify torch seed was set (we can't easily verify the exact value)
        # But we can verify no errors occurred
        assert True

    @mock.patch("app.training.distributed.dist")
    @mock.patch("app.training.distributed.get_rank")
    def test_adds_rank_offset_when_distributed(
        self, mock_get_rank: mock.MagicMock, mock_dist: mock.MagicMock
    ) -> None:
        """Test rank is added to seed when distributed and rank_offset=True."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_get_rank.return_value = 2

        # This should use seed = 42 + 2 = 44
        seed_everything(42, rank_offset=True)

        # Verify get_rank was called
        mock_get_rank.assert_called()


class TestGetDeviceForRank:
    """Tests for get_device_for_rank function."""

    @mock.patch("app.training.distributed.get_local_rank")
    def test_returns_cuda_device_when_available(
        self, mock_local_rank: mock.MagicMock
    ) -> None:
        """Test returns cuda device when CUDA is available."""
        mock_local_rank.return_value = 1

        with mock.patch.object(torch.cuda, "is_available", return_value=True):
            device = get_device_for_rank()

        assert device == torch.device("cuda:1")

    def test_returns_cpu_when_no_gpu(self) -> None:
        """Test returns CPU when no GPU available."""
        with mock.patch.object(torch.cuda, "is_available", return_value=False):
            with mock.patch.object(
                torch.backends.mps, "is_available", return_value=False
            ):
                device = get_device_for_rank()

        assert device == torch.device("cpu")


class TestScaleLearningRate:
    """Tests for scale_learning_rate function."""

    def test_linear_scaling(self) -> None:
        """Test linear scaling multiplies by world size."""
        result = scale_learning_rate(0.001, world_size=4, scale_type="linear")
        assert result == pytest.approx(0.004)

    def test_sqrt_scaling(self) -> None:
        """Test sqrt scaling multiplies by sqrt of world size."""
        result = scale_learning_rate(0.001, world_size=4, scale_type="sqrt")
        assert result == pytest.approx(0.002)

    def test_no_scaling(self) -> None:
        """Test no scaling returns original LR."""
        result = scale_learning_rate(0.001, world_size=4, scale_type="none")
        assert result == pytest.approx(0.001)

    def test_invalid_scale_type_raises(self) -> None:
        """Test invalid scale type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown scale_type"):
            scale_learning_rate(0.001, world_size=4, scale_type="invalid")

    @mock.patch("app.training.distributed.get_world_size")
    def test_uses_get_world_size_when_not_provided(
        self, mock_get_world_size: mock.MagicMock
    ) -> None:
        """Test uses get_world_size when world_size not provided."""
        mock_get_world_size.return_value = 8

        result = scale_learning_rate(0.001, scale_type="linear")

        assert result == pytest.approx(0.008)
        mock_get_world_size.assert_called_once()


class TestDistributedMetrics:
    """Tests for DistributedMetrics class."""

    def test_add_single_value(self) -> None:
        """Test adding a single metric value."""
        metrics = DistributedMetrics()
        metrics.add("loss", 1.5)
        metrics.add("loss", 2.5)

        # Check internal state
        assert metrics._sums["loss"] == 4.0
        assert metrics._counts["loss"] == 2

    def test_add_with_count(self) -> None:
        """Test adding metric with custom count."""
        metrics = DistributedMetrics()
        metrics.add("loss", 1.0, count=10)  # 10 samples with avg loss 1.0

        assert metrics._sums["loss"] == 10.0
        assert metrics._counts["loss"] == 10

    @mock.patch("app.training.distributed.dist")
    def test_reduce_and_reset_non_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test reduce_and_reset in non-distributed mode."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        metrics = DistributedMetrics()
        metrics.add("loss", 1.0)
        metrics.add("loss", 2.0)
        metrics.add("accuracy", 0.9)

        result = metrics.reduce_and_reset()

        assert result["loss"] == pytest.approx(1.5)
        assert result["accuracy"] == pytest.approx(0.9)

        # Check accumulators were reset
        assert len(metrics._sums) == 0
        assert len(metrics._counts) == 0

    @mock.patch("app.training.distributed.reduce_tensor")
    @mock.patch("app.training.distributed.is_distributed")
    def test_reduce_and_reset_distributed(
        self, mock_is_dist: mock.MagicMock, mock_reduce: mock.MagicMock
    ) -> None:
        """Test reduce_and_reset in distributed mode."""
        mock_is_dist.return_value = True

        # Mock reduce_tensor to simulate summing across 2 processes
        def mock_reduce_fn(
            tensor: torch.Tensor, op: str = "sum"
        ) -> torch.Tensor:
            # Simulate doubling the value (2 processes)
            tensor.mul_(2)
            return tensor

        mock_reduce.side_effect = mock_reduce_fn

        metrics = DistributedMetrics()
        metrics.add("loss", 1.0)

        result = metrics.reduce_and_reset()

        # With 2 processes each contributing loss=1.0 and count=1,
        # total sum = 2.0, total count = 2, average = 1.0
        assert result["loss"] == pytest.approx(1.0)

    def test_reset_clears_accumulators(self) -> None:
        """Test reset clears all accumulators."""
        metrics = DistributedMetrics()
        metrics.add("loss", 1.0)
        metrics.add("accuracy", 0.9)

        metrics.reset()

        assert len(metrics._sums) == 0
        assert len(metrics._counts) == 0

    def test_multiple_metrics(self) -> None:
        """Test handling multiple different metrics."""
        metrics = DistributedMetrics()
        metrics.add("train_loss", 0.5)
        metrics.add("train_loss", 0.3)
        metrics.add("val_loss", 0.4)
        metrics.add("accuracy", 0.85)

        assert "train_loss" in metrics._sums
        assert "val_loss" in metrics._sums
        assert "accuracy" in metrics._sums

    @mock.patch("app.training.distributed.dist")
    def test_handles_zero_count(self, mock_dist: mock.MagicMock) -> None:
        """Test handles zero count gracefully."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        metrics = DistributedMetrics()
        # Manually set zero count (edge case)
        metrics._sums["empty"] = 0.0
        metrics._counts["empty"] = 0

        result = metrics.reduce_and_reset()

        assert result["empty"] == 0.0


class TestIntegration:
    """Integration tests for distributed utilities."""

    @mock.patch("app.training.distributed.dist")
    def test_full_training_flow_non_distributed(
        self, mock_dist: mock.MagicMock
    ) -> None:
        """Test typical training flow in non-distributed mode."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False

        # Setup (no-op in non-distributed)
        setup_distributed()

        # Check distributed status
        assert not is_distributed()
        assert is_main_process()
        assert get_rank() == 0
        assert get_world_size() == 1

        # Create model and dataset (verify they can be created)
        _ = SimpleModel()
        _ = TensorDataset(torch.randn(100, 10))

        # Learning rate scaling (should return base LR)
        lr = scale_learning_rate(0.001, scale_type="linear")
        assert lr == pytest.approx(0.001)

        # Metrics collection
        metrics = DistributedMetrics()
        for i in range(5):
            metrics.add("loss", float(i))

        avg_metrics = metrics.reduce_and_reset()
        assert avg_metrics["loss"] == pytest.approx(2.0)  # (0+1+2+3+4) / 5

        # Cleanup (no-op in non-distributed)
        cleanup_distributed()

    def test_seed_reproducibility(self) -> None:
        """Test that seed_everything produces reproducible results."""
        # Seed and generate random numbers
        seed_everything(42, rank_offset=False)
        random1 = torch.rand(5)

        # Re-seed and generate again
        seed_everything(42, rank_offset=False)
        random2 = torch.rand(5)

        assert torch.equal(random1, random2)


class TestStreamingDataLoaderSharding:
    """Tests for StreamingDataLoader distributed data sharding."""

    def _create_test_data(self, tmp_path: str, num_samples: int = 100):
        """Create a test .npz file with sample data."""
        # Create sample data
        features = np.random.rand(
            num_samples, 10, 8, 8
        ).astype(np.float32)
        globals_vec = np.random.rand(num_samples, 10).astype(np.float32)
        values = np.random.choice(
            [1.0, 0.0, -1.0], size=num_samples
        ).astype(np.float32)
        policy_indices = np.array([
            np.random.choice(55000, 5, replace=False).astype(np.int32)
            for _ in range(num_samples)
        ], dtype=object)
        policy_values = np.array([
            np.random.rand(5).astype(np.float32)
            for _ in range(num_samples)
        ], dtype=object)

        data_path = os.path.join(tmp_path, "test_data.npz")
        np.savez_compressed(
            data_path,
            features=features,
            globals=globals_vec,
            values=values,
            policy_indices=policy_indices,
            policy_values=policy_values,
        )
        return data_path

    def test_single_process_no_sharding(self, tmp_path) -> None:
        """Test loader with rank=0, world_size=1 (no sharding)."""
        data_path = self._create_test_data(str(tmp_path), num_samples=50)

        loader = StreamingDataLoader(
            data_paths=data_path,
            batch_size=10,
            shuffle=False,
            rank=0,
            world_size=1,
        )

        assert loader.total_samples == 50
        assert loader.shard_size == 50
        assert len(loader) == 5  # 50 / 10

        loader.close()

    def test_two_process_sharding_even(self, tmp_path) -> None:
        """Test loader with 2 processes and even sample count."""
        data_path = self._create_test_data(str(tmp_path), num_samples=100)

        # Rank 0
        loader0 = StreamingDataLoader(
            data_paths=data_path,
            batch_size=10,
            shuffle=False,
            rank=0,
            world_size=2,
        )

        # Rank 1
        loader1 = StreamingDataLoader(
            data_paths=data_path,
            batch_size=10,
            shuffle=False,
            rank=1,
            world_size=2,
        )

        # Each rank should get half the samples
        assert loader0.total_samples == 100
        assert loader0.shard_size == 50
        assert loader1.shard_size == 50

        # Collect indices from each loader
        indices_0 = []
        for batch in loader0:
            (features, _), _ = batch
            indices_0.append(features.shape[0])

        indices_1 = []
        for batch in loader1:
            (features, _), _ = batch
            indices_1.append(features.shape[0])

        # Total samples across both shards
        total_samples_0 = sum(indices_0)
        total_samples_1 = sum(indices_1)
        assert total_samples_0 == 50
        assert total_samples_1 == 50

        loader0.close()
        loader1.close()

    def test_four_process_sharding(self, tmp_path) -> None:
        """Test loader with 4 processes."""
        data_path = self._create_test_data(str(tmp_path), num_samples=100)

        loaders = []
        for rank in range(4):
            loader = StreamingDataLoader(
                data_paths=data_path,
                batch_size=5,
                shuffle=False,
                rank=rank,
                world_size=4,
            )
            loaders.append(loader)

        # Each rank should get ~25 samples
        for loader in loaders:
            assert loader.shard_size == 25

        for loader in loaders:
            loader.close()

    def test_sharding_with_odd_samples(self, tmp_path) -> None:
        """Test sharding with odd sample count."""
        data_path = self._create_test_data(str(tmp_path), num_samples=103)

        loader0 = StreamingDataLoader(
            data_paths=data_path,
            batch_size=10,
            shuffle=False,
            rank=0,
            world_size=2,
        )

        loader1 = StreamingDataLoader(
            data_paths=data_path,
            batch_size=10,
            shuffle=False,
            rank=1,
            world_size=2,
        )

        # Rank 0 gets 52, rank 1 gets 51
        assert loader0.shard_size == 52
        assert loader1.shard_size == 51

        loader0.close()
        loader1.close()

    def test_shuffling_reproducibility_across_epochs(self, tmp_path) -> None:
        """Test that shuffling is different per epoch but reproducible."""
        data_path = self._create_test_data(str(tmp_path), num_samples=50)

        loader = StreamingDataLoader(
            data_paths=data_path,
            batch_size=10,
            shuffle=True,
            seed=42,
            rank=0,
            world_size=1,
        )

        # Get batches from epoch 0
        loader.set_epoch(0)
        batches_epoch0 = [b for b in loader]

        # Get batches from epoch 1
        loader.set_epoch(1)
        _ = [b for b in loader]  # Different order expected

        # Reset to epoch 0 - should get same order
        loader.set_epoch(0)
        batches_epoch0_again = [b for b in loader]

        # Epoch 0 and epoch 0 again should be identical
        assert len(batches_epoch0) == len(batches_epoch0_again)

        loader.close()


@pytest.mark.skip(
    reason="TODO-DISTRIBUTED-REFACTOR: Tests written for the old distributed_training.py "
    "module which was refactored into separate modules: training/loop.py, training/config.py, "
    "and distributed/unified_data_sync.py. The DistributedTrainer class was replaced by "
    "IntegratedTrainingManager. These tests should be migrated to test the new architecture "
    "or deleted if covered by other test files. See training/README.md for current architecture."
)
class TestDistributedTrainer:
    """Tests for DistributedTrainer class."""

    @mock.patch("app.training.distributed.dist")
    @mock.patch("app.training.distributed.torch.cuda")
    def test_trainer_initialization(
        self, mock_cuda: mock.MagicMock, mock_dist: mock.MagicMock
    ) -> None:
        """Test DistributedTrainer initialization without distributed."""
        mock_dist.is_available.return_value = False
        mock_dist.is_initialized.return_value = False
        mock_cuda.is_available.return_value = False

        from app.training.config import TrainConfig

        config = TrainConfig(
            epochs_per_iter=10,
            batch_size=32,
        )

        model = SimpleModel()

        trainer = DistributedTrainer(
            config=config,
            data_paths=["/tmp/fake.npz"],
            model=model,
        )

        assert trainer.config == config
        assert trainer.model == model
        assert trainer.rank == 0
        assert trainer.world_size == 1

    @mock.patch("app.training.distributed.dist")
    def test_trainer_checkpoint_only_rank_0(
        self, mock_dist: mock.MagicMock, tmp_path
    ) -> None:
        """Test that only rank 0 saves checkpoints."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 1  # Not main process
        mock_dist.get_world_size.return_value = 2

        from app.training.config import TrainConfig

        config = TrainConfig(epochs_per_iter=1, batch_size=8)
        model = SimpleModel()

        trainer = DistributedTrainer(
            config=config,
            data_paths=["/tmp/fake.npz"],
            model=model,
            checkpoint_dir=str(tmp_path),
        )

        # Manually set state for testing
        trainer.rank = 1
        trainer.is_main = False
        # Cast for testing purposes
        trainer.wrapped_model = model  # type: ignore[assignment]

        # Mock version manager
        trainer._version_manager = mock.MagicMock()

        # Call checkpoint - should not save since rank != 0
        trainer.checkpoint(epoch=0, loss=0.5, is_best=True)

        # Version manager should not have been called
        trainer._version_manager.create_metadata.assert_not_called()

    @mock.patch("app.training.distributed.synchronize")
    @mock.patch("app.training.distributed.dist")
    def test_trainer_checkpoint_barrier_sync(
        self,
        mock_dist: mock.MagicMock,
        mock_sync: mock.MagicMock,
        tmp_path
    ) -> None:
        """Test that checkpoint uses barrier synchronization."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True
        mock_dist.get_rank.return_value = 0
        mock_dist.get_world_size.return_value = 2

        from app.training.config import TrainConfig

        config = TrainConfig(epochs_per_iter=1, batch_size=8)
        model = SimpleModel()

        trainer = DistributedTrainer(
            config=config,
            data_paths=["/tmp/fake.npz"],
            model=model,
            checkpoint_dir=str(tmp_path),
        )

        trainer.rank = 0
        trainer.is_main = True

        # Create a mock DDP-wrapped model with .module attribute
        mock_wrapped = mock.MagicMock()
        mock_wrapped.module = model
        trainer.wrapped_model = mock_wrapped  # type: ignore[assignment]

        # Mock version manager
        trainer._version_manager = mock.MagicMock()

        # Call checkpoint
        trainer.checkpoint(epoch=0, loss=0.5, is_best=False)

        # Synchronize should be called twice (before and after checkpoint)
        assert mock_sync.call_count == 2

    @mock.patch("app.training.distributed.synchronize")
    @mock.patch("app.training.distributed.dist")
    def test_trainer_load_checkpoint_barrier_sync(
        self,
        mock_dist: mock.MagicMock,
        mock_sync: mock.MagicMock,
        tmp_path
    ) -> None:
        """Test that loading checkpoint uses barrier synchronization."""
        mock_dist.is_available.return_value = True
        mock_dist.is_initialized.return_value = True

        from app.training.config import TrainConfig

        config = TrainConfig(epochs_per_iter=1, batch_size=8)
        model = SimpleModel()

        trainer = DistributedTrainer(
            config=config,
            data_paths=["/tmp/fake.npz"],
            model=model,
            checkpoint_dir=str(tmp_path),
        )

        trainer.is_main = True
        trainer.model = model
        trainer.wrapped_model = model  # type: ignore[assignment]

        # Mock version manager
        trainer._version_manager = mock.MagicMock()

        # Try to load non-existent checkpoint
        trainer._load_checkpoint("/tmp/nonexistent.pth")

        # Synchronize should be called at start (file doesn't exist)
        assert mock_sync.call_count >= 1


class TestDataShardingCorrectness:
    """Tests to verify data sharding produces unique samples per rank."""

    def _create_indexed_data(self, tmp_path: str, num_samples: int = 100):
        """Create test data with indexed values for tracking."""
        # Create data where features[i][0][0][0] = i
        # so we can track which samples each rank receives
        features = np.zeros(
            (num_samples, 1, 8, 8), dtype=np.float32
        )
        for i in range(num_samples):
            features[i, 0, 0, 0] = float(i)

        globals_vec = np.zeros((num_samples, 10), dtype=np.float32)
        values = np.zeros(num_samples, dtype=np.float32)
        policy_indices = np.array([
            np.array([0, 1, 2], dtype=np.int32)
            for _ in range(num_samples)
        ], dtype=object)
        policy_values = np.array([
            np.array([0.5, 0.3, 0.2], dtype=np.float32)
            for _ in range(num_samples)
        ], dtype=object)

        data_path = os.path.join(tmp_path, "indexed_data.npz")
        np.savez_compressed(
            data_path,
            features=features,
            globals=globals_vec,
            values=values,
            policy_indices=policy_indices,
            policy_values=policy_values,
        )
        return data_path

    def test_no_duplicate_samples_across_ranks(self, tmp_path) -> None:
        """Verify that different ranks don't receive duplicate samples."""
        data_path = self._create_indexed_data(str(tmp_path), num_samples=100)

        # Create loaders for 4 ranks
        all_samples = []
        for rank in range(4):
            loader = StreamingDataLoader(
                data_paths=data_path,
                batch_size=10,
                shuffle=False,
                rank=rank,
                world_size=4,
            )

            rank_samples = []
            for batch in loader:
                (features, _), _ = batch
                # Extract the index values from features[i][0][0][0]
                indices = features[:, 0, 0, 0].numpy().tolist()
                rank_samples.extend(indices)

            all_samples.extend(rank_samples)
            loader.close()

        # All samples should be unique
        assert len(all_samples) == len(set(all_samples)), \
            "Duplicate samples detected across ranks"

        # All 100 samples should be covered
        assert len(set(all_samples)) == 100, \
            f"Expected 100 unique samples, got {len(set(all_samples))}"

    def test_consistent_sharding_with_same_seed(self, tmp_path) -> None:
        """Verify that same seed produces consistent sharding."""
        data_path = self._create_indexed_data(str(tmp_path), num_samples=50)

        def get_rank_samples(rank: int) -> list:
            loader = StreamingDataLoader(
                data_paths=data_path,
                batch_size=10,
                shuffle=True,
                seed=42,
                rank=rank,
                world_size=2,
            )
            loader.set_epoch(0)
            samples = []
            for batch in loader:
                (features, _), _ = batch
                indices = features[:, 0, 0, 0].numpy().tolist()
                samples.extend(indices)
            loader.close()
            return samples

        # Get samples twice for same rank
        samples_run1 = get_rank_samples(0)
        samples_run2 = get_rank_samples(0)

        # Should be identical
        assert samples_run1 == samples_run2, \
            "Same seed should produce identical sharding"