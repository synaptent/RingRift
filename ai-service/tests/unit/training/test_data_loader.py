"""Tests for Streaming DataLoader.

Tests the core data loading infrastructure including FileHandle,
StreamingDataLoader, PrefetchIterator, and utility functions.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from app.training.data_loader import (
    FileHandle,
    StreamingDataLoader,
    PrefetchIterator,
    get_sample_count,
)


class TestFileHandle:
    """Tests for FileHandle class."""

    @pytest.fixture
    def sample_npz(self, tmp_path):
        """Create a sample .npz file for testing."""
        num_samples = 100
        feature_dim = 64
        global_dim = 10
        max_policy_size = 50

        # Create sample data
        features = np.random.randn(num_samples, feature_dim).astype(np.float32)
        globals_ = np.random.randn(num_samples, global_dim).astype(np.float32)
        values = np.random.randn(num_samples).astype(np.float32)

        # Create sparse policy data (some samples have empty policies)
        policy_indices = []
        policy_values = []
        for i in range(num_samples):
            if i % 10 == 0:  # 10% have empty policies (terminal states)
                policy_indices.append(np.array([], dtype=np.int32))
                policy_values.append(np.array([], dtype=np.float32))
            else:
                num_moves = np.random.randint(1, max_policy_size)
                indices = np.random.choice(1000, size=num_moves, replace=False).astype(np.int32)
                probs = np.random.dirichlet(np.ones(num_moves)).astype(np.float32)
                policy_indices.append(indices)
                policy_values.append(probs)

        # Save as npz
        npz_path = tmp_path / "test_data.npz"
        np.savez(
            npz_path,
            features=features,
            globals=globals_,
            values=values,
            policy_indices=np.array(policy_indices, dtype=object),
            policy_values=np.array(policy_values, dtype=object),
        )
        return npz_path

    @pytest.fixture
    def sample_npz_v2(self, tmp_path):
        """Create a v2 format .npz file with multi-player values."""
        num_samples = 50
        feature_dim = 64
        global_dim = 10
        max_players = 4

        features = np.random.randn(num_samples, feature_dim).astype(np.float32)
        globals_ = np.random.randn(num_samples, global_dim).astype(np.float32)
        values = np.random.randn(num_samples).astype(np.float32)
        values_mp = np.random.randn(num_samples, max_players).astype(np.float32)
        num_players = np.random.randint(2, max_players + 1, size=num_samples).astype(np.int32)

        policy_indices = [np.array([0, 1, 2], dtype=np.int32) for _ in range(num_samples)]
        policy_values = [np.array([0.5, 0.3, 0.2], dtype=np.float32) for _ in range(num_samples)]

        npz_path = tmp_path / "test_data_v2.npz"
        np.savez(
            npz_path,
            features=features,
            globals=globals_,
            values=values,
            values_mp=values_mp,
            num_players=num_players,
            policy_indices=np.array(policy_indices, dtype=object),
            policy_values=np.array(policy_values, dtype=object),
        )
        return npz_path

    def test_open_npz_file(self, sample_npz):
        """Should open .npz file successfully."""
        handle = FileHandle(str(sample_npz))
        assert handle.num_samples > 0
        assert handle._format == 'npz'
        handle.close()

    def test_file_not_found(self, tmp_path):
        """Should raise FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            FileHandle(str(tmp_path / "nonexistent.npz"))

    def test_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported format."""
        bad_file = tmp_path / "data.txt"
        bad_file.write_text("not a data file")
        with pytest.raises(ValueError, match="Unsupported file format"):
            FileHandle(str(bad_file))

    def test_invalid_npz_format(self, tmp_path):
        """Should raise ValueError for npz missing features."""
        bad_npz = tmp_path / "bad.npz"
        np.savez(bad_npz, values=np.array([1, 2, 3]))
        with pytest.raises(ValueError, match="missing 'features' key"):
            FileHandle(str(bad_npz))

    def test_filter_empty_policies(self, sample_npz):
        """Should filter out samples with empty policies."""
        handle_filtered = FileHandle(str(sample_npz), filter_empty_policies=True)
        handle_unfiltered = FileHandle(str(sample_npz), filter_empty_policies=False)

        # Unfiltered should have more samples (includes terminal states)
        assert handle_unfiltered.num_samples >= handle_filtered.num_samples
        # Original has 10% empty policies
        assert handle_filtered.num_samples == 90  # 100 - 10

        handle_filtered.close()
        handle_unfiltered.close()

    def test_get_sample(self, sample_npz):
        """Should retrieve individual samples correctly."""
        handle = FileHandle(str(sample_npz))

        features, globals_, value, policy_indices, policy_values = handle.get_sample(0)

        assert features.shape[0] == 64  # feature_dim
        assert globals_.shape[0] == 10  # global_dim
        assert isinstance(value, float)
        assert len(policy_indices) == len(policy_values)

        handle.close()

    def test_get_sample_out_of_range(self, sample_npz):
        """Should raise IndexError for out-of-range index."""
        handle = FileHandle(str(sample_npz))
        with pytest.raises(IndexError):
            handle.get_sample(handle.num_samples + 100)
        handle.close()

    def test_detect_v2_multi_player_format(self, sample_npz_v2):
        """Should detect v2 multi-player value format."""
        handle = FileHandle(str(sample_npz_v2))

        assert handle.has_multi_player_values is True
        assert handle.max_players == 4

        handle.close()

    def test_v1_format_no_multi_player(self, sample_npz):
        """Should not have multi-player values for v1 format."""
        handle = FileHandle(str(sample_npz))

        assert handle.has_multi_player_values is False

        handle.close()

    def test_close(self, sample_npz):
        """Should close file handle properly."""
        handle = FileHandle(str(sample_npz))
        handle.close()
        assert handle._data is None

    def test_close_prevents_get_sample(self, sample_npz):
        """Should raise RuntimeError when accessing closed handle."""
        handle = FileHandle(str(sample_npz))
        handle.close()

        with pytest.raises(RuntimeError, match="File handle is closed"):
            handle.get_sample(0)

    def test_get_batch(self, sample_npz):
        """Should retrieve batch of samples as tuple."""
        handle = FileHandle(str(sample_npz))
        indices = np.array([0, 1, 2, 3, 4])

        features, globals_, values, pol_indices, pol_values = handle.get_batch(indices)

        assert features.shape[0] == 5
        assert globals_.shape[0] == 5
        assert values.shape == (5,)
        assert len(pol_indices) == 5
        assert len(pol_values) == 5

        handle.close()

    def test_get_batch_with_mp(self, sample_npz_v2):
        """Should retrieve batch with multi-player values."""
        handle = FileHandle(str(sample_npz_v2))
        indices = np.array([0, 1, 2])

        result = handle.get_batch_with_mp(indices)
        features, globals_, values, pol_indices, pol_values, values_mp, num_players = result

        assert features.shape[0] == 3
        assert values_mp is not None
        assert values_mp.shape == (3, 4)
        assert num_players is not None
        assert num_players.shape == (3,)

        handle.close()

    def test_victory_type_weights(self, tmp_path):
        """Should compute victory type weights correctly."""
        num_samples = 100
        features = np.random.randn(num_samples, 10).astype(np.float32)
        globals_ = np.zeros((num_samples, 5), dtype=np.float32)
        values = np.zeros(num_samples, dtype=np.float32)

        # Create imbalanced victory types: 80% type 0, 20% type 1
        victory_types = np.array([0] * 80 + [1] * 20, dtype=np.int32)

        policy_indices = [np.array([0], dtype=np.int32) for _ in range(num_samples)]
        policy_values = [np.array([1.0], dtype=np.float32) for _ in range(num_samples)]

        npz_path = tmp_path / "victory_types.npz"
        np.savez(
            npz_path,
            features=features,
            globals=globals_,
            values=values,
            victory_types=victory_types,
            policy_indices=np.array(policy_indices, dtype=object),
            policy_values=np.array(policy_values, dtype=object),
        )

        handle = FileHandle(str(npz_path))
        weights = handle.get_victory_type_weights()

        assert weights is not None
        assert len(weights) == num_samples
        # Type 1 samples should have higher weight (inverse frequency)
        assert weights[80] > weights[0]  # Type 1 vs Type 0

        handle.close()

    def test_victory_type_weights_none_if_missing(self, sample_npz):
        """Should return None if victory_types not in data."""
        handle = FileHandle(str(sample_npz))
        weights = handle.get_victory_type_weights()
        assert weights is None
        handle.close()


class TestStreamingDataLoader:
    """Tests for StreamingDataLoader class."""

    @pytest.fixture
    def sample_data_files(self, tmp_path):
        """Create multiple sample data files."""
        files = []
        for i in range(3):
            num_samples = 50
            features = np.random.randn(num_samples, 32).astype(np.float32)
            globals_ = np.random.randn(num_samples, 8).astype(np.float32)
            values = np.random.randn(num_samples).astype(np.float32)
            policy_indices = [np.array([0, 1], dtype=np.int32) for _ in range(num_samples)]
            policy_values = [np.array([0.6, 0.4], dtype=np.float32) for _ in range(num_samples)]

            file_path = tmp_path / f"data_{i}.npz"
            np.savez(
                file_path,
                features=features,
                globals=globals_,
                values=values,
                policy_indices=np.array(policy_indices, dtype=object),
                policy_values=np.array(policy_values, dtype=object),
            )
            files.append(file_path)

        return files

    @pytest.fixture
    def single_data_file(self, tmp_path):
        """Create a single data file."""
        num_samples = 50
        features = np.random.randn(num_samples, 32).astype(np.float32)
        globals_ = np.random.randn(num_samples, 8).astype(np.float32)
        values = np.random.randn(num_samples).astype(np.float32)
        policy_indices = [np.array([0, 1], dtype=np.int32) for _ in range(num_samples)]
        policy_values = [np.array([0.6, 0.4], dtype=np.float32) for _ in range(num_samples)]

        file_path = tmp_path / "data.npz"
        np.savez(
            file_path,
            features=features,
            globals=globals_,
            values=values,
            policy_indices=np.array(policy_indices, dtype=object),
            policy_values=np.array(policy_values, dtype=object),
        )
        return file_path

    def test_initialization_with_file_list(self, sample_data_files):
        """Should load specific files from list."""
        files = [str(f) for f in sample_data_files[:2]]
        loader = StreamingDataLoader(files, batch_size=16)
        assert loader.total_samples == 100  # 2 files * 50 samples

    def test_initialization_with_single_file(self, single_data_file):
        """Should load a single file."""
        loader = StreamingDataLoader(str(single_data_file), batch_size=16)
        assert loader.total_samples == 50

    def test_batch_iteration_yields_tensors(self, sample_data_files):
        """Should iterate through batches and yield torch tensors."""
        files = [str(f) for f in sample_data_files]
        loader = StreamingDataLoader(files, batch_size=16, policy_size=1000)

        total_samples = 0
        for (features, globals_), (values, policies) in loader:
            assert isinstance(features, torch.Tensor)
            assert isinstance(globals_, torch.Tensor)
            assert isinstance(values, torch.Tensor)
            assert isinstance(policies, torch.Tensor)
            total_samples += features.shape[0]

        assert total_samples == 150

    def test_batch_size_respected(self, sample_data_files):
        """Should respect batch size (except possibly last batch)."""
        batch_size = 32
        files = [str(f) for f in sample_data_files]
        loader = StreamingDataLoader(files, batch_size=batch_size)

        batch_count = 0
        for (features, _), _ in loader:
            if batch_count < len(loader) - 1:  # All but last batch
                assert features.shape[0] == batch_size
            batch_count += 1

    def test_shuffle_with_set_epoch(self, single_data_file):
        """Should shuffle indices differently between epochs."""
        loader = StreamingDataLoader(
            str(single_data_file), batch_size=16, shuffle=True, seed=42
        )

        # Get values from first batch of epoch 0
        batch0 = next(iter(loader))
        values0 = batch0[1][0].clone()

        # Set epoch to reseed shuffle
        loader.set_epoch(1)

        # Get values from first batch of epoch 1
        batch1 = next(iter(loader))
        values1 = batch1[1][0].clone()

        # With high probability, order should be different
        # (seed 42 vs seed 43 for epoch)
        assert values0.shape == values1.shape

    def test_no_shuffle(self, single_data_file):
        """Should maintain order when shuffle=False."""
        loader = StreamingDataLoader(
            str(single_data_file), batch_size=16, shuffle=False
        )

        # First pass
        batch1 = next(iter(loader))
        values1 = batch1[1][0].clone()

        # Second pass (same order)
        batch2 = next(iter(loader))
        values2 = batch2[1][0].clone()

        # Should be identical
        assert torch.allclose(values1, values2)

    def test_len_returns_batch_count(self, sample_data_files):
        """Should return correct number of batches from len()."""
        batch_size = 16
        files = [str(f) for f in sample_data_files]
        loader = StreamingDataLoader(files, batch_size=batch_size)

        expected_batches = (150 + batch_size - 1) // batch_size  # ceil(150/16) = 10
        assert len(loader) == expected_batches

    def test_drop_last(self, sample_data_files):
        """Should drop last incomplete batch when drop_last=True."""
        batch_size = 32
        files = [str(f) for f in sample_data_files]
        loader = StreamingDataLoader(
            files, batch_size=batch_size, drop_last=True
        )

        batches = list(loader)

        # All batches should have exactly batch_size
        for (features, _), _ in batches:
            assert features.shape[0] == batch_size

    def test_nonexistent_file_skipped(self, tmp_path, single_data_file):
        """Should skip nonexistent files and continue with valid ones."""
        files = [str(single_data_file), str(tmp_path / "nonexistent.npz")]
        loader = StreamingDataLoader(files, batch_size=16)

        # Should only have samples from the valid file
        assert loader.total_samples == 50

    def test_all_files_nonexistent(self, tmp_path):
        """Should handle all files being nonexistent."""
        files = [str(tmp_path / "missing1.npz"), str(tmp_path / "missing2.npz")]
        loader = StreamingDataLoader(files, batch_size=16)

        assert loader.total_samples == 0
        assert len(list(loader)) == 0

    def test_policy_size_parameter(self, single_data_file):
        """Should respect policy_size for dense policy conversion."""
        loader = StreamingDataLoader(
            str(single_data_file),
            batch_size=16,
            policy_size=100,
        )

        (_, _), (_, policies) = next(iter(loader))
        assert policies.shape[1] == 100

    def test_distributed_sharding(self, sample_data_files):
        """Should shard data for distributed training."""
        files = [str(f) for f in sample_data_files]

        loader_rank0 = StreamingDataLoader(files, batch_size=10, rank=0, world_size=2)
        loader_rank1 = StreamingDataLoader(files, batch_size=10, rank=1, world_size=2)

        # Each rank gets half the samples
        assert loader_rank0.shard_size == 75  # 150 / 2
        assert loader_rank1.shard_size == 75


class TestPrefetchIterator:
    """Tests for PrefetchIterator class."""

    def test_basic_iteration(self):
        """Should iterate through base iterator."""
        def source():
            for i in range(10):
                yield torch.tensor([i])

        prefetch_iter = PrefetchIterator(iter(source()), prefetch_count=2)

        results = list(prefetch_iter)
        assert len(results) == 10
        assert results[0].item() == 0
        assert results[-1].item() == 9

    def test_nested_structure_preserved(self):
        """Should preserve nested tuple structure."""
        def source():
            for i in range(3):
                features = torch.randn(4, 8)
                globals_ = torch.randn(4, 4)
                values = torch.randn(4, 1)
                policies = torch.randn(4, 100)
                yield ((features, globals_), (values, policies))

        prefetch_iter = PrefetchIterator(iter(source()), prefetch_count=2)

        batches = list(prefetch_iter)
        assert len(batches) == 3

        (features, globals_), (values, policies) = batches[0]
        assert features.shape == (4, 8)
        assert globals_.shape == (4, 4)
        assert values.shape == (4, 1)
        assert policies.shape == (4, 100)

    def test_prefetch_count(self):
        """Should prefetch correct number of items."""
        fetch_times = []

        def slow_generator():
            for i in range(5):
                fetch_times.append(time.time())
                time.sleep(0.05)  # Simulate slow data loading
                yield torch.tensor([i])

        prefetch_iter = PrefetchIterator(iter(slow_generator()), prefetch_count=3)

        # Start iteration
        result = next(prefetch_iter)
        assert result.item() == 0

        # Give time for prefetching
        time.sleep(0.2)

        # Should have prefetched more items
        assert len(fetch_times) >= 2

    def test_empty_iterator(self):
        """Should handle empty iterator."""
        prefetch_iter = PrefetchIterator(iter([]), prefetch_count=2)
        results = list(prefetch_iter)
        assert len(results) == 0

    def test_exception_propagation(self):
        """Should propagate exceptions from base iterator."""
        def failing_generator():
            yield torch.tensor([0])
            yield torch.tensor([1])
            raise ValueError("Test error")

        prefetch_iter = PrefetchIterator(iter(failing_generator()), prefetch_count=1)

        # Iterate through until exception
        results = []
        with pytest.raises(ValueError, match="Test error"):
            for item in prefetch_iter:
                results.append(item.item())

        # Should have received at least the first item before exception
        assert 0 in results

    def test_pin_memory_parameter(self):
        """Should accept pin_memory parameter."""
        def source():
            for i in range(2):
                yield torch.randn(4, 8)

        # Should not raise
        prefetch_iter = PrefetchIterator(
            iter(source()), prefetch_count=2, pin_memory=False
        )
        results = list(prefetch_iter)
        assert len(results) == 2

    def test_device_transfer_parameter(self):
        """Should accept transfer_to_device parameter."""
        def source():
            for i in range(2):
                yield torch.randn(4, 8)

        device = torch.device('cpu')
        prefetch_iter = PrefetchIterator(
            iter(source()),
            prefetch_count=2,
            transfer_to_device=device,
            non_blocking=False
        )

        results = list(prefetch_iter)
        assert len(results) == 2
        assert results[0].device.type == 'cpu'


class TestGetSampleCount:
    """Tests for get_sample_count function."""

    def test_npz_file(self, tmp_path):
        """Should count samples in .npz file."""
        num_samples = 42
        features = np.random.randn(num_samples, 10).astype(np.float32)
        globals_ = np.random.randn(num_samples, 5).astype(np.float32)
        values = np.random.randn(num_samples).astype(np.float32)
        policy_indices = [np.array([0], dtype=np.int32) for _ in range(num_samples)]
        policy_values = [np.array([1.0], dtype=np.float32) for _ in range(num_samples)]

        npz_path = tmp_path / "count_test.npz"
        np.savez(
            npz_path,
            features=features,
            globals=globals_,
            values=values,
            policy_indices=np.array(policy_indices, dtype=object),
            policy_values=np.array(policy_values, dtype=object),
        )

        count = get_sample_count(str(npz_path))
        assert count == num_samples

    def test_unsupported_format(self, tmp_path):
        """Should raise ValueError for unsupported formats."""
        text_file = tmp_path / "data.txt"
        text_file.write_text("test")

        with pytest.raises(ValueError, match="Unsupported file format"):
            get_sample_count(str(text_file))

    def test_nonexistent_file(self, tmp_path):
        """Should raise FileNotFoundError for nonexistent file."""
        with pytest.raises(FileNotFoundError):
            get_sample_count(str(tmp_path / "nonexistent.npz"))


class TestBatchSparseToSensePolicies:
    """Tests for _batch_sparse_to_dense_policies method."""

    def _create_minimal_loader(self, policy_size):
        """Create a minimal StreamingDataLoader instance for testing conversion method."""
        loader = StreamingDataLoader.__new__(StreamingDataLoader)
        loader.policy_size = policy_size
        loader._file_handles = []  # Prevent __del__ errors
        return loader

    def test_basic_conversion(self):
        """Should convert sparse policies to dense batch."""
        loader = self._create_minimal_loader(policy_size=100)

        pol_indices = [
            np.array([0, 10, 50], dtype=np.int64),
            np.array([5, 25, 75], dtype=np.int64),
            np.array([], dtype=np.int64),  # Empty policy
        ]
        pol_values = [
            np.array([0.5, 0.3, 0.2], dtype=np.float32),
            np.array([0.4, 0.4, 0.2], dtype=np.float32),
            np.array([], dtype=np.float32),
        ]

        result = loader._batch_sparse_to_dense_policies(pol_indices, pol_values, 3)

        assert result.shape == (3, 100)
        assert result[0, 0] == pytest.approx(0.5)
        assert result[0, 10] == pytest.approx(0.3)
        assert result[0, 50] == pytest.approx(0.2)
        assert result[1, 5] == pytest.approx(0.4)
        assert result[2].sum() == pytest.approx(0.0)  # Empty policy

    def test_all_empty_policies(self):
        """Should handle batch of all empty policies."""
        loader = self._create_minimal_loader(policy_size=50)

        pol_indices = [np.array([], dtype=np.int64) for _ in range(3)]
        pol_values = [np.array([], dtype=np.float32) for _ in range(3)]

        result = loader._batch_sparse_to_dense_policies(pol_indices, pol_values, 3)

        assert result.shape == (3, 50)
        assert result.sum() == pytest.approx(0.0)


class TestPrefetchLoader:
    """Tests for prefetch_loader convenience function."""

    @pytest.fixture
    def data_file(self, tmp_path):
        """Create a data file for testing."""
        num_samples = 10
        features = np.random.randn(num_samples, 32).astype(np.float32)
        globals_ = np.random.randn(num_samples, 8).astype(np.float32)
        values = np.random.randn(num_samples).astype(np.float32)
        policy_indices = [np.array([0, 1], dtype=np.int32) for _ in range(num_samples)]
        policy_values = [np.array([0.6, 0.4], dtype=np.float32) for _ in range(num_samples)]

        file_path = tmp_path / "data.npz"
        np.savez(
            file_path,
            features=features,
            globals=globals_,
            values=values,
            policy_indices=np.array(policy_indices, dtype=object),
            policy_values=np.array(policy_values, dtype=object),
        )
        return file_path

    def test_basic_prefetch_loader(self, data_file):
        """Should create prefetching iterator from loader."""
        from app.training.data_loader import prefetch_loader

        loader = StreamingDataLoader(str(data_file), batch_size=4, policy_size=1000)
        prefetch = prefetch_loader(loader, prefetch_count=2)

        batches = list(prefetch)
        assert len(batches) == 3  # ceil(10 / 4) = 3


class TestMultiPlayerIteration:
    """Tests for iter_with_mp method."""

    @pytest.fixture
    def mp_data_file(self, tmp_path):
        """Create a data file with multi-player values."""
        num_samples = 20
        features = np.random.randn(num_samples, 32).astype(np.float32)
        globals_ = np.random.randn(num_samples, 8).astype(np.float32)
        values = np.random.randn(num_samples).astype(np.float32)
        values_mp = np.random.randn(num_samples, 4).astype(np.float32)
        num_players = np.random.randint(2, 5, size=num_samples).astype(np.int32)
        policy_indices = [np.array([0, 1], dtype=np.int32) for _ in range(num_samples)]
        policy_values = [np.array([0.6, 0.4], dtype=np.float32) for _ in range(num_samples)]

        file_path = tmp_path / "mp_data.npz"
        np.savez(
            file_path,
            features=features,
            globals=globals_,
            values=values,
            values_mp=values_mp,
            num_players=num_players,
            policy_indices=np.array(policy_indices, dtype=object),
            policy_values=np.array(policy_values, dtype=object),
        )
        return file_path

    def test_iter_with_mp_yields_extra_tensors(self, mp_data_file):
        """Should yield multi-player values when using iter_with_mp."""
        loader = StreamingDataLoader(str(mp_data_file), batch_size=5)

        assert loader.has_multi_player_values is True
        assert loader.max_players == 4

        batch_count = 0
        for (features, globals_), (values, policies), values_mp, num_players in loader.iter_with_mp():
            assert isinstance(features, torch.Tensor)
            assert isinstance(values_mp, torch.Tensor)
            assert isinstance(num_players, torch.Tensor)
            assert values_mp.shape[1] == 4  # max_players
            batch_count += 1

        assert batch_count == 4  # 20 / 5 = 4
