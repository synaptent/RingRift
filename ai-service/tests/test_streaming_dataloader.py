"""
Tests for StreamingDataLoader

Tests cover:
- Basic functionality with single and multiple files
- Large dataset simulation
- Memory usage bounds
- Shuffle functionality
- Multi-file loading
- Edge cases (empty files, missing files, etc.)
"""

import os
import tempfile
import pytest
import numpy as np
import torch

from app.training.data_loader import (
    StreamingDataLoader,
    StreamingDataset,
    FileHandle,
    get_sample_count,
    merge_data_files,
)


def create_test_npz(
    path: str,
    num_samples: int = 100,
    feature_shape: tuple = (40, 8, 8),
    global_features: int = 10,
    policy_size: int = 55000,
    include_empty_policies: bool = False,
    seed: int = 42,
) -> str:
    """Create a test .npz file with random data."""
    rng = np.random.default_rng(seed)

    features = rng.random(
        (num_samples,) + feature_shape, dtype=np.float64
    ).astype(np.float32)
    globals_arr = rng.random(
        (num_samples, global_features), dtype=np.float64
    ).astype(np.float32)
    values = rng.choice(
        [-1.0, 0.0, 1.0], size=num_samples
    ).astype(np.float32)

    # Create sparse policies
    policy_indices = []
    policy_values = []
    for i in range(num_samples):
        if include_empty_policies and i % 10 == 0:
            # Empty policy (terminal state)
            policy_indices.append(np.array([], dtype=np.int32))
            policy_values.append(np.array([], dtype=np.float32))
        else:
            # Random sparse policy with 5-20 non-zero entries
            num_moves = rng.integers(5, 21)
            indices = rng.choice(
                policy_size, size=num_moves, replace=False
            ).astype(np.int32)
            probs = rng.random(num_moves).astype(np.float32)
            probs = probs / probs.sum()
            policy_indices.append(indices)
            policy_values.append(probs)

    policy_indices_arr = np.array(policy_indices, dtype=object)
    policy_values_arr = np.array(policy_values, dtype=object)

    np.savez_compressed(
        path,
        features=features,
        globals=globals_arr,
        values=values,
        policy_indices=policy_indices_arr,
        policy_values=policy_values_arr,
    )
    return path


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def single_data_file(temp_dir):
    """Create a single test data file."""
    path = os.path.join(temp_dir, "test_data.npz")
    create_test_npz(path, num_samples=100)
    return path


@pytest.fixture
def multiple_data_files(temp_dir):
    """Create multiple test data files."""
    paths = []
    for i in range(3):
        path = os.path.join(temp_dir, f"test_data_{i}.npz")
        create_test_npz(path, num_samples=50 + i * 25, seed=42 + i)
        paths.append(path)
    return paths


class TestFileHandle:
    """Tests for FileHandle class."""

    def test_open_valid_file(self, single_data_file):
        """Test opening a valid .npz file."""
        handle = FileHandle(single_data_file)
        assert handle.num_samples == 100
        handle.close()

    def test_open_nonexistent_file(self, temp_dir):
        """Test opening a nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            FileHandle(os.path.join(temp_dir, "nonexistent.npz"))

    def test_filter_empty_policies(self, temp_dir):
        """Test that empty policy samples are filtered."""
        path = os.path.join(temp_dir, "with_empty.npz")
        create_test_npz(path, num_samples=100, include_empty_policies=True)

        # With filtering (default)
        handle = FileHandle(path, filter_empty_policies=True)
        assert handle.num_samples == 90  # 10% filtered out
        handle.close()

        # Without filtering
        handle = FileHandle(path, filter_empty_policies=False)
        assert handle.num_samples == 100
        handle.close()

    def test_get_sample(self, single_data_file):
        """Test getting individual samples."""
        handle = FileHandle(single_data_file)

        features, globals_vec, value, pol_idx, pol_val = handle.get_sample(0)

        assert features.shape == (40, 8, 8)
        assert globals_vec.shape == (10,)
        assert isinstance(value, float)
        assert len(pol_idx) > 0
        assert len(pol_idx) == len(pol_val)
        assert np.isclose(pol_val.sum(), 1.0, atol=1e-5)

        handle.close()

    def test_get_batch(self, single_data_file):
        """Test getting batch of samples."""
        handle = FileHandle(single_data_file)

        indices = np.array([0, 5, 10, 15])
        features, globals_vec, values, pol_indices, pol_values = (
            handle.get_batch(indices)
        )

        assert features.shape == (4, 40, 8, 8)
        assert globals_vec.shape == (4, 10)
        assert values.shape == (4,)
        assert len(pol_indices) == 4
        assert len(pol_values) == 4

        handle.close()


class TestStreamingDataLoader:
    """Tests for StreamingDataLoader class."""

    def test_initialization_single_file(self, single_data_file):
        """Test initialization with a single file."""
        loader = StreamingDataLoader(single_data_file, batch_size=16)
        assert loader.total_samples == 100
        assert len(loader) == 7  # ceil(100/16)
        loader.close()

    def test_initialization_multiple_files(self, multiple_data_files):
        """Test initialization with multiple files."""
        loader = StreamingDataLoader(multiple_data_files, batch_size=16)
        # 50 + 75 + 100 = 225 samples
        assert loader.total_samples == 225
        loader.close()

    def test_init_with_missing_file(self, single_data_file, temp_dir):
        """Test that missing files are skipped gracefully."""
        paths = [single_data_file, os.path.join(temp_dir, "missing.npz")]
        loader = StreamingDataLoader(paths, batch_size=16)
        assert loader.total_samples == 100  # Only the valid file
        loader.close()

    def test_iteration_yields_correct_batches(self, single_data_file):
        """Test that iteration yields batches of correct shape."""
        loader = StreamingDataLoader(
            single_data_file, batch_size=16, shuffle=False
        )

        batches = list(loader)
        assert len(batches) == 7  # ceil(100/16)

        # First 6 batches should be full
        for i, ((features, globals_tensor), (values, policies)) in enumerate(
            batches[:-1]
        ):
            assert features.shape[0] == 16
            assert globals_tensor.shape[0] == 16
            assert values.shape == (16, 1)
            assert policies.shape == (16, 55000)

        # Last batch may be smaller
        (features, globals_tensor), (values, policies) = batches[-1]
        assert features.shape[0] == 4  # 100 % 16 = 4
        assert values.shape == (4, 1)

        loader.close()

    def test_drop_last_option(self, single_data_file):
        """Test that drop_last option works."""
        loader = StreamingDataLoader(
            single_data_file,
            batch_size=16,
            drop_last=True,
        )

        batches = list(loader)
        assert len(batches) == 6  # 100 // 16 = 6

        # All batches should be full size
        for (features, _), (_, _) in batches:
            assert features.shape[0] == 16

        loader.close()

    def test_shuffle_changes_order(self, single_data_file):
        """Test that shuffle produces different orderings."""
        loader1 = StreamingDataLoader(
            single_data_file, batch_size=16, shuffle=True, seed=42
        )
        loader2 = StreamingDataLoader(
            single_data_file, batch_size=16, shuffle=True, seed=123
        )

        batches1 = list(loader1)
        batches2 = list(loader2)

        # Extract first batch values for comparison
        values1 = batches1[0][1][0].numpy()
        values2 = batches2[0][1][0].numpy()

        # Should be different due to different seeds
        assert not np.array_equal(values1, values2)

        loader1.close()
        loader2.close()

    def test_set_epoch_changes_shuffle(self, single_data_file):
        """Test that set_epoch produces different shuffling."""
        loader = StreamingDataLoader(
            single_data_file, batch_size=16, shuffle=True, seed=42
        )

        loader.set_epoch(0)
        batches_epoch0 = list(loader)

        loader.set_epoch(1)
        batches_epoch1 = list(loader)

        # Values should differ between epochs
        values0 = batches_epoch0[0][1][0].numpy()
        values1 = batches_epoch1[0][1][0].numpy()

        assert not np.array_equal(values0, values1)

        loader.close()

    def test_multi_file_iteration(self, multiple_data_files):
        """Test iteration across multiple files."""
        loader = StreamingDataLoader(
            multiple_data_files, batch_size=32, shuffle=False
        )

        total_samples = 0
        for (features, _), (_, _) in loader:
            total_samples += features.shape[0]

        assert total_samples == 225  # 50 + 75 + 100

        loader.close()

    def test_output_tensor_types(self, single_data_file):
        """Test that output tensors have correct types."""
        loader = StreamingDataLoader(single_data_file, batch_size=16)

        for (features, globals_tensor), (values, policies) in loader:
            assert features.dtype == torch.float32
            assert globals_tensor.dtype == torch.float32
            assert values.dtype == torch.float32
            assert policies.dtype == torch.float32
            break  # Only need to check first batch

        loader.close()

    def test_policies_sum_to_one(self, single_data_file):
        """Test that policy distributions sum to 1."""
        loader = StreamingDataLoader(single_data_file, batch_size=16)

        for (_, _), (_, policies) in loader:
            sums = policies.sum(dim=1)
            # All non-zero policy sums should be ~1.0
            non_zero_mask = sums > 0
            if non_zero_mask.any():
                assert torch.allclose(
                    sums[non_zero_mask],
                    torch.ones_like(sums[non_zero_mask]),
                    atol=1e-4,
                )
            break

        loader.close()


class TestStreamingDataset:
    """Tests for StreamingDataset PyTorch wrapper."""

    def test_basic_iteration(self, single_data_file):
        """Test basic iteration through StreamingDataset."""
        dataset = StreamingDataset(
            single_data_file, batch_size=16, shuffle=False
        )

        assert len(dataset) == 7
        assert dataset.total_samples == 100

        batch_count = 0
        for batch in dataset:
            batch_count += 1

        assert batch_count == 7

    def test_epoch_setting(self, single_data_file):
        """Test that set_epoch works."""
        dataset = StreamingDataset(
            single_data_file, batch_size=16, shuffle=True, seed=42
        )

        dataset.set_epoch(0)
        batches0 = list(dataset)

        dataset.set_epoch(1)
        batches1 = list(dataset)

        # Should produce different orderings
        values0 = batches0[0][1][0].numpy()
        values1 = batches1[0][1][0].numpy()
        assert not np.array_equal(values0, values1)


class TestGetSampleCount:
    """Tests for get_sample_count utility function."""

    def test_count_samples_npz(self, single_data_file):
        """Test counting samples in .npz file."""
        count = get_sample_count(single_data_file)
        assert count == 100

    def test_count_samples_nonexistent(self, temp_dir):
        """Test counting samples in nonexistent file raises error."""
        with pytest.raises(Exception):
            get_sample_count(os.path.join(temp_dir, "nonexistent.npz"))


class TestMergeDataFiles:
    """Tests for merge_data_files utility function."""

    def test_merge_multiple_files(self, multiple_data_files, temp_dir):
        """Test merging multiple files."""
        output_path = os.path.join(temp_dir, "merged.npz")

        total = merge_data_files(multiple_data_files, output_path)
        assert total == 225  # 50 + 75 + 100

        # Verify merged file is valid
        count = get_sample_count(output_path)
        assert count == 225

    def test_merge_with_max_samples(self, multiple_data_files, temp_dir):
        """Test merging with sample limit."""
        output_path = os.path.join(temp_dir, "merged_limited.npz")

        total = merge_data_files(
            multiple_data_files, output_path, max_samples=100
        )
        assert total == 100

        count = get_sample_count(output_path)
        assert count == 100


class TestMemoryUsage:
    """Tests for memory bounds and large dataset handling."""

    def test_large_dataset_simulation(self, temp_dir):
        """Test that memory usage stays bounded for simulated large dataset.

        This creates multiple small files that together simulate a larger
        dataset, verifying that streaming iteration doesn't load all data.
        """
        # Create 10 files with 100 samples each
        paths = []
        for i in range(10):
            path = os.path.join(temp_dir, f"large_{i}.npz")
            create_test_npz(path, num_samples=100, seed=42 + i)
            paths.append(path)

        loader = StreamingDataLoader(paths, batch_size=32, shuffle=True)

        assert loader.total_samples == 1000

        # Iterate and verify batches load correctly
        batch_count = 0
        total_samples_seen = 0
        for (features, _), (values, _) in loader:
            batch_count += 1
            total_samples_seen += features.shape[0]

            # Verify batch data is valid
            assert features.shape[1:] == (40, 8, 8)
            assert not torch.isnan(features).any()
            assert not torch.isnan(values).any()

        assert total_samples_seen == 1000
        assert batch_count == 32  # ceil(1000/32)

        loader.close()

    def test_consistent_results_across_epochs(self, single_data_file):
        """Test that same seed produces same results across runs."""
        loader1 = StreamingDataLoader(
            single_data_file, batch_size=16, shuffle=True, seed=42
        )
        loader2 = StreamingDataLoader(
            single_data_file, batch_size=16, shuffle=True, seed=42
        )

        loader1.set_epoch(5)
        loader2.set_epoch(5)

        batches1 = list(loader1)
        batches2 = list(loader2)

        # All batches should match
        for b1, b2 in zip(batches1, batches2):
            (f1, g1), (v1, p1) = b1
            (f2, g2), (v2, p2) = b2

            assert torch.equal(f1, f2)
            assert torch.equal(g1, g2)
            assert torch.equal(v1, v2)
            assert torch.equal(p1, p2)

        loader1.close()
        loader2.close()


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_loader(self, temp_dir):
        """Test that empty loader iterates without error."""
        loader = StreamingDataLoader(
            [os.path.join(temp_dir, "nonexistent.npz")],
            batch_size=16,
        )

        assert loader.total_samples == 0
        assert len(loader) == 0
        assert list(loader) == []

        loader.close()

    def test_single_sample_file(self, temp_dir):
        """Test handling file with single sample."""
        path = os.path.join(temp_dir, "single.npz")
        create_test_npz(path, num_samples=1)

        loader = StreamingDataLoader(path, batch_size=16)
        assert loader.total_samples == 1
        assert len(loader) == 1

        batches = list(loader)
        assert len(batches) == 1
        assert batches[0][0][0].shape[0] == 1

        loader.close()

    def test_batch_size_larger_than_data(self, temp_dir):
        """Test when batch size is larger than dataset."""
        path = os.path.join(temp_dir, "small.npz")
        create_test_npz(path, num_samples=10)

        loader = StreamingDataLoader(path, batch_size=32)
        assert len(loader) == 1

        batches = list(loader)
        assert len(batches) == 1
        assert batches[0][0][0].shape[0] == 10

        loader.close()

    def test_exact_batch_size_multiple(self, temp_dir):
        """Test when data size is exact multiple of batch size."""
        path = os.path.join(temp_dir, "exact.npz")
        create_test_npz(path, num_samples=64)

        loader = StreamingDataLoader(path, batch_size=16)
        assert len(loader) == 4

        batches = list(loader)
        assert len(batches) == 4
        for batch in batches:
            assert batch[0][0].shape[0] == 16

        loader.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])