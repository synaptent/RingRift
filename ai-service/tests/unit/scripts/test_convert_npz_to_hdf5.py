"""
Tests for NPZ to HDF5 conversion utility.

These tests verify that the conversion script correctly converts training data
from NPZ format to HDF5 format with data integrity preserved.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest

# Check if h5py is available
try:
    import h5py
    HAS_H5PY = True
except ImportError:
    HAS_H5PY = False

# Import conversion functions
from scripts.convert_npz_to_hdf5 import (
    convert_npz_to_hdf5,
    convert_directory,
    compute_array_checksum,
    HAS_H5PY as SCRIPT_HAS_H5PY,
)


@pytest.fixture
def sample_npz_data() -> dict[str, np.ndarray]:
    """Create sample training data similar to real RingRift data."""
    num_samples = 100
    feature_size = (14, 8, 8)  # channels x board_size x board_size
    globals_size = 20
    policy_size = 55000

    # Dense features and globals
    features = np.random.randn(num_samples, *feature_size).astype(np.float32)
    globals_vec = np.random.randn(num_samples, globals_size).astype(np.float32)
    values = np.random.uniform(-1, 1, num_samples).astype(np.float32)

    # Sparse policies (variable length per sample)
    policy_indices = np.empty(num_samples, dtype=object)
    policy_values = np.empty(num_samples, dtype=object)

    for i in range(num_samples):
        # Random number of non-zero policy entries (10-100)
        num_moves = np.random.randint(10, 100)
        policy_indices[i] = np.random.choice(
            policy_size, size=num_moves, replace=False
        ).astype(np.int32)
        policy_values[i] = np.random.dirichlet(
            np.ones(num_moves)
        ).astype(np.float32)

    # String object array (like phases, victory_types)
    phases = np.array(['ring_placement', 'main_game', 'endgame'] * (num_samples // 3 + 1),
                      dtype=object)[:num_samples]

    # Scalar values (metadata)
    board_type = np.array('square8')  # scalar string
    board_size = np.array(8, dtype=np.int64)  # scalar int

    return {
        'features': features,
        'globals': globals_vec,
        'values': values,
        'policy_indices': policy_indices,
        'policy_values': policy_values,
        'phases': phases,
        'board_type': board_type,
        'board_size': board_size,
    }


@pytest.fixture
def sample_npz_file(sample_npz_data: dict[str, np.ndarray], tmp_path: Path) -> Path:
    """Save sample data to NPZ file."""
    npz_path = tmp_path / "test_data.npz"
    np.savez(npz_path, **sample_npz_data)
    return npz_path


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestConvertNpzToHdf5:
    """Test cases for NPZ to HDF5 conversion."""

    def test_basic_conversion(
        self, sample_npz_file: Path, tmp_path: Path
    ):
        """Test basic conversion creates valid HDF5 file."""
        hdf5_path = tmp_path / "output.h5"

        result = convert_npz_to_hdf5(
            npz_path=sample_npz_file,
            hdf5_path=hdf5_path,
        )

        assert result['success'] is True
        assert hdf5_path.exists()
        assert result['num_samples'] == 100
        assert result['num_keys'] == 8  # features, globals, values, policy_*, phases, board_*

    def test_conversion_preserves_dense_data(
        self, sample_npz_file: Path, sample_npz_data: dict[str, np.ndarray],
        tmp_path: Path
    ):
        """Test that dense arrays are preserved exactly."""
        hdf5_path = tmp_path / "output.h5"

        convert_npz_to_hdf5(
            npz_path=sample_npz_file,
            hdf5_path=hdf5_path,
        )

        with h5py.File(hdf5_path, 'r') as hf:
            np.testing.assert_array_equal(
                hf['features'][:], sample_npz_data['features']
            )
            np.testing.assert_array_equal(
                hf['globals'][:], sample_npz_data['globals']
            )
            np.testing.assert_array_almost_equal(
                hf['values'][:], sample_npz_data['values'], decimal=5
            )

    def test_conversion_preserves_sparse_data(
        self, sample_npz_file: Path, sample_npz_data: dict[str, np.ndarray],
        tmp_path: Path
    ):
        """Test that sparse policy arrays are preserved."""
        hdf5_path = tmp_path / "output.h5"

        convert_npz_to_hdf5(
            npz_path=sample_npz_file,
            hdf5_path=hdf5_path,
        )

        with h5py.File(hdf5_path, 'r') as hf:
            for i in range(10):  # Check first 10 samples
                np.testing.assert_array_equal(
                    hf['policy_indices'][i],
                    sample_npz_data['policy_indices'][i]
                )
                np.testing.assert_array_almost_equal(
                    hf['policy_values'][i],
                    sample_npz_data['policy_values'][i],
                    decimal=5
                )

    def test_conversion_preserves_string_object_arrays(
        self, sample_npz_file: Path, sample_npz_data: dict[str, np.ndarray],
        tmp_path: Path
    ):
        """Test that string object arrays are preserved."""
        hdf5_path = tmp_path / "output.h5"

        convert_npz_to_hdf5(
            npz_path=sample_npz_file,
            hdf5_path=hdf5_path,
        )

        with h5py.File(hdf5_path, 'r') as hf:
            for i in range(10):  # Check first 10 samples
                hdf5_val = hf['phases'][i]
                # HDF5 returns bytes, decode to str
                if isinstance(hdf5_val, bytes):
                    hdf5_val = hdf5_val.decode('utf-8')
                assert hdf5_val == sample_npz_data['phases'][i]

    def test_conversion_preserves_scalar_values(
        self, sample_npz_file: Path, sample_npz_data: dict[str, np.ndarray],
        tmp_path: Path
    ):
        """Test that scalar values (strings and numerics) are preserved."""
        hdf5_path = tmp_path / "output.h5"

        convert_npz_to_hdf5(
            npz_path=sample_npz_file,
            hdf5_path=hdf5_path,
        )

        with h5py.File(hdf5_path, 'r') as hf:
            # Scalar string
            board_type = hf['board_type'][()]
            if isinstance(board_type, bytes):
                board_type = board_type.decode('utf-8')
            assert board_type == sample_npz_data['board_type'].item()

            # Scalar numeric
            assert hf['board_size'][()] == sample_npz_data['board_size'].item()

    def test_conversion_with_verification(
        self, sample_npz_file: Path, tmp_path: Path
    ):
        """Test conversion with checksum verification."""
        hdf5_path = tmp_path / "output.h5"

        result = convert_npz_to_hdf5(
            npz_path=sample_npz_file,
            hdf5_path=hdf5_path,
            verify=True,
        )

        assert result['success'] is True
        assert result.get('verified') is True

    def test_conversion_with_compression(
        self, sample_npz_file: Path, tmp_path: Path
    ):
        """Test conversion with gzip compression."""
        hdf5_path = tmp_path / "output.h5"

        result = convert_npz_to_hdf5(
            npz_path=sample_npz_file,
            hdf5_path=hdf5_path,
            compress=True,
        )

        assert result['success'] is True
        # Compressed file should generally be smaller
        # (though random data may not compress well)
        assert hdf5_path.exists()

    def test_hdf5_metadata(
        self, sample_npz_file: Path, tmp_path: Path
    ):
        """Test that HDF5 file contains proper metadata."""
        hdf5_path = tmp_path / "output.h5"

        convert_npz_to_hdf5(
            npz_path=sample_npz_file,
            hdf5_path=hdf5_path,
        )

        with h5py.File(hdf5_path, 'r') as hf:
            assert hf.attrs['num_samples'] == 100
            assert hf.attrs['format_version'] == '1.0'
            assert 'source_file' in hf.attrs
            assert 'converted_at' in hf.attrs

    def test_missing_input_file(self, tmp_path: Path):
        """Test error handling for missing input file."""
        with pytest.raises(FileNotFoundError):
            convert_npz_to_hdf5(
                npz_path=tmp_path / "nonexistent.npz",
                hdf5_path=tmp_path / "output.h5",
            )

    def test_invalid_npz_format(self, tmp_path: Path):
        """Test error handling for NPZ without 'features' key."""
        bad_npz = tmp_path / "bad.npz"
        np.savez(bad_npz, other_data=np.zeros(10))

        with pytest.raises(ValueError, match="missing 'features' key"):
            convert_npz_to_hdf5(
                npz_path=bad_npz,
                hdf5_path=tmp_path / "output.h5",
            )


@pytest.mark.skipif(not HAS_H5PY, reason="h5py not installed")
class TestConvertDirectory:
    """Test cases for batch directory conversion."""

    def test_directory_conversion(self, sample_npz_data: dict, tmp_path: Path):
        """Test converting multiple NPZ files in a directory."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()

        # Create multiple NPZ files
        for i in range(3):
            npz_path = input_dir / f"data_{i}.npz"
            np.savez(npz_path, **sample_npz_data)

        result = convert_directory(
            input_dir=input_dir,
            output_dir=output_dir,
        )

        assert result['success'] is True
        assert result['files_converted'] == 3
        assert result['files_failed'] == 0
        assert (output_dir / "data_0.h5").exists()
        assert (output_dir / "data_1.h5").exists()
        assert (output_dir / "data_2.h5").exists()

    def test_skip_existing(self, sample_npz_data: dict, tmp_path: Path):
        """Test that existing HDF5 files are skipped."""
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create NPZ files
        for i in range(2):
            npz_path = input_dir / f"data_{i}.npz"
            np.savez(npz_path, **sample_npz_data)

        # Pre-create one HDF5 file
        with h5py.File(output_dir / "data_0.h5", 'w') as hf:
            hf.create_dataset('dummy', data=[1, 2, 3])

        result = convert_directory(
            input_dir=input_dir,
            output_dir=output_dir,
            skip_existing=True,
        )

        assert result['files_converted'] == 1
        assert result['files_skipped'] == 1

    def test_empty_directory(self, tmp_path: Path):
        """Test handling of empty input directory."""
        input_dir = tmp_path / "empty"
        input_dir.mkdir()

        result = convert_directory(
            input_dir=input_dir,
            output_dir=tmp_path / "output",
        )

        assert result['success'] is False
        assert 'No NPZ files found' in result.get('error', '')


class TestComputeArrayChecksum:
    """Test cases for checksum computation."""

    def test_checksum_deterministic(self):
        """Test that checksums are deterministic."""
        arr = np.random.randn(100, 10).astype(np.float32)
        cs1 = compute_array_checksum(arr)
        cs2 = compute_array_checksum(arr)
        assert cs1 == cs2

    def test_checksum_different_data(self):
        """Test that different data produces different checksums."""
        arr1 = np.ones((10, 10), dtype=np.float32)
        arr2 = np.zeros((10, 10), dtype=np.float32)
        assert compute_array_checksum(arr1) != compute_array_checksum(arr2)

    def test_checksum_length(self):
        """Test checksum is truncated to 16 chars."""
        arr = np.random.randn(10).astype(np.float32)
        cs = compute_array_checksum(arr)
        assert len(cs) == 16
