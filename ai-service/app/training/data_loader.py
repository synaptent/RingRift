"""
Streaming DataLoader for RingRift Training Infrastructure

This module provides disk-backed streaming data loading that doesn't load
the entire dataset into RAM, enabling scalable neural network training
beyond the previous 50K sample cap.

Key Features:
- Memory-mapped file reading for .npz files
- Optional HDF5 support for larger datasets
- Batch loading with configurable batch size
- Index shuffling per epoch (shuffles indices, not data)
- Multi-file support for distributed data
- Lazy sample counting without loading full data
"""

import logging
import os
from typing import Any, Iterator, List, Optional, Tuple, Union

import numpy as np

try:
    import h5py
    HDF5_AVAILABLE = True
except ImportError:
    h5py = None  # type: ignore
    HDF5_AVAILABLE = False

import torch
from torch.utils.data import IterableDataset

logger = logging.getLogger(__name__)


class FileHandle:
    """
    Wrapper for memory-mapped data file access.

    Supports .npz (numpy compressed) and .hdf5 formats with lazy loading
    and memory-efficient access patterns.
    """

    def __init__(self, path: str, filter_empty_policies: bool = True):
        """
        Initialize a file handle for the given data file.

        Args:
            path: Path to data file (.npz or .hdf5)
            filter_empty_policies: If True, filters out samples with empty
                policy arrays (terminal states) to prevent NaN losses
        """
        self.path = path
        self.filter_empty_policies = filter_empty_policies
        self._data: Any = None
        self._format: Optional[str] = None
        self._total_samples = 0
        self._valid_indices: np.ndarray = np.array([], dtype=np.int64)

        self._open()

    def _open(self) -> None:
        """Open file and determine format."""
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Data file not found: {self.path}")

        ext = os.path.splitext(self.path)[1].lower()

        if ext in ('.npz', '.npy'):
            self._format = 'npz'
            # Use mmap_mode='r' for memory-efficient access
            self._data = np.load(
                self.path,
                mmap_mode='r',
                allow_pickle=True,
            )

            if 'features' not in self._data:
                raise ValueError(
                    f"Invalid data format in {self.path}: "
                    "missing 'features' key"
                )

            self._total_samples = len(self._data['values'])

            # Build valid indices (filtering empty policies if requested)
            if self.filter_empty_policies and 'policy_indices' in self._data:
                policy_indices_arr = self._data['policy_indices']
                self._valid_indices = np.array([
                    i for i in range(self._total_samples)
                    if len(policy_indices_arr[i]) > 0
                ], dtype=np.int64)

                filtered = self._total_samples - len(self._valid_indices)
                if filtered > 0:
                    logger.debug(
                        f"Filtered {filtered} terminal states from "
                        f"{self.path}"
                    )
            else:
                self._valid_indices = np.arange(
                    self._total_samples, dtype=np.int64
                )

        elif ext in ('.h5', '.hdf5'):
            if not HDF5_AVAILABLE or h5py is None:
                raise ImportError(
                    "h5py is required for HDF5 support. "
                    "Install with: pip install h5py"
                )
            self._format = 'hdf5'
            self._data = h5py.File(self.path, 'r')

            if 'features' not in self._data:
                raise ValueError(
                    f"Invalid data format in {self.path}: "
                    "missing 'features' key"
                )

            self._total_samples = len(self._data['values'])

            # For HDF5, we need different handling of sparse policies
            # They might be stored differently - handle gracefully
            if self.filter_empty_policies and 'policy_indices' in self._data:
                # HDF5 may store sparse policies as variable-length arrays
                # or as separate datasets. Check and handle appropriately.
                try:
                    policy_indices_ds = self._data['policy_indices']
                    self._valid_indices = np.array([
                        i for i in range(self._total_samples)
                        if len(policy_indices_ds[i]) > 0
                    ], dtype=np.int64)
                except Exception:
                    # If we can't filter, use all indices
                    self._valid_indices = np.arange(
                        self._total_samples, dtype=np.int64
                    )
            else:
                self._valid_indices = np.arange(
                    self._total_samples, dtype=np.int64
                )
        else:
            raise ValueError(f"Unsupported file format: {ext}")

        logger.debug(
            f"Opened {self.path}: {len(self._valid_indices)} valid samples "
            f"out of {self._total_samples} total"
        )

    @property
    def num_samples(self) -> int:
        """Number of valid samples in this file."""
        return len(self._valid_indices)

    def get_sample(self, idx: int) -> Tuple[
        np.ndarray, np.ndarray, float, np.ndarray, np.ndarray
    ]:
        """
        Get a single sample by index.

        Args:
            idx: Index into valid_indices array

        Returns:
            Tuple of (features, globals, value, policy_indices, policy_values)
        """
        if self._data is None:
            raise RuntimeError("File handle is closed")

        # Map to actual data index through valid_indices
        actual_idx = int(self._valid_indices[idx])

        if self._format == 'npz':
            # Memory-mapped access - copies the specific slice
            features = np.array(self._data['features'][actual_idx])
            globals_vec = np.array(self._data['globals'][actual_idx])
            value = float(self._data['values'][actual_idx])
            policy_indices = np.asarray(
                self._data['policy_indices'][actual_idx]
            )
            policy_values = np.asarray(
                self._data['policy_values'][actual_idx]
            )
        else:
            # HDF5 access
            features = np.array(self._data['features'][actual_idx])
            globals_vec = np.array(self._data['globals'][actual_idx])
            value = float(self._data['values'][actual_idx])
            policy_indices = np.asarray(
                self._data['policy_indices'][actual_idx]
            )
            policy_values = np.asarray(
                self._data['policy_values'][actual_idx]
            )

        return features, globals_vec, value, policy_indices, policy_values

    def get_batch(
        self, indices: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List, List]:
        """
        Get a batch of samples by indices.

        Args:
            indices: Array of indices into valid_indices

        Returns:
            Tuple of (features_batch, globals_batch, values_batch,
                     policy_indices_list, policy_values_list)
        """
        if self._data is None:
            raise RuntimeError("File handle is closed")

        # Map to actual data indices
        actual_indices = self._valid_indices[indices]

        if self._format == 'npz':
            # For npz with mmap, we need to load each index
            # (can't do fancy indexing on mmap'd arrays efficiently)
            features_list = []
            globals_list = []
            values_list: List[float] = []
            policy_indices_list: List[np.ndarray] = []
            policy_values_list: List[np.ndarray] = []

            for actual_idx in actual_indices:
                idx = int(actual_idx)
                features_list.append(
                    np.array(self._data['features'][idx])
                )
                globals_list.append(
                    np.array(self._data['globals'][idx])
                )
                values_list.append(float(self._data['values'][idx]))
                policy_indices_list.append(
                    np.asarray(self._data['policy_indices'][idx])
                )
                policy_values_list.append(
                    np.asarray(self._data['policy_values'][idx])
                )

            features_batch = np.stack(features_list, axis=0)
            globals_batch = np.stack(globals_list, axis=0)
            values_batch = np.array(values_list, dtype=np.float32)

        else:
            # HDF5 supports fancy indexing
            sorted_indices = np.sort(actual_indices)
            # Need to reorder after loading
            reorder = np.argsort(np.argsort(actual_indices))

            features_batch = np.array(
                self._data['features'][sorted_indices.tolist()]
            )[reorder]
            globals_batch = np.array(
                self._data['globals'][sorted_indices.tolist()]
            )[reorder]
            values_batch = np.array(
                self._data['values'][sorted_indices.tolist()],
                dtype=np.float32
            )[reorder]

            policy_indices_list = [
                np.asarray(self._data['policy_indices'][int(i)])
                for i in actual_indices
            ]
            policy_values_list = [
                np.asarray(self._data['policy_values'][int(i)])
                for i in actual_indices
            ]

        return (
            features_batch,
            globals_batch,
            values_batch,
            policy_indices_list,
            policy_values_list,
        )
    
    def close(self) -> None:
        """Close file handle and release resources."""
        if self._data is not None:
            if self._format == 'hdf5' and hasattr(self._data, 'close'):
                self._data.close()
            self._data = None

    def __del__(self) -> None:
        self.close()


class StreamingDataLoader:
    """
    Streaming DataLoader for large-scale training data.

    Implements disk-backed streaming that doesn't load entire dataset into RAM.
    Supports multiple data files, batch loading, and index shuffling per epoch.

    Example usage:
        >>> loader = StreamingDataLoader(
        ...     data_paths=['data1.npz', 'data2.npz'],
        ...     batch_size=32,
        ...     shuffle=True
        ... )
        >>> for states_batch, labels_batch in loader:
        ...     # states_batch is (features, globals)
        ...     # labels_batch is (values, policies)
        ...     train_step(states_batch, labels_batch)
    """

    def __init__(
        self,
        data_paths: Union[str, List[str]],
        batch_size: int = 32,
        shuffle: bool = True,
        filter_empty_policies: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False,
        policy_size: int = 55000,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize the streaming data loader.

        Args:
            data_paths: Path(s) to data files (.npz or .hdf5)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle at the start of each epoch
            filter_empty_policies: Filter terminal states with empty policies
            seed: Random seed for reproducible shuffling
            drop_last: Whether to drop the last incomplete batch
            policy_size: Size of dense policy vector (default 55000)
            rank: Process rank for distributed training (0-indexed)
            world_size: Total number of processes for distributed training
        """
        if isinstance(data_paths, str):
            data_paths = [data_paths]

        self.data_paths = data_paths
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.filter_empty_policies = filter_empty_policies
        self.seed = seed
        self.drop_last = drop_last
        self.policy_size = policy_size
        self.rank = rank
        self.world_size = world_size

        self._rng = np.random.default_rng(seed)
        self._file_handles: List[FileHandle] = []
        self._file_offsets: List[int] = []  # Cumulative sample counts
        self._total_samples = 0

        self._initialize_files()

    def _initialize_files(self) -> None:
        """Open all data files and compute sample counts."""
        self._file_handles = []
        self._file_offsets = [0]
        self._total_samples = 0

        for path in self.data_paths:
            if not os.path.exists(path):
                logger.warning(f"Data file not found, skipping: {path}")
                continue

            try:
                handle = FileHandle(
                    path,
                    filter_empty_policies=self.filter_empty_policies
                )
                self._file_handles.append(handle)
                self._total_samples += handle.num_samples
                self._file_offsets.append(self._total_samples)

                logger.info(
                    f"Loaded file: {path} ({handle.num_samples} samples)"
                )
            except Exception as e:
                logger.warning(f"Failed to open {path}: {e}")

        if not self._file_handles:
            logger.warning("No valid data files found")
        else:
            logger.info(
                f"StreamingDataLoader initialized: {self._total_samples} "
                f"total samples across {len(self._file_handles)} files"
            )

    def _global_to_file_index(self, global_idx: int) -> Tuple[int, int]:
        """
        Convert global sample index to (file_index, local_index).

        Uses binary search for efficient lookup.
        """
        file_idx = int(np.searchsorted(
            self._file_offsets[1:], global_idx, side='right'
        ))
        local_idx = global_idx - self._file_offsets[file_idx]
        return file_idx, local_idx

    def _sparse_to_dense_policy(
        self,
        indices: np.ndarray,
        values: np.ndarray
    ) -> np.ndarray:
        """Convert sparse policy representation to dense vector."""
        policy = np.zeros(self.policy_size, dtype=np.float32)
        if len(indices) > 0:
            indices_arr = np.asarray(indices, dtype=np.int64)
            values_arr = np.asarray(values, dtype=np.float32)
            policy[indices_arr] = values_arr
        return policy

    def __iter__(self) -> Iterator[Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
    ]]:
        """
        Iterate over batches of data.

        In distributed mode, each process gets a unique subset of the data
        based on its rank. The data is sharded so that process with rank `r`
        gets samples at indices r, r + world_size, r + 2*world_size, etc.

        Yields:
            Tuple of ((features, globals), (values, policies)):
            - features: torch.Tensor of shape (batch_size, C, H, W)
            - globals: torch.Tensor of shape (batch_size, G)
            - values: torch.Tensor of shape (batch_size, 1)
            - policies: torch.Tensor of shape (batch_size, policy_size)
        """
        if self._total_samples == 0:
            return

        # Create index array for this epoch
        indices = np.arange(self._total_samples)

        if self.shuffle:
            self._rng.shuffle(indices)

        # In distributed mode, shard the indices
        # Each rank gets every world_size-th sample starting at rank
        if self.world_size > 1:
            indices = indices[self.rank::self.world_size]

        # Determine number of complete batches for this shard
        shard_size = len(indices)
        if self.drop_last:
            num_batches = shard_size // self.batch_size
        else:
            num_batches = (shard_size + self.batch_size - 1) // self.batch_size

        if num_batches == 0:
            return

        for batch_idx in range(num_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, shard_size)
            batch_indices = indices[start_idx:end_idx]

            # Skip incomplete batch if drop_last is True
            if self.drop_last and len(batch_indices) < self.batch_size:
                continue

            # Group indices by file for efficient batch loading
            # Dictionary: file_idx -> list of (batch_position, local_idx)
            file_groups: dict = {}
            for batch_pos, global_idx in enumerate(batch_indices):
                f_idx, l_idx = self._global_to_file_index(int(global_idx))
                if f_idx not in file_groups:
                    file_groups[f_idx] = []
                file_groups[f_idx].append((batch_pos, l_idx))

            # Pre-allocate batch arrays
            batch_size_actual = len(batch_indices)
            features_batch = np.zeros((batch_size_actual, 1), dtype=np.float32)
            globals_batch = np.zeros((batch_size_actual, 1), dtype=np.float32)
            values_batch = np.zeros(batch_size_actual, dtype=np.float32)
            policies_batch = np.zeros(
                (batch_size_actual, self.policy_size), dtype=np.float32
            )
            initialized = False

            # Load from each file
            for file_idx, positions in file_groups.items():
                handle = self._file_handles[file_idx]
                local_indices = np.array([p[1] for p in positions])

                # Load batch from this file
                features, globals_vec, values, pol_indices, pol_values = \
                    handle.get_batch(local_indices)

                # Initialize arrays on first file with correct shape
                if not initialized:
                    features_batch = np.zeros(
                        (batch_size_actual,) + features.shape[1:],
                        dtype=np.float32
                    )
                    globals_batch = np.zeros(
                        (batch_size_actual,) + globals_vec.shape[1:],
                        dtype=np.float32
                    )
                    initialized = True

                # Place samples in correct batch positions
                for i, (batch_pos, _) in enumerate(positions):
                    features_batch[batch_pos] = features[i]
                    globals_batch[batch_pos] = globals_vec[i]
                    values_batch[batch_pos] = values[i]
                    policies_batch[batch_pos] = self._sparse_to_dense_policy(
                        pol_indices[i], pol_values[i]
                    )

            # Convert to torch tensors
            features_tensor = torch.from_numpy(features_batch)
            globals_tensor = torch.from_numpy(globals_batch)
            values_tensor = torch.from_numpy(values_batch).unsqueeze(1)
            policies_tensor = torch.from_numpy(policies_batch)

            yield (
                (features_tensor, globals_tensor),
                (values_tensor, policies_tensor)
            )

    def __len__(self) -> int:
        """Total number of batches per epoch for this rank's shard."""
        if self._total_samples == 0:
            return 0

        # Calculate shard size (samples for this rank)
        if self.world_size > 1:
            # Each rank gets approximately total_samples / world_size samples
            total = self._total_samples
            shard_size = (total + self.world_size - 1) // self.world_size
        else:
            shard_size = self._total_samples

        if self.drop_last:
            return shard_size // self.batch_size
        return (shard_size + self.batch_size - 1) // self.batch_size

    @property
    def shard_size(self) -> int:
        """Number of samples in this rank's shard."""
        if self.world_size > 1:
            # Calculate exact shard size for this rank
            base_size = self._total_samples // self.world_size
            remainder = self._total_samples % self.world_size
            # Earlier ranks get one extra sample if there's a remainder
            if self.rank < remainder:
                return base_size + 1
            return base_size
        return self._total_samples

    @property
    def total_samples(self) -> int:
        """Total samples across all files without loading them."""
        return self._total_samples

    def set_epoch(self, epoch: int) -> None:
        """
        Set the epoch for shuffling.

        Call this at the start of each epoch to ensure different shuffling
        when using a fixed seed.

        Args:
            epoch: Current epoch number
        """
        if self.seed is not None:
            self._rng = np.random.default_rng(self.seed + epoch)

    def close(self) -> None:
        """Close all file handles and release resources."""
        for handle in self._file_handles:
            handle.close()
        self._file_handles = []

    def __del__(self) -> None:
        self.close()


class StreamingDataset(IterableDataset):  # type: ignore[type-arg]
    """
    PyTorch IterableDataset wrapper for StreamingDataLoader.

    This allows integration with PyTorch DataLoader for features like
    multi-worker loading and automatic batching.

    Example usage:
        >>> dataset = StreamingDataset(['data.npz'], shuffle=True)
        >>> loader = DataLoader(dataset, batch_size=None)
        >>> for batch in loader:
        ...     train_step(batch)
    """

    def __init__(
        self,
        data_paths: Union[str, List[str]],
        batch_size: int = 32,
        shuffle: bool = True,
        filter_empty_policies: bool = True,
        seed: Optional[int] = None,
        drop_last: bool = False,
        policy_size: int = 55000,
        rank: int = 0,
        world_size: int = 1,
    ):
        """
        Initialize the streaming dataset.

        Args:
            data_paths: Path(s) to data files (.npz or .hdf5)
            batch_size: Number of samples per batch
            shuffle: Whether to shuffle at the start of each epoch
            filter_empty_policies: Filter terminal states with empty policies
            seed: Random seed for reproducible shuffling
            drop_last: Whether to drop the last incomplete batch
            policy_size: Size of dense policy vector
            rank: Process rank for distributed training (0-indexed)
            world_size: Total number of processes for distributed training
        """
        self.loader = StreamingDataLoader(
            data_paths=data_paths,
            batch_size=batch_size,
            shuffle=shuffle,
            filter_empty_policies=filter_empty_policies,
            seed=seed,
            drop_last=drop_last,
            policy_size=policy_size,
            rank=rank,
            world_size=world_size,
        )
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        """Set the epoch for shuffling."""
        self._epoch = epoch
        self.loader.set_epoch(epoch)

    def __iter__(self) -> Iterator[Tuple[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor]
    ]]:
        """Iterate over batches."""
        self.loader.set_epoch(self._epoch)
        return iter(self.loader)

    def __len__(self) -> int:
        """Total number of batches."""
        return len(self.loader)

    @property
    def total_samples(self) -> int:
        """Total samples in the dataset."""
        return self.loader.total_samples


def get_sample_count(data_path: str) -> int:
    """
    Get the number of samples in a data file without loading the full data.

    This is useful for dataset statistics and memory planning.

    Args:
        data_path: Path to .npz or .hdf5 data file

    Returns:
        Number of samples in the file
    """
    ext = os.path.splitext(data_path)[1].lower()

    if ext in ('.npz', '.npy'):
        # Load with mmap and check only the values array shape
        data = np.load(data_path, mmap_mode='r', allow_pickle=True)
        if 'values' in data:
            return len(data['values'])
        elif 'features' in data:
            return len(data['features'])
        return 0

    elif ext in ('.h5', '.hdf5'):
        if not HDF5_AVAILABLE or h5py is None:
            raise ImportError("h5py is required for HDF5 support")
        with h5py.File(data_path, 'r') as f:
            if 'values' in f:
                return len(f['values'])  # type: ignore
            elif 'features' in f:
                return len(f['features'])  # type: ignore
        return 0

    raise ValueError(f"Unsupported file format: {ext}")


def merge_data_files(
    input_paths: List[str],
    output_path: str,
    max_samples: Optional[int] = None,
    use_hdf5: bool = False,
) -> int:
    """
    Merge multiple data files into a single file.

    Useful for consolidating distributed training data or converting
    between formats.

    Args:
        input_paths: List of input file paths
        output_path: Path for merged output file
        max_samples: Optional maximum number of samples to include
        use_hdf5: If True, output as HDF5; otherwise NPZ

    Returns:
        Number of samples in merged file
    """
    # Collect all data
    all_features: List[np.ndarray] = []
    all_globals: List[np.ndarray] = []
    all_values: List[float] = []
    all_policy_indices: List[np.ndarray] = []
    all_policy_values: List[np.ndarray] = []

    total_count = 0

    for path in input_paths:
        if not os.path.exists(path):
            logger.warning(f"File not found, skipping: {path}")
            continue

        handle = FileHandle(path, filter_empty_policies=True)

        for i in range(handle.num_samples):
            if max_samples is not None and total_count >= max_samples:
                break

            feat, glob, val, pol_idx, pol_val = handle.get_sample(i)
            all_features.append(feat)
            all_globals.append(glob)
            all_values.append(val)
            all_policy_indices.append(pol_idx)
            all_policy_values.append(pol_val)
            total_count += 1

        handle.close()

        if max_samples is not None and total_count >= max_samples:
            break

    if total_count == 0:
        logger.warning("No samples found in input files")
        return 0

    # Convert to arrays
    features_arr = np.array(all_features, dtype=np.float32)
    globals_arr = np.array(all_globals, dtype=np.float32)
    values_arr = np.array(all_values, dtype=np.float32)
    policy_indices_arr = np.array(all_policy_indices, dtype=object)
    policy_values_arr = np.array(all_policy_values, dtype=object)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if use_hdf5:
        if not HDF5_AVAILABLE or h5py is None:
            raise ImportError("h5py is required for HDF5 output")

        with h5py.File(output_path, 'w') as f:
            f.create_dataset(
                'features', data=features_arr, compression='gzip'
            )
            f.create_dataset(
                'globals', data=globals_arr, compression='gzip'
            )
            f.create_dataset(
                'values', data=values_arr, compression='gzip'
            )
            # HDF5 variable-length arrays for sparse policies
            dt = h5py.vlen_dtype(np.dtype('int32'))
            f.create_dataset(
                'policy_indices', (total_count,), dtype=dt,
                data=[
                    np.asarray(x, dtype=np.int32)
                    for x in policy_indices_arr
                ]
            )
            dt_f = h5py.vlen_dtype(np.dtype('float32'))
            f.create_dataset(
                'policy_values', (total_count,), dtype=dt_f,
                data=[
                    np.asarray(x, dtype=np.float32)
                    for x in policy_values_arr
                ]
            )
    else:
        np.savez_compressed(
            output_path,
            features=features_arr,
            globals=globals_arr,
            values=values_arr,
            policy_indices=policy_indices_arr,
            policy_values=policy_values_arr,
        )

    logger.info(f"Merged {total_count} samples to {output_path}")
    return total_count