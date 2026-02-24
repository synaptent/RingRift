"""
PyTorch datasets for RingRift training data.

Extracted from train.py (December 2025) for modularity.

Classes:
    RingRiftDataset: Base dataset for self-play positions
    WeightedRingRiftDataset: Dataset with position-weighted sampling
"""

import logging
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from app.ai.neural_net import (
    HEX8_BOARD_SIZE,
    HEX_BOARD_SIZE,
    get_policy_size_for_board,
)
from app.models import BoardType
from app.training.hex_augmentation import HexSymmetryTransform
from app.utils.numpy_utils import safe_load_npz

logger = logging.getLogger(__name__)

# Minimum number of samples required for meaningful training.
# Fewer than this produces unreliable gradients and garbage models.
MIN_SAMPLES_FOR_TRAINING = 100


class RingRiftDataset(Dataset):
    """
    Dataset of self-play positions for a single board geometry.

    Each .npz file is expected to be *homogeneous* in board type/size so that
    mini-batches drawn by a vanilla DataLoader contain only one spatial shape.
    This keeps the training input compatible with the CNN and with
    NeuralNetAI.evaluate_batch, which enforces same-board-per-batch semantics.

    Future multi-board runs can either:
      - use separate datasets per board type/size, or
      - introduce a higher-level sampler/collate_fn that groups samples by
        geometry before feeding them to the network.

    Note: Terminal states (samples with empty policy arrays) can either be
    filtered out during loading or retained with masked policy loss,
    depending on the training configuration.

    Args:
        data_path: Path to the .npz training data file
        board_type: Board geometry type (for augmentation)
        augment_hex: Enable D6 symmetry augmentation for hex boards
    """

    def __init__(
        self,
        data_path: str,
        board_type: BoardType = BoardType.SQUARE8,
        augment_hex: bool = False,
        use_multi_player_values: bool = False,
        filter_empty_policies: bool = True,
        return_num_players: bool = False,
        return_auxiliary_targets: bool = False,
        return_heuristics: bool = False,
    ):
        self.data_path = data_path
        self.board_type = board_type
        self.augment_hex = augment_hex and board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
        # When True and the underlying dataset provides 'values_mp' and
        # 'num_players', __getitem__ will surface vector value targets
        # suitable for multi-player value heads.
        self.use_multi_player_values = use_multi_player_values
        self.filter_empty_policies = filter_empty_policies
        self.return_num_players = return_num_players
        # When True, includes auxiliary targets (game_length, piece_count, outcome)
        # for multi-task learning if available in the data.
        self.return_auxiliary_targets = return_auxiliary_targets
        # When True and 'heuristics' array exists, includes heuristic features
        # for v5-heavy model training (49 hand-crafted features per state).
        self.return_heuristics = return_heuristics
        self.hex_transform: HexSymmetryTransform | None = None

        # Initialize hex transform if augmentation enabled
        if self.augment_hex:
            # Set board_size based on board_type: hex8 uses 9x9, hexagonal uses 25x25
            hex_board_size = 9 if board_type == BoardType.HEX8 else 25
            self.hex_transform = HexSymmetryTransform(board_size=hex_board_size)
            logger.info(f"Hex symmetry augmentation enabled (D6 group, board_size={hex_board_size})")

        self.length = 0
        # Memory-mapped file object (np.lib.npyio.NpzFile) or in-memory dict
        self.data = None
        # Track if policy data is available (value-only datasets won't have it)
        self.has_policy = True
        # Optional metadata inferred from the underlying npz file to aid
        # future multi-board training tooling.
        self.spatial_shape = None  # (H, W) of feature maps, if known
        self.board_type_meta = None
        self.board_size_meta = None
        # List of valid sample indices (filtered when empty-policy filtering is enabled)
        self.valid_indices = None
        # Multi-player value support metadata
        self.has_multi_player_values = False
        self.num_players_arr: np.ndarray | None = None
        # Effective dense policy vector length inferred from data
        self.policy_size: int = 0
        # Auxiliary task targets (multi-task learning)
        self.has_auxiliary_targets = False
        self.game_lengths_arr: np.ndarray | None = None
        self.piece_counts_arr: np.ndarray | None = None
        self.outcomes_arr: np.ndarray | None = None
        # Heuristic features for v5-heavy training (December 2025)
        self.has_heuristics = False
        self.heuristics_arr: np.ndarray | None = None
        self.num_heuristic_features: int = 0

        if os.path.exists(data_path):
            try:
                # Load data and cache arrays in memory. NPZ files don't support
                # true mmap (they're zip files), so each array access would
                # re-read from disk. For training efficiency, we load everything
                # into RAM upfront. For very large datasets, consider using
                # HDF5 or raw .npy files instead.
                # safe_load_npz tries without pickle first for security
                npz_data = safe_load_npz(data_path)
                # Convert to dict to force-load all arrays into memory
                # Skip metadata arrays that require pickle (game_ids, phases, move_types)
                # These are not needed for training - only the numerical arrays are used
                training_keys = {'features', 'globals', 'values', 'values_mp',
                                'policy_indices', 'policy_values', 'move_numbers',
                                'total_game_moves', 'num_players', 'player_numbers',
                                'quality_score', 'heuristics', 'opponent_elo',
                                'board_type', 'board_size', 'history_length',
                                'feature_version', 'policy_encoding', 'encoder_version',
                                'in_channels', 'export_version', 'encoder_type',
                                'base_channels', 'spatial_size', 'policy_head_size'}
                self.data = {}
                pickle_reload_needed = False
                for k in npz_data.keys():
                    if k in training_keys:
                        try:
                            self.data[k] = np.asarray(npz_data[k])
                        except (ValueError, TypeError) as e:
                            # Object arrays (like policy_indices/policy_values) need allow_pickle=True
                            if "allow_pickle=False" in str(e) or "pickle" in str(e).lower():
                                pickle_reload_needed = True
                            else:
                                logger.debug(f"Skipping array '{k}' due to load error: {e}")

                # If we had pickle errors, reload with allow_pickle=True
                if pickle_reload_needed:
                    logger.info(f"Reloading {data_path} with allow_pickle=True for object arrays")
                    npz_data = np.load(data_path, allow_pickle=True)
                    for k in npz_data.keys():
                        if k in training_keys and k not in self.data:
                            try:
                                self.data[k] = np.asarray(npz_data[k])
                            except Exception as e:
                                logger.debug(f"Skipping array '{k}' after pickle reload: {e}")
                    # else: skip metadata arrays like game_ids, phases, move_types

                # ================================================================
                # Early validation: Fail fast on encoder/dimension mismatches
                # Added Dec 2025 to catch issues at load time, not forward pass
                # ================================================================
                available_keys = set(self.data.keys())

                # Extract encoder metadata for validation
                self._encoder_type = None
                self._base_channels = None
                self._in_channels_meta = None
                self._spatial_size_meta = None
                self._policy_head_size_meta = None

                if "encoder_type" in available_keys:
                    raw = self.data["encoder_type"]
                    self._encoder_type = str(raw.item() if raw.ndim == 0 else raw)
                if "base_channels" in available_keys:
                    raw = self.data["base_channels"]
                    self._base_channels = int(raw.item() if raw.ndim == 0 else raw)
                if "in_channels" in available_keys:
                    raw = self.data["in_channels"]
                    self._in_channels_meta = int(raw.item() if raw.ndim == 0 else raw)
                if "spatial_size" in available_keys:
                    raw = self.data["spatial_size"]
                    self._spatial_size_meta = int(raw.item() if raw.ndim == 0 else raw)
                if "policy_head_size" in available_keys:
                    raw = self.data["policy_head_size"]
                    self._policy_head_size_meta = int(raw.item() if raw.ndim == 0 else raw)

                # Validate feature dimensions if metadata available
                if 'features' in self.data:
                    feat_shape = self.data['features'].shape
                    if len(feat_shape) >= 4:
                        actual_channels = feat_shape[1]
                        actual_h, actual_w = feat_shape[2], feat_shape[3]

                        # Validate channel count matches in_channels metadata
                        if self._in_channels_meta is not None:
                            if actual_channels != self._in_channels_meta:
                                raise ValueError(
                                    f"========================================\n"
                                    f"CHANNEL MISMATCH IN NPZ DATA\n"
                                    f"========================================\n"
                                    f"File: {data_path}\n"
                                    f"Features have {actual_channels} channels\n"
                                    f"Metadata says in_channels={self._in_channels_meta}\n"
                                    f"Encoder type: {self._encoder_type or 'unknown'}\n"
                                    f"\n"
                                    f"This indicates corrupted or inconsistent data.\n"
                                    f"========================================"
                                )

                        # Validate spatial dimensions match metadata
                        if self._spatial_size_meta is not None:
                            if actual_h != self._spatial_size_meta or actual_w != self._spatial_size_meta:
                                raise ValueError(
                                    f"========================================\n"
                                    f"SPATIAL DIMENSION MISMATCH IN NPZ DATA\n"
                                    f"========================================\n"
                                    f"File: {data_path}\n"
                                    f"Features have {actual_h}x{actual_w} spatial size\n"
                                    f"Metadata says spatial_size={self._spatial_size_meta}\n"
                                    f"Board type: {board_type.name}\n"
                                    f"\n"
                                    f"This indicates board type mismatch or data corruption.\n"
                                    f"========================================"
                                )

                        # Cross-validate spatial_size with board_type metadata
                        if self._spatial_size_meta is not None and board_type is not None:
                            expected_sizes = {
                                "SQUARE8": 8, "SQUARE19": 19, "HEX8": 9, "HEXAGONAL": 25,
                                "square8": 8, "square19": 19, "hex8": 9, "hexagonal": 25,
                            }
                            board_type_str = board_type.name if hasattr(board_type, 'name') else str(board_type)
                            expected_for_board = expected_sizes.get(board_type_str)
                            if expected_for_board is not None and self._spatial_size_meta != expected_for_board:
                                raise ValueError(
                                    f"========================================\n"
                                    f"METADATA INCONSISTENCY IN NPZ DATA\n"
                                    f"========================================\n"
                                    f"File: {data_path}\n"
                                    f"board_type metadata: {board_type_str}\n"
                                    f"spatial_size metadata: {self._spatial_size_meta}\n"
                                    f"Expected spatial_size for {board_type_str}: {expected_for_board}\n\n"
                                    f"This data was likely exported with incorrect encoder.\n"
                                    f"Re-export with the correct board-type-aware encoder.\n"
                                    f"========================================"
                                )

                        # Log validation success
                        logger.debug(
                            f"NPZ validation passed: {actual_channels}ch, "
                            f"{actual_h}x{actual_w} spatial, "
                            f"encoder={self._encoder_type or 'unknown'}"
                        )

                if 'features' in self.data:
                    total_samples = len(self.data['values'])

                    # Check if policy data is available (value-only datasets won't have it)
                    self.has_policy = 'policy_indices' in self.data and 'policy_values' in self.data
                    policy_indices_arr = None  # Initialize for later use in policy_size inference
                    if not self.has_policy:
                        logger.info(
                            f"No policy data in {data_path} - value-only training mode"
                        )
                        # For value-only training, all samples are valid
                        self.valid_indices = list(range(total_samples))
                    elif self.filter_empty_policies:
                        # Optionally filter out samples with empty policies
                        # (terminal states). When disabled, training will
                        # mask policy loss for those samples instead.
                        policy_indices_arr = self.data['policy_indices']
                        self.valid_indices = [
                            i for i in range(total_samples)
                            if len(policy_indices_arr[i]) > 0
                        ]

                        filtered_count = total_samples - len(self.valid_indices)
                        if filtered_count > 0:
                            logger.info(
                                f"Filtered {filtered_count} terminal states "
                                f"with empty policies out of {total_samples} "
                                f"total samples"
                            )
                    else:
                        # has_policy=True but filter_empty_policies=False
                        # Still need policy_indices_arr for policy_size inference
                        policy_indices_arr = self.data['policy_indices']
                        self.valid_indices = list(range(total_samples))

                    self.length = len(self.valid_indices)

                    if self.length == 0:
                        logger.warning(
                            f"All {total_samples} samples in {data_path} "
                            f"have empty policies (terminal states). "
                            f"Dataset is empty."
                        )
                    else:
                        logger.info(
                            f"Loaded {self.length} valid training samples "
                            f"from {data_path} (in-memory)"
                        )

                    # Optional per-dataset metadata for multi-board training.
                    # Newer datasets may include scalar or per-sample arrays
                    # named 'board_type' and/or 'board_size'. Older datasets
                    # will simply omit these keys.
                    # Note: available_keys was already set above during validation
                    if "board_type" in available_keys:
                        self.board_type_meta = self.data["board_type"]
                    if "board_size" in available_keys:
                        self.board_size_meta = self.data["board_size"]

                    # Dec 28, 2025: Validate board_type metadata matches constructor arg
                    # This prevents cross-config contamination (training hex8 with square8 data)
                    if self.board_type_meta is not None:
                        # Extract board_type string from metadata (may be 0-d array)
                        meta_raw = self.board_type_meta
                        if hasattr(meta_raw, 'item'):
                            meta_board_type_str = str(meta_raw.item())
                        elif hasattr(meta_raw, '__iter__') and len(meta_raw) > 0:
                            meta_board_type_str = str(meta_raw.flat[0])
                        else:
                            meta_board_type_str = str(meta_raw)

                        # Normalize to lowercase for comparison
                        meta_board_type_str = meta_board_type_str.lower().strip()
                        expected_board_type_str = (
                            board_type.name.lower() if hasattr(board_type, 'name')
                            else str(board_type).lower()
                        )

                        if meta_board_type_str != expected_board_type_str:
                            raise ValueError(
                                f"========================================\n"
                                f"CROSS-CONFIG CONTAMINATION DETECTED\n"
                                f"========================================\n"
                                f"File: {data_path}\n"
                                f"NPZ board_type metadata: '{meta_board_type_str}'\n"
                                f"Expected board_type: '{expected_board_type_str}'\n"
                                f"\n"
                                f"This data was exported for a different board type.\n"
                                f"Training with this data would produce a garbage model.\n"
                                f"\n"
                                f"Either:\n"
                                f"  1. Use --board-type {meta_board_type_str} to match the data\n"
                                f"  2. Re-export data with the correct board type\n"
                                f"========================================"
                            )

                    # Multi-player value targets: optional 'values_mp'
                    # (N, MAX_PLAYERS) and 'num_players' (N,) arrays.
                    if "values_mp" in available_keys and "num_players" in available_keys:
                        self.has_multi_player_values = True
                        self.num_players_arr = np.asarray(
                            self.data["num_players"],
                            dtype=np.int32,
                        )
                    elif self.return_num_players:
                        # Dataset doesn't include num_players metadata; disable
                        # return to avoid incompatible batching in multi-player loss.
                        logger.info(
                            "num_players metadata missing in %s; disabling return_num_players",
                            data_path,
                        )
                        self.return_num_players = False

                    # Auxiliary task targets: optional 'game_lengths', 'piece_counts',
                    # 'outcomes' arrays for multi-task learning (2025-12).
                    has_game_lengths = "game_lengths" in available_keys
                    has_piece_counts = "piece_counts" in available_keys
                    has_outcomes = "outcomes" in available_keys
                    if has_game_lengths or has_piece_counts or has_outcomes:
                        self.has_auxiliary_targets = True
                        if has_game_lengths:
                            self.game_lengths_arr = np.asarray(
                                self.data["game_lengths"], dtype=np.float32
                            )
                        if has_piece_counts:
                            self.piece_counts_arr = np.asarray(
                                self.data["piece_counts"], dtype=np.float32
                            )
                        if has_outcomes:
                            self.outcomes_arr = np.asarray(
                                self.data["outcomes"], dtype=np.int64
                            )
                        logger.info(
                            "Auxiliary targets available in %s: game_lengths=%s, "
                            "piece_counts=%s, outcomes=%s",
                            data_path,
                            has_game_lengths,
                            has_piece_counts,
                            has_outcomes,
                        )
                    elif self.return_auxiliary_targets:
                        logger.info(
                            "Auxiliary targets not found in %s; will derive from values",
                            data_path,
                        )

                    # Heuristic features for v5-heavy training (December 2025)
                    if "heuristics" in available_keys:
                        self.has_heuristics = True
                        self.heuristics_arr = np.asarray(
                            self.data["heuristics"], dtype=np.float32
                        )
                        self.num_heuristic_features = self.heuristics_arr.shape[1]
                        logger.info(
                            "Heuristic features available in %s: %d features per sample",
                            data_path,
                            self.num_heuristic_features,
                        )
                    elif self.return_heuristics:
                        logger.warning(
                            "Heuristics requested but not found in %s; "
                            "will return zeros. Re-export with --include-heuristics.",
                            data_path,
                        )

                    # Infer the canonical spatial shape (H, W) once so that
                    # callers can route samples into same-board batches if
                    # mixed-geometry datasets are ever introduced.
                    try:
                        # Use first valid sample if available
                        if self.valid_indices:
                            first_valid = self.valid_indices[0]
                            sample = self.data["features"][first_valid]
                        else:
                            sample = self.data["features"][0]
                        if sample.ndim >= 3:
                            self.spatial_shape = tuple(sample.shape[-2:])
                    except (KeyError, IndexError, AttributeError, TypeError):
                        # Best-effort only; training will still work as long
                        # as individual samples are well-formed.
                        self.spatial_shape = None

                    # Infer effective policy_size from sparse indices.
                    try:
                        max_index = -1
                        if policy_indices_arr is not None:
                            for i in self.valid_indices or []:
                                indices = np.asarray(
                                    policy_indices_arr[i],
                                    dtype=np.int64,
                                )
                                if indices.size == 0:
                                    continue
                                local_max = int(indices.max())
                                if local_max > max_index:
                                    max_index = local_max
                        if max_index >= 0:
                            self.policy_size = max_index + 1
                            logger.info(
                                "Inferred policy_size=%d from %s",
                                self.policy_size,
                                data_path,
                            )
                        else:
                            # Fallback to board-default if no non-empty policies
                            self.policy_size = get_policy_size_for_board(
                                self.board_type
                            )
                    except Exception as e:
                        logger.warning(
                            "Failed to infer policy_size from %s (%s); "
                            "falling back to board default.",
                            data_path,
                            e,
                        )
                        self.policy_size = get_policy_size_for_board(
                            self.board_type
                        )
                else:
                    print("Invalid data format in npz")
                    self.length = 0
            except Exception as e:
                print(f"Error loading data: {e}")
                self.length = 0

            # Validate minimum sample count to prevent training on insufficient data.
            # Too few samples produce unreliable gradients and garbage models.
            if self.length > 0 and self.length < MIN_SAMPLES_FOR_TRAINING:
                raise ValueError(
                    f"Dataset too small: {self.length} samples "
                    f"(min={MIN_SAMPLES_FOR_TRAINING}). "
                    f"Refusing to train on insufficient data. "
                    f"File: {data_path}"
                )
        else:
            print(f"Data file {data_path} not found, generating dummy data")
            # Generate dummy data in memory for testing
            # Ensure all dummy samples have non-empty policies
            dummy_count = 100
            # Use board-appropriate policy size instead of hardcoded value
            dummy_policy_size = get_policy_size_for_board(self.board_type)
            # Keep demo dummy data structurally compatible with the default
            # model constructors in train_model():
            # - Square boards: 14 base channels × (history_length+1=4) = 56
            # - Hex boards:    10 base channels × (history_length+1=4) = 40
            dummy_input_channels = 40 if self.board_type in (BoardType.HEXAGONAL, BoardType.HEX8) else 56
            # Model expects 20 global features (see neural_net.py global_features default)
            dummy_global_features = 20
            if self.board_type == BoardType.SQUARE19:
                dummy_h = 19
                dummy_w = 19
            elif self.board_type == BoardType.HEXAGONAL:
                dummy_h = HEX_BOARD_SIZE
                dummy_w = HEX_BOARD_SIZE
            elif self.board_type == BoardType.HEX8:
                dummy_h = HEX8_BOARD_SIZE
                dummy_w = HEX8_BOARD_SIZE
            else:
                dummy_h = 8
                dummy_w = 8
            self.data = {
                'features': np.random.rand(
                    dummy_count, dummy_input_channels, dummy_h, dummy_w
                ).astype(np.float32),
                'globals': np.random.rand(dummy_count, dummy_global_features).astype(np.float32),
                'values': np.random.choice(
                    [1.0, 0.0, -1.0],
                    size=dummy_count,
                ).astype(np.float32),
                'policy_indices': np.array([
                    np.random.choice(dummy_policy_size, 5, replace=False).astype(np.int32)
                    for _ in range(dummy_count)
                ], dtype=object),
                'policy_values': np.array([
                    np.random.rand(5).astype(np.float32)
                    for _ in range(dummy_count)
                ], dtype=object),
            }
            # Use board-appropriate policy size
            self.policy_size = dummy_policy_size
            self.valid_indices = list(range(dummy_count))
            self.length = dummy_count
            self.spatial_shape = (dummy_h, dummy_w)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.length == 0:
            raise IndexError("Dataset is empty")

        if self.data is None:
            raise RuntimeError(
                "RingRiftDataset backing store is not initialised. "
                "This usually indicates a failed load."
            )

        # Map through valid_indices to get actual data index
        # This skips terminal states with empty policies
        if self.valid_indices is not None:
            actual_idx = self.valid_indices[idx]
        else:
            actual_idx = idx

        # Access data from memory-mapped arrays. We copy to ensure we have a
        # writable tensor if needed, and to detach from the mmap backing
        # store.
        features = np.array(self.data['features'][actual_idx])
        globals_vec = np.array(self.data['globals'][actual_idx])
        value = np.array(self.data['values'][actual_idx])

        # Policy is stored as object array of arrays (sparse). mmap does not
        # support object arrays directly, so these may be fully loaded into
        # memory depending on how the npz was written. For very large datasets
        # a CSR-style encoding would be preferable, but for now we assume the
        # object array fits in memory or is handled by OS paging.
        # Handle value-only datasets (no policy data)
        if self.has_policy:
            policy_indices = self.data['policy_indices'][actual_idx]
            policy_values = self.data['policy_values'][actual_idx]
        else:
            policy_indices = np.array([], dtype=np.int32)
            policy_values = np.array([], dtype=np.float32)

        # Apply hex symmetry augmentation on-the-fly if enabled
        # This expands effective dataset size by 12x without extra memory
        if self.augment_hex and self.hex_transform is not None:
            # Pick a random transformation from the D6 group (0-11)
            transform_id = random.randint(0, 11)

            if transform_id != 0:  # 0 is identity, skip for efficiency
                # Transform the feature tensor
                features = self.hex_transform.transform_board(
                    features, transform_id
                )

                # Transform sparse policy (only if we have policy data)
                if self.has_policy and len(policy_indices) > 0:
                    indices_arr = np.asarray(policy_indices, dtype=np.int32)
                    values_arr = np.asarray(policy_values, dtype=np.float32)
                    policy_indices, policy_values = (
                        self.hex_transform.transform_sparse_policy(
                            indices_arr, values_arr, transform_id
                        )
                    )

        # Reconstruct dense policy vector on-the-fly.
        # When empty policies are allowed, the vector may remain all zeros.
        if self.policy_size <= 0:
            # Defensive fallback; should not normally happen
            self.policy_size = get_policy_size_for_board(self.board_type)
        policy_vector = torch.zeros(self.policy_size, dtype=torch.float32)

        if len(policy_indices) > 0:
            # Convert to proper numpy arrays with correct dtype
            # The object array may contain arrays that need explicit casting
            indices_arr = np.asarray(policy_indices, dtype=np.int64)
            values_arr = np.asarray(policy_values, dtype=np.float32)

            # Phase 4 Validation: Check indices are in valid range
            max_valid_idx = self.policy_size - 1
            if np.any(indices_arr < 0) or np.any(indices_arr > max_valid_idx):
                invalid_indices = indices_arr[(indices_arr < 0) | (indices_arr > max_valid_idx)]
                raise ValueError(
                    f"Policy index out of range [0, {max_valid_idx}] at sample {idx}: "
                    f"invalid indices = {invalid_indices[:5].tolist()}"
                )

            policy_vector[indices_arr] = torch.from_numpy(values_arr)

            # Phase 4 Validation: Check policy vector normalization
            policy_sum = policy_vector.sum().item()
            if not (0.99 < policy_sum < 1.01):
                # Only raise error for severely denormalized vectors
                if policy_sum < 0.5 or policy_sum > 1.5:
                    raise ValueError(
                        f"Policy vector severely denormalized at sample {idx}: "
                        f"sum={policy_sum:.6f}, "
                        f"indices={indices_arr[:5].tolist()}, "
                        f"values={values_arr[:5].tolist()}"
                    )

        # Scalar vs multi-player value targets:
        # - Scalar: shape (1,) tensor containing float
        # - Multi-player: shape (MAX_PLAYERS,) tensor from 'values_mp'
        if self.use_multi_player_values and self.has_multi_player_values:
            values_mp = np.asarray(
                self.data["values_mp"][actual_idx],
                dtype=np.float32,
            )
            value_tensor = torch.from_numpy(values_mp)
        else:
            value_tensor = torch.tensor(
                [value.item()],
                dtype=torch.float32,
            )

        # Build auxiliary targets dict if requested
        aux_targets = None
        if self.return_auxiliary_targets:
            aux_targets = {}
            # Game length (normalized by /100 for stability)
            if self.game_lengths_arr is not None:
                aux_targets["game_length"] = torch.tensor(
                    self.game_lengths_arr[actual_idx] / 100.0,
                    dtype=torch.float32,
                )
            # Piece count (normalized)
            if self.piece_counts_arr is not None:
                aux_targets["piece_count"] = torch.tensor(
                    self.piece_counts_arr[actual_idx] / 20.0,  # Normalize by typical max
                    dtype=torch.float32,
                )
            # Outcome (0=loss, 1=draw, 2=win)
            if self.outcomes_arr is not None:
                aux_targets["outcome"] = torch.tensor(
                    self.outcomes_arr[actual_idx],
                    dtype=torch.long,
                )
            else:
                # Derive outcome from value if not pre-computed
                val = value.item()
                if val > 0.3:
                    outcome = 2  # Win
                elif val < -0.3:
                    outcome = 0  # Loss
                else:
                    outcome = 1  # Draw
                aux_targets["outcome"] = torch.tensor(outcome, dtype=torch.long)

        # Build heuristic tensor if requested (for v5-heavy training)
        heuristic_tensor = None
        if self.return_heuristics:
            if self.has_heuristics and self.heuristics_arr is not None:
                heuristic_tensor = torch.from_numpy(
                    self.heuristics_arr[actual_idx].astype(np.float32)
                )
            else:
                # Return zeros if heuristics not available in data
                # Use 21 as default (fast_heuristic_features component scores)
                num_features = self.num_heuristic_features if self.num_heuristic_features > 0 else 21
                heuristic_tensor = torch.zeros(num_features, dtype=torch.float32)

        if self.return_num_players:
            num_players_val = 0
            if self.num_players_arr is not None:
                try:
                    num_players_val = int(self.num_players_arr[actual_idx])
                except (ValueError, TypeError, IndexError):
                    num_players_val = 0
            if self.return_auxiliary_targets:
                result = (
                    torch.from_numpy(features),
                    torch.from_numpy(globals_vec),
                    value_tensor,
                    policy_vector,
                    torch.tensor(num_players_val, dtype=torch.int64),
                    aux_targets,
                )
                if self.return_heuristics:
                    result = result + (heuristic_tensor,)
                return result
            result = (
                torch.from_numpy(features),
                torch.from_numpy(globals_vec),
                value_tensor,
                policy_vector,
                torch.tensor(num_players_val, dtype=torch.int64),
            )
            if self.return_heuristics:
                result = result + (heuristic_tensor,)
            return result

        if self.return_auxiliary_targets:
            result = (
                torch.from_numpy(features),
                torch.from_numpy(globals_vec),
                value_tensor,
                policy_vector,
                aux_targets,
            )
            if self.return_heuristics:
                result = result + (heuristic_tensor,)
            return result

        result = (
            torch.from_numpy(features),
            torch.from_numpy(globals_vec),
            value_tensor,
            policy_vector,
        )
        if self.return_heuristics:
            result = result + (heuristic_tensor,)
        return result


class WeightedRingRiftDataset(RingRiftDataset):
    """
    Dataset with position-weighted sampling for curriculum learning.

    Extends RingRiftDataset to compute per-sample weights based on:
    - Game progress (late-game positions weighted higher)
    - Game phase (territory/line decisions weighted higher)

    The weights can be used with torch.utils.data.WeightedRandomSampler
    for biased sampling during training.

    Args:
        data_path: Path to the .npz training data file
        board_type: Board geometry type (for augmentation)
        augment_hex: Enable D6 symmetry augmentation for hex boards
        weighting: Weighting strategy - one of:
            - 'uniform': No weighting (weight = 1.0 for all)
            - 'late_game': Higher weight for late-game positions
            - 'phase_emphasis': Higher weight for decision phases
            - 'combined': Combines late_game and phase_emphasis
            - 'source': Weight by data source quality (Gumbel 3x, MCTS 3x, policy 1.5x)
            - 'combined_source': Combines late_game, phase_emphasis, AND source weighting
                Recommended for AlphaZero-style training with mixed data sources.
    """

    # Phase weights for phase_emphasis and combined strategies
    PHASE_WEIGHTS = {
        # Canonical GamePhase values (snake_case)
        'ring_placement': 0.8,
        'movement': 1.0,
        'capture': 1.2,
        'chain_capture': 1.3,
        'line_processing': 1.5,
        'territory_processing': 1.5,
        # Final cleanup phase when a player is blocked with stacks but has
        # no legal placements, movements, or captures. We weight this in the
        # same band as other decision/cleanup phases so that forced-elimination
        # samples participate normally in phase-emphasis curricula.
        'forced_elimination': 1.5,
        # Legacy / alias names (for backwards compatibility)
        'RING_PLACEMENT': 0.8,
        'MOVEMENT': 1.0,
        'CAPTURE': 1.2,
        'CHAIN_CAPTURE': 1.3,
        'LINE_DECISION': 1.5,
        'TERRITORY_DECISION': 1.5,
        'FORCED_ELIMINATION': 1.5,
        'ring_movement': 1.0,
        'line_decision': 1.5,
        'territory_decision': 1.5,
    }

    def __init__(
        self,
        data_path: str,
        board_type: BoardType = BoardType.SQUARE8,
        augment_hex: bool = False,
        weighting: str = 'late_game',
        use_multi_player_values: bool = False,
        filter_empty_policies: bool = True,
        return_num_players: bool = False,
        return_heuristics: bool = False,
    ):
        super().__init__(
            data_path,
            board_type,
            augment_hex,
            use_multi_player_values=use_multi_player_values,
            filter_empty_policies=filter_empty_policies,
            return_num_players=return_num_players,
            return_heuristics=return_heuristics,
        )

        self.weighting = weighting
        self.sample_weights: np.ndarray | None = None

        if self.length > 0:
            self._compute_weights()

    def _compute_weights(self) -> None:
        """Compute per-sample weights based on weighting strategy."""
        weights = np.ones(self.length, dtype=np.float32)

        # Load metadata if available
        move_numbers = None
        total_game_moves = None
        phases = None

        if self.data is not None:
            if 'move_numbers' in self.data:
                move_numbers = self.data['move_numbers']
            if 'total_game_moves' in self.data:
                total_game_moves = self.data['total_game_moves']
            if 'phases' in self.data:
                phases = self.data['phases']

        if self.weighting == 'uniform':
            # No weighting
            pass

        elif self.weighting == 'late_game':
            # Weight positions higher toward end of game
            # w = 0.5 + 0.5 * (move_num / total_moves)
            if move_numbers is not None and total_game_moves is not None:
                for i, orig_idx in enumerate(self.valid_indices):
                    move_num = move_numbers[orig_idx]
                    total = max(total_game_moves[orig_idx], 1)
                    progress = move_num / total
                    weights[i] = 0.5 + 0.5 * progress
            else:
                logger.warning(
                    "late_game weighting requested but move_numbers/total_game_moves "
                    "not in dataset. Using uniform weights."
                )

        elif self.weighting == 'phase_emphasis':
            # Boost territory/line decision phases
            if phases is not None:
                for i, orig_idx in enumerate(self.valid_indices):
                    phase = str(phases[orig_idx])
                    weights[i] = self.PHASE_WEIGHTS.get(phase, 1.0)
            else:
                logger.warning(
                    "phase_emphasis weighting requested but phases not in dataset. "
                    "Using uniform weights."
                )

        elif self.weighting == 'combined':
            # Combine late_game and phase_emphasis
            late_game_available = (
                move_numbers is not None and total_game_moves is not None
            )
            phase_available = phases is not None

            for i, orig_idx in enumerate(self.valid_indices):
                weight = 1.0

                # Late game factor
                if late_game_available:
                    move_num = move_numbers[orig_idx]
                    total = max(total_game_moves[orig_idx], 1)
                    progress = move_num / total
                    weight *= (0.5 + 0.5 * progress)

                # Phase factor
                if phase_available:
                    phase = str(phases[orig_idx])
                    weight *= self.PHASE_WEIGHTS.get(phase, 1.0)

                weights[i] = weight

        elif self.weighting == 'source':
            # Weight by data source quality - Gumbel MCTS games get 3x weight
            # This prioritizes high-quality NN+MCTS data for self-improvement
            engine_modes = None
            if self.data is not None and 'engine_modes' in self.data:
                engine_modes = self.data['engine_modes']

            if engine_modes is not None:
                from app.training.source_weighting import get_quality_tier
                for i, orig_idx in enumerate(self.valid_indices):
                    mode = str(engine_modes[orig_idx])
                    tier = get_quality_tier(mode)
                    if tier == 'high':
                        weights[i] = 3.0  # Gumbel/MCTS
                    elif tier == 'medium':
                        weights[i] = 1.5  # Policy-only/descent
                    else:
                        weights[i] = 1.0  # Heuristic/unknown
            else:
                logger.warning(
                    "source weighting requested but engine_modes not in dataset. "
                    "Using uniform weights."
                )

        elif self.weighting == 'combined_source':
            # Combine late_game, phase_emphasis, AND source weighting
            late_game_available = (
                move_numbers is not None and total_game_moves is not None
            )
            phase_available = phases is not None
            engine_modes = None
            if self.data is not None and 'engine_modes' in self.data:
                engine_modes = self.data['engine_modes']

            from app.training.source_weighting import get_quality_tier

            for i, orig_idx in enumerate(self.valid_indices):
                weight = 1.0

                # Late game factor
                if late_game_available:
                    move_num = move_numbers[orig_idx]
                    total = max(total_game_moves[orig_idx], 1)
                    progress = move_num / total
                    weight *= (0.5 + 0.5 * progress)

                # Phase factor
                if phase_available:
                    phase = str(phases[orig_idx])
                    weight *= self.PHASE_WEIGHTS.get(phase, 1.0)

                # Source quality factor (Gumbel 3x)
                if engine_modes is not None:
                    mode = str(engine_modes[orig_idx])
                    tier = get_quality_tier(mode)
                    if tier == 'high':
                        weight *= 3.0
                    elif tier == 'medium':
                        weight *= 1.5

                weights[i] = weight

        elif self.weighting == 'chain_emphasis':
            # Emphasize chain initiation decisions, de-emphasize forced continuations
            # Chain initiations (overtaking_capture) are strategic decisions -> high weight
            # Forced continuations (continue_capture_segment) often have only 1 legal move -> low weight
            move_types = None
            if self.data is not None and 'move_types' in self.data:
                move_types = self.data['move_types']

            if move_types is not None:
                for i, orig_idx in enumerate(self.valid_indices):
                    move_type = str(move_types[orig_idx])
                    if move_type == 'overtaking_capture':
                        # Chain initiation - strategic decision worth learning
                        weights[i] = 2.0
                    elif move_type == 'continue_capture_segment':
                        # Forced continuation - often only one legal move
                        weights[i] = 0.5
                    else:
                        # Standard moves - normal weight
                        weights[i] = 1.0
            else:
                logger.warning(
                    "chain_emphasis weighting requested but move_types not in dataset. "
                    "Using uniform weights."
                )

        elif self.weighting == 'combined_chain':
            # Full combination: late_game + phase + source + chain_emphasis
            late_game_available = (
                move_numbers is not None and total_game_moves is not None
            )
            phase_available = phases is not None
            engine_modes = None
            if self.data is not None and 'engine_modes' in self.data:
                engine_modes = self.data['engine_modes']
            move_types = None
            if self.data is not None and 'move_types' in self.data:
                move_types = self.data['move_types']

            from app.training.source_weighting import get_quality_tier

            for i, orig_idx in enumerate(self.valid_indices):
                weight = 1.0

                # Late game factor (0.5 to 1.0)
                if late_game_available:
                    move_num = move_numbers[orig_idx]
                    total = max(total_game_moves[orig_idx], 1)
                    progress = move_num / total
                    weight *= (0.5 + 0.5 * progress)

                # Phase factor
                if phase_available:
                    phase = str(phases[orig_idx])
                    weight *= self.PHASE_WEIGHTS.get(phase, 1.0)

                # Source quality factor (Gumbel 3x)
                if engine_modes is not None:
                    mode = str(engine_modes[orig_idx])
                    tier = get_quality_tier(mode)
                    if tier == 'high':
                        weight *= 3.0
                    elif tier == 'medium':
                        weight *= 1.5

                # Chain emphasis factor
                if move_types is not None:
                    move_type = str(move_types[orig_idx])
                    if move_type == 'overtaking_capture':
                        weight *= 2.0  # Chain initiation
                    elif move_type == 'continue_capture_segment':
                        weight *= 0.5  # Forced continuation

                weights[i] = weight

        elif self.weighting == 'quality':
            # Weight by per-game quality score (computed during export)
            # Quality scores range [0, 1] - scale to [0.5, 2.0] for weighting
            quality_scores = None
            if self.data is not None and 'quality_score' in self.data:
                quality_scores = self.data['quality_score']

            if quality_scores is not None:
                for i, orig_idx in enumerate(self.valid_indices):
                    q = float(quality_scores[orig_idx])
                    # Scale [0, 1] to [0.5, 2.0]
                    weights[i] = 0.5 + 1.5 * q
            else:
                logger.warning(
                    "quality weighting requested but quality_score not in dataset. "
                    "Using uniform weights. Re-export with export_replay_dataset.py to include."
                )

        elif self.weighting == 'opponent_elo':
            # Weight higher for games against stronger opponents
            # Stronger opponents produce more valuable training signal
            opponent_elo = None
            if self.data is not None and 'opponent_elo' in self.data:
                opponent_elo = self.data['opponent_elo']

            if opponent_elo is not None:
                # Normalize Elo to [0, 1] range based on reasonable bounds
                # Assume Elo range is ~400 (random) to ~2000 (strong)
                min_elo, max_elo = 400.0, 2000.0
                for i, orig_idx in enumerate(self.valid_indices):
                    elo = float(opponent_elo[orig_idx])
                    normalized = (elo - min_elo) / (max_elo - min_elo)
                    normalized = max(0.0, min(1.0, normalized))  # Clamp
                    # Scale [0, 1] to [0.5, 2.0]
                    weights[i] = 0.5 + 1.5 * normalized
            else:
                logger.warning(
                    "opponent_elo weighting requested but opponent_elo not in dataset. "
                    "Using uniform weights."
                )

        elif self.weighting == 'quality_combined':
            # Full combination: quality + opponent_elo + late_game + source
            # December 2025: Highest-quality weighting combining all signals
            late_game_available = (
                move_numbers is not None and total_game_moves is not None
            )
            engine_modes = None
            if self.data is not None and 'engine_modes' in self.data:
                engine_modes = self.data['engine_modes']
            quality_scores = None
            if self.data is not None and 'quality_score' in self.data:
                quality_scores = self.data['quality_score']
            opponent_elo = None
            if self.data is not None and 'opponent_elo' in self.data:
                opponent_elo = self.data['opponent_elo']

            from app.training.source_weighting import get_quality_tier

            for i, orig_idx in enumerate(self.valid_indices):
                weight = 1.0

                # Quality score factor (0.5 to 2.0)
                if quality_scores is not None:
                    q = float(quality_scores[orig_idx])
                    weight *= (0.5 + 1.5 * q)

                # Opponent Elo factor (0.7 to 1.3 - smaller range)
                if opponent_elo is not None:
                    elo = float(opponent_elo[orig_idx])
                    normalized = (elo - 400.0) / (2000.0 - 400.0)
                    normalized = max(0.0, min(1.0, normalized))
                    weight *= (0.7 + 0.6 * normalized)

                # Late game factor (0.5 to 1.0)
                if late_game_available:
                    move_num = move_numbers[orig_idx]
                    total = max(total_game_moves[orig_idx], 1)
                    progress = move_num / total
                    weight *= (0.5 + 0.5 * progress)

                # Source quality factor (1.0 to 3.0 for Gumbel)
                if engine_modes is not None:
                    mode = str(engine_modes[orig_idx])
                    tier = get_quality_tier(mode)
                    if tier == 'high':
                        weight *= 3.0
                    elif tier == 'medium':
                        weight *= 1.5

                weights[i] = weight

        elif self.weighting == 'precomputed':
            # Use pre-computed sample weights from NPZ file (December 2025)
            # These weights combine source quality + freshness scoring during export
            # Export with: python scripts/export_replay_dataset.py --quality-weighted
            precomputed = None
            if self.data is not None and 'sample_weights' in self.data:
                precomputed = self.data['sample_weights']

            if precomputed is not None:
                for i, orig_idx in enumerate(self.valid_indices):
                    weights[i] = float(precomputed[orig_idx])
                logger.info(
                    "Using pre-computed sample weights (source quality + freshness)"
                )
            else:
                logger.warning(
                    "precomputed weighting requested but sample_weights not in dataset. "
                    "Re-export with --quality-weighted flag. Using uniform weights."
                )

        elif self.weighting == 'precomputed_combined':
            # Combine pre-computed weights with late_game and phase emphasis
            # December 2025: Best of both worlds - export-time quality + runtime heuristics
            late_game_available = (
                move_numbers is not None and total_game_moves is not None
            )
            phase_available = phases is not None
            precomputed = None
            if self.data is not None and 'sample_weights' in self.data:
                precomputed = self.data['sample_weights']

            for i, orig_idx in enumerate(self.valid_indices):
                weight = 1.0

                # Pre-computed weight (source quality + freshness)
                if precomputed is not None:
                    weight *= float(precomputed[orig_idx])

                # Late game factor (0.5 to 1.0)
                if late_game_available:
                    move_num = move_numbers[orig_idx]
                    total = max(total_game_moves[orig_idx], 1)
                    progress = move_num / total
                    weight *= (0.5 + 0.5 * progress)

                # Phase factor
                if phase_available:
                    phase = str(phases[orig_idx])
                    weight *= self.PHASE_WEIGHTS.get(phase, 1.0)

                weights[i] = weight

            if precomputed is not None:
                logger.info(
                    "Using pre-computed weights combined with late_game + phase emphasis"
                )
            else:
                logger.warning(
                    "precomputed_combined weighting requested but sample_weights not in dataset. "
                    "Only using late_game + phase emphasis."
                )

        else:
            logger.warning(
                f"Unknown weighting strategy '{self.weighting}'. Using uniform."
            )

        # Normalize weights to sum to length (maintains expected gradient scale)
        weight_sum = weights.sum()
        if weight_sum > 0:
            weights = weights * (self.length / weight_sum)

        self.sample_weights = weights
        logger.info(
            f"Computed {self.weighting} weights: "
            f"min={weights.min():.3f}, max={weights.max():.3f}, "
            f"mean={weights.mean():.3f}"
        )

    def get_sampler(self) -> 'torch.utils.data.WeightedRandomSampler':
        """
        Get a WeightedRandomSampler using the computed weights.

        Returns
        -------
        WeightedRandomSampler
            Sampler that samples indices according to computed weights.
        """
        from torch.utils.data import WeightedRandomSampler

        if self.sample_weights is None:
            weights = torch.ones(self.length)
        else:
            weights = torch.from_numpy(self.sample_weights)

        return WeightedRandomSampler(
            weights=weights,
            num_samples=self.length,
            replacement=True,
        )


class StreamingRingRiftDataset(Dataset):
    """Memory-efficient streaming dataset for very large training files.

    Unlike RingRiftDataset which loads everything into RAM, this class uses:
    - Memory-mapped numpy arrays (for .npy files)
    - HDF5 datasets (for .h5/.hdf5 files)

    This enables training on datasets larger than available RAM (10M+ samples).

    Note: NPZ files are compressed zip archives and cannot be memory-mapped.
    Convert large NPZ files to HDF5 using `convert_npz_to_hdf5()` first.

    Usage:
        # For HDF5 files (recommended for very large datasets):
        dataset = StreamingRingRiftDataset("data/training/large_dataset.h5", board_type=BoardType.HEX8)

        # Convert NPZ to HDF5:
        from app.training.datasets import convert_npz_to_hdf5
        convert_npz_to_hdf5("data/training/large.npz", "data/training/large.h5")

    Performance:
        - Initial load: O(1) - just opens file handles
        - Per-sample access: Slightly slower than RAM (~10-20%)
        - Memory usage: O(batch_size) instead of O(dataset_size)

    Args:
        data_path: Path to HDF5 file (.h5 or .hdf5)
        board_type: Board geometry type (for augmentation)
        augment_hex: Enable D6 symmetry augmentation for hex boards
    """

    def __init__(
        self,
        data_path: str,
        board_type: BoardType = BoardType.SQUARE8,
        augment_hex: bool = False,
        use_multi_player_values: bool = False,
        filter_empty_policies: bool = True,
        return_num_players: bool = False,
    ):
        self.data_path = data_path
        self.board_type = board_type
        self.augment_hex = augment_hex and board_type in (BoardType.HEXAGONAL, BoardType.HEX8)
        self.use_multi_player_values = use_multi_player_values
        self.filter_empty_policies = filter_empty_policies
        self.return_num_players = return_num_players

        self.length = 0
        self._h5_file = None
        self._features = None
        self._globals = None
        self._values = None
        self._values_mp = None
        self._policy_indices = None
        self._policy_values = None
        self._num_players = None
        self.has_policy = True
        self.has_multi_player_values = False
        self.valid_indices = None
        self.policy_size = 0
        self.hex_transform = None
        self.spatial_shape = None

        # Initialize hex transform if augmentation enabled
        if self.augment_hex:
            hex_board_size = 9 if board_type == BoardType.HEX8 else 25
            self.hex_transform = HexSymmetryTransform(board_size=hex_board_size)
            logger.info(f"Streaming dataset: Hex augmentation enabled (D6, size={hex_board_size})")

        if not os.path.exists(data_path):
            logger.error(f"Streaming dataset file not found: {data_path}")
            return

        try:
            import h5py
        except ImportError:
            logger.error("h5py required for streaming datasets. Install with: pip install h5py")
            return

        try:
            # Open HDF5 file in read mode (keeps file handle open for lazy reads)
            self._h5_file = h5py.File(data_path, 'r')

            # Get references to datasets (no data loaded yet)
            self._features = self._h5_file['features']
            self._globals = self._h5_file['globals']
            self._values = self._h5_file['values']

            # Check for policy data
            if 'policy_indices' in self._h5_file and 'policy_values' in self._h5_file:
                self._policy_indices = self._h5_file['policy_indices']
                self._policy_values = self._h5_file['policy_values']
                self.has_policy = True
            else:
                self.has_policy = False
                logger.info(f"No policy data in {data_path} - value-only streaming mode")

            # Check for multi-player values
            if 'values_mp' in self._h5_file and 'num_players' in self._h5_file:
                self._values_mp = self._h5_file['values_mp']
                self._num_players = self._h5_file['num_players']
                self.has_multi_player_values = True

            total_samples = len(self._features)

            # Build valid indices (filtering empty policies if needed)
            if not self.has_policy:
                self.valid_indices = list(range(total_samples))
            elif self.filter_empty_policies:
                # For HDF5, we need to scan policy lengths
                # This is a one-time cost at initialization
                self.valid_indices = []
                for i in range(total_samples):
                    if len(self._policy_indices[i]) > 0:
                        self.valid_indices.append(i)

                filtered = total_samples - len(self.valid_indices)
                if filtered > 0:
                    logger.info(f"Streaming: Filtered {filtered} terminal states")
            else:
                self.valid_indices = list(range(total_samples))

            self.length = len(self.valid_indices)

            # Get spatial shape from first sample
            if self.length > 0:
                sample_shape = self._features[0].shape
                if len(sample_shape) >= 3:
                    self.spatial_shape = tuple(sample_shape[-2:])

            # Infer policy size from metadata or data
            if 'policy_size' in self._h5_file.attrs:
                self.policy_size = int(self._h5_file.attrs['policy_size'])
            else:
                self.policy_size = get_policy_size_for_board(self.board_type)

            logger.info(
                f"Streaming dataset opened: {self.length} samples from {data_path} "
                f"(zero RAM loaded, policy_size={self.policy_size})"
            )

        except Exception as e:
            logger.error(f"Failed to open streaming dataset {data_path}: {e}")
            if self._h5_file is not None:
                self._h5_file.close()
            self._h5_file = None
            self.length = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.length == 0 or self._h5_file is None:
            raise IndexError("Dataset is empty or not loaded")

        # Map through valid_indices
        actual_idx = self.valid_indices[idx] if self.valid_indices else idx

        # Read only this sample from disk (HDF5 handles efficient chunk access)
        features = np.array(self._features[actual_idx])
        globals_vec = np.array(self._globals[actual_idx])
        value = np.array(self._values[actual_idx])

        # Policy data
        if self.has_policy:
            policy_indices = np.array(self._policy_indices[actual_idx])
            policy_values = np.array(self._policy_values[actual_idx])
        else:
            policy_indices = np.array([], dtype=np.int32)
            policy_values = np.array([], dtype=np.float32)

        # Apply hex augmentation
        if self.augment_hex and self.hex_transform is not None:
            transform_id = random.randint(0, 11)
            if transform_id != 0:
                features = self.hex_transform.transform_board(features, transform_id)
                if self.has_policy and len(policy_indices) > 0:
                    policy_indices, policy_values = self.hex_transform.transform_sparse_policy(
                        policy_indices.astype(np.int32),
                        policy_values.astype(np.float32),
                        transform_id
                    )

        # Build dense policy vector
        policy_vector = torch.zeros(self.policy_size, dtype=torch.float32)
        if len(policy_indices) > 0:
            indices_arr = np.asarray(policy_indices, dtype=np.int64)
            values_arr = np.asarray(policy_values, dtype=np.float32)
            policy_vector[indices_arr] = torch.from_numpy(values_arr)

        # Value tensor
        if self.use_multi_player_values and self.has_multi_player_values:
            values_mp = np.asarray(self._values_mp[actual_idx], dtype=np.float32)
            value_tensor = torch.from_numpy(values_mp)
        else:
            value_tensor = torch.tensor([value.item()], dtype=torch.float32)

        # Return with optional num_players
        if self.return_num_players and self._num_players is not None:
            num_players_val = int(self._num_players[actual_idx])
            return (
                torch.from_numpy(features),
                torch.from_numpy(globals_vec),
                value_tensor,
                policy_vector,
                torch.tensor(num_players_val, dtype=torch.int64),
            )

        return (
            torch.from_numpy(features),
            torch.from_numpy(globals_vec),
            value_tensor,
            policy_vector,
        )

    def close(self):
        """Close the HDF5 file handle."""
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self):
        self.close()


def convert_npz_to_hdf5(
    npz_path: str,
    hdf5_path: str,
    chunk_size: int = 1000,
    compression: str = "gzip",
) -> bool:
    """Convert NPZ training data to HDF5 format for streaming.

    HDF5 supports true memory-mapping and chunk-based access, making it
    suitable for datasets larger than RAM.

    Args:
        npz_path: Path to input NPZ file
        hdf5_path: Path for output HDF5 file
        chunk_size: Chunk size for HDF5 datasets (affects read performance)
        compression: Compression algorithm ("gzip", "lzf", or None)

    Returns:
        True if conversion succeeded
    """
    try:
        import h5py
    except ImportError:
        logger.error("h5py required for conversion. Install with: pip install h5py")
        return False

    if not os.path.exists(npz_path):
        logger.error(f"Source NPZ not found: {npz_path}")
        return False

    logger.info(f"Converting {npz_path} to HDF5 format...")

    try:
        # Load NPZ - safe_load_npz tries without pickle first for security
        npz_data = safe_load_npz(npz_path)
        data = {k: np.asarray(v) for k, v in npz_data.items()}

        if 'features' not in data or 'values' not in data:
            logger.error("NPZ missing required 'features' or 'values' arrays")
            return False

        n_samples = len(data['features'])
        logger.info(f"Converting {n_samples:,} samples...")

        # Create HDF5 file
        with h5py.File(hdf5_path, 'w') as h5f:
            # Core arrays with chunking for efficient streaming
            feat_shape = data['features'].shape
            h5f.create_dataset(
                'features',
                data=data['features'],
                chunks=(min(chunk_size, n_samples),) + feat_shape[1:],
                compression=compression,
            )

            global_shape = data['globals'].shape
            h5f.create_dataset(
                'globals',
                data=data['globals'],
                chunks=(min(chunk_size, n_samples),) + global_shape[1:],
                compression=compression,
            )

            h5f.create_dataset(
                'values',
                data=data['values'],
                chunks=(min(chunk_size, n_samples),),
                compression=compression,
            )

            # Policy arrays (variable-length - use special dtype)
            if 'policy_indices' in data and 'policy_values' in data:
                # HDF5 variable-length arrays
                dt_indices = h5py.special_dtype(vlen=np.int32)
                dt_values = h5py.special_dtype(vlen=np.float32)

                pi_ds = h5f.create_dataset(
                    'policy_indices',
                    shape=(n_samples,),
                    dtype=dt_indices,
                )
                pv_ds = h5f.create_dataset(
                    'policy_values',
                    shape=(n_samples,),
                    dtype=dt_values,
                )

                for i in range(n_samples):
                    pi_ds[i] = np.asarray(data['policy_indices'][i], dtype=np.int32)
                    pv_ds[i] = np.asarray(data['policy_values'][i], dtype=np.float32)

            # Multi-player values
            if 'values_mp' in data:
                h5f.create_dataset(
                    'values_mp',
                    data=data['values_mp'],
                    chunks=(min(chunk_size, n_samples),) + data['values_mp'].shape[1:],
                    compression=compression,
                )

            if 'num_players' in data:
                h5f.create_dataset(
                    'num_players',
                    data=data['num_players'],
                    compression=compression,
                )

            # Store metadata as attributes
            for key in ['board_type', 'board_size', 'policy_size', 'encoder_type',
                        'in_channels', 'spatial_size']:
                if key in data:
                    val = data[key]
                    if hasattr(val, 'item'):
                        val = val.item()
                    h5f.attrs[key] = val

        # Verify file size
        npz_size = os.path.getsize(npz_path) / (1024 * 1024)
        h5_size = os.path.getsize(hdf5_path) / (1024 * 1024)
        logger.info(
            f"Conversion complete: {npz_size:.1f}MB -> {h5_size:.1f}MB "
            f"({h5_size/npz_size*100:.0f}%)"
        )

        return True

    except Exception as e:
        logger.error(f"Conversion failed: {e}")
        if os.path.exists(hdf5_path):
            os.remove(hdf5_path)
        return False


def auto_select_dataset(
    data_path: str,
    board_type: BoardType = BoardType.SQUARE8,
    memory_threshold_mb: float = 4000.0,
    **kwargs,
) -> Dataset:
    """Automatically select appropriate dataset class based on file size.

    For files larger than memory_threshold_mb, uses StreamingRingRiftDataset
    (requires HDF5 format). Otherwise uses standard RingRiftDataset.

    Args:
        data_path: Path to training data (.npz or .h5)
        board_type: Board geometry type
        memory_threshold_mb: Size threshold in MB for streaming mode
        **kwargs: Additional arguments passed to dataset constructor

    Returns:
        Appropriate Dataset instance
    """
    if not os.path.exists(data_path):
        logger.warning(f"File not found: {data_path}, falling back to standard dataset")
        return RingRiftDataset(data_path, board_type, **kwargs)

    file_size_mb = os.path.getsize(data_path) / (1024 * 1024)

    # Check file extension
    is_hdf5 = data_path.endswith('.h5') or data_path.endswith('.hdf5')

    if file_size_mb > memory_threshold_mb:
        if not is_hdf5:
            logger.warning(
                f"Large file ({file_size_mb:.0f}MB > {memory_threshold_mb:.0f}MB threshold) "
                f"but not HDF5 format. Convert with convert_npz_to_hdf5() for streaming. "
                f"Loading into RAM instead..."
            )
            return RingRiftDataset(data_path, board_type, **kwargs)

        logger.info(
            f"Using streaming dataset for large file ({file_size_mb:.0f}MB)"
        )
        return StreamingRingRiftDataset(data_path, board_type, **kwargs)

    elif is_hdf5:
        # HDF5 file but small enough - still use streaming for consistency
        logger.info(f"Using streaming dataset for HDF5 file ({file_size_mb:.0f}MB)")
        return StreamingRingRiftDataset(data_path, board_type, **kwargs)

    else:
        # Standard NPZ, small enough for RAM
        return RingRiftDataset(data_path, board_type, **kwargs)


__all__ = [
    'MIN_SAMPLES_FOR_TRAINING',
    'RingRiftDataset',
    'WeightedRingRiftDataset',
    'StreamingRingRiftDataset',
    'convert_npz_to_hdf5',
    'auto_select_dataset',
]
