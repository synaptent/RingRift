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

logger = logging.getLogger(__name__)


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

        if os.path.exists(data_path):
            try:
                # Load data and cache arrays in memory. NPZ files don't support
                # true mmap (they're zip files), so each array access would
                # re-read from disk. For training efficiency, we load everything
                # into RAM upfront. For very large datasets, consider using
                # HDF5 or raw .npy files instead.
                npz_data = np.load(data_path, allow_pickle=True)
                # Convert to dict to force-load all arrays into memory
                self.data = {k: np.asarray(v) for k, v in npz_data.items()}

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
                    except Exception:
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

        if self.return_num_players:
            num_players_val = 0
            if self.num_players_arr is not None:
                try:
                    num_players_val = int(self.num_players_arr[actual_idx])
                except Exception:
                    num_players_val = 0
            if self.return_auxiliary_targets:
                return (
                    torch.from_numpy(features),
                    torch.from_numpy(globals_vec),
                    value_tensor,
                    policy_vector,
                    torch.tensor(num_players_val, dtype=torch.int64),
                    aux_targets,
                )
            return (
                torch.from_numpy(features),
                torch.from_numpy(globals_vec),
                value_tensor,
                policy_vector,
                torch.tensor(num_players_val, dtype=torch.int64),
            )

        if self.return_auxiliary_targets:
            return (
                torch.from_numpy(features),
                torch.from_numpy(globals_vec),
                value_tensor,
                policy_vector,
                aux_targets,
            )

        return (
            torch.from_numpy(features),
            torch.from_numpy(globals_vec),
            value_tensor,
            policy_vector,
        )


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
    ):
        super().__init__(
            data_path,
            board_type,
            augment_hex,
            use_multi_player_values=use_multi_player_values,
            filter_empty_policies=filter_empty_policies,
            return_num_players=return_num_players,
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


__all__ = [
    'RingRiftDataset',
    'WeightedRingRiftDataset',
]
