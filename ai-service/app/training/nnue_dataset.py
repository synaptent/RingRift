"""NNUE Training Dataset for RingRift Minimax.

This module provides dataset classes for training the NNUE (Efficiently
Updatable Neural Network) evaluator used by Minimax at difficulty 4+.

Training data is extracted from self-play game databases (SQLite), where
each position is labeled with the game outcome from that player's perspective.

Key classes:
- NNUESQLiteDataset: PyTorch Dataset that loads from SQLite game DBs
- NNUEDataGenerator: Utility for generating training data NPZ files

The NNUE model predicts game outcome (win/loss/draw) from position features,
providing a learned evaluation function for alpha-beta search.
"""

import gzip
import json
import logging
import os
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, Sampler

from ..ai.nnue import (
    extract_features_from_gamestate,
    get_feature_dim,
    get_board_size,
    FEATURE_PLANES,
)
from ..models import BoardType, GameState, Move
from ..rules.default_engine import DefaultRulesEngine

logger = logging.getLogger(__name__)


# =============================================================================
# GPU-Accelerated Feature Extraction
# =============================================================================

def _parse_state_to_arrays(
    state: GameState,
    board_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse GameState into numpy arrays for batch GPU processing.

    Returns:
        Tuple of (stack_owner, stack_height, territory_owner) arrays,
        each of shape (board_size, board_size)
    """
    stack_owner = np.zeros((board_size, board_size), dtype=np.int32)
    stack_height = np.zeros((board_size, board_size), dtype=np.int32)
    territory_owner = np.zeros((board_size, board_size), dtype=np.int32)

    board = state.board

    # Parse stacks
    for pos_key, ring_stack in (board.stacks or {}).items():
        try:
            parts = pos_key.split(",")
            if len(parts) >= 2:
                x, y = int(parts[0]), int(parts[1])
                if 0 <= x < board_size and 0 <= y < board_size:
                    owner = getattr(ring_stack, 'controlling_player', 0)
                    height = getattr(ring_stack, 'stack_height', 0)
                    stack_owner[y, x] = owner
                    stack_height[y, x] = height
        except (ValueError, AttributeError):
            continue

    # Parse territories
    for territory_key, territory in (board.territories or {}).items():
        try:
            pnum = getattr(territory, 'player', 0)
            if pnum > 0:
                for pos in (getattr(territory, 'spaces', None) or []):
                    if 0 <= pos.x < board_size and 0 <= pos.y < board_size:
                        territory_owner[pos.y, pos.x] = pnum
        except (ValueError, AttributeError):
            continue

    return stack_owner, stack_height, territory_owner


def extract_features_batch_gpu(
    states: List[GameState],
    player_numbers: List[int],
    board_type: BoardType,
    num_players: int = 2,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Extract NNUE features for a batch of states using GPU acceleration.

    This function batches multiple game states and processes them on GPU
    for significantly faster feature extraction during dataset generation.

    Args:
        states: List of GameState objects
        player_numbers: List of player perspectives for each state
        board_type: Board type
        num_players: Number of players
        device: GPU device (uses CUDA if available, falls back to CPU)

    Returns:
        (batch, feature_dim) numpy array of features
    """
    if not states:
        return np.array([], dtype=np.float32)

    batch_size = len(states)
    board_size = get_board_size(board_type)
    feature_dim = get_feature_dim(board_type)

    # Determine device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    # Parse all states to numpy arrays
    stack_owners = np.zeros((batch_size, board_size, board_size), dtype=np.int32)
    stack_heights = np.zeros((batch_size, board_size, board_size), dtype=np.int32)
    territory_owners = np.zeros((batch_size, board_size, board_size), dtype=np.int32)
    current_players = np.array(player_numbers, dtype=np.int64)

    for i, state in enumerate(states):
        so, sh, to = _parse_state_to_arrays(state, board_size)
        stack_owners[i] = so
        stack_heights[i] = sh
        territory_owners[i] = to

    # Move to device
    stack_owner_t = torch.from_numpy(stack_owners).to(device)
    stack_height_t = torch.from_numpy(stack_heights).to(device)
    territory_owner_t = torch.from_numpy(territory_owners).to(device)
    current_player_t = torch.from_numpy(current_players).to(device)

    # Use vectorized GPU extraction
    features_t = _extract_features_batch_vectorized(
        stack_owner_t, stack_height_t, territory_owner_t,
        current_player_t, num_players
    )

    # Move back to CPU and return numpy
    return features_t.cpu().numpy()


def _extract_features_batch_vectorized(
    stack_owner: torch.Tensor,      # (batch, H, W)
    stack_height: torch.Tensor,     # (batch, H, W)
    territory_owner: torch.Tensor,  # (batch, H, W)
    current_player: torch.Tensor,   # (batch,)
    num_players: int = 2,
) -> torch.Tensor:
    """Fully vectorized batch feature extraction.

    Feature planes (12 total) - PERSPECTIVE ROTATED:
    - Planes 0-3: Ring/stack presence (plane 0 = current player)
    - Planes 4-7: Stack height normalized (plane 4 = current player)
    - Planes 8-11: Territory ownership (plane 8 = current player)
    """
    batch_size = stack_owner.shape[0]
    H, W = stack_owner.shape[1], stack_owner.shape[2]
    device = stack_owner.device

    # Initialize features: (batch, 12, H, W)
    features = torch.zeros(batch_size, FEATURE_PLANES, H, W, device=device, dtype=torch.float32)

    # Current player expanded: (batch, 1, 1)
    cp = current_player.view(batch_size, 1, 1)

    # Process each player's features with perspective rotation
    for p in range(1, num_players + 1):
        # Masks for this player: (batch, H, W)
        owner_mask = (stack_owner == p).float()
        height_feature = torch.clamp((stack_height.float() * owner_mask) / 5.0, 0, 1)
        territory_mask = (territory_owner == p).float()

        # Compute rotation: (batch,) - 0 for current player, 1-3 for opponents
        is_current = (torch.tensor(p, device=device) == current_player)
        rotation = torch.where(
            is_current,
            torch.zeros(batch_size, dtype=torch.long, device=device),
            (torch.tensor(p, device=device) - current_player) % num_players
        )

        # Scatter to appropriate planes
        batch_idx = torch.arange(batch_size, device=device)

        # Use advanced indexing to place features in rotated planes
        for b in range(batch_size):
            rot = rotation[b].item()
            features[b, rot] = owner_mask[b]
            features[b, 4 + rot] = height_feature[b]
            features[b, 8 + rot] = territory_mask[b]

    # Flatten to (batch, feature_dim)
    return features.view(batch_size, -1)


@dataclass
class NNUESample:
    """A single training sample for NNUE."""
    features: np.ndarray  # Shape: (feature_dim,)
    value: float  # Game outcome from player perspective: +1 win, -1 loss, 0 draw
    player_number: int
    game_id: str
    move_number: int


@dataclass
class DataValidationResult:
    """Result of data validation."""
    is_valid: bool
    total_samples: int = 0
    valid_samples: int = 0
    invalid_samples: int = 0
    nan_count: int = 0
    inf_count: int = 0
    value_out_of_range: int = 0
    zero_feature_count: int = 0
    class_balance: Dict[str, int] = None  # wins/losses/draws
    feature_stats: Dict[str, float] = None  # mean, std, min, max
    errors: List[str] = None

    def __post_init__(self):
        if self.class_balance is None:
            self.class_balance = {}
        if self.feature_stats is None:
            self.feature_stats = {}
        if self.errors is None:
            self.errors = []


def validate_nnue_sample(sample: NNUESample, feature_dim: int) -> Tuple[bool, Optional[str]]:
    """Validate a single NNUE training sample.

    Args:
        sample: The sample to validate
        feature_dim: Expected feature dimension

    Returns:
        Tuple of (is_valid, error_message)
    """
    # Check feature shape
    if sample.features.shape[0] != feature_dim:
        return False, f"Feature dim mismatch: expected {feature_dim}, got {sample.features.shape[0]}"

    # Check for NaN
    if np.isnan(sample.features).any():
        return False, "Features contain NaN"

    # Check for Inf
    if np.isinf(sample.features).any():
        return False, "Features contain Inf"

    # Check value range
    if not -1.0 <= sample.value <= 1.0:
        return False, f"Value out of range: {sample.value}"

    # Check player number
    if sample.player_number < 1 or sample.player_number > 8:
        return False, f"Invalid player number: {sample.player_number}"

    # Check for all-zero features (likely extraction failure)
    if np.allclose(sample.features, 0):
        return False, "All-zero features"

    return True, None


def validate_nnue_dataset(
    samples: List[NNUESample],
    feature_dim: int,
    log_errors: bool = True,
) -> DataValidationResult:
    """Validate an entire NNUE dataset.

    Args:
        samples: List of samples to validate
        feature_dim: Expected feature dimension
        log_errors: Whether to log validation errors

    Returns:
        DataValidationResult with validation statistics
    """
    result = DataValidationResult(is_valid=True, total_samples=len(samples))
    class_counts = {"wins": 0, "losses": 0, "draws": 0}
    all_features = []
    errors = []

    for i, sample in enumerate(samples):
        is_valid, error = validate_nnue_sample(sample, feature_dim)

        if is_valid:
            result.valid_samples += 1
            all_features.append(sample.features)

            # Track class balance
            if sample.value > 0.5:
                class_counts["wins"] += 1
            elif sample.value < -0.5:
                class_counts["losses"] += 1
            else:
                class_counts["draws"] += 1
        else:
            result.invalid_samples += 1
            errors.append(f"Sample {i} ({sample.game_id}:{sample.move_number}): {error}")

            # Categorize error
            if "NaN" in error:
                result.nan_count += 1
            elif "Inf" in error:
                result.inf_count += 1
            elif "out of range" in error:
                result.value_out_of_range += 1
            elif "zero" in error.lower():
                result.zero_feature_count += 1

    result.class_balance = class_counts
    result.errors = errors[:100]  # Limit to first 100 errors

    # Compute feature statistics if we have valid samples
    if all_features:
        features_array = np.stack(all_features)
        result.feature_stats = {
            "mean": float(np.mean(features_array)),
            "std": float(np.std(features_array)),
            "min": float(np.min(features_array)),
            "max": float(np.max(features_array)),
            "sparsity": float(np.mean(np.abs(features_array) < 1e-6)),
        }

    # Check class imbalance (warn if > 2:1 ratio)
    if class_counts["wins"] > 0 and class_counts["losses"] > 0:
        ratio = max(class_counts["wins"], class_counts["losses"]) / min(class_counts["wins"], class_counts["losses"])
        if ratio > 2.0:
            result.errors.append(f"Warning: Class imbalance detected (ratio: {ratio:.1f})")

    # Mark as invalid if too many errors
    if result.invalid_samples > result.total_samples * 0.1:
        result.is_valid = False
        result.errors.insert(0, f"Too many invalid samples: {result.invalid_samples}/{result.total_samples}")

    if log_errors and result.errors:
        for error in result.errors[:10]:
            logger.warning(error)
        if len(result.errors) > 10:
            logger.warning(f"... and {len(result.errors) - 10} more validation errors")

    return result


def validate_database_integrity(db_path: str) -> Tuple[bool, Dict[str, Any]]:
    """Validate SQLite database integrity for training.

    Args:
        db_path: Path to SQLite database

    Returns:
        Tuple of (is_valid, stats_dict)
    """
    stats = {
        "path": db_path,
        "exists": os.path.exists(db_path),
        "total_games": 0,
        "completed_games": 0,
        "games_with_snapshots": 0,
        "total_snapshots": 0,
        "integrity_ok": False,
        "errors": [],
    }

    if not stats["exists"]:
        stats["errors"].append("Database file does not exist")
        return False, stats

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check SQLite integrity
        cursor.execute("PRAGMA integrity_check")
        integrity = cursor.fetchone()[0]
        stats["integrity_ok"] = integrity == "ok"
        if integrity != "ok":
            stats["errors"].append(f"SQLite integrity check failed: {integrity}")

        # Count games
        cursor.execute("SELECT COUNT(*) FROM games")
        stats["total_games"] = cursor.fetchone()[0]

        # Count completed games
        cursor.execute("SELECT COUNT(*) FROM games WHERE game_status = 'completed'")
        stats["completed_games"] = cursor.fetchone()[0]

        # Count games with snapshots (if table exists)
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='game_state_snapshots'"
        )
        if cursor.fetchone():
            cursor.execute("SELECT COUNT(DISTINCT game_id) FROM game_state_snapshots")
            stats["games_with_snapshots"] = cursor.fetchone()[0]

            cursor.execute("SELECT COUNT(*) FROM game_state_snapshots")
            stats["total_snapshots"] = cursor.fetchone()[0]

        conn.close()

        # Validate stats
        if stats["completed_games"] == 0:
            stats["errors"].append("No completed games in database")

        return len(stats["errors"]) == 0, stats

    except Exception as e:
        stats["errors"].append(f"Database error: {str(e)}")
        return False, stats


@dataclass
class NNUEDatasetConfig:
    """Configuration for NNUE dataset generation."""
    board_type: BoardType = BoardType.SQUARE8
    num_players: int = 2
    sample_every_n_moves: int = 1  # Sample every Nth position
    min_game_length: int = 10  # Skip very short games
    include_draws: bool = True
    late_game_weight: float = 1.0  # Weight for positions near end of game
    balance_outcomes: bool = False  # Balance wins/losses in dataset


class NNUESQLiteDataset(Dataset):
    """PyTorch Dataset that loads NNUE training samples from SQLite game DBs.

    This dataset:
    1. Loads completed games from SQLite databases
    2. Replays each game to extract position snapshots
    3. Labels each position with the game outcome
    4. Extracts NNUE features from each position

    For efficiency, positions can be cached to NPZ files.
    """

    def __init__(
        self,
        db_paths: List[str],
        config: Optional[NNUEDatasetConfig] = None,
        cache_path: Optional[str] = None,
        max_samples: Optional[int] = None,
        use_gpu_extraction: bool = False,
        gpu_batch_size: int = 256,
    ):
        """Initialize the NNUE dataset.

        Args:
            db_paths: List of paths to SQLite game databases
            config: Dataset configuration
            cache_path: Optional path to cache extracted features as NPZ
            max_samples: Optional limit on number of samples
            use_gpu_extraction: Use GPU-accelerated batch feature extraction
            gpu_batch_size: Batch size for GPU extraction
        """
        self.db_paths = db_paths
        self.config = config or NNUEDatasetConfig()
        self.cache_path = cache_path
        self.max_samples = max_samples
        self.use_gpu_extraction = use_gpu_extraction and torch.cuda.is_available()
        self.gpu_batch_size = gpu_batch_size

        self.feature_dim = get_feature_dim(self.config.board_type)
        self.samples: List[NNUESample] = []

        # Load from cache if available
        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
            if self.use_gpu_extraction:
                logger.info("Using GPU-accelerated feature extraction")
            self._extract_samples()
            if cache_path:
                self._save_to_cache(cache_path)

    def _extract_samples(self) -> None:
        """Extract training samples from SQLite databases."""
        logger.info(f"Extracting NNUE samples from {len(self.db_paths)} databases")

        for db_path in self.db_paths:
            if not os.path.exists(db_path):
                logger.warning(f"Database not found: {db_path}")
                continue

            try:
                samples = self._extract_from_db(db_path)
                self.samples.extend(samples)
                logger.info(f"Extracted {len(samples)} samples from {db_path}")
            except Exception as e:
                logger.error(f"Failed to extract from {db_path}: {e}")

        if self.max_samples and len(self.samples) > self.max_samples:
            # Shuffle and truncate
            np.random.shuffle(self.samples)
            self.samples = self.samples[:self.max_samples]

        logger.info(f"Total NNUE samples: {len(self.samples)}")

        # Validate the extracted samples
        if self.samples:
            validation = validate_nnue_dataset(self.samples, self.feature_dim, log_errors=True)
            logger.info(
                f"Validation: {validation.valid_samples}/{validation.total_samples} valid "
                f"(wins={validation.class_balance.get('wins', 0)}, "
                f"losses={validation.class_balance.get('losses', 0)}, "
                f"draws={validation.class_balance.get('draws', 0)})"
            )
            if validation.feature_stats:
                logger.info(
                    f"Feature stats: mean={validation.feature_stats.get('mean', 0):.4f}, "
                    f"std={validation.feature_stats.get('std', 0):.4f}, "
                    f"sparsity={validation.feature_stats.get('sparsity', 0):.2%}"
                )
            if not validation.is_valid:
                logger.warning("Dataset validation failed - training may have issues")

    def _extract_from_cached_features(self, db_path: str) -> Optional[List[NNUESample]]:
        """Try to extract samples from cached NNUE features table.

        Returns None if no cached features exist, otherwise returns the samples.
        This is much faster than replay-based extraction.
        """
        samples: List[NNUESample] = []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Check if the game_nnue_features table exists
        cursor.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name='game_nnue_features'
        """)
        if not cursor.fetchone():
            conn.close()
            return None

        board_type_str = self.config.board_type.value.lower()

        # Query cached features matching our criteria
        query = """
            SELECT f.game_id, f.move_number, f.player_perspective, f.features, f.value, f.feature_dim
            FROM game_nnue_features f
            JOIN games g ON f.game_id = g.game_id
            WHERE f.board_type = ?
              AND g.num_players = ?
              AND g.game_status = 'completed'
              AND g.winner IS NOT NULL
              AND g.total_moves >= ?
              AND COALESCE(g.excluded_from_training, 0) = 0
        """
        try:
            cursor.execute(query, (
                board_type_str,
                self.config.num_players,
                self.config.min_game_length,
            ))
        except sqlite3.OperationalError:
            # Table might not have proper joins, fallback
            conn.close()
            return None

        count = 0
        for row in cursor:
            try:
                # Decompress features
                features = np.frombuffer(
                    gzip.decompress(row['features']), dtype=np.float32
                ).copy()

                # Validate feature dimension
                if len(features) != row['feature_dim']:
                    continue

                # Apply sampling rate
                if row['move_number'] % self.config.sample_every_n_moves != 0:
                    continue

                samples.append(NNUESample(
                    features=features,
                    value=row['value'],
                    player_number=row['player_perspective'],
                    game_id=row['game_id'],
                    move_number=row['move_number'],
                ))
                count += 1
            except Exception:
                continue

        conn.close()

        if count > 0:
            logger.info(f"Loaded {count} cached NNUE features from {db_path}")
            return samples
        return None

    def _extract_from_db(self, db_path: str) -> List[NNUESample]:
        """Extract samples from a single SQLite database."""
        # Try cached features first (instant extraction)
        cached_samples = self._extract_from_cached_features(db_path)
        if cached_samples is not None:
            return cached_samples

        # Fall back to snapshot/replay extraction
        samples: List[NNUESample] = []
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        # Get completed games with winners
        board_type_str = self.config.board_type.value.lower()
        query = """
            SELECT game_id, board_type, num_players, winner, total_moves
            FROM games
            WHERE game_status = 'completed'
              AND winner IS NOT NULL
              AND board_type = ?
              AND num_players = ?
              AND total_moves >= ?
              AND COALESCE(excluded_from_training, 0) = 0
        """
        cursor.execute(query, (
            board_type_str,
            self.config.num_players,
            self.config.min_game_length,
        ))
        games = cursor.fetchall()

        # For GPU extraction, batch collect states first
        if self.use_gpu_extraction:
            samples = self._extract_from_db_gpu_batch(conn, cursor, games)
            conn.close()
            return samples

        for game_row in games:
            game_id = game_row['game_id']
            winner = game_row['winner']
            total_moves = game_row['total_moves']

            # Get state snapshots for this game
            snapshot_query = """
                SELECT move_number, state_json, compressed
                FROM game_state_snapshots
                WHERE game_id = ?
                ORDER BY move_number
            """
            cursor.execute(snapshot_query, (game_id,))
            snapshots = cursor.fetchall()

            # Determine expected positions based on sampling rate
            # Use snapshots if we have at least 1 - faster than full replay
            # Note: Lowered threshold from max(expected_samples//2, 5) to 1
            # because most cluster DBs only store final state snapshot
            if not snapshots:
                # No snapshots at all - must use slow replay
                samples.extend(self._extract_via_replay(
                    conn, game_id, winner, total_moves
                ))
                continue

            # Check if snapshots have actual board data (not just initial empty state)
            # If snapshots only have move 0 (initial) or have empty boards, use replay
            has_useful_snapshots = False
            for snap in snapshots:
                move_num = snap['move_number']
                # Skip initial state check - it's always empty
                if move_num > 0:
                    # Check if this snapshot has board data
                    try:
                        snap_json = snap['state_json']
                        if snap['compressed']:
                            snap_json = gzip.decompress(snap_json.encode()).decode()
                        snap_dict = json.loads(snap_json)
                        board_data = snap_dict.get('board', {})
                        stacks = board_data.get('stacks', {})
                        if stacks:  # Has actual board state
                            has_useful_snapshots = True
                            break
                    except Exception:
                        continue

            if not has_useful_snapshots:
                # Snapshots are initial-only or have empty boards - use replay
                samples.extend(self._extract_via_replay(
                    conn, game_id, winner, total_moves
                ))
                continue

            # Sample positions from snapshots (only if we have good coverage)
            for i, snapshot in enumerate(snapshots):
                move_num = snapshot['move_number']

                # Sample every Nth position
                if move_num % self.config.sample_every_n_moves != 0:
                    continue

                # Parse state JSON
                state_json = snapshot['state_json']
                if snapshot['compressed']:
                    state_json = gzip.decompress(state_json.encode()).decode()

                try:
                    state_dict = json.loads(state_json)
                    game_state = GameState(**state_dict)
                except Exception as e:
                    logger.debug(f"Failed to parse state for {game_id}:{move_num}: {e}")
                    continue

                # Get current player from game state
                current_player = game_state.current_player
                if current_player is None or current_player < 1:
                    current_player = 1

                # Calculate value: +1 if current player wins, -1 if loses
                if winner == current_player:
                    value = 1.0
                elif winner is None or winner == 0:
                    value = 0.0  # Draw
                else:
                    value = -1.0  # Loss

                # Skip draws if configured
                if value == 0.0 and not self.config.include_draws:
                    continue

                # Extract NNUE features
                try:
                    features = extract_features_from_gamestate(
                        game_state, current_player
                    )
                except Exception as e:
                    logger.debug(f"Feature extraction failed for {game_id}:{move_num}: {e}")
                    continue

                sample = NNUESample(
                    features=features,
                    value=value,
                    player_number=current_player,
                    game_id=game_id,
                    move_number=move_num,
                )
                samples.append(sample)

                if self.max_samples and len(samples) >= self.max_samples:
                    conn.close()
                    return samples

        conn.close()
        return samples

    def _extract_from_db_gpu_batch(
        self,
        conn: sqlite3.Connection,
        cursor: sqlite3.Cursor,
        games: List[Any],
    ) -> List[NNUESample]:
        """GPU-accelerated batch extraction from database.

        Collects states in batches, then processes them on GPU for faster
        feature extraction.
        """
        samples: List[NNUESample] = []

        # Collect batch data: (state, player_number, value, game_id, move_number)
        batch_states: List[GameState] = []
        batch_players: List[int] = []
        batch_values: List[float] = []
        batch_game_ids: List[str] = []
        batch_move_nums: List[int] = []

        for game_row in games:
            game_id = game_row['game_id']
            winner = game_row['winner']

            # Get state snapshots for this game
            snapshot_query = """
                SELECT move_number, state_json, compressed
                FROM game_state_snapshots
                WHERE game_id = ?
                ORDER BY move_number
            """
            cursor.execute(snapshot_query, (game_id,))
            snapshots = cursor.fetchall()

            if not snapshots:
                # Fall back to replay for games without snapshots
                replay_samples = self._extract_via_replay(
                    conn, game_id, winner, game_row['total_moves']
                )
                samples.extend(replay_samples)
                continue

            # Check if snapshots have actual board data (not just initial empty state)
            has_useful_snapshots = False
            for snap in snapshots:
                move_num = snap['move_number']
                if move_num > 0:
                    try:
                        snap_json = snap['state_json']
                        if snap['compressed']:
                            snap_json = gzip.decompress(snap_json.encode()).decode()
                        snap_dict = json.loads(snap_json)
                        board_data = snap_dict.get('board', {})
                        stacks = board_data.get('stacks', {})
                        if stacks:
                            has_useful_snapshots = True
                            break
                    except Exception:
                        continue

            if not has_useful_snapshots:
                # Snapshots are initial-only or have empty boards - use replay
                replay_samples = self._extract_via_replay(
                    conn, game_id, winner, game_row['total_moves']
                )
                samples.extend(replay_samples)
                continue

            for snapshot in snapshots:
                move_num = snapshot['move_number']

                # Sample every Nth position
                if move_num % self.config.sample_every_n_moves != 0:
                    continue

                # Parse state JSON
                state_json = snapshot['state_json']
                if snapshot['compressed']:
                    state_json = gzip.decompress(state_json.encode()).decode()

                try:
                    state_dict = json.loads(state_json)
                    game_state = GameState(**state_dict)
                except Exception:
                    continue

                # Get current player from game state
                current_player = game_state.current_player
                if current_player is None or current_player < 1:
                    current_player = 1

                # Calculate value
                if winner == current_player:
                    value = 1.0
                elif winner is None or winner == 0:
                    if not self.config.include_draws:
                        continue
                    value = 0.0
                else:
                    value = -1.0

                # Add to batch
                batch_states.append(game_state)
                batch_players.append(current_player)
                batch_values.append(value)
                batch_game_ids.append(game_id)
                batch_move_nums.append(move_num)

                # Process batch when full
                if len(batch_states) >= self.gpu_batch_size:
                    batch_samples = self._process_gpu_batch(
                        batch_states, batch_players, batch_values,
                        batch_game_ids, batch_move_nums
                    )
                    samples.extend(batch_samples)

                    # Clear batch
                    batch_states = []
                    batch_players = []
                    batch_values = []
                    batch_game_ids = []
                    batch_move_nums = []

                    if self.max_samples and len(samples) >= self.max_samples:
                        return samples[:self.max_samples]

        # Process remaining batch
        if batch_states:
            batch_samples = self._process_gpu_batch(
                batch_states, batch_players, batch_values,
                batch_game_ids, batch_move_nums
            )
            samples.extend(batch_samples)

        return samples

    def _process_gpu_batch(
        self,
        states: List[GameState],
        players: List[int],
        values: List[float],
        game_ids: List[str],
        move_nums: List[int],
    ) -> List[NNUESample]:
        """Process a batch of states using GPU feature extraction."""
        try:
            features = extract_features_batch_gpu(
                states, players,
                self.config.board_type,
                self.config.num_players,
            )

            samples = []
            for i in range(len(states)):
                sample = NNUESample(
                    features=features[i],
                    value=values[i],
                    player_number=players[i],
                    game_id=game_ids[i],
                    move_number=move_nums[i],
                )
                samples.append(sample)
            return samples

        except Exception as e:
            # Fall back to CPU extraction on error
            logger.warning(f"GPU batch extraction failed, falling back to CPU: {e}")
            samples = []
            for i, state in enumerate(states):
                try:
                    features = extract_features_from_gamestate(state, players[i])
                    sample = NNUESample(
                        features=features,
                        value=values[i],
                        player_number=players[i],
                        game_id=game_ids[i],
                        move_number=move_nums[i],
                    )
                    samples.append(sample)
                except Exception:
                    continue
            return samples

    def _extract_via_replay(
        self,
        conn: sqlite3.Connection,
        game_id: str,
        winner: int,
        total_moves: int,
    ) -> List[NNUESample]:
        """Extract samples by replaying moves when snapshots aren't available.

        This method loads the initial state and all moves for a game, then
        replays the game to extract position features at sampled positions.
        """
        samples: List[NNUESample] = []
        cursor = conn.cursor()

        # Get initial state
        cursor.execute(
            "SELECT initial_state_json, compressed FROM game_initial_state WHERE game_id = ?",
            (game_id,)
        )
        initial_row = cursor.fetchone()
        if not initial_row:
            logger.debug(f"Skipping {game_id}: no initial state")
            return []

        initial_json = initial_row[0]
        if initial_row[1]:  # compressed
            initial_json = gzip.decompress(initial_json.encode()).decode()

        try:
            state_dict = json.loads(initial_json)
            state = GameState(**state_dict)
        except Exception as e:
            logger.debug(f"Failed to parse initial state for {game_id}: {e}")
            return []

        # Get all moves for this game
        cursor.execute(
            """
            SELECT move_number, move_json
            FROM game_moves
            WHERE game_id = ?
            ORDER BY move_number
            """,
            (game_id,)
        )
        moves = cursor.fetchall()

        if not moves:
            logger.debug(f"Skipping {game_id}: no moves found")
            return []

        # Create rules engine for move application
        engine = DefaultRulesEngine()

        # Replay game and sample positions
        for move_row in moves:
            move_number = move_row[0]
            move_json_str = move_row[1]

            # Sample every Nth position
            if move_number % self.config.sample_every_n_moves == 0:
                # Get current player from state
                current_player = state.current_player
                if current_player is None or current_player < 1:
                    current_player = 1

                # Calculate value: +1 if current player wins, -1 if loses
                if winner == current_player:
                    value = 1.0
                elif winner is None or winner == 0:
                    value = 0.0  # Draw
                else:
                    value = -1.0  # Loss

                # Skip draws if configured
                if value == 0.0 and not self.config.include_draws:
                    pass  # Skip but continue replaying
                else:
                    # Extract NNUE features
                    try:
                        features = extract_features_from_gamestate(
                            state, current_player
                        )
                        sample = NNUESample(
                            features=features,
                            value=value,
                            player_number=current_player,
                            game_id=game_id,
                            move_number=move_number,
                        )
                        samples.append(sample)
                    except Exception as e:
                        logger.debug(f"Feature extraction failed for {game_id}:{move_number}: {e}")

            # Apply move to advance state
            try:
                move_dict = json.loads(move_json_str)
                move = Move(**move_dict)
                state = engine.apply_move(state, move)
            except Exception as e:
                logger.debug(f"Failed to apply move {move_number} for {game_id}: {e}")
                break  # Stop replay on error

            # Check if we've collected enough samples
            if self.max_samples and len(samples) >= self.max_samples:
                break

        return samples

    def _load_from_cache(self, cache_path: str) -> None:
        """Load samples from NPZ cache file with memory-mapped loading for large files."""
        logger.info(f"Loading NNUE dataset from cache: {cache_path}")

        # Use memory-mapped mode for large files (>100MB)
        file_size_mb = os.path.getsize(cache_path) / (1024 * 1024)
        mmap_mode = 'r' if file_size_mb > 100 else None

        if mmap_mode:
            logger.info(f"Using memory-mapped loading for {file_size_mb:.1f}MB cache file")

        data = np.load(cache_path, allow_pickle=True, mmap_mode=mmap_mode)

        features = data['features']
        values = data['values']
        player_numbers = data['player_numbers']
        game_ids = data['game_ids']
        move_numbers = data['move_numbers']

        # For large datasets, store as contiguous arrays for faster access
        if len(values) > 100000:
            logger.info("Converting to tensor-backed storage for fast access")
            self._features_array = np.array(features, dtype=np.float32)
            self._values_array = np.array(values, dtype=np.float32)
            self._player_numbers_array = np.array(player_numbers, dtype=np.int32)
            self._game_ids_array = np.array(game_ids, dtype=object)
            self._move_numbers_array = np.array(move_numbers, dtype=np.int32)
            self._use_tensor_cache = True

            # Still create sample objects for metadata access
            for i in range(len(values)):
                sample = NNUESample(
                    features=self._features_array[i],
                    value=float(self._values_array[i]),
                    player_number=int(self._player_numbers_array[i]),
                    game_id=str(self._game_ids_array[i]),
                    move_number=int(self._move_numbers_array[i]),
                )
                self.samples.append(sample)
        else:
            self._use_tensor_cache = False
            for i in range(len(values)):
                sample = NNUESample(
                    features=features[i],
                    value=float(values[i]),
                    player_number=int(player_numbers[i]),
                    game_id=str(game_ids[i]),
                    move_number=int(move_numbers[i]),
                )
                self.samples.append(sample)

        if self.max_samples and len(self.samples) > self.max_samples:
            self.samples = self.samples[:self.max_samples]

        logger.info(f"Loaded {len(self.samples)} samples from cache")

    def _save_to_cache(self, cache_path: str) -> None:
        """Save samples to NPZ cache file."""
        logger.info(f"Saving NNUE dataset to cache: {cache_path}")

        features = np.stack([s.features for s in self.samples])
        values = np.array([s.value for s in self.samples], dtype=np.float32)
        player_numbers = np.array([s.player_number for s in self.samples], dtype=np.int32)
        game_ids = np.array([s.game_id for s in self.samples], dtype=object)
        move_numbers = np.array([s.move_number for s in self.samples], dtype=np.int32)

        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        np.savez_compressed(
            cache_path,
            features=features,
            values=values,
            player_numbers=player_numbers,
            game_ids=game_ids,
            move_numbers=move_numbers,
            board_type=self.config.board_type.value,
            num_players=self.config.num_players,
        )

        logger.info(f"Saved {len(self.samples)} samples to {cache_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single sample as (features, value) tensors."""
        # Fast path: use cached arrays directly
        if getattr(self, '_use_tensor_cache', False):
            features = torch.from_numpy(self._features_array[idx]).float()
            value = torch.tensor([self._values_array[idx]], dtype=torch.float32)
            return features, value

        # Standard path
        sample = self.samples[idx]
        features = torch.from_numpy(sample.features).float()
        value = torch.tensor([sample.value], dtype=torch.float32)
        return features, value

    def get_move_numbers(self) -> np.ndarray:
        """Get move numbers for all samples (for phase-based weighting)."""
        return np.array([s.move_number for s in self.samples], dtype=np.int32)

    def compute_phase_balanced_weights(
        self,
        early_end: int = 40,
        mid_end: int = 80,
        target_balance: Tuple[float, float, float] = (0.25, 0.35, 0.40),
    ) -> np.ndarray:
        """Compute sample weights to balance early/mid/late game phases.

        This enables training on a more balanced distribution of game phases,
        counteracting the natural bias toward late-game positions.

        Args:
            early_end: Move number where early game ends (default 40)
            mid_end: Move number where mid game ends (default 80)
            target_balance: Target proportion (early, mid, late) that should
                           sum to 1.0. Default (0.25, 0.35, 0.40) gives more
                           weight to early/mid game positions.

        Returns:
            Array of sample weights for use with WeightedRandomSampler
        """
        move_numbers = self.get_move_numbers()
        n_samples = len(move_numbers)

        if n_samples == 0:
            return np.array([])

        # Classify samples by phase
        early_mask = move_numbers < early_end
        mid_mask = (move_numbers >= early_end) & (move_numbers < mid_end)
        late_mask = move_numbers >= mid_end

        n_early = np.sum(early_mask)
        n_mid = np.sum(mid_mask)
        n_late = np.sum(late_mask)

        logger.info(f"Phase counts - Early: {n_early}, Mid: {n_mid}, Late: {n_late}")

        # Compute weights to achieve target balance
        weights = np.ones(n_samples, dtype=np.float64)

        target_early, target_mid, target_late = target_balance

        # Weight = (target_proportion * total) / actual_count
        # This makes expected samples per phase proportional to target
        if n_early > 0:
            weights[early_mask] = (target_early * n_samples) / n_early
        if n_mid > 0:
            weights[mid_mask] = (target_mid * n_samples) / n_mid
        if n_late > 0:
            weights[late_mask] = (target_late * n_samples) / n_late

        # Normalize to sum to 1 (for sampling probabilities)
        weights = weights / weights.sum()

        # Log effective balance
        eff_early = weights[early_mask].sum() if n_early > 0 else 0
        eff_mid = weights[mid_mask].sum() if n_mid > 0 else 0
        eff_late = weights[late_mask].sum() if n_late > 0 else 0
        logger.info(
            f"Effective phase balance - Early: {eff_early:.1%}, "
            f"Mid: {eff_mid:.1%}, Late: {eff_late:.1%}"
        )

        return weights

    def get_balanced_sampler(
        self,
        early_end: int = 40,
        mid_end: int = 80,
        target_balance: Tuple[float, float, float] = (0.25, 0.35, 0.40),
    ) -> "torch.utils.data.WeightedRandomSampler":
        """Get a WeightedRandomSampler that balances game phases.

        Args:
            early_end: Move number where early game ends
            mid_end: Move number where mid game ends
            target_balance: Target (early, mid, late) proportions

        Returns:
            WeightedRandomSampler for use with DataLoader
        """
        from torch.utils.data import WeightedRandomSampler

        weights = self.compute_phase_balanced_weights(
            early_end=early_end,
            mid_end=mid_end,
            target_balance=target_balance,
        )

        return WeightedRandomSampler(
            weights=torch.from_numpy(weights).double(),
            num_samples=len(self.samples),
            replacement=True,
        )


class NNUEStreamingDataset(IterableDataset):
    """Streaming dataset for large-scale NNUE training.

    Streams samples from SQLite databases without loading all into memory.
    Suitable for training on very large game databases.

    Features:
    - Memory-efficient streaming from SQLite
    - Multi-worker support with database sharding
    - Buffered shuffling for randomization
    - Epoch-based reseeding for different shuffles each epoch
    """

    def __init__(
        self,
        db_paths: List[str],
        config: Optional[NNUEDatasetConfig] = None,
        shuffle_games: bool = True,
        seed: Optional[int] = None,
        buffer_size: int = 10000,
        epoch: int = 0,
    ):
        self.db_paths = db_paths
        self.config = config or NNUEDatasetConfig()
        self.shuffle_games = shuffle_games
        self.base_seed = seed if seed is not None else 42
        self.buffer_size = buffer_size
        self.epoch = epoch
        self.feature_dim = get_feature_dim(self.config.board_type)

    def set_epoch(self, epoch: int) -> None:
        """Set current epoch for shuffling variance across epochs."""
        self.epoch = epoch

    def _get_worker_info(self) -> Tuple[int, int]:
        """Get worker ID and total workers for sharding."""
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:
            return 0, 1
        return worker_info.id, worker_info.num_workers

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over training samples with multi-worker sharding and buffered shuffle."""
        worker_id, num_workers = self._get_worker_info()

        # Create RNG with epoch-based seed for different shuffles each epoch
        rng = np.random.default_rng(self.base_seed + self.epoch + worker_id * 1000)

        # Shard databases across workers
        db_paths = [p for i, p in enumerate(self.db_paths) if i % num_workers == worker_id]

        if self.shuffle_games:
            rng.shuffle(db_paths)

        # Use buffer for shuffling samples within a window
        buffer: List[Tuple[torch.Tensor, torch.Tensor]] = []

        for db_path in db_paths:
            if not os.path.exists(db_path):
                continue

            try:
                for sample in self._stream_from_db(db_path):
                    buffer.append(sample)

                    # When buffer is full, shuffle and yield half
                    if len(buffer) >= self.buffer_size:
                        rng.shuffle(buffer)
                        # Yield first half of buffer
                        for item in buffer[:self.buffer_size // 2]:
                            yield item
                        # Keep second half
                        buffer = buffer[self.buffer_size // 2:]
            except Exception as e:
                logger.error(f"Error streaming from {db_path}: {e}")

        # Yield remaining samples in buffer
        if buffer:
            rng.shuffle(buffer)
            for item in buffer:
                yield item

    def _stream_from_db(self, db_path: str) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Stream samples from a single database."""
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        board_type_str = self.config.board_type.value.lower()
        query = """
            SELECT g.game_id, g.winner, g.total_moves,
                   s.move_number, s.state_json, s.compressed
            FROM games g
            JOIN game_state_snapshots s ON g.game_id = s.game_id
            WHERE g.game_status = 'completed'
              AND g.winner IS NOT NULL
              AND g.board_type = ?
              AND g.num_players = ?
              AND g.total_moves >= ?
              AND COALESCE(g.excluded_from_training, 0) = 0
            ORDER BY g.game_id, s.move_number
        """
        cursor.execute(query, (
            board_type_str,
            self.config.num_players,
            self.config.min_game_length,
        ))

        for row in cursor:
            move_num = row['move_number']

            # Sample every Nth position
            if move_num % self.config.sample_every_n_moves != 0:
                continue

            winner = row['winner']
            state_json = row['state_json']
            if row['compressed']:
                state_json = gzip.decompress(state_json.encode()).decode()

            try:
                state_dict = json.loads(state_json)
                game_state = GameState(**state_dict)
            except Exception:
                continue

            current_player = game_state.current_player or 1

            # Calculate value
            if winner == current_player:
                value = 1.0
            elif winner is None or winner == 0:
                if not self.config.include_draws:
                    continue
                value = 0.0
            else:
                value = -1.0

            # Extract features
            try:
                features = extract_features_from_gamestate(game_state, current_player)
            except Exception:
                continue

            features_tensor = torch.from_numpy(features).float()
            value_tensor = torch.tensor([value], dtype=torch.float32)
            yield features_tensor, value_tensor

        conn.close()


class PrioritizedExperienceSampler(Sampler):
    """Prioritized Experience Replay (PER) sampler for NNUE training.

    Samples training examples based on prediction error priority.
    Samples with higher errors are sampled more frequently.

    Uses proportional prioritization: P(i) = p_i^alpha / sum(p_j^alpha)
    With importance sampling weights: w_i = (N * P(i))^(-beta)

    Args:
        dataset_size: Number of samples in the dataset
        alpha: Prioritization exponent (0 = uniform, 1 = full prioritization)
        beta: Importance sampling correction (0 = no correction, 1 = full)
        beta_schedule: Whether to anneal beta during training
        epsilon: Small constant to ensure non-zero priority
        initial_priority: Initial priority for unseen samples
    """

    def __init__(
        self,
        dataset_size: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_schedule: bool = True,
        epsilon: float = 1e-6,
        initial_priority: float = 1.0,
    ):
        self.dataset_size = dataset_size
        self.alpha = alpha
        self.beta = beta
        self.beta_start = beta
        self.beta_schedule = beta_schedule
        self.epsilon = epsilon

        # Initialize priorities to initial_priority
        self.priorities = np.full(dataset_size, initial_priority, dtype=np.float64)

        # Track which samples have been seen
        self.seen = np.zeros(dataset_size, dtype=bool)

        # Current epoch for beta annealing
        self.epoch = 0
        self.total_epochs = 100

        logger.info(f"PER sampler initialized: {dataset_size} samples, alpha={alpha}, beta={beta}")

    def __len__(self) -> int:
        return self.dataset_size

    def __iter__(self) -> Iterator[int]:
        """Generate sample indices based on priority."""
        # Compute sampling probabilities
        probs = self._compute_probabilities()

        # Sample indices
        indices = np.random.choice(
            self.dataset_size,
            size=self.dataset_size,
            replace=True,  # With replacement for prioritized sampling
            p=probs,
        )

        return iter(indices.tolist())

    def _compute_probabilities(self) -> np.ndarray:
        """Compute sampling probabilities from priorities."""
        priorities_alpha = np.power(self.priorities + self.epsilon, self.alpha)
        probs = priorities_alpha / priorities_alpha.sum()
        return probs

    def update_priorities(self, indices: List[int], errors: np.ndarray) -> None:
        """Update priorities based on prediction errors.

        Args:
            indices: List of sample indices that were used
            errors: Corresponding prediction errors (|predicted - target|)
        """
        for idx, error in zip(indices, errors):
            self.priorities[idx] = abs(error) + self.epsilon
            self.seen[idx] = True

    def get_importance_weights(self, indices: List[int]) -> torch.Tensor:
        """Compute importance sampling weights for a batch.

        Args:
            indices: Batch sample indices

        Returns:
            Tensor of importance sampling weights
        """
        # Current beta (annealed if scheduled)
        beta = self.beta
        if self.beta_schedule and self.total_epochs > 0:
            beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.epoch / self.total_epochs))

        probs = self._compute_probabilities()
        batch_probs = probs[indices]

        # w_i = (N * P(i))^(-beta)
        weights = np.power(self.dataset_size * batch_probs, -beta)

        # Normalize by max weight for stability
        weights = weights / weights.max()

        return torch.from_numpy(weights).float()

    def set_epoch(self, epoch: int, total_epochs: int = 100) -> None:
        """Set current epoch for beta annealing."""
        self.epoch = epoch
        self.total_epochs = total_epochs

    def get_stats(self) -> Dict[str, Any]:
        """Get sampler statistics."""
        return {
            "mean_priority": float(self.priorities.mean()),
            "max_priority": float(self.priorities.max()),
            "min_priority": float(self.priorities.min()),
            "std_priority": float(self.priorities.std()),
            "seen_ratio": float(self.seen.sum() / self.dataset_size),
            "current_beta": self.beta if not self.beta_schedule else min(
                1.0, self.beta_start + (1.0 - self.beta_start) * (self.epoch / max(1, self.total_epochs))
            ),
        }


def generate_nnue_dataset(
    db_paths: List[str],
    output_path: str,
    config: Optional[NNUEDatasetConfig] = None,
    max_samples: Optional[int] = None,
) -> int:
    """Generate an NPZ dataset file for NNUE training.

    Args:
        db_paths: List of SQLite database paths
        output_path: Path for output NPZ file
        config: Dataset configuration
        max_samples: Maximum samples to extract

    Returns:
        Number of samples extracted
    """
    dataset = NNUESQLiteDataset(
        db_paths=db_paths,
        config=config,
        cache_path=output_path,
        max_samples=max_samples,
    )
    return len(dataset)


def count_available_samples(
    db_paths: List[str],
    config: Optional[NNUEDatasetConfig] = None,
) -> Dict[str, int]:
    """Count available training samples without loading them.

    Args:
        db_paths: List of SQLite database paths
        config: Dataset configuration

    Returns:
        Dict with counts per database and total
    """
    config = config or NNUEDatasetConfig()
    counts = {}
    total = 0

    for db_path in db_paths:
        if not os.path.exists(db_path):
            counts[db_path] = 0
            continue

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()

            board_type_str = config.board_type.value.lower()
            query = """
                SELECT COUNT(*) FROM (
                    SELECT s.move_number
                    FROM games g
                    JOIN game_state_snapshots s ON g.game_id = s.game_id
                    WHERE g.game_status = 'completed'
                      AND g.winner IS NOT NULL
                      AND g.board_type = ?
                      AND g.num_players = ?
                      AND g.total_moves >= ?
                      AND s.move_number % ? = 0
                      AND COALESCE(g.excluded_from_training, 0) = 0
                )
            """
            cursor.execute(query, (
                board_type_str,
                config.num_players,
                config.min_game_length,
                config.sample_every_n_moves,
            ))
            count = cursor.fetchone()[0]
            counts[db_path] = count
            total += count
            conn.close()
        except Exception as e:
            logger.error(f"Error counting samples in {db_path}: {e}")
            counts[db_path] = 0

    counts['total'] = total
    return counts


def extract_features_from_state(
    state_dict: Dict[str, Any],
    board_type: str,
    num_players: int,
    current_player: int = 1,
) -> np.ndarray:
    """Extract NNUE features from a JSON state dictionary.

    This function converts a dictionary representation of a game state
    (as stored in holdout databases) into NNUE feature vectors.

    Args:
        state_dict: Dictionary representation of board state
        board_type: Board type string (e.g., "square8", "hexagonal")
        num_players: Number of players in the game
        current_player: Current player number (1-indexed)

    Returns:
        numpy array of NNUE features
    """
    from ..models import BoardType as BT

    # Parse board type
    board_type_lower = board_type.lower()
    if "hex" in board_type_lower:
        bt = BT.HEXAGONAL
    elif "19" in board_type_lower:
        bt = BT.SQUARE19
    else:
        bt = BT.SQUARE8

    # Get feature dimension for this board
    feature_dim = get_feature_dim(bt)

    try:
        # Try to reconstruct a minimal GameState from the dict
        # The state_dict may be a full GameState or just board data
        if "board" in state_dict or "boardType" in state_dict:
            # Full game state format - use pydantic to parse
            from ..models import GameState
            game_state = GameState.model_validate(state_dict)
            return extract_features_from_gamestate(game_state, current_player)

        # Otherwise, try to extract from board-only format
        # This handles legacy formats where only board data is stored
        rings = state_dict.get("rings", state_dict.get("cells", {}))
        stacks = state_dict.get("stacks", {})
        territories = state_dict.get("territories", state_dict.get("territory", {}))

        # Build feature vector manually for simple board formats
        # This is a simplified extraction for compatibility
        features = np.zeros(feature_dim, dtype=np.float32)

        # Encode ring positions
        for pos_key, ring_data in rings.items():
            if isinstance(ring_data, dict):
                owner = ring_data.get("owner", ring_data.get("player", 0))
            else:
                owner = ring_data if isinstance(ring_data, int) else 0

            if owner > 0:
                # Simple encoding: hash position to feature index
                pos_hash = hash(pos_key) % (feature_dim // 4)
                # Rotate perspective to current player
                rotated_owner = ((owner - current_player) % num_players) + 1
                features[pos_hash + (rotated_owner - 1) * (feature_dim // 4)] = 1.0

        # Encode stack positions
        for pos_key, stack_data in stacks.items():
            if isinstance(stack_data, dict):
                owner = stack_data.get("owner", stack_data.get("player", 0))
                height = stack_data.get("height", stack_data.get("count", 1))
            elif isinstance(stack_data, list):
                owner = stack_data[0] if stack_data else 0
                height = len(stack_data)
            else:
                owner = stack_data if isinstance(stack_data, int) else 0
                height = 1

            if owner > 0:
                pos_hash = hash(pos_key) % (feature_dim // 4)
                rotated_owner = ((owner - current_player) % num_players) + 1
                offset = feature_dim // 2
                features[offset + pos_hash + (rotated_owner - 1) * (feature_dim // 8)] = height / 5.0

        return features

    except Exception as e:
        logger.warning(f"Failed to extract features from state dict: {e}")
        return np.zeros(feature_dim, dtype=np.float32)
