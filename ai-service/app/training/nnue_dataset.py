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
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset

from ..ai.nnue import (
    extract_features_from_gamestate,
    get_feature_dim,
)
from ..models import BoardType, GameState, Move
from ..rules.default_engine import DefaultRulesEngine

logger = logging.getLogger(__name__)


@dataclass
class NNUESample:
    """A single training sample for NNUE."""
    features: np.ndarray  # Shape: (feature_dim,)
    value: float  # Game outcome from player perspective: +1 win, -1 loss, 0 draw
    player_number: int
    game_id: str
    move_number: int


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
    ):
        """Initialize the NNUE dataset.

        Args:
            db_paths: List of paths to SQLite game databases
            config: Dataset configuration
            cache_path: Optional path to cache extracted features as NPZ
            max_samples: Optional limit on number of samples
        """
        self.db_paths = db_paths
        self.config = config or NNUEDatasetConfig()
        self.cache_path = cache_path
        self.max_samples = max_samples

        self.feature_dim = get_feature_dim(self.config.board_type)
        self.samples: List[NNUESample] = []

        # Load from cache if available
        if cache_path and os.path.exists(cache_path):
            self._load_from_cache(cache_path)
        else:
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

    def _extract_from_db(self, db_path: str) -> List[NNUESample]:
        """Extract samples from a single SQLite database."""
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
        """Load samples from NPZ cache file."""
        logger.info(f"Loading NNUE dataset from cache: {cache_path}")
        data = np.load(cache_path, allow_pickle=True)

        features = data['features']
        values = data['values']
        player_numbers = data['player_numbers']
        game_ids = data['game_ids']
        move_numbers = data['move_numbers']

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
    """

    def __init__(
        self,
        db_paths: List[str],
        config: Optional[NNUEDatasetConfig] = None,
        shuffle_games: bool = True,
        seed: Optional[int] = None,
    ):
        self.db_paths = db_paths
        self.config = config or NNUEDatasetConfig()
        self.shuffle_games = shuffle_games
        self.rng = np.random.default_rng(seed)
        self.feature_dim = get_feature_dim(self.config.board_type)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate over training samples."""
        db_paths = self.db_paths.copy()
        if self.shuffle_games:
            self.rng.shuffle(db_paths)

        for db_path in db_paths:
            if not os.path.exists(db_path):
                continue

            try:
                yield from self._stream_from_db(db_path)
            except Exception as e:
                logger.error(f"Error streaming from {db_path}: {e}")

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
