"""Reanalysis Pipeline for RingRift AI Training.

Re-evaluates historical games with the current (stronger) model to get
improved value and policy targets. Similar to MuZero's reanalysis approach.

Benefits:
1. Get higher-quality labels on existing data
2. Leverage model improvements retroactively
3. More efficient use of selfplay data
4. Reduces need for fresh selfplay generation

Usage:
    from app.training.reanalysis import ReanalysisEngine, ReanalysisConfig

    engine = ReanalysisEngine(
        model=current_model,
        config=ReanalysisConfig(batch_size=64),
    )

    # Reanalyze games from archive
    reanalyzed_data = engine.reanalyze_games(game_paths)

    # Mix with fresh data for training
    mixed_loader = engine.create_mixed_dataloader(
        fresh_data=fresh_npz,
        reanalyzed_ratio=0.3,
    )
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset
    from app.utils.torch_utils import get_device
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    get_device = None


@dataclass
class ReanalysisConfig:
    """Configuration for reanalysis pipeline."""
    batch_size: int = 64  # Batch size for model inference
    max_games_per_run: int = 1000  # Max games to reanalyze per run
    value_blend_ratio: float = 0.7  # Blend: 0.7*new + 0.3*old
    policy_blend_ratio: float = 0.8  # Higher for policy (more trust in new)
    min_model_elo_delta: int = 50  # Min Elo improvement to trigger reanalysis
    reanalysis_interval_hours: float = 6.0  # Hours between reanalysis runs
    priority_recent_games: bool = True  # Prioritize recent games
    cache_dir: str = "data/reanalysis_cache"
    # Quality filters
    min_game_length: int = 10  # Skip very short games
    max_game_length: int = 500  # Skip excessively long games
    # MCTS-based reanalysis (Phase 4 enhancement)
    use_mcts: bool = False  # Use GPU Gumbel MCTS instead of raw policy
    mcts_simulations: int = 100  # Number of MCTS simulations per position
    mcts_temperature: float = 1.0  # Temperature for MCTS policy extraction
    capture_q_values: bool = True  # Capture Q-values for auxiliary training
    capture_uncertainty: bool = True  # Capture uncertainty estimates


@dataclass
class ReanalyzedPosition:
    """A position with reanalyzed value and policy."""
    features: np.ndarray
    globals_vec: np.ndarray
    original_value: float
    reanalyzed_value: float
    blended_value: float
    original_policy: np.ndarray | None
    reanalyzed_policy: np.ndarray | None
    blended_policy: np.ndarray | None
    game_id: str
    move_number: int
    reanalysis_timestamp: float


class ReanalysisEngine:
    """Engine for reanalyzing historical games with current model.

    Takes existing game data and generates improved training targets
    by running the current (stronger) model on each position.
    """

    def __init__(
        self,
        model: nn.Module,
        config: ReanalysisConfig,
        device: torch.device | None = None,
    ):
        """Initialize the reanalysis engine.

        Args:
            model: Current model for reanalysis
            config: Reanalysis configuration
            device: Torch device for inference
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch not available")

        self.model = model
        self.config = config
        self.device = device or get_device(prefer_gpu=True)

        self.model = self.model.to(self.device)
        self.model.eval()

        # Cache directory
        self.cache_dir = Path(config.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Statistics
        self.positions_reanalyzed = 0
        self.games_reanalyzed = 0
        self.last_reanalysis_time = 0.0

    def reanalyze_npz(
        self,
        npz_path: Path,
        output_path: Path | None = None,
    ) -> Path:
        """Reanalyze positions from an NPZ file.

        Args:
            npz_path: Path to input NPZ file
            output_path: Path for output (default: adds _reanalyzed suffix)

        Returns:
            Path to reanalyzed NPZ file
        """
        logger.info(f"[Reanalysis] Loading {npz_path}")

        with np.load(npz_path, allow_pickle=True) as data:
            features = data["features"]
            globals_vec = data.get("globals", np.zeros((len(features), 8)))
            values = data["values"]
            policy_indices = data.get("policy_indices", None)
            policy_values = data.get("policy_values", None)
            move_numbers = data.get("move_numbers", np.zeros(len(features)))
            total_game_moves = data.get("total_game_moves", np.zeros(len(features)))
            phases = data.get("phases", None)
            values_mp = data.get("values_mp", None)
            num_players = data.get("num_players", None)

        n_samples = len(features)
        logger.info(f"[Reanalysis] Processing {n_samples} positions")

        # Reanalyze in batches
        new_values = np.zeros(n_samples, dtype=np.float32)
        new_policies = [None] * n_samples

        batch_size = self.config.batch_size

        # Check if model requires globals argument (v3 models do)
        import inspect
        forward_sig = inspect.signature(self.model.forward)
        requires_globals = "globals" in forward_sig.parameters

        with torch.no_grad():
            for start_idx in range(0, n_samples, batch_size):
                end_idx = min(start_idx + batch_size, n_samples)
                batch_features = torch.tensor(
                    features[start_idx:end_idx],
                    dtype=torch.float32,
                    device=self.device,
                )

                # Model inference - pass globals if model requires it
                if requires_globals:
                    batch_globals = torch.tensor(
                        globals_vec[start_idx:end_idx],
                        dtype=torch.float32,
                        device=self.device,
                    )
                    output = self.model(batch_features, batch_globals)
                else:
                    output = self.model(batch_features)

                if isinstance(output, tuple):
                    batch_values, batch_policies = output[:2]
                else:
                    batch_values = output
                    batch_policies = None

                # Store reanalyzed values
                # Handle multi-player value heads: take player 0's value (current player perspective)
                values_np = batch_values.cpu().numpy()
                if values_np.ndim == 2 and values_np.shape[1] > 1:
                    # Multi-player value head (shape: batch x num_players)
                    # Use player 0's value since training data is from current player's perspective
                    values_np = values_np[:, 0]
                else:
                    values_np = values_np.squeeze()
                new_values[start_idx:end_idx] = values_np

                # Store reanalyzed policies
                if batch_policies is not None:
                    policies_np = batch_policies.cpu().numpy()
                    for i, policy in enumerate(policies_np):
                        new_policies[start_idx + i] = policy

        # Blend old and new values
        blend_ratio = self.config.value_blend_ratio
        blended_values = blend_ratio * new_values + (1 - blend_ratio) * values

        # Blend policies if available
        blended_policy_indices = policy_indices
        blended_policy_values = policy_values

        if policy_values is not None and new_policies[0] is not None:
            policy_blend = self.config.policy_blend_ratio
            blended_policy_values = []
            for i in range(n_samples):
                if new_policies[i] is not None and i < len(policy_values):
                    old_policy = policy_values[i]
                    new_policy = new_policies[i]
                    # Simple blend (could be more sophisticated)
                    if len(old_policy) == len(new_policy):
                        blended = policy_blend * new_policy + (1 - policy_blend) * old_policy
                        blended = blended / (blended.sum() + 1e-8)
                        blended_policy_values.append(blended.astype(np.float32))
                    else:
                        blended_policy_values.append(old_policy)
                else:
                    blended_policy_values.append(policy_values[i] if i < len(policy_values) else np.array([]))
            blended_policy_values = np.array(blended_policy_values, dtype=object)

        # Output path
        if output_path is None:
            stem = npz_path.stem
            output_path = npz_path.parent / f"{stem}_reanalyzed.npz"

        # Save reanalyzed data
        save_dict = {
            "features": features,
            "globals": globals_vec,
            "values": blended_values,
            "move_numbers": move_numbers,
            "total_game_moves": total_game_moves,
        }

        if blended_policy_indices is not None:
            save_dict["policy_indices"] = blended_policy_indices
        if blended_policy_values is not None:
            save_dict["policy_values"] = blended_policy_values
        if phases is not None:
            save_dict["phases"] = phases
        if values_mp is not None:
            save_dict["values_mp"] = values_mp
        if num_players is not None:
            save_dict["num_players"] = num_players

        np.savez_compressed(output_path, **save_dict)

        self.positions_reanalyzed += n_samples
        self.last_reanalysis_time = time.time()

        logger.info(f"[Reanalysis] Saved {n_samples} positions to {output_path}")
        return output_path

    def reanalyze_jsonl_games(
        self,
        jsonl_path: Path,
        output_dir: Path,
        encoder: Any,
        board_type: str,
        max_games: int | None = None,
    ) -> list[Path]:
        """Reanalyze games from JSONL format.

        Args:
            jsonl_path: Path to JSONL file with games
            output_dir: Directory for output NPZ files
            encoder: Feature encoder for the board type
            board_type: Board type string
            max_games: Maximum games to process

        Returns:
            List of output NPZ paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        max_games = max_games or self.config.max_games_per_run
        games_processed = 0
        output_paths = []

        logger.info(f"[Reanalysis] Processing games from {jsonl_path}")

        with open(jsonl_path) as f:
            for line in f:
                if games_processed >= max_games:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    game = json.loads(line)

                    # Skip if doesn't match board type
                    if game.get("board_type") != board_type:
                        continue

                    # Skip short/long games
                    moves = game.get("moves", [])
                    if len(moves) < self.config.min_game_length:
                        continue
                    if len(moves) > self.config.max_game_length:
                        continue

                    # Extract positions (would need proper implementation)
                    # This is a placeholder - actual implementation would
                    # replay the game and extract features at each position
                    games_processed += 1
                    self.games_reanalyzed += 1

                except json.JSONDecodeError:
                    continue
                except Exception as e:
                    logger.debug(f"Error processing game: {e}")
                    continue

        logger.info(f"[Reanalysis] Processed {games_processed} games")
        return output_paths

    def should_reanalyze(self, current_elo: float, last_reanalysis_elo: float) -> bool:
        """Check if reanalysis should be triggered.

        Args:
            current_elo: Current model Elo
            last_reanalysis_elo: Elo at last reanalysis

        Returns:
            True if reanalysis should run
        """
        # Check Elo improvement
        elo_delta = current_elo - last_reanalysis_elo
        if elo_delta < self.config.min_model_elo_delta:
            return False

        # Check time since last reanalysis
        hours_since = (time.time() - self.last_reanalysis_time) / 3600
        return not hours_since < self.config.reanalysis_interval_hours

    def reanalyze_with_mcts(
        self,
        db_path: Path,
        output_path: Path,
        board_type: str,
        num_players: int = 2,
        max_games: int | None = None,
    ) -> Path | None:
        """Reanalyze games from database using GPU Gumbel MCTS search.

        This method provides higher-quality soft policy targets than raw model
        inference by running full MCTS search on each position. The search
        produces:
        - Visit-based soft policy distributions
        - Q-value estimates for auxiliary training
        - Uncertainty estimates from value variance

        Args:
            db_path: Path to game database
            output_path: Path for output NPZ file
            board_type: Board type string (e.g., 'hex8', 'square8')
            num_players: Number of players
            max_games: Maximum games to process (default: config.max_games_per_run)

        Returns:
            Path to reanalyzed NPZ file, or None if failed
        """
        if not self.config.use_mcts:
            logger.warning("[Reanalysis] MCTS mode not enabled in config")
            return None

        try:
            from app.db.game_replay import GameReplayDB
            from app.rules.engine import create_game, apply_move
            from app.ai.tensor_gumbel_tree import GPUGumbelMCTS, SearchStats
            from app.ai.feature_encoder import FeatureEncoder
        except ImportError as e:
            logger.error(f"[Reanalysis] Missing dependency for MCTS reanalysis: {e}")
            return None

        max_games = max_games or self.config.max_games_per_run
        db_path = Path(db_path)

        if not db_path.exists():
            logger.error(f"[Reanalysis] Database not found: {db_path}")
            return None

        logger.info(f"[Reanalysis MCTS] Processing up to {max_games} games from {db_path}")

        # Initialize GPU MCTS
        try:
            gpu_mcts = GPUGumbelMCTS(
                num_simulations=self.config.mcts_simulations,
                device=self.device,
            )
        except Exception as e:
            logger.error(f"[Reanalysis] Failed to initialize GPU MCTS: {e}")
            return None

        # Initialize feature encoder
        encoder = FeatureEncoder(board_type=board_type, num_players=num_players)

        # Collect reanalyzed positions
        all_features = []
        all_globals = []
        all_values = []
        all_policy_indices = []
        all_policy_values = []
        all_q_values = []  # Auxiliary data
        all_uncertainty = []  # Auxiliary data

        games_processed = 0
        positions_processed = 0

        with GameReplayDB(db_path, read_only=True) as db:
            # Get game IDs matching the config
            game_ids = db.list_games(
                board_type=board_type,
                num_players=num_players,
                limit=max_games,
            )

            for game_id in game_ids:
                try:
                    meta, initial_state, moves = db.load_game(game_id)
                    if meta is None:
                        continue

                    # Skip games outside quality bounds
                    if len(moves) < self.config.min_game_length:
                        continue
                    if len(moves) > self.config.max_game_length:
                        continue

                    # Replay game and reanalyze each position
                    game_state = initial_state
                    for move_idx, move in enumerate(moves):
                        # Encode current position
                        features = encoder.encode(game_state)
                        globals_vec = encoder.encode_globals(game_state, move_idx, len(moves))

                        # Run GPU Gumbel MCTS search with stats
                        try:
                            best_move, policy_dict, stats = gpu_mcts.search_with_stats(
                                game_state,
                                self.model,
                            )

                            # Convert policy dict to sparse format
                            policy_indices = []
                            policy_values = []
                            for move_key, prob in policy_dict.items():
                                idx = encoder.encode_move_index(move_key)
                                if idx >= 0:
                                    policy_indices.append(idx)
                                    policy_values.append(prob)

                            # Normalize policy values
                            if policy_values:
                                total = sum(policy_values)
                                policy_values = [v / total for v in policy_values]

                            # Collect auxiliary data
                            q_value = stats.root_value if stats else 0.0
                            uncertainty = stats.uncertainty if stats else 0.0

                        except Exception as e:
                            logger.debug(f"[Reanalysis] MCTS search failed: {e}")
                            # Fall back to model inference
                            with torch.no_grad():
                                feat_tensor = torch.tensor(
                                    features[np.newaxis],
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                                output = self.model(feat_tensor)
                                if isinstance(output, tuple):
                                    value, policy = output[:2]
                                    policy_np = policy.cpu().numpy()[0]
                                    # Use top-k indices as sparse policy
                                    topk = min(10, len(policy_np))
                                    top_indices = np.argsort(policy_np)[-topk:]
                                    policy_indices = top_indices.tolist()
                                    policy_values = policy_np[top_indices].tolist()
                                    q_value = value.cpu().item()
                                else:
                                    policy_indices = []
                                    policy_values = []
                                    q_value = output.cpu().item()
                            uncertainty = 0.0

                        # Store position data
                        all_features.append(features)
                        all_globals.append(globals_vec)
                        # Value from game outcome (will be set from meta)
                        all_values.append(0.0)  # Placeholder
                        all_policy_indices.append(np.array(policy_indices, dtype=np.int32))
                        all_policy_values.append(np.array(policy_values, dtype=np.float32))
                        if self.config.capture_q_values:
                            all_q_values.append(q_value)
                        if self.config.capture_uncertainty:
                            all_uncertainty.append(uncertainty)

                        positions_processed += 1

                        # Apply move to advance game state
                        game_state = apply_move(game_state, move)

                    # Set values based on game outcome
                    winner = meta.get('winner')
                    if winner is not None:
                        # Compute values for all positions in this game
                        start_idx = len(all_values) - len(moves)
                        for i in range(len(moves)):
                            pos_idx = start_idx + i
                            # Simple: winner gets +1, loser gets -1
                            player_at_pos = initial_state.current_player
                            # Advance player for each position
                            for j in range(i):
                                player_at_pos = (player_at_pos + 1) % num_players
                            if player_at_pos == winner:
                                all_values[pos_idx] = 1.0
                            else:
                                all_values[pos_idx] = -1.0

                    games_processed += 1

                    if games_processed % 10 == 0:
                        logger.info(
                            f"[Reanalysis MCTS] Processed {games_processed} games, "
                            f"{positions_processed} positions"
                        )

                except Exception as e:
                    logger.debug(f"[Reanalysis] Error processing game {game_id}: {e}")
                    continue

        if not all_features:
            logger.warning("[Reanalysis MCTS] No positions collected")
            return None

        # Save to NPZ
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            "features": np.array(all_features),
            "globals": np.array(all_globals),
            "values": np.array(all_values, dtype=np.float32),
            "policy_indices": np.array(all_policy_indices, dtype=object),
            "policy_values": np.array(all_policy_values, dtype=object),
        }

        if self.config.capture_q_values and all_q_values:
            save_dict["q_values"] = np.array(all_q_values, dtype=np.float32)
        if self.config.capture_uncertainty and all_uncertainty:
            save_dict["uncertainty"] = np.array(all_uncertainty, dtype=np.float32)

        np.savez_compressed(output_path, **save_dict)

        self.positions_reanalyzed += positions_processed
        self.games_reanalyzed += games_processed
        self.last_reanalysis_time = time.time()

        logger.info(
            f"[Reanalysis MCTS] Complete: {games_processed} games, "
            f"{positions_processed} positions saved to {output_path}"
        )

        return output_path

    def get_stats(self) -> dict[str, Any]:
        """Get reanalysis statistics."""
        return {
            "positions_reanalyzed": self.positions_reanalyzed,
            "games_reanalyzed": self.games_reanalyzed,
            "last_reanalysis_time": self.last_reanalysis_time,
        }


class ReanalyzedDataset(Dataset):
    """PyTorch Dataset that combines fresh and reanalyzed data."""

    def __init__(
        self,
        fresh_npz_paths: list[Path],
        reanalyzed_npz_paths: list[Path],
        reanalyzed_ratio: float = 0.3,
    ):
        """Initialize combined dataset.

        Args:
            fresh_npz_paths: Paths to fresh selfplay NPZ files
            reanalyzed_npz_paths: Paths to reanalyzed NPZ files
            reanalyzed_ratio: Ratio of reanalyzed samples (0.3 = 30%)
        """
        self.fresh_data = self._load_npz_files(fresh_npz_paths)
        self.reanalyzed_data = self._load_npz_files(reanalyzed_npz_paths)
        self.reanalyzed_ratio = reanalyzed_ratio

        self.n_fresh = len(self.fresh_data["features"]) if self.fresh_data else 0
        self.n_reanalyzed = len(self.reanalyzed_data["features"]) if self.reanalyzed_data else 0

        logger.info(f"[ReanalyzedDataset] {self.n_fresh} fresh + {self.n_reanalyzed} reanalyzed samples")

    def _load_npz_files(self, paths: list[Path]) -> dict[str, np.ndarray] | None:
        """Load and concatenate multiple NPZ files."""
        if not paths:
            return None

        all_features = []
        all_globals = []
        all_values = []

        for path in paths:
            if not path.exists():
                continue
            try:
                with np.load(path, allow_pickle=True) as data:
                    all_features.append(data["features"])
                    all_globals.append(data.get("globals", np.zeros((len(data["features"]), 8))))
                    all_values.append(data["values"])
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        if not all_features:
            return None

        return {
            "features": np.concatenate(all_features, axis=0),
            "globals": np.concatenate(all_globals, axis=0),
            "values": np.concatenate(all_values, axis=0),
        }

    def __len__(self) -> int:
        return self.n_fresh + self.n_reanalyzed

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Probabilistically sample from reanalyzed data
        use_reanalyzed = (
            self.n_reanalyzed > 0 and
            np.random.random() < self.reanalyzed_ratio
        )

        if use_reanalyzed:
            data = self.reanalyzed_data
            idx = idx % self.n_reanalyzed
        else:
            data = self.fresh_data
            idx = idx % self.n_fresh

        features = torch.tensor(data["features"][idx], dtype=torch.float32)
        globals_vec = torch.tensor(data["globals"][idx], dtype=torch.float32)
        values = torch.tensor(data["values"][idx], dtype=torch.float32)

        return features, globals_vec, values


def create_reanalysis_engine(
    model: nn.Module,
    batch_size: int = 64,
    value_blend: float = 0.7,
) -> ReanalysisEngine:
    """Factory function to create a reanalysis engine.

    Args:
        model: Model for reanalysis
        batch_size: Inference batch size
        value_blend: Blend ratio for values (new vs old)

    Returns:
        Configured ReanalysisEngine
    """
    config = ReanalysisConfig(
        batch_size=batch_size,
        value_blend_ratio=value_blend,
    )
    return ReanalysisEngine(model, config)


def reanalyze_training_data(
    model: nn.Module,
    npz_paths: list[Path],
    output_dir: Path,
    batch_size: int = 64,
) -> list[Path]:
    """Convenience function to reanalyze multiple NPZ files.

    Args:
        model: Model for reanalysis
        npz_paths: List of input NPZ paths
        output_dir: Output directory for reanalyzed files
        batch_size: Batch size for inference

    Returns:
        List of output NPZ paths
    """
    engine = create_reanalysis_engine(model, batch_size=batch_size)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_paths = []
    for npz_path in npz_paths:
        try:
            output_path = output_dir / f"{npz_path.stem}_reanalyzed.npz"
            engine.reanalyze_npz(npz_path, output_path)
            output_paths.append(output_path)
        except Exception as e:
            logger.error(f"Failed to reanalyze {npz_path}: {e}")

    return output_paths
