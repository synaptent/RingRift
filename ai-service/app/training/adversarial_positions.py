"""Adversarial Position Generation for RingRift AI.

Generates challenging board positions that expose model weaknesses.
Used to improve training data quality and model robustness.

Strategies:
1. Uncertainty-based: Positions where models disagree or are uncertain
2. Gradient-based: Positions that maximize prediction error
3. Search-based: Positions found by targeted game tree search
4. Replay-based: Positions where model made mistakes in past games
5. Perturbation-based: Slight modifications to known hard positions

Usage:
    from app.training.adversarial_positions import AdversarialGenerator

    generator = AdversarialGenerator(model_path="models/best.pt")

    # Generate adversarial positions
    positions = generator.generate(num_positions=100)

    # Evaluate model on adversarial positions
    accuracy = generator.evaluate_model_robustness(positions)
"""

from __future__ import annotations

import json
import logging
import random
import sqlite3
import time

from app.utils.checksum_utils import compute_bytes_checksum
from app.utils.datetime_utils import iso_now
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class AdversarialStrategy(Enum):
    """Strategies for generating adversarial positions."""
    UNCERTAINTY = "uncertainty"      # High model uncertainty
    DISAGREEMENT = "disagreement"    # Multiple models disagree
    GRADIENT = "gradient"            # Maximize loss via gradients
    SEARCH = "search"               # Targeted game tree search
    REPLAY = "replay"               # Mistakes from past games
    PERTURBATION = "perturbation"   # Perturb known hard positions
    BOUNDARY = "boundary"           # Near decision boundaries


@dataclass
class AdversarialPosition:
    """An adversarial board position."""
    position_id: str
    board_type: str
    num_players: int
    board_state: np.ndarray
    move_history: List[Any]
    strategy: AdversarialStrategy
    difficulty_score: float  # How challenging (0-1)
    uncertainty_score: float  # Model uncertainty (0-1)
    disagreement_score: float  # Multi-model disagreement (0-1)
    ground_truth_value: Optional[float] = None
    ground_truth_policy: Optional[np.ndarray] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""


@dataclass
class AdversarialConfig:
    """Configuration for adversarial generation."""
    # General settings
    num_positions: int = 100
    strategies: List[AdversarialStrategy] = field(default_factory=lambda: [
        AdversarialStrategy.UNCERTAINTY,
        AdversarialStrategy.DISAGREEMENT,
        AdversarialStrategy.REPLAY,
    ])
    strategy_weights: Dict[str, float] = field(default_factory=lambda: {
        "uncertainty": 0.3,
        "disagreement": 0.2,
        "replay": 0.3,
        "perturbation": 0.1,
        "search": 0.1,
    })

    # Uncertainty thresholds
    min_uncertainty: float = 0.3
    min_disagreement: float = 0.25

    # Search settings
    search_depth: int = 3
    search_width: int = 10

    # Perturbation settings
    max_perturbations: int = 3
    perturbation_prob: float = 0.1

    # Filtering
    dedupe: bool = True
    max_game_length: int = 200
    min_game_length: int = 5


class PositionEvaluator:
    """Evaluates positions for adversarial potential."""

    def __init__(
        self,
        model_paths: Optional[List[Path]] = None,
        device: str = "cpu",
    ):
        """Initialize evaluator.

        Args:
            model_paths: Paths to models for evaluation
            device: Device for inference
        """
        self.model_paths = model_paths or []
        self.device = device
        self.models = []
        self._load_models()

    def _load_models(self):
        """Load models for evaluation."""
        try:
            import torch
            for path in self.model_paths:
                if path.exists():
                    model = torch.load(path, map_location=self.device)
                    if hasattr(model, "eval"):
                        model.eval()
                    self.models.append(model)
        except ImportError:
            logger.warning("PyTorch not available, using dummy evaluator")

    def evaluate_uncertainty(
        self,
        features: np.ndarray,
    ) -> Tuple[float, np.ndarray, float]:
        """Evaluate model uncertainty on a position.

        Args:
            features: Board features

        Returns:
            Tuple of (uncertainty, policy, value)
        """
        if not self.models:
            # Dummy evaluation
            return random.random(), np.ones(64) / 64, 0.5

        try:
            import torch
            import torch.nn.functional as F

            with torch.no_grad():
                input_tensor = torch.tensor(features, dtype=torch.float32)
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                input_tensor = input_tensor.to(self.device)

                policies = []
                values = []

                for model in self.models:
                    output = model(input_tensor)
                    if isinstance(output, tuple):
                        policy_logits, value = output
                    else:
                        policy_logits = output
                        value = torch.tensor([0.5])

                    policy = F.softmax(policy_logits, dim=-1)
                    policies.append(policy.cpu().numpy().flatten())
                    values.append(float(value.cpu().numpy().flatten()[0]))

                # Compute uncertainty from disagreement
                if len(policies) > 1:
                    stacked = np.stack(policies)
                    # Policy entropy as uncertainty
                    mean_policy = np.mean(stacked, axis=0)
                    entropy = -np.sum(mean_policy * np.log(mean_policy + 1e-10))
                    max_entropy = np.log(len(mean_policy))
                    uncertainty = entropy / max_entropy

                    # Value disagreement
                    value_std = np.std(values)
                    uncertainty = 0.7 * uncertainty + 0.3 * min(value_std * 2, 1.0)
                else:
                    uncertainty = 0.0

                return uncertainty, policies[0], values[0]

        except Exception as e:
            logger.error(f"Error evaluating uncertainty: {e}")
            return random.random(), np.ones(64) / 64, 0.5

    def evaluate_disagreement(
        self,
        features: np.ndarray,
    ) -> float:
        """Evaluate disagreement between multiple models.

        Returns disagreement score (0-1).
        """
        if len(self.models) < 2:
            return 0.0

        try:
            import torch
            import torch.nn.functional as F

            with torch.no_grad():
                input_tensor = torch.tensor(features, dtype=torch.float32)
                if input_tensor.dim() == 3:
                    input_tensor = input_tensor.unsqueeze(0)
                input_tensor = input_tensor.to(self.device)

                best_moves = []
                for model in self.models:
                    output = model(input_tensor)
                    if isinstance(output, tuple):
                        policy_logits = output[0]
                    else:
                        policy_logits = output
                    best_move = torch.argmax(policy_logits).item()
                    best_moves.append(best_move)

                # Count unique best moves
                from collections import Counter
                move_counts = Counter(best_moves)
                most_common = move_counts.most_common(1)[0][1]
                agreement = most_common / len(best_moves)

                return 1.0 - agreement

        except Exception as e:
            logger.error(f"Error evaluating disagreement: {e}")
            return 0.0


class AdversarialGenerator:
    """Generates adversarial board positions."""

    def __init__(
        self,
        model_paths: Optional[List[Path]] = None,
        config: Optional[AdversarialConfig] = None,
        game_db_path: Optional[Path] = None,
        device: str = "cpu",
    ):
        """Initialize generator.

        Args:
            model_paths: Paths to models for evaluation
            config: Generation configuration
            game_db_path: Path to game database for replay-based generation
            device: Device for inference
        """
        self.model_paths = model_paths or []
        self.config = config or AdversarialConfig()
        self.game_db_path = game_db_path
        self.device = device

        self.evaluator = PositionEvaluator(model_paths, device)
        self._seen_positions: Set[str] = set()

    def _hash_position(self, board_state: np.ndarray) -> str:
        """Generate hash for position deduplication."""
        return compute_bytes_checksum(board_state.tobytes(), algorithm="md5", truncate=16)

    def generate(
        self,
        num_positions: Optional[int] = None,
        board_type: str = "square8",
        num_players: int = 2,
    ) -> List[AdversarialPosition]:
        """Generate adversarial positions.

        Args:
            num_positions: Number of positions to generate
            board_type: Board type
            num_players: Number of players

        Returns:
            List of adversarial positions
        """
        num_positions = num_positions or self.config.num_positions
        positions = []

        # Calculate positions per strategy based on weights
        strategy_counts = {}
        for strategy in self.config.strategies:
            weight = self.config.strategy_weights.get(strategy.value, 0.1)
            strategy_counts[strategy] = int(num_positions * weight)

        # Ensure we generate at least num_positions
        remaining = num_positions - sum(strategy_counts.values())
        if remaining > 0 and self.config.strategies:
            strategy_counts[self.config.strategies[0]] += remaining

        # Generate positions for each strategy
        for strategy, count in strategy_counts.items():
            if count <= 0:
                continue

            logger.info(f"Generating {count} positions using {strategy.value} strategy")

            if strategy == AdversarialStrategy.UNCERTAINTY:
                new_positions = self._generate_uncertainty_based(
                    count, board_type, num_players
                )
            elif strategy == AdversarialStrategy.DISAGREEMENT:
                new_positions = self._generate_disagreement_based(
                    count, board_type, num_players
                )
            elif strategy == AdversarialStrategy.REPLAY:
                new_positions = self._generate_replay_based(
                    count, board_type, num_players
                )
            elif strategy == AdversarialStrategy.PERTURBATION:
                new_positions = self._generate_perturbation_based(
                    count, board_type, num_players
                )
            elif strategy == AdversarialStrategy.SEARCH:
                new_positions = self._generate_search_based(
                    count, board_type, num_players
                )
            elif strategy == AdversarialStrategy.BOUNDARY:
                new_positions = self._generate_boundary_based(
                    count, board_type, num_players
                )
            else:
                new_positions = []

            positions.extend(new_positions)

        # Deduplicate if configured
        if self.config.dedupe:
            unique_positions = []
            for pos in positions:
                pos_hash = self._hash_position(pos.board_state)
                if pos_hash not in self._seen_positions:
                    self._seen_positions.add(pos_hash)
                    unique_positions.append(pos)
            positions = unique_positions

        logger.info(f"Generated {len(positions)} adversarial positions")
        return positions

    def _generate_uncertainty_based(
        self,
        count: int,
        board_type: str,
        num_players: int,
    ) -> List[AdversarialPosition]:
        """Generate positions with high model uncertainty."""
        positions = []

        # Generate random positions and keep high-uncertainty ones
        attempts = 0
        max_attempts = count * 10

        while len(positions) < count and attempts < max_attempts:
            attempts += 1

            # Generate random position
            board_state, move_history = self._generate_random_position(
                board_type, num_players
            )

            if board_state is None:
                continue

            # Evaluate uncertainty
            features = self._state_to_features(board_state, board_type)
            uncertainty, policy, value = self.evaluator.evaluate_uncertainty(features)

            if uncertainty >= self.config.min_uncertainty:
                position = AdversarialPosition(
                    position_id=f"unc_{len(positions)}_{int(time.time())}",
                    board_type=board_type,
                    num_players=num_players,
                    board_state=board_state,
                    move_history=move_history,
                    strategy=AdversarialStrategy.UNCERTAINTY,
                    difficulty_score=uncertainty,
                    uncertainty_score=uncertainty,
                    disagreement_score=0.0,
                    created_at=iso_now(),
                )
                positions.append(position)

        return positions

    def _generate_disagreement_based(
        self,
        count: int,
        board_type: str,
        num_players: int,
    ) -> List[AdversarialPosition]:
        """Generate positions where models disagree."""
        positions = []

        if len(self.evaluator.models) < 2:
            logger.warning("Need at least 2 models for disagreement-based generation")
            return positions

        attempts = 0
        max_attempts = count * 10

        while len(positions) < count and attempts < max_attempts:
            attempts += 1

            board_state, move_history = self._generate_random_position(
                board_type, num_players
            )

            if board_state is None:
                continue

            features = self._state_to_features(board_state, board_type)
            disagreement = self.evaluator.evaluate_disagreement(features)
            uncertainty, _, _ = self.evaluator.evaluate_uncertainty(features)

            if disagreement >= self.config.min_disagreement:
                position = AdversarialPosition(
                    position_id=f"dis_{len(positions)}_{int(time.time())}",
                    board_type=board_type,
                    num_players=num_players,
                    board_state=board_state,
                    move_history=move_history,
                    strategy=AdversarialStrategy.DISAGREEMENT,
                    difficulty_score=disagreement,
                    uncertainty_score=uncertainty,
                    disagreement_score=disagreement,
                    created_at=iso_now(),
                )
                positions.append(position)

        return positions

    def _generate_replay_based(
        self,
        count: int,
        board_type: str,
        num_players: int,
    ) -> List[AdversarialPosition]:
        """Generate positions from past game mistakes."""
        positions = []

        if not self.game_db_path or not self.game_db_path.exists():
            logger.warning("No game database for replay-based generation")
            return self._generate_uncertainty_based(count, board_type, num_players)

        try:
            conn = sqlite3.connect(str(self.game_db_path))
            cursor = conn.cursor()

            # Find games where AI lost or played suboptimally
            cursor.execute("""
                SELECT game_id, move_history, winner
                FROM games
                WHERE status = 'completed'
                AND board_type = ?
                AND num_players = ?
                ORDER BY created_at DESC
                LIMIT ?
            """, (board_type, num_players, count * 5))

            games = cursor.fetchall()
            conn.close()

            for game_id, move_history_json, winner in games:
                if len(positions) >= count:
                    break

                try:
                    move_history = json.loads(move_history_json or "[]")
                except json.JSONDecodeError:
                    continue

                # Sample positions from losing games
                if len(move_history) < self.config.min_game_length:
                    continue

                # Pick random positions from the game
                num_samples = min(3, len(move_history) - 5)
                for _ in range(num_samples):
                    if len(positions) >= count:
                        break

                    # Random position index
                    idx = random.randint(5, len(move_history) - 1)
                    partial_history = move_history[:idx]

                    # Reconstruct board state
                    board_state = self._reconstruct_board(
                        board_type, num_players, partial_history
                    )

                    if board_state is None:
                        continue

                    features = self._state_to_features(board_state, board_type)
                    uncertainty, _, _ = self.evaluator.evaluate_uncertainty(features)

                    position = AdversarialPosition(
                        position_id=f"rep_{len(positions)}_{game_id[:8]}",
                        board_type=board_type,
                        num_players=num_players,
                        board_state=board_state,
                        move_history=partial_history,
                        strategy=AdversarialStrategy.REPLAY,
                        difficulty_score=0.5 + 0.5 * uncertainty,
                        uncertainty_score=uncertainty,
                        disagreement_score=0.0,
                        metadata={"source_game": game_id, "move_idx": idx},
                        created_at=iso_now(),
                    )
                    positions.append(position)

        except Exception as e:
            logger.error(f"Error in replay-based generation: {e}")
            return self._generate_uncertainty_based(count, board_type, num_players)

        return positions

    def _generate_perturbation_based(
        self,
        count: int,
        board_type: str,
        num_players: int,
    ) -> List[AdversarialPosition]:
        """Generate positions by perturbing known difficult ones."""
        positions = []

        # First get some base positions
        base_positions = self._generate_uncertainty_based(
            count // 2, board_type, num_players
        )

        for base in base_positions:
            if len(positions) >= count:
                break

            # Create perturbations
            for _ in range(min(self.config.max_perturbations, count - len(positions))):
                perturbed_state = self._perturb_position(
                    base.board_state.copy(),
                    board_type,
                )

                if perturbed_state is None:
                    continue

                features = self._state_to_features(perturbed_state, board_type)
                uncertainty, _, _ = self.evaluator.evaluate_uncertainty(features)

                position = AdversarialPosition(
                    position_id=f"pert_{len(positions)}_{int(time.time())}",
                    board_type=board_type,
                    num_players=num_players,
                    board_state=perturbed_state,
                    move_history=base.move_history.copy(),
                    strategy=AdversarialStrategy.PERTURBATION,
                    difficulty_score=uncertainty,
                    uncertainty_score=uncertainty,
                    disagreement_score=0.0,
                    metadata={"base_position": base.position_id},
                    created_at=iso_now(),
                )
                positions.append(position)

        return positions

    def _generate_search_based(
        self,
        count: int,
        board_type: str,
        num_players: int,
    ) -> List[AdversarialPosition]:
        """Generate positions through targeted search."""
        positions = []

        # Use search to find positions that maximize uncertainty
        for i in range(count):
            # Start from random position
            board_state, move_history = self._generate_random_position(
                board_type, num_players
            )

            if board_state is None:
                continue

            # Greedy search for higher uncertainty
            best_state = board_state
            best_history = move_history
            best_uncertainty = 0.0

            for _ in range(self.config.search_depth):
                # Generate candidate moves
                candidates = self._generate_candidate_moves(
                    best_state, board_type, self.config.search_width
                )

                for candidate_state, candidate_history in candidates:
                    features = self._state_to_features(candidate_state, board_type)
                    uncertainty, _, _ = self.evaluator.evaluate_uncertainty(features)

                    if uncertainty > best_uncertainty:
                        best_uncertainty = uncertainty
                        best_state = candidate_state
                        best_history = candidate_history

            if best_uncertainty >= self.config.min_uncertainty:
                position = AdversarialPosition(
                    position_id=f"search_{i}_{int(time.time())}",
                    board_type=board_type,
                    num_players=num_players,
                    board_state=best_state,
                    move_history=best_history,
                    strategy=AdversarialStrategy.SEARCH,
                    difficulty_score=best_uncertainty,
                    uncertainty_score=best_uncertainty,
                    disagreement_score=0.0,
                    metadata={"search_depth": self.config.search_depth},
                    created_at=iso_now(),
                )
                positions.append(position)

        return positions

    def _generate_boundary_based(
        self,
        count: int,
        board_type: str,
        num_players: int,
    ) -> List[AdversarialPosition]:
        """Generate positions near decision boundaries."""
        positions = []

        # Find positions where value is close to 0.5 (uncertain outcome)
        attempts = 0
        max_attempts = count * 10

        while len(positions) < count and attempts < max_attempts:
            attempts += 1

            board_state, move_history = self._generate_random_position(
                board_type, num_players
            )

            if board_state is None:
                continue

            features = self._state_to_features(board_state, board_type)
            uncertainty, policy, value = self.evaluator.evaluate_uncertainty(features)

            # Look for positions near decision boundary (value ~0.5)
            boundary_distance = abs(value - 0.5)
            if boundary_distance < 0.15:  # Within 15% of decision boundary
                position = AdversarialPosition(
                    position_id=f"bound_{len(positions)}_{int(time.time())}",
                    board_type=board_type,
                    num_players=num_players,
                    board_state=board_state,
                    move_history=move_history,
                    strategy=AdversarialStrategy.BOUNDARY,
                    difficulty_score=1.0 - boundary_distance * 2,
                    uncertainty_score=uncertainty,
                    disagreement_score=0.0,
                    metadata={"value": value, "boundary_distance": boundary_distance},
                    created_at=iso_now(),
                )
                positions.append(position)

        return positions

    def _generate_random_position(
        self,
        board_type: str,
        num_players: int,
    ) -> Tuple[Optional[np.ndarray], List[Any]]:
        """Generate a random valid board position."""
        # Board size based on type
        if board_type == "square8":
            size = 8
        elif board_type == "square19":
            size = 19
        elif board_type == "hexagonal":
            size = 11
        else:
            size = 8

        # Create empty board
        board_state = np.zeros((size, size), dtype=np.int8)

        # Randomly place some pieces
        num_moves = random.randint(
            self.config.min_game_length,
            min(self.config.max_game_length, size * size // 2)
        )

        move_history = []
        player = 0

        for i in range(num_moves):
            # Find empty positions
            empty = list(zip(*np.where(board_state == 0)))
            if not empty:
                break

            # Random move
            pos = random.choice(empty)
            board_state[pos] = player + 1
            move_history.append({"position": pos, "player": player})
            player = (player + 1) % num_players

        return board_state, move_history

    def _reconstruct_board(
        self,
        board_type: str,
        num_players: int,
        move_history: List[Any],
    ) -> Optional[np.ndarray]:
        """Reconstruct board state from move history."""
        try:
            if board_type == "square8":
                size = 8
            elif board_type == "square19":
                size = 19
            elif board_type == "hexagonal":
                size = 11
            else:
                size = 8

            board_state = np.zeros((size, size), dtype=np.int8)

            for move in move_history:
                if isinstance(move, dict):
                    pos = move.get("position") or move.get("pos")
                    player = move.get("player", 0)
                elif isinstance(move, (list, tuple)) and len(move) >= 2:
                    pos = move[:2]
                    player = move[2] if len(move) > 2 else 0
                else:
                    continue

                if pos and len(pos) >= 2:
                    r, c = int(pos[0]), int(pos[1])
                    if 0 <= r < size and 0 <= c < size:
                        board_state[r, c] = player + 1

            return board_state

        except Exception as e:
            logger.error(f"Error reconstructing board: {e}")
            return None

    def _state_to_features(
        self,
        board_state: np.ndarray,
        board_type: str,
    ) -> np.ndarray:
        """Convert board state to model input features."""
        # Simple feature representation
        # Channel 0: Empty
        # Channel 1: Player 1
        # Channel 2: Player 2
        # etc.

        size = board_state.shape[0]
        num_channels = 4  # Empty + 3 players max

        features = np.zeros((num_channels, size, size), dtype=np.float32)

        # Empty positions
        features[0] = (board_state == 0).astype(np.float32)

        # Player positions
        for p in range(1, num_channels):
            features[p] = (board_state == p).astype(np.float32)

        return features

    def _perturb_position(
        self,
        board_state: np.ndarray,
        board_type: str,
    ) -> Optional[np.ndarray]:
        """Create slight perturbation of a position."""
        size = board_state.shape[0]

        # Random perturbations
        for _ in range(random.randint(1, self.config.max_perturbations)):
            if random.random() < self.config.perturbation_prob:
                # Swap two pieces
                filled = list(zip(*np.where(board_state > 0)))
                empty = list(zip(*np.where(board_state == 0)))

                if filled and empty:
                    from_pos = random.choice(filled)
                    to_pos = random.choice(empty)
                    board_state[to_pos] = board_state[from_pos]
                    board_state[from_pos] = 0

            elif random.random() < self.config.perturbation_prob:
                # Add a piece
                empty = list(zip(*np.where(board_state == 0)))
                if empty:
                    pos = random.choice(empty)
                    board_state[pos] = random.randint(1, 2)

            elif random.random() < self.config.perturbation_prob:
                # Remove a piece
                filled = list(zip(*np.where(board_state > 0)))
                if filled:
                    pos = random.choice(filled)
                    board_state[pos] = 0

        return board_state

    def _generate_candidate_moves(
        self,
        board_state: np.ndarray,
        board_type: str,
        num_candidates: int,
    ) -> List[Tuple[np.ndarray, List[Any]]]:
        """Generate candidate next moves for search."""
        candidates = []
        empty = list(zip(*np.where(board_state == 0)))

        if not empty:
            return candidates

        # Sample random moves
        for _ in range(min(num_candidates, len(empty))):
            pos = random.choice(empty)
            empty.remove(pos)

            new_state = board_state.copy()
            new_state[pos] = 1  # Assume player 1's turn

            candidates.append((new_state, [{"position": pos, "player": 0}]))

        return candidates

    def evaluate_model_robustness(
        self,
        positions: List[AdversarialPosition],
    ) -> Dict[str, float]:
        """Evaluate model robustness on adversarial positions.

        Args:
            positions: Adversarial positions to test

        Returns:
            Dict with robustness metrics
        """
        if not positions:
            return {"robustness": 1.0, "avg_uncertainty": 0.0}

        uncertainties = []
        disagreements = []

        for pos in positions:
            features = self._state_to_features(pos.board_state, pos.board_type)
            uncertainty, _, _ = self.evaluator.evaluate_uncertainty(features)
            disagreement = self.evaluator.evaluate_disagreement(features)

            uncertainties.append(uncertainty)
            disagreements.append(disagreement)

        return {
            "robustness": 1.0 - np.mean(uncertainties),
            "avg_uncertainty": np.mean(uncertainties),
            "max_uncertainty": np.max(uncertainties),
            "avg_disagreement": np.mean(disagreements),
            "num_positions": len(positions),
        }

    def save_positions(
        self,
        positions: List[AdversarialPosition],
        output_path: Path,
    ):
        """Save positions to file.

        Args:
            positions: Positions to save
            output_path: Output file path
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = []
        for pos in positions:
            data.append({
                "position_id": pos.position_id,
                "board_type": pos.board_type,
                "num_players": pos.num_players,
                "board_state": pos.board_state.tolist(),
                "move_history": pos.move_history,
                "strategy": pos.strategy.value,
                "difficulty_score": pos.difficulty_score,
                "uncertainty_score": pos.uncertainty_score,
                "disagreement_score": pos.disagreement_score,
                "metadata": pos.metadata,
                "created_at": pos.created_at,
            })

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved {len(positions)} positions to {output_path}")

    @staticmethod
    def load_positions(input_path: Path) -> List[AdversarialPosition]:
        """Load positions from file.

        Args:
            input_path: Input file path

        Returns:
            List of adversarial positions
        """
        with open(input_path) as f:
            data = json.load(f)

        positions = []
        for item in data:
            positions.append(AdversarialPosition(
                position_id=item["position_id"],
                board_type=item["board_type"],
                num_players=item["num_players"],
                board_state=np.array(item["board_state"]),
                move_history=item["move_history"],
                strategy=AdversarialStrategy(item["strategy"]),
                difficulty_score=item["difficulty_score"],
                uncertainty_score=item["uncertainty_score"],
                disagreement_score=item["disagreement_score"],
                metadata=item.get("metadata", {}),
                created_at=item.get("created_at", ""),
            ))

        return positions


def generate_adversarial_training_data(
    model_paths: List[Path],
    game_db_path: Path,
    output_path: Path,
    num_positions: int = 1000,
    board_type: str = "square8",
    num_players: int = 2,
) -> Dict[str, Any]:
    """Generate adversarial training data for model improvement.

    Args:
        model_paths: Paths to current models
        game_db_path: Path to game database
        output_path: Where to save generated positions
        num_positions: Number of positions to generate
        board_type: Board type
        num_players: Number of players

    Returns:
        Generation statistics
    """
    config = AdversarialConfig(num_positions=num_positions)
    generator = AdversarialGenerator(
        model_paths=model_paths,
        config=config,
        game_db_path=game_db_path,
    )

    # Generate positions
    positions = generator.generate(
        num_positions=num_positions,
        board_type=board_type,
        num_players=num_players,
    )

    # Evaluate robustness
    metrics = generator.evaluate_model_robustness(positions)

    # Save positions
    generator.save_positions(positions, output_path)

    return {
        "num_generated": len(positions),
        "strategies_used": list(set(p.strategy.value for p in positions)),
        "avg_difficulty": np.mean([p.difficulty_score for p in positions]),
        "robustness_metrics": metrics,
        "output_path": str(output_path),
    }
