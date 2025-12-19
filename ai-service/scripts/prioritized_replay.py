#!/usr/bin/env python3
"""Prioritized Experience Replay for RingRift AI Training.

Implements prioritized experience replay (PER) where training samples are
weighted by their "surprise" - how much the model can learn from them.

Benefits:
- More efficient learning by focusing on informative samples
- Faster convergence on difficult positions
- Automatic curriculum through surprise weighting
- Reduces wasted compute on already-learned positions

Reference: Schaul et al., "Prioritized Experience Replay" (2015)

Usage:
    # Build prioritized buffer from game database
    python scripts/prioritized_replay.py --build \
        --db data/games/selfplay.db \
        --output data/replay_buffer.pkl

    # Sample batch from buffer
    python scripts/prioritized_replay.py --sample \
        --buffer data/replay_buffer.pkl \
        --batch-size 256

    # Update priorities after training
    python scripts/prioritized_replay.py --update \
        --buffer data/replay_buffer.pkl \
        --indices 0,1,2,3 \
        --priorities 0.5,0.2,0.8,0.1
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sqlite3
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("prioritized_replay")


# PER hyperparameters
DEFAULT_ALPHA = 0.6  # Priority exponent (0 = uniform, 1 = full prioritization)
DEFAULT_BETA = 0.4  # Importance sampling exponent (anneals to 1)
DEFAULT_BETA_INCREMENT = 0.001  # Beta increment per sampling
DEFAULT_EPSILON = 1e-6  # Small constant to prevent zero priorities
DEFAULT_MAX_PRIORITY = 1.0  # Initial priority for new samples


@dataclass
class ReplayExperience:
    """A single experience in the replay buffer."""
    game_id: str
    move_idx: int
    board_type: str
    num_players: int
    state_encoding: Optional[np.ndarray] = None  # Board state features
    action: int = 0  # Move index
    value_target: float = 0.0  # Game outcome
    policy_target: Optional[np.ndarray] = None  # MCTS policy
    priority: float = DEFAULT_MAX_PRIORITY
    td_error: float = 0.0  # Temporal difference error


class SumTree:
    """Sum tree data structure for efficient prioritized sampling.

    Each leaf contains a priority value. Parent nodes contain the sum of
    their children. Allows O(log n) sampling proportional to priority.
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = [None] * capacity
        self.write_ptr = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float):
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, s: float) -> int:
        """Find leaf node for a given cumulative priority."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self) -> float:
        """Return total priority sum."""
        return self.tree[0]

    def add(self, priority: float, data: Any):
        """Add experience with given priority."""
        idx = self.write_ptr + self.capacity - 1

        self.data[self.write_ptr] = data
        self.update(idx, priority)

        self.write_ptr = (self.write_ptr + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def update(self, idx: int, priority: float):
        """Update priority at given index."""
        change = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, change)

    def get(self, s: float) -> Tuple[int, float, Any]:
        """Sample leaf by cumulative priority.

        Returns (tree_index, priority, data).
        """
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, self.tree[idx], self.data[data_idx]

    def min_priority(self) -> float:
        """Return minimum priority in the tree."""
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 0.0
        return min(self.tree[leaf_start:leaf_end])


class PrioritizedReplayBuffer:
    """Prioritized experience replay buffer using sum tree."""

    def __init__(
        self,
        capacity: int,
        alpha: float = DEFAULT_ALPHA,
        beta: float = DEFAULT_BETA,
        beta_increment: float = DEFAULT_BETA_INCREMENT,
        epsilon: float = DEFAULT_EPSILON,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        self.tree = SumTree(capacity)
        self.max_priority = DEFAULT_MAX_PRIORITY

    def add(self, experience: ReplayExperience):
        """Add experience with max priority (will be updated after training)."""
        priority = self.max_priority ** self.alpha
        self.tree.add(priority, experience)

    def sample(self, batch_size: int) -> Tuple[List[int], List[ReplayExperience], np.ndarray]:
        """Sample batch with priorities.

        Returns:
            (indices, experiences, importance_sampling_weights)
        """
        indices = []
        experiences = []
        priorities = []

        # Divide priority range into segments for stratified sampling
        segment = self.tree.total() / batch_size

        # Anneal beta
        self.beta = min(1.0, self.beta + self.beta_increment)

        for i in range(batch_size):
            # Sample from segment
            a = segment * i
            b = segment * (i + 1)
            s = np.random.uniform(a, b)

            idx, priority, experience = self.tree.get(s)

            if experience is not None:
                indices.append(idx)
                experiences.append(experience)
                priorities.append(priority)

        # Compute importance sampling weights
        priorities = np.array(priorities) + self.epsilon
        min_prob = (self.tree.min_priority() + self.epsilon) / (self.tree.total() + self.epsilon)
        max_weight = (min_prob * self.tree.n_entries) ** (-self.beta)

        probs = priorities / (self.tree.total() + self.epsilon)
        weights = (probs * self.tree.n_entries) ** (-self.beta)
        weights = weights / max_weight  # Normalize

        return indices, experiences, weights.astype(np.float32)

    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors."""
        for idx, error in zip(indices, td_errors):
            priority = (abs(error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)
            self.max_priority = max(self.max_priority, priority)

    def __len__(self) -> int:
        return self.tree.n_entries

    def save(self, path: Path):
        """Save buffer to disk."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "capacity": self.capacity,
                "alpha": self.alpha,
                "beta": self.beta,
                "beta_increment": self.beta_increment,
                "epsilon": self.epsilon,
                "tree": self.tree,
                "max_priority": self.max_priority,
            }, f)
        logger.info(f"Saved buffer ({len(self)} experiences) to {path}")

    @classmethod
    def load(cls, path: Path) -> "PrioritizedReplayBuffer":
        """Load buffer from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        buffer = cls(
            capacity=data["capacity"],
            alpha=data["alpha"],
            beta=data["beta"],
            beta_increment=data["beta_increment"],
            epsilon=data["epsilon"],
        )
        buffer.tree = data["tree"]
        buffer.max_priority = data["max_priority"]
        return buffer


def compute_td_error(
    experience: ReplayExperience,
    model_value: float,
) -> float:
    """Compute TD error for prioritization.

    TD error = |target - predicted|
    Higher error = more to learn from this sample.
    """
    return abs(experience.value_target - model_value)


def build_buffer_from_database(
    db_path: Path,
    board_type: Optional[str] = None,
    num_players: Optional[int] = None,
    capacity: int = 100000,
    max_games: int = 1000,
) -> PrioritizedReplayBuffer:
    """Build prioritized replay buffer from game database."""
    buffer = PrioritizedReplayBuffer(capacity=capacity)

    if not db_path.exists():
        logger.warning(f"Database not found: {db_path}")
        return buffer

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row

    query = """
        SELECT game_id, board_type, num_players, winner, move_history
        FROM games
        WHERE status = 'completed'
    """
    params = []

    if board_type:
        query += " AND board_type = ?"
        params.append(board_type)
    if num_players:
        query += " AND num_players = ?"
        params.append(num_players)

    query += " ORDER BY RANDOM() LIMIT ?"
    params.append(max_games)

    cursor = conn.execute(query, params)
    total_experiences = 0

    for row in cursor:
        try:
            move_history = json.loads(row["move_history"] or "[]")
        except json.JSONDecodeError:
            continue

        winner = row["winner"]
        game_id = row["game_id"]
        board_type = row["board_type"]
        num_players = row["num_players"]

        # Create experience for each move
        for move_idx, move in enumerate(move_history):
            player = move_idx % num_players

            # Compute value target (1 if won, 0 if lost, 0.5 if draw)
            if winner is None:
                value_target = 0.5
            elif winner == player:
                value_target = 1.0
            else:
                value_target = 0.0

            experience = ReplayExperience(
                game_id=game_id,
                move_idx=move_idx,
                board_type=board_type,
                num_players=num_players,
                action=move.get("action_idx", 0),
                value_target=value_target,
                priority=DEFAULT_MAX_PRIORITY,
            )

            buffer.add(experience)
            total_experiences += 1

            if total_experiences >= capacity:
                break

        if total_experiences >= capacity:
            break

    conn.close()
    logger.info(f"Built buffer with {len(buffer)} experiences from {max_games} games")
    return buffer


def demonstrate_prioritized_sampling(buffer: PrioritizedReplayBuffer, batch_size: int = 32):
    """Demonstrate prioritized sampling with importance weights."""
    if len(buffer) < batch_size:
        logger.warning(f"Buffer too small ({len(buffer)} < {batch_size})")
        return

    print("\n" + "=" * 70)
    print("PRIORITIZED SAMPLING DEMONSTRATION")
    print("=" * 70)

    indices, experiences, weights = buffer.sample(batch_size)

    print(f"\nSampled {len(experiences)} experiences")
    print(f"Weight range: [{weights.min():.4f}, {weights.max():.4f}]")
    print(f"Weight mean: {weights.mean():.4f}")

    # Show some samples
    print("\nSample experiences:")
    for i in range(min(5, len(experiences))):
        exp = experiences[i]
        print(f"  {i+1}. Game {exp.game_id[:8]}... move {exp.move_idx}, "
              f"value={exp.value_target:.1f}, weight={weights[i]:.4f}")

    # Simulate priority update
    print("\nSimulating priority update...")
    fake_td_errors = np.random.exponential(0.3, size=len(indices))
    buffer.update_priorities(indices, fake_td_errors)
    print(f"Updated {len(indices)} priorities")


def main():
    parser = argparse.ArgumentParser(
        description="Prioritized Experience Replay for RingRift AI"
    )

    parser.add_argument(
        "--build",
        action="store_true",
        help="Build buffer from database",
    )
    parser.add_argument(
        "--sample",
        action="store_true",
        help="Sample batch from buffer",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="Update priorities",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Demonstrate prioritized sampling",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=str(AI_SERVICE_ROOT / "data" / "games" / "selfplay.db"),
        help="Game database path",
    )
    parser.add_argument(
        "--buffer",
        type=str,
        default=str(AI_SERVICE_ROOT / "data" / "replay_buffer.pkl"),
        help="Replay buffer path",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output path (for --build)",
    )
    parser.add_argument(
        "--board",
        type=str,
        help="Filter by board type",
    )
    parser.add_argument(
        "--players",
        type=int,
        help="Filter by player count",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=100000,
        help="Buffer capacity",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=1000,
        help="Maximum games to load",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for sampling",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help="Priority exponent",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=DEFAULT_BETA,
        help="Importance sampling exponent",
    )
    parser.add_argument(
        "--indices",
        type=str,
        help="Comma-separated indices for update",
    )
    parser.add_argument(
        "--priorities",
        type=str,
        help="Comma-separated priorities for update",
    )

    args = parser.parse_args()

    if args.build:
        output_path = Path(args.output or args.buffer)
        buffer = build_buffer_from_database(
            db_path=Path(args.db),
            board_type=args.board,
            num_players=args.players,
            capacity=args.capacity,
            max_games=args.max_games,
        )
        buffer.save(output_path)
        print(f"\nBuffer saved to {output_path}")
        return 0

    if args.sample or args.demo:
        buffer_path = Path(args.buffer)
        if not buffer_path.exists():
            logger.error(f"Buffer not found: {buffer_path}")
            return 1

        buffer = PrioritizedReplayBuffer.load(buffer_path)
        logger.info(f"Loaded buffer with {len(buffer)} experiences")

        demonstrate_prioritized_sampling(buffer, args.batch_size)
        return 0

    if args.update:
        if not args.indices or not args.priorities:
            logger.error("--indices and --priorities required for update")
            return 1

        buffer_path = Path(args.buffer)
        buffer = PrioritizedReplayBuffer.load(buffer_path)

        indices = [int(x) for x in args.indices.split(",")]
        priorities = np.array([float(x) for x in args.priorities.split(",")])

        buffer.update_priorities(indices, priorities)
        buffer.save(buffer_path)
        print(f"Updated {len(indices)} priorities")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
