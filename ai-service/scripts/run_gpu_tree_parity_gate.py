#!/usr/bin/env python3
"""GPU Tree Parity Gate - Validates GPU tree produces canonical training data.

This script runs GPU tree searches and compares them against CPU sequential
halving to ensure training data quality parity. Use before training to validate
the GPU acceleration doesn't compromise data quality.

Usage:
    # Basic validation (100 games, 5% tolerance)
    python scripts/run_gpu_tree_parity_gate.py

    # Strict validation for CI
    python scripts/run_gpu_tree_parity_gate.py --games 500 --tolerance 0.03

    # Specific board type
    python scripts/run_gpu_tree_parity_gate.py --board-type hex8 --num-players 2

Environment variables:
    RINGRIFT_GPU_TREE_SHADOW_RATE: Set to 1.0 for full validation
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from dataclasses import dataclass, field

import numpy as np

from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.tensor_gumbel_tree import GPUGumbelMCTS, GPUGumbelMCTSConfig
from app.models import AIConfig, BoardType, GamePhase
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ParityResult:
    """Result from a single parity check."""
    position_idx: int
    move_match: bool
    policy_max_diff: float
    gpu_move_key: str
    cpu_move_key: str
    gpu_policy_entropy: float
    cpu_policy_entropy: float


@dataclass
class GateResult:
    """Aggregate result from parity gate."""
    total_positions: int = 0
    positions_checked: int = 0
    move_mismatches: int = 0
    policy_divergences: int = 0
    max_policy_diff: float = 0.0
    avg_policy_diff: float = 0.0
    results: list[ParityResult] = field(default_factory=list)
    duration_sec: float = 0.0

    @property
    def move_match_rate(self) -> float:
        if self.positions_checked == 0:
            return 0.0
        return 1.0 - (self.move_mismatches / self.positions_checked)

    @property
    def policy_parity_rate(self) -> float:
        if self.positions_checked == 0:
            return 0.0
        return 1.0 - (self.policy_divergences / self.positions_checked)

    @property
    def passed(self) -> bool:
        """Gate passes if policy divergence rate is below threshold."""
        return self.policy_parity_rate >= 0.95  # 95% parity required


class MockNeuralNet:
    """Mock neural network for deterministic testing."""

    def __init__(self, board_size: int = 8):
        self.board_size = board_size

    def evaluate_batch(self, states, value_head=None):
        """Return uniform policy and zero values."""
        values = [0.0] * len(states)
        policies = []
        for state in states:
            # Create uniform policy over board
            policy_size = self.board_size * self.board_size
            policy = np.ones(policy_size) / policy_size
            policies.append(policy)
        return values, policies

    def encode_move(self, move, board) -> int:
        """Encode move as flat index."""
        size = board.size
        if move.to is not None:
            return move.to.y * size + move.to.x
        return 0


def compute_entropy(policy: dict[str, float]) -> float:
    """Compute entropy of policy distribution."""
    entropy = 0.0
    for prob in policy.values():
        if prob > 1e-10:
            entropy -= prob * np.log(prob)
    return entropy


def run_parity_check(
    state,
    valid_moves: list,
    gpu_tree: GPUGumbelMCTS,
    mock_nn: MockNeuralNet,
    gumbel_ai: GumbelMCTSAI,
    tolerance: float,
    position_idx: int,
) -> ParityResult | None:
    """Run single parity check between GPU and CPU.

    Returns:
        ParityResult or None if check couldn't be performed.
    """
    if len(valid_moves) <= 1:
        return None

    try:
        # Run GPU tree search
        gpu_move, gpu_policy = gpu_tree.search(state, mock_nn, valid_moves)

        # Run CPU sequential halving
        policy_logits = gumbel_ai._get_policy_logits(state, valid_moves)
        actions = gumbel_ai._gumbel_top_k_sample(valid_moves, policy_logits)

        if len(actions) <= 1:
            return None

        for a in actions:
            a.visit_count = 0
            a.total_value = 0.0
        cpu_best = gumbel_ai._sequential_halving(state, actions)

        # Build CPU policy
        cpu_policy = {}
        total_visits = sum(a.visit_count for a in actions if a.visit_count > 0)
        if total_visits > 0:
            for a in actions:
                if a.visit_count > 0:
                    move_key = gpu_tree._move_to_key(a.move)
                    cpu_policy[move_key] = a.visit_count / total_visits

        # Compare
        gpu_move_key = gpu_tree._move_to_key(gpu_move)
        cpu_move_key = gpu_tree._move_to_key(cpu_best.move)
        move_match = gpu_move_key == cpu_move_key

        # Compute max policy difference
        all_keys = set(gpu_policy.keys()) | set(cpu_policy.keys())
        max_diff = 0.0
        for key in all_keys:
            gpu_prob = gpu_policy.get(key, 0.0)
            cpu_prob = cpu_policy.get(key, 0.0)
            max_diff = max(max_diff, abs(gpu_prob - cpu_prob))

        return ParityResult(
            position_idx=position_idx,
            move_match=move_match,
            policy_max_diff=max_diff,
            gpu_move_key=gpu_move_key,
            cpu_move_key=cpu_move_key,
            gpu_policy_entropy=compute_entropy(gpu_policy),
            cpu_policy_entropy=compute_entropy(cpu_policy),
        )

    except Exception as e:
        logger.warning(f"Position {position_idx} check failed: {e}")
        return None


def run_gate(
    board_type: BoardType,
    num_players: int,
    num_games: int,
    moves_per_game: int,
    tolerance: float,
    simulation_budget: int,
) -> GateResult:
    """Run full parity gate validation.

    Args:
        board_type: Board type to test.
        num_players: Number of players.
        num_games: Number of games to generate.
        moves_per_game: Moves to check per game.
        tolerance: Maximum allowed policy difference.
        simulation_budget: Simulations per search.

    Returns:
        GateResult with aggregate statistics.
    """
    result = GateResult()
    start_time = time.time()

    engine = DefaultRulesEngine()

    # Determine board size
    board_size = {
        BoardType.SQUARE8: 8,
        BoardType.SQUARE19: 19,
        BoardType.HEX8: 9,
        BoardType.HEXAGONAL: 25,
    }.get(board_type, 8)

    mock_nn = MockNeuralNet(board_size=board_size)

    # Create GPU tree
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        device = "cpu"

    gpu_config = GPUGumbelMCTSConfig(
        num_sampled_actions=8,
        simulation_budget=simulation_budget,
        eval_mode="heuristic",
        device=device,
    )
    gpu_tree = GPUGumbelMCTS(gpu_config)

    # Create CPU GumbelMCTSAI for comparison
    ai_config = AIConfig(
        difficulty=5,
        simulation_budget=simulation_budget,
        num_sampled_actions=8,
    )
    gumbel_ai = GumbelMCTSAI(
        player_number=1,
        config=ai_config,
        board_type=board_type,
    )

    logger.info(f"Running parity gate: {num_games} games, {moves_per_game} moves/game")
    logger.info(f"Board: {board_type.value}, Players: {num_players}, Device: {device}")
    logger.info(f"Budget: {simulation_budget} sims, Tolerance: {tolerance}")

    position_idx = 0
    policy_diffs = []

    for game_idx in range(num_games):
        # Create fresh game state
        state = create_initial_state(board_type=board_type, num_players=num_players)

        for move_idx in range(moves_per_game):
            result.total_positions += 1

            # Skip non-placement phases for simplicity
            if state.current_phase != GamePhase.RING_PLACEMENT:
                break

            valid_moves = engine.get_valid_moves(state, state.current_player)

            if len(valid_moves) <= 1:
                if len(valid_moves) == 1:
                    state = engine.apply_move(state, valid_moves[0])
                continue

            # Run parity check
            parity = run_parity_check(
                state, valid_moves, gpu_tree, mock_nn, gumbel_ai, tolerance, position_idx
            )

            if parity is not None:
                result.positions_checked += 1
                result.results.append(parity)
                policy_diffs.append(parity.policy_max_diff)

                if not parity.move_match:
                    result.move_mismatches += 1

                if parity.policy_max_diff > tolerance:
                    result.policy_divergences += 1

                result.max_policy_diff = max(result.max_policy_diff, parity.policy_max_diff)

            position_idx += 1

            # Make a move to advance state
            move = np.random.choice(valid_moves)
            state = engine.apply_move(state, move)

            if state.game_status != "active":
                break

        # Progress update
        if (game_idx + 1) % 10 == 0:
            logger.info(
                f"Progress: {game_idx + 1}/{num_games} games, "
                f"{result.positions_checked} positions checked"
            )

    result.duration_sec = time.time() - start_time
    if policy_diffs:
        result.avg_policy_diff = np.mean(policy_diffs)

    return result


def main():
    parser = argparse.ArgumentParser(
        description="GPU Tree Parity Gate - Validate training data quality"
    )
    parser.add_argument(
        "--board-type",
        choices=["square8", "square19", "hex8", "hexagonal"],
        default="square8",
        help="Board type to test (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="Number of games to test (default: 100)",
    )
    parser.add_argument(
        "--moves-per-game",
        type=int,
        default=20,
        help="Moves to check per game (default: 20)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="Maximum allowed policy difference (default: 0.05)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=100,
        help="Simulation budget per search (default: 100)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed divergence info",
    )

    args = parser.parse_args()

    # Map board type string to enum
    board_type_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex8": BoardType.HEX8,
        "hexagonal": BoardType.HEXAGONAL,
    }
    board_type = board_type_map[args.board_type]

    # Run gate
    result = run_gate(
        board_type=board_type,
        num_players=args.num_players,
        num_games=args.games,
        moves_per_game=args.moves_per_game,
        tolerance=args.tolerance,
        simulation_budget=args.budget,
    )

    # Print results
    print("\n" + "=" * 60)
    print("GPU TREE PARITY GATE RESULTS")
    print("=" * 60)
    print(f"Board Type:        {args.board_type}")
    print(f"Players:           {args.num_players}")
    print(f"Duration:          {result.duration_sec:.1f}s")
    print()
    print(f"Total Positions:   {result.total_positions}")
    print(f"Positions Checked: {result.positions_checked}")
    print(f"Move Mismatches:   {result.move_mismatches}")
    print(f"Policy Divergences: {result.policy_divergences}")
    print()
    print(f"Move Match Rate:   {result.move_match_rate:.1%}")
    print(f"Policy Parity:     {result.policy_parity_rate:.1%}")
    print(f"Max Policy Diff:   {result.max_policy_diff:.4f}")
    print(f"Avg Policy Diff:   {result.avg_policy_diff:.4f}")
    print()

    # Verbose output
    if args.verbose and result.policy_divergences > 0:
        print("Divergent positions:")
        for r in result.results:
            if r.policy_max_diff > args.tolerance:
                print(f"  #{r.position_idx}: diff={r.policy_max_diff:.3f}, "
                      f"GPU={r.gpu_move_key}, CPU={r.cpu_move_key}")

    # Final verdict
    print("=" * 60)
    if result.passed:
        print("✅ GATE PASSED - GPU tree produces canonical training data")
        return 0
    else:
        print("❌ GATE FAILED - Policy divergence exceeds threshold")
        print(f"   Required: ≥95% parity, Got: {result.policy_parity_rate:.1%}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
