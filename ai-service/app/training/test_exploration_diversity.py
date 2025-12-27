"""Test exploration bonuses for diverse data generation.

Compares move diversity and position coverage with different
exploration settings (beta, gamma values).

Usage:
    python -m app.training.test_exploration_diversity --games 30
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.ai.factory import AIFactory
from app.ai.gmo_ai import GMOAI, GMOConfig
from app.game_engine import GameEngine
from app.models import AIConfig, AIType, BoardType, Move
from app.training.train_gmo_selfplay import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


@dataclass
class DiversityMetrics:
    """Metrics for measuring move/data diversity."""
    unique_moves: int  # Number of unique moves played
    total_moves: int  # Total moves played
    move_entropy: float  # Entropy of move distribution (higher = more diverse)
    position_coverage: float  # Fraction of board positions touched
    move_type_distribution: dict[str, int]  # Count per move type
    avg_game_length: float
    win_rate: float


def compute_move_hash(move: Move) -> str:
    """Create a hashable string representation of a move."""
    from_str = f"{move.from_pos.x},{move.from_pos.y}" if move.from_pos else "none"
    to_str = f"{move.to.x},{move.to.y}" if move.to else "none"
    return f"{move.type.value}:{from_str}:{to_str}:{move.placement_count}"


def compute_entropy(counts: list[int]) -> float:
    """Compute Shannon entropy from counts."""
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * np.log2(p) for p in probs)


def play_games_and_collect_data(
    gmo_config: GMOConfig,
    num_games: int,
    checkpoint_path: str | None = None,
) -> tuple[list[Move], DiversityMetrics]:
    """Play games with given config and collect diversity data.

    Returns:
        Tuple of (all_moves, metrics)
    """
    ai_config = AIConfig(difficulty=6)
    gmo_ai = GMOAI(player_number=1, config=ai_config, gmo_config=gmo_config)

    # Load checkpoint
    if checkpoint_path:
        gmo_ai.load_checkpoint(checkpoint_path)
    else:
        default_path = Path("models/gmo/gmo_best.pt")
        if default_path.exists():
            gmo_ai.load_checkpoint(default_path)

    # Create diverse opponents
    opponents = [
        ("random", AIFactory.create(AIType.RANDOM, player_number=2, config=AIConfig(difficulty=1))),
        ("heuristic", AIFactory.create(AIType.HEURISTIC, player_number=2, config=AIConfig(difficulty=4))),
    ]

    all_gmo_moves = []
    move_counter = Counter()
    move_type_counter = Counter()
    positions_touched = set()
    game_lengths = []
    wins = 0

    for game_idx in range(num_games):
        # Alternate opponents
        opp_name, opponent = opponents[game_idx % len(opponents)]

        # Alternate playing as P1/P2
        gmo_player = 1 if game_idx % 2 == 0 else 2
        opp_player = 2 if gmo_player == 1 else 1

        # Create fresh opponent with correct player number
        if opp_name == "random":
            opponent = AIFactory.create(AIType.RANDOM, player_number=opp_player, config=AIConfig(difficulty=1))
        else:
            opponent = AIFactory.create(AIType.HEURISTIC, player_number=opp_player, config=AIConfig(difficulty=4))

        gmo_ai.player_number = gmo_player
        # Use game-specific seed for varied but reproducible behavior
        game_seed = (game_idx * 12345 + 7919) & 0xFFFFFFFF
        gmo_ai.reset_for_new_game(rng_seed=game_seed)

        state = create_initial_state(
            game_id=f"diversity_{game_idx}",
            board_type=BoardType.SQUARE8,
            rng_seed=game_idx * 12345,
        )

        move_count = 0
        game_moves = []

        while state.game_status.value == "active" and move_count < 500:
            current_player = state.current_player
            legal_moves = GameEngine.get_valid_moves(state, current_player)

            if not legal_moves:
                requirement = GameEngine.get_phase_requirement(state, current_player)
                if requirement is not None:
                    move = GameEngine.synthesize_bookkeeping_move(requirement, state)
                    if move:
                        state = GameEngine.apply_move(state, move)
                        move_count += 1
                        continue
                break

            if current_player == gmo_player:
                move = gmo_ai.select_move(state)
                if move:
                    game_moves.append(move)
                    move_hash = compute_move_hash(move)
                    move_counter[move_hash] += 1
                    move_type_counter[move.type.value] += 1

                    # Track positions
                    if move.from_pos:
                        positions_touched.add((move.from_pos.x, move.from_pos.y))
                    if move.to:
                        positions_touched.add((move.to.x, move.to.y))
            else:
                move = opponent.select_move(state)

            if move is None:
                requirement = GameEngine.get_phase_requirement(state, current_player)
                if requirement is not None:
                    move = GameEngine.synthesize_bookkeeping_move(requirement, state)

            if move is None:
                break

            state = GameEngine.apply_move(state, move)
            move_count += 1

        all_gmo_moves.extend(game_moves)
        game_lengths.append(move_count)

        if state.winner == gmo_player:
            wins += 1

    # Compute metrics
    unique_moves = len(move_counter)
    total_moves = sum(move_counter.values())
    move_entropy = compute_entropy(list(move_counter.values()))
    position_coverage = len(positions_touched) / 64.0  # 8x8 board
    avg_game_length = np.mean(game_lengths) if game_lengths else 0
    win_rate = wins / num_games if num_games > 0 else 0

    metrics = DiversityMetrics(
        unique_moves=unique_moves,
        total_moves=total_moves,
        move_entropy=move_entropy,
        position_coverage=position_coverage,
        move_type_distribution=dict(move_type_counter),
        avg_game_length=avg_game_length,
        win_rate=win_rate,
    )

    return all_gmo_moves, metrics


def run_diversity_comparison(
    num_games: int = 30,
    checkpoint_path: str | None = None,
    output_dir: str = "data/diversity",
) -> dict:
    """Compare diversity metrics across different exploration settings."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Define configurations to test
    configs = [
        ("baseline", GMOConfig(beta=0.1, gamma=0.0)),  # Current best
        ("high_beta", GMOConfig(beta=0.5, gamma=0.0)),  # More uncertainty exploration
        ("with_novelty", GMOConfig(beta=0.1, gamma=0.2)),  # Add novelty bonus
        ("high_explore", GMOConfig(beta=0.5, gamma=0.3)),  # Maximum exploration
        ("low_optim", GMOConfig(beta=0.3, gamma=0.1, optim_steps=3)),  # Fewer optimization steps
    ]

    results = {}

    for config_name, gmo_config in configs:
        logger.info(f"\nTesting config: {config_name} (beta={gmo_config.beta}, gamma={gmo_config.gamma})")

        _moves, metrics = play_games_and_collect_data(
            gmo_config=gmo_config,
            num_games=num_games,
            checkpoint_path=checkpoint_path,
        )

        results[config_name] = {
            "config": {
                "beta": gmo_config.beta,
                "gamma": gmo_config.gamma,
                "optim_steps": gmo_config.optim_steps,
            },
            "metrics": {
                "unique_moves": metrics.unique_moves,
                "total_moves": metrics.total_moves,
                "diversity_ratio": metrics.unique_moves / max(metrics.total_moves, 1),
                "move_entropy": metrics.move_entropy,
                "position_coverage": metrics.position_coverage,
                "avg_game_length": metrics.avg_game_length,
                "win_rate": metrics.win_rate,
                "move_type_distribution": metrics.move_type_distribution,
            }
        }

        logger.info(f"  Unique moves: {metrics.unique_moves}/{metrics.total_moves} "
                    f"({100*metrics.unique_moves/max(metrics.total_moves,1):.1f}%)")
        logger.info(f"  Move entropy: {metrics.move_entropy:.2f} bits")
        logger.info(f"  Position coverage: {100*metrics.position_coverage:.1f}%")
        logger.info(f"  Win rate: {100*metrics.win_rate:.1f}%")

    # Summary comparison
    logger.info("\n" + "=" * 70)
    logger.info("DIVERSITY COMPARISON SUMMARY")
    logger.info("=" * 70)
    logger.info(f"{'Config':<15} {'Unique':<8} {'Entropy':<10} {'Coverage':<10} {'Win Rate':<10}")
    logger.info("-" * 70)

    for config_name, data in results.items():
        m = data["metrics"]
        logger.info(
            f"{config_name:<15} "
            f"{m['unique_moves']:<8} "
            f"{m['move_entropy']:<10.2f} "
            f"{100*m['position_coverage']:<10.1f}% "
            f"{100*m['win_rate']:<10.1f}%"
        )

    # Find best for diversity (entropy) while maintaining reasonable win rate
    best_diversity = max(
        [(name, data) for name, data in results.items() if data["metrics"]["win_rate"] >= 0.4],
        key=lambda x: x[1]["metrics"]["move_entropy"],
        default=(None, None)
    )

    if best_diversity[0]:
        logger.info(f"\nBest for diversity (with win rate >= 40%): {best_diversity[0]}")
        logger.info(f"  Config: beta={best_diversity[1]['config']['beta']}, "
                    f"gamma={best_diversity[1]['config']['gamma']}")

    # Save results
    results_file = output_path / f"diversity_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(results_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "num_games_per_config": num_games,
            "results": results,
        }, f, indent=2)

    logger.info(f"\nResults saved to {results_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Test exploration bonuses for data diversity")
    parser.add_argument("--games", type=int, default=30,
                        help="Games per configuration")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to GMO checkpoint")
    parser.add_argument("--output-dir", type=str, default="data/diversity",
                        help="Output directory")

    args = parser.parse_args()

    run_diversity_comparison(
        num_games=args.games,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
