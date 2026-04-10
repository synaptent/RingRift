#!/usr/bin/env python3
"""Validate trained CMA-ES weights against all baseline personas.

This script runs a comprehensive round-robin tournament to verify that
trained weights generalize well across all opponent types.

Usage:
    # Validate from JSON file
    python scripts/validate_trained_weights.py --weights path/to/weights.json

    # Validate from training output directory
    python scripts/validate_trained_weights.py --training-dir logs/cmaes/robust/xxx/square8_2p

    # Quick validation (fewer games)
    python scripts/validate_trained_weights.py --weights weights.json --quick

    # Full validation (more games, higher confidence)
    python scripts/validate_trained_weights.py --weights weights.json --full
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.ai.gpu_parallel_games import evaluate_candidate_fitness_gpu
from app.ai.heuristic_weights import (
    HEURISTIC_V1_BALANCED,
    HEURISTIC_V1_AGGRESSIVE,
    HEURISTIC_V1_TERRITORIAL,
    HEURISTIC_V1_DEFENSIVE,
)
from app.ai.multi_opponent_fitness import (
    evaluate_multi_opponent,
    compute_aggregate_fitness,
    BASELINE_OPPONENTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of weight validation."""

    weights: dict[str, float]
    per_opponent: dict[str, float]
    aggregate: float
    self_play: float
    games_per_opponent: int
    total_games: int

    # Success criteria
    min_acceptable: float = 0.40  # No catastrophic losses
    target_aggregate: float = 0.55  # Better than random

    @property
    def passed(self) -> bool:
        """Check if weights pass validation criteria."""
        min_rate = min(self.per_opponent.values())
        return min_rate >= self.min_acceptable and self.aggregate >= self.target_aggregate

    @property
    def worst_opponent(self) -> tuple[str, float]:
        """Get the opponent with lowest win rate."""
        worst = min(self.per_opponent.items(), key=lambda x: x[1])
        return worst

    @property
    def best_opponent(self) -> tuple[str, float]:
        """Get the opponent with highest win rate."""
        best = max(self.per_opponent.items(), key=lambda x: x[1])
        return best

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "weights": self.weights,
            "per_opponent": self.per_opponent,
            "aggregate": self.aggregate,
            "self_play": self.self_play,
            "games_per_opponent": self.games_per_opponent,
            "total_games": self.total_games,
            "passed": self.passed,
            "worst_opponent": self.worst_opponent[0],
            "worst_rate": self.worst_opponent[1],
            "best_opponent": self.best_opponent[0],
            "best_rate": self.best_opponent[1],
        }

    def print_report(self) -> None:
        """Print validation report."""
        print("\n" + "=" * 60)
        print("VALIDATION REPORT")
        print("=" * 60)

        print(f"\nGames per opponent: {self.games_per_opponent}")
        print(f"Total games: {self.total_games}")

        print("\nPer-opponent win rates:")
        for name, rate in sorted(self.per_opponent.items(), key=lambda x: -x[1]):
            status = "OK" if rate >= self.min_acceptable else "FAIL"
            print(f"  {name:12s}: {rate:6.1%}  [{status}]")

        print(f"\nSelf-play: {self.self_play:.1%} (should be ~50%)")
        print(f"Aggregate fitness: {self.aggregate:.3f}")
        print(f"  (0.4 * min + 0.6 * mean = 0.4 * {min(self.per_opponent.values()):.3f} + 0.6 * {np.mean(list(self.per_opponent.values())):.3f})")

        print(f"\nBest against: {self.best_opponent[0]} ({self.best_opponent[1]:.1%})")
        print(f"Worst against: {self.worst_opponent[0]} ({self.worst_opponent[1]:.1%})")

        print("\n" + "-" * 60)
        if self.passed:
            print("VALIDATION PASSED")
            print(f"  - Min win rate ({min(self.per_opponent.values()):.1%}) >= {self.min_acceptable:.0%}")
            print(f"  - Aggregate ({self.aggregate:.3f}) >= {self.target_aggregate:.0%}")
        else:
            print("VALIDATION FAILED")
            if min(self.per_opponent.values()) < self.min_acceptable:
                print(f"  - Min win rate ({min(self.per_opponent.values()):.1%}) < {self.min_acceptable:.0%}")
            if self.aggregate < self.target_aggregate:
                print(f"  - Aggregate ({self.aggregate:.3f}) < {self.target_aggregate:.0%}")
        print("=" * 60 + "\n")


def load_weights_from_file(path: Path) -> dict[str, float]:
    """Load weights from JSON file."""
    with open(path) as f:
        data = json.load(f)

    # Handle different formats
    if "weights" in data:
        return data["weights"]
    elif "best_weights" in data:
        return data["best_weights"]
    else:
        # Assume it's just the weights dict
        return data


def load_weights_from_training_dir(path: Path) -> dict[str, float]:
    """Load best weights from training output directory."""
    # Look for common output files
    candidates = [
        path / "best_weights.json",
        path / "final_weights.json",
        path / "checkpoint_best.json",
    ]

    for candidate in candidates:
        if candidate.exists():
            return load_weights_from_file(candidate)

    # Try to find any JSON file with weights
    for json_file in path.glob("*.json"):
        try:
            weights = load_weights_from_file(json_file)
            if "material_weight" in weights:
                logger.info(f"Loaded weights from {json_file}")
                return weights
        except (json.JSONDecodeError, KeyError):
            continue

    raise FileNotFoundError(f"No weights file found in {path}")


def validate_weights(
    weights: dict[str, float],
    games_per_opponent: int = 64,
    self_play_games: int = 32,
    board_size: int = 8,
    num_players: int = 2,
    max_moves: int = 200,
    use_gpu: bool = True,
) -> ValidationResult:
    """Run comprehensive validation of weights.

    Args:
        weights: Heuristic weights to validate
        games_per_opponent: Games per opponent (higher = more confidence)
        self_play_games: Self-play games
        board_size: Board size
        num_players: Number of players
        max_moves: Max moves per game
        use_gpu: Use GPU acceleration

    Returns:
        ValidationResult with detailed metrics
    """
    logger.info(f"Validating weights against {len(BASELINE_OPPONENTS)} opponents")
    logger.info(f"  Games per opponent: {games_per_opponent}")
    logger.info(f"  Self-play games: {self_play_games}")

    # Use multi-opponent evaluation
    result = evaluate_multi_opponent(
        candidate_weights=weights,
        games_per_opponent=games_per_opponent,
        self_play_games=self_play_games,
        board_size=board_size,
        num_players=num_players,
        max_moves=max_moves,
    )

    total_games = games_per_opponent * len(BASELINE_OPPONENTS) + self_play_games

    return ValidationResult(
        weights=weights,
        per_opponent=result.per_opponent,
        aggregate=result.aggregate,
        self_play=result.self_play,
        games_per_opponent=games_per_opponent,
        total_games=total_games,
    )


def validate_against_baselines_comparison(
    weights: dict[str, float],
    games: int = 32,
) -> dict[str, dict[str, float]]:
    """Compare trained weights against each baseline playing each other.

    This provides context: how do baselines perform against each other?
    """
    logger.info("Running baseline comparison for context...")

    results = {}
    personas = list(BASELINE_OPPONENTS.keys())

    # Trained weights vs each baseline
    results["trained"] = {}
    for name, opponent in BASELINE_OPPONENTS.items():
        win_rate = evaluate_candidate_fitness_gpu(
            candidate_weights=weights,
            opponent_weights=opponent,
            num_games=games,
        )
        results["trained"][name] = win_rate

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Validate trained CMA-ES weights",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input options (mutually exclusive)
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--weights", type=Path,
        help="Path to weights JSON file"
    )
    input_group.add_argument(
        "--training-dir", type=Path,
        help="Path to training output directory"
    )

    # Validation mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--quick", action="store_true",
        help="Quick validation (16 games/opponent)"
    )
    mode_group.add_argument(
        "--full", action="store_true",
        help="Full validation (128 games/opponent)"
    )

    # Game parameters
    parser.add_argument("--board-size", type=int, default=8)
    parser.add_argument("--num-players", type=int, default=2)
    parser.add_argument("--games-per-opponent", type=int, default=64)
    parser.add_argument("--output", type=Path, help="Save results to JSON")

    args = parser.parse_args()

    # Load weights
    if args.weights:
        weights = load_weights_from_file(args.weights)
        logger.info(f"Loaded weights from {args.weights}")
    else:
        weights = load_weights_from_training_dir(args.training_dir)
        logger.info(f"Loaded weights from {args.training_dir}")

    # Determine games per opponent
    if args.quick:
        games_per_opponent = 16
        self_play_games = 8
    elif args.full:
        games_per_opponent = 128
        self_play_games = 64
    else:
        games_per_opponent = args.games_per_opponent
        self_play_games = games_per_opponent // 2

    # Run validation
    result = validate_weights(
        weights=weights,
        games_per_opponent=games_per_opponent,
        self_play_games=self_play_games,
        board_size=args.board_size,
        num_players=args.num_players,
    )

    # Print report
    result.print_report()

    # Save results if requested
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        logger.info(f"Results saved to {args.output}")

    # Exit with appropriate code
    sys.exit(0 if result.passed else 1)


if __name__ == "__main__":
    main()
