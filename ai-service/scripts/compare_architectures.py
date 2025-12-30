#!/usr/bin/env python3
"""Architecture Comparison Script for RingRift Neural Networks.

Runs head-to-head comparisons between different architecture versions (v3, v4, v5, v5_heavy, v6)
to determine relative playing strength. Results are stored in architecture_tracker for
informing training compute allocation.

Usage:
    # Run full comparison on hex8_2p
    python scripts/compare_architectures.py run --board-type hex8 --num-players 2

    # Run comparison with specific models
    python scripts/compare_architectures.py run \
        --board-type hex8 --num-players 2 \
        --model v3=models/hex8_2p_trained_v3.pth \
        --model v5=models/canonical_hex8_2p.pth

    # Quick test run (fewer games)
    python scripts/compare_architectures.py run --board-type hex8 --num-players 2 --quick

    # Generate report from existing results
    python scripts/compare_architectures.py report --input data/architecture_comparison.json
"""

from __future__ import annotations

import argparse
import itertools
import json
import logging
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Default model paths for each architecture (hex8_2p)
DEFAULT_MODELS_HEX8_2P: dict[str, str] = {
    "v3": "models/hex8_2p_trained_v3.pth",
    "v4": "models/canonical_hex8_2p.pth",  # v4 checkpoints may need verification
    "v5": "models/canonical_hex8_2p.pth",
    "v5_heavy": "models/canonical_hex8_2p_v5heavy.pth",
    "v6": "models/hex8_2p_v6_test.pth",
}

DEFAULT_MODELS_SQUARE8_2P: dict[str, str] = {
    "v3": "models/square8_2p_trained_v3.pth",
    "v4": "models/canonical_square8_2p.pth",
    "v5": "models/canonical_square8_2p.pth",
    "v5_heavy": "models/canonical_square8_2p.pth",  # May need dedicated v5_heavy
    "v6": "models/canonical_square8_2p.pth",  # May need dedicated v6
}


@dataclass
class ComparisonConfig:
    """Configuration for architecture comparison."""

    architectures: list[str] = field(default_factory=lambda: ["v3", "v4", "v5", "v5_heavy", "v6"])
    board_type: str = "hex8"
    num_players: int = 2
    games_per_matchup: int = 100
    model_paths: dict[str, str] = field(default_factory=dict)
    output_path: str = "data/architecture_comparison.json"
    seed: int = 42
    max_workers: int = 4
    harness: str = "policy_only"  # gumbel_mcts, mcts, policy_only


@dataclass
class MatchupResult:
    """Result of a single architecture matchup."""

    arch_a: str
    arch_b: str
    wins_a: int
    wins_b: int
    draws: int
    games_played: int
    elo_diff: float = 0.0  # Estimated Elo difference (arch_a - arch_b)

    @property
    def win_rate_a(self) -> float:
        """Win rate of architecture A."""
        if self.games_played == 0:
            return 0.5
        return (self.wins_a + 0.5 * self.draws) / self.games_played

    @property
    def win_rate_b(self) -> float:
        """Win rate of architecture B."""
        return 1.0 - self.win_rate_a


@dataclass
class ComparisonReport:
    """Full comparison report with all matchup results."""

    config: ComparisonConfig
    matchups: list[MatchupResult] = field(default_factory=list)
    elo_ratings: dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "config": {
                "architectures": self.config.architectures,
                "board_type": self.config.board_type,
                "num_players": self.config.num_players,
                "games_per_matchup": self.config.games_per_matchup,
                "harness": self.config.harness,
            },
            "matchups": [
                {
                    "arch_a": m.arch_a,
                    "arch_b": m.arch_b,
                    "wins_a": m.wins_a,
                    "wins_b": m.wins_b,
                    "draws": m.draws,
                    "games_played": m.games_played,
                    "elo_diff": m.elo_diff,
                    "win_rate_a": m.win_rate_a,
                }
                for m in self.matchups
            ],
            "elo_ratings": self.elo_ratings,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ComparisonReport":
        """Load from dictionary."""
        config = ComparisonConfig(
            architectures=data["config"]["architectures"],
            board_type=data["config"]["board_type"],
            num_players=data["config"]["num_players"],
            games_per_matchup=data["config"]["games_per_matchup"],
            harness=data["config"].get("harness", "policy_only"),
        )
        matchups = [
            MatchupResult(
                arch_a=m["arch_a"],
                arch_b=m["arch_b"],
                wins_a=m["wins_a"],
                wins_b=m["wins_b"],
                draws=m["draws"],
                games_played=m["games_played"],
                elo_diff=m.get("elo_diff", 0.0),
            )
            for m in data["matchups"]
        ]
        return cls(
            config=config,
            matchups=matchups,
            elo_ratings=data.get("elo_ratings", {}),
            timestamp=data.get("timestamp", time.time()),
        )


def load_ai_for_architecture(
    arch: str,
    model_path: str,
    board_type: str,
    num_players: int,
    player_number: int = 1,
) -> Any:
    """Load an AI instance for the given architecture and model.

    Args:
        arch: Architecture version (v3, v4, v5, v5_heavy, v6)
        model_path: Path to model checkpoint
        board_type: Board type (hex8, square8, etc.)
        num_players: Number of players
        player_number: Which player this AI controls

    Returns:
        AI instance ready for game play
    """
    from app.ai.policy_only_ai import PolicyOnlyAI
    from app.models import AIConfig, BoardType

    # Convert board_type string to enum
    board_enum = BoardType(board_type) if isinstance(board_type, str) else board_type

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    logger.debug(f"Loading {arch} model from {model_path}")

    # Extract model ID from path
    model_id = str(path)
    if model_id.startswith("models/"):
        model_id = model_id[7:]
    if model_id.endswith(".pth"):
        model_id = model_id[:-4]

    # Create AIConfig for PolicyOnlyAI
    # difficulty=5 is mid-level, but PolicyOnlyAI ignores it
    config = AIConfig(
        difficulty=5,
        nn_model_id=model_id,
    )

    # Use PolicyOnlyAI for direct neural network evaluation
    return PolicyOnlyAI(player_number, config, board_type=board_enum)


def play_single_matchup_game(
    ai_a: Any,
    ai_b: Any,
    board_type: str,
    num_players: int,
    ai_a_player: int,
    seed: int | None = None,
) -> int:
    """Play a single game between two AIs using the game gauntlet infrastructure.

    Args:
        ai_a: First AI instance
        ai_b: Second AI instance
        board_type: Board type
        num_players: Number of players
        ai_a_player: Which player number AI A controls (1 or 2)
        seed: Random seed for reproducibility

    Returns:
        1 if ai_a wins, -1 if ai_b wins, 0 for draw
    """
    from app.models import BoardType
    from app.training.game_gauntlet import play_single_game

    board_enum = BoardType(board_type) if isinstance(board_type, str) else board_type

    # Set up opponent AIs dict for the game
    ai_b_player = 2 if ai_a_player == 1 else 1
    opponent_ais = {ai_b_player: ai_b}

    # Play the game using the existing infrastructure
    result = play_single_game(
        candidate_ai=ai_a,
        opponent_ai=ai_b,  # Fallback, but opponent_ais takes precedence
        board_type=board_enum,
        num_players=num_players,
        candidate_player=ai_a_player,
        max_moves=500,
        seed=seed,
        opponent_ais=opponent_ais,
    )

    # Convert GameResult to win/loss/draw
    if result.winner == ai_a_player:
        return 1
    elif result.winner == ai_b_player:
        return -1
    else:
        return 0  # Draw


def run_matchup(
    arch_a: str,
    arch_b: str,
    model_path_a: str,
    model_path_b: str,
    config: ComparisonConfig,
) -> MatchupResult:
    """Run a full matchup between two architectures.

    Args:
        arch_a: First architecture name
        arch_b: Second architecture name
        model_path_a: Path to first model
        model_path_b: Path to second model
        config: Comparison configuration

    Returns:
        MatchupResult with game outcomes
    """
    logger.info(f"Starting matchup: {arch_a} vs {arch_b} ({config.games_per_matchup} games)")

    wins_a = 0
    wins_b = 0
    draws = 0

    # Load AIs once
    ai_a = load_ai_for_architecture(
        arch_a, model_path_a, config.board_type, config.num_players, player_number=1
    )
    ai_b = load_ai_for_architecture(
        arch_b, model_path_b, config.board_type, config.num_players, player_number=2
    )

    for game_idx in range(config.games_per_matchup):
        # Alternate starting player for fairness
        ai_a_player = 1 if game_idx % 2 == 0 else 2

        # Fixed seed for reproducibility
        game_seed = config.seed * 1000 + game_idx

        # Update AI player numbers
        ai_a.player_number = ai_a_player
        ai_b.player_number = 2 if ai_a_player == 1 else 1

        result = play_single_matchup_game(
            ai_a, ai_b, config.board_type, config.num_players, ai_a_player, game_seed
        )

        if result == 1:
            wins_a += 1
        elif result == -1:
            wins_b += 1
        else:
            draws += 1

        if (game_idx + 1) % 10 == 0:
            logger.info(
                f"  {arch_a} vs {arch_b}: {game_idx + 1}/{config.games_per_matchup} "
                f"({wins_a}-{wins_b}-{draws})"
            )

    # Calculate Elo difference from win rate
    win_rate = (wins_a + 0.5 * draws) / config.games_per_matchup
    win_rate = max(0.01, min(0.99, win_rate))  # Clamp to avoid division by zero
    elo_diff = 400 * math.log10(win_rate / (1 - win_rate))

    result = MatchupResult(
        arch_a=arch_a,
        arch_b=arch_b,
        wins_a=wins_a,
        wins_b=wins_b,
        draws=draws,
        games_played=config.games_per_matchup,
        elo_diff=elo_diff,
    )

    logger.info(
        f"Matchup complete: {arch_a} vs {arch_b} = {wins_a}-{wins_b}-{draws} "
        f"(Elo diff: {elo_diff:+.0f})"
    )

    return result


def calculate_elo_ratings(
    matchups: list[MatchupResult],
    anchor_arch: str = "v5",
    anchor_elo: float = 1500.0,
) -> dict[str, float]:
    """Calculate Elo ratings from matchup results using Bradley-Terry model.

    Uses iterative MLE fitting to find ratings that best explain observed win rates.

    Args:
        matchups: List of matchup results
        anchor_arch: Architecture to anchor at known Elo
        anchor_elo: Elo rating for anchor architecture

    Returns:
        Dictionary mapping architecture to Elo rating
    """
    # Collect all architectures
    architectures = set()
    for m in matchups:
        architectures.add(m.arch_a)
        architectures.add(m.arch_b)

    architectures = sorted(architectures)
    n_arch = len(architectures)
    arch_to_idx = {a: i for i, a in enumerate(architectures)}

    # Initialize ratings
    ratings = {a: anchor_elo for a in architectures}

    # Iterative update (simplified Bradley-Terry)
    for _ in range(100):  # Max iterations
        new_ratings = ratings.copy()

        for arch in architectures:
            if arch == anchor_arch:
                continue

            # Calculate expected vs observed wins
            total_expected = 0.0
            total_observed = 0.0
            total_games = 0

            for m in matchups:
                if m.arch_a == arch:
                    opponent = m.arch_b
                    observed = m.wins_a + 0.5 * m.draws
                    games = m.games_played
                elif m.arch_b == arch:
                    opponent = m.arch_a
                    observed = m.wins_b + 0.5 * m.draws
                    games = m.games_played
                else:
                    continue

                # Expected win rate based on current ratings
                diff = ratings[arch] - ratings[opponent]
                expected_rate = 1.0 / (1.0 + 10 ** (-diff / 400))

                total_expected += expected_rate * games
                total_observed += observed
                total_games += games

            if total_games > 0 and total_expected > 0:
                # Adjust rating based on performance
                adjustment = 32 * (total_observed - total_expected) / total_games
                new_ratings[arch] = ratings[arch] + adjustment

        # Check convergence
        max_change = max(abs(new_ratings[a] - ratings[a]) for a in architectures)
        ratings = new_ratings

        if max_change < 0.1:
            break

    # Re-anchor to ensure anchor_arch has correct rating
    if anchor_arch in ratings:
        offset = anchor_elo - ratings[anchor_arch]
        ratings = {a: r + offset for a, r in ratings.items()}

    return ratings


def run_comparison(config: ComparisonConfig) -> ComparisonReport:
    """Run full architecture comparison.

    Args:
        config: Comparison configuration

    Returns:
        ComparisonReport with all results
    """
    logger.info(f"Starting architecture comparison")
    logger.info(f"  Board: {config.board_type}, Players: {config.num_players}")
    logger.info(f"  Architectures: {config.architectures}")
    logger.info(f"  Games per matchup: {config.games_per_matchup}")

    # Verify all models exist
    missing_models = []
    for arch in config.architectures:
        if arch not in config.model_paths:
            missing_models.append(arch)
        elif not Path(config.model_paths[arch]).exists():
            missing_models.append(f"{arch} ({config.model_paths[arch]})")

    if missing_models:
        logger.error(f"Missing models: {missing_models}")
        raise FileNotFoundError(f"Models not found: {missing_models}")

    # Run all pairwise matchups
    matchups: list[MatchupResult] = []
    total_matchups = len(list(itertools.combinations(config.architectures, 2)))

    for idx, (arch_a, arch_b) in enumerate(itertools.combinations(config.architectures, 2)):
        logger.info(f"\n=== Matchup {idx + 1}/{total_matchups}: {arch_a} vs {arch_b} ===")

        result = run_matchup(
            arch_a, arch_b,
            config.model_paths[arch_a],
            config.model_paths[arch_b],
            config,
        )
        matchups.append(result)

    # Calculate overall Elo ratings
    elo_ratings = calculate_elo_ratings(matchups)

    report = ComparisonReport(
        config=config,
        matchups=matchups,
        elo_ratings=elo_ratings,
    )

    return report


def print_report(report: ComparisonReport) -> None:
    """Print a formatted comparison report."""
    print("\n" + "=" * 60)
    print(f"Architecture Comparison Report")
    print(f"Board: {report.config.board_type}, Players: {report.config.num_players}")
    print(f"Games per matchup: {report.config.games_per_matchup}")
    print("=" * 60)

    # Sort by Elo rating
    sorted_archs = sorted(
        report.elo_ratings.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    print("\n## Elo Rankings\n")
    print(f"{'Rank':<6}{'Architecture':<15}{'Elo':<10}{'95% CI':<20}")
    print("-" * 51)

    for rank, (arch, elo) in enumerate(sorted_archs, 1):
        # Calculate confidence interval from matchup data
        total_games = sum(
            m.games_played for m in report.matchups
            if m.arch_a == arch or m.arch_b == arch
        )
        ci_width = 200 / math.sqrt(max(1, total_games / 10))  # Approximate
        print(f"{rank:<6}{arch:<15}{elo:<10.0f}({elo - ci_width:.0f}-{elo + ci_width:.0f})")

    print("\n## Head-to-Head Results\n")
    print(f"{'Matchup':<25}{'Result':<15}{'Win %':<10}{'Elo Diff':<12}")
    print("-" * 62)

    for m in report.matchups:
        matchup_str = f"{m.arch_a} vs {m.arch_b}"
        result_str = f"{m.wins_a}-{m.wins_b}-{m.draws}"
        win_pct = f"{m.win_rate_a * 100:.1f}%"
        elo_str = f"{m.elo_diff:+.0f}"
        print(f"{matchup_str:<25}{result_str:<15}{win_pct:<10}{elo_str:<12}")

    print("\n" + "=" * 60)


def save_report(report: ComparisonReport, path: str) -> None:
    """Save report to JSON file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    logger.info(f"Report saved to {path}")


def update_architecture_tracker(report: ComparisonReport) -> None:
    """Update architecture tracker with comparison results."""
    try:
        from app.training.architecture_tracker import record_evaluation

        for arch, elo in report.elo_ratings.items():
            # Find total games for this architecture
            total_games = sum(
                m.games_played for m in report.matchups
                if m.arch_a == arch or m.arch_b == arch
            )

            record_evaluation(
                architecture=arch,
                board_type=report.config.board_type,
                num_players=report.config.num_players,
                elo=elo,
                training_hours=0.0,  # Comparison, not training
                games_evaluated=total_games,
            )

        logger.info("Architecture tracker updated with comparison results")
    except ImportError as e:
        logger.warning(f"Could not update architecture tracker: {e}")


def emit_comparison_completed_event(report: ComparisonReport) -> None:
    """Emit ARCHITECTURE_COMPARISON_COMPLETED event for integration with training pipeline.

    This event triggers the ArchitectureFeedbackController to update
    allocation weights based on the comparison results.
    """
    try:
        from app.coordination.event_router import get_router

        router = get_router()

        # Emit event with full comparison data
        event_data = {
            "elo_ratings": report.elo_ratings,
            "matchups": [
                {
                    "arch_a": m.arch_a,
                    "arch_b": m.arch_b,
                    "wins_a": m.wins_a,
                    "wins_b": m.wins_b,
                    "draws": m.draws,
                    "games_played": m.games_played,
                    "elo_diff": m.elo_diff,
                    "win_rate_a": m.win_rate_a,
                }
                for m in report.matchups
            ],
            "config": {
                "board_type": report.config.board_type,
                "num_players": report.config.num_players,
                "games_per_matchup": report.config.games_per_matchup,
                "harness": report.config.harness,
            },
            "timestamp": report.timestamp,
        }

        router.emit("ARCHITECTURE_COMPARISON_COMPLETED", event_data)
        logger.info(
            f"Emitted ARCHITECTURE_COMPARISON_COMPLETED event "
            f"({len(report.elo_ratings)} architectures)"
        )

    except ImportError as e:
        logger.warning(f"Could not emit comparison event (event_router not available): {e}")
    except Exception as e:
        logger.warning(f"Error emitting comparison event: {e}")


def get_default_models(board_type: str, num_players: int) -> dict[str, str]:
    """Get default model paths for a configuration."""
    key = f"{board_type}_{num_players}p"

    if key == "hex8_2p":
        return DEFAULT_MODELS_HEX8_2P.copy()
    elif key == "square8_2p":
        return DEFAULT_MODELS_SQUARE8_2P.copy()
    else:
        # Try to find canonical models
        models = {}
        for arch in ["v3", "v4", "v5", "v5_heavy", "v6"]:
            canonical = f"models/canonical_{board_type}_{num_players}p.pth"
            if Path(canonical).exists():
                models[arch] = canonical
        return models


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Compare neural network architectures")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run architecture comparison")
    run_parser.add_argument("--board-type", default="hex8", help="Board type")
    run_parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    run_parser.add_argument("--games-per-matchup", type=int, default=100, help="Games per matchup")
    run_parser.add_argument("--architectures", nargs="+", default=["v3", "v4", "v5", "v5_heavy", "v6"],
                          help="Architectures to compare")
    run_parser.add_argument("--model", action="append", default=[], metavar="ARCH=PATH",
                          help="Specify model path for architecture (e.g., v3=models/v3.pth)")
    run_parser.add_argument("--output", default="data/architecture_comparison.json",
                          help="Output path for results")
    run_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    run_parser.add_argument("--quick", action="store_true", help="Quick test (10 games)")
    run_parser.add_argument("--update-tracker", action="store_true",
                          help="Update architecture tracker with results")
    run_parser.add_argument("--emit-event", action="store_true",
                          help="Emit ARCHITECTURE_COMPARISON_COMPLETED event")

    # Report command
    report_parser = subparsers.add_parser("report", help="Generate report from results")
    report_parser.add_argument("--input", required=True, help="Input JSON file")
    report_parser.add_argument("--format", choices=["text", "json", "markdown"], default="text")

    args = parser.parse_args()

    if args.command == "run":
        # Build config
        config = ComparisonConfig(
            architectures=args.architectures,
            board_type=args.board_type,
            num_players=args.num_players,
            games_per_matchup=10 if args.quick else args.games_per_matchup,
            output_path=args.output,
            seed=args.seed,
        )

        # Get default models
        config.model_paths = get_default_models(args.board_type, args.num_players)

        # Override with command-line models
        for model_spec in args.model:
            if "=" in model_spec:
                arch, path = model_spec.split("=", 1)
                config.model_paths[arch] = path

        # Filter architectures to those with models
        config.architectures = [a for a in config.architectures if a in config.model_paths]

        if len(config.architectures) < 2:
            logger.error("Need at least 2 architectures with models for comparison")
            logger.info(f"Available: {list(config.model_paths.keys())}")
            sys.exit(1)

        # Run comparison
        report = run_comparison(config)

        # Print and save results
        print_report(report)
        save_report(report, config.output_path)

        # Update tracker if requested
        if args.update_tracker:
            update_architecture_tracker(report)

        # Emit event if requested
        if args.emit_event:
            emit_comparison_completed_event(report)

    elif args.command == "report":
        # Load and display existing results
        with open(args.input) as f:
            data = json.load(f)

        report = ComparisonReport.from_dict(data)

        if args.format == "json":
            print(json.dumps(report.to_dict(), indent=2))
        else:
            print_report(report)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
