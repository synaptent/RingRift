#!/usr/bin/env python3
"""Validate trained weights against current defaults and optionally promote them.

This script plays games between candidate weights and the current default weights
to determine if the candidate should be promoted. Weights are only promoted if
they demonstrate statistically significant improvement.

Usage:
    # Validate candidate weights against defaults
    python scripts/validate_and_promote_weights.py \
        --candidate logs/cmaes/phase2_square8_2p/runs/*/best_weights.json \
        --games 100 \
        --num-players 2

    # Validate and auto-promote if better (for CI/CD)
    python scripts/validate_and_promote_weights.py \
        --candidate logs/cmaes/merged/player_specific_profiles.json \
        --games 200 \
        --auto-promote \
        --min-win-rate 0.52 \
        --confidence 0.95

    # Validate multiple player counts
    python scripts/validate_and_promote_weights.py \
        --candidate-2p logs/cmaes/phase2_square8_2p/runs/*/best_weights.json \
        --candidate-3p logs/cmaes/phase2_square8_3p/runs/*/best_weights.json \
        --candidate-4p logs/cmaes/phase2_square8_4p/runs/*/best_weights.json \
        --games 50 \
        --auto-promote
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.ai.heuristic_weights import (
    BASE_V1_BALANCED_WEIGHTS,
    HeuristicWeights,
)
from app.models import BoardType
from app.training.significance import wilson_score_interval


@dataclass
class ValidationResult:
    """Result of validating candidate weights against defaults."""

    candidate_path: str
    num_players: int
    games_played: int
    candidate_wins: int
    draws: int
    default_wins: int
    win_rate: float
    is_significant: bool
    confidence: float
    recommendation: str

    def to_dict(self) -> dict:
        return {
            "candidate_path": self.candidate_path,
            "num_players": self.num_players,
            "games_played": self.games_played,
            "candidate_wins": self.candidate_wins,
            "draws": self.draws,
            "default_wins": self.default_wins,
            "win_rate": self.win_rate,
            "is_significant": self.is_significant,
            "confidence": self.confidence,
            "recommendation": self.recommendation,
        }


def load_candidate_weights(path: str) -> Tuple[HeuristicWeights, dict]:
    """Load candidate weights from a file."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Handle different file formats
    if "weights" in data:
        return data["weights"], data
    elif "profiles" in data:
        # Multi-profile file - return first profile
        profiles = data["profiles"]
        first_key = next(iter(profiles))
        return profiles[first_key], data
    else:
        # Assume entire file is weights
        return data, {"weights": data}


def play_validation_games(
    candidate_weights: HeuristicWeights,
    default_weights: HeuristicWeights,
    num_games: int,
    num_players: int,
    board_type: BoardType = BoardType.SQUARE8,
    eval_randomness: float = 0.15,
) -> Tuple[int, int, int]:
    """Play validation games between candidate and default weights.

    Returns:
        Tuple of (candidate_wins, draws, default_wins)
    """
    from app.ai.heuristic_ai import HeuristicAI
    from app.models import AIConfig, GameState
    from app.rules.factory import get_rules_engine

    rules = get_rules_engine()

    candidate_wins = 0
    draws = 0
    default_wins = 0

    # Create AI configs
    candidate_config = AIConfig(
        ai_type="heuristic",
        difficulty=5,
        eval_randomness=eval_randomness,
    )
    default_config = AIConfig(
        ai_type="heuristic",
        difficulty=5,
        eval_randomness=eval_randomness,
    )

    for game_idx in range(num_games):
        # Alternate who plays first
        candidate_seat = game_idx % num_players

        # Create initial state
        state = GameState.create_initial(board_type, num_players)

        # Create AIs for each seat
        ais: List[HeuristicAI] = []
        for seat in range(num_players):
            if seat == candidate_seat:
                ai = HeuristicAI(candidate_config, rules)
                ai.apply_weights(candidate_weights)
            else:
                ai = HeuristicAI(default_config, rules)
                ai.apply_weights(default_weights)
            ais.append(ai)

        # Play game
        move_count = 0
        max_moves = 200

        while not state.game_over and move_count < max_moves:
            current_player = state.current_player_index
            ai = ais[current_player]
            move = ai.select_move(state)
            if move is None:
                break
            state = rules.apply_move(state, move)
            move_count += 1

        # Determine winner
        if state.game_over and state.winner is not None:
            if state.winner == candidate_seat:
                candidate_wins += 1
            else:
                default_wins += 1
        else:
            draws += 1

        if (game_idx + 1) % 10 == 0:
            print(
                f"  Progress: {game_idx + 1}/{num_games} games | "
                f"Candidate: {candidate_wins}W/{draws}D/{default_wins}L"
            )

    return candidate_wins, draws, default_wins


def validate_weights(
    candidate_path: str,
    num_games: int,
    num_players: int,
    min_win_rate: float = 0.52,
    confidence: float = 0.95,
    board_type: BoardType = BoardType.SQUARE8,
) -> ValidationResult:
    """Validate candidate weights against current defaults.

    Args:
        candidate_path: Path to candidate weights file
        num_games: Number of games to play
        num_players: Number of players per game
        min_win_rate: Minimum win rate required for promotion
        confidence: Confidence level for statistical significance

    Returns:
        ValidationResult with recommendation
    """
    print(f"\nValidating {candidate_path}")
    print(f"  Players: {num_players}, Games: {num_games}")

    # Load candidate weights
    candidate_weights, _ = load_candidate_weights(candidate_path)

    # Get current default weights
    default_weights = BASE_V1_BALANCED_WEIGHTS

    # Play validation games
    candidate_wins, draws, default_wins = play_validation_games(
        candidate_weights,
        default_weights,
        num_games,
        num_players,
        board_type,
    )

    # Calculate statistics
    total_decisive = candidate_wins + default_wins
    win_rate = candidate_wins / total_decisive if total_decisive > 0 else 0.5

    # Calculate confidence interval
    if total_decisive <= 0:
        lower, upper = 0.0, 1.0
    else:
        lower, upper = wilson_score_interval(
            candidate_wins,
            total_decisive,
            confidence=confidence,
        )

    # Determine if statistically significant improvement
    is_significant = lower > 0.5 and win_rate >= min_win_rate

    # Generate recommendation
    if is_significant:
        recommendation = "PROMOTE"
    elif upper < 0.5:
        recommendation = "REJECT"
    else:
        recommendation = "INCONCLUSIVE"

    result = ValidationResult(
        candidate_path=candidate_path,
        num_players=num_players,
        games_played=num_games,
        candidate_wins=candidate_wins,
        draws=draws,
        default_wins=default_wins,
        win_rate=win_rate,
        is_significant=is_significant,
        confidence=confidence,
        recommendation=recommendation,
    )

    print(f"\n  Results: {candidate_wins}W / {draws}D / {default_wins}L")
    print(f"  Win rate: {win_rate:.2%} (CI: {lower:.2%} - {upper:.2%})")
    print(f"  Recommendation: {recommendation}")

    return result


def promote_weights(
    candidate_path: str,
    profile_id: str,
    output_path: Optional[str] = None,
) -> str:
    """Promote validated weights to a profile.

    Creates or updates a trained profiles JSON file that can be loaded
    via RINGRIFT_TRAINED_HEURISTIC_PROFILES environment variable.
    """
    candidate_weights, _ = load_candidate_weights(candidate_path)

    if output_path is None:
        output_path = "data/trained_heuristic_profiles.json"

    # Load existing profiles or create new
    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = {"profiles": {}, "metadata": {"updates": []}}

    # Update profile
    data["profiles"][profile_id] = candidate_weights
    data["metadata"]["updates"].append(
        {
            "profile_id": profile_id,
            "source": candidate_path,
            "timestamp": datetime.now().isoformat(),
        }
    )

    # Write output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nPromoted {profile_id} to {output_path}")
    return output_path


def expand_glob_patterns(patterns: Optional[List[str]]) -> List[str]:
    """Expand glob patterns to actual file paths."""
    if not patterns:
        return []
    paths = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        paths.extend(expanded)
    return sorted(set(paths))


def main():
    parser = argparse.ArgumentParser(
        description="Validate trained weights against defaults",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--candidate",
        "-c",
        action="append",
        dest="candidates",
        help="Candidate weight file(s) to validate",
    )
    parser.add_argument(
        "--candidate-2p",
        action="append",
        dest="candidates_2p",
        help="Candidate weight file(s) for 2-player validation",
    )
    parser.add_argument(
        "--candidate-3p",
        action="append",
        dest="candidates_3p",
        help="Candidate weight file(s) for 3-player validation",
    )
    parser.add_argument(
        "--candidate-4p",
        action="append",
        dest="candidates_4p",
        help="Candidate weight file(s) for 4-player validation",
    )
    parser.add_argument(
        "--games",
        "-g",
        type=int,
        default=50,
        help="Number of games per validation (default: 50)",
    )
    parser.add_argument(
        "--num-players",
        "-n",
        type=int,
        default=2,
        choices=[2, 3, 4],
        help="Number of players for general --candidate validation",
    )
    parser.add_argument(
        "--min-win-rate",
        type=float,
        default=0.52,
        help="Minimum win rate required for promotion (default: 0.52)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="Confidence level for significance testing (default: 0.95)",
    )
    parser.add_argument(
        "--auto-promote",
        action="store_true",
        help="Automatically promote weights that pass validation",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/trained_heuristic_profiles.json",
        help="Output path for promoted profiles",
    )
    parser.add_argument(
        "--board",
        choices=["square8", "square19", "hex"],
        default="square8",
        help="Board type for validation games",
    )
    parser.add_argument(
        "--report",
        "-r",
        help="Path to save validation report JSON",
    )

    args = parser.parse_args()

    # Map board string to BoardType
    board_map = {
        "square8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "hex": BoardType.HEXAGONAL,
    }
    board_type = board_map[args.board]

    results: List[ValidationResult] = []

    # Validate general candidates
    if args.candidates:
        files = expand_glob_patterns(args.candidates)
        for path in files:
            result = validate_weights(
                path,
                args.games,
                args.num_players,
                args.min_win_rate,
                args.confidence,
                board_type,
            )
            results.append(result)

            if args.auto_promote and result.recommendation == "PROMOTE":
                profile_id = f"heuristic_v1_{args.num_players}p"
                promote_weights(path, profile_id, args.output)

    # Validate player-specific candidates
    for num_players, candidates in [
        (2, args.candidates_2p),
        (3, args.candidates_3p),
        (4, args.candidates_4p),
    ]:
        if candidates:
            files = expand_glob_patterns(candidates)
            for path in files:
                result = validate_weights(
                    path,
                    args.games,
                    num_players,
                    args.min_win_rate,
                    args.confidence,
                    board_type,
                )
                results.append(result)

                if args.auto_promote and result.recommendation == "PROMOTE":
                    profile_id = f"heuristic_v1_{num_players}p"
                    promote_weights(path, profile_id, args.output)

    # Print summary
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)

    for result in results:
        status = "PASS" if result.recommendation == "PROMOTE" else result.recommendation
        print(
            f"  {result.num_players}p | {result.win_rate:.1%} win rate | "
            f"{status} | {os.path.basename(result.candidate_path)}"
        )

    # Save report if requested
    if args.report:
        report = {
            "timestamp": datetime.now().isoformat(),
            "parameters": {
                "games": args.games,
                "min_win_rate": args.min_win_rate,
                "confidence": args.confidence,
                "board": args.board,
            },
            "results": [r.to_dict() for r in results],
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.report)), exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {args.report}")

    # Exit with appropriate code for CI/CD
    all_passed = all(r.recommendation != "REJECT" for r in results)
    any_promoted = any(r.recommendation == "PROMOTE" for r in results)

    if not results:
        print("\nNo candidates validated")
        sys.exit(1)
    elif all_passed and any_promoted:
        print("\nValidation PASSED - weights promoted")
        sys.exit(0)
    elif all_passed:
        print("\nValidation INCONCLUSIVE - no weights promoted")
        sys.exit(0)
    else:
        print("\nValidation FAILED - some weights rejected")
        sys.exit(1)


if __name__ == "__main__":
    main()
