#!/usr/bin/env python3
"""Elo-based model promotion gate with confidence intervals.

Uses Wilson score confidence intervals to ensure promotions are statistically
significant. A model is only promoted if its lower CI bound exceeds the
promotion threshold.

Usage:
    python scripts/elo_promotion_gate.py --candidate model_v5 --baseline model_v4
    python scripts/elo_promotion_gate.py --candidate model_v5 --threshold 0.55 --confidence 0.95
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.tournament.elo import EloRating, EloCalculator
from app.training.significance import wilson_score_interval, wilson_lower_bound

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def evaluate_promotion(
    candidate_wins: int,
    candidate_losses: int,
    candidate_draws: int = 0,
    threshold: float = 0.55,
    confidence: float = 0.95,
) -> dict:
    """Evaluate whether a candidate model should be promoted.

    Args:
        candidate_wins: Games won by candidate
        candidate_losses: Games lost by candidate
        candidate_draws: Games drawn
        threshold: Win rate threshold for promotion
        confidence: Confidence level for CI

    Returns:
        Dict with promotion decision and statistics
    """
    total_games = candidate_wins + candidate_losses + candidate_draws
    decisive_games = candidate_wins + candidate_losses

    if total_games == 0:
        return {
            "promote": False,
            "reason": "no_games_played",
            "total_games": 0,
        }

    # Calculate win rate (excluding draws for decisive comparison)
    if decisive_games > 0:
        win_rate = candidate_wins / decisive_games
        ci_lower, ci_upper = wilson_score_interval(candidate_wins, decisive_games, confidence)
    else:
        # All draws - use overall rate
        win_rate = 0.5
        ci_lower, ci_upper = 0.5, 0.5

    # Promotion requires CI lower bound >= threshold
    promote = ci_lower >= threshold

    # Calculate Elo difference estimate
    if win_rate > 0 and win_rate < 1:
        # Elo difference = 400 * log10(win_rate / (1 - win_rate))
        import math
        elo_diff = 400 * math.log10(win_rate / (1 - win_rate))
    elif win_rate >= 1:
        elo_diff = 400  # Cap at +400
    else:
        elo_diff = -400  # Cap at -400

    return {
        "promote": promote,
        "win_rate": round(win_rate, 4),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "threshold": threshold,
        "confidence": confidence,
        "total_games": total_games,
        "decisive_games": decisive_games,
        "wins": candidate_wins,
        "losses": candidate_losses,
        "draws": candidate_draws,
        "estimated_elo_diff": round(elo_diff, 1),
        "reason": "ci_lower_exceeds_threshold" if promote else "ci_lower_below_threshold",
        "margin": round(ci_lower - threshold, 4),
    }


def games_needed_for_significance(
    target_win_rate: float = 0.55,
    threshold: float = 0.55,
    confidence: float = 0.95,
) -> int:
    """Estimate games needed to achieve statistical significance.

    If the true win rate is target_win_rate, approximately how many games
    are needed for the CI lower bound to exceed threshold?
    """
    from statistics import NormalDist

    # Z-score for confidence level
    z = NormalDist().inv_cdf(0.5 + confidence / 2)

    # For Wilson interval, need approximately:
    # p - z * sqrt(p(1-p)/n) >= threshold
    # Solving for n: n >= z^2 * p(1-p) / (p - threshold)^2

    p = target_win_rate
    margin = p - threshold

    if margin <= 0:
        return 10000  # Very large number if target <= threshold

    n_approx = (z ** 2) * p * (1 - p) / (margin ** 2)
    return int(n_approx) + 1


def run_promotion_tournament(
    candidate_path: str,
    baseline_path: str,
    board_type: str = "square8",
    num_players: int = 2,
    min_games: int = 30,
    max_games: int = 200,
    threshold: float = 0.55,
    confidence: float = 0.95,
) -> dict:
    """Run an adaptive tournament to determine promotion.

    Plays games until either:
    - CI lower bound > threshold (promote)
    - CI upper bound < threshold (reject)
    - Max games reached (inconclusive)
    """
    from app.training.tournament import run_tournament

    batch_size = 20
    total_wins = 0
    total_losses = 0
    total_draws = 0
    games_played = 0

    while games_played < max_games:
        # Run a batch of games
        games_this_batch = min(batch_size, max_games - games_played)

        results = run_tournament(
            model_a_path=candidate_path,
            model_b_path=baseline_path,
            num_games=games_this_batch,
            board_type=board_type,
            num_players=num_players,
            seed=42 + games_played,
        )

        total_wins += results["model_a_wins"]
        total_losses += results["model_b_wins"]
        total_draws += results["draws"]
        games_played += results["total_games"]

        # Evaluate after minimum games
        if games_played >= min_games:
            eval_result = evaluate_promotion(
                total_wins, total_losses, total_draws,
                threshold=threshold, confidence=confidence
            )

            # Early stop if we can confidently promote or reject
            if eval_result["ci_lower"] >= threshold:
                eval_result["stop_reason"] = "promote_confident"
                return eval_result
            elif eval_result["ci_upper"] < threshold:
                eval_result["stop_reason"] = "reject_confident"
                return eval_result

    # Max games reached
    final_result = evaluate_promotion(
        total_wins, total_losses, total_draws,
        threshold=threshold, confidence=confidence
    )
    final_result["stop_reason"] = "max_games_reached"
    return final_result


def main():
    parser = argparse.ArgumentParser(description="Elo-based promotion gate")
    parser.add_argument("--candidate", required=True, help="Candidate model path or ID")
    parser.add_argument("--baseline", help="Baseline model path or ID")
    parser.add_argument("--threshold", type=float, default=0.55, help="Win rate threshold")
    parser.add_argument("--confidence", type=float, default=0.95, help="Confidence level")
    parser.add_argument("--min-games", type=int, default=30, help="Minimum games to play")
    parser.add_argument("--max-games", type=int, default=200, help="Maximum games to play")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--estimate-games", action="store_true",
                        help="Estimate games needed for given win rate")
    parser.add_argument("--target-rate", type=float, default=0.60,
                        help="Target win rate for estimation")
    parser.add_argument("--output", help="Output JSON file")

    args = parser.parse_args()

    if args.estimate_games:
        n = games_needed_for_significance(
            target_win_rate=args.target_rate,
            threshold=args.threshold,
            confidence=args.confidence,
        )
        print(f"Estimated games needed for {args.target_rate:.0%} true win rate:")
        print(f"  Threshold: {args.threshold:.0%}")
        print(f"  Confidence: {args.confidence:.0%}")
        print(f"  Games needed: ~{n}")
        return 0

    if not args.baseline:
        print("Error: --baseline required for tournament")
        return 1

    # Run tournament
    print(f"Running promotion tournament:")
    print(f"  Candidate: {args.candidate}")
    print(f"  Baseline: {args.baseline}")
    print(f"  Threshold: {args.threshold:.0%}")
    print(f"  Confidence: {args.confidence:.0%}")
    print()

    result = run_promotion_tournament(
        candidate_path=args.candidate,
        baseline_path=args.baseline,
        board_type=args.board,
        num_players=args.players,
        min_games=args.min_games,
        max_games=args.max_games,
        threshold=args.threshold,
        confidence=args.confidence,
    )

    # Print results
    print("=" * 60)
    print("PROMOTION GATE RESULT")
    print("=" * 60)
    print(f"Decision: {'PROMOTE' if result['promote'] else 'REJECT'}")
    print(f"Win rate: {result['win_rate']:.1%} ({result['wins']}/{result['decisive_games']})")
    print(f"95% CI: [{result['ci_lower']:.1%}, {result['ci_upper']:.1%}]")
    print(f"Threshold: {result['threshold']:.1%}")
    print(f"Margin: {result['margin']:+.1%}")
    print(f"Est. Elo diff: {result['estimated_elo_diff']:+.0f}")
    print(f"Games played: {result['total_games']}")
    print(f"Stop reason: {result['stop_reason']}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    return 0 if result["promote"] else 1


if __name__ == "__main__":
    sys.exit(main())
