#!/usr/bin/env python3
"""Fast policy-only gauntlet evaluation using PolicyOnlyAI.

This script evaluates models quickly without MCTS search, using only the policy head.
Use for fast baseline verification when MCTS gauntlet is too slow.

Usage:
    python scripts/policy_gauntlet.py --model models/canonical_hex8_2p.pth --games 20
    python scripts/policy_gauntlet.py --model models/canonical_square8_3p.pth --board-type square8 --num-players 3
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from app.ai.policy_only_ai import PolicyOnlyAI
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI
from app.rules import get_rules_engine, create_game_state
from app.models import BoardType, GamePhase

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run_game(
    model_ai,
    opponent_ai,
    board_type: BoardType,
    num_players: int,
    model_player: int = 0,
) -> int:
    """Run a single game and return winner index."""
    state = create_game_state(
        board_type=board_type.value,
        num_players=num_players,
    )
    engine = get_rules_engine()

    # Map players to AIs (1-indexed player numbers)
    ais = {}
    for i in range(num_players):
        if i == model_player:
            ais[i + 1] = model_ai
        else:
            ais[i + 1] = opponent_ai

    # Play game
    move_count = 0
    while state.current_phase != GamePhase.GAME_OVER:
        current_player = state.current_player
        ai = ais[current_player]
        move = ai.select_move(state)
        if move is None:
            logger.warning(f"No move from player {current_player}, ending game")
            return -1
        state = engine.apply_move(state, move)
        move_count += 1

        if move_count > 1000:  # Safety limit
            logger.warning("Game exceeded 1000 moves, ending as draw")
            return -1

    return state.winner if state.winner else -1


def run_gauntlet(
    model_path: str,
    board_type: str,
    num_players: int,
    games: int,
    opponent_type: str = "random",
) -> dict:
    """Run gauntlet evaluation against specified opponent."""
    bt = BoardType(board_type)

    # Create model AI (policy-only for speed)
    model_ai = PolicyOnlyAI(
        model_path=model_path,
        board_type=bt,
        num_players=num_players,
    )

    # Create opponent AI
    if opponent_type == "random":
        opponent_ai = RandomAI()
    elif opponent_type == "heuristic":
        opponent_ai = HeuristicAI()
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")

    # Track results
    wins = 0
    losses = 0
    draws = 0

    for i in range(games):
        # Alternate who goes first for fairness (0-indexed position)
        model_player = i % num_players

        # run_game returns 1-indexed winner (1-4) or -1 for draw
        winner = run_game(model_ai, opponent_ai, bt, num_players, model_player)

        # model_player is 0-indexed, winner is 1-indexed
        if winner == model_player + 1:
            wins += 1
            result_str = "WIN"
        elif winner == -1:
            draws += 1
            result_str = "DRAW"
        else:
            losses += 1
            result_str = "LOSS"

        print(f"Game {i + 1}/{games}: {result_str} (model as P{model_player + 1}, winner P{winner})")

    win_rate = wins / games if games > 0 else 0
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "games": games,
        "win_rate": win_rate,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Fast policy-only gauntlet evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Path to model checkpoint")
    parser.add_argument("--board-type", default="hex8", choices=["hex8", "square8", "square19", "hexagonal"])
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--games", type=int, default=20, help="Games per opponent")
    parser.add_argument("--opponents", default="random,heuristic", help="Comma-separated opponent types")
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Policy-Only Gauntlet: {model_path.name}")
    print(f"Config: {args.board_type} {args.num_players}p, {args.games} games/opponent")
    print("=" * 60)

    # Required thresholds for passing
    thresholds = {
        "random": 0.70 if args.num_players == 2 else 0.50,
        "heuristic": 0.50 if args.num_players == 2 else 0.35,
    }

    all_results = {}
    all_passed = True

    for opponent in args.opponents.split(","):
        opponent = opponent.strip()
        print(f"\n--- vs {opponent.upper()} ---")

        results = run_gauntlet(
            model_path=str(model_path),
            board_type=args.board_type,
            num_players=args.num_players,
            games=args.games,
            opponent_type=opponent,
        )

        all_results[opponent] = results
        threshold = thresholds.get(opponent, 0.50)
        passed = results["win_rate"] >= threshold

        print(f"\nResult: {results['wins']}/{results['games']} = {results['win_rate']:.1%}")
        print(f"Threshold: {threshold:.0%} - {'PASSED' if passed else 'FAILED'}")

        if not passed:
            all_passed = False

    # Summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    total_wins = sum(r["wins"] for r in all_results.values())
    total_games = sum(r["games"] for r in all_results.values())
    print(f"Total: {total_wins}/{total_games} ({total_wins / total_games:.1%})")
    for opp, res in all_results.items():
        print(f"  vs {opp}: {res['wins']}/{res['games']} ({res['win_rate']:.1%})")
    print(f"\nOVERALL: {'PASSED' if all_passed else 'FAILED'}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
