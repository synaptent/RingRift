#!/usr/bin/env python3
"""Sharded gauntlet - run a subset of games on a single node for distributed evaluation.

Usage:
    # Run games 0-4 of 10 total games vs random
    python scripts/sharded_gauntlet.py --model models/canonical_hex8_2p.pth \\
        --shard 0 --total-shards 2 --opponent random --games 10

    # Aggregate results from multiple shards
    python scripts/sharded_gauntlet.py --aggregate --results-dir /tmp/gauntlet_results

This script enables distributing gauntlet games across multiple nodes by:
1. Each node runs a shard (subset) of games
2. Results are written to JSON files
3. Final aggregation combines all shard results
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.ai.policy_only_ai import PolicyOnlyAI
from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.random_ai import RandomAI
from app.ai.heuristic_ai import HeuristicAI
from app.rules import get_rules_engine, create_game_state
from app.models import BoardType, GamePhase, AIConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def create_model_ai(model_path: str, board_type: BoardType, player_number: int, use_mcts: bool = False):
    """Create AI for the model being evaluated."""
    config = AIConfig(
        difficulty=5,
        nn_model_id=model_path,
        policy_temperature=0.1 if not use_mcts else 1.0,
        allow_fresh_weights=False,
    )
    if use_mcts:
        return GumbelMCTSAI(
            player_number=player_number,
            config=config,
            board_type=board_type,
            simulation_budget=64,  # Lower budget for faster games
        )
    return PolicyOnlyAI(
        player_number=player_number,
        config=config,
        board_type=board_type,
    )


def run_single_game(
    model_ai,
    opponent_ais: dict,
    board_type: BoardType,
    num_players: int,
    model_player: int,
) -> int:
    """Run a single game and return winner (1-indexed) or -1 for draw."""
    state = create_game_state(
        board_type=board_type.value,
        num_players=num_players,
    )
    engine = get_rules_engine()

    ais = {model_player + 1: model_ai}
    ais.update(opponent_ais)

    move_count = 0
    while state.current_phase != GamePhase.GAME_OVER:
        current_player = state.current_player
        ai = ais[current_player]
        move = ai.select_move(state)
        if move is None:
            return -1
        state = engine.apply_move(state, move)
        move_count += 1
        if move_count > 1000:
            return -1

    return state.winner if state.winner else -1


def run_shard(
    model_path: str,
    board_type: str,
    num_players: int,
    opponent_type: str,
    total_games: int,
    shard_index: int,
    total_shards: int,
    use_mcts: bool = False,
) -> dict:
    """Run a shard of games and return results."""
    bt = BoardType(board_type)

    # Calculate which games this shard handles
    games_per_shard = total_games // total_shards
    remainder = total_games % total_shards

    # Distribute remainder across first shards
    start_game = shard_index * games_per_shard + min(shard_index, remainder)
    if shard_index < remainder:
        games_per_shard += 1
    end_game = start_game + games_per_shard

    logger.info(f"Shard {shard_index}/{total_shards}: games {start_game}-{end_game-1} of {total_games}")

    wins = 0
    losses = 0
    draws = 0
    game_results = []

    for game_num in range(start_game, end_game):
        # Alternate starting player for fairness
        model_player = game_num % num_players

        # Create model AI
        model_ai = create_model_ai(model_path, bt, model_player + 1, use_mcts)

        # Create opponent AIs
        opponent_config = AIConfig(difficulty=5)
        opponent_ais = {}
        for p in range(1, num_players + 1):
            if p == model_player + 1:
                continue
            if opponent_type == "random":
                opponent_ais[p] = RandomAI(player_number=p, config=opponent_config)
            elif opponent_type == "heuristic":
                opponent_ais[p] = HeuristicAI(player_number=p, config=opponent_config)
            else:
                raise ValueError(f"Unknown opponent type: {opponent_type}")

        # Run game
        winner = run_single_game(model_ai, opponent_ais, bt, num_players, model_player)

        if winner == model_player + 1:
            wins += 1
            result = "win"
        elif winner == -1:
            draws += 1
            result = "draw"
        else:
            losses += 1
            result = "loss"

        game_results.append({
            "game_num": game_num,
            "model_player": model_player + 1,
            "winner": winner,
            "result": result,
        })

        logger.info(f"Game {game_num}: {result.upper()} (model as P{model_player + 1}, winner P{winner})")

    return {
        "shard_index": shard_index,
        "total_shards": total_shards,
        "opponent": opponent_type,
        "games_played": len(game_results),
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "game_results": game_results,
    }


def aggregate_results(results_dir: str) -> dict:
    """Aggregate results from multiple shard files."""
    results_path = Path(results_dir)

    aggregated = {}

    for result_file in results_path.glob("shard_*.json"):
        with open(result_file) as f:
            shard_result = json.load(f)

        opponent = shard_result["opponent"]
        if opponent not in aggregated:
            aggregated[opponent] = {
                "wins": 0,
                "losses": 0,
                "draws": 0,
                "games": 0,
            }

        aggregated[opponent]["wins"] += shard_result["wins"]
        aggregated[opponent]["losses"] += shard_result["losses"]
        aggregated[opponent]["draws"] += shard_result["draws"]
        aggregated[opponent]["games"] += shard_result["games_played"]

    # Calculate win rates and pass/fail
    thresholds = {
        "random": 0.70,
        "heuristic": 0.50,
    }

    all_passed = True
    for opponent, stats in aggregated.items():
        stats["win_rate"] = stats["wins"] / stats["games"] if stats["games"] > 0 else 0
        threshold = thresholds.get(opponent, 0.50)
        stats["threshold"] = threshold
        stats["passed"] = stats["win_rate"] >= threshold
        if not stats["passed"]:
            all_passed = False

    return {
        "opponents": aggregated,
        "all_passed": all_passed,
    }


def main():
    parser = argparse.ArgumentParser(description="Sharded gauntlet evaluation")
    parser.add_argument("--model", help="Path to model checkpoint")
    parser.add_argument("--board-type", default="hex8", choices=["hex8", "square8", "square19", "hexagonal"])
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--opponent", default="random", choices=["random", "heuristic"])
    parser.add_argument("--games", type=int, default=20, help="Total games for this opponent")
    parser.add_argument("--shard", type=int, default=0, help="Shard index (0-based)")
    parser.add_argument("--total-shards", type=int, default=1, help="Total number of shards")
    parser.add_argument("--use-mcts", action="store_true", help="Use MCTS instead of policy-only")
    parser.add_argument("--output", help="Output JSON file for results")
    parser.add_argument("--aggregate", action="store_true", help="Aggregate results from shards")
    parser.add_argument("--results-dir", help="Directory containing shard results")

    args = parser.parse_args()

    if args.aggregate:
        if not args.results_dir:
            print("ERROR: --results-dir required for aggregation", file=sys.stderr)
            sys.exit(1)
        results = aggregate_results(args.results_dir)
        print("\n" + "=" * 60)
        print("AGGREGATED RESULTS")
        print("=" * 60)
        for opponent, stats in results["opponents"].items():
            status = "PASSED" if stats["passed"] else "FAILED"
            print(f"vs {opponent}: {stats['wins']}/{stats['games']} ({stats['win_rate']:.1%}) - {status}")
        print(f"\nOVERALL: {'PASSED' if results['all_passed'] else 'FAILED'}")
        return

    if not args.model:
        print("ERROR: --model required", file=sys.stderr)
        sys.exit(1)

    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Sharded Gauntlet: {model_path.name}")
    print(f"Config: {args.board_type} {args.num_players}p")
    print(f"Shard: {args.shard}/{args.total_shards}, Opponent: {args.opponent}, Games: {args.games}")
    print("=" * 60)

    results = run_shard(
        model_path=str(model_path),
        board_type=args.board_type,
        num_players=args.num_players,
        opponent_type=args.opponent,
        total_games=args.games,
        shard_index=args.shard,
        total_shards=args.total_shards,
        use_mcts=args.use_mcts,
    )

    win_rate = results["wins"] / results["games_played"] if results["games_played"] > 0 else 0
    print(f"\nShard Result: {results['wins']}/{results['games_played']} ({win_rate:.1%})")

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results written to: {output_path}")


if __name__ == "__main__":
    main()
