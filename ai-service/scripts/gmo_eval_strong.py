#!/usr/bin/env python3
"""Evaluate GMO against stronger baselines (Policy-only, MCTS, Descent).

This script tests GMO against higher-tier AI opponents to understand
its position in the difficulty ladder.

Usage:
    python scripts/gmo_eval_strong.py --games 50
    python scripts/gmo_eval_strong.py --opponents policy,mcts_100,descent
"""

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.factory import AIFactory
from archive.deprecated_ai.gmo_ai import GMOAI, GMOConfig
from app.game_engine import GameEngine
from app.models import AIConfig, AIType, BoardType, GameStatus
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

GMO_CHECKPOINT = PROJECT_ROOT / "models" / "gmo" / "gmo_best.pt"


@dataclass
class MatchResult:
    """Results from a matchup."""
    opponent: str
    gmo_wins: int
    opponent_wins: int
    draws: int
    gmo_winrate: float
    avg_game_length: float
    avg_gmo_time_per_move: float
    avg_opponent_time_per_move: float


def create_gmo(player_number: int, device: str = "cpu") -> GMOAI:
    """Create GMO AI with trained checkpoint."""
    ai_config = AIConfig(difficulty=5)
    gmo_config = GMOConfig(device=device)
    ai = GMOAI(player_number=player_number, config=ai_config, gmo_config=gmo_config)
    if GMO_CHECKPOINT.exists():
        ai.load_checkpoint(GMO_CHECKPOINT)
    return ai


def create_opponent(opponent_type: str, player_number: int):
    """Create opponent AI by type."""
    opponent_map = {
        "random": (AIType.RANDOM, AIConfig(difficulty=1)),
        "heuristic": (AIType.HEURISTIC, AIConfig(difficulty=3)),
        "policy": (AIType.POLICY_ONLY, AIConfig(difficulty=5)),
        "policy_only": (AIType.POLICY_ONLY, AIConfig(difficulty=5)),
        "descent": (AIType.DESCENT, AIConfig(difficulty=5, think_time=500)),
        "descent_3": (AIType.DESCENT, AIConfig(difficulty=3, think_time=300)),
        "mcts_25": (AIType.MCTS, AIConfig(difficulty=3, mcts_simulations=25)),
        "mcts_100": (AIType.MCTS, AIConfig(difficulty=5, mcts_simulations=100)),
        "mcts_400": (AIType.MCTS, AIConfig(difficulty=7, mcts_simulations=400)),
        "minimax": (AIType.MINIMAX, AIConfig(difficulty=4)),
        "minimax_deep": (AIType.MINIMAX, AIConfig(difficulty=6)),
    }

    if opponent_type.lower() not in opponent_map:
        raise ValueError(f"Unknown opponent type: {opponent_type}. "
                        f"Available: {list(opponent_map.keys())}")

    ai_type, config = opponent_map[opponent_type.lower()]
    return AIFactory.create(ai_type, player_number=player_number, config=config)


def play_game(
    ai1,
    ai2,
    board_type: BoardType = BoardType.SQUARE8,
    max_moves: int = 500,
    game_idx: int = 0,
) -> tuple[int | None, int, float, float]:
    """Play a game and return (winner, moves, ai1_time, ai2_time)."""
    state = create_initial_state(board_type=board_type, num_players=2)

    ais = {1: ai1, 2: ai2}

    # Reset AIs with unique seeds per game for variety
    for p, ai in ais.items():
        if hasattr(ai, 'reset_for_new_game'):
            ai.reset_for_new_game(rng_seed=game_idx * 1000 + p)
    times = {1: 0.0, 2: 0.0}
    num_moves = 0

    for _ in range(max_moves):
        if state.game_status != GameStatus.ACTIVE:
            break

        current_player = state.current_player
        current_ai = ais[current_player]

        legal_moves = GameEngine.get_valid_moves(state, current_player)
        if not legal_moves:
            # Check for phase requirements (no-action moves)
            phase_req = GameEngine.get_phase_requirement(state, current_player)
            if phase_req:
                bookkeeping_move = GameEngine.synthesize_bookkeeping_move(phase_req, state)
                state = GameEngine.apply_move(state, bookkeeping_move)
                num_moves += 1
                continue
            else:
                break

        start = time.time()
        move = current_ai.select_move(state)
        if move is None:
            break
        times[current_player] += time.time() - start

        state = GameEngine.apply_move(state, move)
        num_moves += 1

    winner = state.winner if state.game_status == GameStatus.COMPLETED else None
    return winner, num_moves, times[1], times[2]


def evaluate_matchup(
    opponent_type: str,
    num_games: int = 50,
    device: str = "cpu",
) -> MatchResult:
    """Evaluate GMO vs a specific opponent."""
    logger.info(f"Evaluating GMO vs {opponent_type} ({num_games} games)...")

    gmo_wins = 0
    opponent_wins = 0
    draws = 0
    total_moves = 0
    total_gmo_time = 0.0
    total_opponent_time = 0.0
    total_gmo_moves = 0
    total_opponent_moves = 0

    games_per_side = num_games // 2

    # GMO as player 1
    for game_idx in range(games_per_side):
        gmo = create_gmo(1, device)
        opponent = create_opponent(opponent_type, 2)

        winner, moves, p1_time, p2_time = play_game(gmo, opponent, game_idx=game_idx)
        total_moves += moves
        total_gmo_time += p1_time
        total_opponent_time += p2_time
        total_gmo_moves += moves // 2 + (moves % 2)
        total_opponent_moves += moves // 2

        if winner == 1:
            gmo_wins += 1
        elif winner == 2:
            opponent_wins += 1
        else:
            draws += 1

        if (game_idx + 1) % 10 == 0:
            logger.info(f"  P1 games: {game_idx + 1}/{games_per_side}")

    # GMO as player 2
    for game_idx in range(games_per_side):
        opponent = create_opponent(opponent_type, 1)
        gmo = create_gmo(2, device)

        winner, moves, p1_time, p2_time = play_game(opponent, gmo, game_idx=games_per_side + game_idx)
        total_moves += moves
        total_opponent_time += p1_time
        total_gmo_time += p2_time
        total_opponent_moves += moves // 2 + (moves % 2)
        total_gmo_moves += moves // 2

        if winner == 2:
            gmo_wins += 1
        elif winner == 1:
            opponent_wins += 1
        else:
            draws += 1

        if (game_idx + 1) % 10 == 0:
            logger.info(f"  P2 games: {game_idx + 1}/{games_per_side}")

    total_games = gmo_wins + opponent_wins + draws

    return MatchResult(
        opponent=opponent_type,
        gmo_wins=gmo_wins,
        opponent_wins=opponent_wins,
        draws=draws,
        gmo_winrate=gmo_wins / total_games * 100 if total_games > 0 else 0,
        avg_game_length=total_moves / total_games if total_games > 0 else 0,
        avg_gmo_time_per_move=total_gmo_time / total_gmo_moves if total_gmo_moves > 0 else 0,
        avg_opponent_time_per_move=total_opponent_time / total_opponent_moves if total_opponent_moves > 0 else 0,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate GMO against strong baselines"
    )
    parser.add_argument(
        "--opponents", type=str,
        default="random,heuristic,policy,mcts_100,descent",
        help="Comma-separated list of opponents"
    )
    parser.add_argument(
        "--games", type=int, default=50,
        help="Games per opponent"
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--output", type=Path,
        help="Output JSON file for results"
    )

    args = parser.parse_args()

    opponents = [o.strip() for o in args.opponents.split(",")]
    results: list[MatchResult] = []

    print("\n" + "="*70)
    print("GMO EVALUATION vs STRONG BASELINES")
    print("="*70)

    for opponent in opponents:
        try:
            result = evaluate_matchup(
                opponent,
                num_games=args.games,
                device=args.device,
            )
            results.append(result)

            print(f"\nGMO vs {opponent}:")
            print(f"  Win rate: {result.gmo_winrate:.1f}%")
            print(f"  Record: {result.gmo_wins}W / {result.opponent_wins}L / {result.draws}D")
            print(f"  Avg game length: {result.avg_game_length:.1f} moves")
            print(f"  GMO time/move: {result.avg_gmo_time_per_move*1000:.1f}ms")
            print(f"  {opponent} time/move: {result.avg_opponent_time_per_move*1000:.1f}ms")

        except Exception as e:
            logger.error(f"Failed to evaluate vs {opponent}: {e}")
            continue

    # Summary table
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"\n{'Opponent':<15} {'Win Rate':>10} {'Record':>15} {'Avg Length':>12}")
    print("-"*55)

    for r in results:
        record = f"{r.gmo_wins}W/{r.opponent_wins}L/{r.draws}D"
        print(f"{r.opponent:<15} {r.gmo_winrate:>9.1f}% {record:>15} {r.avg_game_length:>11.1f}")

    # Difficulty ladder placement
    print("\n" + "-"*55)
    print("\nDifficulty Ladder Placement:")

    ladder = [
        ("Random", "random"),
        ("Heuristic", "heuristic"),
        ("Policy-only", "policy"),
        ("MCTS-100", "mcts_100"),
        ("Descent", "descent"),
    ]

    result_map = {r.opponent: r.gmo_winrate for r in results}

    for name, key in ladder:
        if key in result_map:
            wr = result_map[key]
            if wr >= 60:
                status = "BEATS"
            elif wr >= 40:
                status = "COMPETITIVE"
            else:
                status = "LOSES TO"
            print(f"  {name:<15} - GMO {status} ({wr:.0f}%)")

    # Save results if requested
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output_data = {
            "timestamp": datetime.now().isoformat(),
            "games_per_opponent": args.games,
            "results": [
                {
                    "opponent": r.opponent,
                    "gmo_wins": r.gmo_wins,
                    "opponent_wins": r.opponent_wins,
                    "draws": r.draws,
                    "gmo_winrate": r.gmo_winrate,
                    "avg_game_length": r.avg_game_length,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
