#!/usr/bin/env python3
"""Generate training data by playing neural net against HeuristicAI.

This generates high-quality training data by:
1. Playing Gumbel MCTS vs HeuristicAI (both sides)
2. Extracting soft policy targets from MCTS
3. Using game outcomes for value training

This is valuable because HeuristicAI is currently stronger, so the
neural net learns from games against a better opponent.

Usage:
    python scripts/train_vs_heuristic.py \
        --num-games 1000 \
        --model-id distilled_sq8_2p_v3 \
        --output data/vs_heuristic/games.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.ai.gumbel_mcts_ai import GumbelMCTSAI
from app.ai.heuristic_ai import HeuristicAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GameState, Move
from app.rules.default_engine import DefaultRulesEngine
from app.training.initial_state import create_initial_state

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def serialize_move(move: Move, mcts_policy: dict[str, float] | None = None) -> dict[str, Any]:
    """Serialize move to JSON-compatible dict."""
    move_data = {
        "type": move.type.value,
        "player": move.player,
    }
    if mcts_policy:
        move_data["mcts_policy"] = mcts_policy
    if move.from_pos:
        move_data["from"] = {"x": move.from_pos.x, "y": move.from_pos.y}
    if move.to:
        move_data["to"] = {"x": move.to.x, "y": move.to.y}
    if move.capture_target:
        move_data["capture_target"] = {"x": move.capture_target.x, "y": move.capture_target.y}
    return move_data


def serialize_state(state: GameState) -> dict[str, Any]:
    """Serialize game state to JSON-compatible dict."""
    data = state.model_dump() if hasattr(state, 'model_dump') else state.dict()
    for key, value in data.items():
        if hasattr(value, 'isoformat'):
            data[key] = value.isoformat()
    return data


def find_move_index(move: Move, legal_moves: list[Move]) -> int:
    """Find the index of a move in the legal moves list."""
    for i, legal in enumerate(legal_moves):
        if (move.type == legal.type and
            getattr(move, 'from_pos', None) == getattr(legal, 'from_pos', None) and
            getattr(move, 'to', None) == getattr(legal, 'to', None)):
            return i
    return -1


def play_game(
    nn_player: int,
    model_id: str,
    board_type: BoardType,
    simulations: int,
    seed: int,
    hybrid_alpha: float | None = None,
) -> dict[str, Any] | None:
    """Play a game between neural net and heuristic."""
    random.seed(seed)
    np.random.seed(seed)

    # Create players
    nn_config = AIConfig(
        difficulty=5,
        nn_model_id=model_id,
        think_time=500,
        gumbel_simulation_budget=simulations,
        rngSeed=seed,
        heuristic_blend_alpha=hybrid_alpha,
    )
    nn_ai = GumbelMCTSAI(player_number=nn_player, config=nn_config, board_type=board_type)

    heuristic_player = 3 - nn_player  # 1 -> 2, 2 -> 1
    heuristic_ai = HeuristicAI(player_number=heuristic_player, config=AIConfig(difficulty=5))

    state = create_initial_state(board_type, 2)
    initial_state = serialize_state(state)
    moves_data = []
    engine = DefaultRulesEngine()
    game_engine = GameEngine()

    for move_num in range(500):
        status = state.game_status.value
        if status not in ['active', 'in_progress']:
            break

        current = state.current_player
        legal_moves = engine.get_valid_moves(state, current)
        if not legal_moves:
            break

        if current == nn_player:
            # Neural net move with MCTS policy
            move = nn_ai.select_move(state)
            if move is None:
                break

            # Extract soft policy from MCTS
            mcts_policy = {}
            if hasattr(nn_ai, '_last_search_actions') and nn_ai._last_search_actions:
                total_visits = sum(a.visit_count for a in nn_ai._last_search_actions)
                if total_visits > 0:
                    for i, action in enumerate(nn_ai._last_search_actions):
                        if action.visit_count > 0:
                            prob = action.visit_count / total_visits
                            if prob > 1e-6:
                                mcts_policy[str(i)] = float(prob)

            if not mcts_policy:
                idx = find_move_index(move, legal_moves)
                if idx >= 0:
                    mcts_policy[str(idx)] = 1.0

            moves_data.append(serialize_move(move, mcts_policy))
        else:
            # Heuristic move (one-hot policy)
            move = heuristic_ai.select_move(state)
            if move is None:
                break

            idx = find_move_index(move, legal_moves)
            policy = {str(idx): 1.0} if idx >= 0 else None
            moves_data.append(serialize_move(move, policy))

        state = game_engine.apply_move(state, move)

    board_size = 8 if board_type == BoardType.SQUARE8 else 19

    return {
        "game_id": f"vs_heuristic_{nn_player}_{seed}",
        "board_type": board_type.value,
        "board_size": board_size,
        "num_players": 2,
        "winner": state.winner,
        "move_count": len(moves_data),
        "game_status": state.game_status.value if hasattr(state.game_status, 'value') else str(state.game_status),
        "nn_player": nn_player,
        "nn_won": state.winner == nn_player,
        "engine_mode": "vs_heuristic",
        "moves": moves_data,
        "initial_state": initial_state,
        "timestamp": datetime.now().isoformat(),
        "source": "train_vs_heuristic.py",
    }


def main():
    parser = argparse.ArgumentParser(description="Generate training data vs HeuristicAI")
    parser.add_argument("--num-games", type=int, default=500, help="Number of games")
    parser.add_argument("--model-id", type=str, default="distilled_sq8_2p_v3", help="Neural net model")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--simulations", type=int, default=400, help="MCTS simulations (higher = stronger, slower)")
    parser.add_argument("--hybrid-alpha", type=float, default=None, help="Hybrid NN+Heuristic blend alpha (0.6 recommended)")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    args = parser.parse_args()

    board_type = BoardType.SQUARE8 if "square8" in args.board.lower() else BoardType.SQUARE19

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("TRAINING VS HEURISTIC")
    logger.info("=" * 60)
    logger.info(f"Model: {args.model_id}")
    logger.info(f"Games: {args.num_games}")
    logger.info(f"Simulations: {args.simulations}")
    if args.hybrid_alpha is not None:
        logger.info(f"Hybrid Mode: alpha={args.hybrid_alpha} (NN={args.hybrid_alpha:.0%}, Heuristic={1-args.hybrid_alpha:.0%})")
    logger.info(f"Output: {output_path}")
    logger.info("=" * 60)

    nn_wins = 0
    heuristic_wins = 0
    games_generated = 0

    with open(output_path, 'w') as f:
        for game_idx in range(args.num_games):
            seed = int(time.time() * 1000) % (2**31) + game_idx * 10000

            # Alternate which side the neural net plays
            nn_player = 1 if game_idx % 2 == 0 else 2

            try:
                game = play_game(
                    nn_player=nn_player,
                    model_id=args.model_id,
                    board_type=board_type,
                    simulations=args.simulations,
                    seed=seed,
                    hybrid_alpha=args.hybrid_alpha,
                )

                if game:
                    if game.get("nn_won"):
                        nn_wins += 1
                    elif game.get("winner") is not None:
                        heuristic_wins += 1

                    f.write(json.dumps(game) + "\n")
                    games_generated += 1

                    if (game_idx + 1) % 50 == 0:
                        nn_rate = 100 * nn_wins / games_generated if games_generated > 0 else 0
                        logger.info(
                            f"Game {game_idx + 1}/{args.num_games} | "
                            f"NN: {nn_wins} ({nn_rate:.1f}%) | "
                            f"Heuristic: {heuristic_wins}"
                        )
            except Exception as e:
                logger.warning(f"Failed game {game_idx}: {e}")

    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Games: {games_generated}")
    if games_generated > 0:
        logger.info(f"NN wins: {nn_wins} ({100*nn_wins/games_generated:.1f}%)")
        logger.info(f"Heuristic wins: {heuristic_wins} ({100*heuristic_wins/games_generated:.1f}%)")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
