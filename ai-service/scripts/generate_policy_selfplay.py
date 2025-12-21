#!/usr/bin/env python3
"""Generate selfplay games using PolicyOnlyAI for balanced training data.

Unlike Gumbel MCTS, PolicyOnlyAI doesn't use the value head, avoiding
any value head bias in the generated games. This produces more balanced
outcomes suitable for training unbiased value heads.

Usage:
    python scripts/generate_policy_selfplay.py \
        --num-games 500 \
        --board square8 \
        --model-id distilled_sq8_2p \
        --output data/selfplay/policy_balanced.jsonl
"""

from __future__ import annotations

import json
import logging
import os
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

from app.ai.policy_only_ai import PolicyOnlyAI
from app.models import AIConfig, BoardType
from app.rules.default_engine import DefaultRulesEngine
from app.training.env import RingRiftEnv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def parse_board_type(board_str: str) -> BoardType:
    """Parse board type string to enum."""
    board_str = board_str.lower()
    if "square8" in board_str or "sq8" in board_str:
        return BoardType.SQUARE8
    elif "square19" in board_str or "sq19" in board_str:
        return BoardType.SQUARE19
    elif "hex" in board_str:
        return BoardType.HEXAGONAL
    return BoardType.SQUARE8


def serialize_state(state) -> dict[str, Any]:
    """Serialize GameState to JSON-compatible dict."""
    data = state.model_dump() if hasattr(state, 'model_dump') else state.dict()
    for key, value in data.items():
        if hasattr(value, 'isoformat'):
            data[key] = value.isoformat()
    return data


def serialize_move(move, policy_probs: dict[str, float] | None = None) -> dict[str, Any]:
    """Serialize a Move object to a JSON-compatible dict."""
    move_data = {
        "type": move.type.value,
        "player": move.player,
    }
    if policy_probs:
        move_data["mcts_policy"] = policy_probs  # Keep same key for compatibility
    if move.from_pos:
        move_data["from"] = {"x": move.from_pos.x, "y": move.from_pos.y}
    if move.to:
        move_data["to"] = {"x": move.to.x, "y": move.to.y}
    if move.capture_target:
        move_data["capture_target"] = {"x": move.capture_target.x, "y": move.capture_target.y}
    return move_data


def find_move_index(move, legal_moves: list) -> int:
    """Find the index of a move in the legal moves list."""
    move_type = getattr(move, 'type', None)
    move_from = getattr(move, 'from_pos', None)
    move_to = getattr(move, 'to', None)

    for i, legal in enumerate(legal_moves):
        legal_type = getattr(legal, 'type', None)
        legal_from = getattr(legal, 'from_pos', None)
        legal_to = getattr(legal, 'to', None)

        if move_type != legal_type:
            continue
        if move_from is not None and legal_from is not None:
            if move_from.x != legal_from.x or move_from.y != legal_from.y:
                continue
        elif move_from is not None or legal_from is not None:
            continue
        if move_to is not None and legal_to is not None:
            if move_to.x != legal_to.x or move_to.y != legal_to.y:
                continue
        elif move_to is not None or legal_to is not None:
            continue
        return i
    return -1


def generate_game(
    env: RingRiftEnv,
    ai_players: dict[int, PolicyOnlyAI],
    game_idx: int,
    max_moves: int = 500,
) -> dict[str, Any] | None:
    """Generate a single game with policy distributions."""
    state = env.reset()
    initial_state = serialize_state(state)
    moves_data = []
    done = False
    move_count = 0
    engine = DefaultRulesEngine()

    while not done and move_count < max_moves:
        current_player = state.current_player
        ai = ai_players.get(current_player)
        if ai is None:
            break

        legal_moves = engine.get_valid_moves(state, current_player)
        move = ai.select_move(state)

        # Get policy distribution from PolicyOnlyAI with integer indices
        policy_probs = {}
        if hasattr(ai, '_get_policy_scores') and legal_moves:
            try:
                scores = ai._get_policy_scores(state, legal_moves)
                if scores is not None and len(scores) > 0:
                    # Convert to probabilities with temperature
                    temp = getattr(ai, 'temperature', 1.0)
                    if temp <= 0.01:
                        probs = np.zeros_like(scores)
                        probs[np.argmax(scores)] = 1.0
                    else:
                        # Softmax with temperature
                        x_scaled = np.log(np.maximum(scores, 1e-10)) / temp
                        x_scaled = x_scaled - np.max(x_scaled)
                        exp_x = np.exp(x_scaled)
                        probs = exp_x / np.sum(exp_x)

                    # Store with integer indices (matching legal_moves order)
                    for idx, prob in enumerate(probs):
                        if prob > 1e-6:
                            policy_probs[str(idx)] = float(prob)
            except Exception:
                pass  # Fall through to one-hot encoding

        # If no policy distribution available, create one-hot
        if not policy_probs and legal_moves:
            idx = find_move_index(move, legal_moves)
            if idx >= 0:
                policy_probs[str(idx)] = 1.0

        moves_data.append(serialize_move(move, policy_probs if policy_probs else None))
        state, _, done, step_info = env.step(move)
        move_count += 1

        # Record auto-generated bookkeeping moves
        for auto_move in step_info.get("auto_generated_moves", []):
            moves_data.append(serialize_move(auto_move))
            move_count += 1

    board_size = 8 if env.board_type == BoardType.SQUARE8 else (19 if env.board_type == BoardType.SQUARE19 else 11)
    return {
        "game_id": f"policy_{env.board_type.value}_{env.num_players}p_{game_idx}_{int(time.time())}",
        "board_type": env.board_type.value,
        "board_size": board_size,
        "num_players": env.num_players,
        "winner": state.winner,
        "move_count": move_count,
        "game_status": state.game_status.value if hasattr(state.game_status, 'value') else str(state.game_status),
        "victory_type": getattr(state, 'victory_type', None),
        "engine_mode": "policy-only",
        "moves": moves_data,
        "initial_state": initial_state,
        "timestamp": datetime.now().isoformat(),
        "source": "generate_policy_selfplay.py",
    }


def main():
    import argparse
    import random
    import numpy as np

    parser = argparse.ArgumentParser(description="Generate PolicyOnlyAI selfplay games")
    parser.add_argument("--num-games", type=int, default=100, help="Number of games")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--num-players", type=int, default=2, help="Number of players")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--model-id", type=str, default=None, help="Model ID")
    parser.add_argument("--max-moves", type=int, default=500, help="Max moves per game")
    parser.add_argument("--temperature", type=float, default=1.0, help="Policy temperature")
    parser.add_argument("--seed", type=int, default=None, help="Base random seed (default: time-based)")
    args = parser.parse_args()

    # Initialize RNG with proper seeding
    if args.seed is None:
        base_seed = int(time.time() * 1000) % (2**31)
    else:
        base_seed = args.seed

    random.seed(base_seed)
    np.random.seed(base_seed)
    logger.info(f"Base random seed: {base_seed}")

    board_type = parse_board_type(args.board)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Generating {args.num_games} PolicyOnlyAI games")
    logger.info(f"Board: {board_type.value}, Players: {args.num_players}")
    logger.info(f"Model: {args.model_id or 'default'}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Output: {output_path}")

    env = RingRiftEnv(board_type=board_type, num_players=args.num_players)

    # Create AI players with unique RNG seeds per player
    ai_players = {}
    for pn in range(1, args.num_players + 1):
        player_seed = base_seed + pn * 1000  # Unique seed per player
        config = AIConfig(
            difficulty=5,
            self_play=True,
            nn_model_id=args.model_id,
            temperature=args.temperature,
            rngSeed=player_seed,
        )
        ai_players[pn] = PolicyOnlyAI(player_number=pn, config=config, board_type=board_type)

    start_time = time.time()
    games_generated = 0
    p1_wins = 0
    p2_wins = 0

    with open(output_path, 'w') as f:
        for game_idx in range(args.num_games):
            # Re-seed RNG for each game to ensure variety
            game_seed = base_seed + game_idx * 10000
            random.seed(game_seed)
            np.random.seed(game_seed)

            # Recreate AI players with fresh seeds per game
            for pn in range(1, args.num_players + 1):
                player_seed = game_seed + pn * 1000
                config = AIConfig(
                    difficulty=5,
                    self_play=True,
                    nn_model_id=args.model_id,
                    temperature=args.temperature,
                    rngSeed=player_seed,
                )
                ai_players[pn] = PolicyOnlyAI(player_number=pn, config=config, board_type=board_type)

            try:
                game = generate_game(env, ai_players, game_idx, args.max_moves)
                if game:
                    winner = game.get("winner")
                    if winner == 1:
                        p1_wins += 1
                    elif winner == 2:
                        p2_wins += 1

                    f.write(json.dumps(game) + "\n")
                    games_generated += 1

                    if (game_idx + 1) % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = games_generated / elapsed
                        logger.info(
                            f"Game {game_idx + 1}/{args.num_games} "
                            f"({rate:.2f}/s) | P1: {p1_wins} | P2: {p2_wins}"
                        )
            except Exception as e:
                logger.warning(f"Failed game {game_idx}: {e}")

    elapsed = time.time() - start_time
    logger.info("=" * 60)
    logger.info("GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Games: {games_generated}")
    if games_generated > 0:
        logger.info(f"P1 wins: {p1_wins} ({100*p1_wins/games_generated:.1f}%)")
        logger.info(f"P2 wins: {p2_wins} ({100*p2_wins/games_generated:.1f}%)")
        logger.info(f"Time: {elapsed:.1f}s ({games_generated/elapsed:.2f}/s)")
    else:
        logger.warning("No games generated!")
    logger.info(f"Output: {output_path}")


if __name__ == "__main__":
    main()
