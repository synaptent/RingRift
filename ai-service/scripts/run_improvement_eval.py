#!/usr/bin/env python3
"""Run improvement evaluation between a new model and baseline model.

This script is designed to be called via SSH from the P2P orchestrator
for evaluating newly trained models against the current best model.

Usage:
    python scripts/run_improvement_eval.py \
        --new-model models/square8_2p_v42.pth \
        --baseline-model models/square8_2p_v41.pth \
        --board square8 \
        --players 2 \
        --games 50

Output (JSON to stdout):
    {
        "success": true,
        "new_model_wins": 28,
        "baseline_wins": 18,
        "draws": 4,
        "total_games": 50,
        "win_rate": 0.56,
        "duration_seconds": 120.5
    }
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))


def run_improvement_evaluation(
    new_model_path: str,
    baseline_model_path: str,
    board_type: str = "square8",
    num_players: int = 2,
    games: int = 50,
    ai_type: str = "descent",
    think_time_ms: int = 2000,
) -> dict[str, Any]:
    """Run head-to-head evaluation between two models.

    Args:
        new_model_path: Path to the new model being evaluated
        baseline_model_path: Path to the baseline/best model
        board_type: Board type (square8, square19, hex8, hexagonal)
        num_players: Number of players (2, 3, or 4)
        games: Number of games to play
        ai_type: AI type for neural network inference
        think_time_ms: Think time per move in milliseconds

    Returns:
        Dict with evaluation results
    """
    start_time = time.time()

    try:
        from app.ai.factory import AIFactory
        from app.models import AIConfig, AIType, BoardType, GameStatus
        from app.rules import MutableGameState
        from app.training.generate_data import create_initial_state

        # Map board type string to enum
        board_type_map = {
            "square8": BoardType.SQUARE8,
            "square19": BoardType.SQUARE19,
            "hex8": BoardType.HEX8,
            "hex": BoardType.HEXAGONAL,
            "hexagonal": BoardType.HEXAGONAL,
        }
        board_type_enum = board_type_map.get(board_type, BoardType.SQUARE8)

        # Map AI type string to enum
        ai_type_map = {
            "descent": AIType.DESCENT,
            "mcts": AIType.MCTS,
            "gumbel_mcts": AIType.GUMBEL_MCTS,
            "policy_only": AIType.POLICY_ONLY,
        }
        ai_type_enum = ai_type_map.get(ai_type, AIType.DESCENT)

        # Create AI configs for both models
        new_model_config = AIConfig(
            difficulty=5,  # Mid-level difficulty for evaluation
            ai_type=ai_type_enum,
            think_time=think_time_ms,
            randomness=0.0,
            nn_model_id=new_model_path,
        )

        baseline_config = AIConfig(
            difficulty=5,  # Mid-level difficulty for evaluation
            ai_type=ai_type_enum,
            think_time=think_time_ms,
            randomness=0.0,
            nn_model_id=baseline_model_path,
        )

        # Create AI instances (player 1 and 2)
        new_ai = AIFactory.create(ai_type_enum, 1, new_model_config)
        baseline_ai = AIFactory.create(ai_type_enum, 2, baseline_config)

        results = {
            "new_model_wins": 0,
            "baseline_wins": 0,
            "draws": 0,
        }

        for game_num in range(games):
            # Alternate who plays first
            if game_num % 2 == 0:
                player_ais = [new_ai, baseline_ai] + [baseline_ai] * (num_players - 2)
                new_is_player = 0
            else:
                player_ais = [baseline_ai, new_ai] + [baseline_ai] * (num_players - 2)
                new_is_player = 1

            # Create game state
            state = create_initial_state(board_type=board_type, num_players=num_players)
            mutable = MutableGameState.from_immutable(state)

            max_moves = 500
            move_count = 0

            while not mutable.is_game_over() and move_count < max_moves:
                current_player = mutable.current_player
                ai = player_ais[current_player]

                # Get AI move
                game_state = mutable.to_immutable()
                action = ai.get_action(game_state, current_player)

                if action is None:
                    break

                mutable.make_move(action)
                move_count += 1

            # Determine winner
            final_state = mutable.to_immutable()
            if final_state.status == GameStatus.COMPLETED:
                winner = final_state.winner_player_number
                if winner is not None:
                    winner_idx = winner - 1  # 1-indexed to 0-indexed
                    if winner_idx == new_is_player:
                        results["new_model_wins"] += 1
                    else:
                        results["baseline_wins"] += 1
                else:
                    results["draws"] += 1
            else:
                results["draws"] += 1

            # Progress update every 10 games
            if (game_num + 1) % 10 == 0:
                print(f"[ImprovementEval] Progress: {game_num + 1}/{games} games", file=sys.stderr)

        duration = time.time() - start_time
        total = results["new_model_wins"] + results["baseline_wins"] + results["draws"]

        return {
            "success": True,
            "new_model_wins": results["new_model_wins"],
            "baseline_wins": results["baseline_wins"],
            "draws": results["draws"],
            "total_games": total,
            "win_rate": results["new_model_wins"] / total if total > 0 else 0.5,
            "duration_seconds": round(duration, 2),
            "new_model_path": new_model_path,
            "baseline_model_path": baseline_model_path,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "duration_seconds": round(time.time() - start_time, 2),
        }


def main():
    parser = argparse.ArgumentParser(description="Run improvement evaluation between two models")
    parser.add_argument("--new-model", required=True, help="Path to new model")
    parser.add_argument("--baseline-model", required=True, help="Path to baseline model")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--games", type=int, default=50, help="Number of games")
    parser.add_argument("--ai-type", default="descent", help="AI type for inference")
    parser.add_argument("--think-time", type=int, default=2000, help="Think time in ms")

    args = parser.parse_args()

    # Validate model paths exist
    new_model_path = Path(args.new_model)
    baseline_model_path = Path(args.baseline_model)

    if not new_model_path.exists():
        result = {"success": False, "error": f"New model not found: {args.new_model}"}
        print(json.dumps(result))
        sys.exit(1)

    if not baseline_model_path.exists():
        result = {"success": False, "error": f"Baseline model not found: {args.baseline_model}"}
        print(json.dumps(result))
        sys.exit(1)

    print(f"[ImprovementEval] Evaluating {args.new_model} vs {args.baseline_model}", file=sys.stderr)
    print(f"[ImprovementEval] Board: {args.board}, Players: {args.players}, Games: {args.games}", file=sys.stderr)

    result = run_improvement_evaluation(
        new_model_path=str(new_model_path),
        baseline_model_path=str(baseline_model_path),
        board_type=args.board,
        num_players=args.players,
        games=args.games,
        ai_type=args.ai_type,
        think_time_ms=args.think_time,
    )

    # Output JSON to stdout for parsing
    print(json.dumps(result))

    sys.exit(0 if result.get("success") else 1)


if __name__ == "__main__":
    main()
