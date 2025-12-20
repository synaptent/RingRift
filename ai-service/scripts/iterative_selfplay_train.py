#!/usr/bin/env python3
"""Iterative self-play training loop.

Implements the AlphaZero-style training cycle:
1. Generate games using current best model with Gumbel MCTS
2. Export training data with soft policy targets
3. Train new model on accumulated data
4. Evaluate new model vs current best
5. If improved, update best model and repeat

Usage:
    python scripts/iterative_selfplay_train.py \
        --iterations 10 \
        --games-per-iter 1000 \
        --board square8 \
        --base-model distilled_sq8_2p_v3
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import subprocess
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
from app.ai.policy_only_ai import PolicyOnlyAI
from app.game_engine import GameEngine
from app.models import AIConfig, BoardType, GameState, Move, MoveType, Position
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


def generate_gumbel_game(
    env_state: GameState,
    ai_players: dict[int, GumbelMCTSAI],
    game_idx: int,
    board_type: BoardType,
    max_moves: int = 500,
) -> dict[str, Any] | None:
    """Generate a single game with Gumbel MCTS soft policy targets."""
    state = env_state
    initial_state = serialize_state(state)
    moves_data = []
    done = False
    move_count = 0
    engine = DefaultRulesEngine()
    game_engine = GameEngine()

    while not done and move_count < max_moves:
        status = state.game_status.value
        if status not in ['active', 'in_progress']:
            break

        current_player = state.current_player
        ai = ai_players.get(current_player)
        if ai is None:
            break

        legal_moves = engine.get_valid_moves(state, current_player)
        if not legal_moves:
            break

        # Get move and MCTS policy distribution
        move = ai.select_move(state)
        if move is None:
            break

        # Extract soft policy from Gumbel MCTS search
        mcts_policy = {}
        if hasattr(ai, '_last_search_actions') and ai._last_search_actions:
            total_visits = sum(a.visit_count for a in ai._last_search_actions)
            if total_visits > 0:
                for i, action in enumerate(ai._last_search_actions):
                    if action.visit_count > 0:
                        prob = action.visit_count / total_visits
                        if prob > 1e-6:
                            mcts_policy[str(i)] = float(prob)

        # Fallback to one-hot if no policy
        if not mcts_policy:
            for i, legal in enumerate(legal_moves):
                if (move.type == legal.type and
                    getattr(move, 'from_pos', None) == getattr(legal, 'from_pos', None) and
                    getattr(move, 'to', None) == getattr(legal, 'to', None)):
                    mcts_policy[str(i)] = 1.0
                    break

        moves_data.append(serialize_move(move, mcts_policy if mcts_policy else None))
        state = game_engine.apply_move(state, move)
        move_count += 1

    board_size = 8 if board_type == BoardType.SQUARE8 else 19

    return {
        "game_id": f"gumbel_{board_type.value}_2p_{game_idx}_{int(time.time())}",
        "board_type": board_type.value,
        "board_size": board_size,
        "num_players": 2,
        "winner": state.winner,
        "move_count": move_count,
        "game_status": state.game_status.value if hasattr(state.game_status, 'value') else str(state.game_status),
        "victory_type": getattr(state, 'victory_type', None),
        "engine_mode": "gumbel_mcts",
        "moves": moves_data,
        "initial_state": initial_state,
        "timestamp": datetime.now().isoformat(),
        "source": "iterative_selfplay_train.py",
    }


def generate_games(
    model_id: str,
    board_type: BoardType,
    num_games: int,
    output_path: Path,
    simulations: int = 100,
) -> dict[str, int]:
    """Generate selfplay games using Gumbel MCTS."""
    logger.info(f"Generating {num_games} games with {model_id}...")

    p1_wins = 0
    p2_wins = 0
    games_generated = 0

    with open(output_path, 'w') as f:
        for game_idx in range(num_games):
            seed = int(time.time() * 1000) % (2**31) + game_idx * 10000
            random.seed(seed)
            np.random.seed(seed)

            # Create AI players with Gumbel MCTS
            ai_players = {}
            for pn in range(1, 3):
                player_seed = seed + pn * 1000
                config = AIConfig(
                    difficulty=5,
                    self_play=True,
                    nn_model_id=model_id,
                    think_time=500,
                    gumbel_simulation_budget=simulations,
                    rngSeed=player_seed,
                )
                ai_players[pn] = GumbelMCTSAI(
                    player_number=pn, config=config, board_type=board_type
                )

            state = create_initial_state(board_type, 2)

            try:
                game = generate_gumbel_game(state, ai_players, game_idx, board_type)
                if game:
                    winner = game.get("winner")
                    if winner == 1:
                        p1_wins += 1
                    elif winner == 2:
                        p2_wins += 1

                    f.write(json.dumps(game) + "\n")
                    games_generated += 1

                    if (game_idx + 1) % 50 == 0:
                        logger.info(
                            f"Game {game_idx + 1}/{num_games} | "
                            f"P1: {p1_wins} ({100*p1_wins/games_generated:.1f}%) | "
                            f"P2: {p2_wins} ({100*p2_wins/games_generated:.1f}%)"
                        )
            except Exception as e:
                logger.warning(f"Failed game {game_idx}: {e}")

    logger.info(f"Generated {games_generated} games -> {output_path}")
    return {"p1_wins": p1_wins, "p2_wins": p2_wins, "total": games_generated}


def export_to_npz(
    jsonl_path: Path,
    npz_path: Path,
    board_type: str,
) -> int:
    """Export JSONL to NPZ format."""
    logger.info(f"Exporting {jsonl_path} -> {npz_path}...")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "export_gumbel_kl_dataset.py"),
        "--input", str(jsonl_path),
        "--output", str(npz_path),
        "--board-type", board_type,
        "--num-players", "2",
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Export failed: {result.stderr}")
        return 0

    # Parse sample count from output
    for line in result.stdout.split('\n'):
        if 'Samples created:' in line:
            return int(line.split(':')[1].strip())
    return 0


def train_model(
    data_path: Path,
    output_path: Path,
    model_id: str,
    board_type: str,
    epochs: int = 30,
) -> bool:
    """Train a new model on the data."""
    logger.info(f"Training {model_id} on {data_path}...")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "train_distilled_model.py"),
        "--data", str(data_path),
        "--output", str(output_path),
        "--board-type", board_type,
        "--num-players", "2",
        "--epochs", str(epochs),
        "--batch-size", "256",
        "--learning-rate", "0.002",
        "--model-id", model_id,
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Training failed: {result.stderr}")
        return False

    logger.info(f"Training complete -> {output_path}")
    return True


def evaluate_model(
    model_a: str,
    model_b: str,
    board_type: BoardType,
    num_games: int = 20,
) -> tuple[int, int]:
    """Evaluate model A vs model B, returns (a_wins, b_wins)."""
    logger.info(f"Evaluating {model_a} vs {model_b}...")

    a_wins = 0
    b_wins = 0

    for i in range(num_games):
        seed = 42 + i * 100
        random.seed(seed)
        np.random.seed(seed)

        # Alternate colors
        if i % 2 == 0:
            p1_model, p2_model = model_a, model_b
        else:
            p1_model, p2_model = model_b, model_a

        p1_config = AIConfig(
            difficulty=5, nn_model_id=p1_model, temperature=0.5, rngSeed=seed
        )
        p2_config = AIConfig(
            difficulty=5, nn_model_id=p2_model, temperature=0.5, rngSeed=seed + 1
        )

        p1_ai = PolicyOnlyAI(player_number=1, config=p1_config, board_type=board_type)
        p2_ai = PolicyOnlyAI(player_number=2, config=p2_config, board_type=board_type)

        state = create_initial_state(board_type, 2)
        engine = GameEngine()

        for _ in range(300):
            status = state.game_status.value
            if status not in ['active', 'in_progress']:
                break
            current = state.current_player
            ai = p1_ai if current == 1 else p2_ai
            move = ai.select_move(state)
            if move is None:
                break
            state = engine.apply_move(state, move)

        winner = state.winner
        if i % 2 == 0:  # A was P1
            if winner == 1:
                a_wins += 1
            elif winner == 2:
                b_wins += 1
        else:  # A was P2
            if winner == 2:
                a_wins += 1
            elif winner == 1:
                b_wins += 1

    logger.info(f"Evaluation: {model_a}={a_wins}, {model_b}={b_wins}")
    return a_wins, b_wins


def main():
    parser = argparse.ArgumentParser(description="Iterative self-play training")
    parser.add_argument("--iterations", type=int, default=5, help="Training iterations")
    parser.add_argument("--games-per-iter", type=int, default=500, help="Games per iteration")
    parser.add_argument("--board", type=str, default="square8", help="Board type")
    parser.add_argument("--base-model", type=str, default="distilled_sq8_2p_v3", help="Starting model")
    parser.add_argument("--simulations", type=int, default=100, help="MCTS simulations per move")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs per iteration")
    parser.add_argument("--eval-games", type=int, default=20, help="Evaluation games")
    parser.add_argument("--output-dir", type=str, default="models/iterative", help="Output directory")
    args = parser.parse_args()

    board_type = BoardType.SQUARE8 if "square8" in args.board.lower() else BoardType.SQUARE19
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path("data/iterative")
    data_dir.mkdir(parents=True, exist_ok=True)

    current_best = args.base_model
    all_data_files = []

    logger.info("=" * 60)
    logger.info("ITERATIVE SELF-PLAY TRAINING")
    logger.info("=" * 60)
    logger.info(f"Starting model: {current_best}")
    logger.info(f"Iterations: {args.iterations}")
    logger.info(f"Games per iteration: {args.games_per_iter}")
    logger.info(f"MCTS simulations: {args.simulations}")
    logger.info("=" * 60)

    for iteration in range(1, args.iterations + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"ITERATION {iteration}/{args.iterations}")
        logger.info(f"{'='*60}")

        # 1. Generate games with current best
        games_path = data_dir / f"iter_{iteration:03d}_games.jsonl"
        stats = generate_games(
            model_id=current_best,
            board_type=board_type,
            num_games=args.games_per_iter,
            output_path=games_path,
            simulations=args.simulations,
        )

        # 2. Export to NPZ
        npz_path = data_dir / f"iter_{iteration:03d}_data.npz"
        samples = export_to_npz(games_path, npz_path, args.board)
        if samples == 0:
            logger.error(f"No samples exported for iteration {iteration}")
            continue

        all_data_files.append(npz_path)
        logger.info(f"Accumulated {len(all_data_files)} data files, {samples} new samples")

        # 3. Train new model (on latest data for now)
        new_model_id = f"iter_{iteration:03d}"
        new_model_path = output_dir / f"{new_model_id}.pth"

        success = train_model(
            data_path=npz_path,
            output_path=new_model_path,
            model_id=new_model_id,
            board_type=args.board,
            epochs=args.epochs,
        )

        if not success:
            logger.error(f"Training failed for iteration {iteration}")
            continue

        # 4. Evaluate new model vs current best
        new_wins, old_wins = evaluate_model(
            model_a=new_model_id,
            model_b=current_best,
            board_type=board_type,
            num_games=args.eval_games,
        )

        win_rate = new_wins / (new_wins + old_wins) if (new_wins + old_wins) > 0 else 0.5

        logger.info(f"New model win rate: {win_rate*100:.1f}%")

        # 5. Update best model if improved
        if new_wins > old_wins:
            logger.info(f"NEW BEST MODEL: {new_model_id} ({new_wins}/{new_wins+old_wins})")
            current_best = new_model_id

            # Copy to best model location
            best_path = output_dir / "best.pth"
            shutil.copy(new_model_path, best_path)
        else:
            logger.info(f"Keeping {current_best} as best ({old_wins}/{new_wins+old_wins})")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Final best model: {current_best}")
    logger.info(f"Total iterations: {args.iterations}")
    logger.info(f"Data files: {len(all_data_files)}")


if __name__ == "__main__":
    main()
