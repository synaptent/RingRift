#!/usr/bin/env python
"""Benchmark NNUE policy head quality.

This script evaluates:
1. Move prediction accuracy (top-1, top-3, top-5)
2. Accuracy by game phase (early, mid, late)
3. Accuracy by move type
4. Impact on search efficiency (nodes visited with/without policy ordering)

Usage:
    # Benchmark on a database
    python scripts/benchmark_policy.py --db data/games/selfplay.db

    # Benchmark specific model
    python scripts/benchmark_policy.py --db data/games/*.db \\
        --model models/nnue/nnue_policy_square8_2p.pt

Output:
    - Console: Summary statistics
    - JSON report: {run_dir}/policy_benchmark_report.json
"""

from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Set up path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import sqlite3
import torch

from app.ai.nnue import extract_features_from_gamestate, get_board_size
from app.ai.nnue_policy import (
    RingRiftNNUEWithPolicy,
    pos_to_flat_index,
)
from app.models import BoardType, GameState, Move, GamePhase
from app.rules.default_engine import DefaultRulesEngine

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("benchmark_policy")


def parse_board_type(value: str) -> BoardType:
    """Parse board type string to enum."""
    mapping = {
        "square8": BoardType.SQUARE8,
        "sq8": BoardType.SQUARE8,
        "square19": BoardType.SQUARE19,
        "sq19": BoardType.SQUARE19,
        "hexagonal": BoardType.HEXAGONAL,
        "hex": BoardType.HEXAGONAL,
    }
    key = value.lower()
    if key not in mapping:
        raise ValueError(f"Unknown board type: {value}")
    return mapping[key]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Benchmark NNUE policy head quality",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db",
        type=str,
        nargs="+",
        required=True,
        help="Path(s) to SQLite game database(s)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to policy model (default: auto-detect)",
    )
    parser.add_argument(
        "--board-type",
        type=str,
        default="square8",
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=100,
        help="Maximum games to benchmark (default: 100)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5000,
        help="Maximum positions to evaluate (default: 5000)",
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Output directory for report",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    return parser.parse_args(argv)


def load_policy_model(
    model_path: Optional[str],
    board_type: BoardType,
    num_players: int,
) -> Optional[RingRiftNNUEWithPolicy]:
    """Load policy model from checkpoint."""
    if model_path is None:
        model_path = os.path.join(
            PROJECT_ROOT, "models", "nnue",
            f"nnue_policy_{board_type.value}_{num_players}p.pt"
        )

    if not os.path.exists(model_path):
        logger.warning(f"Policy model not found at {model_path}")
        return None

    try:
        # weights_only=False needed for our trusted checkpoints that may have numpy scalars
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        hidden_dim = checkpoint.get("hidden_dim", 256)
        num_hidden_layers = checkpoint.get("num_hidden_layers", 2)

        model = RingRiftNNUEWithPolicy(
            board_type=board_type,
            hidden_dim=hidden_dim,
            num_hidden_layers=num_hidden_layers,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        logger.info(f"Loaded policy model from {model_path}")
        return model
    except Exception as e:
        logger.error(f"Failed to load policy model: {e}")
        return None


def get_game_phase(state: GameState, move_number: int, total_moves: int) -> str:
    """Categorize game phase."""
    progress = move_number / max(total_moves, 1)
    if progress < 0.25:
        return "early"
    elif progress < 0.65:
        return "mid"
    else:
        return "late"


def evaluate_position(
    model: RingRiftNNUEWithPolicy,
    state: GameState,
    legal_moves: List[Move],
    played_move: Move,
    board_type: BoardType,
) -> Dict[str, Any]:
    """Evaluate model predictions for a single position.

    Returns:
        Dict with:
        - predicted_rank: Rank of the played move (0 = top prediction)
        - num_legal_moves: Total legal moves
        - top_1_correct: Whether played move was top prediction
        - top_3_correct: Whether played move was in top 3
        - top_5_correct: Whether played move was in top 5
    """
    board_size = get_board_size(board_type)
    current_player = state.current_player or 1

    # Extract features
    features = extract_features_from_gamestate(state, current_player)
    features_tensor = torch.from_numpy(features[None, ...]).float()

    # Get policy scores
    with torch.no_grad():
        _, from_logits, to_logits = model(features_tensor, return_policy=True)
        from_logits = from_logits[0].numpy()
        to_logits = to_logits[0].numpy()

    # Score all legal moves
    center = board_size // 2
    center_idx = center * board_size + center
    move_scores = []

    for move in legal_moves:
        from_pos = getattr(move, 'from_pos', None)
        if from_pos is None:
            from_idx = center_idx
        else:
            from_idx = pos_to_flat_index(from_pos, board_size, board_type)

        to_pos = getattr(move, 'to', None)
        if to_pos is None:
            to_pos = from_pos
        if to_pos is None:
            to_idx = center_idx
        else:
            to_idx = pos_to_flat_index(to_pos, board_size, board_type)

        score = from_logits[from_idx] + to_logits[to_idx]
        move_scores.append(score)

    # Find rank of played move
    played_idx = find_move_index(played_move, legal_moves)
    if played_idx < 0:
        return None  # Move not found in legal moves

    played_score = move_scores[played_idx]
    rank = sum(1 for s in move_scores if s > played_score)

    return {
        "predicted_rank": rank,
        "num_legal_moves": len(legal_moves),
        "top_1_correct": rank == 0,
        "top_3_correct": rank < 3,
        "top_5_correct": rank < 5,
    }


def find_move_index(played_move: Move, legal_moves: List[Move]) -> int:
    """Find index of played move in legal moves list."""
    played_type = getattr(played_move, 'type', None)
    played_from = getattr(played_move, 'from_pos', None)
    played_to = getattr(played_move, 'to', None)

    for i, legal in enumerate(legal_moves):
        legal_type = getattr(legal, 'type', None)
        legal_from = getattr(legal, 'from_pos', None)
        legal_to = getattr(legal, 'to', None)

        if played_type != legal_type:
            continue

        if played_from is not None and legal_from is not None:
            if played_from.x != legal_from.x or played_from.y != legal_from.y:
                continue
        elif played_from is not None or legal_from is not None:
            continue

        if played_to is not None and legal_to is not None:
            if played_to.x != legal_to.x or played_to.y != legal_to.y:
                continue
        elif played_to is not None or legal_to is not None:
            continue

        return i

    return -1


def benchmark_on_database(
    model: RingRiftNNUEWithPolicy,
    db_path: str,
    board_type: BoardType,
    num_players: int,
    max_games: int,
    max_positions: int,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """Benchmark policy model on a database."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # Get games
    board_type_str = board_type.value.lower()
    query = """
        SELECT game_id, winner, total_moves
        FROM games
        WHERE game_status = 'completed'
          AND winner IS NOT NULL
          AND board_type = ?
          AND num_players = ?
          AND total_moves >= 10
        ORDER BY RANDOM()
        LIMIT ?
    """
    cursor.execute(query, (board_type_str, num_players, max_games))
    games = cursor.fetchall()

    if not games:
        logger.warning(f"No suitable games found in {db_path}")
        conn.close()
        return {}

    engine = DefaultRulesEngine()
    results = {
        "total_positions": 0,
        "top_1_correct": 0,
        "top_3_correct": 0,
        "top_5_correct": 0,
        "ranks": [],
        "by_phase": defaultdict(lambda: {"total": 0, "top_1": 0, "top_3": 0}),
        "by_move_type": defaultdict(lambda: {"total": 0, "top_1": 0, "top_3": 0}),
    }

    positions_evaluated = 0

    for game_row in games:
        if positions_evaluated >= max_positions:
            break

        game_id = game_row['game_id']
        total_moves = game_row['total_moves']

        # Get initial state
        cursor.execute(
            "SELECT initial_state_json, compressed FROM game_initial_state WHERE game_id = ?",
            (game_id,)
        )
        initial_row = cursor.fetchone()
        if not initial_row:
            continue

        initial_json = initial_row['initial_state_json']
        if initial_row['compressed']:
            initial_json = gzip.decompress(initial_json.encode()).decode()

        try:
            state_dict = json.loads(initial_json)
            state = GameState(**state_dict)
        except Exception:
            continue

        # Get moves
        cursor.execute(
            "SELECT move_number, move_json FROM game_moves WHERE game_id = ? ORDER BY move_number",
            (game_id,)
        )
        moves = cursor.fetchall()

        if not moves:
            continue

        # Replay and evaluate
        for move_row in moves:
            if positions_evaluated >= max_positions:
                break

            move_number = move_row['move_number']
            move_json_str = move_row['move_json']

            # Sample ~50% of positions
            if rng.random() > 0.5:
                # Apply move and continue
                try:
                    move_dict = json.loads(move_json_str)
                    move = Move(**move_dict)
                    state = engine.apply_move(state, move)
                except Exception:
                    break
                continue

            current_player = state.current_player or 1

            try:
                legal_moves = engine.get_valid_moves(state, current_player)
                if not legal_moves or len(legal_moves) < 2:
                    # Skip trivial positions
                    move_dict = json.loads(move_json_str)
                    move = Move(**move_dict)
                    state = engine.apply_move(state, move)
                    continue

                move_dict = json.loads(move_json_str)
                played_move = Move(**move_dict)

                result = evaluate_position(
                    model, state, legal_moves, played_move, board_type
                )

                if result is not None:
                    positions_evaluated += 1
                    results["total_positions"] += 1
                    results["ranks"].append(result["predicted_rank"])

                    if result["top_1_correct"]:
                        results["top_1_correct"] += 1
                    if result["top_3_correct"]:
                        results["top_3_correct"] += 1
                    if result["top_5_correct"]:
                        results["top_5_correct"] += 1

                    # By phase
                    phase = get_game_phase(state, move_number, total_moves)
                    results["by_phase"][phase]["total"] += 1
                    if result["top_1_correct"]:
                        results["by_phase"][phase]["top_1"] += 1
                    if result["top_3_correct"]:
                        results["by_phase"][phase]["top_3"] += 1

                    # By move type
                    move_type = str(played_move.type.value if hasattr(played_move.type, 'value') else played_move.type)
                    results["by_move_type"][move_type]["total"] += 1
                    if result["top_1_correct"]:
                        results["by_move_type"][move_type]["top_1"] += 1
                    if result["top_3_correct"]:
                        results["by_move_type"][move_type]["top_3"] += 1

                # Apply move
                state = engine.apply_move(state, played_move)

            except Exception as e:
                logger.debug(f"Error evaluating position: {e}")
                try:
                    move_dict = json.loads(move_json_str)
                    move = Move(**move_dict)
                    state = engine.apply_move(state, move)
                except Exception:
                    break

    conn.close()
    return results


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    board_type = parse_board_type(args.board_type)
    rng = np.random.default_rng(args.seed)

    # Expand database paths
    db_paths: List[str] = []
    for pattern in args.db:
        expanded = glob.glob(pattern)
        if expanded:
            db_paths.extend(expanded)
        elif os.path.exists(pattern):
            db_paths.append(pattern)

    if not db_paths:
        logger.error("No database files found")
        return 1

    # Load model
    model = load_policy_model(args.model, board_type, args.num_players)
    if model is None:
        logger.error("No policy model available for benchmarking")
        return 1

    # Benchmark
    logger.info(f"Benchmarking on {len(db_paths)} database(s)...")

    all_results = {
        "total_positions": 0,
        "top_1_correct": 0,
        "top_3_correct": 0,
        "top_5_correct": 0,
        "ranks": [],
        "by_phase": defaultdict(lambda: {"total": 0, "top_1": 0, "top_3": 0}),
        "by_move_type": defaultdict(lambda: {"total": 0, "top_1": 0, "top_3": 0}),
    }

    for db_path in db_paths:
        logger.info(f"Processing {db_path}...")
        results = benchmark_on_database(
            model, db_path, board_type, args.num_players,
            args.max_games, args.max_positions, rng
        )

        # Merge results
        all_results["total_positions"] += results.get("total_positions", 0)
        all_results["top_1_correct"] += results.get("top_1_correct", 0)
        all_results["top_3_correct"] += results.get("top_3_correct", 0)
        all_results["top_5_correct"] += results.get("top_5_correct", 0)
        all_results["ranks"].extend(results.get("ranks", []))

        for phase, data in results.get("by_phase", {}).items():
            all_results["by_phase"][phase]["total"] += data["total"]
            all_results["by_phase"][phase]["top_1"] += data["top_1"]
            all_results["by_phase"][phase]["top_3"] += data["top_3"]

        for mt, data in results.get("by_move_type", {}).items():
            all_results["by_move_type"][mt]["total"] += data["total"]
            all_results["by_move_type"][mt]["top_1"] += data["top_1"]
            all_results["by_move_type"][mt]["top_3"] += data["top_3"]

    # Compute statistics
    total = all_results["total_positions"]
    if total == 0:
        logger.error("No positions evaluated")
        return 1

    top_1_acc = all_results["top_1_correct"] / total
    top_3_acc = all_results["top_3_correct"] / total
    top_5_acc = all_results["top_5_correct"] / total
    mean_rank = np.mean(all_results["ranks"])
    median_rank = np.median(all_results["ranks"])

    # Print summary
    print("\n" + "=" * 60)
    print("POLICY BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Total positions evaluated: {total}")
    print(f"\nOverall Accuracy:")
    print(f"  Top-1: {top_1_acc:.1%}")
    print(f"  Top-3: {top_3_acc:.1%}")
    print(f"  Top-5: {top_5_acc:.1%}")
    print(f"  Mean rank: {mean_rank:.2f}")
    print(f"  Median rank: {median_rank:.1f}")

    print(f"\nAccuracy by Game Phase:")
    for phase in ["early", "mid", "late"]:
        data = all_results["by_phase"].get(phase, {"total": 0, "top_1": 0, "top_3": 0})
        if data["total"] > 0:
            t1 = data["top_1"] / data["total"]
            t3 = data["top_3"] / data["total"]
            print(f"  {phase:6s}: Top-1={t1:.1%}, Top-3={t3:.1%} (n={data['total']})")

    print(f"\nAccuracy by Move Type (top 5):")
    sorted_types = sorted(
        all_results["by_move_type"].items(),
        key=lambda x: x[1]["total"],
        reverse=True
    )[:5]
    for mt, data in sorted_types:
        if data["total"] > 0:
            t1 = data["top_1"] / data["total"]
            print(f"  {mt[:25]:25s}: Top-1={t1:.1%} (n={data['total']})")

    print("=" * 60)

    # Save report
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = args.run_dir or os.path.join(PROJECT_ROOT, "runs", f"policy_benchmark_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    report = {
        "board_type": board_type.value,
        "num_players": args.num_players,
        "total_positions": total,
        "top_1_accuracy": top_1_acc,
        "top_3_accuracy": top_3_acc,
        "top_5_accuracy": top_5_acc,
        "mean_rank": float(mean_rank),
        "median_rank": float(median_rank),
        "by_phase": dict(all_results["by_phase"]),
        "by_move_type": dict(all_results["by_move_type"]),
        "db_paths": db_paths,
        "model_path": args.model,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    report_path = os.path.join(run_dir, "policy_benchmark_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved report to {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
