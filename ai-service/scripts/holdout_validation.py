#!/usr/bin/env python3
"""Holdout validation set management for NNUE model generalization testing.

This script manages a holdout set of games that are NEVER used for training.
It allows evaluating models on unseen data to detect overfitting.

Features:
1. Reserve a percentage of new games for holdout
2. Evaluate NNUE models on holdout set
3. Track generalization metrics over time
4. Detect model overfitting (train/holdout gap)

Usage:
    # Reserve 10% of recent games for holdout
    python scripts/holdout_validation.py --reserve --percent 10

    # Evaluate a model on holdout set
    python scripts/holdout_validation.py --evaluate --model models/square8_2p.pt

    # Show holdout statistics
    python scripts/holdout_validation.py --stats

    # Check for overfitting across all models
    python scripts/holdout_validation.py --check-overfitting
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
SCRIPT_DIR = Path(__file__).parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(AI_SERVICE_ROOT))

# Configuration
HOLDOUT_DB_PATH = AI_SERVICE_ROOT / "data" / "holdouts" / "holdout_validation.db"
GAMES_DIR = AI_SERVICE_ROOT / "data" / "games"
SELFPLAY_DIR = AI_SERVICE_ROOT / "data" / "selfplay"
MODELS_DIR = AI_SERVICE_ROOT / "models"

# Holdout configuration
DEFAULT_HOLDOUT_PERCENT = 10  # 10% of games reserved for holdout
HOLDOUT_SEED = 42  # Deterministic selection based on game_id hash
OVERFIT_THRESHOLD = 0.15  # Alert if holdout loss > train loss by this much

# Stratification by game phase
PHASE_OPENING_MAX_MOVES = 10  # First N moves are "opening"
PHASE_MIDGAME_MAX_MOVES = 40  # Moves N to M are "midgame"
# Beyond midgame = "endgame"

GAME_PHASES = ["opening", "midgame", "endgame"]


@dataclass
class HoldoutGame:
    """A game reserved for holdout validation."""
    game_id: str
    board_type: str
    num_players: int
    source_file: str
    reserved_at: str
    num_positions: int


@dataclass
class EvaluationResult:
    """Result of evaluating a model on holdout set."""
    model_path: str
    board_type: str
    num_players: int
    holdout_loss: float
    holdout_accuracy: float
    train_loss: float  # From training metrics if available
    num_samples: int
    evaluated_at: str
    overfit_gap: float  # holdout_loss - train_loss


@dataclass
class StratifiedEvaluation:
    """Stratified evaluation result by game phase."""
    model_path: str
    board_type: str
    num_players: int
    phase: str  # "opening", "midgame", "endgame"
    loss: float
    accuracy: float
    num_samples: int
    evaluated_at: str


def classify_move_phase(move_number: int, total_moves: int) -> str:
    """Classify which game phase a move belongs to.

    Args:
        move_number: The move number (0-indexed)
        total_moves: Total moves in the game

    Returns:
        "opening", "midgame", or "endgame"
    """
    if move_number < PHASE_OPENING_MAX_MOVES:
        return "opening"
    elif move_number < PHASE_MIDGAME_MAX_MOVES:
        return "midgame"
    else:
        return "endgame"


def get_db_connection() -> sqlite3.Connection:
    """Get connection to holdout database, creating if needed."""
    HOLDOUT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(HOLDOUT_DB_PATH)
    conn.row_factory = sqlite3.Row

    # Create tables if needed
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS holdout_games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            source_file TEXT NOT NULL,
            reserved_at TEXT NOT NULL,
            num_positions INTEGER DEFAULT 0,
            game_data TEXT  -- Full game JSON for position extraction
        );

        CREATE INDEX IF NOT EXISTS idx_holdout_board ON holdout_games(board_type, num_players);

        CREATE TABLE IF NOT EXISTS evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_path TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            holdout_loss REAL NOT NULL,
            holdout_accuracy REAL NOT NULL,
            train_loss REAL DEFAULT 0,
            num_samples INTEGER NOT NULL,
            evaluated_at TEXT NOT NULL,
            overfit_gap REAL DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_eval_model ON evaluations(model_path, board_type, num_players);

        CREATE TABLE IF NOT EXISTS metrics_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_path TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            metric_type TEXT NOT NULL,
            value REAL NOT NULL,
            recorded_at TEXT NOT NULL
        );

        -- Stratified evaluations by game phase
        CREATE TABLE IF NOT EXISTS stratified_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_path TEXT NOT NULL,
            board_type TEXT NOT NULL,
            num_players INTEGER NOT NULL,
            phase TEXT NOT NULL,  -- opening, midgame, endgame
            loss REAL NOT NULL,
            accuracy REAL NOT NULL,
            num_samples INTEGER NOT NULL,
            evaluated_at TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS idx_strat_eval_model ON stratified_evaluations(model_path, board_type, num_players, phase);
    """)
    conn.commit()
    return conn


def is_holdout_game(game_id: str, holdout_percent: int = DEFAULT_HOLDOUT_PERCENT) -> bool:
    """Deterministically decide if a game should be in holdout set.

    Uses hash of game_id to make selection reproducible.
    """
    hash_val = int(hashlib.md5(f"{game_id}:{HOLDOUT_SEED}".encode()).hexdigest(), 16)
    return (hash_val % 100) < holdout_percent


def reserve_games_from_db(
    db_path: Path,
    holdout_percent: int = DEFAULT_HOLDOUT_PERCENT,
    dry_run: bool = False,
) -> List[HoldoutGame]:
    """Reserve games from a SQLite game database for holdout.

    Supports both legacy schema (game_state column) and v7 schema
    (separate game_moves and game_initial_state tables).
    """
    if not db_path.exists():
        return []

    reserved = []
    conn = get_db_connection()

    try:
        game_conn = sqlite3.connect(db_path)
        game_conn.row_factory = sqlite3.Row

        # Check which schema we have
        cursor = game_conn.execute("PRAGMA table_info(games)")
        columns = {col[1] for col in cursor.fetchall()}
        has_game_state = "game_state" in columns
        has_total_moves = "total_moves" in columns

        # Check for v7 schema tables
        cursor = game_conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        has_game_moves = "game_moves" in tables
        has_initial_state = "game_initial_state" in tables

        # Query games based on schema
        try:
            if has_game_state:
                # Legacy schema
                cursor = game_conn.execute("""
                    SELECT game_id, board_type, num_players, game_state
                    FROM games
                """)
            else:
                # v7 schema - just get metadata, we'll fetch moves separately
                cursor = game_conn.execute("""
                    SELECT game_id, board_type, num_players, total_moves
                    FROM games
                """)
        except sqlite3.OperationalError:
            game_conn.close()
            return []

        # Get existing holdout game IDs
        existing = set(
            row[0] for row in
            conn.execute("SELECT game_id FROM holdout_games").fetchall()
        )

        for row in cursor:
            game_id = row["game_id"]

            # Skip if already in holdout
            if game_id in existing:
                continue

            # Check if should be holdout
            if not is_holdout_game(game_id, holdout_percent):
                continue

            # Get position count and game data based on schema
            num_positions = 0
            game_data = None

            if has_game_state:
                # Legacy schema
                game_data = row["game_state"]
                if game_data:
                    try:
                        state = json.loads(game_data)
                        num_positions = len(state.get("moveHistory", []))
                    except json.JSONDecodeError:
                        pass
            elif has_total_moves:
                # v7 schema - use total_moves from games table
                num_positions = row["total_moves"] or 0

                # Reconstruct game_data from v7 tables for evaluation later
                if has_game_moves and has_initial_state:
                    try:
                        # Get initial state
                        init_row = game_conn.execute(
                            "SELECT state_json FROM game_initial_state WHERE game_id = ?",
                            (game_id,)
                        ).fetchone()

                        # Get moves
                        moves_cursor = game_conn.execute(
                            "SELECT move_json FROM game_moves WHERE game_id = ? ORDER BY move_index",
                            (game_id,)
                        )
                        moves = [json.loads(m[0]) for m in moves_cursor]

                        # Build game_data structure
                        game_data = json.dumps({
                            "initialState": json.loads(init_row[0]) if init_row else {},
                            "moveHistory": moves,
                            "board_type": row["board_type"],
                            "num_players": row["num_players"],
                        })
                    except Exception:
                        # If reconstruction fails, still reserve but without game_data
                        pass

            holdout = HoldoutGame(
                game_id=game_id,
                board_type=row["board_type"],
                num_players=row["num_players"],
                source_file=str(db_path),
                reserved_at=datetime.utcnow().isoformat(),
                num_positions=num_positions,
            )

            if not dry_run:
                conn.execute("""
                    INSERT OR IGNORE INTO holdout_games
                    (game_id, board_type, num_players, source_file, reserved_at, num_positions, game_data)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    holdout.game_id,
                    holdout.board_type,
                    holdout.num_players,
                    holdout.source_file,
                    holdout.reserved_at,
                    holdout.num_positions,
                    game_data,
                ))

            reserved.append(holdout)

        game_conn.close()

        if not dry_run:
            conn.commit()

    finally:
        conn.close()

    return reserved


def reserve_games_from_all_sources(
    holdout_percent: int = DEFAULT_HOLDOUT_PERCENT,
    dry_run: bool = False,
) -> Dict[str, List[HoldoutGame]]:
    """Reserve games from all available sources."""
    results = {}

    # Process all SQLite game databases
    for db_file in GAMES_DIR.glob("*.db"):
        reserved = reserve_games_from_db(db_file, holdout_percent, dry_run)
        if reserved:
            results[str(db_file)] = reserved
            print(f"  {db_file.name}: {len(reserved)} games reserved")

    return results


def get_holdout_stats() -> Dict[str, Any]:
    """Get statistics about the holdout set."""
    conn = get_db_connection()

    stats = {
        "total_games": 0,
        "total_positions": 0,
        "by_config": {},
        "recent_evaluations": [],
    }

    try:
        # Total counts
        row = conn.execute("""
            SELECT COUNT(*), COALESCE(SUM(num_positions), 0)
            FROM holdout_games
        """).fetchone()
        stats["total_games"] = row[0]
        stats["total_positions"] = row[1]

        # By config
        rows = conn.execute("""
            SELECT board_type, num_players, COUNT(*), COALESCE(SUM(num_positions), 0)
            FROM holdout_games
            GROUP BY board_type, num_players
        """).fetchall()

        for row in rows:
            key = f"{row[0]}_{row[1]}p"
            stats["by_config"][key] = {
                "games": row[2],
                "positions": row[3],
            }

        # Recent evaluations
        rows = conn.execute("""
            SELECT model_path, board_type, num_players, holdout_loss,
                   holdout_accuracy, train_loss, overfit_gap, evaluated_at
            FROM evaluations
            ORDER BY evaluated_at DESC
            LIMIT 10
        """).fetchall()

        for row in rows:
            stats["recent_evaluations"].append({
                "model": Path(row[0]).name if row[0] else "unknown",
                "config": f"{row[1]}_{row[2]}p",
                "holdout_loss": row[3],
                "holdout_accuracy": row[4],
                "train_loss": row[5],
                "overfit_gap": row[6],
                "evaluated_at": row[7],
            })

    finally:
        conn.close()

    return stats


def extract_positions_for_config(
    board_type: str,
    num_players: int,
    max_positions: int = 10000,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract feature vectors and values from holdout games.

    Returns (features, values) numpy arrays ready for model evaluation.
    """
    conn = get_db_connection()

    try:
        # Import feature extraction
        from app.training.nnue_dataset import extract_features_from_state

        features_list = []
        values_list = []

        rows = conn.execute("""
            SELECT game_data FROM holdout_games
            WHERE board_type = ? AND num_players = ?
            ORDER BY reserved_at DESC
        """, (board_type, num_players)).fetchall()

        for row in rows:
            if len(features_list) >= max_positions:
                break

            game_data = row[0]
            if not game_data:
                continue

            try:
                state = json.loads(game_data)
                move_history = state.get("moveHistory", [])

                # Sample positions from the game
                for i, move in enumerate(move_history):
                    if len(features_list) >= max_positions:
                        break

                    # Get board state at this position
                    position_state = move.get("state_before", move.get("boardState"))
                    if not position_state:
                        continue

                    # Extract features
                    features = extract_features_from_state(
                        position_state,
                        board_type,
                        num_players,
                        current_player=move.get("player", 0),
                    )

                    # Get value from game outcome
                    winner = state.get("winner")
                    current_player = move.get("player", 0)
                    if winner is not None:
                        value = 1.0 if winner == current_player else -1.0
                    else:
                        value = 0.0

                    features_list.append(features)
                    values_list.append(value)

            except Exception:
                continue

        if not features_list:
            return np.array([]), np.array([])

        return np.array(features_list), np.array(values_list)

    finally:
        conn.close()


def evaluate_model_on_holdout(
    model_path: Path,
    board_type: str,
    num_players: int,
    train_loss: float = 0.0,
) -> Optional[EvaluationResult]:
    """Evaluate a model on the holdout set."""
    import torch

    # Extract holdout positions
    features, values = extract_positions_for_config(board_type, num_players)

    if len(features) == 0:
        print(f"No holdout positions found for {board_type}_{num_players}p")
        return None

    # Load model
    try:
        from app.training.nnue_model import RingRiftNNUE

        model = RingRiftNNUE(board_type=board_type)
        state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
    except Exception as e:
        print(f"Error loading model {model_path}: {e}")
        return None

    # Evaluate
    with torch.no_grad():
        features_tensor = torch.from_numpy(features).float()
        values_tensor = torch.from_numpy(values).float().unsqueeze(1)

        predictions = model(features_tensor)

        # MSE loss
        loss = torch.nn.functional.mse_loss(predictions, values_tensor).item()

        # Accuracy (correct sign prediction)
        correct = ((predictions > 0) == (values_tensor > 0)).float().mean().item()

    overfit_gap = loss - train_loss if train_loss > 0 else 0.0

    result = EvaluationResult(
        model_path=str(model_path),
        board_type=board_type,
        num_players=num_players,
        holdout_loss=loss,
        holdout_accuracy=correct,
        train_loss=train_loss,
        num_samples=len(features),
        evaluated_at=datetime.utcnow().isoformat(),
        overfit_gap=overfit_gap,
    )

    # Save to database
    conn = get_db_connection()
    try:
        conn.execute("""
            INSERT INTO evaluations
            (model_path, board_type, num_players, holdout_loss, holdout_accuracy,
             train_loss, num_samples, evaluated_at, overfit_gap)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            result.model_path,
            result.board_type,
            result.num_players,
            result.holdout_loss,
            result.holdout_accuracy,
            result.train_loss,
            result.num_samples,
            result.evaluated_at,
            result.overfit_gap,
        ))
        conn.commit()
    finally:
        conn.close()

    return result


def evaluate_model_stratified(
    model_path: Path,
    board_type: str,
    num_players: int,
    max_positions_per_phase: int = 3000,
) -> Dict[str, StratifiedEvaluation]:
    """Evaluate a model on holdout set with stratification by game phase.

    Returns dict mapping phase -> StratifiedEvaluation.
    """
    import torch

    conn = get_db_connection()
    results = {}

    try:
        from app.training.nnue_dataset import extract_features_from_state

        # Organize positions by phase
        phase_features: Dict[str, List] = {phase: [] for phase in GAME_PHASES}
        phase_values: Dict[str, List] = {phase: [] for phase in GAME_PHASES}

        rows = conn.execute("""
            SELECT game_data FROM holdout_games
            WHERE board_type = ? AND num_players = ?
            ORDER BY reserved_at DESC
        """, (board_type, num_players)).fetchall()

        for row in rows:
            # Check if we have enough for all phases
            if all(len(phase_features[p]) >= max_positions_per_phase for p in GAME_PHASES):
                break

            game_data = row[0]
            if not game_data:
                continue

            try:
                state = json.loads(game_data)
                move_history = state.get("moveHistory", [])
                total_moves = len(move_history)

                for i, move in enumerate(move_history):
                    phase = classify_move_phase(i, total_moves)

                    if len(phase_features[phase]) >= max_positions_per_phase:
                        continue

                    position_state = move.get("state_before", move.get("boardState"))
                    if not position_state:
                        continue

                    features = extract_features_from_state(
                        position_state,
                        board_type,
                        num_players,
                        current_player=move.get("player", 0),
                    )

                    winner = state.get("winner")
                    current_player = move.get("player", 0)
                    if winner is not None:
                        value = 1.0 if winner == current_player else -1.0
                    else:
                        value = 0.0

                    phase_features[phase].append(features)
                    phase_values[phase].append(value)

            except Exception:
                continue

        # Load model
        try:
            from app.training.nnue_model import RingRiftNNUE

            model = RingRiftNNUE(board_type=board_type)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
        except Exception as e:
            print(f"Error loading model {model_path}: {e}")
            return {}

        # Evaluate each phase
        for phase in GAME_PHASES:
            features = phase_features[phase]
            values = phase_values[phase]

            if not features:
                continue

            with torch.no_grad():
                features_tensor = torch.from_numpy(np.array(features)).float()
                values_tensor = torch.from_numpy(np.array(values)).float().unsqueeze(1)

                predictions = model(features_tensor)

                loss = torch.nn.functional.mse_loss(predictions, values_tensor).item()
                correct = ((predictions > 0) == (values_tensor > 0)).float().mean().item()

            evaluated_at = datetime.utcnow().isoformat()

            result = StratifiedEvaluation(
                model_path=str(model_path),
                board_type=board_type,
                num_players=num_players,
                phase=phase,
                loss=loss,
                accuracy=correct,
                num_samples=len(features),
                evaluated_at=evaluated_at,
            )
            results[phase] = result

            # Save to database
            conn.execute("""
                INSERT INTO stratified_evaluations
                (model_path, board_type, num_players, phase, loss, accuracy, num_samples, evaluated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                result.model_path,
                result.board_type,
                result.num_players,
                result.phase,
                result.loss,
                result.accuracy,
                result.num_samples,
                result.evaluated_at,
            ))

        conn.commit()

    finally:
        conn.close()

    return results


def get_stratified_summary(board_type: str, num_players: int) -> Dict[str, Any]:
    """Get summary of stratified evaluation results for a config."""
    conn = get_db_connection()

    try:
        summary = {
            "board_type": board_type,
            "num_players": num_players,
            "phases": {},
            "weakest_phase": None,
            "strongest_phase": None,
        }

        for phase in GAME_PHASES:
            row = conn.execute("""
                SELECT loss, accuracy, num_samples, evaluated_at
                FROM stratified_evaluations
                WHERE board_type = ? AND num_players = ? AND phase = ?
                ORDER BY evaluated_at DESC
                LIMIT 1
            """, (board_type, num_players, phase)).fetchone()

            if row:
                summary["phases"][phase] = {
                    "loss": row[0],
                    "accuracy": row[1],
                    "num_samples": row[2],
                    "evaluated_at": row[3],
                }

        # Determine weakest/strongest
        if summary["phases"]:
            by_accuracy = sorted(summary["phases"].items(), key=lambda x: x[1]["accuracy"])
            summary["weakest_phase"] = by_accuracy[0][0]
            summary["strongest_phase"] = by_accuracy[-1][0]

        return summary

    finally:
        conn.close()


def check_for_overfitting(threshold: float = OVERFIT_THRESHOLD) -> List[Dict[str, Any]]:
    """Check all recent evaluations for signs of overfitting."""
    conn = get_db_connection()
    overfit_models = []

    try:
        # Get most recent evaluation for each model/config
        rows = conn.execute("""
            SELECT model_path, board_type, num_players, holdout_loss,
                   train_loss, overfit_gap, evaluated_at
            FROM evaluations
            WHERE train_loss > 0
            ORDER BY evaluated_at DESC
        """).fetchall()

        seen = set()
        for row in rows:
            key = (row[0], row[1], row[2])
            if key in seen:
                continue
            seen.add(key)

            if row[5] > threshold:  # overfit_gap
                overfit_models.append({
                    "model": Path(row[0]).name,
                    "config": f"{row[1]}_{row[2]}p",
                    "holdout_loss": row[3],
                    "train_loss": row[4],
                    "gap": row[5],
                    "evaluated_at": row[6],
                })

    finally:
        conn.close()

    return overfit_models


def print_stats():
    """Print holdout set statistics."""
    stats = get_holdout_stats()

    print("\n" + "=" * 60)
    print("Holdout Validation Set Statistics")
    print("=" * 60)

    print(f"\nTotal holdout games: {stats['total_games']}")
    print(f"Total positions: {stats['total_positions']}")

    if stats["by_config"]:
        print("\nBy configuration:")
        for config, data in sorted(stats["by_config"].items()):
            print(f"  {config}: {data['games']} games, {data['positions']} positions")

    if stats["recent_evaluations"]:
        print("\nRecent evaluations:")
        for eval_data in stats["recent_evaluations"]:
            gap_str = ""
            if eval_data["train_loss"] > 0:
                gap_str = f", gap={eval_data['overfit_gap']:.4f}"
            print(
                f"  {eval_data['model']} ({eval_data['config']}): "
                f"loss={eval_data['holdout_loss']:.4f}, "
                f"acc={eval_data['holdout_accuracy']:.4f}"
                f"{gap_str}"
            )


def main():
    parser = argparse.ArgumentParser(description="Holdout validation management")
    parser.add_argument("--reserve", action="store_true", help="Reserve games for holdout")
    parser.add_argument("--percent", type=int, default=DEFAULT_HOLDOUT_PERCENT, help="Percentage to reserve")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate model on holdout")
    parser.add_argument("--evaluate-stratified", action="store_true", help="Evaluate model with phase stratification")
    parser.add_argument("--model", type=str, help="Model path to evaluate")
    parser.add_argument("--board", type=str, help="Board type")
    parser.add_argument("--players", type=int, help="Number of players")
    parser.add_argument("--train-loss", type=float, default=0.0, help="Training loss for comparison")
    parser.add_argument("--stats", action="store_true", help="Show holdout statistics")
    parser.add_argument("--stratified-summary", action="store_true", help="Show stratified evaluation summary")
    parser.add_argument("--check-overfitting", action="store_true", help="Check for overfitting")
    parser.add_argument("--dry-run", action="store_true", help="Don't actually reserve games")

    args = parser.parse_args()

    if args.reserve:
        print(f"Reserving {args.percent}% of games for holdout set...")
        results = reserve_games_from_all_sources(args.percent, args.dry_run)
        total = sum(len(games) for games in results.values())
        print(f"\nTotal games reserved: {total}")
        if args.dry_run:
            print("(dry run - no changes made)")

    elif args.evaluate:
        if not args.model:
            parser.error("--evaluate requires --model")

        model_path = Path(args.model)
        if not model_path.exists():
            model_path = MODELS_DIR / args.model

        # Infer board type and players from model name if not provided
        board_type = args.board
        num_players = args.players

        if not board_type or not num_players:
            name = model_path.stem
            parts = name.split("_")
            if not board_type:
                board_type = parts[0] if parts else "square8"
            if not num_players:
                for p in parts:
                    if p.endswith("p") and p[:-1].isdigit():
                        num_players = int(p[:-1])
                        break
                if not num_players:
                    num_players = 2

        print(f"Evaluating {model_path.name} on {board_type}_{num_players}p holdout set...")
        result = evaluate_model_on_holdout(
            model_path, board_type, num_players, args.train_loss
        )

        if result:
            print(f"\nResults:")
            print(f"  Holdout loss: {result.holdout_loss:.4f}")
            print(f"  Holdout accuracy: {result.holdout_accuracy:.4f}")
            print(f"  Samples evaluated: {result.num_samples}")
            if result.train_loss > 0:
                print(f"  Training loss: {result.train_loss:.4f}")
                print(f"  Overfit gap: {result.overfit_gap:.4f}")
                if result.overfit_gap > OVERFIT_THRESHOLD:
                    print(f"  ⚠️  WARNING: Model may be overfitting!")

    elif args.evaluate_stratified:
        if not args.model:
            parser.error("--evaluate-stratified requires --model")

        model_path = Path(args.model)
        if not model_path.exists():
            model_path = MODELS_DIR / args.model

        # Infer board type and players from model name if not provided
        board_type = args.board
        num_players = args.players

        if not board_type or not num_players:
            name = model_path.stem
            parts = name.split("_")
            if not board_type:
                board_type = parts[0] if parts else "square8"
            if not num_players:
                for p in parts:
                    if p.endswith("p") and p[:-1].isdigit():
                        num_players = int(p[:-1])
                        break
                if not num_players:
                    num_players = 2

        print(f"Evaluating {model_path.name} on {board_type}_{num_players}p with phase stratification...")
        results = evaluate_model_stratified(model_path, board_type, num_players)

        if results:
            print(f"\nStratified Results:")
            for phase in GAME_PHASES:
                if phase in results:
                    r = results[phase]
                    print(f"  {phase.capitalize():10s}: loss={r.loss:.4f}, accuracy={r.accuracy:.4f} ({r.num_samples} samples)")
                else:
                    print(f"  {phase.capitalize():10s}: No data")

            # Identify weakest phase
            by_acc = sorted([(p, r.accuracy) for p, r in results.items()], key=lambda x: x[1])
            print(f"\n  Weakest phase: {by_acc[0][0]} ({by_acc[0][1]:.4f})")
            print(f"  Strongest phase: {by_acc[-1][0]} ({by_acc[-1][1]:.4f})")
        else:
            print("No results - check if holdout data exists for this config")

    elif args.stratified_summary:
        board_type = args.board or "square8"
        num_players = args.players or 2

        summary = get_stratified_summary(board_type, num_players)
        print(f"\nStratified Summary for {board_type}_{num_players}p:")

        if summary["phases"]:
            for phase in GAME_PHASES:
                if phase in summary["phases"]:
                    p = summary["phases"][phase]
                    print(f"  {phase.capitalize():10s}: accuracy={p['accuracy']:.4f}, loss={p['loss']:.4f}")
            print(f"\n  Weakest phase: {summary['weakest_phase']}")
            print(f"  Strongest phase: {summary['strongest_phase']}")
        else:
            print("  No stratified evaluations found")

    elif args.check_overfitting:
        print("Checking for overfitting...")
        overfit = check_for_overfitting()

        if overfit:
            print(f"\n⚠️  Found {len(overfit)} potentially overfitting models:")
            for m in overfit:
                print(f"  {m['model']} ({m['config']}): gap={m['gap']:.4f}")
        else:
            print("\n✓ No overfitting detected")

    elif args.stats or True:  # Default to stats
        print_stats()


if __name__ == "__main__":
    main()
