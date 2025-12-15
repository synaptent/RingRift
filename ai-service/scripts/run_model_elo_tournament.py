#!/usr/bin/env python3
"""Run comprehensive model-vs-model tournament with persistent Elo tracking.

This script addresses the gap where 182+ trained models exist but no cross-model
Elo leaderboard tracks their relative strengths.

Features:
1. Discovers all trained models (.pth files)
2. Runs round-robin or Swiss tournaments between models
3. Persists Elo ratings to SQLite database
4. Generates leaderboard reports

Usage:
    # Run tournament between all v3/v4/v5 models
    python scripts/run_model_elo_tournament.py --board square8 --players 2

    # Run quick tournament with top N models only
    python scripts/run_model_elo_tournament.py --board square8 --players 2 --top-n 10

    # View current leaderboard without running games
    python scripts/run_model_elo_tournament.py --leaderboard-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sqlite3
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.tournament.elo import EloCalculator, EloRating
from app.models import (
    AIConfig, AIType, BoardType, GamePhase, GameStatus,
    BoardState, GameState, Player, TimeControl,
)
from app.rules.default_engine import DefaultRulesEngine
from app.utils.victory_type import derive_victory_type
from app.training.generate_data import create_initial_state


# ============================================
# Game Execution with Neural Networks
# ============================================

def create_ai_from_model(
    model_def: Dict[str, Any],
    player_number: int,
    board_type: BoardType,
) -> "BaseAI":
    """Create an AI instance from a model definition.

    Supports neural network models (model_path to .pth file), NNUE models (.pt files),
    and baseline players (model_path starting with __BASELINE_).
    """
    from app.ai.base import BaseAI
    from app.ai.random_ai import RandomAI
    from app.ai.heuristic_ai import HeuristicAI
    from app.ai.mcts_ai import MCTSAI
    from app.ai.neural_net import NeuralNetAI
    from app.ai.minimax_ai import MinimaxAI

    model_path = model_def.get("model_path", "")
    ai_type = model_def.get("ai_type", "neural_net")

    if model_path == "__BASELINE_RANDOM__" or ai_type == "random":
        config = AIConfig(ai_type=AIType.RANDOM, board_type=board_type, difficulty=1)
        return RandomAI(player_number, config)

    elif model_path == "__BASELINE_HEURISTIC__" or ai_type == "heuristic":
        config = AIConfig(ai_type=AIType.HEURISTIC, board_type=board_type, difficulty=5)
        return HeuristicAI(player_number, config)

    elif model_path.startswith("__BASELINE_MCTS") or ai_type == "mcts":
        mcts_sims = model_def.get("mcts_simulations", 100)
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=7,
            mcts_iterations=mcts_sims,
            use_neural_net=False,  # Pure heuristic MCTS for baselines
        )
        return MCTSAI(player_number, config)

    elif ai_type == "nnue" or model_path.endswith(".pt"):
        # NNUE model - use MinimaxAI with NNUE evaluation
        config = AIConfig(
            ai_type=AIType.MINIMAX,
            board_type=board_type,
            difficulty=5,  # D5 uses NNUE evaluation
            use_neural_net=True,  # Enable NNUE
        )
        return MinimaxAI(player_number, config)

    else:
        # Neural network model - use MCTS with neural net guidance
        # Extract model_id from model_path (strip .pth extension)
        model_id = Path(model_path).stem if model_path else None
        mcts_iters = model_def.get("mcts_iterations", 800)
        config = AIConfig(
            ai_type=AIType.MCTS,
            board_type=board_type,
            difficulty=8,
            use_neural_net=True,
            model_id=model_id,
            mcts_iterations=mcts_iters,
        )
        return MCTSAI(player_number, config)


def play_model_vs_model_game(
    model_a: Dict[str, Any],
    model_b: Dict[str, Any],
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 10000,
) -> Dict[str, Any]:
    """Play a single game between two models (NN or baseline).

    Returns dict with: winner (model_a, model_b, or draw), game_length, duration_sec
    """
    import uuid
    from app.rules.default_engine import DefaultRulesEngine
    from app.training.generate_data import create_initial_state

    start_time = time.time()
    game_id = str(uuid.uuid4())

    # Create initial state
    state = create_initial_state(board_type, num_players)
    engine = DefaultRulesEngine()

    # Create AIs for both models
    # Model A plays as player 1, Model B plays as player 2
    ai_a = create_ai_from_model(model_a, 1, board_type)
    ai_b = create_ai_from_model(model_b, 2, board_type)

    move_count = 0
    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = state.current_player
        ai = ai_a if current_player == 1 else ai_b

        # Select move (BaseAI.select_move handles getting valid moves internally)
        move = ai.select_move(state)
        if move is None:
            break

        # Apply move
        state = engine.apply_move(state, move)
        move_count += 1

    duration = time.time() - start_time

    # Determine winner
    winner = "draw"
    if state.game_status == GameStatus.COMPLETED:
        if state.winner == 1:
            winner = "model_a"
        elif state.winner == 2:
            winner = "model_b"

    return {
        "winner": winner,
        "game_length": move_count,
        "duration_sec": duration,
        "game_id": game_id,
        "final_status": state.game_status.value if hasattr(state.game_status, "value") else str(state.game_status),
    }


def play_nn_vs_nn_game(
    model_a_path: str,
    model_b_path: str,
    board_type: BoardType = BoardType.SQUARE8,
    num_players: int = 2,
    max_moves: int = 10000,
    mcts_simulations: int = 100,
    save_game_history: bool = True,
    ai_type: str = "descent",
    ai_type_a: Optional[str] = None,
    ai_type_b: Optional[str] = None,
) -> Dict[str, Any]:
    """Play a single game between two neural network models.

    Returns dict with: winner (model_a, model_b, or draw), game_length, duration_sec, game_record
    If save_game_history=True, also returns full game record for training data export.
    The game_record follows canonical JSONL format suitable for NPZ conversion.

    If ai_type_a and ai_type_b are provided, they are used for model A and B respectively,
    enabling cross-inference matches (e.g., model_a uses MCTS, model_b uses Descent).
    Otherwise, ai_type is used for both models.
    """
    import time
    import uuid
    from datetime import datetime
    from app.ai.neural_net import NeuralNetAI, clear_model_cache

    def _timeout_tiebreak_winner(final_state: GameState) -> Optional[int]:
        """Deterministically select a winner for evaluation-only timeouts."""
        players = getattr(final_state, "players", None) or []
        if not players:
            return None

        territory_counts: Dict[int, int] = {}
        try:
            for p_id in final_state.board.collapsed_spaces.values():
                territory_counts[int(p_id)] = territory_counts.get(int(p_id), 0) + 1
        except Exception:
            pass

        marker_counts: Dict[int, int] = {int(p.player_number): 0 for p in players}
        try:
            for marker in final_state.board.markers.values():
                owner = int(marker.player)
                marker_counts[owner] = marker_counts.get(owner, 0) + 1
        except Exception:
            pass

        last_actor: Optional[int] = None
        try:
            if final_state.move_history:
                last_actor = int(final_state.move_history[-1].player)
        except Exception:
            last_actor = None

        sorted_players = sorted(
            players,
            key=lambda p: (
                territory_counts.get(int(p.player_number), 0),
                int(getattr(p, "eliminated_rings", 0) or 0),
                marker_counts.get(int(p.player_number), 0),
                1 if last_actor == int(p.player_number) else 0,
                -int(p.player_number),
            ),
            reverse=True,
        )
        if not sorted_players:
            return None
        return int(sorted_players[0].player_number)

    def _winner_label_for_player(player_num: int) -> str:
        # Players are assigned model_a/model_b alternating by position index:
        # P1,P3 -> model_a; P2,P4 -> model_b.
        return "model_a" if ((int(player_num) - 1) % 2) == 0 else "model_b"

    # Move history for training data export
    move_history = []

    start_time = time.time()

    # Use canonical create_initial_state for proper setup
    game_state = create_initial_state(board_type=board_type, num_players=num_players)
    game_state.id = str(uuid.uuid4())

    # Capture initial state snapshot for NPZ export (required for training data)
    # Use .dict() for pydantic v1 compatibility
    initial_state_snapshot = game_state.dict() if save_game_history else None

    # Create AI instances - alternate between model A and model B
    # Player 1 -> model_a, Player 2 -> model_b
    ai_configs = []
    model_paths = [model_a_path, model_b_path]

    # Determine AI types for each player
    # Use per-player types if provided, otherwise use the shared ai_type
    effective_ai_type_a = ai_type_a if ai_type_a else ai_type
    effective_ai_type_b = ai_type_b if ai_type_b else ai_type
    ai_types = [effective_ai_type_a, effective_ai_type_b]

    for i in range(num_players):
        model_idx = i % 2  # Alternate models for multiplayer
        player_ai_type = ai_types[model_idx]
        ai_type_enum = AIType.MCTS if player_ai_type == "mcts" else AIType.DESCENT
        config = AIConfig(
            type=ai_type_enum,
            difficulty=10,
            nn_model_id=model_paths[model_idx],  # Pass full path
            mcts_simulations=mcts_simulations,
            think_time=5000,
            use_neural_net=True,
        )
        ai_configs.append(config)

    # Create neural net AIs
    ais = []
    try:
        for i, config in enumerate(ai_configs):
            ai = NeuralNetAI(player_number=i + 1, config=config, board_type=board_type)
            ais.append(ai)
    except Exception as e:
        clear_model_cache()
        return {
            "winner": "error",
            "game_length": 0,
            "duration_sec": time.time() - start_time,
            "error": str(e),
        }

    rules_engine = DefaultRulesEngine()
    move_count = 0

    # Play the game
    while game_state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_player = game_state.current_player
        current_ai = ais[current_player - 1]
        current_ai.player_number = current_player

        try:
            move = current_ai.select_move(game_state)
        except Exception as e:
            # AI error - opponent wins
            winner_idx = 1 if current_player == 1 else 0
            clear_model_cache()
            return {
                "winner": "model_b" if winner_idx == 1 else "model_a",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
                "error": f"AI error: {e}",
            }

        if not move:
            # No valid moves - opponent wins
            winner_idx = 1 if current_player == 1 else 0
            clear_model_cache()
            return {
                "winner": "model_b" if winner_idx == 1 else "model_a",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
            }

        # Record move for training data in canonical format (matching run_random_selfplay.py)
        if save_game_history:
            # Move object uses 'type' attribute (not 'move_type')
            move_type_val = move.type.value if hasattr(move.type, 'value') else str(move.type)
            move_record = {
                'type': move_type_val,
                'player': current_player,
            }
            # Add position data (handle both key-based and coordinate-based moves)
            if hasattr(move, 'to_key') and move.to_key:
                move_record['to_key'] = move.to_key
            if hasattr(move, 'to') and move.to:
                move_record['to'] = {'x': move.to.x, 'y': move.to.y}
            if hasattr(move, 'from_key') and move.from_key:
                move_record['from_key'] = move.from_key
            if hasattr(move, 'from_pos') and move.from_pos:
                move_record['from'] = {'x': move.from_pos.x, 'y': move.from_pos.y}
            if hasattr(move, 'ring_index') and move.ring_index is not None:
                move_record['ring_index'] = move.ring_index
            move_history.append(move_record)

        try:
            game_state = rules_engine.apply_move(game_state, move)
        except Exception as e:
            winner_idx = 1 if current_player == 1 else 0
            clear_model_cache()
            return {
                "winner": "model_b" if winner_idx == 1 else "model_a",
                "game_length": move_count,
                "duration_sec": time.time() - start_time,
                "error": f"Move error: {e}",
            }

        move_count += 1

    # Determine winner
    duration = time.time() - start_time
    clear_model_cache()

    # Derive victory type for canonical format
    victory_type, stalemate_tb = derive_victory_type(game_state, max_moves)
    status = "completed" if game_state.game_status == GameStatus.COMPLETED else str(game_state.game_status.value)

    # Evaluation-only timeout tie-break (avoid draw-heavy tournaments).
    winner_player: Optional[int] = None
    if game_state.winner is not None:
        try:
            winner_player = int(game_state.winner)
        except Exception:
            winner_player = None

    timed_out = bool(move_count >= max_moves and winner_player is None)
    evaluation_tiebreak_player: Optional[int] = None
    if winner_player is None:
        evaluation_tiebreak_player = _timeout_tiebreak_winner(game_state)

    # Build game record for training data export in canonical format (matching run_random_selfplay.py)
    game_record = None
    if save_game_history:
        game_record = {
            # === Core game identifiers ===
            'game_id': game_state.id,
            'board_type': board_type.value if hasattr(board_type, 'value') else str(board_type),
            'num_players': num_players,
            # === Game outcome ===
            'winner': game_state.winner if game_state.game_status == GameStatus.COMPLETED else None,
            'move_count': move_count,
            'total_moves': move_count,  # Alias for compatibility
            'status': status,
            'game_status': status,
            'victory_type': victory_type,
            'stalemate_tiebreaker': stalemate_tb,
            'termination_reason': f"status:{status}:{victory_type}",
            'completed': game_state.game_status == GameStatus.COMPLETED,
            # === Engine/opponent metadata ===
            'engine_mode': 'nn_vs_nn_tournament',
            'opponent_type': 'nn_tournament',
            'player_types': ['neural_net'] * num_players,
            'model_a': model_a_path,
            'model_b': model_b_path,
            # === Training data (required for NPZ export) ===
            'moves': move_history,
            'initial_state': initial_state_snapshot,
            # === Timing metadata ===
            'game_time_seconds': duration,
            'duration_sec': duration,
            'timestamp': datetime.now().isoformat(),
            'created_at': datetime.now().isoformat(),
            # === Source tracking ===
            'source': 'run_model_elo_tournament.py',
        }
        if evaluation_tiebreak_player is not None:
            if timed_out:
                game_record["timeout_tiebreak_winner"] = int(evaluation_tiebreak_player)
                game_record["timeout_tiebreak_winner_model"] = _winner_label_for_player(int(evaluation_tiebreak_player))
            else:
                game_record["evaluation_tiebreak_winner"] = int(evaluation_tiebreak_player)
                game_record["evaluation_tiebreak_winner_model"] = _winner_label_for_player(int(evaluation_tiebreak_player))

    winner_label = "draw"
    if winner_player is not None:
        winner_label = _winner_label_for_player(winner_player)
    elif evaluation_tiebreak_player is not None:
        winner_label = _winner_label_for_player(evaluation_tiebreak_player)

    return {
        "winner": winner_label,
        "game_length": move_count,
        "duration_sec": duration,
        "game_record": game_record,
    }


def run_model_matchup(
    db: EloDatabase,
    model_a: Dict[str, Any],
    model_b: Dict[str, Any],
    board_type: str,
    num_players: int,
    games: int,
    tournament_id: str,
    nn_ai_type: str = "descent",
    use_both_ai_types: bool = False,
    save_games_dir: Optional[Path] = None,
) -> Dict[str, int]:
    """Run multiple games between two models and update Elo.

    If save_games_dir is provided, games are saved to JSONL for training data.
    If use_both_ai_types is True, runs half games with MCTS and half with Descent,
    ensuring neural networks are evaluated using both inference methods.
    """
    board_type_enum = BoardType.SQUARE8
    if board_type == "square19":
        board_type_enum = BoardType.SQUARE19
    elif board_type == "hex" or board_type == "hexagonal":
        board_type_enum = BoardType.HEXAGONAL

    results = {"model_a_wins": 0, "model_b_wins": 0, "draws": 0, "errors": 0}

    # Setup game saving directory
    if save_games_dir is None:
        save_games_dir = AI_SERVICE_ROOT / "data" / "holdouts" / "elo_tournaments"
    save_games_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = save_games_dir / f"tournament_{tournament_id}_{board_type}_{num_players}p.jsonl"

    # Check if either model is a baseline (requires play_model_vs_model_game)
    is_baseline_match = (
        model_a.get("model_path", "").startswith("__BASELINE") or
        model_b.get("model_path", "").startswith("__BASELINE") or
        model_a.get("ai_type") in ("random", "heuristic", "mcts") or
        model_b.get("ai_type") in ("random", "heuristic", "mcts")
    )

    for game_num in range(games):
        # Alternate who plays first
        if game_num % 2 == 0:
            play_a, play_b = model_a, model_b
            id_a, id_b = model_a["model_id"], model_b["model_id"]
        else:
            play_a, play_b = model_b, model_a
            id_a, id_b = model_b["model_id"], model_a["model_id"]

        # Select AI type(s) for this game
        if use_both_ai_types:
            # Cross-inference evaluation: cycle through all 4 AI type combinations
            # This ensures each NN is evaluated with both MCTS and Descent against both
            ai_type_combos = [
                ("descent", "descent"),   # Both use descent
                ("mcts", "mcts"),         # Both use MCTS
                ("mcts", "descent"),      # A uses MCTS, B uses descent
                ("descent", "mcts"),      # A uses descent, B uses MCTS
            ]
            combo_idx = game_num % len(ai_type_combos)
            ai_type_a, ai_type_b = ai_type_combos[combo_idx]
        else:
            ai_type_a = ai_type_b = nn_ai_type

        if is_baseline_match:
            # Use generic model-vs-model for baseline players
            result = play_model_vs_model_game(
                model_a=play_a,
                model_b=play_b,
                board_type=board_type_enum,
                num_players=num_players,
                max_moves=10000,
            )
        else:
            # Use NN-specific game play for neural networks
            result = play_nn_vs_nn_game(
                model_a_path=play_a["model_path"],
                model_b_path=play_b["model_path"],
                board_type=board_type_enum,
                num_players=num_players,
                max_moves=10000,
                mcts_simulations=50,  # Faster games
                save_game_history=True,  # Record for training
                ai_type_a=ai_type_a,
                ai_type_b=ai_type_b,
            )

        # Save game record to JSONL for training data
        game_record = result.get("game_record")
        if game_record:
            game_record["tournament_id"] = tournament_id
            game_record["game_num"] = game_num
            try:
                with open(jsonl_path, "a") as f:
                    f.write(json.dumps(game_record) + "\n")
            except Exception as e:
                print(f"Warning: Failed to save game record: {e}")

        # Map back to original model_a/model_b
        winner_id = None
        if result["winner"] == "model_a":
            winner_id = id_a
        elif result["winner"] == "model_b":
            winner_id = id_b

        # Update stats based on original model_a vs model_b
        if winner_id == model_a["model_id"]:
            results["model_a_wins"] += 1
            winner = model_a["model_id"]
        elif winner_id == model_b["model_id"]:
            results["model_b_wins"] += 1
            winner = model_b["model_id"]
        elif result["winner"] == "error":
            results["errors"] += 1
            continue
        else:
            results["draws"] += 1
            winner = "draw"

        # Update Elo and record match using unified database
        update_elo_after_match(
            db,
            model_a["model_id"],
            model_b["model_id"],
            winner,
            board_type,
            num_players,
            tournament_id,
            game_length=result.get("game_length", 0),
            duration_sec=result.get("duration_sec", 0.0),
        )

    return results


# ============================================
# Persistent Elo Database
# ============================================

# Canonical Elo database - tracks trained model ratings and history
ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "unified_elo.db"
LEGACY_ELO_DB_PATH = AI_SERVICE_ROOT / "data" / "elo_leaderboard.db"  # Legacy, merged into unified

# Import unified Elo database module
from app.tournament.unified_elo_db import (
    EloDatabase,
    get_elo_database,
    UnifiedEloRating,
)
UNIFIED_DB_AVAILABLE = True


def init_elo_database(db_path: Path = ELO_DB_PATH) -> EloDatabase:
    """Initialize unified Elo database.

    Uses the unified EloDatabase class for consistent schema across all tournament scripts.
    """
    return EloDatabase(db_path)


def discover_models(
    models_dir: Path,
    board_type: str = "square8",
    num_players: int = 2,
    include_nnue: bool = False,
) -> List[Dict[str, Any]]:
    """Discover all trained models for a given board type.

    Args:
        models_dir: Directory to search for NN models
        board_type: Board type (square8, square19, hexagonal)
        num_players: Number of players (2, 3, 4)
        include_nnue: If True, also discover NNUE models in models/nnue/
    """
    models = []

    # Look for .pth files matching the board/player config
    pattern = f"{board_type.replace('square', 'sq')}_{num_players}p"

    for f in models_dir.glob("*.pth"):
        name = f.stem

        # Extract version info
        version = "unknown"
        if "ringrift_v5" in name:
            version = "v5"
        elif "ringrift_v4" in name:
            version = "v4"
        elif "ringrift_v3" in name:
            version = "v3"
        elif "nn_baseline" in name:
            version = "baseline"

        # Check if it matches the board/player pattern
        if pattern in name or "ringrift_v" in name:
            models.append({
                "model_id": name,
                "model_path": str(f),
                "board_type": board_type,
                "num_players": num_players,
                "version": version,
                "size_mb": f.stat().st_size / (1024 * 1024),
                "created_at": f.stat().st_mtime,
                "model_type": "nn",
            })

    # Also discover NNUE models if requested
    if include_nnue:
        nnue_dir = models_dir / "nnue"
        if nnue_dir.exists():
            nnue_pattern = f"nnue_{board_type}_{num_players}p"
            for f in nnue_dir.glob("*.pt"):
                name = f.stem
                # Check if it matches the board/player pattern
                if nnue_pattern in name or f"nnue_policy_{board_type}_{num_players}p" in name:
                    models.append({
                        "model_id": f"nnue_{name}",
                        "model_path": str(f),
                        "board_type": board_type,
                        "num_players": num_players,
                        "version": "nnue",
                        "size_mb": f.stat().st_size / (1024 * 1024),
                        "created_at": f.stat().st_mtime,
                        "model_type": "nnue",
                        "ai_type": "nnue",
                    })

    return sorted(models, key=lambda x: x["created_at"], reverse=True)


def get_baseline_players(board_type: str, num_players: int) -> List[Dict[str, Any]]:
    """Get baseline player definitions for Elo calibration.

    These provide anchor points for the Elo scale:
    - random: ~800-1000 Elo (worst baseline)
    - heuristic: ~1200-1400 Elo (decent play)
    - mcts_100: ~1400-1600 Elo (strong baseline)
    """
    now = time.time()
    baselines = [
        {
            "model_id": f"baseline_random_{board_type}_{num_players}p",
            "model_path": "__BASELINE_RANDOM__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "random",
        },
        {
            "model_id": f"baseline_heuristic_{board_type}_{num_players}p",
            "model_path": "__BASELINE_HEURISTIC__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "heuristic",
        },
        {
            "model_id": f"baseline_mcts_100_{board_type}_{num_players}p",
            "model_path": "__BASELINE_MCTS_100__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "mcts",
            "mcts_simulations": 100,
        },
        {
            "model_id": f"baseline_mcts_500_{board_type}_{num_players}p",
            "model_path": "__BASELINE_MCTS_500__",
            "board_type": board_type,
            "num_players": num_players,
            "version": "baseline",
            "size_mb": 0,
            "created_at": now,
            "ai_type": "mcts",
            "mcts_simulations": 500,
        },
    ]
    return baselines


def register_models(db: EloDatabase, models: List[Dict[str, Any]]):
    """Register discovered models in the database."""
    for m in models:
        # Determine AI type from model path
        model_path = m.get("model_path", "")
        if model_path.startswith("__BASELINE_RANDOM__"):
            ai_type = "random"
            participant_type = "baseline"
        elif model_path.startswith("__BASELINE_HEURISTIC__"):
            ai_type = "heuristic"
            participant_type = "baseline"
        elif model_path.startswith("__BASELINE_MCTS"):
            ai_type = "mcts"
            participant_type = "baseline"
        else:
            ai_type = "neural_net"
            participant_type = "model"

        db.register_participant(
            participant_id=m["model_id"],
            participant_type=participant_type,
            ai_type=ai_type,
            use_neural_net=(ai_type == "neural_net"),
            model_path=m.get("model_path"),
            model_version=m.get("version"),
        )


def get_leaderboard(
    db: EloDatabase,
    board_type: str = None,
    num_players: int = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Get current Elo leaderboard."""
    rows = db.get_leaderboard(board_type=board_type, num_players=num_players, min_games=0, limit=limit)

    results = []
    for row in rows:
        games = row.get("games_played", 0)
        wins = row.get("wins", 0)
        win_rate = (wins / games * 100) if games > 0 else 0

        results.append({
            "rank": len(results) + 1,
            "model_id": row.get("participant_id"),  # Keep as model_id for backward compat
            "participant_id": row.get("participant_id"),
            "board_type": row.get("board_type"),
            "num_players": row.get("num_players"),
            "rating": round(row.get("rating", 1500.0), 1),
            "games_played": games,
            "wins": wins,
            "losses": row.get("losses", 0),
            "draws": row.get("draws", 0),
            "win_rate": round(win_rate, 1),
            "version": row.get("model_version", "unknown"),
            "last_update": datetime.fromtimestamp(row["last_update"]).isoformat() if row.get("last_update") else None,
        })

    return results


def update_elo_after_match(
    db: EloDatabase,
    model_a: str,
    model_b: str,
    winner: str,  # model_a, model_b, or "draw"
    board_type: str,
    num_players: int,
    tournament_id: str = None,
    k_factor: float = 32.0,
    game_length: int = 0,
    duration_sec: float = 0.0,
):
    """Update Elo ratings after a match using unified EloDatabase."""
    # Determine rankings from winner
    if winner == model_a:
        is_draw = False
    elif winner == model_b:
        # Swap so winner is first in the call
        model_a, model_b = model_b, model_a
        is_draw = False
    else:
        is_draw = True

    db.record_two_player_result(
        winner_id=model_a,
        loser_id=model_b,
        board_type=board_type,
        num_players=num_players,
        tournament_id=tournament_id or "model_elo_tournament",
        is_draw=is_draw,
        game_length=game_length,
        duration_sec=duration_sec,
    )


def print_leaderboard(leaderboard: List[Dict[str, Any]], title: str = "Elo Leaderboard"):
    """Pretty print the leaderboard."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")

    if not leaderboard:
        print("  No models found in leaderboard.")
        return

    print(f"{'Rank':<6}{'Model':<50}{'Elo':>8}{'Games':>8}{'Win%':>8}{'Type':>10}")
    print("-" * 90)

    for entry in leaderboard:
        # Support both model_id and participant_id keys
        model_id = entry.get("model_id") or entry.get("participant_id", "unknown")
        model_short = model_id[:48] if len(model_id) > 48 else model_id
        version = entry.get("version") or entry.get("participant_type", "?")
        print(f"{entry['rank']:<6}{model_short:<50}{entry['rating']:>8.1f}{entry['games_played']:>8}{entry['win_rate']:>7.1f}%{version:>10}")

    print(f"\nTotal models: {len(leaderboard)}")


# All supported board/player configurations for Elo tracking
ALL_CONFIGS = [
    ("square8", 2),
    ("square8", 3),
    ("square8", 4),
    ("square19", 2),
    ("square19", 3),
    ("square19", 4),
    ("hexagonal", 2),
    ("hexagonal", 3),
    ("hexagonal", 4),
]


def run_all_config_tournaments(args):
    """Run tournaments for all board/player configurations.

    This ensures there's an Elo ranking for each combination of board type and number of players.
    """
    import uuid

    db_path = Path(args.db) if args.db else ELO_DB_PATH
    db = init_elo_database(db_path)
    models_dir = AI_SERVICE_ROOT / "models"

    print(f"\n{'='*80}")
    print(f" Running Elo Tournaments for All Configurations")
    print(f"{'='*80}")

    overall_start = time.time()
    total_games_all = 0

    for board_type, num_players in ALL_CONFIGS:
        config_label = f"{board_type} {num_players}p"
        print(f"\n{'='*60}")
        print(f" Configuration: {config_label}")
        print(f"{'='*60}")

        # Discover models for this config
        if args.baselines_only:
            models = get_baseline_players(board_type, num_players)
            print(f"Using {len(models)} baseline players for {config_label}")
        else:
            models = discover_models(models_dir, board_type, num_players)
            print(f"Discovered {len(models)} models for {config_label}")
            if args.include_baselines:
                baselines = get_baseline_players(board_type, num_players)
                models.extend(baselines)
                print(f"Added {len(baselines)} baseline players")

        if args.top_n and not args.baselines_only:
            models = models[:args.top_n]
            print(f"Using top {args.top_n} most recent models")

        if len(models) < 2:
            print(f"  Skipping {config_label}: need at least 2 models")
            continue

        # Register models
        register_models(db, models)

        if args.leaderboard_only:
            leaderboard = get_leaderboard(db, board_type, num_players)
            print_leaderboard(leaderboard, f"Elo Leaderboard - {config_label}")
            continue

        if not args.run:
            # Just show plan
            matchups = []
            for i, m1 in enumerate(models):
                for m2 in models[i+1:]:
                    matchups.append((m1, m2))
            print(f"  Would run {len(matchups) * args.games} games ({len(matchups)} matchups × {args.games} games)")
            continue

        # Run tournament for this config
        tournament_id = str(uuid.uuid4())[:8]
        matchups = []
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                matchups.append((m1, m2))

        print(f"Running tournament {tournament_id}: {len(matchups)} matchups × {args.games} games")

        config_start = time.time()
        games_completed = 0

        for matchup_idx, (m1, m2) in enumerate(matchups):
            print(f"  [{matchup_idx + 1}/{len(matchups)}] {m1['model_id'][:30]} vs {m2['model_id'][:30]}", end=" ")

            try:
                results = run_model_matchup(
                    db=db,
                    model_a=m1,
                    model_b=m2,
                    board_type=board_type,
                    num_players=num_players,
                    games=args.games,
                    tournament_id=tournament_id,
                    nn_ai_type=args.ai_type,
                    use_both_ai_types=args.both_ai_types,
                )
                games_completed += args.games
                print(f"A={results['model_a_wins']} B={results['model_b_wins']} D={results['draws']}")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

        config_elapsed = time.time() - config_start
        total_games_all += games_completed
        print(f"  Completed {games_completed} games in {config_elapsed:.1f}s")

        # Show updated leaderboard
        leaderboard = get_leaderboard(db, board_type, num_players, limit=10)
        print_leaderboard(leaderboard, f"Top 10 - {config_label}")

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*80}")
    print(f" All Tournaments Complete")
    print(f"{'='*80}")
    print(f"Total games: {total_games_all}")
    print(f"Total time: {overall_elapsed:.1f}s")

    db.close()


def run_continuous_tournament(args):
    """Run tournaments continuously in daemon mode.

    This provides continuous model evaluation for the unified AI improvement loop:
    - Shadow tournaments every 15 minutes (10 games)
    - Full tournaments every hour (50 games)
    - Checkpoint monitoring for new models
    - Event emission for pipeline integration
    """
    import signal

    # Try to import event bus for pipeline integration
    emit_events = args.emit_events
    if emit_events:
        try:
            from app.distributed.data_events import (
                emit_evaluation_completed,
                emit_error,
                get_event_bus,
            )
            import asyncio
        except ImportError:
            print("[ContinuousTournament] Warning: Event bus not available, disabling event emission")
            emit_events = False

    db_path = Path(args.db) if args.db else ELO_DB_PATH
    db = init_elo_database(db_path)
    models_dir = AI_SERVICE_ROOT / "models"

    interval = args.continuous_interval
    checkpoint_watch_dir = Path(args.checkpoint_watch) if args.checkpoint_watch else None
    known_checkpoints = set()

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        print("\n[ContinuousTournament] Shutdown requested")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"[ContinuousTournament] Starting continuous evaluation daemon")
    print(f"  Interval: {interval}s ({interval // 60} min)")
    print(f"  Games per matchup: {args.games}")
    print(f"  Board: {args.board}, Players: {args.players}")
    if checkpoint_watch_dir:
        print(f"  Watching: {checkpoint_watch_dir}")

    iteration = 0
    while running:
        iteration += 1
        iteration_start = time.time()

        print(f"\n{'='*60}")
        print(f"[ContinuousTournament] Iteration {iteration} at {datetime.now().isoformat()}")
        print(f"{'='*60}")

        try:
            # Check for new checkpoints if watching
            if checkpoint_watch_dir and checkpoint_watch_dir.exists():
                for pth_file in checkpoint_watch_dir.glob("*.pth"):
                    file_key = f"{pth_file}:{pth_file.stat().st_mtime}"
                    if file_key not in known_checkpoints:
                        known_checkpoints.add(file_key)
                        print(f"[ContinuousTournament] New checkpoint: {pth_file.name}")

            # Discover models
            if args.baselines_only:
                models = get_baseline_players(args.board, args.players)
            else:
                models = discover_models(models_dir, args.board, args.players, include_nnue=args.include_nnue)
                if args.include_baselines:
                    models.extend(get_baseline_players(args.board, args.players))

            if args.top_n:
                models = models[:args.top_n]

            if len(models) < 2:
                print(f"[ContinuousTournament] Not enough models ({len(models)}), waiting...")
                time.sleep(interval)
                continue

            # Register models
            register_models(db, models)

            # Filter archived models
            active_models = [m for m in models if not is_model_archived(db, m["model_id"], args.board, args.players)]
            if len(active_models) < 2:
                print(f"[ContinuousTournament] Not enough active models ({len(active_models)}), waiting...")
                time.sleep(interval)
                continue
            models = active_models

            # Generate matchups
            if args.elo_matchmaking:
                matchups = generate_elo_based_matchups(models, db, args.board, args.players, args.elo_range)
            else:
                matchups = []
                for i, m1 in enumerate(models):
                    for m2 in models[i+1:]:
                        matchups.append((m1, m2))

            # Limit matchups for shadow mode
            if args.quick and len(matchups) > 10:
                # Sample matchups for quick evaluation
                import random
                matchups = random.sample(matchups, min(10, len(matchups)))

            print(f"[ContinuousTournament] Running {len(matchups)} matchups × {args.games} games")

            import uuid
            tournament_id = f"cont_{str(uuid.uuid4())[:8]}"
            games_completed = 0
            total_wins = 0
            total_games = 0

            for matchup_idx, (m1, m2) in enumerate(matchups):
                if not running:
                    break

                try:
                    results = run_model_matchup(
                        db=db,
                        model_a=m1,
                        model_b=m2,
                        board_type=args.board,
                        num_players=args.players,
                        games=args.games,
                        tournament_id=tournament_id,
                        nn_ai_type=args.ai_type,
                        use_both_ai_types=args.both_ai_types,
                    )
                    games_completed += args.games
                    total_wins += results["model_a_wins"] + results["model_b_wins"]
                    total_games += results["model_a_wins"] + results["model_b_wins"] + results["draws"]
                    print(f"  [{matchup_idx + 1}/{len(matchups)}] A={results['model_a_wins']} B={results['model_b_wins']} D={results['draws']}")

                except Exception as e:
                    print(f"  [{matchup_idx + 1}/{len(matchups)}] ERROR: {e}")
                    continue

            iteration_duration = time.time() - iteration_start

            # Get best model Elo for reporting
            leaderboard = get_leaderboard(db, args.board, args.players, limit=1)
            best_elo = leaderboard[0]["rating"] if leaderboard else 1500.0
            win_rate = total_wins / total_games if total_games > 0 else 0.5

            print(f"[ContinuousTournament] Completed {games_completed} games in {iteration_duration:.1f}s")
            print(f"  Best Elo: {best_elo:.0f}, Win rate: {win_rate:.1%}")

            # Emit event for pipeline integration
            if emit_events:
                config_key = f"{args.board}_{args.players}p"
                try:
                    asyncio.run(emit_evaluation_completed(
                        config=config_key,
                        elo=best_elo,
                        games_played=games_completed,
                        win_rate=win_rate,
                        source="run_model_elo_tournament.py",
                    ))
                except Exception as e:
                    print(f"[ContinuousTournament] Failed to emit event: {e}")

            # Print summary output for parsing by shadow tournament service
            print(f"Win rate: {win_rate * 100:.1f}%")
            print(f"Elo: {best_elo:.0f}")

        except Exception as e:
            print(f"[ContinuousTournament] Error in iteration: {e}")
            import traceback
            traceback.print_exc()

            if emit_events:
                try:
                    asyncio.run(emit_error(
                        component="continuous_tournament",
                        error=str(e),
                        source="run_model_elo_tournament.py",
                    ))
                except Exception:
                    pass

        # Wait for next iteration
        if running:
            elapsed = time.time() - iteration_start
            sleep_time = max(0, interval - elapsed)
            if sleep_time > 0:
                print(f"[ContinuousTournament] Sleeping {sleep_time:.0f}s until next iteration")
                try:
                    time.sleep(sleep_time)
                except KeyboardInterrupt:
                    running = False

    print("[ContinuousTournament] Stopped")
    db.close()


def generate_elo_based_matchups(
    models: List[Dict[str, Any]],
    db: EloDatabase,
    board_type: str,
    num_players: int,
    max_elo_diff: int = 200,
) -> List[Tuple[Dict, Dict]]:
    """Generate matchups between models with similar Elo ratings.

    This produces more informative games than random matchups, as close
    games provide more Elo information than one-sided blowouts.
    """
    # Get current Elo ratings for all models
    model_elos = {}
    for model in models:
        rating = db.get_rating(model["model_id"], board_type, num_players)
        model_elos[model["model_id"]] = rating.rating

    # Sort models by Elo
    sorted_models = sorted(models, key=lambda m: model_elos.get(m["model_id"], 1500), reverse=True)

    matchups = []
    used = set()

    # Pair adjacent models in Elo ranking (closest ratings play each other)
    for i, m1 in enumerate(sorted_models):
        if m1["model_id"] in used:
            continue

        # Find best opponent (closest Elo within range, not already paired)
        best_opponent = None
        best_diff = float("inf")

        for m2 in sorted_models:
            if m2["model_id"] == m1["model_id"] or m2["model_id"] in used:
                continue

            elo_diff = abs(model_elos[m1["model_id"]] - model_elos[m2["model_id"]])
            if elo_diff <= max_elo_diff and elo_diff < best_diff:
                best_diff = elo_diff
                best_opponent = m2

        if best_opponent:
            matchups.append((m1, best_opponent))
            used.add(m1["model_id"])
            used.add(best_opponent["model_id"])

    # Add remaining unmatched models paired with closest available
    unmatched = [m for m in sorted_models if m["model_id"] not in used]
    for i in range(0, len(unmatched) - 1, 2):
        matchups.append((unmatched[i], unmatched[i + 1]))

    return matchups


def archive_low_elo_models(
    db: EloDatabase,
    board_type: str,
    num_players: int,
    elo_threshold: int = 1400,
    min_games: int = 50,
) -> List[str]:
    """Archive models with low Elo after sufficient games.

    Archived models are marked in the database and excluded from future tournaments.
    Returns list of archived model IDs.
    """
    conn = db._get_connection()
    cursor = conn.cursor()

    # Find models to archive (using unified schema with participant_id)
    cursor.execute("""
        SELECT participant_id, rating, games_played
        FROM elo_ratings
        WHERE board_type = ? AND num_players = ?
          AND rating < ? AND games_played >= ?
    """, (board_type, num_players, elo_threshold, min_games))

    to_archive = []
    for row in cursor.fetchall():
        model_id, rating, games = row
        to_archive.append({
            "model_id": model_id,
            "rating": rating,
            "games_played": games,
        })

    if not to_archive:
        return []

    # Create archived_models table if not exists
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS archived_models (
            model_id TEXT,
            board_type TEXT,
            num_players INTEGER,
            final_rating REAL,
            games_played INTEGER,
            archived_at REAL,
            PRIMARY KEY (model_id, board_type, num_players)
        )
    """)

    # Archive the models
    archived = []
    now = time.time()
    for model in to_archive:
        cursor.execute("""
            INSERT OR REPLACE INTO archived_models
            (model_id, board_type, num_players, final_rating, games_played, archived_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (model["model_id"], board_type, num_players, model["rating"], model["games_played"], now))
        archived.append(model["model_id"])

    conn.commit()
    return archived


def is_model_archived(db: EloDatabase, model_id: str, board_type: str, num_players: int) -> bool:
    """Check if a model has been archived."""
    conn = db._get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT 1 FROM archived_models
            WHERE model_id = ? AND board_type = ? AND num_players = ?
        """, (model_id, board_type, num_players))
        return cursor.fetchone() is not None
    except sqlite3.OperationalError:
        # Table doesn't exist yet - no models archived
        return False


def main():
    parser = argparse.ArgumentParser(description="Run model Elo tournament")
    parser.add_argument("--board", default="square8", help="Board type")
    parser.add_argument("--players", type=int, default=2, help="Number of players")
    parser.add_argument("--games", type=int, default=10, help="Games per matchup")
    parser.add_argument("--top-n", type=int, help="Only include top N models by recency")
    parser.add_argument("--leaderboard-only", action="store_true", help="Just show leaderboard")
    parser.add_argument("--run", action="store_true", help="Actually run games (otherwise just shows plan)")
    parser.add_argument("--mcts-sims", type=int, default=50, help="MCTS simulations per move")
    parser.add_argument("--db", type=str, help="Path to Elo database")
    parser.add_argument("--all-configs", action="store_true", help="Run tournament for all board/player configurations")
    parser.add_argument("--elo-matchmaking", action="store_true", help="Use Elo-based matchmaking (pair similar-rated models)")
    parser.add_argument("--elo-range", type=int, default=200, help="Max Elo difference for matchmaking (default: 200)")
    parser.add_argument("--archive-threshold", type=int, default=1400, help="Archive models below this Elo after 50+ games")
    parser.add_argument("--archive", action="store_true", help="Archive low-Elo models")
    parser.add_argument("--include-baselines", action="store_true", help="Include baseline players (Random, Heuristic, MCTS)")
    parser.add_argument("--baselines-only", action="store_true", help="Run tournament with only baseline players (for calibration)")
    parser.add_argument("--ai-type", choices=["mcts", "descent"], default="descent", help="AI type for neural networks (default: descent)")
    parser.add_argument("--both-ai-types", action="store_true", help="Use BOTH MCTS and Descent AI types (half games each) for comprehensive NN evaluation")
    parser.add_argument("--include-nnue", action="store_true", help="Include NNUE models from models/nnue/ directory")
    parser.add_argument("--nnue-only", action="store_true", help="Run tournament with only NNUE models")

    # Continuous evaluation modes (for unified AI loop integration)
    parser.add_argument("--quick", action="store_true", help="Quick mode: 10 games per matchup for fast shadow evaluation")
    parser.add_argument("--continuous", action="store_true", help="Continuous daemon mode: run tournaments periodically")
    parser.add_argument("--continuous-interval", type=int, default=900, help="Interval between continuous runs (seconds, default: 900)")
    parser.add_argument("--checkpoint-watch", type=str, help="Directory to watch for new model checkpoints")
    parser.add_argument("--emit-events", action="store_true", help="Emit data events for pipeline integration")

    args = parser.parse_args()

    # Quick mode: reduce games for fast shadow evaluation
    if args.quick:
        args.games = 10
        args.run = True  # Quick mode implies --run

    # Continuous mode: run as daemon
    if args.continuous:
        run_continuous_tournament(args)
        return

    # If --all-configs, loop through all configurations
    if args.all_configs:
        run_all_config_tournaments(args)
        return

    db_path = Path(args.db) if args.db else ELO_DB_PATH
    db = init_elo_database(db_path)

    # Discover models
    models_dir = AI_SERVICE_ROOT / "models"
    if args.baselines_only:
        models = get_baseline_players(args.board, args.players)
        print(f"\nUsing {len(models)} baseline players for {args.board} {args.players}p")
    elif args.nnue_only:
        # Only NNUE models
        models = discover_models(models_dir, args.board, args.players, include_nnue=True)
        models = [m for m in models if m.get("model_type") == "nnue"]
        print(f"\nDiscovered {len(models)} NNUE models for {args.board} {args.players}p")
    else:
        models = discover_models(models_dir, args.board, args.players, include_nnue=args.include_nnue)
        nn_count = len([m for m in models if m.get("model_type") != "nnue"])
        nnue_count = len([m for m in models if m.get("model_type") == "nnue"])
        print(f"\nDiscovered {nn_count} NN models for {args.board} {args.players}p")
        if nnue_count > 0:
            print(f"Also found {nnue_count} NNUE models")
        if args.include_baselines:
            baselines = get_baseline_players(args.board, args.players)
            models.extend(baselines)
            print(f"Added {len(baselines)} baseline players")

    if args.top_n and not args.baselines_only:
        models = models[:args.top_n]
        print(f"Using top {args.top_n} most recent models")

    # Register models
    register_models(db, models)

    # Filter out archived models
    active_models = [m for m in models if not is_model_archived(db, m["model_id"], args.board, args.players)]
    if len(active_models) < len(models):
        print(f"Filtered out {len(models) - len(active_models)} archived models")
        models = active_models

    # Handle archiving if requested
    if args.archive:
        archived = archive_low_elo_models(
            db, args.board, args.players,
            elo_threshold=args.archive_threshold,
            min_games=50,
        )
        if archived:
            print(f"\nArchived {len(archived)} low-Elo models:")
            for model_id in archived:
                print(f"  - {model_id}")
            # Re-filter models
            models = [m for m in models if m["model_id"] not in archived]

    # Show leaderboard
    leaderboard = get_leaderboard(db, args.board, args.players)
    print_leaderboard(leaderboard, f"Current Elo Leaderboard - {args.board} {args.players}p")

    if args.leaderboard_only:
        db.close()
        return

    if len(models) < 2:
        print("\nNeed at least 2 models to run a tournament!")
        db.close()
        return

    # Generate matchups (Elo-based or round-robin)
    if args.elo_matchmaking:
        print(f"\nUsing Elo-based matchmaking (max diff: {args.elo_range})")
        matchups = generate_elo_based_matchups(
            models, db, args.board, args.players, args.elo_range
        )
    else:
        # Standard round-robin matchups
        matchups = []
        for i, m1 in enumerate(models):
            for m2 in models[i+1:]:
                matchups.append((m1, m2))

    print(f"\n{'='*80}")
    print(f" Tournament Plan")
    print(f"{'='*80}")
    print(f"Models: {len(models)}")
    print(f"Matchups: {len(matchups)}")
    print(f"Games per matchup: {args.games}")
    print(f"Total games needed: {len(matchups) * args.games}")

    # Check if --run flag provided
    if not args.run:
        print("\nSample matchups:")
        for m1, m2 in matchups[:5]:
            print(f"  {m1['model_id'][:40]} vs {m2['model_id'][:40]}")

        if len(matchups) > 5:
            print(f"  ... and {len(matchups) - 5} more")

        print("\nAdd --run flag to execute games and update Elo ratings.")
        db.close()
        return

    # Run the tournament
    import uuid
    tournament_id = str(uuid.uuid4())[:8]

    print(f"\n{'='*80}")
    print(f" Running Tournament {tournament_id}")
    print(f"{'='*80}")

    total_games = len(matchups) * args.games
    games_completed = 0
    start_time = time.time()

    for matchup_idx, (m1, m2) in enumerate(matchups):
        print(f"\nMatchup {matchup_idx + 1}/{len(matchups)}: {m1['model_id'][:35]} vs {m2['model_id'][:35]}")

        try:
            results = run_model_matchup(
                db=db,
                model_a=m1,
                model_b=m2,
                board_type=args.board,
                num_players=args.players,
                games=args.games,
                tournament_id=tournament_id,
                nn_ai_type=args.ai_type,
                use_both_ai_types=args.both_ai_types,
            )

            games_completed += args.games
            elapsed = time.time() - start_time
            rate = games_completed / elapsed if elapsed > 0 else 0

            print(f"  Results: A={results['model_a_wins']} B={results['model_b_wins']} D={results['draws']} E={results['errors']}")
            print(f"  Progress: {games_completed}/{total_games} games ({rate:.1f} games/sec)")

        except Exception as e:
            import traceback
            print(f"  Error in matchup: {e}")
            traceback.print_exc()
            continue

    # Show final leaderboard
    final_leaderboard = get_leaderboard(db, args.board, args.players, limit=100)
    print_leaderboard(final_leaderboard, f"Final Elo Leaderboard - {args.board} {args.players}p (Tournament {tournament_id})")

    # Summary
    elapsed = time.time() - start_time
    print(f"\nTournament completed in {elapsed:.1f} seconds")
    print(f"Total games played: {games_completed}")

    db.close()


if __name__ == "__main__":
    main()
