#!/usr/bin/env python3
"""Simple canonical game generator without coordination dependencies."""

import argparse
import json
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.game_engine import GameEngine
from app.ai.heuristic_ai import HeuristicAI
from app.models.core import AIConfig, BoardType, GameStatus
from app.training.initial_state import create_initial_state


def create_db(db_path: str):
    """Create a SQLite DB with canonical schema."""
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_records (
            id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            winner INTEGER,
            final_scores TEXT,
            created_at TEXT,
            metadata TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS game_moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            move_number INTEGER,
            player INTEGER,
            move_type TEXT,
            move_data TEXT,
            game_phase TEXT,
            FOREIGN KEY(game_id) REFERENCES game_records(id)
        )
    """)
    conn.commit()
    return conn


def save_game(conn, game_id, state, moves, board_type):
    """Save a game to the database."""
    cur = conn.cursor()
    # Get scores from player objects
    scores = {}
    for player in state.players:
        scores[player.player_number] = player.territory_spaces

    cur.execute(
        "INSERT INTO game_records VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            game_id,
            board_type.value,
            state.max_players,
            state.winner,
            json.dumps(scores),
            datetime.now().isoformat(),
            json.dumps({"source": "canonical_heuristic_gen"}),
        ),
    )

    for i, (player, move, phase) in enumerate(moves):
        move_data = {"type": move.type.value if hasattr(move.type, "value") else str(move.type)}
        if hasattr(move, "from_position") and move.from_position:
            move_data["from"] = list(move.from_position)
        if hasattr(move, "to_position") and move.to_position:
            move_data["to"] = list(move.to_position)
        if hasattr(move, "direction") and move.direction:
            move_data["dir"] = move.direction.value if hasattr(move.direction, "value") else str(move.direction)

        cur.execute(
            "INSERT INTO game_moves (game_id, move_number, player, move_type, move_data, game_phase) VALUES (?, ?, ?, ?, ?, ?)",
            (
                game_id,
                i,
                player,
                move_data.get("type", ""),
                json.dumps(move_data),
                phase.value if hasattr(phase, "value") else str(phase),
            ),
        )
    conn.commit()


def run_game(engine, ais, board_type, num_players):
    """Run a single game."""
    state = create_initial_state(board_type=board_type, num_players=num_players)
    moves = []
    move_count = 0
    max_moves = 1000

    while state.game_status == GameStatus.ACTIVE and move_count < max_moves:
        current_ai = ais[state.current_player]
        move = current_ai.select_move(state)
        if move is None:
            break
        moves.append((state.current_player, move, state.current_phase))
        state = engine.apply_move(state, move)
        move_count += 1

    return state, moves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-type", default="square19", choices=["square8", "square19", "hexagonal"])
    parser.add_argument("--num-players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--num-games", type=int, default=50)
    parser.add_argument("--output", default="data/canonical/canonical.db")
    args = parser.parse_args()

    board_type = BoardType(args.board_type)
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    conn = create_db(args.output)
    engine = GameEngine()
    config = AIConfig(difficulty=2)

    print(f"Generating {args.num_games} canonical {args.board_type} {args.num_players}p games...")

    for g in range(args.num_games):
        ais = {p: HeuristicAI(p, config) for p in range(1, args.num_players + 1)}
        state, moves = run_game(engine, ais, board_type, args.num_players)
        game_id = str(uuid.uuid4())
        save_game(conn, game_id, state, moves, board_type)

        if (g + 1) % 10 == 0 or g == 0:
            print(f"  [{g+1}/{args.num_games}] winner={state.winner}, moves={len(moves)}")

    conn.close()
    print(f"Done! Saved to {args.output}")


if __name__ == "__main__":
    main()
