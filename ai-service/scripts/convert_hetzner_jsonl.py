#!/usr/bin/env python3
"""Convert gzipped JSONL files from Hetzner to SQLite databases."""

import subprocess
import json
import sqlite3
from pathlib import Path

JSONL_DIR = Path("data/tournaments/hetzner_sync")
OUTPUT_DIR = Path("data/games")


def convert_jsonl_to_sqlite(jsonl_path: Path, output_db: Path, board_type: str, num_players: int):
    """Convert gzipped JSONL using system gzip."""
    if output_db.exists():
        output_db.unlink()

    conn = sqlite3.connect(str(output_db))
    c = conn.cursor()

    c.execute('''CREATE TABLE games (
        game_id TEXT PRIMARY KEY,
        board_type TEXT NOT NULL,
        num_players INTEGER NOT NULL,
        winner INTEGER,
        move_count INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    )''')

    c.execute('''CREATE TABLE moves (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_id TEXT NOT NULL,
        move_number INTEGER NOT NULL,
        board_state TEXT NOT NULL,
        move TEXT NOT NULL,
        outcome TEXT,
        ply_to_end INTEGER
    )''')

    game_info = {}
    moves_inserted = 0

    proc = subprocess.Popen(
        ['gzip', '-dc', str(jsonl_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        errors='replace'
    )

    lines_read = 0
    for line in proc.stdout:
        lines_read += 1
        try:
            data = json.loads(line.strip())
            game_id = data.get('game_id', '')
            if not game_id:
                continue

            state = data.get('state', {})
            outcome = data.get('outcome', {})

            if game_id not in game_info:
                winner = outcome.get('winner', -1) if isinstance(outcome, dict) else -1
                scores = outcome.get('scores', [])
                np = len(scores) if scores else num_players
                game_info[game_id] = {'winner': winner, 'num_players': np, 'move_count': 0}

            game_info[game_id]['move_count'] = max(
                game_info[game_id]['move_count'],
                data.get('move_number', 0)
            )

            c.execute(
                '''INSERT INTO moves (game_id, move_number, board_state, move, outcome, ply_to_end)
                   VALUES (?, ?, ?, ?, ?, ?)''',
                (game_id, data.get('move_number', 0),
                 json.dumps(state),
                 json.dumps(data.get('move', {})),
                 json.dumps(outcome),
                 data.get('ply_to_end', 0))
            )
            moves_inserted += 1

            if moves_inserted % 50000 == 0:
                conn.commit()
                print(f"  {moves_inserted:,} moves, {len(game_info):,} games...")

        except json.JSONDecodeError:
            continue
        except Exception as e:
            continue

    proc.wait()

    for game_id, info in game_info.items():
        c.execute(
            'INSERT OR REPLACE INTO games VALUES (?, ?, ?, ?, ?, datetime("now"))',
            (game_id, board_type, info['num_players'], info['winner'], info['move_count'])
        )

    conn.commit()
    c.execute('CREATE INDEX IF NOT EXISTS idx_moves_game ON moves(game_id)')
    c.execute('CREATE INDEX IF NOT EXISTS idx_games_type ON games(board_type, num_players)')
    conn.commit()
    conn.close()

    return len(game_info), moves_inserted


def main():
    files = [
        ("cpu_hexagonal_2p.jsonl", "hexagonal", 2),
        ("cpu_hexagonal_4p.jsonl", "hexagonal", 4),
        ("cpu_square19_2p.jsonl", "square19", 2),
        ("cpu_square19_3p.jsonl", "square19", 3),
        ("cpu_square19_4p.jsonl", "square19", 4),
    ]

    total_games = 0
    total_moves = 0

    for filename, board_type, num_players in files:
        jsonl_file = JSONL_DIR / filename
        if not jsonl_file.exists():
            print(f"Skipping {filename}")
            continue

        output_db = OUTPUT_DIR / f"hetzner_{board_type}_{num_players}p.db"
        print(f"\nConverting {filename} -> {output_db.name}")

        games, moves = convert_jsonl_to_sqlite(jsonl_file, output_db, board_type, num_players)
        print(f"  Final: {games:,} games, {moves:,} moves")
        total_games += games
        total_moves += moves

    print(f"\n=== TOTAL: {total_games:,} games, {total_moves:,} moves ===")


if __name__ == '__main__':
    main()
