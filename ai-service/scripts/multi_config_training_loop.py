#!/usr/bin/env python3
"""
Multi-config training loop that monitors game databases and triggers training
when enough new games are available for each board/player configuration.

Key improvements:
- Uses config-specific databases with full move data
- Only counts games that have moves (required for training)
- Supports multiple database sources per config
- More efficient database queries
"""
import os
import sys
import time
import sqlite3
import subprocess
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Base paths
BASE_DIR = os.path.expanduser("~/ringrift/ai-service")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Database sources for each config - databases that have games WITH moves
# Format: (board_type, num_players) -> list of database paths
CONFIG_DATABASES: Dict[Tuple[str, int], List[str]] = {
    ("square8", 2): [
        "data/games/selfplay.db",
        "data/selfplay/p2p/square8_2p",  # Directory with multiple DBs
    ],
    ("square8", 3): [
        "data/games/selfplay.db",
        "data/selfplay/p2p/square8_3p",
    ],
    ("square8", 4): [
        "data/games/selfplay.db",
        "data/selfplay/p2p/square8_4p",
        "data/selfplay/diverse/square8_4p.db",
    ],
    ("square19", 2): [
        "data/selfplay/diverse/square19_2p.db",
        "data/selfplay/p2p/square19_2p",
        "data/games/selfplay.db",
    ],
    ("square19", 3): [
        "data/selfplay/diverse/square19_3p.db",
        "data/selfplay/p2p/square19_3p",
    ],
    ("square19", 4): [
        "data/selfplay/diverse/square19_4p.db",
        "data/games/new_square19_4p.db",
        "data/selfplay/p2p/square19_4p",
    ],
    ("hexagonal", 2): [
        "data/selfplay/diverse/hex_2p.db",
        "data/selfplay/p2p/hexagonal_2p",
    ],
    ("hexagonal", 3): [
        "data/selfplay/diverse/hex_3p.db",
        "data/selfplay/p2p/hexagonal_3p",
    ],
    ("hexagonal", 4): [
        "data/selfplay/diverse/hex_4p.db",
        "data/selfplay/p2p/hexagonal_4p",
    ],
}

# Training thresholds - trigger training when this many NEW games are available
THRESHOLDS: Dict[Tuple[str, int], int] = {
    ("square8", 2): 500,
    ("square8", 3): 200,
    ("square8", 4): 200,
    ("square19", 2): 50,  # Lower thresholds for configs with less data
    ("square19", 3): 50,
    ("square19", 4): 50,
    ("hexagonal", 2): 50,
    ("hexagonal", 3): 50,
    ("hexagonal", 4): 50,
}

# Export settings per config type (games with more moves need higher sample_every)
# Format: (board_type, num_players) -> (max_games, sample_every, epochs)
EXPORT_SETTINGS: Dict[Tuple[str, int], Tuple[int, int, int]] = {
    # Square8: ~100-200 moves per game
    ("square8", 2): (100, 5, 5),
    ("square8", 3): (100, 5, 5),
    ("square8", 4): (100, 5, 5),
    # Square19: ~300-500 moves per game
    ("square19", 2): (50, 20, 5),
    ("square19", 3): (50, 20, 5),
    ("square19", 4): (50, 20, 5),
    # Hexagonal: ~800-1200 moves per game - need aggressive sampling
    ("hexagonal", 2): (30, 50, 5),
    ("hexagonal", 3): (30, 50, 5),
    ("hexagonal", 4): (30, 50, 5),
}

# Default settings if config not found
DEFAULT_MAX_GAMES = 50
DEFAULT_SAMPLE_EVERY = 20
DEFAULT_EPOCHS = 5

# Track last training count per config
last_trained_counts: Dict[Tuple[str, int], int] = {k: 0 for k in THRESHOLDS}


def find_databases(path: str) -> List[str]:
    """Find all .db files in a path (file or directory)."""
    full_path = os.path.join(BASE_DIR, path) if not path.startswith("/") else path

    if os.path.isfile(full_path):
        return [full_path]
    elif os.path.isdir(full_path):
        dbs = []
        for root, _, files in os.walk(full_path):
            for f in files:
                if f.endswith(".db"):
                    dbs.append(os.path.join(root, f))
        return dbs
    return []


def count_games_with_moves(db_path: str, board_type: str, num_players: int) -> int:
    """Count games that have move data (required for training export)."""
    if not os.path.exists(db_path):
        return 0
    try:
        conn = sqlite3.connect(db_path, timeout=10)
        cur = conn.cursor()
        # Only count games that have associated moves
        cur.execute("""
            SELECT COUNT(DISTINCT g.game_id)
            FROM games g
            INNER JOIN game_moves gm ON g.game_id = gm.game_id
            WHERE g.board_type = ? AND g.num_players = ?
            AND COALESCE(g.excluded_from_training, 0) = 0
        """, (board_type, num_players))
        result = cur.fetchone()
        conn.close()
        return result[0] if result else 0
    except Exception as e:
        return 0


def get_config_counts() -> Dict[Tuple[str, int], Tuple[int, str]]:
    """Get game counts for each config, returning (count, best_db_path)."""
    results = {}

    for config, db_paths in CONFIG_DATABASES.items():
        board_type, num_players = config
        total_count = 0
        best_db = None
        best_count = 0

        for path in db_paths:
            for db_path in find_databases(path):
                count = count_games_with_moves(db_path, board_type, num_players)
                total_count += count
                if count > best_count:
                    best_count = count
                    best_db = db_path

        results[config] = (total_count, best_db)

    return results


def run_training(board_type: str, num_players: int, db_path: str, current_count: int) -> bool:
    """Run export and training for a config."""
    key = (board_type, num_players)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = f"{board_type[:3]}{num_players}p"

    # Get config-specific export settings
    max_games, sample_every, epochs = EXPORT_SETTINGS.get(
        key, (DEFAULT_MAX_GAMES, DEFAULT_SAMPLE_EVERY, DEFAULT_EPOCHS)
    )

    print(f"[{ts}] Training {short} from {os.path.basename(db_path)} "
          f"(max_games={max_games}, sample_every={sample_every})...", flush=True)

    # Export training data
    npz = os.path.join(BASE_DIR, f"data/training/selfplay_{short}_{ts}.npz")
    exp_cmd = [
        sys.executable, os.path.join(BASE_DIR, "scripts/export_replay_dataset.py"),
        "--db", db_path,
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--output", npz,
        "--sample-every", str(sample_every),
        "--max-games", str(max_games),
    ]

    print(f"  Exporting {short}...", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR

    # Longer timeout for hex games (1000+ moves per game)
    export_timeout = 1800 if board_type == "hexagonal" else 600

    try:
        r = subprocess.run(exp_cmd, capture_output=True, text=True, timeout=export_timeout, env=env)
        if r.returncode != 0:
            print(f"  Export failed: {r.stderr[:500] if r.stderr else r.stdout[:500]}", flush=True)
            last_trained_counts[key] = current_count  # Skip to avoid repeated failures
            return False
        if "No samples generated" in (r.stdout or ""):
            print(f"  No samples generated, skipping training", flush=True)
            last_trained_counts[key] = current_count
            return False
    except subprocess.TimeoutExpired:
        print(f"  Export timeout after {export_timeout}s", flush=True)
        last_trained_counts[key] = current_count
        return False

    # Verify NPZ was created
    if not os.path.exists(npz):
        print(f"  NPZ file not created, skipping training", flush=True)
        last_trained_counts[key] = current_count
        return False

    # Run training
    run_dir = os.path.join(BASE_DIR, f"data/training/runs/auto_{short}_{ts}")
    train_cmd = [
        sys.executable, os.path.join(BASE_DIR, "scripts/run_nn_training_baseline.py"),
        "--board", board_type,
        "--num-players", str(num_players),
        "--run-dir", run_dir,
        "--data-path", npz,
        "--epochs", str(epochs),
    ]

    print(f"  Training {short} for {epochs} epochs...", flush=True)
    try:
        r = subprocess.run(train_cmd, capture_output=True, text=True, timeout=900, env=env)
        if r.returncode != 0:
            print(f"  Training failed: {r.stderr[:500] if r.stderr else r.stdout[:500]}", flush=True)
            last_trained_counts[key] = current_count
            return False
    except subprocess.TimeoutExpired:
        print(f"  Training timeout after 900s", flush=True)
        last_trained_counts[key] = current_count
        return False

    print(f"  {short} training complete!", flush=True)
    last_trained_counts[key] = current_count
    return True


def main():
    global last_trained_counts

    print("=" * 60, flush=True)
    print("Multi-Config Training Loop v2", flush=True)
    print(f"Base dir: {BASE_DIR}", flush=True)
    print("Configs: " + ", ".join(f"{bt[:3]}{np}p" for bt, np in THRESHOLDS.keys()), flush=True)
    print("=" * 60, flush=True)

    iteration = 0
    while True:
        try:
            iteration += 1
            counts = get_config_counts()
            ts = datetime.now().strftime("%H:%M:%S")

            # Status line
            status_parts = []
            training_candidates = []

            for config, threshold in THRESHOLDS.items():
                board_type, num_players = config
                count, best_db = counts.get(config, (0, None))
                last = last_trained_counts.get(config, 0)
                new_games = count - last
                short = f"{board_type[:3]}{num_players}p"

                if count > 0:
                    status_parts.append(f"{short}:{count}(+{new_games}/{threshold})")

                # Check if ready for training
                if new_games >= threshold and best_db:
                    training_candidates.append((config, best_db, count, new_games))

            print(f"[{ts}] iter={iteration} | {' '.join(status_parts)}", flush=True)

            # Train ONE config per iteration (prioritize configs with most new games)
            if training_candidates:
                training_candidates.sort(key=lambda x: x[3], reverse=True)  # Sort by new_games
                config, best_db, count, new_games = training_candidates[0]
                board_type, num_players = config
                run_training(board_type, num_players, best_db, count)

            time.sleep(60)

        except KeyboardInterrupt:
            print("\nShutting down...", flush=True)
            break
        except Exception as e:
            print(f"Error: {e}", flush=True)
            import traceback
            traceback.print_exc()
            time.sleep(60)


if __name__ == "__main__":
    os.chdir(BASE_DIR)
    main()
