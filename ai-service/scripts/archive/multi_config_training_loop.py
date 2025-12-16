#!/usr/bin/env python3
"""
Multi-config training loop that monitors game databases and triggers training
when enough new games are available for each board/player configuration.

Key improvements:
- Uses config-specific databases with full move data
- Only counts games that have moves (required for training)
- Supports multiple database sources per config
- More efficient database queries
- BALANCE MODE: Prioritizes least-represented board/player combos
- Counts existing trained models per config for balanced training
"""
import os
import sys
import time
import sqlite3
import subprocess
import glob
import re
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Base paths - auto-detect from script location or use env var
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("RINGRIFT_BASE_DIR", os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Database sources for each config - databases that have games WITH moves
# Format: (board_type, num_players) -> list of database paths
# Updated to use verified canonical DBs with game_moves table
CONFIG_DATABASES: Dict[Tuple[str, int], List[str]] = {
    # NOTE: data/games/selfplay.db is the PRIMARY source - it receives synced
    # game data WITH game_moves from all cluster nodes via simple_game_sync.py
    ("square8", 2): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/canonical/canonical_square8_2p.db",
        "data/selfplay/5090_imports/5090_quad_selfplay.db",
        "data/selfplay/mcts_nn_v5/square8_2p.db",
        "data/selfplay/fallback",
        "data/selfplay/vast_sync",
    ],
    ("square8", 3): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/selfplay/vast_sync/ssh1_4060ti/p2p/square8_3p",
        "data/selfplay/vast_sync/ssh1_4060ti/p2p_hybrid/square8_3p",
    ],
    ("square8", 4): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/selfplay/vast_sync",
    ],
    ("square19", 2): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/selfplay/diverse/square19_2p.db",
        "data/selfplay/diverse_synced/square19_2p.db",
    ],
    ("square19", 3): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/selfplay/diverse/square19_3p.db",
        "data/selfplay/diverse_synced/square19_3p.db",
        "data/selfplay/vast_sync/ssh1_4060ti/p2p/square19_3p",
    ],
    ("square19", 4): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/selfplay/diverse/square19_4p.db",
        "data/selfplay/diverse_synced/square19_4p.db",
    ],
    ("hexagonal", 2): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/selfplay/vast_sync",
    ],
    ("hexagonal", 3): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/selfplay/vast_sync/ssh1_4060ti/p2p/hexagonal_3p",
    ],
    ("hexagonal", 4): [
        "data/games/selfplay.db",  # Primary - synced with game_moves
        "data/selfplay/vast_sync",
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

# Export settings per config: (max_games, sample_every, epochs)
# Replay bug is now fixed so all moves can be replayed correctly
EXPORT_SETTINGS: Dict[Tuple[str, int], Tuple[int, int, int]] = {
    # Square8: ~100-200 moves per game
    ("square8", 2): (50, 5, 5),
    ("square8", 3): (50, 5, 5),
    ("square8", 4): (50, 5, 5),
    # Square19: ~300-500 moves per game
    ("square19", 2): (30, 5, 5),
    ("square19", 3): (30, 5, 5),
    ("square19", 4): (30, 5, 5),
    # Hexagonal: ~800-1200 moves per game
    ("hexagonal", 2): (50, 5, 5),
    ("hexagonal", 3): (50, 5, 5),
    ("hexagonal", 4): (50, 5, 5),
}

# Default settings if config not found
DEFAULT_MAX_GAMES = 50
DEFAULT_SAMPLE_EVERY = 20
DEFAULT_EPOCHS = 5

# Track last training count per config
last_trained_counts: Dict[Tuple[str, int], int] = {k: 0 for k in THRESHOLDS}

# Board type name variants for model file matching
BOARD_VARIANTS = {
    "square8": ["square8", "sq8"],
    "square19": ["square19", "sq19"],
    "hexagonal": ["hex", "hexagonal"],
}


def count_trained_models(board_type: str, num_players: int) -> int:
    """Count existing trained models for a specific board/player config.

    Searches the models directory for .pth files matching the config pattern.
    Returns count of unique model files (not checkpoints).
    """
    models_dir = os.path.join(BASE_DIR, "models")
    if not os.path.exists(models_dir):
        return 0

    variants = BOARD_VARIANTS.get(board_type, [board_type])
    count = 0
    seen_bases = set()

    for variant in variants:
        # Pattern: {variant}_{num_players}p*.pth
        pattern = os.path.join(models_dir, f"*{variant}*{num_players}p*.pth")
        for pth_file in glob.glob(pattern):
            # Normalize to base model (strip checkpoint suffixes like _20251215_123456)
            base = os.path.basename(pth_file)
            # Remove timestamp suffixes to count unique models
            base_clean = re.sub(r'_\d{8}_\d{6}\.pth$', '.pth', base)
            if base_clean not in seen_bases:
                seen_bases.add(base_clean)
                count += 1

    return count


def get_model_counts() -> Dict[Tuple[str, int], int]:
    """Get count of trained models for each config."""
    return {config: count_trained_models(board_type, num_players)
            for config in THRESHOLDS
            for board_type, num_players in [config]}


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

        # Check if game_moves table exists
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='game_moves'")
        if not cur.fetchone():
            conn.close()
            return 0

        # Try with different board type name variants
        variants = BOARD_VARIANTS.get(board_type, [board_type])
        total = 0
        for variant in variants:
            try:
                # Try query without excluded_from_training first (more compatible)
                cur.execute("""
                    SELECT COUNT(DISTINCT g.game_id)
                    FROM games g
                    INNER JOIN game_moves gm ON g.game_id = gm.game_id
                    WHERE g.board_type = ? AND g.num_players = ?
                """, (variant, num_players))
                result = cur.fetchone()
                if result and result[0] > 0:
                    total += result[0]
            except Exception:
                pass
        conn.close()
        return total
    except Exception as e:
        return 0


def get_config_counts() -> Dict[Tuple[str, int], Tuple[int, List[str]]]:
    """Get game counts for each config, returning (total_count, all_db_paths_with_games)."""
    results = {}

    for config, db_paths in CONFIG_DATABASES.items():
        board_type, num_players = config
        total_count = 0
        dbs_with_games = []

        for path in db_paths:
            for db_path in find_databases(path):
                count = count_games_with_moves(db_path, board_type, num_players)
                if count > 0:
                    total_count += count
                    dbs_with_games.append(db_path)

        results[config] = (total_count, dbs_with_games)

    return results


def run_training(board_type: str, num_players: int, db_paths: List[str], current_count: int) -> bool:
    """Run export and training for a config using multiple data sources with deduplication."""
    key = (board_type, num_players)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = f"{board_type[:3]}{num_players}p"

    # Get config-specific export settings
    max_games, sample_every, epochs = EXPORT_SETTINGS.get(
        key, (DEFAULT_MAX_GAMES, DEFAULT_SAMPLE_EVERY, DEFAULT_EPOCHS)
    )

    db_names = [os.path.basename(p) for p in db_paths[:3]]  # Show first 3
    if len(db_paths) > 3:
        db_names.append(f"...+{len(db_paths)-3} more")
    print(f"[{ts}] Training {short} from {len(db_paths)} DB(s): {', '.join(db_names)} "
          f"(max_games={max_games}, sample_every={sample_every})...", flush=True)

    # Export training data from all sources with deduplication
    npz = os.path.join(BASE_DIR, f"data/training/selfplay_{short}_{ts}.npz")
    exp_cmd = [
        sys.executable, os.path.join(BASE_DIR, "scripts/export_replay_dataset.py"),
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--output", npz,
        "--sample-every", str(sample_every),
        "--max-games", str(max_games),
    ]
    # Add all database paths with separate --db arguments
    for db_path in db_paths:
        exp_cmd.extend(["--db", db_path])

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
    print("Multi-Config Training Loop v3 (BALANCED)", flush=True)
    print(f"Base dir: {BASE_DIR}", flush=True)
    print("Configs: " + ", ".join(f"{bt[:3]}{np}p" for bt, np in THRESHOLDS.keys()), flush=True)
    print("Mode: BALANCE - prioritizes least-represented combos", flush=True)
    print("=" * 60, flush=True)

    iteration = 0
    while True:
        try:
            iteration += 1
            counts = get_config_counts()
            model_counts = get_model_counts()
            ts = datetime.now().strftime("%H:%M:%S")

            # Status line with model counts
            status_parts = []
            training_candidates = []

            for config, threshold in THRESHOLDS.items():
                board_type, num_players = config
                count, db_paths = counts.get(config, (0, []))
                last = last_trained_counts.get(config, 0)
                new_games = count - last
                models = model_counts.get(config, 0)
                short = f"{board_type[:3]}{num_players}p"

                if count > 0:
                    # Show games and model count: hex2p:1500(+100/50)[3db]M:2
                    status_parts.append(f"{short}:{count}(+{new_games}/{threshold})M:{models}")

                # Check if ready for training
                if new_games >= threshold and db_paths:
                    training_candidates.append((config, db_paths, count, new_games, models))

            print(f"[{ts}] iter={iteration} | {' '.join(status_parts)}", flush=True)

            # BALANCE MODE: Prioritize configs with FEWEST trained models
            # This ensures we balance training across all 9 board/player combinations
            # Ties broken by most new games available
            if training_candidates:
                # Sort by: (1) fewest models ASCENDING, (2) most new games DESCENDING
                training_candidates.sort(key=lambda x: (x[4], -x[3]))
                config, db_paths, count, new_games, models = training_candidates[0]
                board_type, num_players = config
                short = f"{board_type[:3]}{num_players}p"
                print(f"[{ts}] BALANCE: Training {short} (has only {models} models, {new_games} new games)", flush=True)
                run_training(board_type, num_players, db_paths, count)

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
