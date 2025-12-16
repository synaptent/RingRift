#!/usr/bin/env python3
"""
Multi-config training loop that monitors game databases AND JSONL files,
triggering training when enough new games are available.

Key improvements:
- Uses config-specific databases with full move data
- Supports JSONL files directly via jsonl_to_npz.py (no DB conversion needed)
- Only counts games that have moves (required for training)
- Supports multiple database AND JSONL sources per config
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
import json
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set

# Base paths - auto-detect from script location or use env var
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("RINGRIFT_BASE_DIR", os.path.dirname(SCRIPT_DIR))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Database sources for each config - databases that have games WITH moves
# Format: (board_type, num_players) -> list of database paths
# Uses canonical selfplay.db and diverse_synced per-config DBs
# NOTE: selfplay_stats.db is for monitoring ONLY (no game_moves table)
# NOTE: jsonl_converted_*.db have NO game_moves table - don't use for training
CONFIG_DATABASES: Dict[Tuple[str, int], List[str]] = {
    # Square8 configs - use selfplay.db (canonical, has moves)
    ("square8", 2): [
        "data/games/selfplay.db",  # Canonical DB with game_moves
    ],
    ("square8", 3): [
        "data/games/selfplay.db",
    ],
    ("square8", 4): [
        "data/games/selfplay.db",
    ],
    # Square19 configs - use diverse_synced per-config DBs
    ("square19", 2): [
        "data/games/selfplay.db",
        "data/selfplay/diverse_synced/square19_2p.db",  # 72 games with moves
    ],
    ("square19", 3): [
        "data/games/selfplay.db",
        "data/selfplay/diverse_synced/square19_3p.db",  # 50 games with moves
    ],
    ("square19", 4): [
        "data/games/selfplay.db",
        "data/selfplay/diverse_synced/square19_4p.db",  # 46 games with moves
    ],
    # Hexagonal configs - use selfplay.db
    ("hexagonal", 2): [
        "data/games/selfplay.db",
    ],
    ("hexagonal", 3): [
        "data/games/selfplay.db",
    ],
    ("hexagonal", 4): [
        "data/games/selfplay.db",
    ],
}

# JSONL source directories for each config
# These are converted directly to NPZ using jsonl_to_npz.py (no DB needed)
# JSONL from tournaments and hybrid selfplay are canonical format
CONFIG_JSONL_DIRS: Dict[Tuple[str, int], List[str]] = {
    # All configs can use canonical JSONL from tournaments
    ("square8", 2): [
        "data/selfplay/canonical",  # Canonical selfplay JSONL
        "data/games",  # Tournament JSONL (games.jsonl, etc.)
    ],
    ("square8", 3): [
        "data/selfplay/canonical",
        "data/games",
    ],
    ("square8", 4): [
        "data/selfplay/canonical",
        "data/games",
    ],
    ("square19", 2): [
        "data/selfplay/canonical",
        "data/games",
    ],
    ("square19", 3): [
        "data/selfplay/canonical",
        "data/games",
    ],
    ("square19", 4): [
        "data/selfplay/canonical",
        "data/games",
    ],
    ("hexagonal", 2): [
        "data/selfplay/canonical",
        "data/games",
    ],
    ("hexagonal", 3): [
        "data/selfplay/canonical",
        "data/games",
    ],
    ("hexagonal", 4): [
        "data/selfplay/canonical",
        "data/games",
    ],
}

# Training thresholds - trigger training when this many NEW games are available
# Set low initially to trigger training quickly, adjust based on game generation rate
THRESHOLDS: Dict[Tuple[str, int], int] = {
    ("square8", 2): 100,   # Square8 has most data, can use higher threshold
    ("square8", 3): 50,
    ("square8", 4): 50,
    ("square19", 2): 30,   # Square19/hex have fewer games, lower thresholds
    ("square19", 3): 30,
    ("square19", 4): 30,
    ("hexagonal", 2): 30,
    ("hexagonal", 3): 30,
    ("hexagonal", 4): 30,
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

# Short names for display - must be unique per board type
BOARD_SHORT_NAMES = {
    "square8": "sq8",
    "square19": "sq19",
    "hexagonal": "hex",
}


def short_name(board_type: str, num_players: int) -> str:
    """Generate a unique short name for a board/player config."""
    prefix = BOARD_SHORT_NAMES.get(board_type, board_type[:4])
    return f"{prefix}_{num_players}p"


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


def find_jsonl_files(path: str) -> List[str]:
    """Find all .jsonl files in a path (file or directory)."""
    full_path = os.path.join(BASE_DIR, path) if not path.startswith("/") else path

    if os.path.isfile(full_path) and full_path.endswith(".jsonl"):
        return [full_path]
    elif os.path.isdir(full_path):
        jsonl_files = []
        for root, _, files in os.walk(full_path):
            for f in files:
                if f.endswith(".jsonl"):
                    jsonl_files.append(os.path.join(root, f))
        return jsonl_files
    return []


def count_jsonl_games(jsonl_path: str, board_type: str, num_players: int,
                       max_lines: int = 10000) -> Tuple[int, Set[str]]:
    """Count games in JSONL file matching board_type and num_players.

    Returns (count, set of game_ids) for deduplication.
    Only counts games with canonical format (has 'moves' array).

    For performance, limits to first max_lines lines (default 10000).
    This is a reasonable estimate for large files.
    """
    if not os.path.exists(jsonl_path):
        return 0, set()

    variants = BOARD_VARIANTS.get(board_type, [board_type])
    count = 0
    game_ids = set()
    lines_read = 0

    try:
        with open(jsonl_path, 'r') as f:
            for line in f:
                lines_read += 1
                if lines_read > max_lines:
                    break  # Stop after max_lines for performance

                line = line.strip()
                if not line:
                    continue
                try:
                    game = json.loads(line)
                    game_board = game.get("board_type", "")
                    game_players = game.get("num_players", 0)
                    game_id = game.get("game_id", "")

                    # Check board type variants
                    board_match = game_board in variants or game_board == board_type

                    # Check players match
                    players_match = game_players == num_players

                    # Must have moves array (canonical format)
                    has_moves = "moves" in game and len(game.get("moves", [])) > 0

                    if board_match and players_match and has_moves:
                        if game_id and game_id not in game_ids:
                            game_ids.add(game_id)
                            count += 1
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        return 0, set()

    return count, game_ids


# Cache for JSONL file metadata to avoid repeated parsing
_jsonl_metadata_cache: Dict[str, Dict[Tuple[str, int], int]] = {}
_jsonl_cache_time: Dict[str, float] = {}
JSONL_CACHE_TTL = 300  # 5 minute cache TTL


def get_jsonl_file_metadata(jsonl_path: str, max_lines: int = 5000) -> Dict[Tuple[str, int], Set[str]]:
    """Parse JSONL file once and return game counts per config.

    Returns dict of (board_type, num_players) -> set of game_ids.
    Cached for 5 minutes to avoid repeated parsing.
    """
    # Check cache
    cache_key = jsonl_path
    if cache_key in _jsonl_cache_time:
        if time.time() - _jsonl_cache_time[cache_key] < JSONL_CACHE_TTL:
            return _jsonl_metadata_cache.get(cache_key, {})

    # Parse file
    result: Dict[Tuple[str, int], Set[str]] = {}

    if not os.path.exists(jsonl_path):
        return result

    try:
        lines_read = 0
        with open(jsonl_path, 'r') as f:
            for line in f:
                lines_read += 1
                if lines_read > max_lines:
                    break

                line = line.strip()
                if not line:
                    continue

                try:
                    game = json.loads(line)
                    game_board = game.get("board_type", "")
                    game_players = game.get("num_players", 0)
                    game_id = game.get("game_id", "")
                    has_moves = "moves" in game and len(game.get("moves", [])) > 0

                    if not has_moves or not game_id:
                        continue

                    # Normalize board type
                    board_type = None
                    if "hex" in game_board.lower():
                        board_type = "hexagonal"
                    elif "square19" in game_board.lower() or "sq19" in game_board.lower():
                        board_type = "square19"
                    elif "square8" in game_board.lower() or "sq8" in game_board.lower():
                        board_type = "square8"
                    else:
                        board_type = game_board

                    if board_type and game_players in (2, 3, 4):
                        key = (board_type, game_players)
                        if key not in result:
                            result[key] = set()
                        result[key].add(game_id)

                except json.JSONDecodeError:
                    continue
    except Exception:
        pass

    # Update cache
    _jsonl_metadata_cache[cache_key] = result
    _jsonl_cache_time[cache_key] = time.time()

    return result


def get_jsonl_counts(board_type: str, num_players: int) -> Tuple[int, List[str]]:
    """Get total JSONL game counts for a config, returning (total_count, jsonl_files_with_games)."""
    config = (board_type, num_players)
    jsonl_dirs = CONFIG_JSONL_DIRS.get(config, [])

    total_count = 0
    jsonl_with_games = []
    seen_game_ids: Set[str] = set()  # Dedupe across files

    for path in jsonl_dirs:
        for jsonl_path in find_jsonl_files(path):
            # Use cached metadata
            metadata = get_jsonl_file_metadata(jsonl_path)
            game_ids = metadata.get(config, set())

            # Only count games not seen before (deduplication)
            new_ids = game_ids - seen_game_ids
            if new_ids:
                total_count += len(new_ids)
                seen_game_ids.update(new_ids)
                jsonl_with_games.append(jsonl_path)

    return total_count, jsonl_with_games


def count_games_with_moves(db_path: str, board_type: str, num_players: int) -> int:
    """Count games that have move data and are not excluded from training.

    This must match the filtering in export_replay_dataset.py which uses
    GameReplayDB.query_games with exclude_training_excluded=True (default).
    """
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

        # Check if excluded_from_training column exists
        cur.execute("PRAGMA table_info(games)")
        columns = {row[1] for row in cur.fetchall()}
        has_excluded_col = "excluded_from_training" in columns

        # Try with different board type name variants
        variants = BOARD_VARIANTS.get(board_type, [board_type])
        total = 0
        for variant in variants:
            try:
                # Count games with moves, excluding training-excluded games
                # COALESCE handles NULL values (treat as not excluded)
                if has_excluded_col:
                    cur.execute("""
                        SELECT COUNT(DISTINCT g.game_id)
                        FROM games g
                        INNER JOIN game_moves gm ON g.game_id = gm.game_id
                        WHERE g.board_type = ? AND g.num_players = ?
                        AND COALESCE(g.excluded_from_training, 0) = 0
                    """, (variant, num_players))
                else:
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


def get_config_counts() -> Dict[Tuple[str, int], Tuple[int, List[str], int, List[str]]]:
    """Get game counts for each config from both DB and JSONL sources.

    Returns dict of config -> (db_count, db_paths, jsonl_count, jsonl_paths).
    """
    results = {}

    for config, db_paths in CONFIG_DATABASES.items():
        board_type, num_players = config
        db_count = 0
        dbs_with_games = []

        # Count DB games
        for path in db_paths:
            for db_path in find_databases(path):
                count = count_games_with_moves(db_path, board_type, num_players)
                if count > 0:
                    db_count += count
                    dbs_with_games.append(db_path)

        # Count JSONL games
        jsonl_count, jsonl_with_games = get_jsonl_counts(board_type, num_players)

        results[config] = (db_count, dbs_with_games, jsonl_count, jsonl_with_games)

    return results


def run_training(board_type: str, num_players: int, db_paths: List[str],
                  jsonl_paths: List[str], current_count: int) -> bool:
    """Run export and training for a config using DB and/or JSONL sources.

    Supports three modes:
    1. DB only: Uses export_replay_dataset.py
    2. JSONL only: Uses jsonl_to_npz.py (faster, no DB needed)
    3. Both: Exports both, merges NPZ files
    """
    key = (board_type, num_players)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = short_name(board_type, num_players)

    # Get config-specific export settings
    max_games, sample_every, epochs = EXPORT_SETTINGS.get(
        key, (DEFAULT_MAX_GAMES, DEFAULT_SAMPLE_EVERY, DEFAULT_EPOCHS)
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR

    # Longer timeout for hex games (1000+ moves per game)
    export_timeout = 1800 if board_type == "hexagonal" else 600

    # Output NPZ path
    npz = os.path.join(BASE_DIR, f"data/training/selfplay_{short}_{ts}.npz")
    os.makedirs(os.path.dirname(npz), exist_ok=True)

    # Determine export mode and log sources
    has_db = len(db_paths) > 0
    has_jsonl = len(jsonl_paths) > 0

    source_info = []
    if has_db:
        db_names = [os.path.basename(p) for p in db_paths[:3]]
        if len(db_paths) > 3:
            db_names.append(f"...+{len(db_paths)-3}")
        source_info.append(f"{len(db_paths)} DB(s): {', '.join(db_names)}")
    if has_jsonl:
        jsonl_names = [os.path.basename(p) for p in jsonl_paths[:3]]
        if len(jsonl_paths) > 3:
            jsonl_names.append(f"...+{len(jsonl_paths)-3}")
        source_info.append(f"{len(jsonl_paths)} JSONL(s): {', '.join(jsonl_names)}")

    print(f"[{ts}] Training {short} from {' + '.join(source_info)} "
          f"(max_games={max_games}, sample_every={sample_every})...", flush=True)

    # Export based on available sources
    npz_files = []

    # Export from JSONL if available (preferred - faster, no DB overhead)
    if has_jsonl:
        jsonl_npz = os.path.join(BASE_DIR, f"data/training/jsonl_{short}_{ts}.npz")
        exp_cmd = [
            sys.executable, os.path.join(BASE_DIR, "scripts/jsonl_to_npz.py"),
            "--output", jsonl_npz,
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--sample-every", str(sample_every),
            "--max-games", str(max_games),
            "--gpu-selfplay",  # Use GPU selfplay format (simplified move sequences)
        ]
        # Add each JSONL file as --input
        for jsonl_path in jsonl_paths:
            exp_cmd.extend(["--input", jsonl_path])

        print(f"  Exporting from JSONL...", flush=True)
        try:
            r = subprocess.run(exp_cmd, capture_output=True, text=True, timeout=export_timeout, env=env)
            if r.returncode == 0 and os.path.exists(jsonl_npz):
                npz_files.append(jsonl_npz)
                print(f"  JSONL export complete", flush=True)
            else:
                print(f"  JSONL export failed: {r.stderr[:300] if r.stderr else r.stdout[:300]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  JSONL export timeout", flush=True)

    # Export from DB if available (and JSONL didn't produce enough)
    if has_db:
        db_npz = os.path.join(BASE_DIR, f"data/training/db_{short}_{ts}.npz")
        exp_cmd = [
            sys.executable, os.path.join(BASE_DIR, "scripts/export_replay_dataset.py"),
            "--board-type", board_type,
            "--num-players", str(num_players),
            "--output", db_npz,
            "--sample-every", str(sample_every),
            "--max-games", str(max_games),
        ]
        for db_path in db_paths:
            exp_cmd.extend(["--db", db_path])

        print(f"  Exporting from DB...", flush=True)
        try:
            r = subprocess.run(exp_cmd, capture_output=True, text=True, timeout=export_timeout, env=env)
            if r.returncode == 0 and os.path.exists(db_npz):
                npz_files.append(db_npz)
                print(f"  DB export complete", flush=True)
            else:
                print(f"  DB export failed: {r.stderr[:300] if r.stderr else r.stdout[:300]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  DB export timeout", flush=True)

    # Check if we have any data
    if not npz_files:
        print(f"  No data exported, skipping training", flush=True)
        last_trained_counts[key] = current_count
        return False

    # If multiple NPZ files, use the first one (TODO: merge them)
    # For now, prefer JSONL (first in list) as it's canonical format
    final_npz = npz_files[0]
    if len(npz_files) > 1:
        print(f"  Using {os.path.basename(final_npz)} (TODO: merge {len(npz_files)} sources)", flush=True)

    # Rename to final output path
    if final_npz != npz:
        os.rename(final_npz, npz)

    # Clean up other NPZ files
    for f in npz_files[1:]:
        if os.path.exists(f):
            os.remove(f)

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
    print("Multi-Config Training Loop v5 (BALANCED + JSONL)", flush=True)
    print(f"Base dir: {BASE_DIR}", flush=True)
    print("Configs: " + ", ".join(short_name(bt, np) for bt, np in THRESHOLDS.keys()), flush=True)
    print("Mode: BALANCE - prioritizes least-represented combos", flush=True)
    print("Sources: DB + JSONL (tournaments, canonical selfplay)", flush=True)
    print("Note: Excludes games marked excluded_from_training=1", flush=True)
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
                # New format: (db_count, db_paths, jsonl_count, jsonl_paths)
                db_count, db_paths, jsonl_count, jsonl_paths = counts.get(config, (0, [], 0, []))
                total_count = db_count + jsonl_count
                last = last_trained_counts.get(config, 0)
                new_games = total_count - last
                models = model_counts.get(config, 0)
                sn = short_name(board_type, num_players)

                if total_count > 0:
                    # Show games (DB+JSONL) and model count: hex_2p:1500(+100/50)M:2
                    if jsonl_count > 0:
                        status_parts.append(f"{sn}:{db_count}+{jsonl_count}j(+{new_games}/{threshold})M:{models}")
                    else:
                        status_parts.append(f"{sn}:{total_count}(+{new_games}/{threshold})M:{models}")

                # Check if ready for training (need at least some data paths)
                if new_games >= threshold and (db_paths or jsonl_paths):
                    training_candidates.append((config, db_paths, jsonl_paths, total_count, new_games, models))

            print(f"[{ts}] iter={iteration} | {' '.join(status_parts)}", flush=True)

            # BALANCE MODE: Prioritize configs with FEWEST trained models
            # This ensures we balance training across all 9 board/player combinations
            # Ties broken by most new games available
            if training_candidates:
                # Sort by: (1) fewest models ASCENDING, (2) most new games DESCENDING
                training_candidates.sort(key=lambda x: (x[5], -x[4]))
                config, db_paths, jsonl_paths, total_count, new_games, models = training_candidates[0]
                board_type, num_players = config
                sn = short_name(board_type, num_players)
                print(f"[{ts}] BALANCE: Training {sn} (has only {models} models, {new_games} new games)", flush=True)
                run_training(board_type, num_players, db_paths, jsonl_paths, total_count)

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
