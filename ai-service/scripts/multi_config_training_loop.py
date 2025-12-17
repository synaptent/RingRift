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
- ADAPTIVE CURRICULUM: Prioritizes configs with lower ELO (weaker models)
- OPTIMIZED HYPERPARAMETERS: Uses tuned hyperparameters from config/hyperparameters.json

Version: v7 (OPTIMIZED HP + ADAPTIVE CURRICULUM)

Environment Variables:
    RINGRIFT_ENABLE_AUTO_HP_TUNING: Set to "1" to enable automatic HP tuning (default: disabled)
    RINGRIFT_MIN_GAMES_FOR_HP_TUNING: Min games before HP tuning is considered (default: 500)
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
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set, Any

# Try to import advanced training features
try:
    from app.training.feedback_accelerator import (
        get_feedback_accelerator,
        FeedbackAccelerator,
    )
    HAS_FEEDBACK_ACCELERATOR = True
except ImportError:
    HAS_FEEDBACK_ACCELERATOR = False

try:
    from app.config.hyperparameters import (
        get_hyperparameters,
        is_optimized,
        needs_tuning,
        get_hyperparameter_info,
        get_all_configs as get_all_hp_configs,
    )
    HAS_HYPERPARAMETERS = True
except ImportError:
    HAS_HYPERPARAMETERS = False

# Base paths - auto-detect from script location or use env var
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.environ.get("RINGRIFT_BASE_DIR", os.path.dirname(SCRIPT_DIR))

# HP tuning settings
ENABLE_AUTO_HP_TUNING = os.environ.get("RINGRIFT_ENABLE_AUTO_HP_TUNING", "0") == "1"
MIN_GAMES_FOR_HP_TUNING = int(os.environ.get("RINGRIFT_MIN_GAMES_FOR_HP_TUNING", "500"))

# Policy training with auto KL loss (uses MCTS visit distributions when available)
# Enabled by default as of 2025-12-17 - auto-detects and falls back gracefully
ENABLE_POLICY_TRAINING = os.environ.get("RINGRIFT_ENABLE_POLICY_TRAINING", "1") == "1"
POLICY_AUTO_KL_LOSS = os.environ.get("RINGRIFT_POLICY_AUTO_KL_LOSS", "1") == "1"  # Auto-detect KL loss
POLICY_KL_MIN_COVERAGE = float(os.environ.get("RINGRIFT_POLICY_KL_MIN_COVERAGE", "0.3"))  # 30% MCTS coverage threshold
POLICY_KL_MIN_SAMPLES = int(os.environ.get("RINGRIFT_POLICY_KL_MIN_SAMPLES", "50"))  # Min samples with MCTS policy

# Track HP tuning recommendations
_hp_tuning_recommendations: Dict[Tuple[str, int], bool] = {}
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
    # Hex8 (radius 4 hexagonal) configs - use policy_c selfplay
    ("hex8", 2): [
        "data/selfplay/hex8_policy_c/games.db",
    ],
    ("hex8", 3): [
        "data/selfplay/hex8_policy_c/games.db",
    ],
    ("hex8", 4): [
        "data/selfplay/hex8_policy_c/games.db",
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
    # Hex8 configs - use hex8-specific selfplay directories
    ("hex8", 2): [
        "data/selfplay/hex8_policy_c",
    ],
    ("hex8", 3): [
        "data/selfplay/hex8_policy_c",
    ],
    ("hex8", 4): [
        "data/selfplay/hex8_policy_c",
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
    # Hex8 thresholds - start low to get initial models quickly
    ("hex8", 2): 30,
    ("hex8", 3): 30,
    ("hex8", 4): 30,
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
    # Hex8 (radius 4): ~100-200 moves per game
    ("hex8", 2): (50, 5, 5),
    ("hex8", 3): (50, 5, 5),
    ("hex8", 4): (50, 5, 5),
}

# Default settings if config not found
DEFAULT_MAX_GAMES = 50
DEFAULT_SAMPLE_EVERY = 20
DEFAULT_EPOCHS = 5

# Track last training count per config
last_trained_counts: Dict[Tuple[str, int], int] = {k: 0 for k in THRESHOLDS}

# Unified ELO database path
UNIFIED_ELO_DB = os.path.join(DATA_DIR, "unified_elo.db")

# Board type name variants for model file matching
BOARD_VARIANTS = {
    "square8": ["square8", "sq8"],
    "square19": ["square19", "sq19"],
    "hexagonal": ["hex", "hexagonal"],
    "hex8": ["hex8"],
}

# Short names for display - must be unique per board type
BOARD_SHORT_NAMES = {
    "square8": "sq8",
    "square19": "sq19",
    "hexagonal": "hex",
    "hex8": "hex8",
}


def short_name(board_type: str, num_players: int) -> str:
    """Generate a unique short name for a board/player config."""
    prefix = BOARD_SHORT_NAMES.get(board_type, board_type[:4])
    return f"{prefix}_{num_players}p"


def get_config_elo(board_type: str, num_players: int) -> float:
    """Get the best ELO rating for a config from unified ELO database.

    Returns 1500.0 (default ELO) if no ratings found or database unavailable.
    Lower ELO means the model is weaker and should get more training attention.
    """
    if not os.path.exists(UNIFIED_ELO_DB):
        return 1500.0

    try:
        conn = sqlite3.connect(UNIFIED_ELO_DB)
        cursor = conn.execute("""
            SELECT MAX(rating) FROM elo_ratings
            WHERE board_type = ? AND num_players = ?
        """, (board_type, num_players))
        row = cursor.fetchone()
        conn.close()

        if row and row[0]:
            return float(row[0])
    except Exception:
        pass

    return 1500.0  # Default ELO


def check_nas_results(board_type: str, num_players: int) -> Optional[Dict[str, Any]]:
    """Check if there are NAS results available for this config.

    Returns best architecture params if found, None otherwise.
    """
    nas_dir = Path(BASE_DIR) / "logs" / "nas"
    if not nas_dir.exists():
        return None

    # Look for best_architecture.json files
    best_arch = None
    best_perf = 0.0

    for run_dir in nas_dir.iterdir():
        if not run_dir.is_dir():
            continue

        best_file = run_dir / "best_architecture.json"
        if not best_file.exists():
            continue

        try:
            with open(best_file) as f:
                data = json.load(f)

            # Check if this is for our config
            state_file = run_dir / "nas_state.json"
            if state_file.exists():
                with open(state_file) as f:
                    state = json.load(f)
                if state.get("board_type") != board_type or state.get("num_players") != num_players:
                    continue

            if data.get("performance", 0) > best_perf:
                best_perf = data["performance"]
                best_arch = data.get("params")

        except Exception:
            continue

    return best_arch


def check_nas_recommendation(board_type: str, num_players: int) -> Optional[str]:
    """Check if NAS should be recommended for this config.

    Returns recommendation message if NAS is recommended.
    """
    config_key = f"{board_type}_{num_players}p"

    # Check if ELO has plateaued
    if HAS_FEEDBACK_ACCELERATOR:
        try:
            accelerator = get_feedback_accelerator()
            config_momentum = accelerator._configs.get(config_key)
            if config_momentum and config_momentum.consecutive_plateaus >= 3:
                return (
                    f"NAS RECOMMENDED for {config_key} (ELO plateau detected). Run:\n"
                    f"  python scripts/launch_distributed_nas.py --board {board_type} --players {num_players} --strategy evolutionary --generations 50"
                )
        except Exception:
            pass

    # Check for existing NAS results
    existing_arch = check_nas_results(board_type, num_players)
    if existing_arch:
        return None  # Already have NAS results

    # Check if this is a main config with lots of data
    if board_type == "square8" and num_players == 2:
        return (
            f"NAS available for {config_key} (main config). Run:\n"
            f"  python scripts/launch_distributed_nas.py --board {board_type} --players {num_players} --strategy evolutionary --generations 50"
        )

    return None


def check_hp_tuning_recommendation(board_type: str, num_players: int, total_games: int) -> Optional[str]:
    """Check if HP tuning should be recommended for a config.

    Returns a recommendation message if HP tuning is recommended, None otherwise.
    """
    config_key = (board_type, num_players)

    # Skip if already recommended this session
    if _hp_tuning_recommendations.get(config_key):
        return None

    # Skip if HP module not available
    if not HAS_HYPERPARAMETERS:
        return None

    # Check if config needs tuning
    try:
        if not needs_tuning(board_type, num_players, min_confidence="medium"):
            return None  # Already optimized
    except Exception:
        return None

    # Check if enough games for tuning
    if total_games < MIN_GAMES_FOR_HP_TUNING:
        return None

    # Mark as recommended
    _hp_tuning_recommendations[config_key] = True

    recommendation = (
        f"HP TUNING RECOMMENDED for {board_type}_{num_players}p "
        f"({total_games} games available). Run:\n"
        f"  python scripts/tune_hyperparameters.py --board {board_type} --players {num_players} --trials 30"
    )

    return recommendation


def trigger_hp_tuning(board_type: str, num_players: int, trials: int = 20) -> bool:
    """Trigger HP tuning for a config. Returns True if started successfully."""
    if not ENABLE_AUTO_HP_TUNING:
        return False

    tune_cmd = [
        sys.executable, os.path.join(BASE_DIR, "scripts/tune_hyperparameters.py"),
        "--board", board_type,
        "--players", str(num_players),
        "--trials", str(trials),
    ]

    try:
        # Start in background
        subprocess.Popen(
            tune_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(f"  [HP Tuning] Started background tuning for {board_type}_{num_players}p", flush=True)
        return True
    except Exception as e:
        print(f"  [HP Tuning] Failed to start: {e}", flush=True)
        return False


def get_all_config_elos() -> Dict[Tuple[str, int], float]:
    """Get ELO ratings for all configs at once for efficiency."""
    elos = {k: 1500.0 for k in THRESHOLDS}

    if not os.path.exists(UNIFIED_ELO_DB):
        return elos

    try:
        conn = sqlite3.connect(UNIFIED_ELO_DB)
        cursor = conn.execute("""
            SELECT board_type, num_players, MAX(rating) as best_elo
            FROM elo_ratings
            GROUP BY board_type, num_players
        """)
        for row in cursor:
            key = (row[0], row[1])
            if key in elos and row[2]:
                elos[key] = float(row[2])
        conn.close()
    except Exception:
        pass

    return elos


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
                  jsonl_paths: List[str], current_count: int, iteration: int = 0) -> bool:
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

        # Use parallel export for hex boards (10-20x faster)
        use_parallel = board_type == "hexagonal"
        parallel_script = os.path.join(BASE_DIR, "scripts/export_replay_dataset_parallel.py")

        if use_parallel and os.path.exists(parallel_script):
            exp_cmd = [
                sys.executable, parallel_script,
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--output", db_npz,
                "--sample-every", str(sample_every),
                "--max-games", str(max_games),
                "--encoder-version", "v3",
            ]
            export_mode = "DB (parallel)"
        else:
            exp_cmd = [
                sys.executable, os.path.join(BASE_DIR, "scripts/export_replay_dataset.py"),
                "--board-type", board_type,
                "--num-players", str(num_players),
                "--output", db_npz,
                "--sample-every", str(sample_every),
                "--max-games", str(max_games),
            ]
            export_mode = "DB"

        for db_path in db_paths:
            exp_cmd.extend(["--db", db_path])

        print(f"  Exporting from {export_mode}...", flush=True)
        try:
            r = subprocess.run(exp_cmd, capture_output=True, text=True, timeout=export_timeout, env=env)
            if r.returncode == 0 and os.path.exists(db_npz):
                npz_files.append(db_npz)
                print(f"  {export_mode} export complete", flush=True)
            else:
                print(f"  {export_mode} export failed: {r.stderr[:300] if r.stderr else r.stdout[:300]}", flush=True)
        except subprocess.TimeoutExpired:
            print(f"  {export_mode} export timeout", flush=True)

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

    # Run training with optimized hyperparameters
    run_dir = os.path.join(BASE_DIR, f"data/training/runs/auto_{short}_{ts}")

    # Get adaptive training intensity from feedback accelerator
    epochs_multiplier = 1.0
    lr_multiplier = 1.0
    intensity_status = "normal"

    if HAS_FEEDBACK_ACCELERATOR:
        try:
            config_key = f"{board_type}_{num_players}p"
            accelerator = get_feedback_accelerator()
            decision = accelerator.get_training_decision(config_key)
            epochs_multiplier = decision.epochs_multiplier
            lr_multiplier = decision.learning_rate_multiplier
            intensity_status = decision.intensity.value
            print(f"  [Feedback] {config_key}: intensity={intensity_status}, "
                  f"epochs_mult={epochs_multiplier:.1f}, lr_mult={lr_multiplier:.1f}", flush=True)
        except Exception as e:
            print(f"  [Feedback] Warning: {e}", flush=True)

    # Apply intensity multipliers
    adjusted_epochs = max(1, int(epochs * epochs_multiplier))

    train_cmd = [
        sys.executable, os.path.join(BASE_DIR, "scripts/run_nn_training_baseline.py"),
        "--board", board_type,
        "--num-players", str(num_players),
        "--run-dir", run_dir,
        "--data-path", npz,
        "--epochs", str(adjusted_epochs),
        "--use-optimized-hyperparams",  # Load tuned HP from config/hyperparameters.json
    ]

    # Apply LR multiplier if non-default
    if lr_multiplier != 1.0 and HAS_HYPERPARAMETERS:
        try:
            hp = get_hyperparameters(board_type, num_players)
            base_lr = hp.get("learning_rate", 0.001)
            adjusted_lr = base_lr * lr_multiplier
            train_cmd.extend(["--learning-rate", f"{adjusted_lr:.6f}"])
        except Exception:
            pass

    # Check if this config has optimized hyperparameters
    hp_status = "defaults"
    if HAS_HYPERPARAMETERS:
        try:
            hp_info = get_hyperparameter_info(board_type, num_players)
            if hp_info.get("optimized"):
                hp_status = f"OPTIMIZED ({hp_info.get('tuning_method', 'unknown')})"
        except Exception:
            pass

    print(f"  Training {short} for {adjusted_epochs} epochs (HP: {hp_status}, intensity: {intensity_status})...", flush=True)
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

    # Check if HP tuning should be recommended
    hp_recommendation = check_hp_tuning_recommendation(board_type, num_players, current_count)
    if hp_recommendation:
        print(f"\n  ⚡ {hp_recommendation}\n", flush=True)
        # Optionally trigger auto HP tuning if enabled
        if ENABLE_AUTO_HP_TUNING:
            trigger_hp_tuning(board_type, num_players, trials=20)

    # Check for NAS results or recommendations (only occasionally)
    if iteration % 10 == 0:  # Check every 10 training iterations
        nas_arch = check_nas_results(board_type, num_players)
        if nas_arch:
            print(f"  🧬 NAS architecture available for {short}: {nas_arch.get('num_res_blocks', '?')} blocks, "
                  f"{nas_arch.get('channels', '?')} channels", flush=True)
        else:
            nas_rec = check_nas_recommendation(board_type, num_players)
            if nas_rec:
                print(f"\n  🧬 {nas_rec}\n", flush=True)

    return True


def run_policy_training(board_type: str, num_players: int, db_paths: List[str],
                         jsonl_paths: List[str], current_count: int, iteration: int = 0) -> bool:
    """Run policy training with auto KL loss detection.

    This trains a policy network using MCTS visit distributions when available.
    Gracefully falls back to cross-entropy loss if insufficient MCTS data.
    """
    if not ENABLE_POLICY_TRAINING:
        return False

    key = (board_type, num_players)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = short_name(board_type, num_players)

    # Get config-specific export settings
    max_games, sample_every, epochs = EXPORT_SETTINGS.get(
        key, (DEFAULT_MAX_GAMES, DEFAULT_SAMPLE_EVERY, DEFAULT_EPOCHS)
    )

    env = os.environ.copy()
    env["PYTHONPATH"] = BASE_DIR

    # Build policy training command
    run_dir = os.path.join(BASE_DIR, f"data/training/runs/policy_{short}_{ts}")

    train_cmd = [
        sys.executable, os.path.join(BASE_DIR, "scripts/train_nnue_policy.py"),
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--epochs", str(epochs),
        "--save-path", os.path.join(BASE_DIR, f"models/policy_{short}_{ts}.pth"),
        "--sample-every-n", str(sample_every),
        "--max-samples", str(max_games * 50),  # Approximate samples per game
    ]

    # Add database sources
    for db_path in db_paths:
        train_cmd.extend(["--db", db_path])

    # Add auto KL loss flags
    if POLICY_AUTO_KL_LOSS:
        train_cmd.append("--auto-kl-loss")
        train_cmd.extend(["--kl-min-coverage", str(POLICY_KL_MIN_COVERAGE)])
        train_cmd.extend(["--kl-min-samples", str(POLICY_KL_MIN_SAMPLES)])

    print(f"[{ts}] Policy training {short} with auto-KL (coverage>={POLICY_KL_MIN_COVERAGE:.0%})...", flush=True)

    try:
        r = subprocess.run(train_cmd, capture_output=True, text=True, timeout=1200, env=env)
        if r.returncode != 0:
            # Check if it's a "no data" error vs actual failure
            if "No training samples" in r.stderr or "No training samples" in r.stdout:
                print(f"  Policy training skipped: no policy training data available", flush=True)
                return False
            print(f"  Policy training failed: {r.stderr[:500] if r.stderr else r.stdout[:500]}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print(f"  Policy training timeout after 1200s", flush=True)
        return False

    # Check output for KL loss status
    output = r.stdout + r.stderr
    if "Auto-enabled KL loss" in output:
        print(f"  Policy training complete with KL loss (MCTS data detected)!", flush=True)
    elif "KL loss not auto-enabled" in output:
        print(f"  Policy training complete with cross-entropy (no MCTS data)", flush=True)
    else:
        print(f"  Policy training complete!", flush=True)

    return True


def main():
    global last_trained_counts

    print("=" * 60, flush=True)
    print("Multi-Config Training Loop v7 (OPTIMIZED HP + ADAPTIVE CURRICULUM)", flush=True)
    print(f"Base dir: {BASE_DIR}", flush=True)
    print("Configs: " + ", ".join(short_name(bt, np) for bt, np in THRESHOLDS.keys()), flush=True)
    print("=" * 60, flush=True)

    # Show advanced features status
    print("Advanced Features:", flush=True)
    print(f"  - Hyperparameters: {'ENABLED' if HAS_HYPERPARAMETERS else 'disabled'}", flush=True)
    print(f"  - Feedback Accelerator: {'ENABLED' if HAS_FEEDBACK_ACCELERATOR else 'disabled'}", flush=True)
    print(f"  - Adaptive Curriculum: ENABLED (ELO-based)", flush=True)
    print(f"  - Auto HP Tuning: {'ENABLED' if ENABLE_AUTO_HP_TUNING else 'disabled (set RINGRIFT_ENABLE_AUTO_HP_TUNING=1)'}", flush=True)
    if ENABLE_POLICY_TRAINING:
        print(f"  - Policy Training: ENABLED (auto-KL: {POLICY_AUTO_KL_LOSS}, "
              f"coverage>={POLICY_KL_MIN_COVERAGE:.0%}, min={POLICY_KL_MIN_SAMPLES} samples)", flush=True)
    else:
        print(f"  - Policy Training: disabled (set RINGRIFT_ENABLE_POLICY_TRAINING=1)", flush=True)

    # Show hyperparameter status for all configs
    if HAS_HYPERPARAMETERS:
        try:
            hp_configs = get_all_hp_configs()
            optimized = [k for k, v in hp_configs.items() if v.get("optimized")]
            needs_tune = [k for k, v in hp_configs.items() if not v.get("optimized")]
            if optimized:
                print(f"  HP Optimized: {', '.join(optimized)}", flush=True)
            if needs_tune:
                print(f"  HP Needs Tuning: {', '.join(needs_tune[:3])}{'...' if len(needs_tune) > 3 else ''}", flush=True)
        except Exception:
            pass

    # Show feedback accelerator status
    if HAS_FEEDBACK_ACCELERATOR:
        try:
            accelerator = get_feedback_accelerator()
            curriculum_weights = accelerator.get_curriculum_weights()
            if curriculum_weights:
                accelerated = [k for k, v in curriculum_weights.items() if v > 1.2]
                if accelerated:
                    print(f"  Accelerated configs: {', '.join(accelerated)}", flush=True)
        except Exception:
            pass

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

            # ADAPTIVE CURRICULUM MODE: Prioritize configs based on:
            # 1. Fewest trained models (balance training across configs)
            # 2. Lowest ELO (weaker models need more training)
            # 3. Most new games available (tie-breaker)
            if training_candidates:
                # Get ELO ratings for adaptive curriculum
                elo_scores = get_all_config_elos()

                # Sort by: (1) fewest models ASCENDING, (2) lowest ELO ASCENDING, (3) most new games DESCENDING
                # This prioritizes under-trained configs, then weaker models
                def sort_key(candidate):
                    config, _, _, _, new_games, models = candidate
                    elo = elo_scores.get(config, 1500.0)
                    return (models, elo, -new_games)

                training_candidates.sort(key=sort_key)
                config, db_paths, jsonl_paths, total_count, new_games, models = training_candidates[0]
                board_type, num_players = config
                sn = short_name(board_type, num_players)
                elo = elo_scores.get(config, 1500.0)
                print(f"[{ts}] ADAPTIVE: Training {sn} (models={models}, ELO={elo:.0f}, new_games={new_games})", flush=True)
                run_training(board_type, num_players, db_paths, jsonl_paths, total_count, iteration)

                # Run policy training with auto KL loss if enabled
                if ENABLE_POLICY_TRAINING and db_paths:
                    run_policy_training(board_type, num_players, db_paths, jsonl_paths, total_count, iteration)

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
