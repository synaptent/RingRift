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
import glob
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

# Try to import advanced training features
try:
    from app.training.feedback_accelerator import (
        get_feedback_accelerator,
    )
    HAS_FEEDBACK_ACCELERATOR = True
except ImportError:
    HAS_FEEDBACK_ACCELERATOR = False

try:
    from app.training.dynamic_export import get_export_settings as get_dynamic_export_settings
    HAS_DYNAMIC_EXPORT = True
except ImportError:
    HAS_DYNAMIC_EXPORT = False

try:
    HAS_DISTRIBUTED_LOCK = True
except ImportError:
    HAS_DISTRIBUTED_LOCK = False

try:
    HAS_MODEL_REGISTRY = True
except ImportError:
    HAS_MODEL_REGISTRY = False

try:
    from app.config.hyperparameters import (
        get_all_configs as get_all_hp_configs,
        get_hyperparameter_info,
        get_hyperparameters,
        needs_tuning,
    )
    HAS_HYPERPARAMETERS = True
except ImportError:
    HAS_HYPERPARAMETERS = False

# PFSP (Prioritized Fictitious Self-Play) opponent pool for diverse training
try:
    from app.training.advanced_training import (
        OpponentStats,
        PFSPOpponentPool,
    )
    HAS_PFSP = True
except ImportError:
    HAS_PFSP = False
    PFSPOpponentPool = None
    OpponentStats = None

# CMA-ES Auto-Tuner for hyperparameter optimization on Elo plateau
try:
    from app.training.advanced_training import (
        CMAESAutoTuner,
        PlateauConfig,
    )
    HAS_CMAES = True
except ImportError:
    HAS_CMAES = False
    CMAESAutoTuner = None
    PlateauConfig = None

# Incremental NPZ export for faster data pipeline
try:
    from app.training.incremental_export import (
        IncrementalExporter,
    )
    HAS_INCREMENTAL_EXPORT = True
except ImportError:
    HAS_INCREMENTAL_EXPORT = False
    IncrementalExporter = None

# Integrated Training Enhancements (2025-12)
# Unified module for auxiliary tasks, gradient surgery, batch scheduling,
# background eval, ELO weighting, curriculum learning, and data augmentation
try:
    from app.training.integrated_enhancements import (
        IntegratedEnhancementsConfig,
        IntegratedTrainingManager,
        create_integrated_manager,
    )
    HAS_INTEGRATED_ENHANCEMENTS = True
except ImportError:
    HAS_INTEGRATED_ENHANCEMENTS = False
    IntegratedTrainingManager = None
    IntegratedEnhancementsConfig = None

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

# Incremental NPZ export - dramatically faster by only processing new games
# Enabled by default when the module is available
ENABLE_INCREMENTAL_EXPORT = os.environ.get("RINGRIFT_ENABLE_INCREMENTAL_EXPORT", "1") == "1" and HAS_INCREMENTAL_EXPORT

# Track HP tuning recommendations
_hp_tuning_recommendations: dict[tuple[str, int], bool] = {}
DATA_DIR = os.path.join(BASE_DIR, "data")


def merge_npz_files(npz_files: list[str], output_path: str) -> int:
    """Merge multiple NPZ training data files into one.

    Args:
        npz_files: List of NPZ file paths to merge
        output_path: Output file path for merged data

    Returns:
        Total number of samples in merged file

    Note:
        Automatically filters out files with mismatched feature dimensions.
        Uses v3 encoder (64 features) as the target. Files with v2 encoder
        (40 features) are skipped with a warning.
    """
    if not npz_files:
        return 0

    # Target feature dimension for v3 encoder
    TARGET_FEATURE_DIM = 64  # v3 encoder has 64 features, v2 has 40

    if len(npz_files) == 1:
        # Single file - check dimensions first
        if os.path.exists(npz_files[0]):
            try:
                with np.load(npz_files[0], allow_pickle=True) as data:
                    if data["features"].shape[1] != TARGET_FEATURE_DIM:
                        print(f"  Warning: Skipping {npz_files[0]} - wrong feature dim ({data['features'].shape[1]} != {TARGET_FEATURE_DIM})", flush=True)
                        return 0
            except Exception:
                pass
        if npz_files[0] != output_path:
            os.rename(npz_files[0], output_path)
        return -1  # Unknown count, caller should check

    # Collect all data from files
    all_features = []
    all_globals = []
    all_values = []
    all_values_mp = []
    all_num_players = []
    all_policy_indices = []
    all_policy_values = []
    all_move_numbers = []
    all_total_game_moves = []
    all_phases = []

    skipped_v2_count = 0
    target_dim = None  # Will be set from first valid file

    total_samples = 0
    for npz_path in npz_files:
        if not os.path.exists(npz_path):
            continue

        try:
            with np.load(npz_path, allow_pickle=True) as data:
                n_samples = len(data["features"])
                if n_samples == 0:
                    continue

                # Check feature dimensions - skip mismatched files
                feature_dim = data["features"].shape[1]
                if target_dim is None:
                    # Use TARGET_FEATURE_DIM (v3) as the standard
                    target_dim = TARGET_FEATURE_DIM

                if feature_dim != target_dim:
                    skipped_v2_count += 1
                    if skipped_v2_count <= 3:  # Only log first few
                        print(f"  Warning: Skipping {os.path.basename(npz_path)} - feature dim {feature_dim} != {target_dim} (v2/v3 mismatch)", flush=True)
                    continue

                all_features.append(data["features"])
                all_globals.append(data["globals"])
                all_values.append(data["values"])

                # Optional fields - may not exist in all files
                if "values_mp" in data:
                    all_values_mp.append(data["values_mp"])
                if "num_players" in data:
                    all_num_players.append(data["num_players"])
                if "policy_indices" in data:
                    all_policy_indices.extend(data["policy_indices"])
                if "policy_values" in data:
                    all_policy_values.extend(data["policy_values"])
                if "move_numbers" in data:
                    all_move_numbers.append(data["move_numbers"])
                if "total_game_moves" in data:
                    all_total_game_moves.append(data["total_game_moves"])
                if "phases" in data:
                    all_phases.extend(data["phases"])

                total_samples += n_samples
        except Exception as e:
            print(f"  Warning: Failed to load {npz_path}: {e}", flush=True)
            continue

    # Log summary of skipped files
    if skipped_v2_count > 0:
        print(f"  Note: Skipped {skipped_v2_count} files with v2 encoder (40 features) - using v3 only (64 features)", flush=True)

    if total_samples == 0:
        if skipped_v2_count > 0:
            print(f"  Warning: All {skipped_v2_count} NPZ files had wrong feature dimensions - no v3 data available", flush=True)
        return 0

    # Concatenate arrays
    features_arr = np.concatenate(all_features, axis=0)
    globals_arr = np.concatenate(all_globals, axis=0)
    values_arr = np.concatenate(all_values, axis=0)

    # Build save dict with required fields
    save_dict = {
        "features": features_arr,
        "globals": globals_arr,
        "values": values_arr,
    }

    # Add optional fields if present
    if all_values_mp:
        save_dict["values_mp"] = np.concatenate(all_values_mp, axis=0)
    if all_num_players:
        save_dict["num_players"] = np.concatenate(all_num_players, axis=0)
    if all_policy_indices:
        save_dict["policy_indices"] = np.array(all_policy_indices, dtype=object)
    if all_policy_values:
        save_dict["policy_values"] = np.array(all_policy_values, dtype=object)
    if all_move_numbers:
        save_dict["move_numbers"] = np.concatenate(all_move_numbers, axis=0)
    if all_total_game_moves:
        save_dict["total_game_moves"] = np.concatenate(all_total_game_moves, axis=0)
    if all_phases:
        save_dict["phases"] = np.array(all_phases, dtype=object)

    # Save merged file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **save_dict)

    # Clean up source files
    for npz_path in npz_files:
        if npz_path != output_path and os.path.exists(npz_path):
            os.remove(npz_path)

    return total_samples

# Database sources for each config - databases that have games WITH moves
# Format: (board_type, num_players) -> list of database paths
# Uses canonical selfplay.db and diverse_synced per-config DBs
# NOTE: selfplay_stats.db is for monitoring ONLY (no game_moves table)
# NOTE: jsonl_converted_*.db have NO game_moves table - don't use for training
CONFIG_DATABASES: dict[tuple[str, int], list[str]] = {
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
# JSONL from tournaments, hybrid selfplay, GPU selfplay, and MCTS selfplay
# NOTE: Prior to 2025-12-17, only canonical and data/games were searched - this
# caused a critical data flow mismatch where selfplay data wasn't reaching training!
CONFIG_JSONL_DIRS: dict[tuple[str, int], list[str]] = {
    # Square8 2p - highest priority, include all selfplay sources
    ("square8", 2): [
        "data/selfplay/canonical",  # Canonical selfplay JSONL
        "data/games",  # Tournament JSONL (games.jsonl, etc.)
        "data/selfplay/gpu",  # GPU selfplay (high throughput)
        "data/selfplay/mcts_square8_2p",  # MCTS policy selfplay
        "data/selfplay/mcts_cluster_collected_v3",  # Cluster MCTS data
        "data/selfplay/mcts_cluster_collected_v2",  # Older cluster MCTS
        "data/selfplay/hybrid_test",  # Hybrid selfplay test data
        "data/selfplay/reanalyzed_square8_2p",  # Reanalyzed games
        "data/selfplay/cluster_h100",  # H100 cluster selfplay
    ],
    ("square8", 3): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/gpu",
    ],
    ("square8", 4): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/gpu",
    ],
    ("square19", 2): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/gpu",
        "data/selfplay/daemon_square19_2p",
    ],
    ("square19", 3): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/daemon_square19_3p",
    ],
    ("square19", 4): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/daemon_square19_4p",
    ],
    ("hexagonal", 2): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/daemon_hexagonal_2p",
    ],
    ("hexagonal", 3): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/daemon_hexagonal_3p",
    ],
    ("hexagonal", 4): [
        "data/selfplay/canonical",
        "data/games",
        "data/selfplay/daemon_hexagonal_4p",
    ],
    # Hex8 configs - use hex8-specific selfplay directories
    ("hex8", 2): [
        "data/selfplay/hex8_policy_c",
        "data/selfplay/hex8_combined",  # Combined hex8 selfplay
    ],
    ("hex8", 3): [
        "data/selfplay/hex8_policy_c",
        "data/selfplay/hex8_combined",
    ],
    ("hex8", 4): [
        "data/selfplay/hex8_policy_c",
        "data/selfplay/hex8_combined",
    ],
}

# Training thresholds - trigger training when this many NEW games are available
# Set low initially to trigger training quickly, adjust based on game generation rate
THRESHOLDS: dict[tuple[str, int], int] = {
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
EXPORT_SETTINGS: dict[tuple[str, int], tuple[int, int, int]] = {
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


def get_export_settings_for_config(
    board_type: str,
    num_players: int,
    db_paths: list[str] | None = None,
) -> tuple[int, int, int]:
    """Get export settings for a config, using dynamic settings if available.

    Returns:
        Tuple of (max_games, sample_every, epochs)
    """
    # Try dynamic export first (computes optimal settings based on data size)
    if HAS_DYNAMIC_EXPORT and db_paths:
        try:
            settings = get_dynamic_export_settings(db_paths, board_type, num_players)
            max_games = settings.max_games if settings.max_games else DEFAULT_MAX_GAMES
            sample_every = settings.sample_every
            epochs = settings.epochs
            print(f"  [dynamic_export] {board_type}_{num_players}p: "
                  f"tier={settings.data_tier}, max_games={max_games}, "
                  f"sample_every={sample_every}, epochs={epochs}", flush=True)
            return (max_games, sample_every, epochs)
        except Exception as e:
            print(f"  [dynamic_export] Failed to compute settings: {e}", flush=True)

    # Fall back to static settings
    return EXPORT_SETTINGS.get(
        (board_type, num_players),
        (DEFAULT_MAX_GAMES, DEFAULT_SAMPLE_EVERY, DEFAULT_EPOCHS)
    )


# Track last training count per config
last_trained_counts: dict[tuple[str, int], int] = dict.fromkeys(THRESHOLDS, 0)

# Unified ELO database path
UNIFIED_ELO_DB = os.path.join(DATA_DIR, "unified_elo.db")

# PFSP opponent pools per config (for diverse selfplay training)
PFSP_POOLS: dict[tuple[str, int], Any] = {}
if HAS_PFSP:
    for config in [("square8", 2), ("square8", 4), ("hex8", 2), ("hexagonal", 2)]:
        try:
            PFSP_POOLS[config] = PFSPOpponentPool(
                max_pool_size=30,
                hard_opponent_weight=0.6,
                diversity_weight=0.25,
                recency_weight=0.15,
            )
        except Exception as e:
            print(f"[PFSP] Failed to initialize pool for {config}: {e}")

# CMA-ES Auto-Tuners per config (for hyperparameter optimization on plateau)
CMAES_TUNERS: dict[tuple[str, int], Any] = {}
LAST_CMAES_ELO: dict[tuple[str, int], float] = {}

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


def check_nas_results(board_type: str, num_players: int) -> dict[str, Any] | None:
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


def check_nas_recommendation(board_type: str, num_players: int) -> str | None:
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


def check_hp_tuning_recommendation(board_type: str, num_players: int, total_games: int) -> str | None:
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


def get_all_config_elos() -> dict[tuple[str, int], float]:
    """Get ELO ratings for all configs at once for efficiency."""
    elos = dict.fromkeys(THRESHOLDS, 1500.0)

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


def get_model_counts() -> dict[tuple[str, int], int]:
    """Get count of trained models for each config."""
    return {config: count_trained_models(board_type, num_players)
            for config in THRESHOLDS
            for board_type, num_players in [config]}


def find_databases(path: str) -> list[str]:
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


def find_jsonl_files(path: str) -> list[str]:
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


def auto_discover_jsonl_dirs() -> dict[str, list[str]]:
    """Auto-discover all selfplay JSONL directories.

    Scans data/selfplay for subdirectories containing JSONL files.
    Returns a mapping of directory name to list of JSONL files.

    This prevents data flow mismatches where selfplay generates data
    but training can't find it because CONFIG_JSONL_DIRS is out of date.
    """
    selfplay_root = os.path.join(BASE_DIR, "data/selfplay")
    if not os.path.exists(selfplay_root):
        return {}

    discovered = {}
    for entry in os.listdir(selfplay_root):
        dir_path = os.path.join(selfplay_root, entry)
        if os.path.isdir(dir_path):
            jsonl_files = find_jsonl_files(dir_path)
            if jsonl_files:
                discovered[entry] = jsonl_files

    return discovered


def get_dynamic_jsonl_dirs(board_type: str, num_players: int) -> list[str]:
    """Get JSONL directories for a config, including auto-discovered ones.

    Combines static CONFIG_JSONL_DIRS with dynamically discovered directories
    that might contain relevant data based on naming patterns.
    """
    config = (board_type, num_players)
    static_dirs = list(CONFIG_JSONL_DIRS.get(config, []))

    # Auto-discover directories matching naming patterns
    discovered = auto_discover_jsonl_dirs()

    # Patterns that indicate relevance to this config
    board_patterns = BOARD_VARIANTS.get(board_type, [board_type])
    player_pattern = f"{num_players}p"

    for dir_name, _jsonl_files in discovered.items():
        # Skip if already in static config
        rel_path = f"data/selfplay/{dir_name}"
        if rel_path in static_dirs:
            continue

        # Check if directory name suggests relevance to this config
        dir_lower = dir_name.lower()
        is_relevant = any(bp in dir_lower for bp in board_patterns) or player_pattern in dir_lower

        # Also check for generic selfplay directories that might have mixed content
        is_generic = any(pat in dir_lower for pat in ['gpu', 'mcts', 'hybrid', 'cluster', 'daemon'])

        if is_relevant or is_generic:
            static_dirs.append(rel_path)

    return static_dirs


def count_jsonl_games(jsonl_path: str, board_type: str, num_players: int,
                       max_lines: int = 10000) -> tuple[int, set[str]]:
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
        with open(jsonl_path) as f:
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

                    if board_match and players_match and has_moves and game_id and game_id not in game_ids:
                        game_ids.add(game_id)
                        count += 1
                except json.JSONDecodeError:
                    continue
    except Exception:
        return 0, set()

    return count, game_ids


# Cache for JSONL file metadata to avoid repeated parsing
_jsonl_metadata_cache: dict[str, dict[tuple[str, int], int]] = {}
_jsonl_cache_time: dict[str, float] = {}
JSONL_CACHE_TTL = 300  # 5 minute cache TTL


def get_jsonl_file_metadata(jsonl_path: str, max_lines: int = 5000) -> dict[tuple[str, int], set[str]]:
    """Parse JSONL file once and return game counts per config.

    Returns dict of (board_type, num_players) -> set of game_ids.
    Cached for 5 minutes to avoid repeated parsing.
    """
    # Check cache
    cache_key = jsonl_path
    if cache_key in _jsonl_cache_time and time.time() - _jsonl_cache_time[cache_key] < JSONL_CACHE_TTL:
        return _jsonl_metadata_cache.get(cache_key, {})

    # Parse file
    result: dict[tuple[str, int], set[str]] = {}

    if not os.path.exists(jsonl_path):
        return result

    try:
        lines_read = 0
        with open(jsonl_path) as f:
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


def get_jsonl_counts(board_type: str, num_players: int) -> tuple[int, list[str]]:
    """Get total JSONL game counts for a config, returning (total_count, jsonl_files_with_games).

    Uses dynamic directory discovery to find all selfplay JSONL sources.
    """
    config = (board_type, num_players)
    # Use dynamic discovery to find all relevant JSONL directories
    jsonl_dirs = get_dynamic_jsonl_dirs(board_type, num_players)

    total_count = 0
    jsonl_with_games = []
    seen_game_ids: set[str] = set()  # Dedupe across files

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
    except Exception:
        return 0


def get_config_counts() -> dict[tuple[str, int], tuple[int, list[str], int, list[str]]]:
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


def check_cmaes_auto_tuning(board_type: str, num_players: int, iteration: int) -> None:
    """Check for Elo plateau and trigger CMA-ES auto-tuning if needed.

    This monitors Elo progression and automatically triggers hyperparameter
    optimization when a plateau is detected (no Elo gain over several iterations).
    """
    if not HAS_CMAES:
        return

    key = (board_type, num_players)
    short = short_name(board_type, num_players)
    current_elo = get_config_elo(board_type, num_players)

    # Check every 5 iterations to avoid too frequent checks
    if iteration % 5 != 0:
        return

    # Check if Elo has improved since last check
    last_elo = LAST_CMAES_ELO.get(key, 0.0)
    LAST_CMAES_ELO[key] = current_elo

    if last_elo == 0.0:
        return  # First check, no comparison

    elo_delta = current_elo - last_elo

    # Detect plateau: less than 5 Elo gain over check period
    if elo_delta < 5.0:
        print(f"  [CMA-ES] Elo plateau detected for {short}: {last_elo:.0f} -> {current_elo:.0f} (delta={elo_delta:.1f})", flush=True)

        # Check if already running a tuner for this config
        if key in CMAES_TUNERS and CMAES_TUNERS[key] is not None:
            print(f"  [CMA-ES] Auto-tuner already active for {short}", flush=True)
            return

        # Initialize and trigger CMA-ES auto-tuning
        try:
            plateau_config = PlateauConfig(
                patience=3,
                min_improvement=5.0,
                window_size=5,
            )
            tuner = CMAESAutoTuner(
                plateau_config=plateau_config,
                population_size=8,
                sigma=0.3,
            )
            CMAES_TUNERS[key] = tuner

            # Trigger HP tuning via the standard mechanism
            print(f"  [CMA-ES] Triggering auto-tuning for {short} due to plateau", flush=True)
            trigger_hp_tuning(board_type, num_players, trials=15)

        except Exception as e:
            print(f"  [CMA-ES] Failed to initialize auto-tuner: {e}", flush=True)

    elif elo_delta >= 10.0:
        # Clear tuner if Elo is improving again
        if key in CMAES_TUNERS:
            print(f"  [CMA-ES] Elo improving for {short}: +{elo_delta:.1f}, clearing plateau state", flush=True)
            CMAES_TUNERS[key] = None


def get_pfsp_opponent(board_type: str, num_players: int) -> str | None:
    """Get a PFSP-weighted opponent for selfplay.

    Returns model path selected based on PFSP prioritization (hard opponents
    with some diversity and recency weighting).
    """
    if not HAS_PFSP:
        return None

    key = (board_type, num_players)
    if key not in PFSP_POOLS:
        return None

    try:
        pool = PFSP_POOLS[key]
        opponent = pool.sample_opponent()
        if opponent:
            return opponent.model_path
    except Exception:
        pass

    return None


def update_pfsp_stats(board_type: str, num_players: int, model_id: str,
                       win: bool, game_length: int) -> None:
    """Update PFSP opponent statistics after a game.

    Args:
        board_type: Board type for the config
        num_players: Number of players
        model_id: ID of the opponent model
        win: Whether the current model won against this opponent
        game_length: Number of moves in the game
    """
    if not HAS_PFSP:
        return

    key = (board_type, num_players)
    if key not in PFSP_POOLS:
        return

    try:
        pool = PFSP_POOLS[key]
        pool.update_opponent_stats(
            model_id=model_id,
            win=win,
            game_length=game_length,
        )
    except Exception:
        pass


def run_training(board_type: str, num_players: int, db_paths: list[str],
                  jsonl_paths: list[str], current_count: int, iteration: int = 0) -> bool:
    """Run export and training for a config using DB and/or JSONL sources.

    Supports three modes:
    1. DB only: Uses export_replay_dataset.py
    2. JSONL only: Uses jsonl_to_npz.py (faster, no DB needed)
    3. Both: Exports both, merges NPZ files
    """
    key = (board_type, num_players)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = short_name(board_type, num_players)

    # Get config-specific export settings (uses dynamic settings if available)
    all_db_paths = db_paths if db_paths else []
    max_games, sample_every, epochs = get_export_settings_for_config(
        board_type, num_players, all_db_paths
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

        print("  Exporting from JSONL...", flush=True)
        try:
            r = subprocess.run(exp_cmd, capture_output=True, text=True, timeout=export_timeout, env=env)
            if r.returncode == 0 and os.path.exists(jsonl_npz):
                npz_files.append(jsonl_npz)
                print("  JSONL export complete", flush=True)
            else:
                print(f"  JSONL export failed: {r.stderr[:300] if r.stderr else r.stdout[:300]}", flush=True)
        except subprocess.TimeoutExpired:
            print("  JSONL export timeout", flush=True)

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
        print("  No data exported, skipping training", flush=True)
        last_trained_counts[key] = current_count
        return False

    # Merge multiple NPZ files if we have more than one
    if len(npz_files) > 1:
        print(f"  Merging {len(npz_files)} NPZ sources into training data...", flush=True)
        merged_samples = merge_npz_files(npz_files, npz)
        if merged_samples > 0:
            print(f"  Merged {merged_samples} total samples from {len(npz_files)} sources", flush=True)
        elif merged_samples == 0:
            print("  Warning: Merge produced no samples", flush=True)
            last_trained_counts[key] = current_count
            return False
    else:
        # Single file - just rename to final output path
        final_npz = npz_files[0]
        if final_npz != npz:
            os.rename(final_npz, npz)

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
        # Advanced training optimizations (2024-12)
        "--spectral-norm",  # Gradient stability
        "--cyclic-lr", "--cyclic-lr-period", "5",  # Cyclic LR with triangular waves
        "--mixed-precision", "--amp-dtype", "bfloat16",  # BF16 for speed+stability
        "--warmup-epochs", "5",  # LR warmup
        # 2024-12 Advanced Training Improvements
        "--value-whitening",  # Value head whitening for stable training
        "--ema",  # Exponential Moving Average for better generalization
        "--stochastic-depth", "--stochastic-depth-prob", "0.1",  # Stochastic depth regularization
        "--adaptive-warmup",  # Adaptive warmup based on dataset size
        "--hard-example-mining", "--hard-example-top-k", "0.3",  # Focus on difficult examples
        # 2024-12 Advanced Optimizer Enhancements
        "--lookahead", "--lookahead-k", "5", "--lookahead-alpha", "0.5",  # Lookahead optimizer
        "--adaptive-clip",  # Adaptive gradient clipping
        "--board-nas",  # Board-specific neural architecture search
        "--online-bootstrap", "--bootstrap-temperature", "1.5", "--bootstrap-start-epoch", "10",  # Online bootstrapping
        # 2024-12 Phase 2 Advanced Training
        "--prefetch-gpu",  # GPU prefetching for improved throughput
        "--difficulty-curriculum",  # Difficulty-aware curriculum learning
        "--quantized-eval",  # Fast quantized inference for validation
        # Note: Experimental Phase 2 features (enable via config if needed):
        # --use-attention, --use-moe, --use-multitask (architectural changes)
        # --use-lamb, --gradient-compression (distributed training)
        # --contrastive-pretrain (self-supervised pretraining)
        # 2025-12 Training Improvements
        "--policy-label-smoothing", "0.05",  # Prevent overconfident predictions
        "--sampling-weights", "victory_type",  # Balance across victory types for better generalization
        # 2024-12 Phase 3 Advanced Training
        "--grokking-detection",  # Monitor for delayed generalization
        # Note: Optional Phase 3 features (enable via config if needed):
        # --use-sam (Sharpness-Aware Minimization - better generalization)
        # --td-lambda (Temporal Difference learning)
        # --auxiliary-targets (auxiliary prediction heads)
        # --pruning (post-training structured pruning)
        # --self-play (integrated self-play data generation)
        # --distillation (knowledge distillation from teacher)
        # 2024-12 Phase 4: Training Stability
        "--adaptive-accumulation",  # Dynamic gradient accumulation based on memory
        # Note: Optional Phase 4 features:
        # --curriculum-schedule (automatic progression through difficulty)
        # --loss-landscape-smoothing (better convergence)
        # --gradient-noise-injection (regularization via noise)
        # 2024-12 Phase 5: Production Optimization
        "--dynamic-loss-scaling",  # Adaptive FP16 loss scaling for stability
        # Note: Optional Phase 5 features:
        # --activation-checkpointing (memory efficiency for large models)
        # --flash-attention (faster attention when available)
        # --streaming-npz (for very large datasets)
        # --profiling (detailed training profiler)
    ]

    # D6 hex symmetry augmentation for hex boards (12x effective data)
    if board_type in ('hex8', 'hexagonal', 'hex'):
        train_cmd.append("--augment-hex-symmetry")

    # 2025-12 Integrated Enhancements - Apply curriculum and ELO weighting
    if HAS_INTEGRATED_ENHANCEMENTS:
        try:
            enhancement_config = IntegratedEnhancementsConfig(
                elo_weighting_enabled=True,
                curriculum_learning_enabled=True,
                augmentation_enabled=True,
            )
            enhancement_manager = create_integrated_manager(
                config_dict={
                    "elo_weighting_enabled": True,
                    "curriculum_learning_enabled": True,
                    "augmentation_enabled": True,
                },
                board_type=board_type,
            )
            curriculum_params = enhancement_manager.get_curriculum_parameters()
            if curriculum_params:
                print(f"  [Enhancements] Curriculum stage: {curriculum_params.get('name', 'default')}", flush=True)
        except Exception as e:
            print(f"  [Enhancements] Warning: {e}", flush=True)

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
        print("  Training timeout after 900s", flush=True)
        last_trained_counts[key] = current_count
        return False

    print(f"  {short} training complete!", flush=True)
    last_trained_counts[key] = current_count

    # PFSP: Add trained model to opponent pool for diverse selfplay
    if HAS_PFSP and key in PFSP_POOLS:
        try:
            # Find the trained model path (latest in run_dir)
            model_files = glob.glob(os.path.join(run_dir, "*.pth"))
            if model_files:
                latest_model = max(model_files, key=os.path.getmtime)
                model_id = Path(latest_model).stem
                PFSP_POOLS[key].add_opponent(
                    model_id=model_id,
                    model_path=latest_model,
                    elo=get_config_elo(board_type, num_players),
                    win_rate=0.5,  # Start with 50% assumed win rate
                )
                print(f"  [PFSP] Added {model_id} to opponent pool for {short}", flush=True)
        except Exception as e:
            print(f"  [PFSP] Warning: Failed to add model to pool: {e}", flush=True)

    # CMA-ES: Check for Elo plateau and trigger auto-tuning if needed
    check_cmaes_auto_tuning(board_type, num_players, iteration)

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


def run_policy_training(board_type: str, num_players: int, db_paths: list[str],
                         jsonl_paths: list[str], current_count: int, iteration: int = 0) -> bool:
    """Run policy training with auto KL loss detection.

    This trains a policy network using MCTS visit distributions when available.
    Gracefully falls back to cross-entropy loss if insufficient MCTS data.
    """
    if not ENABLE_POLICY_TRAINING:
        return False

    key = (board_type, num_players)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    short = short_name(board_type, num_players)

    # Get config-specific export settings (uses dynamic settings if available)
    all_db_paths = db_paths if db_paths else []
    max_games, sample_every, epochs = get_export_settings_for_config(
        board_type, num_players, all_db_paths
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

    # Add JSONL sources (for MCTS policy data)
    for jsonl_path in jsonl_paths:
        train_cmd.extend(["--jsonl", jsonl_path])

    # Add auto KL loss flags
    if POLICY_AUTO_KL_LOSS:
        train_cmd.append("--auto-kl-loss")
        train_cmd.extend(["--kl-min-coverage", str(POLICY_KL_MIN_COVERAGE)])
        train_cmd.extend(["--kl-min-samples", str(POLICY_KL_MIN_SAMPLES)])

    sources = []
    if db_paths:
        sources.append(f"{len(db_paths)} DB(s)")
    if jsonl_paths:
        sources.append(f"{len(jsonl_paths)} JSONL(s)")
    print(f"[{ts}] Policy training {short} from {' + '.join(sources)} with auto-KL (coverage>={POLICY_KL_MIN_COVERAGE:.0%})...", flush=True)

    try:
        r = subprocess.run(train_cmd, capture_output=True, text=True, timeout=1200, env=env)
        if r.returncode != 0:
            # Check if it's a "no data" error vs actual failure
            if "No training samples" in r.stderr or "No training samples" in r.stdout:
                print("  Policy training skipped: no policy training data available", flush=True)
                return False
            print(f"  Policy training failed: {r.stderr[:500] if r.stderr else r.stdout[:500]}", flush=True)
            return False
    except subprocess.TimeoutExpired:
        print("  Policy training timeout after 1200s", flush=True)
        return False

    # Check output for KL loss status
    output = r.stdout + r.stderr
    if "Auto-enabled KL loss" in output:
        print("  Policy training complete with KL loss (MCTS data detected)!", flush=True)
    elif "KL loss not auto-enabled" in output:
        print("  Policy training complete with cross-entropy (no MCTS data)", flush=True)
    else:
        print("  Policy training complete!", flush=True)

    return True


def main():
    global last_trained_counts

    print("=" * 60, flush=True)
    print("Multi-Config Training Loop v7 (OPTIMIZED HP + ADAPTIVE CURRICULUM)", flush=True)
    print(f"Base dir: {BASE_DIR}", flush=True)
    print("Configs: " + ", ".join(short_name(bt, np) for bt, np in THRESHOLDS), flush=True)
    print("=" * 60, flush=True)

    # Show advanced features status
    print("Advanced Features:", flush=True)
    print(f"  - Hyperparameters: {'ENABLED' if HAS_HYPERPARAMETERS else 'disabled'}", flush=True)
    print(f"  - Feedback Accelerator: {'ENABLED' if HAS_FEEDBACK_ACCELERATOR else 'disabled'}", flush=True)
    print("  - Adaptive Curriculum: ENABLED (ELO-based)", flush=True)
    print(f"  - Auto HP Tuning: {'ENABLED' if ENABLE_AUTO_HP_TUNING else 'disabled (set RINGRIFT_ENABLE_AUTO_HP_TUNING=1)'}", flush=True)
    if ENABLE_POLICY_TRAINING:
        print(f"  - Policy Training: ENABLED (auto-KL: {POLICY_AUTO_KL_LOSS}, "
              f"coverage>={POLICY_KL_MIN_COVERAGE:.0%}, min={POLICY_KL_MIN_SAMPLES} samples)", flush=True)
    else:
        print("  - Policy Training: disabled (set RINGRIFT_ENABLE_POLICY_TRAINING=1)", flush=True)
    # PFSP status
    if HAS_PFSP:
        pfsp_configs = list(PFSP_POOLS.keys())
        print(f"  - PFSP Opponent Pools: ENABLED ({len(pfsp_configs)} configs)", flush=True)
    else:
        print("  - PFSP Opponent Pools: disabled (import failed)", flush=True)
    # CMA-ES status
    if HAS_CMAES:
        print("  - CMA-ES Auto-Tuning: ENABLED (plateau detection)", flush=True)
    else:
        print("  - CMA-ES Auto-Tuning: disabled (import failed)", flush=True)
    # Incremental export status
    if ENABLE_INCREMENTAL_EXPORT:
        print("  - Incremental Export: ENABLED (fast data pipeline)", flush=True)
    else:
        print("  - Incremental Export: disabled (set RINGRIFT_ENABLE_INCREMENTAL_EXPORT=1)", flush=True)

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
                if ENABLE_POLICY_TRAINING and (db_paths or jsonl_paths):
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
