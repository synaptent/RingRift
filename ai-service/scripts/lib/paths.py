"""Path utilities for scripts.

Re-exports common paths from app/utils/paths.py and provides additional
script-specific path utilities.

Usage:
    from scripts.lib.paths import (
        AI_SERVICE_ROOT,
        DATA_DIR,
        MODELS_DIR,
        LOGS_DIR,
        get_game_db_path,
        get_training_data_path,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

# Compute project root from this file's location
_SCRIPT_LIB_DIR = Path(__file__).resolve().parent
_SCRIPTS_DIR = _SCRIPT_LIB_DIR.parent
AI_SERVICE_ROOT = _SCRIPTS_DIR.parent

# =============================================================================
# Primary Directories
# =============================================================================

# Data directories
DATA_DIR = AI_SERVICE_ROOT / "data"
GAMES_DIR = DATA_DIR / "games"
SELFPLAY_DIR = DATA_DIR / "selfplay"
TRAINING_DIR = DATA_DIR / "training"
METRICS_DIR = DATA_DIR / "metrics"
HOLDOUT_DIR = DATA_DIR / "holdouts"
QUARANTINE_DIR = DATA_DIR / "quarantine"
BACKUP_DIR = DATA_DIR / "backups"

# Model directories
MODELS_DIR = AI_SERVICE_ROOT / "models"
NNUE_MODELS_DIR = MODELS_DIR / "nnue"
ARCHIVED_MODELS_DIR = MODELS_DIR / "archived"

# Log directories
LOGS_DIR = AI_SERVICE_ROOT / "logs"
TRAINING_LOGS_DIR = LOGS_DIR / "training"

# Configuration directories
CONFIG_DIR = AI_SERVICE_ROOT / "config"
SCRIPTS_DIR = _SCRIPTS_DIR

# Runtime directories
RUNS_DIR = AI_SERVICE_ROOT / "runs"
LOCKS_DIR = RUNS_DIR / "locks"


# =============================================================================
# Common Database Paths
# =============================================================================

UNIFIED_ELO_DB = DATA_DIR / "unified_elo.db"
WORK_QUEUE_DB = DATA_DIR / "work_queue.db"


# =============================================================================
# Path Utilities
# =============================================================================

def get_game_db_path(config_key: str) -> Path:
    """Get the selfplay database path for a config.

    Args:
        config_key: Board config key (e.g., "square8_2p")

    Returns:
        Path to the selfplay database
    """
    return GAMES_DIR / f"selfplay_{config_key}.db"


def get_training_data_path(config_key: str, suffix: str = ".npz") -> Path:
    """Get the training data path for a config.

    Args:
        config_key: Board config key (e.g., "square8_2p")
        suffix: File extension (default: ".npz")

    Returns:
        Path to the training data file
    """
    return TRAINING_DIR / f"training_{config_key}{suffix}"


def get_model_path(
    config_key: str,
    model_type: str = "nnue",
    filename: Optional[str] = None,
) -> Path:
    """Get the model path for a config.

    Args:
        config_key: Board config key (e.g., "square8_2p")
        model_type: Type of model ("nnue", "policy", etc.)
        filename: Specific filename (if None, uses default naming)

    Returns:
        Path to the model file
    """
    base_dir = MODELS_DIR / model_type if model_type else MODELS_DIR

    if filename:
        return base_dir / filename

    return base_dir / f"{model_type}_{config_key}.pt"


def get_log_path(script_name: str) -> Path:
    """Get log file path for a script.

    Args:
        script_name: Name of the script

    Returns:
        Path to the log file
    """
    return LOGS_DIR / f"{script_name}.log"


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path to ensure

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    """Ensure the parent directory of a file exists.

    Args:
        path: File path whose parent should exist

    Returns:
        The same path (for chaining)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# Ensure critical directories exist on import
for _dir in [DATA_DIR, GAMES_DIR, MODELS_DIR, LOGS_DIR, RUNS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)
