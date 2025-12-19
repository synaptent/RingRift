"""Centralized path definitions for the AI service.

This module provides a single source of truth for all project paths.
Import paths from here instead of computing them in each file.

Usage:
    from app.utils.paths import AI_SERVICE_ROOT, MODELS_DIR, DATA_DIR

    # Or for scripts that run before sys.path is set up:
    from app.utils.paths import (
        get_project_root,
        get_models_dir,
        get_data_dir,
    )
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

# =============================================================================
# Core Project Root
# =============================================================================

def get_project_root() -> Path:
    """Get the AI service project root directory.

    This works regardless of where the code is called from, as long as
    this file is in app/utils/paths.py.
    """
    return Path(__file__).resolve().parents[2]


# The project root - use this instead of computing Path(__file__).parents[n]
AI_SERVICE_ROOT = get_project_root()


# =============================================================================
# Primary Directories
# =============================================================================

# Data directories
DATA_DIR = AI_SERVICE_ROOT / "data"
GAMES_DIR = DATA_DIR / "games"
SELFPLAY_DIR = DATA_DIR / "selfplay"
METRICS_DIR = DATA_DIR / "metrics"
HOLDOUT_DIR = DATA_DIR / "holdouts"
QUARANTINE_DIR = DATA_DIR / "quarantine"
BACKUP_DIR = DATA_DIR / "backups"

# Model directories
MODELS_DIR = AI_SERVICE_ROOT / "models"
ARCHIVED_MODELS_DIR = MODELS_DIR / "archived"

# Log directories
LOGS_DIR = AI_SERVICE_ROOT / "logs"
TRAINING_LOGS_DIR = LOGS_DIR / "training"
DEPLOYMENT_LOGS_DIR = LOGS_DIR / "deployment"

# Runtime directories
RUNS_DIR = AI_SERVICE_ROOT / "runs"
PROMOTION_DIR = RUNS_DIR / "promotion"
LOCKS_DIR = RUNS_DIR / "locks"

# Configuration directories
CONFIG_DIR = AI_SERVICE_ROOT / "config"
SCRIPTS_DIR = AI_SERVICE_ROOT / "scripts"


# =============================================================================
# Common Database Paths
# =============================================================================

UNIFIED_ELO_DB = DATA_DIR / "unified_elo.db"
TRAINING_METRICS_DB = METRICS_DIR / "training_metrics.db"
WORK_QUEUE_DB = DATA_DIR / "work_queue.db"


# =============================================================================
# Common Configuration Files
# =============================================================================

PROMOTION_HISTORY_FILE = PROMOTION_DIR / "promoted_models.json"
GAUNTLET_RESULTS_FILE = DATA_DIR / "aggregated_gauntlet_results.json"
MODEL_REGISTRY_FILE = DATA_DIR / "model_registry.json"


# =============================================================================
# Helper Functions
# =============================================================================

def get_models_dir(board_type: Optional[str] = None) -> Path:
    """Get the models directory, optionally for a specific board type.

    Args:
        board_type: Optional board type (e.g., 'square8', 'hex')

    Returns:
        Path to models directory
    """
    if board_type:
        return MODELS_DIR / board_type
    return MODELS_DIR


def get_data_dir(subdir: Optional[str] = None) -> Path:
    """Get the data directory, optionally with a subdirectory.

    Args:
        subdir: Optional subdirectory name

    Returns:
        Path to data directory
    """
    if subdir:
        return DATA_DIR / subdir
    return DATA_DIR


def get_games_db_path(config_key: str) -> Path:
    """Get the path to a games database for a config.

    Args:
        config_key: Configuration key (e.g., 'square8_2p')

    Returns:
        Path to the games database
    """
    return GAMES_DIR / f"{config_key}.db"


def get_selfplay_db_path(config_key: str) -> Path:
    """Get the path to a selfplay database for a config.

    Args:
        config_key: Configuration key (e.g., 'square8_2p')

    Returns:
        Path to the selfplay database
    """
    return SELFPLAY_DIR / f"selfplay_{config_key}.db"


def get_model_path(model_name: str, board_type: Optional[str] = None) -> Path:
    """Get the path to a model file.

    Args:
        model_name: Model filename
        board_type: Optional board type for subdirectory

    Returns:
        Path to the model file
    """
    base = get_models_dir(board_type)
    return base / model_name


def get_log_path(name: str, subdir: Optional[str] = None) -> Path:
    """Get a log file path.

    Args:
        name: Log filename
        subdir: Optional subdirectory

    Returns:
        Path to the log file
    """
    base = LOGS_DIR / subdir if subdir else LOGS_DIR
    return base / name


def ensure_dir(path: Path) -> Path:
    """Ensure a directory exists, creating it if necessary.

    Args:
        path: Directory path

    Returns:
        The same path (for chaining)
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def ensure_parent_dir(path: Path) -> Path:
    """Ensure the parent directory of a path exists.

    Args:
        path: File path

    Returns:
        The same path (for chaining)
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


# =============================================================================
# Environment-Based Overrides
# =============================================================================

def get_env_path(env_var: str, default: Path) -> Path:
    """Get a path from environment variable or use default.

    Args:
        env_var: Environment variable name
        default: Default path if env var not set

    Returns:
        Path from environment or default
    """
    env_value = os.environ.get(env_var)
    if env_value:
        return Path(env_value)
    return default


# Allow override of key directories via environment
if os.environ.get("AI_SERVICE_DATA_DIR"):
    DATA_DIR = Path(os.environ["AI_SERVICE_DATA_DIR"])
    GAMES_DIR = DATA_DIR / "games"
    SELFPLAY_DIR = DATA_DIR / "selfplay"
    METRICS_DIR = DATA_DIR / "metrics"
    HOLDOUT_DIR = DATA_DIR / "holdouts"

if os.environ.get("AI_SERVICE_MODELS_DIR"):
    MODELS_DIR = Path(os.environ["AI_SERVICE_MODELS_DIR"])
    ARCHIVED_MODELS_DIR = MODELS_DIR / "archived"

if os.environ.get("AI_SERVICE_LOGS_DIR"):
    LOGS_DIR = Path(os.environ["AI_SERVICE_LOGS_DIR"])
    TRAINING_LOGS_DIR = LOGS_DIR / "training"
    DEPLOYMENT_LOGS_DIR = LOGS_DIR / "deployment"
