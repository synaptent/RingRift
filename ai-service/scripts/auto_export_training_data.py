#!/usr/bin/env python3
"""Automated Training Data Export Pipeline.

Continuously exports selfplay game data to NPZ format for neural network training.
Tracks exported games to avoid duplicate processing.

Usage:
    # Export all board types once
    python scripts/auto_export_training_data.py --once

    # Run as daemon (continuous export)
    python scripts/auto_export_training_data.py --daemon

    # Export specific board type
    python scripts/auto_export_training_data.py --once --board hexagonal --players 2

    # Cron example (every 2 hours):
    0 */2 * * * cd /path/to/ai-service && python scripts/auto_export_training_data.py --once >> logs/auto_export.log 2>&1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

LOG_DIR = AI_SERVICE_ROOT / "logs"
LOG_FILE = LOG_DIR / "auto_export.log"
STATE_FILE = AI_SERVICE_ROOT / "data" / "export_state.json"
DATA_DIR = AI_SERVICE_ROOT / "data"
TRAINING_DIR = DATA_DIR / "training"
SELFPLAY_DIR = DATA_DIR / "selfplay"

LOG_DIR.mkdir(parents=True, exist_ok=True)
TRAINING_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [AutoExport] %(levelname)s: %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


@dataclass
class ExportConfig:
    """Export configuration for a board/player combination."""
    board_type: str
    num_players: int
    db_patterns: List[str] = field(default_factory=list)
    output_prefix: str = ""
    min_new_games: int = 100  # Minimum new games before re-export
    encoder_version: str = "v3"
    board_aware_encoding: bool = True


# Default configurations for all board/player combinations
DEFAULT_CONFIGS = [
    ExportConfig(
        board_type="hexagonal",
        num_players=2,
        db_patterns=[
            "diverse/hex_2p.db",
            "diverse_synced/hex_2p.db",
            "aggregated/*/diverse_synced/hex_2p.db",
        ],
        output_prefix="hex_2p",
        min_new_games=50,
    ),
    ExportConfig(
        board_type="hexagonal",
        num_players=3,
        db_patterns=[
            "diverse/hex_3p.db",
            "diverse_synced/hex_3p.db",
        ],
        output_prefix="hex_3p",
        min_new_games=50,
    ),
    ExportConfig(
        board_type="hexagonal",
        num_players=4,
        db_patterns=[
            "diverse/hex_4p.db",
            "diverse_synced/hex_4p.db",
        ],
        output_prefix="hex_4p",
        min_new_games=50,
    ),
    ExportConfig(
        board_type="square19",
        num_players=2,
        db_patterns=[
            "diverse/square19_2p.db",
            "diverse_synced/square19_2p.db",
            "aggregated/*/diverse_synced/square19_2p.db",
        ],
        output_prefix="sq19_2p",
        min_new_games=50,
    ),
    ExportConfig(
        board_type="square19",
        num_players=3,
        db_patterns=[
            "diverse/square19_3p.db",
            "diverse_synced/square19_3p.db",
        ],
        output_prefix="sq19_3p",
        min_new_games=50,
    ),
    ExportConfig(
        board_type="square19",
        num_players=4,
        db_patterns=[
            "diverse/square19_4p.db",
            "diverse_synced/square19_4p.db",
        ],
        output_prefix="sq19_4p",
        min_new_games=50,
    ),
    ExportConfig(
        board_type="hex8",
        num_players=2,
        db_patterns=[
            "diverse/hex8_2p.db",
            "diverse_synced/hex8_2p.db",
        ],
        output_prefix="hex8_2p",
        min_new_games=100,
    ),
    ExportConfig(
        board_type="square8",
        num_players=2,
        db_patterns=[
            "diverse/square8_2p.db",
            "diverse_synced/square8_2p.db",
        ],
        output_prefix="sq8_2p",
        min_new_games=200,
    ),
]


@dataclass
class ExportState:
    """Tracks export state for incremental processing."""
    last_export_time: Dict[str, float] = field(default_factory=dict)
    last_game_count: Dict[str, int] = field(default_factory=dict)
    last_export_hash: Dict[str, str] = field(default_factory=dict)


def load_state() -> ExportState:
    """Load export state from file."""
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE) as f:
                data = json.load(f)
            return ExportState(
                last_export_time=data.get("last_export_time", {}),
                last_game_count=data.get("last_game_count", {}),
                last_export_hash=data.get("last_export_hash", {}),
            )
        except Exception as e:
            logger.warning(f"Failed to load state: {e}")
    return ExportState()


def save_state(state: ExportState) -> None:
    """Save export state to file."""
    try:
        with open(STATE_FILE, "w") as f:
            json.dump(asdict(state), f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save state: {e}")


def find_databases(config: ExportConfig) -> List[Path]:
    """Find all database files matching the config patterns."""
    databases = []
    for pattern in config.db_patterns:
        full_pattern = SELFPLAY_DIR / pattern
        # Handle glob patterns
        if "*" in pattern:
            parent = full_pattern.parent
            if parent.exists():
                for match in parent.parent.glob(pattern.split("/")[-2] + "/" + pattern.split("/")[-1]):
                    if match.exists() and match.stat().st_size > 0:
                        databases.append(match)
        else:
            if full_pattern.exists() and full_pattern.stat().st_size > 0:
                databases.append(full_pattern)
    return databases


def count_available_games(databases: List[Path]) -> int:
    """Count total available games across all databases."""
    total = 0
    for db_path in databases:
        try:
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM games WHERE game_status='completed' AND excluded_from_training=0"
            )
            count = cursor.fetchone()[0]
            total += count
            conn.close()
        except Exception as e:
            logger.warning(f"Error counting games in {db_path}: {e}")
    return total


def run_export(config: ExportConfig, databases: List[Path]) -> Tuple[bool, str]:
    """Run the export script for a configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = TRAINING_DIR / f"{config.output_prefix}_auto_{timestamp}.npz"

    # Build command
    cmd = [
        sys.executable,
        str(AI_SERVICE_ROOT / "scripts" / "export_replay_dataset.py"),
        "--board-type", config.board_type,
        "--num-players", str(config.num_players),
        "--output", str(output_file),
        "--encoder-version", config.encoder_version,
    ]

    # Add database paths
    for db in databases:
        cmd.extend(["--db", str(db)])

    if config.board_aware_encoding:
        cmd.append("--board-aware-encoding")

    logger.info(f"Running export: {' '.join(cmd[:6])}... ({len(databases)} DBs)")

    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = str(AI_SERVICE_ROOT)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=3600,  # 1 hour timeout
        )

        if result.returncode == 0:
            if output_file.exists():
                size_mb = output_file.stat().st_size / (1024 * 1024)
                logger.info(f"Export successful: {output_file.name} ({size_mb:.1f} MB)")
                return True, str(output_file)
            else:
                logger.warning(f"Export completed but file not created")
                return False, "No output file"
        else:
            logger.error(f"Export failed: {result.stderr[:500]}")
            return False, result.stderr[:500]

    except subprocess.TimeoutExpired:
        logger.error("Export timed out after 1 hour")
        return False, "Timeout"
    except Exception as e:
        logger.error(f"Export error: {e}")
        return False, str(e)


def should_export(config: ExportConfig, state: ExportState, databases: List[Path]) -> bool:
    """Determine if export should run for this configuration."""
    key = f"{config.board_type}_{config.num_players}p"

    # Check if databases exist
    if not databases:
        logger.debug(f"No databases found for {key}")
        return False

    # Count available games
    current_count = count_available_games(databases)
    last_count = state.last_game_count.get(key, 0)
    new_games = current_count - last_count

    logger.info(f"{key}: {current_count} total games, {new_games} new since last export")

    # Check if enough new games
    if new_games < config.min_new_games:
        logger.debug(f"Not enough new games ({new_games} < {config.min_new_games})")
        return False

    return True


def process_config(config: ExportConfig, state: ExportState) -> bool:
    """Process a single export configuration."""
    key = f"{config.board_type}_{config.num_players}p"
    logger.info(f"Processing {key}...")

    databases = find_databases(config)

    if not should_export(config, state, databases):
        return False

    success, result = run_export(config, databases)

    if success:
        # Update state
        state.last_export_time[key] = time.time()
        state.last_game_count[key] = count_available_games(databases)
        state.last_export_hash[key] = hashlib.md5(result.encode()).hexdigest()
        save_state(state)
        return True

    return False


def run_once(board_filter: Optional[str] = None, players_filter: Optional[int] = None) -> int:
    """Run export once for all configurations."""
    logger.info("=" * 60)
    logger.info("Starting automated training data export")
    logger.info("=" * 60)

    state = load_state()
    exports_run = 0

    for config in DEFAULT_CONFIGS:
        # Apply filters if specified
        if board_filter and config.board_type != board_filter:
            continue
        if players_filter and config.num_players != players_filter:
            continue

        if process_config(config, state):
            exports_run += 1

    logger.info(f"Completed: {exports_run} exports run")
    return exports_run


def run_daemon(interval_seconds: int = 3600) -> None:
    """Run as continuous daemon."""
    logger.info(f"Starting daemon mode (interval: {interval_seconds}s)")

    while True:
        try:
            run_once()
        except Exception as e:
            logger.error(f"Daemon iteration error: {e}")

        logger.info(f"Sleeping for {interval_seconds} seconds...")
        time.sleep(interval_seconds)


def main():
    parser = argparse.ArgumentParser(description="Automated training data export")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--daemon", action="store_true", help="Run as continuous daemon")
    parser.add_argument("--interval", type=int, default=3600, help="Daemon check interval (seconds)")
    parser.add_argument("--board", type=str, help="Filter by board type")
    parser.add_argument("--players", type=int, help="Filter by player count")
    parser.add_argument("--force", action="store_true", help="Force export regardless of game count")

    args = parser.parse_args()

    if args.force:
        # Reset state to force exports
        for config in DEFAULT_CONFIGS:
            config.min_new_games = 0

    if args.daemon:
        run_daemon(args.interval)
    elif args.once:
        run_once(args.board, args.players)
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
