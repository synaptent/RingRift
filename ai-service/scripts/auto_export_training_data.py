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
import os
import sqlite3
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

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

from scripts.lib.logging_config import setup_script_logging
from scripts.lib.state_manager import StateManager

logger = setup_script_logging("auto_export_training_data")

# Unified game discovery - finds all game databases across all storage patterns
try:
    from app.utils.game_discovery import GameDiscovery
    HAS_GAME_DISCOVERY = True
except ImportError:
    HAS_GAME_DISCOVERY = False
    GameDiscovery = None


@dataclass
class ExportConfig:
    """Export configuration for a board/player combination."""
    board_type: str
    num_players: int
    db_patterns: list[str] = field(default_factory=list)
    output_prefix: str = ""
    min_new_games: int = 100  # Minimum new games before re-export
    encoder_version: str = "v3"
    board_aware_encoding: bool = True
    allow_noncanonical: bool = False
    allow_pending_gate: bool = False
    registry_path: Path | None = None


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
    last_export_time: dict[str, float] = field(default_factory=dict)
    last_game_count: dict[str, int] = field(default_factory=dict)
    last_export_hash: dict[str, str] = field(default_factory=dict)


# Use StateManager for persistent state
_state_manager = StateManager(STATE_FILE, ExportState)


def load_state() -> ExportState:
    """Load export state from file."""
    return _state_manager.load()


def save_state(state: ExportState) -> None:
    """Save export state to file."""
    _state_manager.save(state)


def find_databases(config: ExportConfig) -> list[Path]:
    """Find all database files matching the config patterns.

    First tries configured patterns, then falls back to unified GameDiscovery
    to find databases in any storage location.
    """
    databases = []
    seen_paths: set[Path] = set()

    # Try configured patterns first
    for pattern in config.db_patterns:
        full_pattern = SELFPLAY_DIR / pattern
        # Handle glob patterns
        if "*" in pattern:
            parent = full_pattern.parent
            if parent.exists():
                for match in parent.parent.glob(pattern.split("/")[-2] + "/" + pattern.split("/")[-1]):
                    if match.exists() and match.stat().st_size > 0 and match not in seen_paths:
                        databases.append(match)
                        seen_paths.add(match)
        else:
            if full_pattern.exists() and full_pattern.stat().st_size > 0 and full_pattern not in seen_paths:
                databases.append(full_pattern)
                seen_paths.add(full_pattern)

    # If no databases found via patterns, use unified game discovery
    if not databases and HAS_GAME_DISCOVERY:
        logger.info(f"No databases found via patterns for {config.board_type}/{config.num_players}p, "
                    "trying unified game discovery...")
        discovery = GameDiscovery(AI_SERVICE_ROOT)
        for db_info in discovery.find_databases_for_config(config.board_type, config.num_players):
            if db_info.path not in seen_paths and db_info.game_count > 0:
                databases.append(db_info.path)
                seen_paths.add(db_info.path)
                logger.info(f"  Found via discovery: {db_info.path} ({db_info.game_count:,} games)")

    return databases


def count_available_games(databases: list[Path]) -> int:
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


def run_export(config: ExportConfig, databases: list[Path]) -> tuple[bool, str]:
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

    if getattr(config, "allow_noncanonical", False):
        cmd.append("--allow-noncanonical")
    if getattr(config, "allow_pending_gate", False):
        cmd.append("--allow-pending-gate")
    if getattr(config, "registry_path", None):
        cmd.extend(["--registry", str(config.registry_path)])

    logger.info(f"Running export: {' '.join(cmd[:6])}... ({len(databases)} DBs)")

    # Feb 2026: Best-effort cross-process export coordination
    _config_key = f"{config.board_type}_{config.num_players}p"
    _release_slot = False
    try:
        from app.coordination.export_coordinator import get_export_coordinator
        _coord = get_export_coordinator()
        if not _coord.try_acquire(_config_key):
            logger.info(f"Skipping: cross-process export slot unavailable for {_config_key}")
            return False, "Export slot unavailable"
        _release_slot = True
    except Exception:
        pass  # Fail open if coordinator unavailable

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
                logger.warning("Export completed but file not created")
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
    finally:
        if _release_slot:
            try:
                _coord.release(_config_key)
            except Exception:
                pass


def should_export(config: ExportConfig, state: ExportState, databases: list[Path]) -> bool:
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


def run_once(board_filter: str | None = None, players_filter: int | None = None) -> int:
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
    parser.add_argument(
        "--allow-noncanonical",
        action="store_true",
        help="Allow exporting from non-canonical DBs for legacy/experimental runs.",
    )
    parser.add_argument(
        "--allow-pending-gate",
        action="store_true",
        help="Allow DBs marked pending_gate in TRAINING_DATA_REGISTRY.md.",
    )
    parser.add_argument(
        "--registry",
        type=str,
        default=None,
        help="Path to TRAINING_DATA_REGISTRY.md (default: repo root)",
    )

    args = parser.parse_args()

    if args.force:
        # Reset state to force exports
        for config in DEFAULT_CONFIGS:
            config.min_new_games = 0

    for config in DEFAULT_CONFIGS:
        config.allow_noncanonical = bool(args.allow_noncanonical)
        config.allow_pending_gate = bool(args.allow_pending_gate)
        config.registry_path = Path(args.registry) if args.registry else None

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
