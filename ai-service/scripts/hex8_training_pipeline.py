#!/usr/bin/env python3
"""
Hex8 Training Pipeline

Monitors selfplay databases across cluster nodes, consolidates data,
exports to NPZ format, and trains a HexNeuralNet_v2 model for hex8.

Usage:
    python scripts/hex8_training_pipeline.py --monitor  # Monitor and auto-train
    python scripts/hex8_training_pipeline.py --train    # Train immediately from existing data
    python scripts/hex8_training_pipeline.py --collect  # Just collect/consolidate data
"""

import argparse
import asyncio
import logging
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Try to import SyncCoordinator for enhanced data collection
# Unified sync coordinator (consolidated from deprecated DataSyncManager)
try:
    from app.distributed.sync_coordinator import SyncCoordinator
    HAS_SYNC_COORDINATOR = True

    def get_sync_coordinator():
        return SyncCoordinator.get_instance()
except ImportError:
    HAS_SYNC_COORDINATOR = False
    SyncCoordinator = None

# Unified resource guard - 80% utilization limits (enforced 2025-12-16)
try:
    from app.utils.resource_guard import (
        can_proceed as resource_can_proceed,
        check_disk_space,
        check_memory,
        wait_for_resources,
        LIMITS as RESOURCE_LIMITS,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    resource_can_proceed = lambda **kwargs: True  # type: ignore
    check_disk_space = lambda *args, **kwargs: True  # type: ignore
    check_memory = lambda *args, **kwargs: True  # type: ignore
    wait_for_resources = lambda *args, **kwargs: True  # type: ignore
    RESOURCE_LIMITS = None  # type: ignore

# Unified logging setup (use app.core.logging_config instead of basicConfig)
try:
    from app.core.logging_config import setup_logging
    logger = setup_logging(
        "hex8_training_pipeline",
        log_dir="logs",
        format_style="default",
    )
except ImportError:
    # Fallback if app.core not available
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    logger = logging.getLogger(__name__)

# Configuration - use unified config as single source of truth
try:
    from app.config.unified_config import get_training_threshold
    MIN_GAMES_FOR_TRAINING = get_training_threshold()
except ImportError:
    MIN_GAMES_FOR_TRAINING = 500  # Default from unified_config.py

TARGET_GAMES = MIN_GAMES_FOR_TRAINING * 2  # Target for robust training
POLL_INTERVAL_SECONDS = 300  # Check every 5 minutes

# Remote database locations - loaded from config
def _load_remote_databases() -> dict:
    """Load remote database hosts from config/distributed_hosts.yaml."""
    config_path = Path(__file__).parent.parent / "config" / "distributed_hosts.yaml"
    if not config_path.exists():
        logger.warning("[Pipeline] No config found at %s", config_path)
        return {}

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}

        databases = {}
        for name, info in config.get("hosts", {}).items():
            if info.get("status") not in ("ready", "active"):
                continue

            # Get SSH connection info
            ssh_host = info.get("tailscale_ip") or info.get("ssh_host")
            if not ssh_host:
                continue

            ssh_user = info.get("ssh_user", "ubuntu")
            ringrift_path = info.get("ringrift_path", "~/ringrift/ai-service")
            db_path = f"{ringrift_path}/data/games/hex8_{name.replace('-', '_')}.db"

            entry = {
                "ssh_host": f"{ssh_user}@{ssh_host}",
                "db_path": db_path,
            }

            # Add optional SSH key and port
            if info.get("ssh_key"):
                entry["ssh_key"] = info["ssh_key"]
            if info.get("ssh_port") and info["ssh_port"] != 22:
                entry["ssh_port"] = info["ssh_port"]

            databases[name] = entry

        return databases
    except Exception as e:
        logger.warning("[Pipeline] Error loading config: %s", e)
        return {}


REMOTE_DATABASES = _load_remote_databases()

# Local database (from current local selfplay)
LOCAL_DB = Path("data/games/hex8_training.db")
CONSOLIDATED_DB = Path("data/games/hex8_consolidated.db")
TRAINING_NPZ = Path("data/training/hex8_2p_consolidated.npz")


def get_local_game_count(db_path: Path) -> int:
    """Get number of completed games in a local database."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL"
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.warning(f"Error reading {db_path}: {e}")
        return 0


def get_remote_game_count(node_name: str, config: Dict) -> int:
    """Get number of completed games from a remote database using Python."""
    ssh_host = config["ssh_host"]
    db_path = config["db_path"]
    ssh_key = config.get("ssh_key", "~/.ssh/id_cluster")
    ssh_port = config.get("ssh_port")

    # Expand ~ to absolute path for reliability
    db_abs_path = db_path.replace("~", "$HOME")

    # Build SSH command using Python to query SQLite
    # Use bash -c to properly expand $HOME and handle the python command
    python_cmd = (
        f"bash -c 'python3 -c \"import sqlite3; "
        f"conn=sqlite3.connect(\\\"{db_abs_path}\\\"); "
        f"print(conn.execute(\\\"SELECT COUNT(*) FROM games WHERE winner IS NOT NULL\\\").fetchone()[0])\" "
        f"2>/dev/null || echo 0'"
    )

    ssh_cmd = ["ssh"]
    if ssh_port:
        ssh_cmd.extend(["-p", str(ssh_port)])
    ssh_cmd.extend(["-i", os.path.expanduser(ssh_key), "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=no"])
    ssh_cmd.append(ssh_host)
    ssh_cmd.append(python_cmd)

    try:
        result = subprocess.run(
            ssh_cmd, capture_output=True, text=True, timeout=45
        )
        # Parse output, handling Vast welcome messages
        lines = result.stdout.strip().split("\n")
        for line in reversed(lines):
            line = line.strip()
            if line.isdigit():
                return int(line)
        return 0
    except Exception as e:
        logger.warning(f"Error getting count from {node_name}: {e}")
        return 0


def collect_game_counts() -> Dict[str, int]:
    """Collect game counts from all sources."""
    counts = {}

    # Local database
    counts["local"] = get_local_game_count(LOCAL_DB)

    # Remote databases
    for node_name, config in REMOTE_DATABASES.items():
        counts[node_name] = get_remote_game_count(node_name, config)

    return counts


def rsync_remote_db(node_name: str, config: Dict, dest_dir: Path) -> Optional[Path]:
    """Rsync a remote database to local destination."""
    ssh_host = config["ssh_host"]
    db_path = config["db_path"]
    ssh_key = config.get("ssh_key", "~/.ssh/id_cluster")
    ssh_port = config.get("ssh_port")

    dest_path = dest_dir / f"hex8_{node_name}.db"

    # Build rsync command
    rsync_cmd = ["rsync", "-avz", "--progress"]
    ssh_opts = f"-i {os.path.expanduser(ssh_key)} -o ConnectTimeout=30"
    if ssh_port:
        ssh_opts += f" -p {ssh_port}"
    rsync_cmd.extend(["-e", f"ssh {ssh_opts}"])
    rsync_cmd.append(f"{ssh_host}:{db_path}")
    rsync_cmd.append(str(dest_path))

    try:
        logger.info(f"Syncing {node_name} database...")
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            logger.info(f"Successfully synced {node_name}")
            return dest_path
        else:
            logger.warning(f"Rsync failed for {node_name}: {result.stderr}")
            return None
    except Exception as e:
        logger.warning(f"Error syncing {node_name}: {e}")
        return None


def consolidate_databases(db_paths: List[Path], output_path: Path) -> int:
    """Consolidate multiple SQLite databases into one."""
    if output_path.exists():
        output_path.unlink()

    total_games = 0
    total_moves = 0

    # Create output database with schema
    conn = sqlite3.connect(str(output_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS games (
            game_id TEXT PRIMARY KEY,
            board_type TEXT,
            num_players INTEGER,
            winner INTEGER,
            num_moves INTEGER,
            created_at TEXT,
            completed_at TEXT,
            source TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS moves (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            game_id TEXT,
            move_number INTEGER,
            player INTEGER,
            move_type TEXT,
            from_pos TEXT,
            to_pos TEXT,
            placement_count INTEGER,
            state_before TEXT,
            state_after TEXT,
            FOREIGN KEY (game_id) REFERENCES games(game_id)
        )
    """)

    for db_path in db_paths:
        if not db_path.exists():
            continue

        source_name = db_path.stem
        logger.info(f"Processing {db_path}...")

        try:
            src_conn = sqlite3.connect(str(db_path))

            # Copy games
            games = src_conn.execute(
                "SELECT game_id, board_type, num_players, winner, num_moves, "
                "created_at, completed_at FROM games WHERE winner IS NOT NULL"
            ).fetchall()

            for game in games:
                try:
                    conn.execute(
                        "INSERT OR IGNORE INTO games VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (*game, source_name),
                    )
                    total_games += 1
                except sqlite3.IntegrityError:
                    pass  # Duplicate game_id

            # Copy moves for included games
            game_ids = [g[0] for g in games]
            for game_id in game_ids:
                moves = src_conn.execute(
                    "SELECT game_id, move_number, player, move_type, from_pos, "
                    "to_pos, placement_count, state_before, state_after "
                    "FROM moves WHERE game_id = ?",
                    (game_id,),
                ).fetchall()

                for move in moves:
                    try:
                        conn.execute(
                            "INSERT INTO moves (game_id, move_number, player, move_type, "
                            "from_pos, to_pos, placement_count, state_before, state_after) "
                            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                            move,
                        )
                        total_moves += 1
                    except Exception:
                        pass

            src_conn.close()

        except Exception as e:
            logger.error(f"Error processing {db_path}: {e}")

    conn.commit()
    conn.close()

    logger.info(f"Consolidated {total_games} games with {total_moves} moves")
    return total_games


def export_to_npz(db_path: Path, output_path: Path) -> bool:
    """Export consolidated database to NPZ format for training."""
    logger.info(f"Exporting {db_path} to {output_path}...")

    cmd = [
        sys.executable,
        "scripts/export_replay_dataset.py",
        "--db", str(db_path),
        "--board-type", "hex8",
        "--num-players", "2",
        "--require-completed",
        "--min-moves", "10",
        "--encoder-version", "v2",
        "--output", str(output_path),
    ]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600, cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            logger.info(f"Successfully exported to {output_path}")
            return True
        else:
            logger.error(f"Export failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Export error: {e}")
        return False


def train_hex8_model(
    npz_path: Path,
    use_label_smoothing: bool = True,
    use_hex_augmentation: bool = True,
    enable_curriculum: bool = True,
) -> bool:
    """Train a HexNeuralNet_v2 model on hex8 data.

    Args:
        npz_path: Path to training data NPZ file
        use_label_smoothing: Enable policy label smoothing (0.05) for regularization
        use_hex_augmentation: Enable D6 symmetry augmentation (12x data expansion)
        enable_curriculum: Enable curriculum learning (weights late-game positions higher)
    """
    logger.info(f"Training hex8 model from {npz_path}...")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"ringrift_hex8_2p_{timestamp}"

    cmd = [
        sys.executable,
        "-m", "app.training.train",
        "--data-path", str(npz_path),
        "--board-type", "hex8",
        "--model-id", model_name,
        "--epochs", "50",
        "--batch-size", "64",
        "--learning-rate", "2e-3",
        "--early-stopping-patience", "10",
        "--checkpoint-dir", "models",
    ]

    # Add new training improvements
    if use_label_smoothing:
        cmd.extend(["--policy-label-smoothing", "0.05"])
        logger.info("Using policy label smoothing (0.05)")

    if use_hex_augmentation:
        cmd.append("--augment-hex-symmetry")
        logger.info("Using D6 hex symmetry augmentation (12x)")

    # Enable curriculum learning for better late-game position weighting
    if enable_curriculum:
        cmd.extend(["--use-integrated-enhancements", "--enable-curriculum"])
        logger.info("Using curriculum learning (late-game position weighting)")

    try:
        logger.info(f"Starting training with command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=7200, cwd=Path(__file__).parent.parent
        )
        if result.returncode == 0:
            logger.info(f"Training completed successfully!")
            logger.info(result.stdout[-2000:] if len(result.stdout) > 2000 else result.stdout)
            return True
        else:
            logger.error(f"Training failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"Training error: {e}")
        return False


def monitor_and_train():
    """Monitor selfplay progress and trigger training when ready."""
    logger.info("Starting hex8 training pipeline monitor...")
    logger.info(f"Target: {TARGET_GAMES} games, minimum: {MIN_GAMES_FOR_TRAINING}")

    while True:
        counts = collect_game_counts()
        total = sum(counts.values())

        logger.info(f"\n{'='*50}")
        logger.info(f"Hex8 Selfplay Progress: {total}/{TARGET_GAMES} games")
        logger.info("Per-node breakdown:")
        for node, count in sorted(counts.items()):
            logger.info(f"  {node}: {count} games")

        if total >= MIN_GAMES_FOR_TRAINING:
            logger.info(f"\nReached {total} games - starting data collection and training!")
            run_collection_and_training()
            break
        else:
            logger.info(f"\nWaiting for more data. Next check in {POLL_INTERVAL_SECONDS}s...")
            time.sleep(POLL_INTERVAL_SECONDS)


async def _collect_via_sync_coordinator() -> List[Path]:
    """Collect databases using SyncCoordinator with fallback transport methods."""
    collected = []
    try:
        coordinator = get_sync_coordinator()

        # Sync games from cluster using aria2 → ssh → p2p fallback chain
        stats = await coordinator.sync_games()

        if stats.files_synced > 0:
            logger.info(f"SyncCoordinator synced {stats.files_synced} files via {stats.transport_used}")
            # Games are synced to the provider's selfplay_dir
            # Look for hex8 databases there
            selfplay_dir = coordinator._provider.selfplay_dir
            for db_path in selfplay_dir.glob("*hex8*.db"):
                if db_path.exists():
                    collected.append(db_path)
                    logger.info(f"SyncCoordinator collected: {db_path}")

    except Exception as e:
        logger.warning(f"SyncCoordinator collection failed: {e}")

    return collected


def run_collection_and_training():
    """Collect data from all sources and run training."""
    # Create temp directory for synced databases
    sync_dir = Path("data/games/hex8_sync")
    sync_dir.mkdir(parents=True, exist_ok=True)

    db_paths = []
    failed_nodes = set(REMOTE_DATABASES.keys())

    # Try SyncCoordinator first for better connectivity (aria2, ssh, p2p fallback)
    if HAS_SYNC_COORDINATOR:
        logger.info("Trying SyncCoordinator for enhanced data collection...")
        collected = asyncio.run(_collect_via_sync_coordinator())
        if collected:
            db_paths.extend(collected)
            # Track which nodes succeeded via SyncCoordinator (based on filename pattern)
            for db_path in collected:
                # Filename format: nodename_hex8_*.db
                stem = db_path.stem
                for node_name in list(failed_nodes):
                    if node_name in stem:
                        failed_nodes.discard(node_name)
            logger.info(f"SyncCoordinator collected {len(collected)} databases")

    # Fallback to direct SSH/rsync for nodes that failed via DataSync
    if failed_nodes:
        logger.info(f"Falling back to SSH for {len(failed_nodes)} nodes: {failed_nodes}")
        for node_name in list(failed_nodes):
            if node_name in REMOTE_DATABASES:
                config = REMOTE_DATABASES[node_name]
                db_path = rsync_remote_db(node_name, config, sync_dir)
                if db_path:
                    db_paths.append(db_path)
                    failed_nodes.discard(node_name)

    # Log any nodes that still failed
    if failed_nodes:
        logger.warning(f"Failed to collect from nodes: {failed_nodes}")

    # Add local database
    if LOCAL_DB.exists():
        db_paths.append(LOCAL_DB)

    if not db_paths:
        logger.error("No databases to consolidate!")
        return False

    # Consolidate
    total_games = consolidate_databases(db_paths, CONSOLIDATED_DB)

    if total_games < MIN_GAMES_FOR_TRAINING:
        logger.warning(f"Only {total_games} games consolidated, need {MIN_GAMES_FOR_TRAINING}")
        return False

    # Export to NPZ
    TRAINING_NPZ.parent.mkdir(parents=True, exist_ok=True)
    if not export_to_npz(CONSOLIDATED_DB, TRAINING_NPZ):
        return False

    # Train model
    if not train_hex8_model(TRAINING_NPZ):
        return False

    logger.info("\n" + "="*50)
    logger.info("HEX8 TRAINING PIPELINE COMPLETE!")
    logger.info(f"Total games: {total_games}")
    logger.info(f"Training data: {TRAINING_NPZ}")
    logger.info("="*50)

    return True


def main():
    parser = argparse.ArgumentParser(description="Hex8 Training Pipeline")
    parser.add_argument("--monitor", action="store_true", help="Monitor and auto-train")
    parser.add_argument("--train", action="store_true", help="Train immediately")
    parser.add_argument("--collect", action="store_true", help="Just collect/consolidate")
    parser.add_argument("--status", action="store_true", help="Show current status")
    args = parser.parse_args()

    # Ensure we're in the right directory
    os.chdir(Path(__file__).parent.parent)

    # Create log directory
    Path("logs").mkdir(exist_ok=True)

    if args.status:
        counts = collect_game_counts()
        total = sum(counts.values())
        print(f"\nHex8 Selfplay Status: {total} total games")
        print("-" * 40)
        for node, count in sorted(counts.items()):
            print(f"  {node}: {count} games")
        return

    if args.collect:
        run_collection_and_training()
    elif args.train:
        if TRAINING_NPZ.exists():
            train_hex8_model(TRAINING_NPZ)
        else:
            logger.error(f"Training data not found: {TRAINING_NPZ}")
            logger.info("Run with --collect first to gather data")
    elif args.monitor:
        monitor_and_train()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
