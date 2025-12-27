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
import os
import sqlite3
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

# Ensure ai-service root on path for scripts/lib imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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
        LIMITS as RESOURCE_LIMITS,
        can_proceed as resource_can_proceed,
        check_disk_space,
        check_memory,
        wait_for_resources,
    )
    HAS_RESOURCE_GUARD = True
except ImportError:
    HAS_RESOURCE_GUARD = False
    def resource_can_proceed(**kwargs):
        return True  # type: ignore
    def check_disk_space(*args, **kwargs):
        return True  # type: ignore
    def check_memory(*args, **kwargs):
        return True  # type: ignore
    def wait_for_resources(*args, **kwargs):
        return True  # type: ignore
    RESOURCE_LIMITS = None  # type: ignore

# Unified game discovery - finds all game databases across all storage patterns
try:
    from app.utils.game_discovery import GameDiscovery
    HAS_GAME_DISCOVERY = True
except ImportError:
    HAS_GAME_DISCOVERY = False
    GameDiscovery = None

# Dec 2025: Bandwidth limiting for rsync transfers
try:
    from app.config.cluster_config import get_node_bandwidth_kbs
    HAS_BANDWIDTH_CONFIG = True
except ImportError:
    HAS_BANDWIDTH_CONFIG = False
    def get_node_bandwidth_kbs(node_name: str, config_path=None) -> int:
        return 50 * 1024  # Default 50 MB/s in KB/s

from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("hex8_training_pipeline")

# Configuration - use unified config as single source of truth
try:
    from app.config.unified_config import get_training_threshold
    MIN_GAMES_FOR_TRAINING = get_training_threshold()
except ImportError:
    MIN_GAMES_FOR_TRAINING = 500  # Default from unified_config.py

TARGET_GAMES = MIN_GAMES_FOR_TRAINING * 2  # Target for robust training
POLL_INTERVAL_SECONDS = 300  # Check every 5 minutes

# Databases that contain hex8 game data (board_type='hex8' or 'hexagonal' with 2p)
# The actual data is in central databases, not per-node hex8_*.db files
CENTRAL_DATABASES = [
    "data/games/selfplay.db",
    "data/games/jsonl_aggregated.db",
    "data/selfplay/canonical_hex8_2p.db",
]

# Board types to consider as hex8 data
HEX8_BOARD_TYPES = ("hex8", "hexagonal")


# Remote database locations - loaded from config
def _load_remote_databases() -> dict:
    """Load remote database hosts from config/distributed_hosts.yaml.

    Note: hex8 data lives in central databases (selfplay.db, jsonl_aggregated.db)
    rather than per-node hex8_{nodename}.db files. This function returns hosts
    for syncing those central databases.
    """
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

            # Hex8 data is in central databases, not per-node files
            # We'll sync selfplay.db and jsonl_aggregated.db from each node
            entry = {
                "ssh_host": f"{ssh_user}@{ssh_host}",
                "ringrift_path": ringrift_path,
                "central_dbs": [f"{ringrift_path}/{db}" for db in CENTRAL_DATABASES],
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

# Local database paths
LOCAL_CENTRAL_DBS = [Path(db) for db in CENTRAL_DATABASES]
CONSOLIDATED_DB = Path("data/games/hex8_consolidated.db")
TRAINING_NPZ = Path("data/training/hex8_2p_consolidated.npz")


def get_local_hex8_game_count() -> dict[str, int]:
    """Get number of hex8 games from all local databases using unified discovery.

    Returns dict mapping db_name -> count of hex8/hexagonal games.
    """
    counts = {}

    # Use unified game discovery if available
    if HAS_GAME_DISCOVERY:
        discovery = GameDiscovery()
        # Get hex8 2p games (primary focus)
        for db_info in discovery.find_databases_for_config("hex8", 2):
            if db_info.game_count > 0:
                counts[f"local:{db_info.path.name}"] = db_info.game_count
        # Also get hexagonal 2p games (same board topology)
        for db_info in discovery.find_databases_for_config("hexagonal", 2):
            key = f"local:{db_info.path.name}"
            if key not in counts and db_info.game_count > 0:
                counts[key] = db_info.game_count
        return counts

    # Fallback to manual search if game discovery not available
    for db_path in LOCAL_CENTRAL_DBS:
        if not db_path.exists():
            continue
        try:
            conn = sqlite3.connect(str(db_path))
            # Count hex8 and hexagonal board types
            cursor = conn.execute(
                "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL "
                "AND board_type IN (?, ?)",
                HEX8_BOARD_TYPES,
            )
            count = cursor.fetchone()[0]
            conn.close()
            if count > 0:
                counts[db_path.name] = count
        except Exception as e:
            logger.warning(f"Error reading {db_path}: {e}")
    return counts


def get_local_game_count(db_path: Path) -> int:
    """Get number of completed hex8 games in a local database."""
    if not db_path.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.execute(
            "SELECT COUNT(*) FROM games WHERE winner IS NOT NULL "
            "AND board_type IN (?, ?)",
            HEX8_BOARD_TYPES,
        )
        count = cursor.fetchone()[0]
        conn.close()
        return count
    except Exception as e:
        logger.warning(f"Error reading {db_path}: {e}")
        return 0


def get_remote_hex8_game_count(node_name: str, config: dict) -> dict[str, int]:
    """Get number of hex8 games from remote central databases.

    Returns dict mapping db_name -> count of hex8/hexagonal games.
    """
    ssh_host = config["ssh_host"]
    ssh_key = config.get("ssh_key", "~/.ssh/id_cluster")
    ssh_port = config.get("ssh_port")
    ringrift_path = config.get("ringrift_path", "~/ringrift/ai-service")

    counts = {}
    for db_rel_path in CENTRAL_DATABASES:
        db_path = f"{ringrift_path}/{db_rel_path}"
        db_abs_path = db_path.replace("~", "$HOME")

        # Query for hex8/hexagonal board types
        python_cmd = (
            f"bash -c 'python3 -c \"import sqlite3; import os; "
            f"db=os.path.expandvars(\\\"{db_abs_path}\\\"); "
            f"conn=sqlite3.connect(db) if os.path.exists(db) else None; "
            f"print(conn.execute(\\\"SELECT COUNT(*) FROM games WHERE winner IS NOT NULL "
            f"AND board_type IN (\\\\\\\"hex8\\\\\\\", \\\\\\\"hexagonal\\\\\\\")\\\").fetchone()[0] if conn else 0)\" "
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
                    count = int(line)
                    if count > 0:
                        db_name = Path(db_rel_path).name
                        counts[f"{node_name}:{db_name}"] = count
                    break
        except Exception as e:
            logger.warning(f"Error getting count from {node_name}/{db_rel_path}: {e}")

    return counts


def get_remote_game_count(node_name: str, config: dict) -> int:
    """Get total number of hex8 games from a remote node's central databases."""
    counts = get_remote_hex8_game_count(node_name, config)
    return sum(counts.values())


def collect_game_counts() -> dict[str, int]:
    """Collect hex8 game counts from all sources (local and remote central databases)."""
    counts = {}

    # Local central databases
    local_counts = get_local_hex8_game_count()
    for db_name, count in local_counts.items():
        counts[f"local:{db_name}"] = count

    # Remote databases - sample a few key nodes to avoid long waits
    # Focus on nodes most likely to have hex8 data
    key_nodes = ["lambda-a10", "lambda-h100", "vast-3070b"]
    for node_name, config in REMOTE_DATABASES.items():
        # Only check key nodes in quick mode
        if node_name not in key_nodes and len(counts) > 5:
            continue
        remote_counts = get_remote_hex8_game_count(node_name, config)
        counts.update(remote_counts)

    return counts


def rsync_remote_db(node_name: str, config: dict, dest_dir: Path, db_name: str = "selfplay.db") -> Path | None:
    """Rsync a remote central database to local destination.

    Args:
        node_name: Name of the remote node
        config: Node configuration dict
        dest_dir: Local destination directory
        db_name: Name of the database file to sync (default: selfplay.db)

    Returns:
        Path to synced file or None if failed
    """
    ssh_host = config["ssh_host"]
    ssh_key = config.get("ssh_key", "~/.ssh/id_cluster")
    ssh_port = config.get("ssh_port")
    ringrift_path = config.get("ringrift_path", "~/ringrift/ai-service")

    # Find the right remote path
    remote_db_path = f"{ringrift_path}/data/games/{db_name}"
    dest_path = dest_dir / f"{node_name}_{db_name}"

    # Build rsync command with checksum verification (December 2025)
    bwlimit_kbs = get_node_bandwidth_kbs(node_name)  # Dec 2025: Bandwidth limit
    rsync_cmd = ["rsync", "-avz", "--progress", "--checksum", f"--bwlimit={bwlimit_kbs}"]
    ssh_opts = f"-i {os.path.expanduser(ssh_key)} -o ConnectTimeout=30 -o StrictHostKeyChecking=no"
    if ssh_port:
        ssh_opts += f" -p {ssh_port}"
    rsync_cmd.extend(["-e", f"ssh {ssh_opts}"])
    rsync_cmd.append(f"{ssh_host}:{remote_db_path}")
    rsync_cmd.append(str(dest_path))

    try:
        logger.info(f"Syncing {node_name}/{db_name}...")
        result = subprocess.run(rsync_cmd, capture_output=True, text=True, timeout=600)
        if result.returncode == 0:
            logger.info(f"Successfully synced {node_name}/{db_name}")
            return dest_path
        else:
            logger.warning(f"Rsync failed for {node_name}/{db_name}: {result.stderr}")
            return None
    except Exception as e:
        logger.warning(f"Error syncing {node_name}/{db_name}: {e}")
        return None


def consolidate_databases(db_paths: list[Path], output_path: Path) -> int:
    """Consolidate hex8 games from multiple SQLite databases into one.

    Only copies games where board_type is 'hex8' or 'hexagonal'.
    """
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

            # Check which columns exist in source database
            cursor = src_conn.execute("PRAGMA table_info(games)")
            columns = {row[1] for row in cursor.fetchall()}

            # Build SELECT based on available columns
            select_cols = ["game_id", "board_type", "num_players", "winner"]
            if "num_moves" in columns:
                select_cols.append("num_moves")
            else:
                select_cols.append("0 as num_moves")  # Default value
            if "created_at" in columns:
                select_cols.append("created_at")
            else:
                select_cols.append("NULL as created_at")
            if "completed_at" in columns:
                select_cols.append("completed_at")
            else:
                select_cols.append("NULL as completed_at")

            select_sql = ", ".join(select_cols)

            # Copy only hex8/hexagonal games
            games = src_conn.execute(
                f"SELECT {select_sql} FROM games WHERE winner IS NOT NULL "
                "AND board_type IN (?, ?)",
                HEX8_BOARD_TYPES,
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

            # Check if moves table exists and copy moves for included games
            cursor = src_conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='moves'")
            has_moves_table = cursor.fetchone() is not None

            if has_moves_table:
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
                        except (sqlite3.Error, ValueError, TypeError):
                            pass

            src_conn.close()
            logger.info(f"  -> Added {len(games)} hex8 games from {source_name}")

        except Exception as e:
            logger.error(f"Error processing {db_path}: {e}")

    conn.commit()
    conn.close()

    logger.info(f"Consolidated {total_games} hex8 games with {total_moves} moves")
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
            logger.info("Training completed successfully!")
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


async def _collect_via_sync_coordinator() -> list[Path]:
    """Collect databases using SyncCoordinator with fallback transport methods.

    Returns paths to databases that may contain hex8 data. The consolidation
    step will filter to only hex8/hexagonal games.
    """
    collected = []
    try:
        coordinator = get_sync_coordinator()

        # Sync games from cluster using aria2 → ssh → p2p fallback chain
        stats = await coordinator.sync_games()

        if stats.files_synced > 0:
            logger.info(f"SyncCoordinator synced {stats.files_synced} files via {stats.transport_used}")

            # Look for databases that might contain hex8 data
            # Check both the selfplay_dir and games directories
            search_dirs = [
                coordinator._provider.selfplay_dir,
                coordinator._provider.selfplay_dir.parent / "games",
            ]

            for search_dir in search_dirs:
                if not search_dir.exists():
                    continue
                # Look for central databases that contain hex8 data
                for pattern in ["selfplay.db", "jsonl_aggregated.db", "*hex*.db"]:
                    for db_path in search_dir.glob(pattern):
                        if db_path.exists() and db_path.stat().st_size > 0:
                            if db_path not in collected:
                                collected.append(db_path)
                                logger.info(f"SyncCoordinator found: {db_path}")

    except Exception as e:
        logger.warning(f"SyncCoordinator collection failed: {e}")

    return collected


def run_collection_and_training():
    """Collect hex8 data from all local databases and run training.

    Uses unified game discovery to find ALL databases containing hex8 data,
    regardless of where they're stored (central DBs, selfplay dirs, p2p dirs, etc.).
    """
    # Create temp directory for synced databases
    sync_dir = Path("data/games/hex8_sync")
    sync_dir.mkdir(parents=True, exist_ok=True)

    db_paths = []
    seen_paths: set[Path] = set()

    # Use unified game discovery if available (preferred method)
    if HAS_GAME_DISCOVERY:
        logger.info("Using unified game discovery to find all hex8 databases...")
        discovery = GameDiscovery()

        # Get all databases with hex8 games
        for db_info in discovery.find_databases_for_config("hex8", 2):
            if db_info.path not in seen_paths and db_info.game_count > 0:
                db_paths.append(db_info.path)
                seen_paths.add(db_info.path)
                logger.info(f"Found {db_info.path}: {db_info.game_count:,} hex8 games")

        # Also get hexagonal 2p games (same board topology)
        for db_info in discovery.find_databases_for_config("hexagonal", 2):
            if db_info.path not in seen_paths and db_info.game_count > 0:
                db_paths.append(db_info.path)
                seen_paths.add(db_info.path)
                logger.info(f"Found {db_info.path}: {db_info.game_count:,} hexagonal games")
    else:
        # Fallback to manual search
        logger.info("Falling back to manual database search...")

        # Add local central databases (primary source of hex8 data)
        for db_path in LOCAL_CENTRAL_DBS:
            if db_path.exists() and db_path.stat().st_size > 0:
                db_paths.append(db_path)
                logger.info(f"Found local database: {db_path}")

        # Also check for any existing hex8 databases in data/games
        games_dir = Path("data/games")
        for db_path in games_dir.glob("hex8*.db"):
            if db_path.stat().st_size > 0 and db_path not in db_paths:
                db_paths.append(db_path)
                logger.info(f"Found hex8 database: {db_path}")

        # Also check hexagonal databases
        for db_path in games_dir.glob("hexagonal*.db"):
            if db_path.stat().st_size > 0 and db_path not in db_paths:
                db_paths.append(db_path)
                logger.info(f"Found hexagonal database: {db_path}")

    # Try SyncCoordinator for any additional remote data
    if HAS_SYNC_COORDINATOR:
        logger.info("Trying SyncCoordinator for enhanced data collection...")
        try:
            collected = asyncio.run(_collect_via_sync_coordinator())
            if collected:
                for db_path in collected:
                    if db_path not in seen_paths:
                        db_paths.append(db_path)
                        seen_paths.add(db_path)
                logger.info(f"SyncCoordinator collected {len(collected)} additional databases")
        except Exception as e:
            logger.warning(f"SyncCoordinator failed: {e}")

    if not db_paths:
        logger.error("No databases found with hex8 data!")
        return False

    logger.info(f"\nConsolidating hex8 data from {len(db_paths)} databases...")

    # Consolidate (will filter to hex8/hexagonal games only)
    total_games = consolidate_databases(db_paths, CONSOLIDATED_DB)

    if total_games < MIN_GAMES_FOR_TRAINING:
        logger.warning(f"Only {total_games} hex8 games consolidated, need {MIN_GAMES_FOR_TRAINING}")
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
