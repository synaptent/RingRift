#!/usr/bin/env python3
"""Clean up phantom Elo entries - models that exist in the database but not as files.

This script identifies and optionally removes Elo entries for models that:
1. Have a model_path in the database but the actual model file doesn't exist
2. Have player count mismatch (e.g., 4-player model registered as 2-player)

Usage:
    # Dry run (default) - show phantom entries without deleting
    python scripts/cleanup_phantom_elo_entries.py

    # Filter by config
    python scripts/cleanup_phantom_elo_entries.py --config square8_2p

    # Actually delete phantom entries
    python scripts/cleanup_phantom_elo_entries.py --execute

    # Check cluster nodes too (slower but more thorough)
    python scripts/cleanup_phantom_elo_entries.py --cluster-check

    # Check OWC backup drive on mac-studio (requires SSH access)
    python scripts/cleanup_phantom_elo_entries.py --owc-check

    # Validate player count in model files
    python scripts/cleanup_phantom_elo_entries.py --validate-players
"""

import argparse
import json
import logging
import shutil
import sqlite3
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.config.cluster_config import get_cluster_nodes

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Baselines that don't have model files (they're built-in AI types)
BASELINE_PREFIXES = [
    "baseline_random",
    "baseline_heuristic",
    "none:random",
    "none:heuristic",
    "random",
    "heuristic",
    "tier1_",
    "tier2_",
    "tier3_",
    "d1",
    "d2",
    "d3",
    "d4",
    "d5",
]

# Directories to search for models
MODEL_DIRS = [
    Path("models"),
    Path("models_essential"),
    Path("models/recovered_high_elo"),
    Path("backups/models"),
    Path("data/models"),
]

# OWC backup configuration (mac-studio)
OWC_CONFIG = {
    "host": "100.107.168.125",
    "user": "armand",
    "base_path": "/Volumes/RingRift-Data",
    "model_dirs": [
        "canonical_models",
        "model_checkpoints",
        "trained_models",
        "model_registry",
    ],
}


def is_baseline(participant_id: str) -> bool:
    """Check if participant is a baseline AI (doesn't need model file)."""
    pid_lower = participant_id.lower()
    for prefix in BASELINE_PREFIXES:
        if pid_lower.startswith(prefix.lower()):
            return True
    if "random" in pid_lower or "heuristic" in pid_lower:
        return True
    return False


def find_model_locally(model_path: str | None, participant_id: str) -> bool:
    """Check if model exists locally."""
    if not model_path:
        return False

    path = Path(model_path)

    # Check absolute path
    if path.is_absolute() and path.exists():
        return True

    # Check relative paths in known directories
    for model_dir in MODEL_DIRS:
        # Try as-is
        candidate = model_dir / path
        if candidate.exists():
            return True
        # Try just the filename
        candidate = model_dir / path.name
        if candidate.exists():
            return True

    # Check cwd
    if path.exists():
        return True

    # Try to find by participant_id pattern
    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            continue
        for pattern in [f"*{participant_id}*", f"*{participant_id.split('_')[0]}*"]:
            matches = list(model_dir.glob(pattern))
            if matches:
                return True

    return False


def check_model_on_cluster(model_path: str, nodes: list) -> str | None:
    """Check if model exists on any cluster node. Returns node name if found."""
    if not model_path:
        return None

    path = Path(model_path)
    remote_paths = [
        f"~/ringrift/ai-service/models/{path.name}",
        f"/workspace/ringrift/ai-service/models/{path.name}",
        f"/root/ringrift/ai-service/models/{path.name}",
        str(model_path),
    ]

    for node in nodes:
        ssh_host = node.ssh_host or node.tailscale_ip
        if not ssh_host:
            continue

        for remote_path in remote_paths:
            try:
                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                     f"{node.ssh_user}@{ssh_host}", f"test -f {remote_path}"],
                    capture_output=True,
                    timeout=10,
                )
                if result.returncode == 0:
                    return node.name
            except (subprocess.TimeoutExpired, Exception):
                continue

    return None


def find_model_by_participant_id(participant_id: str) -> bool:
    """Try to find a model file based on participant ID pattern."""
    # Common model file patterns to try
    patterns = [
        f"{participant_id}.pth",
        f"{participant_id}.pt",
        f"*{participant_id}*.pth",
        f"*{participant_id}*.pt",
    ]

    for model_dir in MODEL_DIRS:
        if not model_dir.exists():
            continue

        for pattern in patterns:
            matches = list(model_dir.glob(pattern))
            if matches:
                return True

    # For nn_baseline patterns, check if they might be temporary training artifacts
    # that were never saved (common cause of phantoms)
    return False


def check_model_on_owc(model_path: str, participant_id: str) -> tuple[bool, str | None]:
    """Check if model exists on OWC backup drive.

    Returns (found, path) tuple.
    """
    if not model_path:
        return False, None

    path = Path(model_path)
    patterns_to_check = [
        path.name,
        f"{participant_id}.pth",
        f"*{participant_id}*.pth",
    ]

    for owc_dir in OWC_CONFIG["model_dirs"]:
        for pattern in patterns_to_check:
            remote_path = f"{OWC_CONFIG['base_path']}/{owc_dir}/{pattern}"
            try:
                # Use find for glob patterns, test -f for exact paths
                if "*" in pattern:
                    cmd = f"find {OWC_CONFIG['base_path']}/{owc_dir} -name '{pattern}' -type f | head -1"
                else:
                    cmd = f"test -f '{remote_path}' && echo '{remote_path}'"

                result = subprocess.run(
                    ["ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                     f"{OWC_CONFIG['user']}@{OWC_CONFIG['host']}", cmd],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True, result.stdout.strip()
            except (subprocess.TimeoutExpired, Exception) as e:
                logger.debug(f"OWC check failed for {pattern}: {e}")
                continue

    return False, None


def validate_model_player_count(model_path: str, expected_players: int) -> tuple[bool, int | None]:
    """Validate that a model file has the expected player count.

    Returns (is_valid, actual_players) tuple.
    """
    try:
        import torch
        path = Path(model_path)

        if not path.exists():
            return True, None  # Can't validate non-existent files

        checkpoint = torch.load(path, map_location="cpu", weights_only=False)

        # Try to get num_players from metadata
        actual_players = checkpoint.get("num_players")

        if actual_players is None:
            # Infer from value head shape
            state_dict = checkpoint.get("model_state_dict", checkpoint.get("state_dict", checkpoint))
            if isinstance(state_dict, dict):
                for key in ["value_fc2.weight", "value_head.fc2.weight"]:
                    if key in state_dict:
                        actual_players = state_dict[key].shape[0]
                        break

        if actual_players is None:
            logger.debug(f"Could not determine player count for {model_path}")
            return True, None  # Can't determine, assume valid

        return actual_players == expected_players, actual_players

    except Exception as e:
        logger.debug(f"Could not validate player count for {model_path}: {e}")
        return True, None  # Assume valid on error


def validate_remote_model_player_count(
    remote_path: str, expected_players: int, host: str, user: str
) -> tuple[bool, int | None]:
    """Validate player count of a remote model via SSH.

    Returns (is_valid, actual_players) tuple.
    """
    try:
        # Use Python on remote host to extract player count
        python_cmd = f"""
import torch
import sys
try:
    cp = torch.load('{remote_path}', map_location='cpu', weights_only=False)
    np = cp.get('num_players')
    if np is None:
        sd = cp.get('model_state_dict', cp.get('state_dict', cp))
        if isinstance(sd, dict):
            for k in ['value_fc2.weight', 'value_head.fc2.weight']:
                if k in sd:
                    np = sd[k].shape[0]
                    break
    print(np if np else 'unknown')
except Exception as e:
    print(f'error: {{e}}')
"""
        result = subprocess.run(
            ["ssh", "-o", "ConnectTimeout=10", "-o", "StrictHostKeyChecking=no",
             f"{user}@{host}", f"python3 -c \"{python_cmd}\""],
            capture_output=True,
            text=True,
            timeout=30,
        )

        if result.returncode == 0:
            output = result.stdout.strip()
            if output.isdigit():
                actual = int(output)
                return actual == expected_players, actual
            elif output == "unknown":
                return True, None

        return True, None  # Assume valid on error

    except (subprocess.TimeoutExpired, Exception) as e:
        logger.debug(f"Remote player count validation failed: {e}")
        return True, None


def get_phantom_entries(
    db_path: Path,
    config_filter: str | None = None,
    cluster_check: bool = False,
    owc_check: bool = False,
    validate_players: bool = False,
) -> list[dict]:
    """Find all phantom Elo entries (in DB but file doesn't exist or has wrong player count).

    Checks:
    1. Participants with model_path set but file doesn't exist
    2. Elo ratings for participants that don't exist in participants table
       AND don't correspond to any actual model file
    3. (Optional) Models on OWC backup drive
    4. (Optional) Player count mismatches in model files
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    phantoms = []
    recovered = []  # Models found on OWC that could be recovered
    mismatches = []  # Player count mismatches
    cluster_nodes = []

    if cluster_check:
        try:
            cluster_nodes = list(get_cluster_nodes().values())
            logger.info(f"Will check {len(cluster_nodes)} cluster nodes")
        except Exception as e:
            logger.warning(f"Could not load cluster nodes: {e}")

    if owc_check:
        logger.info(f"Will check OWC backup at {OWC_CONFIG['host']}:{OWC_CONFIG['base_path']}")

    # Query 1: Participants with model_path but file doesn't exist
    query1 = """
        SELECT DISTINCT p.participant_id, p.model_path, p.ai_type,
               r.board_type, r.num_players, r.rating, r.games_played
        FROM participants p
        LEFT JOIN elo_ratings r ON p.participant_id = r.participant_id
        WHERE p.model_path IS NOT NULL AND p.model_path != ''
    """
    params1 = []

    if config_filter:
        parts = config_filter.replace("_", " ").split()
        if len(parts) >= 2:
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
            query1 += " AND r.board_type = ? AND r.num_players = ?"
            params1.extend([board_type, num_players])

    for row in conn.execute(query1, params1).fetchall():
        participant_id = row["participant_id"]
        if is_baseline(participant_id):
            continue

        model_path = row["model_path"]
        num_players = row["num_players"]
        found_locally = find_model_locally(model_path, participant_id)

        # Check player count if found locally and validation requested
        if found_locally and validate_players and num_players:
            is_valid, actual_players = validate_model_player_count(model_path, num_players)
            if not is_valid:
                mismatches.append({
                    "participant_id": participant_id,
                    "model_path": model_path,
                    "board_type": row["board_type"],
                    "num_players": num_players,
                    "actual_players": actual_players,
                    "rating": row["rating"],
                    "games_played": row["games_played"],
                    "ai_type": row["ai_type"],
                    "source": "player_count_mismatch",
                    "reason": f"Expected {num_players} players, model has {actual_players}",
                })
                continue

        if found_locally:
            continue

        if cluster_check and cluster_nodes:
            if check_model_on_cluster(model_path, cluster_nodes):
                continue

        # Check OWC backup
        if owc_check:
            found_on_owc, owc_path = check_model_on_owc(model_path, participant_id)
            if found_on_owc:
                # Validate player count on OWC
                if validate_players and num_players:
                    is_valid, actual = validate_remote_model_player_count(
                        owc_path, num_players, OWC_CONFIG["host"], OWC_CONFIG["user"]
                    )
                    if not is_valid:
                        mismatches.append({
                            "participant_id": participant_id,
                            "model_path": model_path,
                            "owc_path": owc_path,
                            "board_type": row["board_type"],
                            "num_players": num_players,
                            "actual_players": actual,
                            "rating": row["rating"],
                            "games_played": row["games_played"],
                            "ai_type": row["ai_type"],
                            "source": "owc_player_mismatch",
                            "reason": f"OWC model has {actual} players, registered as {num_players}",
                        })
                        continue

                recovered.append({
                    "participant_id": participant_id,
                    "model_path": model_path,
                    "owc_path": owc_path,
                    "board_type": row["board_type"],
                    "num_players": num_players,
                    "rating": row["rating"],
                    "games_played": row["games_played"],
                    "ai_type": row["ai_type"],
                })
                continue

        phantoms.append({
            "participant_id": participant_id,
            "model_path": model_path,
            "board_type": row["board_type"],
            "num_players": row["num_players"],
            "rating": row["rating"],
            "games_played": row["games_played"],
            "ai_type": row["ai_type"],
            "source": "participants_table",
        })

    # Query 2: Elo ratings for non-existent participants (orphaned ratings)
    # These are entries in elo_ratings that either:
    # - Have no corresponding participants entry, OR
    # - Don't correspond to any actual model file
    query2 = """
        SELECT DISTINCT r.participant_id, r.board_type, r.num_players,
               r.rating, r.games_played
        FROM elo_ratings r
        LEFT JOIN participants p ON r.participant_id = p.participant_id
        WHERE p.participant_id IS NULL
    """
    params2 = []

    if config_filter:
        parts = config_filter.replace("_", " ").split()
        if len(parts) >= 2:
            board_type = parts[0]
            num_players = int(parts[1].replace("p", ""))
            query2 += " AND r.board_type = ? AND r.num_players = ?"
            params2.extend([board_type, num_players])

    for row in conn.execute(query2, params2).fetchall():
        participant_id = row["participant_id"]
        if is_baseline(participant_id):
            continue

        # Check if model might exist as a file
        if find_model_by_participant_id(participant_id):
            continue

        if cluster_check and cluster_nodes:
            if check_model_on_cluster(f"{participant_id}.pth", cluster_nodes):
                continue

        # Check OWC backup for orphaned entries
        if owc_check:
            found_on_owc, owc_path = check_model_on_owc(None, participant_id)
            if found_on_owc:
                num_players = row["num_players"]
                if validate_players and num_players:
                    is_valid, actual = validate_remote_model_player_count(
                        owc_path, num_players, OWC_CONFIG["host"], OWC_CONFIG["user"]
                    )
                    if not is_valid:
                        mismatches.append({
                            "participant_id": participant_id,
                            "model_path": None,
                            "owc_path": owc_path,
                            "board_type": row["board_type"],
                            "num_players": num_players,
                            "actual_players": actual,
                            "rating": row["rating"],
                            "games_played": row["games_played"],
                            "ai_type": None,
                            "source": "owc_orphan_player_mismatch",
                            "reason": f"OWC model has {actual} players, registered as {num_players}",
                        })
                        continue

                recovered.append({
                    "participant_id": participant_id,
                    "model_path": None,
                    "owc_path": owc_path,
                    "board_type": row["board_type"],
                    "num_players": num_players,
                    "rating": row["rating"],
                    "games_played": row["games_played"],
                    "ai_type": None,
                })
                continue

        phantoms.append({
            "participant_id": participant_id,
            "model_path": None,
            "board_type": row["board_type"],
            "num_players": row["num_players"],
            "rating": row["rating"],
            "games_played": row["games_played"],
            "ai_type": None,
            "source": "orphaned_elo_rating",
        })

    conn.close()

    # Log summary
    if recovered:
        logger.info(f"Found {len(recovered)} models recoverable from OWC backup")
    if mismatches:
        logger.warning(f"Found {len(mismatches)} player count mismatches")

    return {
        "phantoms": phantoms,
        "recovered": recovered,
        "mismatches": mismatches,
    }


def delete_phantom_entries(db_path: Path, phantoms: list[dict]) -> dict:
    """Delete phantom entries from the database."""
    conn = sqlite3.connect(db_path)

    deleted = {
        "participants": 0,
        "elo_ratings": 0,
        "match_history": 0,
    }

    participant_ids = [p["participant_id"] for p in phantoms]

    if not participant_ids:
        conn.close()
        return deleted

    placeholders = ",".join(["?" for _ in participant_ids])

    # Delete from elo_ratings
    cursor = conn.execute(
        f"DELETE FROM elo_ratings WHERE participant_id IN ({placeholders})",
        participant_ids
    )
    deleted["elo_ratings"] = cursor.rowcount

    # Delete from match_history
    # The participant_ids are stored as JSON array in participant_ids column
    # We need to delete matches where any phantom participated
    for pid in participant_ids:
        # Match entries where this participant was involved
        # participant_ids is a JSON array like '["model_a", "model_b"]'
        cursor = conn.execute(
            "DELETE FROM match_history WHERE participant_ids LIKE ?",
            (f'%"{pid}"%',)
        )
        deleted["match_history"] += cursor.rowcount

    # Delete from participants
    cursor = conn.execute(
        f"DELETE FROM participants WHERE participant_id IN ({placeholders})",
        participant_ids
    )
    deleted["participants"] = cursor.rowcount

    conn.commit()
    conn.close()

    return deleted


def backup_database(db_path: Path) -> Path:
    """Create a backup of the database before modifications."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = db_path.parent / f"{db_path.stem}_backup_{timestamp}{db_path.suffix}"
    shutil.copy2(db_path, backup_path)
    logger.info(f"Created backup: {backup_path}")
    return backup_path


def main():
    parser = argparse.ArgumentParser(
        description="Clean up phantom Elo entries (models in DB but files don't exist)"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/unified_elo.db"),
        help="Path to Elo database"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Filter by config (e.g., 'square8_2p', 'hex8_4p')"
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually delete phantom entries (default: dry run)"
    )
    parser.add_argument(
        "--cluster-check",
        action="store_true",
        help="Also check cluster nodes for model files (slower)"
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Output phantom entries to JSON file"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not args.db.exists():
        logger.error(f"Database not found: {args.db}")
        sys.exit(1)

    logger.info(f"Scanning database: {args.db}")
    if args.config:
        logger.info(f"Filtering by config: {args.config}")

    # Find phantom entries
    phantoms = get_phantom_entries(
        args.db,
        config_filter=args.config,
        cluster_check=args.cluster_check
    )

    if not phantoms:
        logger.info("No phantom entries found!")
        return

    # Report findings
    logger.info(f"\nFound {len(phantoms)} phantom entries:")
    print("\n" + "=" * 80)
    print(f"{'Participant ID':<50} {'Rating':>8} {'Games':>8}")
    print("=" * 80)

    by_config = {}
    for p in phantoms:
        config_key = f"{p['board_type']}_{p['num_players']}p"
        if config_key not in by_config:
            by_config[config_key] = []
        by_config[config_key].append(p)

    for config_key in sorted(by_config.keys()):
        print(f"\n[{config_key}]")
        entries = sorted(by_config[config_key], key=lambda x: -(x["rating"] or 0))
        for p in entries[:20]:  # Show top 20 per config
            rating = p["rating"] or 0
            games = p["games_played"] or 0
            print(f"  {p['participant_id']:<48} {rating:>8.1f} {games:>8}")
        if len(entries) > 20:
            print(f"  ... and {len(entries) - 20} more")

    print("\n" + "=" * 80)
    print(f"Total phantom entries: {len(phantoms)}")

    # Output to JSON if requested
    if args.output_json:
        with open(args.output_json, "w") as f:
            json.dump(phantoms, f, indent=2)
        logger.info(f"Wrote phantom entries to: {args.output_json}")

    # Delete if --execute flag is set
    if args.execute:
        print("\n")
        confirm = input(f"Delete {len(phantoms)} phantom entries? (yes/no): ")
        if confirm.lower() != "yes":
            logger.info("Aborted")
            return

        # Create backup first
        backup_path = backup_database(args.db)

        # Delete entries
        deleted = delete_phantom_entries(args.db, phantoms)

        logger.info("Deletion complete:")
        logger.info(f"  - Participants deleted: {deleted['participants']}")
        logger.info(f"  - Elo ratings deleted: {deleted['elo_ratings']}")
        logger.info(f"  - Match history deleted: {deleted['match_history']}")
        logger.info(f"Backup saved to: {backup_path}")
    else:
        print("\nThis was a DRY RUN. Use --execute to actually delete entries.")


if __name__ == "__main__":
    main()
