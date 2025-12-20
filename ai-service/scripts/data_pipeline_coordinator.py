#!/usr/bin/env python3
"""Data Pipeline Coordinator - Automated data collection and training orchestration.

This coordinator manages the complete data pipeline:
1. Monitors Gumbel data generation across cluster nodes
2. Collects completed data files to local storage
3. Triggers KL training when data threshold is met
4. Tracks model versions and training history

Usage:
    # Run coordinator daemon (continuous monitoring)
    python scripts/data_pipeline_coordinator.py daemon --interval 300

    # One-time status check
    python scripts/data_pipeline_coordinator.py status

    # Collect data from all nodes
    python scripts/data_pipeline_coordinator.py collect

    # Trigger training manually
    python scripts/data_pipeline_coordinator.py train --min-games 500

    # Full pipeline: collect + train if threshold met
    python scripts/data_pipeline_coordinator.py run --min-games 1000
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATA_DIR = PROJECT_ROOT / "data" / "gumbel_selfplay"
MODELS_DIR = PROJECT_ROOT / "models" / "nnue"
PIPELINE_STATE_FILE = PROJECT_ROOT / "data" / "pipeline_state.json"

SSH_KEY = os.path.expanduser("~/.ssh/id_cluster")
REMOTE_DATA_PATH = "/workspace/ringrift/ai-service/data/gumbel_selfplay"
REMOTE_LOG_PATH = "/workspace/ringrift/ai-service/logs/gumbel_gen_a40.log"

# Canonical data file and promotion settings
CANONICAL_DATA_FILE = DATA_DIR / "sq8_gumbel_kl_canonical.jsonl"
PRODUCTION_MODEL = MODELS_DIR / "nnue_policy_square8_2p.pt"
AB_TEST_GAMES = 20
AB_TEST_THINK_TIME = 200
MIN_WIN_RATE_FOR_PROMOTION = 0.55  # Must beat production by at least 55%

# Training thresholds (games)
TRAINING_THRESHOLDS = [1000, 1500, 2000, 3000, 5000]


@dataclass
class NodeStatus:
    """Status of a cluster node."""
    instance_id: str
    host: str
    port: int
    gpu_name: str = ""
    is_reachable: bool = False
    is_generating: bool = False
    games_completed: int = 0
    games_target: int = 0
    last_check: str = ""
    data_files: list[str] = field(default_factory=list)
    error: str = ""


@dataclass
class PipelineState:
    """Persistent state for the pipeline coordinator."""
    total_games_collected: int = 0
    total_games_generated: int = 0
    last_collection_time: str = ""
    last_training_time: str = ""
    last_training_games: int = 0
    current_model_version: str = ""
    training_history: list[dict] = field(default_factory=list)
    node_statuses: dict[str, dict] = field(default_factory=dict)

    def save(self, path: Path = PIPELINE_STATE_FILE):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: Path = PIPELINE_STATE_FILE) -> "PipelineState":
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                return cls(**data)
        return cls()


def get_cluster_nodes() -> list[tuple[str, str, int]]:
    """Get list of cluster nodes from vastai."""
    try:
        result = subprocess.run(
            ["vastai", "show", "instances"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            logger.error(f"Failed to get instances: {result.stderr}")
            return []

        nodes = []
        for line in result.stdout.strip().split("\n")[1:]:  # Skip header
            parts = line.split()
            if len(parts) >= 10 and parts[2] == "running":
                instance_id = parts[0]
                gpu_model = parts[4]
                ssh_host = parts[9]
                ssh_port = int(parts[10])
                nodes.append((instance_id, gpu_model, ssh_host, ssh_port))

        return nodes
    except Exception as e:
        logger.error(f"Error getting cluster nodes: {e}")
        return []


def check_node_status(instance_id: str, host: str, port: int) -> NodeStatus:
    """Check status of a single node."""
    status = NodeStatus(
        instance_id=instance_id,
        host=host,
        port=port,
        last_check=datetime.now().isoformat(),
    )

    try:
        # Check SSH connectivity and get GPU info
        ssh_cmd = [
            "ssh", "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(port),
            f"root@{host}",
            "nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1"
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            status.is_reachable = True
            status.gpu_name = result.stdout.strip().split("\n")[-1]  # Last line (skip welcome)
        else:
            status.error = "SSH connection failed"
            return status

        # Check if generation is running (more robust check)
        ssh_cmd[-1] = "pgrep -f 'generate_gumbel_selfplay' 2>/dev/null | head -1"
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            output = result.stdout.strip().split("\n")[-1]
            status.is_generating = output.isdigit() and int(output) > 0

        # Get data files and game counts
        ssh_cmd[-1] = f"ls -1 {REMOTE_DATA_PATH}/sq8_gumbel_aligned_*.jsonl 2>/dev/null"
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            files = [f for f in result.stdout.strip().split("\n") if f and f.startswith("/")]
            status.data_files = files

        # Count games in files
        if status.data_files:
            ssh_cmd[-1] = f"wc -l {REMOTE_DATA_PATH}/sq8_gumbel_aligned_*.jsonl 2>/dev/null | tail -1"
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
            if result.returncode == 0:
                try:
                    line = result.stdout.strip().split("\n")[-1]
                    status.games_completed = int(line.split()[0])
                except (ValueError, IndexError):
                    pass

    except subprocess.TimeoutExpired:
        status.error = "SSH timeout"
    except Exception as e:
        status.error = str(e)

    return status


def collect_data_from_node(
    instance_id: str,
    host: str,
    port: int,
    local_dir: Path,
) -> tuple[int, list[str]]:
    """Collect data files from a node."""
    collected_files = []
    total_games = 0

    try:
        # Get list of files
        ssh_cmd = [
            "ssh", "-i", SSH_KEY,
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(port),
            f"root@{host}",
            f"ls -1 {REMOTE_DATA_PATH}/sq8_gumbel_aligned_*.jsonl 2>/dev/null"
        ]
        result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return 0, []

        remote_files = [f for f in result.stdout.strip().split("\n") if f and f.startswith("/")]

        for remote_file in remote_files:
            filename = os.path.basename(remote_file)
            # Add node identifier to filename to avoid collisions
            local_filename = f"{instance_id}_{filename}"
            local_path = local_dir / local_filename

            # Skip if already collected (check by size)
            if local_path.exists():
                continue

            # Download file
            scp_cmd = [
                "scp", "-i", SSH_KEY,
                "-o", "StrictHostKeyChecking=no",
                "-P", str(port),
                f"root@{host}:{remote_file}",
                str(local_path),
            ]
            result = subprocess.run(scp_cmd, capture_output=True, timeout=120)
            if result.returncode == 0:
                collected_files.append(local_filename)
                # Count games in file
                with open(local_path) as f:
                    games = sum(1 for _ in f)
                    total_games += games
                logger.info(f"Collected {local_filename} ({games} games)")

    except Exception as e:
        logger.error(f"Error collecting from {instance_id}: {e}")

    return total_games, collected_files


def is_aligned_data_file(filepath: Path) -> bool:
    """Check if a data file has properly aligned move indices.

    Aligned files have scattered move indices (42, 57, 72, 129, etc.)
    Broken files have sequential indices (0, 1, 2, 3, etc.)
    """
    try:
        with open(filepath) as f:
            first_line = f.readline()
            if not first_line.strip():
                return False
            data = json.loads(first_line)
            if 'moves' not in data or not data['moves']:
                return False
            policy = data['moves'][0].get('mcts_policy', {})
            if not policy:
                return False
            keys = sorted([int(k) for k in policy.keys()])
            if len(keys) < 2:
                return False
            # Check if keys are sequential (broken) or scattered (aligned)
            gaps = [keys[i+1] - keys[i] for i in range(len(keys)-1)]
            return not all(g == 1 for g in gaps)
    except Exception:
        return False


def get_aligned_data_files(data_dir: Path) -> list[Path]:
    """Get only data files with properly aligned move indices."""
    aligned = []
    for jsonl_file in data_dir.glob("*.jsonl"):
        if is_aligned_data_file(jsonl_file):
            aligned.append(jsonl_file)
    return aligned


def count_local_games(data_dir: Path, aligned_only: bool = False) -> int:
    """Count total games in local data directory."""
    total = 0
    if aligned_only:
        files = get_aligned_data_files(data_dir)
    else:
        files = list(data_dir.glob("*.jsonl"))

    for jsonl_file in files:
        with open(jsonl_file) as f:
            total += sum(1 for _ in f)
    return total


def trigger_training(
    data_dir: Path,
    output_dir: Path,
    min_games: int = 500,
    epochs: int = 30,
    pretrained: Path | None = None,
) -> tuple[bool, dict]:
    """Trigger KL training on collected data."""
    # Find all properly aligned data files (validates actual content)
    data_files = get_aligned_data_files(data_dir)
    if not data_files:
        return False, {"error": "No aligned data files found"}

    # Count total games
    total_games = 0
    for f in data_files:
        with open(f) as fp:
            total_games += sum(1 for _ in fp)

    if total_games < min_games:
        return False, {"error": f"Insufficient data: {total_games} < {min_games}"}

    logger.info(f"Starting KL training with {total_games} games from {len(data_files)} files")

    # Build training command
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"nnue_policy_kl_{timestamp}"

    cmd = [
        "python", str(PROJECT_ROOT / "scripts" / "train_nnue_policy.py"),
        "--jsonl", *[str(f) for f in data_files],
        "--board-type", "square8",
        "--num-players", "2",
        "--epochs", str(epochs),
        "--batch-size", "64",
        "--use-kl-loss",
        "--save-path", str(output_dir / f"{model_name}.pt"),
    ]

    if pretrained and pretrained.exists():
        cmd.extend(["--pretrained", str(pretrained)])

    # Run training
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour max
            cwd=PROJECT_ROOT,
        )

        training_result = {
            "model_name": model_name,
            "timestamp": timestamp,
            "total_games": total_games,
            "data_files": len(data_files),
            "epochs": epochs,
            "success": result.returncode == 0,
        }

        if result.returncode == 0:
            logger.info(f"Training completed: {model_name}")
            # Extract final metrics from output
            for line in result.stdout.split("\n"):
                if "Best validation loss:" in line:
                    training_result["val_loss"] = float(line.split(":")[-1].strip())
                if "Final policy accuracy:" in line:
                    training_result["accuracy"] = float(line.split(":")[-1].strip())
        else:
            logger.error(f"Training failed: {result.stderr[-500:]}")
            training_result["error"] = result.stderr[-500:]

        return result.returncode == 0, training_result

    except subprocess.TimeoutExpired:
        return False, {"error": "Training timeout"}
    except Exception as e:
        return False, {"error": str(e)}


def run_ab_test(
    model_a: Path,
    model_b: Path,
    num_games: int = AB_TEST_GAMES,
    think_time: int = AB_TEST_THINK_TIME,
) -> tuple[int, int, int, float]:
    """Run A/B test between two models.

    Returns: (a_wins, b_wins, draws, win_rate_a)
    """
    logger.info(f"Running A/B test: {model_a.name} vs {model_b.name} ({num_games} games)")

    cmd = [
        "python", str(PROJECT_ROOT / "scripts" / "ab_test_policy_models.py"),
        "--model-a", str(model_a),
        "--model-b", str(model_b),
        "--num-games", str(num_games),
        "--think-time", str(think_time),
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=PROJECT_ROOT,
        )

        if result.returncode != 0:
            logger.error(f"A/B test failed: {result.stderr[-300:]}")
            return 0, 0, 0, 0.0

        # Parse results from output
        a_wins, b_wins, draws = 0, 0, 0
        for line in result.stdout.split("\n"):
            if "Model A wins:" in line:
                a_wins = int(line.split(":")[1].split("(")[0].strip())
            elif "Model B wins:" in line:
                b_wins = int(line.split(":")[1].split("(")[0].strip())
            elif "Draws:" in line:
                draws = int(line.split(":")[1].split("(")[0].strip())

        total = a_wins + b_wins + draws
        win_rate = a_wins / total if total > 0 else 0.0

        logger.info(f"A/B test results: A={a_wins}, B={b_wins}, D={draws}, WR={win_rate:.1%}")
        return a_wins, b_wins, draws, win_rate

    except subprocess.TimeoutExpired:
        logger.error("A/B test timeout")
        return 0, 0, 0, 0.0
    except Exception as e:
        logger.error(f"A/B test error: {e}")
        return 0, 0, 0, 0.0


def promote_model(
    new_model: Path,
    production_model: Path = PRODUCTION_MODEL,
) -> bool:
    """Promote a new model to production with backup."""
    try:
        # Backup current production
        if production_model.exists():
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = production_model.parent / f"{production_model.stem}_backup_{timestamp}.pt"
            import shutil
            shutil.copy2(production_model, backup_path)
            logger.info(f"Backed up production model to {backup_path.name}")

        # Copy new model to production
        import shutil
        shutil.copy2(new_model, production_model)
        logger.info(f"Promoted {new_model.name} to production")
        return True

    except Exception as e:
        logger.error(f"Failed to promote model: {e}")
        return False


def train_and_evaluate(
    data_dir: Path,
    output_dir: Path,
    min_games: int,
    run_ab_test_flag: bool = True,
) -> tuple[bool, dict]:
    """Train a new model and optionally A/B test against production.

    Returns: (promoted, result_dict)
    """
    # Train new model
    success, result = trigger_training(data_dir, output_dir, min_games=min_games, epochs=100)

    if not success:
        return False, result

    new_model = output_dir / f"{result['model_name']}.pt"

    if not run_ab_test_flag:
        logger.info("Skipping A/B test (disabled)")
        return False, result

    if not PRODUCTION_MODEL.exists():
        logger.info("No production model exists, promoting directly")
        if promote_model(new_model):
            result["promoted"] = True
            return True, result
        return False, result

    # Run A/B test
    a_wins, b_wins, draws, win_rate = run_ab_test(new_model, PRODUCTION_MODEL)
    result["ab_test"] = {
        "new_wins": a_wins,
        "production_wins": b_wins,
        "draws": draws,
        "win_rate": win_rate,
    }

    # Promote if better than threshold
    if win_rate >= MIN_WIN_RATE_FOR_PROMOTION:
        logger.info(f"New model wins {win_rate:.1%} >= {MIN_WIN_RATE_FOR_PROMOTION:.1%}, promoting!")
        if promote_model(new_model):
            result["promoted"] = True
            return True, result
    else:
        logger.info(f"New model wins {win_rate:.1%} < {MIN_WIN_RATE_FOR_PROMOTION:.1%}, not promoting")
        result["promoted"] = False

    return False, result


def get_next_training_threshold(current_games: int) -> int | None:
    """Get the next training threshold based on current game count."""
    for threshold in TRAINING_THRESHOLDS:
        if current_games >= threshold:
            continue
        return threshold
    return None


def print_status(state: PipelineState, node_statuses: list[NodeStatus]):
    """Print current pipeline status."""
    print("\n" + "=" * 70)
    print("DATA PIPELINE STATUS")
    print("=" * 70)

    # Count aligned games
    aligned_games = count_local_games(DATA_DIR, aligned_only=True)
    total_games = count_local_games(DATA_DIR, aligned_only=False)

    print(f"\nAligned games (usable): {aligned_games}")
    print(f"Total games collected: {total_games}")
    print(f"Last collection: {state.last_collection_time or 'Never'}")
    print(f"Last training: {state.last_training_time or 'Never'}")
    print(f"Current model: {state.current_model_version or 'None'}")

    print(f"\n{'Node':<12} {'GPU':<20} {'Status':<12} {'Games':<10} {'Generating'}")
    print("-" * 70)

    for status in node_statuses:
        if status.is_reachable:
            gen_status = "Yes" if status.is_generating else "No"
            print(f"{status.instance_id:<12} {status.gpu_name[:20]:<20} {'Online':<12} {status.games_completed:<10} {gen_status}")
        else:
            print(f"{status.instance_id:<12} {'-':<20} {'Offline':<12} {'-':<10} {'-'}")

    total_cluster_games = sum(s.games_completed for s in node_statuses if s.is_reachable)
    print("-" * 70)
    print(f"{'Total':<12} {'':<20} {'':<12} {total_cluster_games:<10}")
    print("=" * 70)


def run_daemon(interval: int, min_games_for_training: int):
    """Run coordinator as daemon with periodic checks."""
    logger.info(f"Starting pipeline daemon (interval={interval}s, min_games={min_games_for_training})")

    while True:
        try:
            state = PipelineState.load()

            # Get cluster nodes
            nodes = get_cluster_nodes()
            if not nodes:
                logger.warning("No cluster nodes found")
                time.sleep(interval)
                continue

            # Check status of each node
            node_statuses = []
            for instance_id, gpu_model, host, port in nodes:
                status = check_node_status(instance_id, host, port)
                status.gpu_name = gpu_model
                node_statuses.append(status)
                state.node_statuses[instance_id] = asdict(status)

            # Collect data from nodes with completed games
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            total_new_games = 0

            for status in node_statuses:
                if status.is_reachable and status.games_completed > 0:
                    games, files = collect_data_from_node(
                        status.instance_id,
                        status.host,
                        status.port,
                        DATA_DIR,
                    )
                    total_new_games += games

            if total_new_games > 0:
                state.total_games_collected = count_local_games(DATA_DIR)
                state.last_collection_time = datetime.now().isoformat()
                logger.info(f"Collected {total_new_games} new games, total: {state.total_games_collected}")

            # Check if we should trigger training
            games_since_training = state.total_games_collected - state.last_training_games
            if games_since_training >= min_games_for_training:
                logger.info(f"Training threshold reached: {games_since_training} games since last training")

                pretrained = MODELS_DIR / "nnue_policy_square8_2p.pt"
                success, result = trigger_training(
                    DATA_DIR,
                    MODELS_DIR,
                    min_games=min_games_for_training,
                    pretrained=pretrained if pretrained.exists() else None,
                )

                if success:
                    state.last_training_time = datetime.now().isoformat()
                    state.last_training_games = state.total_games_collected
                    state.current_model_version = result.get("model_name", "")
                    state.training_history.append(result)
                    logger.info(f"Training completed: {result}")

            state.save()
            print_status(state, node_statuses)

        except KeyboardInterrupt:
            logger.info("Daemon stopped by user")
            break
        except Exception as e:
            logger.error(f"Daemon error: {e}")

        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(description="Data Pipeline Coordinator")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Status command
    subparsers.add_parser("status", help="Show current pipeline status")

    # Collect command
    subparsers.add_parser("collect", help="Collect data from all nodes")

    # Train command
    train_parser = subparsers.add_parser("train", help="Trigger training")
    train_parser.add_argument("--min-games", type=int, default=500)
    train_parser.add_argument("--epochs", type=int, default=30)

    # Run command (collect + train)
    run_parser = subparsers.add_parser("run", help="Collect and train if threshold met")
    run_parser.add_argument("--min-games", type=int, default=1000)

    # Daemon command
    daemon_parser = subparsers.add_parser("daemon", help="Run as daemon")
    daemon_parser.add_argument("--interval", type=int, default=300, help="Check interval in seconds")
    daemon_parser.add_argument("--min-games", type=int, default=500, help="Min games to trigger training")

    # Auto command (collect + train + A/B test + promote)
    auto_parser = subparsers.add_parser("auto", help="Full automated pipeline with A/B gate")
    auto_parser.add_argument("--min-games", type=int, default=1000, help="Min games for training")
    auto_parser.add_argument("--skip-ab", action="store_true", help="Skip A/B testing")
    auto_parser.add_argument("--skip-collect", action="store_true", help="Skip collection")

    args = parser.parse_args()
    state = PipelineState.load()

    if args.command == "status":
        nodes = get_cluster_nodes()
        node_statuses = []
        for instance_id, gpu_model, host, port in nodes:
            status = check_node_status(instance_id, host, port)
            status.gpu_name = gpu_model
            node_statuses.append(status)
        print_status(state, node_statuses)

    elif args.command == "collect":
        nodes = get_cluster_nodes()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        total = 0

        for instance_id, gpu_model, host, port in nodes:
            status = check_node_status(instance_id, host, port)
            if status.is_reachable and status.games_completed > 0:
                games, _ = collect_data_from_node(instance_id, host, port, DATA_DIR)
                total += games

        state.total_games_collected = count_local_games(DATA_DIR)
        state.last_collection_time = datetime.now().isoformat()
        state.save()
        logger.info(f"Collection complete: {total} new games, {state.total_games_collected} total")

    elif args.command == "train":
        pretrained = MODELS_DIR / "nnue_policy_square8_2p.pt"
        success, result = trigger_training(
            DATA_DIR,
            MODELS_DIR,
            min_games=args.min_games,
            epochs=args.epochs,
            pretrained=pretrained if pretrained.exists() else None,
        )
        if success:
            state.last_training_time = datetime.now().isoformat()
            state.last_training_games = state.total_games_collected
            state.current_model_version = result.get("model_name", "")
            state.training_history.append(result)
            state.save()
        print(json.dumps(result, indent=2))

    elif args.command == "run":
        # Collect
        nodes = get_cluster_nodes()
        DATA_DIR.mkdir(parents=True, exist_ok=True)

        for instance_id, gpu_model, host, port in nodes:
            status = check_node_status(instance_id, host, port)
            if status.is_reachable and status.games_completed > 0:
                collect_data_from_node(instance_id, host, port, DATA_DIR)

        state.total_games_collected = count_local_games(DATA_DIR)
        state.last_collection_time = datetime.now().isoformat()

        # Train if threshold met
        if state.total_games_collected >= args.min_games:
            pretrained = MODELS_DIR / "nnue_policy_square8_2p.pt"
            success, result = trigger_training(
                DATA_DIR,
                MODELS_DIR,
                min_games=args.min_games,
                pretrained=pretrained if pretrained.exists() else None,
            )
            if success:
                state.last_training_time = datetime.now().isoformat()
                state.last_training_games = state.total_games_collected
                state.current_model_version = result.get("model_name", "")
                state.training_history.append(result)

        state.save()

    elif args.command == "auto":
        logger.info("=" * 60)
        logger.info("AUTOMATED PIPELINE: Collect → Train → A/B Test → Promote")
        logger.info("=" * 60)

        # Step 1: Collect (optional)
        if not args.skip_collect:
            logger.info("\n[Step 1/4] Collecting data from cluster...")
            nodes = get_cluster_nodes()
            DATA_DIR.mkdir(parents=True, exist_ok=True)

            for instance_id, gpu_model, host, port in nodes:
                status = check_node_status(instance_id, host, port)
                if status.is_reachable and status.games_completed > 0:
                    collect_data_from_node(instance_id, host, port, DATA_DIR)

            state.total_games_collected = count_local_games(DATA_DIR, aligned_only=True)
            state.last_collection_time = datetime.now().isoformat()
            logger.info(f"Collection complete: {state.total_games_collected} aligned games")
        else:
            state.total_games_collected = count_local_games(DATA_DIR, aligned_only=True)
            logger.info(f"[Step 1/4] Skipped collection. Local games: {state.total_games_collected}")

        # Check threshold
        if state.total_games_collected < args.min_games:
            logger.info(f"Insufficient data: {state.total_games_collected} < {args.min_games}")
            state.save()
            sys.exit(0)

        # Step 2-4: Train + A/B Test + Promote
        logger.info(f"\n[Step 2-4] Training with {state.total_games_collected} games...")
        promoted, result = train_and_evaluate(
            DATA_DIR,
            MODELS_DIR,
            min_games=args.min_games,
            run_ab_test_flag=not args.skip_ab,
        )

        state.last_training_time = datetime.now().isoformat()
        state.last_training_games = state.total_games_collected
        state.current_model_version = result.get("model_name", "")
        state.training_history.append(result)
        state.save()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {result.get('model_name', 'N/A')}")
        logger.info(f"Val Loss: {result.get('val_loss', 'N/A')}")
        logger.info(f"Accuracy: {result.get('accuracy', 'N/A')}")
        if "ab_test" in result:
            ab = result["ab_test"]
            logger.info(f"A/B Test: {ab['new_wins']}-{ab['production_wins']} ({ab['win_rate']:.1%})")
        logger.info(f"Promoted: {result.get('promoted', False)}")
        logger.info("=" * 60)

    elif args.command == "daemon":
        run_daemon(args.interval, args.min_games)


if __name__ == "__main__":
    main()
