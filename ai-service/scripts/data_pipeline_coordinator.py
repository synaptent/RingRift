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
    data_files: List[str] = field(default_factory=list)
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
    training_history: List[Dict] = field(default_factory=list)
    node_statuses: Dict[str, Dict] = field(default_factory=dict)

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


def get_cluster_nodes() -> List[Tuple[str, str, int]]:
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
) -> Tuple[int, List[str]]:
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


def get_aligned_data_files(data_dir: Path) -> List[Path]:
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
    pretrained: Optional[Path] = None,
) -> Tuple[bool, Dict]:
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


def print_status(state: PipelineState, node_statuses: List[NodeStatus]):
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

    elif args.command == "daemon":
        run_daemon(args.interval, args.min_games)


if __name__ == "__main__":
    main()
