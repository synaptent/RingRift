#!/usr/bin/env python
"""Distributed NNUE training for RingRift.

This script provides distributed NNUE training capabilities:
- Run training on a remote high-memory host (e.g., Mac Studio)
- Collect training data from databases on multiple hosts
- Track memory usage during training
- Support training on heavy boards (square19, hexagonal)
- Mode-based host selection (local, lan, aws, hybrid)

Deployment Modes:
    - local:  Train locally only (no remote hosts)
    - lan:    Use local network Mac cluster hosts
    - aws:    Use AWS cloud hosts only (limited to square8 due to 16GB memory)
    - hybrid: Use both LAN and AWS hosts (maximum parallelism)

Usage:
    # Train on local machine with local data
    python scripts/run_distributed_nnue_training.py --db data/games/*.db --epochs 50

    # Train on remote host (Mac Studio) with local data
    python scripts/run_distributed_nnue_training.py --db data/games/*.db \\
        --remote-host mac-studio --epochs 50

    # Train using LAN cluster hosts (auto-selects best host)
    python scripts/run_distributed_nnue_training.py --mode lan \\
        --db data/games/*.db --epochs 50

    # Train on AWS staging (square8 only due to 16GB memory)
    python scripts/run_distributed_nnue_training.py --mode aws \\
        --board square8 --db data/games/*.db --epochs 50

    # Collect data from multiple hosts and train remotely
    python scripts/run_distributed_nnue_training.py \\
        --collect-from mac-studio,mbp-64gb \\
        --remote-host mac-studio \\
        --board square19 --epochs 100

    # Track memory during training
    python scripts/run_distributed_nnue_training.py --db data/games/*.db \\
        --track-memory --epochs 50

Requirements:
    - config/distributed_hosts.yaml for remote operations
    - SSH key-based authentication for remote hosts
    - PyTorch installed on remote hosts

AWS Configuration (in distributed_hosts.yaml):
    aws-staging:
      ssh_host: "3.236.54.231"
      ssh_user: "ubuntu"
      ssh_key: "~/.ssh/ringrift-staging-key.pem"
      ringrift_path: "/home/ubuntu/ringrift"
      memory_gb: 16
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.distributed import (
    HostConfig,
    SSHExecutor,
    MemoryTracker,
    RemoteMemoryMonitor,
    load_remote_hosts,
    detect_host_memory,
    get_eligible_hosts_for_board,
    get_high_memory_hosts,
    get_ssh_executor,
    format_memory_profile,
    write_memory_report,
    BOARD_MEMORY_REQUIREMENTS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Deployment modes for host selection
VALID_MODES = ["local", "lan", "aws", "hybrid"]


def get_hosts_for_mode(mode: str, hosts_config: Dict[str, HostConfig]) -> List[str]:
    """Get list of host names based on deployment mode.

    Args:
        mode: Deployment mode (local, lan, aws, hybrid)
        hosts_config: Dict of host name -> HostConfig

    Returns:
        List of eligible host names for the given mode.
    """
    if mode == "local":
        return []

    # Categorize hosts as LAN or AWS based on configuration
    lan_hosts: List[str] = []
    aws_hosts: List[str] = []

    for name, host in hosts_config.items():
        # AWS hosts typically have ssh_user set (e.g., "ubuntu") or /home/ in work_dir
        is_aws = (
            host.ssh_user == "ubuntu"
            or (host.work_directory and "/home/" in host.work_directory)
            or "aws" in name.lower()
        )
        if is_aws:
            aws_hosts.append(name)
        else:
            lan_hosts.append(name)

    if mode == "lan":
        return lan_hosts
    elif mode == "aws":
        return aws_hosts
    elif mode == "hybrid":
        return lan_hosts + aws_hosts

    return []


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Distributed NNUE training for RingRift",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data sources
    parser.add_argument(
        "--db",
        type=str,
        nargs="*",
        default=[],
        help="Local database paths (supports glob patterns)",
    )
    parser.add_argument(
        "--collect-from",
        type=str,
        help="Comma-separated hosts to collect databases from",
    )
    parser.add_argument(
        "--remote-db-pattern",
        type=str,
        default="data/games/*.db",
        help="Glob pattern for databases on remote hosts",
    )

    # Remote execution
    parser.add_argument(
        "--mode",
        type=str,
        default="lan",
        choices=VALID_MODES,
        help="Deployment mode: local (no remote), lan (Mac cluster), aws (cloud), hybrid (both)",
    )
    parser.add_argument(
        "--remote-host",
        type=str,
        help="Run training on this remote host (e.g., mac-studio). Overrides --mode",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Force local training even if remote host specified",
    )

    # Board configuration
    parser.add_argument(
        "--board",
        type=str,
        default="square8",
        choices=["square8", "square19", "hexagonal"],
        help="Board type (default: square8)",
    )
    parser.add_argument(
        "--num-players",
        type=int,
        default=2,
        help="Number of players (default: 2)",
    )

    # Training parameters
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of training epochs (default: 50)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size (default: 256)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
        help="Learning rate (default: 1e-3)",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=256,
        help="NNUE hidden dimension (default: 256)",
    )
    parser.add_argument(
        "--sample-every-n",
        type=int,
        default=1,
        help="Sample every N positions (default: 1)",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum training samples (default: all)",
    )

    # Memory tracking
    parser.add_argument(
        "--track-memory",
        action="store_true",
        help="Track memory usage during training",
    )
    parser.add_argument(
        "--memory-sample-interval",
        type=float,
        default=5.0,
        help="Memory sampling interval in seconds (default: 5.0)",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for reports (default: auto)",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Model ID for checkpoint",
    )

    # Other
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run in demo mode with synthetic data",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be executed without running",
    )

    return parser.parse_args(argv)


def collect_remote_databases(
    hosts: List[str],
    db_pattern: str,
    local_dir: str,
) -> List[str]:
    """Collect databases from remote hosts.

    Args:
        hosts: List of host names to collect from.
        db_pattern: Glob pattern for databases on remote hosts.
        local_dir: Local directory to store collected files.

    Returns:
        List of local paths to collected databases.
    """
    collected = []
    hosts_config = load_remote_hosts()

    os.makedirs(local_dir, exist_ok=True)

    for host_name in hosts:
        if host_name not in hosts_config:
            logger.warning(f"Unknown host: {host_name}")
            continue

        host = hosts_config[host_name]
        executor = SSHExecutor(host)

        # Find databases on remote host
        logger.info(f"Finding databases on {host_name}...")
        find_cmd = f"ls -1 {db_pattern} 2>/dev/null"
        result = executor.run(find_cmd, timeout=30)

        if result.returncode != 0 or not result.stdout.strip():
            logger.warning(f"No databases found on {host_name}")
            continue

        remote_paths = result.stdout.strip().split('\n')
        logger.info(f"Found {len(remote_paths)} databases on {host_name}")

        # Copy each database
        for remote_path in remote_paths:
            db_name = os.path.basename(remote_path)
            local_path = os.path.join(local_dir, f"{host_name}_{db_name}")

            logger.info(f"Copying {remote_path} from {host_name}...")

            scp_cmd = ["scp", "-o", "ConnectTimeout=10"]
            if host.ssh_key:
                scp_cmd.extend(["-i", host.ssh_key_path])
            # Build remote target with optional ssh_user (e.g., ubuntu@host for AWS)
            ssh_target = f"{host.ssh_user}@{host.ssh_host}" if host.ssh_user else host.ssh_host
            scp_cmd.extend([
                f"{ssh_target}:{remote_path}",
                local_path,
            ])

            try:
                subprocess.run(scp_cmd, check=True, timeout=300)
                collected.append(local_path)
                logger.info(f"Collected: {local_path}")
            except Exception as e:
                logger.error(f"Failed to copy {remote_path}: {e}")

    return collected


def run_remote_training(
    host_name: str,
    db_paths: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run NNUE training on a remote host.

    Args:
        host_name: Remote host name.
        db_paths: List of database paths (local to remote host).
        args: Command line arguments.

    Returns:
        Training report dict.
    """
    hosts_config = load_remote_hosts()
    if host_name not in hosts_config:
        raise ValueError(f"Unknown host: {host_name}")

    host = hosts_config[host_name]
    executor = SSHExecutor(host)

    # Check host memory
    memory_info = detect_host_memory(host_name)
    required_memory = BOARD_MEMORY_REQUIREMENTS.get(args.board, 8)

    if memory_info.total_gb < required_memory:
        raise RuntimeError(
            f"Host {host_name} has {memory_info.total_gb}GB but "
            f"{args.board} requires {required_memory}GB"
        )

    logger.info(f"Running training on {host_name} ({memory_info})")

    # Build training command
    cmd_parts = [
        "PYTHONPATH=.",
        "python", "scripts/train_nnue.py",
        "--board-type", args.board,
        "--num-players", str(args.num_players),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--hidden-dim", str(args.hidden_dim),
        "--sample-every-n", str(args.sample_every_n),
        "--seed", str(args.seed),
    ]

    if db_paths:
        cmd_parts.extend(["--db"] + db_paths)
    elif args.demo:
        cmd_parts.append("--demo")
    else:
        # Use default pattern on remote
        cmd_parts.extend(["--db", f"data/games/*{args.board}*.db"])

    if args.max_samples:
        cmd_parts.extend(["--max-samples", str(args.max_samples)])

    if args.model_id:
        cmd_parts.extend(["--model-id", args.model_id])

    cmd = " ".join(cmd_parts)
    logger.info(f"Remote command: {cmd}")

    # Start memory monitoring if requested
    memory_monitor = None
    if args.track_memory:
        memory_monitor = RemoteMemoryMonitor(
            ssh_executor=executor,
            process_pattern="train_nnue",
            sample_interval=args.memory_sample_interval,
        )
        memory_monitor.start()

    # Run training
    try:
        result = executor.run(cmd, timeout=3600 * 4)  # 4 hour timeout

        if result.returncode != 0:
            logger.error(f"Training failed on {host_name}: {result.stderr}")
            return {"error": result.stderr}

        # Parse output for report path
        output_lines = result.stdout.strip().split('\n')
        report = {"success": True, "output": result.stdout}

        # Try to find and fetch the report JSON
        for line in output_lines:
            if "nnue_training_report.json" in line:
                # Extract path and fetch report
                parts = line.split()
                for part in parts:
                    if "nnue_training_report.json" in part:
                        # Fetch report from remote
                        cat_result = executor.run(f"cat {part}", timeout=30)
                        if cat_result.returncode == 0:
                            try:
                                report = json.loads(cat_result.stdout)
                            except json.JSONDecodeError:
                                pass
                        break

        return report

    finally:
        if memory_monitor:
            memory_monitor.stop()
            peak_rss = memory_monitor.get_peak_rss()
            logger.info(f"Peak memory on {host_name}: {peak_rss} MB")


def run_local_training(
    db_paths: List[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    """Run NNUE training locally.

    Args:
        db_paths: List of database paths.
        args: Command line arguments.

    Returns:
        Training report dict.
    """
    # Import here to avoid circular imports and allow remote-only usage
    from scripts.train_nnue import main as train_main

    # Build argument list
    train_args = [
        "--board-type", args.board,
        "--num-players", str(args.num_players),
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--hidden-dim", str(args.hidden_dim),
        "--sample-every-n", str(args.sample_every_n),
        "--seed", str(args.seed),
    ]

    if db_paths:
        train_args.extend(["--db"] + db_paths)
    elif args.demo:
        train_args.append("--demo")

    if args.max_samples:
        train_args.extend(["--max-samples", str(args.max_samples)])

    if args.model_id:
        train_args.extend(["--model-id", args.model_id])

    if args.output_dir:
        train_args.extend(["--run-dir", args.output_dir])

    # Track memory if requested
    memory_tracker = None
    if args.track_memory:
        memory_tracker = MemoryTracker(
            operation_name="nnue_training",
            sample_interval=args.memory_sample_interval,
            use_tracemalloc=True,
        )
        memory_tracker.start()

    try:
        result = train_main(train_args)
        report = {"success": result == 0}
    finally:
        if memory_tracker:
            profile = memory_tracker.stop()
            report["memory_profile"] = profile.to_dict()
            logger.info(format_memory_profile(profile))

    return report


def select_training_host(board_type: str, mode: str = "lan") -> Optional[str]:
    """Select the best host for training based on memory requirements and mode.

    Args:
        board_type: Board type being trained.
        mode: Deployment mode (local, lan, aws, hybrid).

    Returns:
        Host name, or None to train locally.
    """
    if mode == "local":
        return None

    hosts_config = load_remote_hosts()
    if not hosts_config:
        return None

    # Filter hosts by mode
    mode_hosts = get_hosts_for_mode(mode, hosts_config)
    if not mode_hosts:
        logger.warning(f"No hosts available for mode '{mode}'")
        return None

    # Check memory requirements for AWS (16GB) vs board type
    required_memory = BOARD_MEMORY_REQUIREMENTS.get(board_type, 8)
    if mode == "aws" and board_type != "square8":
        logger.warning(
            f"AWS mode only supports square8 (16GB RAM). "
            f"Requested {board_type} requires {required_memory}GB. "
            f"Training locally instead."
        )
        return None

    # Find hosts with enough memory
    eligible = get_eligible_hosts_for_board(board_type, mode_hosts)

    if not eligible:
        logger.warning(f"No {mode} hosts have enough memory for {board_type}")
        return None

    # Prefer high-memory hosts
    high_mem = get_high_memory_hosts(eligible)
    if high_mem:
        return high_mem[0]

    return eligible[0]


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point."""
    args = parse_args(argv)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir or f"runs/nnue_distributed_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Collect local databases
    db_paths: List[str] = []
    for pattern in args.db:
        expanded = glob.glob(pattern)
        if expanded:
            db_paths.extend(expanded)
        elif os.path.exists(pattern):
            db_paths.append(pattern)

    # Collect remote databases if requested
    if args.collect_from:
        collect_hosts = [h.strip() for h in args.collect_from.split(',')]
        collect_dir = os.path.join(output_dir, "collected_dbs")

        logger.info(f"Collecting databases from: {collect_hosts}")
        collected = collect_remote_databases(
            hosts=collect_hosts,
            db_pattern=args.remote_db_pattern,
            local_dir=collect_dir,
        )
        db_paths.extend(collected)
        logger.info(f"Total databases: {len(db_paths)}")

    if not db_paths and not args.demo:
        logger.error("No databases found. Use --db, --collect-from, or --demo")
        return 1

    # Determine where to run training
    if args.local:
        training_host = None
    elif args.remote_host:
        training_host = args.remote_host
    else:
        # Auto-select based on board type and mode
        training_host = select_training_host(args.board, args.mode)

    if args.dry_run:
        logger.info(f"Dry run - would train on: {training_host or 'local'}")
        logger.info(f"Databases: {db_paths}")
        return 0

    # Run training
    logger.info(f"Starting NNUE training for {args.board}")
    logger.info(f"Training host: {training_host or 'local'}")
    logger.info(f"Databases: {len(db_paths)}")

    if training_host:
        report = run_remote_training(
            host_name=training_host,
            db_paths=db_paths,
            args=args,
        )
    else:
        report = run_local_training(
            db_paths=db_paths,
            args=args,
        )

    # Save report
    report["training_host"] = training_host or "local"
    report["board_type"] = args.board
    report["db_count"] = len(db_paths)
    report["timestamp"] = timestamp

    report_path = os.path.join(output_dir, "distributed_training_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Saved report to {report_path}")

    if report.get("error"):
        logger.error(f"Training failed: {report['error']}")
        return 1

    logger.info("Distributed NNUE training complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
