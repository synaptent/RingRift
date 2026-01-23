#!/usr/bin/env python3
"""Bootstrap Training - One-command startup for the RingRift training infrastructure.

This script auto-detects the node type and starts the appropriate daemon profile
for 24/7 automated training. It provides a single entry point for all cluster nodes.

Usage:
    # Auto-detect and start (recommended)
    python scripts/bootstrap_training.py

    # Force a specific profile
    python scripts/bootstrap_training.py --profile training_node
    python scripts/bootstrap_training.py --profile coordinator
    python scripts/bootstrap_training.py --profile selfplay

    # Dry run (show what would be started)
    python scripts/bootstrap_training.py --dry-run

    # Start with P2P orchestrator
    python scripts/bootstrap_training.py --with-p2p

Profiles:
    coordinator   - Cluster coordinator (leader election, work queue, scheduling)
    training_node - GPU training node (selfplay, training, evaluation)
    selfplay      - Dedicated selfplay node (GPU selfplay only)
    ephemeral     - Ephemeral/Vast node (aggressive sync, temporary)
    cpu_only      - CPU-only node (heuristic selfplay, sync)

Auto-detection:
    - Checks GPU type (GH200, H100, A100, consumer)
    - Checks GPU memory
    - Checks hostname patterns
    - Falls back to training_node if GPU available, cpu_only otherwise

Created: December 2025
Purpose: Phase 4 ACTIVATION - enable 24/7 automated training
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import socket
import subprocess
import sys
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# =============================================================================
# Node Detection
# =============================================================================


def detect_gpu_info() -> dict:
    """Detect GPU type and capabilities.

    Returns:
        Dictionary with GPU info:
        - has_gpu: bool
        - gpu_name: str
        - gpu_memory_gb: float
        - gpu_count: int
        - gpu_tier: str (high, mid, low, none)
    """
    result = {
        "has_gpu": False,
        "gpu_name": "None",
        "gpu_memory_gb": 0.0,
        "gpu_count": 0,
        "gpu_tier": "none",
    }

    try:
        import torch

        if not torch.cuda.is_available():
            return result

        result["has_gpu"] = True
        result["gpu_count"] = torch.cuda.device_count()
        result["gpu_name"] = torch.cuda.get_device_name(0)

        # Get GPU memory
        props = torch.cuda.get_device_properties(0)
        result["gpu_memory_gb"] = props.total_memory / (1024 ** 3)

        # Determine tier
        gpu_name = result["gpu_name"].upper()
        if "GH200" in gpu_name or "H100" in gpu_name or "H200" in gpu_name:
            result["gpu_tier"] = "high"
        elif "A100" in gpu_name or "A40" in gpu_name:
            result["gpu_tier"] = "high"
        elif "4090" in gpu_name or "5090" in gpu_name:
            result["gpu_tier"] = "mid"
        elif "3090" in gpu_name or "4080" in gpu_name or "5080" in gpu_name:
            result["gpu_tier"] = "mid"
        elif "A10" in gpu_name or "3080" in gpu_name or "4070" in gpu_name:
            result["gpu_tier"] = "low"
        else:
            result["gpu_tier"] = "low"

    except ImportError:
        pass
    except Exception as e:
        logger.debug(f"GPU detection error: {e}")

    return result


def detect_node_type() -> str:
    """Detect the type of node based on hostname and environment.

    Returns:
        Node type: 'coordinator', 'lambda', 'vast', 'hetzner', 'unknown'
    """
    hostname = socket.gethostname().lower()

    # Check hostname patterns
    if "coordinator" in hostname or "master" in hostname or "leader" in hostname:
        return "coordinator"
    elif "lambda" in hostname or "gh200" in hostname or "h100" in hostname:
        return "lambda"
    elif "vast" in hostname or "workspace" in hostname:
        return "vast"
    elif "hetzner" in hostname or "cpu" in hostname:
        return "hetzner"

    # Check environment
    if os.environ.get("VAST_CONTAINERLABEL"):
        return "vast"

    # Check for Lambda-specific paths
    if Path("/home/ubuntu/ringrift").exists():
        return "lambda"

    return "unknown"


def select_profile(gpu_info: dict, node_type: str, force_profile: str = None) -> str:
    """Select the appropriate daemon profile.

    Args:
        gpu_info: GPU detection results
        node_type: Detected node type
        force_profile: Force a specific profile (overrides auto-detection)

    Returns:
        Profile name
    """
    if force_profile:
        return force_profile

    # Coordinator detection
    if node_type == "coordinator":
        return "coordinator"

    # Ephemeral/Vast nodes - need aggressive sync
    if node_type == "vast":
        return "ephemeral"

    # CPU-only nodes
    if not gpu_info["has_gpu"]:
        return "cpu_only"

    # High-tier GPU - training node
    if gpu_info["gpu_tier"] == "high":
        return "training_node"

    # Mid-tier GPU - selfplay focus
    if gpu_info["gpu_tier"] == "mid":
        return "selfplay"

    # Low-tier GPU - selfplay only
    return "selfplay"


# =============================================================================
# Startup
# =============================================================================


async def start_p2p_orchestrator(gpu_info: dict, node_type: str) -> subprocess.Popen | None:
    """Start the P2P orchestrator in the background.

    Args:
        gpu_info: GPU detection results
        node_type: Detected node type

    Returns:
        Popen object for the P2P process, or None if not started
    """
    script_path = Path(__file__).parent / "p2p_orchestrator.py"
    if not script_path.exists():
        logger.warning("P2P orchestrator script not found")
        return None

    hostname = socket.gethostname()
    node_id = f"{node_type}-{hostname}"

    cmd = [
        sys.executable,
        str(script_path),
        "--node-id", node_id,
        "--port", "8770",
    ]

    # High-tier nodes can be leaders
    if gpu_info["gpu_tier"] == "high" or node_type == "coordinator":
        cmd.append("--can-be-leader")

    logger.info(f"Starting P2P orchestrator: {' '.join(cmd)}")

    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "p2p_orchestrator.log"

    with open(log_file, "a") as f:
        proc = subprocess.Popen(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    logger.info(f"P2P orchestrator started (PID: {proc.pid}, log: {log_file})")
    return proc


async def start_daemon_manager(profile: str, verbose: bool = False) -> None:
    """Start the daemon manager with the specified profile.

    Args:
        profile: Daemon profile to use
        verbose: Enable verbose output
    """
    from app.coordination.daemon_manager import (
        DaemonManager,
        DaemonManagerConfig,
        get_daemon_manager,
    )
    from app.coordination.daemon_adapters import register_all_adapters_with_manager

    config = DaemonManagerConfig(
        health_check_interval=30.0,
        auto_restart_failed=True,
    )

    manager = get_daemon_manager(config)
    register_all_adapters_with_manager()

    # Get daemons for profile
    daemons = manager.get_profile_daemons(profile)

    if not daemons:
        logger.error(f"No daemons defined for profile: {profile}")
        return

    logger.info(f"Starting profile '{profile}' with {len(daemons)} daemons...")

    # Start all daemons
    results = await manager.start_all(daemons)

    success = sum(1 for v in results.values() if v)
    failed = len(results) - success

    if verbose:
        for daemon_type, started in results.items():
            status = "✓" if started else "✗"
            logger.info(f"  {status} {daemon_type.value}")

    logger.info(f"Started {success}/{len(results)} daemons ({failed} failed)")

    # Run in foreground
    import signal
    shutdown_event = asyncio.Event()

    def signal_handler():
        logger.info("Shutdown signal received")
        shutdown_event.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, signal_handler)

    logger.info("Running in foreground. Press Ctrl+C to stop.")
    await shutdown_event.wait()

    logger.info("Shutting down...")
    await manager.shutdown()


def print_summary(gpu_info: dict, node_type: str, profile: str) -> None:
    """Print bootstrap summary."""
    print("=" * 60)
    print("RINGRIFT TRAINING BOOTSTRAP")
    print("=" * 60)
    print(f"Hostname:    {socket.gethostname()}")
    print(f"Node Type:   {node_type}")
    print(f"Profile:     {profile}")
    print("-" * 60)
    print(f"GPU:         {gpu_info['gpu_name']}")
    print(f"GPU Count:   {gpu_info['gpu_count']}")
    print(f"GPU Memory:  {gpu_info['gpu_memory_gb']:.1f} GB")
    print(f"GPU Tier:    {gpu_info['gpu_tier']}")
    print("=" * 60)


async def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Bootstrap RingRift training infrastructure",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Profiles:
  coordinator   Cluster coordinator (leader, work queue, scheduling)
  training_node GPU training node (selfplay, training, evaluation)
  selfplay      Dedicated selfplay node
  ephemeral     Ephemeral/Vast node (aggressive sync)
  cpu_only      CPU-only node (heuristic selfplay)

Examples:
  python scripts/bootstrap_training.py
  python scripts/bootstrap_training.py --profile training_node
  python scripts/bootstrap_training.py --with-p2p
  python scripts/bootstrap_training.py --dry-run
        """,
    )

    parser.add_argument(
        "--profile",
        choices=["coordinator", "training_node", "selfplay", "ephemeral", "cpu_only"],
        help="Force a specific profile (default: auto-detect)",
    )
    parser.add_argument(
        "--with-p2p", action="store_true",
        help="Also start P2P orchestrator",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show what would be started without actually starting",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Detect node capabilities
    logger.info("Detecting node capabilities...")
    gpu_info = detect_gpu_info()
    node_type = detect_node_type()

    # Select profile
    profile = select_profile(gpu_info, node_type, args.profile)

    # Print summary
    print_summary(gpu_info, node_type, profile)

    if args.dry_run:
        print("\n[DRY RUN] Would start the following:")
        print(f"  Profile: {profile}")
        if args.with_p2p:
            print("  P2P orchestrator: yes")
        print("\nRun without --dry-run to start.")
        return 0

    # Start P2P orchestrator if requested
    p2p_proc = None
    if args.with_p2p:
        p2p_proc = await start_p2p_orchestrator(gpu_info, node_type)
        if p2p_proc:
            await asyncio.sleep(2)  # Let P2P initialize

    try:
        # Start daemon manager
        await start_daemon_manager(profile, args.verbose)
        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted")
        return 130
    except Exception as e:
        logger.error(f"Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1
    finally:
        # Clean up P2P if we started it
        if p2p_proc:
            logger.info("Stopping P2P orchestrator...")
            p2p_proc.terminate()
            try:
                p2p_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p2p_proc.kill()


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
