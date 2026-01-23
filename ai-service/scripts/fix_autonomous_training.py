#!/usr/bin/env python3
"""Fix autonomous training loop to maximize Elo improvement.

December 29, 2025 - Emergency fix script to ensure all configs train.

Issues addressed:
1. TrainingTriggerDaemon has no state - NPZ files not scanned
2. Selfplay distributed evenly instead of focusing on data-starved configs
3. Some configs have sufficient data but training not triggered

Usage:
    python scripts/fix_autonomous_training.py --scan      # Scan NPZ and show status
    python scripts/fix_autonomous_training.py --trigger   # Trigger training for ready configs
    python scripts/fix_autonomous_training.py --focus     # Focus selfplay on starved configs
    python scripts/fix_autonomous_training.py --all       # All of the above
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Training threshold (match TrainingTriggerDaemon)
MIN_SAMPLES_THRESHOLD = 10000

# All 12 canonical configs
ALL_CONFIGS = [
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
]


def get_best_npz_for_config(board_type: str, num_players: int) -> tuple[Path | None, int]:
    """Find the best (largest) NPZ file for a config and return sample count."""
    training_dir = Path("data/training")
    config_key = f"{board_type}_{num_players}p"

    best_path = None
    best_samples = 0

    # Check various naming patterns
    patterns = [
        f"{config_key}_combined.npz",
        f"{config_key}_consolidated.npz",
        f"{config_key}_canonical.npz",
        f"{config_key}.npz",
        f"{config_key}_fresh.npz",
        f"{config_key}_cluster.npz",
        f"canonical_{config_key}.npz",
    ]

    for pattern in patterns:
        npz_path = training_dir / pattern
        if npz_path.exists():
            try:
                data = np.load(npz_path, allow_pickle=True)
                if "features" in data:
                    samples = len(data["features"])
                elif "states" in data:
                    samples = len(data["states"])
                elif "values" in data:
                    samples = len(data["values"])
                else:
                    samples = 0

                if samples > best_samples:
                    best_samples = samples
                    best_path = npz_path
            except Exception as e:
                logger.warning(f"Error reading {npz_path}: {e}")

    return best_path, best_samples


def scan_all_configs() -> dict[str, dict]:
    """Scan all configs and return their training readiness status."""
    results = {}

    for board_type, num_players in ALL_CONFIGS:
        config_key = f"{board_type}_{num_players}p"
        npz_path, samples = get_best_npz_for_config(board_type, num_players)

        model_path = Path(f"models/canonical_{config_key}.pth")
        model_exists = model_path.exists()

        results[config_key] = {
            "board_type": board_type,
            "num_players": num_players,
            "npz_path": str(npz_path) if npz_path else None,
            "samples": samples,
            "threshold": MIN_SAMPLES_THRESHOLD,
            "can_train": samples >= MIN_SAMPLES_THRESHOLD,
            "model_exists": model_exists,
            "gap": max(0, MIN_SAMPLES_THRESHOLD - samples),
        }

    return results


def print_status(results: dict[str, dict]) -> None:
    """Print training status for all configs."""
    print("\n" + "=" * 80)
    print("AUTONOMOUS TRAINING STATUS - December 29, 2025")
    print("=" * 80)

    ready = []
    blocked = []

    for config_key, info in sorted(results.items()):
        status = "✅ READY" if info["can_train"] else "❌ BLOCKED"
        samples_str = f"{info['samples']:,}" if info['samples'] else "0"
        gap_str = f"(need {info['gap']:,} more)" if info['gap'] > 0 else ""

        print(f"{config_key:15} {status:12} {samples_str:>12} samples {gap_str}")

        if info["can_train"]:
            ready.append(config_key)
        else:
            blocked.append(config_key)

    print("=" * 80)
    print(f"Ready for training: {len(ready)}/12 configs")
    print(f"Blocked (need data): {len(blocked)}/12 configs")

    if blocked:
        print(f"\nBlocked configs: {', '.join(blocked)}")

    return ready, blocked


def trigger_training_for_config(config_key: str, info: dict) -> bool:
    """Trigger training for a specific config."""
    if not info["can_train"]:
        logger.warning(f"Cannot train {config_key}: insufficient samples")
        return False

    if not info["npz_path"]:
        logger.warning(f"Cannot train {config_key}: no NPZ file found")
        return False

    board_type = info["board_type"]
    num_players = info["num_players"]
    npz_path = info["npz_path"]

    # Build training command
    cmd = [
        sys.executable, "-m", "app.training.train",
        "--board-type", board_type,
        "--num-players", str(num_players),
        "--data-path", npz_path,
        "--epochs", "30",
        "--batch-size", "256" if "hex" in board_type else "512",
        "--save-path", f"models/{config_key}_autonomous.pth",
        "--allow-stale-data",  # Allow training on existing NPZ files
    ]

    # Add init weights if model exists
    model_path = Path(f"models/canonical_{config_key}.pth")
    if model_path.exists():
        cmd.extend(["--init-weights", str(model_path)])

    logger.info(f"Triggering training for {config_key}: {' '.join(cmd)}")

    # Run in background with nohup
    log_path = f"logs/train_{config_key}_autonomous.log"
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    logger.info(f"Training started for {config_key} (PID: {process.pid}, log: {log_path})")
    return True


def focus_selfplay_on_starved_configs(blocked_configs: list[str]) -> None:
    """Adjust selfplay scheduler to focus on data-starved configs.

    This updates the curriculum weights to prioritize blocked configs.
    """
    if not blocked_configs:
        logger.info("No blocked configs - selfplay distribution is optimal")
        return

    logger.info(f"Focusing selfplay on blocked configs: {blocked_configs}")

    # Create a curriculum weight adjustment
    # Higher weight = more selfplay jobs for that config
    weights = {}
    for board_type, num_players in ALL_CONFIGS:
        config_key = f"{board_type}_{num_players}p"
        if config_key in blocked_configs:
            weights[config_key] = 5.0  # 5x priority for blocked configs
        else:
            weights[config_key] = 1.0  # Normal priority

    # Write to a config file that SelfplayScheduler can read
    weights_path = Path("data/coordination/selfplay_priority_weights.json")
    weights_path.parent.mkdir(parents=True, exist_ok=True)

    import json
    with open(weights_path, "w") as f:
        json.dump({
            "weights": weights,
            "reason": "focus_on_data_starved",
            "updated": "2025-12-29",
            "blocked_configs": blocked_configs,
        }, f, indent=2)

    logger.info(f"Wrote priority weights to {weights_path}")

    # Also try to emit an event to notify the scheduler
    try:
        from app.coordination.event_router import publish_sync
        from app.distributed.data_events import DataEventType

        publish_sync(
            DataEventType.CURRICULUM_REBALANCED,
            payload={
                "weights": weights,
                "reason": "autonomous_fix_focus_starved",
                "blocked_configs": blocked_configs,
            },
            source="fix_autonomous_training",
        )
        logger.info("Emitted CURRICULUM_REBALANCED event")
    except Exception as e:
        logger.warning(f"Could not emit event: {e}")


def main():
    parser = argparse.ArgumentParser(description="Fix autonomous training loop")
    parser.add_argument("--scan", action="store_true", help="Scan NPZ files and show status")
    parser.add_argument("--trigger", action="store_true", help="Trigger training for ready configs")
    parser.add_argument("--focus", action="store_true", help="Focus selfplay on starved configs")
    parser.add_argument("--all", action="store_true", help="Do all of the above")
    parser.add_argument("--max-concurrent", type=int, default=3, help="Max concurrent training jobs")

    args = parser.parse_args()

    if not any([args.scan, args.trigger, args.focus, args.all]):
        args.scan = True  # Default to scan

    if args.all:
        args.scan = args.trigger = args.focus = True

    # Always scan first
    logger.info("Scanning NPZ files for all 12 configs...")
    results = scan_all_configs()

    if args.scan:
        ready, blocked = print_status(results)
    else:
        ready = [k for k, v in results.items() if v["can_train"]]
        blocked = [k for k, v in results.items() if not v["can_train"]]

    if args.trigger:
        print("\n" + "=" * 80)
        print("TRIGGERING TRAINING FOR READY CONFIGS")
        print("=" * 80)

        # Sort by samples (fewer first - they'll finish faster)
        ready_sorted = sorted(ready, key=lambda k: results[k]["samples"])

        triggered = 0
        for config_key in ready_sorted:
            if triggered >= args.max_concurrent:
                logger.info(f"Reached max concurrent limit ({args.max_concurrent}), skipping remaining")
                break

            info = results[config_key]
            if trigger_training_for_config(config_key, info):
                triggered += 1

        print(f"\nTriggered {triggered} training jobs")

    if args.focus:
        print("\n" + "=" * 80)
        print("FOCUSING SELFPLAY ON DATA-STARVED CONFIGS")
        print("=" * 80)
        focus_selfplay_on_starved_configs(blocked)

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Configs ready for training: {len(ready)}/12")
    print(f"Configs needing more data: {len(blocked)}/12")

    if blocked:
        print(f"\nTo generate more data for blocked configs, run:")
        for config_key in blocked:
            info = results[config_key]
            print(f"  # {config_key}: need {info['gap']:,} more samples")


if __name__ == "__main__":
    main()
