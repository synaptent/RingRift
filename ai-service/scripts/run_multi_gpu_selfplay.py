#!/usr/bin/env python3
"""Multi-GPU parallel self-play coordinator.

Launches multiple selfplay workers across available GPUs for maximum throughput.
Each GPU runs an independent selfplay process, with results aggregated into
a shared database.

Uses unified SelfplayConfig for configuration (December 2025).

Usage:
    # Auto-detect GPUs and distribute work
    python scripts/run_multi_gpu_selfplay.py --num-games 10000 --board square8

    # Specify specific GPUs
    python scripts/run_multi_gpu_selfplay.py --gpus 0,1,2 --num-games 6000

    # Run on all available GPUs with custom batch size
    python scripts/run_multi_gpu_selfplay.py --all-gpus --batch-size 64

Output:
    - Per-GPU: data/selfplay/multi_gpu/{gpu_id}/games.jsonl
    - Merged: data/selfplay/multi_gpu/merged/stats.json
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Add app/ to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.training.selfplay_config import SelfplayConfig, create_argument_parser, EngineMode

import torch

# Unified logging setup
from scripts.lib.logging_config import setup_script_logging

logger = setup_script_logging("run_multi_gpu_selfplay")

AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]


def get_available_gpus() -> List[int]:
    """Get list of available CUDA GPUs."""
    if not torch.cuda.is_available():
        return []

    count = torch.cuda.device_count()
    gpus = []

    for i in range(count):
        try:
            # Check if GPU is usable
            torch.cuda.set_device(i)
            _ = torch.cuda.get_device_properties(i)
            gpus.append(i)
        except Exception:
            pass

    return gpus


def run_worker(
    gpu_id: int,
    num_games: int,
    board_type: str,
    num_players: int,
    batch_size: int,
    output_dir: Path,
    engine_mode: str,
    result_queue: mp.Queue,
) -> None:
    """Worker process that runs selfplay on a specific GPU."""
    # Set GPU device before imports to avoid memory allocation issues
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    try:
        # Now import GPU-dependent code
        from scripts.run_gpu_selfplay import run_gpu_selfplay

        worker_output = output_dir / f"gpu_{gpu_id}"
        worker_output.mkdir(parents=True, exist_ok=True)

        start_time = time.time()

        run_gpu_selfplay(
            num_games=num_games,
            board_type=board_type,
            num_players=num_players,
            batch_size=batch_size,
            output_dir=str(worker_output),
            engine_mode=engine_mode,
            save_npz=False,
        )

        elapsed = time.time() - start_time

        # Read stats from output
        stats_path = worker_output / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
        else:
            stats = {}

        result_queue.put({
            "gpu_id": gpu_id,
            "success": True,
            "games": num_games,
            "elapsed": elapsed,
            "games_per_sec": num_games / elapsed if elapsed > 0 else 0,
            "stats": stats,
        })

    except Exception as e:
        logger.exception(f"Worker GPU {gpu_id} failed")
        result_queue.put({
            "gpu_id": gpu_id,
            "success": False,
            "error": str(e),
        })


def merge_results(output_dir: Path, gpu_ids: List[int]) -> Dict[str, Any]:
    """Merge results from all GPU workers."""
    merged_stats = {
        "total_games": 0,
        "total_elapsed": 0,
        "per_gpu": {},
        "merged_at": datetime.utcnow().isoformat() + "Z",
    }

    for gpu_id in gpu_ids:
        stats_path = output_dir / f"gpu_{gpu_id}" / "stats.json"
        if stats_path.exists():
            with open(stats_path) as f:
                stats = json.load(f)
                merged_stats["total_games"] += stats.get("total_games", 0)
                merged_stats["per_gpu"][str(gpu_id)] = stats

    # Save merged stats
    merged_dir = output_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    with open(merged_dir / "stats.json", "w") as f:
        json.dump(merged_stats, f, indent=2)

    return merged_stats


def main():
    # Use unified argument parser from SelfplayConfig
    parser = create_argument_parser(
        description="Multi-GPU parallel self-play coordinator",
        include_ramdrive=False,  # Multi-GPU manages its own output
    )
    # Add script-specific arguments for multi-GPU coordination
    parser.add_argument(
        "--gpus",
        type=str,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2')",
    )
    parser.add_argument(
        "--all-gpus",
        action="store_true",
        help="Use all available GPUs",
    )
    parsed = parser.parse_args()

    # Create config from parsed args (uses canonical board type normalization)
    config = SelfplayConfig(
        board_type=parsed.board,
        num_players=parsed.num_players,
        num_games=parsed.num_games,
        engine_mode=parsed.engine_mode,
        batch_size=parsed.batch_size,
        output_dir=parsed.output_dir or str(AI_SERVICE_ROOT / "data" / "selfplay" / "multi_gpu"),
        source='run_multi_gpu_selfplay.py',
    )

    # Build args object for backwards compatibility with rest of script
    args = type('Args', (), {
        'num_games': config.num_games,
        'board': config.board_type,
        'num_players': config.num_players,
        'batch_size': config.batch_size,
        'output_dir': config.output_dir,
        'engine_mode': config.engine_mode.value if isinstance(config.engine_mode, EngineMode) else config.engine_mode,
        'gpus': parsed.gpus,
        'all_gpus': parsed.all_gpus,
    })()

    # Determine GPUs to use
    available_gpus = get_available_gpus()

    if not available_gpus:
        logger.warning("No CUDA GPUs available. Running on CPU with single worker.")
        gpu_ids = [0]  # Will use CPU
    elif args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
        invalid = [g for g in gpu_ids if g not in available_gpus]
        if invalid:
            logger.error(f"Invalid GPU IDs: {invalid}. Available: {available_gpus}")
            return 1
    elif args.all_gpus:
        gpu_ids = available_gpus
    else:
        # Default to first GPU
        gpu_ids = [available_gpus[0]]

    # Distribute games across GPUs
    num_gpus = len(gpu_ids)
    games_per_gpu = args.num_games // num_gpus
    remainder = args.num_games % num_gpus

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("MULTI-GPU SELF-PLAY COORDINATOR")
    logger.info("=" * 60)
    logger.info(f"Total games: {args.num_games}")
    logger.info(f"GPUs: {gpu_ids}")
    logger.info(f"Games per GPU: ~{games_per_gpu}")
    logger.info(f"Board: {args.board} ({args.num_players}p)")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 60)

    # Launch workers
    result_queue = mp.Queue()
    processes = []

    start_time = time.time()

    for i, gpu_id in enumerate(gpu_ids):
        # Add remainder games to first workers
        worker_games = games_per_gpu + (1 if i < remainder else 0)

        p = mp.Process(
            target=run_worker,
            args=(
                gpu_id,
                worker_games,
                args.board,
                args.num_players,
                args.batch_size,
                output_dir,
                args.engine_mode,
                result_queue,
            ),
        )
        p.start()
        processes.append(p)
        logger.info(f"Launched worker on GPU {gpu_id} for {worker_games} games")

    # Wait for all workers
    results = []
    for _ in processes:
        result = result_queue.get(timeout=3600)  # 1 hour timeout
        results.append(result)
        if result.get("success"):
            logger.info(
                f"GPU {result['gpu_id']} completed: {result['games']} games "
                f"in {result['elapsed']:.1f}s ({result['games_per_sec']:.1f} games/sec)"
            )
        else:
            logger.error(f"GPU {result['gpu_id']} failed: {result.get('error')}")

    # Wait for processes to exit
    for p in processes:
        p.join(timeout=60)

    total_elapsed = time.time() - start_time

    # Merge results
    merged = merge_results(output_dir, gpu_ids)

    # Print summary
    logger.info("=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total games: {merged['total_games']}")
    logger.info(f"Total time: {total_elapsed:.1f}s")
    logger.info(f"Throughput: {merged['total_games'] / total_elapsed:.1f} games/sec")

    successful = sum(1 for r in results if r.get("success"))
    logger.info(f"Workers: {successful}/{len(gpu_ids)} succeeded")

    return 0 if successful == len(gpu_ids) else 1


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    sys.exit(main())
