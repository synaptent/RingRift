#!/usr/bin/env python3
"""Continuous policy-bearing selfplay worker for trainer supplemental data."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
AI_SERVICE_ROOT = SCRIPT_DIR.parent
if str(AI_SERVICE_ROOT) not in sys.path:
    sys.path.insert(0, str(AI_SERVICE_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("policy_selfplay_worker")


def _timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def run_batch(args: argparse.Namespace) -> bool:
    from scripts.generate_gumbel_selfplay import GumbelSelfplayConfig, run_selfplay
    from scripts.ingest_policy_selfplay import ingest_policy_selfplay_files

    raw_dir = Path(args.raw_output_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    supplemental_dir = Path(args.supplemental_output_dir)
    supplemental_dir.mkdir(parents=True, exist_ok=True)
    state_dir = Path(args.state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    batch_name = f"{args.config_key}_{_timestamp_slug()}"
    jsonl_path = raw_dir / f"{batch_name}.jsonl"

    config = GumbelSelfplayConfig(
        board_type=args.board_type,
        num_players=args.num_players,
        num_games=args.batch_games,
        simulation_budget=args.simulation_budget,
        output_path=str(jsonl_path),
        nn_model_id=args.model,
        opponent_type=args.opponent_type,
        use_gpu_tree=not args.disable_gpu_tree,
        allow_fresh_weights=args.allow_fresh_weights,
    )

    logger.info(
        "Starting selfplay batch config=%s games=%s budget=%s model=%s output=%s",
        args.config_key,
        args.batch_games,
        args.simulation_budget,
        args.model,
        jsonl_path,
    )
    results = run_selfplay(config)
    if not results:
        logger.warning("No selfplay games completed for %s", batch_name)
        return False

    summary = ingest_policy_selfplay_files(
        input_paths=[jsonl_path],
        output_dir=supplemental_dir,
        state_dir=state_dir,
        board_type=args.board_type,
        num_players=args.num_players,
        policy_entropy_threshold=args.policy_entropy_threshold,
        completion_rate_threshold=args.completion_rate_threshold,
        min_value_std=args.min_value_std,
        remote_host=args.remote_host,
        remote_dir=args.remote_dir,
        remote_user=args.remote_user,
        remote_key=args.remote_key,
        remote_port=args.remote_port,
    )
    logger.info(
        "Completed selfplay batch config=%s games_kept=%s shard=%s",
        args.config_key,
        summary.games_kept,
        summary.output_npz,
    )
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Continuous policy-bearing Gumbel selfplay worker")
    parser.add_argument("--config-key", required=True)
    parser.add_argument("--board-type", required=True, choices=["square8", "square19", "hex8", "hexagonal"])
    parser.add_argument("--num-players", required=True, type=int, choices=[2, 3, 4])
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-games", type=int, default=32)
    parser.add_argument("--simulation-budget", type=int, default=800)
    parser.add_argument("--raw-output-dir", required=True)
    parser.add_argument("--supplemental-output-dir", required=True)
    parser.add_argument("--state-dir", required=True)
    parser.add_argument("--sleep-seconds", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--disable-gpu-tree", action="store_true")
    parser.add_argument("--allow-fresh-weights", action="store_true")
    parser.add_argument("--opponent-type", default="selfplay")
    parser.add_argument("--policy-entropy-threshold", type=float, default=0.5)
    parser.add_argument("--completion-rate-threshold", type=float, default=0.95)
    parser.add_argument("--min-value-std", type=float, default=1e-6)
    parser.add_argument("--remote-host", default="")
    parser.add_argument("--remote-dir", default="")
    parser.add_argument("--remote-user", default="ubuntu")
    parser.add_argument("--remote-key", default="")
    parser.add_argument("--remote-port", type=int, default=22)
    args = parser.parse_args()

    while True:
        try:
            run_batch(args)
        except Exception as exc:
            logger.error("Selfplay worker batch failed: %s", exc, exc_info=True)
        if args.once:
            break
        time.sleep(max(args.sleep_seconds, 1))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
