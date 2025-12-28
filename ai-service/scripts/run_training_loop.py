#!/usr/bin/env python3
"""One-Command Training Loop - Automated selfplay → training pipeline.

This script provides a single entry point that:
1. Initializes the pipeline orchestrator with auto-trigger enabled
2. Runs selfplay to generate training data
3. Automatically triggers: sync → export → combine → train → evaluate → promote

Usage:
    # Basic usage (uses defaults)
    python scripts/run_training_loop.py --board-type hex8 --num-players 2

    # Full options
    python scripts/run_training_loop.py \
        --board-type hex8 --num-players 2 \
        --selfplay-games 1000 \
        --training-epochs 50 \
        --engine gumbel \
        --auto-promote

    # Skip selfplay, just trigger pipeline on existing data
    python scripts/run_training_loop.py \
        --board-type hex8 --num-players 2 \
        --skip-selfplay

    # Use specific training data (skip best-data selection)
    python scripts/run_training_loop.py \
        --board-type hex8 --num-players 2 \
        --data-path data/training/hex8_2p_custom.npz

Pipeline Flow:
    Selfplay → SELFPLAY_COMPLETE event
                    ↓
    Orchestrator auto-triggers:
    1. Data sync (if cluster nodes configured)
    2. NPZ export from databases
    3. NPZ combination (quality-weighted combining of historical + fresh data)
    4. Training with early stopping (uses combined NPZ or best single file)
    5. Evaluation (Elo rating)
    6. Promotion (automatic if model wins gauntlet, disable with --no-auto-promote)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Any

# Setup path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("training_loop")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="One-command training loop with automatic pipeline triggering",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required config
    parser.add_argument(
        "--board-type", "-b",
        type=str,
        required=True,
        choices=["square8", "square19", "hex8", "hexagonal"],
        help="Board type for training",
    )
    parser.add_argument(
        "--num-players", "-p",
        type=int,
        required=True,
        choices=[2, 3, 4],
        help="Number of players",
    )

    # Selfplay options
    selfplay_group = parser.add_argument_group("Selfplay Options")
    selfplay_group.add_argument(
        "--selfplay-games", "-n",
        type=int,
        default=1000,
        help="Number of selfplay games to generate (default: 1000)",
    )
    selfplay_group.add_argument(
        "--engine", "-e",
        type=str,
        default="gumbel-mcts",
        choices=["heuristic", "gumbel-mcts", "mcts", "nnue-guided", "mixed"],
        help="Selfplay engine mode (default: gumbel-mcts)",
    )
    selfplay_group.add_argument(
        "--skip-selfplay",
        action="store_true",
        help="Skip selfplay generation, trigger pipeline on existing data",
    )

    # Training options
    training_group = parser.add_argument_group("Training Options")
    training_group.add_argument(
        "--training-epochs",
        type=int,
        default=50,
        help="Max training epochs (default: 50, uses early stopping)",
    )
    training_group.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="Training batch size (default: 512)",
    )
    training_group.add_argument(
        "--model-version",
        type=str,
        default="v2",
        help="Model version/architecture (default: v2)",
    )
    training_group.add_argument(
        "--best-data",
        action="store_true",
        default=True,
        help="Use best available training data: combined NPZ if available, else largest fresh (default: True)",
    )
    training_group.add_argument(
        "--no-best-data",
        action="store_false",
        dest="best_data",
        help="Disable best-data selection, use explicit data path instead",
    )
    training_group.add_argument(
        "--data-path",
        type=str,
        default=None,
        help="Explicit path to training NPZ file (disables --best-data)",
    )

    # Pipeline options
    pipeline_group = parser.add_argument_group("Pipeline Options")
    pipeline_group.add_argument(
        "--auto-promote",
        action="store_true",
        default=True,
        help="Automatically promote model if it wins gauntlet evaluation (default: True)",
    )
    pipeline_group.add_argument(
        "--no-auto-promote",
        action="store_false",
        dest="auto_promote",
        help="Disable automatic model promotion after evaluation",
    )
    pipeline_group.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Skip Elo evaluation after training",
    )
    pipeline_group.add_argument(
        "--sync-from-cluster",
        action="store_true",
        help="Sync game data from cluster nodes before training",
    )
    pipeline_group.add_argument(
        "--pipeline-timeout",
        type=int,
        default=7200,
        help="Pipeline timeout in seconds (default: 7200 = 2 hours)",
    )

    # Misc
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be done without executing",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose logging",
    )

    return parser.parse_args()


def bootstrap_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    """Initialize the pipeline orchestrator with auto-trigger enabled.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary of initialized coordinators
    """
    from app.coordination.coordination_bootstrap import bootstrap_coordination

    logger.info("Bootstrapping coordination infrastructure with auto-trigger enabled...")

    # Determine effective data path (explicit or best-data selection)
    # If explicit data path is provided, disable best_data
    effective_best_data = args.best_data and args.data_path is None

    result = bootstrap_coordination(
        pipeline_auto_trigger=True,
        enable_selfplay=True,
        enable_training=True,
        enable_pipeline=True,
        enable_sync=args.sync_from_cluster,
        # Pass training config from CLI args (December 2025)
        training_epochs=args.training_epochs,
        training_batch_size=args.batch_size,
        training_model_version=args.model_version,
        # Pass data selection config (December 2025)
        training_use_best_data=effective_best_data,
        training_data_path=args.data_path,
    )

    active = [k for k, v in result.items() if v]
    logger.info(f"Initialized {len(active)} coordinators: {', '.join(active[:5])}...")
    logger.info(f"Training config: epochs={args.training_epochs}, batch_size={args.batch_size}, model_version={args.model_version}")

    # Log data selection mode
    if args.data_path:
        logger.info(f"Data selection: explicit path = {args.data_path}")
    elif effective_best_data:
        logger.info("Data selection: best available (combined NPZ if available, else largest fresh)")
    else:
        logger.info("Data selection: disabled (will use default pipeline behavior)")

    return result


def run_selfplay(args: argparse.Namespace) -> bool:
    """Run selfplay game generation.

    Args:
        args: Parsed command-line arguments

    Returns:
        True if selfplay completed successfully
    """
    from app.training.selfplay_config import SelfplayConfig, EngineMode

    # Create config
    config = SelfplayConfig(
        board_type=args.board_type,
        num_players=args.num_players,
        num_games=args.selfplay_games,
        engine_mode=EngineMode(args.engine),
        emit_pipeline_events=True,  # Enable pipeline event emission
    )

    logger.info(f"Starting selfplay: {config.config_key}")
    logger.info(f"  Games: {config.num_games}")
    logger.info(f"  Engine: {config.engine_mode.value}")
    logger.info(f"  Output: {config.record_db}")

    if args.dry_run:
        logger.info("[DRY RUN] Would run selfplay here")
        return True

    # Import and run selfplay
    from scripts.selfplay import get_runner_for_config, emit_selfplay_complete_event

    runner = get_runner_for_config(config)
    stats = runner.run()

    if stats.games_completed > 0:
        logger.info(f"Selfplay completed: {stats.games_completed} games, {stats.total_samples} samples")
        # Emit the event (runner should have done this, but ensure it)
        emit_selfplay_complete_event(config, stats)
        return True
    else:
        logger.error("Selfplay failed - no games completed")
        return False


async def wait_for_pipeline_completion(
    args: argparse.Namespace,
    timeout_seconds: int,
) -> bool:
    """Wait for the pipeline to complete all stages.

    Args:
        args: Parsed command-line arguments
        timeout_seconds: Maximum time to wait

    Returns:
        True if pipeline completed successfully
    """
    from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator

    orchestrator = get_pipeline_orchestrator()
    config_key = f"{args.board_type}_{args.num_players}p"

    start_time = time.time()
    last_stage = None
    initial_iterations = None

    logger.info(f"Waiting for pipeline completion (timeout: {timeout_seconds}s)...")

    while time.time() - start_time < timeout_seconds:
        status = orchestrator.get_status()

        if status is None:
            logger.warning(f"No pipeline status available, waiting...")
            await asyncio.sleep(10)
            continue

        current_stage = status.get("current_stage", "unknown")
        iterations_completed = status.get("iterations_completed", 0)
        models_trained = status.get("total_models_trained", 0)

        # Track initial iteration count to detect completion
        if initial_iterations is None:
            initial_iterations = iterations_completed

        # Log stage transitions
        if current_stage != last_stage:
            logger.info(f"Pipeline stage: {current_stage}")
            last_stage = current_stage

        # Check for completion: stage is IDLE and a new iteration completed
        if current_stage in ("idle", "complete"):
            if iterations_completed > initial_iterations:
                if args.auto_promote and not args.skip_evaluation:
                    # Check for evaluation/promotion completion
                    if status.get("promotions", 0) > 0 or models_trained > 0:
                        logger.info("Pipeline completed: training + evaluation done")
                        return True
                    # Wait a bit more for evaluation
                    logger.info(f"Training complete, waiting for evaluation... (models trained: {models_trained})")
                else:
                    logger.info("Pipeline completed: training done")
                    return True

        # Check for failures via circuit breaker status
        cb_status = status.get("circuit_breaker", {})
        if cb_status.get("state") == "open":
            logger.error("Pipeline circuit breaker is OPEN - too many failures")
            return False

        await asyncio.sleep(5)

    logger.error(f"Pipeline timed out after {timeout_seconds}s")
    return False


def trigger_manual_pipeline(args: argparse.Namespace) -> bool:
    """Manually trigger pipeline stages when --skip-selfplay is used.

    Args:
        args: Parsed command-line arguments

    Returns:
        True if triggered successfully
    """
    from app.coordination.event_router import get_router, StageEvent

    config_key = f"{args.board_type}_{args.num_players}p"

    logger.info(f"Triggering pipeline for existing data: {config_key}")

    if args.dry_run:
        logger.info("[DRY RUN] Would trigger pipeline here")
        return True

    # Emit a synthetic SELFPLAY_COMPLETE event to trigger the pipeline
    router = get_router()
    router.publish(
        event_type=StageEvent.SELFPLAY_COMPLETE,
        payload={
            "config_key": config_key,
            "board_type": args.board_type,
            "num_players": args.num_players,
            "games_completed": 0,  # Existing data
            "total_samples": 0,
            "source": "manual_trigger",
        },
    )

    logger.info("Pipeline triggered")
    return True


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_key = f"{args.board_type}_{args.num_players}p"

    # Print header
    logger.info("=" * 70)
    logger.info("RingRift Training Loop")
    logger.info("=" * 70)
    logger.info(f"Config: {config_key}")
    logger.info(f"Mode: {'Selfplay + Train' if not args.skip_selfplay else 'Train Existing Data'}")
    if args.auto_promote:
        logger.info("Auto-promote: ENABLED")
    logger.info("=" * 70)

    # Handle Ctrl+C gracefully
    shutdown_requested = False

    def signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            logger.info("Force exit")
            sys.exit(1)
        shutdown_requested = True
        logger.info("Shutdown requested, finishing current operation...")

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        # Step 1: Bootstrap coordination
        bootstrap_pipeline(args)

        # Step 1b: Start daemons (sync, health check, etc.) if not dry-run
        if not args.dry_run:
            try:
                from app.coordination.daemon_manager import get_daemon_manager, DaemonType
                manager = get_daemon_manager()
                # Start essential daemons for pipeline operation
                asyncio.run(manager.start(DaemonType.EVENT_ROUTER))
                asyncio.run(manager.start(DaemonType.DATA_PIPELINE))
                if args.sync_from_cluster:
                    asyncio.run(manager.start(DaemonType.AUTO_SYNC))  # Dec 2025: Using unified sync
                logger.info("Essential daemons started")
            except Exception as e:
                logger.warning(f"Could not start daemons: {e}")

        # Step 2: Run selfplay or trigger manual pipeline
        if args.skip_selfplay:
            if not trigger_manual_pipeline(args):
                return 1
        else:
            if not run_selfplay(args):
                return 1

        if shutdown_requested:
            logger.info("Shutdown requested, exiting before pipeline wait")
            return 130

        # Step 3: Wait for pipeline completion
        if not args.dry_run:
            success = asyncio.run(
                wait_for_pipeline_completion(args, args.pipeline_timeout)
            )

            if not success:
                logger.error("Pipeline did not complete successfully")
                return 1

        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("TRAINING LOOP COMPLETE")
        logger.info("=" * 70)
        logger.info(f"Config: {config_key}")

        if not args.dry_run:
            # Try to get final model path from orchestrator status
            try:
                from app.coordination.data_pipeline_orchestrator import get_pipeline_orchestrator
                orchestrator = get_pipeline_orchestrator()
                status = orchestrator.get_status()
                # Log relevant completion info
                if status.get("total_models_trained", 0) > 0:
                    logger.info(f"Models trained: {status['total_models_trained']}")
                if status.get("promotions", 0) > 0:
                    logger.info(f"Promotions: {status['promotions']}")
            except (ImportError, AttributeError, RuntimeError):
                pass

        logger.info("=" * 70)

        return 0

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.exception(f"Training loop failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
