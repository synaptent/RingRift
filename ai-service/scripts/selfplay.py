#!/usr/bin/env python3
"""Unified Selfplay CLI - Single entry point for all selfplay variants.

This consolidates 30+ selfplay scripts into one unified CLI that leverages
the SelfplayRunner base class and SelfplayConfig for all selfplay needs.

Usage:
    # Quick heuristic selfplay (fast, for data bootstrapping)
    python scripts/selfplay.py --board square8 --num-players 2 --engine heuristic

    # GPU Gumbel MCTS selfplay (high quality training data)
    python scripts/selfplay.py --board hex8 --num-players 2 --engine gumbel --num-games 500

    # Full options
    python scripts/selfplay.py \
        --board square8 \
        --num-players 4 \
        --num-games 1000 \
        --engine nnue-guided \
        --output-dir data/games/selfplay_sq8_4p \
        --batch-size 256 \
        --use-gpu

Engine Modes:
    heuristic     Fast heuristic AI (good for bootstrap)
    gumbel        Gumbel MCTS (best quality, slower)
    mcts          Standard MCTS
    nnue-guided   NNUE evaluation with search
    policy-only   Direct policy network
    nn-descent    Neural descent search
    mixed         Mix of engines for diversity
    random        Random moves (baseline)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Setup path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.training.selfplay_config import (
    EngineMode,
    SelfplayConfig,
    parse_selfplay_args,
)
from app.training.selfplay_runner import (
    GameResult,
    GumbelMCTSSelfplayRunner,
    HeuristicSelfplayRunner,
    RunStats,
    SelfplayRunner,
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def get_runner_for_config(config: SelfplayConfig) -> SelfplayRunner:
    """Get the appropriate runner for the engine mode.

    Args:
        config: Selfplay configuration

    Returns:
        Configured SelfplayRunner subclass
    """
    mode = config.engine_mode

    # Map engine modes to runner classes
    if mode in (EngineMode.HEURISTIC, EngineMode.RANDOM):
        logger.info(f"Using HeuristicSelfplayRunner for {mode.value}")
        return HeuristicSelfplayRunner(config)

    elif mode in (EngineMode.GUMBEL_MCTS, EngineMode.MCTS):
        logger.info(f"Using GumbelMCTSSelfplayRunner for {mode.value}")
        return GumbelMCTSSelfplayRunner(config)

    elif mode in (EngineMode.NNUE_GUIDED, EngineMode.POLICY_ONLY,
                  EngineMode.NN_MINIMAX, EngineMode.NN_DESCENT):
        # These modes use neural network - try GPU runner
        try:
            from app.training.gpu_mcts_selfplay import GPUMCTSSelfplayRunner
            logger.info(f"Using GPUMCTSSelfplayRunner for {mode.value}")
            return GPUMCTSSelfplayRunner(config)
        except ImportError:
            logger.warning(f"GPU runner not available, falling back to Gumbel MCTS")
            return GumbelMCTSSelfplayRunner(config)

    elif mode in (EngineMode.MIXED, EngineMode.DIVERSE):
        # Mixed mode - alternate between engines
        logger.info(f"Using mixed engine strategy for {mode.value}")
        # Default to Gumbel MCTS for quality
        return GumbelMCTSSelfplayRunner(config)

    elif mode in (EngineMode.GMO, EngineMode.EBMO, EngineMode.IG_GMO, EngineMode.CAGE):
        # Experimental AI modes
        try:
            from app.ai.experimental import get_experimental_runner
            logger.info(f"Using experimental runner for {mode.value}")
            return get_experimental_runner(mode, config)
        except ImportError:
            logger.warning(f"Experimental AI {mode.value} not available, falling back to heuristic")
            return HeuristicSelfplayRunner(config)

    else:
        # Default fallback
        logger.warning(f"Unknown engine mode {mode}, using heuristic")
        return HeuristicSelfplayRunner(config)


def emit_selfplay_complete_event(config: SelfplayConfig, stats: RunStats) -> None:
    """Emit SELFPLAY_COMPLETE event for pipeline automation.

    Args:
        config: Selfplay configuration
        stats: Run statistics
    """
    try:
        from app.coordination.event_router import get_router
        from app.coordination.stage_events import StageEvent

        router = get_router()
        router.publish(
            event_type=StageEvent.SELFPLAY_COMPLETE,
            payload={
                "config_key": config.config_key,
                "board_type": config.board_type,
                "num_players": config.num_players,
                "games_completed": stats.games_completed,
                "total_samples": stats.total_samples,
                "database_path": config.record_db,
                "duration_seconds": stats.elapsed_seconds,
            },
            source="selfplay_cli",
        )
        logger.info(f"[Pipeline] Emitted SELFPLAY_COMPLETE event for {config.config_key}")
    except ImportError:
        logger.warning(
            "[Pipeline] Could not emit SELFPLAY_COMPLETE event - "
            "coordination module not available"
        )
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to emit SELFPLAY_COMPLETE event: {e}")


def main():
    """Main entry point for unified selfplay CLI."""
    # Parse args using unified config
    config = parse_selfplay_args(
        description="Unified Selfplay CLI - Generate training data via self-play"
    )

    # Log configuration summary
    logger.info("=" * 60)
    logger.info("Unified Selfplay")
    logger.info("=" * 60)
    logger.info(f"Board: {config.board_type}")
    logger.info(f"Players: {config.num_players}")
    logger.info(f"Games: {config.num_games}")
    logger.info(f"Engine: {config.engine_mode.value}")
    if config.output_dir:
        logger.info(f"Output: {config.output_dir}")
    if config.record_db:
        logger.info(f"Database: {config.record_db}")
    if config.emit_pipeline_events:
        logger.info("Pipeline events: ENABLED")
    logger.info("=" * 60)

    # Get appropriate runner
    runner = get_runner_for_config(config)

    # Run selfplay
    try:
        stats = runner.run()

        # Print final summary
        logger.info("")
        logger.info("=" * 60)
        logger.info("SELFPLAY COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Games completed: {stats.games_completed}")
        logger.info(f"Games failed: {stats.games_failed}")
        logger.info(f"Total moves: {stats.total_moves}")
        logger.info(f"Total samples: {stats.total_samples}")
        logger.info(f"Duration: {stats.elapsed_seconds:.1f}s")
        logger.info(f"Throughput: {stats.games_per_second:.2f} games/sec")

        if stats.wins_by_player:
            logger.info("Wins by player:")
            for player, wins in sorted(stats.wins_by_player.items()):
                pct = 100 * wins / max(1, stats.games_completed)
                logger.info(f"  Player {player}: {wins} ({pct:.1f}%)")

        logger.info("=" * 60)

        # Emit pipeline event if requested
        if config.emit_pipeline_events and stats.games_completed > 0:
            emit_selfplay_complete_event(config, stats)

        # Return success if games completed
        return 0 if stats.games_completed > 0 else 1

    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Selfplay failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
