#!/usr/bin/env python3
"""Unified Selfplay CLI - Single entry point for all selfplay variants.

This consolidates 30+ selfplay scripts into one unified CLI that leverages
the SelfplayRunner base class and SelfplayConfig for all selfplay needs.

Usage:
    # Quick heuristic selfplay (fast, for data bootstrapping)
    python scripts/selfplay.py --board square8 --num-players 2 --engine-mode heuristic-only

    # GPU Gumbel MCTS selfplay (high quality training data)
    python scripts/selfplay.py --board hex8 --num-players 2 --engine-mode gumbel-mcts --num-games 500

    # Full options
    python scripts/selfplay.py \
        --board square8 \
        --num-players 4 \
        --num-games 1000 \
        --engine-mode nnue-guided \
        --output-dir data/games/selfplay_sq8_4p \
        --batch-size 256 \
        --use-gpu

Engine Modes:
    heuristic-only  Fast heuristic AI (good for bootstrap)
    gumbel-mcts     Gumbel MCTS (best quality, slower)
    mcts            Standard MCTS
    nnue-guided     NNUE evaluation with search
    policy-only     Direct policy network
    nn-descent      Neural descent search
    mixed           Mix of engines for diversity
    random          Random moves (baseline)
    gnn             Pure GNN policy network (requires PyTorch Geometric)
    hybrid          CNN-GNN hybrid model (requires PyTorch Geometric)
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

# Disable torch.compile on GH200 nodes (driver incompatibility)
# Set this before any torch imports to prevent compile() errors
if not os.environ.get('RINGRIFT_DISABLE_TORCH_COMPILE'):
    os.environ['RINGRIFT_DISABLE_TORCH_COMPILE'] = '1'

# Setup path for imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.training.selfplay_config import (
    EngineMode,
    SelfplayConfig,
    parse_selfplay_args,
)
from app.training.selfplay_runner import (
    GameResult,
    GNNSelfplayRunner,
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

    elif mode == EngineMode.GNN:
        # Pure GNN policy network
        logger.info(f"Using GNNSelfplayRunner for {mode.value}")
        return GNNSelfplayRunner(config, model_tier="gnn")

    elif mode == EngineMode.HYBRID:
        # CNN-GNN hybrid network
        logger.info(f"Using GNNSelfplayRunner (hybrid) for {mode.value}")
        return GNNSelfplayRunner(config, model_tier="hybrid")

    elif mode in (EngineMode.GMO, EngineMode.EBMO, EngineMode.IG_GMO, EngineMode.CAGE):
        # Deprecated experimental AI modes - emit warning and fall back
        import warnings
        warnings.warn(
            f"{mode.value} is deprecated. Use 'gnn' or 'hybrid' instead.",
            DeprecationWarning,
        )
        logger.warning(f"{mode.value} is deprecated, falling back to heuristic")
        return HeuristicSelfplayRunner(config)

    else:
        # Default fallback
        logger.warning(f"Unknown engine mode {mode}, using heuristic")
        return HeuristicSelfplayRunner(config)


def emit_selfplay_complete_event(config: SelfplayConfig, stats: RunStats) -> None:
    """Emit SELFPLAY_COMPLETE event for pipeline automation.

    December 2025 - Phase 3A.2: Events are now always emitted at selfplay
    completion to enable automated pipeline triggering.

    Args:
        config: Selfplay configuration
        stats: Run statistics
    """
    import socket

    try:
        from app.coordination.event_router import publish_sync, StageEvent

        publish_sync(
            event_type=StageEvent.SELFPLAY_COMPLETE,
            payload={
                "config_key": config.config_key,
                "board_type": config.board_type,
                "num_players": config.num_players,
                "games_completed": stats.games_completed,
                "games_count": stats.games_completed,  # Alias for compatibility
                "total_samples": stats.total_samples,
                "database_path": str(config.record_db) if config.record_db else None,
                "db_path": str(config.record_db) if config.record_db else None,  # Alias
                "duration_seconds": stats.elapsed_seconds,
                "node_id": socket.gethostname(),
            },
            source="selfplay_cli",
        )
        logger.info(f"[Pipeline] Emitted SELFPLAY_COMPLETE for {config.config_key}")
    except ImportError:
        logger.debug(
            "[Pipeline] Could not emit SELFPLAY_COMPLETE - "
            "coordination module not available"
        )
    except Exception as e:
        logger.warning(f"[Pipeline] Failed to emit SELFPLAY_COMPLETE: {e}")


def main():
    """Main entry point for unified selfplay CLI."""
    # Parse args using unified config
    config = parse_selfplay_args(
        description="Unified Selfplay CLI - Generate training data via self-play"
    )

    # Query ImprovementOptimizer for priority boost (December 2025)
    # When on a promotion streak, boost search budget for higher quality games
    priority_boost = 0.0
    try:
        from app.training.improvement_optimizer import get_selfplay_priority_boost

        config_key = f"{config.board_type}_{config.num_players}p"
        priority_boost = get_selfplay_priority_boost(config_key)

        if priority_boost > 0.05:
            # On a promotion streak - boost search budget for higher quality data
            original_sims = config.mcts_simulations
            boost_factor = 1.0 + priority_boost * 2  # +0.15 boost → 1.3x simulations
            config.mcts_simulations = min(3200, int(original_sims * boost_factor))
            logger.info(
                f"[ImprovementOptimizer] Priority boost: +{priority_boost:.2f} → "
                f"mcts_simulations {original_sims} → {config.mcts_simulations}"
            )
        elif priority_boost < -0.05:
            # Config underperforming - reduce search budget to save compute
            original_sims = config.mcts_simulations
            reduction = 1.0 + priority_boost  # -0.10 boost → 0.9x simulations
            config.mcts_simulations = max(200, int(original_sims * reduction))
            logger.info(
                f"[ImprovementOptimizer] Priority reduction: {priority_boost:.2f} → "
                f"mcts_simulations {original_sims} → {config.mcts_simulations}"
            )
    except ImportError:
        pass  # ImprovementOptimizer not available
    except Exception as e:
        logger.debug(f"[ImprovementOptimizer] Could not get priority boost: {e}")

    # Query FeedbackAccelerator for games multiplier based on Elo momentum (December 2025)
    # When on an improvement streak, generate more games to capitalize on positive momentum
    games_multiplier = 1.0
    try:
        from app.training.feedback_accelerator import get_selfplay_multiplier

        config_key = f"{config.board_type}_{config.num_players}p"
        games_multiplier = get_selfplay_multiplier(config_key)

        if abs(games_multiplier - 1.0) > 0.05:
            original_games = config.num_games
            config.num_games = max(10, int(original_games * games_multiplier))
            logger.info(
                f"[FeedbackAccelerator] Elo momentum multiplier: {games_multiplier:.2f}x → "
                f"num_games {original_games} → {config.num_games}"
            )
    except ImportError:
        pass  # FeedbackAccelerator not available
    except Exception as e:
        logger.debug(f"[FeedbackAccelerator] Could not get games multiplier: {e}")

    # Log configuration summary
    logger.info("=" * 60)
    logger.info("Unified Selfplay")
    logger.info("=" * 60)
    logger.info(f"Board: {config.board_type}")
    logger.info(f"Players: {config.num_players}")
    logger.info(f"Games: {config.num_games}")
    logger.info(f"Engine: {config.engine_mode.value}")
    logger.info(f"MCTS sims: {config.mcts_simulations}")
    if priority_boost != 0.0:
        logger.info(f"Priority boost: {priority_boost:+.2f}")
    if games_multiplier != 1.0:
        logger.info(f"Games multiplier: {games_multiplier:.2f}x (Elo momentum)")
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

        # Always emit pipeline event for automation (Phase 3A.2: Dec 2025)
        # Event emission is now unconditional to enable automated pipeline
        if stats.games_completed > 0:
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
