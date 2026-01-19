"""Command-line interface for RingRift training.

This module provides the CLI argument parsing and main entry point for training.
The actual training logic is in train.py.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path

from app.models import BoardType
from app.training.config import TrainConfig, get_model_version_for_board

logger = logging.getLogger(__name__)


def _discover_best_npz_for_config(
    board_type: str,
    num_players: int,
    fallback_path: str,
) -> str:
    """Discover the best NPZ training file for a given config.

    Uses DataCatalog's NPZ discovery to find the most recent/suitable
    training data for the specified board type and player count.

    Args:
        board_type: Board type (e.g., 'hex8', 'square8')
        num_players: Number of players (2, 3, or 4)
        fallback_path: Path to use if no NPZ files found

    Returns:
        Path to the best NPZ file, or fallback_path if none found
    """
    try:
        from app.distributed.data_catalog import get_data_catalog

        catalog = get_data_catalog()
        best_npz = catalog.get_best_npz_for_training(
            board_type=board_type,
            num_players=num_players,
            prefer_recent=True,
        )

        if best_npz is not None:
            logger.info(
                f"[NPZ Discovery] Found training data: {best_npz.path} "
                f"({best_npz.sample_count:,} samples, "
                f"{best_npz.age_hours:.1f}h old)"
            )
            return str(best_npz.path)

        logger.info(
            f"[NPZ Discovery] No NPZ files found for {board_type}_{num_players}p, "
            f"using fallback: {fallback_path}"
        )
        return fallback_path

    except ImportError as e:
        logger.debug(f"[NPZ Discovery] DataCatalog not available: {e}")
        return fallback_path
    except (RuntimeError, ConnectionError, TimeoutError, OSError) as e:
        # Dec 2025: Narrowed from broad Exception to expected error types
        # These are the specific errors that can occur during catalog discovery
        logger.warning(f"[NPZ Discovery] Error during discovery: {e}")
        return fallback_path


def _preflight_cluster_check(
    board_type: str,
    num_players: int,
    sync_threshold: int = 5000,
    sync_timeout: float = 300.0,
) -> dict[str, int]:
    """Check cluster-wide data availability, trigger sync if beneficial.

    Jan 2026: Part of Phase 1 of Cluster Manifest Training Integration.

    Args:
        board_type: Board type (e.g., 'hex8', 'square8')
        num_players: Number of players (2, 3, or 4)
        sync_threshold: Minimum games required locally before training
        sync_timeout: Maximum seconds to wait for sync (default: 300)

    Returns:
        Dict with local, cluster, total game counts and sync status
    """
    config_key = f"{board_type}_{num_players}p"

    try:
        from app.distributed.data_catalog import get_data_registry

        registry = get_data_registry()
        status = registry.get_cluster_status().get(config_key, {})

        local_games = status.get("local", 0)
        cluster_games = status.get("cluster", 0)
        owc_games = status.get("owc", 0)
        s3_games = status.get("s3", 0)
        total_games = status.get("total", local_games)

        result = {
            "local": local_games,
            "cluster": cluster_games,
            "owc": owc_games,
            "s3": s3_games,
            "total": total_games,
            "sync_triggered": False,
            "sync_completed": False,
        }

        logger.info(
            f"[Cluster Preflight] {config_key}: local={local_games:,}, "
            f"cluster={cluster_games:,}, total={total_games:,}"
        )

        # Check if sync is beneficial
        if local_games >= sync_threshold:
            logger.info(
                f"[Cluster Preflight] Local data sufficient "
                f"({local_games:,} >= {sync_threshold:,}), skipping sync"
            )
            return result

        # Only trigger sync if cluster has more data than local
        remote_games = cluster_games + owc_games + s3_games
        if remote_games <= local_games:
            logger.info(
                f"[Cluster Preflight] No additional data on cluster "
                f"(remote={remote_games:,} <= local={local_games:,})"
            )
            return result

        # Trigger priority sync
        logger.info(
            f"[Cluster Preflight] Triggering priority sync: "
            f"local={local_games:,} < threshold={sync_threshold:,}, "
            f"cluster has {remote_games:,} more games"
        )

        result["sync_triggered"] = True

        try:
            import asyncio
            from app.coordination.sync_facade import get_sync_facade

            async def _run_priority_sync() -> bool:
                """Run priority sync in async context."""
                facade = get_sync_facade()
                response = await asyncio.wait_for(
                    facade.trigger_priority_sync(
                        reason="training_preflight",
                        config_key=config_key,
                        data_type="games",
                    ),
                    timeout=sync_timeout,
                )
                return response.success

            # Run in event loop
            loop = asyncio.new_event_loop()
            try:
                sync_success = loop.run_until_complete(_run_priority_sync())
                result["sync_completed"] = sync_success
            finally:
                loop.close()

            if sync_success:
                # Refresh counts after sync
                refreshed = registry.get_cluster_status().get(config_key, {})
                result["local"] = refreshed.get("local", local_games)
                logger.info(
                    f"[Cluster Preflight] Sync completed, "
                    f"local now={result['local']:,}"
                )
            else:
                logger.warning(
                    "[Cluster Preflight] Sync did not complete within timeout, "
                    "proceeding with existing data"
                )

        except ImportError as e:
            logger.warning(f"[Cluster Preflight] SyncFacade not available: {e}")
        except (RuntimeError, TimeoutError, asyncio.TimeoutError) as e:
            logger.warning(f"[Cluster Preflight] Sync failed: {e}")

        return result

    except ImportError as e:
        logger.debug(f"[Cluster Preflight] DataRegistry not available: {e}")
        return {"local": 0, "cluster": 0, "total": 0, "sync_triggered": False}
    except (RuntimeError, ConnectionError, TimeoutError, OSError) as e:
        logger.warning(f"[Cluster Preflight] Error during check: {e}")
        return {"local": 0, "cluster": 0, "total": 0, "sync_triggered": False}


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command line arguments.

    Args:
        args: Optional list of argument strings. If None, uses sys.argv.
              Useful for testing.
    """
    parser = argparse.ArgumentParser(
        description='Train RingRift Neural Network AI'
    )

    # Config file (overrides individual arguments)
    parser.add_argument(
        '--config', type=str, default=None,
        help='Path to TrainingPipelineConfig YAML/JSON file. Overrides individual arguments.'
    )

    # Data and model paths
    parser.add_argument(
        '--data-path', type=str, default=None,
        help='Path to training data (.npz file)'
    )
    parser.add_argument(
        '--save-path', type=str, default=None,
        help='Path to save best model weights'
    )

    # Training configuration
    parser.add_argument(
        '--epochs', type=int, default=None,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--batch-size', type=int, default=None,
        help='Training batch size. If not specified, auto-detects optimal size based on GPU memory.'
    )
    parser.add_argument(
        '--auto-tune-batch-size', action='store_true', default=True,
        help='Auto-tune batch size based on GPU memory (default: enabled). Use --no-auto-tune-batch-size to disable.'
    )
    parser.add_argument(
        '--no-auto-tune-batch-size', action='store_true',
        help='Disable auto-tuning batch size. Uses --batch-size value (default: 256) or explicit value.'
    )
    parser.add_argument(
        '--target-memory-fraction', type=float, default=None,
        help='Target GPU memory utilization fraction (0.0-1.0). Default is 0.50 (50%%), '
             'or 0.35 if --safe-mode is specified. Higher values risk OOM errors.'
    )
    parser.add_argument(
        '--safe-mode', action='store_true',
        help='Enable extra conservative batch size tuning (35%% memory target). '
             'Use when experiencing OOM errors or on nodes with memory pressure.'
    )
    parser.add_argument(
        '--track-calibration', action='store_true',
        help='Track value head calibration metrics during training'
    )

    # 2024-12 Hot Data Buffer and Integrated Enhancements
    parser.add_argument(
        '--use-hot-data-buffer', action='store_true',
        help='Enable hot data buffer for priority experience replay'
    )
    parser.add_argument(
        '--hot-buffer-size', type=int, default=10000,
        help='Size of hot data buffer (default: 10000)'
    )
    parser.add_argument(
        '--hot-buffer-mix-ratio', type=float, default=0.3,
        help='Ratio of samples from hot buffer vs regular data (default: 0.3)'
    )
    parser.add_argument(
        '--use-integrated-enhancements', action='store_true',
        help='Enable integrated training enhancements (curriculum, augmentation, etc.)'
    )
    parser.add_argument(
        '--enable-curriculum', action='store_true',
        help='Enable curriculum learning (progressive difficulty)'
    )
    parser.add_argument(
        '--enable-augmentation', action='store_true',
        help='Enable data augmentation (symmetry transforms)'
    )
    parser.add_argument(
        '--enable-elo-weighting', action='store_true',
        help='Enable ELO-based sample weighting'
    )
    parser.add_argument(
        '--enable-auxiliary-tasks', action='store_true',
        help='Enable auxiliary prediction tasks (outcome classification)'
    )
    parser.add_argument(
        '--enable-batch-scheduling', action='store_true',
        help='Enable dynamic batch size scheduling (linear ramp-up)'
    )
    parser.add_argument(
        '--enable-background-eval', action='store_true',
        help='Enable background Elo evaluation during training'
    )

    # Quality-weighted training (December 2025)
    # Resurrected from archive/deprecated_ai/ebmo_network.py
    parser.add_argument(
        '--enable-quality-weighting', action='store_true',
        help='Enable quality-weighted training (weight samples by MCTS visit counts)'
    )
    parser.add_argument(
        '--quality-weight-blend', type=float, default=0.5,
        help='Blend factor for quality weighting [0=uniform, 1=fully quality-weighted] (default: 0.5)'
    )
    parser.add_argument(
        '--quality-ranking-weight', type=float, default=0.1,
        help='Weight for ranking loss term (default: 0.1)'
    )

    parser.add_argument(
        '--learning-rate', type=float, default=0.0005,
        help='Initial learning rate (default: 0.0005 for stable training)'
    )
    parser.add_argument(
        '--seed', type=int, default=None,
        help='Random seed for reproducibility'
    )
    parser.add_argument(
        '--policy-label-smoothing', type=float, default=0.0,
        help='Policy label smoothing factor (default: 0.0)'
    )
    parser.add_argument(
        '--weight-decay', type=float, default=None,
        help='Weight decay for optimizer'
    )
    parser.add_argument(
        '--label-smoothing', type=float, default=0.0,
        help='Alias for --policy-label-smoothing'
    )
    parser.add_argument(
        '--filter-empty-policies', action='store_true',
        help='Filter out samples with empty policy targets'
    )
    parser.add_argument(
        '--min-quality-score', type=float, default=0.0,
        help='Minimum quality score for training samples (0.0-1.0). '
             'Samples below this threshold will be excluded from training. '
             'Requires NPZ file with quality_score field. (December 2025)'
    )
    parser.add_argument(
        '--feature-version', type=int, default=None,
        help='Feature encoding version (1=legacy, 2=enhanced features)'
    )

    # Early stopping
    parser.add_argument(
        '--early-stopping-patience', type=int, default=None,
        help='Number of epochs without improvement before stopping (0 to disable)'
    )

    # Checkpointing
    parser.add_argument(
        '--checkpoint-dir', type=str, default='checkpoints',
        help='Directory to save checkpoints'
    )
    parser.add_argument(
        '--checkpoint-interval', type=int, default=5,
        help='Epochs between checkpoints'
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to checkpoint to resume from (loads model, optimizer, scheduler, etc.)'
    )
    parser.add_argument(
        '--init-weights', type=str, default=None,
        help='Path to model weights for initialization (transfer learning). '
             'Only loads model weights, not optimizer state. Useful for 2p->4p transfer.'
    )
    parser.add_argument(
        '--init-weights-strict', action='store_true',
        help='Require all weights to match when using --init-weights (default: allow partial loading)'
    )
    parser.add_argument(
        '--freeze-policy', action='store_true',
        help='Freeze policy head and feature extractor, only train value head. '
             'Useful for fine-tuning value estimates without changing policy.'
    )

    # Learning rate scheduling
    parser.add_argument(
        '--warmup-epochs', type=int, default=None,
        help='Number of warmup epochs'
    )
    parser.add_argument(
        '--lr-scheduler', type=str, default=None,
        choices=['cosine', 'step', 'plateau', 'warmrestart'],
        help='Learning rate scheduler type'
    )
    parser.add_argument(
        '--lr-min', type=float, default=None,
        help='Minimum learning rate for scheduler'
    )
    parser.add_argument(
        '--lr-t0', type=int, default=10,
        help='Initial period for warm restart scheduler'
    )
    parser.add_argument(
        '--lr-t-mult', type=int, default=2,
        help='Period multiplier for warm restart scheduler'
    )

    # Board type and model configuration
    parser.add_argument(
        '--board-type', type=str, default=None,
        choices=['square8', 'square19', 'hex8', 'hexagonal'],
        help='Board type (default: square8)'
    )
    parser.add_argument(
        '--model-version', type=str, default=None,
        choices=['v2', 'v2_lite', 'v3', 'v3_lite', 'v3-flat', 'v4', 'v5', 'v5-gnn', 'v5-heavy',
                 'v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl'],
        help='Model architecture version (for CNN models). v3 uses spatial policy heads (default), '
             'v3-flat uses flat policy heads (fallback). v5/v5-heavy use heuristic features. '
             'v5-heavy-large/v5-heavy-xl are scaled up for 2000+ Elo (~25-35M params). '
             'Note: v6/v6-xl are deprecated aliases for v5-heavy-large/v5-heavy-xl.'
    )
    parser.add_argument(
        '--model-type', type=str, default='cnn',
        choices=['cnn', 'gnn', 'hybrid'],
        help='Model type: cnn (default), gnn (Graph Neural Network), or hybrid (CNN-GNN). '
             'GNN/hybrid require PyTorch Geometric.'
    )
    parser.add_argument(
        '--num-res-blocks', type=int, default=None,
        help='Number of residual blocks (overrides default for model version)'
    )
    parser.add_argument(
        '--num-filters', type=int, default=None,
        help='Number of filters per layer (overrides default for model version)'
    )
    parser.add_argument(
        '--memory-tier', type=str, default=None,
        choices=['v4', 'v3-high', 'v3-low', 'v5', 'v5.1', 'v5-heavy-large', 'v5-heavy-xl', 'v6', 'v6-xl'],
        help='Memory tier for model architecture. Determines num_filters and num_res_blocks. '
             'When resuming training, tier is auto-detected from checkpoint if not specified. '
             'Tiers: v4=128ch/13blocks, v3-high=192ch/12blocks, v3-low=96ch/6blocks, '
             'v5=160ch/11blocks, v5-heavy-large=256ch/18blocks, v5-heavy-xl=320ch/20blocks. '
             'Note: v6/v6-xl are deprecated aliases for v5-heavy-large/v5-heavy-xl.'
    )

    # Multi-player support
    parser.add_argument(
        '--multi-player', action='store_true',
        help='Enable multi-player mode (3+ players)'
    )
    parser.add_argument(
        '--num-players', type=int, default=2,
        help='Number of players for multi-player mode'
    )

    # Data augmentation - Jan 15, 2026: Auto-detect by default for hex boards
    parser.add_argument(
        '--augment-hex-symmetry', type=str, default='auto', nargs='?', const='always',
        choices=['auto', 'always', 'never'],
        help='Apply hex symmetry augmentation (D6 group, 6x data). '
             'auto=enable for hex boards (default), always=force on, never=disable'
    )
    parser.add_argument(
        # Dec 29 2025: Added quality-based weighting options (+20-40 Elo improvement)
        '--sampling-weights', type=str, default='quality_combined',
        choices=[
            'uniform', 'late_game', 'phase_emphasis', 'combined', 'source',
            'combined_source', 'chain_emphasis', 'combined_chain',
            # New in Dec 2025: quality-weighted options
            'quality',          # Uses quality_score from export
            'opponent_elo',     # Weights by opponent strength
            'quality_combined', # Best: quality + opponent_elo + late_game + source
        ],
        help='Sample weighting strategy. "quality_combined" (default) combines all signals: '
             'quality_score, opponent_elo, late_game, and source weighting for maximum '
             'training efficiency. Use "uniform" for no weighting.'
    )

    # Distributed training
    parser.add_argument(
        '--distributed', action='store_true',
        help='Enable distributed training via PyTorch DDP'
    )
    parser.add_argument(
        '--local-rank', type=int, default=-1,
        help='Local rank for distributed training'
    )
    parser.add_argument(
        '--scale-lr', action='store_true',
        help='Scale learning rate by world size'
    )
    parser.add_argument(
        '--lr-scale-mode', type=str, default='linear',
        choices=['linear', 'sqrt'],
        help='Learning rate scaling mode for distributed training'
    )
    parser.add_argument(
        '--find-unused-parameters', action='store_true',
        help='Enable find_unused_parameters for DDP'
    )

    # Fault tolerance (2025-12)
    parser.add_argument(
        '--disable-circuit-breaker', action='store_true',
        help='Disable circuit breaker for fault tolerance'
    )
    parser.add_argument(
        '--disable-anomaly-detection', action='store_true',
        help='Disable anomaly detection during training'
    )
    parser.add_argument(
        '--gradient-clip-mode', type=str, default='adaptive',
        choices=['none', 'fixed', 'adaptive'],
        help='Gradient clipping mode'
    )
    parser.add_argument(
        '--gradient-clip-max-norm', type=float, default=1.0,
        help='Maximum gradient norm for clipping'
    )
    parser.add_argument(
        '--anomaly-spike-threshold', type=float, default=3.0,
        help='Threshold for loss spike anomaly detection'
    )
    parser.add_argument(
        '--anomaly-gradient-threshold', type=float, default=100.0,
        help='Threshold for gradient anomaly detection'
    )
    parser.add_argument(
        '--disable-graceful-shutdown', action='store_true',
        help='Disable graceful shutdown handling'
    )

    # Pipeline automation (2025-12)
    parser.add_argument(
        '--enable-pipeline-auto-trigger', action='store_true',
        help='Enable automatic triggering of pipeline stages (sync → export → train → evaluate → promote). '
             'Requires bootstrap_coordination() to be called. Also respects COORDINATOR_AUTO_TRIGGER_PIPELINE env var.'
    )

    # Auto-promotion after training (January 2026)
    parser.add_argument(
        '--auto-promote', action='store_true',
        help='Automatically run gauntlet evaluation and promote if criteria met. '
             'Uses OR logic: promote if (Elo >= heuristic) OR (meets win rate floors with significance). '
             'Games per opponent controlled by --auto-promote-games.'
    )
    parser.add_argument(
        '--auto-promote-games', type=int, default=30,
        help='Number of games per opponent for auto-promotion gauntlet (default: 30)'
    )
    parser.add_argument(
        '--auto-promote-sync', action='store_true', default=True,
        help='Sync promoted model to cluster (default: True). Use --no-auto-promote-sync to disable.'
    )
    parser.add_argument(
        '--no-auto-promote-sync', action='store_true',
        help='Disable syncing promoted model to cluster'
    )

    # Gradient checkpointing (January 2026)
    parser.add_argument(
        '--gradient-checkpointing', action='store_true',
        help='Enable gradient checkpointing to reduce GPU memory usage. '
             'Trades ~20-30%% compute overhead for ~40-60%% memory savings. '
             'Useful for training large models (e.g., hexagonal) on memory-constrained GPUs.'
    )

    # Mixed precision training (January 2026)
    parser.add_argument(
        '--mixed-precision', action='store_true',
        help='Enable mixed precision training (AMP) to reduce GPU memory usage. '
             'Can reduce memory by ~40-50%% with minimal accuracy impact. '
             'Recommended for large models (v3, v5-heavy) on memory-constrained GPUs.'
    )
    parser.add_argument(
        '--amp-dtype', type=str, default='bfloat16',
        choices=['bfloat16', 'float16'],
        help='Data type for mixed precision training (default: bfloat16). '
             'bfloat16 is more stable, float16 may be faster on older GPUs.'
    )

    # Training data freshness check (2025-12)
    # MANDATORY by default to prevent stale data training (Phase 1.5)
    # This prevents 95% of stale data training incidents by failing early
    parser.add_argument(
        '--skip-freshness-check', action='store_true',
        help='Skip training data freshness check entirely (DANGEROUS - only for debugging). '
             'By default, training will fail if data is older than max-data-age-hours.'
    )
    parser.add_argument(
        '--max-data-age-hours', type=float, default=1.0,
        help='Maximum age in hours for "fresh" training data (default: 1.0). '
             'Training fails if data is older unless --allow-stale-data is specified.'
    )
    parser.add_argument(
        '--allow-stale-data', action='store_true',
        help='Allow training on stale data with warning instead of failing. '
             'Not recommended - may degrade model quality. Use only when fresh data is unavailable.'
    )

    # Stale fallback for 48-hour autonomous operation (December 2025)
    parser.add_argument(
        '--disable-stale-fallback', action='store_true',
        help='Disable stale data fallback (strict mode). By default, training will proceed '
             'with stale data after --max-sync-failures failures or --max-sync-duration timeout.'
    )
    parser.add_argument(
        '--max-sync-failures', type=int, default=5,
        help='Maximum sync failures before allowing stale fallback (default: 5). '
             'Set higher for stricter data freshness requirements.'
    )
    parser.add_argument(
        '--max-sync-duration', type=float, default=2700.0,
        help='Maximum time (seconds) to wait for sync before fallback (default: 2700 = 45 min). '
             'Training will proceed with stale data if sync takes longer than this.'
    )

    # Adaptive training intensity (2025-12)
    parser.add_argument(
        '--use-adaptive-intensity', action='store_true',
        help='Use FeedbackAccelerator to adjust training intensity based on Elo momentum. '
             'Improves learning velocity during positive feedback loops.'
    )

    # Hard example mining for curriculum learning (2025-12)
    parser.add_argument(
        '--enable-hard-example-mining', action='store_true',
        help='Enable hard example mining for curriculum learning. '
             'Tracks per-sample losses to focus training on difficult examples.'
    )
    parser.add_argument(
        '--hard-example-top-k', type=float, default=0.3,
        help='Fraction of hard examples to prioritize (default: 0.3 = top 30%%)'
    )

    # Outcome-weighted policy loss (2025-12)
    # Inspired by EBMO outcome-contrastive loss
    parser.add_argument(
        '--enable-outcome-weighted-policy', action='store_true',
        help='Enable outcome-weighted policy loss. Winners moves get higher weight, '
             'losers lower. Improves move quality learning (+5-10 Elo expected).'
    )
    parser.add_argument(
        '--outcome-weight-scale', type=float, default=0.5,
        help='Scale for outcome weighting (0=no effect, 1=full). Default: 0.5'
    )

    # Regularization - Jan 15, 2026: Increased to 0.12 based on hyperparameter tuning
    parser.add_argument(
        '--dropout', type=float, default=0.12,
        help='Dropout rate (default: 0.12 - forces robust feature learning)'
    )

    # CMA-ES Heuristic Optimization (offline, expert use only)
    parser.add_argument(
        '--cmaes-heuristic', action='store_true',
        help='Run CMA-ES heuristic weight optimization (offline mode)'
    )
    parser.add_argument(
        '--cmaes-tier-id', type=str, default='sq8_heuristic_baseline_v1',
        help='Tier ID for CMA-ES heuristic optimization'
    )
    parser.add_argument(
        '--cmaes-base-profile-id', type=str, default='baseline_v1',
        help='Base heuristic profile ID for CMA-ES optimization'
    )
    parser.add_argument(
        '--cmaes-generations', type=int, default=20,
        help='Number of CMA-ES generations'
    )
    parser.add_argument(
        '--cmaes-population-size', type=int, default=16,
        help='CMA-ES population size'
    )
    parser.add_argument(
        '--cmaes-seed', type=int, default=42,
        help='Random seed for CMA-ES optimization'
    )
    parser.add_argument(
        '--cmaes-games-per-candidate', type=int, default=50,
        help='Games per candidate for fitness evaluation'
    )

    # Curriculum training mode (2025-12)
    parser.add_argument(
        '--curriculum', action='store_true',
        help='Run curriculum training (iterative self-play with promotion)'
    )
    parser.add_argument(
        '--curriculum-generations', type=int, default=10,
        help='Number of curriculum generations'
    )
    parser.add_argument(
        '--curriculum-games-per-gen', type=int, default=1000,
        help='Games to generate per generation'
    )
    parser.add_argument(
        '--curriculum-training-epochs', type=int, default=10,
        help='Training epochs per generation'
    )
    parser.add_argument(
        '--curriculum-eval-games', type=int, default=100,
        help='Evaluation games for promotion decision'
    )
    parser.add_argument(
        '--curriculum-promotion-threshold', type=float, default=0.60,
        help='Win rate threshold for promotion (Dec 2025: tightened from 0.55)'
    )
    parser.add_argument(
        '--curriculum-data-retention', type=int, default=3,
        help='Number of past generations to retain data from'
    )
    parser.add_argument(
        '--curriculum-num-players', type=int, default=2,
        help='Number of players for curriculum games'
    )
    parser.add_argument(
        '--curriculum-engine', type=str, default='descent',
        choices=['descent', 'mcts', 'heuristic', 'neural', 'mix'],
        help='Engine type for self-play generation'
    )
    parser.add_argument(
        '--curriculum-engine-mix', type=str, default=None,
        help='Comma-separated engine:weight pairs for mix mode (e.g., "descent:0.5,mcts:0.3,heuristic:0.2")'
    )
    parser.add_argument(
        '--curriculum-engine-ratio', type=float, default=0.8,
        help='Ratio of games using primary engine (rest use random opponent)'
    )
    parser.add_argument(
        '--curriculum-output-dir', type=str, default='curriculum_runs',
        help='Output directory for curriculum artifacts (default: curriculum_runs)',
    )
    parser.add_argument(
        '--curriculum-base-model',
        type=str,
        default=None,
        help='Path to initial model checkpoint for curriculum training',
    )

    # Autonomous training mode (December 2025)
    parser.add_argument(
        '--autonomous', action='store_true',
        help='Enable autonomous training mode. Treats validation errors as '
             'warnings (stale data, pending gate DBs, non-canonical sources). '
             'Equivalent to RINGRIFT_AUTONOMOUS_MODE=1. Use for unattended training.'
    )
    # NOTE: --allow-stale-data and --max-data-age-hours already defined above (line ~390)

    # Cluster data awareness (January 2026)
    parser.add_argument(
        '--use-cluster-data', action='store_true',
        help='Enable cluster-wide data awareness. Before training, checks cluster '
             'manifest for total available games and triggers sync if local data '
             'is insufficient. Requires P2P orchestrator running.'
    )
    parser.add_argument(
        '--cluster-sync-threshold', type=int, default=5000,
        help='Minimum games required locally before training. If local < threshold '
             'and cluster has more, triggers priority sync (default: 5000).'
    )
    parser.add_argument(
        '--skip-cluster-preflight', action='store_true',
        help='Skip cluster preflight check even with --use-cluster-data. '
             'Use when you want cluster awareness for metrics but not auto-sync.'
    )

    return parser.parse_args(args)


def main() -> None:
    """Main entry point for training."""
    # Import here to avoid circular imports and reduce startup time
    from app.training.train import run_cmaes_heuristic_optimization, train_model

    args = parse_args()

    # Handle autonomous mode (December 2025)
    if getattr(args, 'autonomous', False):
        os.environ["RINGRIFT_AUTONOMOUS_MODE"] = "1"
        logger.info("[TrainCLI] Autonomous mode enabled via --autonomous flag")

    # Handle allow-stale-data flag (implicit in autonomous mode)
    if getattr(args, 'allow_stale_data', False) or getattr(args, 'autonomous', False):
        os.environ["RINGRIFT_ALLOW_STALE_DATA"] = "1"

    # Initialize pipeline auto-trigger if requested (2025-12)
    # This wires up the coordination infrastructure for automatic stage progression
    if getattr(args, 'enable_pipeline_auto_trigger', False):
        try:
            from app.coordination.coordination_bootstrap import bootstrap_coordination
            bootstrap_result = bootstrap_coordination(pipeline_auto_trigger=True)
            logger.info(
                f"[TrainCLI] Pipeline auto-trigger enabled. "
                f"Initialized {len([v for v in bootstrap_result.values() if v])} coordinators."
            )
        except ImportError:
            logger.warning(
                "[TrainCLI] --enable-pipeline-auto-trigger requires app.coordination module. "
                "Continuing without pipeline automation."
            )
        except (RuntimeError, ValueError, ConnectionError) as e:
            # Dec 2025: Narrowed from broad Exception to expected error types
            logger.warning(f"[TrainCLI] Failed to enable pipeline auto-trigger: {e}")

    # Load config file if provided (overrides individual arguments)
    if args.config:
        from app.training.config import TrainingPipelineConfig
        try:
            pipeline_config = TrainingPipelineConfig.load(args.config)
            logger.info(f"Loaded config from {args.config}")

            # Apply config values to args (config takes precedence if args not set)
            if args.data_path is None:
                args.data_path = pipeline_config.data.data_dir
            if args.epochs is None:
                args.epochs = pipeline_config.train.epochs_per_iter
            if args.batch_size is None:
                args.batch_size = pipeline_config.train.batch_size
            if args.learning_rate is None:
                args.learning_rate = pipeline_config.train.learning_rate
            if args.checkpoint_dir is None or args.checkpoint_dir == 'checkpoints':
                args.checkpoint_dir = pipeline_config.checkpoint.checkpoint_dir
            if args.board_type is None:
                args.board_type = pipeline_config.train.board_type.value

            # Log config summary
            logger.info(f"  Board type: {pipeline_config.train.board_type.value}")
            logger.info(f"  Learning rate: {pipeline_config.train.learning_rate}")
            logger.info(f"  Batch size: {pipeline_config.train.batch_size}")

        except (FileNotFoundError, ValueError, KeyError, OSError) as e:
            # Dec 2025: Narrowed from broad Exception to config-specific errors
            # FileNotFoundError: config file doesn't exist
            # ValueError: invalid config values
            # KeyError: missing required config keys
            # OSError: file access issues
            logger.error(f"Failed to load config from {args.config}: {e}")
            raise

    # Curriculum training mode: iterative self-play with model promotion
    if getattr(args, "curriculum", False):
        from app.training.curriculum import CurriculumConfig, CurriculumTrainer

        # Map board type string to enum
        board_type_map = {
            'square8': BoardType.SQUARE8,
            'square19': BoardType.SQUARE19,
            'hex8': BoardType.HEX8,
            'hexagonal': BoardType.HEXAGONAL,
        }
        board_type = board_type_map.get(
            args.board_type or 'square8',
            BoardType.SQUARE8,
        )

        config = CurriculumConfig(
            board_type=board_type,
            generations=args.curriculum_generations,
            games_per_generation=args.curriculum_games_per_gen,
            training_epochs=args.curriculum_training_epochs,
            eval_games=args.curriculum_eval_games,
            promotion_threshold=args.curriculum_promotion_threshold,
            data_retention=args.curriculum_data_retention,
            num_players=args.curriculum_num_players,
            # Learning rate from standard args if provided
            learning_rate=args.learning_rate or 1e-3,
            batch_size=args.batch_size or 32,
            base_seed=args.seed or 42,
            output_dir=args.curriculum_output_dir,
            # Engine configuration
            engine=args.curriculum_engine,
            engine_mix=args.curriculum_engine_mix,
            engine_ratio=args.curriculum_engine_ratio,
        )

        trainer = CurriculumTrainer(config, args.curriculum_base_model)
        results = trainer.run()

        # Print summary
        print("\n" + "=" * 60)
        print("CURRICULUM TRAINING COMPLETE")
        print("=" * 60)
        promoted_count = sum(1 for r in results if r.promoted)
        print(f"Total generations: {len(results)}")
        print(f"Promotions: {promoted_count}")
        print(f"Output directory: {trainer.run_dir}")
        print()
        for r in results:
            status = "PROMOTED" if r.promoted else "skipped"
            print(
                f"Gen {r.generation}: {status} (win={r.win_rate:.1%}, "
                f"loss={r.training_loss:.4f})"
            )
        return

    # Offline heuristic CMA-ES optimisation mode is explicitly opt-in and
    # does not affect the neural-network training path.
    if getattr(args, "cmaes_heuristic", False):
        report = run_cmaes_heuristic_optimization(
            tier_id=args.cmaes_tier_id,
            base_profile_id=args.cmaes_base_profile_id,
            generations=args.cmaes_generations,
            population_size=args.cmaes_population_size,
            rng_seed=args.cmaes_seed,
            games_per_candidate=args.cmaes_games_per_candidate,
        )
        out_dir = Path("results") / "ai_eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        out_path = out_dir / f"cmaes_heuristic_square8_{ts}.json"
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Wrote CMA-ES heuristic report to {out_path}")
        return

    # Create config
    config = TrainConfig()

    # Override config from CLI args
    if args.epochs is not None:
        config.epochs_per_iter = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.learning_rate is not None:
        config.learning_rate = args.learning_rate
    if args.seed is not None:
        config.seed = args.seed
    if args.policy_label_smoothing > 0:
        config.policy_label_smoothing = args.policy_label_smoothing
    if hasattr(args, 'weight_decay') and args.weight_decay is not None:
        config.weight_decay = args.weight_decay
    if hasattr(args, 'label_smoothing') and args.label_smoothing > 0:
        config.policy_label_smoothing = args.label_smoothing
    if args.feature_version is not None:
        config.feature_version = args.feature_version
    if args.filter_empty_policies:
        config.allow_empty_policies = False
    if args.board_type is not None:
        board_type_map = {
            'square8': BoardType.SQUARE8,
            'square19': BoardType.SQUARE19,
            'hex8': BoardType.HEX8,
            'hexagonal': BoardType.HEXAGONAL,
        }
        config.board_type = board_type_map[args.board_type]

    # Determine paths
    data_path = args.data_path
    if data_path is None:
        # Try NPZ discovery to find best training data for this config
        data_path = _discover_best_npz_for_config(
            board_type=args.board_type or 'hex8',
            num_players=args.num_players or 2,
            fallback_path=os.path.join(config.data_dir, "dataset.npz"),
        )

    # ==========================================================================
    # Cluster Data Preflight Check (January 2026)
    # ==========================================================================
    # Phase 1 of Cluster Manifest Training Integration: Check cluster-wide data
    # availability before training and trigger sync if local data is insufficient.
    if getattr(args, 'use_cluster_data', False) and not getattr(args, 'skip_cluster_preflight', False):
        board_type_str = args.board_type or 'hex8'
        num_players_val = args.num_players or 2
        sync_threshold = getattr(args, 'cluster_sync_threshold', 5000)

        logger.info(
            f"[TrainCLI] Running cluster preflight check for {board_type_str}_{num_players_val}p "
            f"(threshold={sync_threshold:,})"
        )

        preflight_result = _preflight_cluster_check(
            board_type=board_type_str,
            num_players=num_players_val,
            sync_threshold=sync_threshold,
            sync_timeout=300.0,
        )

        if preflight_result.get("sync_triggered"):
            if preflight_result.get("sync_completed"):
                logger.info(
                    f"[TrainCLI] Cluster sync completed. Local games: {preflight_result['local']:,}"
                )
            else:
                logger.warning(
                    "[TrainCLI] Cluster sync was triggered but did not complete. "
                    "Proceeding with existing data."
                )

    # Board-aware default model version (centralized in config.py)
    # Auto-detects from data if data_path provided (Dec 2025)
    model_version = args.model_version
    if model_version is None:
        model_version = get_model_version_for_board(config.board_type, data_path=data_path)

    # December 29, 2025: Use canonical model paths by default
    # December 30, 2025: Include architecture in default filename to support
    # multi-architecture training (e.g., canonical_hex8_2p_v4.pth)
    if args.save_path:
        save_path = args.save_path
    else:
        board_type = args.board_type or 'square8'
        num_players = args.num_players or 2
        # Include architecture in filename only if explicitly specified
        # This avoids breaking backward compat with existing v5 models
        if args.model_version and args.model_version != 'v5':
            save_path = os.path.join(
                config.model_dir,
                f"canonical_{board_type}_{num_players}p_{args.model_version}.pth",
            )
        else:
            # Default v5 gets the canonical name without suffix for backward compat
            save_path = os.path.join(
                config.model_dir,
                f"canonical_{board_type}_{num_players}p.pth",
            )

    # ==========================================================================
    # Adaptive Training Intensity (2025-12)
    # ==========================================================================
    # Use FeedbackAccelerator to adjust training parameters based on Elo momentum.
    # This creates positive feedback loops: improving models train faster,
    # plateauing models get more resources, regressing models conserve resources.
    # NOTE: As of Dec 2025, adaptive intensity is ENABLED BY DEFAULT. The
    # --use-adaptive-intensity flag is now deprecated (kept for backwards compat).
    try:
        from app.training.feedback_accelerator import get_feedback_accelerator

        # Build config key from board type and num players
        board_str = config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type)
        config_key = f"{board_str}_{args.num_players}p"

        accelerator = get_feedback_accelerator()
        intensity = accelerator.get_training_intensity(config_key)

        # Apply intensity multipliers only if non-default
        intensity_level = intensity.get('intensity', 'normal')
        epochs_mult = intensity.get('epochs_multiplier', 1.0)
        lr_mult = intensity.get('learning_rate_multiplier', 1.0)

        if intensity_level != 'normal' or epochs_mult != 1.0 or lr_mult != 1.0:
            original_epochs = config.epochs_per_iter
            original_lr = config.learning_rate

            config.epochs_per_iter = int(config.epochs_per_iter * epochs_mult)
            config.learning_rate = config.learning_rate * lr_mult

            logger.info(
                f"[TrainCLI] Adaptive intensity applied for {config_key}: "
                f"intensity={intensity_level}, "
                f"epochs={original_epochs}→{config.epochs_per_iter} ({epochs_mult:.2f}x), "
                f"lr={original_lr:.6f}→{config.learning_rate:.6f} ({lr_mult:.2f}x)"
            )
    except ImportError:
        # FeedbackAccelerator not available - continue with default intensity
        pass
    except (RuntimeError, ValueError, AttributeError) as e:
        # Dec 2025: Narrowed from broad Exception to FeedbackAccelerator-specific errors
        logger.debug(f"[TrainCLI] Adaptive intensity not applied: {e}")

    # ==========================================================================
    # Subscribe to HYPERPARAMETER_UPDATED events (Phase 14, December 2025)
    # ==========================================================================
    # Subscribe to feedback events from GauntletFeedbackController to allow
    # runtime adjustment of training parameters based on gauntlet evaluation.
    _hyperparameter_updates: dict = {}

    def _subscribe_to_hyperparameter_events(config_key: str) -> None:
        """Subscribe to HYPERPARAMETER_UPDATED events for this config."""
        try:
            from app.coordination.event_router import get_event_bus

            bus = get_event_bus()
            if bus is None:
                return

            def on_hyperparameter_updated(event):
                """Handle HYPERPARAMETER_UPDATED from GauntletFeedbackController."""
                payload = event.payload if hasattr(event, "payload") else event
                event_config = payload.get("config", "")

                if event_config != config_key:
                    return

                parameter = payload.get("parameter", "")
                new_value = payload.get("new_value", None)
                reason = payload.get("reason", "unknown")

                if parameter and new_value is not None:
                    _hyperparameter_updates[parameter] = {
                        "value": new_value,
                        "reason": reason,
                    }
                    logger.info(
                        f"[FeedbackLoop] Received hyperparameter update for {config_key}: "
                        f"{parameter}={new_value} (reason: {reason})"
                    )

            bus.subscribe("HYPERPARAMETER_UPDATED", on_hyperparameter_updated)
            logger.debug(f"[TrainCLI] Subscribed to HYPERPARAMETER_UPDATED for {config_key}")

        except ImportError:
            pass  # Event system not available
        except (RuntimeError, ConnectionError) as e:
            # Dec 2025: Narrowed from broad Exception to event-system specific errors
            logger.debug(f"[TrainCLI] Failed to subscribe to hyperparameter events: {e}")

    # Build config key and subscribe to feedback events
    board_str = config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type)
    config_key = f"{board_str}_{args.num_players}p"
    _subscribe_to_hyperparameter_events(config_key)

    # Apply any pending hyperparameter updates received before training starts
    def _apply_hyperparameter_updates(config: TrainConfig) -> None:
        """Apply pending hyperparameter updates to training config."""
        for param, update in _hyperparameter_updates.items():
            value = update["value"]
            reason = update["reason"]

            if param == "temperature_scale":
                # Temperature affects selfplay, not direct training
                logger.info(f"[FeedbackLoop] Note: temperature_scale update received (for selfplay)")
            elif param == "quality_threshold_boost":
                # Quality affects data loading, not direct training
                logger.info(f"[FeedbackLoop] Note: quality_threshold_boost update received")
            elif param == "epoch_multiplier" and isinstance(value, (int, float)):
                old_epochs = config.epochs_per_iter
                config.epochs_per_iter = int(config.epochs_per_iter * value)
                logger.info(
                    f"[FeedbackLoop] Epochs adjusted: {old_epochs} -> {config.epochs_per_iter} "
                    f"(multiplier={value}, reason={reason})"
                )
            elif param == "learning_rate" and isinstance(value, (int, float)):
                old_lr = config.learning_rate
                config.learning_rate = float(value)
                logger.info(
                    f"[FeedbackLoop] Learning rate adjusted: {old_lr} -> {config.learning_rate} "
                    f"(reason={reason})"
                )
            elif param == "batch_size" and isinstance(value, int):
                old_batch = config.batch_size
                config.batch_size = value
                logger.info(
                    f"[FeedbackLoop] Batch size adjusted: {old_batch} -> {config.batch_size} "
                    f"(reason={reason})"
                )

    _apply_hyperparameter_updates(config)

    # Route to GNN/hybrid training if model-type specified (2025-12)
    model_type = getattr(args, 'model_type', 'cnn')
    if model_type in ('gnn', 'hybrid'):
        import subprocess
        import sys

        logger.info(f"[TrainCLI] Routing to GNN training pipeline (model_type={model_type})")

        board_type_str = config.board_type.value if hasattr(config.board_type, 'value') else str(config.board_type)

        # Determine output directory from save_path
        save_dir = str(Path(save_path).parent) if save_path else f"models/gnn_{board_type_str}_{args.num_players}p"

        # Build command for GNN training script
        cmd = [
            sys.executable, "-m", "app.training.train_gnn_policy",
            "--data-path", data_path,
            "--board-type", board_type_str,
            "--output-dir", save_dir,
            "--epochs", str(config.epochs_per_iter),
            "--batch-size", str(config.batch_size),
            "--lr", str(config.learning_rate),
        ]

        logger.info(f"[TrainCLI] Executing: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=False)

        if result.returncode != 0:
            logger.error(f"[TrainCLI] GNN training failed with exit code {result.returncode}")
            sys.exit(result.returncode)

        logger.info("[TrainCLI] GNN training completed successfully")
        return

    # December 2025: Apply memory tier to architecture parameters
    # Memory tier provides a convenient way to set num_filters and num_res_blocks
    memory_tier = getattr(args, 'memory_tier', None)
    if memory_tier:
        TIER_DEFAULTS = {
            "v4": (128, 13),
            "v3-high": (192, 12),
            "v3-low": (96, 6),
            "v5": (160, 11),
            "v5.1": (160, 11),
            # Canonical names for scaled-up v5-heavy
            "v5-heavy-large": (256, 18),
            "v5-heavy-xl": (320, 20),
            # Deprecated aliases (will be removed Q2 2026)
            "v6": (256, 18),
            "v6-xl": (320, 20),
        }
        tier_filters, tier_blocks = TIER_DEFAULTS.get(memory_tier, (128, 13))

        # Only override if not explicitly set via --num-filters/--num-res-blocks
        if getattr(args, 'num_filters', None) is None:
            args.num_filters = tier_filters
            logger.info(f"[TrainCLI] Using memory tier '{memory_tier}': num_filters={tier_filters}")
        if getattr(args, 'num_res_blocks', None) is None:
            args.num_res_blocks = tier_blocks
            logger.info(f"[TrainCLI] Using memory tier '{memory_tier}': num_res_blocks={tier_blocks}")
    elif getattr(args, 'resume', None):
        # If resuming without explicit tier, note that auto-detection will happen in train.py
        logger.info(
            "[TrainCLI] Resuming training - architecture will be auto-detected from checkpoint. "
            "Use --memory-tier to override if needed."
        )

    # Jan 15, 2026: Convert tri-state augment-hex-symmetry to boolean
    # auto = enable for hex boards, always = force on, never = disable
    augment_hex = args.augment_hex_symmetry
    if augment_hex == 'auto':
        augment_hex = config.board_type in (BoardType.HEX8, BoardType.HEXAGONAL)
        if augment_hex:
            logger.info(f"[TrainCLI] Auto-enabled hex symmetry augmentation for {config.board_type}")
    elif augment_hex == 'always':
        augment_hex = True
        logger.info("[TrainCLI] Hex symmetry augmentation forced ON")
    else:  # 'never'
        augment_hex = False

    # Run CNN training
    train_model(
        config=config,
        data_path=data_path,
        save_path=save_path,
        early_stopping_patience=args.early_stopping_patience,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_interval=args.checkpoint_interval,
        warmup_epochs=args.warmup_epochs,
        lr_scheduler=args.lr_scheduler,
        lr_min=args.lr_min,
        lr_t0=args.lr_t0,
        lr_t_mult=args.lr_t_mult,
        resume_path=args.resume,
        init_weights_path=getattr(args, 'init_weights', None),
        init_weights_strict=getattr(args, 'init_weights_strict', False),
        freeze_policy=getattr(args, 'freeze_policy', False),
        augment_hex_symmetry=augment_hex,
        distributed=args.distributed,
        local_rank=args.local_rank,
        scale_lr=args.scale_lr,
        lr_scale_mode=args.lr_scale_mode,
        find_unused_parameters=args.find_unused_parameters,
        sampling_weights=args.sampling_weights,
        multi_player=args.multi_player,
        num_players=args.num_players,
        model_version=model_version,
        num_res_blocks=getattr(args, 'num_res_blocks', None),
        num_filters=getattr(args, 'num_filters', None),
        # January 2026: Auto-tune batch size enabled by default for optimal GPU utilization
        auto_tune_batch_size=not getattr(args, 'no_auto_tune_batch_size', False),
        # January 2026: Conservative memory targeting (50% default, 35% safe mode)
        target_memory_fraction=getattr(args, 'target_memory_fraction', None),
        safe_mode=getattr(args, 'safe_mode', False),
        track_calibration=getattr(args, 'track_calibration', False),
        # 2024-12 Hot Data Buffer and Integrated Enhancements
        use_hot_data_buffer=getattr(args, 'use_hot_data_buffer', False),
        hot_buffer_size=getattr(args, 'hot_buffer_size', 10000),
        hot_buffer_mix_ratio=getattr(args, 'hot_buffer_mix_ratio', 0.3),
        use_integrated_enhancements=getattr(args, 'use_integrated_enhancements', True),  # Dec 2025: Enable by default
        enable_curriculum=getattr(args, 'enable_curriculum', False),
        enable_augmentation=getattr(args, 'enable_augmentation', False),
        enable_elo_weighting=getattr(args, 'enable_elo_weighting', True),  # Dec 2025: Enable for +20-35 Elo
        enable_auxiliary_tasks=getattr(args, 'enable_auxiliary_tasks', True),  # Dec 2025: Enable for +5-15 Elo
        enable_batch_scheduling=getattr(args, 'enable_batch_scheduling', False),
        enable_background_eval=getattr(args, 'enable_background_eval', True),  # Dec 2025: Enable for feedback loop
        # Quality-weighted training (Dec 2025) - resurrected from ebmo_network.py
        # Dec 27 2025: Enabled by default for ML acceleration (+5-10 Elo)
        enable_quality_weighting=getattr(args, 'enable_quality_weighting', True),
        quality_weight_blend=getattr(args, 'quality_weight_blend', 0.5),
        quality_ranking_weight=getattr(args, 'quality_ranking_weight', 0.1),
        # Fault tolerance (2025-12)
        enable_circuit_breaker=not getattr(args, 'disable_circuit_breaker', False),
        enable_anomaly_detection=not getattr(args, 'disable_anomaly_detection', False),
        gradient_clip_mode=getattr(args, 'gradient_clip_mode', 'adaptive'),
        gradient_clip_max_norm=getattr(args, 'gradient_clip_max_norm', 1.0),
        anomaly_spike_threshold=getattr(args, 'anomaly_spike_threshold', 3.0),
        anomaly_gradient_threshold=getattr(args, 'anomaly_gradient_threshold', 100.0),
        enable_graceful_shutdown=not getattr(args, 'disable_graceful_shutdown', False),
        # Regularization (2025-12)
        dropout=getattr(args, 'dropout', 0.08),
        # GNN support (2025-12)
        model_type=getattr(args, 'model_type', 'cnn'),
        # Training data freshness check (2025-12) - mandatory by default
        skip_freshness_check=getattr(args, 'skip_freshness_check', False),
        max_data_age_hours=getattr(args, 'max_data_age_hours', 1.0),
        allow_stale_data=getattr(args, 'allow_stale_data', False) or getattr(args, 'autonomous', False),
        # Stale fallback for 48-hour autonomous operation (December 2025)
        disable_stale_fallback=getattr(args, 'disable_stale_fallback', False),
        max_sync_failures=getattr(args, 'max_sync_failures', 5),
        max_sync_duration=getattr(args, 'max_sync_duration', 2700.0),
        # Quality-aware sample filtering (December 2025)
        min_quality_score=getattr(args, 'min_quality_score', 0.0),
        # Hard example mining for curriculum learning (2025-12)
        # Dec 27 2025: Enabled by default for ML acceleration (+5-10 Elo)
        hard_example_mining=getattr(args, 'enable_hard_example_mining', True),
        hard_example_top_k=getattr(args, 'hard_example_top_k', 0.3),
        # Outcome-weighted policy loss (2025-12)
        # Dec 27 2025: Enabled by default for ML acceleration (+5-10 Elo)
        enable_outcome_weighted_policy=getattr(args, 'enable_outcome_weighted_policy', True),
        outcome_weight_scale=getattr(args, 'outcome_weight_scale', 0.5),
        # Auto-promotion after training (January 2026)
        auto_promote=getattr(args, 'auto_promote', False),
        auto_promote_games=getattr(args, 'auto_promote_games', 30),
        auto_promote_sync=getattr(args, 'auto_promote_sync', True) and not getattr(args, 'no_auto_promote_sync', False),
        # Gradient checkpointing (January 2026)
        gradient_checkpointing=getattr(args, 'gradient_checkpointing', False),
        # Mixed precision training (January 2026)
        mixed_precision=getattr(args, 'mixed_precision', False),
        amp_dtype=getattr(args, 'amp_dtype', 'bfloat16'),
    )


if __name__ == "__main__":
    main()
