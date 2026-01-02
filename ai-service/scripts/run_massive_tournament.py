#!/usr/bin/env python3
"""Massive tournament for OWC model archive Elo evaluation.

This script evaluates all unique models from an external archive (e.g., OWC drive)
against diverse baseline opponents using the existing distributed tournament
infrastructure.

Features:
- SHA256 deduplication (16K files → ~450-600 unique models)
- Distributed execution across P2P cluster (35+ nodes)
- Diverse opponents (random, heuristic, MCTS variants)
- Automatic retry on node failures
- Checkpoint/resume support for interrupted runs
- Games recorded to database for training

Leverages existing infrastructure:
- ModelDeduplicator for SHA256 deduplication
- launch_distributed_elo_tournament for distributed execution with retries
- EloService for Elo persistence

Usage:
    # Dry run - scan and report unique models
    python scripts/run_massive_tournament.py --source /Volumes/RingRift-Data --dry-run

    # Full tournament for all configs
    python scripts/run_massive_tournament.py --source /Volumes/RingRift-Data --games 25

    # Single config only
    python scripts/run_massive_tournament.py --source /Volumes/RingRift-Data --config hex8_2p

    # Resume interrupted tournament
    python scripts/run_massive_tournament.py --resume results/massive_tournament_abc123.json

January 2, 2026: Created as thin wrapper around existing tournament infrastructure.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# Add ai-service to path
AI_SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(AI_SERVICE_ROOT))

from app.utils.model_deduplicator import ModelDeduplicator, UniqueModel

# Import from existing tournament infrastructure
from scripts.launch_distributed_elo_tournament import (
    run_distributed_tournament,
    discover_healthy_nodes,
    calculate_elo,
    AI_TYPE_CONFIGS,
    AI_TYPE_CONFIGS_LIGHTWEIGHT,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default baseline opponents for diverse evaluation
DEFAULT_BASELINES = ["random", "heuristic", "mcts_100", "mcts_200"]

# Results directory
RESULTS_DIR = Path("results/massive_tournament")


@dataclass
class TournamentCheckpoint:
    """Checkpoint for resumable tournament state."""

    tournament_id: str
    source_path: str
    started_at: float
    last_updated: float = field(default_factory=time.time)

    # Config-level progress
    configs_completed: list[str] = field(default_factory=list)
    configs_pending: list[str] = field(default_factory=list)
    current_config: str | None = None

    # Overall stats
    total_models: int = 0
    total_games_played: int = 0
    total_games_target: int = 0

    # Per-config results
    config_results: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Model registry (SHA256 -> model info for resume)
    model_registry: dict[str, dict[str, Any]] = field(default_factory=dict)

    def save(self, path: Path) -> None:
        """Save checkpoint to file."""
        self.last_updated = time.time()
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)
        logger.info(f"Checkpoint saved: {path}")

    @classmethod
    def load(cls, path: Path) -> TournamentCheckpoint:
        """Load checkpoint from file."""
        with open(path) as f:
            data = json.load(f)
        return cls(**data)


def parse_config_key(config_key: str) -> tuple[str, int]:
    """Parse config key like 'hex8_2p' into (board_type, num_players)."""
    parts = config_key.rsplit("_", 1)
    if len(parts) != 2 or not parts[1].endswith("p"):
        raise ValueError(f"Invalid config key: {config_key}")
    board_type = parts[0]
    num_players = int(parts[1][:-1])
    return board_type, num_players


def print_dedup_report(models: list[UniqueModel], deduplicator: ModelDeduplicator) -> None:
    """Print detailed deduplication report."""
    print("\n" + "=" * 70)
    print("MODEL DEDUPLICATION REPORT")
    print("=" * 70)

    # Group by config
    by_config = deduplicator.group_by_config(models)

    print(f"\nTotal unique models: {len(models)}")
    print(f"Configs found: {len(by_config)}")

    print(f"\n{'Config':<20} {'Models':<10} {'Architectures':<30}")
    print("-" * 60)

    for config_key in sorted(by_config.keys()):
        config_models = by_config[config_key]
        archs = Counter(m.architecture for m in config_models)
        arch_str = ", ".join(f"{a}:{c}" for a, c in archs.most_common(3))
        print(f"{config_key:<20} {len(config_models):<10} {arch_str:<30}")

    # Family distribution
    families = Counter(m.model_family for m in models)
    print(f"\nTop model families:")
    for family, count in families.most_common(10):
        print(f"  {family}: {count}")

    # Size distribution
    total_size = sum(m.file_size for m in models)
    print(f"\nTotal size: {total_size / (1024**3):.2f} GB")
    print("=" * 70)


def print_config_results(
    config_key: str,
    ratings: dict[str, float],
    models: list[UniqueModel],
    results_count: int,
) -> None:
    """Print results for a single config."""
    print(f"\n{'=' * 70}")
    print(f"RESULTS: {config_key}")
    print(f"{'=' * 70}")
    print(f"Games completed: {results_count}")

    # Separate model ratings from baseline ratings
    baseline_names = set(DEFAULT_BASELINES)
    model_ratings = {k: v for k, v in ratings.items() if k not in baseline_names}
    baseline_ratings = {k: v for k, v in ratings.items() if k in baseline_names}

    # Print baselines
    print(f"\nBaseline Elo ratings:")
    for name, elo in sorted(baseline_ratings.items(), key=lambda x: -x[1]):
        print(f"  {name:<20} {elo:>7.1f}")

    # Print top models
    print(f"\nTop 10 models:")
    sorted_models = sorted(model_ratings.items(), key=lambda x: -x[1])
    for i, (agent_id, elo) in enumerate(sorted_models[:10], 1):
        # Find the original model info
        sha256_prefix = agent_id.replace("model_", "")
        model_info = next(
            (m for m in models if m.sha256.startswith(sha256_prefix)),
            None,
        )
        name = model_info.model_family if model_info else agent_id
        print(f"  {i:>2}. {name:<35} {elo:>7.1f}")

    print(f"{'=' * 70}")


async def run_massive_tournament(
    source_path: Path,
    games_per_pairing: int = 25,
    config_filter: str | None = None,
    baselines: list[str] | None = None,
    max_parallel: int = 2,
    no_retry: bool = False,
    checkpoint_path: Path | None = None,
    resume_from: TournamentCheckpoint | None = None,
) -> dict[str, dict[str, float]]:
    """Run massive tournament across all configs.

    Args:
        source_path: Path to model archive (e.g., /Volumes/RingRift-Data)
        games_per_pairing: Games per (model, opponent) pair
        config_filter: Optional config key to filter (e.g., "hex8_2p")
        baselines: Baseline opponents (default: random, heuristic, mcts)
        max_parallel: Max parallel matches per node
        no_retry: Disable retry on failures
        checkpoint_path: Path to save checkpoints
        resume_from: Optional checkpoint to resume from

    Returns:
        Dict of config_key -> ratings dict
    """
    baselines = baselines or DEFAULT_BASELINES
    tournament_id = resume_from.tournament_id if resume_from else f"massive_{uuid.uuid4().hex[:8]}"

    # Setup checkpoint
    if checkpoint_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint_path = RESULTS_DIR / f"tournament_{tournament_id}.json"

    # Initialize or resume checkpoint
    if resume_from:
        checkpoint = resume_from
        logger.info(f"Resuming tournament {tournament_id}")
        logger.info(f"  Completed configs: {checkpoint.configs_completed}")
        logger.info(f"  Pending configs: {checkpoint.configs_pending}")
    else:
        checkpoint = TournamentCheckpoint(
            tournament_id=tournament_id,
            source_path=str(source_path),
            started_at=time.time(),
        )

    # Phase 1: Deduplicate models (or reload from checkpoint)
    deduplicator = ModelDeduplicator()

    if resume_from and resume_from.model_registry:
        # Reconstruct models from checkpoint
        logger.info("Reconstructing models from checkpoint...")
        unique_models = [
            UniqueModel.from_dict(data)
            for data in resume_from.model_registry.values()
        ]
        logger.info(f"Loaded {len(unique_models)} models from checkpoint")
    else:
        logger.info(f"Scanning {source_path} for models...")
        unique_models = await deduplicator.scan_directory(source_path)
        logger.info(f"Found {len(unique_models)} unique models (deduped)")

        # Save to checkpoint
        for model in unique_models:
            checkpoint.model_registry[model.sha256] = model.to_dict()
        checkpoint.total_models = len(unique_models)

    # Phase 2: Group by config
    models_by_config = deduplicator.group_by_config(unique_models)

    # Filter if specified
    if config_filter:
        if config_filter in models_by_config:
            models_by_config = {config_filter: models_by_config[config_filter]}
        else:
            logger.error(f"Config {config_filter} not found. Available: {list(models_by_config.keys())}")
            return {}

    # Skip completed configs
    if resume_from:
        for completed in checkpoint.configs_completed:
            if completed in models_by_config:
                logger.info(f"Skipping completed config: {completed}")
                del models_by_config[completed]

    checkpoint.configs_pending = list(models_by_config.keys())
    checkpoint.save(checkpoint_path)

    # Calculate target games
    total_models = sum(len(m) for m in models_by_config.values())
    games_per_model = len(baselines) * games_per_pairing
    checkpoint.total_games_target = total_models * games_per_model
    logger.info(f"Target: {checkpoint.total_games_target} games ({total_models} models × {games_per_model} games each)")

    # Phase 3: Run tournament per config
    all_results: dict[str, dict[str, float]] = {}

    for config_key, models in models_by_config.items():
        checkpoint.current_config = config_key
        logger.info(f"\n{'='*60}")
        logger.info(f"Starting config: {config_key} ({len(models)} models)")
        logger.info(f"{'='*60}")

        board_type, num_players = parse_config_key(config_key)

        # Register models as agents
        agents = []
        for model in models:
            agent_id = f"model_{model.sha256[:12]}"
            AI_TYPE_CONFIGS[agent_id] = {
                "ai_type": "descent",
                "model_path": str(model.canonical_path),
            }
            agents.append(agent_id)

        # Add baselines
        agents.extend(baselines)

        logger.info(f"Running tournament: {len(models)} models + {len(baselines)} baselines")
        logger.info(f"Matchups: {len(models) * len(baselines)} model-baseline pairs")
        logger.info(f"Games: {len(models) * len(baselines) * games_per_pairing} total")

        # Run distributed tournament (uses existing parallelism/retry/sharding)
        try:
            results, ratings = run_distributed_tournament(
                agents=agents,
                games_per_pairing=games_per_pairing,
                board_type=board_type,
                num_players=num_players,
                retry_failed=not no_retry,
                max_parallel_per_node=max_parallel,
            )

            # Store results
            all_results[config_key] = ratings
            checkpoint.config_results[config_key] = {
                "ratings": ratings,
                "games_played": len(results),
                "models_evaluated": len(models),
            }
            checkpoint.total_games_played += len(results)

            # Print results
            print_config_results(config_key, ratings, models, len(results))

        except Exception as e:
            logger.error(f"Config {config_key} failed: {e}")
            checkpoint.config_results[config_key] = {"error": str(e)}

        # Mark config completed
        checkpoint.configs_completed.append(config_key)
        if config_key in checkpoint.configs_pending:
            checkpoint.configs_pending.remove(config_key)
        checkpoint.save(checkpoint_path)

    # Final summary
    elapsed = time.time() - checkpoint.started_at
    logger.info(f"\n{'='*60}")
    logger.info("TOURNAMENT COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Duration: {elapsed/3600:.1f} hours")
    logger.info(f"Games played: {checkpoint.total_games_played}")
    logger.info(f"Configs completed: {len(checkpoint.configs_completed)}")
    logger.info(f"Checkpoint: {checkpoint_path}")

    return all_results


async def main():
    parser = argparse.ArgumentParser(
        description="Run massive tournament on model archive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run - scan and report
  python scripts/run_massive_tournament.py --source /Volumes/RingRift-Data --dry-run

  # Full tournament
  python scripts/run_massive_tournament.py --source /Volumes/RingRift-Data --games 25

  # Single config
  python scripts/run_massive_tournament.py --source /Volumes/RingRift-Data --config hex8_2p

  # Resume interrupted
  python scripts/run_massive_tournament.py --resume results/massive_tournament/tournament_abc123.json
        """,
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="Path to model archive (e.g., /Volumes/RingRift-Data)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Specific config to evaluate (e.g., hex8_2p)",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=25,
        help="Games per (model, opponent) pair (default: 25)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report without running tournament",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=2,
        help="Max parallel matches per node (default: 2)",
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Disable retry on failures",
    )
    parser.add_argument(
        "--baselines",
        type=str,
        default="random,heuristic,mcts_100,mcts_200",
        help="Comma-separated baseline opponents",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        help="Resume from checkpoint file",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show cluster status and exit",
    )

    args = parser.parse_args()

    # Status check
    if args.status:
        nodes, all_hosts = discover_healthy_nodes()
        print(f"\nCluster: {len(nodes)} healthy nodes out of {len(all_hosts)} total")
        return

    # Resume mode
    if args.resume:
        if not args.resume.exists():
            print(f"Error: Checkpoint not found: {args.resume}")
            sys.exit(1)
        checkpoint = TournamentCheckpoint.load(args.resume)
        source_path = Path(checkpoint.source_path)
        print(f"Resuming tournament {checkpoint.tournament_id}")
        print(f"  Source: {source_path}")
        print(f"  Progress: {len(checkpoint.configs_completed)}/{len(checkpoint.configs_completed) + len(checkpoint.configs_pending)} configs")
    else:
        if not args.source:
            print("Error: --source required (or use --resume)")
            sys.exit(1)
        source_path = args.source
        checkpoint = None

        if not source_path.exists():
            print(f"Error: Source path not found: {source_path}")
            sys.exit(1)

    # Dry run - just scan and report
    if args.dry_run:
        deduplicator = ModelDeduplicator()
        print(f"Scanning {source_path}...")
        models = await deduplicator.scan_directory(source_path)
        print_dedup_report(models, deduplicator)
        return

    # Parse baselines
    baselines = [b.strip() for b in args.baselines.split(",")]
    for b in baselines:
        if b not in AI_TYPE_CONFIGS and b not in AI_TYPE_CONFIGS_LIGHTWEIGHT:
            print(f"Warning: Unknown baseline '{b}', will use AI_TYPE_CONFIGS")

    # Run tournament
    results = await run_massive_tournament(
        source_path=source_path,
        games_per_pairing=args.games,
        config_filter=args.config,
        baselines=baselines,
        max_parallel=args.max_parallel,
        no_retry=args.no_retry,
        resume_from=checkpoint,
    )

    if results:
        print(f"\nTournament complete! {len(results)} configs evaluated.")
    else:
        print("\nNo results collected.")


if __name__ == "__main__":
    asyncio.run(main())
