"""High-tier training configuration for 2000+ Elo models.

This module provides configuration defaults optimized for training models
that exceed 2000 Elo across all 12 board/player configurations.

Key changes from default training:
1. Gumbel MCTS as default selfplay engine (5-50x speedup, better policy targets)
2. Multi-config curriculum for balanced training across all 12 configs
3. Crossboard promotion thresholds enforced
4. Vector value head for multi-player games

December 2025: Created for 2000+ Elo target across all configurations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.models import BoardType
from app.training.selfplay_config import EngineMode, SelfplayConfig

logger = logging.getLogger(__name__)

# All 12 board/player configurations
ALL_CONFIGS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hex8", 2), ("hex8", 3), ("hex8", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

# 9-config subset (3 boards x 3 player counts) used in some pipelines
CANONICAL_CONFIGS = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]

# Target Elo thresholds by tier
TIER_ELO_TARGETS = {
    "D6": 1600.0,
    "D7": 1800.0,
    "D8": 2000.0,
    "D9": 2200.0,
    "D10": 2400.0,
}

# Default tier for high-tier training
DEFAULT_TARGET_TIER = "D8"
DEFAULT_TARGET_ELO = 2000.0


@dataclass
class HighTierSelfplayConfig:
    """Selfplay configuration optimized for high-tier (2000+ Elo) training.

    Uses Gumbel MCTS by default for sample-efficient search and GPU acceleration.
    Generates data across all configurations to ensure balanced training.
    """

    # Engine settings - Gumbel MCTS by default
    engine_mode: EngineMode = EngineMode.GUMBEL_MCTS
    simulation_budget: int = 150  # Gumbel simulations per move
    temperature: float = 1.0  # Exploration temperature
    temperature_threshold: int = 30  # Move after which to use greedy

    # Multi-config settings
    all_configs: bool = True  # Generate for all 12 configs by default
    config_weights: dict[str, float] = field(default_factory=dict)
    round_robin: bool = True  # Rotate through configs

    # Quality settings
    record_mcts_distribution: bool = True  # Soft policy targets
    validate_parity: bool = True  # Canonical spec compliance

    # GPU settings
    use_gpu: bool = True
    batch_neural_net: bool = True  # Batch NN calls for GPU efficiency

    # Output
    output_format: str = "db"  # Database for efficient storage
    games_per_config: int = 1000  # Per-config target

    def to_selfplay_config(
        self,
        board_type: str = "square8",
        num_players: int = 2,
    ) -> SelfplayConfig:
        """Convert to standard SelfplayConfig."""
        return SelfplayConfig(
            board_type=board_type,
            num_players=num_players,
            engine_mode=self.engine_mode,
            mcts_simulations=self.simulation_budget,
            temperature=self.temperature,
            temperature_threshold=self.temperature_threshold,
            use_gpu=self.use_gpu,
            num_games=self.games_per_config,
        )


@dataclass
class HighTierTrainingConfig:
    """Training configuration for 2000+ Elo models.

    Combines selfplay, curriculum, and promotion settings for high-tier training.
    """

    # Target tier
    target_tier: str = DEFAULT_TARGET_TIER
    target_elo: float = DEFAULT_TARGET_ELO

    # Selfplay settings
    selfplay: HighTierSelfplayConfig = field(default_factory=HighTierSelfplayConfig)

    # Curriculum settings
    use_curriculum: bool = True
    curriculum_stages: list[str] = field(
        default_factory=lambda: ["D6", "D7", "D8"]
    )

    # Multi-config training
    train_all_configs: bool = True
    config_batch_size: int = 3  # Train 3 configs in parallel

    # Value head
    use_vector_value: bool = True  # Vector head for multi-player

    # Promotion settings
    crossboard_promotion: bool = True  # Require 2000+ on all configs
    allow_partial_promotion: bool = False  # No partial promotions
    auto_promote: bool = True  # Auto-update ladder on success

    # Quality gates
    require_parity_validation: bool = True
    min_games_per_config: int = 200  # For Elo estimation


def get_high_tier_selfplay_config() -> HighTierSelfplayConfig:
    """Get default high-tier selfplay configuration.

    Returns:
        HighTierSelfplayConfig with Gumbel MCTS defaults
    """
    return HighTierSelfplayConfig()


def get_high_tier_training_config(
    target_tier: str = "D8",
    target_elo: float | None = None,
) -> HighTierTrainingConfig:
    """Get high-tier training configuration.

    Args:
        target_tier: Target tier (D6-D10)
        target_elo: Optional explicit Elo target

    Returns:
        HighTierTrainingConfig for specified tier
    """
    if target_elo is None:
        target_elo = TIER_ELO_TARGETS.get(target_tier, DEFAULT_TARGET_ELO)

    return HighTierTrainingConfig(
        target_tier=target_tier,
        target_elo=target_elo,
    )


def create_multi_config_selfplay_args(
    config: HighTierSelfplayConfig,
    output_dir: str,
) -> list[list[str]]:
    """Create argument lists for multi-config selfplay generation.

    Args:
        config: High-tier selfplay configuration
        output_dir: Output directory for generated data

    Returns:
        List of argument lists, one per configuration
    """
    all_args = []

    configs = ALL_CONFIGS if config.all_configs else CANONICAL_CONFIGS

    for board_type, num_players in configs:
        args = [
            "--board", board_type,
            "--num-players", str(num_players),
            "--num-games", str(config.games_per_config),
            "--simulation-budget", str(config.simulation_budget),
            "--temperature", str(config.temperature),
            f"--output-dir={output_dir}",
        ]

        if config.use_gpu:
            args.append("--use-gpu")

        if config.validate_parity:
            args.append("--validate-parity")

        all_args.append(args)

    return all_args


def should_use_gumbel_engine(tier: str) -> bool:
    """Check if Gumbel MCTS should be used for a tier.

    Returns True for D7+ tiers where search quality matters.
    """
    high_tiers = {"D7", "D8", "D9", "D10"}
    return tier in high_tiers


def get_engine_mode_for_tier(tier: str) -> EngineMode:
    """Get appropriate engine mode for a training tier.

    Args:
        tier: Training tier (D1-D10)

    Returns:
        Recommended EngineMode for the tier
    """
    if should_use_gumbel_engine(tier):
        return EngineMode.GUMBEL_MCTS
    elif tier in {"D5", "D6"}:
        return EngineMode.NN_DESCENT
    else:
        return EngineMode.NNUE_GUIDED


def get_crossboard_promotion_config(
    candidate_id: str,
    target_tier: str = "D8",
    target_elo: float = 2000.0,
) -> dict[str, Any]:
    """Get configuration for crossboard tier promotion.

    Args:
        candidate_id: Model identifier
        target_tier: Target tier for promotion
        target_elo: Target Elo threshold

    Returns:
        Configuration dict for crossboard_tier_orchestrator
    """
    return {
        "candidate_id": candidate_id,
        "target_tier": target_tier,
        "target_elo": target_elo,
        "configs": [f"{b}_{p}p" for b, p in CANONICAL_CONFIGS],
        "parallel": 3,
        "games_per_config": 200,
        "allow_partial": False,
        "auto_promote": True,
        "require_parity": True,
    }


__all__ = [
    'ALL_CONFIGS',
    'CANONICAL_CONFIGS',
    'DEFAULT_TARGET_ELO',
    'DEFAULT_TARGET_TIER',
    'HighTierSelfplayConfig',
    'HighTierTrainingConfig',
    'TIER_ELO_TARGETS',
    'create_multi_config_selfplay_args',
    'get_crossboard_promotion_config',
    'get_engine_mode_for_tier',
    'get_high_tier_selfplay_config',
    'get_high_tier_training_config',
    'should_use_gumbel_engine',
]
