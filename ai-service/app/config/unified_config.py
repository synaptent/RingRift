"""Unified Configuration Module for RingRift AI Self-Improvement System.

This module provides a SINGLE SOURCE OF TRUTH for all configuration values
used across the distributed training, evaluation, and promotion system.

All scripts should import config from this module instead of hardcoding values
or using scattered environment variables.

Usage:
    from app.config.unified_config import get_config, UnifiedConfig

    config = get_config()  # Loads from config/unified_loop.yaml

    # Access training thresholds
    threshold = config.training.trigger_threshold_games

    # Access evaluation settings
    shadow_games = config.evaluation.shadow_games_per_config

    # Access all 9 board configurations
    for board_config in config.get_all_board_configs():
        print(f"{board_config.board_type}_{board_config.num_players}p")

Environment Variable Overrides:
    - RINGRIFT_CONFIG_PATH: Override config file path
    - RINGRIFT_TRAINING_THRESHOLD: Override training trigger threshold
    - RINGRIFT_ELO_DB: Override Elo database path
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import yaml

# Import canonical threshold constants
try:
    from app.config.thresholds import (
        INITIAL_ELO_RATING,
        ELO_K_FACTOR,
        ELO_DROP_ROLLBACK,
        MIN_GAMES_FOR_ELO,
        TRAINING_TRIGGER_GAMES,
        TRAINING_MIN_INTERVAL_SECONDS,
    )
except ImportError:
    INITIAL_ELO_RATING = 1500.0

# Import centralized path utilities
try:
    from app.utils.paths import AI_SERVICE_ROOT, ensure_dir, ensure_parent_dir
except ImportError:
    AI_SERVICE_ROOT = Path(__file__).parent.parent.parent
    def ensure_dir(path: Path) -> Path:
        path.mkdir(parents=True, exist_ok=True)
        return path
    def ensure_parent_dir(path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    ELO_K_FACTOR = 32
    ELO_DROP_ROLLBACK = 50.0
    MIN_GAMES_FOR_ELO = 30
    TRAINING_TRIGGER_GAMES = 500
    TRAINING_MIN_INTERVAL_SECONDS = 1200

logger = logging.getLogger(__name__)

# Default config path relative to ai-service root
DEFAULT_CONFIG_PATH = "config/unified_loop.yaml"

# Singleton instance
_config_instance: Optional[UnifiedConfig] = None


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion from remote hosts."""
    poll_interval_seconds: int = 60
    ephemeral_poll_interval_seconds: int = 15  # Aggressive sync for RAM disk hosts
    sync_method: str = "incremental"
    deduplication: bool = True
    min_games_per_sync: int = 5
    remote_db_pattern: str = "data/games/*.db"
    remote_selfplay_patterns: List[str] = field(default_factory=lambda: [
        "data/selfplay/gpu_*/games*.jsonl",
        "data/selfplay/p2p_gpu/*/games*.jsonl",
        "data/games/gpu_selfplay/*/games*.jsonl",
    ])
    # sync_disabled: When true, data sync is disabled on this machine (orchestrator-only mode)
    # Set to true on machines with limited disk space that shouldn't collect data locally
    sync_disabled: bool = False
    use_external_sync: bool = False  # Whether to use external unified_data_sync.py
    checksum_validation: bool = True
    retry_max_attempts: int = 3
    retry_base_delay_seconds: int = 5
    dead_letter_enabled: bool = True
    wal_enabled: bool = True
    wal_db_path: str = "data/sync_wal.db"
    elo_replication_enabled: bool = True
    elo_replication_interval_seconds: int = 60


@dataclass
class TrainingConfig:
    """Configuration for automatic training triggers.

    This is THE SINGLE SOURCE OF TRUTH for training thresholds.
    Do not hardcode these values elsewhere.
    """
    trigger_threshold_games: int = 500  # Canonical threshold
    min_interval_seconds: int = 1200  # 20 minutes
    max_concurrent_jobs: int = 1
    prefer_gpu_hosts: bool = True
    nn_training_script: str = "scripts/run_nn_training_baseline.py"
    export_script: str = "scripts/export_replay_dataset.py"
    hex_encoder_version: str = "v3"
    warm_start: bool = True
    validation_split: float = 0.1

    # NNUE-specific thresholds (higher requirements)
    nnue_min_games: int = 10000
    nnue_policy_min_games: int = 5000
    cmaes_min_games: int = 20000


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation (shadow + full tournaments)."""
    shadow_interval_seconds: int = 900  # 15 minutes
    shadow_games_per_config: int = 15
    full_tournament_interval_seconds: int = 3600  # 1 hour
    full_tournament_games: int = 50
    baseline_models: List[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100", "mcts_500"])
    min_games_for_elo: int = 30
    elo_k_factor: int = 32


@dataclass
class PromotionConfig:
    """Configuration for automatic model promotion."""
    auto_promote: bool = True
    elo_threshold: int = 25
    min_games: int = 50
    significance_level: float = 0.05
    sync_to_cluster: bool = True
    cooldown_seconds: int = 1800  # 30 minutes
    max_promotions_per_day: int = 10
    regression_test: bool = True

    # Aliases for compatibility with PromotionCriteria
    @property
    def min_elo_improvement(self) -> float:
        return float(self.elo_threshold)

    @property
    def min_win_rate(self) -> float:
        return 0.52  # Default win rate threshold

    @property
    def statistical_significance(self) -> float:
        return self.significance_level


@dataclass
class CurriculumConfig:
    """Configuration for adaptive curriculum (Elo-weighted training)."""
    adaptive: bool = True
    rebalance_interval_seconds: int = 3600  # 1 hour
    max_weight_multiplier: float = 1.5
    min_weight_multiplier: float = 0.7
    ema_alpha: float = 0.3
    min_games_for_weight: int = 100

    # Event-driven rebalancing (new)
    rebalance_on_elo_change: bool = True
    elo_change_threshold: int = 50  # Trigger rebalance on 50+ Elo change


@dataclass
class SafeguardsConfig:
    """Process safeguards to prevent uncoordinated sprawl."""
    max_python_processes_per_host: int = 20
    max_selfplay_processes: int = 2
    max_tournament_processes: int = 1
    max_training_processes: int = 1
    single_orchestrator: bool = True
    orchestrator_host: str = "lambda-h100"
    kill_orphans_on_start: bool = True
    process_watchdog: bool = True
    watchdog_interval_seconds: int = 60
    max_process_age_hours: int = 4
    max_subprocess_depth: int = 2
    subprocess_timeout_seconds: int = 3600


@dataclass
class BoardConfig:
    """Configuration for a specific board type and player count."""
    board_type: str
    num_players: int

    @property
    def config_key(self) -> str:
        return f"{self.board_type}_{self.num_players}p"


@dataclass
class RegressionConfig:
    """Configuration for regression testing before promotion."""
    hard_block: bool = True
    test_script: str = "scripts/run_regression_tests.py"
    timeout_seconds: int = 600


@dataclass
class AlertingConfig:
    """Alerting thresholds for monitoring."""
    sync_failure_threshold: int = 5
    training_timeout_hours: int = 4
    elo_drop_threshold: int = 50
    games_per_hour_min: int = 100


@dataclass
class SafetyConfig:
    """Safety thresholds to prevent bad models from being promoted."""
    overfit_threshold: float = 0.15  # Max gap between train/val loss
    min_memory_gb: int = 64  # Minimum RAM required to run unified loop
    max_consecutive_failures: int = 3  # Stop after N consecutive failures
    parity_failure_rate_max: float = 0.10  # Block training if parity failures exceed this
    data_quality_score_min: float = 0.70  # Minimum data quality to proceed


@dataclass
class PlateauDetectionConfig:
    """Plateau detection and automatic hyperparameter search."""
    elo_plateau_threshold: float = 15.0  # Elo gain below this triggers plateau detection
    elo_plateau_lookback: int = 5  # Number of evaluations to look back
    win_rate_degradation_threshold: float = 0.40  # Win rate below this triggers retraining
    plateau_count_for_cmaes: int = 2  # Trigger CMA-ES after this many consecutive plateaus
    plateau_count_for_nas: int = 4  # Trigger NAS after this many consecutive plateaus


@dataclass
class ReplayBufferConfig:
    """Prioritized experience replay buffer settings."""
    priority_alpha: float = 0.6  # Priority exponent
    importance_beta: float = 0.4  # Importance sampling exponent
    capacity: int = 100000  # Max experiences in buffer
    rebuild_interval_seconds: int = 7200  # Rebuild buffer every 2 hours


@dataclass
class ClusterConfig:
    """Cluster orchestration settings (previously hardcoded in cluster_orchestrator.py)."""
    # Target performance
    target_selfplay_games_per_hour: int = 1000  # Target selfplay rate across cluster

    # Health monitoring
    health_check_interval_seconds: int = 60  # Seconds between cluster health checks
    sync_interval_seconds: int = 300  # Seconds between data sync with cluster

    # Host sync intervals (in iterations, where 1 iteration ~= 5 minutes)
    sync_interval: int = 6  # Sync every 6 iterations (30 minutes)
    model_sync_interval: int = 12  # Sync models every 12 iterations (1 hour)
    model_sync_enabled: bool = True

    # Elo calibration
    elo_calibration_interval: int = 72  # Every 72 iterations (6 hours)
    elo_calibration_games: int = 50

    # Elo-driven curriculum learning
    elo_curriculum_enabled: bool = True
    elo_match_window: int = 200
    elo_underserved_threshold: int = 100

    # Auto-scaling
    auto_scale_interval: int = 12
    underutilized_cpu_threshold: int = 30
    underutilized_python_jobs: int = 10
    scale_up_games_per_host: int = 50

    # Adaptive game count
    adaptive_games_min: int = 30
    adaptive_games_max: int = 150


@dataclass
class SSHConfig:
    """SSH execution settings (shared across all orchestrators)."""
    max_retries: int = 3
    base_delay_seconds: float = 2.0
    max_delay_seconds: float = 30.0
    connect_timeout_seconds: int = 10
    command_timeout_seconds: int = 3600  # 1 hour max
    # SSH transport specific (from ssh_transport.py)
    transport_command_timeout_seconds: int = 30  # For P2P commands via SSH
    retry_delay_seconds: float = 1.0
    address_cache_ttl_seconds: int = 300  # 5 minutes


@dataclass
class SlurmConfig:
    """Slurm execution settings for stable HPC clusters."""
    enabled: bool = False
    partition_training: str = "gpu-train"
    partition_selfplay: str = "gpu-selfplay"
    partition_tournament: str = "cpu-eval"
    account: Optional[str] = None
    qos: Optional[str] = None
    default_time_training: str = "08:00:00"
    default_time_selfplay: str = "02:00:00"
    default_time_tournament: str = "02:00:00"
    gpus_training: int = 1
    cpus_training: int = 16
    mem_training: str = "64G"
    gpus_selfplay: int = 0
    cpus_selfplay: int = 8
    mem_selfplay: str = "16G"
    gpus_tournament: int = 0
    cpus_tournament: int = 8
    mem_tournament: str = "16G"
    job_dir: str = "data/slurm/jobs"
    log_dir: str = "data/slurm/logs"
    shared_root: Optional[str] = None
    repo_subdir: str = "ai-service"
    venv_activate: Optional[str] = None
    setup_commands: List[str] = field(default_factory=list)
    extra_sbatch_args: List[str] = field(default_factory=list)
    poll_interval_seconds: int = 20


@dataclass
class DistributedConfig:
    """Configuration for distributed system components.

    Centralizes settings previously hardcoded across:
    - dynamic_registry.py
    - ssh_transport.py
    - unified_data_sync.py
    """
    # Node health state thresholds
    degraded_failure_threshold: int = 2   # Failures before node marked degraded
    offline_failure_threshold: int = 5    # Failures before node marked offline
    recovery_success_threshold: int = 2   # Successes needed to recover from degraded

    # API check intervals (seconds)
    vast_api_check_interval_seconds: int = 300   # 5 minutes
    aws_api_check_interval_seconds: int = 300    # 5 minutes
    tailscale_check_interval_seconds: int = 120  # 2 minutes

    # P2P orchestrator settings
    p2p_port: int = 8770
    p2p_base_url: str = "http://localhost:8770"

    # Gossip sync
    gossip_port: int = 8771

    # Data server (for aria2 transport)
    data_server_port: int = 8766


@dataclass
class SelfplayDefaults:
    """Default selfplay settings shared across all selfplay scripts.

    Note: This is for APPLICATION-LEVEL defaults loaded from config files.
    For per-run configuration, use :class:`app.training.selfplay_config.SelfplayConfig`.
    """
    # Game generation
    default_games_per_config: int = 50
    min_games_for_training: int = 500
    max_games_per_session: int = 1000

    # Worker management
    max_concurrent_workers: int = 4
    worker_timeout_seconds: int = 7200  # 2 hours
    checkpoint_interval_games: int = 100

    # Quality settings
    mcts_simulations: int = 200
    temperature: float = 0.5
    noise_fraction: float = 0.25


# Backward compatibility alias (deprecated December 2025)
# Use SelfplayDefaults for app-level defaults or
# app.training.selfplay_config.SelfplayConfig for per-run config
SelfplayConfig = SelfplayDefaults


@dataclass
class TournamentConfig:
    """Tournament settings (shared across all tournament scripts)."""
    # Default game counts
    default_games_per_matchup: int = 20
    shadow_games: int = 15
    full_tournament_games: int = 50

    # Time limits
    game_timeout_seconds: int = 300  # 5 minutes
    tournament_timeout_seconds: int = 7200  # 2 hours

    # Elo calculation
    k_factor: int = 32
    initial_elo: int = 1500
    min_games_for_rating: int = 30

    # Baseline models
    baseline_models: List[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100", "mcts_500"])


@dataclass
class IntegratedEnhancementsConfig:
    """Configuration for integrated training enhancements.

    Centralizes all advanced training feature flags and parameters.
    See app/training/integrated_enhancements.py for implementation.

    This is the CANONICAL location for this config. The scripts/unified_loop/config.py
    version should import from here.
    """
    # Master toggle
    enabled: bool = True

    # Auxiliary Tasks (Multi-Task Learning)
    auxiliary_tasks_enabled: bool = False
    aux_game_length_weight: float = 0.1
    aux_piece_count_weight: float = 0.1
    aux_outcome_weight: float = 0.05

    # Gradient Surgery (PCGrad)
    gradient_surgery_enabled: bool = False
    gradient_surgery_method: str = "pcgrad"  # "pcgrad" or "cagrad"
    gradient_conflict_threshold: float = 0.0  # Threshold for detecting conflicts

    # Batch Scheduling
    batch_scheduling_enabled: bool = False
    batch_initial_size: int = 64
    batch_final_size: int = 512
    batch_warmup_steps: int = 1000
    batch_rampup_steps: int = 10000
    batch_schedule_type: str = "linear"  # "linear", "exponential", "step"

    # Background Evaluation
    background_eval_enabled: bool = False
    eval_interval_steps: int = 1000
    eval_games_per_check: int = 20
    eval_elo_checkpoint_threshold: float = 10.0
    eval_elo_drop_threshold: float = ELO_DROP_ROLLBACK
    eval_auto_checkpoint: bool = True
    eval_checkpoint_dir: str = "data/eval_checkpoints"

    # ELO Weighting (uses thresholds.py constants)
    elo_weighting_enabled: bool = True
    elo_base_rating: float = INITIAL_ELO_RATING
    elo_weight_scale: float = 400.0
    elo_min_weight: float = 0.5
    elo_max_weight: float = 2.0

    # Curriculum Learning
    curriculum_enabled: bool = True
    curriculum_auto_advance: bool = True
    curriculum_checkpoint_path: str = "data/curriculum_state.json"

    # Data Augmentation
    augmentation_enabled: bool = True
    augmentation_mode: str = "all"  # "all", "random", "light"
    augmentation_probability: float = 1.0

    # Reanalysis
    reanalysis_enabled: bool = False
    reanalysis_blend_ratio: float = 0.5
    reanalysis_interval_steps: int = 5000
    reanalysis_batch_size: int = 1000


@dataclass
class HealthConfig:
    """Configuration for component health monitoring."""
    enabled: bool = True
    check_interval_seconds: int = 30
    # Thresholds for component health (seconds since last successful operation)
    data_collector_stale_threshold: int = 300  # 5 minutes
    evaluator_stale_threshold: int = 1800  # 30 minutes
    training_stale_threshold: int = 7200  # 2 hours (training can be slow)
    # Recovery settings
    auto_restart_on_failure: bool = True
    max_restart_attempts: int = 3
    restart_cooldown_seconds: int = 60
    # Alert settings
    alert_on_degraded: bool = True
    alert_on_unhealthy: bool = True


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = False  # Disabled by default - resource intensive
    population_size: int = 8
    exploit_interval_steps: int = 1000
    tunable_params: List[str] = field(default_factory=lambda: ["learning_rate", "batch_size", "temperature"])
    check_interval_seconds: int = 1800  # Check PBT status every 30 min
    auto_start: bool = False  # Auto-start PBT when training completes


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = False  # Disabled by default - very resource intensive
    strategy: str = "evolutionary"  # evolutionary, random, bayesian
    population_size: int = 20
    generations: int = 50
    check_interval_seconds: int = 3600  # Check NAS status every hour
    auto_start_on_plateau: bool = False  # Start NAS when Elo plateaus


@dataclass
class PERConfig:
    """Configuration for Prioritized Experience Replay.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = True  # Enabled by default - improves training efficiency
    alpha: float = 0.6  # Priority exponent
    beta: float = 0.4  # Importance sampling exponent
    buffer_capacity: int = 100000
    rebuild_interval_seconds: int = 7200  # Rebuild buffer every 2 hours


@dataclass
class DataLoadingConfig:
    """Configuration for data loading and streaming.

    Centralizes settings previously scattered across:
    - data_loader.py (StreamingDataLoader)
    - streaming_pipeline.py (StreamingDataPipeline)
    - datasets.py (RingRiftDataset)
    """
    # Policy encoding
    policy_size: int = 55000  # Policy vector size
    max_policy_size: int = 60000  # Maximum policy size for safety

    # Batch settings
    batch_size: int = 512
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True

    # Filtering
    filter_empty_policies: bool = True
    max_samples: Optional[int] = None

    # Streaming buffer
    buffer_size: int = 10000
    poll_interval_seconds: float = 5.0
    dedupe_window: int = 50000

    # Multi-source loading
    enable_multi_source: bool = True
    max_sources: int = 10

    # Phase-based weighting (consolidated from 3 locations)
    phase_weights: Dict[str, float] = field(default_factory=lambda: {
        "ring_placement": 0.8,
        "movement": 1.0,
        "capture": 1.2,
        "late_game": 1.5,
    })

    # Victory type weighting
    victory_type_weights: Dict[str, float] = field(default_factory=lambda: {
        "elimination": 1.0,
        "territory": 1.2,
        "ring_out": 0.8,
        "timeout": 0.5,
    })


@dataclass
class QualityConfig:
    """Configuration for game quality scoring and weighting.

    Consolidates quality computation from:
    - game_quality_scorer.py
    - quality_extractor.py
    - streaming_pipeline.py
    - unified_manifest.py

    This is the SINGLE SOURCE OF TRUTH for quality scoring parameters.
    """
    # Game quality component weights (must sum to 1.0)
    outcome_weight: float = 0.25
    length_weight: float = 0.25
    phase_balance_weight: float = 0.20
    diversity_weight: float = 0.15
    source_reputation_weight: float = 0.15

    # Sync priority weights (for quality_extractor.py compatibility)
    sync_elo_weight: float = 0.4
    sync_length_weight: float = 0.3
    sync_decisive_weight: float = 0.3

    # Elo normalization bounds
    min_elo: float = 1200.0
    max_elo: float = 2400.0
    default_elo: float = 1500.0

    # Game length normalization
    min_game_length: int = 10
    max_game_length: int = 200
    optimal_game_length: int = 80

    # Sampling weights
    quality_weight: float = 0.4
    recency_weight: float = 0.3
    priority_weight: float = 0.3

    # Recency decay
    recency_half_life_hours: float = 24.0

    # Thresholds
    min_quality_for_training: float = 0.3
    min_quality_for_priority_sync: float = 0.5
    high_quality_threshold: float = 0.7

    # Decisive outcome credit
    decisive_bonus: float = 1.0
    draw_credit: float = 0.3

    def get_sync_weights(self) -> Dict[str, float]:
        """Get weights for sync quality extraction."""
        return {
            "elo_weight": self.sync_elo_weight,
            "length_weight": self.sync_length_weight,
            "decisive_weight": self.sync_decisive_weight,
        }


@dataclass
class StoragePathsConfig:
    """Storage paths for a specific provider."""
    selfplay_games: str = "data/selfplay"
    model_checkpoints: str = "models/checkpoints"
    training_data: str = "data/training"
    elo_database: str = "data/unified_elo.db"
    sync_staging: str = "data/sync_staging"
    local_scratch: str = "/tmp/ringrift"
    nfs_base: Optional[str] = None  # NFS base path if applicable
    use_nfs_for_sync: bool = False
    skip_rsync_to_nfs_nodes: bool = False


@dataclass
class ProviderDetectionConfig:
    """Rules for detecting which storage provider to use."""
    check_path: str = ""
    hostname_patterns: List[str] = field(default_factory=list)
    os_type: Optional[str] = None


@dataclass
class StorageConfig:
    """Storage configuration with provider-specific paths."""
    default: StoragePathsConfig = field(default_factory=StoragePathsConfig)
    lambda_: StoragePathsConfig = field(default_factory=lambda: StoragePathsConfig(
        nfs_base="/lambda/nfs/RingRift",
        selfplay_games="/lambda/nfs/RingRift/selfplay",
        model_checkpoints="/lambda/nfs/RingRift/models",
        training_data="/lambda/nfs/RingRift/training_data",
        elo_database="/lambda/nfs/RingRift/elo/unified_elo.db",
        sync_staging="/lambda/nfs/RingRift/sync_staging",
        local_scratch="/tmp/ringrift",
        use_nfs_for_sync=True,
        skip_rsync_to_nfs_nodes=True,
    ))
    vast: StoragePathsConfig = field(default_factory=lambda: StoragePathsConfig(
        selfplay_games="/workspace/data/selfplay",
        model_checkpoints="/workspace/models",
        training_data="/workspace/data/training",
        elo_database="/workspace/data/unified_elo.db",
        sync_staging="/workspace/data/sync_staging",
        local_scratch="/dev/shm/ringrift",
        use_nfs_for_sync=False,
    ))
    mac: StoragePathsConfig = field(default_factory=lambda: StoragePathsConfig(
        local_scratch="/tmp/ringrift",
    ))
    # Provider detection rules
    provider_detection: Dict[str, ProviderDetectionConfig] = field(default_factory=dict)


@dataclass
class FeedbackConfig:
    """Configuration for pipeline feedback controller integration.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = True  # Enable closed-loop feedback
    # Performance-based training triggers
    elo_plateau_threshold: float = 15.0  # Elo gain below this triggers plateau detection
    elo_plateau_lookback: int = 5  # Number of evaluations to look back
    win_rate_degradation_threshold: float = 0.40  # Win rate below this triggers retraining
    # Data quality gates
    max_parity_failure_rate: float = 0.10  # Block training if parity failures exceed this
    min_data_quality_score: float = 0.70  # Minimum data quality to proceed with training
    # CMA-ES/NAS auto-trigger
    plateau_count_for_cmaes: int = 2  # Trigger CMA-ES after this many consecutive plateaus
    plateau_count_for_nas: int = 4  # Trigger NAS after this many consecutive plateaus


@dataclass
class P2PClusterConfig:
    """Configuration for P2P distributed cluster integration.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = False  # Enable P2P cluster coordination
    p2p_base_url: str = "http://localhost:8770"  # P2P orchestrator URL
    auth_token: Optional[str] = None  # Auth token (defaults to RINGRIFT_CLUSTER_AUTH_TOKEN env)
    model_sync_enabled: bool = True  # Auto-sync models to cluster
    model_sync_on_promotion: bool = True  # Auto-sync when model is promoted
    target_selfplay_games_per_hour: int = 1000  # Target selfplay rate across cluster
    auto_scale_selfplay: bool = True  # Auto-scale selfplay workers
    use_distributed_tournament: bool = True  # Use cluster for tournament evaluation
    tournament_nodes_per_eval: int = 3  # Number of nodes per evaluation
    health_check_interval: int = 60  # Seconds between cluster health checks
    unhealthy_threshold: int = 3  # Failures before marking unhealthy
    sync_interval_seconds: int = 300  # Seconds between data sync with cluster
    # Gossip sync integration (P2P data replication)
    gossip_sync_enabled: bool = True  # Gossip-based data replication
    gossip_port: int = 8771  # Port for gossip protocol


@dataclass
class ModelPruningConfig:
    """Configuration for automated model pruning/evaluation.

    Migrated from scripts/unified_loop/config.py for consolidation.
    """
    enabled: bool = True  # Enable automatic model pruning
    threshold: int = 100  # Trigger pruning when model count exceeds this
    check_interval_seconds: int = 3600  # Check model count every hour
    top_quartile_keep: float = 0.25  # Keep top 25% of models
    games_per_baseline: int = 10  # Games per baseline for evaluation
    parallel_workers: int = 50  # Parallel evaluation workers
    archive_models: bool = True  # Archive pruned models instead of deleting
    prefer_high_cpu_hosts: bool = True  # Schedule on high-CPU hosts
    evaluation_timeout_seconds: int = 7200  # 2 hour timeout
    dry_run: bool = False  # If true, log but don't prune
    use_elo_based_culling: bool = True  # Use fast ELO-based culling instead of games


@dataclass
class UnifiedConfig:
    """Master configuration class that loads from unified_loop.yaml.

    This is the SINGLE SOURCE OF TRUTH for all configuration values.
    """
    version: str = "1.2"  # Bumped for config consolidation
    execution_backend: str = "auto"

    # Sub-configurations
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    safeguards: SafeguardsConfig = field(default_factory=SafeguardsConfig)
    regression: RegressionConfig = field(default_factory=RegressionConfig)
    alerting: AlertingConfig = field(default_factory=AlertingConfig)
    health: HealthConfig = field(default_factory=HealthConfig)

    # Safety and quality thresholds
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    plateau_detection: PlateauDetectionConfig = field(default_factory=PlateauDetectionConfig)
    replay_buffer: ReplayBufferConfig = field(default_factory=ReplayBufferConfig)

    # New unified sections (previously scattered as hardcoded constants)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    ssh: SSHConfig = field(default_factory=SSHConfig)
    slurm: SlurmConfig = field(default_factory=SlurmConfig)
    selfplay: SelfplayDefaults = field(default_factory=SelfplayDefaults)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)

    # Integrated training enhancements (December 2025)
    enhancements: IntegratedEnhancementsConfig = field(default_factory=IntegratedEnhancementsConfig)

    # Advanced training features (migrated from scripts/unified_loop/config.py)
    pbt: PBTConfig = field(default_factory=PBTConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    per: PERConfig = field(default_factory=PERConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    p2p: P2PClusterConfig = field(default_factory=P2PClusterConfig)
    model_pruning: ModelPruningConfig = field(default_factory=ModelPruningConfig)

    # Storage configuration (provider-specific paths)
    storage: StorageConfig = field(default_factory=StorageConfig)

    # Data loading configuration (December 2025)
    data_loading: DataLoadingConfig = field(default_factory=DataLoadingConfig)

    # Quality scoring configuration (December 2025)
    quality: QualityConfig = field(default_factory=QualityConfig)

    # Paths
    hosts_config_path: str = "config/distributed_hosts.yaml"
    elo_db: str = "data/unified_elo.db"
    data_manifest_db: str = "data/data_manifest.db"
    log_dir: str = "logs/unified_loop"

    # Metrics
    metrics_enabled: bool = True
    metrics_port: int = 9090

    # Board configurations
    _board_configs: List[Dict[str, Any]] = field(default_factory=list)

    # Source file for debugging
    _source_path: Optional[str] = None

    @classmethod
    def from_yaml(cls, path: str | Path) -> UnifiedConfig:
        """Load configuration from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning(f"Config file not found: {path}, using defaults")
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls._from_dict(data)
        config._source_path = str(path)
        return config

    @classmethod
    def _from_dict(cls, data: Dict[str, Any]) -> UnifiedConfig:
        """Create config from dictionary."""
        config = cls()

        # Load version
        config.version = data.get("version", config.version)
        config.execution_backend = data.get("execution_backend", config.execution_backend)

        # Load sub-configurations
        if "data_ingestion" in data:
            config.data_ingestion = DataIngestionConfig(**data["data_ingestion"])

        if "training" in data:
            training_data = data["training"]
            config.training = TrainingConfig(
                trigger_threshold_games=training_data.get("trigger_threshold_games", 500),
                min_interval_seconds=training_data.get("min_interval_seconds", 1200),
                max_concurrent_jobs=training_data.get("max_concurrent_jobs", 1),
                prefer_gpu_hosts=training_data.get("prefer_gpu_hosts", True),
                nn_training_script=training_data.get("nn_training_script", "scripts/run_nn_training_baseline.py"),
                export_script=training_data.get("export_script", "scripts/export_replay_dataset.py"),
                hex_encoder_version=training_data.get("hex_encoder_version", "v3"),
                warm_start=training_data.get("warm_start", True),
                validation_split=training_data.get("validation_split", 0.1),
            )

        if "evaluation" in data:
            eval_data = data["evaluation"]
            config.evaluation = EvaluationConfig(
                shadow_interval_seconds=eval_data.get("shadow_interval_seconds", 900),
                shadow_games_per_config=eval_data.get("shadow_games_per_config", 15),
                full_tournament_interval_seconds=eval_data.get("full_tournament_interval_seconds", 3600),
                full_tournament_games=eval_data.get("full_tournament_games", 50),
                baseline_models=eval_data.get("baseline_models", ["random", "heuristic", "mcts_100", "mcts_500"]),
                min_games_for_elo=eval_data.get("min_games_for_elo", 30),
                elo_k_factor=eval_data.get("elo_k_factor", 32),
            )

        if "promotion" in data:
            promo_data = data["promotion"]
            config.promotion = PromotionConfig(
                auto_promote=promo_data.get("auto_promote", True),
                elo_threshold=promo_data.get("elo_threshold", 25),
                min_games=promo_data.get("min_games", 50),
                significance_level=promo_data.get("significance_level", 0.05),
                sync_to_cluster=promo_data.get("sync_to_cluster", True),
                cooldown_seconds=promo_data.get("cooldown_seconds", 1800),
                max_promotions_per_day=promo_data.get("max_promotions_per_day", 10),
                regression_test=promo_data.get("regression_test", True),
            )

        if "curriculum" in data:
            curr_data = data["curriculum"]
            config.curriculum = CurriculumConfig(
                adaptive=curr_data.get("adaptive", True),
                rebalance_interval_seconds=curr_data.get("rebalance_interval_seconds", 3600),
                max_weight_multiplier=curr_data.get("max_weight_multiplier", 1.5),
                min_weight_multiplier=curr_data.get("min_weight_multiplier", 0.7),
                ema_alpha=curr_data.get("ema_alpha", 0.3),
                min_games_for_weight=curr_data.get("min_games_for_weight", 100),
            )

        if "safeguards" in data:
            safe_data = data["safeguards"]
            config.safeguards = SafeguardsConfig(
                max_python_processes_per_host=safe_data.get("max_python_processes_per_host", 20),
                max_selfplay_processes=safe_data.get("max_selfplay_processes", 2),
                max_tournament_processes=safe_data.get("max_tournament_processes", 1),
                max_training_processes=safe_data.get("max_training_processes", 1),
                single_orchestrator=safe_data.get("single_orchestrator", True),
                orchestrator_host=safe_data.get("orchestrator_host", "lambda-h100"),
                kill_orphans_on_start=safe_data.get("kill_orphans_on_start", True),
                process_watchdog=safe_data.get("process_watchdog", True),
                watchdog_interval_seconds=safe_data.get("watchdog_interval_seconds", 60),
                max_process_age_hours=safe_data.get("max_process_age_hours", 4),
                max_subprocess_depth=safe_data.get("max_subprocess_depth", 2),
                subprocess_timeout_seconds=safe_data.get("subprocess_timeout_seconds", 3600),
            )

        if "regression" in data:
            reg_data = data["regression"]
            config.regression = RegressionConfig(
                hard_block=reg_data.get("hard_block", True),
                test_script=reg_data.get("test_script", "scripts/run_regression_tests.py"),
                timeout_seconds=reg_data.get("timeout_seconds", 600),
            )

        if "alerting" in data:
            alert_data = data["alerting"]
            config.alerting = AlertingConfig(
                sync_failure_threshold=alert_data.get("sync_failure_threshold", 5),
                training_timeout_hours=alert_data.get("training_timeout_hours", 4),
                elo_drop_threshold=alert_data.get("elo_drop_threshold", 50),
                games_per_hour_min=alert_data.get("games_per_hour_min", 100),
            )

        # Load safety and quality thresholds
        if "safety" in data:
            safety_data = data["safety"]
            config.safety = SafetyConfig(
                overfit_threshold=safety_data.get("overfit_threshold", 0.15),
                min_memory_gb=safety_data.get("min_memory_gb", 64),
                max_consecutive_failures=safety_data.get("max_consecutive_failures", 3),
                parity_failure_rate_max=safety_data.get("parity_failure_rate_max", 0.10),
                data_quality_score_min=safety_data.get("data_quality_score_min", 0.70),
            )

        if "plateau_detection" in data:
            plateau_data = data["plateau_detection"]
            config.plateau_detection = PlateauDetectionConfig(
                elo_plateau_threshold=plateau_data.get("elo_plateau_threshold", 15.0),
                elo_plateau_lookback=plateau_data.get("elo_plateau_lookback", 5),
                win_rate_degradation_threshold=plateau_data.get("win_rate_degradation_threshold", 0.40),
                plateau_count_for_cmaes=plateau_data.get("plateau_count_for_cmaes", 2),
                plateau_count_for_nas=plateau_data.get("plateau_count_for_nas", 4),
            )

        if "replay_buffer" in data:
            rb_data = data["replay_buffer"]
            config.replay_buffer = ReplayBufferConfig(
                priority_alpha=rb_data.get("priority_alpha", 0.6),
                importance_beta=rb_data.get("importance_beta", 0.4),
                capacity=rb_data.get("capacity", 100000),
                rebuild_interval_seconds=rb_data.get("rebuild_interval_seconds", 7200),
            )

        # Load new unified sections (previously hardcoded constants)
        if "cluster" in data:
            cluster_data = data["cluster"]
            config.cluster = ClusterConfig(
                target_selfplay_games_per_hour=cluster_data.get("target_selfplay_games_per_hour", 1000),
                health_check_interval_seconds=cluster_data.get("health_check_interval_seconds", 60),
                sync_interval_seconds=cluster_data.get("sync_interval_seconds", 300),
                sync_interval=cluster_data.get("sync_interval", 6),
                model_sync_interval=cluster_data.get("model_sync_interval", 12),
                model_sync_enabled=cluster_data.get("model_sync_enabled", True),
                elo_calibration_interval=cluster_data.get("elo_calibration_interval", 72),
                elo_calibration_games=cluster_data.get("elo_calibration_games", 50),
                elo_curriculum_enabled=cluster_data.get("elo_curriculum_enabled", True),
                elo_match_window=cluster_data.get("elo_match_window", 200),
                elo_underserved_threshold=cluster_data.get("elo_underserved_threshold", 100),
                auto_scale_interval=cluster_data.get("auto_scale_interval", 12),
                underutilized_cpu_threshold=cluster_data.get("underutilized_cpu_threshold", 30),
                underutilized_python_jobs=cluster_data.get("underutilized_python_jobs", 10),
                scale_up_games_per_host=cluster_data.get("scale_up_games_per_host", 50),
                adaptive_games_min=cluster_data.get("adaptive_games_min", 30),
                adaptive_games_max=cluster_data.get("adaptive_games_max", 150),
            )

        if "ssh" in data:
            ssh_data = data["ssh"]
            config.ssh = SSHConfig(
                max_retries=ssh_data.get("max_retries", 3),
                base_delay_seconds=ssh_data.get("base_delay_seconds", 2.0),
                max_delay_seconds=ssh_data.get("max_delay_seconds", 30.0),
                connect_timeout_seconds=ssh_data.get("connect_timeout_seconds", 10),
                command_timeout_seconds=ssh_data.get("command_timeout_seconds", 3600),
            )

        if "slurm" in data:
            config.slurm = SlurmConfig(**data["slurm"])

        if "selfplay" in data:
            sp_data = data["selfplay"]
            config.selfplay = SelfplayDefaults(
                default_games_per_config=sp_data.get("default_games_per_config", 50),
                min_games_for_training=sp_data.get("min_games_for_training", 500),
                max_games_per_session=sp_data.get("max_games_per_session", 1000),
                max_concurrent_workers=sp_data.get("max_concurrent_workers", 4),
                worker_timeout_seconds=sp_data.get("worker_timeout_seconds", 7200),
                checkpoint_interval_games=sp_data.get("checkpoint_interval_games", 100),
                mcts_simulations=sp_data.get("mcts_simulations", 200),
                temperature=sp_data.get("temperature", 0.5),
                noise_fraction=sp_data.get("noise_fraction", 0.25),
            )

        if "tournament" in data:
            tourn_data = data["tournament"]
            config.tournament = TournamentConfig(
                default_games_per_matchup=tourn_data.get("default_games_per_matchup", 20),
                shadow_games=tourn_data.get("shadow_games", 15),
                full_tournament_games=tourn_data.get("full_tournament_games", 50),
                game_timeout_seconds=tourn_data.get("game_timeout_seconds", 300),
                tournament_timeout_seconds=tourn_data.get("tournament_timeout_seconds", 7200),
                k_factor=tourn_data.get("k_factor", 32),
                initial_elo=tourn_data.get("initial_elo", 1500),
                min_games_for_rating=tourn_data.get("min_games_for_rating", 30),
                baseline_models=tourn_data.get("baseline_models", ["random", "heuristic", "mcts_100", "mcts_500"]),
            )

        # Load advanced training feature configs (migrated from scripts/unified_loop/config.py)
        if "pbt" in data:
            config.pbt = PBTConfig(**data["pbt"])

        if "nas" in data:
            config.nas = NASConfig(**data["nas"])

        if "per" in data:
            config.per = PERConfig(**data["per"])

        if "feedback" in data:
            config.feedback = FeedbackConfig(**data["feedback"])

        if "p2p" in data:
            config.p2p = P2PClusterConfig(**data["p2p"])

        if "model_pruning" in data:
            config.model_pruning = ModelPruningConfig(**data["model_pruning"])

        # Load storage configuration
        if "storage" in data:
            storage_data = data["storage"]
            storage_config = StorageConfig()

            # Load default paths
            if "default" in storage_data:
                storage_config.default = StoragePathsConfig(**storage_data["default"])

            # Load Lambda paths
            if "lambda" in storage_data:
                storage_config.lambda_ = StoragePathsConfig(**storage_data["lambda"])

            # Load Vast paths
            if "vast" in storage_data:
                storage_config.vast = StoragePathsConfig(**storage_data["vast"])

            # Load Mac paths
            if "mac" in storage_data:
                storage_config.mac = StoragePathsConfig(**storage_data["mac"])

            config.storage = storage_config

        # Load provider detection rules
        if "provider_detection" in data:
            detection_data = data["provider_detection"]
            for provider, rules in detection_data.items():
                config.storage.provider_detection[provider] = ProviderDetectionConfig(
                    check_path=rules.get("check_path", ""),
                    hostname_patterns=rules.get("hostname_patterns", []),
                    os_type=rules.get("os_type"),
                )

        # Load paths
        config.hosts_config_path = data.get("hosts_config_path", config.hosts_config_path)
        config.elo_db = data.get("elo_db", config.elo_db)
        config.data_manifest_db = data.get("data_manifest_db", config.data_manifest_db)
        config.log_dir = data.get("log_dir", config.log_dir)
        config.metrics_enabled = data.get("metrics_enabled", config.metrics_enabled)
        config.metrics_port = data.get("metrics_port", config.metrics_port)

        # Load data loading configuration (December 2025)
        if "data_loading" in data:
            dl_data = data["data_loading"]
            config.data_loading = DataLoadingConfig(
                policy_size=dl_data.get("policy_size", 55000),
                max_policy_size=dl_data.get("max_policy_size", 60000),
                batch_size=dl_data.get("batch_size", 512),
                num_workers=dl_data.get("num_workers", 4),
                prefetch_factor=dl_data.get("prefetch_factor", 2),
                pin_memory=dl_data.get("pin_memory", True),
                filter_empty_policies=dl_data.get("filter_empty_policies", True),
                max_samples=dl_data.get("max_samples"),
                buffer_size=dl_data.get("buffer_size", 10000),
                poll_interval_seconds=dl_data.get("poll_interval_seconds", 5.0),
                dedupe_window=dl_data.get("dedupe_window", 50000),
                enable_multi_source=dl_data.get("enable_multi_source", True),
                max_sources=dl_data.get("max_sources", 10),
                phase_weights=dl_data.get("phase_weights", {
                    "ring_placement": 0.8,
                    "movement": 1.0,
                    "capture": 1.2,
                    "late_game": 1.5,
                }),
                victory_type_weights=dl_data.get("victory_type_weights", {
                    "elimination": 1.0,
                    "territory": 1.2,
                    "ring_out": 0.8,
                    "timeout": 0.5,
                }),
            )

        # Load quality configuration (December 2025)
        if "quality" in data:
            q_data = data["quality"]
            config.quality = QualityConfig(
                outcome_weight=q_data.get("outcome_weight", 0.25),
                length_weight=q_data.get("length_weight", 0.25),
                phase_balance_weight=q_data.get("phase_balance_weight", 0.20),
                diversity_weight=q_data.get("diversity_weight", 0.15),
                source_reputation_weight=q_data.get("source_reputation_weight", 0.15),
                sync_elo_weight=q_data.get("sync_elo_weight", 0.4),
                sync_length_weight=q_data.get("sync_length_weight", 0.3),
                sync_decisive_weight=q_data.get("sync_decisive_weight", 0.3),
                min_elo=q_data.get("min_elo", 1200.0),
                max_elo=q_data.get("max_elo", 2400.0),
                default_elo=q_data.get("default_elo", 1500.0),
                min_game_length=q_data.get("min_game_length", 10),
                max_game_length=q_data.get("max_game_length", 200),
                optimal_game_length=q_data.get("optimal_game_length", 80),
                quality_weight=q_data.get("quality_weight", 0.4),
                recency_weight=q_data.get("recency_weight", 0.3),
                priority_weight=q_data.get("priority_weight", 0.3),
                recency_half_life_hours=q_data.get("recency_half_life_hours", 24.0),
                min_quality_for_training=q_data.get("min_quality_for_training", 0.3),
                min_quality_for_priority_sync=q_data.get("min_quality_for_priority_sync", 0.5),
                high_quality_threshold=q_data.get("high_quality_threshold", 0.7),
                decisive_bonus=q_data.get("decisive_bonus", 1.0),
                draw_credit=q_data.get("draw_credit", 0.3),
            )

        # Load board configurations
        config._board_configs = data.get("configurations", [])

        return config

    def get_all_board_configs(self) -> List[BoardConfig]:
        """Get all 9 board configurations."""
        configs = []
        for bc in self._board_configs:
            board_type = bc.get("board_type", "")
            for num_players in bc.get("num_players", []):
                configs.append(BoardConfig(board_type=board_type, num_players=num_players))

        # Fallback to default 9 configs if not specified
        if not configs:
            for board in ["square8", "square19", "hexagonal"]:
                for players in [2, 3, 4]:
                    configs.append(BoardConfig(board_type=board, num_players=players))

        return configs

    def get_elo_db_path(self, base_path: Optional[Path] = None) -> Path:
        """Get absolute path to Elo database."""
        if base_path is None:
            base_path = Path(__file__).parent.parent.parent
        return base_path / self.elo_db

    def validate(self) -> List[str]:
        """Validate configuration values and return list of errors.

        Returns:
            List of error messages. Empty list means valid.
        """
        errors: List[str] = []
        valid_backends = {"auto", "local", "ssh", "p2p", "slurm"}
        backend_value = str(self.execution_backend or "auto").lower()
        if backend_value not in valid_backends:
            errors.append(
                f"execution_backend={self.execution_backend!r} invalid "
                f"(expected one of: {', '.join(sorted(valid_backends))})"
            )

        # Training thresholds
        if self.training.trigger_threshold_games < 10:
            errors.append(f"training.trigger_threshold_games={self.training.trigger_threshold_games} too low (min: 10)")
        if self.training.trigger_threshold_games > 100000:
            errors.append(f"training.trigger_threshold_games={self.training.trigger_threshold_games} too high (max: 100000)")
        if self.training.min_interval_seconds < 60:
            errors.append(f"training.min_interval_seconds={self.training.min_interval_seconds} too low (min: 60)")
        if self.training.validation_split < 0 or self.training.validation_split > 0.5:
            errors.append(f"training.validation_split={self.training.validation_split} out of range (0-0.5)")

        # Evaluation thresholds
        if self.evaluation.shadow_games_per_config < 1:
            errors.append(f"evaluation.shadow_games_per_config={self.evaluation.shadow_games_per_config} too low (min: 1)")
        if self.evaluation.min_games_for_elo < 1:
            errors.append(f"evaluation.min_games_for_elo={self.evaluation.min_games_for_elo} too low (min: 1)")
        if self.evaluation.elo_k_factor < 1 or self.evaluation.elo_k_factor > 100:
            errors.append(f"evaluation.elo_k_factor={self.evaluation.elo_k_factor} out of range (1-100)")

        # Promotion thresholds
        if self.promotion.elo_threshold < 0:
            errors.append(f"promotion.elo_threshold={self.promotion.elo_threshold} cannot be negative")
        if self.promotion.significance_level < 0.001 or self.promotion.significance_level > 0.5:
            errors.append(f"promotion.significance_level={self.promotion.significance_level} out of range (0.001-0.5)")

        # Curriculum weights
        if self.curriculum.max_weight_multiplier < 1.0:
            errors.append(f"curriculum.max_weight_multiplier={self.curriculum.max_weight_multiplier} must be >= 1.0")
        if self.curriculum.min_weight_multiplier > 1.0:
            errors.append(f"curriculum.min_weight_multiplier={self.curriculum.min_weight_multiplier} must be <= 1.0")
        if self.curriculum.min_weight_multiplier > self.curriculum.max_weight_multiplier:
            errors.append("curriculum.min_weight_multiplier > max_weight_multiplier")

        # Safeguards
        if self.safeguards.max_python_processes_per_host < 1:
            errors.append(f"safeguards.max_python_processes_per_host={self.safeguards.max_python_processes_per_host} too low")
        if self.safeguards.max_process_age_hours < 0.5:
            errors.append(f"safeguards.max_process_age_hours={self.safeguards.max_process_age_hours} too low (min: 0.5)")

        # Safety thresholds
        if self.safety.overfit_threshold < 0 or self.safety.overfit_threshold > 1:
            errors.append(f"safety.overfit_threshold={self.safety.overfit_threshold} out of range (0-1)")
        if self.safety.data_quality_score_min < 0 or self.safety.data_quality_score_min > 1:
            errors.append(f"safety.data_quality_score_min={self.safety.data_quality_score_min} out of range (0-1)")

        # Tournament config
        if self.tournament.k_factor < 1 or self.tournament.k_factor > 100:
            errors.append(f"tournament.k_factor={self.tournament.k_factor} out of range (1-100)")
        if self.tournament.initial_elo < 0:
            errors.append(f"tournament.initial_elo={self.tournament.initial_elo} cannot be negative")

        # Selfplay config
        if self.selfplay.mcts_simulations < 1:
            errors.append(f"selfplay.mcts_simulations={self.selfplay.mcts_simulations} too low (min: 1)")
        if self.selfplay.temperature < 0:
            errors.append(f"selfplay.temperature={self.selfplay.temperature} cannot be negative")

        return errors

    def validate_or_raise(self) -> None:
        """Validate and raise ValueError if invalid."""
        errors = self.validate()
        if errors:
            raise ValueError(f"Config validation failed:\n  " + "\n  ".join(errors))

    def apply_env_overrides(self) -> None:
        """Apply environment variable overrides."""
        # Training threshold override
        if "RINGRIFT_TRAINING_THRESHOLD" in os.environ:
            self.training.trigger_threshold_games = int(os.environ["RINGRIFT_TRAINING_THRESHOLD"])
            logger.info(f"Training threshold overridden to {self.training.trigger_threshold_games}")

        # Also support the old env var name for backwards compatibility
        if "RINGRIFT_MIN_GAMES_FOR_TRAINING" in os.environ:
            self.training.trigger_threshold_games = int(os.environ["RINGRIFT_MIN_GAMES_FOR_TRAINING"])
            logger.info(f"Training threshold overridden to {self.training.trigger_threshold_games} (via legacy env var)")

        # Elo database override
        if "RINGRIFT_ELO_DB" in os.environ:
            self.elo_db = os.environ["RINGRIFT_ELO_DB"]
            logger.info(f"Elo DB overridden to {self.elo_db}")


def get_config(config_path: Optional[str | Path] = None, force_reload: bool = False) -> UnifiedConfig:
    """Get the unified configuration singleton.

    Args:
        config_path: Optional path to config file. Defaults to config/unified_loop.yaml
        force_reload: Force reload from file even if already loaded

    Returns:
        UnifiedConfig instance (singleton)
    """
    global _config_instance

    if _config_instance is not None and not force_reload:
        return _config_instance

    # Determine config path
    if config_path is None:
        config_path = os.environ.get("RINGRIFT_CONFIG_PATH", DEFAULT_CONFIG_PATH)

    # Make path absolute relative to ai-service root
    config_path = Path(config_path)
    if not config_path.is_absolute():
        ai_service_root = Path(__file__).parent.parent.parent
        config_path = ai_service_root / config_path

    # Load config
    _config_instance = UnifiedConfig.from_yaml(config_path)
    _config_instance.apply_env_overrides()

    # Validate configuration
    validation_errors = _config_instance.validate()
    if validation_errors:
        for error in validation_errors:
            logger.warning(f"Config validation: {error}")

    logger.info(f"Loaded unified config from {config_path}")
    logger.info(f"  Training threshold: {_config_instance.training.trigger_threshold_games}")
    logger.info(f"  Elo DB: {_config_instance.elo_db}")

    return _config_instance


def get_training_threshold() -> int:
    """Convenience function to get the training threshold.

    Use this instead of hardcoding values like MIN_NEW_GAMES_FOR_TRAINING.
    """
    return get_config().training.trigger_threshold_games


def get_elo_db_path() -> Path:
    """Convenience function to get the Elo database path."""
    return get_config().get_elo_db_path()


def get_min_elo_improvement() -> float:
    """Get the minimum Elo improvement required for promotion.

    CANONICAL SOURCE: Use this instead of hardcoding values like 25.0.
    Other modules (model_lifecycle, unified_loop_extensions) should
    use this function to ensure consistency.
    """
    return float(get_config().promotion.elo_threshold)


def get_target_selfplay_rate() -> int:
    """Get the target selfplay games per hour.

    CANONICAL SOURCE: Use this instead of hardcoding values like 1000.
    Other modules (p2p_integration, unified_ai_loop) should
    use this function to ensure consistency.
    """
    return get_config().cluster.target_selfplay_games_per_hour


# =========================================================================
# Data Collection Config Helpers
# =========================================================================

def get_data_collector_poll_interval() -> int:
    """Get the data collector poll interval in seconds.

    CANONICAL SOURCE: Use this instead of hardcoding values like 60.
    Scripts (streaming_data_collector) should use this function.
    Default: 60 seconds
    """
    cfg = get_config()
    return getattr(cfg, 'data_collector_poll_interval', 60)


def get_data_collector_max_failures() -> int:
    """Get the max consecutive failures before disabling a host.

    CANONICAL SOURCE: Use this instead of hardcoding values like 5.
    Scripts (streaming_data_collector) should use this function.
    Default: 5
    """
    cfg = get_config()
    return getattr(cfg, 'data_collector_max_failures', 5)


# =========================================================================
# Tournament Config Helpers
# =========================================================================

def get_shadow_tournament_interval() -> int:
    """Get the shadow tournament interval in seconds.

    CANONICAL SOURCE: Use this instead of hardcoding values like 900 (15 min).
    Scripts (shadow_tournament_service) should use this function.
    Default: 900 seconds (15 minutes)
    """
    cfg = get_config()
    return getattr(cfg, 'shadow_tournament_interval', 900)


def get_full_tournament_interval() -> int:
    """Get the full tournament interval in seconds.

    CANONICAL SOURCE: Use this instead of hardcoding values like 3600 (1 hour).
    Scripts (shadow_tournament_service) should use this function.
    Default: 3600 seconds (1 hour)
    """
    cfg = get_config()
    return getattr(cfg, 'full_tournament_interval', 3600)


def get_tournament_games_per_matchup() -> int:
    """Get the number of games per matchup in tournaments.

    CANONICAL SOURCE: Use this instead of hardcoding values like 20 or 50.
    Scripts (improvement_cycle_manager, shadow_tournament_service) should use this.
    Default: 20
    """
    cfg = get_config()
    return getattr(cfg, 'tournament_games_per_matchup', 20)


def get_regression_elo_threshold() -> float:
    """Get the Elo threshold for detecting regressions.

    CANONICAL SOURCE: Use this instead of hardcoding values like 30.0.
    Scripts (shadow_tournament_service) should use this function.
    Default: 30.0 Elo points
    """
    cfg = get_config()
    return getattr(cfg, 'regression_elo_threshold', 30.0)


def get_promotion_elo_threshold() -> float:
    """Get the Elo gain required for automatic model promotion.

    CANONICAL SOURCE: Use this instead of hardcoding values like 20.0.
    Scripts (model_promotion_manager) should use this function.
    Default: 20.0 Elo points
    """
    cfg = get_config()
    return getattr(cfg, 'promotion_elo_threshold', 20.0)


def get_promotion_min_games() -> int:
    """Get the minimum games required before a model can be promoted.

    CANONICAL SOURCE: Use this instead of hardcoding values like 50.
    Scripts (model_promotion_manager) should use this function.
    Default: 50 games
    """
    cfg = get_config()
    return getattr(cfg, 'promotion_min_games', 50)


def get_promotion_check_interval() -> int:
    """Get the interval between auto-promotion checks in seconds.

    CANONICAL SOURCE: Use this instead of hardcoding values like 300.
    Scripts (model_promotion_manager) should use this function.
    Default: 300 seconds (5 minutes)
    """
    cfg = get_config()
    return getattr(cfg, 'promotion_check_interval', 300)


def get_rollback_elo_threshold() -> float:
    """Get the Elo drop threshold for triggering automatic rollback.

    CANONICAL SOURCE: Use this instead of hardcoding values like 50.0.
    Scripts (model_promotion_manager) should use this function.
    Default: 50.0 Elo points
    """
    cfg = get_config()
    return getattr(cfg, 'rollback_elo_threshold', 50.0)


def get_rollback_min_games() -> int:
    """Get the minimum games before rollback can be considered.

    CANONICAL SOURCE: Use this instead of hardcoding values like 20.
    Scripts (model_promotion_manager) should use this function.
    Default: 20 games
    """
    cfg = get_config()
    return getattr(cfg, 'rollback_min_games', 20)


# Constants for backwards compatibility
# These should be deprecated in favor of get_config()
def _get_legacy_threshold() -> int:
    """Legacy accessor - prefer get_config().training.trigger_threshold_games"""
    return get_config().training.trigger_threshold_games


# Export commonly used constants (computed at import time for backwards compat)
# NOTE: These are evaluated once at import - use get_training_threshold() for dynamic access
ALL_BOARD_CONFIGS: List[Tuple[str, int]] = [
    ("square8", 2), ("square8", 3), ("square8", 4),
    ("square19", 2), ("square19", 3), ("square19", 4),
    ("hexagonal", 2), ("hexagonal", 3), ("hexagonal", 4),
]


# =============================================================================
# Integrated Enhancements Factory
# =============================================================================

def create_training_manager(
    model: Optional[Any] = None,
    board_type: str = "square8",
    config: Optional[UnifiedConfig] = None,
) -> Optional[Any]:
    """Create an IntegratedTrainingManager from UnifiedConfig.

    Factory function that creates a training enhancement manager using
    the canonical IntegratedEnhancementsConfig from unified config.

    Args:
        model: PyTorch model (optional, can be set later)
        board_type: Board type for augmentation
        config: Optional UnifiedConfig (defaults to get_config())

    Returns:
        IntegratedTrainingManager instance or None if disabled/unavailable

    Example:
        from app.config.unified_config import create_training_manager

        manager = create_training_manager(model=my_model, board_type="square8")
        if manager:
            manager.initialize_all()
            batch_size = manager.get_batch_size()
    """
    if config is None:
        config = get_config()

    if not config.enhancements.enabled:
        return None

    try:
        from app.training.integrated_enhancements import (
            IntegratedTrainingManager,
            IntegratedEnhancementsConfig as EnhancementsConfig,
        )

        # Map UnifiedConfig.enhancements to IntegratedEnhancementsConfig
        enh = config.enhancements
        manager_config = EnhancementsConfig(
            auxiliary_tasks_enabled=enh.auxiliary_tasks_enabled,
            aux_game_length_weight=enh.aux_game_length_weight,
            aux_piece_count_weight=enh.aux_piece_count_weight,
            aux_outcome_weight=enh.aux_outcome_weight,
            gradient_surgery_enabled=enh.gradient_surgery_enabled,
            gradient_surgery_method=enh.gradient_surgery_method,
            batch_scheduling_enabled=enh.batch_scheduling_enabled,
            batch_initial_size=enh.batch_initial_size,
            batch_final_size=enh.batch_final_size,
            batch_warmup_steps=enh.batch_warmup_steps,
            batch_rampup_steps=enh.batch_rampup_steps,
            batch_schedule_type=enh.batch_schedule_type,
            background_eval_enabled=enh.background_eval_enabled,
            eval_interval_steps=enh.eval_interval_steps,
            eval_elo_checkpoint_threshold=enh.eval_elo_checkpoint_threshold,
            elo_weighting_enabled=enh.elo_weighting_enabled,
            elo_base_rating=enh.elo_base_rating,
            elo_weight_scale=enh.elo_weight_scale,
            elo_min_weight=enh.elo_min_weight,
            elo_max_weight=enh.elo_max_weight,
            curriculum_enabled=enh.curriculum_enabled,
            curriculum_auto_advance=enh.curriculum_auto_advance,
            augmentation_enabled=enh.augmentation_enabled,
            augmentation_mode=enh.augmentation_mode,
            reanalysis_enabled=enh.reanalysis_enabled,
            reanalysis_blend_ratio=enh.reanalysis_blend_ratio,
        )

        return IntegratedTrainingManager(manager_config, model, board_type)

    except ImportError as e:
        logger.warning(f"Failed to import integrated enhancements: {e}")
        return None


# =============================================================================
# Storage Path Helpers
# =============================================================================

def detect_storage_provider() -> str:
    """Detect the current storage provider based on environment.

    Returns:
        Provider name: "lambda", "vast", "mac", or "default"
    """
    import fnmatch
    import platform
    import socket

    config = get_config()
    hostname = socket.gethostname()

    # Check detection rules in priority order
    for provider, rules in config.storage.provider_detection.items():
        # Check OS type
        if rules.os_type and platform.system() == rules.os_type:
            return provider

        # Check path existence
        if rules.check_path and Path(rules.check_path).exists():
            return provider

        # Check hostname patterns
        for pattern in rules.hostname_patterns:
            if fnmatch.fnmatch(hostname, pattern):
                return provider

    # Fallback detection without explicit rules
    # Lambda: NFS mount
    if Path("/lambda/nfs").exists():
        return "lambda"

    # Vast: /workspace directory
    if Path("/workspace").exists():
        return "vast"

    # Mac: Darwin OS
    if platform.system() == "Darwin":
        return "mac"

    return "default"


def get_storage_paths(provider: Optional[str] = None) -> StoragePathsConfig:
    """Get storage paths for the current or specified provider.

    Args:
        provider: Optional provider name. Auto-detected if not specified.

    Returns:
        StoragePathsConfig with paths for the provider.

    Example:
        from app.config.unified_config import get_storage_paths

        paths = get_storage_paths()
        selfplay_dir = paths.selfplay_games
        model_dir = paths.model_checkpoints

        # For specific provider
        lambda_paths = get_storage_paths("lambda")
    """
    if provider is None:
        provider = detect_storage_provider()

    config = get_config()

    if provider == "lambda":
        return config.storage.lambda_
    elif provider == "vast":
        return config.storage.vast
    elif provider == "mac":
        return config.storage.mac
    else:
        return config.storage.default


def get_selfplay_dir(provider: Optional[str] = None) -> Path:
    """Get the selfplay games directory for the current provider."""
    return Path(get_storage_paths(provider).selfplay_games)


def get_model_checkpoint_dir(provider: Optional[str] = None) -> Path:
    """Get the model checkpoint directory for the current provider."""
    return Path(get_storage_paths(provider).model_checkpoints)


def get_training_data_dir(provider: Optional[str] = None) -> Path:
    """Get the training data directory for the current provider."""
    return Path(get_storage_paths(provider).training_data)


def get_nfs_base(provider: Optional[str] = None) -> Optional[Path]:
    """Get the NFS base path if available for the provider."""
    paths = get_storage_paths(provider)
    if paths.nfs_base:
        return Path(paths.nfs_base)
    return None


def should_use_nfs_sync(provider: Optional[str] = None) -> bool:
    """Check if NFS should be used for data sync (no rsync needed)."""
    return get_storage_paths(provider).use_nfs_for_sync


def should_skip_rsync_to_node(node_id: str) -> bool:
    """Check if rsync should be skipped for a node (has NFS access).

    Args:
        node_id: The node identifier

    Returns:
        True if node has NFS access and rsync should be skipped
    """
    # Lambda nodes with NFS don't need rsync between each other
    if node_id.startswith("lambda-"):
        return get_storage_paths("lambda").skip_rsync_to_nfs_nodes
    return False


def ensure_storage_dirs(provider: Optional[str] = None) -> None:
    """Ensure all storage directories exist for the provider.

    Creates directories if they don't exist.
    """
    paths = get_storage_paths(provider)

    for path_str in [
        paths.selfplay_games,
        paths.model_checkpoints,
        paths.training_data,
        paths.sync_staging,
        paths.local_scratch,
    ]:
        path = Path(path_str)
        if not path.is_absolute():
            # Make relative paths absolute from ai-service root
            path = AI_SERVICE_ROOT / path

        ensure_dir(path)

    # Also create parent dir for elo database
    elo_path = Path(paths.elo_database)
    if not elo_path.is_absolute():
        elo_path = AI_SERVICE_ROOT / elo_path
    ensure_parent_dir(elo_path)
