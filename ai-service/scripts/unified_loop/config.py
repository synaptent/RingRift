"""Unified AI Loop Configuration Classes.

This module contains all configuration dataclasses and event types
for the unified AI improvement loop.
Extracted from unified_ai_loop.py for better modularity.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import yaml


# =============================================================================
# Configuration Dataclasses
# =============================================================================

@dataclass
class DataIngestionConfig:
    """Configuration for streaming data collection.

    When use_external_sync=True, the internal data collector is disabled and
    the loop expects an external unified_data_sync.py service to handle data sync.
    This allows using advanced features like P2P fallback, WAL, and content dedup.
    """
    poll_interval_seconds: int = 60
    ephemeral_poll_interval_seconds: int = 15  # Aggressive sync for RAM disk hosts
    sync_method: str = "incremental"  # "incremental" or "full"
    deduplication: bool = True
    min_games_per_sync: int = 10
    remote_db_pattern: str = "data/games/*.db"
    # External sync: when True, skip internal data collection and rely on external
    # unified_data_sync.py service (provides P2P fallback, WAL, content dedup)
    use_external_sync: bool = False
    # Hardening options
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

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    trigger_threshold_games: int = 500  # Canonical: 500 (was 1000)
    min_interval_seconds: int = 1200  # Canonical: 20 min (was 30)
    max_concurrent_jobs: int = 1
    prefer_gpu_hosts: bool = True
    # Simplified 3-signal trigger system (2024-12)
    use_simplified_triggers: bool = True  # Use 3-signal system instead of 8+ signals
    staleness_hours: float = 6.0  # Hours before config is "stale"
    min_win_rate_threshold: float = 0.45  # Below this triggers urgent training
    bootstrap_threshold: int = 50  # Low threshold for configs with 0 models
    nnue_training_script: str = "scripts/train_nnue.py"
    nn_training_script: str = "scripts/run_nn_training_baseline.py"
    export_script: str = "scripts/export_replay_dataset.py"
    # Encoder version for hex boards: "v3" uses HexStateEncoderV3 (16 channels)
    hex_encoder_version: str = "v3"
    # Verbose logging for training scheduler
    verbose: bool = False
    # Optimized training settings
    batch_size: int = 256  # Higher batch size for better GPU utilization
    sampling_weights: str = "victory_type"  # Balance across victory types
    warmup_epochs: int = 5  # LR warmup for stability
    use_optimized_hyperparams: bool = True  # Load from hyperparameters.json
    # Advanced training optimizations (2024-12 improvements)
    use_spectral_norm: bool = True  # Gradient stability via spectral normalization
    use_lars: bool = False  # LARS optimizer for distributed large-batch training
    use_cyclic_lr: bool = True  # Cyclic LR with triangular waves
    cyclic_lr_period: int = 5  # Cycle period in epochs
    use_gradient_profiling: bool = False  # Track gradient norms for diagnostics
    use_mixed_precision: bool = True  # FP16/BF16 mixed precision training
    amp_dtype: str = "bfloat16"  # Prefer BF16 for stability
    gradient_accumulation: int = 1  # Gradient accumulation steps
    # Knowledge distillation (optional)
    use_knowledge_distill: bool = False  # Train from teacher model
    teacher_model_path: Optional[str] = None  # Path to teacher model
    distill_alpha: float = 0.5  # Blend weight (0=pure label, 1=pure teacher)
    distill_temperature: float = 2.0  # Softening temperature
    # Advanced NNUE policy training options (2024-12)
    use_swa: bool = True  # Stochastic Weight Averaging for better generalization
    swa_start_fraction: float = 0.75  # Start SWA at 75% of epochs
    use_ema: bool = True  # Exponential Moving Average for smoother weights
    ema_decay: float = 0.999  # EMA decay rate
    use_progressive_batch: bool = True  # Progressive batch sizing
    min_batch_size: int = 64  # Starting batch size
    max_batch_size: int = 512  # Maximum batch size
    focal_gamma: float = 2.0  # Focal loss gamma for hard sample mining
    label_smoothing_warmup: int = 5  # Warmup epochs for label smoothing
    policy_label_smoothing: float = 0.05  # Policy label smoothing factor (0.05-0.1 recommended)
    use_hex_augmentation: bool = True  # D6 symmetry augmentation for hex boards
    hex_augment_count: int = 6  # Number of D6 symmetry augmentations (1-12)
    policy_dropout: float = 0.1  # Dropout rate for policy head regularization
    # Learning rate finder (auto-discovers optimal LR before training)
    use_lr_finder: bool = False  # Run LR finder before training
    lr_finder_iterations: int = 100  # Number of iterations for LR sweep
    # 2024-12 Advanced Training Improvements
    use_value_whitening: bool = True  # Value head whitening for stable training
    value_whitening_momentum: float = 0.99  # Momentum for running stats
    use_stochastic_depth: bool = True  # Stochastic depth regularization
    stochastic_depth_prob: float = 0.1  # Drop probability
    use_adaptive_warmup: bool = True  # Adaptive warmup based on dataset size
    use_hard_example_mining: bool = True  # Focus on difficult examples
    hard_example_top_k: float = 0.3  # Top 30% hardest examples
    use_dynamic_batch: bool = False  # Dynamic batch scheduling (optional)
    dynamic_batch_schedule: str = "linear"  # linear, exponential, or step
    transfer_from_model: Optional[str] = None  # Cross-board transfer learning
    transfer_freeze_epochs: int = 5  # Freeze transferred layers for N epochs
    # Advanced optimizer enhancements
    use_lookahead: bool = True  # Lookahead optimizer wrapper
    lookahead_k: int = 5  # Slow weight update interval
    lookahead_alpha: float = 0.5  # Interpolation factor
    use_adaptive_clip: bool = True  # Adaptive gradient clipping
    use_gradient_noise: bool = False  # Gradient noise injection (optional)
    gradient_noise_variance: float = 0.01  # Initial noise variance
    # Architecture search and pretraining
    use_board_nas: bool = True  # Board-specific neural architecture search
    use_self_supervised: bool = False  # Self-supervised pre-training (optional)
    ss_epochs: int = 10  # Self-supervised pre-training epochs
    ss_projection_dim: int = 128  # Projection dimension for contrastive learning
    ss_temperature: float = 0.07  # Contrastive loss temperature
    use_online_bootstrap: bool = True  # Online bootstrapping with soft labels
    bootstrap_temperature: float = 1.5  # Soft label temperature
    bootstrap_start_epoch: int = 10  # Epoch to start bootstrapping
    # Phase 2 Advanced Training (2024-12)
    use_prefetch_gpu: bool = True  # GPU prefetching for improved throughput
    use_attention: bool = False  # Positional attention (experimental)
    attention_heads: int = 4  # Number of attention heads
    use_moe: bool = False  # Mixture of Experts (experimental)
    moe_experts: int = 4  # Number of experts
    moe_top_k: int = 2  # Top-k expert selection
    use_multitask: bool = False  # Multi-task learning heads
    multitask_weight: float = 0.1  # Auxiliary task weight
    use_difficulty_curriculum: bool = True  # Difficulty-aware curriculum
    curriculum_initial_threshold: float = 0.9  # Start with easy samples
    curriculum_final_threshold: float = 0.3  # End with all samples
    use_lamb: bool = False  # LAMB optimizer for large batch
    use_gradient_compression: bool = False  # Gradient compression for distributed
    compression_ratio: float = 0.1  # Keep top 10% of gradients
    use_quantized_eval: bool = True  # Quantized inference for validation
    use_contrastive: bool = False  # Contrastive representation learning
    contrastive_weight: float = 0.1  # Contrastive loss weight
    # NNUE policy training script
    nnue_policy_script: str = "scripts/train_nnue_policy.py"
    nnue_curriculum_script: str = "scripts/train_nnue_policy_curriculum.py"


@dataclass
class EvaluationConfig:
    """Configuration for continuous evaluation.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    OPTIMIZED: Reduced default from 900s to 300s since parallel execution is 3x faster
    """
    shadow_interval_seconds: int = 300  # 5 minutes (reduced from 15)
    shadow_games_per_config: int = 15  # Canonical: 15 (was 10)
    full_tournament_interval_seconds: int = 3600  # 1 hour
    full_tournament_games: int = 50
    baseline_models: List[str] = field(default_factory=lambda: ["random", "heuristic", "mcts_100", "mcts_500"])
    # Adaptive interval settings - go faster when cluster is healthy
    adaptive_interval_enabled: bool = True
    adaptive_interval_min_seconds: int = 120  # Can go as low as 2 min
    adaptive_interval_max_seconds: int = 600  # Cap at 10 min during high load


@dataclass
class PromotionConfig:
    """Configuration for automatic model promotion.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    auto_promote: bool = True
    elo_threshold: int = 25  # Canonical: 25 (was 20)
    min_games: int = 50
    significance_level: float = 0.05
    sync_to_cluster: bool = True

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
    """Configuration for adaptive curriculum.

    NOTE: Defaults match app/config/unified_config.py (single source of truth)
    """
    adaptive: bool = True
    rebalance_interval_seconds: int = 3600  # 1 hour
    max_weight_multiplier: float = 1.5  # Canonical: 1.5 (was 2.0)
    min_weight_multiplier: float = 0.7  # Canonical: 0.7 (was 0.5)


@dataclass
class PBTConfig:
    """Configuration for Population-Based Training."""
    enabled: bool = False  # Disabled by default - resource intensive
    population_size: int = 8
    exploit_interval_steps: int = 1000
    tunable_params: List[str] = field(default_factory=lambda: ["learning_rate", "batch_size", "temperature"])
    check_interval_seconds: int = 1800  # Check PBT status every 30 min
    auto_start: bool = False  # Auto-start PBT when training completes


@dataclass
class NASConfig:
    """Configuration for Neural Architecture Search."""
    enabled: bool = False  # Disabled by default - very resource intensive
    strategy: str = "evolutionary"  # evolutionary, random, bayesian
    population_size: int = 20
    generations: int = 50
    check_interval_seconds: int = 3600  # Check NAS status every hour
    auto_start_on_plateau: bool = False  # Start NAS when Elo plateaus


@dataclass
class PERConfig:
    """Configuration for Prioritized Experience Replay."""
    enabled: bool = True  # Enabled by default - improves training efficiency
    alpha: float = 0.6  # Priority exponent
    beta: float = 0.4  # Importance sampling exponent
    buffer_capacity: int = 100000
    rebuild_interval_seconds: int = 7200  # Rebuild buffer every 2 hours


@dataclass
class FeedbackConfig:
    """Configuration for pipeline feedback controller integration."""
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
    """Configuration for P2P distributed cluster integration."""
    enabled: bool = False  # Enable P2P cluster coordination
    p2p_base_url: str = "http://localhost:8770"  # P2P orchestrator URL
    auth_token: Optional[str] = None  # Auth token (defaults to RINGRIFT_CLUSTER_AUTH_TOKEN env)
    model_sync_enabled: bool = True  # Auto-sync models to cluster
    target_selfplay_games_per_hour: int = 1000  # Target selfplay rate across cluster
    auto_scale_selfplay: bool = True  # Auto-scale selfplay workers
    use_distributed_tournament: bool = True  # Use cluster for tournament evaluation
    health_check_interval: int = 60  # Seconds between cluster health checks
    sync_interval_seconds: int = 300  # Seconds between data sync with cluster


@dataclass
class ModelPruningConfig:
    """Configuration for automated model pruning/evaluation."""
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
class UnifiedLoopConfig:
    """Complete configuration for the unified AI loop."""
    data_ingestion: DataIngestionConfig = field(default_factory=DataIngestionConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    promotion: PromotionConfig = field(default_factory=PromotionConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    pbt: PBTConfig = field(default_factory=PBTConfig)
    nas: NASConfig = field(default_factory=NASConfig)
    per: PERConfig = field(default_factory=PERConfig)
    feedback: FeedbackConfig = field(default_factory=FeedbackConfig)
    p2p: P2PClusterConfig = field(default_factory=P2PClusterConfig)
    model_pruning: ModelPruningConfig = field(default_factory=ModelPruningConfig)

    # Host configuration
    hosts_config_path: str = "config/remote_hosts.yaml"

    # Database paths
    elo_db: str = "data/unified_elo.db"  # Canonical Elo database
    data_manifest_db: str = "data/data_manifest.db"

    # Logging
    log_dir: str = "logs/unified_loop"
    verbose: bool = False

    # Metrics
    metrics_port: int = 9091  # Note: 9090 is reserved for Prometheus itself
    metrics_enabled: bool = True

    # Operation modes
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, path: Path) -> "UnifiedLoopConfig":
        """Load configuration from YAML file."""
        if not path.exists():
            return cls()

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        config = cls()

        if "data_ingestion" in data:
            for k, v in data["data_ingestion"].items():
                if hasattr(config.data_ingestion, k):
                    setattr(config.data_ingestion, k, v)

        if "training" in data:
            for k, v in data["training"].items():
                if hasattr(config.training, k):
                    setattr(config.training, k, v)

        if "evaluation" in data:
            for k, v in data["evaluation"].items():
                if hasattr(config.evaluation, k):
                    setattr(config.evaluation, k, v)

        if "promotion" in data:
            for k, v in data["promotion"].items():
                if hasattr(config.promotion, k):
                    setattr(config.promotion, k, v)

        if "curriculum" in data:
            for k, v in data["curriculum"].items():
                if hasattr(config.curriculum, k):
                    setattr(config.curriculum, k, v)

        if "pbt" in data:
            for k, v in data["pbt"].items():
                if hasattr(config.pbt, k):
                    setattr(config.pbt, k, v)

        if "nas" in data:
            for k, v in data["nas"].items():
                if hasattr(config.nas, k):
                    setattr(config.nas, k, v)

        if "per" in data:
            for k, v in data["per"].items():
                if hasattr(config.per, k):
                    setattr(config.per, k, v)

        if "feedback" in data:
            for k, v in data["feedback"].items():
                if hasattr(config.feedback, k):
                    setattr(config.feedback, k, v)

        if "p2p" in data:
            for k, v in data["p2p"].items():
                if hasattr(config.p2p, k):
                    setattr(config.p2p, k, v)

        if "model_pruning" in data:
            for k, v in data["model_pruning"].items():
                if hasattr(config.model_pruning, k):
                    setattr(config.model_pruning, k, v)

        for key in ["hosts_config_path", "elo_db", "data_manifest_db", "log_dir",
                    "verbose", "metrics_port", "metrics_enabled", "dry_run"]:
            if key in data:
                setattr(config, key, data[key])

        return config


# =============================================================================
# Event System
# =============================================================================

class DataEventType(Enum):
    """Types of data pipeline events."""
    # Data collection events
    NEW_GAMES_AVAILABLE = "new_games"
    DATA_SYNC_STARTED = "sync_started"
    DATA_SYNC_COMPLETED = "sync_completed"
    DATA_SYNC_FAILED = "sync_failed"
    # Training events
    TRAINING_THRESHOLD_REACHED = "training_threshold"
    TRAINING_STARTED = "training_started"
    TRAINING_PROGRESS = "training_progress"
    TRAINING_COMPLETED = "training_completed"
    TRAINING_FAILED = "training_failed"
    # Evaluation events
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_PROGRESS = "evaluation_progress"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_FAILED = "evaluation_failed"
    ELO_UPDATED = "elo_updated"
    # Promotion events
    PROMOTION_CANDIDATE = "promotion_candidate"
    PROMOTION_STARTED = "promotion_started"
    MODEL_PROMOTED = "model_promoted"
    PROMOTION_FAILED = "promotion_failed"
    PROMOTION_REJECTED = "promotion_rejected"
    # Curriculum events
    CURRICULUM_REBALANCED = "curriculum_rebalanced"
    WEIGHT_UPDATED = "weight_updated"
    ELO_SIGNIFICANT_CHANGE = "elo_significant_change"  # Triggers event-driven curriculum rebalance
    # System events
    DAEMON_STARTED = "daemon_started"
    DAEMON_STOPPED = "daemon_stopped"
    HOST_ONLINE = "host_online"
    HOST_OFFLINE = "host_offline"
    ERROR = "error"
    # PBT events
    PBT_STARTED = "pbt_started"
    PBT_GENERATION_COMPLETE = "pbt_generation_complete"
    PBT_COMPLETED = "pbt_completed"
    # NAS events
    NAS_STARTED = "nas_started"
    NAS_GENERATION_COMPLETE = "nas_generation_complete"
    NAS_COMPLETED = "nas_completed"
    NAS_BEST_ARCHITECTURE = "nas_best_architecture"
    # PER events
    PER_BUFFER_REBUILT = "per_buffer_rebuilt"
    PER_PRIORITIES_UPDATED = "per_priorities_updated"
    # Optimization events
    CMAES_TRIGGERED = "cmaes_triggered"
    CMAES_COMPLETED = "cmaes_completed"
    NAS_TRIGGERED = "nas_triggered"
    PLATEAU_DETECTED = "plateau_detected"
    HYPERPARAMETER_UPDATED = "hyperparameter_updated"
    # Tier gating events
    TIER_PROMOTION = "tier_promotion"
    # Parity validation events
    PARITY_VALIDATION_STARTED = "parity_validation_started"
    PARITY_VALIDATION_COMPLETED = "parity_validation_completed"
    # P2P cluster events
    P2P_CLUSTER_HEALTHY = "p2p_cluster_healthy"
    P2P_CLUSTER_UNHEALTHY = "p2p_cluster_unhealthy"
    P2P_NODES_DEAD = "p2p_nodes_dead"
    P2P_SELFPLAY_SCALED = "p2p_selfplay_scaled"
    P2P_MODEL_SYNCED = "p2p_model_synced"


@dataclass
class DataEvent:
    """A data pipeline event."""
    event_type: DataEventType
    payload: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


# =============================================================================
# State Management
# =============================================================================

@dataclass
class HostState:
    """State for a remote host."""
    name: str
    ssh_host: str
    ssh_user: str = "ubuntu"
    ssh_port: int = 22
    last_sync_time: float = 0.0
    last_game_count: int = 0
    consecutive_failures: int = 0
    enabled: bool = True


@dataclass
class ConfigState:
    """State for a board/player configuration."""
    board_type: str
    num_players: int
    game_count: int = 0
    games_since_training: int = 0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0  # For dynamic threshold calculation
    current_elo: float = 1500.0
    elo_trend: float = 0.0  # Positive = improving
    training_weight: float = 1.0
    # Win rate tracking for training feedback (Phase 2.4)
    win_rate: float = 0.5  # Latest win rate from evaluations (0.5 = default/unknown)
    win_rate_trend: float = 0.0  # Change in win rate (positive = improving)
    consecutive_high_win_rate: int = 0  # Count of evals with win_rate > 0.7
