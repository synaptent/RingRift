"""Unified AI Loop Configuration Classes.

This module contains all configuration dataclasses and event types
for the unified AI improvement loop.
Extracted from unified_ai_loop.py for better modularity.

CONSOLIDATION NOTE (2025-12-18):
================================
Core configuration classes have been migrated to the canonical location:
    app.config.unified_config

Event types consolidated to canonical location:
    app.distributed.data_events

The following classes are now re-exported from canonical:
- PBTConfig, NASConfig, PERConfig, FeedbackConfig, P2PClusterConfig, ModelPruningConfig
- DataEventType, DataEvent (from app.distributed.data_events)

This module retains:
- UnifiedLoopConfig (script-specific root config)
- Runtime state classes (ConfigState, FeedbackState, HostState)

For new integrations, prefer importing from app.config.unified_config.
See ai-service/docs/CONSOLIDATION_ROADMAP.md for consolidation status.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from app.training.integrated_enhancements import IntegratedTrainingManager

import yaml

# Import canonical threshold constants
try:
    from app.config.thresholds import (
        ELO_DROP_ROLLBACK,
        INITIAL_ELO_RATING,
    )
except ImportError:
    INITIAL_ELO_RATING = 1500.0
    ELO_DROP_ROLLBACK = 50.0

# Re-export migrated classes from canonical location for backward compatibility
try:
    from app.config.unified_config import (
        CurriculumConfig,
        DataIngestionConfig,
        EvaluationConfig,
        FeedbackConfig,
        IntegratedEnhancementsConfig,  # Canonical location as of 2025-12-17
        ModelPruningConfig,
        NASConfig,
        P2PClusterConfig,
        PBTConfig,
        PERConfig,
        PromotionConfig,
        TrainingConfig,
    )
    _HAS_CANONICAL_ENHANCEMENTS = True
except ImportError:
    # Fallback: define locally if canonical import fails (shouldn't happen in normal use)
    _HAS_CANONICAL_ENHANCEMENTS = False

# Import DataEventType and DataEvent from canonical location
from app.coordination.event_router import DataEvent, DataEventType

# IntegratedEnhancementsConfig is now imported from canonical location (app.config.unified_config)
# The definitions below are ONLY used as fallback if canonical imports fail (standalone execution)
if not _HAS_CANONICAL_ENHANCEMENTS:
    @dataclass
    class CurriculumConfig:
        """Fallback: Configuration for adaptive curriculum.

        NOTE: This is a fallback definition. The canonical location is:
            app.config.unified_config.CurriculumConfig
        """
        adaptive: bool = True
        rebalance_interval_seconds: int = 3600  # 1 hour
        max_weight_multiplier: float = 1.5  # Canonical: 1.5 (was 2.0)
        min_weight_multiplier: float = 0.7  # Canonical: 0.7 (was 0.5)

    @dataclass
    class IntegratedEnhancementsConfig:
        """Fallback: Configuration for integrated training enhancements.

        NOTE: This is a fallback definition. The canonical location is:
            app.config.unified_config.IntegratedEnhancementsConfig

        Only used when running standalone without app package.
        """
        enabled: bool = True
        auxiliary_tasks_enabled: bool = False
        aux_game_length_weight: float = 0.1
        aux_piece_count_weight: float = 0.1
        aux_outcome_weight: float = 0.05
        gradient_surgery_enabled: bool = False
        gradient_surgery_method: str = "pcgrad"
        gradient_conflict_threshold: float = 0.0
        batch_scheduling_enabled: bool = False
        batch_initial_size: int = 64
        batch_final_size: int = 512
        batch_warmup_steps: int = 1000
        batch_rampup_steps: int = 10000
        batch_schedule_type: str = "linear"
        background_eval_enabled: bool = False
        eval_interval_steps: int = 1000
        eval_games_per_check: int = 20
        eval_elo_checkpoint_threshold: float = 10.0
        eval_elo_drop_threshold: float = 50.0  # ELO_DROP_ROLLBACK default
        eval_auto_checkpoint: bool = True
        eval_checkpoint_dir: str = "data/eval_checkpoints"
        elo_weighting_enabled: bool = True
        elo_base_rating: float = 1500.0  # INITIAL_ELO_RATING default
        elo_weight_scale: float = 400.0
        elo_min_weight: float = 0.5
        elo_max_weight: float = 2.0
        curriculum_enabled: bool = True
        curriculum_auto_advance: bool = True
        curriculum_checkpoint_path: str = "data/curriculum_state.json"
        augmentation_enabled: bool = True
        augmentation_mode: str = "all"
        augmentation_probability: float = 1.0
        reanalysis_enabled: bool = False
        reanalysis_blend_ratio: float = 0.5
        reanalysis_interval_steps: int = 5000
        reanalysis_batch_size: int = 1000


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
    enhancements: IntegratedEnhancementsConfig = field(default_factory=IntegratedEnhancementsConfig)

    # Host configuration
    hosts_config_path: str = "config/distributed_hosts.yaml"

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
    def from_yaml(cls, path: Path) -> UnifiedLoopConfig:
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

        if "enhancements" in data:
            for k, v in data["enhancements"].items():
                if hasattr(config.enhancements, k):
                    setattr(config.enhancements, k, v)

        for key in ["hosts_config_path", "elo_db", "data_manifest_db", "log_dir",
                    "verbose", "metrics_port", "metrics_enabled", "dry_run"]:
            if key in data:
                setattr(config, key, data[key])

        if config.promotion.hosts_config_path is None:
            config.promotion.hosts_config_path = config.hosts_config_path

        return config


# =============================================================================
# Event System - Re-exported from canonical module (2025-12-18)
# =============================================================================
# DataEventType and DataEvent are now imported from the canonical module
# app.distributed.data_events for consolidation and consistency.



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
class FeedbackState:
    """Consolidated feedback state for training decisions (2025-12).

    DEPRECATION NOTE (December 29, 2025):
    This class will be consolidated with CanonicalFeedbackState in
    app.coordination.feedback_state in Q1 2026. New code should prefer:

        from app.coordination.feedback_state import CanonicalFeedbackState

    Field mapping for migration:
        curriculum_weight -> curriculum_weight (same)
        elo_current -> elo_current (same)
        data_quality_score -> quality_score
        win_rate -> win_rate (same)
        urgency_score -> compute_urgency() method in MonitoringFeedbackState

    Groups all feedback signals into a single structure:
    - Curriculum: weights and staleness
    - Quality: parity failures, data health
    - Elo: current rating and trends
    - Win rate: performance metrics
    """
    # Curriculum feedback (0.5-2.0, weight > 1 = needs more training)
    curriculum_weight: float = 1.0
    curriculum_last_update: float = 0.0

    # Data quality feedback
    parity_failure_rate: float = 0.0  # Rolling average of parity failures (0-1)
    parity_checks_total: int = 0  # Total parity checks performed
    data_quality_score: float = 1.0  # Composite quality metric (0-1)

    # Elo feedback (uses INITIAL_ELO_RATING from app.config.thresholds)
    elo_current: float = INITIAL_ELO_RATING
    elo_trend: float = 0.0  # Positive = improving, negative = declining
    elo_peak: float = INITIAL_ELO_RATING  # Historical peak Elo
    elo_plateau_count: int = 0  # Consecutive evaluations without gain

    # Win rate feedback
    win_rate: float = 0.5  # Latest win rate (0-1)
    win_rate_trend: float = 0.0  # Change over recent evals
    consecutive_high_win_rate: int = 0  # Streak above 70%
    consecutive_low_win_rate: int = 0  # Streak below 50%

    # Training urgency metrics
    urgency_score: float = 0.0  # Composite urgency (0-1, higher = more urgent)
    last_urgency_update: float = 0.0

    def update_parity(self, passed: bool, alpha: float = 0.1) -> None:
        """Update rolling parity failure rate."""
        result = 0.0 if passed else 1.0
        self.parity_failure_rate = alpha * result + (1 - alpha) * self.parity_failure_rate
        self.parity_checks_total += 1

    def update_elo(self, new_elo: float, plateau_threshold: float = 15.0) -> None:
        """Update Elo with trend and plateau detection."""
        old_elo = self.elo_current
        self.elo_trend = new_elo - old_elo
        self.elo_current = new_elo
        self.elo_peak = max(self.elo_peak, new_elo)

        # Plateau detection
        if abs(self.elo_trend) < plateau_threshold:
            self.elo_plateau_count += 1
        else:
            self.elo_plateau_count = 0

    def update_win_rate(self, new_win_rate: float) -> None:
        """Update win rate with trend tracking."""
        old_win_rate = self.win_rate
        self.win_rate_trend = new_win_rate - old_win_rate
        self.win_rate = new_win_rate

        # Track consecutive high/low streaks
        if new_win_rate > 0.7:
            self.consecutive_high_win_rate += 1
            self.consecutive_low_win_rate = 0
        elif new_win_rate < 0.5:
            self.consecutive_low_win_rate += 1
            self.consecutive_high_win_rate = 0
        else:
            self.consecutive_high_win_rate = 0
            self.consecutive_low_win_rate = 0

    def compute_urgency(self) -> float:
        """Compute composite urgency score for training prioritization.

        Returns value 0-1 where higher = more urgent training need.
        """
        import time
        urgency = 0.0

        # Factor 1: Low win rate increases urgency
        if self.win_rate < 0.5:
            urgency += (0.5 - self.win_rate) * 0.4  # Up to 0.2 contribution

        # Factor 2: Declining win rate increases urgency
        if self.win_rate_trend < 0:
            urgency += min(0.2, abs(self.win_rate_trend) * 2)

        # Factor 3: Elo plateau increases urgency (stagnation)
        plateau_factor = min(0.2, self.elo_plateau_count * 0.04)
        urgency += plateau_factor

        # Factor 4: High curriculum weight (needs training)
        if self.curriculum_weight > 1.0:
            urgency += min(0.2, (self.curriculum_weight - 1.0) * 0.2)

        # Factor 5: Good data quality is a prerequisite (reduces urgency if bad)
        if self.parity_failure_rate > 0.1:
            urgency *= 0.5  # De-prioritize if data quality is poor

        self.urgency_score = min(1.0, urgency)
        self.last_urgency_update = time.time()
        return self.urgency_score

    def compute_data_quality(
        self,
        sample_diversity: float = 1.0,
        avg_game_length: float = 50.0,
        min_game_length: float = 10.0,
        max_game_length: float = 200.0,
    ) -> float:
        """Compute composite data quality score (0-1).

        Factors:
        - Parity pass rate (inverse of failure rate)
        - Sample diversity (0-1, higher = more diverse positions)
        - Game length normalization (penalize too short or too long)

        Args:
            sample_diversity: Diversity score from data collection (0-1)
            avg_game_length: Average game length in moves
            min_game_length: Expected minimum reasonable length
            max_game_length: Expected maximum reasonable length

        Returns:
            Composite quality score (0-1)
        """
        quality = 0.0

        # Factor 1: Parity pass rate (40% weight)
        parity_score = 1.0 - self.parity_failure_rate
        quality += parity_score * 0.4

        # Factor 2: Sample diversity (30% weight)
        quality += max(0, min(1.0, sample_diversity)) * 0.3

        # Factor 3: Game length normalization (30% weight)
        # Penalize games that are too short (likely errors) or too long (stalemates)
        if avg_game_length < min_game_length:
            length_score = avg_game_length / min_game_length
        elif avg_game_length > max_game_length:
            length_score = max(0.5, max_game_length / avg_game_length)
        else:
            # Optimal range
            length_score = 1.0
        quality += length_score * 0.3

        self.data_quality_score = min(1.0, max(0.0, quality))
        return self.data_quality_score

    def is_data_quality_acceptable(self, threshold: float = 0.7) -> bool:
        """Check if data quality meets minimum threshold.

        Args:
            threshold: Minimum acceptable quality score (0-1)

        Returns:
            True if data quality is acceptable
        """
        return self.data_quality_score >= threshold

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization/logging."""
        return {
            'curriculum_weight': self.curriculum_weight,
            'parity_failure_rate': self.parity_failure_rate,
            'data_quality_score': self.data_quality_score,
            'elo_current': self.elo_current,
            'elo_trend': self.elo_trend,
            'elo_plateau_count': self.elo_plateau_count,
            'win_rate': self.win_rate,
            'win_rate_trend': self.win_rate_trend,
            'urgency_score': self.urgency_score,
        }


@dataclass
class ConfigState:
    """State for a board/player configuration.

    Note: Uses INITIAL_ELO_RATING from app.config.thresholds as default.
    """
    board_type: str
    num_players: int
    game_count: int = 0
    games_since_training: int = 0
    last_training_time: float = 0.0
    last_evaluation_time: float = 0.0
    last_promotion_time: float = 0.0  # For dynamic threshold calculation
    current_elo: float = INITIAL_ELO_RATING
    elo_trend: float = 0.0  # Positive = improving
    training_weight: float = 1.0
    # Win rate tracking for training feedback (Phase 2.4)
    win_rate: float = 0.5  # Latest win rate from evaluations (0.5 = default/unknown)
    win_rate_trend: float = 0.0  # Change in win rate (positive = improving)
    consecutive_high_win_rate: int = 0  # Count of evals with win_rate > 0.7
    # Consolidated feedback state (2025-12)
    feedback: FeedbackState = field(default_factory=FeedbackState)


@dataclass
class UnifiedLoopState:
    """Backward-compatible unified loop runtime state."""

    started_at: str = ""
    hosts: dict[str, HostState] = field(default_factory=dict)
    configs: dict[str, ConfigState] = field(default_factory=dict)
    curriculum_weights: dict[str, float] = field(default_factory=dict)
    training_in_progress: bool = False
    training_config: str = ""
    training_started_at: float = 0.0
    total_data_syncs: int = 0
    total_training_runs: int = 0
    total_evaluations: int = 0
    total_promotions: int = 0
    total_games_pending: int = 0
    current_temperature_preset: str = "default"
    regression_detected: bool = False
    last_curriculum_rebalance: float = 0.0
    last_calibration_time: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize the state for tests and legacy checkpoint flows."""
        return {
            "started_at": self.started_at,
            "hosts": {name: asdict(host) for name, host in self.hosts.items()},
            "configs": {name: asdict(config) for name, config in self.configs.items()},
            "curriculum_weights": dict(self.curriculum_weights),
            "training_in_progress": self.training_in_progress,
            "training_config": self.training_config,
            "training_started_at": self.training_started_at,
            "total_data_syncs": self.total_data_syncs,
            "total_training_runs": self.total_training_runs,
            "total_evaluations": self.total_evaluations,
            "total_promotions": self.total_promotions,
            "total_games_pending": self.total_games_pending,
            "current_temperature_preset": self.current_temperature_preset,
            "regression_detected": self.regression_detected,
            "last_curriculum_rebalance": self.last_curriculum_rebalance,
            "last_calibration_time": self.last_calibration_time,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UnifiedLoopState":
        """Restore the state from the serialized representation."""
        state = cls()
        state.started_at = data.get("started_at", "")
        state.curriculum_weights = dict(data.get("curriculum_weights", {}))
        state.training_in_progress = bool(data.get("training_in_progress", False))
        state.training_config = data.get("training_config", "")
        state.training_started_at = float(data.get("training_started_at", 0.0))
        state.total_data_syncs = int(data.get("total_data_syncs", 0))
        state.total_training_runs = int(data.get("total_training_runs", 0))
        state.total_evaluations = int(data.get("total_evaluations", 0))
        state.total_promotions = int(data.get("total_promotions", 0))
        state.total_games_pending = int(data.get("total_games_pending", 0))
        state.current_temperature_preset = data.get("current_temperature_preset", "default")
        state.regression_detected = bool(data.get("regression_detected", False))
        state.last_curriculum_rebalance = float(data.get("last_curriculum_rebalance", 0.0))
        state.last_calibration_time = float(data.get("last_calibration_time", 0.0))

        for host_name, host_data in data.get("hosts", {}).items():
            state.hosts[host_name] = HostState(**host_data)
        for config_key, config_data in data.get("configs", {}).items():
            config_payload = dict(config_data)
            feedback_data = dict(config_payload.pop("feedback", {}))
            config_payload["feedback"] = FeedbackState(**feedback_data)
            state.configs[config_key] = ConfigState(**config_payload)
        return state


@dataclass
class ConfigPriorityQueue:
    """Minimal priority queue retained for unified-loop compatibility."""

    _trained_model_counts: dict[str, int] = field(default_factory=dict)

    def _update_trained_model_counts(self) -> None:
        """Legacy hook retained for training scheduler compatibility."""
        return None

    def get_prioritized_configs(
        self, configs: dict[str, ConfigState]
    ) -> list[tuple[str, float]]:
        """Return configs ordered by training need."""
        prioritized: list[tuple[str, float]] = []
        for config_key, config_state in configs.items():
            priority = float(config_state.games_since_training)
            priority += max(0.0, config_state.training_weight - 1.0) * 100.0
            prioritized.append((config_key, priority))
        return sorted(prioritized, key=lambda item: (-item[1], item[0]))


# =============================================================================
# Integrated Enhancements Factory
# =============================================================================

def create_integrated_manager_from_config(
    training_config: TrainingConfig,
    model: Any | None = None,
    board_type: str = "square8",
) -> IntegratedTrainingManager | None:
    """Create an IntegratedTrainingManager from TrainingConfig.

    Args:
        training_config: TrainingConfig with enhancement settings
        model: PyTorch model (optional, can be set later)
        board_type: Board type for augmentation

    Returns:
        IntegratedTrainingManager or None if disabled
    """
    if not training_config.use_integrated_enhancements:
        return None

    try:
        from app.training.integrated_enhancements import (
            IntegratedEnhancementsConfig,
            IntegratedTrainingManager,
        )

        # Map TrainingConfig to IntegratedEnhancementsConfig
        enhancement_config = IntegratedEnhancementsConfig(
            # Auxiliary Tasks
            auxiliary_tasks_enabled=training_config.auxiliary_tasks_enabled,
            aux_game_length_weight=training_config.aux_game_length_weight,
            aux_piece_count_weight=training_config.aux_piece_count_weight,
            aux_outcome_weight=training_config.aux_outcome_weight,
            # Gradient Surgery
            gradient_surgery_enabled=training_config.gradient_surgery_enabled,
            gradient_surgery_method=training_config.gradient_surgery_method,
            # Batch Scheduling
            batch_scheduling_enabled=training_config.batch_scheduling_enabled,
            batch_initial_size=training_config.batch_initial_size,
            batch_final_size=training_config.batch_final_size,
            batch_schedule_type=training_config.batch_schedule_type,
            # Background Evaluation
            background_eval_enabled=training_config.background_eval_enabled,
            eval_interval_steps=training_config.eval_interval_steps,
            eval_elo_checkpoint_threshold=training_config.eval_elo_checkpoint_threshold,
            # ELO Weighting
            elo_weighting_enabled=training_config.elo_weighting_enabled,
            elo_base_rating=training_config.elo_base_rating,
            elo_weight_scale=training_config.elo_weight_scale,
            elo_min_weight=training_config.elo_min_weight,
            elo_max_weight=training_config.elo_max_weight,
            # Curriculum Learning
            curriculum_enabled=training_config.curriculum_enabled,
            curriculum_auto_advance=training_config.curriculum_auto_advance,
            # Augmentation
            augmentation_enabled=training_config.augmentation_enabled,
            augmentation_mode=training_config.augmentation_mode,
            # Reanalysis
            reanalysis_enabled=training_config.reanalysis_enabled,
            reanalysis_blend_ratio=training_config.reanalysis_blend_ratio,
        )

        return IntegratedTrainingManager(enhancement_config, model, board_type)

    except ImportError as e:
        import logging
        logging.getLogger(__name__).warning(
            f"[Config] Failed to import integrated enhancements: {e}"
        )
        return None


# =============================================================================
# Integration with app.config.unified_config
# =============================================================================

def sync_with_unified_config(loop_config: UnifiedLoopConfig) -> UnifiedLoopConfig:
    """Sync defaults from app.config.unified_config to keep values aligned.

    This ensures that the unified loop uses the same canonical values as the
    rest of the codebase. Call this after loading a UnifiedLoopConfig.

    Args:
        loop_config: The config to sync

    Returns:
        The same config with any unset values populated from unified_config
    """
    try:
        from app.config.unified_config import get_config as get_unified_config

        unified = get_unified_config()

        # Sync training thresholds if using defaults
        if loop_config.training.trigger_threshold_games == 500:  # Default
            loop_config.training.trigger_threshold_games = unified.training.trigger_threshold_games

        # Sync promotion thresholds if using defaults
        if loop_config.promotion.elo_threshold == 25:  # Default
            loop_config.promotion.elo_threshold = int(unified.promotion.min_elo_improvement)

        return loop_config

    except ImportError:
        # app.config.unified_config not available, use local defaults
        return loop_config


def get_canonical_training_threshold() -> int:
    """Get the canonical training threshold from unified_config.

    Returns:
        Training threshold from unified_config, or default (500)
    """
    try:
        from app.config.unified_config import get_training_threshold
        return get_training_threshold()
    except ImportError:
        return 500  # Default
