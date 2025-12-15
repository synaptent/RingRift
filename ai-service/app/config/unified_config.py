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

logger = logging.getLogger(__name__)

# Default config path relative to ai-service root
DEFAULT_CONFIG_PATH = "config/unified_loop.yaml"

# Singleton instance
_config_instance: Optional[UnifiedConfig] = None


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion from remote hosts."""
    poll_interval_seconds: int = 60
    sync_method: str = "incremental"
    deduplication: bool = True
    min_games_per_sync: int = 5
    remote_db_pattern: str = "data/games/*.db"
    checksum_validation: bool = True
    retry_max_attempts: int = 3
    retry_base_delay_seconds: int = 5
    dead_letter_enabled: bool = True


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
class ClusterConfig:
    """Cluster orchestration settings (previously hardcoded in cluster_orchestrator.py)."""
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


@dataclass
class SelfplayConfig:
    """Selfplay settings (shared across all selfplay scripts)."""
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
class UnifiedConfig:
    """Master configuration class that loads from unified_loop.yaml.

    This is the SINGLE SOURCE OF TRUTH for all configuration values.
    """
    version: str = "1.1"

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

    # New unified sections (previously scattered as hardcoded constants)
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    ssh: SSHConfig = field(default_factory=SSHConfig)
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    tournament: TournamentConfig = field(default_factory=TournamentConfig)

    # Paths
    hosts_config_path: str = "config/remote_hosts.yaml"
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

        # Load new unified sections (previously hardcoded constants)
        if "cluster" in data:
            cluster_data = data["cluster"]
            config.cluster = ClusterConfig(
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

        if "selfplay" in data:
            sp_data = data["selfplay"]
            config.selfplay = SelfplayConfig(
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

        # Load paths
        config.hosts_config_path = data.get("hosts_config_path", config.hosts_config_path)
        config.elo_db = data.get("elo_db", config.elo_db)
        config.data_manifest_db = data.get("data_manifest_db", config.data_manifest_db)
        config.log_dir = data.get("log_dir", config.log_dir)
        config.metrics_enabled = data.get("metrics_enabled", config.metrics_enabled)
        config.metrics_port = data.get("metrics_port", config.metrics_port)

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
