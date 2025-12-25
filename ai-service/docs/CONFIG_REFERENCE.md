# Configuration Reference

This document provides a comprehensive reference for all configuration options in the RingRift AI service.

## Overview

Configuration is managed through `app/config/unified_config.py` which provides a single source of truth for all settings. Configuration is loaded from `config/unified_loop.yaml` and can be overridden via environment variables.

```python
from app.config.unified_config import get_config

config = get_config()
```

## Environment Variable Overrides

| Variable                              | Description                                 | Default                     |
| ------------------------------------- | ------------------------------------------- | --------------------------- |
| `RINGRIFT_CONFIG_PATH`                | Override config file path                   | `config/unified_loop.yaml`  |
| `RINGRIFT_TRAINING_THRESHOLD`         | Training trigger threshold                  | 500                         |
| `RINGRIFT_ELO_DB`                     | Elo database path                           | `data/elo/elo.db`           |
| `ADMIN_API_KEY`                       | Admin API key (`X-Admin-Key`) for /admin/\* | Generated per boot if unset |
| `CORS_ORIGINS`                        | Comma-separated CORS allowlist              | `*`                         |
| `GAME_REPLAY_DB_PATH`                 | Replay DB path for `/api/replay/*`          | `data/games/selfplay.db`    |
| `RINGRIFT_START_DAEMONS`              | Start daemon manager on boot (`1` = on)     | `0`                         |
| `RINGRIFT_ENV`                        | Environment label for error sanitization    | `development`               |
| `RINGRIFT_AI_TIMEOUT`                 | AI operation timeout (seconds)              | `30.0`                      |
| `RINGRIFT_AI_INSTANCE_CACHE`          | Enable AI instance cache (`1` = on)         | `1`                         |
| `RINGRIFT_AI_INSTANCE_CACHE_TTL_SEC`  | AI instance cache TTL (seconds)             | `1800`                      |
| `RINGRIFT_AI_INSTANCE_CACHE_MAX`      | Max cached AI instances                     | `512`                       |
| `RINGRIFT_TRAINED_HEURISTIC_PROFILES` | Path to trained heuristic profiles JSON     | None                        |

---

## DataIngestionConfig

Configuration for data ingestion from remote hosts.

| Option                             | Type | Default            | Description                                 |
| ---------------------------------- | ---- | ------------------ | ------------------------------------------- |
| `poll_interval_seconds`            | int  | 60                 | Interval between data sync polls            |
| `ephemeral_poll_interval_seconds`  | int  | 15                 | Aggressive sync interval for RAM disk hosts |
| `sync_method`                      | str  | "incremental"      | Sync method ("incremental" or "full")       |
| `deduplication`                    | bool | True               | Enable game deduplication                   |
| `min_games_per_sync`               | int  | 5                  | Minimum games before triggering sync        |
| `remote_db_pattern`                | str  | "data/games/\*.db" | Pattern for remote database files           |
| `sync_disabled`                    | bool | False              | Disable sync on this machine                |
| `use_external_sync`                | bool | False              | Use external unified_data_sync.py           |
| `checksum_validation`              | bool | True               | Validate checksums during sync              |
| `retry_max_attempts`               | int  | 3                  | Maximum retry attempts                      |
| `retry_base_delay_seconds`         | int  | 5                  | Base delay between retries                  |
| `dead_letter_enabled`              | bool | True               | Enable dead letter queue for failed syncs   |
| `wal_enabled`                      | bool | True               | Enable write-ahead logging                  |
| `wal_db_path`                      | str  | "data/sync_wal.db" | WAL database path                           |
| `elo_replication_enabled`          | bool | True               | Enable Elo database replication             |
| `elo_replication_interval_seconds` | int  | 60                 | Elo replication interval                    |

---

## TrainingConfig

Configuration for automatic training triggers. **This is the single source of truth for training thresholds.**

| Option                    | Type  | Default                               | Description                                     |
| ------------------------- | ----- | ------------------------------------- | ----------------------------------------------- |
| `trigger_threshold_games` | int   | 500                                   | Games needed to trigger training                |
| `min_interval_seconds`    | int   | 1200                                  | Minimum interval between training runs (20 min) |
| `max_concurrent_jobs`     | int   | 1                                     | Maximum concurrent training jobs                |
| `prefer_gpu_hosts`        | bool  | True                                  | Prefer GPU hosts for training                   |
| `nn_training_script`      | str   | "scripts/run_nn_training_baseline.py" | Neural network training script                  |
| `export_script`           | str   | "scripts/export_replay_dataset.py"    | Dataset export script                           |
| `hex_encoder_version`     | str   | "v3"                                  | Hex encoder version                             |
| `warm_start`              | bool  | True                                  | Enable warm start from previous weights         |
| `validation_split`        | float | 0.1                                   | Validation data split ratio                     |
| `nnue_min_games`          | int   | 10000                                 | Minimum games for NNUE training                 |
| `nnue_policy_min_games`   | int   | 5000                                  | Minimum games for NNUE policy training          |
| `cmaes_min_games`         | int   | 20000                                 | Minimum games for CMA-ES training               |

---

## EvaluationConfig

Configuration for model evaluation (shadow and full tournaments).

| Option                             | Type      | Default                                         | Description                            |
| ---------------------------------- | --------- | ----------------------------------------------- | -------------------------------------- |
| `shadow_interval_seconds`          | int       | 900                                             | Shadow evaluation interval (15 min)    |
| `shadow_games_per_config`          | int       | 15                                              | Games per configuration in shadow eval |
| `full_tournament_interval_seconds` | int       | 3600                                            | Full tournament interval (1 hour)      |
| `full_tournament_games`            | int       | 50                                              | Games in full tournament               |
| `baseline_models`                  | List[str] | ["random", "heuristic", "mcts_100", "mcts_500"] | Baseline models for comparison         |
| `min_games_for_elo`                | int       | 30                                              | Minimum games for Elo calculation      |
| `elo_k_factor`                     | int       | 32                                              | Elo K-factor                           |

---

## PromotionConfig

Configuration for automatic model promotion.

| Option                   | Type  | Default | Description                           |
| ------------------------ | ----- | ------- | ------------------------------------- |
| `auto_promote`           | bool  | True    | Enable automatic promotion            |
| `elo_threshold`          | int   | 25      | Minimum Elo improvement for promotion |
| `min_games`              | int   | 50      | Minimum games before promotion        |
| `significance_level`     | float | 0.05    | Statistical significance level        |
| `sync_to_cluster`        | bool  | True    | Sync promoted models to cluster       |
| `cooldown_seconds`       | int   | 1800    | Cooldown between promotions (30 min)  |
| `max_promotions_per_day` | int   | 10      | Maximum promotions per day            |
| `regression_test`        | bool  | True    | Run regression tests before promotion |

---

## CurriculumConfig

Configuration for adaptive curriculum (Elo-weighted training).

| Option                       | Type  | Default | Description                            |
| ---------------------------- | ----- | ------- | -------------------------------------- |
| `adaptive`                   | bool  | True    | Enable adaptive curriculum             |
| `rebalance_interval_seconds` | int   | 3600    | Curriculum rebalance interval (1 hour) |
| `max_weight_multiplier`      | float | 1.5     | Maximum weight multiplier              |
| `min_weight_multiplier`      | float | 0.7     | Minimum weight multiplier              |
| `ema_alpha`                  | float | 0.3     | EMA smoothing factor                   |
| `min_games_for_weight`       | int   | 100     | Minimum games for weight calculation   |
| `rebalance_on_elo_change`    | bool  | True    | Rebalance on significant Elo changes   |
| `elo_change_threshold`       | int   | 50      | Elo change threshold for rebalance     |

---

## SafeguardsConfig

Process safeguards to prevent uncoordinated process sprawl.

| Option                          | Type | Default       | Description                      |
| ------------------------------- | ---- | ------------- | -------------------------------- |
| `max_python_processes_per_host` | int  | 20            | Max Python processes per host    |
| `max_selfplay_processes`        | int  | 2             | Max selfplay processes           |
| `max_tournament_processes`      | int  | 1             | Max tournament processes         |
| `max_training_processes`        | int  | 1             | Max training processes           |
| `single_orchestrator`           | bool | True          | Enforce single orchestrator      |
| `orchestrator_host`             | str  | "lambda-h100" | Designated orchestrator host     |
| `kill_orphans_on_start`         | bool | True          | Kill orphan processes on startup |
| `process_watchdog`              | bool | True          | Enable process watchdog          |
| `watchdog_interval_seconds`     | int  | 60            | Watchdog check interval          |
| `max_process_age_hours`         | int  | 4             | Maximum process age before kill  |
| `max_subprocess_depth`          | int  | 2             | Maximum subprocess nesting depth |
| `subprocess_timeout_seconds`    | int  | 3600          | Subprocess timeout (1 hour)      |

---

## ClusterConfig

Cluster orchestration settings.

| Option                           | Type | Default | Description                           |
| -------------------------------- | ---- | ------- | ------------------------------------- |
| `target_selfplay_games_per_hour` | int  | 1000    | Target selfplay rate                  |
| `health_check_interval_seconds`  | int  | 60      | Health check interval                 |
| `sync_interval_seconds`          | int  | 300     | Data sync interval (5 min)            |
| `sync_interval`                  | int  | 6       | Sync every N iterations               |
| `model_sync_interval`            | int  | 12      | Model sync every N iterations         |
| `model_sync_enabled`             | bool | True    | Enable model sync                     |
| `elo_calibration_interval`       | int  | 72      | Elo calibration interval (iterations) |
| `elo_calibration_games`          | int  | 50      | Games per Elo calibration             |
| `elo_curriculum_enabled`         | bool | True    | Enable Elo-driven curriculum          |
| `elo_match_window`               | int  | 200     | Elo match window                      |
| `elo_underserved_threshold`      | int  | 100     | Underserved config threshold          |
| `auto_scale_interval`            | int  | 12      | Auto-scale check interval             |
| `underutilized_cpu_threshold`    | int  | 30      | CPU underutilization threshold        |
| `underutilized_python_jobs`      | int  | 10      | Min jobs for scaling                  |
| `scale_up_games_per_host`        | int  | 50      | Games per host for scale-up           |
| `adaptive_games_min`             | int  | 30      | Minimum adaptive games                |
| `adaptive_games_max`             | int  | 150     | Maximum adaptive games                |

---

## SSHConfig

SSH execution settings shared across orchestrators.

| Option                              | Type  | Default | Description                        |
| ----------------------------------- | ----- | ------- | ---------------------------------- |
| `max_retries`                       | int   | 3       | Maximum SSH retries                |
| `base_delay_seconds`                | float | 2.0     | Base retry delay                   |
| `max_delay_seconds`                 | float | 30.0    | Maximum retry delay                |
| `connect_timeout_seconds`           | int   | 10      | SSH connection timeout             |
| `command_timeout_seconds`           | int   | 3600    | Command execution timeout (1 hour) |
| `transport_command_timeout_seconds` | int   | 30      | P2P transport command timeout      |
| `retry_delay_seconds`               | float | 1.0     | Retry delay between attempts       |
| `address_cache_ttl_seconds`         | int   | 300     | Address cache TTL (5 min)          |

---

## SlurmConfig

Slurm execution settings for HPC clusters.

| Option                    | Type | Default           | Description                   |
| ------------------------- | ---- | ----------------- | ----------------------------- |
| `enabled`                 | bool | False             | Enable Slurm backend          |
| `partition_training`      | str  | "gpu-train"       | Training partition            |
| `partition_selfplay`      | str  | "gpu-selfplay"    | Selfplay partition            |
| `partition_tournament`    | str  | "cpu-eval"        | Tournament partition          |
| `account`                 | str  | None              | Slurm account                 |
| `qos`                     | str  | None              | Quality of service            |
| `default_time_training`   | str  | "08:00:00"        | Default training time limit   |
| `default_time_selfplay`   | str  | "02:00:00"        | Default selfplay time limit   |
| `default_time_tournament` | str  | "02:00:00"        | Default tournament time limit |
| `gpus_training`           | int  | 1                 | GPUs for training             |
| `cpus_training`           | int  | 16                | CPUs for training             |
| `mem_training`            | str  | "64G"             | Memory for training           |
| `gpus_selfplay`           | int  | 0                 | GPUs for selfplay             |
| `cpus_selfplay`           | int  | 8                 | CPUs for selfplay             |
| `mem_selfplay`            | str  | "16G"             | Memory for selfplay           |
| `job_dir`                 | str  | "data/slurm/jobs" | Slurm job directory           |
| `log_dir`                 | str  | "data/slurm/logs" | Slurm log directory           |
| `poll_interval_seconds`   | int  | 20                | Job status poll interval      |

---

## SafetyConfig

Safety thresholds to prevent bad models from being promoted.

| Option                     | Type  | Default | Description                |
| -------------------------- | ----- | ------- | -------------------------- |
| `overfit_threshold`        | float | 0.15    | Max train/val loss gap     |
| `min_memory_gb`            | int   | 64      | Minimum RAM required       |
| `max_consecutive_failures` | int   | 3       | Failures before stopping   |
| `parity_failure_rate_max`  | float | 0.10    | Max parity failure rate    |
| `data_quality_score_min`   | float | 0.70    | Minimum data quality score |

---

## PlateauDetectionConfig

Plateau detection and automatic hyperparameter search.

| Option                           | Type  | Default | Description                          |
| -------------------------------- | ----- | ------- | ------------------------------------ |
| `elo_plateau_threshold`          | float | 15.0    | Elo gain below this triggers plateau |
| `elo_plateau_lookback`           | int   | 5       | Evaluations to look back             |
| `win_rate_degradation_threshold` | float | 0.40    | Win rate degradation threshold       |
| `plateau_count_for_cmaes`        | int   | 2       | Plateaus before CMA-ES trigger       |
| `plateau_count_for_nas`          | int   | 4       | Plateaus before NAS trigger          |

---

## AlertingConfig

Alerting thresholds for monitoring.

| Option                   | Type | Default | Description                |
| ------------------------ | ---- | ------- | -------------------------- |
| `sync_failure_threshold` | int  | 5       | Sync failures before alert |
| `training_timeout_hours` | int  | 4       | Training timeout           |
| `elo_drop_threshold`     | int  | 50      | Elo drop alert threshold   |
| `games_per_hour_min`     | int  | 100     | Minimum games per hour     |

---

## ReplayBufferConfig

Prioritized experience replay buffer settings.

| Option                     | Type  | Default | Description                       |
| -------------------------- | ----- | ------- | --------------------------------- |
| `priority_alpha`           | float | 0.6     | Priority exponent                 |
| `importance_beta`          | float | 0.4     | Importance sampling exponent      |
| `capacity`                 | int   | 100000  | Maximum buffer capacity           |
| `rebuild_interval_seconds` | int   | 7200    | Buffer rebuild interval (2 hours) |

---

## DistributedConfig

Distributed system component settings.

| Option                       | Type | Default | Description                      |
| ---------------------------- | ---- | ------- | -------------------------------- |
| `degraded_failure_threshold` | int  | 2       | Failures before node is degraded |
| `offline_failure_threshold`  | int  | 5       | Failures before node is offline  |
| `recovery_success_threshold` | int  | 2       | Successes needed for recovery    |

---

## YAML Configuration File

The default configuration file is at `config/unified_loop.yaml`. Example structure:

```yaml
data_ingestion:
  poll_interval_seconds: 60
  sync_method: incremental

training:
  trigger_threshold_games: 500
  min_interval_seconds: 1200
  prefer_gpu_hosts: true

evaluation:
  shadow_games_per_config: 15
  full_tournament_games: 50

promotion:
  auto_promote: true
  elo_threshold: 25

cluster:
  target_selfplay_games_per_hour: 1000
  health_check_interval_seconds: 60
```

---

## Convenience Functions

The config module provides several convenience functions:

```python
from app.config.unified_config import (
    get_config,                  # Get singleton config
    get_training_threshold,      # Training game threshold
    get_elo_db_path,            # Elo database path
    get_min_elo_improvement,     # Promotion Elo threshold
    get_default_board_configs,   # All 9 board configurations
)

# Access nested config
config = get_config()
threshold = config.training.trigger_threshold_games

# Or use convenience functions
threshold = get_training_threshold()
```
