# Event Payload Schemas (Auto-Generated)

Generated: January 2026 (Sprint 14)

Total events: 253
Previously documented: 23
Newly documented: 230

---

## Training Events (16)

### RESOURCE_CONSTRAINT 🆕

**Value**: `resource_constraint`

**Payload Fields**:

| Field             | Type   |
| ----------------- | ------ |
| `ram_total_gb`    | any    |
| `ram_used_gb`     | any    |
| `ram_utilization` | any    |
| `resource_type`   | string |
| `source`          | any    |
| `timestamp`       | any    |

**Emitters**:

- `app/coordination/memory_monitor_daemon.py:510`

---

### RESOURCE_CONSTRAINT_DETECTED 🆕

**Value**: `resource_constraint_detected`

**Payload**: (no fields detected)

---

### TRAINING_BLOCKED_BY_QUALITY

**Value**: `training_blocked_by_quality`

**Payload Fields**:

| Field             | Type          |
| ----------------- | ------------- |
| `config_key`      | any           |
| `data_age_hours`  | any           |
| `db_path`         | string (path) |
| `games_available` | any           |
| `quality_deficit` | any           |
| `quality_score`   | any           |
| `reason`          | string        |
| `recommendation`  | string        |
| `source`          | string        |
| `threshold`       | number        |
| `threshold_hours` | any           |
| `timestamp`       | timestamp     |

**Emitters**:

- `app/coordination/data_pipeline_orchestrator.py:2200`
- `app/coordination/data_pipeline_orchestrator.py:2854`
- `app/coordination/feedback_loop_controller.py:1178`

---

### TRAINING_COMPLETED

**Value**: `training_completed`

**Payload Fields**:

| Field        | Type          |
| ------------ | ------------- |
| `config_key` | any           |
| `final_loss` | any           |
| `model_path` | string (path) |
| `source`     | string        |

**Emitters**:

- `scripts/p2p_orchestrator.py:3967`

---

### TRAINING_EARLY_STOPPED 🆕

**Value**: `training_early_stopped`

**Payload**: (no fields detected)

---

### TRAINING_FAILED 🆕

**Value**: `training_failed`

**Payload**: (no fields detected)

---

### TRAINING_LOCK_ACQUIRED 🆕

**Value**: `training_lock_acquired`

**Payload**: (no fields detected)

---

### TRAINING_LOSS_ANOMALY 🆕

**Value**: `training_loss_anomaly`

**Payload**: (no fields detected)

---

### TRAINING_LOSS_TREND 🆕

**Value**: `training_loss_trend`

**Payload**: (no fields detected)

---

### TRAINING_PROGRESS 🆕

**Value**: `training_progress`

**Payload**: (no fields detected)

---

### TRAINING_ROLLBACK_COMPLETED 🆕

**Value**: `training_rollback_completed`

**Payload**: (no fields detected)

---

### TRAINING_ROLLBACK_NEEDED 🆕

**Value**: `training_rollback_needed`

**Payload**: (no fields detected)

---

### TRAINING_SLOT_UNAVAILABLE 🆕

**Value**: `training_slot_unavailable`

**Payload**: (no fields detected)

---

### TRAINING_STARTED 🆕

**Value**: `training_started`

**Payload**: (no fields detected)

---

### TRAINING_THRESHOLD_REACHED

**Value**: `training_threshold`

**Payload Fields**:

| Field         | Type    |
| ------------- | ------- |
| `board_type`  | any     |
| `config`      | any     |
| `num_players` | integer |
| `priority`    | any     |
| `reason`      | string  |

**Emitters**:

- `app/coordination/pipeline_event_handler_mixin.py:422`
- `app/coordination/pipeline_event_handler_mixin.py:477`
- `scripts/master_loop.py:1994`

---

### TRAINING_TIMEOUT_REACHED 🆕

**Value**: `training_timeout_reached`

**Payload Fields**:

| Field           | Type      |
| --------------- | --------- |
| `config_key`    | any       |
| `grace_seconds` | any       |
| `pid`           | any       |
| `timeout_hours` | any       |
| `timestamp`     | timestamp |

**Emitters**:

- `app/coordination/training_trigger_daemon.py:3553`

---

## Selfplay Events (9)

### GAME_SYNCED 🆕

**Value**: `game_synced`

**Payload**: (no fields detected)

---

### NEW_GAMES_AVAILABLE

**Value**: `new_games`

**Payload Fields**:

| Field         | Type      |
| ------------- | --------- |
| `config_key`  | any       |
| `count`       | any       |
| `source`      | any       |
| `source_node` | any       |
| `timestamp`   | timestamp |
| `trigger`     | string    |

**Emitters**:

- `app/coordination/data_pipeline_orchestrator.py:2666`
- `app/coordination/owc_import_daemon.py:512`
- `app/coordination/unified_data_plane_daemon.py:596`

---

### ORPHAN_GAMES_DETECTED 🆕

**Value**: `orphan_games_detected`

**Payload**: (no fields detected)

---

### ORPHAN_GAMES_REGISTERED 🆕

**Value**: `orphan_games_registered`

**Payload**: (no fields detected)

---

### P2P_SELFPLAY_SCALED 🆕

**Value**: `p2p_selfplay_scaled`

**Payload**: (no fields detected)

---

### SELFPLAY_ALLOCATION_UPDATED 🆕

**Value**: `selfplay_allocation_updated`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/selfplay_scheduler.py:4138`

---

### SELFPLAY_COMPLETE

**Value**: `selfplay_complete`

**Payload**: (no fields detected)

---

### SELFPLAY_RATE_CHANGED 🆕

**Value**: `selfplay_rate_changed`

**Payload Fields**:

| Field            | Type   |
| ---------------- | ------ |
| `change_percent` | any    |
| `config_key`     | any    |
| `new_rate`       | any    |
| `old_rate`       | any    |
| `reason`         | string |

**Emitters**:

- `app/coordination/selfplay_scheduler.py:2027`
- `app/coordination/selfplay_scheduler.py:2897`

---

### SELFPLAY_TARGET_UPDATED 🆕

**Value**: `selfplay_target_updated`

**Payload Fields**:

| Field                 | Type    |
| --------------------- | ------- |
| `anomaly_count`       | integer |
| `board_type`          | any     |
| `config_key`          | any     |
| `curriculum_weight`   | any     |
| `dispatched`          | any     |
| `elo_gap`             | number  |
| `exploration_boost`   | number  |
| `momentum_multiplier` | any     |
| `node_id`             | any     |
| `num_players`         | integer |
| `priority`            | string  |
| `reason`              | string  |
| `search_budget`       | any     |
| `source`              | string  |
| `target_games`        | integer |
| `velocity`            | number  |

**Emitters**:

- `app/coordination/feedback_loop_controller.py:1896`
- `app/coordination/feedback_loop_controller.py:2065`
- `app/coordination/selfplay_scheduler.py:2617`

---

## Evaluation Events (11)

### ELO_SIGNIFICANT_CHANGE 🆕

**Value**: `elo_significant_change`

**Payload**: (no fields detected)

---

### ELO_UPDATED

**Value**: `elo_updated`

**Payload**: (no fields detected)

---

### ELO_VELOCITY_CHANGED 🆕

**Value**: `elo_velocity_changed`

**Payload**: (no fields detected)

---

### EVALUATION_BACKPRESSURE

**Value**: `evaluation_backpressure`

**Payload**: (no fields detected)

---

### EVALUATION_BACKPRESSURE_RELEASED

**Value**: `evaluation_backpressure_released`

**Payload**: (no fields detected)

---

### EVALUATION_COMPLETED

**Value**: `evaluation_completed`

**Payload**: (no fields detected)

---

### EVALUATION_FAILED 🆕

**Value**: `evaluation_failed`

**Payload Fields**:

| Field        | Type          |
| ------------ | ------------- |
| `config_key` | any           |
| `model_path` | string (path) |
| `reason`     | string        |
| `source`     | string        |

**Emitters**:

- `tests/integration/coordination/test_full_event_chain_e2e.py:708`

---

### EVALUATION_PROGRESS 🆕

**Value**: `evaluation_progress`

**Payload**: (no fields detected)

---

### EVALUATION_STARTED 🆕

**Value**: `evaluation_started`

**Payload**: (no fields detected)

---

### HARNESS_EVALUATION_COMPLETED 🆕

**Value**: `harness_evaluation_completed`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/tournament_daemon.py:820`

---

### MODEL_EVALUATION_BLOCKED 🆕

**Value**: `model_evaluation_blocked`

**Payload**: (no fields detected)

---

## Model Events (10)

### CHECKPOINT_LOADED 🆕

**Value**: `checkpoint_loaded`

**Payload**: (no fields detected)

---

### CHECKPOINT_SAVED 🆕

**Value**: `checkpoint_saved`

**Payload**: (no fields detected)

---

### MODEL_CORRUPTED 🆕

**Value**: `model_corrupted`

**Payload**: (no fields detected)

---

### MODEL_DISTRIBUTION_COMPLETE 🆕

**Value**: `model_distribution_complete`

**Payload**: (no fields detected)

---

### MODEL_DISTRIBUTION_FAILED 🆕

**Value**: `model_distribution_failed`

**Payload**: (no fields detected)

---

### MODEL_DISTRIBUTION_STARTED 🆕

**Value**: `model_distribution_started`

**Payload**: (no fields detected)

---

### MODEL_PROMOTED

**Value**: `model_promoted`

**Payload**: (no fields detected)

---

### MODEL_SYNC_REQUESTED 🆕

**Value**: `model_sync_requested`

**Payload**: (no fields detected)

---

### MODEL_UPDATED 🆕

**Value**: `model_updated`

**Payload**: (no fields detected)

---

### P2P_MODEL_SYNCED 🆕

**Value**: `p2p_model_synced`

**Payload**: (no fields detected)

---

## Data Events (16)

### DATABASE_CREATED 🆕

**Value**: `database_created`

**Payload**: (no fields detected)

---

### DATA_BACKUP_COMPLETED 🆕

**Value**: `data_backup_completed`

**Payload**: (no fields detected)

---

### DATA_FRESH 🆕

**Value**: `data_fresh`

**Payload**: (no fields detected)

---

### DATA_QUALITY_ALERT 🆕

**Value**: `data_quality_alert`

**Payload**: (no fields detected)

---

### DATA_STALE 🆕

**Value**: `data_stale`

**Payload**: (no fields detected)

---

### DATA_SYNC_COMPLETED

**Value**: `sync_completed`

**Payload Fields**:

| Field           | Type      |
| --------------- | --------- |
| `bytes_synced`  | any       |
| `config_key`    | any       |
| `duration`      | any       |
| `entry_count`   | any       |
| `games_synced`  | any       |
| `host`          | string    |
| `node_id`       | any       |
| `reason`        | any       |
| `source`        | string    |
| `source_node`   | any       |
| `sources_count` | integer   |
| `sync_type`     | string    |
| `target_nodes`  | any       |
| `timestamp`     | timestamp |

**Emitters**:

- `app/coordination/dual_backup_daemon.py:498`
- `app/coordination/owc_import_daemon.py:529`
- `app/coordination/s3_import_daemon.py:464`

---

### DATA_SYNC_FAILED 🆕

**Value**: `sync_failed`

**Payload Fields**:

| Field          | Type   |
| -------------- | ------ |
| `config_key`   | any    |
| `error`        | any    |
| `host`         | string |
| `reason`       | any    |
| `source`       | string |
| `source_node`  | any    |
| `target_nodes` | any    |

**Emitters**:

- `app/coordination/unified_data_plane_daemon.py:607`
- `scripts/p2p/loops/training_sync_loop.py:198`

---

### DATA_SYNC_STARTED 🆕

**Value**: `sync_started`

**Payload Fields**:

| Field       | Type   |
| ----------- | ------ |
| `host`      | string |
| `source`    | string |
| `sync_type` | string |

**Emitters**:

- `scripts/p2p/loops/training_sync_loop.py:159`

---

### EXPORT_VALIDATION_FAILED 🆕

**Value**: `export_validation_failed`

**Payload Fields**:

| Field         | Type      |
| ------------- | --------- |
| `board_type`  | any       |
| `config_key`  | any       |
| `num_players` | integer   |
| `reason`      | any       |
| `source`      | string    |
| `timestamp`   | timestamp |

**Emitters**:

- `app/coordination/auto_export_daemon.py:1109`
- `app/coordination/auto_export_daemon.py:1252`

---

### HIGH_QUALITY_DATA_AVAILABLE 🆕

**Value**: `high_quality_data_available`

**Payload**: (no fields detected)

---

### LOW_QUALITY_DATA_WARNING 🆕

**Value**: `low_quality_data_warning`

**Payload**: (no fields detected)

---

### NPZ_COMBINATION_COMPLETE

**Value**: `npz_combination_complete`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/npz_combination_daemon.py:270`
- `app/coordination/pipeline_actions.py:754`

---

### NPZ_COMBINATION_FAILED 🆕

**Value**: `npz_combination_failed`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/npz_combination_daemon.py:293`
- `app/coordination/pipeline_actions.py:780`

---

### NPZ_COMBINATION_STARTED 🆕

**Value**: `npz_combination_started`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/npz_combination_daemon.py:317`

---

### NPZ_EXPORT_COMPLETE 🆕

**Value**: `npz_export_complete`

**Payload**: (no fields detected)

---

### NPZ_EXPORT_STARTED 🆕

**Value**: `npz_export_started`

**Payload**: (no fields detected)

---

## Sync Events (9)

### DISTRIBUTION_INCOMPLETE 🆕

**Value**: `distribution_incomplete`

**Payload Fields**:

| Field             | Type          |
| ----------------- | ------------- |
| `actual_nodes`    | integer       |
| `model_path`      | string (path) |
| `required_nodes`  | any           |
| `timeout_seconds` | timestamp     |

**Emitters**:

- `app/coordination/unified_distribution_daemon.py:602`

---

### QUALITY_DISTRIBUTION_CHANGED 🆕

**Value**: `quality_distribution_changed`

**Payload**: (no fields detected)

---

### REPLICATION_ALERT 🆕

**Value**: `replication_alert`

**Payload**: (no fields detected)

---

### SYNC_CHECKSUM_FAILED 🆕

**Value**: `sync_checksum_failed`

**Payload**: (no fields detected)

---

### SYNC_FAILURE_CRITICAL 🆕

**Value**: `sync_failure_critical`

**Payload**: (no fields detected)

---

### SYNC_NODE_UNREACHABLE 🆕

**Value**: `sync_node_unreachable`

**Payload**: (no fields detected)

---

### SYNC_REQUEST

**Value**: `sync_request`

**Payload**: (no fields detected)

---

### SYNC_STALLED 🆕

**Value**: `sync_stalled`

**Payload**: (no fields detected)

---

### SYNC_TRIGGERED 🆕

**Value**: `sync_triggered`

**Payload Fields**:

| Field               | Type      |
| ------------------- | --------- |
| `game_count`        | any       |
| `host`              | any       |
| `nodes_reconnected` | any       |
| `partitions_healed` | any       |
| `paths`             | any       |
| `priority`          | string    |
| `reason`            | string    |
| `source`            | string    |
| `timestamp`         | timestamp |
| `trigger`           | string    |

**Emitters**:

- `app/coordination/data_pipeline_orchestrator.py:2640`
- `app/coordination/data_pipeline_orchestrator.py:3138`
- `app/coordination/training_data_sync_daemon.py:459`

---

## Curriculum Events (4)

### CURRICULUM_ADVANCED 🆕

**Value**: `curriculum_advanced`

**Payload**: (no fields detected)

---

### CURRICULUM_ADVANCEMENT_NEEDED 🆕

**Value**: `curriculum_advancement_needed`

**Payload**: (no fields detected)

---

### CURRICULUM_PROPAGATE 🆕

**Value**: `curriculum_propagate`

**Payload**: (no fields detected)

---

### CURRICULUM_REBALANCED

**Value**: `curriculum_rebalanced`

**Payload**: (no fields detected)

---

## Quality Events (5)

### QUALITY_CHECK_REQUESTED 🆕

**Value**: `quality_check_requested`

**Payload**: (no fields detected)

---

### QUALITY_DEGRADED 🆕

**Value**: `quality_degraded`

**Payload**: (no fields detected)

---

### QUALITY_FEEDBACK_ADJUSTED 🆕

**Value**: `quality_feedback_adjusted`

**Payload**: (no fields detected)

---

### QUALITY_PENALTY_APPLIED 🆕

**Value**: `quality_penalty_applied`

**Payload**: (no fields detected)

---

### QUALITY_SCORE_UPDATED 🆕

**Value**: `quality_score_updated`

**Payload**: (no fields detected)

---

## Health Events (36)

### CLUSTER_P2P_RECOVERY_COMPLETED 🆕

**Value**: `cluster_p2p_recovery_completed`

**Payload**: (no fields detected)

**Emitters**:

- `scripts/master_loop.py:1843`

---

### CLUSTER_UTILIZATION_RECOVERED 🆕

**Value**: `cluster_utilization_recovered`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/cluster_utilization_watchdog.py:366`

---

### COORDINATOR_HEALTHY 🆕

**Value**: `coordinator_healthy`

**Payload**: (no fields detected)

---

### COORDINATOR_HEALTH_DEGRADED 🆕

**Value**: `coordinator_health_degraded`

**Payload**: (no fields detected)

---

### COORDINATOR_INIT_FAILED 🆕

**Value**: `coordinator_init_failed`

**Payload**: (no fields detected)

---

### COORDINATOR_UNHEALTHY 🆕

**Value**: `coordinator_unhealthy`

**Payload**: (no fields detected)

---

### DAEMON_PERMANENTLY_FAILED 🆕

**Value**: `daemon_permanently_failed`

**Payload**: (no fields detected)

---

### ERROR 🆕

**Value**: `error`

**Payload**: (no fields detected)

---

### HANDLER_FAILED 🆕

**Value**: `handler_failed`

**Payload**: (no fields detected)

---

### HEALTH_ALERT 🆕

**Value**: `health_alert`

**Payload Fields**:

| Field    | Type   |
| -------- | ------ |
| `action` | string |
| `alert`  | string |
| `reason` | string |

**Emitters**:

- `scripts/master_loop.py:1727`

---

### HEALTH_CHECK_FAILED

**Value**: `health_check_failed`

**Payload**: (no fields detected)

---

### HEALTH_CHECK_PASSED 🆕

**Value**: `health_check_passed`

**Payload**: (no fields detected)

---

### JOB_SPAWN_FAILED 🆕

**Value**: `job_spawn_failed`

**Payload**: (no fields detected)

---

### NODE_PROVISION_FAILED 🆕

**Value**: `node_provision_failed`

**Payload**: (no fields detected)

---

### NODE_RECOVERED

**Value**: `node_recovered`

**Payload Fields**:

| Field             | Type   |
| ----------------- | ------ |
| `address`         | any    |
| `node_id`         | any    |
| `recovery_source` | string |
| `timestamp`       | any    |

**Emitters**:

- `scripts/p2p/loops/peer_recovery_loop.py:283`

---

### NODE_UNHEALTHY 🆕

**Value**: `node_unhealthy`

**Payload Fields**:

| Field                    | Type   |
| ------------------------ | ------ |
| `disk_used_percent`      | any    |
| `error`                  | any    |
| `gpu_utilization`        | any    |
| `node_id`                | any    |
| `node_ip`                | any    |
| `node_name`              | any    |
| `reason`                 | string |
| `stall_duration_seconds` | any    |

**Emitters**:

- `app/coordination/unified_health_manager.py:1362`
- `app/monitoring/unified_cluster_monitor.py:853`

---

### P2P_CLUSTER_HEALTHY 🆕

**Value**: `p2p_cluster_healthy`

**Payload**: (no fields detected)

---

### P2P_CLUSTER_UNHEALTHY 🆕

**Value**: `p2p_cluster_unhealthy`

**Payload**: (no fields detected)

---

### P2P_HEALTH_RECOVERED 🆕

**Value**: `p2p_health_recovered`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/p2p_recovery_daemon.py:1368`

---

### P2P_RECOVERY_NEEDED 🆕

**Value**: `p2p_recovery_needed`

**Payload Fields**:

| Field                  | Type      |
| ---------------------- | --------- |
| `consecutive_failures` | any       |
| `escalation_level`     | any       |
| `reason`               | any       |
| `timestamp`            | timestamp |

**Emitters**:

- `scripts/p2p/partition_healer.py:855`

---

### PARITY_FAILURE_RATE_CHANGED 🆕

**Value**: `parity_failure_rate_changed`

**Payload**: (no fields detected)

---

### PARTITION_HEALING_FAILED 🆕

**Value**: `partition_healing_failed`

**Payload Fields**:

| Field       | Type      |
| ----------- | --------- |
| `error`     | any       |
| `timestamp` | timestamp |

**Emitters**:

- `scripts/p2p/partition_healer.py:797`

---

### PROGRESS_RECOVERED

**Value**: `progress_recovered`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/progress_watchdog_daemon.py:403`

---

### PROMOTION_FAILED 🆕

**Value**: `promotion_failed`

**Payload**: (no fields detected)

---

### QUALITY_CHECK_FAILED 🆕

**Value**: `quality_check_failed`

**Payload**: (no fields detected)

---

### QUORUM_RECOVERY_STARTED 🆕

**Value**: `quorum_recovery_started`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/p2p_recovery_daemon.py:339`

---

### RECOVERY_COMPLETED 🆕

**Value**: `recovery_completed`

**Payload**: (no fields detected)

---

### RECOVERY_FAILED 🆕

**Value**: `recovery_failed`

**Payload**: (no fields detected)

---

### RECOVERY_INITIATED 🆕

**Value**: `recovery_initiated`

**Payload**: (no fields detected)

---

### REPAIR_FAILED 🆕

**Value**: `repair_failed`

**Payload**: (no fields detected)

---

### SOCKET_LEAK_RECOVERED 🆕

**Value**: `socket_leak_recovered`

**Payload**: (no fields detected)

---

### SSH_LIVENESS_CHECK_FAILED 🆕

**Value**: `ssh_liveness_check_failed`

**Payload**: (no fields detected)

---

### SSH_NODE_RECOVERED 🆕

**Value**: `ssh_node_recovered`

**Payload**: (no fields detected)

---

### TASK_FAILED 🆕

**Value**: `task_failed`

**Payload Fields**:

| Field         | Type    |
| ------------- | ------- |
| `board_type`  | any     |
| `config_key`  | any     |
| `error`       | any     |
| `node_id`     | any     |
| `num_players` | integer |
| `task_id`     | any     |
| `task_type`   | any     |

**Emitters**:

- `scripts/p2p_orchestrator.py:14501`

---

### WORK_FAILED 🆕

**Value**: `work_failed`

**Payload**: (no fields detected)

---

### WORK_QUEUE_RECOVERED 🆕

**Value**: `work_queue_recovered`

**Payload Fields**:

| Field                    | Type |
| ------------------------ | ---- |
| `recovery_time`          | any  |
| `stall_duration_seconds` | any  |

**Emitters**:

- `scripts/p2p/loops/job_loops.py:1123`

---

## P2P Events (24)

### CLUSTER_CAPACITY_CHANGED 🆕

**Value**: `cluster_capacity_changed`

**Payload**: (no fields detected)

---

### CLUSTER_STALL_DETECTED 🆕

**Value**: `cluster_stall_detected`

**Payload Fields**:

| Field             | Type      |
| ----------------- | --------- |
| `stall_threshold` | any       |
| `stalled_nodes`   | any       |
| `timestamp`       | timestamp |

**Emitters**:

- `app/coordination/cluster_status_monitor.py:1044`

---

### CLUSTER_STATUS_CHANGED 🆕

**Value**: `cluster_status_changed`

**Payload Fields**:

| Field           | Type |
| --------------- | ---- |
| `alerts`        | any  |
| `healthy`       | any  |
| `healthy_nodes` | any  |
| `node_count`    | any  |

**Emitters**:

- `app/monitoring/unified_cluster_monitor.py:810`

---

### CLUSTER_UNDERUTILIZED 🆕

**Value**: `cluster_underutilized`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/cluster_utilization_watchdog.py:337`

---

### LEADER_ELECTED 🆕

**Value**: `leader_elected`

**Payload**: (no fields detected)

---

### LEADER_HEARTBEAT_MISSING 🆕

**Value**: `leader_heartbeat_missing`

**Payload**: (no fields detected)

---

### LEADER_LEASE_EXPIRED 🆕

**Value**: `leader_lease_expired`

**Payload**: (no fields detected)

---

### LEADER_LOST 🆕

**Value**: `leader_lost`

**Payload**: (no fields detected)

---

### LEADER_STEPDOWN 🆕

**Value**: `leader_stepdown`

**Payload**: (no fields detected)

---

### NODE_ACTIVATED 🆕

**Value**: `node_activated`

**Payload**: (no fields detected)

---

### NODE_CAPACITY_UPDATED 🆕

**Value**: `node_capacity_updated`

**Payload**: (no fields detected)

---

### NODE_INCOMPATIBLE_WITH_WORKLOAD 🆕

**Value**: `node_incompatible_with_workload`

**Payload**: (no fields detected)

---

### NODE_OVERLOADED 🆕

**Value**: `node_overloaded`

**Payload**: (no fields detected)

---

### NODE_PROVISIONED 🆕

**Value**: `node_provisioned`

**Payload**: (no fields detected)

---

### NODE_RETIRED 🆕

**Value**: `node_retired`

**Payload**: (no fields detected)

---

### NODE_SUSPECT 🆕

**Value**: `node_suspect`

**Payload**: (no fields detected)

---

### NODE_TERMINATED 🆕

**Value**: `node_terminated`

**Payload**: (no fields detected)

---

### P2P_NODES_DEAD 🆕

**Value**: `p2p_nodes_dead`

**Payload**: (no fields detected)

---

### P2P_NODE_DEAD 🆕

**Value**: `p2p_node_dead`

**Payload**: (no fields detected)

---

### QUORUM_AT_RISK 🆕

**Value**: `quorum_at_risk`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/voter_health_daemon.py:598`

---

### QUORUM_LOST 🆕

**Value**: `quorum_lost`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/voter_health_daemon.py:567`

---

### QUORUM_PRIORITY_RECONNECT 🆕

**Value**: `quorum_priority_reconnect`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/p2p_recovery_daemon.py:1295`

---

### QUORUM_RESTORED 🆕

**Value**: `quorum_restored`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/voter_health_daemon.py:583`

---

### SSH_NODE_UNRESPONSIVE 🆕

**Value**: `ssh_node_unresponsive`

**Payload**: (no fields detected)

---

## Resilience Events (7)

### BACKPRESSURE_ACTIVATED 🆕

**Value**: `backpressure_activated`

**Payload**: (no fields detected)

---

### BACKPRESSURE_RELEASED 🆕

**Value**: `backpressure_released`

**Payload**: (no fields detected)

---

### CIRCUIT_BREAKER_CLOSED 🆕

**Value**: `circuit_breaker_closed`

**Payload**: (no fields detected)

---

### CIRCUIT_BREAKER_HALF_OPEN 🆕

**Value**: `circuit_breaker_half_open`

**Payload**: (no fields detected)

---

### CIRCUIT_BREAKER_OPENED 🆕

**Value**: `circuit_breaker_opened`

**Payload**: (no fields detected)

---

### CIRCUIT_BREAKER_THRESHOLD 🆕

**Value**: `circuit_breaker_threshold`

**Payload**: (no fields detected)

---

### CIRCUIT_RESET 🆕

**Value**: `circuit_reset`

**Payload Fields**:

| Field             | Type   |
| ----------------- | ------ |
| `address`         | any    |
| `node_id`         | any    |
| `recovery_source` | string |
| `timestamp`       | any    |

**Emitters**:

- `scripts/p2p/loops/peer_recovery_loop.py:443`

---

## Monitoring Events (8)

### PROGRESS_STALL_DETECTED

**Value**: `progress_stall_detected`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/progress_watchdog_daemon.py:383`

---

### REGRESSION_CLEARED 🆕

**Value**: `regression_cleared`

**Payload**: (no fields detected)

---

### REGRESSION_CRITICAL

**Value**: `regression_critical`

**Payload**: (no fields detected)

---

### REGRESSION_DETECTED

**Value**: `regression_detected`

**Payload**: (no fields detected)

---

### REGRESSION_MINOR 🆕

**Value**: `regression_minor`

**Payload**: (no fields detected)

---

### REGRESSION_MODERATE 🆕

**Value**: `regression_moderate`

**Payload**: (no fields detected)

---

### REGRESSION_SEVERE 🆕

**Value**: `regression_severe`

**Payload**: (no fields detected)

---

### WORK_QUEUE_STALLED 🆕

**Value**: `work_queue_stalled`

**Payload Fields**:

| Field               | Type    |
| ------------------- | ------- |
| `blocked_configs`   | any     |
| `idle_seconds`      | any     |
| `pending_count`     | integer |
| `stall_detected_at` | any     |
| `threshold_seconds` | any     |

**Emitters**:

- `scripts/p2p/loops/job_loops.py:1093`

---

## Daemon Events (6)

### COORDINATOR_HEARTBEAT 🆕

**Value**: `coordinator_heartbeat`

**Payload**: (no fields detected)

---

### COORDINATOR_SHUTDOWN 🆕

**Value**: `coordinator_shutdown`

**Payload**: (no fields detected)

---

### DAEMON_CRASH_LOOP_DETECTED 🆕

**Value**: `daemon_crash_loop_detected`

**Payload**: (no fields detected)

---

### DAEMON_STARTED 🆕

**Value**: `daemon_started`

**Payload**: (no fields detected)

---

### DAEMON_STATUS_CHANGED 🆕

**Value**: `daemon_status_changed`

**Payload**: (no fields detected)

---

### DAEMON_STOPPED 🆕

**Value**: `daemon_stopped`

**Payload**: (no fields detected)

---

## Other Events (92)

### ADAPTIVE_PARAMS_CHANGED 🆕

**Value**: `adaptive_params_changed`

**Payload Fields**:

| Field        | Type |
| ------------ | ---- |
| `config_key` | any  |

**Emitters**:

- `app/coordination/feedback_loop_controller.py:1993`

---

### ARCHITECTURE_WEIGHTS_UPDATED 🆕

**Value**: `architecture_weights_updated`

**Payload Fields**:

| Field        | Type      |
| ------------ | --------- |
| `config_key` | any       |
| `timestamp`  | timestamp |
| `weights`    | any       |

**Emitters**:

- `app/coordination/architecture_feedback_controller.py:353`

---

### BATCH_DISPATCHED 🆕

**Value**: `batch_dispatched`

**Payload**: (no fields detected)

---

### BATCH_SCHEDULED 🆕

**Value**: `batch_scheduled`

**Payload**: (no fields detected)

---

### BUDGET_ALERT 🆕

**Value**: `budget_alert`

**Payload**: (no fields detected)

---

### BUDGET_EXCEEDED 🆕

**Value**: `budget_exceeded`

**Payload**: (no fields detected)

---

### CACHE_INVALIDATED 🆕

**Value**: `cache_invalidated`

**Payload**: (no fields detected)

---

### CAPACITY_LOW 🆕

**Value**: `capacity_low`

**Payload**: (no fields detected)

---

### CAPACITY_RESTORED 🆕

**Value**: `capacity_restored`

**Payload**: (no fields detected)

---

### CMAES_COMPLETED 🆕

**Value**: `cmaes_completed`

**Payload**: (no fields detected)

---

### CMAES_TRIGGERED 🆕

**Value**: `cmaes_triggered`

**Payload**: (no fields detected)

---

### CONFIG_DIVERGENCE_DETECTED 🆕

**Value**: `config_divergence_detected`

**Payload**: (no fields detected)

---

### CONFIG_UPDATED 🆕

**Value**: `config_updated`

**Payload**: (no fields detected)

---

### CONSOLIDATION_COMPLETE 🆕

**Value**: `consolidation_complete`

**Payload**: (no fields detected)

---

### CONSOLIDATION_STARTED 🆕

**Value**: `consolidation_started`

**Payload**: (no fields detected)

---

### CPU_PIPELINE_JOB_COMPLETED 🆕

**Value**: `cpu_pipeline_job_completed`

**Payload**: (no fields detected)

---

### CROSSBOARD_PROMOTION 🆕

**Value**: `crossboard_promotion`

**Payload**: (no fields detected)

---

### DEADLOCK_DETECTED 🆕

**Value**: `deadlock_detected`

**Payload**: (no fields detected)

---

### DISK_CLEANUP_TRIGGERED 🆕

**Value**: `disk_cleanup_triggered`

**Payload**: (no fields detected)

---

### DISK_SPACE_LOW 🆕

**Value**: `disk_space_low`

**Payload**: (no fields detected)

---

### DLQ_EVENTS_PURGED 🆕

**Value**: `dlq_events_purged`

**Payload**: (no fields detected)

---

### DLQ_EVENTS_REPLAYED 🆕

**Value**: `dlq_events_replayed`

**Payload**: (no fields detected)

---

### DLQ_STALE_EVENTS 🆕

**Value**: `dlq_stale_events`

**Payload**: (no fields detected)

---

### EPOCH_ADVANCED 🆕

**Value**: `epoch_advanced`

**Payload**: (no fields detected)

---

### ESCALATION_TIER_CHANGED 🆕

**Value**: `escalation_tier_changed`

**Payload**: (no fields detected)

---

### EXPLORATION_ADJUSTED 🆕

**Value**: `exploration_adjusted`

**Payload**: (no fields detected)

---

### EXPLORATION_BOOST 🆕

**Value**: `exploration_boost`

**Payload**: (no fields detected)

---

### HANDLER_TIMEOUT 🆕

**Value**: `handler_timeout`

**Payload**: (no fields detected)

---

### HOST_OFFLINE 🆕

**Value**: `host_offline`

**Payload**: (no fields detected)

---

### HOST_ONLINE 🆕

**Value**: `host_online`

**Payload**: (no fields detected)

---

### HYPERPARAMETER_UPDATED 🆕

**Value**: `hyperparameter_updated`

**Payload Fields**:

| Field                      | Type    |
| -------------------------- | ------- |
| `batch_size_multiplier`    | any     |
| `config_key`               | any     |
| `enable_cosine_annealing`  | boolean |
| `learning_rate_multiplier` | any     |
| `reason`                   | any     |
| `source`                   | string  |

**Emitters**:

- `app/coordination/feedback_loop_controller.py:1365`

---

### IDLE_RESOURCE_DETECTED 🆕

**Value**: `idle_resource_detected`

**Payload**: (no fields detected)

---

### IDLE_STATE_BROADCAST 🆕

**Value**: `idle_state_broadcast`

**Payload**: (no fields detected)

---

### IDLE_STATE_REQUEST 🆕

**Value**: `idle_state_request`

**Payload**: (no fields detected)

---

### JOB_PREEMPTED 🆕

**Value**: `job_preempted`

**Payload**: (no fields detected)

---

### JOB_SPAWN_VERIFIED 🆕

**Value**: `job_spawn_verified`

**Payload**: (no fields detected)

---

### LOCK_TIMEOUT 🆕

**Value**: `lock_timeout`

**Payload**: (no fields detected)

---

### MEMORY_PRESSURE

**Value**: `memory_pressure`

**Payload Fields**:

| Field             | Type |
| ----------------- | ---- |
| `gpu_total_gb`    | any  |
| `gpu_used_gb`     | any  |
| `gpu_utilization` | any  |
| `ram_utilization` | any  |
| `source`          | any  |
| `timestamp`       | any  |

**Emitters**:

- `app/coordination/memory_monitor_daemon.py:483`

---

### METRICS_UPDATED 🆕

**Value**: `metrics_updated`

**Payload**: (no fields detected)

---

### NAS_BEST_ARCHITECTURE 🆕

**Value**: `nas_best_architecture`

**Payload**: (no fields detected)

---

### NAS_COMPLETED 🆕

**Value**: `nas_completed`

**Payload**: (no fields detected)

---

### NAS_GENERATION_COMPLETE 🆕

**Value**: `nas_generation_complete`

**Payload**: (no fields detected)

---

### NAS_STARTED 🆕

**Value**: `nas_started`

**Payload**: (no fields detected)

---

### NAS_TRIGGERED 🆕

**Value**: `nas_triggered`

**Payload**: (no fields detected)

---

### NETWORK_ISOLATION_DETECTED 🆕

**Value**: `network_isolation_detected`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/p2p_recovery_daemon.py:1406`

---

### OPPONENT_MASTERED 🆕

**Value**: `opponent_mastered`

**Payload**: (no fields detected)

---

### P2P_CONNECTION_RESET_REQUESTED 🆕

**Value**: `p2p_connection_reset_requested`

**Payload**: (no fields detected)

---

### P2P_RESTARTED 🆕

**Value**: `p2p_restarted`

**Payload**: (no fields detected)

---

### P2P_RESTART_TRIGGERED 🆕

**Value**: `p2p_restart_triggered`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/p2p_recovery_daemon.py:1352`

---

### PARITY_VALIDATION_COMPLETED 🆕

**Value**: `parity_validation_completed`

**Payload**: (no fields detected)

---

### PARITY_VALIDATION_STARTED 🆕

**Value**: `parity_validation_started`

**Payload**: (no fields detected)

---

### PARTITION_HEALED 🆕

**Value**: `partition_healed`

**Payload Fields**:

| Field               | Type      |
| ------------------- | --------- |
| `duration_ms`       | any       |
| `nodes_reconnected` | any       |
| `partitions_found`  | any       |
| `partitions_healed` | any       |
| `timestamp`         | timestamp |

**Emitters**:

- `scripts/p2p/partition_healer.py:663`

---

### PARTITION_HEALING_STARTED 🆕

**Value**: `partition_healing_started`

**Payload Fields**:

| Field       | Type      |
| ----------- | --------- |
| `timestamp` | timestamp |

**Emitters**:

- `scripts/p2p/partition_healer.py:784`

---

### PBT_COMPLETED 🆕

**Value**: `pbt_completed`

**Payload**: (no fields detected)

---

### PBT_GENERATION_COMPLETE 🆕

**Value**: `pbt_generation_complete`

**Payload**: (no fields detected)

---

### PBT_STARTED 🆕

**Value**: `pbt_started`

**Payload**: (no fields detected)

---

### PER_BUFFER_REBUILT 🆕

**Value**: `per_buffer_rebuilt`

**Payload**: (no fields detected)

---

### PER_PRIORITIES_UPDATED 🆕

**Value**: `per_priorities_updated`

**Payload**: (no fields detected)

---

### PLATEAU_DETECTED 🆕

**Value**: `plateau_detected`

**Payload Fields**:

| Field            | Type    |
| ---------------- | ------- |
| `config_key`     | any     |
| `current_elo`    | number  |
| `plateau_type`   | string  |
| `recommendation` | string  |
| `source`         | string  |
| `stall_count`    | integer |
| `velocity`       | number  |

**Emitters**:

- `app/coordination/selfplay_scheduler.py:3798`

---

### PROMOTION_CANDIDATE 🆕

**Value**: `promotion_candidate`

**Payload**: (no fields detected)

---

### PROMOTION_REJECTED 🆕

**Value**: `promotion_rejected`

**Payload**: (no fields detected)

---

### PROMOTION_ROLLED_BACK 🆕

**Value**: `promotion_rolled_back`

**Payload**: (no fields detected)

---

### PROMOTION_STARTED 🆕

**Value**: `promotion_started`

**Payload**: (no fields detected)

---

### REGISTRY_UPDATED 🆕

**Value**: `registry_updated`

**Payload**: (no fields detected)

---

### REPAIR_COMPLETED 🆕

**Value**: `repair_completed`

**Payload**: (no fields detected)

---

### S3_BACKUP_COMPLETED 🆕

**Value**: `s3_backup_completed`

**Payload**: (no fields detected)

---

### SCHEDULER_REGISTERED 🆕

**Value**: `scheduler_registered`

**Payload**: (no fields detected)

---

### SOCKET_LEAK_DETECTED 🆕

**Value**: `socket_leak_detected`

**Payload**: (no fields detected)

---

### SPLIT_BRAIN_DETECTED 🆕

**Value**: `split_brain_detected`

**Payload**: (no fields detected)

---

### SPLIT_BRAIN_RESOLVED 🆕

**Value**: `split_brain_resolved`

**Payload**: (no fields detected)

---

### SSH_LIVENESS_CHECK_STARTED 🆕

**Value**: `ssh_liveness_check_started`

**Payload**: (no fields detected)

---

### SSH_LIVENESS_CHECK_SUCCEEDED 🆕

**Value**: `ssh_liveness_check_succeeded`

**Payload**: (no fields detected)

---

### STATE_PERSISTED 🆕

**Value**: `state_persisted`

**Payload**: (no fields detected)

---

### TASK_ABANDONED 🆕

**Value**: `task_abandoned`

**Payload**: (no fields detected)

---

### TASK_CANCELLED 🆕

**Value**: `task_cancelled`

**Payload**: (no fields detected)

---

### TASK_COMPLETED 🆕

**Value**: `task_completed`

**Payload Fields**:

| Field              | Type    |
| ------------------ | ------- |
| `board_type`       | any     |
| `config_key`       | any     |
| `duration_seconds` | any     |
| `node_id`          | any     |
| `num_players`      | integer |
| `task_id`          | any     |
| `task_type`        | any     |

**Emitters**:

- `scripts/p2p_orchestrator.py:14491`

---

### TASK_HEARTBEAT 🆕

**Value**: `task_heartbeat`

**Payload**: (no fields detected)

---

### TASK_ORPHANED 🆕

**Value**: `task_orphaned`

**Payload**: (no fields detected)

---

### TASK_SPAWNED 🆕

**Value**: `task_spawned`

**Payload**: (no fields detected)

---

### TIER_PROMOTION 🆕

**Value**: `tier_promotion`

**Payload**: (no fields detected)

---

### VOTER_DEMOTED 🆕

**Value**: `voter_demoted`

**Payload Fields**:

| Field           | Type |
| --------------- | ---- |
| `active_voters` | any  |
| `reason`        | any  |
| `voter_id`      | any  |

**Emitters**:

- `scripts/p2p/voter_health_monitor.py:432`

---

### VOTER_OFFLINE 🆕

**Value**: `voter_offline`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/p2p_recovery_daemon.py:393`
- `app/coordination/voter_health_daemon.py:530`

---

### VOTER_ONLINE 🆕

**Value**: `voter_online`

**Payload**: (no fields detected)

**Emitters**:

- `app/coordination/p2p_recovery_daemon.py:404`
- `app/coordination/voter_health_daemon.py:547`

---

### VOTER_PROMOTED 🆕

**Value**: `voter_promoted`

**Payload Fields**:

| Field           | Type |
| --------------- | ---- |
| `active_voters` | any  |
| `voter_id`      | any  |

**Emitters**:

- `scripts/p2p/voter_health_monitor.py:488`

---

### WEIGHT_UPDATED 🆕

**Value**: `weight_updated`

**Payload**: (no fields detected)

---

### WORK_CANCELLED 🆕

**Value**: `work_cancelled`

**Payload**: (no fields detected)

---

### WORK_CLAIMED 🆕

**Value**: `work_claimed`

**Payload**: (no fields detected)

---

### WORK_COMPLETED

**Value**: `work_completed`

**Payload**: (no fields detected)

---

### WORK_QUEUED

**Value**: `work_queued`

**Payload**: (no fields detected)

---

### WORK_RETRY 🆕

**Value**: `work_retry`

**Payload**: (no fields detected)

---

### WORK_STARTED 🆕

**Value**: `work_started`

**Payload**: (no fields detected)

---

### WORK_TIMEOUT 🆕

**Value**: `work_timeout`

**Payload**: (no fields detected)

---
