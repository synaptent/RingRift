# P2P Decomposition Plan

Updated: April 11, 2026

This plan tracks the remaining high-value extraction targets for `ai-service/scripts/p2p_orchestrator.py`. The goal is auditability and operational maintainability, not deletion. Extracted mixins must keep behavior unchanged and should follow the existing `scripts/p2p/mixins/*_mixin.py` pattern.

## Current State

The orchestrator has been reduced from roughly 14,363 lines to 2,616 lines through targeted mixin extraction. The mixin modules now contain 11,907 lines across 19 `*_mixin.py` files, while the main orchestrator stays as the compatibility shell, constructor, and remaining glue code.

Verification: `cd ai-service && PYTHONPATH=. python3 -m pytest tests/unit/p2p/ -x -q` passed on April 10, 2026 with 2,615 passed and 2 skipped.

## Phase 3 Targets

These targets were identified from an AST audit of the remaining `P2POrchestrator` class on April 10, 2026. The immediate target is to reduce `ai-service/scripts/p2p_orchestrator.py` from about 4,913 LOC to below 3,000 LOC while preserving behavior.

| Target                       | Methods                                                                                                                                                                                                                                                                                                    | Estimated LOC | Shared State Dependencies                                                                                                                                                                                                   | Status    |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------- |
| StatePersistenceMixin        | `_load_state`, `_save_state`, `_save_cluster_epoch`, `_increment_cluster_epoch`, `record_metric`, `get_metrics_history`, `get_metrics_summary`                                                                                                                                                             | 310           | `state_manager`, `metrics_manager`, `peers`, `local_jobs`, `jobs_lock`, `peers_lock`, `leader_state_lock`, leader lease and voter fields, lease fencing fields, `health_metrics_manager`, `_job_snapshot`, `quorum_manager` | Completed |
| PeerDiscoveryMixin           | `_reconnect_discovered_peer`, `reconnect_missing_peers`, `_check_partition_mode`, `is_partition_readonly`, `get_partition_status`, `_local_has_tailscale`, `_get_tailscale_status`, `_sync_peer_snapshot`, `_prepopulate_voter_peers`, `_cache_local_ips`                                                  | 342           | `peers`, `peers_lock`, `_peer_snapshot`, `self_info`, `network`, distributed host loading, Tailscale/local IP helpers, partition state fields, event emission, heartbeat sender, voter config fields                        | Completed |
| JobManagementMixin expansion | `_get_all_active_jobs_for_reaper`, `_cancel_job_for_reaper`, `_get_job_heartbeats_for_reaper`, `_inline_job_reaper_fallback_loop`, `_spawn_and_track_job`, `_can_spawn_process`, `_check_spawn_rate_limit`, `_record_spawn`, `_get_node_job_preference`, `_record_gpu_job_result`, `_update_gpu_job_count` | 399           | Existing `JobManagementMixin`, `active_jobs`, `jobs_lock`, `jobs_started_at`, `job_manager`, `jobs`, `spawn_timestamps`, `self_info`, event emission, task abandonment callbacks, YAML cluster config                       | Completed |
| ProcessManagementMixin       | `_reap_orphan_processes`, `_cleanup_stale_processes`, `_cleanup_orphan_gpu_processes`, `_run_subprocess_sync`, `_run_subprocess_async`, `_get_max_selfplay_slots_for_node`                                                                                                                                 | 207           | `jobs`, `local_jobs`, `self_info`, process table access, `nvidia-smi`, subprocess helpers, RingRift process-name conventions                                                                                                | Completed |
| HttpSessionMixin             | `http_session`, `http_session_created_at`, `recreate_http_session`, `_auth_headers`, `_get_leader_peer`, `_proxy_to_leader`, `_is_request_authorized`                                                                                                                                                      | 158           | `_http_session`, auth token fields, `_peer_snapshot`, `leadership`, `leader_id`, `self_info`, URL builders, leader eligibility checks, endpoint conflict helpers, aiohttp sessions                                          | Completed |
| GameCountMixin               | `_seed_selfplay_scheduler_game_counts_sync`, `_fetch_game_counts_from_peers`, `_async_seed_game_counts_from_peers_if_needed`, `_game_count_refresh_loop`                                                                                                                                                   | 172           | `data_pipeline_manager`, `selfplay_scheduler`, peer snapshots, endpoint helpers, canonical DB layout, ai-service path helpers, aiohttp client sessions                                                                      | Completed |

The first six targets reduced the file to about 3,333 LOC, which missed the below-3,000 target. Two follow-up clusters were therefore extracted as `AutonomousWorkMixin` and `RelayCommandExecutionMixin`, reducing the main file to 2,591 LOC.

## Phase 3 Completion Status

Extracted modules:

- `ai-service/scripts/p2p/mixins/state_persistence_mixin.py`
- `ai-service/scripts/p2p/mixins/peer_discovery_mixin.py`
- `ai-service/scripts/p2p/mixins/process_management_mixin.py`
- `ai-service/scripts/p2p/mixins/http_session_mixin.py`
- `ai-service/scripts/p2p/mixins/game_count_mixin.py`
- `ai-service/scripts/p2p/mixins/autonomous_work_mixin.py`
- `ai-service/scripts/p2p/mixins/relay_command_execution_mixin.py`

Expanded module:

- `ai-service/scripts/p2p/mixins/job_management_mixin.py`

Final LOC snapshot:

- `ai-service/scripts/p2p_orchestrator.py`: 2,616 LOC.
- `ai-service/scripts/p2p/mixins/*.py`: 11,907 LOC across 19 mixins.
- Reduction from the original 14,363 LOC orchestrator baseline: about 82%.

Phase 20 verification snapshot (April 11, 2026):

- `ai-service/scripts/p2p_orchestrator.py`: 2,616 LOC
- 19 extracted mixins remain under `ai-service/scripts/p2p/mixins/*_mixin.py`
- Largest current mixins:
  - `job_management_mixin.py`: 1,396 LOC
  - `training_pipeline_mixin.py`: 1,290 LOC
  - `heartbeat_loop_mixin.py`: 1,024 LOC
  - `election_logic_mixin.py`: 1,033 LOC
  - `initialization_phases_mixin.py`: 977 LOC
- The main orchestrator remains below the Part 3 target ceiling of 3,000 LOC.

## Target 10: RuntimeLifecycleMixin

Status: Completed on April 10, 2026.

Extracted module: `ai-service/scripts/p2p/mixins/runtime_lifecycle_mixin.py`

Methods:

- `restart_http_server`
- `run`
- `_run_http_setup`
- `_run_start_background_tasks`
- `_run_bootstrap_and_election`
- `_run_game_count_refresh`
- `_run_shutdown`

Estimated LOC removed: about 795.

Shared state dependencies: HTTP app/runner/sites, `monitoring`, auth token state, route registry setup, sync/cooldown/connection-pool wiring, leadership state machine callbacks, relay mode flags, LoopManager, background task list, scheduler state, `self.running`, and leadership election fields.

## Target 11: InitializationPhasesMixin

Status: Completed on April 10, 2026.

Extracted module: `ai-service/scripts/p2p/mixins/initialization_phases_mixin.py`

Methods:

- `_init_settings`
- `_init_state`
- `_init_advanced_features`
- `_init_threading_and_protocols`
- `_get_peers_snapshot_nonblocking`
- `_publish_peers_snapshot`
- `get_peers_ro`
- `get_peers_list_ro`
- `_init_managers`
- `_init_event_wiring`
- `_get_loop_manager`
- `_register_extracted_loops`

Estimated LOC removed: about 956.

Shared state dependencies: node identity, bootstrap and relay config, storage config, quorum manager, partition config, peer snapshots, stability controller, SWIM/Raft/failover setup, state manager, metrics manager, job/training/sync locks, scheduler/coordinator/orchestrator manager instances, event wiring status, SWIM callbacks, and LoopManager registration state.

Historical candidate targets 12 and 13 were superseded by the completed Phase 3 extraction above.
