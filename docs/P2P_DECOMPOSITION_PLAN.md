# P2P Decomposition Plan

Updated: April 10, 2026

This plan tracks the remaining high-value extraction targets for `ai-service/scripts/p2p_orchestrator.py`. The goal is auditability and operational maintainability, not deletion. Extracted mixins must keep behavior unchanged and should follow the existing `scripts/p2p/mixins/*_mixin.py` pattern.

## Current State

The orchestrator has already been reduced from roughly 14,363 lines to 4,913 lines through targeted mixin extraction. The latest extraction moved startup/runtime lifecycle responsibilities into dedicated mixins while preserving the legacy orchestration surface.

Verification: `cd ai-service && PYTHONPATH=. python3 -m pytest tests/unit/p2p/ -x -q --timeout=120` passed on April 10, 2026 with 2,615 passed and 2 skipped.

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

## Target 12: PersistentStateMixin

Status: Candidate.

Candidate methods:

- `_load_state`
- `_save_state`

Estimated LOC: about 253.

Shared state dependencies: `state_manager`, `peers`, `local_jobs`, `leader_state_lock`, `NodeInfo`, `ClusterJob`, `JobType`, `NodeRole`, persisted leader lease fields, voter grant fields, forced leader override, peer and job snapshots, and health metrics persistence.

Extraction notes: This target is cohesive and lower risk than another large operational cluster. It should be extracted only after the current runtime/initialization mixins pass P2P unit tests, because it sits on the startup path and interacts with leadership recovery.

## Target 13: RelayCommandExecutionMixin

Status: Candidate.

Candidate methods:

- `_execute_relay_commands`
- `_action_reset_circuits`
- `_action_emit_alert`

Estimated LOC: about 240.

Shared state dependencies: relay command queues and attempts, `node_id`, relay locks, auth headers, peer URL/session helpers, circuit breaker registry, stability action callbacks, event emission, notification hooks, and subprocess/network command execution paths.

Extraction notes: Keep this target separate from relay HTTP handlers. The goal is to isolate relay command execution and recovery actions while preserving the existing handler routing modules.
