# P2P Orchestrator Decomposition Plan

**Created:** April 5, 2026
**Status:** Targets 1-9 COMPLETE.

## Current State

p2p_orchestrator.py: 14,363 -> 6,657 lines (54% reduction)

| Target | File                                            | LOC   | Status |
| ------ | ----------------------------------------------- | ----- | ------ |
| 1      | `scripts/p2p/startup_infrastructure.py`         | 1,335 | DONE   |
| 5      | `scripts/p2p/entrypoint.py`                     | 907   | DONE   |
| 3      | `scripts/p2p/mixins/training_pipeline_mixin.py` | 1,290 | DONE   |
| 4      | `scripts/p2p/mixins/heartbeat_loop_mixin.py`    | 1,024 | DONE   |
| 2      | `scripts/p2p/mixins/election_logic_mixin.py`    | 1,033 | DONE   |
| 6      | `scripts/p2p/mixins/data_sync_mixin.py`         | 547   | DONE   |
| 7      | `scripts/p2p/mixins/job_management_mixin.py`    | 974   | DONE   |
| 8      | `scripts/p2p/mixins/code_update_mixin.py`       | 327   | DONE   |
| 9      | `scripts/p2p/mixins/status_monitoring_mixin.py` | 952   | DONE   |

## April 2026 Decomposition Update

Targets 6-9 were extracted using the same mixin pattern as `heartbeat_loop_mixin.py` and `training_pipeline_mixin.py`: each mixin inherits from `P2PMixinBase`, accesses orchestrator state through `self.*`, and is wired through `scripts/p2p/mixins/__init__.py`, `scripts/p2p/startup_infrastructure.py`, and `scripts/p2p_orchestrator.py`.

The newly extracted surfaces are:

- `DataSyncMixin`: sync queue handling, data availability polling, file transfer coordination, and data-plane bookkeeping that had accumulated in the orchestrator.
- `JobManagementMixin`: job lifecycle actions, retry/claim bookkeeping, work queue integration, worker status handling, and related peer job orchestration.
- `CodeUpdateMixin`: cluster code update triggers, update status bookkeeping, and remote update coordination.
- `StatusMonitoringMixin`: status snapshots, peer health/status aggregation, diagnostics, and monitoring endpoints used by orchestration loops.

Verification for the extraction ran through the full ai-service unit/contracts gate:

```bash
PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120
```

Latest passing gate after Phase 6: `28518 passed, 116 skipped, 13 warnings`.

## Remaining Decomposition Guidance

The monolith is now substantially smaller, but it is still not a clean orchestration shell. Future extraction should prioritize areas that have low semantic risk and clear state boundaries:

- Transport/bootstrap compatibility code that can move behind facade classes.
- Legacy event compatibility paths that still import router-level `emit_event`.
- Partition-healing and leadership-recovery subflows whose tests can be isolated from normal leader election.
- HTTP route handlers that can delegate to existing mixins or manager classes.

Do not delete legacy P2P surfaces during this cleanup. Keep behavior available and move code behind audited mixin/facade boundaries so it can be reused alongside the minimal training loop without hiding experiment provenance.

## Target 2: Election + Leadership Methods (MEDIUM-HIGH risk)

Status: COMPLETE on April 5, 2026.

Extracted to `scripts/p2p/mixins/election_logic_mixin.py` and wired through:

- `scripts/p2p/mixins/__init__.py`
- `scripts/p2p/startup_infrastructure.py`
- `scripts/p2p_orchestrator.py`

The extracted surface includes:

- `_endpoint_key`, `_endpoint_conflict_keys`, `_is_leader_eligible`
- `_start_election`, `_become_leader`
- `_check_probabilistic_leadership`, `_claim_provisional_leadership`
- `_check_provisional_promotion`, `_promote_provisional_to_leader`
- `_step_down_from_provisional`, `_request_election_from_voters`
- `_check_emergency_coordinator_fallback`
- `_acquire_voter_lease_quorum`, `_determine_leased_leader_from_voters`
- `_query_arbiter_for_leader`, `_renew_leader_lease`

Verification:

- `PYTHONPATH=. python3 -c "from scripts.p2p_orchestrator import P2POrchestrator; print('OK')"`: passed
- `PYTHONPATH=. python3 -m pytest tests/unit/p2p/ -x --tb=short -q`: `2615 passed, 2 skipped`
- Broader seeded `tests/unit` gate reached an unrelated timeout in `tests/unit/coordination/test_health_check_compliance.py`; that file passes in isolation (`38 passed, 9 skipped`), so the first broader failure observed after Target 2 was not an election extraction regression.

### Methods to extract

- `_is_leader_eligible` (~38 LOC) - election heuristic
- `_endpoint_key`, `_endpoint_conflict_keys` (~47 LOC)
- `_start_election` (~249 LOC) - Bully algorithm with many guards
- `_become_leader` (~104 LOC)
- `_check_probabilistic_leadership` (~60 LOC) - provisional leader fallback
- `_claim_provisional_leadership` (~106 LOC)
- `_promote_provisional_to_leader` (~80 LOC)
- `_step_down_from_provisional` (~7 LOC)
- `_request_election_from_voters` (thin delegate)
- `_check_emergency_coordinator_fallback` (~118 LOC)
- `_acquire_voter_lease_quorum` (~130 LOC)
- `_determine_leased_leader_from_voters` (~82 LOC)
- `_renew_leader_lease` (~123 LOC)

### Shared state dependencies (all via self.\*)

- `self.role`, `self.leader_id`, `self.leader_lease_id`, `self.leader_lease_expires`
- `self.peers`, `self._peer_snapshot`, `self.node_id`, `self.voter_node_ids`
- `self.quorum_manager`, `self.leadership`, `self._leadership_sm`
- `self._forced_leader_override`, `self.election_in_progress`, `self._election_lock`

### Risks

1. **Deep coupling to leadership state machine** - `_start_election` and `_become_leader` have extensive locking via `self.leader_state_lock`
2. **Rolling deploy** - nodes on old code import from p2p_orchestrator; keep re-export stubs
3. **Circular imports** - use `TYPE_CHECKING` guards (same pattern as existing 41 mixins)

### Extraction approach

1. Create `scripts/p2p/mixins/election_logic_mixin.py` with `ElectionLogicMixin` class
2. Follow the existing mixin pattern (inherit from `P2PMixinBase`, `TYPE_CHECKING` guard)
3. Move all election methods there - they access state via `self.*` (standard mixin pattern)
4. Add `ElectionLogicMixin` to P2POrchestrator inheritance list
5. Add re-export in `scripts/p2p/mixins/__init__.py` and `startup_infrastructure.py`
6. Verify: `PYTHONPATH=. python3 -c "from scripts.p2p_orchestrator import P2POrchestrator; print('OK')"`
7. Run: `PYTHONPATH=. python3 -m pytest tests/unit/p2p/ -x --tb=short -q`

### Critical files

- `scripts/p2p_orchestrator.py` - the monolith (currently 6,657 lines)
- `scripts/p2p/mixins/health_tracking.py` - existing mixin pattern to follow
- `scripts/p2p/mixins/__init__.py` - must add export
- `scripts/p2p/startup_infrastructure.py` - must add import
- `scripts/p2p/p2p_mixin_base.py` - base class for mixins (995 LOC)
