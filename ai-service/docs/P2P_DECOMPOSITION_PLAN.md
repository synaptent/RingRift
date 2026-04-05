# P2P Orchestrator Decomposition Plan

**Created:** April 5, 2026
**Status:** Targets 1, 3, 4, 5 COMPLETE. Target 2 remaining.

## Current State

p2p_orchestrator.py: 14,363 → 10,355 lines (28% reduction)

| Target | File                                             | LOC        | Status   |
| ------ | ------------------------------------------------ | ---------- | -------- |
| 1      | `scripts/p2p/startup_infrastructure.py`          | 1,333      | DONE     |
| 5      | `scripts/p2p/entrypoint.py`                      | 907        | DONE     |
| 3      | `scripts/p2p/mixins/training_pipeline_mixin.py`  | 1,290      | DONE     |
| 4      | `scripts/p2p/mixins/heartbeat_loop_mixin.py`     | 1,024      | DONE     |
| **2**  | **`scripts/p2p/mixins/election_logic_mixin.py`** | **~1,100** | **TODO** |

## Target 2: Election + Leadership Methods (MEDIUM-HIGH risk)

### Methods to extract

- `_is_leader_eligible` (~38 LOC) — election heuristic
- `_endpoint_key`, `_endpoint_conflict_keys` (~47 LOC)
- `_start_election` (~249 LOC) — Bully algorithm with many guards
- `_become_leader` (~104 LOC)
- `_check_probabilistic_leadership` (~60 LOC) — provisional leader fallback
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

1. **Deep coupling to leadership state machine** — `_start_election` and `_become_leader` have extensive locking via `self.leader_state_lock`
2. **Rolling deploy** — nodes on old code import from p2p_orchestrator; keep re-export stubs
3. **Circular imports** — use `TYPE_CHECKING` guards (same pattern as existing 41 mixins)

### Extraction approach

1. Create `scripts/p2p/mixins/election_logic_mixin.py` with `ElectionLogicMixin` class
2. Follow the existing mixin pattern (inherit from `P2PMixinBase`, `TYPE_CHECKING` guard)
3. Move all election methods there — they access state via `self.*` (standard mixin pattern)
4. Add `ElectionLogicMixin` to P2POrchestrator inheritance list
5. Add re-export in `scripts/p2p/mixins/__init__.py` and `startup_infrastructure.py`
6. Verify: `PYTHONPATH=. python3 -c "from scripts.p2p_orchestrator import P2POrchestrator; print('OK')"`
7. Run: `PYTHONPATH=. python3 -m pytest tests/unit/p2p/ -x --tb=short -q`

### Critical files

- `scripts/p2p_orchestrator.py` — the monolith (currently 10,355 lines)
- `scripts/p2p/mixins/health_tracking.py` — existing mixin pattern to follow
- `scripts/p2p/mixins/__init__.py` — must add export
- `scripts/p2p/startup_infrastructure.py` — must add import
- `scripts/p2p/p2p_mixin_base.py` — base class for mixins (995 LOC)
