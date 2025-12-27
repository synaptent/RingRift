# P2P Orchestrator Module

This package provides the distributed P2P orchestrator for RingRift AI training.
The orchestrator coordinates selfplay, training, and data sync across a cluster of nodes.

## Architecture

```
scripts/p2p/
├── __init__.py           # Package exports, backward compatibility
├── constants.py          # Configuration constants (421 lines)
├── types.py              # Enums: NodeRole, JobType (34 lines)
├── models.py             # Dataclasses: NodeInfo, ClusterJob (771 lines)
├── network.py            # HTTP client, circuit breaker (181 lines)
├── resource.py           # Resource checking utilities (115 lines)
├── cluster_config.py     # Cluster configuration loading (265 lines)
├── client.py             # P2P client for external use (441 lines)
├── utils.py              # General utilities (46 lines)
├── metrics_manager.py    # Metrics recording & history (Dec 26, 2025)
├── resource_detector.py  # System resource detection (Dec 26, 2025)
├── network_utils.py      # Peer address & URL utilities (Dec 26, 2025)
├── handlers/             # HTTP handler mixins
│   ├── work_queue.py     # Work queue handlers (471 lines)
│   ├── election.py       # Election handlers (349 lines)
│   └── ...               # Other handler modules
└── README.md             # This file
```

## Decomposition Plan

The main `p2p_orchestrator.py` (29,767 lines, 510 methods) needs decomposition into:

### Phase 1: HTTP Handler Extraction (TARGET: ~15,000 lines)

Extract handlers into `scripts/p2p/handlers/`:

| Module              | Lines | Methods | Status                             |
| ------------------- | ----- | ------- | ---------------------------------- |
| `work_queue.py`     | 471   | 11      | ✅ Extracted                       |
| `election.py`       | 349   | 5       | ✅ Extracted                       |
| `relay.py`          | 368   | 4       | ✅ Extracted                       |
| `gauntlet.py`       | 382   | 3       | ✅ Extracted                       |
| `gossip.py`         | 226   | 2       | ✅ Extracted                       |
| `admin.py`          | 131   | 3       | ✅ Extracted                       |
| `elo_sync.py`       | 175   | 4       | ✅ Extracted                       |
| `tournament.py`     | 210   | 4       | ✅ Extracted                       |
| `cmaes.py`          | 215   | 4       | ✅ Extracted                       |
| `ssh_tournament.py` | 263   | 4       | ✅ Extracted                       |
| `data_sync.py`      | ~1000 | 12      | ⏸️ Skipped (complex internal deps) |
| `dashboard.py`      | ~2000 | 20      | Pending                            |
| `api.py`            | ~2500 | 25      | Pending (scattered)                |

**Progress: 29,767 → 27,522 lines (-2,245 lines, ~7.5% reduction)**

### Utility Extractions (Dec 26, 2025)

| Module                 | Lines | Methods | Status       |
| ---------------------- | ----- | ------- | ------------ |
| `metrics_manager.py`   | 268   | 6       | ✅ Extracted |
| `resource_detector.py` | 340   | 8       | ✅ Extracted |
| `network_utils.py`     | 422   | 11      | ✅ Extracted |
| `leader_election.py`   | 198   | 5       | ✅ Extracted |

**Progress: 27,522 → 27,320 lines (Phase 2 continued)**
**Total reduction: 29,767 → 27,320 lines (-2,447 lines, ~8.2%)**

### Dec 26, 2025 Session Summary

**Extracted Modules:**
| Module | Lines | Methods | Purpose |
|--------|-------|---------|---------|
| `network_utils.py` | 422 | 11 | URL building, Tailscale detection |
| `leader_election.py` | 198 | 5 | Voter quorum, consistency checks |
| `peer_manager.py` | 353 | 8 | Peer cache, reputation |
| `metrics_manager.py` | 318 | 6 | Metrics recording |
| `resource_detector.py` | 451 | 8 | System resource detection |

**Key Improvements:**

1. Eliminated dynamic `importlib` loading in `app/metrics/__init__.py`
2. Added 20 unit tests for `leader_election.py`
3. Tailscale URL building now in reusable mixin

**Remaining Large Methods (for future extraction):**

- `_gossip_state_to_peers` (line 21548) - State synchronization
- `_check_dead_peers_async` (line 20749) - Dead peer detection
- `_dispatch_training_job` (line 10015) - Job dispatch
- `_run_gpu_selfplay_job` (line 6980) - GPU selfplay

These methods have complex state dependencies requiring careful extraction.

### Mixin Integration (Dec 26, 2025)

P2POrchestrator now inherits from:

**NetworkUtilsMixin** - Peer address parsing, URL building, Tailscale detection:

- `_parse_peer_address` → provided by mixin
- `_url_for_peer` → provided by mixin
- `_urls_for_peer` → provided by mixin
- `_is_tailscale_host` → provided by mixin
- `_get_tailscale_ip_for_peer` → provided by mixin (Dec 26)
- `_tailscale_urls_for_voter` → provided by mixin (Dec 26)
- `_local_has_tailscale` → P2POrchestrator-specific override retained

**PeerManagerMixin** - Peer cache and reputation management:

- `_update_peer_reputation` → provided by mixin (EMA-based reputation)
- `_save_peer_to_cache` → provided by mixin (SQLite persistence with pruning)
- `_get_bootstrap_peers_by_reputation` → provided by mixin (prioritized peer list)
- `_get_cached_peer_count`, `_clear_peer_cache`, `_prune_stale_peers` → utility methods
- `_get_peer_health_score`, `_record_p2p_sync_result` → orchestrator overrides (use full peer state + circuit breaker)

**LeaderElectionMixin** - Core leader election logic (Dec 26):

- `_has_voter_quorum` → provided by mixin (fixed min 3 quorum)
- `_release_voter_grant_if_self` → provided by mixin
- `_get_voter_quorum_status` → provided by mixin (debug helper)
- `_check_leader_consistency` → provided by mixin
- `_is_leader_lease_valid` → orchestrator override (with grace period)

**Note:** `data_sync.py` handlers skipped - they depend on internal methods (`check_disk_has_capacity`, `_handle_sync_pull_request`) and peer lookup state that would require significant refactoring.

### Phase 2: Core Logic Extraction (TARGET: ~10,000 lines)

**Status: In Progress (Dec 26, 2025)**
**Current orchestrator size: 27,320 lines**

#### 2.1 `peer_manager.py` - Peer Discovery & Management (~3,000 lines)

**Status: ✅ Mixin Integrated (Dec 26, 2025)**

PeerManagerMixin provides peer cache management. Orchestrator-specific methods remain:
| Method | Status | Description |
| ------------------------------------ | ------------ | -------------------------- |
| `_update_peer_reputation` | ✅ Mixin | Track peer success/failure |
| `_save_peer_to_cache` | ✅ Mixin | SQLite peer persistence |
| `_get_bootstrap_peers_by_reputation` | ✅ Mixin | Prioritized peer list |
| `_get_peer_health_score` | Override | Uses full peer state + circuit breaker |
| `_record_p2p_sync_result` | Override | Circuit breaker + detailed metrics |
| `_get_cached_peer_count` | ✅ Extracted | Cache count |
| `_clear_peer_cache` | ✅ Extracted | Clear non-seed peers |
| `_prune_stale_peers` | ✅ Extracted | Prune old peers |

Tests: 15 tests in `tests/unit/scripts/test_peer_manager.py`

**Remaining methods to extract:**

| Method                        | Lines     | Description             |
| ----------------------------- | --------- | ----------------------- |
| `_get_tailscale_ip_for_peer`  | 3142-...  | Tailscale IP lookup     |
| `_tailscale_urls_for_voter`   | 3222-...  | Multi-endpoint fallback |
| `_send_heartbeat_to_peer`     | 19266-... | Peer health check       |
| `_bootstrap_from_known_peers` | 19446-... | Initial peer discovery  |
| `_follower_discovery_loop`    | 19799-... | Async discovery loop    |
| `_check_dead_peers`           | 20978-... | Peer failure detection  |
| `_check_dead_peers_async`     | 20897-... | Async version           |
| `_probe_nat_blocked_peers`    | 20577-... | NAT traversal           |
| `_probe_nat_blocked_peer`     | 20769-... | Single peer probe       |
| `_select_best_relay`          | 20652-... | Relay selection         |

**State to track:**

- `_peers: dict[str, NodeInfo]` - Active peers
- `_peer_last_seen: dict[str, float]` - Last heartbeat times
- `_peer_reputations: dict[str, float]` - Success rates
- `_nat_blocked_peers: set[str]` - Peers behind NAT

**Dependencies:** `network_utils.py`, `models.py`, `constants.py`

#### 2.2 `leader_election.py` - Bully Algorithm & Leases (~2,000 lines)

**Methods to extract:**

| Method                                 | Lines     | Description                |
| -------------------------------------- | --------- | -------------------------- |
| `_is_leader`                           | 1347-...  | Leader check (private)     |
| `is_leader`                            | 1409-...  | Leader check (public)      |
| `_load_voter_node_ids`                 | 1650-...  | Load voter config          |
| `_maybe_adopt_voter_node_ids`          | 1704-...  | Dynamic voter update       |
| `_has_voter_quorum`                    | 1740-...  | Quorum check               |
| `_release_voter_grant_if_self`         | 1766-...  | Grant release              |
| `_enable_partition_local_election`     | 1779-...  | Network partition handling |
| `_restore_original_voters`             | 1841-...  | Voter recovery             |
| `_get_eligible_voters`                 | 1876-...  | Eligible voter list        |
| `_manage_dynamic_voters`               | 1919-...  | Dynamic voter management   |
| `_check_leader_health`                 | 1989-...  | Leader health check        |
| `_acquire_voter_lease_quorum`          | 2032-...  | Lease acquisition          |
| `_determine_leased_leader_from_voters` | 2116-...  | Leader determination       |
| `_query_arbiter_for_leader`            | 2197-...  | Arbiter fallback           |
| `_get_leader_peer`                     | 2251-...  | Get leader NodeInfo        |
| `_proxy_to_leader`                     | 2280-...  | Request proxying           |
| `_start_election`                      | 21065-... | Election initiation        |
| `_become_leader`                       | 21157-... | Leader promotion           |
| `_is_leader_eligible`                  | 20844-... | Eligibility check          |
| `_maybe_adopt_leader_from_peers`       | 20864-... | Leader adoption            |

**State to track:**

- `_leader_id: str | None` - Current leader
- `_leader_lease_id: str | None` - Active lease
- `_lease_expires_at: float` - Lease expiry
- `_voter_node_ids: list[str]` - Voter list
- `_is_leader_elected: bool` - Election state

**Dependencies:** `peer_manager.py`, `network.py`, `constants.py`

#### 2.3 `job_scheduler.py` - Work Queue & Auto-Scaling (~3,500 lines)

**Methods to extract:**

| Method                            | Lines     | Description           |
| --------------------------------- | --------- | --------------------- |
| `_can_spawn_process`              | 1468-...  | Resource check        |
| `_count_local_jobs`               | 3502-...  | Job counting          |
| `_stop_all_local_jobs`            | 6288-...  | Job cleanup           |
| `_run_gpu_selfplay_job`           | 7246-...  | GPU selfplay dispatch |
| `_start_auto_training`            | 6034-...  | Auto-training trigger |
| `_dispatch_training_job`          | 10281-... | Training dispatch     |
| `_dispatch_improvement_training`  | 10705-... | Improvement cycle     |
| `_handle_training_job_completion` | 10890-... | Job completion        |
| `_find_running_training_job`      | 10251-... | Running job lookup    |
| `_find_resumable_training_job`    | 10261-... | Resumable job lookup  |
| `handle_start_job`                | 7032-...  | Job start handler     |
| `handle_stop_job`                 | 7071-...  | Job stop handler      |
| `handle_job_kill`                 | 7091-...  | Job kill handler      |
| `handle_restart_stuck_jobs`       | 7169-...  | Stuck job recovery    |

**State to track:**

- `_local_jobs: dict[str, JobInfo]` - Running jobs
- `_job_history: deque` - Completed jobs
- `_auto_training_enabled: bool` - Auto-training flag
- `_last_job_dispatch: dict[str, float]` - Cooldowns

**Dependencies:** `work_queue` module, `peer_manager.py`, `constants.py`

#### 2.4 `gossip_protocol.py` - State Synchronization (~1,500 lines)

**Methods to extract:**

| Method                             | Lines     | Description            |
| ---------------------------------- | --------- | ---------------------- |
| `_gossip_state_to_peers`           | 21814-... | State broadcast        |
| `_get_gossip_known_states`         | 21952-... | Known state collection |
| `_get_peer_endpoints_for_gossip`   | 21963-... | Peer endpoints         |
| `_process_gossip_response`         | 21991-... | Response handling      |
| `_process_gossip_peer_endpoints`   | 22059-... | Endpoint processing    |
| `_try_connect_gossip_peer`         | 22087-... | Peer connection        |
| `_record_gossip_metrics`           | 22134-... | Metrics recording      |
| `_voter_heartbeat_loop`            | 20328-... | Voter heartbeat        |
| `_send_voter_heartbeat`            | 20395-... | Heartbeat send         |
| `_try_voter_alternative_endpoints` | 20431-... | Endpoint fallback      |
| `_discover_voter_peer`             | 20462-... | Voter discovery        |
| `_refresh_voter_mesh`              | 20485-... | Mesh refresh           |

**State to track:**

- `_known_states: dict[str, dict]` - Peer states
- `_gossip_round: int` - Round counter
- `_last_gossip: float` - Last gossip time

**Dependencies:** `peer_manager.py`, `network.py`

#### Extraction Order

1. **`peer_manager.py`** (Week 1)
   - Lowest dependencies, foundational for other modules
   - Test: Peer discovery, reputation tracking, NAT probing

2. **`leader_election.py`** (Week 1-2)
   - Depends on peer_manager
   - Test: Election, lease acquisition, quorum checks

3. **`gossip_protocol.py`** (Week 2)
   - Depends on peer_manager
   - Test: State sync, mesh refresh

4. **`job_scheduler.py`** (Week 2-3)
   - Depends on peer_manager, leader_election
   - Test: Job dispatch, resource checks, auto-scaling

#### Expected Reduction

| Module               | Lines  | Reduction |
| -------------------- | ------ | --------- |
| `peer_manager.py`    | ~2,500 | 9%        |
| `leader_election.py` | ~2,000 | 7%        |
| `gossip_protocol.py` | ~1,500 | 5%        |
| `job_scheduler.py`   | ~3,500 | 13%       |
| **Total Phase 2**    | ~9,500 | **34%**   |

**Target: 27,586 → ~18,000 lines after Phase 2**

### Phase 3: Final Orchestrator (~5,000 lines)

The remaining P2POrchestrator class should coordinate:

- Lifecycle management (start, stop, run loops)
- Route registration
- State persistence
- Top-level coordination

## Handler Groups (Current p2p_orchestrator.py)

### Work Queue Handlers (lines 6776-7112)

- `handle_work_add`
- `handle_work_add_batch`
- `handle_work_claim`
- `handle_work_start`
- `handle_work_complete`
- `handle_work_fail`
- `handle_work_status`
- `handle_populator_status`
- `handle_work_for_node`
- `handle_work_cancel`
- `handle_work_history`

### Election Handlers (lines 7112-7349)

- `handle_election`
- `handle_lease_request`
- `handle_voter_grant_status`
- `handle_election_reset`
- `handle_election_force_leader`

### Gauntlet Handlers (lines 9055-9529)

- `handle_gauntlet_execute`
- `handle_gauntlet_status`
- `handle_gauntlet_quick_eval`

### Dashboard/API Handlers (lines 19073-21370)

- Analytics: `handle_games_analytics`, `handle_training_metrics`
- A/B Testing: `handle_abtest_*`
- Jobs API: `handle_api_jobs_*`
- Dashboard: `handle_dashboard`, `handle_work_queue_dashboard`

## Migration Strategy

1. **Create mixin classes** for handler groups
2. **P2POrchestrator inherits** from mixins
3. **Test after each extraction**
4. **Update imports** for backward compatibility

## Usage

```python
# External client usage
from scripts.p2p import P2PClient

client = P2PClient("http://localhost:8770")
status = await client.get_status()

# Constants
from scripts.p2p.constants import PEER_TIMEOUT, HEARTBEAT_INTERVAL

# Models
from scripts.p2p.models import NodeInfo, ClusterJob
```

## File Locations

- Main orchestrator: `scripts/p2p_orchestrator.py`
- Package: `scripts/p2p/`
- Tests: `tests/unit/scripts/test_p2p_*.py`
