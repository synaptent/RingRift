# AGENTS Guide for `ai-service/`

This file refines the root `AGENTS.md` for the Python AI service subtree.
It applies to all files under `ai-service/**`.

If you touch anything here (DBs, parity, training, AI models), follow these
constraints.

---

## 1. Role of `ai-service`

- Hosts the **Python mirror** of the shared TS engine:
  - Pydantic models mirroring `src/shared/types/game.ts` in `app/models/**`.
  - `app/game_engine/` (package with `__init__.py` and `phase_requirements.py`) implementing GameEngine semantics for replay/parity.
  - `app/rules/**` exposing a stable `RulesEngine` interface and history checks.
- Owns the **replay database** and **training data pipeline**:
  - `app/db/**` – `GameReplayDB`, GameWriter/Recorder, parity DB helpers.
  - `app/training/**` – encoders, dataset generation, training loops, model versioning.
  - `scripts/**` – parity harnesses, canonical DB gates, dataset exporters, training CLIs.

The TS engine + canonical rules specs remain the semantic SSoT; Python code
must conform to them and to the TS engine, not vice versa.

---

## 2. Canonical vs Legacy Data & Models

### 2.1 Canonical policy (must follow)

- **Canonical game data**:
  - Lives in SQLite DBs under `ai-service/data/games` and `ai-service/data/canonical`.
  - Canonical DBs are named `canonical_<board>.db` or `canonical_<board>_<players>p.db`, e.g.:
    - `canonical_square8.db`, `canonical_square8_2p.db`, `canonical_square8_3p.db`, `canonical_square8_4p.db`
    - `canonical_square19.db`
    - `canonical_hex8.db` (radius-4 geometry, 9×9 grid, 61 cells)
    - `canonical_hex.db` (radius-12 geometry, 25×25 grid, 469 cells)
  - Their status and provenance are documented in:
    - `ai-service/TRAINING_DATA_REGISTRY.md`
    - Gate summaries like `db_health.canonical_<board>.json`
- **Legacy / non-canonical data**:
  - Any `selfplay_*`, historical `*.db`, or `legacy_*` directories.
  - Kept only for debugging/historical comparison; **never** use for new training.

**Agents must not:**

- Re‑introduce or depend on obsolete v1/v1_mps models or training NPZ/NPY files.
- Wire new training flows to DBs that are not clearly marked `canonical` in the registry.
- Create new self‑play DBs with ambiguous names; use `canonical_<board>.db` or an explicit `legacy_*/experimental_*` prefix and update docs.

### 2.2 Canonical DB gate

Canonical DBs must pass all of:

1. **Parity gate** (TS↔Python replay parity):
   - `ai-service/scripts/run_canonical_selfplay_parity_gate.py`
   - Or via the unified generator below.
   - Observability: pass `--summary <path>` and watch it with `tail -f <path>`; the gate emits stage/heartbeat progress to stderr every `--heartbeat-seconds` (default: 60).
2. **Canonical history validation**:
   - `app/rules/history_validation.validate_canonical_history_for_game(...)`.
   - CLIs: `scripts/check_canonical_phase_history.py`, `scripts/scan_canonical_phase_dbs.py`.
3. **Canonical generator**:
   - `ai-service/scripts/generate_canonical_selfplay.py`:
     - Runs a canonical self‑play soak.
     - Runs the parity harness.
     - Validates initial-state rules parameters match canonical defaults (no per-game overrides).
     - Runs canonical history validation across all games.
     - Yields `canonical_ok: true` only if everything passes and at least one game exists.

When regenerating canonical DBs, prefer `generate_canonical_selfplay.py` and
update `TRAINING_DATA_REGISTRY.md` once `canonical_ok` is true.

---

## 3. Replay Database Rules (GameReplayDB)

Location: `app/db/game_replay.py`

- **Schema**:
  - `SCHEMA_VERSION` is authoritative for migrations.
  - Use helper methods (`_migrate_v*_to_v*`) for schema changes; do not inline ad‑hoc `ALTER TABLE`.
- **Write‑time canonical enforcement**:
  - `GameReplayDB.__init__(db_path, snapshot_interval=..., enforce_canonical_history=True)`:
    - When `enforce_canonical_history=True` (default), `_store_move_conn` must:
      - Call `app.rules.history_contract.validate_canonical_move("", move.type.value)`.
      - Reject any non‑canonical `(phase, move_type)` combination with a clear error.
  - Only pass `enforce_canonical_history=False` for:
    - Legacy migrations,
    - Explicitly non‑canonical test fixtures (and document why in tests).
- **State reconstruction**:
  - `get_state_at_move(...)` must:
    - Rebuild state by replaying moves via `GameEngine.apply_move(..., trace_mode=True)`.
    - Not trust outdated snapshots from old engine versions.

If you add new MoveType or phase semantics, update:

- TS types in `src/shared/types/game.ts`.
- Python enums in `app/models/core.py`.
- Canonical contract in `app/rules/history_contract.py`.
- Any history validation and parity code that depends on them.

---

## 4. Canonical Phase↔Move Contract & Validators

Shared contract (do not copy/paste elsewhere):

- `app/rules/history_contract.py`:
  - `phase_move_contract()` – canonical mapping `phase -> (move_type, ...)`.
  - `derive_phase_from_move_type(move_type)` – infers phase from MoveType.
  - `validate_canonical_move(phase, move_type)` – single‑pair validation.
- Read‑side history validation:
  - `app/rules/history_validation.py`:
    - `validate_canonical_history_for_game(db, game_id)` – full‑game report.
  - Scripts:
    - `scripts/check_canonical_phase_history.py`
    - `scripts/scan_canonical_phase_dbs.py`

**Constraints for agents:**

- Do not add new MoveType values without updating the shared contract.
- Do not weaken `validate_canonical_move` to accept legacy non‑canonical combinations.
- When adding new validators or scripts, import from `history_contract` instead
  of re‑encoding the mapping.

---

## 5. TS↔Python Parity Tooling

Core scripts:

- `scripts/check_ts_python_replay_parity.py`:
  - Compares TS and Python replays step‑by‑step.
  - Flags semantic and structural divergences.
  - Emits:
    - Parity summaries (`parity_summary.*.json`).
    - Optional fixtures via `--emit-fixtures-dir`.
    - Optional **state bundles** via `--emit-state-bundles-dir`.
- `scripts/diff_state_bundle.py`:
  - Takes a single `.state_bundle.json` and prints:
    - Phase / player / status on both sides.
    - Structural diffs (players, stacks, collapsed territory).

Parity debug expectations:

- Use `--emit-state-bundles-dir` and `diff_state_bundle.py` to debug _one game at a time_.
- Fix underlying engine or recording/parsing logic, not the parity harness, when semantics disagree.
- Prefer small, surgical changes that keep TS and Python in lockstep.

---

## 6. Training Stack & Datasets

Important modules:

- `app/training/encoding.py` – encoders (must reflect 7‑phase + FE semantics).
- `app/training/generate_data.py` – self‑play driven dataset generation (v1).
- `app/training/train.py`, `train_loop.py` – training orchestration.
- `app/training/model_versioning.py` – `ModelVersionManager` for checkpoint metadata.
- `scripts/export_replay_dataset.py` – **preferred path** for building NPZ from replay DBs (v1 format).
- `scripts/run_canonical_square8_training.py` – example canonical training entrypoint (Square‑8).

Agent guidance:

- For **v2 canonical datasets**, prefer thin wrappers that:
  - Take only `canonical_*.db` as inputs.
  - Delegate encoding to existing modules (e.g. `export_replay_dataset.py` or `encoding.py`).
  - Can be wired into CI/automation.
- When adding new training scripts:
  - Enforce that any `--source-db` argument:
    - Has a basename starting with `canonical_`, or
    - Is explicitly documented as legacy/experimental.
  - Surface clear exit messages if a non‑canonical DB is passed.

Do not add dependencies on external services or network APIs from inside `app/`
modules; keep cloud or remote storage logic in `ai-service/cloud/**` and
scripts that users call explicitly.

---

## 7. Testing & Tooling Expectations

- Prefer **pytest** for new tests under `ai-service/tests/**`.
- For DB / parity / rules changes:
  - Add or update focused tests near the affected modules.
  - Avoid long‑running integration tests unless guarded by explicit flags.
- For scripts under `ai-service/scripts/**`:
  - Keep CLIs small and composable.
  - Prefer explicit arguments over environment‑only configuration.
  - Ensure `python -m py_compile <file>` succeeds after edits.

When modifying existing scripts like `training_preflight_check.py`,
`run_canonical_selfplay_parity_gate.py`, or dataset exporters:

- Keep them **idempotent** and safe to run locally.
- Avoid deleting files automatically unless a `--delete-*` flag is explicitly set.

---

## 8. Deletion, Cleanup, and Legacy Artifacts

- Use `scripts/scan_canonical_phase_dbs.py` and related helpers to manage DBs.
- If you add new cleanup utilities:
  - Default to **dry‑run** behavior; require `--delete` / `--archive-dir` flags for destructive actions.
  - Log which files would be affected and why.
- Never silently delete or overwrite:
  - Canonical DBs,
  - Checkpoints,
  - Training NPZ files,
    without explicit user action.

When in doubt, prefer:

- Printing instructions and exit codes,
- Over direct file deletion in automated scripts.

---

## 9. Singleton Pattern

The codebase uses the `@singleton` decorator as the preferred singleton pattern. Use it for new classes that should have only one instance.

### Preferred Pattern: @singleton Decorator

```python
from app.coordination.singleton_mixin import singleton

@singleton
class MyDaemon:
    def __init__(self):
        self._running = False

    def start(self):
        self._running = True
```

### Alternative Patterns (Legacy)

These patterns exist in the codebase but prefer `@singleton` for new code:

- `SingletonMixin` - Mixin class, requires calling `reset_instance()` manually
- `SingletonMeta` - Metaclass approach, more complex
- `ThreadSafeSingletonMixin` - Use when thread safety is critical

### Accessing Singletons

For singletons with accessors, follow this pattern:

```python
# In module:
_instance: Optional[MyDaemon] = None

def get_my_daemon() -> MyDaemon:
    global _instance
    if _instance is None:
        _instance = MyDaemon()
    return _instance

def reset_my_daemon() -> None:
    global _instance
    _instance = None
```

### Import Location

All singleton utilities are in `app/coordination/singleton_mixin.py`:

- `singleton` - Decorator (preferred)
- `SingletonMixin` - Mixin base class
- `SingletonMeta` - Metaclass
- `ThreadSafeSingletonMixin` - Thread-safe mixin
- `create_singleton_accessors` - Helper to create get/reset functions

---

## 10. Circular Import Mitigation

The coordination layer has complex interdependencies. Follow these patterns to avoid circular imports:

### 10.1 Import Order Rules

1. **Low-level modules import nothing from coordination:**
   - `app/config/*` - Configuration only
   - `app/models/*` - Data classes only
   - `app/utils/*` - Pure utilities

2. **Mid-level modules use lazy imports:**
   - `app/coordination/protocols.py` - Defines interfaces
   - `app/coordination/daemon_types.py` - Enums only
   - Import heavy modules inside functions, not at module level

3. **High-level orchestrators can import freely:**
   - `app/coordination/daemon_manager.py`
   - `app/coordination/coordination_bootstrap.py`

### 10.2 Lazy Import Pattern

Use lazy imports for modules that create cycles:

```python
# BAD - Creates circular import
from app.coordination.event_router import get_router

def my_function():
    router = get_router()
    router.emit("MY_EVENT", {})

# GOOD - Lazy import breaks the cycle
def my_function():
    from app.coordination.event_router import get_router
    router = get_router()
    router.emit("MY_EVENT", {})
```

### 10.3 TYPE_CHECKING Pattern

For type hints that would cause cycles:

```python
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.coordination.daemon_manager import DaemonManager

class MyClass:
    def __init__(self, manager: "DaemonManager"):
        self._manager = manager
```

### 10.4 High-Risk Circular Import Areas

These module groups have complex interdependencies:

| Module Group                            | Pattern Used             | Notes                                           |
| --------------------------------------- | ------------------------ | ----------------------------------------------- |
| `daemon_manager` ↔ `daemon_runners`     | Lazy imports in runners  | Factory functions import manager lazily         |
| `event_router` ↔ event emitters         | Lazy imports in emitters | `get_router()` called inside emit functions     |
| `coordination_bootstrap` ↔ coordinators | Forward references       | TYPE_CHECKING for type hints                    |
| P2P managers ↔ orchestrator             | Protocol interfaces      | Managers implement protocols defined separately |

### 10.5 Detection Tools

Run circular import detection:

```bash
# Check for potential circular imports
python scripts/audit_circular_deps.py

# Test imports don't fail
python -c "from app.coordination import *"
```

### 10.6 When Adding New Modules

1. Start with minimal imports at module level
2. Add imports inside functions that need them
3. Test with `python -c "import my_module"`
4. If circular import error, use lazy import pattern
5. Document any intentional lazy imports with comments

---

## 11. Daemon Development Patterns

The coordination layer has 109 daemon types (103 active, 6 deprecated). When creating new daemons, follow these patterns.

### 11.1 Daemon Lifecycle

Use `HandlerBase` as the base class for most daemons:

```python
from app.coordination.handler_base import HandlerBase, HealthCheckResult

class MyDaemon(HandlerBase):
    def __init__(self):
        super().__init__(name="my_daemon", cycle_interval=60.0)
        self._counter = 0

    async def _run_cycle(self) -> None:
        """Main work loop - called every cycle_interval seconds."""
        self._counter += 1
        self.logger.info(f"Cycle {self._counter}")

    def _get_event_subscriptions(self) -> dict:
        """Return event types this daemon subscribes to."""
        return {
            "DATA_SYNC_COMPLETED": self._on_sync_completed,
            "TRAINING_COMPLETED": self._on_training_done,
        }

    async def _on_sync_completed(self, event_data: dict) -> None:
        """Handle DATA_SYNC_COMPLETED events."""
        self.logger.info(f"Sync completed: {event_data}")

    def health_check(self) -> HealthCheckResult:
        """Required for DaemonManager integration."""
        return HealthCheckResult(
            healthy=self._running,
            details={"cycles": self._counter},
        )
```

### 11.2 Registration Steps

1. **Add to `daemon_types.py`**:

```python
class DaemonType(Enum):
    MY_DAEMON = "my_daemon"
```

2. **Add to `daemon_registry.py`**:

```python
DaemonType.MY_DAEMON: DaemonSpec(
    runner_name="create_my_daemon",
    depends_on=(DaemonType.EVENT_ROUTER,),
    category="my_category",
),
```

3. **Add runner to `daemon_runners.py`**:

```python
async def create_my_daemon(self) -> None:
    from app.coordination.my_daemon import MyDaemon
    from app.coordination.daemon_types import mark_daemon_ready, DaemonType

    daemon = MyDaemon.get_instance()
    await daemon.start()
    mark_daemon_ready(DaemonType.MY_DAEMON)

    # Keep running until cancelled
    try:
        while True:
            await asyncio.sleep(60)
    except asyncio.CancelledError:
        await daemon.stop()
```

4. **Add to `DAEMON_DEPENDENCIES`** if ordering matters:

```python
DaemonType.MY_DAEMON: {DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE},
```

5. **Add to `DAEMON_STARTUP_ORDER`** if it must start before/after specific daemons.

### 11.3 Critical Daemon Requirements

Add to `CRITICAL_DAEMONS` in `daemon_types.py` if:

- Other daemons depend on it for events
- Cluster operation fails without it
- It needs faster health checks (15s vs 60s)

### 11.4 Deprecating Daemons

1. Add to `_DEPRECATED_DAEMON_TYPES` in `daemon_types.py`:

```python
_DEPRECATED_DAEMON_TYPES: dict[str, tuple[str, str]] = {
    "my_daemon": ("NEW_DAEMON", "Q2 2026"),
}
```

2. Set `deprecated=True` in `DaemonSpec`:

```python
DaemonType.MY_DAEMON: DaemonSpec(
    runner_name="create_my_daemon",
    deprecated=True,
    deprecated_message="Use NEW_DAEMON instead. Removal: Q2 2026.",
),
```

### 11.5 HandlerBase Helper Methods (January 2026)

`HandlerBase` provides many utility methods to reduce boilerplate in daemon code.

**Event Payload Helpers:**

```python
# Normalize any event type to a consistent dict format
payload = self._normalize_event_payload(event)
# Returns: dict[str, Any] - works with Event objects, dicts, or raw values

# Extract specific fields with defaults (type-safe)
config_key, elo, model_path = self._extract_event_fields(
    event,
    ["config_key", "elo", "model_path"],
    defaults={"elo": 0, "model_path": None}
)
```

**Staleness Helpers:**

```python
# Check if a timestamp is older than threshold
if self._is_stale(last_sync_time, threshold_seconds=3600):
    self.logger.warning("Data is stale, triggering sync")

# Get staleness as a ratio (0.0 = fresh, 1.0 = at threshold, >1.0 = past threshold)
ratio = self._get_staleness_ratio(timestamp, threshold_seconds=3600)
if ratio > 2.0:  # More than 2x threshold
    self._trigger_emergency_sync()

# Get age in seconds or hours
age_secs = self._get_age_seconds(timestamp)
age_hours = self._get_age_hours(timestamp)
```

**Fire-and-Forget Task Helpers:**

```python
# Create async task with automatic error handling
# Won't crash daemon if task fails
self._safe_create_task(
    self._background_operation(config_key),
    context="background_sync",  # For error logging
    error_callback=lambda e: self._record_error(f"Sync failed: {e}"),
)

# Try to emit event with graceful fallback
# Returns True if emitted, False if failed (never raises)
success = self._try_emit_event(
    "MY_EVENT",
    {"config_key": config_key, "status": "complete"},
    emitter_fn=emit_event,  # Function to call
    context="operation_complete",
)
```

**Thread-Safe Queue Helpers:**

```python
# Append to a queue with lock protection
self._append_to_queue(self._pending_items, item, self._queue_lock)

# Get a copy of queue and clear original (atomic)
items = self._pop_queue_copy(self._pending_items, self._queue_lock)
for item in items:
    await self._process(item)
```

**Retry Queue Helpers:**

```python
# Add item with exponential backoff
self._add_to_retry_queue(
    self._retry_queue,
    {"config_key": "hex8_2p", "attempt": 1},
    base_delay=60.0,  # 1 minute
    max_retries=5,
)

# Get items ready for retry vs still waiting
ready, waiting = self._get_ready_retry_items(self._retry_queue)
self._retry_queue = waiting  # Keep items not ready
for item in ready:
    success = await self._retry_operation(item)
    if not success:
        self._add_to_retry_queue(self._retry_queue, item)

# Convenience: separate, process, and return remaining
async def _process_retries(self):
    self._retry_queue = await self._process_retry_queue_items(
        self._retry_queue,
        self._do_retry,  # Async function (item) -> bool
    )
```

### 11.6 Reference Documentation

- **Full registry**: `docs/DAEMON_REGISTRY.md` (129 daemons, dependencies, categories)
- **Types and state**: `app/coordination/daemon_types.py`
- **Specifications**: `app/coordination/daemon_registry.py`
- **Factory runners**: `app/coordination/daemon_runners.py`
- **Base classes**: `app/coordination/handler_base.py` (1,400+ LOC, 92 tests)

### 11.7 Advanced Patterns (Sprint 16.1)

**Scheduled Recheck Pattern** - When blocking on a condition (e.g., quality gate), schedule automatic rechecks:

```python
def _schedule_quality_recheck(
    self, config_key: str, delay_seconds: float = 300, max_rechecks: int = 6
) -> None:
    """Schedule automatic recheck after delay instead of waiting for full cycle."""
    # Cancel existing recheck for this config (avoid duplicates)
    if config_key in self._pending_quality_rechecks:
        old_task = self._pending_quality_rechecks.pop(config_key)
        if not old_task.done():
            old_task.cancel()

    # Check recheck count to avoid infinite loops
    current_count = self._quality_recheck_counts.get(config_key, 0)
    if current_count >= max_rechecks:
        return  # Give up, wait for external update

    task = asyncio.create_task(
        self._run_quality_recheck(config_key, delay_seconds, max_rechecks)
    )
    self._pending_quality_rechecks[config_key] = task
```

**Confirmation Event Pattern** - After important state changes, emit confirmation events for observability:

```python
def _emit_rollback_completed(
    self, config_key: str, old_weight: float, new_weight: float, elo_delta: float
) -> None:
    """Emit confirmation event for monitoring dashboards and alerts."""
    router.publish_sync(
        "CURRICULUM_ROLLBACK_COMPLETED",
        {
            "config_key": config_key,
            "old_weight": old_weight,
            "new_weight": new_weight,
            "elo_delta": elo_delta,
            "weight_reduction_pct": (1 - new_weight / old_weight) * 100,
            "timestamp": time.time(),
        },
        source="my_component",
    )
```

---

## 12. Board Geometry Conventions

RingRift supports multiple board geometries. This section documents the conventions for board types, config keys, and data paths.

### 12.1 Board Type Reference

| Config Key  | Board Type      | Grid Size              | Cell Count | Coordinate System |
| ----------- | --------------- | ---------------------- | ---------- | ----------------- |
| `hex8`      | Hexagonal small | Radius 4 (9×9 grid)    | 61         | Axial (q, r)      |
| `hexagonal` | Hexagonal large | Radius 12 (25×25 grid) | 469        | Axial (q, r)      |
| `square8`   | Square small    | 8×8                    | 64         | Cartesian (x, y)  |
| `square19`  | Square large    | 19×19                  | 361        | Cartesian (x, y)  |

All board types support 2, 3, or 4 players. The **config key** format is `{board_type}_{num_players}p`, e.g., `hex8_2p`, `square19_4p`.

### 12.2 Config Key Parsing Utilities

Use the canonical utilities in `app/coordination/event_utils.py`:

```python
from app.coordination.event_utils import (
    parse_config_key,
    make_config_key,
    ParsedConfigKey,
)

# Parse config key to components
parsed = parse_config_key("hex8_2p")
# -> ParsedConfigKey(board_type='hex8', num_players=2)

# Create config key from components
key = make_config_key("hex8", 2)
# -> "hex8_2p"

# Safe parsing with validation
parsed = parse_config_key("invalid")
# -> None (returns None on invalid input)
```

**DO NOT** use inline regex or string splitting to parse config keys. Always use these utilities for:

- Consistent error handling
- Type-safe return values
- Centralized validation logic

### 12.3 Hex Coordinate System

Hex boards use **axial coordinates** (q, r):

```
        (-2, 0)  (-1, 0)  (0, 0)  (1, 0)  (2, 0)
           (-2, 1)  (-1, 1)  (0, 1)  (1, 1)
              (-2, 2)  (-1, 2)  (0, 2)
```

**Key properties:**

- Center cell is always `(0, 0)`
- `q` increases to the right, `r` increases down-right
- Neighbors differ by at most ±1 in each coordinate
- Cell validity: `|q| + |r| + |q+r|/2 <= radius`

**Conversion utilities** in `app/rules/hex_utils.py`:

- `axial_to_cube(q, r)` → (x, y, z) cube coordinates
- `cube_to_axial(x, y, z)` → (q, r) axial coordinates
- `axial_distance(q1, r1, q2, r2)` → number of steps between cells
- `get_hex_neighbors(q, r)` → list of 6 neighbor coordinates

### 12.4 Data Path Conventions

Training data and models follow consistent naming conventions:

**Canonical Databases** (`data/games/`):

```
canonical_{board_type}.db           # All players for a board type
canonical_{board_type}_{n}p.db      # Specific player count
```

Examples: `canonical_hex8.db`, `canonical_square8_2p.db`

**Training NPZ Files** (`data/training/`):

```
{board_type}_{n}p.npz               # Standard training data
{board_type}_{n}p_quality.npz       # Quality-filtered data
{board_type}_{n}p_v{version}.npz    # Versioned training data
```

**Model Checkpoints** (`models/`):

```
canonical_{board_type}_{n}p.pth     # Production canonical model
ringrift_best_{board_type}_{n}p.pth # Symlink to canonical (backward compat)
{board_type}_{n}p_v{version}.pth    # Versioned model
```

**Discovery Utilities**:

```python
from app.utils.game_discovery import GameDiscovery

discovery = GameDiscovery()
dbs = discovery.find_databases_for_config("hex8", 2)
# Returns list of DatabaseInfo objects for hex8_2p databases
```

### 12.5 Board-Specific Constraints

| Board Type  | Max Stack Height | Line Length | Recovery Threshold |
| ----------- | ---------------- | ----------- | ------------------ |
| `hex8`      | 4                | 4           | 3 buried pieces    |
| `hexagonal` | 4                | 4           | 3 buried pieces    |
| `square8`   | 4                | 4           | 3 buried pieces    |
| `square19`  | 4                | 5           | 4 buried pieces    |

These are defined in `src/shared/engine/rules/constants.ts` (TypeScript) and mirrored in `app/rules/constants.py` (Python).

---

## 13. Daemon Dependencies

The daemon system has 89+ daemons with complex interdependencies. This section documents the dependency rules and startup ordering.

### 13.1 Startup Order Rules

Daemons must start in dependency order. The key rules are:

1. **EVENT_ROUTER first**: All event-driven daemons depend on the event router
2. **Subscribers before emitters**: Event subscribers must be running before emitters start
3. **Infrastructure before application**: Core daemons (health, config) before feature daemons

The canonical startup order in `master_loop.py`:

```python
DAEMON_STARTUP_ORDER = [
    DaemonType.EVENT_ROUTER,           # 1. Event infrastructure
    DaemonType.HEALTH_SERVER,          # 2. Health monitoring
    DaemonType.FEEDBACK_LOOP,          # 3. Event subscribers
    DaemonType.DATA_PIPELINE,          # 4. Pipeline orchestration
    DaemonType.SELFPLAY_COORDINATOR,   # 5. Work coordination
    DaemonType.AUTO_SYNC,              # 6. Data sync (emits events)
    DaemonType.MODEL_DISTRIBUTION,     # 7. Model distribution
    # ... remaining daemons
]
```

### 13.2 Declaring Dependencies

Dependencies are declared in `DaemonSpec` via the `depends_on` field:

```python
from app.coordination.daemon_registry import DaemonSpec
from app.coordination.daemon_types import DaemonType

# In daemon_registry.py
DaemonType.MY_DAEMON: DaemonSpec(
    runner_name="create_my_daemon",
    depends_on=(DaemonType.EVENT_ROUTER, DaemonType.DATA_PIPELINE),
    category="my_category",
),
```

The `DaemonManager` will:

1. Validate all dependencies exist
2. Start dependencies before dependents
3. Block startup if circular dependencies detected

### 13.3 Common Dependency Patterns

| Daemon Category   | Typical Dependencies                    | Purpose                  |
| ----------------- | --------------------------------------- | ------------------------ |
| Event emitters    | `EVENT_ROUTER`                          | Emit events to bus       |
| Event subscribers | `EVENT_ROUTER`, emitters                | React to events          |
| Training daemons  | `DATA_PIPELINE`, `SELFPLAY_COORDINATOR` | Access training data     |
| Sync daemons      | `EVENT_ROUTER`, `DATA_PIPELINE`         | Coordinate data movement |
| Health monitors   | `HEALTH_SERVER`                         | Report health status     |
| Resource managers | `NODE_AVAILABILITY`, `CLUSTER_MONITOR`  | Manage cluster resources |

### 13.4 Critical Daemons

Some daemons are marked as **critical** in `daemon_types.py`. Critical daemons:

- Have faster health checks (15s vs 60s)
- Trigger alerts when unhealthy
- Block cluster operation if not running

```python
CRITICAL_DAEMONS = {
    DaemonType.EVENT_ROUTER,
    DaemonType.DATA_PIPELINE,
    DaemonType.SELFPLAY_COORDINATOR,
    DaemonType.AUTO_SYNC,
    DaemonType.FEEDBACK_LOOP,
}
```

### 13.5 Debugging Dependency Issues

Common symptoms and solutions:

| Symptom                        | Likely Cause                     | Solution                      |
| ------------------------------ | -------------------------------- | ----------------------------- |
| "Event not delivered"          | Subscriber started after emitter | Check startup order           |
| "Daemon timeout waiting for X" | Circular dependency              | Review `depends_on` chain     |
| "Health check failed"          | Dependency daemon crashed        | Check dependency health first |
| "Pipeline stuck at stage X"    | Missing stage handler            | Verify handler subscription   |

**Debug commands:**

```bash
# Check daemon startup order
python -c "from app.coordination.daemon_registry import get_startup_order; print(get_startup_order())"

# Check specific daemon dependencies
python -c "from app.coordination.daemon_registry import DAEMON_REGISTRY; print(DAEMON_REGISTRY[DaemonType.MY_DAEMON].depends_on)"

# Verify event subscriptions
curl -s http://localhost:8790/status | jq '.event_subscriptions'
```

### 13.6 Adding New Dependencies

When adding a new daemon with dependencies:

1. Add the dependency to `DaemonSpec.depends_on`
2. Test startup with `python scripts/master_loop.py --dry-run`
3. Verify no circular dependencies via `scripts/audit_circular_deps.py`
4. Update `DAEMON_STARTUP_ORDER` if manual ordering needed
5. Document the dependency reason in comments

---

## 14. Training Feedback Loop Reference

The training pipeline uses event-driven feedback loops to optimize model quality. This section documents the key event chains and configuration.

### 14.1 Feedback Loop Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Training Feedback Loop                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Selfplay ──► NEW_GAMES_AVAILABLE ──► DataPipeline                  │
│                                            │                         │
│                                            ▼                         │
│                                   TRAINING_THRESHOLD_REACHED         │
│                                            │                         │
│                                            ▼                         │
│                                      Training                        │
│                                            │                         │
│                                            ▼                         │
│                               TRAINING_COMPLETED                     │
│                                            │                         │
│                                            ▼                         │
│                                      Evaluation                      │
│                                            │                         │
│                              ┌─────────────┴─────────────┐          │
│                              ▼                           ▼          │
│                    EVALUATION_COMPLETED         REGRESSION_DETECTED │
│                              │                           │          │
│                              ▼                           ▼          │
│                       MODEL_PROMOTED          Curriculum Adjustment │
│                              │                                      │
│                              ▼                                      │
│                        Distribution                                 │
│                              │                                      │
│                              ▼                                      │
│                   Curriculum Rebalance ──► Back to Selfplay        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### 14.2 Key Event Chains

| Event                        | Emitter             | Subscribers                      | Purpose                  |
| ---------------------------- | ------------------- | -------------------------------- | ------------------------ |
| `NEW_GAMES_AVAILABLE`        | AutoExportDaemon    | DataPipeline, TrainingTrigger    | New selfplay data ready  |
| `TRAINING_THRESHOLD_REACHED` | TrainingTrigger     | TrainingCoordinator              | Enough data for training |
| `TRAINING_COMPLETED`         | TrainingCoordinator | FeedbackLoop, Evaluation         | Training finished        |
| `EVALUATION_COMPLETED`       | EvaluationDaemon    | CurriculumIntegration, Scheduler | Model evaluated          |
| `MODEL_PROMOTED`             | PromotionController | Distribution, Curriculum         | Model passed gauntlet    |
| `REGRESSION_DETECTED`        | RegressionDetector  | Curriculum, TrainingCoordinator  | Model regressed          |
| `REGRESSION_CRITICAL`        | RegressionDetector  | DaemonManager, Alert             | Severe regression        |
| `PROGRESS_STALL_DETECTED`    | ProgressWatchdog    | Scheduler, Recovery              | Elo stalled >24h         |
| `QUALITY_PENALTY`            | QualityMonitor      | CurriculumIntegration            | Low quality selfplay     |

### 14.3 Threshold Configuration

Training thresholds are centralized in `app/config/thresholds.py`:

```python
from app.config.thresholds import (
    TrainingThresholds,
    QualityThresholds,
    EvaluationThresholds,
)

# Training triggers
TrainingThresholds.MIN_SAMPLES = 5000          # Minimum samples for training
TrainingThresholds.CONFIDENCE_MIN_SAMPLES = 1000  # Early trigger with high confidence
TrainingThresholds.CONFIDENCE_TARGET = 0.95    # 95% confidence interval

# Quality gates
QualityThresholds.MIN_GAME_LENGTH = 10         # Games shorter than this are suspect
QualityThresholds.HIGH_QUALITY_SCORE = 0.8     # High quality threshold
QualityThresholds.BLOCKED_QUALITY_SCORE = 0.3  # Quality below this blocks training

# Evaluation
EvaluationThresholds.MIN_WIN_RATE_VS_RANDOM = 0.85   # Must beat random 85%
EvaluationThresholds.MIN_WIN_RATE_VS_HEURISTIC = 0.60  # Must beat heuristic 60%
```

### 14.4 Regression Handling

Regression detection has severity levels:

| Severity   | Elo Drop | Response                             |
| ---------- | -------- | ------------------------------------ |
| `MINOR`    | 10-25    | Log warning, continue                |
| `MODERATE` | 25-50    | Increase selfplay for config         |
| `SEVERE`   | 50-100   | Block promotion, boost exploration   |
| `CRITICAL` | >100     | Rollback model, emergency curriculum |

**Configuration in `coordination_defaults.py`:**

```python
from app.config.coordination_defaults import RegressionConfig

RegressionConfig.MINOR_THRESHOLD = 25
RegressionConfig.MODERATE_THRESHOLD = 50
RegressionConfig.SEVERE_THRESHOLD = 100
RegressionConfig.CRITICAL_THRESHOLD = 150
RegressionConfig.ROLLBACK_ON_CRITICAL = True
```

### 14.5 Curriculum Feedback

When events indicate problems, curriculum weights adjust automatically:

| Event                             | Weight Adjustment | Effect                               |
| --------------------------------- | ----------------- | ------------------------------------ |
| `REGRESSION_DETECTED`             | +50% to config    | More selfplay for regressed config   |
| `QUALITY_PENALTY`                 | -30% to config    | Less selfplay until quality improves |
| `PROGRESS_STALL_DETECTED`         | +100% to config   | Double allocation for stalled config |
| `EVALUATION_COMPLETED` (high Elo) | -20% to config    | Reduce allocation for strong configs |

**Subscribing to curriculum events:**

```python
from app.coordination.curriculum_integration import CurriculumSignalBridge

# The bridge listens to events and adjusts weights
bridge = CurriculumSignalBridge.get_instance()
bridge.wire_to_event_router()  # Called during bootstrap

# Manual weight adjustment (usually not needed)
from app.coordination.selfplay_scheduler import get_selfplay_scheduler
scheduler = get_selfplay_scheduler()
scheduler.boost_config_priority("hex8_2p", multiplier=1.5, duration_hours=4)
```

### 14.6 Monitoring Feedback Loops

**Health endpoint:**

```bash
curl -s http://localhost:8790/status | jq '.feedback_loop'
# Returns: last_event, pending_actions, backpressure_active, etc.
```

**Key metrics:**

- `feedback_loop.events_processed` - Total events handled
- `feedback_loop.curriculum_adjustments` - Weight changes made
- `feedback_loop.regressions_detected` - Regression count
- `pipeline.stage_latencies` - Time per pipeline stage

**Troubleshooting:**

| Issue                    | Check                         | Solution                             |
| ------------------------ | ----------------------------- | ------------------------------------ |
| No training triggers     | `training_trigger` events     | Verify NEW_GAMES_AVAILABLE emitted   |
| Evaluation stuck         | `evaluation_daemon` queue     | Check backpressure, GPU availability |
| Curriculum not adjusting | `curriculum_integration` logs | Verify event subscriptions wired     |
| Model not promoted       | Gauntlet win rates            | Check evaluation thresholds          |

---

## 15. P2P Coordination Events (January 2026)

The P2P layer emits events that the coordination layer subscribes to. This enables automated recovery from network partitions, leader failures, and quorum loss.

### 15.1 Key P2P Events

| Event                        | Emitter                       | Subscribers                | Purpose                                |
| ---------------------------- | ----------------------------- | -------------------------- | -------------------------------------- |
| `PARTITION_HEALED`           | `partition_healer.py:514`     | `DataPipelineOrchestrator` | Trigger priority sync after healing    |
| `P2P_RECOVERY_NEEDED`        | `partition_healer.py:684`     | `P2PRecoveryDaemon`        | Restart orchestrator at max escalation |
| `NETWORK_ISOLATION_DETECTED` | `p2p_recovery_daemon.py:1133` | P2P orchestrator           | Trigger partition healing              |
| `LEADER_ELECTED`             | `p2p_orchestrator.py`         | `LeadershipCoordinator`    | Track leader changes                   |
| `QUORUM_LOST`                | `leader_election.py`          | `P2PRecoveryDaemon`        | Initiate quorum recovery               |
| `HOST_OFFLINE`               | `p2p_orchestrator.py`         | `UnifiedHealthManager`     | Update cluster health                  |
| `CIRCUIT_BREAKER_OPENED`     | `node_circuit_breaker.py`     | `P2PRecoveryDaemon`        | Track failing nodes                    |

### 15.2 Partition Healing Flow

When network partitions occur:

```
1. NETWORK_ISOLATION_DETECTED → partition_healer.trigger_healing_pass()
2. partition_healer attempts gossip-based healing
3. Success: PARTITION_HEALED → DataPipelineOrchestrator triggers priority sync
4. Failure (max escalation): P2P_RECOVERY_NEEDED → P2PRecoveryDaemon restarts orchestrator
```

### 15.3 Circuit Breaker Gossip Replication

Open circuit breaker states are replicated via gossip protocol:

```python
# State collection (gossip_protocol.py)
def _get_circuit_breaker_gossip_state() -> dict:
    """Collect OPEN/HALF_OPEN circuits for gossip."""
    return {
        target: {
            "state": cb.state,
            "failure_count": cb.failure_count,
            "age_seconds": time.time() - cb.opened_at,
        }
        for target, cb in registry.items()
        if cb.state in ("OPEN", "HALF_OPEN")
    }

# State processing (gossip_protocol.py)
def _process_circuit_breaker_states(peer_states: dict) -> None:
    """Apply peer CB states as preemptive failures."""
    for target, state in peer_states.items():
        if state["age_seconds"] < 300:  # Fresh circuits only
            registry.record_failure(target, preemptive=True)
```

**Benefits:**

- Cluster-wide failure awareness within ~15 seconds
- Reduced connection attempts to known-failing targets
- Faster recovery from network issues

### 15.4 Adding New P2P Event Handlers

When adding handlers for P2P events:

1. **Import DataEventType correctly:**

```python
from app.distributed.data_events import DataEventType
# NOT from app.coordination.data_events
```

2. **Subscribe in the appropriate coordinator:**

```python
# In data_pipeline_orchestrator.py subscribe_to_data_events()
router.subscribe(
    DataEventType.PARTITION_HEALED.value,
    self._on_partition_healed,
)
```

3. **Handle gracefully with fallbacks:**

```python
def _on_partition_healed(self, event) -> None:
    try:
        payload = event.payload if hasattr(event, "payload") else event
        # Handle event...
    except ImportError:
        logger.debug("Event emission not available")
    except (AttributeError, KeyError, TypeError) as e:
        self._record_error(f"_on_partition_healed: {e}")
```

### 15.5 P2P Health Monitoring

The P2P layer has 31 health monitoring mechanisms across multiple layers:

| Layer       | Mechanisms | Examples                           |
| ----------- | ---------- | ---------------------------------- |
| Application | 6          | DaemonManager health checks        |
| P2P         | 8          | Peer liveness, quorum health       |
| Gossip      | 5          | Anti-entropy, state sync           |
| Transport   | 4          | Multi-transport failover           |
| Leader      | 4          | Lease epoch, split-brain detection |
| Voter       | 4          | Quorum monitoring, auto-demotion   |

**Health check command:**

```bash
curl -s http://localhost:8770/status | jq '.health'
```

### 15.6 P2P Event Troubleshooting

| Issue                                 | Check                    | Solution                                |
| ------------------------------------- | ------------------------ | --------------------------------------- |
| Partition not healing                 | `/status` endpoint       | Check connectivity, gossip state        |
| No PARTITION_HEALED events            | Logs in partition_healer | Verify injections succeeding            |
| P2P_RECOVERY_NEEDED firing repeatedly | Escalation state         | Reset escalation, check root cause      |
| Circuit breakers not propagating      | Gossip CB state          | Verify gossip interval, state freshness |
