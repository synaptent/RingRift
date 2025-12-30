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

The coordination layer has 89 daemon types (78 active, 11 deprecated). When creating new daemons, follow these patterns.

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

### 11.5 Reference Documentation

- **Full registry**: `docs/DAEMON_REGISTRY.md` (89 daemons, dependencies, categories)
- **Types and state**: `app/coordination/daemon_types.py`
- **Specifications**: `app/coordination/daemon_registry.py`
- **Factory runners**: `app/coordination/daemon_runners.py`
- **Base classes**: `app/coordination/handler_base.py` (550 LOC, 45 tests)
