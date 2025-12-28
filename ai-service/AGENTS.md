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
