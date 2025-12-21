# Integration Consolidation Phase 1 Plan (Legacy Replay Separation)

Status: in_progress (2025-12-20)
Scope: TS shared engine replay helpers + legacy separation; future consolidation lanes scoped below.

References:

- docs/planning/CODEBASE_CONSOLIDATION_PLAN.md
- docs/planning/CANONICAL_ENGINE_PARITY_AND_SSOT_HARDENING.md
- RULES_CANONICAL_SPEC.md (RR-CANON-R073, R075)

## Goals (Phase 1)

1. Keep canonical replay tooling strict to RR-CANON-R075 (no silent phase fixes).
2. Isolate legacy replay compatibility in `src/shared/engine/legacy/**`.
3. Mark legacy replay helpers as deprecated and document the migration path.

## Phase 1 Tasks (This slice)

1. Audit replay reconstruction usage (shared engine, client, tests).
2. Split replay reconstruction into:
   - Canonical helper (strict, no replayCompatibility).
   - Legacy helper (explicit replayCompatibility path under legacy/).
3. Update references that require legacy compatibility (if any).
4. Add comments + deprecation notice in legacy helper.

Acceptance criteria:

- `reconstructStateAtMove` no longer enables replayCompatibility by default.
- Legacy compatibility is opt-in via `src/shared/engine/legacy/legacyReplayHelpers.ts`.
- All existing tests compile; legacy replay access has an explicit import path.

## Recent Progress (2025-12-21)

### Completed

1. **Python phase auto-advance extraction**: Moved `_auto_advance_phase()` from
   `nnue_policy.py` to `app/rules/legacy/phase_auto_advance.py`. This function
   synthesizes bookkeeping moves for legacy selfplay data, violating RR-CANON-R075.
   Now isolated with deprecation warning and metrics tracking.

2. **Server-side legacy normalization**: Updated `TurnEngineAdapter.ts` to normalize
   legacy move types unconditionally (not just in replay mode). Entry points now
   ensure aggregates only see canonical types.

3. **AI engine naming**: Verified factory.py properly normalizes CLI hyphens to
   internal underscores (line 768). No changes needed.

### Pending (Python legacy replay separation)

The following Python code paths need extraction to `app/rules/legacy/`:

1. **`app/db/game_replay.py:_auto_inject_before_move()`** (~200 lines)
   - Auto-injects bookkeeping moves to bridge phase gaps in non-canonical recordings
   - Contains extensive RR-PARITY-FIX comments for phase coercion
   - Target: Extract to `app/rules/legacy/replay_phase_injection.py`
   - Deprecation: Q2 2026 (after canonical data migration complete)

## Next Consolidation Lanes (Future phases)

Phase 2: Python legacy replay separation

- Extract `_auto_inject_before_move()` to legacy module
- Make `get_state_at_move(auto_inject=True)` route to legacy module
- Add deprecation warnings and metrics tracking

Phase 3: AI engine naming consolidation (gumbel/mcts/descent) ✅ COMPLETE

- Verified: factory.py already normalizes engine identifiers
- `gumbel-mcts` CLI → `gumbel_mcts` internal via `agent_key.replace("-", "_")`

Phase 4: Script consolidation (per CODEBASE_CONSOLIDATION_PLAN.md)

- Consolidate duplicate tournament/selfplay/training entrypoints.
- Migrate retained functionality into the unified scripts and document removals.

## Notes

- Legacy replay compatibility should remain available only for migration,
  parity for historical fixtures, or audit tooling. Production paths should
  consume canonical records and stay strict.
