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

## Next Consolidation Lanes (Future phases)

Phase 2: AI engine naming consolidation (gumbel/mcts/descent)

- Normalize engine identifiers across training/selfplay scripts and AI factory.
- Align `gumbel`, `gumbel-mcts`, and `gumbel_mcts` naming with one canonical
  identifier plus well-documented aliases.

Phase 3: Script consolidation (per CODEBASE_CONSOLIDATION_PLAN.md)

- Consolidate duplicate tournament/selfplay/training entrypoints.
- Migrate retained functionality into the unified scripts and document removals.

## Notes

- Legacy replay compatibility should remain available only for migration,
  parity for historical fixtures, or audit tooling. Production paths should
  consume canonical records and stay strict.
