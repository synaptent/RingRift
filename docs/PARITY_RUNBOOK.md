# Parity & Canonical History Runbook

Use this runbook when TS↔Python parity fails or canonical history checks fail.
For the extended workflow, see `docs/runbooks/PARITY_VERIFICATION_RUNBOOK.md`.

## Trigger Signals

- `check_ts_python_replay_parity.py` reports `structure != "good"` or
  `diverged_at != None`.
- `validate_canonical_history_for_game` returns `is_canonical = false`.
- `generate_canonical_selfplay.py` reports `canonical_ok: false`.

## Triage Steps (recommended order)

1. Identify a single failing game id.
2. Re-run parity with state bundles:
   - `PYTHONPATH=. python scripts/check_ts_python_replay_parity.py --db <db> --emit-state-bundles-dir <dir>`
3. Inspect the first divergence:
   - `PYTHONPATH=. python scripts/diff_state_bundle.py --bundle <bundle.json>`
4. Confirm phase/move contract invariants:
   - Ensure every phase transition is recorded as a move.
   - Verify forced elimination is a distinct phase and move type.
5. Fix logic in TS/Python engine code (not data), then rerun parity.

## Canonical History Validation

- For a single DB:
  - `PYTHONPATH=. python scripts/check_canonical_phase_history.py --db <db>`
- For all canonical DBs:
  - `PYTHONPATH=. python scripts/run_canonical_guards.py`

## Escalation

- If divergence is in phase ordering or FE/no-action semantics, update both:
  - `src/shared/engine/**`
  - `ai-service/app/game_engine.py`
  - `ai-service/app/rules/history_contract.py`
  - 관련 tests and docs.
