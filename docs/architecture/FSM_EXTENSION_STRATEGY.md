# FSM Extension Strategy

**Context:** We now have an explicit Turn FSM (`src/shared/engine/fsm/TurnStateMachine.ts`) and adapter (`src/shared/engine/fsm/FSMAdapter.ts`). FSM validation (shadow/active) is wired into `turnOrchestrator.ts`. Extending the FSM approach can improve determinism, parity, and debugging if applied deliberately.

## Target Areas (with Pros / Cons / Risk–Reward)

1) **Turn Orchestrator Integration (TS)**
   - **Pros:** Single source of truth for phase transitions; eliminates duplicated logic; clearer invariant enforcement; easier diffing vs Python.
   - **Cons/Risks:** Large refactor; potential regressions in live sandbox; needs decision-resolution plumbing.
   - **Reward:** High correctness and maintainability payoff; simplifies future rule changes.
2) **Move Validation Unification**
   - **Pros:** FSM-active validation becomes authoritative; removes divergent legacy validators; fewer phase-invariant escapes.
   - **Cons/Risks:** Might reject legacy recordings/tests; needs migration plan for fixtures.
   - **Reward:** High—reduces class of phase/move bugs; cleaner API surface.
3) **Decision Surfaces (line/territory/FE)**
   - **Pros:** PendingDecision emission can be purely FSM-driven; predictable loops for multiple regions/lines; fewer “silent skips”.
   - **Cons/Risks:** Needs UI/client adjustments; careful UX telemetry alignment.
   - **Reward:** Medium–High—reduces off-phase moves; better UX/state explainability.
4) **Python Phase Machine Parity**
   - **Pros:** Shared FSM semantics in Python (`app/rules/phase_machine.py`) keep parity tight; easier bundle diffing.
   - **Cons/Risks:** Requires tight mirroring or codegen; dual maintenance if manual.
   - **Reward:** High—directly reduces TS↔Python divergences.
5) **UI/Telemetry (GameHUD/Replay/Explanation)**
   - **Pros:** UI can render phase/event timelines directly from FSM actions; better teaching overlays and replay debugging.
   - **Cons/Risks:** Needs adapter layer for existing view models; risk of churn in tests.
   - **Reward:** Medium—clarity and debuggability for players and devs.
6) **Testing & Fixtures**
   - **Pros:** FSM golden tests and property-based sequences catch regressions early; reusable fixtures for both TS/Python.
   - **Cons/Risks:** More test plumbing; fixture drift if not shared.
   - **Reward:** Medium–High—guards against future regression.
7) **Data / Training Pipeline**
   - **Pros:** Parity and canonical-history gates can consume FSM validation results; clearer labels for phase/move legality in datasets.
   - **Cons/Risks:** Needs export format updates; retraining triggers.
   - **Reward:** Medium—cleaner data provenance and model training signals.
8) **Observability / Tooling**
   - **Pros:** FSM traces as structured logs; easier to diff recorded vs expected transitions; better invariant snapshots.
   - **Cons/Risks:** Log volume; requires viewer tooling.
   - **Reward:** Medium—debug speedup.

## Recommendations (prioritized)

P0-P1:
- Migrate turnOrchestrator phase advancement to rely on FSM transitions/adapters; keep shadow/active validation always on in dev/test.
- Unify move validation on FSM-active; retire redundant phase-move guards once fixtures updated.
- Extend Python `phase_machine` to mirror the FSM transitions verbatim; add bundle tests around end-of-game current_player ownership.
- Use FSM to drive PendingDecision surfaces for territory/line/FE to remove ad-hoc skips.

P2:
- Expose FSM action traces to UI/telemetry and replay panels for debugging/teaching.
- Add property-based FSM tests plus cross-language fixture generation.
- Feed FSM validation outcomes into data export to tag non-canonical sequences.

## Work Plan by Area

### A) Turn Orchestrator → FSM Control
- Replace manual phase-routing branches in `turnOrchestrator.ts` with FSM `transition` + adapter events.
- Keep a compatibility layer for delegates/pending decisions; map FSM-required events to existing move shapes.
- Add feature flag for “FSM-driven orchestrator” and enable in test/parity first.

### B) Move Validation Unification
- Make FSM-active the default validator; route `validateMove` through `validateMoveWithFSM`.
- Update fixtures/tests that assume legacy coercions; add targeted tests for rejection cases (off-phase moves, missing bookkeeping).
- Remove duplicated phase-invariant checks once parity is clean.

### C) Decision Surfaces (line/territory/FE)
- Emit pending decisions from FSM state (pendingLines/pendingRegions/FE options) instead of re-deriving in orchestrator.
- Ensure multiple-region and multi-line flows loop through FSM until empty; forbid auto-advance without explicit no_* moves.
- Add tests for “single region still requires explicit process_territory_region” and “FE surfaces only after territory”.

### D) Python Parity
- Align `app/rules/phase_machine.py` transitions to the TS FSM table (consider generating from shared JSON/YAML spec).
- Add parity bundles focused on end-of-turn and game_over ownership; fix current_player divergence.
- Keep `GameEngine.get_phase_requirement` consistent with FSM-derived requirements.

### E) UI/Telemetry
- Add an adapter to surface FSM actions/decisions to `GameHUD`/Replay/Explanation builders.
- Log FSM state/event traces in replay harness for debuggability; gate by env flag.

### F) Testing/Fixtures
- Expand `tests/unit/fsm` with property-based random event sequences that must preserve invariants.
- Generate cross-language fixtures: TS FSM expected transitions → Python replay assertions.
- Add targeted tests for FE entry/exit, territory loops, and no-rings skip rules.

### G) Data/Training
- Thread FSM validation results into data export metadata (canonical: bool, phase_move_ok: bool).
- Block dataset generation on FSM-active validation pass for new DBs.

### H) Observability/Tooling
- Add optional trace emission (JSONL) for FSM transitions during self-play/replay.
- Provide a small CLI to diff FSM traces between TS and Python for a given game ID.

## Immediate Next Steps (suggested sequence)
1. Fix end-of-game current_player mismatch by aligning TS/Python phase-machine transitions at game_over (use existing state bundle).
2. Implement FSM-driven pending decisions for territory/FE to prevent early FE and off-phase no_* moves.
3. Add Python phase-machine parity tests generated from the TS FSM transition table.
4. Enable FSM-active validation by default in test/parity workflows; update fixtures that relied on skip_placement with zero rings.
5. Prototype FSM-driven orchestrator flag; run parity + canonical history gates on small DB to validate.
