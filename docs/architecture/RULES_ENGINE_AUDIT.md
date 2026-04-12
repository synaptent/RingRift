# Rules Engine Audit

Updated: April 11, 2026

This audit records the Phase 19 rules-quality review so the move/phase surface can be checked without rediscovering which files are canonical and which ones are legacy compatibility layers.

## Canonical Surfaces

- Canonical TS rules type surface: [`src/shared/types/game.ts`](../../src/shared/types/game.ts)
- Canonical Python mirror: [`ai-service/app/models/core.py`](../../ai-service/app/models/core.py)
- Canonical storage contract: [`ai-service/app/rules/history_contract.py`](../../ai-service/app/rules/history_contract.py)
- Python rules adapter under audit: [`ai-service/app/rules/default_engine.py`](../../ai-service/app/rules/default_engine.py)

## Important Distinction

[`src/shared/engine/types.ts`](../../src/shared/engine/types.ts) is not the canonical move inventory. Its `ActionType` union is a legacy engine-interface subset used by older shared-engine helpers. It omits newer bookkeeping, recovery, and terminal move types such as `no_*_action`, `skip_recovery`, `forced_elimination`, `resign`, and `timeout`.

Move completeness audits should use [`src/shared/types/game.ts`](../../src/shared/types/game.ts), not [`src/shared/engine/types.ts`](../../src/shared/engine/types.ts).

## DefaultRulesEngine Coverage

After the Phase 19 update, `DefaultRulesEngine.validate_move()` covers:

- Placement: `place_ring`, `skip_placement`, `no_placement_action`, `swap_sides`
- Movement and recovery: `move_stack`, `recovery_slide`, `skip_recovery`, `no_movement_action`
- Capture: `overtaking_capture`, `continue_capture_segment`, `chain_capture`, `skip_capture`
- Line processing: `process_line`, `choose_line_option`, `choose_line_reward`, `line_formation`, `no_line_action`
- Territory processing: `choose_territory_option`, `process_territory_region`, `eliminate_rings_from_stack`, `territory_claim`, `skip_territory_processing`, `no_territory_action`
- Forced elimination and terminal meta moves: `forced_elimination`, `resign`, `timeout`

Recovery validation now uses the dedicated recovery validator and mutator modules instead of falling through as an unknown move type.

## Intentional Remaining Gaps

- `DefaultRulesEngine` still delegates canonical semantics to `GameEngine.apply_move`. That is intentional; the rules adapter is not the semantic SSoT.
- Mutator shadow contracts remain selective. Board-changing paths have dedicated mutator assertions, while pure bookkeeping / terminal moves (`skip_recovery`, `skip_territory_processing`, `no_territory_action`, `resign`, `timeout`) are pass-through no-geometry moves. They rely on the canonical `GameEngine` for phase/victory transitions instead of a separate board-mutation contract.
- `GameResult.reason` in TS is a broader product/API vocabulary (`resignation`, `draw`, `abandonment`, `game_completed`, etc.) than Python training-data victory labels (`ring_elimination`, `territory`, `lps`, `stalemate`, `timeout`). That is a cross-layer terminology difference, not a rules-engine completeness bug.
