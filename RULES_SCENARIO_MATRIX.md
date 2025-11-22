# RingRift Rules / FAQ → Test Scenario Matrix

**Purpose:** This file is the canonical map from the **rules documents** to the **Jest test suites**.

---

## 0.a Scenario Axis IDs (Movement / Chain / Lines / Territory / Victory)

This section introduces short, axis-oriented IDs that can be used in issues,
PRs, and inline test names. Each axis ID (M*/C*/L*/T*/V\*) points at a concrete
`rulesMatrix` scenario (where applicable) and at at least one backend and, when
relevant, sandbox test.

> Over time, these IDs should grow to cover all high-value rules/FAQ examples.
> When you add a new scenario row elsewhere in this file, consider giving it an
> axis ID here as well.

| Axis ID | Axis      | rulesMatrix ref.id                                                                                                                        | Rule / FAQ focus                                                                                 | Backend test(s)                                                                                                                   | Sandbox / parity test(s)                                                                                                        | Status  |
| ------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------- |
| **M1**  | Movement  | `Rules_8_2_Q2_minimum_distance_square8`                                                                                                   | §8.2, FAQ Q2 – minimum distance (square8)                                                        | `tests/unit/RuleEngine.movement.scenarios.test.ts`; `tests/scenarios/RulesMatrix.Movement.RuleEngine.test.ts`                     | –                                                                                                                               | COVERED |
| **M2**  | Movement  | `Rules_8_2_Q2_markers_any_valid_space_beyond_square8`                                                                                     | §8.2, FAQ Q2–Q3 – landing beyond marker runs                                                     | `tests/scenarios/RulesMatrix.Movement.RuleEngine.test.ts`                                                                         | `tests/unit/ClientSandboxEngine.moveParity.test.ts`; `tests/scenarios/RulesMatrix.Movement.ClientSandboxEngine.test.ts`         | COVERED |
| **M3**  | Movement  | `Rules_9_10_overtaking_capture_vs_move_stack_parity`                                                                                      | §9–10 – overtaking capture vs simple move parity                                                 | `tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`                                                                  | `tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`                                                                | COVERED |
| **C1**  | Chain cap | `Rules_10_3_Q15_3_1_180_degree_reversal_basic`                                                                                            | §10.3, FAQ 15.3.1 – 180° reversal                                                                | `tests/scenarios/ComplexChainCaptures.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts`                     | `tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts`                                                          | COVERED |
| **C2**  | Chain cap | `Rules_10_3_Q15_3_2_cyclic_pattern_triangle_loop`                                                                                         | §10.3, FAQ 15.3.2 – cyclic triangle pattern                                                      | `tests/scenarios/ComplexChainCaptures.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts`                     | `tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts`                                                          | COVERED |
| **C3**  | Chain cap | –                                                                                                                                         | §10.3, FAQ 15.3.x – hex cyclic capture patterns                                                  | `tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`; `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts`            | –                                                                                                                               | COVERED |
| **L1**  | Lines     | `Rules_11_2_Q7_exact_length_line`                                                                                                         | §11.2, FAQ 7 – exact-length line                                                                 | `tests/unit/GameEngine.lines.scenarios.test.ts`; `tests/scenarios/RulesMatrix.GameEngine.test.ts`                                 | `tests/unit/ClientSandboxEngine.lines.test.ts`                                                                                  | COVERED |
| **L2**  | Lines     | `Rules_11_3_Q22_overlength_line_option2_default`                                                                                          | §11.2–11.3, FAQ 22 – overlength, Option 2                                                        | `tests/unit/GameEngine.lines.scenarios.test.ts`; `tests/scenarios/RulesMatrix.GameEngine.test.ts`                                 | `tests/unit/ClientSandboxEngine.lines.test.ts`                                                                                  | COVERED |
| **L3**  | Lines     | –                                                                                                                                         | §§11–12; FAQ 7, 15, 22–23 – combined line + territory turns                                      | `tests/scenarios/LineAndTerritory.test.ts`                                                                                        | –                                                                                                                               | COVERED |
| **L4**  | Lines     | –                                                                                                                                         | §11.2–11.3; FAQ 7, 22 – line_processing decision enumeration (process_line / choose_line_reward) | `tests/unit/GameEngine.lines.scenarios.test.ts` (line_processing_getValidMoves_exposes_process_line_and_choose_line_reward_moves) | –                                                                                                                               | COVERED |
| **T1**  | Territory | –                                                                                                                                         | §12.2–12.3, FAQ 10, 15, 20, 23 – square boards                                                   | `tests/unit/GameEngine.territory.scenarios.test.ts`; `tests/scenarios/LineAndTerritory.test.ts`                                   | `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts`                                                                 | COVERED |
| **T2**  | Territory | –                                                                                                                                         | §12.2–12.3, FAQ 10, 15, 20, 23 – hex boards                                                      | `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`                                                                        | `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`                                                             | COVERED |
| **T3**  | Territory | `Rules_12_2_Q23_region_not_processed_without_self_elimination_square19`, `Rules_12_2_Q23_region_processed_with_self_elimination_square19` | §12.2, FAQ 23 – self-elimination prerequisite for disconnected regions                           | `tests/unit/GameEngine.territory.scenarios.test.ts`                                                                               | `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts`; `tests/scenarios/RulesMatrix.Territory.ClientSandboxEngine.test.ts` | COVERED |
| **T4**  | Territory | –                                                                                                                                         | §12.2–12.3; FAQ 23 – territory_processing decision enumeration (process_territory_region)        | `tests/unit/GameEngine.territory.scenarios.test.ts` (territory_processing_getValidMoves_exposes_process_territory_region_moves)   | –                                                                                                                               | COVERED |
| **V1**  | Victory   | –                                                                                                                                         | §13, FAQ 11, 18, 21, 24 – victory & stalemate                                                    | `tests/unit/GameEngine.victory.scenarios.test.ts`; `tests/scenarios/ForcedEliminationAndStalemate.test.ts`                        | `tests/unit/ClientSandboxEngine.victory.test.ts`; `tests/unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts`                      | COVERED |
| **V2**  | Victory   | –                                                                                                                                         | §4.4, §13.3–13.5, FAQ 11, 24 – forced elimination & stalemate ladder                             | `tests/scenarios/ForcedEliminationAndStalemate.test.ts`; `tests/unit/GameEngine.turnSequence.scenarios.test.ts`                   | –                                                                                                                               | COVERED |

As more scenarios are added in `tests/scenarios/rulesMatrix.ts` (for example,
for territory and victory), consider assigning them M*/C*/L*/T*/V\* IDs here for
quick reference.

---

It answers:

- “For this rule / FAQ example, where is the corresponding test?”
- “Which parts of the rules are fully covered vs partially covered vs not yet encoded?”

It is meant to evolve alongside:

- `ringrift_complete_rules.md`
- `ringrift_compact_rules.md`
- `RULES_ANALYSIS_PHASE2.md`
- `tests/README.md`
- `CURRENT_STATE_ASSESSMENT.md` / `KNOWN_ISSUES.md`

Status legend:

- **COVERED** – there is at least one explicit, named scenario test for this rule/FAQ cluster.
- **PARTIAL** – behaviour is exercised indirectly (e.g. via mechanics/unit tests or broader scenarios), but there is not yet a direct, rule‑tagged scenario.
- **PLANNED** – no meaningful coverage yet; scenario suite still to be added.

Coverage targets (minimum expectations):

- **Turn sequence & forced elimination (§4, FAQ 15.2, 24):**
  - At least one COVERED scenario per relevant board family (square8, square19, hex) exercising skip/placement, movement, capture entry, and forced elimination resolution on the backend.
  - At least one parity/flow test (e.g. seeded AI or WebSocket/AI integration) that demonstrates the same turn structure end-to-end.
- **Non‑capture movement & markers (§8.2–8.3, FAQ 2–3):**
  - Rules-matrix scenarios for minimum distance and marker landing on square8, square19, and hex, plus sandbox movement/parity suites that enforce the unified landing rule.
- **Overtaking captures & chain patterns (§9–10, FAQ 5–6, 9, 12, 14, 15.3.x):**
  - Square and hex examples for basic overtaking, 180° reversal, and cyclic patterns, with backend and sandbox parity where applicable.
- **Lines & graduated rewards (§11, FAQ 7, 22):**
  - Exact-length and overlength line scenarios on square and hex boards, including explicit Option 1 vs Option 2 choices, backed by backend scenario tests and sandbox line-processing parity.
- **Territory disconnection & chain reactions (§12, FAQ 10, 15, 20, 23):**
  - Square and hex scenarios for region discovery, border rules, self‑elimination prerequisites, and multi-step chain reactions, with mirrored sandbox tests for ClientSandboxEngine.
- **Victory & stalemate (§13, FAQ 11, 18, 21, 24):**
  - Backend victory and stalemate-ladder scenarios asserting winner, reason, and tie-break ordering, plus sandbox parity tests that reach the same terminal outcomes.
- **PlayerChoice flows (§4.5, §10.3, §11–12, FAQ 7, 15, 22–23):**
  - At least one test per choice type (line order, line reward, ring elimination, region order, capture direction) wiring: GameEngine → PlayerInteractionManager → WebSocket/AI/sandbox.
- **Backend ↔ sandbox parity & S‑invariant (compact rules §9, §13.5):**
  - Seeded trace/AI-parallel suites where backend and sandbox traces remain in lockstep on actions, phases, hashes, and S, treated as CI gates once P0.2 gaps are closed.

> **Naming convention:** When adding new tests, prefer `describe` / `it` names that reference the rule or FAQ explicitly, e.g. `Q15_3_1_180_degree_reversal` or `Rules_11_2_LineReward_Option1VsOption2`.

---

## 0. Canonical scenario table (high‑leverage rules)

This table summarizes the most critical, high‑leverage scenarios using a unified format. It is **not yet exhaustive**; as more scenarios are added, they should follow this pattern.

| Rule / FAQ Ref                                     | Scenario Name                                        | Board Type(s)              | Engine(s)                  | Test File(s)                                                                                                                                                                                                                                                                                                | Status  |
| -------------------------------------------------- | ---------------------------------------------------- | -------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| §§9–10; FAQ 15.3.1–15.3.2 (180° + cyclic captures) | `complex_chain_captures_square_examples`             | 8×8, 19×19                 | Backend                    | `tests/scenarios/ComplexChainCaptures.test.ts`                                                                                                                                                                                                                                                              | COVERED |
| §§9–10; FAQ 15.3.1                                 | `chain_capture_180_degree_reversal_examples`         | 8×8, 19×19                 | Backend                    | `tests/scenarios/ComplexChainCaptures.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts`                                                                                                                       | COVERED |
| §§9–10; FAQ 15.3.1–15.3.2                          | `chain_capture_rules_matrix_examples`                | 8×8, 19×19                 | Backend, Sandbox           | `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts`                                                                                                                                                                       | COVERED |
| §§9–10; FAQ 15.3.1–15.3.2                          | `hex_cyclic_captures_height3`                        | hex                        | Backend                    | `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts`; `tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`                                                                                                                                                                                      | COVERED |
| §§9–10; Overtaking vs Move                         | `overtaking_capture_vs_move_stack_parity`            | 8×8                        | Backend, Sandbox           | `tests/unit/MovementCaptureParity.RuleEngine_vs_Sandbox.test.ts`                                                                                                                                                                                                                                            | COVERED |
| §11.2–11.3; FAQ 7, 22                              | `line_reward_overlength_square_examples`             | 8×8, 19×19                 | Backend                    | `tests/unit/GameEngine.lines.scenarios.test.ts`                                                                                                                                                                                                                                                             | COVERED |
| §11.2–11.3; FAQ 7, 22                              | `line_reward_rules_matrix_backend_examples`          | 8×8                        | Backend                    | `tests/scenarios/RulesMatrix.GameEngine.test.ts`                                                                                                                                                                                                                                                            | COVERED |
| §11.2–11.3; FAQ 22                                 | `line_reward_option1_full_collapse_square19_planned` | 19×19                      | Backend, Sandbox (planned) | TBD – planned backend and sandbox suites driven by `lineRewardRuleScenarios` id `Rules_11_3_Q22_overlength_line_option1_full_collapse_square19`                                                                                                                                                             | PLANNED |
| §11; FAQ 7                                         | `sandbox_line_processing_parity`                     | 8×8, 19×19, hex            | Sandbox                    | `tests/unit/ClientSandboxEngine.lines.test.ts`                                                                                                                                                                                                                                                              | COVERED |
| §12.2–12.3; FAQ 10, 15, 20, 23                     | `territory_disconnection_square_examples`            | 8×8, 19×19                 | Backend, Sandbox           | `tests/unit/GameEngine.territory.scenarios.test.ts`; `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts` (uses `territoryRuleScenarios` including `Rules_12_2_Q23_region_not_processed_without_self_elimination_square19` and `Rules_12_2_Q23_region_processed_with_self_elimination_square19`) | COVERED |
| §12.2–12.3; FAQ 10, 15, 20, 23                     | `territory_disconnection_hex_examples`               | hex                        | Backend, Sandbox           | `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`; `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`                                                                                                                                                                             | COVERED |
| §§11–12; FAQ 7, 15, 22–23                          | `combined_line_and_territory_turns`                  | 8×8, 19×19, hex            | Backend                    | `tests/scenarios/LineAndTerritory.test.ts` (driven by `lineAndTerritoryRuleScenarios`, e.g. `Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_{square8,square19,hexagonal}`)                                                                                                                  | COVERED |
| §§11–12; FAQ 7, 15, 22–23                          | `combined_line_and_territory_square19_hex`           | 19×19, hex                 | Backend                    | `tests/scenarios/LineAndTerritory.test.ts`                                                                                                                                                                                                                                                                  | COVERED |
| §4.x; §§8–13; FAQ 15.2, 24                         | `forced_elimination_and_stalemate_ladder`            | 8×8                        | Backend                    | `tests/scenarios/ForcedEliminationAndStalemate.test.ts` (uses `victoryRuleScenarios` ids `Rules_4_4_13_5_Q24_forced_elimination_single_blocked_stack` and `Rules_13_4_13_5_Q11_structural_stalemate_rings_in_hand_become_eliminated`)                                                                       | COVERED |
| §4.x; FAQ 15.2, 24                                 | `turn_sequence_forced_elimination_matrix`            | 8×8, 19×19, hex (selected) | Backend                    | `tests/unit/GameEngine.turnSequence.scenarios.test.ts`                                                                                                                                                                                                                                                      | COVERED |
| §4.x; FAQ 15.2                                     | `sandbox_mixed_players_place_then_move`              | 8×8                        | Sandbox                    | `tests/unit/ClientSandboxEngine.mixedPlayers.test.ts`                                                                                                                                                                                                                                                       | COVERED |
| §13; FAQ 11, 18, 21, 24                            | `stalemate_tiebreak_ladder_examples`                 | 8×8, 19×19                 | Backend, Sandbox           | `tests/unit/GameEngine.victory.scenarios.test.ts` (Rules*13_1–13_6*\*), `tests/unit/ClientSandboxEngine.victory.test.ts`                                                                                                                                                                                    | COVERED |
| §13.1–13.2; FAQ 18, 21                             | `victory_threshold_rules_matrix_backend_examples`    | 8×8                        | Backend, Sandbox           | `tests/scenarios/RulesMatrix.Victory.GameEngine.test.ts`; `tests/scenarios/RulesMatrix.Victory.ClientSandboxEngine.test.ts` (driven by `victoryRuleScenarios` ids `Rules_13_1_ring_elimination_threshold_square8` and `Rules_13_2_territory_control_threshold_square8`)                                     | COVERED |
| Compact rules §9; `RULES_ANALYSIS_PHASE2` §4       | `backend_vs_sandbox_trace_parity_seed5`              | 8×8 (seeded AI)            | Backend, Sandbox, Traces   | `tests/unit/Backend_vs_Sandbox.traceParity.test.ts`; `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts`                                                                                                                                                                                               | COVERED |
| Compact rules §9; `RULES_ANALYSIS_PHASE2` §4       | `backend_vs_sandbox_trace_parity_extra_seeds`        | 8×8, 19×19, hex (selected) | Backend, Sandbox, Traces   | `tests/unit/ParityDebug.seed14.trace.test.ts`; `tests/unit/TraceParity.seed14.firstDivergence.test.ts`; `tests/unit/Seed17Move52Parity.GameEngine_vs_Sandbox.test.ts`; `tests/unit/SInvariant.seed17FinalBoard.test.ts`                                                                                     | PARTIAL |
| Compact rules §9; `RULES_ANALYSIS_PHASE2` §4       | `sandbox_ai_simulation_progress_invariant`           | 8×8, 19×19, hex (various)  | Backend, Sandbox           | `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`; `tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts`; `tests/unit/ProgressSnapshot.core.test.ts`; `tests/unit/ProgressSnapshot.sandbox.test.ts`                                                                                                    | COVERED |
| §11.2–11.3; FAQ 7, 22                              | `line_reward_choice_ai_service_backing`              | 8×8                        | Backend, AI Service        | `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`                                                                                                                                                                                                                                       | COVERED |
| §12.3; FAQ 15, 23                                  | `region_order_choice_ai_and_sandbox`                 | 8×8                        | Backend, Sandbox, AI       | `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`; `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`                                                                                                                                                                                    | COVERED |

> When adding new high‑value scenarios (especially for rules/FAQ edge cases not yet encoded), append a row here **and** add a more detailed row in the relevant cluster table below.

---

## 0. Index by Rules Sections

This is a high‑level map from rule sections to clusters below.

| Rules section(s)                            | Cluster                                                           |
| ------------------------------------------- | ----------------------------------------------------------------- |
| §4.x, FAQ 15.2, FAQ 24                      | Turn sequence & forced elimination                                |
| §8.2–8.3, FAQ 2–3                           | Non‑capture movement & markers                                    |
| §9–10, FAQ 5–6, 9, 12, 14, 15.3.1–15.3.2    | Overtaking captures & chain patterns                              |
| §11, FAQ 7, 22                              | Line formation, graduated rewards, line ordering                  |
| §12, FAQ 10, 15, 20, 23                     | Territory disconnection, self‑elimination, border colour, chains  |
| §13, FAQ 11, 18, 21, 24                     | Victory conditions & stalemate ladder                             |
| §4.5, 9–12 (choice hooks), FAQ 7, 15, 22–23 | PlayerChoice flows (line order/reward, elimination, region order) |
| §4, 8–13 (via GameTrace + S‑invariant)      | Backend ↔ sandbox parity & progress invariant                    |

The following sections break these down in more detail.

---

## 1. Turn sequence & forced elimination

**Rules/FAQ:**

- `ringrift_complete_rules.md` §4.x (Turn Structure)
- Compact rules §2.2–2.3 (movement phase & forced elimination)
- FAQ 15.2 (flowchart of a turn), FAQ 24 (forced elimination when blocked)

| Coverage    | Scenario / intent                                                                  | Jest file(s)                                                                                                               | Engines | Notes                                                                                                                                                                                                                                                                    |
| ----------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **COVERED** | Full turn sequence including forced elimination and skipping dead players          | `tests/unit/GameEngine.turnSequence.scenarios.test.ts`                                                                     | Backend | Encodes multi‑player cases where some players are blocked with stacks and others are out of material; includes the rule‑tagged three‑player scenario `Rules_4_2_three_player_skip_and_forced_elimination_backend`, aligning with compact rules §2.2–2.3 and FAQ 15.2/24. |
| **COVERED** | Forced elimination & structural stalemate resolution (S‑invariant, no stacks left) | `tests/scenarios/ForcedEliminationAndStalemate.test.ts`                                                                    | Backend | Scenario‑style tests for forced elimination chains and the final stalemate ladder (converting rings in hand to eliminated rings).                                                                                                                                        |
| **PARTIAL** | Per‑player action availability checks (`hasValidActions`, `resolveBlockedState…`)  | `tests/unit/GameEngine.aiSimulation.test.ts`, `tests/unit/GameEngine.aiSimulation.*.debug.test.ts`                         | Backend | AI simulations hit edge‑case blocked states and call `resolveBlockedStateForCurrentPlayerForTesting`; used as diagnostics.                                                                                                                                               |
| **PARTIAL** | Sandbox turn structure ("place then move", mixed human/AI)                         | `tests/unit/ClientSandboxEngine.mixedPlayers.test.ts`, `tests/unit/ClientSandboxEngine.placementForcedElimination.test.ts` | Sandbox | Validates the unified place‑then‑move semantics and forced elimination in the client‑local engine.                                                                                                                                                                       |

**Planned additions**

- **PLANNED:** Add a small, rule‑tagged matrix in `GameEngine.turnSequence.scenarios.test.ts` so each scenario is keyed explicitly to §4.x / FAQ 15.2 / FAQ 24.

---

## 2. Non‑capture movement & marker interaction

**Rules/FAQ:**

- `ringrift_complete_rules.md` §8.2–8.3
- Compact rules §3.1–3.2
- FAQ 2–3 (basic movement & markers)

| Coverage    | Scenario / intent                                                                      | Jest file(s)                                                                                                  | Engines | Notes                                                                                                                                 |
| ----------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Minimum distance ≥ stack height; path blocking; landing restrictions (no enemy marker) | `tests/unit/RuleEngine.movementCapture.test.ts`                                                               | Backend | Core unit tests validate the movement geometry and blocking rules for non‑capture movement; used by both backend and sandbox engines. |
| **COVERED** | Marker creation, flipping opponent markers, collapsing own markers along paths         | `tests/unit/RuleEngine.movementCapture.test.ts`                                                               | Backend | Confirms departure marker placement, path‑marker flipping/collapse, and behaviour on landing into own marker (self‑elimination hook). |
| **COVERED** | Sandbox parity: movement, markers, mandatory move after placement                      | `tests/unit/ClientSandboxEngine.invariants.test.ts`, `tests/unit/ClientSandboxEngine.moveParity.test.ts`      | Sandbox | Sandbox has dedicated parity/invariant checks for movement + S‑invariant; enforced via client‑local engine.                           |
| **COVERED** | Rules-matrix movement scenarios for minimum distance and blocking (FAQ 2–3)            | `tests/unit/RuleEngine.movement.scenarios.test.ts`, `tests/scenarios/RulesMatrix.Movement.RuleEngine.test.ts` | Backend | Encodes minimum-distance and blocking cases for square8, square19, and hex; RulesMatrix suite provides a shared data-driven layer.    |

**Planned additions**

- **COVERED:** `tests/unit/RuleEngine.movement.scenarios.test.ts` – minimum-distance and blocking examples for FAQ 2–3 on `square8`, `square19`, and `hexagonal` boards.

---

## 3. Overtaking captures & chain patterns

**Rules/FAQ:**

- `ringrift_complete_rules.md` §9–10 (Overtaking Capture, Chain Overtaking)
- Compact rules §4.1–4.3
- FAQ 5–6, 9, 12, 14, 15.3.1–15.3.2 (180° reversal, cyclic captures)

| Coverage    | Scenario / intent                                                                                      | Jest file(s)                                                                                                           | Engines              | Notes                                                                                                                                                                                                                                                                                                  |
| ----------- | ------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **COVERED** | Basic overtaking capture validation (geometry, capHeight checks, own/other stacks, markers along path) | `tests/unit/RuleEngine.movementCapture.test.ts`                                                                        | Backend (RuleEngine) | Validates single‑segment captures against the compact spec (segment geometry and capHeight constraints).                                                                                                                                                                                               |
| **COVERED** | Chain capture enforcement, including 180° reversal and re‑capturing same target                        | `tests/unit/GameEngine.chainCapture.test.ts`                                                                           | Backend (GameEngine) | Exercises `chainCaptureState` and the engine‑driven continuation loop for several sequences, including reversals.                                                                                                                                                                                      |
| **COVERED** | PlayerChoice for capture direction (multi‑option chains), backend enumeration of follow‑ups            | `tests/unit/GameEngine.chainCaptureChoiceIntegration.test.ts`                                                          | Backend (GameEngine) | Uses `getCaptureOptionsFromPosition` + `CaptureDirectionChoice` to test multi‑branch chain captures.                                                                                                                                                                                                   |
| **COVERED** | WebSocket flow for capture_direction in multi‑branch scenarios                                         | `tests/unit/GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`                                             | Backend + WebSocket  | End‑to‑end test covering `CaptureDirectionChoice` over WebSockets and subsequent chain segments.                                                                                                                                                                                                       |
| **COVERED** | Sandbox parity for chain capture, including deterministic capture selection in AI                      | `tests/unit/ClientSandboxEngine.chainCapture.test.ts`, `tests/unit/ClientSandboxEngine.aiMovementCaptures.test.ts`     | Sandbox              | Validates sandbox chain logic and AI’s deterministic selection (lexicographically smallest landing).                                                                                                                                                                                                   |
| **COVERED** | Complex, rule‑/FAQ‑style chain capture examples (180° reverse, cycles) on square boards                | `tests/scenarios/ComplexChainCaptures.test.ts`                                                                         | Backend (scenario)   | Scenario‑style tests that encode key FAQ 15.3.\* examples directly, including `FAQ_15_3_2_CyclicPattern_TriangleLoop` and `FAQ_15_3_1_180_degree_reversal_basic`; reference for future Rust parity.                                                                                                    |
| **COVERED** | Hexagonal chain capture patterns (cross‑board comparison)                                              | `tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`, `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts` | Backend (hex)        | Backend tests encode the FAQ 15.3.x-style hex cyclic triangle via `FAQ_15_3_x_hex_cyclic_chain_capture_around_inner_triangle` and companion maximal-search assertions.                                                                                                                                 |
| **COVERED** | Sandbox 180° reversal parity scenario (mirror of FAQ 15.3.1 using ClientSandboxEngine)                 | 19×19                                                                                                                  | Sandbox              | `tests/unit/ClientSandboxEngine.chainCapture.scenarios.test.ts` – mirrors the backend `FAQ_15_3_1_180_degree_reversal_basic` scenario using the client‑local engine and asserts the same aggregate effects: Blue overtaker grows from height 4 → 6 while Red’s original stack at B shrinks from 3 → 1. |

**Planned additions**

- **NONE CURRENTLY:** FAQ 15.3.1/15.3.2 180° reversal and cyclic chain‑capture patterns are already encoded in `tests/scenarios/ComplexChainCaptures.test.ts` and `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts` with rule‑tagged scenario ids (for example `Rules_10_3_Q15_3_1_180_degree_reversal_basic` and `Rules_10_3_Q15_3_2_cyclic_pattern_triangle_loop`). New high‑value chain‑capture patterns should be added as additional RulesMatrix scenarios and rows in this section.

---

## 4. Line formation & graduated rewards

**Rules/FAQ:**

- `ringrift_complete_rules.md` §11 (Lines & Graduated Rewards)
- Compact rules §5.1–5.3
- FAQ 7, 22

| Coverage    | Scenario / intent                                                                                    | Jest file(s)                                                                                                                         | Engines              | Notes                                                                                                                 |
| ----------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | --------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Line detection & collapse for square and hex boards                                                  | `tests/unit/BoardManager.territoryDisconnection.test.ts` (indirect via line helpers), `tests/unit/ClientSandboxEngine.lines.test.ts` | Backend + Sandbox    | Sandbox suite validates line detection/processing in the client engine; BoardManager helpers are reused backend‑side. |
| **COVERED** | Graduated rewards: Option 1 vs Option 2 (collapse all + elimination vs min collapse, no elimination) | `tests/unit/GameEngine.lines.scenarios.test.ts`                                                                                      | Backend (GameEngine) | Tests specific cases where line length > required length and both options are available.                              |
| **COVERED** | AI‑driven choice for line rewards via Python service                                                 | `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`, `tests/unit/AIEngine.placementMetadata.test.ts`               | Backend + AI         | Confirms `LineRewardChoice` is forwarded to AI service with correct metadata, plus fallbacks.                         |
| **COVERED** | WebSocket‑driven line order and reward choices for human players                                     | `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`, `tests/unit/PlayerInteractionManager.test.ts`                  | Backend + WebSocket  | End‑to‑end coverage of line ordering + reward options over sockets.                                                   |

**Planned additions**

- **PLANNED:** Add a dedicated `LineReward` section in `GameEngine.lines.scenarios.test.ts` where each test name includes the corresponding rule section / FAQ (e.g. `Rules_11_3_OverlengthLine_ChooseOption2`).

---

## 5. Territory disconnection & chain reactions

**Rules/FAQ:**

- `ringrift_complete_rules.md` §12 (Area Disconnection & Collapse)
- Compact rules §6.1–6.4
- FAQ 10, 15, 20, 23

| Coverage    | Scenario / intent                                                                                      | Jest file(s)                                                                                                                         | Engines                | Notes                                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | ---------------------- | -------------------------------------------------------------------------------------------------------- |
| **COVERED** | Territory region discovery and isDisconnected flags (square8/square19/hex)                             | `tests/unit/BoardManager.territoryDisconnection.test.ts`, `tests/unit/BoardManager.territoryDisconnection.hex.test.ts`               | Backend (BoardManager) | Validates adjacency and region discovery semantics, including border/representation criteria.            |
| **COVERED** | Engine‑level processing of disconnected regions (collapse, elimination, self‑elimination prerequisite) | `tests/unit/GameEngine.territory.scenarios.test.ts`, `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`                      | Backend (GameEngine)   | Ensures `canProcessDisconnectedRegion` and `processOneDisconnectedRegion` follow compact rules §6.3–6.4. |
| **COVERED** | Client sandbox parity for territory disconnection (square + hex)                                       | `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts`, `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts` | Sandbox                | Confirms sandbox uses same semantics; important for visual debugging via `/sandbox`.                     |
| **COVERED** | Region order PlayerChoice (choosing which disconnected region to process first)                        | `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`, `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`             | Backend + Sandbox      | Both backend and sandbox interaction paths for `RegionOrderChoice` are tested.                           |
| **COVERED** | Multi‑step territory chain reactions and self‑elimination prerequisite in composed scenarios           | `tests/unit/GameEngine.territory.scenarios.test.ts`, `tests/scenarios/LineAndTerritory.test.ts`                                      | Backend (scenario)     | Encodes combined line+territory steps and verifies that self‑elimination constraints are enforced.       |

**Planned additions**

- **PLANNED:** Additional explicit FAQ‑tagged examples in `territory.scenarios.test.ts` for Q15 and Q20 with comments referencing the diagrams/positions from the rules doc. FAQ Q23's self-elimination prerequisite is covered by the paired backend tests `Q23_disconnected_region_illegal_when_no_self_elimination_available_backend` and `Q23_disconnected_region_processed_when_self_elimination_available_backend` in `tests/unit/GameEngine.territory.scenarios.test.ts`, and by the backend↔sandbox parity suite `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts` (shared initial state, positive/negative/multi-region Q23 territory parity).
- **COVERED:** Compact Q23 mini-region numeric invariants on square8 (`Rules_12_2_Q23_mini_region_square8_numeric_invariant`), exercising precise elimination/territory/S-invariant deltas at the rules layer: `tests/unit/territoryProcessing.rules.test.ts` (backend territoryProcessing helper), `tests/unit/sandboxTerritory.rules.test.ts` (sandbox territory rules), and `tests/unit/sandboxTerritoryEngine.rules.test.ts` (sandbox territory engine self-elimination prerequisite).

---

## 6. Victory conditions & stalemate

**Rules/FAQ:**

- `ringrift_complete_rules.md` §13 (Victory Conditions), §7.4 (Stalemate Resolution)
- Compact rules §7.1–7.4, §9 (progress invariant)
- FAQ 11, 18, 21, 24

| Coverage    | Scenario / intent                                                                  | Jest file(s)                                                                                                                                                                                        | Engines           | Notes                                                                                                                                                                                                                                                                                                                                      |
| ----------- | ---------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **COVERED** | Sandbox ring‑elimination and territory‑majority victories                          | `tests/unit/ClientSandboxEngine.victory.test.ts`                                                                                                                                                    | Sandbox           | Confirms that the local engine detects ring/territory victories per compact rules.                                                                                                                                                                                                                                                         |
| **COVERED** | Backend victory reasons and final scores (winner, `gameResult.reason`, ratings)    | `tests/integration/FullGameFlow.test.ts`, `tests/unit/GameEngine.aiSimulation.test.ts`, `tests/unit/GameEngine.victory.scenarios.test.ts`, `tests/unit/VictoryParity.RuleEngine_vs_Sandbox.test.ts` | Backend           | AI‑vs‑AI flow ends with a terminal state; `GameEngine.victory.scenarios` and `VictoryParity.RuleEngine_vs_Sandbox` now provide direct, rule‑tagged checks for victory reasons and winner mapping (per §13 / FAQ 11, 18, 21, 24). The `FullGameFlow` integration suite is treated as a hard CI gate for end‑to‑end flow and rating updates. |
| **COVERED** | Stalemate ladder priorities (territory > eliminated rings > markers > last action) | `tests/scenarios/ForcedEliminationAndStalemate.test.ts`, `tests/unit/GameEngine.victory.scenarios.test.ts`, `tests/unit/ClientSandboxEngine.victory.test.ts`                                        | Backend + Sandbox | Scenario suite covers forced elimination and terminal states; explicit tiebreak rungs (territory, eliminated rings, markers, last actor) are asserted in Rules*13_3–13_6*\* backend tests and mirrored in sandbox victory tests.                                                                                                           |

**Planned additions**

- `tests/unit/GameEngine.victory.scenarios.test.ts` and `tests/unit/ClientSandboxEngine.victory.test.ts` encode backend and sandbox ring‑elimination, territory‑control, and stalemate ladder scenarios (Rules*13_1–13_6*\*). Future additions here can focus on multi‑player and rating/score integrations.

---

## 7. PlayerChoice flows (engine, WebSocket, AI service, sandbox)

**Rules/FAQ:**

- `ringrift_complete_rules.md` §4.5, §10.3, §11–12 (places where choices are surfaced)
- PlayerChoice types: `LineOrderChoice`, `LineRewardChoice`, `RingEliminationChoice`, `RegionOrderChoice`, `CaptureDirectionChoice`
- FAQ 7 (line choice), 15 (region choice), 22–23 (line/territory details)

| Coverage    | Scenario / intent                                                                                    | Jest file(s)                                                                                                                                                  | Layer(s)                  | Notes                                                                                     |
| ----------- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | ----------------------------------------------------------------------------------------- |
| **COVERED** | Core PlayerInteractionManager wiring (request/response lifecycle)                                    | `tests/unit/PlayerInteractionManager.test.ts`                                                                                                                 | Backend interaction layer | Validates registration, choice routing, and error paths.                                  |
| **COVERED** | WebSocket interaction handler (mapping `player_choice_required`/`player_choice_response` to manager) | `tests/unit/WebSocketInteractionHandler.test.ts`, `tests/unit/WebSocketServer.aiTurn.integration.test.ts`                                                     | Backend + WebSocket       | Covers human and AI choice flows via sockets.                                             |
| **COVERED** | AIInteractionHandler & AIEngine service calls for line reward, ring elimination, region order        | `tests/unit/AIInteractionHandler.test.ts`, `tests/unit/AIEngine.serviceClient.test.ts`, `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts` | Backend + AI service      | Confirms service usage + fallbacks and ensures options metadata matches the compact spec. |
| **COVERED** | CaptureDirectionChoice for multi‑branch chains (backend + WebSocket)                                 | `tests/unit/GameEngine.captureDirectionChoice.test.ts`, `tests/unit/GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`                            | Backend + WebSocket       | Ties capture direction choices back into the chain capture loop.                          |
| **COVERED** | Sandbox choice flows for lines, region order, elimination                                            | `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`, `tests/unit/ClientSandboxEngine.lines.test.ts`, `tests/unit/ClientSandboxEngine.victory.test.ts`  | Sandbox                   | Exercises local AI/human choices in the client‑local engine.                              |

**Planned additions**

- **PLANNED:** Explicit rule/FAQ references in the choice‑centric tests (e.g. note in `regionOrderChoice` tests which FAQ disconnection example they encode).

---

## 8. Backend ↔ sandbox parity & progress invariant

**Rules/FAQ:**

- Compact rules §9 (S invariant), progress commentary in `ringrift_compact_rules.md` §9
- `RULES_ANALYSIS_PHASE2.md` §4 (Progress & Termination)

| Coverage    | Scenario / intent                                                                                                  | Jest file(s)                                                                                                                                                                                             | Engines           | Notes                                                                                                                                                                                                                                                                                                                               |
| ----------- | ------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Trace parity between sandbox AI games and backend replays (semantic comparison of moves & phases)                  | `tests/unit/Backend_vs_Sandbox.traceParity.test.ts`, `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts`                                                                                            | Backend + Sandbox | Uses `GameTrace` and `tests/utils/traces.ts` to compare step‑by‑step state; this suite is treated as a hard CI gate for backend↔sandbox parity on the curated seed‑5 trace.                                                                                                                                                        |
| **COVERED** | AI‑parallel debug runs (backend & sandbox) for seeded games, including mismatch logging and S‑snapshot comparisons | `tests/unit/Backend_vs_Sandbox.aiParallelDebug.test.ts`, `tests/utils/traces.ts`                                                                                                                         | Backend + Sandbox | Heavy diagnostic harness over seeded AI games; the curated runs in `Backend_vs_Sandbox.aiParallelDebug` and `Sandbox_vs_Backend.aiHeuristicCoverage` are treated as CI gates for backend↔sandbox + AI parity.                                                                                                                      |
| **COVERED** | Sandbox AI simulation S‑invariant and stall detection                                                              | `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`, `tests/unit/ClientSandboxEngine.aiStall.seed1.test.ts`, `tests/unit/ProgressSnapshot.core.test.ts`, `tests/unit/ProgressSnapshot.sandbox.test.ts` | Backend + Sandbox | aiSimulation/aiStall suites provide seeded, diagnostic coverage; the ProgressSnapshot core/sandbox tests add explicit, hand-built S-invariant checks (Rules*9*\*), asserting M/C/E counts and that canonical marker→territory+elimination transitions strictly increase S. These invariants act as gating tests for S‑monotonicity. |

**Planned additions**

- **PLANNED:** Add additional seeded parity suites (see `backend_vs_sandbox_trace_parity_extra_seeds` above) once new high‑value traces are identified; they should follow the same hard‑CI‑gate treatment and include a short, rule‑tagged comment block referencing compact rules §9.

---

## 9. How to extend this matrix

1. **When adding a new scenario test:**
   - Decide which rule/FAQ it encodes.
   - Add a row under the appropriate cluster table here.
   - Include:
     - Rule/FAQ references.
     - A brief description.
     - The Jest file path and, if useful, the `describe`/`it` name.
     - Engine(s) it exercises (Backend, Sandbox, WebSocket, AI service).
     - Coverage status: start as **PARTIAL**; upgrade to **COVERED** when the rule is clearly and directly asserted.

2. **When discovering a rules gap or bug:**
   - Add a **PLANNED** row for the missing scenario.
   - Link to any `KNOWN_ISSUES.md` entries.
   - Once fixed and tested, update status to **COVERED**.

3. **When modifying rules docs:**
   - If a rule section is changed or a FAQ is added/removed, scan this matrix for references to keep them in sync.

This file, together with `tests/README.md`, should be treated as the **single source of truth** for how RingRift’s formal rules map to concrete, executable tests.
