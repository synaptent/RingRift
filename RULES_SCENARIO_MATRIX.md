# RingRift Rules / FAQ → Test Scenario Matrix

> **Doc Status (2025-11-30): Active**
>
> - This file is the **canonical map** from RingRift’s rules/FAQ documents to concrete Jest test suites and parity harnesses.
> - It intentionally focuses on **which tests cover which rules/FAQ scenarios**; it does **not** redefine rules semantics or engine APIs.
> - For rules semantics SSoT, see:
>   - [`RULES_CANONICAL_SPEC.md`](RULES_CANONICAL_SPEC.md) (RR-CANON-RXXX invariants and formal rules).
>   - [`ringrift_complete_rules.md`](ringrift_complete_rules.md) and [`ringrift_compact_rules.md`](ringrift_compact_rules.md) for narrative and compact prose.
> - For Move/decision/WebSocket lifecycle SSoT, see:
>   - [`docs/CANONICAL_ENGINE_API.md` §3.9–3.10](docs/CANONICAL_ENGINE_API.md).
>   - `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, and `src/shared/types/websocket.ts`.
> - For TS↔Python rules parity SSoT, see:
>   - `src/shared/engine/contracts/*`, `tests/contracts/contractVectorRunner.test.ts`, and `ai-service/tests/contracts/test_contract_vectors.py`.
>
> **Purpose:** This file is the canonical map from the **rules documents** to the **Jest test suites**.

---

## 0.a Scenario Axis IDs (Movement / Chain / Lines / Territory / Victory)

This section introduces short, axis-oriented IDs that can be used in issues,
PRs, and inline test names. Each axis ID (M*/C*/L*/T*/V\*) points at a concrete
`rulesMatrix` scenario (where applicable) and at at least one backend and, when
relevant, sandbox test.

> Over time, these IDs should grow to cover all high-value rules/FAQ examples.
> When you add a new scenario row elsewhere in this file, consider giving it an
> axis ID here as well.

| Axis ID | Axis      | rulesMatrix ref.id                                                                                                                        | Rule / FAQ focus                                                                                 | Backend test(s)                                                                                                                                                   | Sandbox / parity test(s)                                                                                                                                                                   | Status  |
| ------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------- |
| **M1**  | Movement  | `Rules_8_2_Q2_minimum_distance_square8`                                                                                                   | §8.2, FAQ Q2 – minimum distance (square8)                                                        | `tests/unit/movement.shared.test.ts`; `tests/scenarios/RulesMatrix.Comprehensive.test.ts`                                                                         | `tests/unit/movement.shared.test.ts`                                                                                                                                                       | COVERED |
| **M2**  | Movement  | `Rules_8_2_Q2_markers_any_valid_space_beyond_square8`                                                                                     | §8.2, FAQ Q2–Q3 – landing beyond marker runs                                                     | `tests/scenarios/RulesMatrix.Comprehensive.test.ts`                                                                                                               | `tests/unit/movement.shared.test.ts`; `tests/unit/ClientSandboxEngine.invariants.test.ts`                                                                                                  | COVERED |
| **M3**  | Movement  | `Rules_9_10_overtaking_capture_vs_move_stack_parity`                                                                                      | §9–10 – overtaking capture vs simple move parity                                                 | `tests/scenarios/RulesMatrix.Comprehensive.test.ts`; `tests/unit/RuleEngine.movementCapture.test.ts`                                                              | `tests/unit/movement.shared.test.ts`                                                                                                                                                       | COVERED |
| **C1**  | Chain cap | `Rules_10_3_Q15_3_1_180_degree_reversal_basic`                                                                                            | §10.3, FAQ 15.3.1 – 180° reversal                                                                | `tests/scenarios/ComplexChainCaptures.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts`                                                     | `tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts`                                                                                                                     | COVERED |
| **C2**  | Chain cap | `Rules_10_3_Q15_3_2_cyclic_pattern_triangle_loop`                                                                                         | §10.3, FAQ 15.3.2 – cyclic triangle pattern                                                      | `tests/scenarios/ComplexChainCaptures.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts`                                                     | `tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts`                                                                                                                     | COVERED |
| **C3**  | Chain cap | –                                                                                                                                         | §10.3, FAQ 15.3.x – hex cyclic capture patterns                                                  | `tests/unit/GameEngine.cyclicCapture.hex.scenarios.test.ts`; `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts`                                            | –                                                                                                                                                                                          | COVERED |
| **L1**  | Lines     | `Rules_11_2_Q7_exact_length_line`                                                                                                         | §11.2, FAQ 7 – exact-length line                                                                 | `tests/unit/GameEngine.lines.scenarios.test.ts`; `tests/scenarios/RulesMatrix.GameEngine.test.ts`                                                                 | `tests/unit/ClientSandboxEngine.lines.test.ts`                                                                                                                                             | COVERED |
| **L2**  | Lines     | `Rules_11_3_Q22_overlength_line_option2_default`                                                                                          | §11.2–11.3, FAQ 22 – overlength, Option 2                                                        | `tests/unit/GameEngine.lines.scenarios.test.ts`; `tests/scenarios/RulesMatrix.GameEngine.test.ts`                                                                 | `tests/unit/ClientSandboxEngine.lines.test.ts`                                                                                                                                             | COVERED |
| **L3**  | Lines     | –                                                                                                                                         | §§11–12; FAQ 7, 15, 22–23 – combined line + territory turns                                      | `tests/scenarios/LineAndTerritory.test.ts`                                                                                                                        | –                                                                                                                                                                                          | COVERED |
| **L4**  | Lines     | –                                                                                                                                         | §11.2–11.3; FAQ 7, 22 – line_processing decision enumeration (process_line / choose_line_reward) | `tests/unit/GameEngine.lines.scenarios.test.ts` (line_processing_getValidMoves_exposes_process_line_and_choose_line_reward_moves)                                 | –                                                                                                                                                                                          | COVERED |
| **T1**  | Territory | –                                                                                                                                         | §12.2–12.3, FAQ 10, 15, 20, 23 – square boards                                                   | `tests/unit/GameEngine.territory.scenarios.test.ts`; `tests/scenarios/LineAndTerritory.test.ts`                                                                   | `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts`                                                                                                                            | COVERED |
| **T2**  | Territory | –                                                                                                                                         | §12.2–12.3, FAQ 10, 15, 20, 23 – hex boards                                                      | `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`                                                                                                        | `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`                                                                                                                        | COVERED |
| **T3**  | Territory | `Rules_12_2_Q23_region_not_processed_without_self_elimination_square19`, `Rules_12_2_Q23_region_processed_with_self_elimination_square19` | §12.2, FAQ 23 – self-elimination prerequisite for disconnected regions                           | `tests/unit/GameEngine.territory.scenarios.test.ts`; `tests/unit/territoryDecisionHelpers.shared.test.ts`                                                         | `tests/scenarios/RulesMatrix.Territory.ClientSandboxEngine.test.ts`; `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts` (diagnostic 19×19 parity harness)                          | COVERED |
| **T4**  | Territory | –                                                                                                                                         | §12.2–12.3; FAQ 23 – territory_processing decision enumeration (process_territory_region)        | `tests/unit/territoryDecisionHelpers.shared.test.ts`; `tests/unit/GameEngine.territory.scenarios.test.ts` (legacy getValidMoves surface, partly skipped)          | `tests/unit/ClientSandboxEngine.territoryDecisionPhases.MoveDriven.test.ts`                                                                                                                | COVERED |
| **V1**  | Victory   | –                                                                                                                                         | §13, FAQ 11, 18, 21, 24 – victory & stalemate                                                    | `tests/unit/GameEngine.victory.scenarios.test.ts`; `tests/unit/GameEngine.victory.LPS.scenarios.test.ts`; `tests/scenarios/ForcedEliminationAndStalemate.test.ts` | `tests/unit/ClientSandboxEngine.victory.LPS.crossInteraction.test.ts`; `tests/unit/LPS.CrossInteraction.Parity.test.ts`; `tests/scenarios/RulesMatrix.Victory.ClientSandboxEngine.test.ts` | COVERED |
| **V2**  | Victory   | –                                                                                                                                         | §4.4, §13.3–13.5, FAQ 11, 24 – forced elimination & stalemate ladder                             | `tests/scenarios/ForcedEliminationAndStalemate.test.ts`; `tests/unit/GameEngine.turnSequence.scenarios.test.ts`                                                   | –                                                                                                                                                                                          | COVERED |

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
- `archive/RULES_ANALYSIS_PHASE2.md`
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

| Rule / FAQ Ref                                     | Scenario Name                                           | Board Type(s)                                 | Engine(s)                  | Test File(s)                                                                                                                                                                                                                                                                                                                                                                                                                      | Status                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------------- | ------------------------------------------------------- | --------------------------------------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------- |
| §§9–10; FAQ 15.3.1–15.3.2 (180° + cyclic captures) | `complex_chain_captures_square_examples`                | 8×8, 19×19                                    | Backend                    | `tests/scenarios/ComplexChainCaptures.test.ts`                                                                                                                                                                                                                                                                                                                                                                                    | COVERED                                                                                                                                                                                                                                                                                                                            |
| §§9–10; FAQ 15.3.1                                 | `chain_capture_180_degree_reversal_examples`            | 8×8, 19×19                                    | Backend                    | `tests/scenarios/ComplexChainCaptures.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts`                                                                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |
| §§9–10; FAQ 15.3.1–15.3.2                          | `chain_capture_rules_matrix_examples`                   | 8×8, 19×19                                    | Backend, Sandbox           | `tests/scenarios/RulesMatrix.ChainCapture.GameEngine.test.ts`; `tests/scenarios/RulesMatrix.ChainCapture.ClientSandboxEngine.test.ts`                                                                                                                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |
| §§9–10; FAQ 15.3.1–15.3.2                          | `hex_cyclic_captures_height3`                           | hex                                           | Backend                    | `tests/unit/GameEngine.cyclicCapture.hex.height3.test.ts`                                                                                                                                                                                                                                                                                                                                                                         | COVERED                                                                                                                                                                                                                                                                                                                            |
| §§9–10; Overtaking vs Move                         | `overtaking_capture_vs_move_stack_parity`               | 8×8                                           | Backend, Sandbox           | `tests/unit/RuleEngine.movementCapture.test.ts`; `tests/unit/movement.shared.test.ts`; `tests/unit/captureLogic.shared.test.ts`; overtaking vs move-stack behaviour is also exercised inside the multi-domain parity harnesses (`archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts` (archived), elimination/territory traces, and contract-vector–driven shared-engine tests such as `tests/unit/TraceFixtures.sharedEngineParity.test.ts`). | COVERED                                                                                                                                                                                                                                                                                                                            |
| §11.2–11.3; FAQ 7, 22                              | `line_reward_overlength_square_examples`                | 8×8, 19×19                                    | Backend                    | `tests/unit/GameEngine.lines.scenarios.test.ts`                                                                                                                                                                                                                                                                                                                                                                                   | COVERED                                                                                                                                                                                                                                                                                                                            |
| §11.2–11.3; FAQ 7, 22                              | `line_reward_rules_matrix_backend_examples`             | 8×8                                           | Backend                    | `tests/scenarios/RulesMatrix.Comprehensive.test.ts`                                                                                                                                                                                                                                                                                                                                                                               | COVERED                                                                                                                                                                                                                                                                                                                            |
| §11.2–11.3; FAQ 22                                 | `line_reward_option1_full_collapse_square19`            | 19×19                                         | Backend                    | `tests/scenarios/RulesMatrix.Comprehensive.test.ts` (RulesMatrix lineRewardRuleScenario id `Rules_11_3_Q22_overlength_line_option1_full_collapse_square19`)                                                                                                                                                                                                                                                                       | COVERED                                                                                                                                                                                                                                                                                                                            |
| §11; FAQ 7                                         | `sandbox_line_processing_parity`                        | 8×8, 19×19, hex                               | Sandbox                    | `tests/unit/ClientSandboxEngine.lines.test.ts`                                                                                                                                                                                                                                                                                                                                                                                    | COVERED                                                                                                                                                                                                                                                                                                                            |
| §12.2–12.3; FAQ 10, 15, 20, 23                     | `territory_disconnection_square_examples`               | 8×8, 19×19                                    | Backend, Sandbox           | `tests/unit/GameEngine.territory.scenarios.test.ts`; `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts` (uses `territoryRuleScenarios` including `Rules_12_2_Q23_region_not_processed_without_self_elimination_square19` and `Rules_12_2_Q23_region_processed_with_self_elimination_square19`)                                                                                                                       | COVERED                                                                                                                                                                                                                                                                                                                            |
| §12.2–12.3; FAQ 10, 15, 20, 23                     | `territory_disconnection_hex_examples`                  | hex                                           | Backend, Sandbox           | `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`; `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts`                                                                                                                                                                                                                                                                                                   | COVERED                                                                                                                                                                                                                                                                                                                            |
| §§11–12; FAQ 7, 15, 22–23                          | `combined_line_and_territory_turns`                     | 8×8, 19×19, hex                               | Backend                    | `tests/scenarios/LineAndTerritory.test.ts` (driven by `lineAndTerritoryRuleScenarios`, e.g. `Rules_11_2_12_2_Q7_Q20_overlength_line_then_single_cell_region_{square8,square19,hexagonal}`)                                                                                                                                                                                                                                        | COVERED                                                                                                                                                                                                                                                                                                                            |
| §§11–12; FAQ 7, 15, 22–23                          | `combined_line_and_territory_square19_hex`              | 19×19, hex                                    | Backend                    | `tests/scenarios/LineAndTerritory.test.ts`                                                                                                                                                                                                                                                                                                                                                                                        | COVERED                                                                                                                                                                                                                                                                                                                            |
| §4.x; §§8–13; FAQ 15.2, 24                         | `forced_elimination_and_stalemate_ladder`               | 8×8                                           | Backend                    | `tests/scenarios/ForcedEliminationAndStalemate.test.ts` (uses `victoryRuleScenarios` ids `Rules_4_4_13_5_Q24_forced_elimination_single_blocked_stack` and `Rules_13_4_13_5_Q11_structural_stalemate_rings_in_hand_become_eliminated`)                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |
| §4.x; FAQ 15.2, 24                                 | `turn_sequence_forced_elimination_matrix`               | 8×8, 19×19, hex (selected)                    | Backend                    | `tests/unit/GameEngine.turnSequence.scenarios.test.ts`                                                                                                                                                                                                                                                                                                                                                                            | COVERED                                                                                                                                                                                                                                                                                                                            |
| §4.x; FAQ 15.2                                     | `sandbox_mixed_players_place_then_move`                 | 8×8                                           | Sandbox                    | `tests/unit/SandboxAI.ringPlacementNoopRegression.test.ts`, `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`                                                                                                                                                                                                                                                                                                                 | PARTIAL                                                                                                                                                                                                                                                                                                                            |
| §13; FAQ 11, 18, 21, 24                            | `stalemate_tiebreak_ladder_examples`                    | 8×8, 19×19                                    | Backend, Sandbox           | `tests/unit/GameEngine.victory.scenarios.test.ts` (Rules*13_1–13_6*\*), `tests/unit/GameEngine.victory.LPS.scenarios.test.ts`, `tests/scenarios/ForcedEliminationAndStalemate.test.ts`, `tests/unit/ClientSandboxEngine.victory.LPS.crossInteraction.test.ts`, `tests/unit/LPS.CrossInteraction.Parity.test.ts`, `tests/scenarios/RulesMatrix.Victory.ClientSandboxEngine.test.ts`                                                | COVERED                                                                                                                                                                                                                                                                                                                            |
| §13.1–13.2; FAQ 18, 21                             | `victory_threshold_rules_matrix_backend_examples`       | 8×8                                           | Backend, Sandbox           | `tests/scenarios/RulesMatrix.Victory.GameEngine.test.ts`; `tests/scenarios/RulesMatrix.Victory.ClientSandboxEngine.test.ts` (driven by `victoryRuleScenarios` ids `Rules_13_1_ring_elimination_threshold_square8` and `Rules_13_2_territory_control_threshold_square8`)                                                                                                                                                           | COVERED                                                                                                                                                                                                                                                                                                                            |
|                                                    | Compact rules §9; `archive/RULES_ANALYSIS_PHASE2.md` §4 | `backend_vs_sandbox_trace_parity_seed5`       | 8×8 (seeded AI)            | Backend, Sandbox, Traces                                                                                                                                                                                                                                                                                                                                                                                                          | `archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts` (archived); `tests/unit/Backend_vs_Sandbox.seed5.checkpoints.test.ts`; `tests/unit/Backend_vs_Sandbox.seed5.internalStateParity.test.ts`; `tests/unit/Backend_vs_Sandbox.seed5.bisectParity.test.ts`                                                                                      | PARTIAL |
|                                                    | Compact rules §9; `archive/RULES_ANALYSIS_PHASE2.md` §4 | `backend_vs_sandbox_trace_parity_extra_seeds` | 8×8, 19×19, hex (selected) | Backend, Sandbox, Traces                                                                                                                                                                                                                                                                                                                                                                                                          | `tests/unit/TraceParity.seed14.firstDivergence.test.ts`; `tests/unit/Seed14Move35LineParity.test.ts`; `tests/unit/Seed17Move16And33Parity.GameEngine_vs_Sandbox.test.ts`; `tests/unit/TraceParity.seed17.firstDivergence.test.ts`; `tests/unit/Seed17GeometryParity.GameEngine_vs_Sandbox.test.ts`                                 | PARTIAL |
|                                                    | Compact rules §9; `archive/RULES_ANALYSIS_PHASE2.md` §4 | `sandbox_ai_simulation_progress_invariant`    | 8×8, 19×19, hex (various)  | Backend, Sandbox                                                                                                                                                                                                                                                                                                                                                                                                                  | `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`; `tests/unit/ClientSandboxEngine.aiStallRegression.test.ts`; `tests/unit/ClientSandboxEngine.aiSingleSeedDebug.test.ts`; `tests/scenarios/AI_TerminationFromSeed1Plateau.test.ts`; `tests/unit/ProgressSnapshot.core.test.ts`; `tests/unit/ProgressSnapshot.sandbox.test.ts` | COVERED |
| §11.2–11.3; FAQ 7, 22                              | `line_reward_choice_ai_service_backing`                 | 8×8                                           | Backend, AI Service        | `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`                                                                                                                                                                                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |
| §12.3; FAQ 15, 23                                  | `region_order_choice_ai_and_sandbox`                    | 8×8                                           | Backend, Sandbox, AI       | `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`; `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`                                                                                                                                                                                                                                                                                                          | COVERED                                                                                                                                                                                                                                                                                                                            |
| FAQ Q1–Q6 (Basic Mechanics)                        | `faq_basic_mechanics_comprehensive_suite`               | 8×8, 19×19                                    | Backend, Sandbox           | `tests/scenarios/FAQ_Q01_Q06.test.ts`                                                                                                                                                                                                                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |
| FAQ Q7–Q8 (Line Formation)                         | `faq_line_formation_and_collapse`                       | 8×8, 19×19                                    | Backend                    | `tests/scenarios/FAQ_Q07_Q08.test.ts`                                                                                                                                                                                                                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |
| FAQ Q9–Q14 (Edge Cases & Mechanics)                | `faq_edge_cases_special_mechanics`                      | 8×8, 19×19, hex                               | Backend                    | `tests/scenarios/FAQ_Q09_Q14.test.ts`                                                                                                                                                                                                                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |
| FAQ Q15 (Chain Capture Patterns)                   | `faq_chain_capture_comprehensive_suite`                 | 8×8, 19×19                                    | Backend, Sandbox           | `tests/scenarios/FAQ_Q15.test.ts`                                                                                                                                                                                                                                                                                                                                                                                                 | COVERED                                                                                                                                                                                                                                                                                                                            |
| FAQ Q16–Q18 (Victory & Control)                    | `faq_victory_conditions_control_transfer`               | 8×8, 19×19, hex                               | Backend                    | `tests/scenarios/FAQ_Q16_Q18.test.ts`                                                                                                                                                                                                                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |
| FAQ Q19–Q21, Q24 (Player Counts & Thresholds)      | `faq_player_counts_thresholds_forced_elim`              | 8×8, 19×19, hex                               | Backend                    | `tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`                                                                                                                                                                                                                                                                                                                                                                                         | COVERED                                                                                                                                                                                                                                                                                                                            |
| FAQ Q22–Q23 (Graduated Rewards & Prerequisites)    | `faq_graduated_rewards_territory_prereqs`               | 8×8, 19×19, hex                               | Backend                    | `tests/scenarios/FAQ_Q22_Q23.test.ts`                                                                                                                                                                                                                                                                                                                                                                                             | COVERED                                                                                                                                                                                                                                                                                                                            |

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

| Coverage    | Scenario / intent                                                                  | Jest file(s)                                                                                                      | Engines | Notes                                                                                                                                                                                                                                                                    |
| ----------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **COVERED** | Full turn sequence including forced elimination and skipping dead players          | `tests/unit/GameEngine.turnSequence.scenarios.test.ts`                                                            | Backend | Encodes multi‑player cases where some players are blocked with stacks and others are out of material; includes the rule‑tagged three‑player scenario `Rules_4_2_three_player_skip_and_forced_elimination_backend`, aligning with compact rules §2.2–2.3 and FAQ 15.2/24. |
| **COVERED** | Forced elimination & structural stalemate resolution (S‑invariant, no stacks left) | `tests/scenarios/ForcedEliminationAndStalemate.test.ts`                                                           | Backend | Scenario‑style tests for forced elimination chains and the final stalemate ladder (converting rings in hand to eliminated rings).                                                                                                                                        |
| **PARTIAL** | Per‑player action availability checks (`hasValidActions`, `resolveBlockedState…`)  | `tests/unit/GameEngine.aiSimulation.test.ts`, `tests/unit/GameEngine.aiSimulation.*.debug.test.ts`                | Backend | AI simulations hit edge‑case blocked states and call `resolveBlockedStateForCurrentPlayerForTesting`; used as diagnostics.                                                                                                                                               |
| **PARTIAL** | Sandbox turn structure (ring_placement + forced elimination invariants)            | `tests/unit/SandboxAI.ringPlacementNoopRegression.test.ts`, `tests/unit/ClientSandboxEngine.aiSimulation.test.ts` | Sandbox | Validates sandbox ring_placement and forced-elimination behaviour under AI-driven flows; legacy mixedPlayers/placementForcedElimination tests have been removed.                                                                                                         |

**Planned additions**

- **PLANNED:** Add a small, rule‑tagged matrix in `GameEngine.turnSequence.scenarios.test.ts` so each scenario is keyed explicitly to §4.x / FAQ 15.2 / FAQ 24.

---

## 2. Non‑capture movement & marker interaction

**Rules/FAQ:**

- `ringrift_complete_rules.md` §8.2–8.3
- Compact rules §3.1–3.2
- FAQ 2–3 (basic movement & markers)

| Coverage    | Scenario / intent                                                                      | Jest file(s)                                                                                             | Engines | Notes                                                                                                                                 |
| ----------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Minimum distance ≥ stack height; path blocking; landing restrictions (no enemy marker) | `tests/unit/RuleEngine.movementCapture.test.ts`                                                          | Backend | Core unit tests validate the movement geometry and blocking rules for non‑capture movement; used by both backend and sandbox engines. |
| **COVERED** | Marker creation, flipping opponent markers, collapsing own markers along paths         | `tests/unit/RuleEngine.movementCapture.test.ts`                                                          | Backend | Confirms departure marker placement, path‑marker flipping/collapse, and behaviour on landing into own marker (self‑elimination hook). |
| **COVERED** | Sandbox parity: movement, markers, mandatory move after placement                      | `tests/unit/ClientSandboxEngine.invariants.test.ts`, `tests/unit/ClientSandboxEngine.moveParity.test.ts` | Sandbox | Sandbox has dedicated parity/invariant checks for movement + S‑invariant; enforced via client‑local engine.                           |
| **COVERED** | Rules-matrix movement scenarios for minimum distance and blocking (FAQ 2–3)            | `tests/unit/movement.shared.test.ts`, `tests/scenarios/RulesMatrix.Comprehensive.test.ts`                | Backend | Encodes minimum-distance and blocking cases for square8, square19, and hex; RulesMatrix suite provides a shared data-driven layer.    |

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

| Coverage    | Scenario / intent                                                                                    | Jest file(s)                                                                                                                              | Engines              | Notes                                                                                                                              |
| ----------- | ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Line detection & collapse for square and hex boards                                                  | `tests/unit/lineDetection.shared.test.ts`, `tests/unit/LineDetectionParity.rules.test.ts`, `tests/unit/ClientSandboxEngine.lines.test.ts` | Backend + Sandbox    | Shared line-detection helpers are exercised in unit tests; sandbox suite validates line detection/processing in the client engine. |
| **COVERED** | Graduated rewards: Option 1 vs Option 2 (collapse all + elimination vs min collapse, no elimination) | `tests/unit/GameEngine.lines.scenarios.test.ts`                                                                                           | Backend (GameEngine) | Tests specific cases where line length > required length and both options are available.                                           |
| **COVERED** | AI‑driven choice for line rewards via Python service                                                 | `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`, `tests/unit/AIEngine.placementMetadata.test.ts`                    | Backend + AI         | Confirms `LineRewardChoice` is forwarded to AI service with correct metadata, plus fallbacks.                                      |
| **COVERED** | WebSocket‑driven line order and reward choices for human players                                     | `tests/unit/GameEngine.lineRewardChoiceWebSocketIntegration.test.ts`, `tests/unit/PlayerInteractionManager.test.ts`                       | Backend + WebSocket  | End‑to‑end coverage of line ordering + reward options over sockets.                                                                |

**Planned additions**

- **PLANNED:** Add a dedicated `LineReward` section in `GameEngine.lines.scenarios.test.ts` where each test name includes the corresponding rule section / FAQ (e.g. `Rules_11_3_OverlengthLine_ChooseOption2`).

---

## 5. Territory disconnection & chain reactions

**Rules/FAQ:**

- `ringrift_complete_rules.md` §12 (Area Disconnection & Collapse)
- Compact rules §6.1–6.4
- FAQ 10, 15, 20, 23

| Coverage    | Scenario / intent                                                                                      | Jest file(s)                                                                                                                         | Engines              | Notes                                                                                                    |
| ----------- | ------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ | -------------------- | -------------------------------------------------------------------------------------------------------- |
| **COVERED** | Territory region discovery and isDisconnected flags (square8/square19/hex)                             | `tests/unit/GameEngine.territoryDisconnection.test.ts`, `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`                   | Backend (GameEngine) | Validates adjacency and region discovery semantics, including border/representation criteria.            |
| **COVERED** | Engine‑level processing of disconnected regions (collapse, elimination, self‑elimination prerequisite) | `tests/unit/GameEngine.territory.scenarios.test.ts`, `tests/unit/GameEngine.territoryDisconnection.hex.test.ts`                      | Backend (GameEngine) | Ensures `canProcessDisconnectedRegion` and `processOneDisconnectedRegion` follow compact rules §6.3–6.4. |
| **COVERED** | Client sandbox parity for territory disconnection (square + hex)                                       | `tests/unit/ClientSandboxEngine.territoryDisconnection.test.ts`, `tests/unit/ClientSandboxEngine.territoryDisconnection.hex.test.ts` | Sandbox              | Confirms sandbox uses same semantics; important for visual debugging via `/sandbox`.                     |
| **COVERED** | Region order PlayerChoice (choosing which disconnected region to process first)                        | `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`, `tests/unit/GameEngine.regionOrderChoiceIntegration.test.ts`             | Backend + Sandbox    | Both backend and sandbox interaction paths for `RegionOrderChoice` are tested.                           |
| **COVERED** | Multi‑step territory chain reactions and self‑elimination prerequisite in composed scenarios           | `tests/unit/GameEngine.territory.scenarios.test.ts`, `tests/scenarios/LineAndTerritory.test.ts`                                      | Backend (scenario)   | Encodes combined line+territory steps and verifies that self‑elimination constraints are enforced.       |

**Planned additions**

- **PLANNED:** Additional explicit FAQ‑tagged examples in `territory.scenarios.test.ts` for Q15 and Q20 with comments referencing the diagrams/positions from the rules doc. FAQ Q23's self-elimination prerequisite is covered by the paired backend tests `Q23_disconnected_region_illegal_when_no_self_elimination_available_backend` and `Q23_disconnected_region_processed_when_self_elimination_available_backend` in `tests/unit/GameEngine.territory.scenarios.test.ts`, and by the backend↔sandbox parity suite `tests/unit/TerritoryParity.GameEngine_vs_Sandbox.test.ts` (shared initial state, positive/negative/multi-region Q23 territory parity).
- **COVERED:** Compact Q23 mini-region numeric invariants on square8 (`Rules_12_2_Q23_mini_region_square8_numeric_invariant`), exercising precise elimination/territory/S-invariant deltas at the rules layer: legacy/deprecated `tests/unit/territoryProcessing.rules.test.ts` (backend territoryProcessing helper kept for reference) and legacy‑named `tests/unit/sandboxTerritoryEngine.rules.test.ts` (diagnostic harness that now exercises `ClientSandboxEngine.processDisconnectedRegionsForCurrentPlayer` instead of the removed `sandboxTerritoryEngine.ts`).

---

## 6. Victory conditions & stalemate

**Rules/FAQ:**

- `ringrift_complete_rules.md` §13 (Victory Conditions), §7.4 (Stalemate Resolution)
- Compact rules §7.1–7.4, §9 (progress invariant)
- FAQ 11, 18, 21, 24

| Coverage    | Scenario / intent                                                                  | Jest file(s)                                                                                                                                                                                                                                       | Engines           | Notes                                                                                                                                                                                                                                                                                                                                                               |
| ----------- | ---------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Sandbox ring‑elimination and territory‑majority victories                          | `tests/unit/GameEngine.victory.LPS.scenarios.test.ts`; `tests/unit/ClientSandboxEngine.victory.LPS.crossInteraction.test.ts`; `tests/unit/LPS.CrossInteraction.Parity.test.ts`; `tests/scenarios/RulesMatrix.Victory.ClientSandboxEngine.test.ts`  | Backend + Sandbox | Confirms that backend and sandbox engines detect ring/territory victories per compact rules and that LPS/victory interactions remain in parity.                                                                                                                                                                                                                     |
| **COVERED** | Backend victory reasons and final scores (winner, `gameResult.reason`, ratings)    | `tests/integration/FullGameFlow.test.ts`, `tests/unit/GameEngine.aiSimulation.test.ts`, `tests/unit/GameEngine.victory.scenarios.test.ts`, `tests/unit/GameEngine.victory.LPS.scenarios.test.ts`, `tests/unit/LPS.CrossInteraction.Parity.test.ts` | Backend           | AI‑vs‑AI flow ends with a terminal state; `GameEngine.victory.scenarios` and `GameEngine.victory.LPS.scenarios` now provide direct, rule‑tagged checks for victory reasons, winner mapping, and LPS/victory cross‑interaction (per §13 / FAQ 11, 18, 21, 24). The `FullGameFlow` integration suite remains the hard CI gate for end‑to‑end flow and rating updates. |
| **COVERED** | Stalemate ladder priorities (territory > eliminated rings > markers > last action) | `tests/scenarios/ForcedEliminationAndStalemate.test.ts`, `tests/unit/GameEngine.victory.scenarios.test.ts`, `tests/unit/GameEngine.victory.LPS.scenarios.test.ts`                                                                                  | Backend + Sandbox | Scenario suite covers forced elimination and terminal states; explicit tiebreak rungs (territory, eliminated rings, markers, last actor) are asserted in Rules*13_3–13_6*\* backend tests and kept in parity with sandbox/AI flows via the LPS/victory cross-interaction suites.                                                                                    |

**Planned additions**

- `tests/unit/GameEngine.victory.scenarios.test.ts`, `tests/unit/GameEngine.victory.LPS.scenarios.test.ts`, and `tests/unit/ClientSandboxEngine.victory.LPS.crossInteraction.test.ts` encode backend and sandbox ring‑elimination, territory‑control, LPS plateau, and stalemate ladder scenarios (Rules*13_1–13_6*\*). Future additions here can focus on multi‑player and rating/score integrations.

---

## 7. PlayerChoice flows (engine, WebSocket, AI service, sandbox)

**Rules/FAQ:**

- `ringrift_complete_rules.md` §4.5, §10.3, §11–12 (places where choices are surfaced)
- PlayerChoice types: `LineOrderChoice`, `LineRewardChoice`, `RingEliminationChoice`, `RegionOrderChoice`, `CaptureDirectionChoice`
- FAQ 7 (line choice), 15 (region choice), 22–23 (line/territory details)

| Coverage    | Scenario / intent                                                                                    | Jest file(s)                                                                                                                                                                                                                                         | Layer(s)                  | Notes                                                                                                                                   |
| ----------- | ---------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------- | --------------------------------------------------------------------------------------------------------------------------------------- |
| **COVERED** | Core PlayerInteractionManager wiring (request/response lifecycle)                                    | `tests/unit/PlayerInteractionManager.test.ts`                                                                                                                                                                                                        | Backend interaction layer | Validates registration, choice routing, and error paths.                                                                                |
| **COVERED** | WebSocket interaction handler (mapping `player_choice_required`/`player_choice_response` to manager) | `tests/unit/WebSocketInteractionHandler.test.ts`, `tests/unit/WebSocketServer.aiTurn.integration.test.ts`                                                                                                                                            | Backend + WebSocket       | Covers human and AI choice flows via sockets.                                                                                           |
| **COVERED** | AIInteractionHandler & AIEngine service calls for line reward, ring elimination, region order        | `tests/unit/AIInteractionHandler.test.ts`, `tests/unit/AIEngine.serviceClient.test.ts`, `tests/unit/GameEngine.lineRewardChoiceAIService.integration.test.ts`                                                                                        | Backend + AI service      | Confirms service usage + fallbacks and ensures options metadata matches the compact spec.                                               |
| **COVERED** | CaptureDirectionChoice for multi‑branch chains (backend + WebSocket)                                 | `tests/unit/GameEngine.captureDirectionChoice.test.ts`, `tests/unit/GameEngine.captureDirectionChoiceWebSocketIntegration.test.ts`                                                                                                                   | Backend + WebSocket       | Ties capture direction choices back into the chain capture loop.                                                                        |
| **COVERED** | Sandbox choice flows for lines, region order, elimination                                            | `tests/unit/ClientSandboxEngine.regionOrderChoice.test.ts`, `tests/unit/ClientSandboxEngine.lines.test.ts`, `tests/scenarios/RulesMatrix.Elimination.ClientSandboxEngine.test.ts`, `tests/scenarios/RulesMatrix.Victory.ClientSandboxEngine.test.ts` | Sandbox                   | Exercises local AI/human choices in the client‑local engine via RulesMatrix-backed line, territory, elimination, and victory scenarios. |

**Planned additions**

- **PLANNED:** Explicit rule/FAQ references in the choice‑centric tests (e.g. note in `regionOrderChoice` tests which FAQ disconnection example they encode).

---

## 8. Backend ↔ sandbox parity & progress invariant

**Rules/FAQ:**

- Compact rules §9 (S invariant), progress commentary in `ringrift_compact_rules.md` §9
- `archive/RULES_ANALYSIS_PHASE2.md` §4 (Progress & Termination)

| Coverage | Scenario / intent | Jest file(s) | Engines | Notes |
| -------- | ----------------- | ------------ | ------- | ----- |

---

## 9. FAQ Scenario Test Matrix (Q1-Q24)

This section provides a complete mapping of all FAQ questions to their dedicated test files. Each FAQ test file is designed to be run independently for targeted validation.

**Quick Reference - Run FAQ Tests:**

```bash
# Run all FAQ tests
npm test -- FAQ_

# Run specific FAQ question
npm test -- FAQ_Q15        # Chain captures
npm test -- FAQ_Q07_Q08    # Line formation
npm test -- FAQ_Q22_Q23    # Graduated rewards & territory prerequisites
```

### FAQ Test Coverage Matrix

| FAQ | Topic                           | Test File                                                                              | Board Types            | Status     | Notes                                                                                                                           |
| --- | ------------------------------- | -------------------------------------------------------------------------------------- | ---------------------- | ---------- | ------------------------------------------------------------------------------------------------------------------------------- |
| Q1  | Stack splitting/rearranging     | [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1)         | square8, square19      | ✅ COVERED | Validates immutable stack order, rings always added to bottom                                                                   |
| Q2  | Minimum jump requirement        | [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1)         | square8, square19, hex | ✅ COVERED | Tests height=distance rule, marker counting                                                                                     |
| Q3  | Capture landing distance        | [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1)         | square19               | ✅ COVERED | Validates flexible landing beyond captured piece                                                                                |
| Q4  | Rings under captured top        | [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1)         | square8                | ✅ COVERED | Only top ring captured, rest remain in place                                                                                    |
| Q5  | Multiple ring capture           | [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1)         | square8                | ✅ COVERED | Single jump = one ring; multiple via chain                                                                                      |
| Q6  | Overtaking vs Elimination       | [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1)         | square8                | ✅ COVERED | Overtaking keeps rings in play; elimination counts toward victory                                                               |
| Q7  | Multiple lines of markers       | [`tests/scenarios/FAQ_Q07_Q08.test.ts`](tests/scenarios/FAQ_Q07_Q08.test.ts:1)         | square8, square19      | ✅ COVERED | Exact-length, overlength, intersecting lines                                                                                    |
| Q8  | No rings to remove              | [`tests/scenarios/FAQ_Q07_Q08.test.ts`](tests/scenarios/FAQ_Q07_Q08.test.ts:1)         | square8                | ✅ COVERED | Turn ends if no rings available for exact-length; Option 2 for overlength                                                       |
| Q9  | Chain blocking all moves        | [`tests/scenarios/FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)         | square8                | ✅ COVERED | Mandatory chain even if self-destructive                                                                                        |
| Q10 | Multicolored stacks in regions  | [`tests/scenarios/FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)         | square8                | ✅ COVERED | Only current control matters; buried rings don't count                                                                          |
| Q11 | Stalemate with rings in hand    | [`tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`](tests/scenarios/FAQ_Q19_Q21_Q24.test.ts:1) | square8                | ✅ COVERED | Rings in hand → eliminated; tiebreaker applied                                                                                  |
| Q12 | Chain eliminating all rings     | [`tests/scenarios/FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)         | square8                | ✅ COVERED | Chain continues even to self-elimination                                                                                        |
| Q13 | Moore vs Von Neumann            | [`tests/scenarios/FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)         | square8, hex           | ✅ COVERED | Movement/lines use Moore; territory uses Von Neumann (square) or hex (hex)                                                      |
| Q14 | Capture optional vs mandatory   | [`tests/scenarios/FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)         | square8                | ✅ COVERED | Initial capture optional; chain mandatory once started                                                                          |
| Q15 | Surrounded/disconnected regions | See territory tests below                                                              | square8, square19, hex | ✅ COVERED | Covered by existing territory tests + [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:1) for chain patterns |
| Q16 | Control transfer multicolored   | [`tests/scenarios/FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1)         | square8                | ✅ COVERED | Top ring determines control; can recover buried rings                                                                           |
| Q17 | First ring placement rules      | [`tests/scenarios/FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1)         | square8                | ✅ COVERED | No special rule; standard movement applies                                                                                      |
| Q18 | Multiple victory conditions     | [`tests/scenarios/FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1)         | square8                | ✅ COVERED | Ring elimination precedence; >50% prevents simultaneous wins                                                                    |
| Q19 | 2 or 4 player games             | [`tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`](tests/scenarios/FAQ_Q19_Q21_Q24.test.ts:1) | square8, square19, hex | ✅ COVERED | All player counts 2-4; threshold validation                                                                                     |
| Q20 | Territory rules 8×8 vs 19×19    | [`tests/scenarios/FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1)         | square8, square19      | ✅ COVERED | Both use Von Neumann for territory                                                                                              |
| Q21 | Victory thresholds              | [`tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`](tests/scenarios/FAQ_Q19_Q21_Q24.test.ts:1) | square8, square19, hex | ✅ COVERED | Always >50%; prevents ties                                                                                                      |
| Q22 | Graduated line rewards          | [`tests/scenarios/FAQ_Q22_Q23.test.ts`](tests/scenarios/FAQ_Q22_Q23.test.ts:1)         | square8, square19      | ✅ COVERED | Option 1 vs Option 2 strategic choices                                                                                          |
| Q23 | Self-elimination prerequisite   | [`tests/scenarios/FAQ_Q22_Q23.test.ts`](tests/scenarios/FAQ_Q22_Q23.test.ts:1)         | square8, square19, hex | ✅ COVERED | Must have outside stack to process region                                                                                       |
| Q24 | Forced elimination when blocked | [`tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`](tests/scenarios/FAQ_Q19_Q21_Q24.test.ts:1) | square8                | ✅ COVERED | Must eliminate cap when no moves available                                                                                      |

### FAQ Test File Breakdown

#### [`tests/scenarios/FAQ_Q01_Q06.test.ts`](tests/scenarios/FAQ_Q01_Q06.test.ts:1) - Basic Mechanics

- **FAQ Q1**: Stack order immutability
- **FAQ Q2**: Minimum distance requirements (height 1-4 examples)
- **FAQ Q3**: Landing flexibility during captures
- **FAQ Q4**: Only top ring captured per segment
- **FAQ Q5**: Single vs multiple captures (chain patterns)
- **FAQ Q6**: Overtaking vs Elimination distinction
- **Coverage**: Stack mechanics, movement distance, capture basics
- **Engines**: Backend GameEngine, Sandbox ClientSandboxEngine

#### [`tests/scenarios/FAQ_Q07_Q08.test.ts`](tests/scenarios/FAQ_Q07_Q08.test.ts:1) - Line Formation

- **FAQ Q7**: Multiple line processing, intersecting lines
- **FAQ Q8**: No rings available for elimination
- **Coverage**: Exact-length lines (4 for square8, 5 for square19), overlength lines, line invalidation
- **Engines**: Backend GameEngine

#### [`tests/scenarios/FAQ_Q09_Q14.test.ts`](tests/scenarios/FAQ_Q09_Q14.test.ts:1) - Edge Cases

- **FAQ Q9**: Mandatory chain despite blocking future moves
- **FAQ Q10**: Multicolored stacks in territory evaluation
- **FAQ Q12**: Chain continuing to self-elimination
- **FAQ Q13**: Moore vs Von Neumann adjacency systems
- **FAQ Q14**: Optional initial capture, mandatory chain continuation
- **FAQ Q20**: Territory adjacency comparison across board types
- **Coverage**: Edge cases, adjacency rules, mandatory mechanics
- **Engines**: Backend GameEngine

#### [`tests/scenarios/FAQ_Q15.test.ts`](tests/scenarios/FAQ_Q15.test.ts:1) - Chain Capture Patterns

- **FAQ Q15.3.1**: 180-degree reversal pattern (A→B→A)
- **FAQ Q15.3.2**: Cyclic pattern (A→B→C→A)
- **FAQ Q15.3.3**: Mandatory continuation until no legal captures
- **Coverage**: Complex chain patterns, reversal mechanics, mandatory chains
- **Engines**: Backend GameEngine, Sandbox ClientSandboxEngine

#### [`tests/scenarios/FAQ_Q16_Q18.test.ts`](tests/scenarios/FAQ_Q16_Q18.test.ts:1) - Victory & Control

- **FAQ Q16**: Control transfer in multicolored stacks, recovery of buried rings
- **FAQ Q17**: First placement has no special movement rules
- **FAQ Q18**: Victory condition priority, >50% prevents simultaneous wins
- **Coverage**: Stack control, victory thresholds, board-wide consistency
- **Engines**: Backend GameEngine

#### [`tests/scenarios/FAQ_Q19_Q21_Q24.test.ts`](tests/scenarios/FAQ_Q19_Q21_Q24.test.ts:1) - Player Counts & Forced Elimination

- **FAQ Q19**: 2, 3, and 4 player configurations
- **FAQ Q21**: Victory thresholds always >50% of total
- **FAQ Q24**: Forced elimination when blocked with stacks
- **FAQ Q11**: Rings in hand count as eliminated in stalemate
- **Coverage**: Player count variations, threshold calculations, forced elimination, stalemate
- **Engines**: Backend GameEngine

#### [`tests/scenarios/FAQ_Q22_Q23.test.ts`](tests/scenarios/FAQ_Q22_Q23.test.ts:1) - Graduated Rewards & Territory

- **FAQ Q22**: Strategic choices for overlength lines (Option 1 vs Option 2)
- **FAQ Q23**: Self-elimination prerequisite for processing regions
- **Coverage**: Graduated line rewards, territory processing prerequisites, multiple regions
- **Engines**: Backend GameEngine

### Running FAQ-Specific Tests

```bash
# Run all FAQ scenarios
npm test -- tests/scenarios/FAQ_

# Run specific FAQ clusters
npm test -- FAQ_Q01_Q06     # Basic mechanics (Q1-Q6)
npm test -- FAQ_Q07_Q08     # Line formation (Q7-Q8)
npm test -- FAQ_Q09_Q14     # Edge cases (Q9-Q14)
npm test -- FAQ_Q15         # Chain captures (Q15)
npm test -- FAQ_Q16_Q18     # Victory & control (Q16-Q18)
npm test -- FAQ_Q19_Q21_Q24 # Player counts & thresholds (Q19-Q21, Q24)
npm test -- FAQ_Q22_Q23     # Graduated rewards & territory (Q22-Q23)

# Run with verbose output for debugging
npm test -- FAQ_Q15 --verbose
```

### FAQ Coverage Summary

**Total FAQ Questions**: 24  
**Covered by Dedicated Tests**: 24 (100%)  
**Test Files Created**: 7  
**Total Test Cases**: ~50+ individual test cases

**Board Type Coverage**:

- ✅ Square 8×8: All FAQ questions
- ✅ Square 19×19: All applicable FAQ questions
- ✅ Hexagonal: All applicable FAQ questions (Q13, Q20, Q23)

**Engine Coverage**:

- ✅ Backend GameEngine: All FAQ questions
- ✅ Sandbox ClientSandboxEngine: Selected FAQ questions (Q1-Q6, Q15)

**Validation Level**:

- Each FAQ example from [`ringrift_complete_rules.md`](ringrift_complete_rules.md:1) has at least one test
- Complex scenarios (Q15, Q22-Q23) have multiple test cases
- Cross-FAQ integration tests validate rule interactions

---

| **PARTIAL** | Trace parity between sandbox AI games and backend replays (semantic comparison of moves & phases; currently failing for seed 5 territory/self‑elimination parity, see P0-TESTING-002) | `archive/tests/unit/Backend_vs_Sandbox.traceParity.test.ts` (archived), `tests/unit/Sandbox_vs_Backend.seed5.traceDebug.test.ts` | Backend + Sandbox | Uses `GameTrace` and `tests/utils/traces.ts` to compare step‑by‑step state; this suite currently encodes a known failing case (seed 5, move 45) and will become a hard CI gate once backend↔sandbox parity is restored. |
| **COVERED** | AI‑parallel debug runs (backend & sandbox) for seeded games, including mismatch logging and S‑snapshot comparisons | `tests/unit/Backend_vs_Sandbox.eliminationTrace.test.ts`; `tests/unit/Backend_vs_Sandbox.seed5.prefixDiagnostics.test.ts`; `tests/unit/Sandbox_vs_Backend.aiHeuristicCoverage.test.ts`; `tests/utils/traces.ts` | Backend + Sandbox | Heavy diagnostic harness over seeded AI games; the curated runs in the backend_vs_sandbox elimination/diagnostic suites and `Sandbox_vs_Backend.aiHeuristicCoverage.test.ts` are treated as CI gates for backend↔sandbox + AI parity. |
| **COVERED** | Sandbox AI simulation S‑invariant, plateau regression, and stall detection | `tests/unit/ClientSandboxEngine.aiSimulation.test.ts`, `tests/unit/ClientSandboxEngine.aiStallRegression.test.ts`, `tests/unit/ClientSandboxEngine.aiSingleSeedDebug.test.ts`, `tests/scenarios/AI_TerminationFromSeed1Plateau.test.ts`, `tests/unit/ProgressSnapshot.core.test.ts`, `tests/unit/ProgressSnapshot.sandbox.test.ts` | Backend + Sandbox | aiSimulation/aiStall regression and plateau suites provide seeded, diagnostic coverage; the ProgressSnapshot core/sandbox tests add explicit, hand-built S-invariant checks (Rules*9*\*), asserting M/C/E counts and that canonical marker→territory+elimination transitions strictly increase S. These invariants act as gating tests for S‑monotonicity. |

**Planned additions**

- **PLANNED:** Add additional seeded parity suites (see `backend_vs_sandbox_trace_parity_extra_seeds` above) once new high‑value traces are identified; they should follow the same hard‑CI‑gate treatment and include a short, rule‑tagged comment block referencing compact rules §9.

---

## 10. Contract vectors and Python parity suites

The highest‑value rules/scenario clusters above are also backed by a **shared
contract‑vector + parity layer** centered on the canonical TS orchestrator and
its cross‑language contracts:

- **Contract schemas & serialization (TS SSoT):**
  - `src/shared/engine/contracts/schemas.ts`
  - `src/shared/engine/contracts/serialization.ts`
  - `src/shared/engine/contracts/testVectorGenerator.ts`
- **Contract vectors (v2):**
  - `tests/fixtures/contract-vectors/v2/placement.vectors.json`
  - `tests/fixtures/contract-vectors/v2/movement.vectors.json`
  - `tests/fixtures/contract-vectors/v2/capture.vectors.json`
  - `tests/fixtures/contract-vectors/v2/line_detection.vectors.json`
  - `tests/fixtures/contract-vectors/v2/territory.vectors.json`
- **TS contract runner:** `tests/contracts/contractVectorRunner.test.ts`
- **Python contract runner:** `ai-service/tests/contracts/test_contract_vectors.py`
- **Python parity suites (scenario‑oriented):**
  - `ai-service/tests/parity/test_line_and_territory_scenario_parity.py`
  - `ai-service/tests/parity/test_rules_parity.py`
  - `ai-service/tests/parity/test_rules_parity_fixtures.py`
  - `ai-service/tests/parity/test_active_no_moves_line_processing_regression.py`
  - `ai-service/tests/parity/test_ts_seed_plateau_snapshot_parity.py`

At a high level, the mapping between this matrix and the contract/parity layer
is:

| Scenario / cluster (from tables above)                                            | Contract vectors (v2)                                           | TS tests (orchestrator/contract layer)                                                                                | Python tests (parity layer)                                                                                                                        |
| --------------------------------------------------------------------------------- | --------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Non‑capture movement & basic captures** (M1–M3, parts of §§8–10, FAQ 2–6, 9,12) | `movement.vectors.json`, `capture.vectors.json`                 | `tests/contracts/contractVectorRunner.test.ts` ("movement" and "capture" suites)                                      | `ai-service/tests/contracts/test_contract_vectors.py::test_movement_vectors`, `::test_capture_vectors`                                             |
| **Line detection & basic line rewards** (L1–L4; §11, FAQ 7, 22)                   | `line_detection.vectors.json`                                   | `tests/contracts/contractVectorRunner.test.ts` ("line_detection" suite)                                               | `ai-service/tests/contracts/test_contract_vectors.py::test_line_detection_vectors`                                                                 |
| **Territory detection & region processing** (T1–T4; §12, FAQ 10, 15, 20, 23)      | `territory.vectors.json`                                        | `tests/contracts/contractVectorRunner.test.ts` ("territory" suite)                                                    | `ai-service/tests/contracts/test_contract_vectors.py::test_territory_vectors`; `test_line_and_territory_scenario_parity.py`                        |
| **Placement / skip‑placement semantics** (parts of §§4, 8; FAQ 1–3)               | `placement.vectors.json`                                        | `tests/contracts/contractVectorRunner.test.ts` ("placement" suite)                                                    | `ai-service/tests/contracts/test_contract_vectors.py::test_placement_vectors`                                                                      |
| **Plateau / "active but no moves" regressions** (see compact rules §9 notes)      | Seeded traces under `tests/fixtures/rules-parity/v1/*.json`     | `tests/unit/TraceFixtures.sharedEngineParity.test.ts`; `tests/unit/Seed5TerminalSnapshot.parity.test.ts`              | `ai-service/tests/parity/test_ts_seed_plateau_snapshot_parity.py`; `ai-service/tests/parity/test_ai_plateau_progress.py`                           |
| **Active‑no‑moves line/territory regressions** (T‑edge & line/territory chains)   | Specific v2 vectors + v1 traces (line + territory combinations) | `tests/unit/TerritoryDecision.seed5Move45.parity.test.ts`; `tests/unit/TerritoryDetection.seed5Move45.parity.test.ts` | `ai-service/tests/parity/test_active_no_moves_line_processing_regression.py`; `ai-service/tests/parity/test_line_and_territory_scenario_parity.py` |

> **Operational note:** for end‑to‑end TS↔Python parity on the orchestrator
> surface, prefer running the **contract vector suites** and **Python parity
> tests** over re‑adding ad‑hoc backend‑vs‑sandbox assertions. Those suites
> treat the shared TS orchestrator (`src/shared/engine/orchestration/`) as the
> canonical semantics and validate the Python rules engine against it.
>
> See also: `docs/PYTHON_PARITY_REQUIREMENTS.md` for the full function/type
> parity matrix and shadow‑contract design.

## 11. How to extend this matrix

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
