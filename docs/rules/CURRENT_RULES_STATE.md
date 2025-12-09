# RingRift Rules Implementation - Current State

A central navigation guide for developers to quickly locate all rules-related documentation, implementation details, and verification reports.

**Last Updated:** 2025-12-01

> **Doc Status (2025-12-01): Active (with historical appendix)**
> Index and navigation guide to rules specifications, implementation mapping, and verification/audit docs. This file is **not** a rules semantics SSoT; it points to the true SSoTs and their verification harnesses.

- **Rules semantics SSoT:** The canonical rules documents (`RULES_CANONICAL_SPEC.md` together with `ringrift_complete_rules.md` / `ringrift_compact_rules.md`) are the **single source of truth** for RingRift game semantics. The shared TypeScript engine under `src/shared/engine/` (helpers → domain aggregates → turn orchestrator → contracts), cross-language contracts and vectors (`src/shared/engine/contracts/**`, `tests/fixtures/contract-vectors/v2/**`, `tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`), and rules docs (`RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`, `RULES_SCENARIO_MATRIX.md`, `docs/STRICT_INVARIANT_SOAKS.md`) describe and validate the primary executable implementation of that spec.
  > - **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` and shared types/schemas under `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
  > - Some linked documents (especially under `archive/` and older UX/audit reports) are **partially historical**; this file is kept current but intentionally points at both active and archived material.
  > - For high-level architecture/topology, see `ARCHITECTURE_ASSESSMENT.md`, `ARCHITECTURE_REMEDIATION_PLAN.md`, and `DOCUMENTATION_INDEX.md`.

---

## Quick Status Summary

| Aspect         | Status                 | Notes                                                                                                  |
| -------------- | ---------------------- | ------------------------------------------------------------------------------------------------------ |
| Rules Engine   | ✅ Fully implemented   | TS/Python parity verified (core flows)                                                                 |
| Orchestrator   | ✅ 100% rollout in CI  | All environments use orchestrator adapter                                                              |
| Known Issues   | ⚠️ Frontend UX gaps    | See [PASS18_ASSESSMENT_REPORT_PASS3.md](docs/PASS18_ASSESSMENT_REPORT_PASS3.md)                        |
| Weakest Aspect | Frontend UX Polish     | See [WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md](../../WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md) (Pass 18-3) |
| Test Health    | ✅ 2,670 tests passing | 0 failing, 176 skipped (orchestrator-conditional + diagnostic)                                         |

---

## Document Index

### Rules Specification

| Document                                                                                                 | Description                       | When to Use                                        |
| -------------------------------------------------------------------------------------------------------- | --------------------------------- | -------------------------------------------------- |
| [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md)                                                       | Canonical rules with RR-CANON IDs | Authoritative reference for any rule question      |
| [docs/supplementary/RULES_RULESET_CLARIFICATIONS.md](docs/supplementary/RULES_RULESET_CLARIFICATIONS.md) | Ambiguity resolutions (CLAR-XXX)  | When handling edge cases or unclear scenarios      |
| [ringrift_complete_rules.md](ringrift_complete_rules.md)                                                 | Player-facing complete rules      | Understanding full ruleset from player perspective |
| [ringrift_compact_rules.md](ringrift_compact_rules.md)                                                   | Player-facing compact rules       | Quick reference or onboarding new players          |

### Implementation

| Document                                                           | Description                                       | When to Use                                          |
| ------------------------------------------------------------------ | ------------------------------------------------- | ---------------------------------------------------- |
| [RULES_IMPLEMENTATION_MAPPING.md](RULES_IMPLEMENTATION_MAPPING.md) | Rules → Code mapping (RR-CANON → files/functions) | Finding which code implements a specific rule        |
| [RULES_ENGINE_ARCHITECTURE.md](RULES_ENGINE_ARCHITECTURE.md)       | Engine architecture overview                      | Understanding system design and module relationships |

### Verification & Audit

| Document                                                                                                 | Description                     | When to Use                                   |
| -------------------------------------------------------------------------------------------------------- | ------------------------------- | --------------------------------------------- |
| [archive/RULES_STATIC_VERIFICATION.md](archive/RULES_STATIC_VERIFICATION.md)                             | Static code analysis results    | Reviewing code coverage of rules              |
| [archive/RULES_DYNAMIC_VERIFICATION.md](archive/RULES_DYNAMIC_VERIFICATION.md)                           | Dynamic test mapping            | Finding tests for specific rules              |
| [docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md) | Edge case analysis and handling | Debugging unusual game states                 |
| [docs/supplementary/RULES_DOCS_UX_AUDIT.md](docs/supplementary/RULES_DOCS_UX_AUDIT.md)                   | Documentation/UX audit findings | Improving player-facing documentation         |
| [archive/FINAL_RULES_AUDIT_REPORT.md](archive/FINAL_RULES_AUDIT_REPORT.md)                               | Complete audit report           | Understanding overall rules compliance status |

Additional rules‑UX and onboarding specs:

- [`UX_RULES_COPY_SPEC.md`](../docs/UX_RULES_COPY_SPEC.md:1) – Canonical copy for HUD, VictoryModal, TeachingOverlay, and sandbox rules explanations.
- [`UX_RULES_TELEMETRY_SPEC.md`](../docs/UX_RULES_TELEMETRY_SPEC.md:1) – Rules‑UX telemetry envelope, metrics, and hotspot queries (W‑UX‑1).
- [`UX_RULES_WEIRD_STATES_SPEC.md`](../docs/UX_RULES_WEIRD_STATES_SPEC.md:1) – Weird‑state reason codes and UX mappings for ANM/FE, structural stalemate, and LPS (W‑UX‑2).
- [`UX_RULES_TEACHING_SCENARIOS.md`](../docs/UX_RULES_TEACHING_SCENARIOS.md:1) – Scenario‑driven teaching flows for complex mechanics (W‑UX‑3).
- [`UX_RULES_IMPROVEMENT_LOOP.md`](../docs/UX_RULES_IMPROVEMENT_LOOP.md:1) – Telemetry‑driven rules‑UX improvement process (W‑UX‑4).

### Process & Tools

| Document                                                               | Description                                    | When to Use                                                       |
| ---------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------- |
| [archive/RULES_CHANGE_CHECKLIST.md](archive/RULES_CHANGE_CHECKLIST.md) | Change workflow checklist (archived reference) | Historical reference before making major rules-related changes    |
| [scripts/rules-health-report.sh](scripts/rules-health-report.sh)       | Health report script                           | Automated compliance verification (runs core rules/parity suites) |

### AI & Training

| Document                                                             | Description                | When to Use                                     |
| -------------------------------------------------------------------- | -------------------------- | ----------------------------------------------- |
| [docs/AI_TRAINING_AND_DATASETS.md](docs/AI_TRAINING_AND_DATASETS.md) | AI alignment documentation | Training AI or ensuring rules consistency in AI |

#### Decision timers and countdowns

- **Canonical units (host):** All host-side timeouts are expressed in milliseconds (`ms`), including:
  - Decision-phase deadlines configured via `config.decisionPhaseTimeouts.defaultTimeoutMs` and warnings via `warningBeforeTimeoutMs`, plus the `remainingMs` field in `decision_phase_timeout_warning` events (see [`P18.3-1_DECISION_LIFECYCLE_SPEC.md`](../archive/assessments/P18.3-1_DECISION_LIFECYCLE_SPEC.md:125)).
  - Transport-level `PlayerChoice` deadlines (`timeoutMs` / `deadlineAt`) and per-player move clocks (`timeRemaining`).
  - Reconnect windows (`RECONNECTION_TIMEOUT_MS`) and any move-clock-style inactivity timers on the host.
- **Client countdown state (ms):** Client HUD timers are also stored in ms.
  - Hooks such as [`useDecisionCountdown()`](../../src/client/hooks/useDecisionCountdown.ts:66) accept `baseTimeRemainingMs` (derived from host deadlines) and optional server overrides (`timeoutWarning.data.remainingMs`) and compute an `effectiveTimeRemainingMs` that is always `>= 0`.
  - Internally, helper [`normalizeMs()`](../../src/client/hooks/useDecisionCountdown.ts:61) treats non-numeric inputs as `null` and clamps negative values to `0` for display/merging; these values are never interpreted as an authoritative expiry.
- **Display semantics (seconds):** Visible countdown numbers in the HUD and dialogs are whole seconds computed from ms via [`msToDisplaySeconds()`](../../src/client/utils/countdown.ts:43):
  - Nullish or non-finite inputs return `null` (no countdown rendered).
  - `timeRemainingMs <= 0` displays as `0` seconds.
  - For `timeRemainingMs > 0`, the UI shows `Math.ceil(timeRemainingMs / 1000)`, so `1..1000ms → 1s`, `1001..2000ms → 2s`, etc. This avoids showing `0s` while strictly positive time remains.
- **Severity and styling:** The HUD uses [`getCountdownSeverity()`](../../src/client/utils/countdown.ts:12) directly on the ms value to choose styling buckets:
  - `normal` when `timeRemainingMs > 10_000`.
  - `warning` when `3_000 < timeRemainingMs <= 10_000`.
  - `critical` when `timeRemainingMs <= 3_000` (including `0` and negative values).
  - The combination of `effectiveTimeRemainingMs` (ms), [`msToDisplaySeconds()`](../../src/client/utils/countdown.ts:43) for numeric display, and [`getCountdownSeverity()`](../../src/client/utils/countdown.ts:12) for styling is the canonical pattern for decision timers and related UI.
- **Authority and expiry:** Countdown UI is advisory:
  - Actual expiry of decisions and timeouts is governed by host logic and `GameState`, per [`P18.3-1_DECISION_LIFECYCLE_SPEC.md`](../archive/assessments/P18.3-1_DECISION_LIFECYCLE_SPEC.md:125) (for example, decision-phase timeout handlers and reconnect windows).
  - UI components MUST NOT treat a client-side `0s` display as proof that a decision has expired; they should instead react to changes such as the pending decision disappearing or a `decision_phase_timed_out` / `game_state` update.
  - This aligns with ANM and timeout handling catalogued in [`ACTIVE_NO_MOVES_BEHAVIOUR.md`](ACTIVE_NO_MOVES_BEHAVIOUR.md:1).

#### Pie rule / `swap_sides` status

- **Backend TS & Python:** Both engines now treat `swap_sides` as a first‑class meta‑move for 2‑player games, with identical gating (P2’s interactive turn, `rulesOptions.swapRuleEnabled === true`, exactly after P1’s first non‑swap move, and at most once per game). Production remains TS‑authoritative; Python is used for validation, AI, and training.
- **Client sandbox:** `ClientSandboxEngine` mirrors the backend pie‑rule gate and applies `swap_sides` locally for 2‑player sandbox games, with `/sandbox` exposing a “Swap colours (pie rule)” control once for P2 after P1’s opening move.
- **AI training/eval:** Python training and soak harnesses now enable production‑style pie‑rule availability **by default** for 2‑player games by setting `rulesOptions.swapRuleEnabled: true` on initial states. For experiments that must run without the pie rule, callers can explicitly opt out via the `RINGRIFT_TRAINING_DISABLE_SWAP_RULE=1` flag. CMA‑ES/GA evaluation pipelines emit lightweight diagnostics on how often AIs select `swap_sides` during runs.

---

## Key Engine Modules

### TypeScript Shared Engine

| Module                             | Purpose                                                  |
| ---------------------------------- | -------------------------------------------------------- |
| `src/shared/engine/core.ts`        | Core board helpers, S‑invariant, and progress snapshot   |
| `src/shared/engine/validators/`    | Pure move and placement validation logic                 |
| `src/shared/engine/mutators/`      | Low-level state mutation helpers                         |
| `src/shared/engine/aggregates/`    | Aggregates for placement, movement, capture, lines, etc. |
| `src/shared/engine/orchestration/` | Turn/phase state machine and canonical orchestrator      |
| `src/shared/engine/index.ts`       | Canonical shared-engine entry point and exports          |

### TypeScript Backend

| Module                          | Purpose              |
| ------------------------------- | -------------------- |
| `src/server/game/GameEngine.ts` | Backend orchestrator |
| `src/server/game/RuleEngine.ts` | Rules interface      |

### Python Rules Engine

| Module                          | Purpose              |
| ------------------------------- | -------------------- |
| `ai-service/app/game_engine.py` | Python game engine   |
| `ai-service/app/rules/`         | Python rules modules |

---

## Test Suites

| Test Category            | Location / Examples                                                                                                                                                                                             |
| ------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Unit tests               | Core shared-engine and rules tests under `tests/unit/**` (e.g. `movement.shared.test.ts`, `captureLogic.shared.test.ts`)                                                                                        |
| Scenario tests           | Behavioural and FAQ/RulesMatrix scenarios under `tests/scenarios/**`                                                                                                                                            |
| Host parity tests        | Backend vs sandbox parity suites under `tests/unit/Backend_vs_Sandbox.*.test.ts` and related trace/parity tests                                                                                                 |
| TS/Python parity tests   | TS↔Python integration harness in `src/server/game/test-python-rules-integration.ts` and Python suites in `ai-service/tests/parity/**`                                                                           |
| Determinism & invariants | Determinism/invariant suites in `tests/unit/EngineDeterminism.shared.test.ts`, `tests/unit/NoRandomInCoreRules.test.ts`, `ai-service/tests/test_engine_determinism.py`, plus `./scripts/rules-health-report.sh` |

### Recent rules/parity fixes (2025‑11–2025‑12)

- Mixed line+territory flows for Q7/Q20 on all three board families now have dedicated v2 multi‑phase turn vectors (`sequence:turn.line_then_territory.{square8,square19,hex}`) in `multi_phase_turn.vectors.json`, with orchestrator‑backed TS tests and Python parity/snapshot checks (see `RULES_SCENARIO_MATRIX.md` row `combined_line_then_territory_full_sequence`).
- Q20/Q23‑style two‑region territory and mixed line+multi‑region territory scenarios are captured as v2 sequences (`territory.*two_regions_then_elim` and `sequence:turn.line_then_territory.multi_region.{square8,square19,hex}`) plus backend↔sandbox parity suites and Python metadata/snapshot tests (see `RULES_IMPLEMENTATION_MAPPING.md` “Multi‑region territory & mixed line+territory index”).
- Forced‑elimination semantics are unified behind `applyForcedEliminationForPlayer` / `enumerateForcedEliminationOptions` on the TS side, with sandbox `RingEliminationChoice` flows and v2 `forced_elimination.*` vectors / Python invariants acting as the SSOT for blocked‑player behaviour.
- Extended v2 contract bundles (including `territory_line_endgame`, multi‑region territory, mixed line+territory, and near‑victory territory) all pass on both TS and Python (`tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py` and the parity suites under `ai-service/tests/parity/**`), closing previously open rules‑parity gaps called out in earlier audit reports.

## Getting Started

1. **Understand the rules:** Start with [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md) for authoritative rule definitions.
2. **Find relevant code:** Check [RULES_IMPLEMENTATION_MAPPING.md](RULES_IMPLEMENTATION_MAPPING.md) to locate implementation.
3. **Before making changes:** Review the historical checklist in [archive/RULES_CHANGE_CHECKLIST.md](archive/RULES_CHANGE_CHECKLIST.md) for context on the rules-change workflow.
4. **Verify compliance:** Run `./scripts/rules-health-report.sh` to check rules health.
5. **Review context:** See [archive/FINAL_RULES_AUDIT_REPORT.md](archive/FINAL_RULES_AUDIT_REPORT.md) for audit findings and resolution status.
