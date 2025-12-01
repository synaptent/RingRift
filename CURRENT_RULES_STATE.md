# RingRift Rules Implementation - Current State

A central navigation guide for developers to quickly locate all rules-related documentation, implementation details, and verification reports.

**Last Updated:** November 25, 2025

> **Doc Status (2025-11-27): Active (with historical appendix)**  
> Index and navigation guide to rules specifications, implementation mapping, and verification/audit docs. This file is **not** a rules semantics SSoT; it points to the true SSoTs and their verification harnesses.
>
> - **Rules semantics SSoT:** Shared TypeScript engine under `src/shared/engine/` (helpers → domain aggregates → turn orchestrator → contracts) plus cross-language contracts and vectors (`src/shared/engine/contracts/**`, `tests/fixtures/contract-vectors/v2/**`, `tests/contracts/contractVectorRunner.test.ts`, `ai-service/tests/contracts/test_contract_vectors.py`) and rules docs (`RULES_CANONICAL_SPEC.md`, `RULES_ENGINE_ARCHITECTURE.md`, `RULES_IMPLEMENTATION_MAPPING.md`, `docs/RULES_ENGINE_SURFACE_AUDIT.md`, `RULES_SCENARIO_MATRIX.md`, `docs/STRICT_INVARIANT_SOAKS.md`).
> - **Lifecycle/API SSoT:** `docs/CANONICAL_ENGINE_API.md` and shared types/schemas under `src/shared/types/game.ts`, `src/shared/engine/orchestration/types.ts`, `src/shared/types/websocket.ts`, and `src/shared/validation/websocketSchemas.ts`.
> - Some linked documents (especially under `archive/` and older UX/audit reports) are **partially historical**; this file is kept current but intentionally points at both active and archived material.
> - For high-level architecture/topology, see `ARCHITECTURE_ASSESSMENT.md`, `ARCHITECTURE_REMEDIATION_PLAN.md`, and `DOCUMENTATION_INDEX.md`.

---

## Quick Status Summary

| Aspect         | Status                         | Notes                                      |
| -------------- | ------------------------------ | ------------------------------------------ |
| Rules Engine   | ✅ Fully implemented           | TS/Python parity verified (core flows)     |
| Known Issues   | ⚠️ Host integration regressions | See [WEAKNESS_ASSESSMENT_REPORT.md](WEAKNESS_ASSESSMENT_REPORT.md) |
| Recent Changes | LPS, ring caps, clarifications | Territory processing, line rewards aligned |

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

### Process & Tools

| Document                                                               | Description                                    | When to Use                                                       |
| ---------------------------------------------------------------------- | ---------------------------------------------- | ----------------------------------------------------------------- |
| [archive/RULES_CHANGE_CHECKLIST.md](archive/RULES_CHANGE_CHECKLIST.md) | Change workflow checklist (archived reference) | Historical reference before making major rules-related changes    |
| [scripts/rules-health-report.sh](scripts/rules-health-report.sh)       | Health report script                           | Automated compliance verification (runs core rules/parity suites) |

### AI & Training

| Document                                                             | Description                | When to Use                                     |
| -------------------------------------------------------------------- | -------------------------- | ----------------------------------------------- |
| [docs/AI_TRAINING_AND_DATASETS.md](docs/AI_TRAINING_AND_DATASETS.md) | AI alignment documentation | Training AI or ensuring rules consistency in AI |

#### Pie rule / `swap_sides` status

- **Backend TS & Python:** Both engines now treat `swap_sides` as a first‑class meta‑move for 2‑player games, with identical gating (P2’s interactive turn, `rulesOptions.swapRuleEnabled === true`, exactly after P1’s first non‑swap move, and at most once per game). Production remains TS‑authoritative; Python is used for validation, AI, and training.
- **Client sandbox:** `ClientSandboxEngine` mirrors the backend pie‑rule gate and applies `swap_sides` locally for 2‑player sandbox games, with `/sandbox` exposing a “Swap colours (pie rule)” control once for P2 after P1’s opening move.
- **AI training/eval:** Python training and soak harnesses can opt into production‑style pie‑rule availability by setting `rulesOptions.swapRuleEnabled: true` on initial states (for example via `RINGRIFT_TRAINING_ENABLE_SWAP_RULE=1` in self‑play generators). CMA‑ES/GA evaluation pipelines now emit lightweight diagnostics on how often AIs select `swap_sides` during runs.

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
| TS/Python parity tests   | TS↔Python integration harness in `src/server/game/test-python-rules-integration.ts` and Python suites in `ai-service/tests/parity/**`                                                                          |
| Determinism & invariants | Determinism/invariant suites in `tests/unit/EngineDeterminism.shared.test.ts`, `tests/unit/NoRandomInCoreRules.test.ts`, `ai-service/tests/test_engine_determinism.py`, plus `./scripts/rules-health-report.sh` |

---

## Getting Started

1. **Understand the rules:** Start with [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md) for authoritative rule definitions.
2. **Find relevant code:** Check [RULES_IMPLEMENTATION_MAPPING.md](RULES_IMPLEMENTATION_MAPPING.md) to locate implementation.
3. **Before making changes:** Review the historical checklist in [archive/RULES_CHANGE_CHECKLIST.md](archive/RULES_CHANGE_CHECKLIST.md) for context on the rules-change workflow.
4. **Verify compliance:** Run `./scripts/rules-health-report.sh` to check rules health.
5. **Review context:** See [archive/FINAL_RULES_AUDIT_REPORT.md](archive/FINAL_RULES_AUDIT_REPORT.md) for audit findings and resolution status.
