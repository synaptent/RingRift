# RingRift Rules Implementation - Current State

A central navigation guide for developers to quickly locate all rules-related documentation, implementation details, and verification reports.

**Last Updated:** November 25, 2025

---

## Quick Status Summary

| Aspect         | Status                         | Notes                                      |
| -------------- | ------------------------------ | ------------------------------------------ |
| Rules Engine   | ✅ Fully implemented           | TS/Python parity verified                  |
| Known Issues   | None critical                  | See [KNOWN_ISSUES.md](KNOWN_ISSUES.md)     |
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

| Document                                                                     | Description               | When to Use                             |
| ---------------------------------------------------------------------------- | ------------------------- | --------------------------------------- |
| [deprecated/RULES_CHANGE_CHECKLIST.md](deprecated/RULES_CHANGE_CHECKLIST.md) | Change workflow checklist | Before making any rules-related changes |
| [scripts/rules-health-report.sh](scripts/rules-health-report.sh)             | Health report script      | Automated compliance verification       |

### AI & Training

| Document                                                             | Description                | When to Use                                     |
| -------------------------------------------------------------------- | -------------------------- | ----------------------------------------------- |
| [docs/AI_TRAINING_AND_DATASETS.md](docs/AI_TRAINING_AND_DATASETS.md) | AI alignment documentation | Training AI or ensuring rules consistency in AI |

---

## Key Engine Modules

### TypeScript Shared Engine

| Module                            | Purpose                |
| --------------------------------- | ---------------------- |
| `src/shared/engine/core.ts`       | Core helpers and types |
| `src/shared/engine/validators/`   | Move validation logic  |
| `src/shared/engine/mutators/`     | State mutation logic   |
| `src/shared/engine/GameEngine.ts` | Shared game engine     |

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

| Test Category          | Location                                                        |
| ---------------------- | --------------------------------------------------------------- |
| Unit tests             | `tests/unit/*.shared.test.ts`                                   |
| Scenario tests         | `tests/scenarios/`                                              |
| TS/Python parity tests | `tests/unit/Python_vs_TS.*.test.ts`, `ai-service/tests/parity/` |
| Determinism tests      | See `./scripts/rules-health-report.sh`                          |

---

## Getting Started

1. **Understand the rules:** Start with [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md) for authoritative rule definitions.
2. **Find relevant code:** Check [RULES_IMPLEMENTATION_MAPPING.md](RULES_IMPLEMENTATION_MAPPING.md) to locate implementation.
3. **Before making changes:** Follow [deprecated/RULES_CHANGE_CHECKLIST.md](deprecated/RULES_CHANGE_CHECKLIST.md) workflow.
4. **Verify compliance:** Run `./scripts/rules-health-report.sh` to check rules health.
5. **Review context:** See [archive/FINAL_RULES_AUDIT_REPORT.md](archive/FINAL_RULES_AUDIT_REPORT.md) for audit findings and resolution status.
