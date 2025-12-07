# RingRift Documentation Index

> **Last Updated:** 2025-12-07
> **Organization:** Core docs in root (~16 files), detailed docs in `/docs/` subdirectories

This index catalogs all project documentation organized by topic and location. For a lightweight landing page, see `docs/INDEX.md`.

---

## Quick Start

| Document                                   | Purpose                            |
| ------------------------------------------ | ---------------------------------- |
| [README.md](README.md)                     | Project overview, features, status |
| [QUICKSTART.md](QUICKSTART.md)             | Local development setup            |
| [CONTRIBUTING.md](CONTRIBUTING.md)         | Contribution guidelines            |
| [TODO.md](TODO.md)                         | Active task tracker                |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md)         | Current bugs and gaps              |
| [IMPROVEMENT_PLAN.md](IMPROVEMENT_PLAN.md) | Comprehensive improvement roadmap  |

---

## Core Documentation (Root)

### Project Status & Planning

- [CURRENT_STATE_ASSESSMENT.md](CURRENT_STATE_ASSESSMENT.md) - Implementation status snapshot relative to the goals in PROJECT_GOALS; does not define new goals
- [STRATEGIC_ROADMAP.md](STRATEGIC_ROADMAP.md) - Phased roadmap & SLOs that operationalise the goals in PROJECT_GOALS
- [PROJECT_GOALS.md](PROJECT_GOALS.md) - Canonical project goals, v1.0 success criteria, and scope boundaries (authoritative source for goals/scope)
- [WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md](WEAKNESS_AND_HARDEST_PROBLEM_REPORT.md) - Canonical weakest-aspect & hardest-problem assessment snapshot (answer artifact for original assessment)
- [NEXT_WAVE_REMEDIATION_PLAN.md](NEXT_WAVE_REMEDIATION_PLAN.md) - Next-wave remediation plan derived from weakest-aspect / hardest-problem assessment

### Rules & Game Design

- [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md) - Authoritative rules specification
- [ringrift_complete_rules.md](ringrift_complete_rules.md) - Full rulebook
- [ringrift_simple_human_rules.md](ringrift_simple_human_rules.md) - Human-readable rules
- [docs/rules/ringrift_compact_rules.md](docs/rules/ringrift_compact_rules.md) - Implementation-focused summary

### Architecture

- [RULES_ENGINE_ARCHITECTURE.md](RULES_ENGINE_ARCHITECTURE.md) - Rules engine design
- [AI_ARCHITECTURE.md](AI_ARCHITECTURE.md) - AI service architecture

---

## /docs/ Directory Structure

### /docs/architecture/

Engine and system architecture documentation.

| Document                                                                                                 | Purpose                          |
| -------------------------------------------------------------------------------------------------------- | -------------------------------- |
| [API_REFERENCE.md](docs/architecture/API_REFERENCE.md)                                                   | REST API documentation           |
| [CANONICAL_ENGINE_API.md](docs/architecture/CANONICAL_ENGINE_API.md)                                     | Public engine API specification  |
| [DOMAIN_AGGREGATE_DESIGN.md](docs/architecture/DOMAIN_AGGREGATE_DESIGN.md)                               | Domain model and aggregates      |
| [MODULE_RESPONSIBILITIES.md](docs/architecture/MODULE_RESPONSIBILITIES.md)                               | Module catalog                   |
| [PLAYER_MOVE_TRANSPORT_DECISION.md](docs/architecture/PLAYER_MOVE_TRANSPORT_DECISION.md)                 | WebSocket vs HTTP move transport |
| [STATE_MACHINES.md](docs/architecture/STATE_MACHINES.md)                                                 | Session/AI/choice state machines |
| [TOPOLOGY_MODES.md](docs/architecture/TOPOLOGY_MODES.md)                                                 | Board topology design            |
| [ORCHESTRATOR_ROLLOUT_PLAN.md](docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md)                           | Orchestrator migration plan      |
| [ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md](docs/architecture/ORCHESTRATOR_MIGRATION_COMPLETION_PLAN.md) | Migration completion             |
| [SHARED_ENGINE_CONSOLIDATION_PLAN.md](docs/architecture/SHARED_ENGINE_CONSOLIDATION_PLAN.md)             | Engine consolidation design      |

### /docs/rules/

Rules engine implementation and parity documentation.

| Document                                                                            | Purpose                             |
| ----------------------------------------------------------------------------------- | ----------------------------------- |
| [CURRENT_RULES_STATE.md](docs/rules/CURRENT_RULES_STATE.md)                         | Current rules implementation status |
| [RULES_IMPLEMENTATION_MAPPING.md](docs/rules/RULES_IMPLEMENTATION_MAPPING.md)       | Rules to code mapping               |
| [RULES_SCENARIO_MATRIX.md](docs/rules/RULES_SCENARIO_MATRIX.md)                     | Test scenario coverage              |
| [RULES_ENGINE_SURFACE_AUDIT.md](docs/rules/RULES_ENGINE_SURFACE_AUDIT.md)           | Engine surface area audit           |
| [RULES_SSOT_MAP.md](docs/rules/RULES_SSOT_MAP.md)                                   | Single source of truth mapping      |
| [CONTRACT_VECTORS_DESIGN.md](docs/rules/CONTRACT_VECTORS_DESIGN.md)                 | Cross-language parity vectors       |
| [INVARIANTS_AND_PARITY_FRAMEWORK.md](docs/rules/INVARIANTS_AND_PARITY_FRAMEWORK.md) | Rules invariants catalog            |
| [PARITY_SEED_TRIAGE.md](docs/rules/PARITY_SEED_TRIAGE.md)                           | Parity failure triage               |
| [PYTHON_PARITY_REQUIREMENTS.md](docs/rules/PYTHON_PARITY_REQUIREMENTS.md)           | Python AI parity requirements       |
| [SSOT_BANNER_GUIDE.md](docs/rules/SSOT_BANNER_GUIDE.md)                             | SSOT banner conventions             |
| [ACTIVE_NO_MOVES_BEHAVIOUR.md](docs/rules/ACTIVE_NO_MOVES_BEHAVIOUR.md)             | Active-no-moves edge case           |

### /docs/ai/

AI service and training documentation.

| Document                                                                                     | Purpose                         |
| -------------------------------------------------------------------------------------------- | ------------------------------- |
| [AI_DIFFICULTY_ANALYSIS.md](docs/ai/AI_DIFFICULTY_ANALYSIS.md)                               | Difficulty level analysis       |
| [AI_LARGE_BOARD_PERFORMANCE_ASSESSMENT.md](docs/ai/AI_LARGE_BOARD_PERFORMANCE_ASSESSMENT.md) | Large board performance         |
| [AI_TRAINING_AND_DATASETS.md](docs/ai/AI_TRAINING_AND_DATASETS.md)                           | Training pipelines and datasets |
| [AI_TRAINING_ASSESSMENT_FINAL.md](docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md)                   | Training assessment             |
| [AI_TRAINING_PREPARATION_GUIDE.md](docs/ai/AI_TRAINING_PREPARATION_GUIDE.md)                 | Training preparation            |

### /docs/testing/

Test infrastructure and QA documentation.

| Document                                                                                                                  | Purpose                        |
| ------------------------------------------------------------------------------------------------------------------------- | ------------------------------ |
| [TEST_CATEGORIES.md](docs/testing/TEST_CATEGORIES.md)                                                                     | CI vs diagnostic test types    |
| [TEST_INFRASTRUCTURE.md](docs/testing/TEST_INFRASTRUCTURE.md)                                                             | Test framework setup           |
| [STRICT_INVARIANT_SOAKS.md](docs/testing/STRICT_INVARIANT_SOAKS.md)                                                       | Long-running invariant tests   |
| [E2E_AUTH_AND_GAME_FLOW_TEST_STABILIZATION_SUMMARY.md](docs/testing/E2E_AUTH_AND_GAME_FLOW_TEST_STABILIZATION_SUMMARY.md) | E2E test stabilization         |
| [GO_NO_GO_CHECKLIST.md](docs/testing/GO_NO_GO_CHECKLIST.md)                                                               | Production readiness checklist |
| [LOAD_TEST_BASELINE.md](docs/testing/LOAD_TEST_BASELINE.md)                                                               | Load test baseline targets     |
| [LOAD_TEST_BASELINE_REPORT.md](docs/testing/LOAD_TEST_BASELINE_REPORT.md)                                                 | Load test results report       |
| [LOAD_TEST_WEBSOCKET_MOVE_STRATEGY.md](docs/testing/LOAD_TEST_WEBSOCKET_MOVE_STRATEGY.md)                                 | WebSocket load testing design  |
| [HUD_QA_CHECKLIST.md](docs/testing/HUD_QA_CHECKLIST.md)                                                                   | UI/UX manual QA checklist      |
| [GOLDEN_REPLAYS.md](docs/testing/GOLDEN_REPLAYS.md)                                                                       | Golden replay test system      |

### /docs/runbooks/

Operational runbooks for production incidents.

| Document                            | Purpose                   |
| ----------------------------------- | ------------------------- |
| [INDEX.md](docs/runbooks/INDEX.md)  | Runbook index             |
| [DEPLOYMENT\_\*.md](docs/runbooks/) | Deployment procedures     |
| [AI\_\*.md](docs/runbooks/)         | AI service operations     |
| [DATABASE\_\*.md](docs/runbooks/)   | Database operations       |
| [WEBSOCKET\_\*.md](docs/runbooks/)  | WebSocket troubleshooting |
| [GAME\_\*.md](docs/runbooks/)       | Game health monitoring    |

### /docs/incidents/

Incident response and post-mortems.

| Document                                                                                            | Purpose                      |
| --------------------------------------------------------------------------------------------------- | ---------------------------- |
| [INDEX.md](docs/incidents/INDEX.md)                                                                 | Incident documentation index |
| [TRIAGE_GUIDE.md](docs/incidents/TRIAGE_GUIDE.md)                                                   | Incident triage procedures   |
| [POST_MORTEM_TEMPLATE.md](docs/incidents/POST_MORTEM_TEMPLATE.md)                                   | Post-mortem template         |
| [INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md](docs/incidents/INCIDENT_TERRITORY_MUTATOR_DIVERGENCE.md) | Historical incident          |

### /docs/operations/

Operational configuration and infrastructure documentation.

| Document                                                             | Purpose                     |
| -------------------------------------------------------------------- | --------------------------- |
| [ALERTING_THRESHOLDS.md](docs/operations/ALERTING_THRESHOLDS.md)     | Monitoring alert thresholds |
| [ENVIRONMENT_VARIABLES.md](docs/operations/ENVIRONMENT_VARIABLES.md) | Environment configuration   |
| [OPERATIONS_DB.md](docs/operations/OPERATIONS_DB.md)                 | Database operations         |
| [SECRETS_MANAGEMENT.md](docs/operations/SECRETS_MANAGEMENT.md)       | Secrets and credentials     |

### /docs/security/

Security documentation and threat modeling.

| Document                                                                         | Purpose            |
| -------------------------------------------------------------------------------- | ------------------ |
| [DATA_LIFECYCLE_AND_PRIVACY.md](docs/security/DATA_LIFECYCLE_AND_PRIVACY.md)     | Data handling/GDPR |
| [SECURITY_THREAT_MODEL.md](docs/security/SECURITY_THREAT_MODEL.md)               | Security analysis  |
| [SUPPLY_CHAIN_AND_CI_SECURITY.md](docs/security/SUPPLY_CHAIN_AND_CI_SECURITY.md) | CI/CD security     |

### /docs/planning/

Active planning and roadmap documents.

| Document                                                                                       | Purpose                 |
| ---------------------------------------------------------------------------------------------- | ----------------------- |
| [DEPLOYMENT_REQUIREMENTS.md](docs/planning/DEPLOYMENT_REQUIREMENTS.md)                         | Production requirements |
| [ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md](docs/planning/ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md) | Parity research roadmap |
| [WAVE_PLAN_2025_12.md](docs/planning/WAVE_PLAN_2025_12.md)                                     | December 2025 wave plan |

### /docs/ (Reference Docs)

Reference documentation kept at docs/ root.

| Document                                                                                                                  | Purpose                                               |
| ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [ACCESSIBILITY.md](docs/ACCESSIBILITY.md)                                                                                 | Accessibility guide (keyboard, screen reader, visual) |
| [GAME_COMPARISON_ANALYSIS.md](docs/GAME_COMPARISON_ANALYSIS.md)                                                           | Game comparison studies                               |
| [UX_RULES_COPY_SPEC.md](docs/UX_RULES_COPY_SPEC.md)                                                                       | UX copy for rules display                             |
| [UX_RULES_TELEMETRY_SPEC.md](docs/UX_RULES_TELEMETRY_SPEC.md)                                                             | Rules UX telemetry schema and hotspot metrics         |
| [UX_RULES_WEIRD_STATES_SPEC.md](docs/UX_RULES_WEIRD_STATES_SPEC.md)                                                       | Weird-state rules UX reason codes and copy mapping    |
| [UX_RULES_TEACHING_SCENARIOS.md](docs/UX_RULES_TEACHING_SCENARIOS.md)                                                     | Scenario-driven teaching flows for complex mechanics  |
| [UX_RULES_IMPROVEMENT_LOOP.md](docs/UX_RULES_IMPROVEMENT_LOOP.md)                                                         | Telemetry-driven rules UX improvement process         |
| [archive/plans/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md](docs/archive/plans/GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md) | GameReplayDB ↔ /sandbox replay integration            |

### /docs/supplementary/

Extended analysis and edge case documentation.

| Document                                                                              | Purpose                        |
| ------------------------------------------------------------------------------------- | ------------------------------ |
| [RULES_CONSISTENCY_EDGE_CASES.md](docs/supplementary/RULES_CONSISTENCY_EDGE_CASES.md) | Edge case handling             |
| [RULES_RULESET_CLARIFICATIONS.md](docs/supplementary/RULES_RULESET_CLARIFICATIONS.md) | Ambiguous rules clarifications |
| [RULES_TERMINATION_ANALYSIS.md](docs/supplementary/RULES_TERMINATION_ANALYSIS.md)     | Game termination analysis      |
| [RULES_DOCS_UX_AUDIT.md](docs/supplementary/RULES_DOCS_UX_AUDIT.md)                   | Documentation UX review        |
| [AI_IMPROVEMENT_BACKLOG.md](docs/supplementary/AI_IMPROVEMENT_BACKLOG.md)             | AI improvement ideas           |

---

## /docs/archive/

Historical documents preserved for reference. These documents record completed work and are no longer actively maintained.

### /docs/archive/assessments/

Development pass assessment reports (Pass 8-22, P18 detailed reports).

- PASS8_ASSESSMENT_REPORT.md through PASS22_ASSESSMENT_REPORT.md
- P18.\_\_\_.md detailed pass 18 documents

### /docs/archive/plans/

Completed planning documents and remediation reports.

- ARCHITECTURE_ANALYSIS.md
- ARCHITECTURE_ASSESSMENT.md
- ARCHITECTURE_REMEDIATION_PLAN.md
- DOCUMENTATION_AUDIT_REPORT.md
- LEGACY_CODE_DEPRECATION_REPORT.md
- And other completed planning documents

### Replay, Parity, and DB Health Tooling

Key docs and tools for TS↔Python parity, replay analysis, and replay DB health.

| Artifact                                                                                                     | Purpose                                                                                                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| [docs/planning/ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md](docs/planning/ENGINE_TOOLING_PARITY_RESEARCH_PLAN.md) | Week 1–3 engine/tooling/parity research: parity surfaces, GameReplayDB schema v5, DB health, and replay tooling.                                                                                                 |
| [ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md](ai-service/docs/GAME_REPLAY_DATABASE_SPEC.md)                 | GameReplayDB schema and API, including `metadata_json` and recording helpers.                                                                                                                                    |
| [docs/testing/TEST_CATEGORIES.md](docs/testing/TEST_CATEGORIES.md)                                           | Test suite categories, including parity and replay-related suites.                                                                                                                                               |
| `ai-service/scripts/check_ts_python_replay_parity.py`                                                        | TS↔Python replay parity checker for recorded games; supports emitting divergence fixtures (`--emit-fixtures-dir`) and rich TS/Python state bundles (`--emit-state-bundles-dir`) for the first semantic mismatch. |
| `ai-service/scripts/diff_state_bundle.py`                                                                    | Offline inspector for a single `.state_bundle.json`: reconstructs Python/TS states at a chosen `ts_k` and prints a concise structural diff (players, stacks, collapsed).                                         |
| `ai-service/scripts/cleanup_useless_replay_dbs.py`                                                           | Replay DB health/cleanup script; emits JSON health summaries with `--summary-json`.                                                                                                                              |
| `ai-service/tests/parity/test_differential_replay.py`                                                        | Differential replay tests, including optional golden-game strict parity via env configuration.                                                                                                                   |

---

## AI Service Documentation

| Document                                                                               | Purpose                                                                               |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [ai-service/README.md](ai-service/README.md)                                           | AI service overview                                                                   |
| [ai-service/docs/NEURAL_AI_ARCHITECTURE.md](ai-service/docs/NEURAL_AI_ARCHITECTURE.md) | Neural network architecture                                                           |
| [docs/ai/AI_TRAINING_AND_DATASETS.md](docs/ai/AI_TRAINING_AND_DATASETS.md)             | Training datasets, including canonical `GameRecord` JSONL exports (Dec 2025 updates). |
| [ai-service/docs/GAME_RECORD_SPEC.md](ai-service/docs/GAME_RECORD_SPEC.md)             | GameRecord schema; Phase 1–2 implemented as of Dec 2025 (storage + DB integration).   |

---

## Finding Documentation

### By Topic

- **Getting started?** → [QUICKSTART.md](QUICKSTART.md)
- **Understanding rules?** → [ringrift_simple_human_rules.md](ringrift_simple_human_rules.md)
- **Architecture deep dive?** → [docs/architecture/](docs/architecture/)
- **Deploying to production?** → [docs/runbooks/](docs/runbooks/)
- **Debugging issues?** → [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [docs/incidents/](docs/incidents/)

### By Audience

- **New developers:** README → QUICKSTART → CONTRIBUTING
- **Rules/Game designers:** ringrift\_\*.md → RULES_CANONICAL_SPEC
- **AI/ML engineers:** AI_ARCHITECTURE → docs/ai/
- **Operators:** docs/runbooks/ → docs/operations/ALERTING_THRESHOLDS
