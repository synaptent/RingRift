# RingRift Documentation Index

> **Last Updated:** 2025-12-27
> **Organization:** Core docs in root, organized subdirectories in `/docs/` and `/ai-service/docs/`

This index catalogs all project documentation organized by topic and location. For a lightweight landing page, see `docs/INDEX.md`.

---

## Quick Start

| Document                                   | Purpose                            |
| ------------------------------------------ | ---------------------------------- |
| [README.md](README.md)                     | Project overview, features, status |
| [QUICKSTART.md](QUICKSTART.md)             | Local development setup            |
| [CONTRIBUTING.md](CONTRIBUTING.md)         | Contribution guidelines            |
| [SECURITY.md](SECURITY.md)                 | Security policy and implementation |
| [TERMS_OF_SERVICE.md](TERMS_OF_SERVICE.md) | Terms of Service                   |
| [PRIVACY_POLICY.md](PRIVACY_POLICY.md)     | Privacy Policy                     |
| [TODO.md](TODO.md)                         | Active task tracker                |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md)         | Current bugs and gaps              |

### Getting Started (New Players)

| Document                                                                | Purpose                 |
| ----------------------------------------------------------------------- | ----------------------- |
| [Learn in 2 Minutes](docs/getting-started/LEARN_IN_2_MINUTES.md)        | Ultra-quick rules intro |
| [Guided First Game](docs/getting-started/GUIDED_FIRST_GAME_TUTORIAL.md) | Interactive tutorial    |
| [Audience](docs/getting-started/AUDIENCE.md)                            | Who RingRift is for     |

---

## Core Documentation (Root)

### Project Status & Planning

- [PROJECT_GOALS.md](PROJECT_GOALS.md) - Canonical project goals, v1.0 success criteria, and scope boundaries
- [STRATEGIC_ROADMAP.md](docs/planning/STRATEGIC_ROADMAP.md) - Phased roadmap & SLOs
- [NEXT_AREAS_EXECUTION_PLAN.md](docs/planning/NEXT_AREAS_EXECUTION_PLAN.md) - Sequenced execution lanes for next work

### Rules & Game Design

- [RULES_CANONICAL_SPEC.md](RULES_CANONICAL_SPEC.md) - Authoritative rules specification
- [docs/rules/COMPLETE_RULES.md](docs/rules/COMPLETE_RULES.md) - Full rulebook
- [docs/rules/HUMAN_RULES.md](docs/rules/HUMAN_RULES.md) - Human-readable rules
- [docs/rules/COMPACT_RULES.md](docs/rules/COMPACT_RULES.md) - Implementation-focused summary

### Architecture

- [RULES_ENGINE_ARCHITECTURE.md](docs/architecture/RULES_ENGINE_ARCHITECTURE.md) - Rules engine design
- [AI_ARCHITECTURE.md](docs/architecture/AI_ARCHITECTURE.md) - AI service architecture

---

## /docs/ Directory Structure

### /docs/getting-started/

New player onboarding and tutorials.

| Document                                                                            | Purpose                 |
| ----------------------------------------------------------------------------------- | ----------------------- |
| [LEARN_IN_2_MINUTES.md](docs/getting-started/LEARN_IN_2_MINUTES.md)                 | Ultra-quick rules intro |
| [GUIDED_FIRST_GAME_TUTORIAL.md](docs/getting-started/GUIDED_FIRST_GAME_TUTORIAL.md) | Interactive tutorial    |
| [AUDIENCE.md](docs/getting-started/AUDIENCE.md)                                     | Target audience         |

### /docs/production/

Production deployment and release management.

| Document                                                                               | Purpose                 |
| -------------------------------------------------------------------------------------- | ----------------------- |
| [PRODUCTION_READINESS_CHECKLIST.md](docs/production/PRODUCTION_READINESS_CHECKLIST.md) | Pre-launch verification |
| [PRODUCTION_RUNBOOK.md](docs/production/PRODUCTION_RUNBOOK.md)                         | Day-to-day operations   |
| [RELEASE_INSTRUCTIONS.md](docs/production/RELEASE_INSTRUCTIONS.md)                     | Release process         |
| [RELEASE_NOTES_v0.1.0-beta.md](docs/production/RELEASE_NOTES_v0.1.0-beta.md)           | Current release notes   |

### /docs/architecture/

Engine and system architecture documentation.

| Document                                                                                                   | Purpose                                  |
| ---------------------------------------------------------------------------------------------------------- | ---------------------------------------- |
| [API_REFERENCE.md](docs/architecture/API_REFERENCE.md)                                                     | REST API documentation                   |
| [CANONICAL_ENGINE_API.md](docs/architecture/CANONICAL_ENGINE_API.md)                                       | Public engine API specification          |
| [DOMAIN_AGGREGATE_DESIGN.md](docs/architecture/DOMAIN_AGGREGATE_DESIGN.md)                                 | Domain model and aggregates              |
| [MODULE_RESPONSIBILITIES.md](docs/architecture/MODULE_RESPONSIBILITIES.md)                                 | Module catalog                           |
| [PLAYER_MOVE_TRANSPORT_DECISION.md](docs/architecture/PLAYER_MOVE_TRANSPORT_DECISION.md)                   | WebSocket vs HTTP move transport         |
| [STATE_MACHINES.md](docs/architecture/STATE_MACHINES.md)                                                   | Session/AI/choice state machines         |
| [TOPOLOGY_MODES.md](docs/architecture/TOPOLOGY_MODES.md)                                                   | Board topology design                    |
| [SYNC_ARCHITECTURE.md](docs/architecture/SYNC_ARCHITECTURE.md)                                             | Data/model sync architecture             |
| [ORCHESTRATOR_ROLLOUT_PLAN.md](docs/architecture/ORCHESTRATOR_ROLLOUT_PLAN.md)                             | Orchestrator migration plan              |
| [FSM_MIGRATION_STATUS_2025_12.md](docs/architecture/FSM_MIGRATION_STATUS_2025_12.md)                       | Migration status snapshot                |
| [SHARED_ENGINE_CONSOLIDATION_PLAN.md](docs/architecture/SHARED_ENGINE_CONSOLIDATION_PLAN.md)               | Engine consolidation design              |
| [CLIENT_SANDBOX_ENGINE_REFACTOR_PROPOSAL.md](docs/architecture/CLIENT_SANDBOX_ENGINE_REFACTOR_PROPOSAL.md) | Deferred refactor proposal (not started) |
| [NEURAL_NET_INTEGRATION_DESIGN.md](docs/architecture/NEURAL_NET_INTEGRATION_DESIGN.md)                     | Neural network integration design        |
| [VICTORY_REFACTORING_PLAN.md](docs/architecture/VICTORY_REFACTORING_PLAN.md)                               | Victory detection consolidation plan     |
| [TEST_HYGIENE_NOTES.md](docs/architecture/TEST_HYGIENE_NOTES.md)                                           | Test suite hygiene and cleanup notes     |

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
| [HEX_PARITY_AUDIT.md](docs/rules/HEX_PARITY_AUDIT.md)                               | Hex board parity audit              |

### /docs/ai/

AI service and training documentation.

| Document                                                                                         | Purpose                            |
| ------------------------------------------------------------------------------------------------ | ---------------------------------- |
| [AI_DIFFICULTY_ANALYSIS.md](docs/ai/AI_DIFFICULTY_ANALYSIS.md)                                   | Difficulty level analysis          |
| [AI_LARGE_BOARD_PERFORMANCE_ASSESSMENT.md](docs/ai/AI_LARGE_BOARD_PERFORMANCE_ASSESSMENT.md)     | Large board performance            |
| [AI_TRAINING_AND_DATASETS.md](docs/ai/AI_TRAINING_AND_DATASETS.md)                               | Training pipelines and datasets    |
| [AI_TRAINING_ASSESSMENT_FINAL.md](docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md)                       | Training assessment                |
| [AI_TRAINING_PREPARATION_GUIDE.md](docs/ai/AI_TRAINING_PREPARATION_GUIDE.md)                     | Training preparation               |
| [AI_TRAINING_PLAN.md](ai-service/docs/roadmaps/AI_TRAINING_PLAN.md)                              | Comprehensive AI training pipeline |
| [AI_PIPELINE_PARITY_FIXES.md](docs/ai/AI_PIPELINE_PARITY_FIXES.md)                               | TS↔Python parity fixes (Dec 2025)  |
| [AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md](docs/ai/AI_TIER_TRAINING_AND_PROMOTION_PIPELINE.md) | Tier training and promotion design |
| [AI_CALIBRATION_RUNBOOK.md](docs/ai/AI_CALIBRATION_RUNBOOK.md)                                   | Difficulty calibration procedures  |
| [AI_LADDER_PRODUCTION_RUNBOOK.md](docs/ai/AI_LADDER_PRODUCTION_RUNBOOK.md)                       | Production run quick-start guide   |

### /docs/testing/

Test infrastructure and QA documentation.

| Document                                                                                                                  | Purpose                                                              |
| ------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| [TEST_CATEGORIES.md](docs/testing/TEST_CATEGORIES.md)                                                                     | CI vs diagnostic test types                                          |
| [TEST_INFRASTRUCTURE.md](docs/testing/TEST_INFRASTRUCTURE.md)                                                             | Test framework setup                                                 |
| [STRICT_INVARIANT_SOAKS.md](docs/testing/STRICT_INVARIANT_SOAKS.md)                                                       | Long-running invariant tests                                         |
| [E2E_AUTH_AND_GAME_FLOW_TEST_STABILIZATION_SUMMARY.md](docs/testing/E2E_AUTH_AND_GAME_FLOW_TEST_STABILIZATION_SUMMARY.md) | E2E test stabilization                                               |
| [GO_NO_GO_CHECKLIST.md](docs/testing/GO_NO_GO_CHECKLIST.md)                                                               | Production readiness checklist                                       |
| [LOAD_TEST_BASELINE.md](docs/testing/LOAD_TEST_BASELINE.md)                                                               | Load test baseline targets                                           |
| [LOAD_TEST_BASELINE_REPORT.md](docs/testing/LOAD_TEST_BASELINE_REPORT.md)                                                 | Load test results report                                             |
| [BASELINE_CAPACITY.md](docs/testing/BASELINE_CAPACITY.md)                                                                 | Current/target/AI-heavy capacity runs and how to execute/record them |
| [SKIPPED_TESTS_TRIAGE.md](docs/testing/SKIPPED_TESTS_TRIAGE.md)                                                           | Skipped tests triage and rationale                                   |
| [WEAK_ASSERTION_AUDIT.md](docs/testing/WEAK_ASSERTION_AUDIT.md)                                                           | Weak assertion audit and tracking                                    |
| [LOAD_TEST_WEBSOCKET_MOVE_STRATEGY.md](docs/testing/LOAD_TEST_WEBSOCKET_MOVE_STRATEGY.md)                                 | WebSocket load testing design                                        |
| [HUD_QA_CHECKLIST.md](docs/testing/HUD_QA_CHECKLIST.md)                                                                   | UI/UX manual QA checklist                                            |
| [GOLDEN_REPLAYS.md](docs/testing/GOLDEN_REPLAYS.md)                                                                       | Golden replay test system                                            |

### /docs/runbooks/

Operational runbooks for production incidents.

| Document                                                                       | Purpose                   |
| ------------------------------------------------------------------------------ | ------------------------- |
| [INDEX.md](docs/runbooks/INDEX.md)                                             | Runbook index             |
| [PARITY_VERIFICATION_RUNBOOK.md](docs/runbooks/PARITY_VERIFICATION_RUNBOOK.md) | TS↔Python parity checks   |
| [DEPLOYMENT\_\*.md](docs/runbooks/)                                            | Deployment procedures     |
| [AI\_\*.md](docs/runbooks/)                                                    | AI service operations     |
| [DATABASE\_\*.md](docs/runbooks/)                                              | Database operations       |
| [WEBSOCKET\_\*.md](docs/runbooks/)                                             | WebSocket troubleshooting |
| [GAME\_\*.md](docs/runbooks/)                                                  | Game health monitoring    |

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
| [STAGING_ENVIRONMENT.md](docs/operations/STAGING_ENVIRONMENT.md)     | Staging environment setup   |
| [SLO_VERIFICATION.md](docs/operations/SLO_VERIFICATION.md)           | SLO verification framework  |

### Cluster Monitoring Infrastructure

Production monitoring scripts for the distributed training cluster. See [ai-service/scripts/monitoring/README.md](ai-service/scripts/monitoring/README.md) for full documentation.

| Script/Document                                                                                | Purpose                                       |
| ---------------------------------------------------------------------------------------------- | --------------------------------------------- |
| [cluster_health_check.sh](ai-service/scripts/monitoring/cluster_health_check.sh)               | Node status, quorum, CPU/memory, disk alerts  |
| [selfplay_throughput_monitor.sh](ai-service/scripts/monitoring/selfplay_throughput_monitor.sh) | Game generation throughput monitoring         |
| [training_pipeline_monitor.sh](ai-service/scripts/monitoring/training_pipeline_monitor.sh)     | NNUE training, Elo tournaments, model gating  |
| [setup_cloudwatch.sh](ai-service/scripts/monitoring/setup_cloudwatch.sh)                       | CloudWatch alarms, dashboard, SNS setup       |
| [README.md](ai-service/scripts/monitoring/README.md)                                           | Monitoring quick start and environment config |

**Deployed infrastructure:**

- CloudWatch dashboard: `RingRift-Cluster` (us-east-1)
- SNS topic: `ringrift-alerts` with email subscription
- Slack webhook integration for real-time alerts
- Cron jobs on `aws-staging` (5-min health, 2-min throughput, 10-min training)

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
| [WAVE_2025_12.md](docs/planning/WAVE_2025_12.md)                                               | December 2025 wave plan |

### /docs/ux/

UX and rules teaching documentation.

| Document                                                                         | Purpose                                              |
| -------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [UX_RULES_CONCEPTS_INDEX.md](docs/ux/UX_RULES_CONCEPTS_INDEX.md)                 | High-risk rules concepts index                       |
| [UX_RULES_COPY_SPEC.md](docs/ux/UX_RULES_COPY_SPEC.md)                           | UX copy for rules display                            |
| [UX_RULES_EXPLANATION_MODEL_SPEC.md](docs/ux/UX_RULES_EXPLANATION_MODEL_SPEC.md) | Game-end explanation model                           |
| [UX_RULES_IMPROVEMENT_LOOP.md](docs/ux/UX_RULES_IMPROVEMENT_LOOP.md)             | Telemetry-driven rules UX improvement process        |
| [UX_RULES_TEACHING_GAP_ANALYSIS.md](docs/ux/UX_RULES_TEACHING_GAP_ANALYSIS.md)   | Teaching scenario coverage audit                     |
| [UX_RULES_TEACHING_SCENARIOS.md](docs/ux/UX_RULES_TEACHING_SCENARIOS.md)         | Scenario-driven teaching flows for complex mechanics |
| [UX_RULES_TELEMETRY_SPEC.md](docs/ux/UX_RULES_TELEMETRY_SPEC.md)                 | Rules UX telemetry schema and hotspot metrics        |
| [UX_RULES_WEIRD_STATES_SPEC.md](docs/ux/UX_RULES_WEIRD_STATES_SPEC.md)           | Weird-state rules UX reason codes and copy mapping   |

### /docs/ (Reference Docs)

Reference documentation kept at docs/ root.

| Document                                                                                                                  | Purpose                                               |
| ------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [ACCESSIBILITY.md](docs/ACCESSIBILITY.md)                                                                                 | Accessibility guide (keyboard, screen reader, visual) |
| [INDEX.md](docs/INDEX.md)                                                                                                 | Quick navigation index                                |
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
| [GAME_COMPARISON_ANALYSIS.md](docs/supplementary/GAME_COMPARISON_ANALYSIS.md)         | Game comparison studies        |

**Rules analysis (`docs/supplementary/rules_analysis/`)**

| Document                                                                                                                         | Purpose                                                         |
| -------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------- |
| [rules_analysis_lps_inconsistencies.md](docs/supplementary/rules_analysis/rules_analysis_lps_inconsistencies.md)                 | LPS two-round threshold audit + remaining UX copy gaps          |
| [rules_analysis_lps_fe.md](docs/supplementary/rules_analysis/rules_analysis_lps_fe.md)                                           | Analysis of forced elimination not counting as an LPS real move |
| [rules_analysis_ring_count_inconsistencies.md](docs/supplementary/rules_analysis/rules_analysis_ring_count_inconsistencies.md)   | Ring count mismatches across specs, docs, and engines           |
| [rules_analysis_ring_count_increase.md](docs/supplementary/rules_analysis/rules_analysis_ring_count_increase.md)                 | Historical proposal to raise 19×19/hex ring supplies            |
| [rules_analysis_ring_count_remediation_plan.md](docs/supplementary/rules_analysis/rules_analysis_ring_count_remediation_plan.md) | Remediation steps for ring count inconsistencies                |

---

## /docs/ Active Assessments

Current codebase assessments and improvement plans.

| Document                                                                    | Purpose                                |
| --------------------------------------------------------------------------- | -------------------------------------- |
| [ARCHITECTURAL_IMPROVEMENT_PLAN.md](docs/ARCHITECTURAL_IMPROVEMENT_PLAN.md) | Refactoring opportunities and progress |

### Archived Assessments (Historical)

These dated documents have been moved to `/docs/archive/historical/`:

- CODEBASE_REVIEW_2025_12_11.md - First-principles codebase review
- CODE_QUALITY_AUDIT_2025_12_11.md - Code quality audit
- NEXT_STEPS_2025_12_11.md - Session progress notes

See [docs/production/](docs/production/) for current production readiness documentation.

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
| [ai-service/docs/specs/GAME_REPLAY_DATABASE_SPEC.md](ai-service/docs/specs/GAME_REPLAY_DATABASE_SPEC.md)     | GameReplayDB schema and API, including `metadata_json` and recording helpers.                                                                                                                                    |
| [docs/testing/TEST_CATEGORIES.md](docs/testing/TEST_CATEGORIES.md)                                           | Test suite categories, including parity and replay-related suites.                                                                                                                                               |
| `ai-service/scripts/check_ts_python_replay_parity.py`                                                        | TS↔Python replay parity checker for recorded games; supports emitting divergence fixtures (`--emit-fixtures-dir`) and rich TS/Python state bundles (`--emit-state-bundles-dir`) for the first semantic mismatch. |
| `ai-service/scripts/diff_state_bundle.py`                                                                    | Offline inspector for a single `.state_bundle.json`: reconstructs Python/TS states at a chosen `ts_k` and prints a concise structural diff (players, stacks, collapsed).                                         |
| `ai-service/scripts/cleanup_useless_replay_dbs.py`                                                           | Replay DB health/cleanup script; emits JSON health summaries with `--summary-json`.                                                                                                                              |
| `ai-service/tests/parity/test_differential_replay.py`                                                        | Differential replay tests, including optional golden-game strict parity via env configuration.                                                                                                                   |

---

## AI Service Documentation

### Core AI Service

| Document                                                                                                         | Purpose                                               |
| ---------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------- |
| [ai-service/README.md](ai-service/README.md)                                                                     | AI service overview, API endpoints, difficulty ladder |
| [ai-service/docs/architecture/NEURAL_AI_ARCHITECTURE.md](ai-service/docs/architecture/NEURAL_AI_ARCHITECTURE.md) | Neural network architecture (RingRiftCNN)             |
| [ai-service/docs/architecture/MPS_ARCHITECTURE.md](ai-service/docs/architecture/MPS_ARCHITECTURE.md)             | Apple Silicon MPS-compatible architecture             |

### Training Pipeline (Active - Dec 2025)

| Document                                                                                                             | Purpose                                              |
| -------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------- |
| [ai-service/docs/roadmaps/AI_TRAINING_PLAN.md](ai-service/docs/roadmaps/AI_TRAINING_PLAN.md)                         | **Primary training guide**: CMA-ES, NN, NPZ pipeline |
| [ai-service/docs/infrastructure/ORCHESTRATOR_SELECTION.md](ai-service/docs/infrastructure/ORCHESTRATOR_SELECTION.md) | **Which script to use** - orchestrator decision tree |
| [ai-service/docs/sync_architecture.md](ai-service/docs/sync_architecture.md)                                         | Sync stack overview (SyncFacade, AutoSyncDaemon)     |
| [ai-service/docs/training/UNIFIED_AI_LOOP.md](ai-service/docs/training/UNIFIED_AI_LOOP.md)                           | **Unified AI self-improvement daemon** (canonical)   |
| [ai-service/docs/infrastructure/PIPELINE_ORCHESTRATOR.md](ai-service/docs/infrastructure/PIPELINE_ORCHESTRATOR.md)   | CI/CD pipeline orchestration (deprecated)            |
| [ai-service/docs/training/DISTRIBUTED_SELFPLAY.md](ai-service/docs/training/DISTRIBUTED_SELFPLAY.md)                 | Distributed selfplay across GPU clusters             |
| [docs/ai/AI_TRAINING_AND_DATASETS.md](docs/ai/AI_TRAINING_AND_DATASETS.md)                                           | Training datasets and GameRecord JSONL exports       |

### GPU Infrastructure

| Document                                                                                                                                     | Purpose                                         |
| -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [ai-service/docs/roadmaps/GPU_PIPELINE_ROADMAP.md](ai-service/docs/roadmaps/GPU_PIPELINE_ROADMAP.md)                                         | GPU selfplay pipeline development roadmap       |
| [ai-service/docs/architecture/GPU_ARCHITECTURE_SIMPLIFICATION.md](ai-service/docs/architecture/GPU_ARCHITECTURE_SIMPLIFICATION.md)           | GPU architecture design notes                   |
| [ai-service/docs/infrastructure/GPU_RULES_PARITY_AUDIT.md](ai-service/docs/infrastructure/GPU_RULES_PARITY_AUDIT.md)                         | GPU vs CPU rules parity verification            |
| [ai-service/docs/infrastructure/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md](ai-service/docs/infrastructure/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md) | Cloud training infrastructure (Lambda, Vast.ai) |

### Data Formats & Schemas

| Document                                                                                                 | Purpose                             |
| -------------------------------------------------------------------------------------------------------- | ----------------------------------- |
| [ai-service/docs/specs/GAME_RECORD_SPEC.md](ai-service/docs/specs/GAME_RECORD_SPEC.md)                   | GameRecord schema for JSONL exports |
| [ai-service/docs/specs/GAME_REPLAY_DATABASE_SPEC.md](ai-service/docs/specs/GAME_REPLAY_DATABASE_SPEC.md) | GameReplayDB SQLite schema and API  |
| [ai-service/docs/specs/GAME_NOTATION_SPEC.md](ai-service/docs/specs/GAME_NOTATION_SPEC.md)               | Game notation format specification  |

### Analysis & Status Reports

| Document                                                                                                                                                       | Purpose                         |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| [ai-service/docs/archive/status_reports/TRAINING_PIPELINE_STATUS_2025_12_12.md](ai-service/docs/archive/status_reports/TRAINING_PIPELINE_STATUS_2025_12_12.md) | Training pipeline status update |
| [ai-service/AI_ASSESSMENT_REPORT.md](ai-service/AI_ASSESSMENT_REPORT.md)                                                                                       | AI service technical assessment |
| [ai-service/AI_IMPROVEMENT_PLAN.md](ai-service/AI_IMPROVEMENT_PLAN.md)                                                                                         | AI service improvement roadmap  |

---

## Environments and Infrastructure

RingRift supports multiple deployment and test environments:

- **Local development** – Node.js dev server and optional AI service running on a developer workstation. See [QUICKSTART.md](QUICKSTART.md).
- **Deployment scripts** – Systemd services and Grafana dashboards for production deployment. See [ai-service/deploy/README.md](ai-service/deploy/README.md).
- **Docker-based staging on localhost** – Full stack composed via Docker Compose on a single host (for example `http://localhost:3000` for HTTP and `ws://localhost:3001` for WebSockets), suitable for capacity and SLO runs.
- **Cloud-hosted staging and load-testing environments (optional)** – Operators may provision remote instances on AWS EC2/ECS or another cloud provider to run:
  - The RingRift backend/API and WebSocket endpoints for staging (for example `https://staging.example.com`, `wss://staging.example.com`).
  - One or more load-generator instances (for example k6 workers or equivalent) that execute the same scenarios described in [tests/load/README.md](tests/load/README.md) against a configurable base URL.
  - AI training and self-play infrastructure that uses cloud storage and queues, as outlined in [ai-service/docs/infrastructure/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md](ai-service/docs/infrastructure/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md).

These cloud environments are deployment-specific:

- This repository does **not** assume any particular cloud account ID, region (for example `us-east-1`), VPC topology, or hostname.
- Operators are expected to configure their own regions, networks, TLS certificates, and DNS records, and to inject base URLs and credentials via environment variables, CI/CD secrets, and infrastructure-as-code.

Key reference documents:

- [docs/operations/STAGING_ENVIRONMENT.md](docs/operations/STAGING_ENVIRONMENT.md) – Staging environment topology, Docker-based setup, and how load tests attach to staging.
- [docs/testing/BASELINE_CAPACITY.md](docs/testing/BASELINE_CAPACITY.md) – Baseline capacity scenarios and how to execute and record them.
- [docs/operations/SLO_VERIFICATION.md](docs/operations/SLO_VERIFICATION.md) – SLO verification pipeline that consumes k6 JSON outputs from any environment.
- [ai-service/docs/infrastructure/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md](ai-service/docs/infrastructure/CLOUD_TRAINING_INFRASTRUCTURE_PLAN.md) – Cloud training and distributed self-play infrastructure plan (illustrative, not required for all deployments).

---

## Finding Documentation

### By Topic

- **Getting started?** → [QUICKSTART.md](QUICKSTART.md)
- **Understanding rules?** → [docs/rules/HUMAN_RULES.md](docs/rules/HUMAN_RULES.md)
- **Architecture deep dive?** → [docs/architecture/](docs/architecture/)
- **Deploying to production?** → [docs/runbooks/](docs/runbooks/)
- **Debugging issues?** → [KNOWN_ISSUES.md](KNOWN_ISSUES.md), [docs/incidents/](docs/incidents/)

### By Audience

- **New developers:** README → QUICKSTART → CONTRIBUTING
- **Rules/Game designers:** docs/rules/ → RULES_CANONICAL_SPEC
- **AI/ML engineers:** docs/architecture/AI_ARCHITECTURE → docs/ai/
- **Operators:** docs/runbooks/ → docs/operations/ALERTING_THRESHOLDS
