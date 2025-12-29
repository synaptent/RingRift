# RingRift Documentation Index

> **Last Updated:** 2025-12-29
> **Total Docs:** ~180 active + ~65 archived

This index provides navigation and status tracking for all project documentation.

---

## Quick Start

| Document          | Purpose                 | Status |
| ----------------- | ----------------------- | ------ |
| `README.md`       | Project overview        | Active |
| `QUICKSTART.md`   | Setup guide             | Active |
| `SECURITY.md`     | Security policy         | Active |
| `CONTRIBUTING.md` | Contribution guidelines | Active |
| `AGENTS.md`       | AI agent expectations   | Active |

---

## Rules & Game Design

### Canonical Sources (SSoT)

| Document                  | Purpose                                    | Status   |
| ------------------------- | ------------------------------------------ | -------- |
| `RULES_CANONICAL_SPEC.md` | Formal rules specification (RR-CANON-RXXX) | **SSoT** |
| `rules/COMPLETE_RULES.md` | Full rulebook with examples                | Active   |
| `rules/COMPACT_RULES.md`  | Quick reference rules                      | Active   |
| `rules/HUMAN_RULES.md`    | Simplified human-readable rules            | Active   |

### Rules Analysis & Audits

| Document                                                | Purpose                             | Status    |
| ------------------------------------------------------- | ----------------------------------- | --------- |
| `docs/rules/RULES_DOCS_CONSISTENCY_AUDIT_2025_12_12.md` | Consistency audit                   | Active    |
| `docs/rules/RULES_SSOT_MAP.md`                          | SSoT hierarchy guide                | Active    |
| `docs/rules/CURRENT_RULES_STATE.md`                     | Current rules implementation status | Active    |
| `docs/supplementary/rules_analysis/*`                   | Deep dives (LPS, ring count)        | Reference |

---

## Architecture & Design

| Document                                                | Purpose                  | Status |
| ------------------------------------------------------- | ------------------------ | ------ |
| `docs/architecture/RULES_ENGINE_ARCHITECTURE.md`        | Engine design            | Active |
| `docs/architecture/DOMAIN_AGGREGATE_DESIGN.md`          | Aggregate patterns       | Active |
| `docs/architecture/STATE_MACHINES.md`                   | FSM design               | Active |
| `docs/architecture/CANONICAL_ENGINE_API.md`             | Engine API reference     | Active |
| `docs/architecture/PHASE_ORCHESTRATION_ARCHITECTURE.md` | Phase/turn orchestration | Active |
| `docs/architecture/WEBSOCKET_API.md`                    | WebSocket contract       | Active |

---

## AI Service & Training

### Primary Docs

| Document                                           | Purpose                   | Status |
| -------------------------------------------------- | ------------------------- | ------ |
| `ai-service/README.md`                             | AI service overview       | Active |
| `ai-service/docs/README.md`                        | AI service doc hub + SSoT | Active |
| `ai-service/docs/training/TRAINING_FEATURES.md`    | Training feature guide    | Active |
| `ai-service/docs/CONSOLIDATION_ROADMAP.md`         | Consolidation progress    | Active |
| `ai-service/docs/roadmaps/GPU_PIPELINE_ROADMAP.md` | GPU training pipeline     | Active |
| `ai-service/docs/training/DISTRIBUTED_SELFPLAY.md` | Distributed training      | Active |

### Human Calibration

| Document                                        | Purpose                   | Status |
| ----------------------------------------------- | ------------------------- | ------ |
| `docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md`         | Human calibration process | Active |
| `docs/ai/AI_HUMAN_CALIBRATION_STUDY_DESIGN.md`  | Study methodology         | Active |
| `docs/ai/AI_DIFFICULTY_CALIBRATION_ANALYSIS.md` | Calibration results       | Active |

### Operations

| Document                                  | Purpose               | Status |
| ----------------------------------------- | --------------------- | ------ |
| `docs/ai/AI_CALIBRATION_RUNBOOK.md`       | Calibration runbook   | Active |
| `docs/ai/AI_LADDER_PRODUCTION_RUNBOOK.md` | Production ladder ops | Active |
| `docs/ai/CLUSTER_NODE_CONFIGURATION.md`   | Cluster setup         | Active |

---

## UX & Teaching

| Document                                     | Purpose                       | Status  |
| -------------------------------------------- | ----------------------------- | ------- |
| `docs/ux/RULES_QUICK_REFERENCE_DIAGRAMS.md`  | ASCII rule diagrams           | **New** |
| `docs/ux/UX_RULES_CONCEPTS_INDEX.md`         | Rules concepts navigation map | Active  |
| `docs/ux/UX_RULES_TEACHING_SCENARIOS.md`     | Teaching scenario definitions | Active  |
| `docs/ux/UX_RULES_EXPLANATION_MODEL_SPEC.md` | Game-end explanation model    | Active  |
| `docs/ux/UX_RULES_COPY_SPEC.md`              | UI copy specifications        | Active  |
| `docs/ux/UX_RULES_WEIRD_STATES_SPEC.md`      | Edge case UX handling         | Active  |
| `docs/ux/UX_RULES_TELEMETRY_SPEC.md`         | UX telemetry design           | Active  |
| `docs/getting-started/AUDIENCE.md`           | Target audience positioning   | Active  |

---

## Operations & Runbooks

### Production Operations

| Document                                            | Purpose                     | Status |
| --------------------------------------------------- | --------------------------- | ------ |
| `docs/production/PRODUCTION_RUNBOOK.md`             | Production operations guide | Active |
| `docs/production/PRODUCTION_READINESS_CHECKLIST.md` | Go-live checklist           | Active |
| `docs/operations/CLUSTER_OPERATIONS.md`             | Cluster management          | Active |
| `docs/operations/STAGING_ENVIRONMENT.md`            | Staging setup               | Active |
| `docs/operations/ENVIRONMENT_VARIABLES.md`          | Environment configuration   | Active |
| `docs/operations/ENVIRONMENT_VARIABLES_INTERNAL.md` | Internal env flags appendix | Active |

### Incident Response

| Document                         | Purpose         | Status |
| -------------------------------- | --------------- | ------ |
| `docs/runbooks/INDEX.md`         | Runbook index   | Active |
| `docs/incidents/INDEX.md`        | Incident index  | Active |
| `docs/incidents/TRIAGE_GUIDE.md` | Incident triage | Active |

### Specific Runbooks

See `docs/runbooks/` for 25+ specific runbooks covering:

- AI service issues (`AI_*.md`)
- Database issues (`DATABASE_*.md`)
- Deployment (`DEPLOYMENT_*.md`)
- WebSocket/Redis (`WEBSOCKET_*.md`, `REDIS_*.md`)

---

## Testing

| Document                               | Purpose                | Status |
| -------------------------------------- | ---------------------- | ------ |
| `tests/README.md`                      | Test suite overview    | Active |
| `docs/testing/TEST_CATEGORIES.md`      | Test categorization    | Active |
| `docs/testing/LOAD_TEST_BASELINE.md`   | Load test baselines    | Active |
| `docs/testing/GOLDEN_REPLAYS.md`       | Golden replay testing  | Active |
| `docs/testing/SKIPPED_TESTS_TRIAGE.md` | Skipped test triage    | Active |
| `docs/testing/CLIENT_TEST_PLAN.md`     | Frontend coverage plan | Active |

---

## Planning & Status

| Document                                              | Purpose                   | Status     |
| ----------------------------------------------------- | ------------------------- | ---------- |
| `docs/COMPREHENSIVE_ACTION_PLAN_2025_12_17.md`        | Current action plan       | **Active** |
| `docs/archive/historical/CURRENT_STATE_ASSESSMENT.md` | Project state summary     | Reference  |
| `docs/planning/NN_SELFPLAY_TRAINING_LOOP_PLAN.md`     | NN self-play loop plan    | Active     |
| `docs/planning/SELFPLAY_LOOP_CLOSURE_PLAN.md`         | Self-play bottleneck plan | Active     |
| `PROJECT_GOALS.md`                                    | High-level goals          | Active     |
| `TODO.md`                                             | Active TODO list          | Active     |
| `KNOWN_ISSUES.md`                                     | Known issues tracker      | Active     |
| `docs/production/RELEASE_NOTES_v0.1.0-beta.md`        | Release notes draft       | Active     |

---

## Security

| Document                                        | Purpose          | Status |
| ----------------------------------------------- | ---------------- | ------ |
| `SECURITY.md`                                   | Security policy  | Active |
| `docs/security/SECURITY_THREAT_MODEL.md`        | Threat model     | Active |
| `docs/security/DATA_LIFECYCLE_AND_PRIVACY.md`   | Data privacy     | Active |
| `docs/security/SUPPLY_CHAIN_AND_CI_SECURITY.md` | CI/CD security   | Active |
| `docs/operations/SECRETS_MANAGEMENT.md`         | Secrets handling | Active |

---

## Archived Documentation

Superseded or historical documents are in `docs/archive/`:

| Directory                                   | Contents                               |
| ------------------------------------------- | -------------------------------------- |
| `docs/archive/assessments/`                 | Historical pass assessments (PASS1-22) |
| `docs/archive/plans/`                       | Old planning documents                 |
| `docs/archive/historical/`                  | Historical snapshots and assessments   |
| `docs/archive/historical/ROADMAP_2025Q1.md` | Historical roadmap snapshot            |

---

## Documentation Maintenance

### Status Definitions

- **SSoT**: Single Source of Truth - authoritative for its domain
- **Active**: Current and maintained
- **New**: Recently created (< 7 days)
- **Reference**: Historical context, not actively updated
- **Deprecated**: Scheduled for archival

### Freshness Guidelines

- Active docs should be reviewed every 90 days
- Docs older than 90 days without updates should be reviewed for archival
- Archive superseded docs with a deprecation notice at the top

### Related

- See `DOCUMENTATION_INDEX.md` at repo root for the full catalog
- See `docs/rules/SSOT_BANNER_GUIDE.md` for adding SSoT banners
