# Archive Index

> **Purpose:** This directory contains historical documentation that has been completed, superseded, or is no longer actively maintained. These documents are preserved for reference and context.

## Quick Guide

- **Looking for current docs?** See [DOCUMENTATION_INDEX.md](../../DOCUMENTATION_INDEX.md)
- **Need the latest status?** See [CURRENT_STATE_ASSESSMENT.md](historical/CURRENT_STATE_ASSESSMENT.md)
- **Looking for an incident?** See [docs/incidents/INDEX.md](../incidents/INDEX.md)

---

## /docs/archive/assessments/

Development pass assessment reports documenting work completed during iterative development phases.

| Document                                                       | Description                             |
| -------------------------------------------------------------- | --------------------------------------- |
| PASS8_ASSESSMENT_REPORT.md through PASS22_ASSESSMENT_REPORT.md | Sequential development pass reports     |
| PASS20_COMPLETION_SUMMARY.md                                   | Orchestrator migration completion       |
| PASS22_COMPLETION_SUMMARY.md                                   | Final pass completion summary           |
| P18.\* reports                                                 | Detailed Pass 18 sub-task documentation |
| P9_SWAP_RULE_AI_INTEGRATION.md                                 | Swap rule AI integration details        |

---

## /docs/archive/plans/

Completed planning documents and remediation reports.

| Document                                   | Description                      |
| ------------------------------------------ | -------------------------------- |
| ARCHITECTURE_ANALYSIS.md                   | Original architecture analysis   |
| ARCHITECTURE_ASSESSMENT.md                 | Architecture assessment findings |
| ARCHITECTURE_REMEDIATION_PLAN.md           | Remediation plan (completed)     |
| DEPENDENCY_UPGRADE_PLAN.md                 | Dependency upgrade plan          |
| DOCUMENTATION_AUDIT_REPORT.md              | Previous documentation audit     |
| GAME_REPLAY_DB_SANDBOX_INTEGRATION_PLAN.md | GameReplayDB sandbox integration |
| LEGACY_CODE_DEPRECATION_REPORT.md          | Legacy code removal report       |
| LEGACY_PATH_DEPRECATION_PLAN.md            | Legacy path deprecation plan     |
| PASS20-21_DOCUMENTATION_UPDATE_PLAN.md     | Documentation update plan        |
| WEAKNESS_ASSESSMENT_REPORT.md              | Weakness assessment              |

---

## /docs/archive/ (Root)

Historical design documents moved from drafts after completion.

| Document                              | Description                                  |
| ------------------------------------- | -------------------------------------------- |
| LEGACY_CODE_ELIMINATION_PLAN.md       | Legacy code elimination (completed Nov 2025) |
| ORCHESTRATOR_ROLLOUT_FEATURE_FLAGS.md | Feature flag design (implemented Nov 2025)   |
| PHASE3_ADAPTER_MIGRATION_REPORT.md    | Phase 3 migration (completed Nov 2025)       |
| RULES_ENGINE_CONSOLIDATION_DESIGN.md  | Engine consolidation (completed Nov 2025)    |

---

## Finding Historical Context

If you need to understand historical decisions:

1. **Check pass reports** (PASS8-22) for chronological development history
2. **Check plans/** for original planning and design rationale
3. **Check assessments/** for specific task completion details

All archived documents have been marked with `⚠️ HISTORICAL` banners indicating they are no longer the source of truth. Current authoritative documents are:

- Rules: [RULES_CANONICAL_SPEC.md](../../RULES_CANONICAL_SPEC.md)
- Status: [CURRENT_STATE_ASSESSMENT.md](historical/CURRENT_STATE_ASSESSMENT.md)
- Goals: [PROJECT_GOALS.md](../../PROJECT_GOALS.md)
- API: [docs/architecture/CANONICAL_ENGINE_API.md](../architecture/CANONICAL_ENGINE_API.md)
