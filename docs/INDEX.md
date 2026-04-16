# RingRift Documentation Index

> Last updated: 2026-04-16
> Scope: living documentation only

This index is the curated front door for the project docs. It points to the
documents that are meant to be read now. Historical plans, drafts, and
completed remediation notes live under [archive/INDEX.md](archive/INDEX.md).

## Start Here

| Document                                     | Why it matters                                                   |
| -------------------------------------------- | ---------------------------------------------------------------- |
| [../README.md](../README.md)                 | Fastest external-facing overview of the game and training result |
| [../QUICKSTART.md](../QUICKSTART.md)         | Local setup for the supported path                               |
| [RESULTS.md](RESULTS.md)                     | Checked-in evidence, caveats, and charts                         |
| [REPRODUCIBILITY.md](REPRODUCIBILITY.md)     | Exact training commands, hardware, and artifact locations        |
| [LESSONS_LEARNED.md](LESSONS_LEARNED.md)     | Engineering retrospective on what failed, what worked, and why   |
| [PROJECT_BRIEF.md](PROJECT_BRIEF.md)         | Short technical orientation for new readers                      |
| [RESEARCH_SNAPSHOT.md](RESEARCH_SNAPSHOT.md) | Short shareable research summary                                 |

## Game And Rules

| Document                                                                 | Why it matters                                  |
| ------------------------------------------------------------------------ | ----------------------------------------------- |
| [../RULES_CANONICAL_SPEC.md](../RULES_CANONICAL_SPEC.md)                 | Normative rules single source of truth          |
| [GAME_RULES.md](GAME_RULES.md)                                           | Five-minute player-facing rules explainer       |
| [rules/HUMAN_RULES.md](rules/HUMAN_RULES.md)                             | Human-readable explanation of how to play       |
| [rules/COMPLETE_RULES.md](rules/COMPLETE_RULES.md)                       | Full rulebook and examples                      |
| [rules/COMPACT_RULES.md](rules/COMPACT_RULES.md)                         | Compact implementation-oriented rules reference |
| [UX_RULES_EXPLANATION_MODEL_SPEC.md](UX_RULES_EXPLANATION_MODEL_SPEC.md) | UX/game-end explanation model                   |

## Architecture And Supported Path

| Document                                             | Why it matters                             |
| ---------------------------------------------------- | ------------------------------------------ |
| [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) | External architecture map                  |
| [REPOSITORY_MAP.md](REPOSITORY_MAP.md)               | What is active, legacy, or historical      |
| [DEVELOPER_GUIDE.md](DEVELOPER_GUIDE.md)             | Practical engineer/operator path           |
| [SCRIPT_INVENTORY.md](SCRIPT_INVENTORY.md)           | Supported script entrypoints and ownership |
| [PARITY_RUNBOOK.md](PARITY_RUNBOOK.md)               | TS↔Python parity workflow                  |
| [GPU_PARITY_CHECKLIST.md](GPU_PARITY_CHECKLIST.md)   | GPU-vs-CPU parity checklist                |

## Operations And Product

| Document                                                                     | Why it matters                       |
| ---------------------------------------------------------------------------- | ------------------------------------ |
| [DEPLOYMENT_CHECKLIST.md](DEPLOYMENT_CHECKLIST.md)                           | Deployment guardrail checklist       |
| [production/PRODUCTION_RUNBOOK.md](production/PRODUCTION_RUNBOOK.md)         | Production operating guide           |
| [operations/TRAINING_FLEET_RUNBOOK.md](operations/TRAINING_FLEET_RUNBOOK.md) | Training fleet deploy/reboot runbook |
| [data/training_fleet_manifest.json](data/training_fleet_manifest.json)       | Checked-in training fleet manifest   |
| [incidents/INDEX.md](incidents/INDEX.md)                                     | Incident response entrypoint         |
| [ACCESSIBILITY.md](ACCESSIBILITY.md)                                         | Accessibility behavior and controls  |

## Maintainer Notes

| Document                                                     | Why it matters                                        |
| ------------------------------------------------------------ | ----------------------------------------------------- |
| [CODEBASE_QUALITY_PROGRAM.md](CODEBASE_QUALITY_PROGRAM.md)   | Durable cleanup log and next seams                    |
| [LOAD_TEST_RESULTS.md](LOAD_TEST_RESULTS.md)                 | Current checked-in load-test snapshot                 |
| [SECRET_ROTATION_CHECKLIST.md](SECRET_ROTATION_CHECKLIST.md) | Security remediation checklist retained for operators |

## Archived Material

The following top-level docs were moved out of the active index because they
are historical drafts or completed plans:

- `archive/plans/ARCHITECTURAL_IMPROVEMENT_PLAN.md`
- `archive/plans/P2P_DECOMPOSITION_PLAN.md`
- `archive/plans/PLAN_AI_WORK.md`
- `archive/editorial/BLOG_POST_OUTLINE.md`
- `archive/editorial/CASE_STUDY_DRAFT.md`

Use [archive/INDEX.md](archive/INDEX.md) for historical planning, assessments,
and editorial drafts.
