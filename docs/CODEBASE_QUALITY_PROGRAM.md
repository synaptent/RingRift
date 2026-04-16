# Codebase Quality Program

This document tracks the durable quality program for RingRift. It replaces the
old four-score framework with six project-facing categories that map directly to
the repository's current risks: correctness, navigability, operational
reliability, surface-area discipline, credibility, and playability.

Status is current as of April 16, 2026.

## Current Scores

| Category                | Score | Target | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------- | ----: | -----: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Correctness             | `8.4` |  `9.0` | Full Python unit/contracts gate passes with `33519 passed, 8 skipped, 3 warnings in 1778.98s`; full TypeScript/Jest gate passes with `11742 passed, 200 skipped, 1 todo`, `582` suites passed, `50` suites skipped, and `1` snapshot passed in `282.328s`. `docs/REPRODUCIBILITY.md` points at the checked-in `ai-service/scripts/minimal_alphazero_loop.py` command surface, and its documented flags exist in the CLI.                                                                                           |
| Navigability            | `8.5` |  `9.0` | A cold README pass reaches latest results from the first section, the minimal training loop from the quick-start block, and the engine source from the repository map. `ai-service/app/README.md` now describes the active Python package layout, archive boundary, and canonical import surfaces after the facade drains.                                                                                                                                                                                         |
| Operational reliability | `7.1` |  `8.5` | The repo has strong pieces (`node_roles.yaml`, systemd units, `deploy_minimal_loops.sh`, health/status scripts), but a fresh clone still cannot deploy the full role-aware fleet from checked-in files alone because the runtime host inventory lives in untracked `ai-service/config/distributed_hosts.yaml`. Reboot behavior is split: systemd services are boot-persistent when installed/enabled, while the current minimal-loop canary path uses a `nohup` supervisor that must be restarted after reboot.    |
| Surface-area discipline | `7.8` |  `8.5` | `app.training` is down to `23` root exports, `app.ai` now advertises `15` factory/profile exports, `app.db` advertises `8` root exports, and `app.rules` remains at `4`. The new surface dashboard ratchets top-level package export budgets and the `app.coordination` root file budget. Static AST import analysis now filters package/CLI/module entrypoints and reports `39` direct-unimported `app.*` files; the top candidates are retained until dynamic use is ruled out.                                  |
| Credibility             | `8.1` |  `9.0` | The public docs are materially more honest: README, `docs/RESULTS.md`, `docs/REPRODUCIBILITY.md`, and `docs/LESSONS_LEARNED.md` use real results and caveats instead of vague claims. A secret scan found only placeholders/examples/test fixtures, not private keys or real API tokens. Current checked-in results still lag the operator-reported newest `square8_2p` promotion to about `1782` Elo, so claims remain conservative but need a refresh after evidence is checked in.                              |
| Playability             | `8.3` |  `9.0` | Live `https://ringrift.ai/` now serves the public landing page instead of redirecting first-time visitors to login. The sandbox loads anonymously, labels the opponent as Neural AI, has rule/onboarding affordances, and fits a `375px` viewport without horizontal overflow. A live browser smoke test confirmed no console errors on landing or sandbox pages. Full end-game VictoryModal stats were verified by tests but still need a production-length human playthrough to confirm the entire visible path. |

Overall score: `8.0/10`. The weakest category is operational reliability, not
game logic or presentation. The next improvements should make the checked-in
operational story sufficient for a new operator to understand the fleet,
deploy/restart it safely, and know which paths survive a reboot.

## Guardrails

- Do not restart or churn infrastructure unless a code change directly requires it.
- Do not touch `ai-service/scripts/minimal_alphazero_loop.py` core logic as part of quality cleanup.
- Prefer source-of-truth reduction over adding more wrappers.
- Prefer deleting dead surfaces over maintaining historical compatibility indefinitely.
- Use small, verified commits.
- Keep claims tied to current checked-in evidence, not oral history.

## Operational Reliability Improvement Plan

Operational reliability is the lowest-scoring category. This batch should raise
it by adding checked-in, testable operator guidance without modifying running
training infrastructure.

1. Add a checked-in training fleet manifest under `docs/data/` that summarizes
   active roles, boards, known hosts, supervision mode, and the distinction
   between checked-in role data and private runtime host inventory.
2. Add a training fleet runbook under `docs/operations/` that covers preflight,
   minimal-loop canary deployment, role-aware systemd deployment, status checks,
   reboot behavior, rollback, and known gaps.
3. Add a docs ratchet that keeps the runbook and manifest discoverable from
   `docs/INDEX.md`, `ai-service/scripts/README.md`, and tracked operations
   guidance, with explicit coverage for the `nohup` non-boot-persistent caveat.

## Recent Quality Baseline

- Python gate: `cd ai-service && PYTHONPATH=. python3 -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120` passed with `33519 passed`, `8 skipped`, and `3 warnings`.
- TypeScript gate: `npm test` passed with `11742 passed`, `200 skipped`, `1 todo`, `582` suites passed, `50` suites skipped, and `1` snapshot passed.
- Reproducibility check: `docs/REPRODUCIBILITY.md` remains aligned with the `minimal_alphazero_loop.py` CLI flags used for checked-in results.
- Live product check: `ringrift.ai` serves the public landing page, anonymous sandbox loads, Neural AI labeling is visible, and mobile width does not overflow.
- Surface-area check: root training facade is contracted to `23` exports; coordination and distributed remain the main compatibility surfaces to continue draining.

## Credibility Cleanup Log

April 16, 2026 credibility pass:

- Removed redundant tracked redirect stubs whose archived copies already exist:
  `cluster_status_report.md`, `ai-service/P2P_DEPLOYMENT_REPORT.md`,
  `ai-service/QUICK_WINS_CONSOLIDATION_REPORT.md`,
  `ai-service/cluster_update_report.md`, and
  `ai-service/COMPREHENSIVE_ACTION_PLAN_2025_12_25.md`.
- Removed root-level fresh-clone clutter whose maintained copies are elsewhere
  or whose contents were transient logs/work notes:
  `CLIENT_TEST_PLAN.md`, `CLUSTER_STATUS_CRITICAL.txt`,
  `P2P_INVESTIGATION_SUMMARY.md`, `P2P_STATUS_SUMMARY.txt`,
  `P2P_STATUS_TABLE.md`, `PLAN_selfplay_loop_closure.md`,
  `ROADMAP_2025Q1.md`, `nohup_master.out`, `nohup_p2p.out`,
  `nohup_p2p_fresh.out`, and `progress.md`.
- Archived unreferenced active-tree audit doc:
  `docs/architecture/TEST_INFRASTRUCTURE_AUDIT.md` moved to
  `docs/archive/assessments/TEST_INFRASTRUCTURE_AUDIT.md`.
- Kept referenced high-volume docs in place even when their names are not ideal:
  `docs/ai/AI_TRAINING_ASSESSMENT_FINAL.md`,
  `docs/testing/WEAK_ASSERTION_AUDIT.md`,
  `docs/architecture/APP_IMPORT_AUDIT.md`,
  `docs/architecture/RULES_ENGINE_AUDIT.md`, and
  `docs/planning/SLO_THRESHOLD_ALIGNMENT_AUDIT.md` still have active
  references from docs, runbooks, or SSOT checks.
- Secret-history check command used:
  `git log --all --diff-filter=A -- '*.env' '*.pem' '*.key' '*secret*' '*credential*' '*password*' '*token*'`.
  The tracked additions surfaced secret-handling utilities, placeholder env
  examples, and archived `cluster_nodes.env`, but no tracked private key or real
  provider token was found in the checked output. Local untracked `.env*` files
  contain deployment values and should remain untracked.
- Claims audit boundary: README, `docs/RESULTS.md`, and
  `docs/REPRODUCIBILITY.md` match `docs/data/results_snapshot.json` for the
  checked-in April 15 headline numbers. Detailed per-iteration histories,
  training duration estimates, and checkpoint artifacts require the S3 archive
  named in `docs/REPRODUCIBILITY.md`; they are not fully reproducible from the
  repository alone.

## Active Batch Order

1. Continue draining `app.coordination`, `app.distributed`, `app.metrics`, and `app.tournament` lazy compatibility exports toward explicit submodule imports.
2. Continue reducing the `39` direct-unimported `app.*` candidates after confirming dynamic-entrypoint use.
3. Refresh checked-in result docs after the newest training promotions have durable logs or artifacts.
4. Verify a full production sandbox game to completion and confirm the visible VictoryModal stats path.

## Reassessment Triggers

Re-score the codebase after:

- each full Python and TypeScript gate baseline
- each production deploy or public sandbox behavior change
- each operational-docs or deployment-path cleanup wave
- each major package-surface contraction wave
- each checked-in results refresh

If future context is lost, resume from:

1. this file
2. `docs/RESULTS.md`
3. `docs/REPRODUCIBILITY.md`
4. `docs/architecture/OVERVIEW.md`
5. `ai-service/app/README.md`
6. `ai-service/scripts/README.md`
