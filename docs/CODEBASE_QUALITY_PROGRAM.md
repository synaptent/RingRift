# Codebase Quality Program

This document tracks the durable quality program for RingRift. It replaces the
old four-score framework with six project-facing categories that map directly to
the repository's current risks: correctness, navigability, operational
reliability, surface-area discipline, credibility, and playability.

Status is current as of April 16, 2026.

## Current Scores

| Category                | Score | Target | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| ----------------------- | ----: | -----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness             | `8.4` |  `9.0` | Full Python unit/contracts gate passes with `33508 passed, 8 skipped, 3 warnings in 3223.77s`; full TypeScript/Jest gate passes with `11742 passed, 200 skipped, 1 todo`, `582` suites passed, `50` suites skipped, and `1` snapshot passed in `282.328s`. `docs/REPRODUCIBILITY.md` points at the checked-in `ai-service/scripts/minimal_alphazero_loop.py` command surface, and its documented flags exist in the CLI.                                                                                                                                                                                                                                                                           |
| Navigability            | `8.5` |  `9.0` | A cold README pass reaches latest results from the first section, the minimal training loop from the quick-start block, and the engine source from the repository map. `ai-service/app/README.md` now describes the active Python package layout, archive boundary, and canonical import surfaces after the facade drains.                                                                                                                                                                                                                                                                                                                                                                         |
| Operational reliability | `7.3` |  `8.5` | The repo has strong pieces (`node_roles.yaml`, systemd units, `deploy_minimal_loops.sh`, health/status scripts), and the checked-in fleet docs now have a read-only validator that cross-checks the manifest, runbook, role file, canary deploy script, and systemd units before SSH/deployment. A fresh clone still cannot deploy the full role-aware fleet from checked-in files alone because the runtime host inventory lives in untracked `ai-service/config/distributed_hosts.yaml`. Reboot behavior is split: systemd services are boot-persistent when installed/enabled, while the current minimal-loop canary path uses a `nohup` supervisor that must be restarted after reboot.        |
| Surface-area discipline | `8.3` |  `8.5` | `app.training` is down to `23` root exports, `app.ai` advertises `15` factory/profile exports, `app.distributed` is down from `141` to `136` root exports, `app.db` advertises `8`, and `app.rules` remains at `4`. The surface dashboard now ratchets `app.distributed` at `140` exports and direct-unimported `app.*` files at `25`. Direct-unimported files dropped from `39` to `20` after deleting the 20 largest orphaned modules. Top-level import-cycle analysis reports `0` cycles.                                                                                                                                                                                                       |
| Credibility             | `8.7` |  `9.0` | The public docs are materially more honest: README, `docs/RESULTS.md`, `docs/REPRODUCIBILITY.md`, and `docs/LESSONS_LEARNED.md` use real results and caveats instead of vague claims. Stale root clutter and generated artifacts were removed, the supported documentation path now has a local-link ratchet, and result claims now have an evidence manifest separating repo-verifiable, S3-backed, and operator-reported data. Secret scans found placeholders/examples/test fixtures, not private keys or real provider tokens. The latest `square8_2p` public result is now updated to `1782.0` Elo with S3 and operator-verified node-log provenance instead of relying on chat-only context. |
| Playability             | `8.3` |  `9.0` | Live `https://ringrift.ai/` now serves the public landing page instead of redirecting first-time visitors to login. The sandbox loads anonymously, labels the opponent as Neural AI, has rule/onboarding affordances, and fits a `375px` viewport without horizontal overflow. A live browser smoke test confirmed no console errors on landing or sandbox pages. Full end-game VictoryModal stats were verified by tests but still need a production-length human playthrough to confirm the entire visible path.                                                                                                                                                                                 |

Overall score: `8.2/10`. The weakest category is operational reliability, not
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

April 16, 2026 follow-up:

- Added `ai-service/scripts/validate_training_fleet_docs.py`, a read-only local
  validator for the checked-in fleet story. It performs `34` consistency checks
  across `docs/data/training_fleet_manifest.json`,
  `docs/operations/TRAINING_FLEET_RUNBOOK.md`, `config/node_roles.yaml`,
  `scripts/deploy_minimal_loops.sh`, and the systemd unit files without SSH,
  subprocess deployment, or network calls.
- Added `ai-service/tests/unit/scripts/test_validate_training_fleet_docs.py` so
  the validator itself is part of the Python unit gate.

April 24, 2026 follow-up:

- Added [docs/REVIEWER_GUIDE.md](REVIEWER_GUIDE.md) and
  [docs/data/reviewer_surface_manifest.json](data/reviewer_surface_manifest.json)
  to give cold outside reviewers a focused path through the supported product,
  rules, parity, and minimal-loop surfaces.
- Added `scripts/check_reviewer_surface.py` and wired it into
  `scripts/check_supported_path.sh` plus `npm run reviewer:check` so the
  reviewer surface remains discoverable and does not drift back into historical
  or archive-first navigation.
- Added `jest.rules-coverage.config.js` and `npm run
test:coverage:rules-critical`, then wired the command into
  `scripts/check_supported_path.sh`. This establishes non-zero coverage
  thresholds for the canonical TypeScript rules engine while deliberately
  excluding legacy compatibility files from the reviewer-critical ratchet.

## Recent Quality Baseline

- Python gate: `cd ai-service && PYTHONPATH=. python3 -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120` passed with `33508 passed`, `8 skipped`, and `3 warnings`.
- TypeScript gate: `npm test` passed with `11742 passed`, `200 skipped`, `1 todo`, `582` suites passed, `50` suites skipped, and `1` snapshot passed.
- Reproducibility check: `docs/REPRODUCIBILITY.md` remains aligned with the `minimal_alphazero_loop.py` CLI flags used for checked-in results.
- Live product check: `ringrift.ai` serves the public landing page, anonymous sandbox loads, Neural AI labeling is visible, and mobile width does not overflow.
- Surface-area check: root training facade is contracted to `23` exports; `app.distributed` is contracted to `136/140` exports; direct-unimported `app.*` files are contracted to `20/25`; top-level `app` import-cycle count is `0`.

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
  `nohup_p2p_fresh.out`, `progress.md`, `playwright-report/index.html`,
  and the accidental zero-byte root file named `;\nfi`.
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
- Fresh-clone audit found two credibility defects and fixed both where safe:
  active Markdown docs no longer contain developer-machine absolute-path
  links, and `npm install` now runs `prisma generate` so `@prisma/client`
  exists before a new developer starts the server. The exact
  `npm install && npm run dev` path still requires the documented `.env`,
  Postgres, and Redis setup before the backend can connect.
- Fresh-clone Python contracts exposed an artifact-boundary defect: the test
  suite assumed large `ai-service/models/canonical_*.pth` checkpoints were
  present even though git intentionally does not track them. `ai-service/models`
  now has a tracked README manifest, and the contract accepts either synced
  non-empty checkpoint files or explicit external-artifact manifest entries.
- Supported-doc link integrity is now a contract:
  `ai-service/tests/contracts/test_supported_docs_links.py` checks that the
  public reader path exists, does not contain developer-machine absolute paths,
  and has no broken local Markdown links. This deliberately excludes archived
  planning and diagnostic docs so historical material does not block unrelated
  work.
- Result-claim provenance is now explicit in
  `docs/data/results_evidence_manifest.json`. Public docs link to it from
  `docs/RESULTS.md` and `docs/REPRODUCIBILITY.md`, and it records which claims
  are checked-in, which require S3 artifacts, and which operator-reported
  updates should not be promoted into public claims yet.

## Surface Cleanup Log

April 16, 2026 surface-area pass:

- Deleted the 20 largest direct-unimported `app/` modules after checking for
  direct imports, test-only imports, and daemon dynamic instantiation paths.
  `UNIFIED_DATA_CATALOG` remains a deprecated daemon type, but its runner is a
  no-op and no longer needs `app/coordination/unified_data_catalog.py`.
- Reduced direct-unimported `app.*` files from `39` to `20`, then tightened the
  contract budget from `60` to `25`.
- Removed deprecated cluster-coordinator aliases from the `app.distributed`
  package root. Active code had no `from app.distributed import
ClusterCoordinator`-style imports; legacy callers must import
  `app.distributed.cluster_coordinator` directly and receive that module's
  deprecation warning.
- Reduced `app.distributed` root exports from `141` to `136`, then tightened
  its contract budget from `145` to `140`.
- Ran `scripts/audit_import_graph.py --module-prefix app --report cycles
--max-depth 8`: top-level imports report `0` cycles. Including local imports
  reports lazy-import cycles and is tracked as intentional startup-cycle
  mitigation rather than a top-level import failure signal.

## Active Batch Order

1. Refresh `docs/RESULTS.md`, `docs/data/results_snapshot.json`, and
   `docs/data/results_evidence_manifest.json` after the newest training
   promotions have durable metrics/log artifacts checked in or referenced.
2. Continue reducing the remaining `20` direct-unimported `app.*` candidates
   after confirming dynamic-entrypoint use.
3. Extend the operational preflight checker to validate private inventory shape
   when `ai-service/config/distributed_hosts.yaml` is present locally, without
   checking in host secrets.
4. Extend non-zero coverage ratchets into Python training contracts and parity
   checkpoint-contract modules.
5. Expand link integrity coverage from the supported-doc path into `tests/` and
   deeper rules/UX docs after resolving their intentionally archived references.
6. Verify a full production sandbox game to completion and confirm the visible
   VictoryModal stats path.

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
