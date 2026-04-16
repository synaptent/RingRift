# Codebase Quality Program

This document tracks the durable quality program for RingRift. It replaces the
old four-score framework with six project-facing categories that map directly to
the repository's current risks: correctness, navigability, operational
reliability, surface-area discipline, credibility, and playability.

Status is current as of April 16, 2026.

## Current Scores

| Category                | Score | Target | Evidence                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| ----------------------- | ----: | -----: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Correctness             | `8.4` |  `9.0` | Full Python unit/contracts gate passes with `33519 passed, 8 skipped, 3 warnings in 1778.98s`; full TypeScript/Jest gate passes with `11742 passed, 200 skipped, 1 todo`, `582` suites passed, `50` suites skipped, and `1` snapshot passed in `282.328s`. `docs/REPRODUCIBILITY.md` points at the checked-in `ai-service/scripts/minimal_alphazero_loop.py` command surface, and its documented flags exist in the CLI.                                                                                                                                             |
| Navigability            | `8.5` |  `9.0` | A cold README pass reaches latest results from the first section, the minimal training loop from the quick-start block, and the engine source from the repository map. `ai-service/app/README.md` now describes the active Python package layout, archive boundary, and canonical import surfaces after the facade drains.                                                                                                                                                                                                                                           |
| Operational reliability | `7.1` |  `8.5` | The repo has strong pieces (`node_roles.yaml`, systemd units, `deploy_minimal_loops.sh`, health/status scripts), but a fresh clone still cannot deploy the full role-aware fleet from checked-in files alone because the runtime host inventory lives in untracked `ai-service/config/distributed_hosts.yaml`. Reboot behavior is split: systemd services are boot-persistent when installed/enabled, while the current minimal-loop canary path uses a `nohup` supervisor that must be restarted after reboot.                                                      |
| Surface-area discipline | `7.5` |  `8.5` | `app.training` is down to `23` root exports and has import-hygiene ratchets. `app.coordination` still exposes `594` resolved names via lazy compatibility aliases despite the root file shrinking to `81` lines. `app.distributed` exposes `141` names with a deprecation timeline for legacy cluster symbols. The checked-in AI-service Python surface is `3194` tracked `*.py` files, with `3370` repo-relevant Python files excluding virtualenv payloads. Static AST import analysis found `58` direct-unimported `app.*` modules, with a dynamic-import caveat. |
| Credibility             | `8.1` |  `9.0` | The public docs are materially more honest: README, `docs/RESULTS.md`, `docs/REPRODUCIBILITY.md`, and `docs/LESSONS_LEARNED.md` use real results and caveats instead of vague claims. A secret scan found only placeholders/examples/test fixtures, not private keys or real API tokens. Current checked-in results still lag the operator-reported newest `square8_2p` promotion to about `1782` Elo, so claims remain conservative but need a refresh after evidence is checked in.                                                                                |
| Playability             | `8.3` |  `9.0` | Live `https://ringrift.ai/` now serves the public landing page instead of redirecting first-time visitors to login. The sandbox loads anonymously, labels the opponent as Neural AI, has rule/onboarding affordances, and fits a `375px` viewport without horizontal overflow. A live browser smoke test confirmed no console errors on landing or sandbox pages. Full end-game VictoryModal stats were verified by tests but still need a production-length human playthrough to confirm the entire visible path.                                                   |

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

## Active Batch Order

1. Raise operational reliability with a checked-in fleet manifest, runbook, and docs ratchet.
2. Continue draining `app.coordination` lazy compatibility exports toward explicit submodule imports.
3. Continue reducing direct-unimported `app.*` modules after confirming dynamic-import use.
4. Refresh checked-in result docs after the newest training promotions have durable logs or artifacts.
5. Verify a full production sandbox game to completion and confirm the visible VictoryModal stats path.

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
