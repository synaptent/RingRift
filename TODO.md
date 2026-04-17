# RingRift Task Tracker

**Last Updated:** 2026-04-17
**Project Health:** credible and cleanup-oriented, with active training still producing value
**Current Focus:** preserve the working minimal loop, keep public result claims evidence-backed, and continue targeted repo quality improvements

## Current Research And Runtime Sources

Do not treat this tracker as a duplicate status dashboard. Use the canonical sources instead:

- [RESULTS.md](docs/RESULTS.md) for current supported public claims and headline Elo
- [RESEARCH_SNAPSHOT.md](docs/RESEARCH_SNAPSHOT.md) for the short shareable state
- [docs/data/results_evidence_manifest.json](docs/data/results_evidence_manifest.json) for the evidence boundary behind public result claims
- [`docs/data/training_status.json`](docs/data/training_status.json) for the checked-in machine-readable runtime snapshot
- [CURRENT_STATUS.md](docs/CURRENT_STATUS.md) only for the preserved April 10 owner-facing operational memo
- [docs/operations/TRAINING_FLEET_RUNBOOK.md](docs/operations/TRAINING_FLEET_RUNBOOK.md) for safe training fleet operations

## What Is Proven Right Now

- `hex8_2p` is the strongest result at `1979.8` Elo with `7` promotions.
- `square8_2p` is the second clean improvement path at `1782.0` Elo with `5` promotions after two consecutive `62%` promotions.
- `square8_3p` has `1` multiplayer promotion at `1534.9`, but the evidence is still weak.
- The supported claim remains narrow: only a small subset of configs has strong improvement evidence.
- The `hex8_2p` v4 architecture experiment is the active plateau-break attempt; keep it labeled as an experiment until a completed result has durable evidence.

## Active Priorities

1. Keep the supported minimal loop stable and avoid infrastructure churn that resets selfplay progress.
2. Keep `docs/RESULTS.md`, `docs/data/results_snapshot.json`, and `docs/data/results_evidence_manifest.json` synchronized whenever a result claim changes.
3. Continue the in-repo cleanup program in [docs/CODEBASE_QUALITY_PROGRAM.md](docs/CODEBASE_QUALITY_PROGRAM.md), especially surface-area discipline and operational reliability.
4. Continue reducing source-of-truth drift across docs, snapshots, public entrypoints, and runbooks.
5. Continue contracting `ai-service/app/coordination` and other large package facades toward explicit submodule imports.
6. Execute [docs/planning/AI_QUALITY_STRENGTH_DIVERSITY_PLAN_2026-04-16.md](docs/planning/AI_QUALITY_STRENGTH_DIVERSITY_PLAN_2026-04-16.md) — quality, strength, diversity, and production-experience plan; tracked as [#77](https://github.com/synaptent/RingRift/issues/77) with Week 1 children [#78 A1](https://github.com/synaptent/RingRift/issues/78), [#79 A2](https://github.com/synaptent/RingRift/issues/79), [#80 C2](https://github.com/synaptent/RingRift/issues/80), [#81 D1](https://github.com/synaptent/RingRift/issues/81), [#82 D5](https://github.com/synaptent/RingRift/issues/82).

## Final Quality Sweep Findings

Verified on 2026-04-17:

- README headline result numbers match `docs/data/results_snapshot.json`.
- Local links in tracked top-level Markdown passed: `461` checked, `0` broken.
- `npm install` completes and regenerates Prisma Client, but reports `17` npm audit findings (`4` low, `6` moderate, `7` high) and the existing deprecated Husky install warning.
- `npm run dev` starts the Vite client, but the server logs Redis `ECONNREFUSED` when `docker compose up -d postgres redis` has not been run first. The README documents the prerequisite, but a first-run preflight would reduce confusion.
- `npm test` passed: `583` suites passed, `50` skipped; `11,746` tests passed, `200` skipped, `1` todo.
- `cd ai-service && PYTHONPATH=. python3 -m pytest tests/contracts/ -q` passed: `4,883` tests.

Remaining non-blocking cleanup:

- Triage the `npm audit` findings instead of running `npm audit fix --force` blindly.
- Reduce Jest suite noise: expected console errors/warnings and `act(...)` warnings make the all-green run look less professional.
- Revisit the `200` skipped Jest tests and `1` todo so the skip count stays intentional.
- Move or remove unused tracked root assets `ringrift icon.png` and `ringrift_favicon.ico`; the active web assets are served from `src/client/public/`.
- Keep ignored local workspace clutter (`.env*`, logs, dumps, caches, `PLAN.md` stub) out of release artifacts; these are not tracked fresh-clone clutter but they make this local checkout noisy.

## What Remains Unproven Or Weak

- multiplayer evidence is still weak outside the single `square8_3p` promotion
- larger boards remain slower and less mature
- `square8_4p` has not demonstrated improvement above baseline
- the coordination and legacy operational surface is still much harder to understand than the supported path
- full role-aware fleet deployment still requires private `ai-service/config/distributed_hosts.yaml` inventory that is not checked in

## Execution Rule

When a doc and a snapshot disagree, fix the source-of-truth boundary first instead of updating more summaries by hand.

## Reference Docs

- [RESULTS.md](docs/RESULTS.md)
- [RESEARCH_SNAPSHOT.md](docs/RESEARCH_SNAPSHOT.md)
- [ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md)
- [CODEBASE_QUALITY_PROGRAM.md](docs/CODEBASE_QUALITY_PROGRAM.md)
- [CURRENT_STATUS.md](docs/CURRENT_STATUS.md)
- [PART3_INFRASTRUCTURE_ROADMAP.md](docs/architecture/PART3_INFRASTRUCTURE_ROADMAP.md)
- [TRAINING_INFRASTRUCTURE_STRATEGY.md](docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md)
- [AI_QUALITY_STRENGTH_DIVERSITY_PLAN_2026-04-16.md](docs/planning/AI_QUALITY_STRENGTH_DIVERSITY_PLAN_2026-04-16.md)
