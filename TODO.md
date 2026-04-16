# RingRift Task Tracker

**Last Updated:** 2026-04-16
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
