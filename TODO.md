# RingRift Task Tracker

**Last Updated:** 2026-04-13
**Project Health:** technically credible, cleanup-oriented
**Current Focus:** raise repo quality without destabilizing the supported minimal loop or duplicating stale status narratives

## Current Research And Runtime Sources

Do not treat this tracker as a duplicate status dashboard. Use the canonical sources instead:

- [RESULTS.md](docs/RESULTS.md) for current supported public claims and headline Elo
- [RESEARCH_SNAPSHOT.md](docs/RESEARCH_SNAPSHOT.md) for the short shareable state
- [`docs/data/training_status.json`](docs/data/training_status.json) for the checked-in machine-readable runtime snapshot
- [CURRENT_STATUS.md](docs/CURRENT_STATUS.md) only for the preserved April 10 owner-facing operational memo

## What Is Proven Right Now

- `hex8_2p` is the strongest result at `1979.8` Elo with `7` promotions.
- `square8_2p` is the second clean improvement path at `1601.8` Elo with `2` promotions.
- `square8_3p` has `1` multiplayer promotion at `1534.9`, but the evidence is still weak.
- The supported claim remains narrow: only a small subset of configs has strong improvement evidence.

## Active Priorities

- keep the supported minimal loop stable and avoid infrastructure churn that resets selfplay progress
- continue the in-repo cleanup program in [docs/CODEBASE_QUALITY_PROGRAM.md](docs/CODEBASE_QUALITY_PROGRAM.md)
- keep reducing source-of-truth drift across docs, snapshots, and public entrypoints
- continue contracting `ai-service/app/coordination` toward a smaller public API surface
- add lightweight guards so stale docs and snapshots fail fast instead of drifting silently

## What Remains Unproven Or Weak

- multiplayer evidence is still weak outside the single `square8_3p` promotion
- larger boards remain slower and less mature
- `square8_4p` has not demonstrated improvement above baseline
- the coordination and legacy operational surface is still much harder to understand than the supported path

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
