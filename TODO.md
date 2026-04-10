# RingRift Task Tracker

**Last Updated:** 2026-04-10
**Project Health:** technically credible, operationally mixed
**Current Focus:** finish the Part 3 cleanup so the legacy infrastructure is diagnosable and reusable without displacing the supported minimal loop

## Current Training Snapshot

Source:

```bash
npm --silent run training:status -- --json --ssh > docs/data/training_status.json
```

| Config       | Node                          | Latest Iteration | Evidence                                                                      | Process Status            | Read                                           |
| ------------ | ----------------------------- | ---------------: | ----------------------------------------------------------------------------- | ------------------------- | ---------------------------------------------- |
| `hex8_2p`    | `gh200-8` / `100.121.230.110` |             `32` | `1967.6` Elo, `6` promotions, latest eval rejected at `40%` after `50` games  | Loop and supervisor alive | Strongest result, but clearly plateaued        |
| `square8_2p` | `gh200-9` / `100.127.168.116` |             `31` | `1601.8` Elo, `2` promotions, latest promotion came at `60%` after `50` games | Loop and supervisor dead  | Real 2P improvement, but currently not running |
| `square8_3p` | `gh200-12` / `100.86.51.4`    |             `13` | `1534.9` Elo, `1` promotion, recent seat-fair evals at `22-24%` win rate      | Loop and supervisor alive | Regressing; not trustworthy as a strong result |
| `square8_4p` | `gh200-10` / `100.100.19.96`  |              `5` | `1500.0` Elo, `0` promotions, latest eval around `46%`                        | Loop and supervisor dead  | No proven improvement                          |

## What Is Proven

- The supported minimal loop remains the training path to trust for reproducible results.
- `hex8_2p` and `square8_2p` both have genuine promotion evidence above the `1500` baseline.
- Production smoke currently passes end to end against `ringrift.ai`, including server health, sandbox AI move generation, replay submission, and local `canonical_hex8_2p` model loadability.
- Training observability now exists in repo via `training:status`, `training:dashboard`, `training:validate-db`, and `training:provenance`.
- The P2P orchestrator size target has been achieved: `2591` LOC in the main file with `21` extracted mixins totaling `12618` LOC.

## What Is Not Yet Trustworthy

- `square8_3p` is still regressing under seat-fair evaluation.
- `square8_4p` has not shown improvement above baseline.
- Two of the four active training nodes are currently down.
- The latest remote GitHub Actions status on `main` is red: both `Supported Path` and `ci.yml` failed on commit `bb4c99be1`, even though the local Phase 8 verification passed.
- Supervisor heartbeat files still report `unknown` age on the nodes that are alive, so the operational story is improved but not finished.

## Immediate Priorities

- P0: Diagnose the failing GitHub Actions runs on `main` and get the remote CI surface green again.
- P0: Restart or debug `square8_2p` and `square8_4p`.
- P0: Keep `ai-service/scripts/minimal_alphazero_loop.py` stable; infrastructure cleanup must not leak into the supported loop.
- P1: Finish the remaining Part 3 cleanup phases in order, updating the roadmap after each.
- P1: Keep training-data provenance gates mandatory before any human/sandbox replay data enters exports.

## Reference Docs

- [CURRENT_STATUS.md](docs/CURRENT_STATUS.md)
- [RESULTS.md](docs/RESULTS.md)
- [ARCHITECTURE_OVERVIEW.md](docs/ARCHITECTURE_OVERVIEW.md)
- [PART3_INFRASTRUCTURE_ROADMAP.md](docs/architecture/PART3_INFRASTRUCTURE_ROADMAP.md)
- [TRAINING_INFRASTRUCTURE_STRATEGY.md](docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md)
