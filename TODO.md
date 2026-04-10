# RingRift Task Tracker

**Last Updated:** 2026-04-10
**Project Health:** usable but operationally mixed
**Current Focus:** make the legacy infrastructure observable, diagnosable, and safe to reuse alongside the supported minimal training loop

## Current Training Status

Snapshot source: `docs/data/training_status.json`, generated with:

```bash
npm --silent run training:status -- --json --ssh > docs/data/training_status.json
```

| Config       | Node                          | Current Evidence                                                  | Process Status                               | Read                                                  |
| ------------ | ----------------------------- | ----------------------------------------------------------------- | -------------------------------------------- | ----------------------------------------------------- |
| `hex8_2p`    | `gh200-8` / `100.121.230.110` | `1967.6` Elo, 6 promotions, latest completed eval rejected at 45% | Loop and supervisor alive                    | Proven strongest result; plateaued near 1968          |
| `square8_2p` | `gh200-9` / `100.127.168.116` | `1601.8` Elo, 2 promotions, latest completed eval promoted at 60% | Loop and supervisor dead in latest SSH probe | Good result, but node needs operational restart/debug |
| `square8_3p` | `gh200-12` / `100.86.51.4`    | `1534.9` Elo, 1 promotion, latest completed eval rejected at 22%  | Loop and supervisor alive                    | Regressing under seat-fair eval; treat cautiously     |
| `square8_4p` | `gh200-10` / `100.100.19.96`  | `1500.0` Elo, 0 promotions, latest completed eval about 46%       | Loop and supervisor dead in latest SSH probe | Experimental; not yet proven                          |

## What Is Proven

- The TypeScript engine remains the rules source of truth; Python mirrors it for AI/training.
- The supported minimal loop has produced real promotions on `hex8_2p` and `square8_2p`.
- `hex8_2p` is the strongest current model and is deployed through the live AI ladder.
- Product smoke coverage now checks production health, sandbox AI move generation, replay submission, and local model loadability via `npm run smoke:product`.
- Training observability now has `npm run training:status`, `npm run training:dashboard`, `npm run training:validate-db -- <db>`, and `npm run training:provenance -- <db>`.

## What Is Experimental Or Not Yet Trustworthy

- `square8_3p` has one promotion but recent seat-fair evaluations are poor; do not market it as a strong model yet.
- `square8_4p` has not improved above baseline.
- The legacy coordinator/P2P stack is being audited and decomposed; it is useful infrastructure, but not yet the source of truth for the current research claims.
- Human-submitted sandbox games are being recorded, but should only enter training when provenance and validation gates say they are ready.

## Immediate P0/P1 Work

- P0: Restart or diagnose dead `square8_2p` and `square8_4p` supervisors using `npm run training:status -- --ssh` plus remote logs.
- P0: Keep `scripts/minimal_alphazero_loop.py` and its support libs stable; do not mix infrastructure refactors into the proven loop.
- P1: Finish operational docs and dashboard flow so node health does not require ad-hoc SSH archaeology.
- P1: Continue P2P orchestrator decomposition only behind tests; the goal is reuse, not deletion.
- P1: Use `npm run training:provenance -- <db>` and `npm run training:validate-db -- <db>` before mixing replay DBs into training exports.

## Reference Docs

- `docs/CURRENT_STATUS.md` - owner-facing state snapshot.
- `docs/RESULTS.md` - research result summary and limitations.
- `docs/architecture/TRAINING_INFRASTRUCTURE_STRATEGY.md` - minimal-loop plus legacy-infrastructure strategy.
- `ai-service/AGENTS.md` and root `AGENTS.md` - invariants for code changes.
