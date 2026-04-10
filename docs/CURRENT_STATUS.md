# Current Status

Last updated: 2026-04-10.

This is an owner-facing snapshot for Armand. It is not a marketing document.

## Snapshot Source

The machine-readable snapshot is checked in at `docs/data/training_status.json`.

Refresh it with:

```bash
npm --silent run training:status -- --json --ssh > docs/data/training_status.json
```

## Training Results

| Config       | Best Current Elo | Promotions | Latest Evidence                                       | Status                                      |
| ------------ | ---------------: | ---------: | ----------------------------------------------------- | ------------------------------------------- |
| `hex8_2p`    |         `1967.6` |        `6` | Latest completed eval rejected at 45% after 200 games | Proven strongest result; plateaued          |
| `square8_2p` |         `1601.8` |        `2` | Latest completed eval promoted at 60% after 50 games  | Proven improvement, but node currently down |
| `square8_3p` |         `1534.9` |        `1` | Recent seat-fair evals rejected at 24%, 26%, then 22% | Regressing; treat as weak evidence          |
| `square8_4p` |         `1500.0` |        `0` | Latest completed eval about 46%; no promotion         | Not improved above baseline                 |

## Infrastructure Health

| Node                          | Config       | SSH Probe | Loop  | Supervisor | Read                        |
| ----------------------------- | ------------ | --------- | ----- | ---------- | --------------------------- |
| `gh200-8` / `100.121.230.110` | `hex8_2p`    | OK        | Alive | Alive      | Healthy                     |
| `gh200-9` / `100.127.168.116` | `square8_2p` | OK        | Dead  | Dead       | Needs restart/debug         |
| `gh200-12` / `100.86.51.4`    | `square8_3p` | OK        | Alive | Alive      | Healthy process, weak evals |
| `gh200-10` / `100.100.19.96`  | `square8_4p` | OK        | Dead  | Dead       | Needs restart/debug         |

## What Works

- Production sandbox AI can route through the Python AI service and report `Gumbel MCTS` / neural telemetry instead of forcing local heuristic fallback.
- Product smoke coverage exists via `npm run smoke:product`.
- Training observability exists via `npm run training:status` and `npm run training:dashboard`.
- Replay-data provenance and validation are now inspectable via `npm run training:provenance -- <db>` and `npm run training:validate-db -- <db>`.
- The minimal training loop remains the supported proof harness.

## What Is Broken Or Unproven

- `square8_2p` and `square8_4p` are not currently running despite having recent metrics.
- Supervisor heartbeat ages are reported as `unknown` by the SSH probe for the live loops, so heartbeat-file path/state still needs follow-up.
- `square8_3p` is alive but appears to be failing seat-fair evaluation badly.
- `square8_4p` has not demonstrated improvement.
- The legacy P2P stack is still being decomposed/audited and should not be treated as the research source of truth yet.

## Next Actions

- Restart or diagnose `square8_2p` and `square8_4p` with the hardened supervisor/deploy scripts.
- Investigate why supervisor heartbeat files are not reporting age even when supervisor processes are alive.
- Keep training claims tied to completed metrics and evaluation conditions, not to aspirational Elo labels.
- Continue P2P decomposition behind `tests/unit/p2p/` and update `docs/P2P_DECOMPOSITION_PLAN.md` after each extraction.
