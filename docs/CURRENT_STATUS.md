# Current Status (Historical Snapshot)

Historical snapshot date: April 10, 2026.

This file is preserved as an owner-facing operational memo from that date. It is not the live status entrypoint.

For the current supported public claims and April 13 research state, use:

- [RESULTS.md](/Users/armand/Development/RingRift/docs/RESULTS.md)
- [RESEARCH_SNAPSHOT.md](/Users/armand/Development/RingRift/docs/RESEARCH_SNAPSHOT.md)
- [docs/data/training_status.json](/Users/armand/Development/RingRift/docs/data/training_status.json)

For the current supported operator and trainer path, use:

- [DEVELOPER_GUIDE.md](/Users/armand/Development/RingRift/docs/DEVELOPER_GUIDE.md)
- [MINIMAL_LOOP_CONTRACT.md](/Users/armand/Development/RingRift/docs/architecture/MINIMAL_LOOP_CONTRACT.md)
- [deploy_minimal_loops.sh](/Users/armand/Development/RingRift/ai-service/scripts/deploy_minimal_loops.sh)
- [SCRIPT_INVENTORY.md](/Users/armand/Development/RingRift/docs/SCRIPT_INVENTORY.md)

For current trainer work directories, treat `progress.json` as the live stage-status file and `metrics.jsonl` as the durable iteration log.

This is an owner-facing snapshot for Armand. It is not a marketing document.

## April 10 Snapshot Sources

- Training nodes: `npm --silent run training:status -- --json --ssh`
- Product health: `npm --silent run smoke:product`
- Local infrastructure gate: `cd ai-service && PYTHONPATH=. python -m pytest tests/unit/ tests/contracts/ -x -q --timeout=120`
- Remote CI state: `gh run list -L 5`

The machine-readable training snapshot is checked in at [`docs/data/training_status.json`](/Users/armand/Development/RingRift/docs/data/training_status.json).

## April 10 Training Results

| Config       | Iteration |      Elo | Promotions | Latest Evidence                                            | Status                                      |
| ------------ | --------: | -------: | ---------: | ---------------------------------------------------------- | ------------------------------------------- |
| `hex8_2p`    |      `32` | `1967.6` |        `6` | Latest eval rejected at `40%` after `50` games             | Strongest result, but clearly plateaued     |
| `square8_2p` |      `31` | `1601.8` |        `2` | Latest completed promotion was `60%` after `50` games      | Real improvement, but node is currently off |
| `square8_3p` |      `13` | `1534.9` |        `1` | Recent seat-fair evals rejected at `22%`, `24%`, and `22%` | Regressing; weak evidence                   |
| `square8_4p` |       `5` | `1500.0` |        `0` | Latest completed eval was about `46%`                      | No proven improvement                       |

## April 10 Infrastructure Health

| Node                          | Config       | SSH Probe | Loop  | Supervisor | Heartbeat File | Read                              |
| ----------------------------- | ------------ | --------- | ----- | ---------- | -------------- | --------------------------------- |
| `gh200-8` / `100.121.230.110` | `hex8_2p`    | OK        | Alive | Alive      | `unknown`      | Healthy loop, but plateaued       |
| `gh200-9` / `100.127.168.116` | `square8_2p` | OK        | Dead  | Dead       | `unknown`      | Needs restart/debug               |
| `gh200-12` / `100.86.51.4`    | `square8_3p` | OK        | Alive | Alive      | `unknown`      | Live process, but weak evaluation |
| `gh200-10` / `100.100.19.96`  | `square8_4p` | OK        | Dead  | Dead       | `unknown`      | Needs restart/debug               |

## April 10 Product Health

The current product smoke against `https://ringrift.ai` passed on April 10, 2026:

- server health endpoint reachable
- AI proxy/replay stats endpoint reachable
- sandbox AI move returned `ai_type=gumbel_mcts` with `use_neural_net=True`
- replay-store smoke succeeded and the smoke replay was explicitly excluded from training
- local `canonical_hex8_2p` model loaded successfully through the Python AI service

## April 10 CI Status

Local verification is green:

- `tests/unit + tests/contracts`: `32716 passed, 94 skipped`

Remote GitHub Actions on `main` are currently red:

- `Supported Path` failed on `bb4c99be1`
- `.github/workflows/ci.yml` failed on `bb4c99be1`

The doc state should therefore be read as: local quality gates are strong, but remote CI still needs follow-up.

## P2P / Legacy Infrastructure State

- `ai-service/scripts/p2p_orchestrator.py` is now `2591` LOC, below the `<3000` target.
- The orchestrator currently delegates into `21` mixin modules totaling `12618` LOC.
- The Phase 8 verification gate passed after the test-infrastructure cleanup, so the decomposed legacy surface is substantially more auditable than it was at the start of Part 3.
- The minimal loop remains the supported training harness; the legacy coordinator/P2P stack is being kept and cleaned up for reuse, not for ownership of the core research claims.

## What Worked In This Snapshot

- Production sandbox AI can route through the Python AI service and use neural-backed Gumbel MCTS.
- Product smoke coverage exists and currently passes.
- Training observability exists through `training:status`, `training:dashboard`, `training:validate-db`, and `training:provenance`.
- Replay-data provenance and validation are inspectable without direct SQL work.
- The P2P orchestrator and the largest coordination modules are now behind passing size and contract checks.

## What Was Broken Or Unproven In This Snapshot

- `square8_2p` and `square8_4p` are not currently running.
- Supervisor heartbeat age reporting still returns `unknown` even on live nodes.
- `square8_3p` remains alive but is failing seat-fair evaluation badly enough that it should not be treated as a strong result.
- Remote GitHub Actions are currently failing on `main`.

## Next Actions From This Snapshot

- Fix the remote `Supported Path` and `ci.yml` failures on `main`.
- Restart or diagnose `square8_2p` and `square8_4p`.
- Keep claims tied to completed evaluations, not label inflation.
- Continue the remaining Part 3 phases in order and update the roadmap after each completion.
