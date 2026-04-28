# Outsider Value Roadmap

This roadmap is about making RingRift useful to people who did not live through
the project history. The goal is not more breadth. The goal is a smaller,
clearer surface that proves the project is playable, reproducible, and
technically interesting.

## Priority 0: Keep The Coordinator Boring

External value depends on a stable repo and stable evidence pipeline. The
mac-studio coordinator must not silently fill disk or restart p2p in a loop.

Current safeguards:

- Remote-to-internal DB rehydration is opt-in through `data/sync_policy.yaml`.
- Launchd user logs are capped.
- Hidden stale gauntlet temp DBs are cleaned by `DiskSpaceManagerDaemon`.
- `node_resilience.py` now uses a 90% disk cleanup threshold and reports
  non-empty cleanup failures.

Definition of done:

- `df -h /System/Volumes/Data` stays above 50 GB free during normal operation.
- `~/Library/Logs/RingRift/p2p.log` stays below the configured cap.
- `com.ringrift.resilience`, `com.ringrift.p2p`, and
  `com.ringrift.master-loop` can run together without a restart loop.

## Priority 1: Make The First 30 Minutes Work

An outsider should be able to do three things without understanding the cluster:

1. Play the game:
   `npm run play`
2. Inspect the supported training path:
   `./scripts/run_proven_experiment.sh square8_2p --print-only`
3. Run a small model evaluation:
   `./scripts/run_quick_eval.sh hex8_2p --games 12`

The quick evaluation is intentionally not an Elo claim. It is a smoke/eval path
that proves a canonical checkpoint loads and plays against a built-in baseline.
It must pass checksum-sidecar verification before any result is publishable.

## Priority 2: Publish The Reusable Research Lessons

The most reusable value is not the current strongest model. It is the failure
analysis around silent AlphaZero implementation bugs:

- A model can define a head and never return it.
- A training loop can pass `--num-players 3` while silently disabling
  multiplayer loss.
- Transfer scripts can appear to run while not touching the intended target
  tensors.
- Initialization choices can make a large architecture look alive while key
  conditioning paths are effectively dead.

The canonical writeup is
[`docs/research/SILENT_ALPHAZERO_FAILURES.md`](research/SILENT_ALPHAZERO_FAILURES.md).
The v4-specific retry protocol is
[`docs/research/V4_MULTIPLAYER_DIAGNOSTIC.md`](research/V4_MULTIPLAYER_DIAGNOSTIC.md).

## Priority 3: Finish One Clean Replication Slice

Do not add new architectures until one replication slice is presentable.

Recommended stopping condition:

- Keep the current fv3 seed/reference lanes running to a clean milestone.
- Summarize promotion rate, plateau timing, and value-head health.
- Plot trajectories from `metrics.jsonl` rather than narrating individual logs.

Current snapshot:

- [`docs/research/FV3_REPLICATION_SNAPSHOT_2026-04-28.md`](research/FV3_REPLICATION_SNAPSHOT_2026-04-28.md)

The public claim should be narrow:

> RingRift self-play improvement is reproducible on the supported small-board
> two-player path; multiplayer training is mechanically validated but not yet
> a solved strength result.

## Priority 4: Package Public Artifacts

Public artifacts should be small and directly runnable:

- `canonical_square8_2p.pth`
- `canonical_hex8_2p.pth`
- one parity fixture DB or fixture bundle
- `scripts/run_quick_eval.sh`
- `scripts/run_proven_experiment.sh`
- `docs/RESULTS.md`
- `docs/REPRODUCIBILITY.md`
- `docs/research/SILENT_ALPHAZERO_FAILURES.md`

If a model is too large for Git, publish it as a release asset and keep a
checksum manifest in the repo.

Current artifact gate:

- `canonical_hex8_2p.pth` matches its checked-in `.sha256` sidecar and is the
  current quick-eval default. Its top-level checkpoint metadata has been
  normalized to `hex8/2p`, and it loads through the public evaluator.
- `canonical_square8_2p.pth` now matches its `.sha256` sidecar and loads through
  the public evaluator. The sidecar was regenerated from the active local file
  after confirming that the active file and backup were byte-identical. Its
  checkpoint metadata has also been normalized to `square8/2p`.
- The verified artifacts are published under the GitHub release
  [`public-model-artifacts-2026-04-28`](https://github.com/synaptent/RingRift/releases/tag/public-model-artifacts-2026-04-28).
  The `.pth` files are split into 1 MiB parts because direct large-asset uploads
  failed from the publishing host; the release includes reconstruction
  instructions.
- The public artifact gate is:
  `cd ai-service && PYTHONPATH=. python scripts/audit_public_model_artifacts.py`.

## What Not To Optimize Yet

- Do not expand the GPU fleet until the existing results are analyzed.
- Do not add another architecture family before v4/fv3/v5-heavy are explained.
- Do not polish every daemon. Move unsupported operational surfaces behind a
  manifest and make the supported path obvious.
- Do not claim multiplayer success until promotion trajectories support it.
