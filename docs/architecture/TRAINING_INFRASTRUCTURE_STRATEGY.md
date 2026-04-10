# Training Infrastructure Strategy

RingRift currently has two training infrastructure paths:

- `ai-service/scripts/minimal_alphazero_loop.py`, which is the narrow, reproducible harness used for the current reported AlphaZero-style results.
- The larger coordinator and P2P system under `ai-service/scripts/master_loop.py`, `ai-service/scripts/p2p/**`, and `ai-service/app/coordination/**`, which contains substantial reusable orchestration work.

The project should not keep reinventing cluster infrastructure around the minimal loop. The minimal loop is useful because it is easy to reason about and has produced defensible results. It should remain the supported experiment harness. The larger infrastructure is useful because it already contains supervision, work queues, training dispatch, candidate model handling, S3 transfer, cluster health, and evaluation scheduling. It should be reused where it satisfies the current contracts.

## Current Decision

Use the minimal loop for reproducible proof runs and live canaries.

Use the larger coordinator/P2P system as the source of reusable infrastructure, not as an automatic replacement for the proof harness.

Do not switch active proof runs back to the full coordinator until the coordinator path is verified against the same rules and experiment contracts that made the minimal-loop results credible.

## Required Contracts For Reuse

Any coordinator/P2P path used for current research results must satisfy all of these:

- It uses the corrected Python rules mirror, including canonical territory thresholds and stalemate tiebreak behavior.
- It does not train from legacy or ambiguous replay data unless the run is explicitly marked experimental.
- It writes candidates to `candidate_*` paths and promotes only after evaluation.
- It supports fixed learning-rate runs without hidden decay when the experiment requires `--lr-schedule fixed`.
- It supports separate selfplay and evaluation budgets when the experiment is designed that way.
- It supports short, explicit training windows when post-fix data purity matters.
- Multiplayer evaluation is seat-fair, with candidate seat rotation handled deliberately rather than accidentally.
- It has safe process restart behavior and does not kill unrelated Python/selfplay/P2P jobs.
- It emits enough metrics to reconstruct the exact experiment parameters used for each result.

## What To Reuse First

The most promising pieces of the larger system are:

- P2P work queue and job lifecycle management.
- Training executor candidate handling and S3 push/fetch logic.
- Cluster health and node status monitoring.
- Evaluation worker scheduling.
- Disk, memory, and data-sync guards.
- Existing daemon profiles, especially `lean`, after auditing the specific daemon list.

These are infrastructure concerns. They should be harvested into the supported path or run alongside it only when they do not obscure experiment provenance.

## What Not To Reuse Blindly

Do not blindly route results through older NNUE or general training handlers that:

- pass `--allow-noncanonical` by default,
- write directly to `canonical_*` paths before evaluation,
- run auto-promotion with a different gauntlet from the current staged/seat-fair evaluator,
- discover datasets implicitly from broad filesystem globs,
- use older training defaults that are not encoded in the current minimal-loop metrics,
- start broad daemon profiles whose side effects are not needed for a given experiment.

Those code paths may still be useful historically or after targeted repair, but they should not be used to support the current research claim without an explicit audit.

## Near-Term Plan

1. Keep the current minimal-loop canaries running under safe, process-local supervision.
2. Audit the coordinator/P2P training path against the required contracts above.
3. Reuse the specific mature pieces that solve real problems, starting with health, job lifecycle, candidate transfer, and evaluation scheduling.
4. Only after the audited path matches the minimal-loop experiment semantics, promote it to a supported training path in the README and architecture docs.

This preserves the value of the large infrastructure without letting historical complexity contaminate the current evidence.
