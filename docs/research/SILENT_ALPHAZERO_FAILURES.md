# Silent AlphaZero Failures Found In RingRift

This note collects implementation bugs that are broadly relevant outside
RingRift. Each bug allowed the system to keep running while invalidating the
scientific interpretation of a training lane.

## Failure 1: Defined Head, Missing Forward Output

Symptom:

- `HexNeuralNet_v4` had rank-distribution head parameters.
- Multiplayer training expected a rank distribution output.
- The forward pass did not compute and return `rank_dist`.

Why it is dangerous:

- Checkpoints contain plausible-looking tensors.
- Training and inference can continue if callers tolerate two-output models.
- The architecture looks multiplayer-aware in `state_dict` inspection while the
  actual forward contract is not.

Fix shape:

- Assert the model returns `(value, policy, rank_dist)` for v4.
- Add `return_features=True` coverage so probe/auxiliary paths preserve the
  same output contract.
- Run a dummy forward that checks `rank_dist.shape == (B, P, P)` and
  `rank_dist.sum(dim=-1) == 1`.

RingRift fix:

- `2a659360f fix(ai): return rank distribution from hex v4`

## Failure 2: Multiplayer Config Without Multiplayer Loss

Symptom:

- The minimal loop passed `--num-players 3`.
- It did not pass `--multi-player` to `app.training.train`.
- `train.py` only enables vector value loss and rank-distribution loss when
  `multi_player=True`.

Why it is dangerous:

- Logs correctly say the run is `3p`.
- NPZ target stats include `values_mp`.
- The model returns `rank_dist`.
- The training loss silently ignores the multiplayer targets and rank head.

Fix shape:

- Treat `num_players > 2` as a training-contract requirement, not just metadata.
- Unit-test the exact subprocess command emitted by the training loop.
- Include loss-mode flags in metrics/probe output.

RingRift fix:

- `10ee06181 fix(ai): enable multiplayer loss in minimal loop`

## Failure 3: Transfer Script That Can No-Op Silently

Symptom class:

- Player-count transfer scripts can inspect a checkpoint and produce a new file
  while skipping the intended tensors because key names or output-head versions
  differ.

Why it is dangerous:

- The command exits successfully.
- The target checkpoint exists.
- Downstream training starts from weights that are not what the operator thinks
  they are.

Fix shape:

- Report every transformed tensor by key.
- Fail if expected value/policy/rank head keys are absent.
- Compare source and target `state_dict` statistics and require at least one
  intended tensor family to change.
- Add a smoke test for each architecture/version pair.

RingRift status:

- Player-count transfer has dedicated resizing utilities for rank-distribution
  keys, but any new architecture version must add a transfer-contract test
  before being used as a scientific baseline.

## Failure 4: Initialization That Leaves Conditioning Paths Functionally Dead

Symptom class:

- Large models with FiLM/heuristic-conditioning paths can pass forward-shape
  tests while the conditioning path contributes near-zero or constant signal.

Why it is dangerous:

- The network trains and emits finite losses.
- The parameter count and architecture diagram look correct.
- The feature path has no practical effect until a targeted sensitivity test
  checks it.

Fix shape:

- Perturb only the conditioning input and require output deltas above a small
  threshold.
- Test both train and eval mode.
- Include initialized parameter norms for FiLM/conditioning layers in model
  inspection output.

RingRift status:

- v5-heavy should be presented as an experiment until its conditioning and
  transfer contracts are documented with the same rigor as v4.

## General Pattern

The common failure mode is not a crash. It is a mismatch between what the
experiment name claims and what the actual training graph optimizes.

The reusable guardrail is to test contracts at three levels:

1. Model contract: outputs, shapes, and sensitivity.
2. Training contract: CLI flags activate the intended losses.
3. Experiment contract: metrics record the exact loss weights, target stats,
   architecture version, feature version, and checkpoint provenance.
