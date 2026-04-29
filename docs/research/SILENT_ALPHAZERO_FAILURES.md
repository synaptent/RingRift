# Silent AlphaZero Failures Found In RingRift

This note collects implementation bugs that are broadly relevant outside
RingRift. Each bug allowed the system to keep running while invalidating the
scientific interpretation of a training lane.

## Failure 1: Defined Head, Missing Forward Output

Symptom:

- `HexNeuralNet_v4` had rank-distribution head parameters
  (`rank_dist_fc1`, `rank_dist_fc2`, `rank_dist_fc3`, `rank_softmax`)
  fully defined in `__init__` and saved to every checkpoint.
- Multiplayer training expected a rank-distribution output.
- The forward pass did not compute or return `rank_dist`.

Code citation:

- `ai-service/app/ai/neural_net/hex_architectures.py` — `HexNeuralNet_v4.forward`
  signature pre-fix returned `(v_out, policy_logits)` while the class held
  `rank_dist_fc*` weights. The fix added the rank computation block
  (`rank_hidden = self.relu(self.rank_dist_fc1(combined)) ...`) and changed the
  return to `(v_out, policy_logits, rank_dist)`. Probe paths
  (`return_features=True`) were updated to keep the same contract.
- Test: `ai-service/tests/unit/ai/test_neural_net_architectures.py::TestHexNeuralNet_v4`.

Why it is dangerous:

- Checkpoints contain plausible-looking tensors with the expected shapes.
- Training and inference continue silently if callers tolerate two-output models.
- The architecture looks multiplayer-aware under `state_dict` inspection while
  the actual forward contract is not.

Fix shape:

- Assert the model returns `(value, policy, rank_dist)` for v4 in unit tests.
- Add `return_features=True` coverage so probe/auxiliary paths preserve the
  same output contract.
- Run a dummy forward that checks `rank_dist.shape == (B, P, P)` and
  `rank_dist.sum(dim=-1) ≈ 1`.

RingRift fix:

- `2a659360f fix(ai): return rank distribution from hex v4` (2026-04-27).

## Failure 2: Multiplayer Config Without Multiplayer Loss

Symptom:

- The minimal loop launched training with `--num-players 3` (or 4) on the
  command line and in NPZ metadata.
- It did **not** pass `--multi-player` to the underlying
  `python -m app.training.train` subprocess.
- `app.training.train` only enables vector value loss and rank-distribution
  loss when `multi_player=True`.
- For `NUM_PLAYERS > 2` runs, the training graph silently fell back to the
  scalar two-player loss.

Code citation:

- `ai-service/scripts/minimal_alphazero_loop.py` — inside `train_model()`,
  the subprocess command builder did not branch on player count. The fix
  appended the flag conditionally:
  `if NUM_PLAYERS > 2: cmd.append("--multi-player")` immediately after the
  base argument list.
- Test: `ai-service/tests/unit/scripts/test_minimal_alphazero_loop.py`
  asserts the emitted subprocess command for 3p/4p runs.

Why it is dangerous:

- Logs correctly say the run is `3p`.
- NPZ target stats include `values_mp` and per-player ranks.
- The model returns `rank_dist`.
- The training loss silently ignores the multiplayer targets and rank head,
  invalidating the scientific interpretation of every multiplayer iter.

Fix shape:

- Treat `num_players > 2` as a training-contract requirement, not just
  metadata.
- Unit-test the exact subprocess command emitted by the training loop.
- Include loss-mode flags in metrics/probe output so the actual training
  configuration is recoverable from artifacts.

RingRift fix:

- `10ee06181 fix(ai): enable multiplayer loss in minimal loop` (2026-04-28).

## Failure 3: Transfer Script That Can No-Op Silently

Symptom:

- `scripts/transfer_2p_to_4p.py` was written when value heads ended at
  `value_fc2`. When v4 added `value_fc3` and a separate `rank_dist_fc*` head,
  the transfer script kept matching only `value_fc2`.
- For v4 inputs the script ran end-to-end, wrote a "transferred" output
  checkpoint, and exited 0.
- The actual target tensors (the v4 `value_fc3` and `rank_dist_fc3`) were
  never touched. Downstream training started from weights that did not match
  the requested player count.

Code citation:

- `ai-service/scripts/transfer_2p_to_4p.py` — pre-fix `infer_source_players`
  only walked `value_fc2`. The fix loops over both `value_fc3` and
  `value_fc2`, adds rank-head detection (`rank_dist_fc3`/`rank_dist_fc2`
  shapes match `players * players`), and adds explicit
  `_is_final_value_head_key` / `_is_rank_distribution_head_key` predicates
  so each transformed tensor is identified by name. The fix also strict-loads
  the resulting `state_dict` into a fresh target model so incompatible
  artifacts fail at generation time instead of at training time.
- Test: `ai-service/tests/unit/scripts/test_transfer_2p_to_4p.py` covers
  both v2 (legacy) and v4 (multi-head) transfer contracts.

Why it is dangerous:

- The command exits successfully.
- The target checkpoint exists.
- Downstream training starts from weights that are not what the operator
  thinks they are, and the discrepancy will not show up in any single iter
  of selfplay.

Fix shape:

- Report every transformed tensor by key.
- Fail loudly if expected value/policy/rank head keys are absent.
- Compare source and target `state_dict` statistics and require at least one
  intended tensor family to change.
- Strict-load the produced `state_dict` into the target architecture before
  saving it.
- Add a smoke test for each architecture/version pair.

RingRift fix:

- `19ceb1ceb fix(training): resize multiplayer transfer rank heads`
  (2026-04-24).

## Failure 4: Initialization That Saturates The Value Head At Birth

Symptom:

- v5-heavy checkpoints exhibited `DEAD_VALUE_HEAD` (training probe
  `value_std == 0.000000`, `mean ≈ ±1.0`) on every iter, with no recovery
  after epochs of training.
- FP16 autocast emitted non-finite warnings every iter.
- Weight delta L2 between consecutive iters spiked to ~140+, then collapsed.

Code citation:

- `ai-service/app/ai/neural_net/v5_heavy.py` — `HeuristicEncoder.__init__`
  pre-fix used:
  ```python
  nn.init.ones_(self.film_gamma.weight.data[:, :output_dim // 2])
  nn.init.zeros_(self.film_gamma.weight.data[:, output_dim // 2:])
  ```
  The intent was to make FiLM behave as identity (`gamma ≈ 1`, `beta ≈ 0`),
  but because the heuristic encoder output is ReLU-positive, half-ones in
  `film_gamma.weight` produced a _nonzero_ gamma at birth (`gamma ≈ 3.3`).
  Compounded through six SE blocks, the value head saturated to ±1.0
  before the first gradient step. Post-fix:
  ```python
  nn.init.zeros_(self.film_gamma.weight)
  nn.init.zeros_(self.film_gamma.bias)
  nn.init.zeros_(self.film_beta.weight)
  nn.init.zeros_(self.film_beta.bias)
  ```
  All zeros yields a true identity FiLM at init.
- Test: `ai-service/tests/unit/ai/test_v5_heavy_film_init.py` asserts
  `film_gamma.weight.abs().max() == 0` on freshly constructed models and
  that `value.abs().max() < 0.5` on a random-input forward pass.

Why it is dangerous:

- The network trains, the architecture matches the intended diagram, and the
  parameter count is correct.
- Forward-shape tests pass. The `state_dict` looks healthy.
- Saturation only shows up under a _targeted_ probe (value head std after
  one epoch).
- Any v5-heavy checkpoint saved before the fix is **poisoned** and cannot be
  reused — the bad `film_gamma.weight` is baked in.

Fix shape:

- Smoke-test new architectures with a synthetic forward pass that requires
  `value.abs().max() < 0.5` and `policy.std() < 2.0` on random input.
- For FiLM/conditioning layers, perturb only the conditioning input and
  require nonzero output deltas (sensitivity test).
- Include initialized parameter norms for FiLM/conditioning layers in model
  inspection output.
- Treat all checkpoints saved before the init fix as poisoned; do not reuse
  them as starting weights.

RingRift fix:

- `c7aa48f92 fix(ai): initialize v5-heavy FiLM as identity` (2026-04-24).

## General Pattern

The common failure mode is not a crash. It is a mismatch between what the
experiment name claims and what the actual training graph optimizes.

The reusable guardrail is to test contracts at three levels:

1. **Model contract**: outputs, shapes, and sensitivity.
   - Forward returns the documented tuple shape for every supported
     `return_features` value.
   - Conditioning paths produce nonzero output deltas under input
     perturbation.
   - Random-input forward keeps `value.abs().max()` and `policy.std()`
     within sane bounds at init.
2. **Training contract**: CLI flags activate the intended losses.
   - Unit-test the exact subprocess command emitted by any wrapper
     loop or training launcher.
   - Assert `multi_player`, `rank_dist`-loss, and per-head loss weights
     match the experiment configuration.
3. **Experiment contract**: metrics record the exact loss weights, target
   stats, architecture version, feature version, and checkpoint provenance.
   - Every artifact (checkpoint, NPZ, log line) should let an outsider
     reconstruct _what was actually optimized_, not just _what was
     intended_.

## Reproducer Index

| Failure                                                        | Commit                                                              | Test                                                                  |
| -------------------------------------------------------------- | ------------------------------------------------------------------- | --------------------------------------------------------------------- |
| Hex v4 missing `rank_dist` in `forward()`                      | [`2a659360f`](https://github.com/an0mium/RingRift/commit/2a659360f) | `tests/unit/ai/test_neural_net_architectures.py::TestHexNeuralNet_v4` |
| Minimal loop drops `--multi-player` for `num_players > 2`      | [`10ee06181`](https://github.com/an0mium/RingRift/commit/10ee06181) | `tests/unit/scripts/test_minimal_alphazero_loop.py`                   |
| `transfer_2p_to_4p.py` no-ops on v4 (only matches `value_fc2`) | [`19ceb1ceb`](https://github.com/an0mium/RingRift/commit/19ceb1ceb) | `tests/unit/scripts/test_transfer_2p_to_4p.py`                        |
| v5-heavy FiLM init compounds to saturation through SE blocks   | [`c7aa48f92`](https://github.com/an0mium/RingRift/commit/c7aa48f92) | `tests/unit/ai/test_v5_heavy_film_init.py`                            |
