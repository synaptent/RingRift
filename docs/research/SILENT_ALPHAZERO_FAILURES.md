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

## Failure 5: Quality Gate Resume With A Degenerate Denominator

Symptom:

- The `gh200-14` fv3 reference lane appeared to plateau at `1586.8` Elo after
  rejects at iters 8, 9, 10, and 11.
- Iter 12 then produced a strong staged-evaluation signal: `36` candidate wins,
  `14` best wins, `0` draws, `72.0%` win rate over `50` games, and
  `evaluation.decision == "promote"`.
- The model was not promoted. The quality gate recorded
  `passed=false`, `critical=true`, and
  `MODE_COLLAPSE: 100% of games share the same opening (1/1 games)`.
- Iter 13 then clean-promoted at `38/12/0`, `76.0%` over `50` games, with
  `quality gate passed`.

Code citation:

- `ai-service/scripts/lib/model_quality_gate.py:82-84` defines the opening
  length and mode-collapse threshold.
- `ai-service/scripts/lib/model_quality_gate.py:208-260` computes opening
  repetition and marks `MODE_COLLAPSE` as critical when the repeat rate exceeds
  the threshold.
- `ai-service/scripts/minimal_alphazero_loop.py:662-668` documents the resume
  limitation: eval checkpoints restore outcome counts, but move-level tracker
  data from pre-restart games is not recovered.
- `ai-service/scripts/minimal_alphazero_loop.py:1358-1360` attaches the
  quality tracker to staged evaluation; `:1380-1387` turns a critical verdict
  into `quality_blocked=True`; `:1414-1417` requires both
  `decision == "promote"` and `not quality_blocked`; `:1436-1437` writes the
  quality-gate record into `metrics.jsonl`.
- Test: `ai-service/tests/unit/scripts/test_model_quality_gate.py`
  (`test_small_opening_sample_does_not_trigger_mode_collapse`,
  `test_partial_tracker_skips_critical_value_head_check`) and
  `ai-service/tests/unit/scripts/test_minimal_alphazero_loop.py`
  (`test_staged_evaluate_resume_restores_quality_tracker_state`,
  `test_staged_evaluate_legacy_resume_marks_tracker_partial`).

Why it is dangerous:

- A normal promotion metric is green: the candidate beats the current best.
- The quality gate also looks decisive because the warning is labeled
  `MODE_COLLAPSE`.
- The denominator is the real signal: `1/1` means only one opening sequence was
  recorded, while the staged-evaluation outcome row counted `50` games.
- A reviewer can draw the wrong conclusion in either direction: "the game is
  flawed" or "the model collapsed" instead of "the move-level quality tracker
  lost coverage across an eval resume."

Fix shape:

- Persist `QualityGateTracker` state inside the staged-eval checkpoint, not
  only outcome counts.
- Add explicit coverage fields such as `opening_games_tracked`,
  `opening_sample_coverage`, and `move_tracking_complete`.
- If a legacy checkpoint resumes without tracker state, mark move-level
  tracking partial and skip critical behavioral/value-head gates for that
  iteration rather than blocking promotion on a one-game suffix.
- Keep seat fairness active across resume because per-seat outcomes are already
  replayed into the tracker.

RingRift fix:

- `4e1b7e20e fix(coordination): persist quality-gate tracker state across eval
resume` (2026-04-29): checkpointed quality-tracker state, partial-coverage
  fields, and legacy-resume guards that keep incomplete move/value samples from
  becoming critical blockers.

RingRift evidence:

- `docs/data/training_runs/2026-04-29/fv3_reference_gh200-14.metrics.jsonl`
  contains iters 1-13 copied from the live node.
- `docs/data/training_runs/2026-04-29/fv3_reference_gh200-14.iter012_resume_backfill.json`
  records the control-vs-treatment backfill classification: iter 12 was a
  strength-positive promotion blocked by the legacy partial-sample gate, while
  iter 13 clean-promoted.
- `docs/data/training_runs/2026-04-29/summary.csv` summarizes iter 12 as
  `decision=promote`, `promoted=False`, `win_rate=0.72`,
  `quality_gate_passed=False`, and iter 13 as `decision=promote`,
  `promoted=True`, `win_rate=0.76`, `quality_gate_passed=True`.
- `docs/data/training_runs/2026-04-29/fv3_reference_gh200-14.iter013_final.json`
  records the clean iter 13 promotion row.

## Failure 6: Fixed-Seat Hex Model Widened To Max Players

Symptom:

- The `gh200-8` v4 multiplayer retry launched as a 3-player run with
  `--num-players 3`, `--multi-player`, and a transferred 3-player v4
  checkpoint.
- Training failed before the first epoch with:
  `Model value head mismatch (after model creation): model.num_players=4 but
training expects 3 players`.
- The transfer artifact had 3-player value/rank tensors, but the training
  model-construction path widened hex multiplayer heads to `MAX_PLAYERS`.

Code citation:

- `ai-service/app/training/train_dataset_inference.py` set
  `hex_num_players = MAX_PLAYERS if multi_player else num_players`, and
  `app.training.train` passed that value into `create_training_model`.
- `ai-service/app/training/model_initializer.py` and the legacy
  `train_model_init.py` helper had the same hex widening rule.
- Tests:
  `ai-service/tests/unit/training/test_train_dataset_inference.py`,
  `ai-service/tests/unit/training/test_model_initializer.py`, and
  `ai-service/tests/unit/scripts/test_transfer_2p_to_4p.py`.

Why it is dangerous:

- The checkpoint can strict-load into a 3-player v4 model, so transfer-time
  verification passes.
- The launch command and NPZ metadata both say `3p`.
- Training silently constructs a different model shape than the experiment
  claims, and the error only appears after expensive selfplay has already
  produced data.

Fix shape:

- For hex models, treat multiplayer checkpoints as fixed-seat artifacts:
  construct value and rank heads for the actual run player count.
- Keep transfer tests that assert both top-level and versioned metadata are
  updated for 3-player targets.
- Validate model value-head metadata immediately after construction.

RingRift fix:

- 2026-04-30 current fix: hex model construction now uses the requested
  player count for hex v3/v4/v5 paths instead of widening to `MAX_PLAYERS`.

RingRift evidence:

- `docs/data/training_runs/2026-04-30/gh200-8_v4_retry_num_players_mismatch/`
  preserves the failed gh200-8 retry. The archived `error_index.txt` shows
  iter 1 and iter 2 both failed with `model.num_players=4 but training expects
3 players`.

## Failure 7: Training Failure Advanced To Fresh Selfplay

Symptom:

- After the v4 retry training subprocess raised the value-head mismatch,
  `minimal_alphazero_loop.py` logged `Training failed, skipping`.
- It then advanced to the next iteration and started another 100-game selfplay
  batch against unchanged `best.pth`.
- By the time the failure was stopped, the live lane had entered iter 3
  selfplay without ever producing `candidate_001.pth` or a promotion eval.

Code citation:

- `ai-service/scripts/minimal_alphazero_loop.py` handled training failure by
  incrementing `consec_failures` and continuing to the next iteration. The
  repeated-failure circuit breaker only checked at the top of the next
  iteration, after more selfplay could be spent.
- Test:
  `ai-service/tests/unit/scripts/test_minimal_alphazero_loop.py::test_loop_halts_after_training_failure_before_next_selfplay`.

Why it is dangerous:

- The loop is alive, GPU-active, and producing fresh JSONL/NPZ files.
- The metrics stream has no promotion/eval row because no candidate exists.
- Operators can mistake GPU utilization for productive search while the lane
  is repeatedly generating data from a frozen best model.

Fix shape:

- Treat training subprocess failure as terminal for the current loop run.
- Write `progress.json` with `stage=training_failed`, the error string, and
  the failure count before exiting.
- Require an operator or patched code path to relaunch the lane, so failure
  review happens before more selfplay is spent.

RingRift fix:

- 2026-04-30 current fix: the minimal loop now halts on training failure before
  starting another selfplay iteration against unchanged weights.

RingRift evidence:

- The same `gh200-8_v4_retry_num_players_mismatch` evidence archive records
  `progress.json` at `iteration=3`, `stage=selfplay_started` after iter 1 and
  iter 2 training failures, proving the silent advance pattern.

## Failure 8: Max-Slot Targets With Fixed-Seat Value Heads

Symptom:

- The `gh200-12` square8_3p lane resumed iter 43 after a node reboot and got
  past the previous disk I/O hang.
- Training then failed with:
  `multi_player_value_loss expects pred_values and target_values to share the
same shape; got pred_values=(512, 3) target_values=(512, 4)`.
- The model correctly produced a fixed 3-player value head, while the NPZ
  stored `values_mp` in a 4-slot max-player layout with the inactive fourth
  slot padded.

Code citation:

- `ai-service/app/ai/neural_losses.py::multi_player_value_loss` required the
  full prediction and target tensors to have identical widths before applying
  the active-player mask.
- `ai-service/app/ai/neural_losses.py::build_rank_targets` built rank targets
  at the raw `values_mp` width, which could also mismatch a fixed-seat rank
  head.
- Callers in `ai-service/app/training/train.py` and
  `ai-service/app/training/train_step.py` now pass the rank-head width when
  building rank targets.
- Tests:
  `ai-service/tests/test_multi_player_value_loss.py`,
  `ai-service/tests/unit/ai/test_neural_losses.py`, and
  `ai-service/tests/unit/training/test_train.py::TestBuildRankTargets`.

Why it is dangerous:

- Max-slot `values_mp` targets are valid training data as long as
  `num_players` identifies the active prefix.
- Fixed-seat models are also valid and preferable for 3-player artifacts.
- Requiring full-width equality confuses inactive padding with an active
  contract violation, so a healthy lane can fail only after selfplay/export
  already spent the iteration.

Fix shape:

- Slice predictions and targets to the shared active width before applying the
  active-player mask.
- Reject only when the requested active player count exceeds either tensor's
  width.
- Build rank targets to the model rank-head width for fixed-seat models.

RingRift fix:

- 2026-04-30 current fix: multiplayer value loss now tolerates inactive target
  padding, and rank-target generation can align to a fixed-seat model head.

RingRift evidence:

- `docs/data/training_runs/2026-04-29/gh200-12_sq8_3p_iter43_shape_mismatch/`
  preserves the failed iter 43 training logs and metrics.

## General Pattern

The common failure mode is not a crash. It is a mismatch between what the
experiment name claims and what the actual system is proving: sometimes the
training graph optimizes a different objective, and sometimes a quality gate
with incomplete internal state produces a false conclusion.

The reusable guardrail is to test contracts at four levels:

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
4. **Evaluation contract**: promotion requires both strength and behavioral
   sanity.
   - Staged eval can decide that a candidate is stronger.
   - Quality gates must still be allowed to block promotion when value-head
     health, seat fairness, or behavioral diversity fails.
   - Quality gates also need coverage metadata so incomplete tracker samples do
     not masquerade as full-evaluation evidence.

## Reproducer Index

| Failure                                                        | Commit                                                              | Test                                                                                                                                      |
| -------------------------------------------------------------- | ------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Hex v4 missing `rank_dist` in `forward()`                      | [`2a659360f`](https://github.com/an0mium/RingRift/commit/2a659360f) | `tests/unit/ai/test_neural_net_architectures.py::TestHexNeuralNet_v4`                                                                     |
| Minimal loop drops `--multi-player` for `num_players > 2`      | [`10ee06181`](https://github.com/an0mium/RingRift/commit/10ee06181) | `tests/unit/scripts/test_minimal_alphazero_loop.py`                                                                                       |
| `transfer_2p_to_4p.py` no-ops on v4 (only matches `value_fc2`) | [`19ceb1ceb`](https://github.com/an0mium/RingRift/commit/19ceb1ceb) | `tests/unit/scripts/test_transfer_2p_to_4p.py`                                                                                            |
| v5-heavy FiLM init compounds to saturation through SE blocks   | [`c7aa48f92`](https://github.com/an0mium/RingRift/commit/c7aa48f92) | `tests/unit/ai/test_v5_heavy_film_init.py`                                                                                                |
| Eval resume loses tracker coverage and blocks promotion        | Live guardrail evidence, copied 2026-04-29                          | `tests/unit/scripts/test_model_quality_gate.py`, `tests/unit/scripts/test_minimal_alphazero_loop.py`                                      |
| Hex multiplayer model construction widens 3p to max players    | Current 2026-04-30 fix                                              | `tests/unit/training/test_train_dataset_inference.py`, `tests/unit/training/test_model_initializer.py`                                    |
| Training failure advances to another selfplay iteration        | Current 2026-04-30 fix                                              | `tests/unit/scripts/test_minimal_alphazero_loop.py::test_loop_halts_after_training_failure_before_next_selfplay`                          |
| Fixed-seat model rejects max-slot inactive value padding       | Current 2026-04-30 fix                                              | `tests/test_multi_player_value_loss.py`, `tests/unit/ai/test_neural_losses.py`, `tests/unit/training/test_train.py::TestBuildRankTargets` |
