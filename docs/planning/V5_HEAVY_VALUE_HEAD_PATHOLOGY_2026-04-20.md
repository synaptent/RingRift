# V5-Heavy Value Head Pathology - 2026-04-20

## Scope

This document diagnoses the `hex8_2p_v5_heavy` failure mode observed on `gh200-11`.

Constraints for this pass:

- Read-only diagnosis only.
- No architecture edits yet.
- No trainer-loop edits yet.
- Goal: explain `DEAD_VALUE_HEAD` and identify the most likely fault domain to fix first.

## Executive Summary

The strongest current evidence does **not** support "the trained v5-heavy value head is intrinsically dead" as the primary diagnosis.

The stronger diagnosis is:

1. Runtime inference on `gh200-11` is repeatedly failing with a `40 -> 64` channel mismatch.
2. The legacy v5-heavy inference path initializes a `HexStateEncoder` v2 encoder (`40` channels) even when the loaded checkpoint clearly expects `64` channels.
3. `tensor_gumbel_tree` catches that failure and falls back to `root_value = 0.0`.
4. The quality gate records those fallback root values during evaluation, so `DEAD_VALUE_HEAD` is being triggered by runtime fallback behavior, not yet by a proven training-time dead head.

This explains the observed `value_mean = 0.0` and `value_std = 0.0` on `gh200-11` much more directly than a loss-scaling or heuristic-normalization theory.

There is still a separate concern: the aggregate training loss on v5-heavy (`~92-464`) is far above other active configs (`~3-6`). That is suspicious, but the currently strongest, concrete bug is the runtime encoder mismatch.

## Live Evidence

### 1. Dead value-head signal is real on completed iterations

From `gh200-11:/home/ubuntu/ringrift/ai-service/data/minimal_loop_hex8_2p_v5_heavy/metrics.jsonl`:

- Iteration `5`: `win_rate = 0.475`, `value_samples = 7315`, `value_mean = 0.0`, `value_std = 0.0`
- Iteration `6`: `win_rate = 0.475`, `value_samples = 7315`, `value_mean = 0.0`, `value_std = 0.0`
- Iteration `7`: `win_rate = 0.475`, `value_samples = 7315`, `value_mean = 0.0`, `value_std = 0.0`

So the quality gate is consistently seeing a constant zero root value during evaluation.

Relevant code:

- `ai-service/scripts/lib/model_quality_gate.py:261-308`
- `ai-service/scripts/minimal_alphazero_loop.py:695-700`

### 2. Training targets are healthy, not all-zero

From `gh200-11:/home/ubuntu/ringrift/ai-service/data/minimal_loop_hex8_2p_v5_heavy/iter_007.npz`:

- `features.shape = (2169, 64, 9, 9)`, mean `0.0924`, std `0.2751`
- `globals.shape = (2169, 20)`, mean `0.2592`, std `0.3827`
- `values.shape = (2169,)`, mean `0.0124`, std `0.9999`, min `-1.0`, max `1.0`
- `values_mp.shape = (2169, 4)`, mean `0.0`, std `0.7071`
- `heuristics.shape = (2169, 49)`, mean `0.00246`, std `0.0326`, min `-0.8`, max `0.9`

Conclusions:

- Value targets are not collapsed to zero.
- Heuristic inputs are already small in magnitude.
- Dataset-side "all-zero values" is ruled out.

### 3. Live runtime logs show repeated 40-channel input into a 64-channel checkpoint

From `gh200-11:/home/ubuntu/ringrift/ai-service/logs/training.log`:

Repeated warnings:

`NN batch evaluation failed: Given groups=1, weight of size [160, 64, 3, 3], expected input[2, 40, 9, 9] to have 64 channels, but got 40 channels instead, using heuristic fallback`

Representative log region:

- lines around `326416-326574`

This is the clearest direct failure signal in the live system.

## Code Path Analysis

### A. The quality gate is measuring runtime root values, not direct training-head gradients

`QualityGateTracker.record_move(...)` stores `root_value` into `_values`.

Relevant code:

- `ai-service/scripts/lib/model_quality_gate.py:112-142`

During staged evaluation, the loop pulls `root_value` from the candidate AI's last search stats and records it:

- `ai-service/scripts/minimal_alphazero_loop.py:695-700`

The dead-head warning is then triggered if the standard deviation of those recorded values is below the threshold:

- `ai-service/scripts/lib/model_quality_gate.py:286-293`

Implication:

- `DEAD_VALUE_HEAD` here means "runtime search root values were constant/near-constant during evaluation".
- It does **not** by itself prove that the trained model's value head weights are frozen or detached.

### B. Runtime search falls back to zero on NN evaluation failure

`tensor_gumbel_tree` catches NN batch evaluation failures and uses a heuristic fallback with `root_value = 0.0`.

Relevant code:

- `ai-service/app/ai/tensor_gumbel_tree.py:914-928`
- `ai-service/app/ai/tensor_gumbel_tree.py:924`

Observed live warning text matches this path exactly.

Implication:

- If runtime inference fails repeatedly, the quality gate will see repeated zeros even if the trained model itself is not emitting zeros.

### C. Legacy v5-heavy runtime initialization is wiring the wrong encoder

`NeuralNetAI` dispatches `nn_model_version = "v5-heavy"` into `_init_v5_heavy_model(...)`:

- `ai-service/app/ai/_neural_net_legacy.py:3309-3314`

Inside `_init_v5_heavy_model(...)`:

1. It correctly peeks `conv1.weight` and discovers the checkpoint expects `64` input channels.
   - `ai-service/app/ai/_neural_net_legacy.py:4383-4407`
2. It builds the v5-heavy model with that `64`-channel expectation.
   - `ai-service/app/ai/_neural_net_legacy.py:4415-4432`
3. But it then unconditionally initializes the hex encoder as:
   - `HexStateEncoder(..., feature_version=2)`
   - `self.feature_version = 2`
   - `ai-service/app/ai/_neural_net_legacy.py:4474-4482`

That encoder is the v2 hex encoder family, which is `40` channels, not `64`.

The encoder registry makes the intended distinction explicit:

- Hex v2: `40` channels
  - `ai-service/app/training/encoder_registry.py:92-113`
- Hex v3: `64` channels
  - `ai-service/app/training/encoder_registry.py:115-136`
- Hex v5-heavy: also `64` channels, but disambiguated by heuristic metadata
  - `ai-service/app/training/encoder_registry.py:161-170`
  - `ai-service/app/training/encoder_registry.py:391-463`

This is the most concrete code defect found in the diagnosis.

## What This Explains Well

### Explained strongly

- Why `DEAD_VALUE_HEAD` fires with `value_mean = 0.0` and `value_std = 0.0`
- Why logs show constant `40 -> 64` channel mismatch warnings
- Why the runtime can appear "dead" even though dataset values are healthy

### Not fully explained yet

- Why aggregate v5-heavy training loss is so large (`~92-464`) relative to other configs
- Whether the trained v5-heavy value head is also weak in addition to the runtime mismatch
- Whether self-play generation on this lane is also hitting the same fallback path and contaminating training data quality

Those remain open questions.

## Ruled-Out or Weakened Hypotheses

### 1. All-zero value targets

Ruled out by `iter_007.npz`:

- `values.std ~= 0.9999`
- `values` span `[-1.0, 1.0]`

### 2. Heuristic features are grossly unnormalized and saturating FiLM by magnitude alone

Weakened, not fully ruled out:

- Stored heuristic features are small: std `0.0326`, min `-0.8`, max `0.9`
- `HeuristicEncoder.forward()` normalizes heuristics per sample before encoding
  - `ai-service/app/ai/neural_net/v5_heavy.py:198-216`

This does not support a simple "heuristics are magnitude-100 and blow up FiLM" story.

### 3. Obvious detach/frozen value-head bug inside `v5_heavy.py`

Not supported by current code inspection:

- No obvious `.detach()` on the value path
- No obvious `requires_grad = False`
- No obvious forced `.eval()` misuse inside the forward pass

That does not prove the architecture is correct, but it lowers the priority of changing `v5_heavy.py` first.

## Likely Fix Ownership

### Primary suspected fix domain: runtime / inference path

Most likely first fix:

- `ai-service/app/ai/_neural_net_legacy.py`

Reason:

- It is explicitly constructing a 40-channel hex encoder for a 64-channel v5-heavy checkpoint.

### Secondary hardening: checkpoint metadata / encoder resolution

Likely follow-up hardening:

- `ai-service/app/training/model_versioning.py`
- `ai-service/app/training/encoder_registry.py`

Reason:

- The live checkpoint metadata on `gh200-11` is stale/incomplete (`in_channels`, `feature_version`, `num_heuristics`, `architecture_version` all missing or `None`).
- `detect_model_version_from_channels(...)` defaults `64`-channel hex checkpoints to `"v4"` unless heuristic metadata is present.
- That is survivable when metadata is correct, but fragile when it is not.

## Concrete Code Changes To Test Next

These are proposed follow-ups, not part of this diagnosis pass.

### Option 1: Fix the legacy v5-heavy runtime encoder selection

In `ai-service/app/ai/_neural_net_legacy.py`, update `_init_v5_heavy_model(...)` so that:

- `in_channels == 64` on hex selects the v3-family encoder (`HexStateEncoderV3`) rather than `HexStateEncoder`
- v5-heavy continues to attach the expected heuristic vector path (`49` features)

This is the smallest and highest-confidence fix candidate.

### Option 2: Derive encoder from checkpoint contract, not version string comments

Instead of hardcoding:

- `HexStateEncoder(feature_version=2)`

derive the runtime encoder from:

- checkpoint `in_channels`
- checkpoint metadata if present
- v5-heavy heuristic requirements

This is less brittle than maintaining special-case comments about "v2 encoder with 10 base channels".

### Option 3: Add regression coverage

Add a test that loads a hex v5-heavy checkpoint contract with:

- `conv1.weight.shape[1] == 64`
- `49` heuristic features

and asserts:

- runtime encoder emits `64` channels
- no `40 -> 64` mismatch occurs
- no heuristic fallback path is taken

### Option 4: Harden checkpoint metadata persistence

Ensure v5-heavy checkpoints persist enough metadata to reconstruct the encoder contract without guessing:

- `in_channels`
- `feature_version`
- `num_heuristics`
- `architecture_version`

This is a hardening change, not the first fix.

## Recommended Next Step Order

1. Fix the runtime encoder mismatch in `NeuralNetAI._init_v5_heavy_model(...)`.
2. Re-run a focused v5-heavy evaluation or smoke test and verify:
   - no `40 -> 64` channel mismatch warnings
   - `value_head_health.value_std` is no longer `0.0`
3. Only after that, reassess the remaining high-loss question.
4. If aggregate loss remains pathological after runtime inference is fixed, then inspect:
   - per-component losses
   - self-play data quality on the v5-heavy lane
   - architecture-specific optimization behavior

## Bottom Line

The current best-supported diagnosis is:

- `v5-heavy` on `gh200-11` is suffering from a **runtime encoder mismatch** in the legacy inference path.
- That mismatch is causing repeated NN evaluation failures.
- The evaluator falls back to `root_value = 0.0`.
- The quality gate then reports `DEAD_VALUE_HEAD`.

So the first fix should be in the runtime encoder contract, not in the v5-heavy head architecture itself.
