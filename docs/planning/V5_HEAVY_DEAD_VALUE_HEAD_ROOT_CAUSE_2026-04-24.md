# V5-Heavy Dead Value Head Root Cause - 2026-04-24

## Finding

The gh200-14 `hex8_2p_v5_heavy_fv3` lane repeatedly failed the training probe with
constant value outputs (`mean=-1` or `mean=+1`, `std=0`) and non-finite FP16
outputs even after the fv3 runtime-contract fixes. The fv3 encoder path was not
the root cause.

The root cause was the v5-heavy `HeuristicEncoder` FiLM initialization in
`ai-service/app/ai/neural_net/v5_heavy.py`. The code comment said FiLM should
start as identity (`gamma=1`, `beta=0`), but half of `film_gamma.weight` was
initialized to ones. Because the heuristic encoder output is ReLU-positive, a
fresh checkpoint produced gamma values around `3.3` before training.

Hex v5-heavy applies that gamma after every SE block. With six SE blocks, the
non-identity scale compounded and produced saturated value outputs and very
large policy logits at checkpoint birth.

## Local Canary

Using gh200-14's stopped fv3 artifact and `iter_001.npz`:

- Before the fix: `gamma mean ~= 3.34`, policy-logit std `~= 11.8`, values near
  saturation.
- With true identity FiLM: `gamma=1`, `beta=0`, value std `~= 0.02`, policy-logit
  std `~= 0.11`.

## Operational Consequence

Do not reuse existing v5-heavy canonicals or `best.pth` files created before
this fix. They have the bad `film_gamma.weight` baked into the checkpoint.

For gh200-14 or any future v5-heavy lane:

1. Pull the fix.
2. Regenerate a fresh v5-heavy canonical from patched code.
3. Smoke-test the fresh checkpoint before training:
   - FiLM gamma must be exactly `1`.
   - FiLM beta must be exactly `0`.
   - Fresh value outputs must be finite and not saturated.
   - Fresh policy logits should be small-scale, not O(100).
4. Restart from a clean work directory.
