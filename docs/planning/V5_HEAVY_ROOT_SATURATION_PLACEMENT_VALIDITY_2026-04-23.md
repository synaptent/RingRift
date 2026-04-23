# V5-Heavy Root Saturation: Placement-Validity Shortcut

Date: 2026-04-23

## Summary

`gh200-11`'s `v5-heavy` value-head issue is not a global dead-head failure.
The stronger diagnosis is a **distribution-specific shortcut** in the hex V3
encoder contract: the `placement_validity` planes (channels `14/15`) are so
predictive that the model can drive root values from them alone, especially
through the history stack.

## Evidence

Using `candidate_005.pth` against `iter_005.npz` on `gh200-11`:

- `zero_globals` had essentially no effect on value outputs.
- `zero_board` collapsed outputs toward a negative constant.
- Zeroing **all** `placement_validity` channels was the only channel-group
  ablation that flipped the sign of the midgame bucket.

More specifically:

- Early bucket (`move_numbers <= 2`):
  - baseline: `mean=-0.9964`, `std=0.0051`
  - zero current-frame placement channels: `mean=-0.9709`, `std=0.0401`
  - zero history-frame placement channels: no change
  - conclusion: early negative saturation is driven almost entirely by the
    **current frame** `placement_validity` planes

- Mid bucket (`move_numbers 3..5`):
  - baseline: `mean=+0.9974`, `std=0.0038`
  - zero current-frame placement channels: no material change
  - zero history-frame placement channels: flips to `mean=-0.9965`, `std=0.0050`
  - conclusion: mid positive saturation is driven almost entirely by the
    **history-frame** `placement_validity` planes

This is much more specific than the earlier "seat skew" story. Seat imbalance
may still correlate with the failure, but the concrete shortcut lives in the
encoder planes.

## Root Cause

In [encoding.py](/Users/armand/Development/RingRift/ai-service/app/training/encoding.py:1017),
`HexStateEncoderV3` populates channels `14/15` with:

- a binary "can place ring here" mask
- a related accessibility score

Those planes are then stacked across history into the `64`-channel hex V3/V4/V5
family input. The result is a very strong proxy for game progress and opening
structure that the value head can exploit instead of learning useful position
evaluation.

## Fix Implemented

I added a new opt-in contract:

- `HexStateEncoderV3(feature_version=3)` now leaves channels `14/15` zeroed
- `feature_version=2` remains unchanged for backward compatibility

This preserves the `64`-channel tensor shape while removing the shortcut.

Why opt-in:

- changing `feature_version=2` in place would invalidate live hex `v3/v4/v5`
  checkpoints that were trained on the old semantics
- new training runs can adopt `feature_version=3` explicitly, and checkpoint
  metadata will preserve that choice across runtime loading

## Tests

Added unit coverage in
[test_encoding.py](/Users/armand/Development/RingRift/ai-service/tests/unit/training/test_encoding.py:447):

- `feature_version=2` still produces non-zero placement-validity planes
- `feature_version=3` zeroes those planes
- non-placement channels and globals remain unchanged

## Recommended Follow-Up

1. Do not hot-swap existing `feature_version=2` checkpoints onto this encoder.
2. For the next clean hex `v5-heavy` retry, re-export data with
   `feature_version=3` and train from scratch or from a checkpoint trained on
   the same contract.
3. Keep the earlier training-probe guard (`a28d2c313`) in place; it remains
   useful for catching future root-value collapses early.
