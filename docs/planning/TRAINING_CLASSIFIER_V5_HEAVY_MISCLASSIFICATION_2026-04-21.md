# Training Classifier V5-Heavy Misclassification

Date: 2026-04-21

## Summary

The false `Encoder mismatch: init_weights=..., data=v3` failure is rooted in the
checkpoint encoder-family classifier, not in NPZ export or in the runtime loader.

`prepare_training_model_artifacts()` in
[ai-service/app/training/model_factory.py](/Users/armand/Development/RingRift/ai-service/app/training/model_factory.py:646)
compares:

- `init_encoder_version = get_encoder_version_from_checkpoint(init_weights_path)`
- `data_encoder = "v3"` for `encoding_channels == 64`

For a valid 64-channel `HexNeuralNet_v5_Heavy` checkpoint, the init-weights side
must normalize to the same encoder family as the dataset: `v3`.

The bug was that
[ai-service/app/ai/neural_net/architecture_registry.py](/Users/armand/Development/RingRift/ai-service/app/ai/neural_net/architecture_registry.py:526)
treated v5 metadata as its own encoder label (`"v5-heavy"`) instead of the shared
64-channel hex encoder family (`"v3"`). That made the training path reject a valid
pair:

- checkpoint: 64-channel v5-heavy
- dataset: 64-channel hex v3-family NPZ

## Exact Trigger

Checkpoint input that reproduces the bug:

- `_versioning_metadata.architecture_version = "v5.1.0-hex"`
- `_versioning_metadata.model_class = "HexNeuralNet_v5_Heavy"`
- `conv1.weight.shape = (160, 64, 3, 3)`

Dataset input on the same training run:

- `features.shape[1] = 64`
- `encoder_version = "v3"` / training-side `data_encoder = "v3"`

Before the fix, `get_encoder_version_from_checkpoint()` returned the wrong label
for the checkpoint metadata path:

- v2 metadata -> `v2`
- v3 metadata -> `v3`
- v4 metadata -> `v3`
- v5 metadata -> `v5-heavy` <- wrong for encoder-family matching

That flowed into the fail-fast check in
[model_factory.py](/Users/armand/Development/RingRift/ai-service/app/training/model_factory.py:656)
and triggered:

```text
Encoder mismatch: init_weights=v5-heavy, data=v3
```

Operationally, this is the same classifier bug family that blocked `gh200-11`:
the checkpoint and NPZ were both 64-channel compatible, but the init-weights side
was labeled with an architecture name instead of an encoder family.

## Specific Misclassification Site

The offending branch was in
[architecture_registry.py](/Users/armand/Development/RingRift/ai-service/app/ai/neural_net/architecture_registry.py:539):

```python
elif "v5" in arch_ver:
    return "v5-heavy"
```

That helper is explicitly an encoder-family detector. Its own docstring says it
should return `v2` or `v3`, and the fallback channel-shape path already mapped
64-channel checkpoints to `v3`.

## Correct Classification Logic

Training compatibility should compare encoder families, not full architecture
names:

- 40-channel hex checkpoints -> `v2`
- 64-channel hex checkpoints (`v3`, `v4`, `v5-heavy`) -> `v3`
- 56-channel square checkpoints -> square family / current training-side `v2`

The model architecture detector remains separate:

- `get_model_version_from_checkpoint()` should still return `v5-heavy` for
  `HexNeuralNet_v5_Heavy`
- only `get_encoder_version_from_checkpoint()` should normalize v5-heavy to `v3`

## Fix

Patched:

- [ai-service/app/ai/neural_net/architecture_registry.py](/Users/armand/Development/RingRift/ai-service/app/ai/neural_net/architecture_registry.py:526)

Change:

- normalize metadata-tagged `v5.*` / v5-heavy checkpoints to encoder family `v3`
  instead of returning `v5-heavy`

This keeps the 64-channel v5-heavy checkpoint compatible with 64-channel hex NPZ
training data while preserving full architecture detection through
`get_model_version_from_checkpoint()`.

## Regression Coverage

Added coverage in
[ai-service/tests/test_model_versioning.py](/Users/armand/Development/RingRift/ai-service/tests/test_model_versioning.py:762):

- versioned 64-channel `HexNeuralNet_v5_Heavy` checkpoint:
  - encoder family must be `v3`
  - model version must still be `v5-heavy`
- versioned 40-channel `HexNeuralNet_v2` checkpoint:
  - encoder family must remain `v2`
  - model version must remain `v2`
