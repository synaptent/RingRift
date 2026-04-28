# v4 Multiplayer Diagnostic

This note separates three things that were previously conflated:

- actual v4 architecture behavior
- training-contract bugs around multiplayer loss
- host instability on `gh200-9`

The current conclusion is narrow: v4 multiplayer is still not cleared, but the
next test must run from a code revision that includes both known contract fixes.

## Known Root Causes Fixed

### 1. v4 rank-distribution head was defined but not returned

Fix:

- `2a659360f fix(ai): return rank distribution from hex v4`

Why it mattered:

- `HexNeuralNet_v4` exposed rank-distribution parameters.
- The forward path did not compute and return `rank_dist`.
- Multiplayer diagnostics could inspect a plausible architecture while the
  actual training graph had no rank-distribution output.

Guardrail now present:

```bash
cd ai-service
PYTHONPATH=. pytest tests/unit/ai/test_neural_net_architectures.py::TestHexNeuralNet_v4 -q
```

### 2. Minimal loop did not enable multiplayer loss

Fix:

- `10ee06181 fix(ai): enable multiplayer loss in minimal loop`

Why it mattered:

- `minimal_alphazero_loop.py` passed `--num-players 3`.
- It did not pass `--multi-player` to `app.training.train`.
- `train.py` only enables vector value loss and rank-distribution loss when
  `--multi-player` is set.

Guardrail now present:

```bash
cd ai-service
PYTHONPATH=. pytest \
  tests/unit/scripts/test_minimal_alphazero_loop.py::test_train_model_enables_multi_player_training_for_3p -q
```

## Inconclusive Prior Diagnostic

The `gh200-9` diagnostic on `2a659360f` is not valid evidence for or against
v4. That host was killed repeatedly during the training phase:

- self-play completed once
- NPZ export completed
- training started
- no post-sentinel `Training probe details` line landed
- each process kill rolled the lane back before a completed train/probe cycle

Interpretation:

> `gh200-9` tested host stability, not v4 architecture behavior.

The evidence bundle is still useful for operations, but it should not be used
as a model-quality conclusion.

## Required Retry Conditions

Only retry v4 on a stable host, with code at or after:

- `2a659360f`
- `10ee06181`

Recommended current baseline:

- `58cd60c33` or later

Preflight:

```bash
cd ai-service
PYTHONPATH=. pytest tests/unit/ai/test_neural_net_architectures.py::TestHexNeuralNet_v4 -q
PYTHONPATH=. pytest \
  tests/unit/scripts/test_minimal_alphazero_loop.py::test_train_model_enables_multi_player_training_for_3p -q
```

Launch conditions:

- fresh `RESTART_UTC` sentinel
- fresh work directory or clearly archived old probe logs
- no reuse of `gh200-9` unless the host-kill pattern is proven resolved
- no displacement of productive fv3 or sq8 lanes without an explicit operator
  decision

Training flags that must appear in the command/logs:

- `--model-version v4`
- `--num-players 3`
- `--multi-player`
- `--value-weight 2.0`
- `--rank-dist-weight 0.05`
- `--policy-weight 0.8`
- `--grad-clip 0.5`

## Verdict Matrix

After the first post-sentinel training probe:

| Observation                                | Interpretation                    | Action                                                  |
| ------------------------------------------ | --------------------------------- | ------------------------------------------------------- |
| `value_std > 0.01` and healthy `values_mp` | corrected graph is alive          | keep evidence, continue to first eval verdict           |
| `value_std ~= 0.007` with healthy targets  | v4 still collapses                | preserve checkpoint/logs and stop                       |
| loader, shape, or P0 architecture error    | contract is still broken upstream | preserve traceback, command line, checkpoint, and stop  |
| repeated process kills before probe        | host is invalid for diagnosis     | terminate or move the lane; do not count as v4 evidence |

## Current Project Decision

Do not add a new architecture family until this retry either:

- clears v4 as mechanically healthy, or
- produces a specific post-fix failure mode.

If no stable GH200 is available, the highest-value use of time is analysis and
packaging of the existing fv3 and 2-player results, not another inconclusive
v4 run.
