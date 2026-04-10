# Minimal AlphaZero Loop Contract

Date: 2026-04-10

This document records the operational contract of `ai-service/scripts/minimal_alphazero_loop.py`. The loop is the current narrow proof harness for AlphaZero-style training. It should be treated as a stable contract and not as a general orchestration API.

## Scope

The minimal loop owns a complete local iteration:

- Generate self-play JSONL with the Python training environment.
- Convert JSONL to NPZ with `scripts/jsonl_to_npz.py`.
- Train a candidate checkpoint with `python -m app.training.train`.
- Evaluate the candidate against the current best model with a staged, seat-fair evaluator.
- Promote only by copying the candidate over the work-dir `models/best.pth` after the staged evaluator and quality gate allow it.
- Emit local `metrics.jsonl` and best-effort S3 heartbeats.

It intentionally avoids the legacy coordinator, P2P work queue, daemon event bus, and broad dataset discovery paths.

## Inputs

The loop requires `--model`, which is copied into `<work-dir>/models/best.pth` if that file is missing. The model is used by `GumbelMCTSAI` through `AIConfig(use_neural_net=True, allow_fresh_weights=False, use_gpu_tree=True, nn_model_id=<best path>)`.

The supported board arguments are `hex8`, `hexagonal`, `square8`, and `square19`. The supported player counts are 2, 3, and 4. The global board and player contract is set from the CLI at startup and then used by self-play, export, train, and evaluation.

The current model family contract is `MODEL_VERSION = "v2"` in the minimal loop.

## Work Directory Layout

For iteration `N`, the loop writes:

- `<work-dir>/iter_NNN.jsonl`
- `<work-dir>/iter_NNN.npz`
- `<work-dir>/combined_NNN.npz` when a sliding training window is used
- `<work-dir>/models/best.pth`
- `<work-dir>/models/candidate_NNN.pth`
- `<work-dir>/metrics.jsonl`

Resume is based on the number of `iter_*.npz` files plus the latest metrics entries for promotions and estimated Elo.

## Self-Play Contract

Self-play uses `make_env(TrainingEnvConfig(...))` and caps moves at `int(theoretical_max * 1.5)`. Each game record contains `game_id`, `board_type`, `num_players`, `winner`, `status`, `num_moves`, `moves`, `initial_state`, and `timestamp`.

Each move is serialized from the Pydantic move model, includes the phase if missing, includes `moveNumber`, and includes an `mcts_policy` target when visit counts are available.

The default live training deployments currently use explicit self-play and evaluation budgets passed by the deploy script rather than relying on the generic `--budget` fallback.

## NPZ Export Contract

Export calls `scripts/jsonl_to_npz.py` with:

- `--input <iter_NNN.jsonl>`
- `--output <iter_NNN.npz>`
- `--board-type <board>`
- `--num-players <players>`
- `--gpu-selfplay`

The resulting NPZ must contain at least:

- `features`
- `globals`
- `values`
- `policy_indices`
- `policy_values`
- `move_numbers`
- `total_game_moves`
- `phases`
- `values_mp`
- `num_players`
- `history_length`
- `feature_version`
- `policy_encoding`
- `encoder_type`
- `base_channels`
- `in_channels`
- `board_type`
- `spatial_size`
- `policy_size`
- `data_schema_version`

The loop validates the `features` channel count against `app.training.board_encoding_contract.get_expected_channels(board_type, "v2")`.

## Training Contract

Training is delegated to `python -m app.training.train` with:

- `--data-path <npz or combined_NNN.npz>`
- `--save-path <candidate_NNN.pth>`
- `--board-type <board>`
- `--num-players <players>`
- `--epochs <epochs>`
- `--batch-size <batch size>`
- `--learning-rate <effective lr>`
- `--init-weights <work-dir>/models/best.pth`
- `--no-auto-tune-batch-size`
- `--lr-scheduler <none|cosine|...>`
- `--skip-freshness-check`
- `--sampling-weights uniform`

The loop retries out-of-memory failures by halving batch size up to three attempts. Loop-level learning rate is either fixed or `max(lr_floor, base_lr / sqrt(iteration))`. When `--train-lr-scheduler auto` is used, the train CLI scheduler is `none` for fixed learning rate and `cosine` otherwise.

## Evaluation And Promotion Contract

Evaluation is staged and seat-fair. Candidate seat rotates as `(game_index % num_players) + 1`; all other seats use the current best model. The stage contract is:

- 50 games: promote at 60 percent win rate, early reject below 42 percent.
- 100 games: promote at 56 percent win rate, early reject below 46 percent.
- 200 games: promote at 53 percent win rate, early reject below 48 percent.
- 400 games: promote above 50.1 percent, with no early-reject floor.

The `--promote-threshold` argument is retained for logging compatibility, but the staged decision is authoritative. Promotion only happens when `evaluation.decision == "promote"` and the model quality gate does not block it.

## Quality Gates

Unless explicitly skipped, the loop runs:

- Data quality sentinel before training.
- Training probes after candidate training.
- Model quality gate after evaluation.
- Self-healing circuit breaker after repeated failures.

These gates are part of the proof-run contract. Any coordinator path claiming equivalence must either run them or explicitly document why it does not.

## Metrics And Heartbeats

Each metrics JSONL entry records iteration, timestamps, self-play metrics, training metrics, evaluation metrics, promotion decision, estimated Elo, total promotions, iteration time, budgets, learning-rate settings, git SHA, and effective learning rate.

S3 heartbeats are best effort and are written under `s3://ringrift-models-20251214/consolidated/heartbeats/<node>.json` with node, config, iteration, Elo, promotions, stage, data-quality score, and experiment parameters.
