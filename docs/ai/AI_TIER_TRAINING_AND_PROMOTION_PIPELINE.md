# AI Tier Training and Promotion Pipeline (H-AI-9 – Square-8 2-Player)

> Status: partially implemented. Describes the intended training, evaluation, and promotion loop for Square-8 2-player D2/D4/D6/D8 tiers.  
> Core building blocks (tier eval configs, perf budgets, curriculum harness) and the combined gating CLI are now implemented; remaining orchestration is incremental wiring on top.

This document specifies a concrete, repeatable pipeline for training, evaluating, and promoting AI models for the Square-8 2-player difficulty ladder tiers D2, D4, D6, and D8.

### Current runnable pieces (2025-12-06)

- **Tier gate path:** `ai-service/scripts/run_tier_gate.py` + `app/training/tier_eval_config.py` remain the canonical promotion gate for D2/D4/D6/D8 (Square-8 2p). Use `--output-json` and `--promotion-plan-out` for artefacts.
- **NN baseline demos:** `ai-service/scripts/run_nn_training_baseline.py` with accompanying smokes `ai-service/tests/test_nn_training_baseline_demo.py` and `ai-service/tests/test_neural_net_ai_demo.py` run today but are **demo-scale only** (not production checkpoints). Treat outputs as experimental candidates until slotted into the gate.
- **Distributed self-play soak:** `ai-service/scripts/run_distributed_selfplay_soak.py` is wired for Square-8 2p distributed self-play; use it for dataset generation ahead of tier gates. Promotion criteria below still apply.

The design is constrained to existing infrastructure:

- Ladder definitions in [`python.LadderTierConfig`](ai-service/app/config/ladder_config.py:26) and [`python.get_ladder_tier_config`](ai-service/app/config/ladder_config.py:279).
- Canonical training environment [`python.RingRiftEnv`](ai-service/app/training/env.py:274) via [`python.make_env`](ai-service/app/training/env.py:221) and [`python.TrainingEnvConfig`](ai-service/app/training/env.py:171).
- Tier evaluation configs in [`python.TierEvaluationConfig`](ai-service/app/training/tier_eval_config.py:37) and runner [`python.run_tier_evaluation`](ai-service/app/training/tier_eval_runner.py:319).
- Difficulty-tier gating CLI [`run_tier_gate.py`](ai-service/scripts/run_tier_gate.py:1).
- Perf budgets in [`python.TierPerfBudget`](ai-service/app/config/perf_budgets.py:20) and benchmark helper [`python.run_tier_perf_benchmark`](ai-service/app/training/tier_perf_benchmark.py:84).
- Human calibration experiments in [`AI_HUMAN_CALIBRATION_GUIDE.md`](docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md:1) and calibration telemetry in [`difficultyCalibrationEvents.ts`](src/shared/telemetry/difficultyCalibrationEvents.ts:1) and [`typescript.MetricsService.recordDifficultyCalibrationEvent`](src/server/services/MetricsService.ts:1292).

---

## 1. Scope and invariants

### 1.1 Scope

This pipeline covers only:

- Board type: `square8` (8×8, compact ruleset).
- Players: 2.
- Ladder difficulties: D2, D4, D6, D8 (logical difficulties 2, 4, 6, 8).

Future work can extend the same patterns to other boards and player counts using the multi-board tier configs already in [`python.tier_eval_config`](ai-service/app/training/tier_eval_config.py:196) and [`python.ladder_config`](ai-service/app/config/ladder_config.py:121).

> **Note (2025-12-17):** Minimax is disabled for square19 and hexagonal boards. The search space on these larger boards causes minimax to take tens of minutes per move. D3/D4 tiers on square19 and hex use MCTS instead. See commit `44bf4400`.

### 1.2 Non-goals

- No rules or move-semantics changes; all engines must stay aligned with the TS shared rules engine and parity contracts.
- No changes to the public HTTP API of the AI service in this task.
- No new on-disk formats beyond small JSON/Markdown artefacts and model checkpoints under existing `models/` and training log directories.

### 1.3 Tier invariants

For Square-8 2-player, the canonical ladder currently defines (simplified):

| Tier | Difficulty | Ladder ai_type | Ladder model_id | heuristic_profile_id | Intended strength    |
| ---- | ---------- | -------------- | --------------- | -------------------- | -------------------- |
| D2   | 2          | HEURISTIC      | heuristic_v1_2p | heuristic_v1_2p      | Casual / learning    |
| D4   | 4          | MINIMAX        | v1-minimax-4    | heuristic_v1_2p      | Intermediate         |
| D6   | 6          | MINIMAX        | v1-minimax-6    | heuristic_v1_2p      | Advanced             |
| D8   | 8          | MCTS           | v1-mcts-8       | heuristic_v1_2p      | Strong / near-expert |

(from [`python._build_default_square8_two_player_configs`](ai-service/app/config/ladder_config.py:47))

Pipeline invariants:

- **Ordering:** D2 < D4 < D6 < D8 in strength on Square-8 2-player.
- **Non-regression vs previous production version of each tier:** new Dn must not be substantially weaker than the previous Dn against shared baselines.
- **Perf budgets for D4/D6/D8:** average and p95 move latencies must respect [`AI_TIER_PERF_BUDGETS.md`](docs/ai/AI_TIER_PERF_BUDGETS.md:1) and [`python.TierPerfBudget`](ai-service/app/config/perf_budgets.py:20).

---

## 2. Building blocks

### 2.1 Environment and rules

- Training and evaluation always use [`python.RingRiftEnv`](ai-service/app/training/env.py:274) constructed via [`python.make_env`](ai-service/app/training/env.py:221) with:
  - `board_type=BoardType.SQUARE8`.
  - `num_players=2`.
  - `reward_mode="terminal"` for gating and perf benchmarks.
- `max_moves` defaults to 200 for Square-8 2-player, consistent with [`python.make_env`](ai-service/app/training/env.py:221) and [`python.get_theoretical_max_moves`](ai-service/app/training/env.py:55).
- Rules semantics are provided by [`python.DefaultRulesEngine`](ai-service/app/rules/default_engine.py:1) and [`python.GameEngine`](ai-service/app/game_engine.py:1); both are already wired through the env.

### 2.2 Difficulty ladder and online integration

The FastAPI service endpoint [`python.get_ai_move`](ai-service/app/main.py:236):

- Uses [`python._get_difficulty_profile`](ai-service/app/main.py:1048) for the global 1–10 mapping.
- When possible, refines settings via board-aware [`python.get_ladder_tier_config`](ai-service/app/config/ladder_config.py:279) for `(difficulty, board_type, num_players)`.
- Constructs [`python.AIConfig`](ai-service/app/models/core.py:419) with:
  - `difficulty`, `randomness`, `think_time` from the ladder or profile.
  - `heuristic_profile_id` from ladder or difficulty profile.
  - `nn_model_id` from ladder `model_id` when `ai_type` is MCTS or DESCENT.

Neural-net-backed AIs (MCTS/Descent) resolve `nn_model_id` to a checkpoint under `ai-service/models` via [`python.NeuralNetAI`](ai-service/app/ai/neural_net.py:1049).

Heuristic tiers use profiles in [`python.HEURISTIC_WEIGHT_PROFILES`](ai-service/app/ai/heuristic_weights.py:1).

### 2.3 Training components

We reuse existing training infrastructure described in [`AI_TRAINING_AND_DATASETS.md`](docs/ai/AI_TRAINING_AND_DATASETS.md:1):

- Self-play NPZ datasets via [`python.generate_dataset`](ai-service/app/training/generate_data.py:1).
- Territory / combined-margin JSONL datasets via [`python.generate_territory_dataset`](ai-service/app/training/generate_territory_dataset.py:1).
- Heuristic-weight training via [`python.train_heuristic_weights`](ai-service/app/training/train_heuristic_weights.py:1).
- Neural net training via [`python.train_model`](ai-service/app/training/train.py:1171) and [`python.TrainConfig`](ai-service/app/config/training_config.py:176).
- Model versioning via [`python.ModelVersionManager`](ai-service/app/training/model_versioning.py:311) and helpers like [`python.save_model_checkpoint`](ai-service/app/training/model_versioning.py:865).
- Curriculum-based self-play and promotion via [`python.CurriculumConfig`](ai-service/app/training/curriculum.py:152) and [`python.CurriculumTrainer`](ai-service/app/training/curriculum.py:247).
- Auto-tournaments and Elo-based comparison via [`python.AutoTournamentPipeline`](ai-service/app/training/auto_tournament.py:327).

### 2.4 Evaluation, gating, and perf

- Tier evaluation configs: [`python.TIER_EVAL_CONFIGS`](ai-service/app/training/tier_eval_config.py:53) with [`python.TierEvaluationConfig`](ai-service/app/training/tier_eval_config.py:37), keyed by `"D2"`, `"D4"`, `"D6"`, `"D8"`.
- Runner: [`python.run_tier_evaluation`](ai-service/app/training/tier_eval_runner.py:319) returning [`python.TierEvaluationResult`](ai-service/app/training/tier_eval_runner.py:74).
- Difficulty-tier gate CLI: [`run_tier_gate.py`](ai-service/scripts/run_tier_gate.py:1), difficulty mode via [`python._run_difficulty_mode`](ai-service/scripts/run_tier_gate.py:252).
  - Emits evaluation JSON via `--output-json`.
  - Emits promotion descriptor via `--promotion-plan-out`.
- Perf budgets:
  - [`python.TIER_PERF_BUDGETS`](ai-service/app/config/perf_budgets.py:101) with entries `"D4_SQ8_2P"`, `"D6_SQ8_2P"`, `"D8_SQ8_2P"`.
  - Benchmarks via [`python.run_tier_perf_benchmark`](ai-service/app/training/tier_perf_benchmark.py:84) and CLI [`run_tier_perf_benchmark.py`](ai-service/scripts/run_tier_perf_benchmark.py:1).

### 2.5 Human calibration and telemetry

- Human calibration experiments: [`AI_HUMAN_CALIBRATION_GUIDE.md`](docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md:1) (Templates A/B/C).
- Telemetry schema: [`difficultyCalibrationEvents.ts`](src/shared/telemetry/difficultyCalibrationEvents.ts:1) defining `DifficultyCalibrationEventPayload`.
- Client helper: [`difficultyCalibrationTelemetry.ts`](src/client/utils/difficultyCalibrationTelemetry.ts:1).
- Server route: [`difficultyCalibrationTelemetry.ts`](src/server/routes/difficultyCalibrationTelemetry.ts:1).
- Metrics sink: [`typescript.MetricsService.recordDifficultyCalibrationEvent`](src/server/services/MetricsService.ts:1292), exposing `ringrift_difficulty_calibration_events_total`.

---

## 3. Candidate model shape per tier

We standardise what a _candidate_ is at each tier and how it will be wired into the ladder.

### 3.1 Summary table

| Tier | Engine family (Square-8 2p) | Candidate kind                               | Primary artefact(s)                                                                         | Ladder fields touched on promotion                             |
| ---- | --------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| D2   | HEURISTIC                   | Heuristic profile                            | New entry in [`python.HEURISTIC_WEIGHT_PROFILES`](ai-service/app/ai/heuristic_weights.py:1) | `heuristic_profile_id` (and optionally `model_id`)             |
| D4   | MINIMAX                     | Search persona + heuristic profile           | Minimax config + optional D4-specific heuristic profile                                     | `model_id` (persona tag) and optionally `heuristic_profile_id` |
| D6   | MINIMAX or MINIMAX+NN       | Neural eval checkpoint and/or search persona | Versioned NN checkpoint (`models/<nn_model_id>.pth`) and/or minimax config                  | `model_id` (mapped to `nn_model_id` or persona)                |
| D8   | MCTS (+ optional NN)        | Strongest available model (NN+MCTS/Descent)  | Versioned NN checkpoint and MCTS config                                                     | `model_id` (mapped to `nn_model_id`)                           |

Rules:

- All **neural** candidates are identified by a logical `nn_model_id` string and a versioned checkpoint saved via [`python.save_model_checkpoint`](ai-service/app/training/model_versioning.py:865).
- All **heuristic** candidates are identified by a `heuristic_profile_id` key in [`python.HEURISTIC_WEIGHT_PROFILES`](ai-service/app/ai/heuristic_weights.py:1).
- Search-persona changes must map to distinct ladder `model_id` values (for example `v2-minimax-4`).

---

## 4. Training pipeline per tier (Square-8 2-player)

### 4.1 Shared training conventions

- Board: `BoardType.SQUARE8`, `num_players=2` in [`python.TrainingEnvConfig`](ai-service/app/training/env.py:171).
- Reward: `"terminal"` for gating-oriented training (self-play for value/policy can still use shaping when appropriate).
- Seeding: follow the RNG model from [`AI_TRAINING_AND_DATASETS.md`](docs/ai/AI_TRAINING_AND_DATASETS.md:284) and [`python.RingRiftEnv.reset`](ai-service/app/training/env.py:371).
- Data provenance: all training / eval datasets generated through the Python rules host described in [`AI_TRAINING_AND_DATASETS.md`](docs/ai/AI_TRAINING_AND_DATASETS.md:19).

### 4.2 D2 – heuristic baseline

**Model type**

- Pure heuristic AI using a Square-8 2-player profile in [`python.HEURISTIC_WEIGHT_PROFILES`](ai-service/app/ai/heuristic_weights.py:1), currently `heuristic_v1_2p`.

**Training data & curriculum**

- Primary dataset: territory / combined-margin JSONL as in [`python.generate_territory_dataset`](ai-service/app/training/generate_territory_dataset.py:1) and documented in [`AI_TRAINING_AND_DATASETS.md`](docs/ai/AI_TRAINING_AND_DATASETS.md:134).
  - Board: `square8`.
  - `num_players=2`.
  - Engine mode: `descent-only` or `mixed` at low-to-mid difficulties.
- Optional NPZ self-play dataset via [`python.generate_dataset`](ai-service/app/training/generate_data.py:1) for auxiliary supervision.
- Curriculum:
  - Start from current `heuristic_v1_2p`.
  - Optimise weights via:
    - [`python.train_heuristic_weights`](ai-service/app/training/train_heuristic_weights.py:1) over combined-margin targets; or
    - [`python.run_cmaes_heuristic_optimization`](ai-service/app/training/train.py:231) with a Square-8 heuristic tier spec (for example `sq8_heuristic_baseline_v1`).

**Training configuration**

- Focus on gentle, forgiving behaviour:
  - Loss: regression on combined margin with regularisation to avoid extreme weight oscillations.
  - Conservative hyperparameters, early stopping on validation combined-margin error.
- For CMA-ES, keep:
  - Limited generations (for example 3–5).
  - Small population per generation (for example 8), as defaulted in [`python.run_cmaes_heuristic_optimization`](ai-service/app/training/train.py:231).

**Outputs & versioning**

- Create a new `heuristic_profile_id` such as `sq8_2p_d2_vYYYYMMDD_N` and persist it to `trained_heuristic_profiles.json` (via future Code-mode tasks).
- For gating, refer to the candidate by this `heuristic_profile_id`. Ladder D2 `heuristic_profile_id` will be updated to this id on promotion.

### 4.3 D4 – intermediate minimax tier

**Model type**

- Minimax search (`AIType.MINIMAX`) using:
  - A search persona (depth, pruning, etc.).
  - A Square-8 2-player heuristic profile (shared with D2 or D4-specific).

**Training data & curriculum**

- Data:
  - Mixed self-play on `square8`, 2 players, with difficulties clustered around D2–D6 using [`python.generate_dataset`](ai-service/app/training/generate_data.py:1).
- Tuning:
  - Treat D4 primarily as a search-config tuning problem with minimal heuristic changes.
  - Compare personas:
    - Baseline: `v1-minimax-4`.
    - Candidates: `v2-minimax-4a`, `v2-minimax-4b`, etc., each with specific depth / pruning parameters.

**Training / tuning configuration**

- For each persona:
  - Fix `heuristic_profile_id`.
  - Define search config (depth, pruning, etc.).
  - Run quick tournaments vs:
    - D1 random.
    - D2 heuristic (current production).
    - Previous D4 persona.
  - Use [`run_ai_tournament.py`](ai-service/scripts/run_ai_tournament.py:1) or [`evaluate_ai_models.py`](ai-service/scripts/evaluate_ai_models.py:1) to gather stats.

**Outputs & versioning**

- Candidate D4 persona id, e.g. `v2-minimax-4a`.
- Ladder D4 entry updated to `model_id="v2-minimax-4a"` and possibly a new `heuristic_profile_id` if D4-specific weights are introduced.

### 4.4 D6 – advanced tier

**Model type (current)**

- High-budget minimax search with same heuristic profile as D2/D4, leveraging D6 perf budget.

**Model type (target)**

- NN-backed search (minimax or Descent), where `nn_model_id` selects a shared Square-8 2-player NN.

**Training data & curriculum**

- Self-play:
  - Square-8 2-player NPZ datasets from [`python.generate_dataset`](ai-service/app/training/generate_data.py:1) using engine mixes centred around D4–D6.
- Curriculum:
  - Use [`python.CurriculumTrainer`](ai-service/app/training/curriculum.py:247) with:
    - `board_type=BoardType.SQUARE8`.
    - `num_players=2`.
    - A teacher model id (for example the best existing D6 or a strong D4).
  - Curriculum loops combine:
    - Self-play via [`python.generate_data`](ai-service/app/training/generate_data.py:1) (curriculum-aware).
    - NN training via [`python.train_model`](ai-service/app/training/train.py:1171).
    - Promotion logic inside the curriculum harness.

**Training configuration**

- NN architecture:
  - Use Square-8 configuration in [`python.TrainConfig`](ai-service/app/config/training_config.py:176) with `board_type=SQUARE8`.
  - Write checkpoints to `models/<model_id>.pth` via [`python.save_model_checkpoint`](ai-service/app/training/model_versioning.py:865).
- When training a D6 NN candidate, choose `model_id` names like `sq8_d6_v1`, `sq8_d6_v2` (set in [`python.TrainConfig`](ai-service/app/config/training_config.py:176)).
- After training:
  - Optionally register candidates into [`python.AutoTournamentPipeline`](ai-service/app/training/auto_tournament.py:327) for detailed Elo-based evaluation.
  - Choose a champion checkpoint as the D6 candidate id `CANDIDATE_D6_ID`.

**Outputs & versioning**

- For NN-backed D6:
  - `nn_model_id = "sq8_d6_vN"`.
  - Ladder D6 `model_id` set to `sq8_d6_vN`.
- For pure-minimax D6:
  - Persona tag `v2-minimax-6` as `model_id`, possibly with D6-specific `heuristic_profile_id`.

**Baseline NN demo experiment (A2, non-production)**

- For a cheap, smoke-testable Square-8 NN baseline, use the dedicated
  Python CLI:

  ```bash
  cd ai-service
  PYTHONPATH=. python scripts/run_nn_training_baseline.py \
    --board square8 \
    --num-players 2 \
    --run-dir /tmp/nn_baseline_demo \
    --demo \
    --seed 123
  ```

- This runs a tiny training loop on a synthetic dummy dataset (no real
  self-play data) and writes `nn_training_report.json` into `--run-dir`
  with:
  - `board = "square8"`, `num_players = 2`
  - `mode = "demo"`
  - `model_id` (logical NN id)
  - a small `training_params` block and stubbed `metrics`.

- The demo run is intended only for wiring and CI smoke tests; it is
  **not** a production training path and must not be used to update
  ladder tiers or canonical models.

- For sandbox evaluation, callers can route traffic to the experimental
  `AIType.NEURAL_DEMO` engine in the Python AI service:
  - Set `AI_ENGINE_NEURAL_DEMO_ENABLED=1` in the AI service environment.
  - Construct `AIConfig` with `ai_type=NEURAL_DEMO` and an appropriate
    `nn_model_id` (for example the `model_id` emitted by the baseline
    script).
  - The canonical difficulty ladder (D2–D8) does **not** select
    `NEURAL_DEMO`; this path is reserved for private queues and
    experiments.

### 4.5 D8 – strong / near-expert tier

**Model type**

- Strong MCTS (`AIType.MCTS`) on Square-8 2-player, optionally with NN guidance rerouted via `nn_model_id`.

**Training data & curriculum**

- Self-play:
  - High-quality games produced by D6/D8 MCTS/Descent variants on Square-8 using [`python.generate_dataset`](ai-service/app/training/generate_data.py:1) or long soaks via [`run_self_play_soak.py`](ai-service/scripts/run_self_play_soak.py:1).
- Curriculum:
  - Start from best D6 NN weights (`sq8_d6_vN`), further train under enhanced search conditions and more challenging self-play opponents.
  - Use auto-tournaments vs:
    - Current best D6.
    - Current best D8.
    - Possibly experimental D9/D10 models (for research purposes, not gating).

**Training configuration**

- NN architecture: same as D6, with training hyperparameters adjusted for more complex positions (for example, more epochs, smaller learning rates).
- MCTS config:
  - Bound search budget to D8 perf budget via targeted profiling before candidate evaluation.
  - Encode MCTS parameters into the persona string (`v2-mcts-8a`) coupled to `nn_model_id` if used.

**Outputs & versioning**

- Candidate id: `sq8_d8_vN` as `nn_model_id`.
- Ladder D8 `model_id` set to this id and treated as both the NN and persona identifier.

---

## 5. Automated evaluation and gating

### 5.1 Evaluation profiles per tier

[`python.TIER_EVAL_CONFIGS`](ai-service/app/training/tier_eval_config.py:53) includes:

| Name | board_type | num_players | candidate_difficulty | Opponents (role)                                        | num_games | min_win_rate_vs_baseline | max_regression_vs_previous_tier |
| ---- | ---------- | ----------- | -------------------- | ------------------------------------------------------- | --------- | ------------------------ | ------------------------------- |
| D2   | SQUARE8    | 2           | 2                    | D1 random (baseline)                                    | 200       | 0.60                     | None                            |
| D4   | SQUARE8    | 2           | 4                    | D1 random (baseline), D2 (previous_tier)                | 400       | 0.70                     | 0.05                            |
| D6   | SQUARE8    | 2           | 6                    | D1 random, D2 heuristic (baselines), D4 (previous_tier) | 400       | 0.75                     | 0.05                            |
| D8   | SQUARE8    | 2           | 8                    | D1 random (baseline), D6 (previous_tier)                | 400       | 0.80                     | 0.05                            |

These are consumed by:

- [`run_tier_evaluation.py`](ai-service/scripts/run_tier_evaluation.py:1) for ad-hoc tier eval.
- [`run_tier_gate.py`](ai-service/scripts/run_tier_gate.py:1) for gating.

### 5.2 Gating & promotion rules

For each tier T ∈ {D2, D4, D6, D8}, the automated gate is:

1. Run [`run_tier_gate.py`](ai-service/scripts/run_tier_gate.py:1) in difficulty-tier mode:

   ```bash
   cd ai-service
   python scripts/run_tier_gate.py \
     --tier D6 \
     --seed 1 \
     --candidate-model-id CANDIDATE_D6_ID \
     --output-json logs/tier_eval/D6_candidate.json \
     --promotion-plan-out logs/tier_eval/D6_promotion_plan.json
   ```

2. [`python._run_difficulty_mode`](ai-service/scripts/run_tier_gate.py:252) will:
   - Look up the TierEvaluationConfig via [`python.get_tier_config`](ai-service/app/training/tier_eval_config.py:436).
   - Resolve current production ladder entry via [`python.get_ladder_tier_config`](ai-service/app/config/ladder_config.py:279) to find `current_model_id`.
   - Call [`python.run_tier_evaluation`](ai-service/app/training/tier_eval_runner.py:319).
   - Print a human summary and emit:
     - Full TierEvaluationResult JSON (`--output-json`).
     - A promotion plan JSON via `--promotion-plan-out` with keys:
       - `tier`, `board_type`, `num_players`.
       - `current_model_id`, `candidate_model_id`.
       - `decision`: `"promote"` or `"reject"`.
       - `reason`: metrics summary (`win_rate_vs_baseline`, `win_rate_vs_previous_tier`, `overall_pass`).

3. Decision semantics:
   - `overall_pass` must be true.
   - Criteria:
     - `win_rate_vs_baseline >= min_win_rate_vs_baseline` when applicable.
     - `win_rate_vs_previous_tier >= 0.5 - max_regression_vs_previous_tier` when applicable.

Process-level expectations added by H-AI-9:

- **Non-regression vs previous production version of the same tier:**
  - When available, compare candidate vs previous production T (e.g. using `evaluate_ai_models.py`) and require that the candidate is not clearly worse (for example no extreme <45% win rate).
- **Tier ordering:**
  - After promotion of any tier, run a brief cross-tier smoke evaluation (e.g. D2 vs D4, D4 vs D6, D6 vs D8) and sanity-check that tier strengths remain ordered as intended.

### 5.3 Ladder integration

When a candidate passes automated gating (and perf, see below) and is approved:

- Update [`python._build_default_square8_two_player_configs`](ai-service/app/config/ladder_config.py:47) in a Code-mode change:
  - D2: set `heuristic_profile_id="CANDIDATE_D2_PROFILE"`, optionally `model_id` to the same id.
  - D4: set `model_id="CANDIDATE_D4_PERSONA"`, optionally adjust `heuristic_profile_id`.
  - D6: set `model_id="CANDIDATE_D6_ID"` (NN or minimax persona).
  - D8: set `model_id="CANDIDATE_D8_ID"`.

Promotion plans written by `run_tier_gate.py` should be archived under something like `ai-service/data/promotions/square8_2p/` for traceability.

---

## 6. Performance constraints in the loop

### 6.1 Running perf benchmarks

For D4, D6, D8, perf budgets are defined in [`python._build_square8_two_player_budgets`](ai-service/app/config/perf_budgets.py:51) and described in [`AI_TIER_PERF_BUDGETS.md`](docs/ai/AI_TIER_PERF_BUDGETS.md:1).

For each candidate that _passes_ the tier gate, the pipeline must also run:

```bash
cd ai-service
python scripts/run_tier_perf_benchmark.py \
  --tier D6 \
  --num-games 4 \
  --moves-per-game 16 \
  --seed 1 \
  --output-json logs/tier_perf/D6_candidate.json
```

[`python.run_tier_perf_benchmark`](ai-service/app/training/tier_perf_benchmark.py:84) returns:

- `average_ms`, `p95_ms`.
- Associated `TierPerfBudget` with `max_avg_move_ms`, `max_p95_move_ms`.

The CLI evaluates budgets via [`python._eval_budget`](ai-service/scripts/run_tier_perf_benchmark.py:95):

- `within_avg = average_ms <= max_avg_move_ms`.
- `within_p95 = p95_ms <= max_p95_move_ms`.
- `overall_pass = within_avg and within_p95`.

H-AI-9 treats `overall_pass` for perf as **required** for D4/D6/D8 promotions unless there is an explicit, documented adjustment of budgets.

In practice, instead of invoking the perf benchmark separately, the
combined wrapper [`run_full_tier_gating.py`](ai-service/scripts/run_full_tier_gating.py:1) can be used to run both the difficulty-tier gate and perf check in one step:

```bash
cd ai-service
python scripts/run_full_tier_gating.py \
  --tier D6 \
  --candidate-model-id CANDIDATE_D6_ID \
  --run-dir logs/tier_gate/D6_candidate
```

This writes:

- `D6_tier_eval.json` and `D6_promotion_plan.json` – difficulty-tier gate outputs.
- `D6_perf.json` – perf benchmark metrics and budget evaluation.
- `D6_gate_report.json` – combined summary suitable for CI and manual review.

### 6.2 Trade-offs and escalation

If a candidate is stronger but exceeds perf budgets:

1. **Preferred path:** adjust search parameters (depth, iteration caps, or reroll heuristics) to bring perf under budget, then re-run tier evaluation and gating.
2. **If regression is minor and strength gain is large:**
   - Optionally propose a new `think_time_ms` ladder value and recompute perf budgets in [`perf_budgets.py`](ai-service/app/config/perf_budgets.py:51) via a coordinated change.
   - This must be justified by product/UX decisions (for example “we accept D8 being slower for significantly more depth”).
3. **Otherwise:** reject the candidate and keep current production tier.

---

## 7. Human calibration integration

### 7.1 Mapping calibration experiments to tiers

[`AI_HUMAN_CALIBRATION_GUIDE.md`](docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md:1) defines three experiment templates for Square-8 2-player:

- Template A – New-player quick check (D2/D4, 3-game sets).
- Template B – Intermediate validation (D4/D6, 5-game blocks).
- Template C – Strong-player session (D6/D8, 10+ games).

H-AI-9 integrates these as follows:

- **D2:** Template A with true beginners; ensure at least one win in 3–5 games is common and D2 is not overwhelmingly strong.
- **D4:** Template A (when D2 trivial) and Template B for intermediates; confirm 30–70% win-rate bands and “about right” perceived difficulty.
- **D6:** Template B and C for strong players; aim for 40–60% win rates among strong testers.
- **D8:** Template C; even strong testers should struggle to exceed ~60% win rate.

### 7.2 Telemetry and aggregation

Calibration-mode games in the client should:

- Use Square-8, 2 players, ladder difficulties D2/D4/D6/D8, as recommended in [`AI_HUMAN_CALIBRATION_GUIDE.md`](docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md:26).
- Attach `isCalibrationOptIn=true` in the payload to [`difficultyCalibrationEvents.ts`](src/shared/telemetry/difficultyCalibrationEvents.ts:1).

During such games, the client sends:

- `difficulty_calibration_game_started`.
- `difficulty_calibration_game_completed` with:
  - `result`: win / loss / draw / abandoned.
  - `movesPlayed`.
  - `perceivedDifficulty` (1–5).

On the server, [`typescript.MetricsService.recordDifficultyCalibrationEvent`](src/server/services/MetricsService.ts:1292) increments a counter labelled by `type`, `board_type`, `num_players`, `difficulty`, and `result`, but **only when `isCalibrationOptIn=true`**.

For H-AI-9, a lightweight aggregation (outside this doc) should periodically compute, per `(board_type, num_players, difficulty)`:

- Number of completed calibration games.
- Human win rate (fraction of `result="win"`).
- Distribution of `perceivedDifficulty`.

This can use Prometheus queries or downstream metric export.

### 7.3 Decision rules using human data

Human calibration is **advisory but important**:

- **Promotion only on automated + perf success:**
  - A candidate must pass automated gate and perf first; human data refines whether we roll out broadly or treat as experimental.
- **Adjusting tiers based on human data:**
  - If automated gate passes but humans consistently find a tier “far too easy” with >80% win rate for the target profile, consider:
    - Promoting the candidate to a higher logical difficulty (e.g. D4→D5) while leaving the old D4 intact.
    - Tightening automated gate thresholds for future candidates.
  - If automated gate passes but humans find a tier “far too hard” with <20% win rate and many 4–5 ratings, consider:
    - Weakening the tier (adjust ladder randomness or search budget, then re-run gates).
    - Hiding the tier from default UX or labelling it as “experimental” or “expert only”.
- **Avoid overfitting to small samples:**
  - Treat sample sizes <30 calibration games per `(player_profile, tier)` as inconclusive; use them only to detect gross issues (e.g. AI never loses).
  - Require multiple waves of data before permanently repositioning tiers or changing displayed difficulty descriptions.

---

## 8. Orchestration and automation plan

### 8.1 High-level pipeline stages

For each tier T ∈ {D2, D4, D6, D8}, the pipeline is:

1. **Train candidate for tier T:**
   - Run the appropriate training jobs (heuristic or NN) for Square-8 2-player using [`python.train_heuristic_weights`](ai-service/app/training/train_heuristic_weights.py:1), [`python.train_model`](ai-service/app/training/train.py:1171), or [`python.CurriculumTrainer`](ai-service/app/training/curriculum.py:247).
   - Produce candidate artefacts and assign a candidate id `CANDIDATE_T_ID`.

2. **Run automated evaluation and gating:**
   - `python scripts/run_tier_gate.py --tier T --candidate-model-id CANDIDATE_T_ID ...`.
   - Inspect TierEvaluationResult JSON and promotion plan.

3. **Run perf benchmarks:**
   - For D4/D6/D8, run `python scripts/run_tier_perf_benchmark.py --tier T ...`.
   - Ensure `overall_pass=true`.

4. **Optional cross-tier sanity evaluation:**
   - Use [`run_ai_tournament.py`](ai-service/scripts/run_ai_tournament.py:1) or [`evaluate_ai_models.py`](ai-service/scripts/evaluate_ai_models.py:1) to play:
     - New D2 vs old D2 vs random.
     - New D4 vs D2 / D6.
     - New D6 vs D4 / D8.
     - New D8 vs D6 and, if appropriate, some D9/D10 experiments.

5. **Optional calibration phase:**
   - Roll the candidate to calibration cohorts via feature flags or controlled environments.
   - Monitor aggregated telemetry for win-rate and perceived difficulty.

6. **Approve and merge ladder update:**
   - If all required stages succeed, update ladder configs in [`ladder_config.py`](ai-service/app/config/ladder_config.py:47) in a Code-mode change.
   - Record promotion descriptor and supporting evidence (JSON, perf benchmark outputs, calibration summary).

### 8.2 Orchestration scripts / configs

To make the above reproducible, H-AI-9 defines the following orchestration entrypoints:

1. **`ai-service/scripts/run_tier_training_pipeline.py`** (implemented)
   - Arguments:
     - `--tier {D2,D4,D6,D8}`.
     - `--board square8`.
     - `--num-players 2`.
     - `--run-dir PATH`.
   - Behaviour (conceptual):
     - Generate training data for the tier (self-play, territory, etc.).
     - Run heuristic or NN training as appropriate.
     - Optionally register NN models into [`python.AutoTournamentPipeline`](ai-service/app/training/auto_tournament.py:327).
     - Emit `training_report.json` with:
       - `candidate_id`.
       - Data sources.
       - Key hyperparameters.
       - Training metrics (losses, etc.).

2. **`ai-service/scripts/run_full_tier_gating.py`** (implemented wrapper around existing CLIs)
   - Arguments (current implementation):
     - `--tier T` (e.g. `D4`, `D6`, `D8`).
     - `--candidate-model-id CANDIDATE_T_ID`.
     - `--run-dir PATH` (defaults to `logs/tier_gate`).
     - `--seed`, `--num-games` for the tier evaluation step.
     - `--perf-num-games`, `--perf-moves-per-game` for the perf benchmark.
   - Behaviour:
     - Runs [`run_tier_gate.py`](ai-service/scripts/run_tier_gate.py:1) in difficulty-tier mode and writes `TIER_tier_eval.json` and `TIER_promotion_plan.json` under `--run-dir`.
     - Runs the tier perf benchmark via [`run_tier_perf_benchmark.py`](ai-service/scripts/run_tier_perf_benchmark.py:1) (where a `TierPerfBudget` exists) and writes `TIER_perf.json`.
     - Aggregates into `TIER_gate_report.json`:
       - Embedded TierEvaluationResult metrics (`overall_pass`, `win_rate_vs_baseline`, `win_rate_vs_previous_tier`, etc.).
       - Perf metrics (`average_ms`, `p95_ms`) and budget evaluation (`within_avg`, `within_p95`, `overall_pass`) when a perf budget is defined for the tier.
   - Exit semantics:
     - Returns exit code `0` only when **both** the difficulty gate and perf budget pass; otherwise returns `1` so CI pipelines can key off a single orchestration step.

3. **`ai-service/scripts/run_parity_promotion_gate.py`** (cross-board candidate-vs-baseline parity gate)
   - Purpose:
     - Run a small evaluation matrix of _candidate vs baseline_ AIs across one or more boards using the generic `evaluate_ai_models` harness.
     - Aggregate win-rate and Wilson CI metrics and emit a single JSON summary with `overall_pass` plus per-board breakdowns.
     - Provide an additional, optional non-regression check that can be run **before** tier-specific gating and perf budgets.
   - Arguments (current implementation):
     - `--player1`, `--player2`: AI types for candidate and baseline (for example `neural_network` vs `neural_network`).
     - `--checkpoint`, `--checkpoint2`: optional checkpoints for candidate and baseline.
     - `--boards`: list of boards to evaluate (defaults to `["square8"]`).
     - `--games-per-matrix`, `--max-moves`, `--seed`: evaluation controls.
     - `--min-ci-lower-bound`: minimum acceptable lower bound of the 95% CI for the candidate win rate on each matrix (default `0.5`).
     - `--output-json PATH`: optional JSON report output.
   - Behaviour:
     - For each requested board, call [`python.run_evaluation`](ai-service/scripts/evaluate_ai_models.py:260) and [`python.format_results_json`](ai-service/scripts/evaluate_ai_models.py:777) to obtain win-rate and positional stats.
     - Evaluate a simple non-inferiority rule per board: candidate passes that matrix when the _lower_ bound of its 95% Wilson CI is ≥ `min-ci-lower-bound`.
     - Emit a JSON report with:
       - `gate.overall_pass`: true only if **all** matrices pass.
       - `gate.matrices[board]`: per-board `player1_win_rate`, CI, `piece_advantage_p1`, and `passes` flag.
   - Integration:
     - Can be wired into CI pipelines or manual workflows as a fast “sanity check” ahead of running the heavier tier-specific gate + perf budgets.

4. **Minimal config snippet**
   - Optional `ai-service/config/tier_training_pipeline.square8_2p.json` describing:
     - Which generators to run per tier (NPZ / JSONL).
     - Default evaluation seeds and num_games overrides.
     - Paths for artefacts.

### 8.3 Pipeline status artefacts

Within each `--run-dir`, it is useful to maintain a small status JSON, for example:

```json
{
  "tier": "D6",
  "board_type": "square8",
  "num_players": 2,
  "candidate_id": "sq8_d6_v2",
  "training": {
    "status": "completed",
    "report": "training_report.json"
  },
  "automated_gate": {
    "status": "passed",
    "eval_json": "tier_eval_D6.json",
    "promotion_plan": "promotion_D6.json"
  },
  "perf": {
    "status": "passed",
    "perf_json": "tier_perf_D6.json"
  },
  "human_calibration": {
    "required": true,
    "status": "pending",
    "min_games": 50
  }
}
```

This status file can be updated incrementally by the orchestration scripts and consumed by CI or dashboards.

### 8.4 Running the tier training pipeline (Square-8 2-player)

This section shows a concrete end-to-end invocation for a mid-tier on Square-8 2-player using the D4 minimax tier as an example. The same pattern applies to D2, D6, and D8 by changing the `--tier` argument.

#### 8.4.1 Demo / smoke run (no heavy training or evaluation)

From `ai-service/`, run the training orchestrator in demo mode:

```bash
cd ai-service
PYTHONPATH=. python scripts/run_tier_training_pipeline.py \
  --tier D4 \
  --board square8 \
  --num-players 2 \
  --run-dir logs/tier_gate/D4_demo \
  --demo \
  --seed 123
```

This:

- Writes `training_report.json` and `status.json` into `logs/tier_gate/D4_demo`.
- Records a `candidate_id` (for example `sq8_2p_d4_demo_YYYYMMDD_HHMMSS`) and a snapshot of the env + tier-specific training parameters.
- Uses only stubbed, demo-friendly training logic (no long self-play or neural training); this is the safe path for CI and local smoke tests.

Next, run the combined tier gate + perf wrapper (also in demo mode) for the same run directory:

```bash
cd ai-service
PYTHONPATH=. python scripts/run_full_tier_gating.py \
  --tier D4 \
  --candidate-id "<CANDIDATE_ID_FROM_TRAINING_REPORT>" \
  --run-dir logs/tier_gate/D4_demo \
  --demo
```

You obtain:

- `tier_eval_result.json` – difficulty-tier evaluation summary for D4.
- `promotion_plan.json` – promotion descriptor (`tier`, `current_model_id`, `candidate_model_id`, `decision`, `reason`).
- `tier_perf_report.json` – perf benchmark metrics and budget evaluation for D4 (since a `TierPerfBudget` exists).
- `gate_report.json` – combined summary with `final_decision`, referenced from `status.json` as described in §8.3.

This pair of commands exercises the full tier training + gating pipeline for a mid-tier (D4 on Square-8 2-player) in a way that is fast enough for automated tests.

#### 8.4.2 Full pipeline run (when canonical data is available)

When canonical Square-8 2-player replay DBs exist and have passed the unified gate in [`TRAINING_DATA_REGISTRY.md`](ai-service/TRAINING_DATA_REGISTRY.md:18):

1. Run [`run_tier_training_pipeline.py`](ai-service/scripts/run_tier_training_pipeline.py:1) **without** `--demo` so that it can:
   - Generate or select training datasets derived **only** from DBs marked `canonical` in the registry.
   - Invoke the appropriate heuristic / neural training loops for the tier (for example via [`python.train_model`](ai-service/app/training/train.py:1171) for D6/D8).
   - Emit a richer `training_report.json` with explicit data-source and hyperparameter fields alongside training metrics.

2. Run [`run_full_tier_gating.py`](ai-service/scripts/run_full_tier_gating.py:1) **without** `--demo` (and without `--no-perf`) so that it:
   - Plays the full number of games from the tier’s [`TierEvaluationConfig`](ai-service/app/training/tier_eval_config.py:37).
   - Enforces the relevant [`TierPerfBudget`](ai-service/app/config/perf_budgets.py:20) for D4/D6/D8.
   - Produces a `gate_report.json` whose `final_decision` drives promotion proposals and ladder updates as described in §5.3.

In both cases you should keep each candidate’s artefacts under a dedicated `--run-dir` such as `logs/tier_gate/D4_candidate_YYYYMMDD` so that `training_report.json`, `gate_report.json`, and `status.json` can be archived together.

#### 8.4.3 Canonical data and compute constraints

- **Canonical-only training data:** Real training runs for D4/D6/D8 must be wired so that all self-play or replay-derived datasets are generated from DBs whose status is `canonical` in [`TRAINING_DATA_REGISTRY.md`](ai-service/TRAINING_DATA_REGISTRY.md:18). DBs marked `legacy_noncanonical`, `pending_gate`, or `DEPRECATED_R10` must not be used for new ladder training.
- **Default to demo in CI:** CI and quick local checks should use the `--demo` pipeline described above; heavy training runs are reserved for dedicated training environments with explicit operator sign-off.
- **Alignment with tier config JSON:** The small JSON descriptor [`tier_training_pipeline.square8_2p.json`](ai-service/config/tier_training_pipeline.square8_2p.json:1) mirrors the demo defaults (seeds, gating overrides) and is used by tests as a shape contract. Future Code-mode work can thread this config through the orchestrators to centralise per-tier settings.

---

## 9. Tier-by-tier summary

| Tier | Model type / candidate                           | Training data                                             | Gating criteria (automated)                                                              | Perf handling                                                                  | Human calibration role                                                                                    |
| ---- | ------------------------------------------------ | --------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------- |
| D2   | Heuristic profile                                | Territory / combined-margin JSONL; optional NPZ self-play | ≥60% vs D1 random (`TierEvaluationConfig D2`); basic sanity that it clearly beats random | No dedicated perf budget; must feel responsive and stable                      | Template A with new players; ensure beginners can win at least some games and difficulty feels around 3   |
| D4   | Minimax persona (+ heuristic profile)            | Mixed self-play on Square-8 2p; tournaments vs D2/random  | ≥70% vs baselines; no major regression vs D2 per TierEvaluationConfig                    | Hard D4 perf budget (`TierPerfBudget D4_SQ8_2P`); reject or retune if exceeded | Templates A+B with intermediates; target ~30–70% win rate for intended audience                           |
| D6   | High minimax or NN-backed search (`nn_model_id`) | Self-play centred on D4–D6; curriculum loops              | ≥75% vs baselines; no major regression vs D4 per TierEvaluationConfig                    | Hard D6 perf budget; NN/search config must respect it                          | Templates B+C with strong players; target ~40–60% win rate, with intermediate players typically below 30% |
| D8   | MCTS (+ optional NN) (`nn_model_id`)             | High-quality D6/D8 self-play; curriculum from D6          | ≥80% vs baselines; no major regression vs D6 per TierEvaluationConfig                    | Hard D8 perf budget; allowed near ceiling but not above                        | Template C with very strong players; even strong testers should struggle to exceed ~60% win rate          |

---

## 10. Cross-references and doc updates

Follow-up doc work (in Code/Docs-mode) should:

- Add a short “Training & promotion pipeline” subsection to the future `AI_DIFFICULTY_SPEC.md` that points at this document as the process SSoT for D2/D4/D6/D8 on Square-8 2-player.
- Cross-link the calibration analysis process in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](docs/ai/AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1) as the **human-facing difficulty tuning SSoT**, sitting on top of the automated training and gating loop defined here.
- Add a note to [`AI_HUMAN_CALIBRATION_GUIDE.md`](docs/ai/AI_HUMAN_CALIBRATION_GUIDE.md:1) clarifying that:
  - calibration templates A/B/C feed into the H-AI-9 promotion loop; and
  - calibration telemetry and experiment results are interpreted using the workflow and decision rules in [`AI_DIFFICULTY_CALIBRATION_ANALYSIS.md`](docs/ai/AI_DIFFICULTY_CALIBRATION_ANALYSIS.md:1); and
  - calibration is run on candidates that have already passed automated gates and perf budgets.

This completes the architectural design for H-AI-9 – Tiered Model Training & Promotion Loop (Square-8 2-Player). Implementation work is limited to adding orchestration scripts, wiring model ids / heuristic profiles into the existing training code, and updating ladder configs in controlled, well-documented promotions.
